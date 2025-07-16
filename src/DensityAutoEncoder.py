import torch
import torch.nn as nn
import natten
from natten import na3d, NeighborhoodAttention3D


class PixelUnshuffle3D(nn.Module):

    def __init__(self, downsample_factor=2):
        super().__init__()
        self.f = downsample_factor

    def forward(self, x):
        f = self.f
        N, C, D, H, W = x.shape
        assert D == H == W, "Input density map must be a cubic, ie. D=H=W"
        assert D % f == 0, "Input density map must be divisible by downsample factor"
        x = x.view(N, C, D // f, f, H // f, f, W // f, f)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        return x.view(N, C * f**3, D // f, H // f, W // f)


class PixelShuffle3D(nn.Module):

    def __init__(self, upsample_factor=2):
        super().__init__()
        self.f = upsample_factor

    def forward(self, x):
        f = self.f
        N, C, D, H, W = x.shape
        assert C % (f**3) == 0, "Channels not divisible by factor^3"
        x = x.view(N, C // f**3, f, f, f, D, H, W)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return x.view(N, C // f**3, D * f, H * f, W * f)


class AdaRMSNorm3D(nn.Module):

    # input shape: [B, C, D, H, W]
    def __init__(self, num_channels):
        super().__init__()
        hidden_dim = 2 * num_channels
        self.num_channels = num_channels
        self.eps = 1e-6
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1, 1))
        self.gamma_net = nn.Sequential(nn.Linear(num_channels, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, num_channels),
                                       nn.Sigmoid())

    def forward(self, x):
        B, C, D, H, W = x.shape
        rms = torch.sqrt(
            torch.mean(x.pow(2), dim=(2, 3, 4), keepdim=True) +
            self.eps)  # (B, C, 1, 1, 1)
        x_normed = x / rms  # (B, C, D, H, W)
        channel_stats = torch.mean(x, dim=(2, 3, 4))  # [B, C,]
        gamma_dynamic = self.gamma_net(channel_stats)  # [B, C]
        gamma_dynamic = gamma_dynamic.view(B, C, 1, 1, 1)  # [B, C, 1, 1, 1]
        return x_normed * (self.scale * gamma_dynamic)


class NA3DBlock(nn.Module):

    # NOTE:
    # Make sure (embed_dim / num_heads) % D == 0
    # Or NATTEN cannot find a suitable backend
    def __init__(self,
                 in_channels,
                 embed_dim,
                 kernel_size,
                 num_heads,
                 stride=1,
                 dilation=1,
                 proj_drop=0.0,
                 mlp_ratio=4.0):
        super().__init__()
        self.proj_in = nn.Conv3d(
            in_channels, embed_dim,
            1) if embed_dim != in_channels else nn.Identity()
        self.norm1 = AdaRMSNorm3D(embed_dim)
        self.attn = natten.NeighborhoodAttention3D(embed_dim=embed_dim,
                                                   num_heads=num_heads,
                                                   kernel_size=kernel_size,
                                                   stride=stride,
                                                   dilation=dilation,
                                                   proj_drop=proj_drop)
        self.norm2 = AdaRMSNorm3D(embed_dim)
        self.mlp = nn.Sequential(
            nn.Conv3d(embed_dim, int(mlp_ratio * embed_dim), kernel_size=1),
            nn.GELU(),
            nn.Conv3d(int(embed_dim * mlp_ratio), embed_dim, kernel_size=1))
        self.proj_out = nn.Conv3d(
            embed_dim, in_channels,
            1) if embed_dim != in_channels else nn.Identity()

    def forward(self, x):
        x = self.proj_in(x)
        h = self.norm1(x).permute(0, 2, 3, 4, 1).contiguous()  # [B, D, H, W, C]
        h = self.attn(h).permute(0, 4, 1, 2, 3).contiguous()  # [B, C, D, H, W]
        x = x + h  # Residual connection
        x = x + self.mlp(self.norm2(x))
        x = self.proj_out(x)
        return x


class GlobalAttentionBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 embed_dim,
                 num_heads=8,
                 dropout=0.0,
                 mlp_ratio=4.0):
        super().__init__()
        self.proj_in = nn.Linear(
            in_channels,
            embed_dim) if embed_dim != in_channels else nn.Identity()
        self.norm1 = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(in_channels,
                                          num_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, int(in_channels * mlp_ratio)), nn.GELU(),
            nn.Linear(int(in_channels * mlp_ratio), in_channels))
        self.proj_out = nn.Linear(
            embed_dim,
            in_channels) if embed_dim != in_channels else nn.Identity()

    def forward(self, x):
        # x: [B, D*H*W, C]
        x = self.proj_in(x)  # [B, D*H*W, embed_dim]
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        x = self.proj_out(x)
        return x


class DensityEncoder(nn.Module):

    def __init__(self,
                 in_channels=1,
                 downsampleFactors=(4, 2),
                 NA_num_layers=(2, 2),
                 NA_num_heads=(8, 16),
                 NA_embed_dim=(896, 1344),
                 GA_num_layers=12,
                 GA_num_heads=16,
                 GA_embed_dim=1344,
                 kernel_size=7):
        super().__init__()
        self.downsample1 = PixelUnshuffle3D(
            downsampleFactors[0])  # [B, 64, D//4, H//4, W//4]
        channels = in_channels * downsampleFactors[0]**3
        self.encode1 = nn.Sequential(*[
            NA3DBlock(in_channels=channels,
                      embed_dim=NA_embed_dim[0],
                      kernel_size=kernel_size,
                      num_heads=NA_num_heads[0])
            for _ in range(NA_num_layers[0])
        ])
        self.downsample2 = PixelUnshuffle3D(downsampleFactors[1])
        channels = channels * downsampleFactors[1]**3
        self.encode2 = nn.Sequential(*[
            NA3DBlock(channels,
                      embed_dim=NA_embed_dim[1],
                      kernel_size=kernel_size,
                      num_heads=NA_num_heads[1])
            for _ in range(NA_num_layers[1])
        ])
        self.bottleneck = nn.Sequential(*[
            GlobalAttentionBlock(channels, GA_embed_dim, num_heads=GA_num_heads)
            for _ in range(GA_num_layers)
        ])

    def forward(self, x):
        # x: [B, 1, D, H, W]
        x = self.downsample1(x)  # [B, 64, D/4, H/4, W/4]
        x = self.encode1(x)
        x = self.downsample2(x)
        x = self.encode2(x)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, D*H*W, C]
        x = self.bottleneck(x).transpose(1, 2)  # [B, C, D*H*W]
        x = x.view(B, C, D, H, W)
        return x


class DensityDecoder(nn.Module):

    def __init__(self,
                 in_channels=512,
                 upsampleFactors=(2, 4),
                 NA_num_layers=(2, 2),
                 NA_num_heads=(16, 8),
                 NA_embed_dim=(1344, 896),
                 kernel_size=7):
        super().__init__()
        self.decode1 = nn.Sequential(*[
            NA3DBlock(in_channels,
                      NA_embed_dim[0],
                      kernel_size=kernel_size,
                      num_heads=NA_num_heads[0])
            for _ in range(NA_num_layers[0])
        ])
        self.upsample1 = PixelShuffle3D(upsampleFactors[0])
        channels = in_channels // (upsampleFactors[0]**3)
        self.decode2 = nn.Sequential(*[
            NA3DBlock(channels,
                      NA_embed_dim[1],
                      kernel_size=kernel_size,
                      num_heads=NA_num_heads[1])
            for _ in range(NA_num_layers[1])
        ])
        self.upsample2 = PixelShuffle3D(upsampleFactors[1])
        channels = channels // (upsampleFactors[1]**3)
        self.out_conv = nn.Conv3d(channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.decode1(x)
        x = self.upsample1(x)
        x = self.decode2(x)
        x = self.upsample2(x)
        x = self.out_conv(x)
        return x


if __name__ == '__main__':

    from torchinfo import summary
    from thop import profile

    device = 'cuda:0'
    x = torch.randn(2, 1, 224, 224, 224).to(dtype=torch.bfloat16).to(device)
    model = nn.Sequential(DensityEncoder(),
                          DensityDecoder()).to(dtype=torch.bfloat16).to(device)
    # y = model(x)
    summary(model, input_data=x, device="cuda", mode='train')
    #
    # flops, params = profile(model, inputs=(x,))
    # print(f"GFLOPs: {flops / 1e9:.2f}")
    # print(f'参数量: {params / 1e6:.2f} M')

    # y = model(x)
    # print(y.shape)
