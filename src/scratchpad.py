#!/usr/bin/env python
"""
attention_autoencoder_ddp.py (bf16‑first)
========================================
3‑D Attention Autoencoder *with native **bfloat16** training* and PyTorch DDP.

Highlights compared to the previous version
-------------------------------------------
* **bf16 everywhere**  – model weights, activations, and gradients all stored in
  bfloat16 to cut memory roughly in half on A100/H100 while keeping dynamic
  range.
* **AMP autocast(dtype=bfloat16)**  – ensures ops that still need fp32 (e.g.
  BatchNorm stats) stay safe.
* **GradScaler removed**  – not required for bf16.
* Extra CLI flag `--precision` (default `bf16`, can set `fp32` for debugging).
* `torch.backends.cuda.matmul.allow_tf32 = True` enabled for additional speed.

Run (single node, 8 GPUs):
```bash
pip install torch torchvision natten einops
TORCH_CUDNN_V8_API_ENABLED=1 \
  torchrun --standalone --nnodes=1 --nproc_per_node=8 attention_autoencoder_ddp.py \
  --data_dir /path/to/npy_volumes \
  --batch_size 1 --epochs 200 --precision bf16
```
"""
from __future__ import annotations
import argparse, os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.cuda import amp

import natten  # Neighborhood Attention
from einops import rearrange

################################################################################
# Utility layers
################################################################################


class PixelUnshuffle3D(nn.Module):
    """3‑D extension of torch.nn.PixelUnshuffle."""

    def __init__(self, downscale_factor: int = 2):
        super().__init__()
        self.r = downscale_factor

    def forward(self, x: torch.Tensor):
        b, c, d, h, w = x.shape
        r = self.r
        assert d % r == h % r == w % r == 0, "Dims must be divisible by factor"
        x = x.view(b, c, d // r, r, h // r, r, w // r, r)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        return x.view(b, c * r**3, d // r, h // r, w // r)


class PixelShuffle3D(nn.Module):
    """3‑D extension of torch.nn.PixelShuffle."""

    def __init__(self, upscale_factor: int = 2):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x: torch.Tensor):
        b, c, d, h, w = x.shape
        r = self.r
        assert c % (r**3) == 0, "Channels not divisible by factor³"
        cout = c // r**3
        x = x.view(b, cout, r, r, r, d, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return x.view(b, cout, d * r, h * r, w * r)


################################################################################
# Attention blocks
################################################################################


class NA3DBlock(nn.Module):

    def __init__(self,
                 dim,
                 kernel_size=5,
                 num_heads=8,
                 dilation=1,
                 dropout=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm3d(dim)
        self.attn = natten.NeighborhoodAttention3D(dim=dim,
                                                   kernel_size=kernel_size,
                                                   dilation=dilation,
                                                   num_heads=num_heads,
                                                   dropout=dropout)
        self.norm2 = nn.BatchNorm3d(dim)
        self.mlp = nn.Sequential(nn.Conv3d(dim, dim * 4, 1), nn.GELU(),
                                 nn.Conv3d(dim * 4, dim, 1))

    def forward(self, x):
        h = self.norm1(x)
        h_cl = h.permute(0, 2, 3, 4, 1).contiguous()  # channels‑last for NATTEN
        h_cl = self.attn(h_cl)
        h = h_cl.permute(0, 4, 1, 2, 3).contiguous()
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class GlobalAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim,
                                          num_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Linear(dim * 4, dim))

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


################################################################################
# Model
################################################################################


class AttentionAutoencoder3D(nn.Module):

    def __init__(self,
                 in_chans=1,
                 img_size=112,
                 factor=2,
                 depth_enc=(2, 2),
                 depth_bottleneck=12,
                 num_heads=8,
                 kernel_size=5):
        super().__init__()
        r = factor
        self.pu1 = PixelUnshuffle3D(r)
        c1 = in_chans * r**3
        self.enc1 = nn.Sequential(*[
            NA3DBlock(c1, kernel_size, num_heads) for _ in range(depth_enc[0])
        ])
        self.pu2 = PixelUnshuffle3D(r)
        c2 = c1 * r**3
        self.enc2 = nn.Sequential(*[
            NA3DBlock(c2, kernel_size, num_heads) for _ in range(depth_enc[1])
        ])
        self.bottleneck = nn.Sequential(*[
            GlobalAttentionBlock(c2, num_heads) for _ in range(depth_bottleneck)
        ])
        self.ps2 = PixelShuffle3D(r)
        self.dec2 = nn.Sequential(*[
            NA3DBlock(c1, kernel_size, num_heads) for _ in range(depth_enc[1])
        ])
        self.ps1 = PixelShuffle3D(r)
        self.dec1 = nn.Sequential(*[
            NA3DBlock(in_chans, kernel_size, num_heads)
            for _ in range(depth_enc[0])
        ])
        self.out_conv = nn.Conv3d(in_chans, in_chans, 1)

    def forward(self, x):
        x = self.pu1(x)
        x = self.enc1(x)
        x = self.pu2(x)
        x = self.enc2(x)
        b, c, d, h, w = x.shape
        x = self.bottleneck(x.flatten(2).transpose(1, 2)).transpose(1, 2).view(
            b, c, d, h, w)
        x = self.ps2(x)
        x = self.dec2(x)
        x = self.ps1(x)
        x = self.dec1(x)
        return self.out_conv(x)


################################################################################
# Dataset
################################################################################


class VoxelDataset(Dataset):

    def __init__(self, data_dir: str):
        self.paths = sorted(Path(data_dir).glob("*.npy"))
        if not self.paths:
            raise RuntimeError(f"No .npy files in {data_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        vol = np.load(self.paths[idx]).astype(np.float32)
        return torch.from_numpy(vol)[None]  # (1, D, H, W)


################################################################################
# Training helper
################################################################################


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--factor", type=int, default=2)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    return rank


def cleanup_ddp():
    dist.destroy_process_group()


################################################################################
# Main training loop
################################################################################


def main():
    args = parse_args()
    rank = setup_ddp()
    is_main = rank == 0
    device = torch.device("cuda", rank)

    # Enable TF32 for extra performance (safe with bf16)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = AttentionAutoencoder3D(factor=args.factor).to(device)

    if args.precision == "bf16":
        model = model.to(dtype=torch.bfloat16)
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float32  # effectively no autocast

    model = DDP(model, device_ids=[rank], broadcast_buffers=False)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=1e-4)
    criterion = nn.MSELoss()

    dataset = VoxelDataset(args.data_dir)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        sampler=sampler,
                        num_workers=4,
                        pin_memory=True,
                        persistent_workers=True)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        for step, vols in enumerate(loader):
            vols = vols.to(device, dtype=autocast_dtype, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda", dtype=autocast_dtype):
                recon = model(vols)
                loss = criterion(recon, vols)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if is_main and step % args.log_interval == 0:
                print(
                    f"[R{rank}] Ep {epoch}/{args.epochs} It {step}/{len(loader)} Loss {loss.item():.4f}"
                )
        if is_main:
            ckpt = {
                "epoch": epoch,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "precision": args.precision,
            }
            torch.save(ckpt, Path(args.save_dir) / f"epoch_{epoch}.pt")
            print(
                f"Saved ckpt epoch {epoch}, avg loss {epoch_loss/len(loader):.4f}"
            )

    cleanup_ddp()


if __name__ == "__main__":
    main()
