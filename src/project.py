import torch
from pytorch3d.transforms import quaternion_apply, random_quaternions
from math import sqrt, ceil
import torch.multiprocessing as mp
import torch.distributed as dist
import mrcfile
import os
import numpy as np
# import pandas as pd
import tifffile
from tqdm import tqdm
from pathlib import Path


def project(volume: torch.Tensor,
            quaternions: torch.Tensor,
            padding_factor=2.,
            # griddingCorrection=False,
            device=torch.device('cpu')):
    '''
    volume: shape (D, H, W)
    quaternions: shape (batch, 4) or (4,)
    padding_factor: the padding ratio in real space before calculating Fourier transform;
                    this will not affect the output project image size
    return: 
        2D projections of the input volume and corresponding quaternions
        projections shape: (N, H, W)
        quaternions shape: (N, 4)
    '''

    # Move volume and quaternions to device
    volume = volume.to(device)
    quaternions = quaternions.to(device)  # (N, 4)

    # Input volume must be in a cube box
    D, H, W = volume.shape[-3], volume.shape[-2], volume.shape[-1]
    assert D == H == W, 'Input volume box must be a cube!'

    # Convert input to float32 torch.Tensor
    volume = volume.to(torch.float32)
    quaternions = quaternions.to(torch.float32)  # (N, 4)

    # If a single quaternion is given, treat it as batch_size=1
    if len(quaternions.shape) == 1:
        quaternions = quaternions[None, :]

    # Define batch_size = N
    batch_size = quaternions.shape[0]

    # Zero padding in real-space before projection
    padded_size = round(padding_factor * D)
    padded_size += padded_size % 2  # make sure padded_size is even
    pad_D, pad_H, pad_W = padded_size, padded_size, padded_size
    pad_len1 = (padded_size - D) // 2
    pad_len2 = padded_size - D - pad_len1
    volume = torch.nn.functional.pad(
        volume, (pad_len1, pad_len2, pad_len1, pad_len2, pad_len1, pad_len2),
        mode='constant',
        value=0)

    # if griddingCorrection:
    #     index_i, index_j, index_k = torch.meshgrid(torch.arange(pad_D),
    #                                                torch.arange(pad_H),
    #                                                torch.arange(pad_W),
    #                                                indexing='ij')
    #     rval = torch.sqrt(index_i**2 + index_j**2 + index_k**2)
    #     del index_i, index_j, index_k
    #     correction_sinc = torch.sinc(torch.pi * rval)
    #     del rval
    #     volume /= correction_sinc**2

    # Revert quaternions because rotation grid_sampler applys on sample grids instead of objects
    quaternions[:, 1:] = -quaternions[:, 1:]

    # Unsqueeze and expand
    quaternions = quaternions.view(batch_size, 1, 1, 1,
                                   4).expand(-1, 1, pad_H, pad_W,
                                             -1)  # (N, D=1, H, W, 4)

    # FFT the volume
    # fftshift must be applied both in spatial and frequency space before rotation
    volume = torch.fft.fftshift(volume)  # (pad_D, pad_H, pad_W)
    volume = torch.fft.fftn(volume)
    volume = torch.fft.fftshift(volume)

    # unsqueeze FFTed volume to (N, C, pad_D, pad_H, pad_W)
    volume = volume[None, None, :, :, :]
    volume = volume.expand(batch_size, -1, -1, -1,
                           -1)  # (N, C, pad_D, pad_H, pad_W)

    # Get the coords of the central slice (the slice locates at pad_D//2)
    x = torch.linspace(-1, 1, pad_W)  # (pad_W,)
    y = torch.linspace(-1, 1, pad_H)  # (pad_H,)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # (pad_H, pad_W)
    grid_z = torch.zeros_like(grid_y)  # (pad_H, pad_W)
    central_slice_coords = torch.stack(
        (grid_x, grid_y, grid_z),
        dim=-1)  # shape: (pad_H, pad_W, 3); elements: (X, Y, Z=0)
    # Prepend the Depth dimension (which is 1 because only the central slice is needed) to the central slice coords
    # Convert 2D coords to 3D
    # Then prepend and expand the batch dimension
    central_slice_coords = central_slice_coords.view(
        1, 1, pad_H, pad_W, 3).expand(batch_size, -1, -1, -1,
                                      -1)  # (N, D=1, pad_H, pad_W, 3)
    central_slice_coords = central_slice_coords.to(device)

    # Apply quaternions to central slice's coords
    # coord order should be (X, Y, Z) when applying quaternions
    rotated_central_slice_coords = quaternion_apply(
        quaternions, central_slice_coords)  # (N, D=1, pad_H, pad_W, 3)

    # grid_sample does not support complex numbers
    # so we need to take apart the FFTed volume to real and imag parts
    # grid_sample works in (X, Y, Z) order, where X corresponds to the 1st and Z to the 3rd dimension
    # align_corners=True is necessary
    fft_projs_real = torch.nn.functional.grid_sample(
        volume.real,
        rotated_central_slice_coords,
        mode='bilinear',
        align_corners=True,
        padding_mode='zeros')  # (N, C, D=1, pad_H, pad_W)
    fft_projs_imag = torch.nn.functional.grid_sample(
        volume.imag,
        rotated_central_slice_coords,
        mode='bilinear',
        align_corners=True,
        padding_mode='zeros')
    fft_projs = torch.complex(fft_projs_real,
                              fft_projs_imag)[:, 0,
                                              0, :, :]  # (N, pad_H, pad_W)
    fft_projs = torch.fft.ifftshift(fft_projs, dim=(1, 2))  # (N, pad_H, pad_W)
    fft_projs = torch.fft.ifft2(fft_projs)
    fft_projs = torch.fft.ifftshift(fft_projs, dim=(1, 2))  # (N, pad_H, pad_W)

    # Crop back to the size before padding
    fft_projs = torch.fft.fftshift(torch.fft.fft2(fft_projs))
    fft_projs = fft_projs[:, pad_len1:-pad_len2,
                          pad_len1:-pad_len2]  # (N, H, W)
    projs = torch.fft.ifft2(torch.fft.ifftshift(fft_projs))
    del fft_projs
    projs = projs.real

    quaternions = quaternions[:, 0, 0, 0, :]  # (N, 4)

    return projs, quaternions  # (N, H, W) and (N, 4)


def ddp_project(rank, world_size, volume_path: str, output_path: str,
                batch_size: int, projections_amount: int):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}")

    # Create subdirs for projections and quaternions seperately
    Path(f"{output_path}/projections").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/quaternions").mkdir(parents=True, exist_ok=True)

    mrc = mrcfile.read(volume_path)
    mrc = np.asarray(mrc)
    mrc = torch.from_numpy(mrc).to(device)

    projections_per_rank = projections_amount // world_size
    batch_num = projections_per_rank // batch_size

    # Generate a tqdm progress bar for monitoring
    pbar = tqdm(total=projections_per_rank, desc=f"Rank {rank}", position=rank)

    for batch in range(batch_num):
        # Generate batched random quaternions
        batched_quaternions = random_quaternions(batch_size,
                                                 device=device).to(device)

        # Generate projections
        batched_projections, batched_quaternions = project(mrc,
                                                           batched_quaternions,
                                                           device=device)

        # move to cpu
        batched_projections = batched_projections.cpu()
        batched_quaternions = batched_quaternions.cpu()

        # Save
        for i in range(batch_size):
            count = rank * projections_per_rank + batch * batch_size + i
            torch.save(batched_projections[i],
                       f"{output_path}/projections/proj_{count}.pt")
            torch.save(batched_quaternions[i],
                       f"{output_path}/quaternions/quat_{count}.pt")
            pbar.update(1)

    # close tqdm bar
    pbar.close()

    # Cleanup
    dist.destroy_process_group()


if __name__ == '__main__':
    # world_size = torch.cuda.device_count()  # Use all GPUs for projection
    # volume_path = './data/emd_19182.map'  # TODO: Change volume path
    # output_path = "./data/proj_emd19182"
    # batch_size = 3
    # projections_amount = 6
    #
    # mp.spawn(ddp_project,
    #          args=(world_size, volume_path, output_path, batch_size,
    #                projections_amount),
    #          nprocs=world_size,
    #          join=True)

    import mrcfile
    import sys
    from pytorch3d.transforms import random_quaternions
    mrc = mrcfile.read(sys.argv[1])
    print(mrc.shape)
    mrc = torch.from_numpy(np.asarray(mrc)).to(torch.device('cuda'))
    q = torch.tensor([[1., 0., 0., 0.]]).to(torch.device('cuda'))
    # q = random_quaternions(1)
    projs, _ = project(mrc, q, device=torch.device('cpu'), griddingCorrection=True)
    projs = projs.cpu().numpy()
    print(projs.shape)
    import matplotlib.pyplot as plt
    plt.imshow(projs[0], cmap='gray')
    plt.show()
