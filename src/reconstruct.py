import os
import torch
import mrcfile
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch3d.transforms import quaternion_apply
from torch.fft import fftn, fftshift, fftfreq, ifftshift, ifftn, fft2, ifft2


def _preprocess_before_reconstruct(
    particles,
    volume,
    padding_factor=2.
) -> tuple[int, int, int, int, int, int, torch.Tensor, torch.Tensor]:
    '''
    particles: (N, H, W)
    volume: None or (D, H, W), if given, particles will be added to this volume
    '''

    H, W = particles.shape[-2], particles.shape[-1]
    D = H

    if volume is None:
        volume = torch.zeros((D, H, W), dtype=particles.dtype)
    device = particles.device
    volume = volume.to(device)

    assert volume.shape[-3] == volume.shape[-2] == volume.shape[-1]
    assert volume.shape[-1] == H == W == D

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
    particles = torch.nn.functional.pad(
        particles, (pad_len1, pad_len2, pad_len1, pad_len2),
        mode='constant',
        value=0)

    # Calculate the FFT of volume and particles
    volume = fftshift(volume)
    volume = fftn(volume)
    volume = fftshift(volume)

    particles = fftshift(particles, dim=(1, 2))
    particles = fft2(particles)
    particles = fftshift(particles, dim=(1, 2))

    return D, H, W, pad_D, pad_H, pad_W, volume, particles


def reconstruct(particles: torch.Tensor,
                quaternions: torch.Tensor,
                weights: torch.Tensor,
                volume=None,
                pixel_size=1.0,
                padding_factor=2.):
    '''
    particles: (N, H, W)
    quaternions: (N, 4)
    weight: (N,)
    volume: (D, H, W), if given, particles will be added to this volume
    pixel_size: in Angstrom per pixel
    '''

    batch_size = particles.shape[0]

    # This function does the following things:
    # 1. Zero-pad the volume and particles in real space
    # 2. Perform the Fourier transform to padded volume and particles; in detail, the order is fftshift-fft-fftshift, to avoid aliasing in the following rotation operations
    D, H, W, pad_D, pad_H, pad_W, volume, particles = _preprocess_before_reconstruct(
        particles, volume, padding_factor)

    device = particles.device
    quaternions = quaternions.to(device)  # (N, 4)
    weights = weights.to(device)  # (N,)

    # First put the particles' coords at the central slice, then rotate them according to quaternions
    # Get the coords of the central slice (the slice locates at pad_D//2)
    x = torch.linspace(-pad_W // 2, pad_W // 2 - 1, pad_W)  # (pad_W,)
    y = torch.linspace(-pad_H // 2, pad_H // 2 - 1, pad_H)  # (pad_H,)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # (pad_H, pad_W)
    grid_z = torch.zeros_like(grid_y)  # (pad_H, pad_W)
    particles_coords = torch.stack(
        (grid_x, grid_y, grid_z),
        dim=-1)  # shape: (pad_H, pad_W, 3); elements: (X, Y, Z=0)

    # Prepend the Depth dimension (which is 1 because only the central slice is needed) to the central slice coords
    # Convert 2D coords to 3D
    # Then prepend and expand the batch dimension
    particles_coords = particles_coords.view(1, 1, pad_H, pad_W, 3).expand(
        batch_size, -1, -1, -1, -1)  # (N, D=1, pad_H, pad_W, 3)

    # Move to device
    particles_coords = particles_coords.to(device)

    # Apply **INVERSE** quaternions to central slice's coords to calculate orientations
    # coord order should be (X, Y, Z) when applying quaternions
    quaternions[:, 1:] = -quaternions[:, 1:]
    particles_coords = quaternion_apply(quaternions, particles_coords).squeeze(
        1)  # (N, pad_H, pad_W, 3)

    ceil_coords = torch.floor(particles_coords)
    floor_coords = torch.ceil(particles_coords)

    # Calculate the coordinates of 8 surrounding points for each grid of the particle
    # And convert the coordinates to matrix indices
    c000_index = torch.floor(
        particles_coords).int() + D // 2  # (N, pad_H, pad_W, 3)
    c100_index = torch.stack(
        [ceil_coords[..., 0], floor_coords[..., 1], floor_coords[..., 2]],
        dim=-1).int() + D // 2
    c110_index = torch.stack(
        [ceil_coords[..., 0], ceil_coords[..., 1], floor_coords[..., 2]],
        dim=-1).int() + D // 2
    c010_index = torch.stack(
        [floor_coords[..., 0], ceil_coords[..., 1], floor_coords[..., 2]],
        dim=-1).int() + D // 2
    c001_index = torch.stack(
        [floor_coords[..., 0], floor_coords[..., 1], ceil_coords[..., 2]],
        dim=-1).int() + D // 2
    c101_index = torch.stack(
        [ceil_coords[..., 0], floor_coords[..., 1], ceil_coords[..., 2]],
        dim=-1).int() + D // 2
    c111_index = torch.ceil(particles_coords).int() + D // 2
    c011_index = torch.stack(
        [floor_coords[..., 0], ceil_coords[..., 1], ceil_coords[..., 2]],
        dim=-1).int() + D // 2

    distance_ceil_particles = ceil_coords - particles_coords  # (N, pad_H, pad_W, 3)
    distance_particles_floor = particles_coords - floor_coords # (N, pad_H, pad_W, 3)

    del ceil_coords, floor_coords

    # distance_c1_c = distance_ceil_particles[..., 2]  # (N, pad_H, pad_W)
    # distance_c_c0 = distance_particles_floor[..., 2]
    # distance_c10_c0 = distance_ceil_particles[..., 1]
    # distance_c0_c00 = distance_particles_floor[..., 1]
    # distance_c100_c00 = distance_ceil_particles[..., 0]
    # distance_c00_c000 = distance_particles_floor[..., 0]

    c1 = particles * distance_particles_floor[..., 2]  # (N, pad_H, pad_W)
    c0 = particles * distance_ceil_particles[..., 2]

    c00 = c0 * distance_ceil_particles[..., 1]
    c10 = c0 * distance_particles_floor[..., 1]
    c01 = c1 * distance_ceil_particles[..., 1]
    c11 = c1 * distance_particles_floor[..., 1]
    del c1, c0
    c000 = c00 * distance_ceil_particles[..., 0]
    c100 = c00 * distance_particles_floor[..., 0]
    del c00
    c010 = c10 * distance_ceil_particles[..., 0]
    c110 = c10 * distance_particles_floor[..., 0]
    del c10
    c001 = c01 * distance_ceil_particles[..., 0]
    c101 = c01 * distance_particles_floor[..., 0]
    del c01
    c011 = c11 * distance_ceil_particles[..., 0]
    c111 = c11 * distance_particles_floor[..., 0]
    del c11

    volume


def run_reconstruct():
    pass
