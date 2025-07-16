import numpy as np
import torch
import cv2
import os


def norm(src,
         method='zscore',
         mean=0,
         std=1,
         remove_abnormal=False,
         lb=2,
         ub=98):
    '''
    Normalize ndarray or tensor

    src: input array/tensor
    method: 
        'zscore': z-score normalization, with mean and std args
        '255': norm to [0, 255] uint8
        '01': norm to [0, 1] float32
    mean: mean value for z-score normalization
    std: std value for z-score normalization
    remove_abnormal: whether to remove abnormal values
    lb: lower bound for abnormal value removal
    ub: upper bound for abnormal value removal

    return: normalized array/tensor

    NOTE: torch.Tensor input will be detached and moved to cpu in this function call
    '''

    input_type = type(src)
    if input_type == torch.Tensor:
        src = src.cpu().detach().numpy()
    src = src.astype(np.float32)

    if remove_abnormal:
        src = np.clip(src, np.percentile(src, lb), np.percentile(src, ub))

    if method == 'zscore':
        src = (src - np.mean(src)) / np.std(src)
        src = src * std + mean
    elif method == '255':
        src = (src - np.min(src)) / (np.max(src) - np.min(src)) * 255.
        src = np.clip(src, 0, 255)
        src = src.astype(np.uint8)
    elif method == '01':
        src = (src - np.min(src)) / (np.max(src) - np.min(src))
        src = np.clip(src, 0, 1)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    if input_type == torch.Tensor:
        src = torch.from_numpy(src)

    return src


def gaussian_blur(src, sigma, axis):
    input_type = type(src)
    if input_type == torch.Tensor:
        src = src.cpu().detach().numpy()
    src = src.astype(np.float32)

    if input_type == torch.Tensor:
        src = torch.from_numpy(src)
    return src

def crpad3D(volume:torch.Tensor, resize_factor:float):
    """
    Resize the input volume tensor using Fouier cropping or padding
    volume: (B, D, H, W)
    """
    device = volume.device
    D, H, W = volume.shape[-3:]
    assert resize_factor > 0, "Resize factor must be greater than 0"
    if resize_factor >= 1.0:
        padded_size = round(resize_factor * D)
        padded_size += padded_size % 2  # make sure padded_size is even
        pad_D, pad_H, pad_W = padded_size, padded_size, padded_size
        pad_len1 = (padded_size - D) // 2
        pad_len2 = padded_size - D - pad_len1
        volume = torch.nn.functional.pad(
            volume, (pad_len1, pad_len2, pad_len1, pad_len2, pad_len1, pad_len2),
            mode='constant',
            value=0)
