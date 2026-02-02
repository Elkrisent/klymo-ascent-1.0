import torch
import torch.nn.functional as F
import numpy as np
from math import log10


def calculate_psnr(sr, hr):
    """
    Peak Signal-to-Noise Ratio. Higher is better (typical: 25-35 dB).
    """
    mse = F.mse_loss(sr, hr)
    if mse == 0:
        return float('inf')
    psnr = 10 * log10(1.0 / mse.item())
    return psnr


def calculate_ssim(sr, hr):
    """
    Structural Similarity Index.
    Range: [0, 1], higher is better.
    """
    from skimage.metrics import structural_similarity as ssim

    sr_np = sr.cpu().numpy()
    hr_np = hr.cpu().numpy()

    ssim_val = 0
    for i in range(sr_np.shape[0]):  # loop over batch
        ssim_val += ssim(
            sr_np[i].transpose(1, 2, 0),   # CHW → HWC
            hr_np[i].transpose(1, 2, 0),
            channel_axis=2,                 # replaces deprecated multichannel=True
            data_range=1.0
        )
    return ssim_val / sr_np.shape[0]
