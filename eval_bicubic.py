import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from dataset import SatelliteSRDataset
from utils.metrics import calculate_psnr, calculate_ssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- same split logic as train_swinir.py (random_state=42 keeps it identical) ---
hr_dir = os.path.join(os.path.dirname(__file__), 'worldstrat_processed', 'HR_8bit')

dataset = SatelliteSRDataset(
    hr_dir=hr_dir,
    lr_patch_size=64,  # Full size to match training
    scale=4
)

indices = list(range(len(dataset)))
_, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

val_loader = DataLoader(
    Subset(dataset, val_idx),
    batch_size=4,
    shuffle=False,
    num_workers=0
)

# --- Evaluate bicubic ---
psnr_total = 0
ssim_total = 0

for lr_img, hr_img in val_loader:
    lr_img, hr_img = lr_img.to(device), hr_img.to(device)
    sr_bicubic = F.interpolate(lr_img, scale_factor=4, mode='bicubic', align_corners=False)
    psnr_total += calculate_psnr(sr_bicubic, hr_img)
    ssim_total += calculate_ssim(sr_bicubic, hr_img)

print(f"Bicubic PSNR: {psnr_total/len(val_loader):.2f} dB")
print(f"Bicubic SSIM: {ssim_total/len(val_loader):.4f}")
