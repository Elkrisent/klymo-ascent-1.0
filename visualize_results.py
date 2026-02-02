import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

from models.swinir_satellite import SatelliteSwinIR
from dataset import SatelliteSRDataset  # Fixed import
from utils.metrics import calculate_psnr, calculate_ssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create results directory
os.makedirs('results', exist_ok=True)

# Load model
model = SatelliteSwinIR(scale=4).to(device)

# Try multiple checkpoint locations
checkpoint_path = None
for path in [
    'checkpoints/swinir_best.pth',
    'checkpoints/swinir_epoch100.pth',
    'checkpoints_backup/swinir_epoch100.pth'
]:
    if os.path.exists(path):
        checkpoint_path = path
        break

if checkpoint_path is None:
    raise FileNotFoundError("No checkpoint found! Looked in: checkpoints/, checkpoints_backup/")

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Checkpoint info:")
if 'psnr' in checkpoint:
    print(f"  PSNR: {checkpoint['psnr']:.2f} dB")
if 'epoch' in checkpoint:
    print(f"  Epoch: {checkpoint['epoch']}")

# Load validation data (SAME split as training)
hr_dir = os.path.join(os.path.dirname(__file__), 'worldstrat_processed', 'HR_8bit')

dataset = SatelliteSRDataset(
    hr_dir=hr_dir,
    lr_patch_size=64,  # Match training (you trained with 64)
    scale=4
)

# Same split as train_swinir.py and eval_bicubic.py
indices = list(range(len(dataset)))
_, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

val_loader = DataLoader(
    Subset(dataset, val_idx),
    batch_size=1,
    shuffle=False
)

# Calculate metrics while generating images
psnr_bicubic = 0
psnr_swinir = 0
ssim_bicubic = 0
ssim_swinir = 0

# Generate comparisons
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

sample_count = 0
with torch.no_grad():
    for i, (lr, hr) in enumerate(val_loader):
        if sample_count >= 3:
            break
        
        lr, hr = lr.to(device), hr.to(device)
        
        # Generate SR and bicubic
        sr = model(lr)
        bicubic = F.interpolate(lr, scale_factor=4, mode='bicubic', align_corners=False)
        
        # Calculate metrics
        psnr_bicubic += calculate_psnr(bicubic, hr)
        psnr_swinir += calculate_psnr(sr, hr)
        ssim_bicubic += calculate_ssim(bicubic, hr)
        ssim_swinir += calculate_ssim(sr, hr)
        
        # Convert to numpy for visualization
        lr_np = lr[0].cpu().permute(1, 2, 0).numpy()
        bicubic_np = bicubic[0].cpu().permute(1, 2, 0).numpy()
        sr_np = sr[0].cpu().permute(1, 2, 0).numpy()
        hr_np = hr[0].cpu().permute(1, 2, 0).numpy()
        
        # Plot row
        axes[sample_count, 0].imshow(lr_np)
        axes[sample_count, 0].set_title(f'LR Input ({lr_np.shape[0]}×{lr_np.shape[1]})')
        axes[sample_count, 0].axis('off')
        
        axes[sample_count, 1].imshow(bicubic_np)
        axes[sample_count, 1].set_title('Bicubic Upscale')
        axes[sample_count, 1].axis('off')
        
        axes[sample_count, 2].imshow(sr_np)
        axes[sample_count, 2].set_title('SwinIR (Ours)')
        axes[sample_count, 2].axis('off')
        
        axes[sample_count, 3].imshow(hr_np)
        axes[sample_count, 3].set_title('Ground Truth HR')
        axes[sample_count, 3].axis('off')
        
        sample_count += 1

# Print metrics
print("\n" + "="*60)
print("COMPARISON METRICS (on these 3 samples)")
print("="*60)
print(f"Bicubic - PSNR: {psnr_bicubic/sample_count:.2f} dB, SSIM: {ssim_bicubic/sample_count:.4f}")
print(f"SwinIR  - PSNR: {psnr_swinir/sample_count:.2f} dB, SSIM: {ssim_swinir/sample_count:.4f}")
print(f"Gain    - PSNR: +{(psnr_swinir - psnr_bicubic)/sample_count:.2f} dB")
print("="*60)

plt.suptitle(f'Super-Resolution Comparison (64×64 → 256×256)\n'
             f'SwinIR: {psnr_swinir/sample_count:.2f} dB vs Bicubic: {psnr_bicubic/sample_count:.2f} dB',
             fontsize=16)
plt.tight_layout()
plt.savefig('results/comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved results/comparison.png")