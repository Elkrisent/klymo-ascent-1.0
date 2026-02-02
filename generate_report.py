import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from models.swinir_satellite import SatelliteSwinIR
from dataset import SatelliteSRDataset
from utils.metrics import calculate_psnr, calculate_ssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load SwinIR model ---
model = SatelliteSwinIR(scale=4).to(device)

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
    raise FileNotFoundError("No checkpoint found!")

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# --- Load validation data ---
hr_dir = os.path.join(os.path.dirname(__file__), 'worldstrat_processed', 'HR_8bit')

dataset = SatelliteSRDataset(
    hr_dir=hr_dir,
    lr_patch_size=64,  # Your training patch size
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

# --- Evaluate both methods ---
results = {
    'Bicubic': {'psnr': 0, 'ssim': 0},
    'SwinIR (Ours)': {'psnr': 0, 'ssim': 0}
}

print("\nEvaluating on validation set...")
with torch.no_grad():
    for lr, hr in val_loader:
        lr, hr = lr.to(device), hr.to(device)
        
        # Bicubic baseline
        bicubic = F.interpolate(lr, scale_factor=4, mode='bicubic', align_corners=False)
        results['Bicubic']['psnr'] += calculate_psnr(bicubic, hr)
        results['Bicubic']['ssim'] += calculate_ssim(bicubic, hr)
        
        # SwinIR
        swinir_out = model(lr)
        results['SwinIR (Ours)']['psnr'] += calculate_psnr(swinir_out, hr)
        results['SwinIR (Ours)']['ssim'] += calculate_ssim(swinir_out, hr)

# Average
for method in results:
    results[method]['psnr'] /= len(val_loader)
    results[method]['ssim'] /= len(val_loader)

# Print table
print("\n" + "="*70)
print("FINAL RESULTS - Super-Resolution Performance")
print("="*70)
print(f"{'Method':<20} {'PSNR (dB)':<15} {'SSIM':<15} {'Notes':<20}")
print("-"*70)

bicubic_psnr = results['Bicubic']['psnr']
bicubic_ssim = results['Bicubic']['ssim']
swinir_psnr = results['SwinIR (Ours)']['psnr']
swinir_ssim = results['SwinIR (Ours)']['ssim']

print(f"{'Bicubic':<20} {bicubic_psnr:<15.2f} {bicubic_ssim:<15.4f} {'Baseline':<20}")
print(f"{'SwinIR (Ours)':<20} {swinir_psnr:<15.2f} {swinir_ssim:<15.4f} {'Our Model':<20}")
print("-"*70)
print(f"{'Improvement':<20} {swinir_psnr - bicubic_psnr:<+15.2f} {swinir_ssim - bicubic_ssim:<+15.4f}")
print("="*70)

# Checkpoint info
if 'psnr' in checkpoint:
    print(f"\nCheckpoint validation PSNR: {checkpoint['psnr']:.2f} dB")
if 'epoch' in checkpoint:
    print(f"Training epochs completed: {checkpoint['epoch']}")

print("\nConfiguration:")
print(f"  - Patch size: 64×64 LR → 256×256 SR")
print(f"  - Scale factor: 4×")
print(f"  - Architecture: SwinIR (Transformer-based)")
print(f"  - Hallucination guardrail: LR-Consistency Loss")

# Analysis
if swinir_psnr > bicubic_psnr:
    gain = swinir_psnr - bicubic_psnr
    print(f"\n✅ SUCCESS: Model beats baseline by {gain:.2f} dB")
    if gain >= 1.0:
        print(f"   Strong improvement - judges will be impressed!")
    else:
        print(f"   Modest improvement - could be better with full 128×128 training")
else:
    loss = bicubic_psnr - swinir_psnr
    print(f"\n⚠️  WARNING: Model underperforms baseline by {loss:.2f} dB")
    print(f"   This likely means:")
    print(f"   - Training converged poorly (check training curves)")
    print(f"   - Batch size too small (you used batch_size=1)")
    print(f"   - Patch size too small (64×64 vs intended 128×128)")
    print(f"   → Recommend retraining on Google Colab T4 with full settings")