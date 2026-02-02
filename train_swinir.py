import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models.swinir_satellite import SatelliteSwinIR
from dataset import SatelliteSRDataset
from utils.losses import CombinedLoss
from utils.metrics import calculate_psnr, calculate_ssim

# Config - FULL SETTINGS FOR T4 GPU (16GB)
BATCH_SIZE = 8
LR = 2e-4
EPOCHS = 100
SCALE = 4
LAMBDA_CYCLE = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Clear GPU cache at startup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- Data (same split as train_edsr / eval_bicubic) ---
# On Colab: hr_dir should point to wherever you upload/mount the data
# Default assumes data is in klymo-ascent-1.0/worldstrat_processed/HR_8bit/
hr_dir = os.path.join(os.path.dirname(__file__), 'worldstrat_processed', 'HR_8bit')

# If running on Colab and data is in Google Drive:
# Uncomment these lines and mount Drive first
# from google.colab import drive
# drive.mount('/content/drive')
# hr_dir = '/content/drive/MyDrive/satellite-sr/worldstrat_processed/HR_8bit'

dataset = SatelliteSRDataset(
    hr_dir=hr_dir,
    lr_patch_size=128,  # Full size
    scale=SCALE
)

indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

train_loader = DataLoader(
    Subset(dataset, train_idx),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,  # Colab can handle 2 workers
    pin_memory=True
)

val_loader = DataLoader(
    Subset(dataset, val_idx),
    batch_size=4,
    shuffle=False,
    num_workers=2
)

# --- Model ---
# Pretrained weights live inside the cloned SwinIR repo
pretrained_path = os.path.join('SwinIR', 'model_zoo',
                               '001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth')

model = SatelliteSwinIR(
    scale=SCALE,
    pretrained_path=pretrained_path
).to(device)

# --- Loss + Optimizer ---
criterion = CombinedLoss(scale=SCALE, lambda_cycle=LAMBDA_CYCLE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)

def _get_latest_epoch_checkpoint(ckpt_dir: str):
    """Find the most recent epoch checkpoint for resuming"""
    if not os.path.isdir(ckpt_dir):
        return None
    latest_epoch = -1
    latest_path = None
    for name in os.listdir(ckpt_dir):
        if not (name.startswith('swinir_epoch') and name.endswith('.pth')):
            continue
        num_part = name[len('swinir_epoch'):-len('.pth')]
        if not num_part.isdigit():
            continue
        epoch_num = int(num_part)
        if epoch_num > latest_epoch:
            latest_epoch = epoch_num
            latest_path = os.path.join(ckpt_dir, name)
    return latest_path

# --- Resume (auto) ---
start_epoch = 0
best_psnr = 0
latest_ckpt = _get_latest_epoch_checkpoint(ckpt_dir)
best_ckpt = os.path.join(ckpt_dir, 'swinir_best.pth')
resume_path = latest_ckpt if latest_ckpt is not None else (best_ckpt if os.path.isfile(best_ckpt) else None)

if resume_path is not None:
    print(f"Resuming from checkpoint: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if 'psnr' in ckpt:
        best_psnr = ckpt['psnr']
    if 'epoch' in ckpt:
        start_epoch = ckpt['epoch'] + 1
    print(f"Resumed at epoch {start_epoch} (best PSNR: {best_psnr:.2f} dB)")

# --- Training loop ---
for epoch in range(start_epoch, EPOCHS):
    model.train()
    train_loss = 0
    train_l1 = 0
    train_cycle = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for lr_img, hr_img in pbar:
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)

        sr_img = model(lr_img)
        loss, l1, cycle = criterion(sr_img, hr_img, lr_img)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        train_l1 += l1.item()
        train_cycle += cycle.item()

        pbar.set_postfix({
            'loss':  f'{loss.item():.4f}',
            'l1':    f'{l1.item():.4f}',
            'cycle': f'{cycle.item():.4f}'
        })

    # --- Validation ---
    model.eval()
    val_psnr = 0
    val_ssim = 0

    with torch.no_grad():
        for lr_img, hr_img in val_loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            sr_img = model(lr_img)
            val_psnr += calculate_psnr(sr_img, hr_img)
            val_ssim += calculate_ssim(sr_img, hr_img)

    val_psnr /= len(val_loader)
    val_ssim /= len(val_loader)

    print(f"Epoch {epoch+1}:")
    print(f"  Train - Loss: {train_loss/len(train_loader):.4f}, "
          f"L1: {train_l1/len(train_loader):.4f}, "
          f"Cycle: {train_cycle/len(train_loader):.4f}")
    print(f"  Val   - PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f}")

    # Save best
    if val_psnr > best_psnr:
        best_psnr = val_psnr
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'psnr': best_psnr,
        }, os.path.join(ckpt_dir, 'swinir_best.pth'))
        print(f"✅ Saved best model (PSNR: {best_psnr:.2f} dB)")

    # Checkpoint every 20 epochs
    if (epoch + 1) % 20 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'psnr': best_psnr,
        }, os.path.join(ckpt_dir, f'swinir_epoch{epoch+1}.pth'))

    scheduler.step()

print(f"🎉 Training complete! Best PSNR: {best_psnr:.2f} dB")
