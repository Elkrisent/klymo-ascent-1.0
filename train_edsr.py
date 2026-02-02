import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

from models.edsr import EDSR
from dataset import SatelliteSRDataset          # ← correct module
from utils.metrics import calculate_psnr, calculate_ssim

# Config
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 50
SCALE = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Data ---
# HR-only dataset: LR is generated internally by downsampling HR
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
hr_dir = os.path.join(project_root, "worldstrat_processed", "HR_8bit")

dataset = SatelliteSRDataset(
    hr_dir=hr_dir,
    lr_patch_size=128,
    scale=SCALE
)

# 80/20 split by index
indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

train_loader = DataLoader(
    Subset(dataset, train_idx),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0          # 0 on Windows
)

val_loader = DataLoader(
    Subset(dataset, val_idx),
    batch_size=4,
    shuffle=False,
    num_workers=0
)

# --- Model ---
model = EDSR(scale=SCALE, n_resblocks=16, n_feats=64).to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

os.makedirs('checkpoints', exist_ok=True)

# --- Training loop ---
best_psnr = 0
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for lr_img, hr_img in pbar:
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)

        sr_img = model(lr_img)
        loss = criterion(sr_img, hr_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # --- Validation ---
    model.eval()
    val_psnr = 0
    with torch.no_grad():
        for lr_img, hr_img in val_loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            sr_img = model(lr_img)
            val_psnr += calculate_psnr(sr_img, hr_img)

    val_psnr /= len(val_loader)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Val PSNR = {val_psnr:.2f} dB")

    if val_psnr > best_psnr:
        best_psnr = val_psnr
        torch.save(model.state_dict(), 'checkpoints/edsr_best.pth')
        print(f"✅ Saved best model (PSNR: {best_psnr:.2f} dB)")

    scheduler.step()

print(f"🎉 Training complete! Best PSNR: {best_psnr:.2f} dB")
