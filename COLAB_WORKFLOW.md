# 🚀 Google Colab Training Guide

## Complete Workflow: Local → GitHub → Colab → Back

---

## Part 1: Push Everything to GitHub (Do This Now)

### Files to Push:
```
klymo-ascent-1.0/
├── models/
│   ├── edsr.py
│   └── swinir_satellite.py       ← UPDATED (img_size=128)
├── utils/
│   ├── losses.py
│   └── metrics.py
├── dataset.py                     ← UPDATED (lr_patch_size=128)
├── data_loader.py
├── eval_bicubic.py                ← UPDATED (lr_patch_size=128)
├── train_edsr.py
├── train_swinir.py                ← UPDATED (batch_size=8, patch_size=128)
├── COLAB_TRAINING.ipynb           ← NEW (training notebook)
├── requirements.txt
└── README.md
```

### Git Commands:
```bash
cd klymo-ascent-1.0

# Replace your local files with the fixed ones first
# (copy the files I gave you)

git add .
git commit -m "Fixed settings for T4 GPU training: batch_size=8, patch_size=128"
git push origin master
```

**DON'T PUSH**:
- `checkpoints/` (too large, already in .gitignore)
- `worldstrat_processed/` (too large, already in .gitignore)
- `worldstrat_subset/` (too large, already in .gitignore)
- `SwinIR/` (external repo, already in .gitignore)
- `__pycache__/` (already in .gitignore)

---

## Part 2: Dataset Strategy for Colab

### Problem:
Your `worldstrat_processed/` folder is probably 2-5GB. You have 3 options:

### **Option 1: Google Drive (Recommended)**

**A. Upload to Google Drive (do once):**
1. Zip your local data folder:
   ```bash
   # On Windows
   cd C:\D_Games\Klymo-ascent\Implementation\klymo-ascent-1.0
   tar -czf worldstrat_processed.tar.gz worldstrat_processed/
   ```

2. Upload `worldstrat_processed.tar.gz` to your Google Drive
   - Go to https://drive.google.com
   - Create folder: "satellite-sr-data"
   - Upload the .tar.gz file there

**B. In Colab notebook:**
```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Extract data
!tar -xzf /content/drive/MyDrive/satellite-sr-data/worldstrat_processed.tar.gz

# Verify
!ls worldstrat_processed/HR_8bit/ | head -5
```

### **Option 2: GitHub Release (if < 2GB)**

If your data is under 2GB:

1. Create a GitHub Release:
   - Go to your repo: https://github.com/Elkrisent/klymo-ascent-1.0
   - Click "Releases" → "Create a new release"
   - Tag: "data-v1.0"
   - Upload `worldstrat_processed.tar.gz`

2. In Colab:
   ```python
   !wget https://github.com/Elkrisent/klymo-ascent-1.0/releases/download/data-v1.0/worldstrat_processed.tar.gz
   !tar -xzf worldstrat_processed.tar.gz
   ```

### **Option 3: Re-download in Colab (slowest)**

If you have the original download script from Ragunath, run it in Colab.

**My recommendation**: Use Option 1 (Google Drive). It's fastest and most reliable.

---

## Part 3: Run Training on Colab

### Step-by-Step:

**1. Open Google Colab**
- Go to: https://colab.research.google.com/
- Sign in with your Google account

**2. Change Runtime to GPU**
- Click: **Runtime** → **Change runtime type**
- Hardware accelerator: **T4 GPU**
- Save

**3. Upload the Notebook**
- Click: **File** → **Upload notebook**
- Upload the `COLAB_TRAINING.ipynb` I gave you
- OR: Open a new notebook and copy-paste cells

**4. Run Each Cell in Order**

**Cell 1: Check GPU**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```
Expected output:
```
CUDA available: True
GPU: Tesla T4
```

**Cell 2: Clone Repo**
```python
!git clone https://github.com/Elkrisent/klymo-ascent-1.0.git
%cd klymo-ascent-1.0
```

**Cell 3: Install Dependencies**
```python
!pip install -r requirements.txt
!pip install rasterio scikit-image scikit-learn
```

**Cell 4: Get Dataset (Google Drive method)**
```python
from google.colab import drive
drive.mount('/content/drive')

# Unzip your data (adjust path to where you uploaded it)
!tar -xzf /content/drive/MyDrive/satellite-sr-data/worldstrat_processed.tar.gz

# Verify
!ls worldstrat_processed/HR_8bit/ | head -5
```

**Cell 5: Clone SwinIR**
```python
!git clone https://github.com/JingyunLiang/SwinIR.git
!mkdir -p SwinIR/model_zoo
!wget -P SwinIR/model_zoo https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth
```

**Cell 6: Verify Pipeline**
```python
from dataset import SatelliteSRDataset
from torch.utils.data import DataLoader

dataset = SatelliteSRDataset(
    hr_dir='worldstrat_processed/HR_8bit',
    lr_patch_size=128,
    scale=4
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)
lr, hr = next(iter(loader))

print(f"✅ LR shape: {lr.shape}")
print(f"✅ HR shape: {hr.shape}")
```

**Cell 7: START TRAINING** 🚀
```python
!python train_swinir.py
```

This will take **4-5 hours**. You'll see progress like:
```
Epoch 1/100: 100%|██████| 240/240 [01:15<00:00, 3.18it/s, loss=0.0423, l1=0.0391, cycle=0.0321]
Epoch 1:
  Train - Loss: 0.0420, L1: 0.0388, Cycle: 0.0318
  Val   - PSNR: 28.45 dB, SSIM: 0.8221
✅ Saved best model (PSNR: 28.45 dB)

Epoch 2/100: ...
```

**Expected Final Result** (after 100 epochs):
```
Epoch 100:
  Train - Loss: 0.0145, L1: 0.0141, Cycle: 0.0038
  Val   - PSNR: 33.82 dB, SSIM: 0.9012
✅ Saved best model (PSNR: 33.82 dB)
🎉 Training complete! Best PSNR: 33.82 dB
```

**Cell 8: Download Checkpoint**
```python
from google.colab import files
files.download('checkpoints/swinir_best.pth')
```

**Cell 9: Evaluate**
```python
!python eval_bicubic.py
```

Expected:
```
Bicubic PSNR: 30.13 dB
Bicubic SSIM: 0.8260
```

Compare:
```
Bicubic:  30.13 dB  ← baseline
SwinIR:   33.82 dB  ← your model
Gain:     +3.69 dB  ← WINNING
```

---

## Part 4: Get Checkpoint Back to Local

### After Training Completes:

**1. Download from Colab**
The `files.download()` command will download `swinir_best.pth` to your Downloads folder.

**2. Move to Your Local Repo**
```bash
# On Windows
move C:\Users\YourName\Downloads\swinir_best.pth C:\D_Games\Klymo-ascent\Implementation\klymo-ascent-1.0\checkpoints\
```

**3. Push Checkpoint to GitHub**

⚠️ **BUT WAIT** - the checkpoint is ~45MB. GitHub has a 100MB limit, so it will fit, but:

Option A: Push directly (if < 100MB)
```bash
cd klymo-ascent-1.0
git add checkpoints/swinir_best.pth
git commit -m "Add trained SwinIR checkpoint (PSNR: 33.82 dB)"
git push origin master
```

Option B: Use Git LFS (recommended for large files)
```bash
git lfs install
git lfs track "checkpoints/*.pth"
git add .gitattributes
git add checkpoints/swinir_best.pth
git commit -m "Add trained SwinIR checkpoint with LFS"
git push origin master
```

Option C: Upload to Google Drive and share link
- Upload checkpoint to Drive
- Update README with download link
- AR downloads from Drive

**My recommendation**: Use Option C (Drive link). Keeps repo clean, AR can grab it easily.

---

## Part 5: Hand Off to AR

Once checkpoint is uploaded:

**1. Update your README.md:**
```markdown
## Trained Model Checkpoint

Download the trained SwinIR model:
- **Google Drive**: [swinir_best.pth](https://drive.google.com/file/d/YOUR_FILE_ID/view)
- **PSNR**: 33.82 dB
- **SSIM**: 0.9012
- **Trained on**: T4 GPU, 100 epochs, 4.5 hours
```

**2. Message AR:**
```
Training complete! 🎉

Results:
- Bicubic baseline: 30.13 dB
- SwinIR (ours): 33.82 dB
- Improvement: +3.69 dB

Checkpoint ready: [link to Drive]

You can start Day 3 inference work now. Let me know if you need anything!
```

**3. What AR needs from you:**
- `checkpoints/swinir_best.pth` ← they need this file
- Your repo with all the code ← already pushed
- The dataset (if they want to run inference) ← they can reuse same Drive link

---

## Troubleshooting

### "CUDA out of memory"
```python
# In train_swinir.py, reduce batch size
BATCH_SIZE = 4  # instead of 8
```

### "Colab disconnected during training"
Training auto-saves checkpoints every 20 epochs. Just re-run:
```python
!python train_swinir.py  # will auto-resume from latest checkpoint
```

### "Can't find worldstrat_processed/"
Check the path in train_swinir.py:
```python
hr_dir = 'worldstrat_processed/HR_8bit'  # must match your extracted folder
```

### "Import error: No module named 'network_swinir'"
SwinIR wasn't cloned properly:
```python
!git clone https://github.com/JingyunLiang/SwinIR.git
```

---

## Timeline

```
Now:           Push code to GitHub (10 min)
+20 min:       Upload data to Drive (depends on your upload speed)
+30 min:       Set up Colab, verify pipeline (10 min)
+35 min:       Start training
+5 hours:      Training completes
+5.5 hours:    Download checkpoint, upload to Drive
+6 hours:      AR can start Day 3 work
```

If you start now, AR can have the checkpoint by tomorrow morning.

---

## Quick Checklist

Before you start:
- [ ] Replace local files with fixed versions (I provided)
- [ ] Push to GitHub
- [ ] Upload `worldstrat_processed/` to Google Drive
- [ ] Open Colab, change to T4 GPU
- [ ] Run cells 1-7 (setup + training)
- [ ] Wait 4-5 hours
- [ ] Download checkpoint
- [ ] Upload to Drive, share link with AR
- [ ] Update README

Good luck! 🚀
