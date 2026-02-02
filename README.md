# 🛰️ Satellite Image Super-Resolution with SwinIR

Transform low-resolution Sentinel-2 imagery (10m/pixel) to commercial quality (2.5m/pixel) using deep learning.

![Comparison](results/final_comparison.png)

---

## 🎯 Problem

- **Challenge**: High-resolution satellite imagery is expensive ($$$)
- **Solution**: Use deep learning to upscale free Sentinel-2 data
- **Key Constraint**: No hallucination - must recover real details, not invent them

---

## 🏆 Approach

### Architecture: SwinIR (Transformer-based SR)

- **Why Transformer?** Captures global spatial context (roads, rivers, city blocks)
- **Why SwinIR?** Proven state-of-the-art on image restoration tasks

### Innovation: LR-Consistency Loss

```python
L_total = L1(SR, HR) + λ * L1(Downsample(SR), LR)
```

**How it works**: Penalizes fake structures by checking if downsampled SR matches original LR

---

## 📊 Results

| Method | PSNR (dB) | SSIM | Notes |
|--------|-----------|------|-------|
| Bicubic | 22.3 | 0.751 | Baseline |
| EDSR (CNN) | 27.2 | 0.802 | Strong baseline |
| **SwinIR (Ours)** | **28.9** | **0.857** | 🏆 Best |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Elkrisent/klymo-ascent-1.0.git
cd klymo-ascent-1.0
pip install -r requirements.txt
```

### Run Inference

```python
from inference import tile_inference, load_model
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('checkpoints/swinir_best.pth', device)

# Load your LR image (numpy array, H×W×3, uint8)
import numpy as np
from PIL import Image
lr_image = np.array(Image.open('path/to/lr_image.png'))

# Run super-resolution
sr_image = tile_inference(lr_image, model, device, tile_size=128, overlap=16)

# Save result
Image.fromarray(sr_image).save('sr_output.png')
```

### Run Interactive Demo

```bash
streamlit run app.py
```

### Train Your Own Model

```bash
# Local training
python train_swinir.py --epochs 100 --batch_size 8

# Or use Google Colab (recommended)
# Open COLAB_TRAINING.ipynb in Colab
```

---

## 📁 Project Structure

```
klymo-ascent-1.0/
├── data/                      # Datasets
│   ├── worldstrat_processed/  # Training data (from Drive)
│   └── mystery_location/      # Test images
├── models/                    # Model architectures
│   ├── swinir_satellite.py    # SwinIR implementation
│   └── edsr.py                # EDSR baseline
├── utils/                     # Utilities
│   ├── losses.py              # Custom loss functions
│   └── metrics.py             # PSNR, SSIM
├── checkpoints/               # Trained model weights
│   └── swinir_best.pth        # Best SwinIR checkpoint
├── results/                   # Output images
├── train_swinir.py            # Training script
├── inference.py               # Inference pipeline
├── app.py                     # Streamlit demo
├── COLAB_TRAINING.ipynb       # Colab training notebook
└── TEST_IMAGE.ipynb           # Colab inference notebook
```

---

## 🎓 Technical Details

### SwinIR Architecture

- **Base**: Swin Transformer with shifted windows
- **Depth**: 6 Residual Swin Transformer Blocks (RSTBs)
- **Embedding Dim**: 180
- **Window Size**: 8×8
- **Parameters**: 11.1M
- **Scale**: 4x upsampling

### Training Configuration

- **Dataset**: WorldStrat (500 LR/HR pairs)
- **GPU**: Google Colab T4 (16GB VRAM)
- **Training Time**: ~5 hours (100 epochs)
- **Loss Function**: L1 + Cycle Consistency (λ=0.1)
- **Optimizer**: AdamW (lr=2e-4, weight_decay=1e-4)
- **Batch Size**: 8
- **Patch Size**: 128×128 (LR) → 512×512 (HR)
- **Data Augmentation**: Random flip, rotation

### Hallucination Guardrail

We add a cycle-consistency constraint to prevent fake structures:

```python
# Standard SR loss
loss_sr = L1(SR, HR)

# Cycle consistency loss
LR_reconstructed = Downsample(SR)
loss_cycle = L1(LR_reconstructed, LR)

# Total loss
total_loss = loss_sr + 0.1 * loss_cycle
```

This ensures every pixel in the super-resolved image is grounded in the original satellite signal.

### Inference Optimizations

- **Tiled Processing**: Handles large images (e.g., 2000×2000) without OOM
- **Overlap Blending**: 16-pixel overlap with feathering for seamless stitching
- **GPU Acceleration**: ~10x faster than CPU
- **Memory Efficient**: Processes tiles sequentially

---

## 📝 Notebooks

### COLAB_TRAINING.ipynb
Full training pipeline for Google Colab:
- Environment setup with T4 GPU
- Dataset download from Google Drive
- Training with real-time monitoring
- Checkpoint saving

### TEST_IMAGE.ipynb
Inference demo for judges:
- Load pre-trained model
- Process test images
- Visualize results with comparisons
- Export super-resolved outputs

---

## 🔬 Key Papers

- **SwinIR**: [Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257) (Liang et al., ICCV 2021)
- **Swin Transformer**: [Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) (Liu et al., ICCV 2021)
- **Cycle Consistency**: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) (Zhu et al., ICCV 2017)

---

## 👥 Team

- **Yatin**: Data Pipeline & Google Earth Engine Integration
- **Ragunath**: Model Implementation & Training
- **Aritra**: Inference Pipeline, Demo & Presentation

**Hackathon**: Klymo Ascent ML Track 2026

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- WorldStrat dataset creators
- SwinIR paper authors
- Google Earth Engine
- Google Colab for free GPU access

---

# Processed WorldStrat Dataset (Drive)

This project uses a **processed subset of the WorldStrat dataset**.
Due to size constraints, the dataset is **hosted on Google Drive** and
is not included directly in this repository.

---

## 📍 Dataset Location

Google Drive link:  
🔗 https://drive.google.com/drive/folders/1MTEk4ykc96BMYVdlneSBiDUhwNsYdVmq?usp=sharing

Download the ZIP file:
```

worldstrat_processed.zip

```

---

## 📦 Dataset Contents

After extracting the ZIP, the folder structure is:

```

worldstrat_processed/
├── HR_8bit/
│   └── *.tif
└── LR_8bit/
    └── *.tif

```

### HR_8bit
- High-resolution satellite images
- 3-band RGB
- Data type: `uint8`
- Value range: `[0, 255]`
- **Used as ground truth for training**

### LR_8bit
- Low-resolution satellite images from WorldStrat
- Provided for reference / inspection only
- **Not pixel-aligned with HR images**
- Not used directly for training

---

## 🧠 How the Dataset Is Used

Training uses an **HR-first super-resolution pipeline**.

- HR images are treated as ground truth
- LR inputs are generated **on-the-fly** by downsampling HR patches
- This avoids alignment issues present in WorldStrat LR–HR pairs

The logic is implemented in `dataset.py`.

---

## 🚀 Usage Instructions

1. Download `worldstrat_processed.zip` from Drive
2. Extract it into your project so you have:
```

data/worldstrat_processed/

````
3. Use the dataset loader:

```python
from dataset import SatelliteSRDataset
from torch.utils.data import DataLoader

dataset = SatelliteSRDataset(
 hr_dir="data/worldstrat_processed/HR_8bit",
 lr_patch_size=128,
 scale=4
)

loader = DataLoader(dataset, batch_size=8, shuffle=True)
````

Expected output:

* LR batch: `[B, 3, 128, 128]`
* HR batch: `[B, 3, 512, 512]`

---

## 📝 Notes

* Do not manually pair LR and HR images
* Do not modify image files
* All preprocessing is already completed

If changes are required, update both the dataset and `dataset.py` consistently.

```

