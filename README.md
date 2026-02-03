
```md
# Processed WorldStrat Dataset (Drive)

This project uses a **processed subset of the WorldStrat dataset**.
Due to size constraints, the dataset is **hosted on Google Drive** and
is not included directly in this repository.

---
Website
🔗 https://klymo-satellite-terrathon.streamlit.app/
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

