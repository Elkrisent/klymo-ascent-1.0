import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))
SWINIR_MODELS = os.path.join(WORKSPACE_ROOT, 'SwinIR', 'models')

for _p in (PROJECT_ROOT, WORKSPACE_ROOT, SWINIR_MODELS):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

try:
    from SwinIR.models.network_swinir import SwinIR
except ModuleNotFoundError:
    from network_swinir import SwinIR


class SatelliteSwinIR(nn.Module):
    """
    SwinIR adapted for satellite imagery.
    """
    def __init__(self, scale=4):
        super().__init__()

        self.model = SwinIR(
            upscale=scale,
            in_chans=3,
            img_size=64,          # ✅ MUST match patch_size
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv'
        )

    def forward(self, x):
        return self.model(x)


# --- Quick smoke test ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # pretrained .pth lives inside SwinIR/model_zoo/
    pretrained = os.path.join(_THIS_DIR, '..', 'SwinIR', 'model_zoo',
                              '001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth')

    model = SatelliteSwinIR(scale=4, pretrained_path=pretrained).to(device)
    lr = torch.rand(1, 3, 128, 128).to(device)
    sr = model(lr)
    print(f"✅ SwinIR output shape: {sr.shape}")  # expect [1, 3, 512, 512]
