import os
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)
import torch
import torch.nn as nn
from SwinIR.models.network_swinir import SwinIR


class SatelliteSwinIR(nn.Module):
    """
    SwinIR adapted for satellite imagery.
    """
    def __init__(self, scale=4, pretrained_path=None):
        super().__init__()

        self.model = SwinIR(
            upscale=scale,
            in_chans=3,
            img_size=64,          # Matches your 64×64 patch training
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv'
        )

        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, path):
        print(f"Loading pretrained weights from {path}")
        pretrained_dict = torch.load(path, map_location='cpu')['params']

        model_dict = self.model.state_dict()
        # Only load weights that match in name AND shape
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers")

    def forward(self, x):
        return self.model(x)


# --- Quick smoke test ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # pretrained .pth lives inside SwinIR/model_zoo/
    pretrained = os.path.join(_THIS_DIR, '..', 'SwinIR', 'model_zoo',
                              '001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth')

    model = SatelliteSwinIR(scale=4, pretrained_path=pretrained).to(device)
    lr = torch.rand(1, 3, 64, 64).to(device)  # 64×64 to match img_size
    sr = model(lr)
    print(f"✅ SwinIR output shape: {sr.shape}")  # expect [1, 3, 256, 256]