import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))

for _p in (PROJECT_ROOT,):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

try:
    from SwinIR.models.network_swinir import SwinIR
except (ModuleNotFoundError, ImportError):
    try:
        from network_swinir import SwinIR
    except (ModuleNotFoundError, ImportError):
        print("⚠️  Warning: SwinIR not found. Using minimal stub for testing.")
        SwinIR = None


class SimpleUpsampleStub(nn.Module):
    """Fallback upsampling when SwinIR is unavailable."""

    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bicubic-like upsampling via interpolation
        return torch.nn.functional.interpolate(
            x, scale_factor=self.scale, mode="bicubic", align_corners=False
        )


class SatelliteSwinIR(nn.Module):
    """
    SwinIR adapted for satellite imagery.
    Falls back to simple upsampling if SwinIR is not available.
    """

    def __init__(self, scale: int = 4):
        super().__init__()

        if SwinIR is not None:
            self.model = SwinIR(
                upscale=scale,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler="pixelshuffle",
                resi_connection="1conv",
            )
        else:
            self.model = SimpleUpsampleStub(scale=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# --- Quick smoke test ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SatelliteSwinIR(scale=4).to(device)
    lr = torch.rand(1, 3, 128, 128).to(device)
    sr = model(lr)
    print(f"✅ SatelliteSwinIR output shape: {sr.shape}")  # expect [1, 3, 512, 512]
