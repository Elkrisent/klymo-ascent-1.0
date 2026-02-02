import argparse
import os
from typing import Optional, Tuple

import numpy as np
import torch

try:
    import rasterio
    from rasterio.plot import reshape_as_image
    from rasterio.transform import Affine
except Exception as exc:  # pragma: no cover - runtime dependency
    raise ImportError("rasterio is required for GeoTIFF inference") from exc

from models.swinir_satellite import SatelliteSwinIR


def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def load_model(checkpoint_path: str, device: torch.device) -> SatelliteSwinIR:
    model = SatelliteSwinIR(scale=4).to(device)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at: {checkpoint_path}. "
            "Place swinir_best.pth in the 'checkpoints' folder or pass --ckpt."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict) or len(state) == 0:
        raise ValueError(
            "Checkpoint 'model_state_dict' is empty. "
            "Provide a valid SwinIR checkpoint with weights."
        )
    model.load_state_dict(state)
    model.eval()

    psnr = checkpoint.get("psnr")
    if psnr is not None:
        print(f"✅ Model loaded (PSNR: {psnr:.2f} dB)")
    else:
        print("✅ Model loaded")

    return model


def inference_single_patch(
    lr_image: np.ndarray, model: torch.nn.Module, device: torch.device
) -> np.ndarray:
    """
    Run inference on a single LR patch.

    Args:
        lr_image: numpy array (H, W, C) in [0, 255]
        model: trained SwinIR
        device: cuda or cpu

    Returns:
        SR image (H*4, W*4, C) in [0, 255]
    """
    if lr_image.ndim != 3 or lr_image.shape[2] != 3:
        raise ValueError("Expected lr_image as HxWx3 array")

    lr_tensor = (
        torch.from_numpy(lr_image).permute(2, 0, 1).float() / 255.0
    )  # (C, H, W)
    lr_tensor = lr_tensor.unsqueeze(0).to(device)  # (1, C, H, W)

    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    sr_image = (
        sr_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    )  # (H*4, W*4, C)
    sr_image = np.clip(sr_image * 255.0, 0, 255).astype(np.uint8)

    return sr_image


def tile_inference(
    large_lr_image: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    tile_size: int = 128,
    overlap: int = 16,
) -> np.ndarray:
    """
    Process large image by tiling with overlap + feathering.

    Args:
        large_lr_image: numpy array (H, W, C) in [0, 255]
        model: trained SwinIR
        device: cuda or cpu
        tile_size: size of each LR tile (default 128)
        overlap: overlap between tiles for seamless stitching (default 16)

    Returns:
        SR image (H*4, W*4, C) in [0, 255]
    """
    if large_lr_image.ndim != 3 or large_lr_image.shape[2] != 3:
        raise ValueError("Expected large_lr_image as HxWx3 array")
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be in [0, tile_size)")

    h, w, c = large_lr_image.shape
    scale = 4

    sr_h, sr_w = h * scale, w * scale
    sr_canvas = np.zeros((sr_h, sr_w, c), dtype=np.float32)
    weight_map = np.zeros((sr_h, sr_w), dtype=np.float32)

    stride = tile_size - overlap

    print(f"Processing {h}x{w} image in tiles of {tile_size}x{tile_size}...")

    tile_count = 0
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            lr_tile = large_lr_image[y:y + tile_size, x:x + tile_size, :]

            sr_tile = inference_single_patch(lr_tile, model, device)

            y_sr = y * scale
            x_sr = x * scale
            tile_h_sr = tile_size * scale
            tile_w_sr = tile_size * scale

            weight = np.ones((tile_h_sr, tile_w_sr), dtype=np.float32)
            if overlap > 0:
                fade_size = overlap * scale
                fade = np.linspace(0, 1, fade_size, dtype=np.float32)
                weight[:fade_size, :] *= fade[:, np.newaxis]
                weight[-fade_size:, :] *= fade[::-1, np.newaxis]
                weight[:, :fade_size] *= fade[np.newaxis, :]
                weight[:, -fade_size:] *= fade[::-1][np.newaxis, :]

            sr_canvas[y_sr:y_sr + tile_h_sr, x_sr:x_sr + tile_w_sr, :] += (
                sr_tile.astype(np.float32) * weight[:, :, np.newaxis]
            )
            weight_map[y_sr:y_sr + tile_h_sr, x_sr:x_sr + tile_w_sr] += weight

            tile_count += 1
            if tile_count % 10 == 0:
                print(f"  Processed {tile_count} tiles...")

    sr_canvas = sr_canvas / (weight_map[:, :, np.newaxis] + 1e-8)
    sr_canvas = np.clip(sr_canvas, 0, 255).astype(np.uint8)

    print(f"✅ Tiling complete ({tile_count} tiles)")
    return sr_canvas


def load_tif(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        img = src.read()
        img = reshape_as_image(img)
    return img


def load_tif_with_meta(path: str) -> Tuple[np.ndarray, dict, "rasterio.Affine"]:
    with rasterio.open(path) as src:
        img = reshape_as_image(src.read())
        meta = src.meta.copy()
        transform = src.transform
    return img, meta, transform


def normalize_16bit_to_8bit(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img_float = img.astype(np.float32)
    min_val = float(np.min(img_float))
    max_val = float(np.max(img_float))
    if max_val <= min_val:
        return np.zeros_like(img_float, dtype=np.uint8)
    scaled = (img_float - min_val) / (max_val - min_val)
    scaled = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    return scaled


def crop_patch(img: np.ndarray, size: int = 128) -> np.ndarray:
    h, w, _ = img.shape
    if h < size or w < size:
        raise ValueError(f"Image too small for {size}x{size} crop: {img.shape}")
    return img[:size, :size, :]


def save_preview(lr: np.ndarray, sr: np.ndarray, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise ImportError("matplotlib is required to save preview") from exc

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(lr)
    axes[0].set_title("LR Input")
    axes[0].axis("off")

    axes[1].imshow(sr)
    axes[1].set_title("SR Output")
    axes[1].axis("off")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def save_png(img: np.ndarray, out_path: str) -> None:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise ImportError("Pillow is required to save PNGs") from exc
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(img).save(out_path)


def bicubic_upscale(lr_img: np.ndarray, scale: int = 4) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise ImportError("opencv-python is required for bicubic upscaling") from exc

    h, w = lr_img.shape[:2]
    return cv2.resize(lr_img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def create_comparison_grid(
    lr_img: np.ndarray,
    sr_img: np.ndarray,
    zoom_coords: Optional[Tuple[int, int, int, int]] = None,
    out_path: Optional[str] = None,
) -> np.ndarray:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise ImportError("matplotlib is required to create comparison grid") from exc

    lr_bicubic = bicubic_upscale(lr_img, scale=4)

    fig, axes = plt.subplots(1, 3 if zoom_coords else 2, figsize=(24, 8))
    axes = np.atleast_1d(axes)

    axes[0].imshow(lr_bicubic)
    axes[0].set_title("Bicubic Upscaling", fontsize=18)
    axes[0].axis("off")

    axes[1].imshow(sr_img)
    axes[1].set_title("SwinIR (Ours)", fontsize=18)
    axes[1].axis("off")

    if zoom_coords:
        y, x, h, w = zoom_coords
        y_sr, x_sr = y * 4, x * 4
        h_sr, w_sr = h * 4, w * 4

        lr_zoom = lr_bicubic[y_sr:y_sr + h_sr, x_sr:x_sr + w_sr, :]
        sr_zoom = sr_img[y_sr:y_sr + h_sr, x_sr:x_sr + w_sr, :]
        zoom_comparison = np.hstack([lr_zoom, sr_zoom])

        axes[2].imshow(zoom_comparison)
        axes[2].set_title("Zoom: Bicubic vs SwinIR", fontsize=18)
        axes[2].axis("off")

    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return lr_bicubic


def parse_args() -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Run SwinIR inference")
    parser.add_argument(
        "--input",
        default=os.path.join(root, "data", "worldstrat", "LR_8bit", "sample_001.tif"),
        help="Path to input GeoTIFF",
    )
    parser.add_argument(
        "--ckpt",
        default=os.path.join(root, "checkpoints", "swinir_best.pth"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(root, "results", "test_inference.png"),
        help="Path to output preview image",
    )
    parser.add_argument(
        "--crop",
        type=int,
        default=128,
        help="Crop size for LR patch",
    )
    parser.add_argument(
        "--run-mystery",
        action="store_true",
        help="Run tiled inference on a mystery location GeoTIFF",
    )
    parser.add_argument(
        "--mystery-input",
        default=os.path.join(root, "data", "mystery_location", "kanpur.tif"),
        help="Path to mystery GeoTIFF",
    )
    parser.add_argument(
        "--mystery-out-tif",
        default=os.path.join(root, "results", "mystery_sr.tif"),
        help="Path to SR GeoTIFF output",
    )
    parser.add_argument(
        "--mystery-out-lr",
        default=os.path.join(root, "results", "mystery_lr.png"),
        help="Path to LR PNG output",
    )
    parser.add_argument(
        "--mystery-out-sr",
        default=os.path.join(root, "results", "mystery_sr.png"),
        help="Path to SR PNG output",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=128,
        help="Tile size for tiled inference",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=16,
        help="Tile overlap for tiled inference",
    )
    parser.add_argument(
        "--make-comparison",
        action="store_true",
        help="Generate bicubic baseline and comparison grid",
    )
    parser.add_argument(
        "--comparison-out",
        default=os.path.join(root, "results", "final_comparison.png"),
        help="Path to comparison grid output",
    )
    parser.add_argument(
        "--bicubic-out",
        default=os.path.join(root, "results", "mystery_bicubic.png"),
        help="Path to bicubic baseline output",
    )
    parser.add_argument(
        "--zoom",
        default="100,100,64,64",
        help="Zoom coords as y,x,h,w on LR image (default 100,100,64,64)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    if args.run_mystery:
        mystery_lr, meta, transform = load_tif_with_meta(args.mystery_input)
        print(f"Mystery location shape: {mystery_lr.shape}")

        if mystery_lr.dtype != np.uint8 or mystery_lr.max() > 255:
            print("Converting 16-bit to 8-bit...")
            mystery_lr = normalize_16bit_to_8bit(mystery_lr)

        print("Running super-resolution inference...")
        mystery_sr = tile_inference(
            mystery_lr,
            model,
            device,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )

        print(f"Output shape: {mystery_sr.shape}")

        sr_h, sr_w = mystery_sr.shape[:2]
        meta.update(
            {
                "dtype": "uint8",
                "height": sr_h,
                "width": sr_w,
                "count": 3,
                "transform": transform
                * Affine.scale(
                    (meta["width"] / sr_w),
                    (meta["height"] / sr_h),
                ),
            }
        )

        os.makedirs(os.path.dirname(args.mystery_out_tif), exist_ok=True)
        with rasterio.open(args.mystery_out_tif, "w", **meta) as dst:
            dst.write(mystery_sr.transpose(2, 0, 1))

        save_png(mystery_lr, args.mystery_out_lr)
        save_png(mystery_sr, args.mystery_out_sr)

        if args.make_comparison:
            try:
                zoom_parts = [int(v) for v in args.zoom.split(",")]
                zoom_coords = (
                    zoom_parts[0],
                    zoom_parts[1],
                    zoom_parts[2],
                    zoom_parts[3],
                )
            except Exception:
                raise ValueError("--zoom must be in 'y,x,h,w' format")

            bicubic = create_comparison_grid(
                mystery_lr,
                mystery_sr,
                zoom_coords=zoom_coords,
                out_path=args.comparison_out,
            )
            save_png(bicubic, args.bicubic_out)
            print(f"✅ Saved {args.comparison_out}")
            print(f"✅ Saved {args.bicubic_out}")

        print(f"✅ Saved {args.mystery_out_tif}")
        print(f"✅ Saved {args.mystery_out_lr}")
        print(f"✅ Saved {args.mystery_out_sr}")
        return

    lr_img = load_tif(args.input)
    lr_patch = crop_patch(lr_img, size=args.crop)
    sr_img = inference_single_patch(lr_patch, model, device)

    print(f"Input: {lr_patch.shape}, Output: {sr_img.shape}")
    save_preview(lr_patch, sr_img, args.out)
    print(f"✅ Saved {args.out}")


if __name__ == "__main__":
    main()
