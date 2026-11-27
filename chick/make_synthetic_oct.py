#!/usr/bin/env python3
"""Write synthetic OCT-like test images for a smoke test (no real data).

The images are smooth horizontal bands with speckle noise. They only exercise
the input checks and the full pipeline; they are not anatomically realistic,
and predictions on them mean nothing.

    python make_synthetic_oct.py --output-dir synthetic
      -> synthetic/synthetic_crop.tif   300 x 1000 uint8, for the default mode
      -> synthetic/synthetic_frame.tif 1024 x 1000 uint8, for --auto-crop
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def synthetic_bscan(height: int, width: int, retina_row: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.arange(width)
    sag = 25 * np.sin(np.pi * x / width)  # gentle curvature
    y = np.arange(height)[:, None]
    image = np.full((height, width), 20.0)
    # (offset above the bright outer band, thickness, brightness)
    bands = [(-170, 10, 150), (-155, 25, 90), (-120, 20, 110), (-95, 18, 70), (-72, 10, 105),
             (-60, 45, 45), (-12, 4, 170), (-4, 6, 200), (4, 8, 150), (14, 12, 215)]
    for offset, thickness, brightness in bands:
        top = retina_row + offset - sag[None, :]
        image += brightness * np.exp(-0.5 * ((y - top - thickness / 2) / (thickness / 2.5)) ** 2)
    image *= rng.gamma(shape=4.0, scale=0.25, size=image.shape)  # speckle
    return np.clip(image, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output-dir", type=Path, default=Path("synthetic"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(synthetic_bscan(300, 1000, 212, args.seed)).save(args.output_dir / "synthetic_crop.tif")
    Image.fromarray(synthetic_bscan(1024, 1000, 420, args.seed + 1)).save(args.output_dir / "synthetic_frame.tif")
    print(f"wrote {args.output_dir}/synthetic_crop.tif and {args.output_dir}/synthetic_frame.tif")


if __name__ == "__main__":
    main()
