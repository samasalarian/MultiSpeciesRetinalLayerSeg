#!/usr/bin/env python
"""Generate a SYNTHETIC 300x1000 OCT-like crop for a quick installation / smoke test.

    python make_smoke_test.py            # writes smoke_test/synthetic_crop.png and smoke_test/synthetic_crop.tif

The image is made from horizontal bright/dark bands with gentle curvature and speckle noise, plus a burned-in
yellow scale bar to exercise the coloured-overlay cleanup. It contains no real data. The model's output on it
only shows that the pipeline runs; it is not a meaningful segmentation.
"""
from pathlib import Path

import numpy as np
from PIL import Image

H, W = 300, 1000


def make(seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(W)
    top = 95 + 12 * np.cos((x - 500) / 1000 * np.pi)              # gently curved inner retinal surface
    # (thickness px, brightness) of synthetic bands from top to bottom
    bands = [(10, 0.75), (22, 0.55), (14, 0.35), (8, 0.55), (40, 0.25), (3, 0.6),
             (8, 0.85), (5, 0.4), (10, 0.95), (25, 0.5)]
    img = np.full((H, W), 0.12)
    y = np.arange(H)[:, None]
    edge = top[None, :].copy()
    for t, v in bands:
        img[(y >= edge) & (y < edge + t)] = v
        edge = edge + t
    img = img * rng.gamma(4.0, 0.25, size=(H, W))                  # speckle
    img = np.clip(img * 200 + 25, 0, 255).astype(np.uint8)
    rgb = np.repeat(img[..., None], 3, -1)
    rgb[270:273, 820:930] = (255, 255, 0)                          # synthetic yellow scale bar
    return rgb


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "smoke_test"
    out.mkdir(exist_ok=True)
    rgb = make()
    Image.fromarray(rgb).save(out / "synthetic_crop.png")
    Image.fromarray(rgb).save(out / "synthetic_crop.tif")
    print(f"wrote {out / 'synthetic_crop.png'} and {out / 'synthetic_crop.tif'} ({H}x{W}, 8-bit RGB, synthetic)")
