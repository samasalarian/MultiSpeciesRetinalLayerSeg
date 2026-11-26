#!/usr/bin/env python
"""Generate a SYNTHETIC 300x1000 OCT-like crop for a quick installation / smoke test.

    python make_smoke_test.py            # writes smoke_test/synthetic_crop.png and smoke_test/synthetic_crop.tif
    python make_smoke_test.py --check smoke_test_output/synthetic_crop   # verifies the outputs of predict.py

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


def check(result_dir):
    """Verify the outputs of `python predict.py smoke_test/synthetic_crop.png --out smoke_test_output`."""
    d = Path(result_dir)
    src = make()
    problems = []

    def need(ok, msg):
        if not ok:
            problems.append(msg)

    for f in ("mask.tif", "boundaries.csv", "boundaries.npy", "info.json", "layers_overlay.tif",
              "boundaries_overlay.tif", "overlay.png"):
        need((d / f).is_file(), f"missing {f}")
    if problems:
        return problems
    m = Image.open(d / "mask.tif")
    need((m.format, m.mode, m.size, getattr(m, "n_frames", 1)) == ("TIFF", "L", (W, H), 1),
         f"mask.tif: {m.format} {m.mode} {m.size}")
    mask = np.array(m)
    keep = mask[0] != 255                                          # columns not excluded as ONH
    need(set(np.unique(mask[:, keep]).tolist()) <= set(range(10)), "mask labels outside 0-9")
    b = np.load(d / "boundaries.npy")
    need(b.shape == (10, W) and np.isfinite(b[:, keep]).all() and (np.diff(b[:, keep], axis=0) >= 0).all(),
         "boundaries not finite / ordered")
    for name in ("layers_overlay.tif", "boundaries_overlay.tif"):
        im = Image.open(d / name)
        need((im.format, im.mode, im.size, getattr(im, "n_frames", 1)) == ("TIFF", "RGB", (W, H), 1),
             f"{name}: {im.format} {im.mode} {im.size} frames={getattr(im, 'n_frames', 1)} (expected TIFF RGB 1000x300, 1 page)")
        a = np.array(im)
        need(a.dtype == np.uint8, f"{name}: dtype {a.dtype}")
        changed = (a != src).any(-1)
        if name == "layers_overlay.tif":
            need(not changed[(mask == 0) | (mask == 255)].any(), f"{name}: OCT altered outside the layers")
            need(changed[(mask >= 1) & (mask <= 9)].mean() > 0.9, f"{name}: layers not coloured")
        else:
            need(changed.mean() < 0.05, f"{name}: too many pixels changed ({changed.mean():.1%})")
            need(not changed[:, ~keep].any(), f"{name}: boundaries drawn in excluded ONH columns")
    return problems


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        probs = check(sys.argv[2])
        print("smoke test PASSED" if not probs else "smoke test FAILED:\n  " + "\n  ".join(probs))
        sys.exit(1 if probs else 0)
    out = Path(__file__).resolve().parent / "smoke_test"
    out.mkdir(exist_ok=True)
    rgb = make()
    Image.fromarray(rgb).save(out / "synthetic_crop.png")
    Image.fromarray(rgb).save(out / "synthetic_crop.tif")
    print(f"wrote {out / 'synthetic_crop.png'} and {out / 'synthetic_crop.tif'} ({H}x{W}, 8-bit RGB, synthetic)")
