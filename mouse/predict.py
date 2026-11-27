#!/usr/bin/env python
"""Mouse retinal OCT layer segmentation - final no-choroid model (9 layers L01-L09, 10 ordered boundaries).

Usage:
    python predict.py INPUT [--out OUTPUT_DIR] [--onh-core START END | --onh-csv FILE] [--device auto|cpu|mps|cuda]

INPUT is one image (.png/.tif/.tiff/.jpg/.bmp) or a folder of them. Each image must be a PREPARED crop: a single
8-bit greyscale or RGB image of exactly 300 rows x 1000 columns (raw OCT frames must be cropped first; see the root README.md).
For each image, OUTPUT_DIR/<image name>/ receives:
    mask.tif        uint8 labels: 0 background, 1-9 = L01-L09 (255 = excluded ONH column, only with --onh-*)
    boundaries.npy  float32 (10, width): row of boundary k in each column (NaN = excluded ONH column)
    boundaries.csv  the same values, one row per column
    layers_overlay.tif      PRIMARY REVIEW: input with L01-L09 filled in their colours (8-bit RGB, same size as input)
    boundaries_overlay.tif  PRIMARY REVIEW: input with the ten boundaries drawn (8-bit RGB, same size as input)
    overlay.png     optional preview: input, layers + boundaries, and a legend in one image
    info.json       settings, ONH handling, and quick checks
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
LAYER_IDS = [f"L{i:02d}" for i in range(1, 10)]
# Same colours as the expert annotation palette (L01-L09)
PALETTE = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
           (0, 255, 255), (255, 128, 0), (128, 255, 0), (128, 0, 255)]
ONH_MARGIN, ONH_BUFFER = 15, 5          # established handling: zone = core +/- 15 columns, plus 5-column buffer
REQUIRED_SIZE = (300, 1000)            # rows x columns of the prepared crops the model was trained and validated on
ACCEPTED_MODES = {"L", "RGB", "RGBA"}  # 8-bit greyscale / colour


def fail(msg):
    sys.exit(f"ERROR: {msg}")


try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import yaml
    from PIL import Image
    from scipy import ndimage as ndi
    import segmentation_models_pytorch as smp
except ImportError as e:
    fail(f"missing dependency ({e.name}). Install with:  pip install -r requirements.txt")


# ----------------------------------------------------------------------------- model (identical to training)
class OrderedBoundaryModel(nn.Module):
    """smp U-Net (ResNet-18) + 3x3-conv boundary head; ordered surfaces s_1 = p_1, s_k = s_(k-1) + softplus(p_k - s_(k-1))."""

    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.net = smp.Unet(m["encoder"], encoder_weights=None, in_channels=m["in_channels"],
                            classes=cfg["labels"]["num_classes"])
        self.bhead = nn.Conv2d(self.net.segmentation_head[0].in_channels, cfg["boundary"]["surfaces"], 3, padding=1)
        self.beta = float(cfg["boundary"]["softplus_beta"])

    def forward(self, x):
        dec = self.net.decoder(self.net.encoder(x))
        return self.net.segmentation_head(dec), self.bhead(dec)


def load_model(device):
    ck, cf = HERE / "model.pt", HERE / "config.yaml"
    for p in (ck, cf):
        if not p.is_file():
            fail(f"{p.name} not found next to predict.py ({HERE}). Keep the package files together.")
    cfg = yaml.safe_load(cf.read_text())
    model = OrderedBoundaryModel(cfg)
    try:
        model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True), strict=True)
    except Exception as e:
        fail(f"could not load model.pt ({e}). The file may be incomplete or from a different model.")
    return model.to(device).eval(), cfg


# ----------------------------------------------------------------------------- preprocessing (identical to training)
def read_image(path):
    try:
        im = Image.open(path)
        n_frames = getattr(im, "n_frames", 1)
        mode = im.mode
        if n_frames > 1:
            fail(f"{path.name}: contains {n_frames} frames/pages; provide one 300x1000 B-scan crop per file")
        if mode not in ACCEPTED_MODES:
            kind = "16-bit" if mode.startswith("I;16") else "32-bit" if mode in ("I", "F") else f"mode '{mode}'"
            fail(f"{path.name}: {kind} image is not supported; provide an 8-bit greyscale or RGB crop")
        a = np.array(im)
    except SystemExit:
        raise
    except Exception as e:
        fail(f"cannot read image {path}: {e}")
    if a.dtype != np.uint8:
        fail(f"{path.name}: expected an 8-bit image, got {a.dtype}")
    if a.shape[:2] != REQUIRED_SIZE:
        fail(f"{path.name}: image is {a.shape[0]} rows x {a.shape[1]} columns; the model requires a prepared crop of "
             f"exactly {REQUIRED_SIZE[0]} x {REQUIRED_SIZE[1]}. Crop the raw OCT frame to the 300-row band containing "
             f"the retina (full 1000-column width) first; do not resize.")
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.ndim != 3 or a.shape[-1] not in (3, 4):
        fail(f"{path}: unsupported image shape {a.shape}")
    if a.shape[-1] == 4:
        if not (a[..., 3] == 255).all():
            fail(f"{path}: image has transparent pixels; save it without transparency")
        a = a[..., :3]
    return a


def preprocess(rgb):
    """Grey channel; burned-in non-grey pixels (scale bar / text) + 1-px ring are inpainted
    (nearest fill, then 5x5 median); per-image z-score."""
    grey = rgb[..., 0].astype(np.float32)
    overlay = ~((rgb[..., 0] == rgb[..., 1]) & (rgb[..., 1] == rgb[..., 2]))
    if overlay.any():
        overlay = ndi.binary_dilation(overlay, iterations=1)
        _, (iy, ix) = ndi.distance_transform_edt(overlay, return_indices=True)
        med = ndi.median_filter(grey[iy, ix], size=5)
        grey = grey.copy()
        grey[overlay] = med[overlay]
    img = ((grey - grey.mean()) / (grey.std() + 1e-6)).astype(np.float32)
    return img, overlay


# ----------------------------------------------------------------------------- inference (identical to evaluation)
@torch.no_grad()
def infer(model, img, device):
    """Replicate-pad to multiples of 32, boundary row softmax over the real rows only, ordered surfaces,
    clip to [0, H]; mask: row y is layer k if round(s_k) <= y < round(s_(k+1))."""
    H, W = img.shape
    fh, fw = -(-H // 32) * 32, -(-W // 32) * 32
    x = torch.from_numpy(img)[None, None]
    x = F.pad(x, (0, fw - W, 0, fh - H), mode="replicate").to(device)
    _, logits = model(x)
    bl = logits[0, :, :H, :W].float()
    prob = torch.softmax(bl, dim=-2)
    p = (prob * torch.arange(H, dtype=prob.dtype, device=prob.device).view(-1, 1)).sum(-2)
    s = [p[0]]
    for k in range(1, p.shape[0]):
        s.append(s[-1] + F.softplus(p[k] - s[-1], beta=model.beta))
    s = torch.stack(s).cpu().numpy().astype(np.float32)
    r = np.clip(np.round(s), 0, H).astype(int)
    mask = np.zeros((H, W), np.uint8)
    ys = np.arange(H)[:, None]
    for k in range(s.shape[0] - 1):
        mask[(ys >= r[k][None]) & (ys < r[k + 1][None])] = k + 1
    return np.clip(s, 0, H), mask


# ----------------------------------------------------------------------------- outputs
LAYER_ALPHA = 0.35                      # fill opacity of the layer colours in layers_overlay.tif (same for all layers)
BOUNDARY_COLOURS = PALETTE + [(255, 255, 255)]   # b01-b09: colour of the layer whose top they mark; b10: white


def layers_overlay(rgb, mask):
    """Input image (unchanged outside layers) with L01-L09 filled at constant opacity; label 0 and 255 (excluded
    ONH columns) stay uncoloured. Same size as the input, 8-bit RGB, no lines, legend or extra rows."""
    base = rgb.astype(np.float32)
    out = base.copy()
    for k, c in enumerate(PALETTE, 1):
        m = mask == k
        out[m] = (1 - LAYER_ALPHA) * base[m] + LAYER_ALPHA * np.array(c, np.float32)
    return np.rint(out).clip(0, 255).astype(np.uint8)


def boundaries_overlay(rgb, surf):
    """Input image with the ten boundaries drawn as 1-px polylines over consecutive non-excluded columns
    (NaN columns = excluded ONH are skipped). Same size as the input, 8-bit RGB, no fills, legend or extra rows."""
    from PIL import ImageDraw
    H, W = rgb.shape[:2]
    pil = Image.fromarray(rgb.copy())
    d = ImageDraw.Draw(pil)
    for k in range(surf.shape[0] - 1, -1, -1):          # draw b10 first so b01..b09 stay on top where they meet
        ys = np.minimum(H - 1, np.rint(surf[k]))
        run = []
        for x in range(W + 1):
            if x < W and np.isfinite(ys[x]):
                run.append((x, int(ys[x])))
                continue
            if len(run) == 1:
                d.point(run, fill=BOUNDARY_COLOURS[k])
            elif run:
                d.line(run, fill=BOUNDARY_COLOURS[k], width=1)
            run = []
    return np.array(pil)


def save_rgb_tiff(arr, path):
    """Single-page, uncompressed, 8-bit RGB TIFF."""
    assert arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[2] == 3
    Image.fromarray(arr).save(path)


def overlay_image(rgb, mask, surf, excl):
    base = np.repeat(rgb[..., :1], 3, -1).astype(np.float32)
    col = np.zeros_like(base)
    for k, c in enumerate(PALETTE, 1):
        col[mask == k] = c
    a = (mask >= 1) & (mask <= 9)
    out = base.copy()
    out[a] = 0.65 * base[a] + 0.35 * col[a]
    H, W = mask.shape
    for k in range(surf.shape[0]):                      # boundary lines: white
        for x in range(W):
            if np.isfinite(surf[k, x]):
                y = int(min(H - 1, round(surf[k, x])))
                out[y, x] = (255, 255, 255)
    if excl.any():                                      # excluded ONH columns: darkened
        out[:, excl] *= 0.35
    out = out.clip(0, 255).astype(np.uint8)
    # side by side: input (top) and overlay (bottom), plus a legend strip
    legend = np.full((24, W, 3), 255, np.uint8)
    pil = Image.fromarray(np.concatenate([np.repeat(rgb[..., :1], 3, -1), out, legend], 0))
    from PIL import ImageDraw
    d = ImageDraw.Draw(pil)
    x0, y0 = 4, 2 * H + 6
    for lid, c in zip(LAYER_IDS, PALETTE):
        d.rectangle([x0, y0, x0 + 12, y0 + 12], fill=c)
        d.text((x0 + 16, y0), lid, fill=(0, 0, 0))
        x0 += 60
    d.text((x0 + 10, y0), "white = boundaries" + ("; dark = excluded ONH" if excl.any() else ""), fill=(0, 0, 0))
    return pil


def onh_columns(core, W):
    a, b = core
    if not (0 <= a <= b < W):
        fail(f"ONH core {a}-{b} is outside the image width 0-{W - 1}")
    lo, hi = max(0, a - ONH_MARGIN - ONH_BUFFER), min(W - 1, b + ONH_MARGIN + ONH_BUFFER)
    excl = np.zeros(W, bool)
    excl[lo:hi + 1] = True
    return excl, [lo, hi]


def run_one(path, out_root, model, cfg, device, core):
    rgb = read_image(path)                               # 8-bit RGB view of the input (greyscale replicated)
    H, W = rgb.shape[:2]
    img, overlay = preprocess(rgb)
    surf, mask = infer(model, img, device)
    excl, excl_range = (onh_columns(core, W) if core else (np.zeros(W, bool), None))
    mask[:, excl] = 255
    surf[:, excl] = np.nan
    od = out_root / path.stem
    od.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(od / "mask.tif")           # single-page uint8 TIFF, uncompressed
    np.save(od / "boundaries.npy", surf)
    with open(od / "boundaries.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["column"] + [f"b{k:02d}" for k in range(1, surf.shape[0] + 1)])
        for x in range(W):
            w.writerow([x] + ["" if np.isnan(v) else f"{v:.3f}" for v in surf[:, x]])
    save_rgb_tiff(layers_overlay(rgb, mask), od / "layers_overlay.tif")          # primary review output
    save_rgb_tiff(boundaries_overlay(rgb, surf), od / "boundaries_overlay.tif")  # primary review output
    overlay_image(rgb, mask, surf, excl).save(od / "overlay.png")                # optional combined preview
    keep = ~excl
    labels = sorted(int(v) for v in np.unique(mask[:, keep])) if keep.any() else []
    checks = {"mask_labels": labels, "labels_within_0_9": set(labels) <= set(range(10)),
              "boundaries_finite": bool(np.isfinite(surf[:, keep]).all()),
              "boundaries_ordered_noncrossing": bool((np.diff(surf[:, keep], axis=0) >= 0).all())}
    info = {"input": path.name, "height": H, "width": W, "device": str(device),
            "mask_file": "mask.tif (uint8; 0 background, 1-9 = L01-L09, 255 = excluded ONH column)",
            "review_files": {"layers_overlay.tif": f"input with L01-L09 filled at opacity {LAYER_ALPHA}",
                             "boundaries_overlay.tif": "input with b01-b09 in the colour of the layer below, b10 white",
                             "overlay.png": "optional preview (input + overlay + legend)"},
            "burned_in_overlay_pixels_inpainted": int(overlay.sum()),
            "onh_core": list(core) if core else None, "onh_excluded_columns": excl_range,
            "boundaries": "b01-b09 = top of L01-L09; b10 = bottom of L09 (RPE/choroid interface); rows, 0 = top",
            "checks": checks}
    (od / "info.json").write_text(json.dumps(info, indent=1))
    print(f"  {path.name}: labels {labels}, boundaries finite={checks['boundaries_finite']}, "
          f"ordered={checks['boundaries_ordered_noncrossing']}" + (f", ONH excluded {excl_range}" if core else "")
          + f" -> {od}")
    return checks


def main():
    ap = argparse.ArgumentParser(description="Final no-choroid OCT layer model: 9 layers + 10 ordered boundaries.")
    ap.add_argument("input", help="an image file or a folder of images")
    ap.add_argument("--out", default="predictions", help="output folder (default: ./predictions)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--onh-core", nargs=2, type=int, metavar=("START", "END"),
                   help="ONH core columns for a single image; core +/- 15 columns + 5-column buffer are excluded")
    g.add_argument("--onh-csv", help="CSV with columns image,core_start,core_end (image = file name) for a folder")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    a = ap.parse_args()

    src = Path(a.input).expanduser()
    if not src.exists():
        fail(f"input not found: {src}")
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS) if src.is_dir() else [src]
    if not files:
        fail(f"no images ({', '.join(sorted(EXTS))}) found in {src}")
    if src.is_file() and src.suffix.lower() not in EXTS:
        fail(f"unsupported file type {src.suffix}; use one of {', '.join(sorted(EXTS))}")
    if a.onh_core and len(files) > 1:
        fail("--onh-core applies to a single image; for a folder use --onh-csv")
    cores = {}
    if a.onh_csv:
        try:
            with open(a.onh_csv) as f:
                for r in csv.DictReader(f):
                    cores[Path(r["image"]).name] = (int(r["core_start"]), int(r["core_end"]))
        except (OSError, KeyError, ValueError) as e:
            fail(f"cannot read --onh-csv {a.onh_csv} (need columns image,core_start,core_end): {e}")

    if a.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        dev = a.device
        if (dev == "cuda" and not torch.cuda.is_available()) or (dev == "mps" and not torch.backends.mps.is_available()):
            fail(f"device '{dev}' is not available on this machine; use --device cpu")
    device = torch.device(dev)
    model, cfg = load_model(device)
    out = Path(a.out).expanduser()
    print(f"Model loaded ({device}); {len(files)} image(s) -> {out}")
    ok = True
    for p in files:
        core = tuple(a.onh_core) if a.onh_core else cores.get(p.name)
        c = run_one(p, out, model, cfg, device, core)
        ok &= c["labels_within_0_9"] and c["boundaries_finite"] and c["boundaries_ordered_noncrossing"]
    if not ok:
        fail("a quick check failed for at least one image; see info.json")
    print("Done.")


if __name__ == "__main__":
    main()
