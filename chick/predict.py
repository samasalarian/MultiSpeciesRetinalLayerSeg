#!/usr/bin/env python3
"""Segment ten retinal boundaries in a chick OCT B-scan with the U-Net or MST model.

    python predict.py --model unet --input scan.tif --output-dir results/scan_unet
    python predict.py --model mst  --input scan.tif --output-dir results/scan_mst

See the repository root README.md for the input requirements and output definitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chick_seg import __version__
from chick_seg.pipeline import InputError, choose_device, load_config, load_model, ordering_report, read_and_validate, run
from chick_seg.render import boundaries_overlay, layers_overlay

OUTPUTS = ("boundaries.csv", "boundaries.npy", "visibility.npy", "info.json", "layers_overlay.tif", "boundaries_overlay.tif")


def write_csv(path: Path, boundaries: list[str], rows: np.ndarray, vis: np.ndarray, visible: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["column", *[f"{b}_{field}" for b in boundaries for field in ("row", "visibility", "visible")]])
        for x in range(rows.shape[1]):
            line: list[object] = [x]
            for c in range(len(boundaries)):
                line += ["" if not visible[c, x] else f"{rows[c, x]:.1f}", f"{vis[c, x]:.6f}", int(visible[c, x])]
            writer.writerow(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=("unet", "mst"), required=True)
    ap.add_argument("--input", type=Path, required=True, help="single-page grayscale TIFF B-scan")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--auto-crop", action="store_true",
                    help="input is a full-height frame: crop 300 rows around the outer retina by intensity first")
    ap.add_argument("--device", default="auto", help="auto (default), cpu, cuda or mps")
    ap.add_argument("--layer-alpha", type=float, default=0.5, help="layer fill opacity in layers_overlay.tif (0-1]")
    ap.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = ap.parse_args()
    if not 0 < args.layer_alpha <= 1:
        ap.error("--layer-alpha must be in (0, 1]")

    cfg = load_config(args.model)
    existing = [name for name in OUTPUTS if (args.output_dir / name).exists()]
    if existing and not args.overwrite:
        print(f"error: {args.output_dir} already has {existing}; use --overwrite or another --output-dir", file=sys.stderr)
        return 2
    try:
        gray, input_info = read_and_validate(args.input, cfg, args.auto_crop)
    except InputError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    device = choose_device(args.device)
    model, digest = load_model(cfg, device)
    try:
        result = run(model, gray, cfg, device, args.auto_crop)
    except InputError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    boundaries = cfg["boundaries"]
    rows, vis, visible = result["rows"], result["visibility"], result["visible"]
    order = ordering_report(rows, visible)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "boundaries.npy", rows)
    np.save(args.output_dir / "visibility.npy", vis)
    write_csv(args.output_dir / "boundaries.csv", boundaries, rows, vis, visible)
    colors = [cfg["boundary_colors"][b] for b in boundaries]
    Image.fromarray(boundaries_overlay(gray, rows, visible, colors)).save(args.output_dir / "boundaries_overlay.tif")
    Image.fromarray(layers_overlay(gray, rows, visible, cfg["layers"], boundaries, args.layer_alpha)).save(args.output_dir / "layers_overlay.tif")

    height = gray.shape[0]
    info = {
        "package_version": __version__,
        "model": args.model,
        "model_description": cfg["description"],
        "checkpoint": cfg["checkpoint"],
        "checkpoint_sha256": digest,
        "checkpoint_key": cfg["checkpoint_key"],
        "input": {"file": args.input.name, "height": int(gray.shape[0]), "width": int(gray.shape[1]),
                  "dtype": input_info["dtype"], "mode": input_info["mode"], "channels": input_info["channels"],
                  "auto_crop": args.auto_crop},
        "preprocessing": result["preprocessing"],
        "boundaries": boundaries,
        "boundary_definition": "row of the lower edge of each named layer (see LAYERS.md)",
        "coordinates": "row index in the input image, 0 = top row, increasing downward; one value per input column; NaN = not visible",
        "visibility_thresholds": cfg["visibility_thresholds"],
        "visible_columns": {b: int(visible[i].sum()) for i, b in enumerate(boundaries)},
        "visible_rows_outside_image": int(((rows[visible] < 0) | (rows[visible] > height - 1)).sum()),
        "ordering_check": order,
        "outputs": {
            "boundaries.npy": f"float32 [{len(boundaries)}, {gray.shape[1]}] boundary rows (NaN where not visible)",
            "visibility.npy": f"float32 [{len(boundaries)}, {gray.shape[1]}] visibility probability",
            "boundaries.csv": "one line per column: <B>_row (blank if not visible), <B>_visibility, <B>_visible",
            "layers_overlay.tif": f"RGB uint8 {gray.shape[0]}x{gray.shape[1]}, layer fills (alpha {args.layer_alpha})",
            "boundaries_overlay.tif": f"RGB uint8 {gray.shape[0]}x{gray.shape[1]}, 1-px boundary lines",
        },
        "runtime": {"device": str(device), "torch": torch.__version__, "python": platform.python_version()},
        "note": "Performance estimates are the cross-validation results in README.md; this output carries no accuracy.",
    }
    (args.output_dir / "info.json").write_text(json.dumps(info, indent=2))
    status = "ok" if order["passed"] else "ORDERING CHECK FAILED"
    print(f"{args.model}: {args.input.name} -> {args.output_dir} ({status}; visible columns "
          f"{min(info['visible_columns'].values())}-{max(info['visible_columns'].values())} of {gray.shape[1]})")
    return 0 if order["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
