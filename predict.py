#!/usr/bin/env python
"""Run the mouse or chick retinal-layer model on every image in input/<species>/.

    python predict.py --species mouse
    python predict.py --species chick --model unet
    python predict.py --species chick --model mst

Each image is processed by the validated species package (mouse/predict.py or
chick/predict.py) unchanged; this wrapper only selects the input files and sorts
the results into output/<species>[/<model>]/<image name>/{tif,data}/.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {
    "mouse": {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"},
    "chick": {".tif", ".tiff"},
}
TIF_FILES = {"layers_overlay.tif", "boundaries_overlay.tif", "mask.tif"}
DATA_FILES = {"boundaries.csv", "boundaries.npy", "visibility.npy", "info.json"}


def command(args: argparse.Namespace, image: Path, tmp: Path) -> list[str]:
    if args.species == "mouse":
        cmd = [sys.executable, str(ROOT / "mouse" / "predict.py"), str(image), "--out", str(tmp), "--device", args.device]
        if args.onh_csv:
            cmd += ["--onh-csv", str(Path(args.onh_csv).expanduser().resolve())]
        return cmd
    cmd = [sys.executable, str(ROOT / "chick" / "predict.py"), "--model", args.model, "--input", str(image),
           "--output-dir", str(tmp), "--device", args.device]
    return cmd + (["--auto-crop"] if args.auto_crop else [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--species", choices=("mouse", "chick"), required=True)
    ap.add_argument("--model", choices=("unet", "mst"), help="chick only (required for chick)")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    ap.add_argument("--auto-crop", action="store_true", help="chick only: inputs are full-height frames (see chick/README.md)")
    ap.add_argument("--onh-csv", help="mouse only: CSV image,core_start,core_end to exclude the optic nerve head")
    args = ap.parse_args()
    if args.species == "chick" and not args.model:
        ap.error("--species chick needs --model unet or --model mst")
    if args.species == "mouse" and (args.model or args.auto_crop):
        ap.error("--model and --auto-crop apply to chick only")
    if args.species == "chick" and args.onh_csv:
        ap.error("--onh-csv applies to mouse only")

    in_dir = ROOT / "input" / args.species
    out_dir = ROOT / "output" / args.species / (args.model or "")
    files, skipped = [], []
    for p in sorted(in_dir.iterdir()) if in_dir.is_dir() else []:
        if p.name.startswith(".") or not p.is_file():
            continue
        (files if p.suffix.lower() in EXTENSIONS[args.species] else skipped).append(p)
    for p in skipped:
        print(f"skipped {p.name}: not a supported {args.species} file ({', '.join(sorted(EXTENSIONS[args.species]))})")
    if not files:
        print(f"error: no input images in {in_dir.relative_to(ROOT)}/", file=sys.stderr)
        return 2
    stems = [p.stem for p in files]
    duplicates = sorted({s for s in stems if stems.count(s) > 1})
    if duplicates:
        print(f"error: several input files share a name: {duplicates}; rename them", file=sys.stderr)
        return 2

    label = args.species + (f" {args.model}" if args.model else "")
    print(f"{label}: {len(files)} image(s) from {in_dir.relative_to(ROOT)}/ -> {out_dir.relative_to(ROOT)}/")
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for image in files:
        with tempfile.TemporaryDirectory(dir=out_dir, prefix=".tmp_") as tmp_name:
            tmp = Path(tmp_name)
            proc = subprocess.run(command(args, image, tmp), cwd=ROOT, capture_output=True, text=True)
            produced = tmp / image.stem if args.species == "mouse" else tmp
            if proc.returncode != 0 or not produced.is_dir():
                failed.append(image.name)
                message = (proc.stderr or proc.stdout).strip().splitlines()
                print(f"  FAILED {image.name}: {message[-1] if message else f'exit code {proc.returncode}'}")
                continue
            dest = out_dir / image.stem
            if dest.exists():
                shutil.rmtree(dest)
            (dest / "tif").mkdir(parents=True)
            (dest / "data").mkdir()
            for f in produced.iterdir():
                if f.name in TIF_FILES:
                    shutil.move(str(f), dest / "tif" / f.name)
                elif f.name in DATA_FILES:
                    shutil.move(str(f), dest / "data" / f.name)
            print(f"  {image.name} -> {dest.relative_to(ROOT)}/")
    print(f"done: {len(files) - len(failed)} succeeded, {len(failed)} failed" + (f" ({', '.join(failed)})" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
