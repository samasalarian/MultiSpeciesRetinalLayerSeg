# Multi-species retinal layer segmentation

Trained models that segment retinal layers in OCT B-scans of mouse and chick.
The mouse model predicts 9 layers and 10 boundaries; the chick U-Net and MST models predict 10 layer boundaries with per-column visibility.

Layer and boundary definitions, names and colours: [`mouse/LAYERS.md`](mouse/LAYERS.md) and [`chick/LAYERS.md`](chick/LAYERS.md).

## Install

The chick weights (`chick/*.pth`) are stored with [Git LFS](https://git-lfs.com). Install Git LFS before cloning, or run `git lfs pull` afterwards.

One environment serves all models (Python 3.12):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Input

Put mouse images in `input/mouse/` and chick images in `input/chick/`.

- **Mouse:** an 8-bit greyscale or RGB retinal crop of exactly 300 rows × 1000 columns, vitreous at the top. Don't resize.
- **Chick:** a single-page TIFF, 1000 pixels wide, vitreous at the top.

Images that fail a check are reported and skipped; the others are still processed.

## Run

```bash
python predict.py --species mouse
python predict.py --species chick --model unet
python predict.py --species chick --model mst
```

Further options (optic nerve head exclusion for mouse, auto-crop of full-height chick frames): `python predict.py --help`.

## Output

Results appear in `output/mouse/<image>/`, `output/chick/unet/<image>/` and `output/chick/mst/<image>/`:

- `tif/layers_overlay.tif`: the input image with each predicted layer filled in its colour.
- `tif/boundaries_overlay.tif`: the input image with each predicted boundary drawn as a 1-pixel line.
- `data/`: boundary rows per column (`boundaries.csv`, `boundaries.npy`), chick visibility (`visibility.npy`) and run details (`info.json`); mouse also writes the label mask `tif/mask.tif`.

Both overlays are 8-bit RGB TIFFs at the input size, with no legend or margin.

## Accuracy

Cross-validation boundary mean absolute error (MAE):

| Model | Boundary MAE |
|---|---|
| Mouse | 0.90 ± 0.08 px |
| Chick U-Net | 1.789 px |
| Chick MST | 2.159 px |

The released checkpoints were trained on the full development datasets; the MAE values above come from cross-validation, not from the released checkpoints.
All development data were normal retinas.
