# Multi-species retinal layer segmentation

Trained models that segment retinal layers in OCT B-scans:

| Species | Model | Predicts |
|---|---|---|
| Mouse | U-Net, no-choroid | 9 layers (L01–L09) and 10 ordered boundaries |
| Chick | U-Net, or MST (Mean Teacher semi-supervised U-Net) | 10 layer boundaries, with per-column visibility |

## Install (one environment for all models)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The chick weights (`chick/*.pth`) are stored with [Git LFS](https://git-lfs.com). Install Git LFS before cloning, or run `git lfs pull` afterwards.

## Input requirements

Put images in `input/mouse/` or `input/chick/`. Every image must have the **vitreous at the top**.

- **Mouse:** a prepared retinal crop, one 8-bit image per file (greyscale or RGB; PNG, TIFF, JPEG or BMP), exactly **300 × 1000** (rows × columns). Do not resize.
- **Chick:** a single-page TIFF (`.tif`/`.tiff`), 8- or 16-bit greyscale, exactly **1000** columns wide and **240–320** rows tall (a crop around the retina). For full-height frames, add `--auto-crop`. Full rules are in [`chick/README.md`](chick/README.md).

Images that don't meet the rules are reported and skipped. The other images are still processed.

## Run

```bash
python predict.py --species mouse
python predict.py --species chick --model unet
python predict.py --species chick --model mst
```

Every supported image in the species' input folder is processed. Results go to:

```text
output/mouse/IMAGE_NAME/
    tif/   layers_overlay.tif  boundaries_overlay.tif  mask.tif
    data/  boundaries.csv  boundaries.npy  info.json
output/chick/unet/IMAGE_NAME/   (and output/chick/mst/IMAGE_NAME/)
    tif/   layers_overlay.tif  boundaries_overlay.tif
    data/  boundaries.csv  boundaries.npy  visibility.npy  info.json
```

Running again replaces earlier results for the same image name. Optional flags: `--device cpu|mps|cuda`, `--auto-crop` (chick) and `--onh-csv FILE` (mouse, to exclude the optic nerve head; see [`mouse/README.md`](mouse/README.md)).

Quick test with synthetic images (contain no real data):

```bash
python mouse/make_smoke_test.py && cp mouse/smoke_test/synthetic_crop.tif input/mouse/
python chick/make_synthetic_oct.py --output-dir chick/synthetic && cp chick/synthetic/synthetic_crop.tif input/chick/
```

## TIFF overlays for review

Both overlays are 8-bit RGB TIFFs at exactly the input size, with no legend or margin.

- `layers_overlay.tif`: the input with each layer filled in its colour.
- `boundaries_overlay.tif`: the input with each boundary drawn as a 1-px line.
- `mask.tif` (mouse only): label image with 0 = background and 1–9 = L01–L09.

The two species define boundaries differently:
- **Mouse** boundaries are the top of each layer, plus the bottom of L09.
- **Chick** boundaries are the lower edge of each named layer. They are drawn only where the model predicts them visible (`NaN` elsewhere). The chick RNFL region is not filled because its upper edge is not one of the ten boundaries.

Colours and definitions: [`mouse/LAYERS.md`](mouse/LAYERS.md), [`chick/LAYERS.md`](chick/LAYERS.md).

## Accuracy (cross-validation boundary MAE)

| Model | Boundary MAE |
|---|---|
| Mouse no-choroid | 0.90 ± 0.08 px |
| Chick U-Net | 1.789 px (subject-macro) |
| Chick MST | 2.159 px (subject-macro) |

The released checkpoints were trained on the complete development datasets. The MAE values above come from cross-validation, not from the released checkpoints. Details are in [`mouse/README.md`](mouse/README.md) and [`chick/README.md`](chick/README.md).
