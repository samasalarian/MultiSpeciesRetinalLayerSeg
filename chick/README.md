# Chick retinal layer segmentation (10 boundaries)

These two models find ten retinal layer boundaries in chick OCT B-scans:

| `--model` | Model | Weights file |
|---|---|---|
| `unet` | Supervised U-Net (ResNet-18 encoder) with a boundary-visibility head | `model_unet.pth` (57.5 MB) |
| `mst` | Mean Teacher semi-supervised U-Net (same architecture); uses the EMA teacher weights | `model_mst.pth` (115.0 MB) |

Both models use the same input, produce the same outputs and run from the same
`predict.py` command. Boundary and layer definitions are in [LAYERS.md](LAYERS.md).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Smoke test on synthetic images (contains no real data)
python make_synthetic_oct.py --output-dir synthetic
python predict.py --model unet --input synthetic/synthetic_crop.tif --output-dir out/unet
python predict.py --model mst  --input synthetic/synthetic_crop.tif --output-dir out/mst

# Your own scan
python predict.py --model unet --input scan.tif --output-dir results/scan_unet
python predict.py --model mst  --input scan.tif --output-dir results/scan_mst

# A full-height frame (for example 1024 x 1000) instead of a retinal crop
python predict.py --model mst --input frame.tif --auto-crop --output-dir results/frame_mst
```

Options: `--device auto|cpu|cuda|mps` (default `auto`), `--layer-alpha` (fill
opacity, default 0.5) and `--overwrite`. The exit code is 0 on success, 2 for a
rejected input or existing outputs, and 1 if the ordering check fails.

## Input requirements (enforced)

| Property | Requirement |
|---|---|
| Format | single-page TIFF (`.tif` / `.tiff`) |
| Pixel type | `uint8` or `uint16` (the models were trained on `uint8`) |
| Channels | grayscale: 1 channel, or RGB/RGBA with identical R = G = B and fully opaque alpha. True colour is rejected. |
| Width | exactly **1000** px (A-scans) |
| Height, default mode | **240–320** px: a crop around the retina like the training crops (246–300 rows, with the bright outer-retinal band around row 212 of a 300-row crop) |
| Height, `--auto-crop` | at least 300 px. A 300-row window is chosen from image intensity alone by locating the bright outer-retinal band, then processed as a crop. This step was used for the unlabeled training scans, but its accuracy was not separately cross-validated. |
| Orientation | standard B-scan: vitreous at the top, choroid at the bottom. Either left–right direction works (training used horizontal flips). Vertically flipped images are not supported and cannot be detected automatically. |

These steps are applied automatically and need nothing from you:
- **Normalization:** each image is clipped to its own 1st–99.5th intensity percentiles and scaled to [0, 1].
- **Padding:** the image is centred and reflect-padded to a height of at least 320 and a multiple of 32, and a width that is a multiple of 32 (1000 becomes 1024).
- **Inference:** the network runs on overlapping 256-px-wide patches spaced 128 px apart, and the patches are averaged.
- **Output frame:** padding is removed and all outputs are in the input image's frame.

## Outputs

`predict.py` writes these files to `--output-dir`. W is the input width and H the input height.

| File | Content |
|---|---|
| `boundaries.npy` | `float32 [10, W]`: row of each boundary in each column, in input-image coordinates (0 = top). `NaN` where the boundary is predicted not visible. |
| `visibility.npy` | `float32 [10, W]`: visibility probability per boundary and column |
| `boundaries.csv` | one line per column: `column`, then for each boundary `<B>_row` (blank if not visible), `<B>_visibility` and `<B>_visible` (0/1) |
| `info.json` | model, checkpoint SHA-256, input properties, preprocessing (crop, padding, normalization range), thresholds, visible-column counts and the ordering check |
| `layers_overlay.tif` | RGB `uint8`, exactly H × W: the input with the 9 fillable layer regions in colour (see [LAYERS.md](LAYERS.md)) |
| `boundaries_overlay.tif` | RGB `uint8`, exactly H × W: the input with 1-px boundary lines in visible columns |

The overlays have no legend, margin or added canvas.

**How visibility works:**
- A boundary counts as visible in a column when its visibility probability is at least 0.5.
- Hidden columns are left empty (`NaN`) and nothing is interpolated across them. Each visible run is decoded with a light continuity term.
- In each column, the visible boundaries are forced into anatomical order with at least 1 px between neighbours, so they never cross.

## Training data

- Both downloadable checkpoints were trained on **all five development subjects**: 10 labeled normal chick OCT scans with expert annotations.
- **MST** also used **three unlabeled scans**, through Mean Teacher consistency training.
- Training settings were fixed before training, using the cross-validation runs:
  - training length: U-Net 30 epochs, MST 93 epochs (the median epoch chosen by validation across folds);
  - learning rate: a constant 1e-4 with AdamW;
  - seed: 42;
  - visibility threshold: 0.5 for every boundary.
- No accuracy is reported for the released checkpoints: they were trained on every labeled scan, so there is nothing held out to test them on.

## Performance estimate (5-fold leave-one-subject-out cross-validation)

These figures come from matched cross-validation runs. In each fold, both models
used the same architecture, preprocessing and training settings, and were trained
without the held-out subject, then tested on it. One difference from the released
checkpoints: in cross-validation, a separate validation subject chose when to stop
training. Values are mean absolute boundary error in pixels: each
subject's errors are averaged first, then the subjects are averaged. Lower is better.

| Boundary | U-Net | MST |
|---|---:|---:|
| **Overall** | **1.789** | **2.159** |
| RNFL | 2.331 | 3.626 |
| GCL | 2.773 | 2.957 |
| IPL | 1.955 | 2.110 |
| INL | 1.717 | 1.692 |
| OPL | 1.413 | 1.880 |
| ONL | 1.129 | 1.794 |
| ELM | 0.952 | 1.458 |
| EZ | 1.296 | 1.510 |
| IZ | 0.933 | 1.366 |
| RPE | 3.326 | 3.277 |

- The U-Net was numerically **0.371 px better overall**. With only five subjects, though, the paired 95% confidence interval for the difference (MST minus U-Net, −0.267 to +1.323 px) includes zero, so this study does not show a reliable difference between the models.
- The cross-validation error was measured at every expert-annotated column, regardless of predicted visibility. The files from `predict.py` hide columns predicted not visible.
- All development data were normal chick retinas. Performance on pathology, other devices, other scan protocols or other species is unknown.

## Environment

Both models run in one environment (`requirements.txt`) with the same
dependencies. The package was tested with Python 3.12 on CPU. The weights load
with `torch.load(..., weights_only=True)` and strict key checking, and their
SHA-256 hashes are verified before every run:

| File | SHA-256 |
|---|---|
| `model_unet.pth` | `8376d34b9d1a3a37ee93266cec9fa7591d65b5f1205ffeb0e5e75ac8e80b7855` |
| `model_mst.pth` | `d08b55fe521978f64b4fe854cdc56c1017a910af307247f4c0db0000a0617e78` |

`chick_seg/vendored.py` holds unmodified copies of the preprocessing, network
wrapper and decoder code used in training and cross-validation. On the
development scans, this package reproduced the original pipeline's outputs bit
for bit.
