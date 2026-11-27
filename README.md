# Multi-species retinal layer segmentation

Trained models that segment retinal layers in OCT B-scans:

| Species | Model | Predicts |
|---|---|---|
| Mouse | U-Net, no-choroid | 9 layers (L01–L09) and 10 ordered boundaries |
| Chick | U-Net, or MST (Mean Teacher semi-supervised U-Net) | 10 layer boundaries, with per-column visibility |

Layer and boundary definitions, names and colours: [`mouse/LAYERS.md`](mouse/LAYERS.md) and [`chick/LAYERS.md`](chick/LAYERS.md).

## Install (one environment for all models)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The chick weights (`chick/*.pth`) are stored with [Git LFS](https://git-lfs.com). Install Git LFS before cloning, or run `git lfs pull` afterwards. The chick weights are checked against their SHA-256 hashes on every run.

## Input requirements

Put mouse images in `input/mouse/` and chick images in `input/chick/`. All images must be standard B-scans with the **vitreous at the top**. Vertically flipped images can't be detected and are not supported.

**Mouse:** a prepared retinal crop.
- One 8-bit greyscale or RGB image per file: `.png`, `.tif`, `.tiff`, `.jpg`, `.jpeg` or `.bmp`.
- Exactly **300 rows × 1000 columns**: a 300-row band of the raw B-scan at full width. Don't resize.
- Match the training crops: the inner retinal surface roughly at rows 60–110, and the bottom of the choroid roughly at rows 215–250.
- Rejected: 16-bit images, multi-page files and transparent pixels.
- Coloured (non-grey) scale bars or text are cleaned automatically. Greyscale annotations can't be detected, so keep them out of the retina.

**Chick:** a single-page TIFF (`.tif` or `.tiff`).
- `uint8` or `uint16` greyscale. RGB/RGBA is accepted only if R = G = B and alpha is fully opaque.
- Exactly **1000 columns** wide.
- **240–320 rows** tall: a crop around the retina, like the training crops (246–300 rows, with the bright outer-retinal band near row 212 of a 300-row crop).
- Full-height frames (at least 300 rows, for example 1024 × 1000) need `--auto-crop` (see Options).
- Either left–right direction works.

The images are normalized, padded and tiled automatically. Files with other extensions are skipped. An image that fails a check is reported, and the other images are still processed.

## Run

```bash
python predict.py --species mouse
python predict.py --species chick --model unet
python predict.py --species chick --model mst
```

Every supported image in `input/mouse/` or `input/chick/` is processed. Running again replaces the earlier results for the same image name. The exit code is 0 if every image succeeded and 1 if any failed.

### Options

- `--device auto|cpu|mps|cuda`: the default `auto` uses a GPU if one is available.
- `--onh-csv FILE` (mouse): excludes the optic nerve head. Use it when a scan crosses the disc, because predictions inside the ONH are not reliable.
  - The CSV has the columns `image,core_start,core_end`, where `image` is the file name.
  - Each core is widened by 15 columns on each side, plus a 5-column buffer.
  - Excluded columns are `255` in `mask.tif` and `NaN` (an empty cell in the CSV) in the boundary files. They are left as in the input in the overlays.
- `--auto-crop` (chick): for full-height frames. A 300-row window is picked from image intensity alone by finding the bright outer-retinal band, and then processed as a crop. Boundary rows are still given in the frame's coordinates. This step was not separately cross-validated.

### Quick test with synthetic images (no real data)

```bash
python mouse/make_smoke_test.py && cp mouse/smoke_test/synthetic_crop.tif input/mouse/
python chick/make_synthetic_oct.py --output-dir chick/synthetic && cp chick/synthetic/synthetic_crop.tif input/chick/
```

Then run the three commands above. The results only show that everything runs; segmentations of synthetic images mean nothing.

## Outputs

```text
output/mouse/IMAGE_NAME/
    tif/   layers_overlay.tif  boundaries_overlay.tif  mask.tif
    data/  boundaries.csv  boundaries.npy  info.json
output/chick/unet/IMAGE_NAME/   (and output/chick/mst/IMAGE_NAME/)
    tif/   layers_overlay.tif  boundaries_overlay.tif
    data/  boundaries.csv  boundaries.npy  visibility.npy  info.json
```

Boundary rows are sub-pixel image rows in the input's coordinates, with 0 at the top. For a mouse crop, add the row where your crop starts to place them in the raw frame.

| File | Mouse | Chick |
|---|---|---|
| `boundaries.npy` | `float32 [10, 1000]`: b01–b09 are the top of L01–L09 and b10 is the bottom of L09. Always ordered. | `float32 [10, 1000]`: the lower edge of each named layer (RNFL … RPE). `NaN` where the boundary is predicted not visible. |
| `boundaries.csv` | `column`, then `b01`…`b10` | `column`, then for each boundary `<B>_row` (empty if not visible), `<B>_visibility` and `<B>_visible` (0/1) |
| `visibility.npy` | – | `float32 [10, 1000]`: visibility probability. A boundary is visible where it is ≥ 0.5. |
| `mask.tif` | 8-bit labels: 0 = background, 1–9 = L01–L09, 255 = excluded ONH column | – |
| `info.json` | settings and quick checks (labels, ordering) | model, checkpoint hash, input, preprocessing, visible-column counts and the ordering check |

Mouse boundaries never cross by construction. Visible chick boundaries are kept in anatomical order at least 1 px apart. Nothing is interpolated across hidden chick columns.

## TIFF overlays for review

Both overlays are the input image as a single-page 8-bit RGB TIFF at exactly the input size, with no legend or margin. Colours are listed in the `LAYERS.md` files.

- **`layers_overlay.tif`:** each layer is filled in its colour over the image, with no lines.
  - Mouse: L01–L09 at 35% opacity. Background and excluded ONH columns are unchanged.
  - Chick: the 9 regions from GCL to RPE at 50% opacity, filled only where both of a region's boundaries are visible. The RNFL region is not filled, because its upper edge (the ILM) is not one of the ten boundaries.
- **`boundaries_overlay.tif`:** each boundary is drawn as a 1-px line over the image, with no fills.
  - Mouse: b01–b09 in the colour of the layer below them, and b10 in white.
  - Chick: each boundary in its own colour, only in the columns where it is visible.

## Accuracy (cross-validation boundary MAE)

| Model | Boundary MAE | Cross-validation |
|---|---|---|
| Mouse no-choroid | 0.90 ± 0.08 px | 4-fold leave-one-animal-group-out, outside ONH zones |
| Chick U-Net | 1.789 px (subject-macro) | 5-fold leave-one-subject-out |
| Chick MST | 2.159 px (subject-macro) | 5-fold leave-one-subject-out |

- **Training vs testing:** the released checkpoints were trained on the complete development datasets, so the MAE values above come from cross-validation, not from the released checkpoints.
- **U-Net vs MST:** the chick U-Net was 0.371 px better than MST. With five subjects, though, the 95% confidence interval for the difference (−0.267 to +1.323 px) includes zero.
- **Scope:** all development data were normal retinas. Accuracy on pathology, other devices or other scan protocols is unknown.
