# Mouse OCT layer segmentation — no-choroid model

Segments a mouse retinal OCT B-scan crop into **9 layers (L01–L09)** and **10 ordered boundaries**. The choroid is not predicted.
See `LAYERS.md` for label numbers, colours and boundary definitions. The layer names are provisional.

## Input requirement: crop first
The model accepts only **prepared crops**:
- exactly **300 rows × 1000 columns**;
- one 8-bit image per file (greyscale or RGB; PNG or TIFF);
- **full image width, no resizing**: the crop is a 300-row band of the raw B-scan containing the retina.
  Match the training crops: the inner retinal surface lies at roughly rows 60–110 of the crop, and the bottom of the choroid at roughly rows 215–250.

Raw OCT frames (for example 1000 × 1024) **must be cropped to 300 × 1000 before inference**. Other sizes, 16-bit images and multi-page TIFFs are rejected with an error message.
Coloured (non-grey) scale bars or text burned into the crop are cleaned automatically. Greyscale annotations cannot be detected, so remove them or keep them outside the retina.

## Install (once)
Requires Python 3.12. From this folder:
```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick test (synthetic image)
```bash
source .venv/bin/activate
python make_smoke_test.py
python predict.py smoke_test/synthetic_crop.png --out smoke_test_output
```
The synthetic image only checks that everything runs. Its segmentation is not meaningful.

## Run on your crops
```bash
source .venv/bin/activate
python predict.py path/to/crop.tif --out results           # one crop
python predict.py path/to/crops_folder --out results       # every .png/.tif/.tiff/.jpg/.bmp in a folder
```
- **Exclude the optic nerve head (recommended when the scan crosses the disc):** `python predict.py crop.tif --onh-core 450 560 --out results`.
  Give the ONH core columns. The core is widened by 15 columns on each side, plus a 5-column buffer.
  For a folder, use `--onh-csv onh.csv` with columns `image,core_start,core_end`, where `image` is the file name.
- **Choose a device:** `--device cpu` (the default `auto` uses a GPU if one is available).

## Output (one folder per image)
| file | contents |
|---|---|
| `mask.tif` | single-page 8-bit TIFF, 300 × 1000: 0 = background, 1–9 = L01–L09, 255 = excluded ONH column |
| `boundaries.npy` / `boundaries.csv` | 10 × 1000 boundary rows (b01–b10), sub-pixel, 0 = top row of the crop. NaN or empty = excluded ONH column |
| `overlay.png` | input (top) and coloured layers with white boundaries (bottom) |
| `info.json` | settings and quick checks (labels 0–9, boundaries finite and ordered) |

Boundary rows are in **crop** coordinates. To place them in the raw frame, add the row where your crop starts.

## Model
- **Architecture:** ResNet-18 U-Net with an ordered-boundary head. The settings are frozen in `config.yaml`.
- **Training:** 14 expert-annotated B-scans from 4 independent animal groups, 1500 iterations, seed 0. `model.pt` SHA-256 `32a77ec7e4bb58b8c7fbec5079c968f20687e39d2ffb7d471065bd03b42b4a0b`.
- **Preprocessing at inference:** grey channel, cleanup of coloured overlays, per-image z-score, replicate padding to 320 × 1024. No other preparation is applied.
- **Expected accuracy:** taken from four-fold leave-one-animal-group-out cross-validation of the same settings: layer Dice 0.933 ± 0.006, mean boundary error 0.90 ± 0.08 px, outside ONH zones.
  This model has no separate held-out test result.
- **Not reliable:** predictions inside the optic nerve head; uncropped or resized images; and scans unlike the training data (other devices or pathology).
