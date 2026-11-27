# Multi-species retinal layer segmentation

Trained models for segmenting retinal layers in OCT B-scans. Each directory is a
self-contained package with its own weights, `predict.py`, requirements and
documentation.

| Directory | Models |
|---|---|
| [`mouse/`](mouse/README.md) | Mouse no-choroid retinal-layer segmentation |
| [`chick/`](chick/README.md) | Chick MST and U-Net boundary models (10 boundaries) |

The chick weights (`chick/*.pth`) are stored with [Git LFS](https://git-lfs.com).
Install Git LFS before cloning, or run `git lfs pull` afterwards, to download them.
