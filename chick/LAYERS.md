# Boundary and layer definitions

The models predict **ten boundaries**. Each boundary is the **lower edge** of the
named retinal layer, as traced by the expert annotators. This follows the
annotation protocol the models were trained on and does not reinterpret the
anatomy.

Coordinates are image rows: 0 is the top row of the input and rows increase
downward. In every column where they are visible, the boundaries are strictly
ordered top to bottom with at least 1 px between neighbours:

`RNFL < GCL < IPL < INL < OPL < ONL < ELM < EZ < IZ < RPE`

## Boundaries (`boundaries.npy` rows, `boundaries.csv` columns)

| Index | Name | Meaning | Colour in `boundaries_overlay.tif` (RGB) |
|---:|---|---|---|
| 0 | RNFL | lower edge of the retinal nerve fibre layer | 255, 0, 0 |
| 1 | GCL | lower edge of the ganglion cell layer | 0, 255, 0 |
| 2 | IPL | lower edge of the inner plexiform layer | 0, 0, 255 |
| 3 | INL | lower edge of the inner nuclear layer | 255, 255, 0 |
| 4 | OPL | lower edge of the outer plexiform layer | 255, 0, 255 |
| 5 | ONL | lower edge of the outer nuclear layer | 0, 255, 255 |
| 6 | ELM | lower edge of the external limiting membrane band | 255, 128, 0 |
| 7 | EZ | lower edge of the ellipsoid zone | 128, 255, 0 |
| 8 | IZ | lower edge of the interdigitation zone | 128, 0, 255 |
| 9 | RPE | lower edge of the retinal pigment epithelium | 255, 0, 192 |

The lowest trace is called `PRE` in the original annotation files. It is
reported here as `RPE`: it is the same trace under a different name.

## Layer regions (`layers_overlay.tif`)

A layer's region is the set of rows strictly below the previous boundary, down
to and including the layer's own boundary. It is filled only in columns where
**both** of those boundaries are visible. Nothing is interpolated across gaps.

| Layer | Upper limit (exclusive) | Lower limit (inclusive) | Fill colour (RGB) |
|---|---|---|---|
| RNFL | not available: **not filled** | RNFL | — |
| GCL | RNFL | GCL | 0, 255, 0 |
| IPL | GCL | IPL | 0, 0, 255 |
| INL | IPL | INL | 255, 255, 0 |
| OPL | INL | OPL | 255, 0, 255 |
| ONL | OPL | ONL | 0, 255, 255 |
| ELM | ONL | ELM | 255, 128, 0 |
| EZ | ELM | EZ | 128, 255, 0 |
| IZ | EZ | IZ | 128, 0, 255 |
| RPE | IZ | RPE | 255, 0, 192 |

**Why RNFL is not filled:** its upper edge (the inner limiting membrane,
the retina–vitreous interface) is not one of the ten trained boundaries.
Drawing an RNFL region would mean inventing that edge, so the package doesn't.
The RNFL boundary line itself is still predicted and drawn in
`boundaries_overlay.tif`.

The fills are blended over the image at the opacity set by `--layer-alpha`
(default 0.5; 1.0 gives fully solid colour).
