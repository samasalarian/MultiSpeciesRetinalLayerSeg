# Layer and boundary reference (no-choroid model)

Names are **provisional**. Expert confirmation was not documented when the model was trained.

| Mask label | Layer ID | Provisional name | Overlay colour (same as annotation palette) |
|---|---|---|---|
| 0 | – | background (vitreous above L01, everything below L09) | none |
| 1 | L01 | RNFL (possibly RNFL/GCL complex) | red (255, 0, 0) |
| 2 | L02 | IPL | green (0, 255, 0) |
| 3 | L03 | INL | blue (0, 0, 255) |
| 4 | L04 | OPL | yellow (255, 255, 0) |
| 5 | L05 | ONL | magenta (255, 0, 255) |
| 6 | L06 | ELM | cyan (0, 255, 255) |
| 7 | L07 | IS/OS | orange (255, 128, 0) |
| 8 | L08 | IDZ | light green (128, 255, 0) |
| 9 | L09 | RPE | purple (128, 0, 255) |
| 255 | – | excluded ONH column (only when an ONH core is given) | left as in the input (darkened in `overlay.png`) |

The choroid (L10 in the annotations) is not predicted; it is part of label 0 below L09.

Boundaries (`boundaries.npy`, shape 10 × image width; row coordinate, 0 = top of image):

| Boundary | Meaning | Line colour in `boundaries_overlay.tif` |
|---|---|---|
| b01 | top of L01 (inner limiting surface) | L01 colour (red) |
| b02–b09 | top of L02–L09 (= bottom of the layer above) | colour of L02–L09 respectively |
| b10 | bottom of L09 (RPE / choroid interface) | white |

Boundaries are ordered by construction (b01 ≤ b02 ≤ … ≤ b10 in every column). A layer whose two boundaries coincide has zero thickness in that column.
