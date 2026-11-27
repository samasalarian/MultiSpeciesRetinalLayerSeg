"""Overlay rendering at the exact input size (no legend, no added canvas)."""

from __future__ import annotations

from typing import Any

import numpy as np


def display_base(gray: np.ndarray) -> np.ndarray:
    """Input intensities as 8-bit RGB: uint8 unchanged, uint16 min-max scaled."""
    if gray.dtype == np.uint8:
        base = gray
    else:
        values = gray.astype(np.float64)
        span = values.max() - values.min()
        base = np.zeros(gray.shape, np.uint8) if span == 0 else np.round((values - values.min()) / span * 255).astype(np.uint8)
    return np.repeat(base[:, :, None], 3, axis=2)


def boundaries_overlay(gray: np.ndarray, rows: np.ndarray, visible: np.ndarray, colors: list[list[int]]) -> np.ndarray:
    """1-px boundary lines in visible columns only; gaps stay unbroken image."""
    out = display_base(gray).copy()
    height = out.shape[0]
    for channel, color in enumerate(colors):
        previous = None
        for x in range(rows.shape[1]):
            if not visible[channel, x]:
                previous = None
                continue
            y = int(round(float(rows[channel, x])))
            # Join to the previous visible column so a steep curve stays connected.
            y0, y1 = (y, y) if previous is None else (min(previous, y), max(previous, y))
            y0, y1 = max(y0, 0), min(y1, height - 1)
            if y0 <= y1:
                out[y0:y1 + 1, x] = color
            previous = y
    return out


def layers_overlay(gray: np.ndarray, rows: np.ndarray, visible: np.ndarray, layers: list[dict[str, Any]], boundaries: list[str], alpha: float) -> np.ndarray:
    """Fill each layer between its upper and lower boundary where both are visible.

    A layer without an upper boundary in the model (RNFL: the ILM is not one of
    the ten boundaries) is not filled; nothing is interpolated across gaps.
    """
    base = display_base(gray).astype(np.float32)
    out = base.copy()
    height = out.shape[0]
    for layer in layers:
        if layer["upper_boundary"] is None:
            continue
        upper = boundaries.index(layer["upper_boundary"])
        lower = boundaries.index(layer["lower_boundary"])
        color = np.asarray(layer["color"], np.float32)
        both = visible[upper] & visible[lower]
        for x in np.flatnonzero(both):
            y0 = int(round(float(rows[upper, x]))) + 1
            y1 = int(round(float(rows[lower, x])))
            y0, y1 = max(y0, 0), min(y1, height - 1)
            if y0 <= y1:
                out[y0:y1 + 1, x] = (1 - alpha) * base[y0:y1 + 1, x] + alpha * color
    return np.round(out).astype(np.uint8)
