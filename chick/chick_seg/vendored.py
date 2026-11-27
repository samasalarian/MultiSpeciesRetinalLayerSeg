"""Inference code copied verbatim from the validated chick training project.

Every function and class below is an unmodified copy (via inspect.getsource) of
the code used to train and cross-validate the released models:

- preprocessing: read_tiff, to_grayscale, robust_normalize, automatic_retina_crop, _padding
- dataset: horizontal_positions, reconstruct_predictions
- losses: vertical_probabilities
- gap_aware: GapAwareOutput, BoundaryVisibilityModel, _huber, _decode_visible_run, _ordered_column_projection, decode_gap_aware_segmentwise

Do not edit these copies; the release test checks they reproduce the project
pipeline bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import ndimage, signal
from torch import nn


class PipelineConfig:
    """Attribute container standing in for the project's PipelineConfig."""

    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)

# --- verbatim from preprocessing.py: read_tiff
def read_tiff(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(path) as image:
        pages = getattr(image, "n_frames", 1)
        if pages != 1:
            raise ValueError(f"Multi-page TIFF stacks are unsupported: {path} has {pages} pages")
        mode = image.mode
        array = np.asarray(image)
    return array, {"mode": mode, "pages": pages, "dtype": str(array.dtype)}


# --- verbatim from preprocessing.py: to_grayscale
def to_grayscale(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"Unsupported OCT image dimensions: {array.shape}")
    rgb = array[..., :3]
    if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2]):
        return rgb[..., 0]
    values = np.tensordot(rgb.astype(np.float32), [0.299, 0.587, 0.114], axes=([-1], [0]))
    return values.astype(np.float32)


# --- verbatim from preprocessing.py: robust_normalize
def robust_normalize(gray: np.ndarray, config: PipelineConfig) -> tuple[np.ndarray, float, float]:
    values = gray.astype(np.float32)
    low, high = np.percentile(
        values, [config.normalization_low_percentile, config.normalization_high_percentile]
    )
    if high <= low:
        raise ValueError(f"Degenerate normalization range {low}..{high}")
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    if config.gaussian_filter_sigma > 0:
        normalized = ndimage.gaussian_filter(normalized, sigma=config.gaussian_filter_sigma)
    return normalized.astype(np.float32), float(low), float(high)


# --- verbatim from preprocessing.py: automatic_retina_crop
def automatic_retina_crop(
    gray: np.ndarray,
    crop_height: int = 300,
    target_outer_retina_row: int = 212,
) -> tuple[np.ndarray, dict[str, int]]:
    """Crop a full OCT frame using image intensity only.

    The broad, bright outer-retinal complex is located from a smoothed
    horizontal median profile. It is placed at the same approximate vertical
    position used by the manually prepared training crops. No annotation or
    prediction is used to choose the crop.
    """
    if gray.ndim != 2:
        raise ValueError(f"automatic_retina_crop expects HxW grayscale data, got {gray.shape}")
    height, width = gray.shape
    if height < crop_height:
        raise ValueError(
            f"Image height {height} is smaller than requested crop height {crop_height}"
        )
    profile = ndimage.gaussian_filter1d(
        np.median(gray.astype(np.float32), axis=1), sigma=3.0, mode="nearest"
    )
    # The full-frame exports can have a brighter inner-retinal surface above
    # the outer-retinal complex. Search the lower retinal zone, find broad
    # local peaks, and select the lowest peak that is close to the strongest.
    search_start = max(0, int(round(height * 0.27)))
    search_stop = min(height, int(round(height * 0.55)))
    if search_stop <= search_start:
        raise ValueError(f"Invalid automatic-crop search range for shape {gray.shape}")
    local_profile = profile[search_start:search_stop]
    peak_offsets, _ = signal.find_peaks(
        local_profile, distance=max(10, crop_height // 12), prominence=1.0
    )
    if peak_offsets.size:
        peak_rows = peak_offsets + search_start
        strongest = float(profile[peak_rows].max())
        plausible = peak_rows[profile[peak_rows] >= 0.90 * strongest]
        outer_retina_row = int(plausible.max())
    else:
        outer_retina_row = search_start + int(np.argmax(local_profile))
    y0 = int(np.clip(outer_retina_row - target_outer_retina_row, 0, height - crop_height))
    y1 = y0 + crop_height
    return gray[y0:y1], {
        "x0": 0,
        "x1": width,
        "y0": y0,
        "y1": y1,
        "detected_outer_retina_row": outer_retina_row,
        "target_outer_retina_row": target_outer_retina_row,
    }


# --- verbatim from preprocessing.py: _padding
def _padding(
    height: int,
    width: int,
    divisor: int,
    minimum_height: int = 0,
) -> tuple[int, int, int, int]:
    padded_height = max(minimum_height, height + (-height) % divisor)
    add_h, add_w = padded_height - height, (-width) % divisor
    top, left = add_h // 2, add_w // 2
    return top, add_h - top, left, add_w - left


# --- verbatim from dataset.py: horizontal_positions
def horizontal_positions(width: int, patch_width: int, stride: int) -> list[int]:
    if patch_width >= width:
        return [0]
    positions = list(range(0, width - patch_width + 1, stride))
    final = width - patch_width
    if positions[-1] != final:
        positions.append(final)
    return positions


# --- verbatim from dataset.py: reconstruct_predictions
def reconstruct_predictions(
    patch_predictions: list[np.ndarray], records: list[PatchRecord], shape: tuple[int, int]
) -> np.ndarray:
    """Average overlapping CxHxW patch predictions into a full image."""
    if not patch_predictions:
        raise ValueError("No patch predictions were supplied")
    height, width = shape
    channels = patch_predictions[0].shape[0]
    total = np.zeros((channels, height, width), dtype=np.float32)
    count = np.zeros((1, height, width), dtype=np.float32)
    for prediction, record in zip(patch_predictions, records):
        total[:, :, record.x0:record.x1] += prediction[..., : record.x1 - record.x0]
        count[:, :, record.x0:record.x1] += 1
    if np.any(count == 0):
        raise ValueError("Patch reconstruction left uncovered pixels")
    return total / count


# --- verbatim from losses.py: vertical_probabilities
def vertical_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Normalize each batch/boundary/column independently over image height."""
    if logits.ndim != 4:
        raise ValueError(f"Expected BxCxHxW logits, got {tuple(logits.shape)}")
    return torch.softmax(logits, dim=2)


# --- verbatim from gap_aware.py: GapAwareOutput
@dataclass
class GapAwareOutput:
    location_logits: torch.Tensor  # Bx5xHxW
    visibility_logits: torch.Tensor  # Bx5xW


# --- verbatim from gap_aware.py: BoundaryVisibilityModel
class BoundaryVisibilityModel(nn.Module):
    """Add an independent per-boundary visibility head to a location U-Net."""

    def __init__(self, location_model: nn.Module, boundaries: int = 5) -> None:
        super().__init__()
        self.location_model = location_model
        self.boundaries = boundaries
        self.visibility_image_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=5, padding=4, dilation=2),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
        )
        self.visibility_projection = nn.Sequential(
            nn.Conv1d(16 + 2 * boundaries, 48, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(48, boundaries, kernel_size=7, padding=3),
        )

    def forward(self, image: torch.Tensor) -> GapAwareOutput:
        location = self.location_model(image)
        image_features = self.visibility_image_encoder(image).mean(dim=2)
        # Visibility may consume localization confidence, but its gradients
        # must not alter the historical segmentation objective.
        distribution = vertical_probabilities(location.detach())
        height = location.shape[2]
        peak = distribution.amax(dim=2) * height
        entropy = -(
            distribution * distribution.clamp_min(1e-8).log()
        ).sum(dim=2) / np.log(height)
        visibility_features = torch.cat([image_features, peak, entropy], dim=1)
        visibility = self.visibility_projection(visibility_features)
        return GapAwareOutput(location, visibility)


# --- verbatim from gap_aware.py: _huber
def _huber(values: np.ndarray, delta: float) -> np.ndarray:
    scaled = np.abs(values) / max(float(delta), 1e-6)
    return np.where(scaled <= 1.0, 0.5 * scaled**2, scaled - 0.5)


# --- verbatim from gap_aware.py: _decode_visible_run
def _decode_visible_run(
    probabilities: np.ndarray,
    top_k: int,
    continuity_weight: float,
    continuity_delta: float,
    transition_cap: float,
) -> np.ndarray:
    """Decode one uninterrupted C-independent run with a small candidate Viterbi."""
    height, width = probabilities.shape
    if width == 1:
        return np.asarray([int(np.argmax(probabilities[:, 0]))], dtype=np.float32)
    count = min(max(int(top_k), 1), height)
    candidates: list[np.ndarray] = []
    unary: list[np.ndarray] = []
    for x in range(width):
        column = probabilities[:, x]
        rows = np.argpartition(column, -count)[-count:]
        rows = np.sort(rows.astype(np.int32))
        candidates.append(rows)
        unary.append(-np.log(np.clip(column[rows], 1e-8, 1.0)))
    costs = unary[0].copy()
    backpointers: list[np.ndarray] = []
    for x in range(1, width):
        previous, current = candidates[x - 1], candidates[x]
        transition = _huber(
            current[:, None] - previous[None, :], continuity_delta
        )
        transition = np.minimum(transition, float(transition_cap))
        combined = costs[None, :] + continuity_weight * transition
        pointer = np.argmin(combined, axis=1)
        costs = unary[x] + combined[np.arange(current.size), pointer]
        backpointers.append(pointer)
    selected = int(np.argmin(costs))
    path = np.empty(width, dtype=np.float32)
    path[-1] = candidates[-1][selected]
    for x in range(width - 2, -1, -1):
        selected = int(backpointers[x][selected])
        path[x] = candidates[x][selected]
    return path


# --- verbatim from gap_aware.py: _ordered_column_projection
def _ordered_column_projection(
    probabilities: np.ndarray,
    active: np.ndarray,
    anchors: np.ndarray,
    min_gap: int,
    anchor_weight: float,
    anchor_delta: float,
) -> np.ndarray:
    """Exact ordered dynamic program for the visible boundaries in one column."""
    height = probabilities.shape[1]
    rows = np.arange(height, dtype=np.float32)
    costs = []
    for channel, anchor in zip(active, anchors):
        unary = -np.log(np.clip(probabilities[channel], 1e-8, 1.0))
        costs.append(unary + anchor_weight * _huber(rows - anchor, anchor_delta))
    dp = costs[0].copy()
    pointers: list[np.ndarray] = []
    gap = max(int(min_gap), 0)
    for boundary_cost in costs[1:]:
        prefix_value = np.minimum.accumulate(dp)
        prefix_index = np.empty(height, dtype=np.int32)
        best = 0
        for row in range(height):
            if dp[row] < dp[best]:
                best = row
            prefix_index[row] = best
        next_dp = np.full(height, np.inf, dtype=np.float64)
        pointer = np.full(height, -1, dtype=np.int32)
        first = gap
        if first < height:
            allowed = np.arange(first, height) - gap
            next_dp[first:] = boundary_cost[first:] + prefix_value[allowed]
            pointer[first:] = prefix_index[allowed]
        dp = next_dp
        pointers.append(pointer)
    selected = int(np.argmin(dp))
    if not np.isfinite(dp[selected]):
        return anchors.astype(np.float32)
    result = np.empty(active.size, dtype=np.float32)
    result[-1] = selected
    for index in range(active.size - 2, -1, -1):
        selected = int(pointers[index][selected])
        result[index] = selected
    return result


# --- verbatim from gap_aware.py: decode_gap_aware_segmentwise
def decode_gap_aware_segmentwise(
    location_probabilities: np.ndarray,
    visibility_probabilities: np.ndarray,
    thresholds: np.ndarray,
    *,
    top_k: int = 8,
    continuity_weight: float = 0.35,
    continuity_delta: float = 2.0,
    transition_cap: float = 8.0,
    min_gap: int = 0,
    anchor_weight: float = 0.2,
    anchor_delta: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode ordered, robust curves independently inside each visible run.

    Visibility gaps are hard reset points: continuity is never evaluated across
    them and returned rendered coordinates remain NaN there.
    """
    location = np.asarray(location_probabilities, dtype=np.float32)
    visibility = np.asarray(visibility_probabilities, dtype=np.float32)
    if location.ndim != 3:
        raise ValueError("Location probabilities must have CxHxW shape")
    channels, _, width = location.shape
    if visibility.shape != (channels, width):
        raise ValueError("Visibility probabilities must have CxW shape")
    threshold_array = np.asarray(thresholds, dtype=np.float32)
    if threshold_array.shape != (channels,):
        raise ValueError("Thresholds must have one value per boundary")
    visible = visibility >= threshold_array[:, None]
    coordinates = np.argmax(location, axis=1).astype(np.float32)
    for channel in range(channels):
        columns = np.flatnonzero(visible[channel])
        runs = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
        for run in runs:
            if not run.size:
                continue
            decoded = _decode_visible_run(
                location[channel][:, run], top_k, continuity_weight,
                continuity_delta, transition_cap,
            )
            coordinates[channel, run] = decoded
    # Enforce only anatomical order at each column. Missing boundaries remain
    # absent and do not induce interpolation or continuity across their gaps.
    for x in range(width):
        active = np.flatnonzero(visible[:, x])
        if active.size < 2:
            continue
        coordinates[active, x] = _ordered_column_projection(
            location[:, :, x], active, coordinates[active, x], min_gap,
            anchor_weight, anchor_delta,
        )
    rendered = coordinates.copy()
    rendered[~visible] = np.nan
    return coordinates, rendered, visible
