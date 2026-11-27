"""Input contract, preprocessing, inference and decoding for the chick models.

All numerical steps call the verbatim project code in `vendored.py`; this file
only validates inputs, wires those steps together exactly as the validated
pipeline did, and maps results back to the input image frame.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .vendored import (
    BoundaryVisibilityModel,
    PipelineConfig,
    _padding,
    automatic_retina_crop,
    decode_gap_aware_segmentwise,
    horizontal_positions,
    read_tiff,
    reconstruct_predictions,
    robust_normalize,
    to_grayscale,
)

PACKAGE_DIR = Path(__file__).resolve().parents[1]


class InputError(ValueError):
    """The input image does not meet the validated input contract."""


@dataclass
class _Patch:
    x0: int
    x1: int


def load_config(model: str) -> dict[str, Any]:
    path = PACKAGE_DIR / f"config_{model}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing model configuration: {path.name}")
    return yaml.safe_load(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(cfg: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, str]:
    """Build the architecture and load the released weights with strict keys."""
    import segmentation_models_pytorch as smp

    checkpoint = PACKAGE_DIR / cfg["checkpoint"]
    digest = sha256(checkpoint)
    if digest != cfg["checkpoint_sha256"]:
        raise RuntimeError(
            f"{checkpoint.name} SHA-256 {digest} does not match {cfg['checkpoint_sha256']}"
        )
    arch = cfg["architecture"]
    channels = len(cfg["boundaries"])
    location = smp.Unet(
        encoder_name=arch["encoder_name"],
        encoder_weights=None,  # trained weights come from the checkpoint
        in_channels=arch["in_channels"],
        classes=channels,
        activation=None,
    )
    model = BoundaryVisibilityModel(location, channels)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    key = cfg["checkpoint_key"]
    if not isinstance(state, dict) or state.get(key) is None:
        raise RuntimeError(f"{checkpoint.name} has no '{key}' weights")
    model.load_state_dict(state[key], strict=True)
    return model.to(device).eval(), digest


def read_and_validate(path: Path, cfg: dict[str, Any], auto_crop: bool) -> tuple[np.ndarray, dict[str, Any]]:
    """Enforce the validated input contract and return the HxW grayscale image."""
    spec = cfg["input"]
    if path.suffix.casefold() not in {".tif", ".tiff"}:
        raise InputError(f"Input must be a single-page TIFF (.tif/.tiff), got '{path.suffix}'")
    try:
        array, info = read_tiff(path)
    except ValueError as error:
        raise InputError(str(error)) from error
    if str(array.dtype) not in spec["dtypes"]:
        raise InputError(f"Pixel type must be one of {spec['dtypes']}, got {array.dtype}")
    if array.ndim == 3:
        if array.shape[2] not in (3, 4):
            raise InputError(f"Expected 1, 3 or 4 channels, got {array.shape[2]}")
        rgb = array[..., :3]
        if not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2])):
            raise InputError("Colour input is not supported: R, G and B must be identical (grayscale OCT)")
        if array.shape[2] == 4 and not np.all(array[..., 3] == np.iinfo(array.dtype).max):
            raise InputError("RGBA input must be fully opaque")
    elif array.ndim != 2:
        raise InputError(f"Unsupported image dimensions {array.shape}")
    gray = to_grayscale(array)
    height, width = gray.shape
    if width != spec["width"]:
        raise InputError(f"Image width must be {spec['width']} px (A-scans), got {width}")
    low, high = spec["precropped_height_range"]
    if auto_crop:
        minimum = spec["auto_crop"]["crop_height"]
        if height < minimum:
            raise InputError(f"--auto-crop needs a full frame at least {minimum} px tall, got {height}")
    elif not low <= height <= high:
        hint = " Use --auto-crop for full-height frames." if height > high else ""
        raise InputError(
            f"Pre-cropped input height must be {low}-{high} px, got {height}.{hint}"
        )
    info["channels"] = 1 if array.ndim == 2 else int(array.shape[2])
    return gray, info


def preprocess(gray: np.ndarray, cfg: dict[str, Any], auto_crop: bool) -> tuple[np.ndarray, dict[str, Any]]:
    """Validated preprocessing: optional intensity-only crop, robust
    normalization over the whole (cropped) image, centred reflect padding."""
    pre = cfg["preprocessing"]
    pc = PipelineConfig(
        normalization_low_percentile=pre["normalization_low_percentile"],
        normalization_high_percentile=pre["normalization_high_percentile"],
        gaussian_filter_sigma=pre["gaussian_filter_sigma"],
    )
    if auto_crop:
        ac = cfg["input"]["auto_crop"]
        cropped, crop = automatic_retina_crop(
            gray, crop_height=ac["crop_height"], target_outer_retina_row=ac["target_outer_retina_row"]
        )
    else:
        cropped, crop = gray, {"x0": 0, "x1": int(gray.shape[1]), "y0": 0, "y1": int(gray.shape[0])}
    try:
        image, low, high = robust_normalize(cropped, pc)
    except ValueError as error:
        raise InputError(f"Image has no usable intensity range: {error}") from error
    top, bottom, left, right = _padding(*image.shape, pre["pad_divisor"], pre["minimum_padded_height"])
    image = np.pad(image, ((top, bottom), (left, right)), mode="reflect")
    return image, {
        "crop": {k: int(v) for k, v in crop.items()},
        "padding": {"top": top, "bottom": bottom, "left": left, "right": right},
        "normalization": {"low": low, "high": high},
        "network_input_shape": list(image.shape),
    }


def infer(model: torch.nn.Module, image: np.ndarray, cfg: dict[str, Any], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping horizontal patches, one at a time, averaged (validated `predict`)."""
    inf = cfg["inference"]
    records = [
        _Patch(x0, min(x0 + inf["patch_width"], image.shape[1]))
        for x0 in horizontal_positions(image.shape[1], inf["patch_width"], inf["stride"])
    ]
    location: list[np.ndarray] = []
    visibility: list[np.ndarray] = []
    with torch.inference_mode():
        for record in records:
            patch = torch.from_numpy(image[:, record.x0:record.x1][None].astype(np.float32))
            out = model(patch[None].to(device))
            location.append(torch.sigmoid(out.location_logits)[0].cpu().numpy())
            p = torch.sigmoid(out.visibility_logits)[0].cpu().numpy()
            visibility.append(np.repeat(p[:, None, :], image.shape[0], axis=1))
    loc = reconstruct_predictions(location, records, image.shape)
    vis = reconstruct_predictions(visibility, records, image.shape)[:, 0]
    return loc, vis


def decode(loc: np.ndarray, vis: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dec = cfg["decoder"]
    thresholds = np.asarray([cfg["visibility_thresholds"][b] for b in cfg["boundaries"]], np.float32)
    return decode_gap_aware_segmentwise(
        loc, vis, thresholds,
        top_k=dec["top_k"], continuity_weight=dec["continuity_weight"],
        continuity_delta=dec["continuity_delta_px"], transition_cap=dec["transition_cap"],
        min_gap=dec["minimum_gap_px"], anchor_weight=dec["anchor_weight"], anchor_delta=dec["anchor_delta"],
    )


def run(model: torch.nn.Module, gray: np.ndarray, cfg: dict[str, Any], device: torch.device, auto_crop: bool) -> dict[str, Any]:
    """Return boundaries and visibility in the coordinates of the input image."""
    image, prep = preprocess(gray, cfg, auto_crop)
    loc, vis = infer(model, image, cfg, device)
    _, rendered, visible = decode(loc, vis, cfg)
    pad = prep["padding"]
    xs = slice(pad["left"], image.shape[1] - pad["right"] if pad["right"] else None)
    rows = rendered[:, xs] - pad["top"] + prep["crop"]["y0"]
    return {
        "rows": rows.astype(np.float32),  # NaN where the boundary is predicted not visible
        "visibility": vis[:, xs].astype(np.float32),
        "visible": visible[:, xs],
        "preprocessing": prep,
    }


def ordering_report(rows: np.ndarray, visible: np.ndarray) -> dict[str, int | bool]:
    """Visible boundaries must be finite and strictly top-to-bottom in every column."""
    nonfinite = int((~np.isfinite(rows[visible])).sum())
    hidden_not_nan = int(np.isfinite(rows[~visible]).sum())
    pairs = violations = 0
    for x in range(rows.shape[1]):
        ys = rows[visible[:, x], x]
        pairs += max(ys.size - 1, 0)
        violations += int(np.sum(np.diff(ys) <= 0))
    return {
        "visible_values_nonfinite": nonfinite,
        "hidden_values_not_nan": hidden_not_nan,
        "adjacent_visible_pairs_checked": pairs,
        "order_violations_or_crossings": violations,
        "passed": nonfinite == 0 and hidden_not_nan == 0 and violations == 0,
    }
