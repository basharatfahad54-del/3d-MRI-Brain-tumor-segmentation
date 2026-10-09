"""Array-level preprocessing utilities for BraTS volumes.

Everything in this module operates on plain NumPy arrays so it can be tested
and reused without TensorFlow or nibabel installed.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:
    from skimage.transform import resize as _skimage_resize
except ImportError:  # pragma: no cover - graceful degradation
    _skimage_resize = None

# BraTS ground-truth label values and their semantics.
BRATS_LABELS: Dict[int, str] = {
    0: "background",
    1: "necrotic_core",
    2: "edema",
    4: "enhancing_tumor",
}
BRATS_CLASS_NAMES: Tuple[str, ...] = (
    "background",
    "necrotic_core",
    "edema",
    "enhancing_tumor",
)

# Contiguous labels used for training (BraTS skips 3, so 4 -> 3).
LABEL_MAP: Dict[int, int] = {0: 0, 1: 1, 2: 2, 4: 3}
INVERSE_LABEL_MAP: Dict[int, int] = {dst: src for src, dst in LABEL_MAP.items()}

# Sub-regions defined by the BraTS challenge (values use contiguous labels).
TUMOR_REGIONS: Dict[str, Tuple[int, ...]] = {
    "whole_tumor": (1, 2, 3),
    "tumor_core": (1, 3),
    "enhancing_tumor": (3,),
}


def _normalise_channel(channel: np.ndarray, nonzero_only: bool, eps: float) -> np.ndarray:
    channel = channel.astype(np.float32, copy=True)
    if nonzero_only:
        mask = channel != 0
        if not np.any(mask):
            return channel
        values = channel[mask]
        mean = float(values.mean())
        std = float(values.std())
        channel[mask] = (values - mean) / (std + eps)
        return channel
    mean = float(channel.mean())
    std = float(channel.std())
    return (channel - mean) / (std + eps)


def zscore_normalize(
    volume: np.ndarray,
    per_channel: bool = True,
    nonzero_only: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """Zero-mean / unit-variance normalisation.

    Parameters
    ----------
    volume:
        Array of shape ``(H, W, D)`` or ``(H, W, D, C)``.
    per_channel:
        When ``True`` and the volume is 4D, each modality is normalised
        independently (recommended for multi-modal MRI).
    nonzero_only:
        Compute statistics over non-zero voxels only, which avoids the large
        background region dominating the statistics.
    """
    volume = np.asarray(volume, dtype=np.float32)
    if per_channel and volume.ndim == 4:
        channels = [
            _normalise_channel(volume[..., c], nonzero_only, eps)
            for c in range(volume.shape[-1])
        ]
        return np.stack(channels, axis=-1)
    return _normalise_channel(volume, nonzero_only, eps)


def resize_volume(
    volume: np.ndarray,
    target_shape: Sequence[int],
    order: int = 1,
) -> np.ndarray:
    """Resample a 3D volume (or 4D multi-channel volume) to ``target_shape``."""
    target_shape = tuple(int(v) for v in target_shape)
    if tuple(volume.shape[:3]) == target_shape and volume.ndim in (3, 4):
        return volume.astype(np.float32, copy=False)
    if _skimage_resize is None:  # pragma: no cover
        raise ImportError(
            "scikit-image is required for resizing: pip install scikit-image"
        )
    if volume.ndim == 4:
        channels = [
            resize_volume(volume[..., c], target_shape, order)
            for c in range(volume.shape[-1])
        ]
        return np.stack(channels, axis=-1).astype(np.float32)
    resized = _skimage_resize(
        volume.astype(np.float32),
        target_shape,
        order=order,
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    )
    return resized.astype(np.float32)


def resize_mask(mask: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Nearest-neighbour resize that preserves integer labels."""
    target_shape = tuple(int(v) for v in target_shape)
    if tuple(mask.shape[:3]) == target_shape:
        return mask.astype(np.uint8, copy=False)
    resized = resize_volume(mask.astype(np.float32), target_shape, order=0)
    return np.rint(resized).astype(np.uint8)


def remap_labels(mask: np.ndarray) -> np.ndarray:
    """Map raw BraTS labels ``{0, 1, 2, 4}`` to contiguous ``{0, 1, 2, 3}``."""
    mask = np.asarray(mask)
    out = np.zeros_like(mask, dtype=np.uint8)
    for src, dst in LABEL_MAP.items():
        out[mask == src] = dst
    return out


def restore_labels(mask: np.ndarray) -> np.ndarray:
    """Inverse of :func:`remap_labels`, recovering values ``{0, 1, 2, 4}``."""
    mask = np.asarray(mask)
    out = np.zeros_like(mask, dtype=np.uint8)
    for src, dst in INVERSE_LABEL_MAP.items():
        out[mask == src] = dst
    return out


def random_patch(
    volume: np.ndarray,
    mask: Optional[np.ndarray],
    patch_size: Sequence[int],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract a random spatial patch, padding when the volume is too small."""
    rng = rng or np.random.default_rng()
    patch_size = tuple(int(v) for v in patch_size)
    spatial = volume.shape[:3]

    pad_width = [(0, max(0, p - s)) for p, s in zip(patch_size, spatial)]
    if any(before or after for before, after in pad_width):
        volume = np.pad(
            volume,
            pad_width + [(0, 0)] * (volume.ndim - 3),
            mode="constant",
        )
        if mask is not None:
            mask = np.pad(mask, pad_width, mode="constant")

    starts = [
        int(rng.integers(0, max(1, vol_dim - patch_dim + 1)))
        for vol_dim, patch_dim in zip(volume.shape[:3], patch_size)
    ]
    slices = tuple(slice(s, s + p) for s, p in zip(starts, patch_size))

    vol_patch = volume[slices]
    mask_patch = mask[slices] if mask is not None else None
    return vol_patch, mask_patch


def center_crop(volume: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Deterministic centre crop to ``target_shape``."""
    target_shape = tuple(int(v) for v in target_shape)
    slices = []
    for current, target in zip(volume.shape[:3], target_shape):
        start = max(0, (current - target) // 2)
        slices.append(slice(start, start + target))
    return volume[tuple(slices)]


def preprocess_volume(
    volume: np.ndarray,
    target_shape: Optional[Sequence[int]] = None,
    normalize: bool = True,
    add_channel: bool = False,
) -> np.ndarray:
    """Normalise and optionally resize a volume."""
    volume = np.asarray(volume, dtype=np.float32)
    if normalize:
        volume = zscore_normalize(volume)
    if target_shape is not None:
        volume = resize_volume(volume, target_shape)
    if add_channel and volume.ndim == 3:
        volume = volume[..., np.newaxis]
    return volume


def preprocess_mask(
    mask: np.ndarray,
    target_shape: Optional[Sequence[int]] = None,
    remap: bool = True,
) -> np.ndarray:
    """Resize (nearest neighbour) and remap a segmentation mask."""
    mask = np.asarray(mask)
    if target_shape is not None:
        mask = resize_mask(mask, target_shape)
    if remap:
        mask = remap_labels(mask)
    return mask.astype(np.uint8)


def preprocess_case(
    volume: np.ndarray,
    mask: Optional[np.ndarray],
    target_shape: Optional[Sequence[int]] = None,
    normalize: bool = True,
    remap: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply the full preprocessing chain to a ``(volume, mask)`` pair."""
    volume = preprocess_volume(volume, target_shape, normalize)
    if mask is not None:
        mask = preprocess_mask(mask, target_shape, remap)
    return volume, mask
