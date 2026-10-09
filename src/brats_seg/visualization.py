"""Matplotlib visualisation helpers for MRI volumes and predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np

import matplotlib

matplotlib.use("Agg", force=False)  # noqa: E402 - must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

from .preprocessing import BRATS_LABELS  # noqa: E402

# Discrete colours for the three tumour classes (background stays transparent).
CLASS_COLORS = {
    1: (1.0, 0.15, 0.15),  # necrotic core - red
    2: (0.15, 0.7, 0.15),  # edema - green
    3: (0.15, 0.35, 1.0),  # enhancing tumour - blue
}
CLASS_TITLES = {1: "Necrotic core", 2: "Edema", 3: "Enhancing tumour"}


def _select_slice(volume: np.ndarray, slice_idx: Optional[int], axis: int) -> int:
    if slice_idx is not None:
        return int(slice_idx)
    return volume.shape[axis] // 2


def _take(volume: np.ndarray, slice_idx: int, axis: int) -> np.ndarray:
    return np.take(volume, slice_idx, axis=axis)


def show_slices(
    volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    modality_idx: int = 0,
    axis: int = 2,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot a 2D slice of a volume with an optional overlaid segmentation mask.

    Parameters
    ----------
    volume:
        ``(H, W, D)`` or ``(H, W, D, C)`` array. 4D volumes are assumed to be
        channel-last, matching :func:`brats_seg.data.load_case`.
    mask:
        Optional integer label map (values ``0, 1, 2, 4`` or ``0, 1, 2, 3``).
    slice_idx:
        Slice to display along ``axis`` (defaults to the middle slice).
    modality_idx:
        Which channel to display when ``volume`` is 4D.
    axis:
        Spatial axis to slice along.
    """
    volume = np.asarray(volume)
    if volume.ndim == 4:
        volume = volume[..., modality_idx]
    slice_idx = _select_slice(volume, slice_idx, axis)

    image_slice = _take(volume, slice_idx, axis)
    mask_slice = _take(mask, slice_idx, axis) if mask is not None else None

    n_cols = 2 if mask_slice is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6), squeeze=False)
    axes = axes.ravel()

    axes[0].imshow(image_slice.T, cmap="gray", origin="lower", interpolation="none")
    axes[0].set_title(f"MRI (modality {modality_idx}) - slice {slice_idx}")
    axes[0].axis("off")

    if mask_slice is not None:
        axes[1].imshow(image_slice.T, cmap="gray", origin="lower", interpolation="none")
        overlay = np.ma.masked_where(mask_slice.T == 0, mask_slice.T)
        axes[1].imshow(overlay, cmap=_class_cmap(), origin="lower", interpolation="none", alpha=0.55)
        axes[1].set_title(f"Segmentation - slice {slice_idx}")
        axes[1].axis("off")
        _add_legend(axes[1])

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _class_cmap():
    from matplotlib.colors import ListedColormap

    return ListedColormap(["black", *[CLASS_COLORS[c] for c in (1, 2, 3)]])


def _add_legend(ax) -> None:
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=CLASS_COLORS[cls], label=CLASS_TITLES[cls]) for cls in (1, 2, 3)
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.6)


def plot_prediction(
    volume: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    prediction: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    modality_idx: int = 0,
    axis: int = 2,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Show the MRI slice next to the ground truth and/or prediction."""
    volume = np.asarray(volume)
    if volume.ndim == 4:
        volume = volume[..., modality_idx]
    slice_idx = _select_slice(volume, slice_idx, axis)
    image_slice = _take(volume, slice_idx, axis)

    panels = []
    if ground_truth is not None:
        panels.append(("Ground truth", _take(ground_truth, slice_idx, axis)))
    if prediction is not None:
        panels.append(("Prediction", _take(prediction, slice_idx, axis)))

    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(6 * (len(panels) + 1), 6), squeeze=False)
    axes = axes.ravel()

    axes[0].imshow(image_slice.T, cmap="gray", origin="lower", interpolation="none")
    axes[0].set_title("MRI")
    axes[0].axis("off")

    for ax, (label, mask_slice) in zip(axes[1:], panels):
        ax.imshow(image_slice.T, cmap="gray", origin="lower", interpolation="none")
        overlay = np.ma.masked_where(mask_slice.T == 0, mask_slice.T)
        ax.imshow(overlay, cmap=_class_cmap(), origin="lower", interpolation="none", alpha=0.55)
        ax.set_title(label)
        ax.axis("off")
        _add_legend(ax)

    fig.suptitle(f"Slice {slice_idx}")
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_history(
    history: Union[Dict[str, list], object],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot training/validation loss and the available metrics."""
    data = history.history if hasattr(history, "history") else dict(history or {})
    metrics = [key for key in data if not key.startswith("val_")]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * max(1, len(metrics)), 4), squeeze=False)
    axes = axes.ravel()

    for ax, key in zip(axes, metrics):
        ax.plot(data[key], label=key)
        val_key = f"val_{key}"
        if val_key in data:
            ax.plot(data[val_key], label=val_key)
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def label_counts(mask: np.ndarray) -> Dict[str, int]:
    """Return the voxel count for each known BraTS class."""
    mask = np.asarray(mask)
    counts: Dict[str, int] = {}
    for value, name in BRATS_LABELS.items():
        counts[name] = int(np.count_nonzero(mask == value))
    return counts
