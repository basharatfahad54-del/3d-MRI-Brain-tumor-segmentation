"""Inference helpers: run a trained model and export BraTS-labelled NIfTI masks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import data, metrics, preprocessing
from .config import Config

logger = logging.getLogger(__name__)


def custom_objects() -> Dict[str, object]:
    """Objects required to deserialise a saved model."""
    return {
        "MeanIoU": metrics.MeanIoU,
        "dice_whole_tumor": metrics.dice_whole_tumor,
        "dice_tumor_core": metrics.dice_tumor_core,
        "dice_enhancing_tumor": metrics.dice_enhancing_tumor,
    }


def load_trained_model(path: str | Path):
    """Load a full Keras model (``.keras``/``.h5``) with project custom objects."""
    import tensorflow as tf

    return tf.keras.models.load_model(str(path), custom_objects=custom_objects())


def predict_array(model, volume: np.ndarray, target_shape: Optional[Sequence[int]] = None) -> np.ndarray:
    """Predict a contiguous-label mask for a single ``(H, W, D, C)`` volume."""
    volume = np.asarray(volume, dtype=np.float32)
    if target_shape is not None:
        volume = preprocessing.resize_volume(volume, target_shape)

    if volume.ndim == 4:
        inputs = volume[np.newaxis, ...]
    elif volume.ndim == 5:
        inputs = volume
    else:
        raise ValueError(f"Expected a 4D or 5D volume, got shape {volume.shape}")

    probabilities = model.predict(inputs, verbose=0)
    labels = np.argmax(probabilities, axis=-1).astype(np.uint8)[0]
    return labels


def predict_case(
    model,
    case_dir: str | Path,
    cfg: Config,
    output_path: Optional[str | Path] = None,
) -> Dict[str, object]:
    """Predict, evaluate (when a mask exists) and optionally save a case.

    Returns a dictionary with the contiguous/predicted labels and, when a
    ground-truth segmentation is available, the BraTS sub-region Dice scores.
    """
    import tensorflow as tf

    volume, mask = data.load_case(
        case_dir,
        modalities=cfg.data.modalities,
        load_seg=False,
        normalize=cfg.data.normalize,
    )
    predicted = predict_array(model, volume, cfg.data.target_shape)

    result: Dict[str, object] = {"labels": predicted}

    if output_path is not None:
        data.save_nifti(preprocessing.restore_labels(predicted), output_path)
        result["output_path"] = str(output_path)

    try:
        _, ground_truth = data.load_case(
            case_dir, modalities=cfg.data.modalities, load_seg=True, normalize=False
        )
    except FileNotFoundError:
        ground_truth = None

    if ground_truth is not None:
        gt = preprocessing.resize_mask(ground_truth, predicted.shape)
        gt = preprocessing.remap_labels(gt)
        gt_tensor = tf.constant(gt, dtype=tf.int32)
        pred_probs = tf.one_hot(tf.constant(predicted, dtype=tf.int32), cfg.model.num_classes)
        result["dice"] = {
            "whole_tumor": float(metrics.dice_whole_tumor(gt_tensor, pred_probs)),
            "tumor_core": float(metrics.dice_tumor_core(gt_tensor, pred_probs)),
            "enhancing_tumor": float(metrics.dice_enhancing_tumor(gt_tensor, pred_probs)),
        }
    return result


def run_prediction(
    cfg: Config,
    model_path: str | Path,
    cases: Optional[List[Path]] = None,
    output_dir: Optional[str | Path] = None,
) -> List[Dict[str, object]]:
    """Run inference over a set of cases and write masks to ``output_dir``."""
    model = load_trained_model(model_path)
    if cases is None:
        cases = data.discover_cases(cfg.data.data_root, require_seg=False)
    output_dir = Path(output_dir or Path(cfg.output_dir) / "predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for case_dir in cases:
        case_dir = Path(case_dir)
        out_path = output_dir / f"{case_dir.name}_pred.nii.gz"
        result = predict_case(model, case_dir, cfg, output_path=out_path)
        results.append(result)
        logger.info("Predicted %s -> %s", case_dir.name, out_path)
    return results
