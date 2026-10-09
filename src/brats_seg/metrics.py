"""Evaluation metrics for BraTS segmentation (Dice per sub-region and mIoU)."""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf

__all__ = [
    "region_dice",
    "dice_whole_tumor",
    "dice_tumor_core",
    "dice_enhancing_tumor",
    "MeanIoU",
]

# Contiguous label convention produced by ``preprocessing.remap_labels``.
WHOLE_TUMOR: Sequence[int] = (1, 2, 3)
TUMOR_CORE: Sequence[int] = (1, 3)
ENHANCING_TUMOR: Sequence[int] = (3,)


def _labels(y_pred: tf.Tensor) -> tf.Tensor:
    """Convert softmax probabilities to integer labels."""
    return tf.argmax(y_pred, axis=-1, output_type=tf.int32)


def _region_mask(labels: tf.Tensor, classes: Sequence[int]) -> tf.Tensor:
    mask = tf.zeros_like(labels, dtype=tf.bool)
    for cls in classes:
        mask = tf.logical_or(mask, tf.equal(labels, cls))
    return tf.cast(mask, tf.float32)


def region_dice(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    classes: Sequence[int],
    smooth: float = 1e-6,
) -> tf.Tensor:
    """Dice coefficient for a union of label classes."""
    y_true = tf.cast(y_true, tf.int32)
    pred_labels = _labels(y_pred)
    true_mask = _region_mask(y_true, classes)
    pred_mask = _region_mask(pred_labels, classes)
    intersection = tf.reduce_sum(true_mask * pred_mask)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(true_mask) + tf.reduce_sum(pred_mask) + smooth
    )


def dice_whole_tumor(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return region_dice(y_true, y_pred, WHOLE_TUMOR)


def dice_tumor_core(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return region_dice(y_true, y_pred, TUMOR_CORE)


def dice_enhancing_tumor(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return region_dice(y_true, y_pred, ENHANCING_TUMOR)


class MeanIoU(tf.keras.metrics.MeanIoU):
    """Thin wrapper so the metric can be serialised alongside the model."""

    def __init__(self, num_classes: int = 4, name: str = "mean_iou", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def get_config(self):
        return {"num_classes": self.num_classes, "name": self.name}
