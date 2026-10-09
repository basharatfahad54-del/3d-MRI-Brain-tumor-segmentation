"""Loss functions tailored to the BraTS class imbalance problem."""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf

__all__ = ["weighted_sparse_ce", "brats_loss", "dice_loss", "combined_loss"]

DEFAULT_CLASS_WEIGHTS: Sequence[float] = (0.1, 0.3, 0.3, 0.3)


def weighted_sparse_ce(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    class_weights: Sequence[float] = DEFAULT_CLASS_WEIGHTS,
) -> tf.Tensor:
    """Class-weighted sparse categorical cross-entropy.

    ``y_true`` holds integer labels of shape ``(batch, D, H, W)`` while
    ``y_pred`` holds softmax probabilities of shape ``(batch, D, H, W, C)``.
    """
    y_true = tf.cast(y_true, tf.int32)
    weights = tf.constant(class_weights, dtype=y_pred.dtype)
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    voxel_weights = tf.gather(weights, y_true)
    return tf.reduce_mean(ce * voxel_weights)


def brats_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    class_weights: Sequence[float] = DEFAULT_CLASS_WEIGHTS,
) -> tf.Tensor:
    """Named wrapper around :func:`weighted_sparse_ce` used across the project."""
    return weighted_sparse_ce(y_true, y_pred, class_weights)


def dice_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    num_classes: int = 4,
    exclude_background: bool = True,
    smooth: float = 1e-6,
) -> tf.Tensor:
    """Soft multi-class Dice loss."""
    y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
    axes = [0, 1, 2, 3]
    intersection = tf.reduce_sum(y_true_onehot * y_pred, axis=axes)
    denominator = tf.reduce_sum(y_true_onehot, axis=axes) + tf.reduce_sum(y_pred, axis=axes)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    if exclude_background:
        dice = dice[1:]
    return 1.0 - tf.reduce_mean(dice)


def combined_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    class_weights: Sequence[float] = DEFAULT_CLASS_WEIGHTS,
    num_classes: int = 4,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> tf.Tensor:
    """Weighted sum of cross-entropy and Dice loss."""
    ce = weighted_sparse_ce(y_true, y_pred, class_weights)
    dc = dice_loss(y_true, y_pred, num_classes=num_classes)
    return ce_weight * ce + dice_weight * dc
