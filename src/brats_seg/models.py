"""3D U-Net implementation for multi-class BraTS segmentation."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

__all__ = ["conv_block", "encoder_block", "decoder_block", "build_brats_unet"]


def conv_block(
    x: tf.Tensor,
    filters: int,
    dropout_rate: float = 0.2,
    name: Optional[str] = None,
) -> tf.Tensor:
    """Two 3x3x3 convolutions with batch norm, ReLU and dropout."""
    x = layers.Conv3D(
        filters, 3, padding="same", kernel_initializer="he_normal", name=_sub(name, "conv1")
    )(x)
    x = layers.BatchNormalization(name=_sub(name, "bn1"))(x)
    x = layers.Activation("relu", name=_sub(name, "relu1"))(x)
    x = layers.Dropout(dropout_rate, name=_sub(name, "drop"))(x)

    x = layers.Conv3D(
        filters, 3, padding="same", kernel_initializer="he_normal", name=_sub(name, "conv2")
    )(x)
    x = layers.BatchNormalization(name=_sub(name, "bn2"))(x)
    x = layers.Activation("relu", name=_sub(name, "relu2"))(x)
    return x


def _sub(name: Optional[str], suffix: str) -> Optional[str]:
    return None if name is None else f"{name}_{suffix}"


def _crop_to_match(x: tf.Tensor, skip: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Crop the larger of two feature maps so their spatial dims line up."""
    if x.shape[1] is None or skip.shape[1] is None:
        return x, skip
    x_spatial = tuple(int(v) for v in x.shape[1:4])
    skip_spatial = tuple(int(v) for v in skip.shape[1:4])
    if x_spatial == skip_spatial:
        return x, skip

    target = [min(a, b) for a, b in zip(x_spatial, skip_spatial)]

    def _crop(tensor: tf.Tensor) -> tf.Tensor:
        croppings = []
        for current, goal in zip(tensor.shape[1:4], target):
            diff = int(current) - goal
            croppings.append((diff // 2, diff - diff // 2))
        if any(before or after for before, after in croppings):
            return layers.Cropping3D(cropping=croppings)(tensor)
        return tensor

    return _crop(x), _crop(skip)


def encoder_block(
    x: tf.Tensor,
    filters: int,
    dropout_rate: float = 0.2,
    name: Optional[str] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Convolutional block followed by max pooling. Returns ``(skip, pooled)``."""
    features = conv_block(x, filters, dropout_rate, name=name)
    pooled = layers.MaxPool3D(pool_size=2, name=_sub(name, "pool"))(features)
    return features, pooled


def decoder_block(
    x: tf.Tensor,
    skip: tf.Tensor,
    filters: int,
    dropout_rate: float = 0.2,
    name: Optional[str] = None,
) -> tf.Tensor:
    """Transposed-convolution upsampling fused with an encoder skip connection."""
    x = layers.Conv3DTranspose(
        filters, kernel_size=2, strides=2, padding="same", name=_sub(name, "up")
    )(x)

    if skip.shape[-1] is not None and skip.shape[-1] != filters:
        skip = layers.Conv3D(filters, 1, padding="same", name=_sub(name, "skip_proj"))(skip)

    x, skip = _crop_to_match(x, skip)
    x = layers.Concatenate(name=_sub(name, "concat"))([x, skip])
    return conv_block(x, filters, dropout_rate, name=name)


def build_brats_unet(
    input_shape: Sequence[int] = (128, 128, 128, 4),
    base_filters: int = 32,
    dropout_rate: float = 0.2,
    num_classes: int = 4,
    name: str = "3D_UNet_BraTS",
) -> models.Model:
    """Build a 3D U-Net for multi-class BraTS segmentation.

    Parameters
    ----------
    input_shape:
        ``(D, H, W, C)``. ``C`` should match the number of MRI modalities.
    base_filters:
        Number of filters in the first encoder level; doubles each level.
    num_classes:
        Number of output classes (4 for BraTS).
    """
    inputs = layers.Input(shape=tuple(input_shape), name="input")

    f1, p1 = encoder_block(inputs, base_filters, dropout_rate, name="enc1")
    f2, p2 = encoder_block(p1, base_filters * 2, dropout_rate, name="enc2")
    f3, p3 = encoder_block(p2, base_filters * 4, dropout_rate, name="enc3")
    f4, p4 = encoder_block(p3, base_filters * 8, dropout_rate, name="enc4")

    bottleneck = conv_block(p4, base_filters * 16, dropout_rate, name="bottleneck")

    d4 = decoder_block(bottleneck, f4, base_filters * 8, dropout_rate, name="dec4")
    d3 = decoder_block(d4, f3, base_filters * 4, dropout_rate, name="dec3")
    d2 = decoder_block(d3, f2, base_filters * 2, dropout_rate, name="dec2")
    d1 = decoder_block(d2, f1, base_filters, dropout_rate, name="dec1")

    outputs = layers.Conv3D(num_classes, 1, activation="softmax", name="output")(d1)
    return models.Model(inputs, outputs, name=name)
