"""Training entry point for the 3D U-Net BraTS segmenter."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from . import data, losses, metrics, models
from .config import Config

logger = logging.getLogger(__name__)

LOSSES = ("ce", "dice", "combined")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow for reproducible runs."""
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_loss_fn(
    name: str,
    class_weights: Tuple[float, ...],
    num_classes: int,
) -> Callable:
    """Return a Keras-compatible loss callable by name."""
    if name == "ce":
        return lambda y_true, y_pred: losses.brats_loss(y_true, y_pred, class_weights)
    if name == "dice":
        return lambda y_true, y_pred: losses.dice_loss(y_true, y_pred, num_classes=num_classes)
    if name == "combined":
        return lambda y_true, y_pred: losses.combined_loss(
            y_true, y_pred, class_weights=class_weights, num_classes=num_classes
        )
    raise ValueError(f"Unknown loss '{name}'. Choose from {LOSSES}.")


def build_model(cfg: Config):
    """Instantiate the 3D U-Net described by ``cfg``."""
    input_shape = cfg.resolved_input_shape()
    if input_shape is None:
        raise ValueError(
            "Could not resolve input shape. Set data.target_shape or model.input_shape."
        )
    return models.build_brats_unet(
        input_shape=input_shape,
        base_filters=cfg.model.base_filters,
        dropout_rate=cfg.model.dropout_rate,
        num_classes=cfg.model.num_classes,
        name=cfg.model.name,
    )


def compile_model(model, cfg: Config, loss_name: str = "ce"):
    """Compile ``model`` with the requested loss and BraTS metrics."""
    import tensorflow as tf

    loss_fn = get_loss_fn(loss_name, tuple(cfg.train.class_weights), cfg.model.num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.train.learning_rate),
        loss=loss_fn,
        metrics=[
            "accuracy",
            metrics.MeanIoU(num_classes=cfg.model.num_classes),
            metrics.dice_whole_tumor,
            metrics.dice_tumor_core,
            metrics.dice_enhancing_tumor,
        ],
    )
    return model


def _callbacks(cfg: Config, output_dir: Path) -> List:
    import tensorflow as tf

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.train.reduce_lr_factor,
            patience=cfg.train.reduce_lr_patience,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.train.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
    ]


def train(
    cfg: Config,
    loss_name: str = "ce",
    cases: Optional[List[Path]] = None,
):
    """Run the full training pipeline and persist artefacts under ``output_dir``.

    Returns ``(model, history)``.
    """
    import tensorflow as tf

    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(output_dir / "config.yaml")

    if cases is None:
        cases = data.discover_cases(cfg.data.data_root, require_seg=True)
    if not cases:
        raise RuntimeError(f"No BraTS cases found under {cfg.data.data_root!r}")

    train_cases, val_cases = data.split_cases(cases, cfg.data.val_split, cfg.data.seed)
    logger.info("Discovered %d cases (%d train / %d val)", len(cases), len(train_cases), len(val_cases))

    batch_size = cfg.train.batch_size
    train_ds = data.build_dataset(
        train_cases, cfg.data, batch_size=batch_size, training=True, with_mask=True
    )
    val_ds = None
    if val_cases:
        val_ds = data.build_dataset(
            val_cases, cfg.data, batch_size=batch_size, training=False, with_mask=True
        )

    model = compile_model(build_model(cfg), cfg, loss_name=loss_name)
    model.summary()

    steps_per_epoch = max(1, len(train_cases) // batch_size)
    validation_steps = max(1, len(val_cases) // batch_size) if val_cases else None

    history = model.fit(
        train_ds,
        epochs=cfg.train.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=_callbacks(cfg, output_dir),
    )

    model.save(output_dir / "final_model.keras")
    with open(output_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history.history, handle, indent=2, default=float)

    metadata = {
        "input_shape": list(cfg.resolved_input_shape() or []),
        "num_classes": cfg.model.num_classes,
        "modalities": list(cfg.data.modalities),
        "loss": loss_name,
        "tensorflow": tf.__version__,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    logger.info("Training complete. Artefacts written to %s", output_dir)
    return model, history
