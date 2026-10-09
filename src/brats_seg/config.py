"""Typed, YAML-friendly configuration objects for the segmentation pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

try:  # PyYAML is optional at runtime, required for YAML config files.
    import yaml
except ImportError:  # pragma: no cover - handled at call sites
    yaml = None


def _sequence_to_tuple(value: Any) -> Any:
    """Normalise YAML lists into tuples for shape-like fields."""
    if isinstance(value, list):
        return tuple(value)
    return value


@dataclass
class DataConfig:
    """How the BraTS dataset is discovered and preprocessed."""

    data_root: str = "data/raw"
    modalities: Sequence[str] = ("flair", "t1", "t1ce", "t2")
    target_shape: Optional[Sequence[int]] = (128, 128, 128)
    patch_size: Optional[Sequence[int]] = None
    normalize: bool = True
    remap_labels: bool = True
    load_seg: bool = True
    val_split: float = 0.2
    seed: int = 42
    shuffle_buffer: int = 8


@dataclass
class ModelConfig:
    """3D U-Net hyper-parameters."""

    input_shape: Optional[Sequence[int]] = None
    base_filters: int = 32
    dropout_rate: float = 0.2
    num_classes: int = 4
    name: str = "3D_UNet_BraTS"


@dataclass
class TrainConfig:
    """Optimisation and callback settings."""

    epochs: int = 100
    batch_size: int = 2
    learning_rate: float = 1e-4
    class_weights: Sequence[float] = (0.1, 0.3, 0.3, 0.3)
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.1
    shuffle_buffer: int = 8


@dataclass
class Config:
    """Top-level configuration aggregate."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "outputs"
    seed: int = 42

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "Config":
        raw = raw or {}
        data = DataConfig(**_filter_fields(DataConfig, raw.get("data")))
        model = ModelConfig(**_filter_fields(ModelConfig, raw.get("model")))
        train = TrainConfig(**_filter_fields(TrainConfig, raw.get("train")))
        cfg = cls(
            data=data,
            model=model,
            train=train,
            output_dir=raw.get("output_dir", "outputs"),
            seed=raw.get("seed", 42),
        )
        cfg._normalise()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        if yaml is None:  # pragma: no cover
            raise ImportError("PyYAML is required to load YAML configs: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        if yaml is None:  # pragma: no cover
            raise ImportError("PyYAML is required to write YAML configs: pip install pyyaml")
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

    # ------------------------------------------------------------------ #
    # Derived values
    # ------------------------------------------------------------------ #
    def resolved_input_shape(self) -> Optional[Tuple[int, ...]]:
        """Return the 4D model input shape ``(D, H, W, C)``.

        Falls back to ``target_shape + (num_modalities,)`` when the model
        config does not pin an explicit shape.
        """
        if self.model.input_shape:
            return tuple(int(v) for v in self.model.input_shape)
        if self.data.target_shape and self.data.modalities:
            return tuple(int(v) for v in self.data.target_shape) + (len(self.data.modalities),)
        return None

    def _normalise(self) -> None:
        for obj, names in (
            (self.data, ("modalities", "target_shape", "patch_size")),
            (self.model, ("input_shape",)),
            (self.train, ("class_weights",)),
        ):
            for name in names:
                setattr(obj, name, _sequence_to_tuple(getattr(obj, name)))


def _filter_fields(dataclass_type: type, values: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not values:
        return {}
    allowed = set(dataclass_type.__dataclass_fields__)
    return {key: value for key, value in values.items() if key in allowed}


def load_config(path: Optional[str | Path]) -> Config:
    """Load a config from ``path`` or return defaults when ``path`` is ``None``."""
    if path is None:
        return Config()
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return Config.from_yaml(path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            return Config.from_dict(json.load(handle))
    raise ValueError(f"Unsupported config format: {path.suffix}")
