"""Dataset discovery, loading and ``tf.data`` pipeline helpers for BraTS 2019."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import preprocessing
from .config import DataConfig

logger = logging.getLogger(__name__)

MODALITIES: Tuple[str, ...] = ("flair", "t1", "t1ce", "t2")
SEG_KEYS: Tuple[str, ...] = ("seg", "segmentation")
NIFTI_EXTENSIONS: Tuple[str, ...] = (".nii.gz", ".nii")

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - optional at import time
    nib = None


def _require_nibabel() -> None:
    if nib is None:  # pragma: no cover
        raise ImportError(
            "nibabel is required to read NIfTI volumes: pip install nibabel"
        )


def find_file(
    case_dir: str | Path,
    key: str,
    case_id: Optional[str] = None,
) -> Optional[Path]:
    """Locate a NIfTI file inside ``case_dir``.

    Two on-disk conventions are supported:

    * ``<case_dir>/<modality>.nii.gz``            (older BraTS layout)
    * ``<case_dir>/<case_id>_<modality>.nii.gz``  (BraTS 2019 layout)
    """
    case_dir = Path(case_dir)
    if case_id is None:
        case_id = case_dir.name

    candidates: List[str] = []
    if case_id:
        candidates += [f"{case_id}_{key}{ext}" for ext in NIFTI_EXTENSIONS]
    candidates += [f"{key}{ext}" for ext in NIFTI_EXTENSIONS]

    for name in candidates:
        candidate = case_dir / name
        if candidate.exists():
            return candidate
    return None


def _find_seg_file(case_dir: str | Path, case_id: Optional[str] = None) -> Optional[Path]:
    for key in SEG_KEYS:
        path = find_file(case_dir, key, case_id)
        if path is not None:
            return path
    return None


def discover_cases(
    data_root: str | Path,
    require_seg: bool = True,
) -> List[Path]:
    """Recursively find BraTS case directories under ``data_root``."""
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    cases: List[Path] = []
    for path in sorted(p for p in data_root.rglob("*") if p.is_dir()):
        if find_file(path, "flair", path.name) is None:
            continue
        if require_seg and _find_seg_file(path, path.name) is None:
            continue
        cases.append(path)
    return cases


def load_volume(
    case_dir: str | Path,
    modalities: Sequence[str] = MODALITIES,
) -> np.ndarray:
    """Load and stack the requested modalities into ``(H, W, D, C)``."""
    _require_nibabel()
    case_dir = Path(case_dir)
    case_id = case_dir.name
    channels = []
    for modality in modalities:
        file_path = find_file(case_dir, modality, case_id)
        if file_path is None:
            raise FileNotFoundError(
                f"Modality '{modality}' not found in {case_dir}"
            )
        channels.append(np.asarray(nib.load(str(file_path)).get_fdata(), dtype=np.float32))
    if not channels:
        raise ValueError("At least one modality must be requested")
    return np.stack(channels, axis=-1)


def load_mask(case_dir: str | Path) -> np.ndarray:
    """Load the segmentation mask for a case."""
    _require_nibabel()
    case_dir = Path(case_dir)
    seg_path = _find_seg_file(case_dir, case_dir.name)
    if seg_path is None:
        raise FileNotFoundError(f"Segmentation file not found in {case_dir}")
    return np.asarray(nib.load(str(seg_path)).get_fdata(), dtype=np.uint8)


def load_case(
    case_dir: str | Path,
    modalities: Sequence[str] = MODALITIES,
    load_seg: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load a full case, returning ``(volume, mask)``.

    ``volume`` has shape ``(H, W, D, C)`` and is optionally z-score normalised.
    ``mask`` is ``None`` when ``load_seg`` is ``False``.
    """
    volume = load_volume(case_dir, modalities)
    if normalize:
        volume = preprocessing.zscore_normalize(volume)
    mask = load_mask(case_dir) if load_seg else None
    return volume, mask


def save_nifti(
    data: np.ndarray,
    path: str | Path,
    affine: Optional[np.ndarray] = None,
) -> Path:
    """Write ``data`` to a NIfTI file, creating parent directories as needed."""
    _require_nibabel()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.asarray(data), affine if affine is not None else np.eye(4))
    nib.save(image, str(path))
    return path


def split_cases(
    cases: Sequence[Path],
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """Deterministically split cases into train / validation lists."""
    cases = list(cases)
    if not 0.0 <= val_split < 1.0:
        raise ValueError("val_split must be in [0, 1)")
    if len(cases) < 2 or val_split == 0:
        return cases, []

    from sklearn.model_selection import train_test_split

    train, val = train_test_split(cases, test_size=val_split, random_state=seed, shuffle=True)
    return sorted(train), sorted(val)


def _prepare_sample(
    case_dir: Path,
    cfg: DataConfig,
    training: bool,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    volume, mask = load_case(
        case_dir,
        modalities=cfg.modalities,
        load_seg=cfg.load_seg,
        normalize=cfg.normalize,
    )
    if cfg.target_shape is not None:
        volume = preprocessing.resize_volume(volume, cfg.target_shape)
        if mask is not None:
            mask = preprocessing.resize_mask(mask, cfg.target_shape)
    if mask is not None and cfg.remap_labels:
        mask = preprocessing.remap_labels(mask)
    if training and cfg.patch_size is not None:
        volume, mask = preprocessing.random_patch(volume, mask, cfg.patch_size, rng=rng)
    return volume.astype(np.float32), (None if mask is None else mask.astype(np.int32))


def _output_signature(
    cfg: DataConfig,
    with_mask: bool,
):
    import tensorflow as tf

    spatial = tuple(cfg.patch_size or cfg.target_shape or (None, None, None))
    volume_shape = spatial + (len(cfg.modalities),)
    specs = [tf.TensorSpec(shape=volume_shape, dtype=tf.float32)]
    if with_mask:
        specs.append(tf.TensorSpec(shape=spatial, dtype=tf.int32))
    return tuple(specs)


def _generator(
    cases: Sequence[Path],
    cfg: DataConfig,
    training: bool,
    with_mask: bool,
) -> Iterator[Tuple[np.ndarray, ...]]:
    rng = np.random.default_rng(cfg.seed if training else None)
    for case_dir in cases:
        volume, mask = _prepare_sample(Path(case_dir), cfg, training, rng)
        if with_mask:
            if mask is None:
                raise ValueError(
                    f"Case {case_dir} has no segmentation mask but a mask is required"
                )
            yield volume, mask
        else:
            yield (volume,)


def build_dataset(
    cases: Sequence[Path],
    cfg: DataConfig,
    batch_size: int = 1,
    training: bool = False,
    with_mask: bool = True,
):
    """Create a ``tf.data.Dataset`` that loads, preprocesses and batches cases."""
    import tensorflow as tf

    dataset = tf.data.Dataset.from_generator(
        lambda: _generator(cases, cfg, training=training, with_mask=with_mask),
        output_signature=_output_signature(cfg, with_mask),
    )
    if training:
        dataset = dataset.shuffle(cfg.shuffle_buffer, seed=cfg.seed).repeat()
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
