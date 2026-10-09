"""Command-line entry points for the BraTS segmentation pipeline.

Each ``*_main`` function is wired to a console script in ``pyproject.toml`` and
is also reachable through the thin wrappers in ``scripts/``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from .config import Config, load_config

DEFAULT_CONFIG = Path("configs/unet3d.yaml")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config(config_path: Optional[str]) -> Config:
    if config_path and Path(config_path).exists():
        return load_config(config_path)
    return Config()


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to a YAML/JSON config file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")


# --------------------------------------------------------------------------- #
# preprocess
# --------------------------------------------------------------------------- #
def preprocess_main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess BraTS cases into .npz archives")
    _add_common(parser)
    parser.add_argument("--input-dir", default=None, help="Override data.data_root")
    parser.add_argument("--output-dir", default="data/processed", help="Where to write .npz files")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit the number of cases")
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    from . import data, preprocessing

    cfg = _load_config(args.config)
    if args.input_dir:
        cfg.data.data_root = args.input_dir

    cases = data.discover_cases(cfg.data.data_root, require_seg=True)
    if args.max_cases:
        cases = cases[: args.max_cases]
    if not cases:
        raise SystemExit(f"No cases found under {cfg.data.data_root!r}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for case_dir in cases:
        volume, mask = data.load_case(
            case_dir, modalities=cfg.data.modalities, load_seg=True, normalize=False
        )
        volume, mask = preprocessing.preprocess_case(
            volume,
            mask,
            target_shape=cfg.data.target_shape,
            normalize=cfg.data.normalize,
            remap=cfg.data.remap_labels,
        )
        npz_path = output_dir / f"{Path(case_dir).name}.npz"
        import numpy as np

        np.savez_compressed(npz_path, volume=volume, mask=mask)
        logging.info("Wrote %s", npz_path)
    return 0


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def train_main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Train the 3D U-Net BraTS segmenter")
    _add_common(parser)
    parser.add_argument("--data-dir", default=None, help="Override data.data_root")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--loss", choices=("ce", "dice", "combined"), default="ce")
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    from .train import train

    cfg = _load_config(args.config)
    if args.data_dir:
        cfg.data.data_root = args.data_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size

    train(cfg, loss_name=args.loss)
    return 0


# --------------------------------------------------------------------------- #
# predict
# --------------------------------------------------------------------------- #
def predict_main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Run inference and export predicted masks")
    _add_common(parser)
    parser.add_argument("--model", required=True, help="Path to a saved .keras/.h5 model")
    parser.add_argument("--data-dir", default=None, help="Override data.data_root")
    parser.add_argument("--output-dir", default=None, help="Where to write predicted NIfTI masks")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    from . import data
    from .predict import run_prediction

    cfg = _load_config(args.config)
    if args.data_dir:
        cfg.data.data_root = args.data_dir

    cases = data.discover_cases(cfg.data.data_root, require_seg=False)
    if args.max_cases:
        cases = cases[: args.max_cases]

    results = run_prediction(cfg, args.model, cases=cases, output_dir=args.output_dir)
    for result in results:
        dice = result.get("dice")
        if dice:
            logging.info("%s", dice)
    return 0


# --------------------------------------------------------------------------- #
# visualize
# --------------------------------------------------------------------------- #
def visualize_main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Render a visualisation of a single case")
    _add_common(parser)
    parser.add_argument("--case", required=True, help="Path to a BraTS case directory")
    parser.add_argument("--output", default=None, help="Optional PNG output path")
    parser.add_argument("--slice", type=int, default=None, help="Slice index to display")
    parser.add_argument("--modality", type=int, default=0, help="Modality channel index")
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    from . import data, visualization

    cfg = _load_config(args.config)
    volume, mask = data.load_case(case_dir=args.case, modalities=cfg.data.modalities, load_seg=True)
    figure = visualization.show_slices(
        volume,
        mask,
        slice_idx=args.slice,
        modality_idx=args.modality,
        save_path=args.output,
        title=Path(args.case).name,
    )
    if args.output is None:
        import matplotlib.pyplot as plt

        plt.show()
    return 0
