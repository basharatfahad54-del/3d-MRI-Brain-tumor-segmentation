#!/usr/bin/env python
"""Wrapper: ``python scripts/train.py --help``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brats_seg.cli import train_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(train_main())
