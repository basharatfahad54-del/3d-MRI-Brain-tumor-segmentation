"""BraTS 2019 3D MRI brain tumor segmentation.

A modular TensorFlow/Keras pipeline that covers the full workflow:

* dataset discovery and loading (:mod:`brats_seg.data`)
* volume / mask preprocessing (:mod:`brats_seg.preprocessing`)
* a configurable 3D U-Net (:mod:`brats_seg.models`)
* BraTS specific losses and metrics (:mod:`brats_seg.losses`, :mod:`brats_seg.metrics`)
* training and inference drivers (:mod:`brats_seg.train`, :mod:`brats_seg.predict`)
* slice visualisation helpers (:mod:`brats_seg.visualization`)

The package is intentionally decoupled from TensorFlow at import time so that
lightweight components (preprocessing, configuration, visualisation) can be
used and tested without a GPU stack installed.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
