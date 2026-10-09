# 3D MRI Brain Tumor Segmentation (BraTS 2019)

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)
![License](https://img.shields.io/badge/license-BSL--1.0-green)
![CI](https://github.com/yourusername/3D-MRI-Brain-Tumor-Segmentation/actions/workflows/ci.yml/badge.svg)

An end-to-end, config-driven deep learning pipeline for **multi-class brain tumor
segmentation** from 3D MRI scans, built around a 3D U-Net and trained on the
[BraTS 2019](https://www.med.upenn.edu/cbica/brats2019/) dataset.

The project is packaged as a reusable Python library (`brats_seg`) with a clean
data / model / training / inference separation, reproduction-friendly CLI tools,
unit tests and continuous integration.

## Features

- **3D U-Net** with batch normalisation, dropout and robust skip-connection shape matching.
- **Multi-modal input**: FLAIR, T1, T1ce and T2 stacked as 4 input channels.
- **Multi-class output**: background, necrotic core, edema and enhancing tumor.
- **BraTS-aware preprocessing**: per-channel z-score normalisation, nearest-neighbour mask resampling and label remapping, plus optional random patch extraction for memory-friendly training.
- **Class-imbalance aware losses**: weighted cross-entropy, soft Dice, or a combined loss.
- **Challenge metrics**: Dice for the whole tumor, tumor core and enhancing tumor regions, and mean IoU.
- **Artefact tracking**: best weights, final model, training history, CSV log, resolved config and metadata.
- **NIfTI export**: predictions are written back as NIfTI masks with the original BraTS label values.

## Dataset

The pipeline expects the standard BraTS 2019 directory layout, one folder per patient.
Both common naming conventions are supported:

```
data/raw/
├── BraTS19_2013_2_1/
│   ├── BraTS19_2013_2_1_flair.nii.gz
│   ├── BraTS19_2013_2_1_t1.nii.gz
│   ├── BraTS19_2013_2_1_t1ce.nii.gz
│   ├── BraTS19_2013_2_1_t2.nii.gz
│   └── BraTS19_2013_2_1_seg.nii.gz
├── BraTS19_2013_10_1/
│   └── ...
```

Older layouts that use bare names (`flair.nii.gz`, `seg.nii.gz`) are also detected.
Raw datasets are not distributed here; download BraTS 2019 and place it under `data/raw/`
(or point `data.data_root` anywhere else).

## Installation

```bash
git clone https://github.com/yourusername/3D-MRI-Brain-Tumor-Segmentation.git
cd 3D-MRI-Brain-Tumor-Segmentation

# Option A: pip
python -m venv .venv && .venv\Scripts\activate      # Windows
# source .venv/bin/activate                          # macOS / Linux
pip install -r requirements-dev.txt
pip install -e .

# Option B: conda
conda env create -f environment.yml
conda activate brats_seg
```

A GPU is strongly recommended for training. TensorFlow will use it automatically
when a compatible CUDA runtime is available.

## Quickstart

### 1. Visualise a case

```bash
python scripts/visualize.py --case data/raw/BraTS19_2013_2_1 --output preview.png
```

### 2. Preprocess (optional cache)

```bash
python scripts/preprocess.py --input-dir data/raw --output-dir data/processed
```

### 3. Train

```bash
python scripts/train.py --data-dir data/raw --output-dir outputs --epochs 100 --loss ce
```

Training writes the following artefacts to `outputs/`:

```
outputs/
├── best_model.weights.h5    # best validation-loss weights
├── final_model.keras        # full saved model
├── history.json             # per-epoch metrics
├── training_log.csv         # CSVLogger output
├── config.yaml              # fully resolved configuration
└── metadata.json
```

### 4. Predict

```bash
python scripts/predict.py --model outputs/final_model.keras \
    --data-dir data/raw --output-dir outputs/predictions
```

Predicted masks are exported as `*_pred.nii.gz` with BraTS labels `{0, 1, 2, 4}`.
When a ground-truth `seg` file is present, the per-region Dice scores are logged.

### 5. Interactive notebook

`notebooks/quickstart.ipynb` walks through the model, preprocessing and
visualisation using synthetic data, so it runs without the dataset.

## Project structure

```
3D-MRI-Brain-Tumor-Segmentation/
├── configs/
│   └── unet3d.yaml            # default pipeline configuration
├── notebooks/
│   └── quickstart.ipynb       # runnable package tour (synthetic data)
├── scripts/                   # thin CLI wrappers
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── visualize.py
├── src/brats_seg/
│   ├── config.py              # typed configuration objects
│   ├── data.py                # dataset discovery, loading, tf.data pipeline
│   ├── preprocessing.py       # normalisation, resizing, label remapping
│   ├── models.py              # 3D U-Net
│   ├── losses.py              # weighted CE, Dice, combined
│   ├── metrics.py             # region Dice, mean IoU
│   ├── visualization.py       # slice / prediction / history plots
│   ├── train.py               # training driver
│   ├── predict.py             # inference + NIfTI export
│   └── cli.py                 # argparse entry points
├── tests/                     # pytest suite
├── legacy/
│   └── original_prototype.py  # original research prototype notebook
├── environment.yml
├── requirements.txt
└── pyproject.toml
```

After `pip install -e .` the CLIs are also available as console scripts:
`brats-preprocess`, `brats-train`, `brats-predict`, `brats-visualize`.

## Configuration

All hyper-parameters live in [`configs/unet3d.yaml`](configs/unet3d.yaml) and can be
overridden from the CLI. Key options:

| Section  | Option          | Description                                              |
|----------|-----------------|----------------------------------------------------------|
| `data`   | `modalities`    | Input modalities (default `flair, t1, t1ce, t2`).        |
| `data`   | `target_shape`  | Resampled volume size, e.g. `[128, 128, 128]`.           |
| `data`   | `patch_size`    | Set (e.g. `[64, 64, 64]`) for random-patch training.     |
| `model`  | `base_filters`  | Filters in the first U-Net level (doubles per level).    |
| `model`  | `num_classes`   | Output classes (4 for BraTS).                            |
| `train`  | `learning_rate` | Adam learning rate.                                      |
| `train`  | `class_weights` | Per-class weights in the cross-entropy loss.             |

## Metrics

Evaluation reports Dice scores for the three BraTS sub-regions:

| Region            | Labels (remapped) | Description                       |
|-------------------|-------------------|-----------------------------------|
| Whole tumor (WT)  | 1, 2, 3           | Necrotic + edema + enhancing.     |
| Tumor core (TC)   | 1, 3              | Necrotic + enhancing.             |
| Enhancing (ET)    | 3                 | Enhancing tumor.                  |

Measured scores depend on the dataset split, preprocessing and training budget, so
this repository does not ship pre-baked numbers. Reproduce them with `scripts/train.py`
and inspect `outputs/history.json` or the per-case Dice logged by `scripts/predict.py`.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
ruff check src tests
```

Tests that exercise the model and metrics are skipped automatically when
TensorFlow is not installed.

## Acknowledgements

- [BraTS 2019 Challenge](https://www.med.upenn.edu/cbica/brats2019/) for the dataset and evaluation protocol.
- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation* (2015).
- Isensee et al., *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation* (2021).

## License

Released under the Boost Software License 1.0 - see [LICENSE](LICENSE) for details.
