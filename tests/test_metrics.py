import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from brats_seg import metrics  # noqa: E402


def test_perfect_prediction_scores_one():
    labels = np.random.randint(0, 4, size=(2, 8, 8, 8)).astype(np.int32)
    probabilities = tf.one_hot(labels, 4)
    for region in (metrics.WHOLE_TUMOR, metrics.TUMOR_CORE, metrics.ENHANCING_TUMOR):
        score = float(metrics.region_dice(labels, probabilities, region))
        assert score == pytest.approx(1.0, abs=1e-5)


def test_disjoint_prediction_scores_zero():
    y_true = np.zeros((1, 4, 4, 4), dtype=np.int32)
    y_true[0, 0, 0, 0] = 1
    y_pred = np.zeros((1, 4, 4, 4, 4), dtype=np.float32)
    y_pred[0, 1, 1, 1, 3] = 1.0  # predict enhancing where truth is necrotic
    score = float(metrics.dice_whole_tumor(y_true, y_pred))
    assert score == pytest.approx(0.0, abs=1e-5)


def test_mean_iou_is_serialisable():
    metric = metrics.MeanIoU(num_classes=4)
    config = metric.get_config()
    assert config["num_classes"] == 4
