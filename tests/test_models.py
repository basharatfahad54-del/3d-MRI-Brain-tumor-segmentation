import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from brats_seg import losses, models  # noqa: E402


def test_build_unet_output_shape():
    model = models.build_brats_unet(input_shape=(16, 16, 16, 4), base_filters=4, dropout_rate=0.0)
    assert model.input_shape == (None, 16, 16, 16, 4)
    assert model.output_shape == (None, 16, 16, 16, 4)


def test_unet_single_forward_pass():
    model = models.build_brats_unet(input_shape=(16, 16, 16, 2), base_filters=4, dropout_rate=0.0)
    batch = np.random.rand(1, 16, 16, 16, 2).astype(np.float32)
    output = model.predict(batch, verbose=0)
    assert output.shape == (1, 16, 16, 16, 4)
    assert np.allclose(output.sum(axis=-1), 1.0, atol=1e-4)


def test_weighted_cross_entropy_is_scalar():
    y_true = np.random.randint(0, 4, size=(1, 8, 8, 8)).astype(np.int32)
    y_pred = tf.random.uniform((1, 8, 8, 8, 4))
    y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
    value = losses.weighted_sparse_ce(y_true, y_pred)
    assert value.shape.rank == 0


def test_dice_loss_is_zero_for_perfect_prediction():
    labels = np.random.randint(0, 4, size=(1, 8, 8, 8)).astype(np.int32)
    probs = tf.one_hot(labels, 4, dtype=tf.float32)
    value = float(losses.dice_loss(labels, probs))
    assert value == pytest.approx(0.0, abs=1e-5)
