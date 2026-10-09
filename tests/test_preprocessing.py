import numpy as np

from brats_seg import preprocessing as pp


def test_remap_restore_roundtrip():
    mask = np.array([0, 1, 2, 4, 4, 2, 1, 0], dtype=np.uint8)
    remapped = pp.remap_labels(mask)
    assert set(np.unique(remapped)) <= {0, 1, 2, 3}
    assert np.array_equal(pp.restore_labels(remapped), mask)


def test_zscore_normalization_per_channel():
    rng = np.random.default_rng(0)
    volume = rng.normal(5.0, 3.0, size=(8, 8, 8, 3)).astype(np.float32)
    normalized = pp.zscore_normalize(volume, nonzero_only=False)
    assert normalized.shape == volume.shape
    assert np.allclose(normalized.mean(axis=(0, 1, 2)), 0.0, atol=1e-5)
    assert np.allclose(normalized.std(axis=(0, 1, 2)), 1.0, atol=1e-5)


def test_resize_volume_and_mask():
    volume = np.random.rand(32, 32, 32, 4).astype(np.float32)
    mask = np.random.randint(0, 4, size=(32, 32, 32)).astype(np.uint8)
    resized_volume = pp.resize_volume(volume, (16, 16, 16))
    resized_mask = pp.resize_mask(mask, (16, 16, 16))
    assert resized_volume.shape == (16, 16, 16, 4)
    assert resized_mask.shape == (16, 16, 16)
    assert set(np.unique(resized_mask)) <= {0, 1, 2, 3}


def test_random_patch_shape_and_padding():
    volume = np.random.rand(20, 20, 20, 2).astype(np.float32)
    mask = np.random.randint(0, 4, size=(20, 20, 20)).astype(np.uint8)
    vol_patch, mask_patch = pp.random_patch(volume, mask, (16, 16, 16))
    assert vol_patch.shape == (16, 16, 16, 2)
    assert mask_patch.shape == (16, 16, 16)

    big_patch, big_mask = pp.random_patch(volume, mask, (30, 30, 30))
    assert big_patch.shape == (30, 30, 30, 2)
    assert big_mask.shape == (30, 30, 30)


def test_preprocess_case_remaps_labels():
    volume = np.random.rand(16, 16, 16, 2).astype(np.float32)
    mask = np.array([0, 1, 2, 4], dtype=np.uint8)
    mask = np.resize(mask, (16, 16, 16))
    processed_volume, processed_mask = pp.preprocess_case(
        volume, mask, target_shape=(8, 8, 8), normalize=True, remap=True
    )
    assert processed_volume.shape == (8, 8, 8, 2)
    assert processed_mask.shape == (8, 8, 8)
    assert processed_mask.max() <= 3
