from brats_seg.config import Config


def test_default_config_resolves_input_shape():
    cfg = Config()
    assert cfg.resolved_input_shape() == (128, 128, 128, 4)


def test_from_dict_normalises_lists_to_tuples():
    cfg = Config.from_dict(
        {
            "data": {"target_shape": [64, 64, 64], "modalities": ["t1ce"]},
            "model": {"base_filters": 8},
            "output_dir": "runs/demo",
        }
    )
    assert cfg.data.target_shape == (64, 64, 64)
    assert cfg.data.modalities == ("t1ce",)
    assert cfg.model.base_filters == 8
    assert cfg.resolved_input_shape() == (64, 64, 64, 1)
    assert cfg.output_dir == "runs/demo"


def test_to_dict_roundtrip():
    cfg = Config()
    cfg.data.val_split = 0.25
    restored = Config.from_dict(cfg.to_dict())
    assert restored.data.val_split == 0.25
    assert restored.train.batch_size == cfg.train.batch_size
