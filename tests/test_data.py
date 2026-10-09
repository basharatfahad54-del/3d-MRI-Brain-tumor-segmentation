from pathlib import Path

from brats_seg import data


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_find_file_supports_both_layouts(tmp_path):
    prefixed = tmp_path / "BraTS19_2013_2_1"
    _touch(prefixed / "BraTS19_2013_2_1_flair.nii.gz")
    assert data.find_file(prefixed, "flair") == prefixed / "BraTS19_2013_2_1_flair.nii.gz"

    plain = tmp_path / "case_plain"
    _touch(plain / "flair.nii.gz")
    assert data.find_file(plain, "flair") == plain / "flair.nii.gz"


def test_discover_cases_requires_segmentation(tmp_path):
    good = tmp_path / "good"
    for modality in data.MODALITIES:
        _touch(good / f"good_{modality}.nii.gz")
    _touch(good / "good_seg.nii.gz")

    no_seg = tmp_path / "no_seg"
    for modality in data.MODALITIES:
        _touch(no_seg / f"no_seg_{modality}.nii.gz")

    cases = data.discover_cases(tmp_path, require_seg=True)
    names = {case.name for case in cases}
    assert names == {"good"}
