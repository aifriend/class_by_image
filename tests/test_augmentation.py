import sys

import numpy as np

from augmentation import aug_pipeline, clean_augmented


def test_transform_returns_requested_number_of_copies():
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    names, images = aug_pipeline.transform("a.png", image, 3)
    assert names == ["a.png"] * 3
    assert all(img.shape == image.shape for img in images)


def test_clean_augmented_moves_matching_files(tmp_path, monkeypatch):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "keep.png").write_bytes(b"x")
    (folder / "left_a1.png").write_bytes(b"x")
    monkeypatch.setattr(sys, "argv", ["clean_augmented", str(folder)])
    clean_augmented.main()
    assert (folder / "keep.png").exists()
    assert (tmp_path / "empty" / "left_a1.png").exists()
