import numpy as np
import pytest
import torch
from PIL import Image

from classification.train_doc_class import GbcCnnService


@pytest.fixture
def tiny_dataset(tmp_path):
    """Two classes with four small random images each, in ImageFolder layout."""
    rng = np.random.default_rng(0)
    root = tmp_path / "dataset"
    for cls in ("invoice", "receipt"):
        (root / cls).mkdir(parents=True)
        for i in range(4):
            arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(root / cls / f"{i}.png")
    return root


def make_service(tiny_dataset, tmp_path):
    service = GbcCnnService(data_dir=str(tiny_dataset), model_path=str(tmp_path / "model"))
    service.BATCH_SIZE = 4
    service.load_data()
    return service


def test_load_data_splits_and_finds_classes(tiny_dataset, tmp_path):
    service = make_service(tiny_dataset, tmp_path)
    assert service.class_name_list == ["invoice", "receipt"]
    assert service.dataset_size_dict == {"train": 7, "val": 1}


def test_checkpoint_is_state_dict_and_reloads_with_weights_only(tiny_dataset, tmp_path):
    service = make_service(tiny_dataset, tmp_path)
    service.make_model(pretrained=False)
    # the file the trainer wrote is the same one it loads next time
    checkpoint = torch.load(service.path_to_model, map_location="cpu", weights_only=True)
    assert isinstance(checkpoint, dict)

    reloaded = make_service(tiny_dataset, tmp_path)
    reloaded.make_model(pretrained=False)
    for key, value in service.model.state_dict().items():
        assert torch.equal(value, reloaded.model.state_dict()[key])


def test_one_epoch_trains_and_saves(tiny_dataset, tmp_path):
    service = make_service(tiny_dataset, tmp_path)
    service.make_model(pretrained=False)
    service.train_model(_num_epochs=1)
    assert service.path_to_model.endswith("_img_v3.model")
    torch.load(service.path_to_model, map_location="cpu", weights_only=True)
