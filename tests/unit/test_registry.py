"""Unit tests for the model and dataset registry."""

import pytest
import torch

from nnbench.registry import get_dataloader, get_model, list_datasets, list_models


@pytest.mark.parametrize("name", ["mobilenet_v2", "resnet18"])
def test_image_models_return_ten_class_logits(name: str) -> None:
    model = get_model(name).eval()
    with torch.inference_mode():
        output = model(torch.rand(2, 3, 32, 32))
    assert output.shape == (2, 10)


def test_distilbert_model_returns_two_class_logits() -> None:
    model = get_model("distilbert_sst2").eval()
    with torch.inference_mode():
        output = model(torch.randint(0, 1000, (2, 16)), torch.ones(2, 16, dtype=torch.long))
    assert output.shape == (2, 2)


def test_registered_models_and_datasets() -> None:
    assert {"mobilenet_v2", "resnet18", "distilbert_sst2"} <= set(list_models())
    assert {"cifar10", "sst2"} <= set(list_datasets())


def test_synthetic_dataloaders_have_expected_batch_shapes() -> None:
    images, labels = next(iter(get_dataloader("cifar10", batch_size=4, synthetic=True)))
    assert images.shape == (4, 3, 32, 32)
    assert labels.shape == (4,)
    text = next(iter(get_dataloader("sst2", batch_size=4, synthetic=True)))
    assert text["input_ids"].shape == (4, 32)
    assert text["labels"].shape == (4,)


def test_unknown_model_has_helpful_error() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("no_such_model")
