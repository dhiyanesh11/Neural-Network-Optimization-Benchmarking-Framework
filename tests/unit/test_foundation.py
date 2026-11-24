"""Tests for the project foundation and experiment configuration."""

import pytest
import torch

from nnbench.config import load_config
from nnbench.seed import set_seed


def test_set_seed_repeats_torch_random_tensor() -> None:
    set_seed(42)
    first = torch.rand(8)
    set_seed(42)
    second = torch.rand(8)
    assert torch.equal(first, second)


def test_sample_config_loads_and_has_required_keys() -> None:
    config = load_config("configs/experiment/default.yaml")
    assert {"seed", "model", "dataset", "optimizers"} <= config.keys()
    assert config["seed"] == 42


def test_config_rejects_missing_required_keys(tmp_path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("model: resnet18\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required keys"):
        load_config(config_path)
