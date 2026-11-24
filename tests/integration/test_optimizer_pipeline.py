"""Integration test: every optimizer feeds the common benchmark runner."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nnbench.metrics.size import model_size_bytes
from nnbench.optimizers import get_optimizer, list_optimizers
from nnbench.runner import BenchmarkRunner


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 512), nn.ReLU(), nn.Linear(512, 2))


def _loader() -> DataLoader:
    torch.manual_seed(23)
    return DataLoader(TensorDataset(torch.rand(8, 4), torch.randint(0, 2, (8,))), batch_size=4)


@pytest.mark.parametrize("name", list_optimizers())
def test_each_optimizer_produces_a_benchmarkable_model(name: str) -> None:
    torch.manual_seed(29)
    model = _model().eval()
    loader = _loader()
    kwargs = {"backend": "eager"} if name == "torch_compile" else {}
    if name == "knowledge_distillation":
        kwargs = {"teacher": _model().eval(), "train_loader": loader, "epochs": 1}
    if name == "qat_int8":
        kwargs = {"epochs": 1}
    optimized = get_optimizer(name, **kwargs).apply(model, calib_loader=loader)
    result = BenchmarkRunner(device="cpu").run(
        optimized,
        loader,
        {"warmup_iters": 1, "measure_iters": 2, "model": "toy", "optimizer": name},
    )
    assert 0 <= result["accuracy"] <= 1
    assert result["model_size_mb"] > 0
    assert result["optimizer"] == name
    if name == "ptq_dynamic_int8":
        assert model_size_bytes(optimized) < model_size_bytes(model)
