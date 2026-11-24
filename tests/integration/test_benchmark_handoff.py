"""Integration check for registry-to-benchmark metrics."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nnbench.runner import BenchmarkRunner


class _ToyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def test_benchmark_runner_returns_standard_metrics() -> None:
    torch.manual_seed(7)
    inputs = torch.rand(8, 4)
    labels = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=4)
    result = BenchmarkRunner(device="cpu").run(
        _ToyClassifier(), loader, {"warmup_iters": 1, "measure_iters": 3, "model": "toy"}
    )
    assert 0 <= result["accuracy"] <= 1
    assert result["num_samples"] == 8
    assert result["model_size_mb"] > 0
    assert result["parameter_count"] == 10
    assert result["latency_p99_ms"] >= result["latency_p50_ms"]
