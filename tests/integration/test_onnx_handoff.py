"""Integration check for a PyTorch and ONNX model using common benchmark keys."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nnbench.export import ORTModel, export_onnx
from nnbench.runner import BenchmarkRunner


def test_onnx_model_benchmarks_with_same_result_schema(tmp_path) -> None:
    torch.manual_seed(43)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).eval()
    inputs = torch.rand(8, 4)
    labels = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=4)
    path = export_onnx(model, inputs[:2], tmp_path / "toy.onnx")
    runner = BenchmarkRunner(device="cpu")
    config = {"warmup_iters": 1, "measure_iters": 2, "model": "toy"}
    pytorch_result = runner.run(model, loader, config)
    onnx_result = runner.run(ORTModel(path), loader, config)
    assert set(onnx_result) == set(pytorch_result)
    assert onnx_result["model_size_mb"] > 0
