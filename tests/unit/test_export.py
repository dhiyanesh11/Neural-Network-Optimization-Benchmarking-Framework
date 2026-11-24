"""Unit tests for ONNX graph export and runtime parity."""

import numpy as np
import onnx
import torch
from torch import nn

from nnbench.export import ORTModel, export_onnx, quantize_onnx_dynamic


def test_onnx_export_validates_and_matches_pytorch(tmp_path) -> None:
    torch.manual_seed(41)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).eval()
    sample = torch.rand(2, 4)
    output_path = export_onnx(model, sample, tmp_path / "model.onnx")
    onnx.checker.check_model(onnx.load(output_path))
    runtime = ORTModel(output_path)
    actual = runtime(sample)
    with torch.inference_mode():
        expected = model(sample)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-4, rtol=1e-4)
    assert runtime(torch.rand(5, 4)).shape == (5, 3)


def test_dynamic_quantized_onnx_runs(tmp_path) -> None:
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2)).eval()
    sample = torch.rand(3, 4)
    exported = export_onnx(model, sample, tmp_path / "model.onnx")
    quantized = quantize_onnx_dynamic(exported, tmp_path / "model_int8.onnx")
    runtime = ORTModel(quantized)
    assert runtime(sample).shape == (3, 2)
