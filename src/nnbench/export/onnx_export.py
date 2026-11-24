"""PyTorch to ONNX export utilities."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn


class _ExportAdapter(nn.Module):
    def __init__(self, model: nn.Module, input_names: list[str]) -> None:
        super().__init__()
        self.model = model
        self.input_names = input_names

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        output = self.model(**dict(zip(self.input_names, inputs, strict=True)))
        if hasattr(output, "logits"):
            output = output.logits
        if isinstance(output, Mapping):
            output = output["logits"]
        return output


def _prepare_inputs(example_inputs: Any) -> tuple[tuple[torch.Tensor, ...], list[str]]:
    if isinstance(example_inputs, Mapping):
        names = list(example_inputs)
        values = tuple(example_inputs[name] for name in names)
    elif isinstance(example_inputs, (tuple, list)):
        names = [f"input_{index}" for index in range(len(example_inputs))]
        values = tuple(example_inputs)
    else:
        names = ["input"]
        values = (example_inputs,)
    if not values or not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError("example_inputs must contain one or more tensors.")
    return values, names


def export_onnx(
    model: nn.Module,
    example_inputs: Any,
    output_path: str | Path,
    *,
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Export an eval-mode model with named inputs and a dynamic batch axis."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    inputs, input_names = _prepare_inputs(example_inputs)
    model_copy = deepcopy(model).cpu().eval()
    adapter = _ExportAdapter(model_copy, input_names).eval()
    dynamic_axes = (
        {name: {0: "batch"} for name in input_names} | {"logits": {0: "batch"}}
        if dynamic_batch
        else None
    )
    with torch.inference_mode():
        torch.onnx.export(
            adapter,
            inputs,
            str(path),
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            dynamo=False,
        )
    validate_onnx(path)
    return path


def validate_onnx(path: str | Path) -> None:
    """Validate an ONNX model with the ONNX checker."""
    try:
        import onnx
    except ImportError as exc:
        raise ImportError("ONNX validation requires the 'export' extra.") from exc
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
