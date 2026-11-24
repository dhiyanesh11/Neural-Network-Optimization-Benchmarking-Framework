"""ONNX Runtime inference and dynamic quantization helpers."""

from pathlib import Path

import numpy as np
import torch
from torch import nn


class ORTModel(nn.Module):
    """Small PyTorch-like adapter around an ONNX Runtime inference session."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        provider: str = "CPUExecutionProvider",
        graph_optimization: str = "all",
    ) -> None:
        super().__init__()
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("ONNX Runtime requires the 'export' extra.") from exc
        options = ort.SessionOptions()
        levels = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        try:
            options.graph_optimization_level = levels[graph_optimization.lower()]
        except KeyError as exc:
            raise ValueError("graph_optimization must be disable, basic, extended, or all") from exc
        available = ort.get_available_providers()
        if provider not in available:
            raise ValueError(
                f"Provider '{provider}' is unavailable. Available providers: {available}"
            )
        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=[provider]
        )
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        if kwargs:
            inputs = kwargs
        elif len(args) == 1 and isinstance(args[0], dict):
            inputs = args[0]
        else:
            inputs = dict(zip(self.input_names, args, strict=True))
        feed = {
            name: (
                value.detach().cpu().numpy()
                if isinstance(value, torch.Tensor)
                else np.asarray(value)
            )
            for name, value in inputs.items()
        }
        outputs = self.session.run([self.output_name], feed)
        return torch.from_numpy(outputs[0])

    def model_size_bytes(self) -> int:
        """Return the ONNX artifact file size."""
        return self.model_path.stat().st_size


def quantize_onnx_dynamic(model_path: str | Path, output_path: str | Path) -> Path:
    """Apply ONNX Runtime dynamic QInt8 weight quantization."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise ImportError("ONNX Runtime quantization requires the 'export' extra.") from exc
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(model_path), str(destination), weight_type=QuantType.QInt8)
    return destination
