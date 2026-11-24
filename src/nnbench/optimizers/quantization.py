"""Dynamic and FX static post-training quantization plugins."""

from typing import Any

import torch
from torch import nn

from nnbench.optimizers.base import BaseOptimizer
from nnbench.runner.benchmark import _batch_parts, _forward


class DynamicInt8Optimizer(BaseOptimizer):
    """Quantize eligible Linear and recurrent weights dynamically to INT8."""

    name = "ptq_dynamic_int8"

    def apply(self, model: nn.Module, calib_loader: Any = None) -> nn.Module:
        candidate = self.clone(model).cpu().eval()
        return torch.ao.quantization.quantize_dynamic(
            candidate, {nn.Linear, nn.LSTM}, dtype=torch.qint8
        )


class StaticInt8Optimizer(BaseOptimizer):
    """Calibrate and convert a traceable model with FX static INT8 quantization."""

    name = "ptq_static_int8"

    def __init__(self, backend: str = "x86", calib_batches: int = 10) -> None:
        self.backend = backend
        self.calib_batches = calib_batches

    def apply(self, model: nn.Module, calib_loader: Any = None) -> nn.Module:
        if calib_loader is None:
            raise ValueError("Static INT8 quantization requires a calibration data loader.")
        try:
            from torch.ao.quantization import get_default_qconfig_mapping
            from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
        except ImportError as exc:
            raise RuntimeError(
                "FX quantization APIs are unavailable in this PyTorch build."
            ) from exc
        candidate = self.clone(model).cpu().eval()
        first = next(iter(calib_loader), None)
        if first is None:
            raise ValueError("Calibration data loader is empty.")
        inputs, _ = _batch_parts(first, torch.device("cpu"))
        example_inputs = tuple(inputs.values()) if isinstance(inputs, dict) else (inputs,)
        prepared = prepare_fx(
            candidate,
            get_default_qconfig_mapping(self.backend),
            example_inputs,
        )
        with torch.inference_mode():
            for index, batch in enumerate(calib_loader):
                if index >= self.calib_batches:
                    break
                batch_inputs, _ = _batch_parts(batch, torch.device("cpu"))
                _forward(prepared, batch_inputs)
        return convert_fx(prepared).eval()


class QuantizationAwareTrainingOptimizer(BaseOptimizer):
    """Fine-tune with fake quantization, then convert to a static INT8 model."""

    name = "qat_int8"

    def __init__(self, backend: str = "x86", epochs: int = 1, learning_rate: float = 1e-4) -> None:
        self.backend = backend
        self.epochs = epochs
        self.learning_rate = learning_rate

    def apply(self, model: nn.Module, calib_loader: Any = None) -> nn.Module:
        if calib_loader is None:
            raise ValueError("QAT requires a labeled training/calibration data loader.")
        if self.epochs < 1:
            raise ValueError("QAT epochs must be at least one.")
        try:
            from torch.ao.quantization import get_default_qat_qconfig_mapping
            from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx
        except ImportError as exc:
            raise RuntimeError(
                "FX quantization APIs are unavailable in this PyTorch build."
            ) from exc
        candidate = self.clone(model).cpu().train()
        first = next(iter(calib_loader), None)
        if first is None:
            raise ValueError("QAT data loader is empty.")
        inputs, _ = _batch_parts(first, torch.device("cpu"))
        example_inputs = tuple(inputs.values()) if isinstance(inputs, dict) else (inputs,)
        prepared = prepare_qat_fx(
            candidate,
            get_default_qat_qconfig_mapping(self.backend),
            example_inputs,
        )
        optimizer = torch.optim.AdamW(prepared.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            for batch in calib_loader:
                batch_inputs, labels = _batch_parts(batch, torch.device("cpu"))
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(_forward(prepared, batch_inputs), labels)
                loss.backward()
                optimizer.step()
        return convert_fx(prepared.eval()).eval()
