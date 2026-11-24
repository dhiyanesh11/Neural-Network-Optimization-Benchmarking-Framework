"""Floating-point precision and torch.compile optimizer plugins."""

from typing import Any

import torch
from torch import nn

from nnbench.optimizers.base import BaseOptimizer


class FP16Optimizer(BaseOptimizer):
    """Convert model parameters and buffers to half precision."""

    name = "fp16"

    def apply(self, model: nn.Module, calib_loader: Any = None) -> nn.Module:
        return self.clone(model).half()


class CompileOptimizer(BaseOptimizer):
    """Compile a copied model with ``torch.compile``."""

    name = "torch_compile"

    def __init__(self, backend: str | None = None, mode: str | None = None) -> None:
        self.backend = backend
        self.mode = mode

    def apply(self, model: nn.Module, calib_loader: Any = None) -> nn.Module:
        candidate = self.clone(model).eval()
        options: dict[str, Any] = {}
        if self.backend is not None:
            options["backend"] = self.backend
        if self.mode is not None:
            options["mode"] = self.mode
        return torch.compile(candidate, **options)
