"""Magnitude pruning plugins."""

from typing import Any

from torch import nn
from torch.nn.utils import prune

from nnbench.optimizers.base import BaseOptimizer


class UnstructuredL1PruningOptimizer(BaseOptimizer):
    """Apply global L1 magnitude pruning to convolution and linear weights."""

    name = "prune_l1"

    def __init__(self, sparsity: float = 0.5) -> None:
        self.sparsity = _validate_sparsity(sparsity)

    def apply(self, model: nn.Module, calib_loader: Any = None) -> nn.Module:
        candidate = self.clone(model)
        parameters = [
            (module, "weight")
            for module in candidate.modules()
            if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight is not None
        ]
        if not parameters:
            raise ValueError("Model has no convolution or linear weights to prune.")
        prune.global_unstructured(
            parameters,
            pruning_method=prune.L1Unstructured,
            amount=self.sparsity,
        )
        for module, name in parameters:
            prune.remove(module, name)
        return candidate


class StructuredL1PruningOptimizer(BaseOptimizer):
    """Prune entire low-magnitude output channels of convolutional layers."""

    name = "prune_structured_l1"

    def __init__(self, sparsity: float = 0.5) -> None:
        self.sparsity = _validate_sparsity(sparsity)

    def apply(self, model: nn.Module, calib_loader: Any = None) -> nn.Module:
        candidate = self.clone(model)
        targets = [module for module in candidate.modules() if isinstance(module, nn.Conv2d)]
        if not targets:
            targets = [module for module in candidate.modules() if isinstance(module, nn.Linear)]
        if not targets:
            raise ValueError("Model has no convolution or linear weights to prune.")
        for module in targets:
            prune.ln_structured(module, name="weight", amount=self.sparsity, n=1, dim=0)
            prune.remove(module, "weight")
        return candidate


def _validate_sparsity(sparsity: float) -> float:
    if not 0 <= sparsity < 1:
        raise ValueError("sparsity must be in the range [0, 1).")
    return float(sparsity)
