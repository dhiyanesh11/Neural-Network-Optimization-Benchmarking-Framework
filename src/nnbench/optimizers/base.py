"""Base interface and model-copy helper for optimization plugins."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from torch import nn


class BaseOptimizer(ABC):
    """Base optimizer that guarantees an independent model copy."""

    name = "base"

    def clone(self, model: nn.Module) -> nn.Module:
        """Deep copy the input model before changes are applied."""
        return deepcopy(model)

    @abstractmethod
    def apply(self, model: nn.Module, calib_loader: Any = None) -> nn.Module:
        """Return an optimized copy of the model."""
