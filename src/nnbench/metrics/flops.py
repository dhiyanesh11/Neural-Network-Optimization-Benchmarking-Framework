"""Optional FLOPs measurement through fvcore."""

from collections.abc import Mapping

import torch
from torch import nn


def estimate_flops(model: nn.Module, example_inputs: torch.Tensor | Mapping[str, torch.Tensor]):
    """Estimate per-forward FLOPs; return ``None`` if fvcore cannot trace the model."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None
    args = (
        tuple(example_inputs.values()) if isinstance(example_inputs, Mapping) else (example_inputs,)
    )
    try:
        return int(FlopCountAnalysis(model, args).total())
    except (RuntimeError, TypeError, ValueError):
        return None
