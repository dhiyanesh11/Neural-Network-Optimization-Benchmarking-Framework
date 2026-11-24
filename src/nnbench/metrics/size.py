"""Model size and parameter-count metrics."""

import os
import tempfile

import torch
from torch import nn


def parameter_count(model: nn.Module, *, trainable_only: bool = False) -> int:
    """Count parameters in a PyTorch model."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def model_size_bytes(model: nn.Module) -> int:
    """Measure serialized state-dict size using a temporary file."""
    artifact_size = getattr(model, "model_size_bytes", None)
    if callable(artifact_size):
        return int(artifact_size())
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, "model.pt")
        torch.save(model.state_dict(), temp_path)
        return os.path.getsize(temp_path)


def model_size_mb(model: nn.Module) -> float:
    """Return the serialized state-dict size in decimal megabytes."""
    return model_size_bytes(model) / 1_000_000
