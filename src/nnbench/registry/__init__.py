"""Model and dataset registry public API."""

from nnbench.registry.datasets import get_dataloader, list_datasets
from nnbench.registry.models import get_model, list_models

__all__ = ["get_dataloader", "get_model", "list_datasets", "list_models"]
