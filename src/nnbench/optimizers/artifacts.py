"""Persistence helpers for optimized model checkpoints."""

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_optimized_model(
    model: nn.Module,
    name: str,
    metadata: dict[str, Any] | None = None,
    output_dir: str | Path = "artifacts/optimized",
) -> Path:
    """Save model weights and a JSON sidecar beneath the optimized artifacts folder."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / f"{name}.pt"
    torch.save(model.state_dict(), checkpoint)
    (directory / f"{name}.json").write_text(
        json.dumps(metadata or {}, indent=2, sort_keys=True), encoding="utf-8"
    )
    return checkpoint
