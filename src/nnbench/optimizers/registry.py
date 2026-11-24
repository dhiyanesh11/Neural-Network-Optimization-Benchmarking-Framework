"""Name-based optimizer plugin registry."""

from collections.abc import Callable
from typing import Any

from nnbench.optimizers.base import BaseOptimizer
from nnbench.optimizers.distillation import KnowledgeDistillationOptimizer
from nnbench.optimizers.precision import CompileOptimizer, FP16Optimizer
from nnbench.optimizers.pruning import StructuredL1PruningOptimizer, UnstructuredL1PruningOptimizer
from nnbench.optimizers.quantization import (
    DynamicInt8Optimizer,
    QuantizationAwareTrainingOptimizer,
    StaticInt8Optimizer,
)

OPTIMIZER_FACTORIES: dict[str, Callable[..., BaseOptimizer]] = {
    "ptq_dynamic_int8": DynamicInt8Optimizer,
    "ptq_static_int8": StaticInt8Optimizer,
    "qat_int8": QuantizationAwareTrainingOptimizer,
    "fp16": FP16Optimizer,
    "prune_l1": UnstructuredL1PruningOptimizer,
    "prune_structured_l1": StructuredL1PruningOptimizer,
    "torch_compile": CompileOptimizer,
    "knowledge_distillation": KnowledgeDistillationOptimizer,
}


def register_optimizer(name: str, factory: Callable[..., BaseOptimizer]) -> None:
    """Register a plugin factory under a config name."""
    if not name.strip():
        raise ValueError("Optimizer name cannot be empty.")
    if name in OPTIMIZER_FACTORIES:
        raise ValueError(f"Optimizer '{name}' is already registered.")
    OPTIMIZER_FACTORIES[name] = factory


def get_optimizer(name: str, **params: Any) -> BaseOptimizer:
    """Construct a registered optimizer using parameters from experiment config."""
    try:
        factory = OPTIMIZER_FACTORIES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown optimizer '{name}'. Registered optimizers: {', '.join(list_optimizers())}"
        ) from exc
    return factory(**params)


def list_optimizers() -> list[str]:
    """Return registered optimizer names."""
    return sorted(OPTIMIZER_FACTORIES)
