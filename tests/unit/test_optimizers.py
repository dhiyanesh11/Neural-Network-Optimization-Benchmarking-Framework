"""Optimizer plugin behavior and copy-safety tests."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nnbench.optimizers import get_optimizer, list_optimizers


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 512), nn.ReLU(), nn.Linear(512, 2))


def _loader() -> DataLoader:
    torch.manual_seed(17)
    return DataLoader(TensorDataset(torch.rand(16, 4), torch.randint(0, 2, (16,))), batch_size=4)


def _sparsity(model: nn.Module) -> float:
    values = torch.cat(
        [
            module.weight.detach().flatten()
            for module in model.modules()
            if isinstance(module, nn.Linear)
        ]
    )
    return float((values == 0).sum() / values.numel())


def test_registry_lists_expected_optimizer_plugins() -> None:
    assert {
        "ptq_dynamic_int8",
        "ptq_static_int8",
        "qat_int8",
        "fp16",
        "prune_l1",
        "prune_structured_l1",
        "torch_compile",
        "knowledge_distillation",
    } <= set(list_optimizers())


def test_unstructured_pruning_hits_target_and_preserves_original() -> None:
    original = _model()
    before = [parameter.detach().clone() for parameter in original.parameters()]
    pruned = get_optimizer("prune_l1", sparsity=0.5).apply(original)
    assert abs(_sparsity(pruned) - 0.5) <= 0.02
    assert all(
        torch.equal(old, new) for old, new in zip(before, original.parameters(), strict=True)
    )
    assert pruned is not original


def test_dynamic_quantization_has_qint8_weights_and_smaller_size() -> None:
    original = _model().eval()
    quantized = get_optimizer("ptq_dynamic_int8").apply(original)
    quantized_linears = [
        module for module in quantized.modules() if isinstance(module, nn.quantized.Linear)
    ]
    assert quantized_linears
    assert all(module.weight().dtype == torch.qint8 for module in quantized_linears)
    from nnbench.metrics.size import model_size_bytes

    assert model_size_bytes(quantized) < model_size_bytes(original)


def test_fp16_returns_a_separate_half_precision_model() -> None:
    original = _model()
    half_model = get_optimizer("fp16").apply(original)
    assert half_model is not original
    assert next(half_model.parameters()).dtype == torch.float16
    assert next(original.parameters()).dtype == torch.float32


def test_static_quantization_requires_calibration_loader() -> None:
    import pytest

    with pytest.raises(ValueError, match="requires a calibration"):
        get_optimizer("ptq_static_int8").apply(_model())


def test_distillation_trains_a_copy() -> None:
    teacher = _model().eval()
    student = _model()
    before = [parameter.detach().clone() for parameter in student.parameters()]
    distilled = get_optimizer(
        "knowledge_distillation", teacher=teacher, train_loader=_loader(), epochs=1
    ).apply(student)
    assert distilled is not student
    assert all(torch.equal(old, new) for old, new in zip(before, student.parameters(), strict=True))
