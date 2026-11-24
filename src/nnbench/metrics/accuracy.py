"""Classification accuracy and F1 metrics."""

from collections.abc import Iterable

import torch


def classification_metrics(predictions: Iterable[int], targets: Iterable[int]) -> dict[str, float]:
    """Return accuracy and macro/weighted F1 from predicted and target labels."""
    predicted = list(predictions)
    expected = list(targets)
    if len(predicted) != len(expected) or not expected:
        raise ValueError("predictions and targets must have the same non-zero length")
    labels = sorted(set(expected) | set(predicted))
    f1_values: list[float] = []
    weighted: list[tuple[float, int]] = []
    for label in labels:
        tp = sum(p == label and y == label for p, y in zip(predicted, expected, strict=True))
        fp = sum(p == label and y != label for p, y in zip(predicted, expected, strict=True))
        fn = sum(p != label and y == label for p, y in zip(predicted, expected, strict=True))
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else 0.0
        support = sum(y == label for y in expected)
        f1_values.append(f1)
        weighted.append((f1, support))
    return {
        "accuracy": sum(p == y for p, y in zip(predicted, expected, strict=True)) / len(expected),
        "f1_macro": sum(f1_values) / len(f1_values),
        "f1_weighted": sum(score * support for score, support in weighted) / len(expected),
    }


def logits_to_labels(logits: torch.Tensor) -> list[int]:
    """Convert a batch of class logits to predicted integer labels."""
    if logits.ndim != 2:
        raise ValueError("classification logits must have shape (batch, classes)")
    return logits.argmax(dim=1).detach().cpu().tolist()
