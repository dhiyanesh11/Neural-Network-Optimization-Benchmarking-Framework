"""Baseline comparisons and Pareto frontier calculations."""

from collections.abc import Iterable
from typing import Any


def _metrics(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("metrics", row)


def compare_to_baseline(
    rows: Iterable[dict[str, Any]], baseline_name: str = "baseline"
) -> list[dict[str, Any]]:
    """Add speedup, compression, and accuracy-drop values relative to each baseline."""
    materialized = list(rows)
    baselines = {
        (row.get("model"), row.get("backend", "pytorch")): row
        for row in materialized
        if row.get("optimizer", "baseline") == baseline_name
    }
    output = []
    for row in materialized:
        key = (row.get("model"), row.get("backend", "pytorch"))
        baseline = baselines.get(key)
        current_metrics = _metrics(row)
        result = dict(row)
        if baseline is not None:
            baseline_metrics = _metrics(baseline)
            base_latency = baseline_metrics.get("latency_mean_ms")
            cur_latency = current_metrics.get("latency_mean_ms")
            base_size = baseline_metrics.get("model_size_mb")
            cur_size = current_metrics.get("model_size_mb")
            base_accuracy = baseline_metrics.get("accuracy")
            cur_accuracy = current_metrics.get("accuracy")
            derived = {
                "speedup": base_latency / cur_latency if base_latency and cur_latency else None,
                "compression_ratio": base_size / cur_size if base_size and cur_size else None,
                "accuracy_drop": (
                    base_accuracy - cur_accuracy
                    if base_accuracy is not None and cur_accuracy is not None
                    else None
                ),
            }
            if "metrics" in result:
                result["metrics"] = {**current_metrics, **derived}
            else:
                result.update(derived)
        output.append(result)
    return output


def pareto_front(
    rows: Iterable[dict[str, Any]],
    *,
    accuracy_key: str = "accuracy",
    latency_key: str = "latency_mean_ms",
    size_key: str = "model_size_mb",
) -> list[dict[str, Any]]:
    """Return rows that maximize accuracy while minimizing latency and size."""
    candidates = [
        row
        for row in rows
        if all(_metrics(row).get(key) is not None for key in (accuracy_key, latency_key, size_key))
    ]
    front: list[dict[str, Any]] = []
    for candidate in candidates:
        values = _metrics(candidate)
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            compared = _metrics(other)
            no_worse = (
                compared[accuracy_key] >= values[accuracy_key]
                and compared[latency_key] <= values[latency_key]
                and compared[size_key] <= values[size_key]
            )
            strictly_better = (
                compared[accuracy_key] > values[accuracy_key]
                or compared[latency_key] < values[latency_key]
                or compared[size_key] < values[size_key]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front
