"""Rule-based recommendations over benchmark results."""

from collections.abc import Iterable, Mapping
from typing import Any

from nnbench.store import ResultStore


def _metrics(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return row.get("metrics", row)


def recommend(
    rows: Iterable[dict[str, Any]],
    constraints: Mapping[str, float],
    *,
    include_baseline: bool = False,
) -> list[dict[str, Any]]:
    """Return feasible runs ordered by accuracy, then latency and size.

    Constraint keys are ``max_latency_ms``, ``max_size_mb``, and ``min_accuracy``.
    Rows without the corresponding metric are ignored.
    """
    unknown = set(constraints) - {"max_latency_ms", "max_size_mb", "min_accuracy"}
    if unknown:
        raise ValueError(f"Unknown recommendation constraints: {', '.join(sorted(unknown))}")
    candidates = []
    for row in rows:
        optimizer = row.get("optimizer", "baseline")
        if optimizer == "baseline" and not include_baseline:
            continue
        metrics = _metrics(row)
        accuracy = metrics.get("accuracy")
        latency = metrics.get("latency_mean_ms", metrics.get("latency_p50_ms"))
        size = metrics.get("model_size_mb")
        if accuracy is None or latency is None or size is None:
            continue
        if "min_accuracy" in constraints and accuracy < constraints["min_accuracy"]:
            continue
        if "max_latency_ms" in constraints and latency > constraints["max_latency_ms"]:
            continue
        if "max_size_mb" in constraints and size > constraints["max_size_mb"]:
            continue
        candidates.append((row, float(accuracy), float(latency), float(size)))
    candidates.sort(
        key=lambda item: (-item[1], item[2], item[3], str(item[0].get("optimizer", "")))
    )
    return [
        {**row, "recommendation_rank": rank}
        for rank, (row, _accuracy, _latency, _size) in enumerate(candidates, start=1)
    ]


def recommendation_message(constraints: Mapping[str, float]) -> str:
    """Describe the absence of feasible results in user-facing language."""
    details = ", ".join(f"{key}={value}" for key, value in sorted(constraints.items()))
    return f"No stored optimization meets the requested constraints ({details})."


def recommend_from_store(
    db_path: str,
    constraints: Mapping[str, float],
    *,
    include_baseline: bool = False,
) -> list[dict[str, Any]]:
    """Query the SQLite store and rank feasible runs."""
    return recommend(
        ResultStore(db_path).query_results(), constraints, include_baseline=include_baseline
    )
