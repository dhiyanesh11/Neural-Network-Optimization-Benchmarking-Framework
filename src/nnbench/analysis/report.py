"""Tabular analysis and CSV/JSON export helpers."""

import csv
import json
from pathlib import Path
from typing import Any


def flatten_result(row: dict[str, Any]) -> dict[str, Any]:
    """Flatten a stored or raw benchmark row for tabular export."""
    result = {
        key: value for key, value in row.items() if key not in {"metrics", "hardware", "params"}
    }
    result.update(row.get("metrics", {}))
    result["params"] = json.dumps(row.get("params", {}), sort_keys=True, default=str)
    result["hardware"] = json.dumps(row.get("hardware", {}), sort_keys=True, default=str)
    return result


def summary_table(
    rows: list[dict[str, Any]], group_by: tuple[str, ...] = ("model", "optimizer", "backend")
) -> list[dict[str, Any]]:
    """Aggregate numeric metrics by model, optimizer, and backend dimensions."""
    buckets: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    for row in rows:
        group = tuple(row.get(key) for key in group_by)
        metrics = row.get("metrics", row)
        bucket = buckets.setdefault(group, {})
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                bucket.setdefault(key, []).append(float(value))
    output = []
    for group, metrics in buckets.items():
        result = dict(zip(group_by, group, strict=True))
        result["runs"] = max((len(values) for values in metrics.values()), default=0)
        result.update({key: sum(values) / len(values) for key, values in metrics.items()})
        output.append(result)
    return output


def export_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write flattened result rows to CSV."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flattened = [flatten_result(row) for row in rows]
    columns = sorted({key for row in flattened for key in row})
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flattened)
    return destination


def export_json(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write results as readable JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return destination
