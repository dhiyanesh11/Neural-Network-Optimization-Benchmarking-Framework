"""SQLite persistence and analysis utility tests."""

import csv

from nnbench.analysis.pareto import compare_to_baseline, pareto_front
from nnbench.analysis.report import export_csv, export_json
from nnbench.store import ResultStore


def test_store_round_trip_and_exports(tmp_path) -> None:
    db_path = tmp_path / "results.sqlite"
    store = ResultStore(db_path)
    run_id = store.save_result(
        {
            "accuracy": 0.91,
            "latency_mean_ms": 4.2,
            "model_size_mb": 1.1,
            "device": "cpu",
        },
        experiment_name="unit",
        config={"seed": 42},
        metadata={"model": "toy", "optimizer": "prune_l1", "params": {"sparsity": 0.5}},
    )
    rows = store.query_results(model="toy", optimizer="prune_l1")
    assert len(rows) == 1
    assert rows[0]["id"] == run_id
    assert rows[0]["metrics"]["accuracy"] == 0.91
    assert rows[0]["params"] == {"sparsity": 0.5}
    csv_path = export_csv(rows, tmp_path / "out.csv")
    json_path = export_json(rows, tmp_path / "out.json")
    with csv_path.open(encoding="utf-8", newline="") as source:
        assert next(csv.DictReader(source))["accuracy"] == "0.91"
    assert json_path.read_text(encoding="utf-8").startswith("[")


def test_pareto_front_and_baseline_derived_metrics() -> None:
    rows = [
        {
            "model": "toy",
            "optimizer": "baseline",
            "metrics": {"accuracy": 0.9, "latency_mean_ms": 10, "model_size_mb": 8},
        },
        {
            "model": "toy",
            "optimizer": "fast",
            "metrics": {"accuracy": 0.89, "latency_mean_ms": 5, "model_size_mb": 4},
        },
        {
            "model": "toy",
            "optimizer": "dominated",
            "metrics": {"accuracy": 0.88, "latency_mean_ms": 11, "model_size_mb": 9},
        },
    ]
    front = pareto_front(rows)
    assert [row["optimizer"] for row in front] == ["baseline", "fast"]
    compared = compare_to_baseline(rows)
    fast = compared[1]["metrics"]
    assert fast["speedup"] == 2
    assert fast["compression_ratio"] == 2
    assert abs(fast["accuracy_drop"] - 0.01) < 1e-9
