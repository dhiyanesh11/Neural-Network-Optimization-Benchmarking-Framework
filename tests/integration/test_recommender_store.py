"""Integration test for recommendations against persisted run metrics."""

from nnbench.selector import recommend_from_store
from nnbench.store import ResultStore


def test_recommendation_respects_constraints_on_saved_results(tmp_path) -> None:
    store = ResultStore(tmp_path / "runs.sqlite")
    for name, metrics in (
        ("baseline", {"accuracy": 0.95, "latency_mean_ms": 20, "model_size_mb": 8}),
        ("prune_l1", {"accuracy": 0.93, "latency_mean_ms": 10, "model_size_mb": 4}),
        ("fp16", {"accuracy": 0.91, "latency_mean_ms": 5, "model_size_mb": 3}),
    ):
        store.save_result(
            metrics,
            experiment_name="selector",
            config={"seed": 42},
            metadata={"model": "toy", "optimizer": name},
        )
    recommendations = recommend_from_store(
        str(store.path), {"max_latency_ms": 12, "max_size_mb": 5, "min_accuracy": 0.9}
    )
    assert {row["optimizer"] for row in recommendations} == {"prune_l1", "fp16"}
    assert all(
        row["metrics"]["accuracy"] >= 0.9
        and row["metrics"]["latency_mean_ms"] <= 12
        and row["metrics"]["model_size_mb"] <= 5
        for row in recommendations
    )
