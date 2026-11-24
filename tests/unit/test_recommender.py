"""Constraint filtering and recommendation ordering tests."""

from nnbench.selector import recommend, recommendation_message


def test_recommender_filters_constraints_and_ranks() -> None:
    rows = [
        {
            "optimizer": "baseline",
            "metrics": {"accuracy": 0.98, "latency_mean_ms": 10, "model_size_mb": 8},
        },
        {
            "optimizer": "fast",
            "metrics": {"accuracy": 0.94, "latency_mean_ms": 4, "model_size_mb": 3},
        },
        {
            "optimizer": "accurate",
            "metrics": {"accuracy": 0.96, "latency_mean_ms": 7, "model_size_mb": 4},
        },
        {
            "optimizer": "too_slow",
            "metrics": {"accuracy": 0.99, "latency_mean_ms": 50, "model_size_mb": 2},
        },
    ]
    ranked = recommend(rows, {"max_latency_ms": 10, "max_size_mb": 5, "min_accuracy": 0.9})
    assert [row["optimizer"] for row in ranked] == ["accurate", "fast"]
    assert ranked[0]["recommendation_rank"] == 1


def test_impossible_constraints_return_empty_with_clear_message() -> None:
    ranked = recommend(
        [{"optimizer": "prune_l1", "accuracy": 0.8, "latency_mean_ms": 20, "model_size_mb": 4}],
        {"min_accuracy": 0.99},
    )
    assert ranked == []
    assert "No stored optimization meets" in recommendation_message({"min_accuracy": 0.99})
