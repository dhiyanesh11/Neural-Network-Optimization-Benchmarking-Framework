"""Unit tests for standardized metric helpers."""

import os
import time

import numpy as np
import torch
from torch import nn

from nnbench.metrics.accuracy import classification_metrics
from nnbench.metrics.latency import latency_summary, measure_latency
from nnbench.metrics.size import model_size_bytes, parameter_count


def test_latency_uses_repeated_samples_and_reports_expected_duration() -> None:
    result = measure_latency(lambda: time.sleep(0.01), warmup_iters=1, measure_iters=3)
    assert 8 <= result["latency_mean_ms"] <= 30
    assert result["latency_p50_ms"] > 0
    assert result["latency_p95_ms"] >= result["latency_p50_ms"]


def test_latency_percentiles_match_numpy() -> None:
    samples = [1.0, 3.0, 8.0, 10.0]
    result = latency_summary(samples, batch_size=2)
    assert result["latency_p50_ms"] == np.percentile(samples, 50)
    assert result["latency_p95_ms"] == np.percentile(samples, 95)
    assert result["latency_p99_ms"] == np.percentile(samples, 99)


def test_model_size_matches_serialized_file_size(tmp_path) -> None:
    model = nn.Linear(5, 2)
    saved = tmp_path / "model.pt"
    torch.save(model.state_dict(), saved)
    assert model_size_bytes(model) == os.path.getsize(saved)


def test_parameter_count_and_classification_metrics() -> None:
    model = nn.Linear(5, 2)
    assert parameter_count(model) == 12
    assert classification_metrics([0, 1, 1], [0, 0, 1]) == {
        "accuracy": 2 / 3,
        "f1_macro": 2 / 3,
        "f1_weighted": 2 / 3,
    }
