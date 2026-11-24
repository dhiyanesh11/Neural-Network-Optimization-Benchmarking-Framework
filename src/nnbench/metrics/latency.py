"""Repeated inference latency and throughput measurements."""

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch


def _sync(device: torch.device | str | None) -> None:
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def latency_summary(samples_ms: list[float], batch_size: int = 1) -> dict[str, float]:
    """Summarize fixed latency samples with percentiles and throughput."""
    if not samples_ms or batch_size < 1:
        raise ValueError("samples_ms must be non-empty and batch_size must be >=1")
    elapsed_s = sum(samples_ms) / 1000.0
    return {
        "latency_mean_ms": float(np.mean(samples_ms)),
        "latency_p50_ms": float(np.percentile(samples_ms, 50)),
        "latency_p95_ms": float(np.percentile(samples_ms, 95)),
        "latency_p99_ms": float(np.percentile(samples_ms, 99)),
        "throughput_samples_per_second": float(len(samples_ms) * batch_size / elapsed_s),
    }


def measure_latency(
    run_once: Callable[[], Any],
    *,
    warmup_iters: int = 20,
    measure_iters: int = 100,
    batch_size: int = 1,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Measure inference latency in milliseconds and throughput for a callable."""
    if warmup_iters < 0 or measure_iters < 1 or batch_size < 1:
        raise ValueError("warmup_iters must be >=0; measure_iters and batch_size must be >=1")
    for _ in range(warmup_iters):
        run_once()
    _sync(device)
    samples: list[float] = []
    for _ in range(measure_iters):
        _sync(device)
        start = time.perf_counter()
        run_once()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return latency_summary(samples, batch_size)
