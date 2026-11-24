"""Consistent accuracy, latency, size, FLOPs, and memory benchmark runner."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from nnbench.metrics.accuracy import classification_metrics, logits_to_labels
from nnbench.metrics.flops import estimate_flops
from nnbench.metrics.latency import measure_latency
from nnbench.metrics.size import model_size_mb, parameter_count


def _batch_parts(batch: Any, device: torch.device) -> tuple[Any, torch.Tensor]:
    if isinstance(batch, Mapping):
        moved = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        labels = moved.pop("labels")
        return moved, labels
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        inputs, labels = batch[0], batch[1]
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(device)
        elif isinstance(inputs, Mapping):
            inputs = {key: value.to(device) for key, value in inputs.items()}
        return inputs, labels.to(device)
    raise TypeError("Each batch must be a (inputs, labels) pair or a mapping with 'labels'.")


def _forward(model: nn.Module, inputs: Any) -> torch.Tensor:
    output = model(**inputs) if isinstance(inputs, Mapping) else model(inputs)
    if hasattr(output, "logits"):
        output = output.logits
    if isinstance(output, Mapping):
        output = output["logits"]
    if not isinstance(output, torch.Tensor):
        raise TypeError("Model output must be a tensor or contain a logits tensor.")
    return output


class BenchmarkRunner:
    """Run model benchmarks with common evaluation and timing settings."""

    def __init__(self, device: str | torch.device | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def run(
        self, model: nn.Module, dataloader: DataLoader, config: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Evaluate a model and return a JSON-serializable metric dictionary."""
        model = model.to(self.device)
        model.eval()
        max_eval_batches = config.get("max_eval_batches")
        predicted: list[int] = []
        targets: list[int] = []
        first_inputs: Any = None
        first_batch_size = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        with torch.inference_mode():
            for index, batch in enumerate(dataloader):
                if max_eval_batches is not None and index >= max_eval_batches:
                    break
                inputs, labels = _batch_parts(batch, self.device)
                logits = _forward(model, inputs)
                predicted.extend(logits_to_labels(logits))
                targets.extend(labels.detach().cpu().tolist())
                if first_inputs is None:
                    first_inputs, first_batch_size = inputs, int(labels.shape[0])
        if first_inputs is None:
            raise ValueError("Cannot benchmark an empty dataloader.")

        def infer() -> torch.Tensor:
            with torch.inference_mode():
                return _forward(model, first_inputs)

        timing = measure_latency(
            infer,
            warmup_iters=int(config.get("warmup_iters", 20)),
            measure_iters=int(config.get("measure_iters", 100)),
            batch_size=first_batch_size,
            device=self.device,
        )
        metrics: dict[str, Any] = {
            **classification_metrics(predicted, targets),
            **timing,
            "num_samples": len(targets),
            "model_size_mb": model_size_mb(model),
            "parameter_count": parameter_count(model),
            "trainable_parameter_count": parameter_count(model, trainable_only=True),
            "flops_per_sample": estimate_flops(model, first_inputs),
            "peak_memory_mb": self._peak_memory_mb(),
            "device": str(self.device),
        }
        batch_sizes = config.get("batch_sizes", [])
        if batch_sizes:
            sweep: list[dict[str, Any]] = []
            for batch_size in batch_sizes:
                sweep_loader = DataLoader(
                    dataloader.dataset,
                    batch_size=int(batch_size),
                    shuffle=False,
                    collate_fn=dataloader.collate_fn,
                    num_workers=dataloader.num_workers,
                )
                sweep_batch = next(iter(sweep_loader), None)
                if sweep_batch is None:
                    continue
                sweep_inputs, sweep_labels = _batch_parts(sweep_batch, self.device)
                sweep_result = measure_latency(
                    lambda inputs=sweep_inputs: _forward(model, inputs),
                    warmup_iters=int(config.get("warmup_iters", 20)),
                    measure_iters=int(config.get("measure_iters", 100)),
                    batch_size=int(sweep_labels.shape[0]),
                    device=self.device,
                )
                sweep.append({"batch_size": int(sweep_labels.shape[0]), **sweep_result})
            metrics["batch_size_sweep"] = sweep
        metrics["model"] = str(config.get("model", model.__class__.__name__))
        metrics["optimizer"] = str(config.get("optimizer", "baseline"))
        return metrics

    def _peak_memory_mb(self) -> float:
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device) / (1024**2)
        try:
            import psutil

            return psutil.Process().memory_info().rss / (1024**2)
        except ImportError:
            return 0.0
