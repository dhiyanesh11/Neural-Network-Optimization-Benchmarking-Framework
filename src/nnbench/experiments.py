"""Config-driven experiment execution pipeline."""

from pathlib import Path
from typing import Any

import torch

from nnbench.config import load_config
from nnbench.export import ORTModel, export_onnx, quantize_onnx_dynamic
from nnbench.optimizers import get_optimizer
from nnbench.optimizers.artifacts import save_optimized_model
from nnbench.registry import get_dataloader, get_model
from nnbench.runner import BenchmarkRunner
from nnbench.runner.benchmark import _batch_parts
from nnbench.seed import set_seed
from nnbench.store import ResultStore
from nnbench.store.metadata import hardware_info


def run_experiment(
    config_path: str | Path,
    *,
    db_path: str | Path = "artifacts/nnbench.sqlite",
    synthetic: bool = False,
    download: bool = False,
) -> list[dict[str, Any]]:
    """Load a YAML config, benchmark baseline and optimizers, and persist every run."""
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    torch.set_num_threads(int(config.get("num_threads", 1)))
    model_name = config["model"]
    dataset_name = config["dataset"]
    batch_size = int(config.get("batch_size", 32))
    samples = int(config.get("synthetic_samples", 32))
    split = str(config.get("split", "test"))
    loader_args = {
        "synthetic": synthetic or bool(config.get("synthetic", False)),
        "num_samples": samples,
        "seed": seed,
        "download": download or bool(config.get("download", False)),
    }
    eval_loader = get_dataloader(dataset_name, split, batch_size, **loader_args)
    calibration_loader = get_dataloader(
        dataset_name, str(config.get("calibration_split", "train")), batch_size, **loader_args
    )
    pretrained = bool(config.get("pretrained", False))
    model = get_model(model_name, pretrained=pretrained)
    checkpoint = config.get("checkpoint")
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    runner = BenchmarkRunner(config.get("device"))
    benchmark_config = {
        **config.get("benchmark", {}),
        "model": model_name,
        "batch_sizes": config.get("batch_sizes", []),
        "max_eval_batches": config.get("max_eval_batches"),
    }
    database = ResultStore(db_path)
    experiment_name = str(config.get("name", Path(config_path).stem))
    artifact_dir = Path(config.get("artifact_dir", "artifacts"))
    backends = config.get("backends", ["pytorch"])
    optimizer_specs = [None, *config.get("optimizers", [])]
    results: list[dict[str, Any]] = []

    for optimizer_spec in optimizer_specs:
        if optimizer_spec is None:
            optimizer_name = "baseline"
            candidate = model
            optimizer_params: dict[str, Any] = {}
        else:
            optimizer_params = dict(optimizer_spec)
            optimizer_name = str(optimizer_params.pop("name"))
            if optimizer_name == "knowledge_distillation":
                optimizer_params.setdefault("teacher", model)
                optimizer_params.setdefault("train_loader", calibration_loader)
            plugin = get_optimizer(optimizer_name, **optimizer_params)
            candidate = plugin.apply(model, calib_loader=calibration_loader)
            if config.get("save_optimized", True):
                save_optimized_model(
                    candidate,
                    f"{model_name}_{optimizer_name}",
                    metadata={"params": optimizer_params, "seed": seed},
                    output_dir=artifact_dir / "optimized",
                )

        for backend in backends:
            if backend == "pytorch":
                backend_model = candidate
            elif backend in {"onnxruntime", "onnxruntime_int8"}:
                sample_batch = next(iter(eval_loader), None)
                if sample_batch is None:
                    raise ValueError("Cannot export a model using an empty evaluation loader.")
                sample_inputs, _ = _batch_parts(sample_batch, torch.device("cpu"))
                model_path = artifact_dir / "onnx" / f"{model_name}_{optimizer_name}.onnx"
                export_onnx(candidate, sample_inputs, model_path)
                if backend == "onnxruntime_int8":
                    int8_path = model_path.with_name(f"{model_path.stem}_int8.onnx")
                    model_path = quantize_onnx_dynamic(model_path, int8_path)
                backend_model = ORTModel(model_path)
            else:
                raise ValueError(f"Unsupported backend '{backend}'.")
            result = runner.run(
                backend_model,
                eval_loader,
                {**benchmark_config, "optimizer": optimizer_name},
            )
            database.save_result(
                result,
                experiment_name=experiment_name,
                config=config,
                metadata={
                    "model": model_name,
                    "optimizer": optimizer_name,
                    "params": optimizer_params,
                    "backend": backend,
                    "hardware": hardware_info(),
                },
            )
            results.append({**result, "backend": backend})
    return results
