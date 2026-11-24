"""Integration test: benchmark runs are stored and rendered as a Pareto chart."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nnbench.analysis.plots import plot_pareto
from nnbench.optimizers import get_optimizer
from nnbench.runner import BenchmarkRunner
from nnbench.store import ResultStore


def test_experiment_results_round_trip_to_plot(tmp_path) -> None:
    torch.manual_seed(53)
    model = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 2)).eval()
    loader = DataLoader(TensorDataset(torch.rand(12, 4), torch.randint(0, 2, (12,))), batch_size=4)
    config = {"seed": 53, "model": "toy", "dataset": "synthetic", "optimizer_params": {}}
    runner = BenchmarkRunner(device="cpu")
    store = ResultStore(tmp_path / "experiment.sqlite")
    baseline = runner.run(model, loader, {"warmup_iters": 1, "measure_iters": 2, "model": "toy"})
    store.save_result(baseline, experiment_name="smoke", config=config)
    optimized_model = get_optimizer("prune_l1", sparsity=0.25).apply(model)
    optimized = runner.run(
        optimized_model,
        loader,
        {"warmup_iters": 1, "measure_iters": 2, "model": "toy", "optimizer": "prune_l1"},
    )
    store.save_result(
        optimized,
        experiment_name="smoke",
        config=config,
        metadata={"optimizer": "prune_l1", "params": {"sparsity": 0.25}},
    )
    rows = store.query_results(experiment_name="smoke")
    assert len(rows) == 2
    figure = plot_pareto(rows)
    assert len(figure.data) > 0
