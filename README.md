# Neural Network Optimization & Benchmarking Framework

`nnbench` is a config-driven Python framework for applying model optimizations and comparing their accuracy, latency, size, and memory use under a consistent protocol.

## Development setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

## Foundation CLI

```powershell
nnbench --help
nnbench config-show configs/experiment/default.yaml
```

The initial config schema requires `seed`, `model`, `dataset`, and `optimizers`. The sample config is `configs/experiment/default.yaml`.

## Checks

```powershell
ruff check src tests
black --check src tests
pytest -q
```
