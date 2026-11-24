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

## Registry and baselines

The registry includes `mobilenet_v2`, `resnet18`, and `distilbert_sst2`, with `cifar10` and `sst2` data loaders. Install optional support with `python -m pip install -e ".[vision,nlp]"`. Pass `synthetic=True` to `get_dataloader` for deterministic, offline smoke tests. Train an image baseline with `python scripts/train_baseline.py --model mobilenet_v2 --epochs 3 --download`; the script stores a checkpoint and its measured validation accuracy in `artifacts/`.

To register another model, add a factory in `src/nnbench/registry/models.py` and expose its stable name through `MODEL_FACTORIES`. Add data loaders to `src/nnbench/registry/datasets.py`; keep synthetic inputs available for fast integration checks.

## Checks

```powershell
ruff check src tests
black --check src tests
pytest -q
```
