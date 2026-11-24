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

## Benchmarking

`BenchmarkRunner.run(model, dataloader, config)` evaluates accuracy and F1, times repeated inference with warm-up, reports p50/p95/p99 latency and throughput, measures serialized model size and parameter counts, and captures process/GPU memory. Set `max_eval_batches` for a short smoke run or `batch_sizes` for a latency sweep. FLOPs are populated when the optional `flops` extra is installed and the model can be traced by fvcore.

## Optimizer plugins

Built-ins are `ptq_dynamic_int8` (dynamic Linear/LSTM quantization), `ptq_static_int8` (calibrated FX conversion), `qat_int8` (fake-quant training then conversion), `fp16`, `prune_l1` (global weight sparsity), `prune_structured_l1` (output channel removal by masking), `torch_compile`, and `knowledge_distillation`. Each accepts its options from config through `get_optimizer(name, **params)` and returns a deep-copied model. Pruning removes the temporary masks after materializing zeros. FP16 inference requires model inputs with floating values; the benchmark runner casts floating inputs to the model's parameter dtype. Optimized checkpoints and metadata can be written with `save_optimized_model`.

## ONNX runtime

Install the `export` extra to enable ONNX and ONNX Runtime. `export_onnx` exports with a dynamic batch axis and validates with `onnx.checker`; `ORTModel` runs the graph with a selectable provider and graph optimization level. `quantize_onnx_dynamic` produces an INT8-weight graph. `ORTModel` can be passed directly to `BenchmarkRunner` and reports the ONNX file size.

## Results store and analysis

`ResultStore` creates the SQLite experiments/runs/metrics schema on first use and stores config hashes, hardware, library details, Git revision, and per-run metrics. Use `query_results` to filter saved runs, `export_csv`/`export_json` for files, `summary_table` for grouped metric averages, `compare_to_baseline` for speedup/compression/accuracy drop, and `pareto_front` or `plot_pareto` for trade-off analysis.

## Checks

```powershell
ruff check src tests
black --check src tests
pytest -q
```
