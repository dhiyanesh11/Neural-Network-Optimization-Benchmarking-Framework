# nnbench User Guide

## Install

Use Python 3.10 or newer. For an editable local install:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[vision,nlp,export,analytics,dashboard]"
```

On Linux or macOS activate with `source .venv/bin/activate`. Install the `dev` extra to run tests and linters.

## Run the offline demo

```powershell
nnbench run experiment=demo --synthetic
nnbench report --out reports
streamlit run dashboard/app.py
```

The demo uses seeded synthetic CIFAR-shaped tensors. Its accuracy is a pipeline smoke metric, not a trained model result. Results are saved in `artifacts/nnbench.sqlite`; model checkpoints are saved in `artifacts/optimized/`. Set `--db` to choose another database file.

## Run a real dataset experiment

Copy `configs/experiment/demo.yaml`, set `synthetic: false`, and use an appropriate registered model/dataset. For CIFAR-10 pass `--download` the first time if the data is absent. For SST-2, install the `nlp` extra; the Hugging Face dataset and tokenizer are downloaded by their loaders. Set `pretrained: true` to request public pretrained image weights or DistilBERT weights where supported.

Example:

```powershell
nnbench run configs/experiment/demo.yaml --download --db artifacts/cifar.sqlite
```

Set `benchmark.warmup_iters` and `benchmark.measure_iters` to control repeated timing. Use `batch_sizes` for a latency sweep and `max_eval_batches` for a bounded smoke run. Set `backends: [pytorch, onnxruntime]` to export and benchmark both runtimes. ONNX INT8 is selected by `onnxruntime_int8`.

## Optimizers

Each entry under `optimizers` names a registered plugin and its keyword parameters. Examples include:

```yaml
optimizers:
  - name: ptq_dynamic_int8
  - name: ptq_static_int8
    calib_batches: 10
  - name: prune_l1
    sparsity: 0.5
  - name: fp16
```

The plugin API deep-copies the model before applying changes. Static quantization, QAT, and distillation require calibration/training data. `torch_compile` may have longer startup time on its first forward pass.

## View and select results

The dashboard includes model filtering, a result table and CSV download, accuracy/latency and accuracy/size Pareto plots, latency distribution, optimizer comparisons, host details, and a constraints recommender. Point it at another result database with `NNBENCH_DB`.

```powershell
nnbench recommend --max-latency-ms 20 --max-size-mb 10 --min-accuracy 0.8
```

The recommender considers stored runs only, excludes baselines by default, and returns an empty result with a message if no run satisfies all limits.

## Add a plugin

Implement `BaseOptimizer.apply(model, calib_loader=None)` in `src/nnbench/optimizers/`, register its factory in `optimizers/registry.py`, and add unit plus pipeline integration tests. For a new model, add a factory to `registry/models.py`; for a dataset, add a loader to `registry/datasets.py`. Keep a synthetic data path for reliable offline checks.

## Container use

```powershell
docker build -f docker/Dockerfile.cpu -t nnbench:cpu .
docker run --rm -v ${PWD}/artifacts:/app/artifacts nnbench:cpu
docker compose up dashboard
```

Use `docker/Dockerfile.cuda` with NVIDIA Container Toolkit and a compatible host driver for GPU access.
