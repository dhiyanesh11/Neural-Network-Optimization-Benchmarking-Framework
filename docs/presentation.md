# Project Presentation

## Slide 1 — Neural Network Optimization & Benchmarking Framework

- Config-driven experiments for model optimization
- Consistent comparison of quality, latency, throughput, size, and memory
- Python, PyTorch, ONNX Runtime, SQLite, Streamlit

## Slide 2 — Problem and goals

- Optimization results are hard to compare when protocols differ
- Quantization, pruning, distillation, and compilation have model-dependent trade-offs
- One config should define a reproducible baseline and its optimized variants

## Slide 3 — System architecture

YAML config → model/data registry → baseline → optimizer plugins → benchmark runner → SQLite → analysis/dashboard/recommender

## Slide 4 — Models and data

- MobileNetV2 and ResNet18 for CIFAR-10
- DistilBERT classifier for SST-2
- Seeded synthetic fixtures keep tests and the demo offline

## Slide 5 — Optimization plugins

- Dynamic/static INT8 and QAT
- FP16 and `torch.compile`
- Unstructured and structured L1 pruning
- Teacher/student knowledge distillation
- Each optimizer works on a copied model

## Slide 6 — Benchmarking protocol

- Fixed seed and thread count
- Warm-up plus repeated inference
- p50/p95/p99 latency and throughput
- Accuracy/F1, model size, parameter count, memory, optional FLOPs

## Slide 7 — Persistence and analysis

- SQLite experiments, runs, and metric tables
- Config hash, hardware, library versions, and Git revision
- Baseline speedup/compression/accuracy drop
- Accuracy/latency and accuracy/size Pareto views

## Slide 8 — User workflow

1. Edit a YAML experiment config
2. Run `nnbench run experiment=demo --synthetic`
3. Review with `nnbench report` or the Streamlit dashboard
4. Use `nnbench recommend` to find stored runs within limits

## Slide 9 — Verification and demo

- 43 unit and integration tests at Module 9 start
- 85% local statement coverage; CI target is at least 80%
- CI runs an offline demo on every push and pull request
- Demo metrics validate wiring, not trained-model performance

## Slide 10 — Limitations and future work

- Train or download real baselines for meaningful comparisons
- Migrate PyTorch quantization and ONNX export APIs
- Rebuild structured-pruned networks to realize shape reductions
- Add edge hardware and energy measurements
