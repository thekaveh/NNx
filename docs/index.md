# nnx

Lightweight PyTorch training / eval / visualization toolkit, with first-class support for graph neural networks (GCN / GraphSAGE / GAT).

Originally extracted from `thekaveh/ml` to underpin training loops, checkpointing, and result visualization across notebook-based experiments; now standalone.

## Why nnx

If you've ever found yourself rewriting the same training loop, the same checkpoint shuffling, and the same metric plotting from project to project, that's nnx's purpose: a tight, opinionated layer that owns the boring parts so you can focus on the model.

- **Generic training loop** with callbacks, early stopping, schedulers, AMP, gradient clipping, gradient accumulation, and seeded reproducibility.
- **Checkpoint management** with FIRST / Q1 / Q2 / Q3 / LAST / BEST tags and a `runs/best` symlink that always points at your lowest-error run.
- **Warm-resume training** from any saved checkpoint, preserving optimizer state.
- **Custom metrics injection** — plug in any `callable(Y_true, Y_pred) -> float` via `NNTrainParams.extra_metrics`.
- **TensorBoard and Weights & Biases callbacks** (opt-in via extras).
- **ONNX export** with a single method call.

## Where to next

- **[Quickstart](quickstart.md)** — five minutes to a trained model.
- **[Concepts](concepts.md)** — what an `NNRun` is, where things land on disk, how the enum-as-factory pattern works.
- **[API Reference](api.md)** — auto-generated from docstrings.

## Status

Alpha. API is stable for the existing `thekaveh/ml` notebook consumer; pre-1.0 means we'll fix bugs (see [CHANGELOG](https://github.com/thekaveh/NNx/blob/main/CHANGELOG.md)) without renaming public APIs unless they're broken in ways notebooks can't work around.
