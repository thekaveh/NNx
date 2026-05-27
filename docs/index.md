# NNx

Lightweight PyTorch training / eval / visualization toolkit, with first-class support for graph neural networks (GCN / GraphSAGE / GAT). Originally extracted from `thekaveh/ml` to underpin training loops, checkpointing, and result visualization across notebook-based experiments; now standalone.

## 1. Why NNx

If you've ever found yourself rewriting the same training loop, the same checkpoint shuffling, and the same metric plotting from project to project, that's NNx's purpose: a tight, opinionated layer that owns the boring parts so you can focus on the model.

### 1.1. Core capabilities

- **Generic training loop** — callbacks, early stopping, schedulers, AMP, gradient clipping, gradient accumulation, and seeded reproducibility.
- **Content-addressed checkpoint management** — FIRST / Q1 / Q2 / Q3 / LAST / BEST tags and a `runs/best` symlink that always points at your lowest-error run.
- **Warm-resume training** — load weights AND optimizer state from any saved checkpoint.
- **Custom metrics injection** — plug in any `callable(Y_true, Y_pred) -> float` via `NNTrainParams.extra_metrics`.
- **TensorBoard and Weights & Biases callbacks** — opt-in via extras.
- **ONNX export** — `NNModel.to_onnx(path, example_input)` with a single method call.

### 1.2. Specializations

- **Fine-tuning (transfer learning)** — glob-pattern layer freezing, external state-dict loading, per-layer-group learning rates.
- **Multi-optimizer `Trainer`** — parallel to `NNModel.train()` for GAN / actor-critic workflows with a name-keyed dict of optimizers scoped via `NNParamGroupSpec`.
- **Diffusion (DDPM)** — noise-prediction training and reverse-diffusion sampling.
- **Training paradigms** — knowledge distillation, contrastive learning, and batch-level augmentation (Mixup / CutMix).
- **Parameter-efficient fine-tuning (PEFT)** — LoRA-wrapped Linear layers and bottleneck residual adapters.

## 2. Where to next

### 2.1. Get running

- [Quickstart](quickstart.md) — five minutes to a trained model, paste-runnable.

### 2.2. Understand the design

- [Concepts](concepts.md) — what an `NNRun` is, where things land on disk, how the enum-as-factory pattern works, how the five specialization subpackages compose.

### 2.3. Look things up

- [API Reference](api.md) — auto-generated from docstrings.

## 3. Status

Alpha. API is stable for the existing `thekaveh/ml` notebook consumer; pre-1.0 means we'll fix bugs (see [CHANGELOG](https://github.com/thekaveh/NNx/blob/main/CHANGELOG.md)) without renaming public APIs unless they're broken in ways notebooks can't work around.
