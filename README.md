<p align="center" class="nnx-banner">
  <img src="docs/assets/nnx-poster.png" alt="NNx neural training banner" width="100%">
</p>

<h1 align="center">NNx</h1>

<p align="center">
  <strong>Lightweight PyTorch training, evaluation, and visualization with first-class graph neural network support.</strong>
</p>

<p align="center">
  Transparent orchestration for durable experiments, with your models and step logic left in your hands.
</p>

<p align="center" class="nnx-status-badges">
  <a href="https://github.com/thekaveh/NNx/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/thekaveh/NNx/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/thekaveh/NNx/actions/workflows/docs.yml"><img alt="Docs" src="https://github.com/thekaveh/NNx/actions/workflows/docs.yml/badge.svg?branch=main"></a>
  <a href="https://pypi.org/project/thekaveh-nnx/"><img alt="PyPI" src="https://img.shields.io/pypi/v/thekaveh-nnx"></a>
  <a href="https://pypi.org/project/thekaveh-nnx/"><img alt="Python" src="https://img.shields.io/badge/python-3.10--3.14-3776AB?logo=python&logoColor=white"></a>
  <a href="https://spdx.org/licenses/Apache-2.0.html"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-2E7D32"></a>
</p>

<p align="center" class="nnx-core-stack">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white">
  <img alt="pandas" src="https://img.shields.io/badge/pandas-2.0%2B-150458?logo=pandas&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white">
</p>

<p align="center" class="nnx-optional-stack">
  <img alt="torchvision" src="https://img.shields.io/badge/torchvision-optional%20%5Bvision%5D-EE4C2C">
  <img alt="PyTorch Geometric" src="https://img.shields.io/badge/PyTorch_Geometric-optional%20%5Bgraph%5D-3C2179">
  <img alt="Plotly" src="https://img.shields.io/badge/Plotly-optional%20%5Bplots%5D-3F4F75?logo=plotly&logoColor=white">
  <img alt="TensorBoard" src="https://img.shields.io/badge/TensorBoard-optional-FF6F00?logo=tensorflow&logoColor=white">
  <img alt="Weights & Biases" src="https://img.shields.io/badge/Weights_%26_Biases-optional-FFBE00?logo=weightsandbiases&logoColor=black">
  <img alt="ONNX" src="https://img.shields.io/badge/ONNX-optional-005CED?logo=onnx&logoColor=white">
  <img alt="torchao" src="https://img.shields.io/badge/torchao-optional-EE4C2C">
  <img alt="Hugging Face" src="https://img.shields.io/badge/Hugging_Face-optional-FFD21E">
  <img alt="safetensors" src="https://img.shields.io/badge/safetensors-optional-4B5563">
  <img alt="FAISS" src="https://img.shields.io/badge/FAISS-optional-0467DF">
</p>

<!-- executive-summary:start -->
NNx is a lightweight PyTorch toolkit for repeatable training, evaluation and visualization. It provides frozen configuration objects, a supervised train, evaluate and predict loop, callbacks, metrics, and content-addressed checkpoints that support resume. Your models and per-step logic stay replaceable, so one loop serves standard, graph and transformer networks and other training methods. NNx is for researchers and engineers who want transparent PyTorch code without a larger training platform.
<!-- executive-summary:end -->

## 1. Overview

### 1.1. Capabilities

- **Training loop** — `NNModel` train, evaluate and predict, with callbacks, schedulers, FP32/FP16/BF16 precision, gradient accumulation and clipping, declared metrics, and a multi-optimizer `Trainer`.
- **Runs and resume** — content-addressed run directories, checkpoint tags, verified warm resume, experiment plans and provenance records.
- **Tasks** — classification, regression, multilabel, ranking, graph classification and link prediction.
- **Training methods** — custom step functions and objectives, distillation, contrastive learning, Mixup and CutMix, Mixture of Experts, diffusion, I-JEPA and DPO.
- **Fine-tuning and transforms** — layer freezing, LoRA, DoRA, IA3, prefix and prompt tuning, quantization, pruning and model surgery.
- **Networks and export** — feed-forward, convolutional, graph, transformer and ViT networks; a small language-model path; ONNX, Hub, safetensors and experimental GGUF export.
- **Evaluation and decisions** — reproducible splits, calibration, abstention, multi-seed comparison and typed decisions.

The [Capability reference](docs/capabilities.md) describes each item with its entry points. [Concepts](docs/concepts.md) explains how the parts fit together.

## 2. Install

### 2.1. Release or source

This documentation describes the source on `main`: NNx 0.4.1 and its unreleased changes. <!-- x-release-please-version -->

Install the newest release from PyPI, then check its version:

```bash
pip install thekaveh-nnx
python -c "import nnx; print(nnx.__version__)"
```

The newest PyPI release can be older than this documentation. An older release lacks the features and extras added after it, as listed in the [Changelog](CHANGELOG.md). In that case, install the current source instead:

```bash
pip install "thekaveh-nnx @ git+https://github.com/thekaveh/NNx.git@main"
pip install "thekaveh-nnx[domains] @ git+https://github.com/thekaveh/NNx.git@main"   # with an extra
```

The requirements and extras below describe this version. `pip` warns when a requested extra does not exist in the version it installs.

### 2.2. Requirements

The core install is torch, NumPy, pandas, scikit-learn and a few small utilities. `import nnx` loads no vision, graph or plotting stack. Those stacks are the `vision`, `graph` and `plots` extras, and `domains` installs all three. Using one without its extra raises an `ImportError` that names the extra.

NNx needs Python 3.10 or later and PyTorch 2.4 or later; the `vision` extra needs torchvision 0.19 or later. It is tested on Python 3.10 through 3.14. The [PyTorch support matrix](docs/external-contracts.md#21-pytorch-support-matrix) lists every tested combination and which have real CUDA evidence. Optional extras keep their own version constraints. The examples in [examples/](examples/) run on CPU.

### 2.3. Optional extras

Add an extra to either install command, for example `"thekaveh-nnx[graph,plots]"`.

```bash
pip install "thekaveh-nnx[vision]"              # NNDataset (torchvision datasets)
pip install "thekaveh-nnx[graph]"               # graph nets, NNGraphDataset, nnx.graph_tasks, nnx.link_tasks
pip install "thekaveh-nnx[plots]"               # VisUtils, nnx.lr_finder, the NNRun chart (Plotly)
pip install "thekaveh-nnx[domains]"             # vision + graph + plots
pip install "thekaveh-nnx[tensorboard]"         # TensorBoardCallback
pip install "thekaveh-nnx[wandb]"               # WandbCallback
pip install "thekaveh-nnx[onnx]"                # NNModel.to_onnx (legacy exporter)
pip install "thekaveh-nnx[onnx-dynamo]"         # NNModel.to_onnx(dynamo=True): adds onnxscript
pip install "thekaveh-nnx[onnx-runtime]"        # ONNX Runtime CPU execution: nnx.export_conformance parity profiles
pip install "thekaveh-nnx[quantize]"            # nnx.quantize_int8 (torchao PTQ INT8)
pip install "thekaveh-nnx[hub]"                 # safetensors checkpoints + HuggingFace Hub publish/load
pip install "thekaveh-nnx[embeddings]"          # nnx.embeddings: FAISS export + sentence-transformers
pip install "thekaveh-nnx[lm]"                  # TransformerNN + HF tokenizer + generate()
pip install "thekaveh-nnx[gguf-write]"          # experimental NNx GGUF writer
pip install "thekaveh-nnx[viz]"                 # nnx.viz: summary + weight_histogram + activation_map + attribute + gradient_flow + netron_export (netron_export also needs [onnx])
pip install "thekaveh-nnx[viz-interactive]"     # adds Netron browser viewer for nnx.viz.netron_export(launch=True)
pip install "thekaveh-nnx[jev]"                 # nnx.decisions.JevProvider (hosted Jev models)
pip install "thekaveh-nnx[optuna]"              # nnx.search: budgeted experiment search
pip install "thekaveh-nnx[notebook]"            # IPython clear_output for legacy function callbacks in notebooks
pip install "thekaveh-nnx[docs]"                # local documentation build toolchain
```

For a reproducible contributor environment, install with the committed resolver state:

```bash
python -m pip install -r requirements-tools.txt
uv sync --all-extras --frozen
```

For an editable install with the test and lint toolchain, see [CONTRIBUTING.md §1](CONTRIBUTING.md#1-getting-set-up).

## 3. Quickstart

End-to-end CPU example — a tiny random-tensor classification run:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    NNModel, NNParams, NNModelParams, NNTrainParams,
    NNOptimParams, NNSchedulerParams,
    Activations, Devices, Losses, Nets, Optims,
    EarlyStopping,
)

# 1. Data
X_train, y_train = torch.randn(256, 8), torch.randint(0, 3, (256,))
X_val,   y_val   = torch.randn(64, 8),  torch.randint(0, 3, (64,))
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=32)

# 2. Model
net_params   = NNParams(input_dim=8, output_dim=3, hidden_dims=[32, 16],
                        dropout_prob=0.1, activation=Activations.RELU)
model_params = NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU,
                             loss=Losses.CROSS_ENTROPY)
model = NNModel(net_params=net_params, params=model_params)

# 3. Train
train_params = NNTrainParams(
    n_epochs=10,
    train_loader=train_loader,
    val_loader=val_loader,
    optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2,
                        momentum=(0.9, 0.999), weight_decay=5e-5),
    scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5,
                                patience=3, cooldown=1, threshold=1e-3),
)
run = model.train(params=train_params, callbacks=[EarlyStopping(patience=5)])

# 4. Use it
print(f"trained {len(run.idps)} iterations; saved under runs/{run.id}/")
logits, classes = model.predict(X=X_val.numpy())  # returns PredictResult(logits=..., classes=...)
```

Run ids are content-addressed, so running the same script again in the same directory raises `FileExistsError`. Delete `runs/` first, or pass `overwrite_existing=True` to `NNTrainParams`.

## 4. Advanced patterns

These patterns use the current source; an older PyPI release can lack some of them (see [Release or source](#21-release-or-source)).

### 4.1. Switching networks

Change the `Nets` enum value passed to `NNModelParams`; NNModel constructs the underlying network for you:

```python
NNModelParams(net=Nets.GRAPH_CONV,  device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
NNModelParams(net=Nets.GRAPH_SAGE,  device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
NNModelParams(net=Nets.GRAPH_ATT,   device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
# Then pass NNParams(..., n_heads=4) for GRAPH_ATT.
# Use NNGraphDataset (PyG NeighborLoader-backed) to feed batches.
```

See [examples/](examples/) for runnable end-to-end scripts.

### 4.2. Probabilities

`predict()` returns logits and classes. For probabilities, declare the task explicitly:

```python
from nnx import ProbabilitySpec
proba = model.predict_proba(X_val.numpy(), ProbabilitySpec(kind="categorical", class_axis=1))
proba.probabilities, proba.decoded, proba.sample_ids  # softmax rows sum to 1; ids align with X_val
```

### 4.3. Reproducibility

```python
from nnx import set_seed, dataloader_worker_init_fn
set_seed(42)                                        # pins torch / numpy / python / cudnn
DataLoader(..., worker_init_fn=dataloader_worker_init_fn)
NNTrainParams(seed=42, ...)                         # pins again at train() entry
```

### 4.4. Warm-resume training

```python
run = model.train(params=NNTrainParams(n_epochs=10, ...))

# Build a fresh NNModel and continue from run's LAST checkpoint
# (optimizer topology/state, scheduler, scaler, epoch, loader generators,
# and Python/NumPy/PyTorch CPU/CUDA/MPS RNG state preserved):
NNModel(net_params=..., params=...).train(params=NNTrainParams(
    n_epochs=10,
    resume_from_run_id=run.id,
    resume_from_checkpoint="last",   # or "best" / "first" / "q1" / "q2" / "q3"
    ...
))
```

Callback and step state continues too. `EarlyStopping`'s patience, the I-JEPA EMA target encoder, objective state and any object passed through `train(..., components=[...])` are versioned components (`nnx.components`), validated before anything changes. `resume_mode` (`"auto"`, `"stateful"` or `"weights_only"`) chooses what is restored, and `run.resume_status` reports it.

`resume_epochs="planned"` continues a run's own plan up to its `n_epochs`; on an experiment plan, use `resuming(..., epochs="planned")`. With deterministic single-process training and in-process loaders, a run split at epoch boundaries equals the uninterrupted run bit for bit. Every checkpoint is a SHA-256-digested resume point, so a torn, mixed or corrupted one is refused with its reason (`ResumePointError`). The multi-optimizer `Trainer` resumes through `NNTrainerParams.builder().resume_from(run_id)`. See [Concepts §14](docs/concepts.md#14-resuming-training).

> **Scope:** Loader and sampler generators are restored by stable seed identity. Exact continuation still requires `train_loader.num_workers=0`, because worker-local RNG state cannot be reconstructed. Resume happens at completed-epoch boundaries.

### 4.5. Custom metrics

```python
NNTrainParams(
    ...,
    extra_metrics={
        # Called as fn(y_true, y_pred): truth first, decoded predictions second.
        "my_metric": lambda y_true, y_pred: float((y_true == y_pred).mean()),
    },
)
# Available on idp.train_edp.extra / idp.val_edp.extra and survives NNRun.load.

# Declared metrics (labels / probabilities / continuous inputs, full-sample values)
# and one monitor for BEST, ReduceLROnPlateau and an EarlyStopping given it:
monitor = MonitorSpec(metric="nll", min_delta=0.01)
NNTrainParams(..., metrics=[MetricSpec("nll"), MetricSpec("brier")], monitor=monitor)
# ... model.train(params, callbacks=[EarlyStopping(monitor=monitor, patience=5)])
```

See [Concepts §6.4](docs/concepts.md#64-named-metrics-and-monitors): custom metrics register with `nnx.register_metric`, training records carry a whole-epoch `train_summary` and the epoch's monitor `selection`, and named-monitor runs never claim `runs/best`.

### 4.6. Visualization

```python
from nnx import VisUtils
fig = VisUtils.confusion_matrix(y_true, y_pred, class_names=["a","b","c"])
fig.show()
df = VisUtils.classification_report(y_true, y_pred)  # DataFrame
```

### 4.7. Auto device detection

```python
from nnx import Devices
NNModelParams(net=Nets.FEED_FWD, device=Devices.get(), loss=Losses.CROSS_ENTROPY)
# Devices.get() picks MPS (Apple) > CUDA > CPU.
```

### 4.8. Precision (FP32 / FP16 / BF16)

```python
from nnx import PrecisionPolicy
NNModelParams(..., precision=PrecisionPolicy("bf16"))                   # CPU, or CUDA with bf16 support
NNModelParams(..., precision=PrecisionPolicy("fp16", fallback="fp32"))  # CUDA fp16, else recorded fp32
```

An unsupported request fails before any run is reserved, unless it names a
fallback. `run.precision` records the requested and effective precision, the
fallback reason and TF32. The legacy `mixed_precision=True` (CUDA FP16,
silently FP32 elsewhere) keeps its meaning and run id. See
[Quickstart §2.2](docs/quickstart.md#22-precision-fp32-fp16-and-bf16).

### 4.9. Scheduler choices

By default the scheduler is `ReduceLROnPlateau` driven by the params dataclass. Pass `kind=` to switch:

```python
from nnx import Schedulers
NNSchedulerParams(..., kind=Schedulers.COSINE_ANNEALING, T_max=100)
# Or: STEP, ONE_CYCLE, LINEAR_WARMUP_DECAY
```

### 4.10. Loading a run

```python
from nnx import NNRun, NNCheckpoint, Checkpoints
run  = NNRun.load(id="<md5>")                              # rehydrate idps + params
ckpt = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
model = NNModel.from_checkpoint(checkpoint=ckpt)
```

`from_checkpoint` also replays persisted topology transforms before loading weights. In particular, QAT-produced `LAST` checkpoints record their torchao recipe and reload as converted quantized models without downstream prepare/convert code.

## 5. Architecture

![NNx architecture](docs/assets/architecture.png)

The architecture separates user-facing orchestration, per-batch extension hooks, callback lifecycle, and persisted run artifacts.

**Reading the diagram top-to-bottom (summary):**

1. **User code** instantiates **`NNModel`** (supervised) or **`Trainer`** (multi-optimizer for GAN / actor-critic).
2. The **`train_step_fn` / `eval_step_fn` / `trainer_step_fn`** hooks are the training extension bus. `diffusion`, `paradigms`, `quantize`, and `embeddings` provide hook-compatible factories; the diffusion, paradigm and embeddings steps share one step-finalization helper. The other specialization packages compose around the loop without hooks: model transforms (`finetune`, `peft`, `prune`, `surgery`), exchange formats (`interop`), inference utilities (`generation`) and diagnostics (`viz`).
3. The **training-loop internals** run the per-batch step hook inside the data loop and save checkpoints once per completed epoch. Each scheduler steps once per completed epoch by default (`clock="epoch"`), or once per committed optimizer update with `NNSchedulerParams(clock="optimizer_update")`. Paradigm and diffusion step factories also apply a NaN guard and gradient clipping.
4. The **Callback bus** fires `on_train_begin / on_epoch_begin / on_epoch_end / on_train_end` to every registered listener (`EarlyStopping`, `LRMonitor`, `ModelCheckpoint`, `TensorBoardCallback`, `WandbCallback`).
5. After `on_epoch_end`, **`NNRun`** and **`NNCheckpoint`** commit durable state in order: history → LAST → phase/BEST → deferred callback checkpoints.

[Concepts §1](docs/concepts.md#1-architecture) gives the full 8-layer breakdown, and [Architecture](docs/architecture.md) gives the exact callback and commit ordering.

## 6. Documentation

The documentation below covers the public API, architecture, extension contracts, and runnable examples.

### 6.1. Conceptual + reference

- [Concepts](docs/concepts.md) — architecture, persistence layout, the callback protocol and every specialization in detail.
- [Capability reference](docs/capabilities.md) — the main capabilities with their entry points, grouped by task.
- [Quickstart](docs/quickstart.md) — a runnable example with variations to copy and adapt.
- [Language modeling](docs/lm.md) — train a small decoder-only `TransformerNN` end to end on CPU and generate text.
- [Direct Preference Optimization](docs/dpo.md) — fine-tune a language model on `(prompt, chosen, rejected)` preference pairs against a frozen reference policy, without reward modeling or RL.
- [Typed decisions](docs/decisions.md) — ask `Choice`, `Boolean` or `Score` questions and get validated, labelled probabilities. Covers the providers, decision jobs and the provider benchmark.
- [I-JEPA](docs/jepa.md) — self-supervised pretraining of a vision encoder by predicting masked patches in latent space.
- [Experimental GGUF export](docs/gguf.md) — write and inspect an NNx-tagged GGUF file, and the current stock-runtime limits.
- [HuggingFace Hub](docs/hub.md) — safetensors checkpoints and `save_pretrained` / `push_to_hub` / `from_pretrained` on `NNModel`.
- [Embeddings + FAISS export](docs/embeddings.md) — train a text embedder with contrastive learning and export a FAISS index.
- [Model surgery](docs/surgery.md) — the `nnx.surgery` primitives, their function-preservation contract, and the load, edit, refine and save workflow.
- [API reference](docs/api.md) — signatures and docstrings for public symbols.
- [Comparison vs Lightning / HF / fastai / Composer](docs/comparison.md) — when to use NNx or another PyTorch training toolkit, compared axis by axis.
- [Architecture](docs/architecture.md) — the system overview and exact training lifecycle, including callback, scheduler, history, and checkpoint ordering.
- [External dependency contracts](docs/external-contracts.md) — optional integrations, version sources and verification coverage; read it before changing dependency ranges or publish workflows.

### 6.2. Workflow + history

- [Examples catalog](examples/README.md) — an ordered tour of the runnable scripts under `examples/`, from the core loop to specialized methods.
- [Test import boundaries](tests/README.md) — when tests should use the public facade and when a deep implementation import is intentional.
- [Contributing](CONTRIBUTING.md) — setup, back-compat invariants, test policy, the omit-when-default rule for params, what we will and won't merge.
- [Security policy](SECURITY.md) — supported versions, private vulnerability reporting, and the checkpoint trust boundary.
- [Changelog](CHANGELOG.md) — release history (Keep-a-Changelog format), back-compat migration notes, and on-disk run.id hash shifts when they occur.

## 7. Project

### 7.1. Status

Alpha. Before 1.0, bug fixes can change behavior (see the [Changelog](CHANGELOG.md)), but public APIs are renamed only when they cannot work as designed.

### 7.2. Contributing

Bug reports and PRs are welcome via GitHub issues. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, verification, and the complete style guide.

### 7.3. License

Apache License 2.0. Copyright 2026 Kaveh Razavi. See [LICENSE](LICENSE).
