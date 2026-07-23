# NNx

Lightweight PyTorch training / eval / visualization toolkit. First-class support for graph neural networks (GCN / GraphSAGE / GAT). Originally extracted from `thekaveh/ml` to underpin training loops, checkpointing, and result visualization across notebook-based experiments; now standalone.

## 1. Overview

NNx owns the boilerplate around supervised training so you can focus on the model: it builds the network from frozen-dataclass configs, runs the train / eval / predict loop, manages checkpoints under a content-addressed `runs/<id>/` directory, dispatches a documented Callback lifecycle, and exposes pluggable extension points for fine-tuning, multi-optimizer training, diffusion, alternative training paradigms, and parameter-efficient fine-tuning.

### 1.1. Architecture

![NNx architecture](docs/assets/architecture.svg)

The architecture separates user-facing orchestration, per-batch extension hooks, callback lifecycle, and persisted run artifacts.

**Reading the diagram top-to-bottom (summary):**

1. **User code** instantiates **`NNModel`** (supervised) or **`Trainer`** (multi-optimizer for GAN / actor-critic).
2. The **`train_step_fn` / `eval_step_fn` / `trainer_step_fn`** hooks are the training extension bus. `diffusion`, `paradigms`, `quantize`, and `embeddings` provide hook-compatible factories; `_step_helpers` supplies shared step finalization. The remaining specialization packages provide model transforms (`finetune`, `peft`, `prune`, `surgery`), exchange formats (`interop`), inference utilities (`generation`), and diagnostics (`viz`) that compose around the loop rather than injecting hooks.
3. The **Training-loop internals** run `_step_scheduler` and `_save_checkpoints` each batch / epoch; paradigm/diffusion step factories additionally route through `finalize_step` (NaN guard + grad-clip).
4. The **Callback bus** fires `on_train_begin / on_epoch_begin / on_epoch_end / on_train_end` to every registered listener (`EarlyStopping`, `LRMonitor`, `ModelCheckpoint`, `TensorBoardCallback`, `WandbCallback`).
5. **`NNRun`** and **`NNCheckpoint`** write to **`runs/<id>/`** atomically after every epoch.

See [docs/concepts.md §1](docs/concepts.md#1-architecture) for the full 8-layer breakdown.

### 1.2. Capabilities at a glance

- **Generic training loop** — callbacks, early stopping, schedulers (`Schedulers` enum: `REDUCE_LR_ON_PLATEAU` / `STEP` / `COSINE_ANNEALING` / `ONE_CYCLE` / `LINEAR_WARMUP_DECAY`), AMP, gradient clipping, gradient accumulation, seeded reproducibility, custom metrics.
- **Content-addressed persistence** — `NNRun` saves `run.yaml` + `idps.csv` + `metadata.yaml` under `runs/<id>/` (where `id` is the md5 of `state()`); incremental writes after every epoch survive `KeyboardInterrupt`. `NNCheckpoint` saves at six tags (FIRST / Q1 / Q2 / Q3 / LAST / BEST) with versioned `.opt.pt` training-state bundles for warm resume.
- **`train_step_fn` hook** — swap the per-batch supervised step for any user-supplied function. Unblocks autoencoder / VAE / link-prediction / recommendation / diffusion / KD / SimCLR / Mixup / CutMix paradigms without modifying NNx internals.
- **Fine-tuning (transfer learning)** — `nnx.finetune.{freeze, unfreeze, load_pretrained, NNParamGroupSpec, build_param_groups}` plus `NNModel.{freeze, unfreeze, export_state_dict}`. Glob-pattern layer freezing, external state-dict loading with optional key remapping, per-layer-group learning rates via `NNOptimParams.param_groups`.
- **Multi-optimizer `Trainer`** — `nnx.trainer.Trainer` parallels `NNModel.train()` for scenarios that need disjoint optimizers (GAN G/D, actor-critic). Accepts a name-keyed dict of `NNOptimParams`; each entry's `NNParamGroupSpec` scopes the optimizer under strict-partition semantics.
- **Diffusion (DDPM)** — `nnx.diffusion.{NoiseSchedulers, DiffusionMLP, diffusion_train_step_factory, sample}`. LINEAR / COSINE noise schedules, a small conditional MLP denoiser, a DDPM-style training step factory that plugs into the `train_step_fn` hook, and a reverse-diffusion sampler.
- **Training paradigms** — `nnx.paradigms.{kd, feature_kd, simclr, mixup, cutmix, moe, jepa, dpo}_train_step_factory` plus `born_again_train`. Hinton-style knowledge distillation (teacher frozen, soft+hard loss mix), FitNets-style **feature distillation** (`feature_kd_train_step_factory` adds an MSE term between named teacher/student intermediate activations via forward hooks, mixed in with `beta`), SimCLR contrastive (NT-Xent loss exposed), Mixup and CutMix batch augmentation, sparse top-k **Mixture-of-Experts** (softmax-gated routing + Switch-style load-balancing aux loss + drop-in `nnx.MoELinear`), **I-JEPA** self-supervised pretraining (masked patches → latent prediction against an EMA target encoder; ships with `JEPAPredictor`, `build_target_encoder`, `update_ema`, `random_block_mask`, and a small `ViTNN` encoder), **Born-Again Networks** (iterated self-distillation across G generations), and **DPO** (Rafailov et al. 2023 — preference-pair fine-tuning against a frozen reference policy via the chosen-vs-rejected log-ratio objective). All share an internal `_step_helpers.finalize_step` for grad-clip + NaN guard.
- **Parameter-efficient fine-tuning (PEFT) — LoRA + DoRA + IA3 + Prefix + Prompt + Adapters** — `nnx.peft.{LoRALinear, apply_lora_to, save_lora_weights, load_lora_weights, AdapterLayer, DoRALinear, apply_dora_to, IA3Linear, apply_ia3_to, save_ia3_weights, load_ia3_weights, PrefixTuner, PromptTuner, save_prefix_weights, load_prefix_weights, save_prompt_weights, load_prompt_weights}`. **LoRA** wraps `nn.Linear` submodules in-place with a frozen base + trainable low-rank residual (B is zero-initialized so output at step 0 equals the pretrained behavior). **DoRA** (NVIDIA ICML 2024 Oral) extends LoRA with a trainable per-output-row magnitude vector and often beats LoRA at the same rank with only `out_features` extra params per layer. **IA3** (NeurIPS 2022) is the smallest adapter in the family: a single learned per-output-dim scaling vector applied to a frozen Linear's output. **PrefixTuner** prepends a learned key/value prefix to every attention layer of a frozen `TransformerNN`; **PromptTuner** prepends learned soft-prompt embeddings ahead of the input tokens. `save_*_weights` persist only the trainable delta for each method.
- **Quantization** — torchao-based **PTQ INT8 weight-only** via `nnx.quantize.quantize_int8(model)` (one call, no calibration data, no retraining; returns a new `NNModel` whose `net.Linear` weights are stored in int8 per-channel with FP32 activations; the quantized model still ONNX-exports) and **QAT 8da4w** via `nnx.quantize.{qat_train_step_factory, QATLifecycleCallback}` (Int8DynActInt4WeightQATQuantizer fake-quant during training, real-quant on convert). Opt-in extra: `pip install thekaveh-nnx[quantize]`.
- **Pruning** — `nnx.prune.magnitude_prune` (mask-based unstructured, checkpoint-safe) and `nnx.prune.semi_structured_24` (2:4 semi-structured via torchao, Ampere+ hardware speedups).
- **Model surgery — Net2Net + drop + low-rank + embedding** — `nnx.surgery.{widen, deepen, drop_layer, low_rank_factorize, expand_embedding}`. `widen` and `deepen` are function-preserving Net2Net edits (Chen/Goodfellow/Shlens, ICLR 2016) — the surged module's forward output matches the original's *before* refinement, so `NNModel.train()` can resume immediately without an accuracy cliff. `low_rank_factorize` is SVD truncation on a Linear (exact at max rank, Eckart-Young-bounded below it). `drop_layer` replaces a named layer with `nn.Identity`; `expand_embedding` grows an Embedding's row count and returns a frozen-mask for the original rows. Every primitive returns a fresh `nn.Module` and composes with `NNModel.train()` for the "refine after surgery" loop.
- **Embeddings — contrastive trainer + FAISS export** — `nnx.embeddings.{ContrastiveTextDataset, train_contrastive, embed_texts, text_contrastive_train_step_factory, export_to_faiss, export_to_safetensors}`. Train a domain-specific text embedder from `(anchor, positive)` pairs via the existing NT-Xent machinery, then export to a FAISS index file that any RAG framework (LangChain / LlamaIndex / Haystack / raw FAISS) can consume. NNx's job ends at the FAISS index — chunking / reranking / prompt orchestration live downstream. Optional dep: `pip install "thekaveh-nnx[embeddings]"` for `faiss-cpu` + `sentence-transformers`.
- **Networks** — `FeedFwdNN`, `FeedFwdMoENN`, `ConvNN`, `GraphConvNN` / `GraphSageNN` / `GraphAttNN` (all built on the shared `GraphNNBase`), `TransformerNN` (decoder-only LM: RMSNorm + RoPE + SwiGLU + tied embeddings + KV-cache), and `ViTNN` (small ViT encoder used as the I-JEPA backbone).
- **Language modeling (opt-in via `thekaveh-nnx[lm]`)** — `TransformerNN` + `NNTransformerParams` + `NNTokenizerParams` (HF Rust BPE wrapper) + `GenerativeNNModel.generate(prompt, ...)` with **KV-cache acceleration** for autoregressive decoding (1.9× speedup on CPU at 128 tokens, larger on GPU / longer contexts within `max_seq_len`; past the window the cache rebuilds per step and converges to full-recompute cost) and greedy / top-k / top-p / repetition-penalty sampling via a `LogitsProcessor` chain. See [docs/lm.md](docs/lm.md) for the full walkthrough; `examples/11_tinystories_lm.py` ships an end-to-end TinyStories-class training run.
- **Experimental GGUF export (opt-in via `thekaveh-nnx[gguf-write]`)** — `nnx.interop.write_gguf(model, tokenizer, path)` writes a structurally valid GGUF artifact with NNx tensor names and `general.architecture=nnx_transformer`. Stock llama.cpp, Ollama, and LM Studio do not implement that architecture; use the output for inspection or a reader explicitly patched for NNx. See [docs/gguf.md](docs/gguf.md).
- **HuggingFace Hub (opt-in via `thekaveh-nnx[hub]`)** — `NNModel` mixes in `PyTorchModelHubMixin`: `save_pretrained` / `push_to_hub` / `from_pretrained`, with safetensors as an opt-in checkpoint format via `NNCheckpoint.to_file(format="safetensors")`. See [docs/hub.md](docs/hub.md).
- **Datasets** — `NNDataset` (torchvision `VisionDataset` wrapper), `NNGraphDataset` (PyG single-graph wrapper using `NeighborLoader`), `NNTabularDataset` (pandas DataFrame → train/val/test loaders), `NNPreferenceDataset` (tokenized `(prompt, chosen, rejected)` preference triples for DPO).
- **Params** — frozen, kw-only, slotted dataclasses for every config knob: `NNParams`, `NNModelParams`, `NNTrainParams`, `NNOptimParams`, `NNSchedulerParams`, `NNTrainerParams`. Every params object round-trips through `state()` / `from_state()`. New fields omit themselves from `state()` when at their default so existing `run.id` hashes are preserved.
- **Fluent params construction** — `NNSchedulerParams.builder()`, `NNOptimParams.builder()`, `NNTransformerParams.builder()`, and `NNTrainerParams.builder()` (the composite, wraps the prior two for the multi-optim Trainer) expose variant-gated `.adam(...)` / `.sgd(...)` / `.one_cycle(...)` / etc. methods so the user can't construct an invalid kind/field combination. `LogitsChain.builder()` extends the pattern to the LM-decoding path — chain custom logit processors in any order; the Builder sorts them into NNx's canonical order (matching `generate()`'s inline-kwargs chain) before decoding runs. All Builders are purely additive; the existing direct-kwarg ctors keep working.
- **Enums-as-factories** — `Nets`, `Losses`, `Optims`, `Schedulers`, `Activations`, `Devices`, `Checkpoints`, `NoiseSchedulers`. Each enum value's `__call__` constructs the underlying object; adding a new option is a single-place change.
- **Callbacks** — `Callback` base class with `on_{train,epoch}_{begin,end}` hooks. Stock: `EarlyStopping`, `LRMonitor`, `ModelCheckpoint` (custom-epoch tags), `TensorBoardCallback` (opt-in via `thekaveh-nnx[tensorboard]`), `WandbCallback` (opt-in via `thekaveh-nnx[wandb]`). Legacy `Callable[[List[IDP]], None]` is still accepted.
- **Visualization** — `VisUtils` (and module-level aliases) returns Plotly `Figure` objects: `confusion_matrix`, `classification_report` (returns a DataFrame), `multi_line_plot`, `scatter_plot`, `two_dim_tsne_checkpoint_logits`.
- **Model-internals viz** — `nnx.viz.summary` (Keras-style parameter table via `torchinfo`), `nnx.viz.weight_histogram` (per-layer Plotly histogram grid), `nnx.viz.activation_map` (forward-hook activation heatmaps), `nnx.viz.attribute` (Captum-backed input attribution: `integrated_gradients` / `gradient_shap` / `deep_lift` / `saliency` / `input_x_gradient` / `occlusion`, returns the attribution tensor plus a Plotly heatmap), `nnx.viz.gradient_flow` (per-layer L2 gradient-norm bar chart for vanishing/exploding diagnostics, call after `loss.backward()`), and `nnx.viz.netron_export` (write the underlying network to a `.onnx` artifact for Netron). Companion to the existing `nnx.vis_utils` run-output viz; opt-in via `pip install thekaveh-nnx[viz]` (pulls `torchinfo` + `captum`; the Netron browser viewer is `thekaveh-nnx[viz-interactive]`).
- **Reproducibility + training diagnostics** — `nnx.set_seed(seed, strict=False)` pins every RNG + cuDNN; `nnx.dataloader_worker_init_fn` for per-worker seeds; `NNTrainParams.seed` runs `set_seed` at `train()` entry. `nnx.lr_finder(model, train_loader, *, loss_fn, ...)` runs a fastai-style exponential LR sweep and returns the Smith-2017 suggested one-cycle `max_lr` plus a Plotly figure; the sweep is non-destructive (model state and training-mode are snapshotted + restored on exit).
- **Type-checked downstream** — NNx ships a PEP 561 `py.typed` marker so consumers' `pyright` / `mypy` honor the public-surface annotations on `NNModel`, the params dataclasses, callbacks, and enums (rather than seeing every symbol as `Any`).
- **ONNX export** — `NNModel.to_onnx(path, example_input)` exports the network via the legacy `torch.onnx.export` (no `onnxscript` dep needed). Pass `dynamo=True` (opt-in via `thekaveh-nnx[onnx-dynamo]`) to dispatch through PyTorch's newer `torch.export`-based exporter (default in torch>=2.9; supports >2 GB models via external data; generally faster).

## 2. Install

### 2.1. Runtime

```bash
pip install thekaveh-nnx                        # latest release from PyPI
```

Python 3.10+. Tested on 3.10 / 3.11 / 3.12. Examples in [examples/](examples/) are runnable on CPU.

### 2.2. Optional extras

```bash
pip install "thekaveh-nnx[tensorboard]"         # TensorBoardCallback
pip install "thekaveh-nnx[wandb]"               # WandbCallback
pip install "thekaveh-nnx[onnx]"                # NNModel.to_onnx validation tooling
pip install "thekaveh-nnx[onnx-dynamo]"         # NNModel.to_onnx(dynamo=True) — torch.export-based exporter
pip install "thekaveh-nnx[quantize]"            # nnx.quantize_int8 (torchao PTQ INT8)
pip install "thekaveh-nnx[hub]"                 # safetensors checkpoints + HuggingFace Hub publish/load
pip install "thekaveh-nnx[embeddings]"          # nnx.embeddings: FAISS export + sentence-transformers
pip install "thekaveh-nnx[lm]"                  # TransformerNN + HF tokenizer + generate()
pip install "thekaveh-nnx[gguf-write]"          # experimental NNx GGUF writer
pip install "thekaveh-nnx[viz]"                 # nnx.viz: summary + weight_histogram + activation_map + attribute + gradient_flow + netron_export
pip install "thekaveh-nnx[viz-interactive]"     # adds Netron browser viewer for nnx.viz.netron_export(launch=True)
pip install "thekaveh-nnx[docs]"                # mkdocs build (mkdocs-material + mkdocstrings)
```

For a reproducible contributor environment, install with the committed resolver state:

```bash
python -m pip install -r requirements-tools.txt
uv sync --all-extras --frozen
```

For local development (editable install from a git checkout, including the test/lint toolchain), see [CONTRIBUTING.md §1](CONTRIBUTING.md#1-getting-set-up).

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

## 4. Advanced patterns

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

### 4.2. Reproducibility

```python
from nnx import set_seed, dataloader_worker_init_fn
set_seed(42)                                        # pins torch / numpy / python / cudnn
DataLoader(..., worker_init_fn=dataloader_worker_init_fn)
NNTrainParams(seed=42, ...)                         # pins again at train() entry
```

### 4.3. Warm-resume training

```python
run = model.train(params=NNTrainParams(n_epochs=10, ...))

# Build a fresh NNModel and continue from run's LAST checkpoint
# (optimizer, scheduler, scaler, epoch, and RNG state preserved):
NNModel(net_params=..., params=...).train(params=NNTrainParams(
    n_epochs=10,
    resume_from_run_id=run.id,
    resume_from_checkpoint="last",   # or "best" / "first" / "q1" / "q2" / "q3"
    ...
))
```

> **Scope:** Warm-resume is supported for the supervised `NNModel.train()` path. `nnx.trainer.Trainer` (multi-optimizer) does not yet ship `.opt.<name>.pt` per-optimizer sidecars; that's a planned follow-up.

### 4.4. Custom metrics

```python
NNTrainParams(
    ...,
    extra_metrics={
        "my_metric": lambda y, y_hat: float((y == y_hat).mean()),
    },
)
# Available on idp.train_edp.extra / idp.val_edp.extra and survives NNRun.load.
```

### 4.5. Visualization

```python
from nnx import VisUtils
fig = VisUtils.confusion_matrix(y_true, y_pred, class_names=["a","b","c"])
fig.show()
df = VisUtils.classification_report(y_true, y_pred)  # DataFrame
```

### 4.6. Auto device detection

```python
from nnx import Devices
NNModelParams(net=Nets.FEED_FWD, device=Devices.get(), loss=Losses.CROSS_ENTROPY)
# Devices.get() picks MPS (Apple) > CUDA > CPU.
```

### 4.7. Mixed precision (CUDA)

```python
NNModelParams(..., mixed_precision=True)   # silently no-op on CPU/MPS
```

### 4.8. Scheduler choices

By default the scheduler is `ReduceLROnPlateau` driven by the params dataclass. Pass `kind=` to switch:

```python
from nnx import Schedulers
NNSchedulerParams(..., kind=Schedulers.COSINE_ANNEALING, T_max=100)
# Or: STEP, ONE_CYCLE, LINEAR_WARMUP_DECAY
```

### 4.9. Loading a run

```python
from nnx import NNRun, NNCheckpoint, Checkpoints
run  = NNRun.load(id="<md5>")                              # rehydrate idps + params
ckpt = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
model = NNModel.from_checkpoint(checkpoint=ckpt)
```

`from_checkpoint` also replays persisted topology transforms before loading weights. In particular, QAT-produced `LAST` checkpoints record their torchao recipe and reload as converted quantized models without downstream prepare/convert code.

## 5. Documentation

The documentation below covers the public API, architecture, extension contracts, and runnable examples.

### 5.1. Conceptual + reference

- [Concepts](docs/concepts.md) — architecture deep-dive, persistence layout, callback protocol, every specialization in detail. Read this when you want to understand how the pieces fit together (callbacks, params hashing, train_step_fn hook, multi-optim Trainer, paradigms, PEFT).
- [Quickstart](docs/quickstart.md) — paste-runnable example with variations. Read this when you want to copy a working snippet and iterate from there.
- [Language modeling](docs/lm.md) — the decoder-only Transformer path: `TransformerNN` + HF tokenizer + `GenerativeNNModel.generate()` with KV-cache. Read this when you want to train a tiny LM end-to-end on CPU.
- [Direct Preference Optimization](docs/dpo.md) — `dpo_train_step_factory` for fine-tuning a TransformerNN against `(prompt, chosen, rejected)` preference pairs via the Rafailov et al. 2023 chosen-vs-rejected log-ratio objective against a frozen reference policy. Read this when you have preference data and want to steer LM behavior post-SFT without reward modeling or RL.
- [I-JEPA](docs/jepa.md) — Joint Embedding Predictive Architecture: masked-patch → latent-prediction self-supervised pretraining against an EMA target encoder. Read this when you want to pretrain a vision encoder without pixel-reconstruction or strong augmentations.
- [Experimental GGUF export](docs/gguf.md) — `nnx.interop.write_gguf` for producing and inspecting an NNx-tagged GGUF artifact; includes the official llama.cpp `llama-quantize` build path and the current stock-runtime limitation.
- [HuggingFace Hub](docs/hub.md) — safetensors checkpoints + `save_pretrained` / `push_to_hub` / `from_pretrained` on `NNModel`. Read this when you want to publish a trained model to the Hub, load from it, or write checkpoints in a format outside-of-Python tools (ComfyUI, vLLM, AutoGPTQ) can read.
- [Embeddings + FAISS export](docs/embeddings.md) — walkthrough for training a domain-specific text embedder via contrastive learning and exporting it to a FAISS index for any RAG stack to consume.
- [Model surgery](docs/surgery.md) — walkthrough of the `nnx.surgery` primitives (`widen` / `deepen` / `drop_layer` / `low_rank_factorize` / `expand_embedding`), the function-preservation contract, before/after parameter-count tables, and the "load checkpoint → surgery → refine via `NNModel.train()` → save" pattern.
- [API reference](docs/api.md) — auto-generated from docstrings via mkdocstrings. Read this when you want the canonical signature / docstring for a public symbol.
- [Comparison vs Lightning / HF / fastai / Composer](docs/comparison.md) — honest scope-explicit comparison: when to use NNx vs Lightning vs HF Transformers vs fastai vs MosaicML Composer, axis by axis. Read this when you're picking a PyTorch training toolkit and want a real decision matrix instead of a marketing page.
- [Architecture diagram](docs/architecture.md) — the §1.1 diagram as a themed page, with a link to the standalone HTML version. Read this when the embedded SVG is hard to follow.
- [External dependency contracts](docs/external-contracts.md) — ledger of optional integrations, version sources, verification coverage, and intentionally gated real-service checks. Read this before changing dependency ranges, external CLI commands, or publish workflows.

### 5.2. Workflow + history

- [Examples catalog](examples/README.md) — ordered tour of the 26 runnable scripts under `examples/`, grouped foundational to specialized (core loop, fine-tuning, paradigms, quantization, embeddings, language modeling, GGUF inspection, self-supervised learning, pruning, surgery, explainability, DPO, and distillation variants).
- [Test import boundaries](tests/README.md) — when tests should use the public facade and when a deep implementation import is intentional.
- [Contributing](CONTRIBUTING.md) — setup, back-compat invariants, test policy, the omit-when-default rule for params, what we will and won't merge.
- [Security policy](SECURITY.md) — supported versions, private vulnerability reporting, and the checkpoint trust boundary.
- [Changelog](CHANGELOG.md) — release history (Keep-a-Changelog format), back-compat migration notes, and on-disk run.id hash shifts when they occur.

## 6. Project

### 6.1. Status

Alpha. API is stable for the existing `thekaveh/ml` notebook consumer; pre-1.0 means we'll fix obvious bugs (see [CHANGELOG](CHANGELOG.md)) without renaming public APIs unless they're broken in ways notebooks can't work around.

### 6.2. Contributing

Bug reports and PRs welcome via GitHub issues. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full setup + style guide. Running locally:

```bash
pytest                                       # full suite (~15s)
pytest tests/test_callbacks.py::test_lr_monitor_records_history  # one test
ruff check src/ tests/ examples/             # lint (gates CI)
ruff format --check src/ tests/ examples/    # format (gates CI)
mkdocs build --strict                        # docs (gates CI)
```

### 6.3. License

Apache License 2.0. Copyright 2026 Kaveh Razavi. See [LICENSE](LICENSE).
