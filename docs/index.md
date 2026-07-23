# NNx

Lightweight PyTorch training / eval / visualization toolkit, with first-class support for graph neural networks (GCN / GraphSAGE / GAT). Originally extracted from `thekaveh/ml` to underpin training loops, checkpointing, and result visualization across notebook-based experiments; now standalone.

## 1. Why NNx

If you've ever found yourself rewriting the same training loop, the same checkpoint shuffling, and the same metric plotting from project to project, that's NNx's purpose: a tight, opinionated layer that owns the boring parts so you can focus on the model.

### 1.1. Core capabilities

- **Generic training loop** — callbacks, early stopping, schedulers, AMP, gradient clipping, gradient accumulation, and seeded reproducibility.
- **Content-addressed checkpoint management** — FIRST / Q1 / Q2 / Q3 / LAST / BEST tags, ordered history → LAST → ancillary commits, and a `runs/best` pointer that advances only after the final durable save.
- **Warm-resume training** — restore model, validated optimizer topology, scheduler, scaler, completed epoch, loader generators, and Python/NumPy/PyTorch CPU/CUDA/MPS RNG state from a matching generation-addressed sidecar.
- **Custom metrics injection** — plug in any `callable(Y_true, Y_pred) -> float` via `NNTrainParams.extra_metrics`.
- **TensorBoard and Weights & Biases callbacks** — opt-in via extras.
- **ONNX export** — `NNModel.to_onnx(path, example_input)` with a single method call. Defaults to the legacy `torch.onnx.export` path (no extra deps); pass `dynamo=True` (with `thekaveh-nnx[onnx-dynamo]` installed) to use PyTorch's newer `torch.export`-based exporter.

### 1.2. Specializations

- **Fine-tuning (transfer learning)** — glob-pattern layer freezing, external state-dict loading, per-layer-group learning rates.
- **Parameter-efficient fine-tuning (PEFT)** — **LoRA + DoRA + IA3 + Prefix-Tuning + Prompt-Tuning + Adapters**. Per-method `save_*_weights` / `load_*_weights` persist only the trainable delta.
- **Multi-optimizer `Trainer`** — parallel to `NNModel.train()` for GAN / actor-critic workflows with a name-keyed dict of optimizers scoped via `NNParamGroupSpec`.
- **Quantization** — PTQ INT8 weight-only (`quantize_int8`) and QAT 8da4w (`qat_train_step_factory` + `QATLifecycleCallback`) via `torchao`.
- **Pruning** — magnitude unstructured (checkpoint-safe) and 2:4 semi-structured via torchao.
- **Model surgery** — `widen` / `deepen` (function-preserving Net2Net), `drop_layer`, `low_rank_factorize` (SVD), `expand_embedding`.
- **Diffusion (DDPM)** — noise-prediction training and reverse-diffusion sampling.
- **Training paradigms** — knowledge distillation (Hinton + FitNets-style feature-KD), contrastive (SimCLR / NT-Xent), Mixup, CutMix, sparse top-k Mixture-of-Experts (`MoELinear` + Switch-style aux loss), I-JEPA self-supervised pretraining, DPO preference fine-tuning, Born-Again iterated self-distillation.
- **Language modeling** — `TransformerNN` (decoder-only: RMSNorm + RoPE + SwiGLU + KV-cache) + `NNTransformerParams` + `NNTokenizerParams` + `GenerativeNNModel.generate()` with greedy / top-k / top-p / repetition-penalty sampling.
- **Embeddings + FAISS** — contrastive text-embedder training + FAISS index export for downstream RAG.
- **Experimental GGUF export** — write and inspect an NNx-tagged `.gguf`, or prepare a bundle for a runtime patched to support the NNx architecture. Stock llama.cpp, Ollama, and LM Studio do not implement `nnx_transformer`.
- **HuggingFace Hub** — `save_pretrained` / `push_to_hub` / `from_pretrained` on `NNModel` via the `PyTorchModelHubMixin`, plus safetensors checkpoint format.
- **Model-internals visualization** — `nnx.viz.summary` (torchinfo) + `weight_histogram` + `activation_map` + `attribute` (Captum) + `gradient_flow` (per-layer gradient-norm diagnostic) + `netron_export`.
- **Training-loop diagnostics** — `nnx.lr_finder(model, train_loader, *, loss_fn, ...)` returns the Smith-2017 suggested one-cycle `max_lr` plus a Plotly figure while restoring model state, mixed per-module modes, loader generators, and all global RNG streams.
- **Type-checked downstream** — PEP 561 `py.typed` marker so consumers' `pyright` / `mypy` honor the public-surface annotations.

## 2. Where to next

### 2.1. Get running

- [Quickstart](quickstart.md) — five minutes to a trained model, paste-runnable.

### 2.2. Understand the design

- [Concepts](concepts.md) — what an `NNRun` is, where things land on disk, how the enum-as-factory pattern works, how the twelve specialization subpackages compose.

### 2.3. Deep-dive guides

- [Language modeling](lm.md) — train a tiny `TransformerNN` end-to-end (CPU-friendly).
- [Direct Preference Optimization](dpo.md) — fine-tune an LM on `(prompt, chosen, rejected)` preference pairs.
- [I-JEPA](jepa.md) — masked-patch latent-prediction self-supervised pretraining.
- [Model surgery](surgery.md) — function-preserving Net2Net + low-rank + drop primitives.
- [Embeddings + FAISS](embeddings.md) — contrastive training + RAG-ready export.
- [HuggingFace Hub](hub.md) — safetensors + Hub publish/load.
- [Experimental GGUF export](gguf.md) — container inspection, quantization notes, and runtime limits.
- [Comparison vs Lightning / HF / fastai / Composer](comparison.md) — scope-explicit decision matrix for picking the right PyTorch training toolkit.
- [Architecture](architecture.md) — package relationships plus exact callback and durable-commit ordering.
- [External dependency contracts](external-contracts.md) — frozen integrations, release publication, and intentionally gated external checks.

### 2.4. Look things up

- [API Reference](api.md) — auto-generated from docstrings (sections 1–20).
- [Examples catalog](_project/Examples.md) — annotated index of the runnable scripts under `examples/`.
- [CONTRIBUTING](_project/Contributing.md) — editable install, dev toolchain, PR workflow.
- [Security policy](_project/Security-Policy.md) — supported versions and private reporting instructions.
- [CHANGELOG](_project/Changelog.md) — user-visible changes per PR.

## 3. Status

Alpha. API is stable for the existing `thekaveh/ml` notebook consumer; pre-1.0 means we'll fix bugs (see [CHANGELOG](_project/Changelog.md)) without renaming public APIs unless they're broken in ways notebooks can't work around.
