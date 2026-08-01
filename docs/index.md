<p align="center" class="nnx-banner">
  <img src="assets/nnx-poster.png" alt="NNx neural training banner" width="100%">
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
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="PyTorch Geometric" src="https://img.shields.io/badge/PyTorch_Geometric-2.4%2B-3C2179">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white">
  <img alt="pandas" src="https://img.shields.io/badge/pandas-2.0%2B-150458?logo=pandas&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="Plotly" src="https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?logo=plotly&logoColor=white">
</p>

<p align="center" class="nnx-optional-stack">
  <img alt="TensorBoard" src="https://img.shields.io/badge/TensorBoard-optional-FF6F00?logo=tensorflow&logoColor=white">
  <img alt="Weights & Biases" src="https://img.shields.io/badge/Weights_%26_Biases-optional-FFBE00?logo=weightsandbiases&logoColor=black">
  <img alt="ONNX" src="https://img.shields.io/badge/ONNX-optional-005CED?logo=onnx&logoColor=white">
  <img alt="torchao" src="https://img.shields.io/badge/torchao-optional-EE4C2C">
  <img alt="Hugging Face" src="https://img.shields.io/badge/Hugging_Face-optional-FFD21E">
  <img alt="safetensors" src="https://img.shields.io/badge/safetensors-optional-4B5563">
  <img alt="FAISS" src="https://img.shields.io/badge/FAISS-optional-0467DF">
</p>

<!-- executive-summary:start -->
NNx is a lightweight PyTorch toolkit for repeatable training, evaluation, and visualization. It owns the routine experiment infrastructure: frozen configuration objects, supervised train/eval/predict orchestration, callbacks, schedulers, metrics, and content-addressed checkpoints that support reliable resume and inspection. Models and per-step logic remain replaceable, so the same loop can serve standard networks, graph neural networks, transformers, diffusion, representation learning, fine-tuning, and multi-optimizer workflows. Focused modules add PEFT, quantization, pruning, model surgery, embeddings, export, and diagnostics without forcing those concerns into the core loop. NNx is aimed at researchers and engineers who want transparent PyTorch code and durable experiments without adopting a larger training platform.
<!-- executive-summary:end -->

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
