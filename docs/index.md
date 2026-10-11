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

## 1. Why NNx

NNx owns the code that repeats across training projects: the loop, checkpoint handling and metric reporting. You write the model and, when you need to, the per-step logic.

### 1.1. Capabilities

- **Training loop** — `NNModel` train, evaluate and predict, with callbacks, schedulers, FP32/FP16/BF16 precision, gradient accumulation and clipping, declared metrics, and a multi-optimizer `Trainer`.
- **Runs and resume** — content-addressed run directories, checkpoint tags, verified warm resume, experiment plans and provenance records.
- **Tasks** — classification, regression, multilabel, ranking, graph classification and link prediction.
- **Training methods** — custom step functions and objectives, distillation, contrastive learning, Mixup and CutMix, Mixture of Experts, diffusion, I-JEPA and DPO.
- **Fine-tuning and transforms** — layer freezing, LoRA, DoRA, IA3, prefix and prompt tuning, quantization, pruning and model surgery.
- **Networks and export** — feed-forward, convolutional, graph, transformer and ViT networks; a small language-model path; ONNX, Hub, safetensors and experimental GGUF export.
- **Evaluation and decisions** — reproducible splits, calibration, abstention, multi-seed comparison and typed decisions.

The [Capability reference](capabilities.md) describes each item with its entry points. [Concepts](concepts.md) explains how the parts fit together.

## 2. Where to next

### 2.1. Get running

- [Install](../README.md#2-install) — a PyPI release or the current source, and the optional extras.
- [Quickstart](quickstart.md) — a runnable example that trains a model in seconds.

### 2.2. Understand the design

- [Concepts](concepts.md) — runs and their on-disk layout, the enum factories, and how the specialization subpackages compose.

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

- [Capability reference](capabilities.md) — the main capabilities with their entry points, grouped by task.
- [API Reference](api.md) — auto-generated from docstrings (sections 1–20).
- [Examples catalog](../examples/README.md) — annotated index of the runnable scripts under `examples/`.
- [CONTRIBUTING](../CONTRIBUTING.md) — editable install, dev toolchain, PR workflow.
- [Security policy](../SECURITY.md) — supported versions and private reporting instructions.
- [CHANGELOG](../CHANGELOG.md) — release history and unreleased changes.

## 3. Status

Alpha. Before 1.0, bug fixes can change behavior (see the [Changelog](../CHANGELOG.md)), but public APIs are renamed only when they cannot work as designed.
