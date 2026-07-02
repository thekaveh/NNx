# 1. External Dependency Contracts

This ledger records NNx integration points whose behavior depends on third-party packages, services, CLIs, or registries. It complements tests by making the consumed contract explicit: supported version source, what NNx relies on, how the contract is checked, and which checks are intentionally gated.

## 1.1. Purpose

NNx is a library, so most integrations are optional extras. A green unit test that mocks or skips an optional boundary does not by itself prove the boundary still matches the upstream contract. This page is the durable review checklist for those boundaries.

## 1.2. Contract Ledger

| Integration | Dependency / boundary | Version source | NNx contract relied on | Verification |
| --- | --- | --- | --- | --- |
| PyTorch training core | `torch>=2.0`, `torchvision>=0.15`, `torch_geometric>=2.4` | `pyproject.toml` runtime deps; CI installs current compatible wheels | `nn.Module` forward/backward/optimizer APIs, PyG graph loaders, torchvision dataset wrappers | Full pytest matrix in CI; graph/dataset/net tests exercise the public paths. |
| ONNX export | `onnx>=1.15`, optional `onnxscript>=0.1` | `pyproject.toml` extras `onnx`, `onnx-dynamo` | `torch.onnx.export` legacy path by default; `dynamo=True` only when the exporter supports the kwarg and `onnxscript` imports | `tests/test_to_onnx_inputs.py`, `tests/test_onnx_dynamo.py`, `tests/test_viz_netron.py`; dynamo dispatch skew is skipped only by the documented helper in `tests/conftest.py`. |
| Netron viewer | `netron>=7.0` | `pyproject.toml` extra `viz-interactive` | `netron.start(path)` launches only when `launch=True`; ONNX file export remains independent of the viewer | `tests/test_viz_netron.py` monkeypatches launch behavior and missing-package errors. |
| HuggingFace Hub | `huggingface_hub>=1.4.0`, `safetensors>=0.7.0` | `pyproject.toml` extra `hub` | `PyTorchModelHubMixin` methods and safetensors checkpoint read/write | `tests/test_hub_mixin.py`, `tests/test_checkpoint_safetensors.py`; real token-authenticated pushes are intentionally not run in CI. |
| GGUF / Ollama | `gguf>=0.19.0`; external `ollama` CLI/runtime | `pyproject.toml` extra `gguf-write`; local Ollama install outside Python deps | GGUF writer tensor names/metadata; Ollama Modelfile syntax and `ollama create -f Modelfile` workflow | `tests/test_interop_gguf_writer.py` covers writer output; examples/docs statically trace Ollama commands because CI does not run an Ollama daemon. |
| Quantization | `torchao>=0.17` | `pyproject.toml` extra `quantize` | PTQ INT8 and QAT 8da4w public quantizer APIs | `tests/test_quantize_ptq.py`, `tests/test_quantize_qat.py`; CUDA-only 2:4 sparsity behavior is hardware-gated. |
| Embeddings / FAISS | `faiss-cpu>=1.7`, `sentence-transformers>=2.7` | `pyproject.toml` extra `embeddings` | FAISS index construction/search; SentenceTransformer-like `forward(list[str]) -> Tensor[B, D]` contract | `tests/test_embeddings_contrastive.py`, `tests/test_embeddings_faiss_export.py`; downstream RAG framework adapters are documented as out of scope. |
| Language modeling data/tokenization | `tokenizers>=0.20`, `datasets>=2.20` | `pyproject.toml` extra `lm` | HF Rust BPE tokenizer training/encode/decode; optional dataset loading for examples | `tests/test_tokenizer_params.py`, `tests/test_generative_nn_model.py`; network-backed dataset downloads are not required for core CI. |
| Experiment logging | `tensorboard>=2.15`, `wandb>=0.16` | `pyproject.toml` extras `tensorboard`, `wandb` | TensorBoard writer lifecycle; W&B run lifecycle and finish semantics | Callback tests cover lifecycle; real W&B service calls are token/network-gated and not run in CI. |
| Package publishing | PyPI trusted publishing; `twine`; GitHub Actions OIDC | `.github/workflows/release.yml`, PyPI project settings | Release tags match package version; built artifacts pass `twine check`; post-publish install imports `nnx` | Release workflow performs matrix tests, package build, metadata check, publish, and fresh-venv verification. |

## 1.3. Review Rules

1. When changing a dependency range, optional extra, external CLI command, or published config shape, update this ledger in the same change.
2. Prefer tests against the real public API for the pinned or resolved version. A mock is acceptable only when the real boundary needs credentials, hardware, a daemon, or network access; record that gate here.
3. If an upstream version break is tolerated temporarily, document the skip or compatibility guard and the condition that lets it be removed.
