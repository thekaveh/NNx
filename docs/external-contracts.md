# External Dependency Contracts

This ledger records NNx integration points that depend on third-party packages,
services, CLIs, or registries. It complements tests by naming the supported
range, exact frozen resolution, consumed contract, and intentional test gates.

## 1. Purpose

Most integrations are optional extras. A mocked or skipped test does not prove
that an external boundary still matches its upstream contract, so changes to an
extra, CLI command, or published format must update this page.

Exact versions below come from `uv.lock` as checked on **2026-07-17**. Supported
ranges remain defined by `pyproject.toml`.

## 2. Contract Ledger

| Integration | Supported / frozen | NNx contract relied on | Verification |
| --- | --- | --- | --- |
| PyTorch training core | `torch>=2.0` / `2.13.0`; `torchvision>=0.15` / `0.28.0`; `torch-geometric>=2.4` / `2.8.0` | `nn.Module`, autograd, optimizer, PyG loader, and torchvision dataset APIs | Full frozen all-extras pytest matrix; graph, dataset, and network tests exercise public paths. |
| ONNX export | `onnx>=1.15` / `1.22.0`; `onnxscript>=0.1` / `0.7.1` | Legacy `torch.onnx.export`; optional `dynamo=True` only when supported | `tests/test_to_onnx_inputs.py`, `tests/test_onnx_dynamo.py`, and `tests/test_viz_netron.py`; known exporter dispatch skew uses the documented guard in `tests/conftest.py`. |
| Netron viewer | `netron>=7.0` / `9.1.3` | `netron.start(path)` only when `launch=True`; ONNX export remains independent | `tests/test_viz_netron.py` covers launch dispatch and missing-package errors. |
| Hugging Face Hub | `huggingface-hub>=1.4.0` / `1.21.0`; `safetensors>=0.7.0` / `0.8.0` | `PyTorchModelHubMixin` and safetensors checkpoint APIs | `tests/test_hub_mixin.py` and `tests/test_checkpoint_safetensors.py`; authenticated pushes are intentionally credential-gated. |
| Experimental GGUF | `gguf>=0.19.0` / `0.19.0` | GGUF container metadata and NNx tensor mapping under `nnx_transformer` | `tests/test_interop_gguf_writer.py` parses writer output. Stock llama.cpp, Ollama, and LM Studio runtime compatibility is explicitly unsupported; Modelfile tests verify text generation only. |
| Quantization | `torchao>=0.17` / `0.17.0` | PTQ INT8 and QAT 8da4w quantizer APIs | `tests/test_quantize_ptq.py` and `tests/test_quantize_qat.py`; CUDA-only 2:4 behavior is hardware-gated. |
| Embeddings / FAISS | `faiss-cpu>=1.7` / `1.14.3`; `sentence-transformers>=2.7` / `5.6.0` | FAISS index/search and SentenceTransformer-like `forward(list[str]) -> Tensor[B, D]` | Embedding contrastive and FAISS export tests; downstream RAG adapters are out of scope. |
| LM data / tokenization | `tokenizers>=0.20` / `0.22.2`; `datasets>=2.20` / `5.0.0` | BPE train/encode/decode and optional remote dataset loading | Tokenizer and generative-model tests; network-backed dataset downloads are not required in core CI. |
| Experiment logging | `tensorboard>=2.15` / `2.21.0`; `wandb>=0.16` / `0.28.0` | Writer/run lifecycle and finish semantics | Callback tests cover lifecycle and TensorBoard event output; real W&B service calls remain credential/network-gated. |
| Package publishing | `setuptools==83.0.0`; `build==1.5.0`; `twine==6.2.0`; PyPI OIDC | Release version/tag agreement, artifact metadata, trusted publish, clean install | Reusable release workflow runs tests, builds, runs `twine check`, publishes, and verifies a fresh install. |
| Maintenance tooling | `uv==0.11.29`; `pip-audit==2.10.1`; Pyright `1.1.411`; Ruff `0.9.10` | Frozen resolution, exact-graph security audit, type and style gates | CI uses `uv sync --frozen --all-extras`; security exports that lock before auditing; Pyright warnings are gating. |

## 3. Review Rules

1. Update this ledger with any dependency range, optional extra, external CLI,
   or published configuration change.
2. Prefer tests against the frozen public API. Use a mock only for credentials,
   hardware, daemons, or network boundaries, and record the gate here.
3. Document tolerated upstream skew together with the condition for removing its
   compatibility guard or skip.
