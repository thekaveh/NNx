# 12. External Dependency Contracts

This ledger records NNx integration points that depend on third-party packages,
services, CLIs, or registries. It complements tests by naming the supported
range, exact frozen resolution, consumed contract, and intentional test gates.

## 1. Purpose

Most integrations are optional extras. A mocked or skipped test does not prove
that an external boundary still matches its upstream contract, so changes to an
extra, CLI command, or published format must update this page.

Exact package versions below come from `uv.lock` as checked on **2026-08-08**.
CLI versions are labeled as an audited local snapshot. Supported package ranges
remain defined by `pyproject.toml`.

## 2. Contract Ledger

| Integration | Supported / frozen | NNx contract relied on | Verification |
| --- | --- | --- | --- |
| PyTorch training core | `torch>=2.4` / `2.13.0`; `torchvision>=0.19` / `0.28.0`; `torch-geometric>=2.4` / `2.8.0.post1` | `nn.Module`, autograd, optimizer, `torch.amp.GradScaler(device)` + `torch.amp.autocast` for CUDA mixed precision, `torch.is_autocast_enabled(device_type)`, SDPA with float masks, `torch.load(weights_only=True)` for training-state sidecars, PyG loader, and torchvision dataset APIs | Full frozen all-extras pytest matrix (current lane) plus the CI `floor-deps` lane on the declared minimum pair; see §2.1 for the tested matrix and what remains unverified. |
| ONNX export | `onnx>=1.15` / `1.22.0`; `onnxscript>=0.1` / `0.7.1` | Legacy `torch.onnx.export`; optional `dynamo=True` only when supported | `tests/test_to_onnx_inputs.py`, `tests/test_onnx_dynamo.py`, and `tests/test_viz_netron.py`; known exporter dispatch skew uses the documented guard in `tests/conftest.py`. |
| Netron viewer | `netron>=7.0` / `9.1.8` | `netron.start(path)` only when `launch=True`; ONNX export remains independent | `tests/test_viz_netron.py` covers launch dispatch and missing-package errors. |
| Hugging Face Hub | `huggingface-hub>=1.4.0` / `1.24.0`; `safetensors>=0.7.0` / `0.8.0` | `PyTorchModelHubMixin` and safetensors checkpoint APIs | `tests/test_hub_mixin.py` and `tests/test_checkpoint_safetensors.py`; authenticated pushes are intentionally credential-gated. |
| Experimental GGUF / Ollama bundle | `gguf>=0.19.0` / `0.19.0`; audited source snapshots: Ollama tag `v0.32.2`, llama.cpp build `8660` | GGUF container metadata and NNx tensor mapping under `nnx_transformer`; the tagged Ollama source's typed Modelfile parameters and bundle layout | `tests/test_interop_gguf_writer.py` parses writer output and verifies bundle structure, rendering, all eleven documented parameter types, templates, and boundary validation. Stock llama.cpp, Ollama, and LM Studio execution remains explicitly unsupported because those runtimes do not implement `nnx_transformer`. |
| Quantization | `torchao>=0.17` / `0.17.0` | PTQ INT8 and QAT 8da4w quantizer APIs | `tests/test_quantize_ptq.py` and `tests/test_quantize_qat.py`; CUDA-only 2:4 behavior is hardware-gated. |
| Embeddings / FAISS | `faiss-cpu>=1.7` / `1.14.3`; `sentence-transformers>=2.7` / `5.6.0` | FAISS index/search and SentenceTransformer-like `forward(list[str]) -> Tensor[B, D]` | Embedding contrastive and FAISS export tests; downstream RAG adapters are out of scope. |
| LM data / tokenization | `tokenizers>=0.20` / `0.22.2`; `datasets>=2.20` / `5.0.0` | BPE train/encode/decode and optional remote dataset loading | Tokenizer and generative-model tests; network-backed dataset downloads are not required in core CI. |
| Experiment logging | `tensorboard>=2.15` / `2.21.0`; `wandb>=0.16` / `0.28.1` | Writer/run lifecycle and finish semantics | Callback tests cover lifecycle and TensorBoard event output; real W&B service calls remain credential/network-gated. |
| Maintenance tooling | `uv==0.12.3`; `pip-audit==2.10.1`; Pyright `1.1.411`; Ruff `0.16.2` | Frozen resolution, exact-graph security audit, type and style gates | Automation installs the same uv version declared in `requirements-tools.txt`; CI uses `uv sync --frozen --all-extras`; security exports that lock before auditing; Pyright warnings are gating. |
| Package publishing | `setuptools==84.0.0`; `uv==0.12.3`; `twine==7.0.0`; PyPI OIDC | Release version/tag agreement, reproducible artifact bytes, exact registry hashes, trusted publish, immutable GitHub release | The top-level dispatch-only release workflow builds once for publication, verifies local/PyPI filename and SHA-256 sets, attaches the same artifacts to the GitHub release, verifies API digests and immutable attestations, then installs from PyPI. |

### 2.1. PyTorch support matrix

The floor in `pyproject.toml` is the oldest torch / torchvision pair on which
the **full core test suite** passes, not merely the oldest release that
imports (FIX-011). Every row states how it was exercised; a row without
hardware evidence says so.

| torch / torchvision | Python | Environment | Evidence | Status |
| --- | --- | --- | --- | --- |
| `2.13.0` / `0.28.0` (frozen `uv.lock`) | 3.10 – 3.14 | GitHub-hosted Ubuntu, CPU, all extras (`uv sync --frozen --all-extras`) | CI `lint-and-test` matrix: the whole suite including extras, ruff, pyright, docs build. | **Tested — current.** |
| `2.4.1` / `0.19.1` (declared floor) | 3.10 | CPU, core dependencies only (no extras): CI `floor-deps` lane on Ubuntu (CPU wheels from `download.pytorch.org/whl/cpu`), and a local macOS arm64 run on 2026-09-22 | Whole core suite: 1714 passed, 70 skipped (extra-gated `importorskip`), docs-projection modules and the Cora download test excluded. | **Tested — minimum.** |
| `2.5.1` / `0.20.1` | 3.10 | Local macOS arm64 CPU run on 2026-09-22, core dependencies only | Whole core suite, same exclusions as the floor row: 1714 passed, 70 skipped. | **Tested — intermediate** (not a CI lane). |
| `2.3.1` / `0.18.1` (with `numpy<2`) | 3.10 | Local macOS arm64 CPU run on 2026-09-22 | 23 failures in library code: `torch.is_autocast_enabled(device_type)` raises `TypeError` in the `TransformerNN` forward (13 tests), SDPA rejects a float `attn_mask` whose dtype differs from a half / bf16 / double query (9), and `torch.load(weights_only=True)` cannot unpickle the training-state sidecar (1). `torch.amp.GradScaler("cuda")` itself exists from 2.3. | **Unsupported** — excluded by metadata. |
| `< 2.3` | — | not run | `torch.amp.GradScaler` does not exist; the CUDA AMP factory would raise `AttributeError`. | **Unsupported** — excluded by metadata. |
| CUDA mixed precision (`mixed_precision=True` on a CUDA device) on any row | — | no CUDA hardware in the local or CI environments used for this matrix | Constructor dispatch is covered by stubs only (`test_grad_scaler_prefers_modern_factory`, `test_grad_scaler_disabled_on_cpu_or_without_mixed_precision`); resume presence validation and unscale-before-clip ordering are covered on CPU with a disabled scaler; `examples/02_resume_training.py::amp_resume_compatibility` runs its enabled branch only when `torch.cuda.is_available()` and reports it as skipped otherwise. | **Unverified** — control-flow coverage, not hardware execution. |

Optional extras carry their own constraints and are *not* part of the floor
lane: `torchao` (`[quantize]`) is validated only against the frozen torch
above, `pyg-lib` / `torch-sparse` (neighbor-sampler iteration) are never
installed, and CUDA-only quantization behaviour is hardware-gated.

NNx uses a release-please-managed static package version. Wheels and sdists from
untagged commits are local test artifacts only and must not be distributed,
because post-release source changes retain the preceding release number until
the next release PR. Release Please dispatches the top-level `release.yml`
workflow, which is the sole distribution path; direct tag pushes do not
publish. Keeping trusted publishing in that workflow preserves the PyPI OIDC
identity while it revalidates the tag SHA, checks tag/version agreement, and
verifies exact artifacts on PyPI and GitHub.
Repository release immutability and protected `v*` tags apply to releases
created after the 2026-07-22 hardening. The historical `v0.2.1` GitHub release
predates that control, remains mutable, and has no attached distribution
attestations. It is retained as published history rather than destructively
recreated; PyPI remains the artifact source for that version.

## 3. Review Rules

1. Update this ledger with any dependency range, optional extra, external CLI,
   or published configuration change.
2. Prefer tests against the frozen public API. Use a mock only for credentials,
   hardware, daemons, or network boundaries, and record the gate here.
3. Document tolerated upstream skew together with the condition for removing its
   compatibility guard or skip.
