# 13. External Dependency Contracts

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
| PyTorch training core | `torch>=2.4` / `2.13.0` (torchvision and torch-geometric are the `vision` / `graph` extras since FEAT-031: see the domain-extras row) | `nn.Module`, autograd, optimizer, `torch.amp.GradScaler(device)` + `torch.autocast(device_type, dtype)` for the FP16 (CUDA) and BF16 (CPU, CUDA) precision policy, `torch.is_autocast_enabled(device_type)`, SDPA with float masks, `torch.load(weights_only=True)` for training-state sidecars | Full frozen all-extras pytest matrix (current lane) plus the CI `floor-deps` lane on the declared minimum pair; see §2.1 for the tested matrix and what remains unverified. |
| Domain extras: vision, graph, plots (FEAT-031) | `torchvision>=0.19` / `0.28.0` (extra `vision`); `torch_geometric>=2.4` / `2.8.0.post1` (extra `graph`); `plotly>=5.18` / frozen in `uv.lock` (extra `plots`, also pulled by `viz`); `domains` = all three | `import nnx` imports none of them; features import them on use through `nnx._optional.require`, which turns only the package's own absence into an `ImportError` naming the extra (a broken dependency of an installed package keeps its own error). Flat names (`NNDataset`, `NNGraphDataset`, the graph nets) resolve lazily to the same classes; `nnx.__all__` lists them only when their extra is installed | `tests/test_core_install.py` (a core-only interpreter: train / predict / reload, no domain module loaded, each domain request naming its extra, `lr_finder` refusing before touching data / model / RNG, the `NNRun` repr without the chart, cause kept for a broken dependency); CI `installed-profiles` and release `artifact-profiles` install the built wheel per profile (core, each extra, `domains`) over the floor and current torch pairs and run `scripts/smoke_core_install.py`, recording the installed closure and three cold-import timings (no threshold). |
| ONNX export | `onnx>=1.15` / `1.22.0`; `onnxscript>=0.1` / `0.7.1` | Legacy `torch.onnx.export`; optional `dynamo=True` only when supported | `tests/test_to_onnx_inputs.py`, `tests/test_onnx_dynamo.py`, and `tests/test_viz_netron.py`; known exporter dispatch skew uses the documented guard in `tests/conftest.py`. Structural (checker) evidence only, except the profiles in the next row. |
| ONNX Runtime execution ([Export conformance](export-conformance.md)) | `onnxruntime>=1.18` / `1.30.0` (`1.23.2` on CPython 3.10, `<1.24`; 1.18 is the first to load IR-10 models), extra `onnx-runtime` | `InferenceSession(path, providers=["CPUExecutionProvider"])` and `run` on FP32 inputs; invalid input dimensions raise | `tests/test_export_conformance.py` and the required CI `export-conformance` job: `FeedFwdNN` 4-8-2 under both exporters, every raw logit at `rtol=1e-4`, `atol=1e-5` for batches 1/2/3/7 of a batch-2 dynamic export, a wrong width refused, artifact SHA-256s verified before loading; the job cannot pass by skipping and keeps the JSON plus ONNX / external-data files. Executed parity is claimed for these two profiles only; quantized, Netron and GGUF / Ollama targets stay structural or container-only. |
| Netron viewer | `netron>=7.0` / `9.1.8` | `netron.start(path)` only when `launch=True`; ONNX export remains independent | `tests/test_viz_netron.py` covers launch dispatch and missing-package errors. |
| Hugging Face Hub | `huggingface-hub>=1.4.0` / `1.24.0`; `safetensors>=0.7.0` / `0.8.0` | `PyTorchModelHubMixin` and safetensors checkpoint APIs | `tests/test_hub_mixin.py` and `tests/test_checkpoint_safetensors.py`; authenticated pushes are intentionally credential-gated. |
| Experimental GGUF / Ollama bundle | `gguf>=0.19.0` / `0.19.0`; audited source snapshots: Ollama tag `v0.32.2`, llama.cpp build `8660` | GGUF container metadata and NNx tensor mapping under `nnx_transformer`; the tagged Ollama source's typed Modelfile parameters and bundle layout | `tests/test_interop_gguf_writer.py` parses writer output and verifies bundle structure, rendering, all eleven documented parameter types, templates, and boundary validation. Stock llama.cpp, Ollama, and LM Studio execution remains explicitly unsupported because those runtimes do not implement `nnx_transformer`. |
| Quantization | `torchao>=0.17` / `0.17.0` | PTQ INT8 and QAT 8da4w quantizer APIs | `tests/test_quantize_ptq.py` and `tests/test_quantize_qat.py`; CUDA-only 2:4 behavior is hardware-gated. |
| Embeddings / FAISS | `faiss-cpu>=1.7` / `1.14.3`; `sentence-transformers>=2.7` / `5.6.0` | FAISS index/search and SentenceTransformer-like `forward(list[str]) -> Tensor[B, D]` | Embedding contrastive and FAISS export tests; downstream RAG adapters are out of scope. |
| LM data / tokenization | `tokenizers>=0.20` / `0.22.2`; `datasets>=2.20` / `5.0.0` | BPE train/encode/decode and optional remote dataset loading | Tokenizer and generative-model tests; network-backed dataset downloads are not required in core CI. |
| Jev models (TypeSafe API) | `typesafe-sdk>=0.7,<0.8` / `0.7.2` (pulls `httpx2`, `tenacity`), extra `jev` | `TypeSafeClient` / `AsyncTypeSafeClient(api_key=, base_url=, timeout=, retry=, transport=)`, `system_one(state, questions, model=)` with `noul` / `choice` / `score` question dictionaries, `SystemOneResponse` (`model`, `usage`, `answers`, `request_id`), the SDK `RetryPolicy` as the only retry owner, and its `TypeSafeAPITimeoutError` / `TypeSafeAuthenticationError` / `TypeSafePermissionDeniedError` / `TypeSafeRateLimitError` / `TypeSafeAPIResponseValidationError` hierarchy | `tests/test_decision_jev.py` (a recording fake of `system_one` returning real SDK responses: translation, alignment, metadata, confidence, typed failures with one attempt, client ownership and closing, failures through a decision job and the benchmark), `tests/test_imports.py` (no SDK imported by `import nnx`; the missing extra named) and `examples/decision_jev.py` (real clients over an offline mock transport, sync and async). Live service behaviour is credential-gated: `scripts/smoke_jev_live.py` pins `jev-1.13.0` and is run by hand, never by CI. |
| Experiment logging | `tensorboard>=2.15` / `2.21.0`; `wandb>=0.16` / `0.28.1` | Writer/run lifecycle and finish semantics | Callback tests cover lifecycle and TensorBoard event output; real W&B service calls remain credential/network-gated. |
| Registered optimizer factories (`nnx.optimizers`) | User-supplied callables registered in-process; no package, entry point or import-by-name | `factory(param_groups, config) -> torch.optim.Optimizer` called once per optimizer with resolved groups (`params`, `lr`, `weight_decay`) and a read-only JSON-like config; the result must own exactly those parameters. `run.yaml` records only `OptimizerFactorySpec(id, version, config)`; resume requires the same identity and parameter topology. | `tests/test_optimizer_factories.py` (registry, one-call contract, ownership checks, failure before the run with no imports, resume validation, offline reload) and `examples/optimizer_factories.py`. Factory behaviour itself is the registrant's contract: NNx validates the returned optimizer's parameter set, not its update rule. |
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
| `2.4.1` / `0.19.1` (declared floor) | 3.10 | CPU, core dependencies plus the `domains` extra (no other extras): CI `floor-deps` lane on Ubuntu (CPU wheels from `download.pytorch.org/whl/cpu`), and a local macOS arm64 run on 2026-09-22 | Whole core suite: 1714 passed, 70 skipped (extra-gated `importorskip`), docs-projection modules and the Cora download test excluded. | **Tested — minimum.** |
| `2.5.1` / `0.20.1` | 3.10 | Local macOS arm64 CPU run on 2026-09-22, the then-core dependencies (graph, vision and plotting stacks included — today's core plus `domains`) | Whole core suite, same exclusions as the floor row: 1714 passed, 70 skipped. | **Tested — intermediate** (not a CI lane). |
| `2.3.1` / `0.18.1` (with `numpy<2`) | 3.10 | Local macOS arm64 CPU run on 2026-09-22 | 23 failures in library code: `torch.is_autocast_enabled(device_type)` raises `TypeError` in the `TransformerNN` forward (13 tests), SDPA rejects a float `attn_mask` whose dtype differs from a half / bf16 / double query (9), and `torch.load(weights_only=True)` cannot unpickle the training-state sidecar (1). `torch.amp.GradScaler("cuda")` itself exists from 2.3. | **Unsupported** — excluded by metadata. |
| `< 2.3` | — | not run | `torch.amp.GradScaler` does not exist; the CUDA AMP factory would raise `AttributeError`. | **Unsupported** — excluded by metadata. |
| CPU BF16 (`PrecisionPolicy("bf16")` on a CPU device) on the CI rows | 3.10 – 3.14 | the CI `lint-and-test` and `floor-deps` lanes (CPU) | `tests/test_precision_execution.py`: seeded classification and KD fixtures with accumulation stay within `nnx.precision.REFERENCE_TOLERANCES["bf16"]` of their FP32 reference; bf16 autocast without a scaler, the backward outside autocast, float32 parameters, and the finite-gradient check before clipping. | **Tested — CPU** (`precision_support()` reports it `"verified"`). |
| CUDA FP16 / BF16 (`PrecisionPolicy("fp16" / "bf16")`, or `mixed_precision=True`, on a CUDA device) on any row | — | no CUDA hardware in the local or CI environments used for this matrix | Resolution is covered by stubs (`tests/test_precision_policy.py`, `test_grad_scaler_prefers_modern_factory`, `test_grad_scaler_disabled_on_cpu_or_without_mixed_precision`); the FP16 update order (unscale → normalize → clip → step through the scaler) and the scaler's checkpoint round trip are simulated on CPU; `test_cuda_cells_are_verified_only_on_cuda_hardware` runs the reference fixtures only on CUDA hardware and otherwise asserts `precision_support()` reports the cells `"unverified"`; `examples/02_resume_training.py::amp_resume_compatibility` runs its enabled branch only when `torch.cuda.is_available()`. | **Unverified** — control-flow coverage, not hardware execution. |

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
