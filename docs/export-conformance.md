# 14. Export conformance

`NNModel.to_onnx` writes a graph; `onnx.checker` proves the file is well
formed. Neither says what a runtime computes from it. `nnx.export_conformance`
separates the two claims: a **profile** fixes everything a parity claim
depends on, executes the export in ONNX Runtime, and records every stage's
outcome on its own.

```text
dependencies ─► export ─► check ─► load ─► parity
  onnx,          to_onnx    onnx.checker   hashes verified,   every raw logit vs the native
  onnxruntime    (state,    (full_check),  then an ORT         model in eval mode, within
  [onnxscript]   grads and  SHA-256 of     InferenceSession    rtol / atol; a wrong feature
                 modes kept) every file    on the provider     width must be refused
```

## 1. Tested profiles

These are the only targets NNx claims **executed parity** for:

| Profile | Exporter | Model | Runtime | Cases | Tolerances |
|---|---|---|---|---|---|
| `feedfwd-fp32-torchscript` | legacy `torch.onnx.export` (`dynamo=False`), opset 17 | `FeedFwdNN` 4-8-2, ReLU, dropout 0, deterministic nontrivial FP32 weights (local generator, seed 0, `normal(0, 0.5)`) | ONNX Runtime, `CPUExecutionProvider`, one thread | exported at batch 2 with `dynamic_batch=True`; batches 1, 2, 3 and 7 must match; width 5 must be refused | `rtol=1e-4`, `atol=1e-5` |
| `feedfwd-fp32-dynamo` | `torch.export`-based (`dynamo=True`), opset 17 | same | same | same | same |

Every other export stays at the level its evidence supports:

| Target | Evidence | Level |
|---|---|---|
| Any other `to_onnx` export (other architectures, registered modules) | `onnx.checker`; `examples/04_onnx_export.py` labels a run `executed` only when ONNX Runtime ran it | structural (checker-only) unless you execute it |
| Quantized INT8 export (`examples/12_quantize_int8.py`, Phase 5) | the file is written | structural — no parity claimed |
| Netron artifact (`nnx.viz.netron_export`) | the file is written for a viewer | structural |
| GGUF / Ollama ([Experimental GGUF export](gguf.md)) | NNx's own parser round trip; stock llama.cpp, Ollama and LM Studio do not implement `nnx_transformer` | container-only |
| FAISS index export ([Embeddings](embeddings.md)) | reloaded and searched by FAISS in `tests/test_embeddings_faiss_export.py` | executed by FAISS (not ONNX) |

A checker pass or a parser round trip never promotes a target to executed
parity. GGUF engines, format conversion, deployment benchmarks and any claim
beyond the two FP32 feed-forward profiles are out of scope.

## 2. Running the profiles

```bash
pip install "thekaveh-nnx[onnx-runtime,onnx-dynamo]"
python scripts/check_export_conformance.py --output conformance.json
#   --artifacts DIR        exported files (default: conformance-artifacts/ next to the JSON)
#   --profile NAME         one profile (repeatable)
#   --source-revision SHA  default: the git checkout NNx is imported from
#   --force                clear the profiles' earlier artifacts (else a non-empty folder is refused)
```

Or from Python:

```python
from nnx.export_conformance import run_profile, execute, verify_artifacts

record = run_profile("feedfwd-fp32-dynamo", "artifacts/")   # a JSON-ready dict
assert record["level"] == "executed"
verify_artifacts(record, "artifacts/")   # refuses a changed, missing or extra file
again = execute(record, "artifacts/")    # re-runs load + parity; hashes checked first
```

Nothing is downloaded. `import nnx` and `import nnx.export_conformance` need
neither `onnx` nor `onnxruntime`; a profile run without them is a
`missing-dependency` **failure**, never a skip.

## 3. Outcomes and failure classes

Each stage is `passed`, `failed` or `not_run`. A stage after a failed one
does not run. The record's `failure` is the first failing stage's class.
Its `level` is `executed` when parity passed, `structural` when only the
checker did, and `none` otherwise. The script's exit status separates the
failure classes; `1` is an error and `2` a usage error (an artifact folder
that already holds a profile's files is refused unless `--force` clears it):

| Exit | Failure | Stage | Meaning |
|---|---|---|---|
| 0 | — | — | every profile passed |
| 10 | `missing-dependency` | dependencies | `onnx`, `onnxruntime` (or `onnxscript` for dynamo) not importable |
| 11 | `export-error` | export | the exporter raised (a dynamo dispatch error included — it is never skipped) |
| 12 | `state-changed` | export | the model's parameters, gradients or mixed train / eval modes changed |
| 13 | `checker-error` | check | `onnx.checker.check_model(path, full_check=True)` refused the file |
| 14 | `hash-mismatch` | load | an artifact's SHA-256 or size differs from the record, or a file is missing or unlisted; nothing is loaded |
| 15 | `load-error` | load | ONNX Runtime could not create a session on the declared provider |
| 16 | `mismatch` | parity | a raw logit is outside `atol + rtol·|native|` or not finite, or the output count, dtype or shape differs |
| 17 | `input-contract` | parity | a valid batch was refused as an invalid argument (e.g. a static batch dimension), a wrong feature width was accepted, or no valid case was compared |
| 18 | `runtime-error` | parity | the runtime failed on a case for any other reason |

## 4. The record

`nnx.export-conformance/1`. Each profile's record holds:

- `source` — `revision`, where it came from (`given`, `git` when this
  checkout is the one imported, or `unknown` for an installed copy), whether
  the tree was `dirty`, and the NNx version.
- `config` — the model (net, dims, activation, dropout, weight generator
  and seed) and `weights_sha256`.
- `exporter` — the exporter and its options (opset, dynamic batch, export
  batch, input / output names); the warnings it emitted are in
  `stages.export.detail`.
- `opset` — the requested opset and the model's actual `opset_import`.
- `artifacts` — the profile's directory and every file the exporter wrote,
  each with its role (`model`, `external-data`, or `sidecar` for a file no
  tensor references), size and SHA-256. External tensors are included.
- `versions` — Python, platform, NNx, torch, numpy, onnx, onnxruntime and
  onnxscript.
- `runtime` — `onnxruntime`, its version, the provider and the session
  options.
- `dtype`, `input_cases` (name, shape, seed, expectation, outcome, max
  absolute and relative error, failure), `tolerances`.
- `model_state` — whether parameters, gradients and modes survived the
  export.
- `stages` — every stage's `status`, `failure` and `detail`.

Every record states a profile this module runs, exactly as `run_profile` writes it: its settings are rebuilt as a `Profile` (FP32 inputs in ONNX Runtime, the recorded session options, a feed-forward model of at most 65,536 units per layer) and compared as canonical text — with the tested profile of that name, whose name stands for its exact settings, or with the rebuilt profile itself, so input cases, exporter options or an artifact layout no profile produces are refused. A `Profile` reusing a tested name with other settings is refused before anything is written. A symlinked artifact or artifact folder never verifies. `validate_record` refuses missing or unknown keys, unknown statuses, a
failure class in the wrong stage, a stage that ran after a failure,
contradictory `status` / `failure` / `level`, malformed hashes and
non-finite tolerances, naming each problem.

## 5. CI evidence

The required `export-conformance` job in `.github/workflows/ci.yml` runs
in four steps:

1. It imports `nnx` in a core-only environment and checks that `onnx`,
   `onnxruntime` and `onnxscript` are absent.
2. It requires the script to fail there with exit code 10.
3. It installs only `onnx-runtime` and `onnx-dynamo` and runs both
   profiles (so a broken extra fails), then adds `dev` and runs the
   conformance tests with `NNX_REQUIRE_ONNX_RUNTIME=1`. A
   missing runtime fails those tests rather than skipping them, and the job
   asserts that the JUnit report has zero skips.
4. It uploads `conformance/` — the JSON report, every ONNX and
   external-data file, and the test report — as the `export-conformance`
   artifact.

`uv.lock` resolves with `--all-extras`. The CPU profiles are not run on the
`floor-deps` lane (torch 2.4.1, core only).
