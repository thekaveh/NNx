# NNx `develop` review: 2026-10-27 → 2026-11-27

> **Update 2026-11-28: implement from `review-tickets/`, not from this file.** These 161 findings were cross-checked against all 196 board issues and rewritten into 142 tickets in the board's format (13 DEC, 46 FIX, 83 FEAT). In that pass:
> - a few findings were narrowed or reframed because a past issue had decided the behaviour on purpose (for example #201 DoRA, #551 bundle depth, #552 fsync, #572 import-only examples, #219 decisions package);
> - one finding, DEAD-13, was dropped, and one (DEAD-10) was resolved by another ticket;
> - choices were moved into DEC tickets, and oversized items were split.
> See `review-tickets/README.md` for the index and the work order.


**Range reviewed:** `f486c85..9ad461e`, the tip of `origin/develop` on 2026-11-27. That is 594 commits and about 29k changed lines in 270 files.
**Scope:** correctness bugs; documentation errors; code, tests, examples and config that are out of sync with each other; out-of-scope code and files; dead code and clutter; code smells, duplication and bloat that are ripe for refactoring.
**Findings:** 161. Each one was checked by reading the code at `9ad461e`; see §0.3 for the limits of that check.

---

## 0. Read this first: instructions for the agent implementing these fixes

### 0.1. What this document is

- Each finding is one self-contained item with a stable **ID**, such as `BUG-03`. Work on **one item at a time**, or on one *group* of tightly linked items (§2).
- Every item has the same fields:

  | Field | Meaning |
  |---|---|
  | **Severity** | **High**: wrong results, data loss, a security risk, or a broken guarantee. **Medium**: a real problem with a workaround, or a misleading doc. **Low**: cosmetic, a small inconsistency, or cleanup. |
  | **Category / Group** | The kind of issue, and which fix group (§2) it belongs to. |
  | **Where** | Files and approximate lines. **Lines are as of commit `9ad461e` and are prefixed `~`. They will drift.** Find the code by the function or class name given, not by the line number alone. |
  | **Problem / Evidence** | What is wrong, with the code quoted. |
  | **Fix** | Numbered steps. Do them in order. |
  | **Tests / Check** | What to add, and the command that must pass afterwards. |
  | **Risk / Note** | What could break, and what to watch for. |

- "Fixed by X" means: do X, then confirm this item is resolved too. Do not implement it separately.
- Some items offer **options (a)/(b)**. Pick the first option unless the item says otherwise, and say which one you picked in the commit message.

### 0.2. Ground rules

1. **Branch from `develop`, not `main`.** The code reviewed here is on `develop`. Run `git fetch origin develop && git checkout -b fix/<item-id> origin/develop`.
2. **Read the code at the cited function before you edit it.** If what you see does not match the item's quoted code, stop and report the mismatch. Do not guess.
3. **Before changing any error or warning message,** grep the tests for it (`grep -rn "<fragment>" tests`). Many tests assert message text.
4. **When moving or renaming a private helper,** keep the old name importable as an alias (`_old_name = new_name`). Tests monkeypatch module attributes by name, and pickles reference class paths.
5. **Every behaviour change needs a CHANGELOG line** under `## [Unreleased]` in `CHANGELOG.md`, in `### Fixed`, `### Changed` or `### Added`. Match the existing bullet style: `- **Bold one-line summary.** Explanation.`
6. **Never bump the version.** `pyproject.toml` and `src/nnx/__init__.py` are managed by release-please.
7. **Generated files must be regenerated, not hand-edited:**
   - `docs/api.md` is generated from docstrings plus `scripts/docs/api_reference_catalog.txt`. Run `uv run python -m scripts.docs.build_api_reference`.
   - `PYPI_README.md` is generated from `README.md`. Run `uv run python -m scripts.docs.build_pypi_readme`.
   - `mkdocs.yml` is generated (and git-ignored) from `docs/manifest.yaml` by `scripts/docs/build_docs.py`.
   - `docs/feature-composition.md` is generated from `docs/feature-composition.yaml`. Run `uv run python scripts/check_feature_composition.py --render`.
8. **Doc headings are numbered** (`## 1.`, `### 1.2.`) on every page except `CHANGELOG.md`. Keep the numbering sequential when you add a section.
9. **Items marked "maintainer decision"** (SCOPE-01, SCOPE-03, SCOPE-04, CLUT-01, SMELL-31) must not be implemented without explicit approval. Raise them, don't do them.
10. **Do not delete or skip a test to make the suite pass.** If an existing test enshrines a bug, the item says which test to rewrite and how.

### 0.3. How these findings were verified

- Five parallel reviewers read the code, git history, tests and docs. One of them re-checked the 17 findings of the earlier `/code-review` run; the other four covered the docs, the examples/scripts/CI/tests, the `nn/` core, and the feature packages. Static tools were used: vulture, radon, ruff, AST scripts, the docs build, an sdist build, `uv lock --check`, and the GitHub Actions run history.
- **Nothing that needs torch was run**, because torch could not be installed in the review environment. Findings about torch's exporters (BUG-03, BUG-04, BUG-06) rest on reading torch, onnxscript and onnx_ir sources, not on a reproduction. **For those items, write the test first and confirm it fails before you fix.**
- Every other finding cites the code it contradicts.

### 0.4. Commands

```bash
uv sync --all-extras --frozen                         # install (CONTRIBUTING §1)
uv run pytest -n logical --dist loadgroup             # full suite
uv run pytest tests/test_x.py::test_name              # one test
uv run pytest -k "keyword"                            # by name
uv run ruff check src/ tests/ examples/ scripts/      # lint
uv run ruff format --check src/ tests/ examples/ scripts/
uv run pyright --warnings                             # type check (gating in CI)
uv run python -m scripts.docs.build_api_reference --check
uv run python -m scripts.docs.build_pypi_readme --check
uv run python -m scripts.docs.build_docs --check
uv run python scripts/check_feature_composition.py --check
```

Before each commit, run ruff check, ruff format, pyright, the tests named in the item, and the docs `--check` commands if you touched docs or docstrings.

---

## 1. Summary

| Part | Prefix | Count | High | Medium | Low |
|---|---|---:|---:|---:|---:|
| A. Correctness bugs | `BUG` | 28 | 7 | 12 | 9 |
| B. CI, security and workflow | `CI` | 7 | 1 | 2 | 4 |
| C. Documentation | `DOC` | 35 | 1 | 6 | 28 |
| D. Out-of-sync examples, tests, config | `SYNC` | 7 | 0 | 4 | 3 |
| E. Out-of-scope code and files | `SCOPE` | 4 | 1 | 2 | 1 |
| F. Dead code and clutter | `DEAD`, `CLUT` | 19 | 0 | 0 | 19 |
| G. Smells, duplication, bloat | `SMELL` | 61 | 2 | 18 | 41 |
| **Total** | | **161** | **12** | **44** | **105** |

**The 12 High items:**

| ID | One line |
|---|---|
| CI-01 | Dependabot PRs run new dependency code on the maintainer's self-hosted Mac (live now). |
| BUG-01 | One sample input that cannot be fingerprinted makes `evaluate()` fail with no report. |
| BUG-02 | Every benchmark report saved by ≤ 0.5.0 now "scored different samples" than a new one. |
| BUG-03 | `to_onnx(dynamo=True)` writes weights under a hidden random file name and leaks it on every export. |
| BUG-04 | Every Conv+BatchNorm ONNX export warns about "dropped" tensors; under `-W error` the export is discarded. |
| BUG-07 | One legacy run directory ending in `.` makes `NNRun.all()` and every `run="best"` read fail. |
| BUG-08 | `save_pretrained` can delete the previous `tokenizer.json` (or a user's own) and leave an unloadable artifact. |
| BUG-10 | On Windows, the atomic writers never flush to disk before renaming. |
| DOC-01 | DPO now refuses empty or all-pad responses mid-epoch, and nothing documents it. |
| SCOPE-01 | `nnx.decisions` (≈4.9k lines of LLM evaluation and orchestration) is a separate product inside a training library. |
| SMELL-01 | Atomic-write and fsync logic is copied about 10 times; this is what produced BUG-10 and BUG-11. |
| SMELL-02 | `nn_model.py` is a 6,826-line god module. |

**Root causes worth knowing:**
- Most regressions come from two large commits late in the range: **3ed0133 (#853)** and **ed93d51 (#855)**. Their behaviour changes did not reach the guides, docstrings or CHANGELOG.
- Durability helpers grew by copy-paste instead of through one shared module (SMELL-01).
- New refusals were added on *read* paths when they only belonged on *write* paths (BUG-07), or without a compatibility marker (BUG-02).

---

## 2. Work plan: fix groups and order

Work top to bottom. Items inside a group touch the same code, so do them in one branch, in the order listed.

| Order | Group | Items (in order) | Why this order |
|---:|---|---|---|
| 1 | **G7 (security first)** | CI-01 | Live exposure; a small YAML change. |
| 2 | **G6 Deprecation sweep** | SYNC-01 → DOC-05 → SYNC-02 → SYNC-03 → SYNC-04 → SYNC-05 → SYNC-06 → SYNC-07 | Quick; stops more deprecated calls landing. |
| 3 | **G1 Decisions benchmark** | BUG-01 → BUG-02 → BUG-14 → DOC-03 → DOC-26 → BUG-16 → DOC-08 → DOC-09 → DOC-10 → SMELL-37 → SMELL-43 | BUG-02 adds `report_version`, which BUG-14 and DOC-26 describe. |
| 4 | **G2 ONNX export** | BUG-04 (resolves BUG-05 and BUG-06) → BUG-03 → DOC-16 → SMELL-29 | BUG-04 changes the check that BUG-03's staging directory feeds. |
| 5 | **G3 Run ids** | BUG-07 | Independent. |
| 6 | **G4 Hub save/load** | BUG-08 → BUG-09 → BUG-17 → BUG-18 → DOC-02 → DOC-12 → DOC-13 → DOC-21 → DOC-22 → SMELL-12 → SMELL-20 → SMELL-21 → SMELL-30 | All in `save_pretrained` / `from_pretrained`. |
| 7 | **G5 Durability and permissions** | BUG-10 → BUG-11 → BUG-12 → BUG-13 → BUG-26 → SMELL-31 (decision) → SMELL-01 | Fix the bugs in place first, then consolidate (SMELL-01). |
| 8 | **G8 Docs and CHANGELOG** | DOC-01 → DOC-04 → DOC-11 → DOC-14 → DOC-15 → DOC-25 → DOC-17 → DOC-18 → DOC-19 → DOC-20 → DOC-23 → DOC-24 → DOC-27 → DOC-28 → DOC-29 → DOC-30 | Text-only; regenerate the generated files at the end. |
| 9 | **G9 Feature bugs** | BUG-15 → DOC-06 → BUG-19 + SMELL-48 → BUG-20 + SMELL-55 → BUG-21 → BUG-22 → BUG-23 → BUG-24 → BUG-25 → BUG-27 → BUG-28 → SMELL-35 → SMELL-56 | Each is small; do BUG-15 before SMELL-35. |
| 10 | **G7 (rest)** | SCOPE-02 (decision) **or** CI-02 → CI-03 → CI-04 → CI-05 → CI-06 → DOC-07; then CI-07 | SCOPE-02 makes CI-02 to CI-06 and DOC-07 unnecessary. |
| 11 | **Dead code and clutter** | DEAD-01 … DEAD-14, CLUT-02 → CLUT-03 → CLUT-05 → CLUT-04 (optional), CLUT-01 (decision) | Low risk; batch them in one PR. |
| 12 | **G10 Refactors** | DOC-31 to DOC-35, then SMELL-04 → SMELL-05 → SMELL-15 → SMELL-02 (one step per PR) → the remaining SMELL items, smallest first | Behaviour-neutral; one per PR. |
| 13 | **G11 Architecture** | SCOPE-01, SCOPE-03, SCOPE-04, SMELL-41, SMELL-42, SMELL-46, SMELL-53 | **Only with maintainer approval.** |

---

## 3. Context the implementing agent needs

- **Package layout:**
  - `src/nnx/nn/nn_model.py`: `NNModel` (training, evaluation, export, Hub I/O)
  - `src/nnx/nn/params/nn_run.py`: run directories and atomic-write helpers
  - `src/nnx/nn/params/nn_checkpoint.py`: checkpoints, `_full_fsync`
  - `src/nnx/history.py`: the epoch journal
  - `src/nnx/decisions/`: the LLM decision and evaluation toolkit
  - `src/nnx/paradigms/`: training paradigms (KD, DPO, JEPA, …)
  - `src/nnx/interop/`: GGUF and Ollama
- **Test layout:** tests live in `tests/`, usually named `test_<module>.py`. `tests/conftest.py` sets the OpenMP environment for FAISS. The `@pytest.mark.mps` tests need Apple-silicon hardware.
- **Optional extras:** `pyproject.toml` `[project.optional-dependencies]` (`hub`, `onnx`, `quantize`, `gguf-write`, `viz`, `embeddings`, `jev`, …). A test that needs an extra uses `pytest.importorskip`.
- **PR numbers cited:**
  - #849 = 582c08a (file modes)
  - #853 = 3ed0133 (atomic exports, benchmark identity, KD placeholders)
  - #855 = ed93d51 (Hub manifest and preflight, run-id rule, dropped-state warning)
  - #857 = ed6c9dd / d429011 (`given_p_false`, MPS gpu-check)
## 4. Part A: correctness bugs (`BUG-*`)

These change what the library does for a user: a wrong result, a refused valid input, a crash, data loss, or a broken safety guarantee. Fix them before anything in Parts C–G.

---

### BUG-01: `evaluate()` fails the whole report when one sample input cannot be fingerprinted

- **Severity:** High · **Category:** regression (PR #853, commit 3ed0133) · **Group:** G1
- **Where:** `src/nnx/decisions/benchmark.py`
  - `_input_digest` (~L1298)
  - `_sample_set` (~L1313–1332)
  - the call in `evaluate` (~L1292)
  - `_fingerprint` (~L205–244)
- **Problem:**
  - `Sample.input` is typed `Any`, but since #853 `evaluate()` fingerprints *every* sample input, with no fallback.
  - `_fingerprint` only handles `str`, `tuple`, tensors, non-object ndarrays and strict JSON. Any other input makes the whole `evaluate()` call raise `BenchmarkError`, so no report is produced.
  - 0.5.0 evaluated the same samples fine. `collect()` never fingerprints, so the same samples collect fine and fail only at replay.
  - The error message ends with `_fingerprint`'s advice `"...; pass id="`. That cannot help, because the `Sample` already has an id.
- **Inputs that now fail:**
  - a dict or list holding `np.float32` / `np.int64` (`np.float64` passes because it subclasses `float`)
  - NaN inside JSON (`allow_nan=False`)
  - a list of tensors
  - a PIL image
  - `bytes`
  - a `str` with a lone surrogate
- **Evidence:**
  ```python
  def _input_digest(sample: Sample) -> str:
      try:
          return _fingerprint(sample.input)
      except BenchmarkError as error:
          raise BenchmarkError(f"sample {sample.id!r} cannot be fingerprinted for the report: {error}") from error
  ...
          sample_set=_sample_set(samples),
  ```
- **Fix:**
  1. Change `_input_digest` to return `Optional[str]`. On `BenchmarkError`, return `None` instead of raising.
  2. In `_sample_set`, compute all digests first. If any is `None`:
     - call `warnings.warn(f"evaluate(): sample {sample.id!r}'s input has no stable digest, so the report records no sample_set; compare_reports cannot check that two reports scored the same samples", UserWarning, stacklevel=3)`
     - `return None`
     - change the return type to `Optional[str]`.
  3. `BenchmarkReport.sample_set` is already `Optional`, and `compare_reports` already skips the check when either side is `None` (`if None not in sets`). Nothing else needs to change.
  4. Do **not** validate in `Sample.__post_init__`: that would break users who only call `collect()`.
  5. Leave `_fingerprint`'s `"pass id="` wording as it is. It is correct for its other caller (`_derived`, the perturbation helpers) and is no longer reachable from `evaluate`.
  6. Docs: see DOC-03. Add a CHANGELOG `### Fixed` line under `[Unreleased]`.
- **Tests:** in `tests/test_decision_benchmark.py`:
  - **Rewrite** `test_evaluate_names_a_sample_whose_input_has_no_stable_digest` (~L1349). It currently *expects* the `BenchmarkError`. It should now assert that, under `pytest.warns(UserWarning, match="'s0'")`, `evaluate(...)` returns a report with `sample_set is None` and normal metrics.
  - Add `test_evaluate_scores_samples_with_numpy_scalar_dict_inputs`: input `{"x": np.float32(1)}` scores, and `compare_reports(report, report)` gives zero deltas.
- **Done when:** both tests pass and `grep -n "pass id=" src/nnx/decisions/benchmark.py` shows the advice only in `_fingerprint`.

---

### BUG-02: reports saved by NNx ≤ 0.5.0 always fail `compare_reports` against new reports

- **Severity:** High · **Category:** backward-compatibility regression (#853) · **Group:** G1
- **Where:** `src/nnx/decisions/benchmark.py`
  - `compare_reports` (~L1604–1625)
  - `_sample_set` prefix (~L1332)
  - `FORMAT` (~L108)
  - report `state()` (~L1486–1490)
- **Problem (two keys, same root cause):**
  1. **`sample_set`:** 0.5.0 wrote `"sha256:" + digest(id, digest, label, slicing)`. #853 added the input digest to it but kept `FORMAT = "nnx.decision-benchmark/1"` and the `"sha256:"` prefix. So an old saved report and a new report of the *same* samples have different digests, and `compare_reports` raises `"the reports scored different samples"`. That message is false.
  2. **`revisions` / `prompt_identities`:** 0.5.0 report `state()` had no such keys, and `compare_reports` reads a missing key as "none". `collect()` has always stamped `prompt_identity` (a digest of `provider.record()`). So for any provider with `record()` (JEV and NLI define it), an old report against a new one raises `"different prompt identities: [] and ['sha256:…']"`.
- **Evidence:**
  ```python
  sets = (left.get("sample_set"), right.get("sample_set"))
  if None not in sets and sets[0] != sets[1]:
      raise BenchmarkError(f"the reports scored different samples of split {left['split']!r}: ...")
  for key, what in (("revisions", "model revisions"), ("prompt_identities", "prompt identities")):
      recorded = (list(left.get(key) or []), list(right.get(key) or []))
      if recorded[0] != recorded[1]:
          raise BenchmarkError(f"the reports scored with different {what}: ...")
  ```
- **Fix:**
  1. Add a module constant `REPORT_VERSION = 2`. Make `BenchmarkReport.state()` always write `"report_version": 2`.
  2. In `compare_reports`, read `lv = left.get("report_version", 1)` and `rv = right.get("report_version", 1)`.
  3. If `lv != rv`:
     - skip **both** the `sample_set` check and the revisions/prompt-identities loop
     - call `warnings.warn("compare_reports: one report was saved before NNx recorded sample inputs and revisions; only its split and metric identity are compared", UserWarning, stacklevel=2)`.
  4. Two v1 reports: compare them as 0.5.0 did (their sample-set digests are comparable with each other; no revision check).
  5. Two v2 reports: run the full checks, as today.
  6. Update the `compare_reports` docstring and `docs/decisions.md` §"Comparison" (~L653–662), and add a CHANGELOG line.
- **Tests:** in `tests/test_decision_benchmark.py`:
  - `test_a_report_saved_before_input_digests_compares_with_a_warning`:
    - `state = report.state()`, then `del state["report_version"]`, set `state["sample_set"] = "sha256:" + "0"*64`, and drop `revisions` / `prompt_identities`
    - use records stamped with a `prompt_identity`
    - assert `compare_reports(state, report)` returns zero deltas under `pytest.warns(UserWarning)`.
  - Keep `test_sample_set_follows_the_input_text`. It must still raise for two current reports.
- **Done when:** old-format states compare with a warning, and new-vs-new behaviour is unchanged.

---

### BUG-03: `to_onnx(dynamo=True)` names the external-data file after the hidden partial file

- **Severity:** High · **Category:** regression (#853, "ONNX export is atomic") · **Group:** G2
- **Where:**
  - `src/nnx/nn/nn_model.py`: `to_onnx` (~L4221). The relevant lines are ~L4344 `partial = _reserve_partial(path)`, ~L4372–4381 (dynamo export into `partial`), and ~L4395–4407.
  - `src/nnx/nn/params/nn_run.py`: `_reserve_partial` (~L290), `_commit_partial` (~L309), `_discard_partial` (~L319).
- **Problem:**
  - `to_onnx` exports into a hidden file `.model.onnx.<rand>.partial` and renames only that file onto `path`.
  - With `dynamo=True`, torch saves weights as external data named `f"{given_file_name}.data"`. onnxscript's `save_model_with_external_data` and onnx_ir externalise every tensor ≥ 256 bytes, and torch ≥ 2.6 defaults to `external_data=True` (from torch source, not run here). So almost every real dynamo export writes `.model.onnx.<rand>.partial.data`.
  - The committed `model.onnx` then references that hidden, random name. Copying `model.onnx` together with the conventional `model.onnx.data` fails to load.
  - Every re-export leaks another (possibly multi-GB) data file.
  - On failure, `_discard_partial` removes only the `.onnx` partial and leaves the `.data` file behind.
  - The legacy exporter on models > 2 GB writes one external file per tensor into the directory, and a failure leaves those behind too.
- **Evidence:**
  ```python
  partial = _reserve_partial(path)
  ...
      torch.onnx.export(self.net, example_input, partial, ..., dynamo=True)
  ...
      _commit_partial(partial, path)
  except BaseException:
      _discard_partial(partial)
  ```
- **Fix:**
  1. In `nn_run.py`, add three helpers:
     - `_reserve_export_dir(path) -> tuple[str, str]`:
       - `staging = tempfile.mkdtemp(prefix=f".{basename}.", suffix=".partial", dir=os.path.dirname(os.path.abspath(path)))`
       - return `(staging, os.path.join(staging, basename))`, so the exporter sees the *final* basename.
     - `_commit_export_dir(staging, path)`:
       - for every entry in `sorted(os.listdir(staging))` except the basename: call `_fsync_file`, then `os.replace` it into the destination directory
       - then fsync and `os.replace` the `.onnx` itself **last**
       - then `os.rmdir(staging)`.
     - `_discard_export_dir(staging)`: `shutil.rmtree(staging, ignore_errors=True)`.
  2. In `to_onnx`, replace `_reserve_partial` / `_commit_partial` / `_discard_partial` with these three helpers. Pass the staged path to all three `torch.onnx.export` calls and to `_unexported_state_keys`.
  3. Optional: before committing, read the old `path`'s external-data locations (`onnx.load(path, load_external_data=False)`, as in `export_conformance._external_locations` ~L528). After committing, delete old locations that the new model no longer references, but only plain files inside the destination directory.
  4. State in the docstring that the data file moves first and the `.onnx` last, and that the pair is not one atomic step.
- **Tests:**
  - `tests/test_onnx_dynamo.py::test_dynamo_external_data_is_named_after_the_target` (`pytest.importorskip("onnxscript")`):
    - use `hidden_dims=[64]` so a weight is ≥ 256 bytes
    - assert `set(os.listdir(tmp_path)) <= {"model.onnx", "model.onnx.data"}` and that no name starts with `"."`
    - assert every external-data `location` is `"model.onnx.data"`.
  - `tests/test_to_onnx_inputs.py::test_a_failed_export_leaves_no_external_data_behind`:
    - a fake `torch.onnx.export` writes `f` and `f + ".data"`, then raises
    - `monkeypatch.setitem(sys.modules, "onnxscript", types.ModuleType("onnxscript"))`, then call with `dynamo=True`
    - assert `sorted(os.listdir(tmp_path)) == ["model.onnx"]` (the pre-existing file).
- **Note:** the existing tests do not catch this. `tests/test_onnx_dynamo.py:55` uses a 4→8→2 net whose tensors are all < 256 bytes. `export_conformance._manifest` (~L581–592) currently hashes the hidden `.partial.data` file for the `feedfwd-fp32-dynamo` profile.

---

### BUG-04: every Conv+BatchNorm export gets a false "dropped state" warning, and `-W error` discards a correct export

- **Severity:** High · **Category:** regression (#855, commit ed93d51) · **Group:** G2
- **Where:**
  - `src/nnx/nn/nn_model.py`: `_unexported_state_keys` (~L2653–2683)
  - the warning in `to_onnx` (~L4394–4403)
- **Problem:**
  - `to_onnx` calls `self.net.eval()` and uses the default `training=TrainingMode.EVAL`. In eval mode the legacy (TorchScript) exporter's eval peephole pass (`_jit_pass_onnx_eval_peephole` / `fuseConvBatchNorm`) fuses BatchNorm into the preceding Conv. This is *not* the constant-folding pass, and it runs regardless of `do_constant_folding`.
  - The fused Conv gets new initializers (`onnx::Conv_NNN`) with new values (W·γ/√(σ²+ε) and a new bias). The BN weight, bias, running_mean and running_var initializers disappear.
  - `_unexported_state_keys` finds no initializer with the same name or the same values, so it reports the conv weight and all four BN tensors as "dropped". For a ResNet that is dozens of keys.
  - The warning is raised *inside* the export `try`. Under `-W error` or pytest `filterwarnings=error`, the warning becomes an exception, the `except` branch runs `_discard_partial`, and a correct export is thrown away.
- **Evidence:**
  ```python
  for key, tensor in net.state_dict().items():
      if not torch.is_tensor(tensor) or key in named or key.endswith("num_batches_tracked"):
          continue
      shape = tuple(tensor.shape)
      candidates = [init for init in initializers if tuple(init.dims) in (shape, shape[::-1])]
      ...
      carried = [numpy_helper.to_array(init) for init in candidates]
      if not any(np.array_equal(array, value) or np.array_equal(array, value.T) for array in carried):
          dropped.append(key)
  ```
- **Fix:** this also fixes BUG-05 and BUG-06.
  1. Replace the ONNX-value heuristic with what the warning really means: *attached tensors the forward pass never reads*. Add:
     ```python
     from torch.overrides import TorchFunctionMode

     def _unread_state_keys(net, example_input):
         seen: set[int] = set()
         def visit(x):
             if isinstance(x, torch.Tensor): seen.add(id(x))
             elif isinstance(x, (list, tuple)): [visit(v) for v in x]
             elif isinstance(x, dict): [visit(v) for v in x.values()]
         class _Reads(TorchFunctionMode):
             def __torch_function__(self, func, types, args=(), kwargs=None):
                 visit(args); visit(kwargs or {})
                 return func(*args, **(kwargs or {}))
         state = net.state_dict(keep_vars=True)   # keep_vars: real Parameter objects, so id() matches
         with torch.no_grad(), _Reads():
             net(*example_input)
         return [k for k, t in state.items()
                 if torch.is_tensor(t) and id(t) not in seen and not k.endswith("num_batches_tracked")]
     ```
  2. Call it while the net is still in eval mode, before `_restore_training_modes` in the `finally`.
  3. Reword the warning to: `"…tensor(s) the forward pass never reads (…); the exported graph does not carry them"`.
  4. **Move the `warnings.warn` after `_commit_partial`**, so a warnings-as-errors filter can no longer discard a good file. The caller still gets the exception, but the file is kept.
  5. Delete the old `_unexported_state_keys`, or keep it only as an optional extra check. If you keep it, also apply the BUG-05 fix.
  6. Document that the check does one extra eval forward pass.
- **Tests:** in `tests/test_to_onnx_dropped_state.py`:
  - Add `test_a_conv_batchnorm_export_names_no_tensor`:
    - net `nn.Sequential(nn.Conv2d(3,4,3,bias=False), nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten(), nn.Linear(144,2))` wrapped in `NNModel(module=..., params=NNModelParams(device=Devices.CPU, loss=Losses.CROSS_ENTROPY))`
    - randomise the BN stats and affine parameters
    - export under `warnings.filterwarnings("error", message=r"to_onnx\(\)", category=UserWarning)` with `dynamic_batch=False`
    - assert the file exists and no initializer is named `1.running_mean`, which proves the fold happened.
  - The existing positive tests (`class_prior`, `unused_scale`) must still pass.
- **Done when:** a Conv+BN export is silent, a net with a truly unused buffer still warns, and under `-W error` the file is still written.

---

### BUG-05: `_unexported_state_keys` is quadratic in same-shape weights (slow, memory-heavy exports)

- **Severity:** Medium · **Category:** performance regression (#855) · **Group:** G2 · **Fixed by BUG-04.** Implement this one separately only if BUG-04 keeps the ONNX-value check.
- **Where:** `src/nnx/nn/nn_model.py` ~L2667–2682.
- **Problem:**
  - For each state key that has no name match, every same- or transposed-shape initializer is converted to NumPy (`numpy_helper.to_array` copies the data) and all the copies are held in a list.
  - The legacy exporter renames every transposed MatMul weight of a `TransformerNN`, so the cost is O(L²) full-matrix copies. Peak memory is roughly all same-shape weights at once. A 12-layer d=768 model does about 1.5k copies (≈10 GB copied) per `to_onnx`.
- **Fix (only if the ONNX check stays):** build an index once, before the loop:
  ```python
  index: dict[tuple, set[str]] = {}
  for init in initializers:
      if init.data_location == TensorProto.EXTERNAL:
          continue
      a = np.ascontiguousarray(numpy_helper.to_array(init))
      index.setdefault((a.shape, a.dtype.str), set()).add(hashlib.sha256(a.tobytes()).hexdigest())
  # per key: hash value and value.T, and look each up in index
  ```
  Keep the "no shape candidate → dropped" and "external → skip" branches.
- **Tests:** `test_the_check_converts_each_initializer_once`. Monkeypatch `onnx.numpy_helper.to_array` with a call counter, export a 2-layer `TransformerNN`, and assert `calls <= len(graph.initializer)`.

---

### BUG-06: on the dynamo path the dropped-state check almost never compares values

- **Severity:** Low · **Category:** ineffective check (#855) · **Group:** G2 · **Fixed by BUG-04.**
- **Where:** `src/nnx/nn/nn_model.py` ~L2675 (`continue` for external initializers).
- **Problem:** with default external data, every candidate ≥ 256 bytes is EXTERNAL and is skipped. Only keys with *no* same-shape initializer at all are ever reported. The check is near-useless on dynamo while it over-reports on legacy (BUG-04).
- **Fix:** BUG-04's forward-read check works the same for both exporters.
- **Tests:** a dynamo variant of `test_an_attached_buffer_the_forward_pass_never_reads_is_named_in_a_warning` (`importorskip("onnxscript")`), using a 64×64 unused buffer so that a same-shape initializer exists.

---

### BUG-07: the new trailing-dot/space run-id rule breaks reading runs created by earlier versions

- **Severity:** High · **Category:** regression (#855; CHANGELOG ~L29) · **Group:** G3
- **Where:** `src/nnx/nn/params/nn_run.py`
  - `_validate_run_id` rule (~L198–204)
  - `_is_run_entry` (~L436–447)
  - `_resolve_run_id` (~L599–610)
  - `_committed_best_checkpoint` (~L644–683)
  - `NNRun.save` best election (~L1270–1274)
  - `NNRun.best_id` (~L1296–1306)
  - `NNRun.all` (~L1400–1411)
- **Problem:** a save-side portability rule ("Windows drops a trailing space or dot") is now enforced on **every read**. Directories such as `runs/exp1.` are legal on Linux and macOS and exist from 0.5.0. Three things go wrong:
  1. `NNRun.all()` filters only with `_is_run_entry`, which does not check the id, then calls `NNRun.load(entry)`. One legacy directory raises `ValueError` and the whole listing fails.
  2. `NNRun.best_id()` can return such an id, which every reader then refuses: `NNRun.load("best")`, `NNCheckpoint.load(run="best")`, `load_provenance`, resume.
  3. `_committed_best_checkpoint` wraps `NNRun._load` in `except Exception: return None`, so the legacy best run looks "ineligible". The next save of *any* run silently moves `runs/best` away from it for good.
- **Evidence:**
  ```python
  if run_id[-1] in " ." or not run_id.strip() or run_id.strip() == "..":
      raise ValueError(f"run id must not end in a space or a dot, ...")
  ...
  return [NNRun.load(id=entry, root=root) for entry in os.listdir(runs_root) if _is_run_entry(runs_root, entry)]
  ...
  best = NNRun.best_id(root)
  return None if best is None else _validate_run_id(best)
  ```
- **Fix:**
  1. Split the validator:
     - `_validate_run_id(run_id)` keeps the traversal, glob and basename checks. It applies the trailing dot/space and `..`-once-stripped rule **only when `os.name == "nt"`**, where such a name really aliases another directory.
     - `_validate_new_run_id(run_id)` = `_validate_run_id` plus that rule on **every** platform.
  2. Use `_validate_new_run_id` wherever a run id names a *new* run. The user-reachable one is `NNCheckpoint.save` (`nn_checkpoint.py` ~L753, via `_checkpoint_path`): add a `new: bool = False` parameter to `_checkpoint_path` and pass `new=True` from `save`.
  3. In `_is_run_entry`, also return `False` when `_validate_run_id(entry)` raises (wrap it in `try`). `NNRun.all`, `best_id` and the election then agree.
  4. Optional: in `_committed_best_checkpoint`, issue a `RuntimeWarning` when it skips the run that currently holds the `best` pointer.
  5. Reword CHANGELOG ~L29: "A **new** run id that Windows would normalise … is refused; existing runs keep loading."
- **Other read callers** pick up the relaxed rule automatically: `history.py:1101`, `provenance.py:560`, `comparison.py:1058,1436`, `bundles.py:852`, `nn_model.py:2891`.
- **Tests:**
  - `tests/test_pass2_r_series.py::test_t7_run_id_refuses_a_name_windows_would_normalise` (~L381): change it to call `_validate_new_run_id`.
  - `tests/test_best_pointer_readers.py`: add `test_a_legacy_run_with_a_trailing_dot_lists_loads_and_can_be_best` with `@pytest.mark.skipif(os.name == "nt", ...)`:
    - train a tiny run, then `shutil.copytree(runs/<id>, runs/"legacy.")`
    - assert `len(NNRun.all()) == 2` and that `NNRun.load("legacy.")` works
    - point best at it (`_point_best`), then assert `NNRun.load("best")` and `NNCheckpoint.load(run="best", …)` work
    - after saving another, worse run, assert `NNRun.best_id() == "legacy."`.
  - Add `test_checkpoint_save_refuses_a_new_trailing_dot_run`.

---

### BUG-08: `save_pretrained` deletes the previous `tokenizer.json` too early, and deletes a user's own tokenizer

- **Severity:** High · **Category:** data loss (#855) · **Group:** G4
- **Where:** `src/nnx/nn/nn_model.py` `NNModel.save_pretrained` (~L4616–4634).
- **Problem:**
  1. **Ordering:** the stale `tokenizer.json` is removed *before* the staged files move in. A failure or interrupt between that `os.remove` and the first `os.replace` (or on the first replace) leaves a directory that is old in every file except that `tokenizer.json` is gone.
     - The old manifest still lists it, so `from_pretrained` raises `"tokenizer.json is listed … but is missing; refusing to load it"` (`_verify_hub_manifest` ~L1442).
     - This contradicts CHANGELOG ~L12 ("A refused or failed `save_pretrained` leaves the previous artifact intact").
     - Plausible trigger: on Windows, `os.replace` raises `PermissionError` while `model.safetensors` is still memory-mapped by an earlier load.
  2. **Wrong file:** the removal checks only that the previous manifest *lists* `tokenizer.json`, not that the file still has the listed digest. A user who replaced it after an NNx save loses it on the next save without a tokenizer. This contradicts CHANGELOG ~L26 ("a user-written one is left alone").
- **Evidence:**
  ```python
  previous = _previous_hub_files(target)
  if previous is not None and _HUB_TOKENIZER_FILENAME in previous and _HUB_TOKENIZER_FILENAME not in produced:
      with contextlib.suppress(FileNotFoundError):
          os.remove(target / _HUB_TOKENIZER_FILENAME)
  for name in sorted(produced, key=lambda produced_name: _HUB_MOVE_ORDER.get(produced_name, 0)):
      ...
      os.replace(os.path.join(staging, name), target / name)
  ```
- **Fix:**
  1. Before the move loop, compute `stale = previous is not None and TOKENIZER in previous and TOKENIZER not in produced`, but delete nothing yet.
  2. After the move loop finishes (the manifest moves last), `if stale:` remove the tokenizer only if the file still matches the old manifest:
     `(target / TOKENIZER).is_file() and _file_digest(str(target / TOKENIZER))[0] == previous[TOKENIZER][0]`.
     Do the removal under `contextlib.suppress(OSError)`.
  3. A failed removal at that point is harmless: the new config and manifest no longer mention a tokenizer.
- **Tests:** in `tests/test_hub_manifest.py`:
  - `test_a_failed_move_keeps_the_previous_tokenizer`:
    - save `_lm_with_tokenizer` to `out`
    - monkeypatch `os.replace` to raise `OSError("busy")` the first time its `src` lies in a `.nnx-hub-` directory
    - assert `_tiny_model().save_pretrained(out)` raises `OSError`, `(out/"tokenizer.json").exists()`, and `GenerativeNNModel.from_pretrained(str(out))` loads.
  - `test_a_user_overwritten_tokenizer_is_kept`: save with a tokenizer, overwrite `tokenizer.json` with `'{"user": true}'`, re-save `_tiny_model()`, and assert the content is kept.

---

### BUG-09: the CHANGELOG promises that a failed `save_pretrained` leaves the previous artifact intact, but a failure mid-move does not

- **Severity:** Medium · **Category:** guarantee overstated (#853 / #855) · **Group:** G4
- **Where:** `src/nnx/nn/nn_model.py` ~L4630–4633; `CHANGELOG.md` ~L12; the `save_pretrained` docstring.
- **Problem:** a failure *after* the first `os.replace` leaves new weights beside the old config and manifest. The load refuses that mix, by design (comment ~L4630), but the previous artifact is no longer intact.
- **Fix:** pick one.
  - **(a) Docs only (simpler):** reword CHANGELOG ~L12 and the docstring to: "A refused save, or one that fails before files move, leaves the previous artifact intact; one interrupted while moving is refused on load."
  - **(b) Make the move reversible:**
    1. Before replacing `target/name`, move the existing `target/name` into `staging/".old"/name`.
    2. On an exception, move the backups back in reverse order and unlink new files that had no backup.
    3. Delete the backups once the manifest is in place.
- **Tests (for b):** `tests/test_hub_mixin.py::test_a_failed_move_restores_the_previous_artifact`: fail the second `os.replace`, then assert `NNModel.from_pretrained` still loads the old weights.

---

### BUG-10: `_fsync_file` opens the file read-only, so on Windows the flush silently never happens

- **Severity:** High · **Category:** durability regression (#853) · **Group:** G5
- **Where:** `src/nnx/nn/params/nn_run.py` `_fsync_file` (~L300–306).
- **Problem:**
  - On Windows, `os.fsync` calls `FlushFileBuffers`, which fails on a handle without write access, with `errno=EBADF`.
  - `EBADF` is in `_UNSUPPORTED_FLUSH` (`nn_checkpoint.py` ~L206–210), so `_full_fsync` swallows the error.
  - Every `_commit_partial` user therefore renames unflushed data into place on Windows: `to_onnx`, `export_state_dict` (`nn_model.py` ~L5019), `write_gguf` (`interop/gguf/writer.py` ~L323), and the Hub weights (`nn_model.py` ~L4757).
  - The project already knows this. `history._fsync_path` (`history.py` ~L306) opens `O_RDWR` *because* "Windows' fsync needs a writable handle", and `tests/test_history_journal.py:1141` tests that.
- **Evidence:**
  ```python
  def _fsync_file(path: str) -> None:
      """Flush the file at ``path`` to stable storage (through a read-only descriptor)."""
      fd = os.open(path, os.O_RDONLY)
      try:
          _full_fsync(fd)
  ```
- **Fix:**
  1. Change the open call to `fd = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))`.
  2. Change the docstring to "(through a writable descriptor: Windows' fsync needs one)".
  3. The partial files are owned by NNx, so `O_RDWR` cannot fail with `EACCES` in practice. Let any other `OSError` propagate.
  4. Do **not** change `nn_checkpoint._fsync_directory` (~L235). `O_RDONLY` is correct for directories.
- **Tests:** `tests/test_to_onnx_inputs.py::test_a_committed_export_is_flushed_writably`, modelled on `test_history_journal.py:1141`:
  - spy on `nn_run.os.open` and record `(path, flags)`
  - call `model.export_state_dict(str(tmp_path/"w.pt"))`
  - assert the open of the `.partial` file had `flags & os.O_RDWR`.

  Add a Hub variant in `tests/test_hub_export_atomic.py`. The existing `tests/test_hub_mixin.py:839` patches `_full_fsync`, so it cannot see the open mode.
- **See also:** SMELL-01 (one shared fsync helper would have prevented this).

---

### BUG-11: `_atomic_write_text` uses plain `os.fsync`, which is weaker on macOS than every other durable write

- **Severity:** Medium · **Category:** durability inconsistency · **Group:** G5
- **Where:** `src/nnx/nn/params/nn_run.py` ~L248–260, inside `_atomic_write_text`.
- **Problem:**
  - The function calls `os.fsync(raw.fileno())` and re-implements the `_UNSUPPORTED_FLUSH` tolerance inline, although `_full_fsync` is already imported at ~L37.
  - `_full_fsync` (`nn_checkpoint.py` ~L213) uses `F_FULLFSYNC` because "on macOS `fsync` only reaches the drive's cache".
  - These files go through `_atomic_write_text`: the history journal manifest (`history.py` ~L565), `run.yaml`, `metadata.yaml`, `idps.csv`, `config.json` and others. On macOS they are less durable than history's documented "#552: flushed to stable storage as a resume point's are".
- **Fix:** replace the inner `try: os.fsync(...) except OSError as error: if error.errno not in _UNSUPPORTED_FLUSH: raise` block with `_full_fsync(raw.fileno())`. `_full_fsync` already tolerates `_UNSUPPORTED_FLUSH`.
- **Tests:** `tests/test_atomic_write_text.py` (~L133/147/172) patches `nn_run.os.fsync`. On Linux that still works, because `F_FULLFSYNC` is absent and the code falls back to `os.fsync`. For the macOS lane, switch those patches to `nn_run._full_fsync`.

---

### BUG-12: `_CREATED_FILE_MODE` is computed with `os.umask(0)` at import time

- **Severity:** Medium · **Category:** thread-safety and stale permissions (#849, commit 582c08a) · **Group:** G5
- **Where:**
  - `src/nnx/nn/params/nn_run.py` ~L279–287 and ~L315
  - users: `nn_model.py` ~L1414/1417 and ~L4777/4780, `generative_nn_model.py` ~L68/71, `interop/ollama.py` ~L173/176.
- **Problem:**
  1. Between `os.umask(0)` and the restore, any file or directory another thread creates gets mode 0666/0777. The window is narrow, but `import nnx` is often done lazily on a worker thread.
  2. The mode is frozen for the process. An application that later tightens its umask (e.g. a daemon setting `0o077`) still gets 0644 for `config.json`, the Hub manifest, the tokenizer, the Modelfile and every `_commit_partial` output (ONNX, GGUF, state dict).
- **Evidence:**
  ```python
  def _created_file_mode() -> int:
      umask = os.umask(0)
      os.umask(umask)
      return 0o666 & ~umask

  # Read once at import: os.umask is process-wide, so it is not re-read per write.
  _CREATED_FILE_MODE = _created_file_mode()
  ```
- **Fix:** let the kernel apply the *current* umask instead of querying it.
  1. Add `_create_exclusive(directory, prefix, suffix, mode) -> tuple[int, str]`. It loops up to 100 times over `name = os.path.join(directory, f"{prefix}{secrets.token_hex(8)}{suffix}")`, tries `os.open(name, os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_BINARY",0)|getattr(os,"O_NOFOLLOW",0), mode)`, and continues on `FileExistsError`.
  2. `_reserve_partial`: create the partial with `mode=0o666` through `_create_exclusive` instead of `mkstemp`. Remove the `os.chmod` from `_commit_partial`.
  3. `_atomic_write_text`: replace `mode: Optional[int]` with `plain_mode: bool = False`.
     - When `True`, create the temp file with `0o666` through `_create_exclusive`.
     - Otherwise keep `mkstemp` (0600), so run and provenance files are unchanged.
     - Update the four call sites to `plain_mode=True`.
  4. Delete `_created_file_mode` and `_CREATED_FILE_MODE`, and their imports in `nn_model.py` (~L122), `generative_nn_model.py` (~L33) and `ollama.py` (~L171).
- **Tests:** `tests/test_atomic_write_text.py::test_the_plain_mode_follows_the_umask_at_write_time`, with `skipif(os.name == "nt")`:
  - `old = os.umask(0o077)`
  - call `_atomic_write_text(p, "x", plain_mode=True)` and `NNModel.export_state_dict(...)`
  - restore the umask
  - assert both files are `0o600`.

  Update `test_exported_files_keep_the_mode_a_plain_write_gives` (~L39) to the new parameter.
- **Fix together with:** BUG-13 and SMELL-31.

---

### BUG-13: atomic writers widen the permissions of an existing destination file

- **Severity:** Medium · **Category:** security-relevant regression (#853 / #849) · **Group:** G5
- **Where:** `src/nnx/nn/params/nn_run.py` `_commit_partial` (~L315) and `_atomic_write_text(mode=…)` (~L268–269).
- **Problem:**
  - Before #853/#849, `to_onnx`, `export_state_dict`, `write_gguf` and `config.json` wrote the existing file in place (`"wb"`/`"w"`), which keeps its mode.
  - Now `_commit_partial` always chmods to `_CREATED_FILE_MODE`. A user's 0600 `model.onnx` or `config.json` becomes 0644 on re-export.
- **Fix:** in `_commit_partial` (or its BUG-12 replacement) and in the `plain_mode` path of `_atomic_write_text`: if `os.path.exists(path)`, call `os.chmod(tmp, stat.S_IMODE(os.stat(path).st_mode))` before `os.replace`. Otherwise use the created mode.
- **Tests:** `tests/test_atomic_write_text.py::test_an_existing_files_mode_is_kept` (POSIX only): chmod an existing export to 0o600, re-export, and assert it is still 0o600.

---

### BUG-14: `compare_reports` refuses reports with different model revisions or prompt identities, so A/B comparison is impossible

- **Severity:** Medium · **Category:** usability regression (#853; deliberate, but it defeats the module's purpose) · **Group:** G1
- **Where:** `src/nnx/decisions/benchmark.py`
  - `compare_reports` (~L1621–1625)
  - module docstring (~L34–36), which is stale; see DOC-26
- **Problem:**
  - The module's purpose is "Score decision providers on identical samples". But `compare_reports` raises when the two reports' records carry different `revision` values, so you cannot compare model v1 and v2 on the same samples.
  - It is worse for providers. `collect()` defaults `prompt_identity` to a digest of `provider.record()` (~L628–636, ~L712), so reports from two *different providers* always differ, and two providers can never be compared unless both are collected with the same explicit `prompt_identity=`.
- **Already enshrined (keep these as the default behaviour):** this is deliberate and documented (`docs/decisions.md` ~L653–662, the `compare_reports` docstring) and tested:
  - `tests/test_decision_benchmark.py:1325 test_compare_refuses_records_differing_only_in_revision_or_prompt`
  - `tests/test_decision_benchmark.py:1339`
- **Fix:**
  1. Add a keyword-only parameter `same_records: bool = True` to `compare_reports`. When it is `False`, skip the revisions/prompt-identities loop, and keep the split, metric-identity and sample-set checks.
  2. Optionally include both reports' revisions and prompt identities in the returned comparison, so the user sees what differed.
  3. Update the `compare_reports` docstring, `docs/decisions.md` ~L653–662 and the module docstring (DOC-26). Add a CHANGELOG `### Added` line.
- **Tests:** `test_compare_allows_other_revisions_when_asked`: stamp two record sets `r@1` and `r@2`, and assert `compare_reports(a, b, same_records=False)` returns deltas. The two existing refusal tests stay as they are.

---

### BUG-15: knowledge distillation at `alpha=1` without a task silently drops the user's `extra_metrics`

- **Severity:** Medium · **Category:** silent data loss (#853) · **Group:** G9
- **Where:**
  - `src/nnx/paradigms/distillation.py`: `kd_train_step_factory.step` (~L201–218) and `feature_kd_train_step_factory.step` (~L390–404)
  - `src/nnx/objectives.py`: the `kd_objective` alpha=1 branch (~L450–455)
- **Problem:**
  - At `alpha == 1` with no `TaskSpec`, the step returns `NNEvaluationDataPoint(loss=loss_val)` only, because the labels are placeholders.
  - Dropping accuracy/F1/precision/recall is deliberate: it is in CHANGELOG ~L16, the docstring (~L147–149) and `tests/test_paradigms_distillation.py:581 test_alpha_1_scores_no_placeholder_label`.
  - But a user who passes `extra_metrics=` to `train()` gets nothing back and no error. Monitors or early stopping on such a metric never fire. That breaks the project's "refuse rather than drop" style (compare `ranking._no_extra_metrics`).
  - The step cannot tell real labels from placeholders, so "keep scoring when labels are real" is not implementable. Refusing is the right fix.
- **Evidence:**
  ```python
  if alpha == 1:  # the labels are placeholders the loss never reads: no classification fields to score
      return NNEvaluationDataPoint(loss=loss_val)
  ```
- **Fix:**
  1. In all three no-task `alpha == 1` branches, **before** the optimizer update (`finalize_step` / `finalize_task_step`), add:
     `if ctx.extra_metrics: raise ValueError("kd_train_step_factory(alpha=1) does not score labels without a TaskSpec: extra_metrics would run on placeholder labels; drop extra_metrics, declare a task, or use alpha < 1")`.
     Use the matching factory or objective name in each branch.
  2. Update the docstrings and CHANGELOG ~L16 ("…are refused").
- **Tests:** `tests/test_paradigms_distillation.py::test_alpha_1_refuses_extra_metrics_without_a_task`, parametrised over `kd`, `feature_kd` and `kd_objective`. Build the context with `extra_metrics={"x": lambda y, p: 0.0}` and assert `ValueError` and that the student's weights are unchanged.
- **Docs:** DOC-06.

---

### BUG-16: the benchmark throws away `BooleanResult.given_p_false`, and equality ignores it

- **Severity:** Low · **Category:** feature undone at a boundary (#857, commit ed6c9dd) · **Group:** G1
- **Where:**
  - `src/nnx/decisions/schema.py`: the field (~L357) and the `p_false` property (~L371–372)
  - `src/nnx/decisions/benchmark.py`: `_answer` (~L646–648) and `_row` (~L1082–1084)
- **Problem:**
  - #857 keeps a provider's own false probability as `given_p_false`, but the benchmark records only `p_true` and replays the false probability as `1 - p_true`.
  - `given_p_false` is `compare=False`, so two results with different `p_false` compare and hash equal.
  - The difference is bounded by `PROBABILITY_TOLERANCE` (1e-6), so the impact is small.
- **Evidence:**
  ```python
  given_p_false: Optional[float] = field(default=None, compare=False)
  ...
  checked: Any = validate_response(sample.question, getattr(result, "p_true", None), provider=provider_id)
  return {"p_true": float(checked.p_true)}
  ...
  return _Row(sample, labels, (record.p_true, 1.0 - record.p_true))
  ```
- **Fix:** pick one.
  - **(A) Keep the feature:**
    1. `Record`: add `p_false: Optional[float] = None`.
       - In `__post_init__`, allow it only together with `p_true`, check it with `_finite_probability`, and check `abs(p_true + p_false - 1) <= PROBABILITY_TOLERANCE`.
       - In `state()`, emit `"p_false"` **only when it is not `None`**.
       - In `from_state`, add `"p_false"` to `known` and pass it through.
    2. In `_answer`, for a Boolean that has `given_p_false`, validate `{"true": p_true, "false": p_false}` and return both values.
    3. In `_row`, use `record.p_false if record.p_false is not None else 1.0 - record.p_true`.
    4. In `schema.py`, drop `compare=False` from `given_p_false`.
    5. Add a CHANGELOG note: `Record.from_state` refuses unknown keys, so records written with `p_false` will not load in NNx 0.5.0.
  - **(B) Simpler:** remove `given_p_false` and revert #857. Then DOC-08, DOC-09 and DOC-10 become "make the docs say p_false = 1 - p_true".
- **Tests (for A):**
  - `tests/test_decision_benchmark.py::test_a_boolean_records_and_replays_the_given_false_probability`: a provider returns `{"true": 0.7, "false": 0.3000005}`; assert `Record.p_false == 0.3000005` and that it survives `write_records`/`read_records`.
  - `tests/test_decision_schema.py::test_boolean_results_with_different_given_false_differ`.
- **Docs:** DOC-08, DOC-09, DOC-10.

---

### BUG-17: the new Hub shape preflight runs a registered model factory twice per `from_pretrained`

- **Severity:** Medium · **Category:** side effects doubled (#855) · **Group:** G4
- **Where:** `src/nnx/nn/nn_model.py` `_declared_module` (~L1293–1321), called at ~L1338 and ~L4960. The real build is at ~L4964.
- **Problem:**
  - For a `ModelSpec`, `_declared_module` calls `build_module(params.net)` under `torch.device("meta")`, and then `from_pretrained` builds the real module. Every successful load runs the user's factory twice.
  - A factory with side effects runs them twice: downloading pretrained weights, `torch.load` (which allocates on CPU even inside a meta context), logging, counters. When the factory fails on meta, its side effects have already happened before `return None`.
  - Built-in `Nets` are pure, so only `ModelSpec` is affected.
- **Evidence:**
  ```python
  if isinstance(params.net, ModelSpec):
      def build() -> torch.nn.Module:
          return build_module(params.net)  # type: ignore[arg-type]
  ...
      with torch.device("meta"), _global_rng_kept():
          module = build()
  ```
- **Fix:**
  1. In `_declared_module`, make the `ModelSpec` branch `return None`. Registered modules are then checked after the build, as before #855.
  2. Optional: add an opt-in `register_model_factory(..., meta_safe: bool = False)`, stored in a `_META_SAFE` set in `models.py`, and build on meta only for factories marked meta-safe.
  3. Update the `_declared_module` docstring and CHANGELOG ~L13 ("…for built-in nets").
- **Tests:** `tests/test_hub_load_preflight.py::test_a_registered_factory_runs_once_per_load`:
  - register a counting factory and save a `ModelSpec` model
  - reset the counter, call `NNModel.from_pretrained(dir)`, and assert `calls == 1`
  - unregister in `finally`.
- **Docs:** DOC-02 (hub.md also over-states what the preflight protects against).

---

### BUG-18: the "no manifest" Hub warning points at huggingface_hub code instead of the caller

- **Severity:** Low · **Category:** wrong warning attribution (#855) · **Group:** G4
- **Where:** `src/nnx/nn/nn_model.py` `_verify_hub_manifest` (~L1427–1433).
- **Problem:**
  - The call chain is: user → `huggingface_hub` `validate_hf_hub_args._inner_fn` → `ModelHubMixin.from_pretrained` → `NNModel._from_pretrained` → `_verify_hub_manifest` → `warn`.
  - `stacklevel=3` therefore lands in `huggingface_hub/hub_mixin.py`. Python's default "once per location" filter keys on that line, so the user sees the warning attributed to library code.
- **Evidence:**
  ```python
  warnings.warn(
      f"{model_id!r} has no {_HUB_MANIFEST_FILENAME}, so its files load without digest verification; "
      "save it again with this NNx to add one",
      UserWarning,
      stacklevel=3,
  )
  ```
- **Fix:**
  1. In `src/nnx/nn/callbacks.py`, give `_warn_at_user_frame` (~L111) a keyword `skip: tuple[str, ...] = ("nnx",)`, and change its frame test to `module_name.split(".")[0] in skip`.
  2. In `_verify_hub_manifest`, replace `warnings.warn(...)` with `_warn_at_user_frame(msg, UserWarning, skip=("nnx", "huggingface_hub"), once_per_location=True)`, imported the same way as at the other call sites.
- **Tests:** `tests/test_hub_manifest.py::test_a_directory_saved_before_manifests_loads_with_a_warning` (~L114): record the warnings and assert `caught[0].filename == __file__`.
- **See also:** SMELL-27 (other raw `warnings.warn` calls with hand-picked stack levels).

---

### BUG-19: DoRA adapters cannot be saved and reloaded, and the prompt-tuning loader skips the shape check

- **Severity:** Medium · **Category:** silent wrong model after reload · **Group:** G9
- **Where:**
  - `src/nnx/peft/lora.py`: `save_lora_weights` / `load_lora_weights` (~L279–335) and `_LORA_OWNED = ("lora_A", "lora_B")` (~L264)
  - `src/nnx/peft/dora.py` (~L24–31)
  - `src/nnx/peft/prompt.py`: `load_prompt_weights` (~L182–217)
- **Problem:**
  - NNx has no DoRA-specific save/load. `save_lora_weights` on a DoRA net silently omits the learned `magnitude` vector; `tests/test_peft_dora.py:194` even asserts that LoRA matrices are "captured".
  - `load_lora_weights` then rebuilds a *different* function: the magnitude is re-initialised from `||W0||`. Nothing warns.
  - Separately, `load_prompt_weights` is the only PEFT loader that skips `_refuse_shape_mismatch`.
- **Fix:**
  1. Add `save_dora_weights` / `load_dora_weights` in `peft/dora.py`, owning `owned_adapter_keys(module, LoRALinear, _LORA_OWNED) | owned_adapter_keys(module, DoRALinear, ("magnitude",))`. Export them from `peft/__init__.py` and add them to `scripts/docs/api_reference_catalog.txt`, then regenerate `docs/api.md`.
  2. In `save_lora_weights`, call `warnings.warn("…contains DoRALinear layers; their magnitude is not saved — use save_dora_weights", UserWarning)` when the module contains a `DoRALinear`.
  3. Add `_refuse_shape_mismatch` to `load_prompt_weights`. Better: do SMELL-48, so all four PEFT pairs share one save/load helper.
- **Tests:** `tests/test_peft_dora.py::test_dora_weights_round_trip`: train a step, save with `save_dora_weights`, load into a fresh DoRA net, and assert identical outputs. Add a `pytest.warns` test for the LoRA saver on a DoRA net.

---

### BUG-20: six paradigm steps report fake `0.0` accuracy/F1/precision/recall instead of "unavailable"

- **Severity:** Medium · **Category:** wrong metrics; contract violation · **Group:** G9
- **Where:**
  - `src/nnx/paradigms/dpo.py` ~L213–225
  - `src/nnx/paradigms/jepa.py` ~L488–495
  - `src/nnx/paradigms/contrastive.py` ~L147–154
  - `src/nnx/embeddings/contrastive_trainer.py` ~L316–323
  - `src/nnx/diffusion/training.py` ~L118–125
  - `src/nnx/paradigms/augmentation.py` ~L176–183 and ~L268–275: mixup/cutmix report a real accuracy but `0.0` for f1/precision/recall
- **Problem:**
  - `NNEvaluationDataPoint` defaults these fields to `None` (`nn_evaluation_data_point.py` ~L89–92). Its docstring says it "leaves the classification fields and `error` as `None` rather than fabricating them". `lm_tasks.py` says "No classification field is fabricated".
  - These steps fabricate zeros instead. A user monitoring `accuracy` with DPO, JEPA or SimCLR sees 0.0 rather than "unavailable", and early stopping or "best" tracking can use the fake number.
  - Inside the diffusion package itself, `diffusion/objective.py` ~L194 emits no classification fields, while `diffusion/training.py`'s docstring says they "stay zero".
  - No test asserts these zeros (grepped).
- **Evidence (DPO):**
  ```python
  return NNEvaluationDataPoint(
      f1=0.0, recall=0.0, accuracy=0.0, precision=0.0,
      loss=loss_val, error=-gap,
      extra={"reward_chosen": ..., "reward_rejected": ..., "reward_accuracy": reward_accuracy},
  )
  ```
- **Fix:**
  1. In each listed `return`, delete the `f1=0.0, recall=0.0, accuracy=0.0, precision=0.0` keyword arguments. For mixup/cutmix, keep `accuracy=acc` and drop the three zeros.
  2. In `dpo.py` ~L220, DPO diagnostics go in `extra=` (documented as the user's `extra_metrics`), while objectives use `metrics=`. Move them to `metrics=`, or keep both for one release.
  3. Fix the `diffusion/training.py` docstring (~L68–70).
  4. Check that epoch aggregation and BEST tracking accept `None` fields (`nn_run._epoch_values`), and check whether saved run YAMLs or `run.id` hashing include these zeros.
- **Tests:** run `pytest tests/test_paradigms_*.py tests/test_diffusion_*.py tests/test_embeddings_contrastive.py`. Add one test per paradigm asserting `edp.accuracy is None` (or a real value for mixup).

---

### BUG-21: `embeddings.export_to_safetensors` silently writes a pickle when `safetensors` is missing

- **Severity:** Medium · **Category:** wrong file format behind a `.safetensors` name · **Group:** G9
- **Where:**
  - `src/nnx/embeddings/faiss_export.py` ~L169–218
  - NNx's own dependency-free safetensors writer: `src/nnx/bundles.py` `_tensor_chunks` / `_save_tensors` (~L689–717)
- **Problem:**
  - Without the `safetensors` package, the function writes `torch.save` output (a pickle) to a path the caller probably named `*.safetensors`. A pickle is not safe to load, which is exactly what safetensors exists to avoid.
  - It also silently skips non-tensor entries.
  - NNx already ships a writer that produces valid safetensors with no optional dependency.
- **Fix:**
  1. Move `_CODES`, `_tensor_bytes`, `_tensor_chunks` and `_save_tensors` from `bundles.py` into a new `src/nnx/_safetensors.py`, and re-import them in `bundles.py` under the same names.
  2. Make `export_to_safetensors` write through `_save_tensors(cleaned, {})` and an atomic write. Delete the pickle fallback.
  3. Raise `TypeError` naming any non-tensor keys instead of dropping them.
- **Tests:** `pytest tests/test_embeddings_faiss_export.py tests/test_run_bundles.py tests/test_bundle_validation.py`. Add a test that, with `safetensors` hidden (`monkeypatch.setitem(sys.modules, "safetensors", None)`), the output still loads with a safetensors reader, or parses as the safetensors format (8-byte header length + JSON).

---

### BUG-22: bundle JSON parsing caps nesting at 64 (everything else allows 256) and turns `1e999` into `inf`

- **Severity:** Low · **Category:** inconsistent limits; a valid run can fail to export · **Group:** G9
- **Where:** `src/nnx/bundles.py`: `_MAX_DEPTH = 64` (~L112), `_no_duplicates` (~L412) and `_parse_json` (~L420–437). The library-wide limit is `_json_depth.py:24` (`MAX_JSON_DEPTH = 256`), applied in `_artifacts.parse_json` (~L82–117).
- **Problem:**
  1. A run config nested 65–256 levels deep is valid everywhere else in NNx but fails the staged read-back in `export_bundle` (~L1321).
  2. `_parse_json` has no `parse_float` hook, so `1e999` parses to `inf`. It is caught later, by `_tensor_refs`, with a different message.
- **Fix:**
  1. Replace `_parse_json` and `_no_duplicates` with `parse_json(data.decode("utf-8"), what, BundleIntegrityError, max_depth=_MAX_DEPTH)` from `_artifacts`, keeping the UTF-8 decode-error mapping.
  2. Set `_MAX_DEPTH = MAX_JSON_DEPTH`, or document why bundles need a tighter limit.
- **Tests:** `pytest tests/test_bundle_validation.py tests/test_run_bundles.py`. Add a bundle test whose config is nested 100 levels deep, and one with `1e999` in a JSON file expecting `BundleIntegrityError`.
- **See also:** SMELL-07 (JSON parsing is implemented four times).

---

### BUG-23: `TopPFilter` and `generate(top_p=...)` accept `True` and give a raw `TypeError` for non-numbers

- **Severity:** Low · **Category:** input validation · **Group:** G9
- **Where:** `src/nnx/generation/logits_processors.py` `TopPFilter.__init__` (~L121–124). Related code is in `generation/pipeline.py` ~L75–112 and `nn/generative_nn_model.py` ~L241–242.
- **Problem:**
  - The only check is `if not (0.0 < top_p <= 1.0)`. So `TopPFilter(True)` and `generate(top_p=True)` are silently treated as `top_p=1`, and `top_p="0.9"` raises a bare `TypeError` from the comparison.
  - The other three processors validate with `require_finite_real` or an explicit bool check.
- **Fix:** in `TopPFilter.__init__`, replace the check with `top_p = require_finite_real(top_p, "top_p", owner="TopPFilter", minimum=0, exclusive_min=True, maximum=1)` (from `nnx._validation`). Check that the keyword names match its signature.
- **Tests:** `pytest tests/test_generation*.py`. Add `pytest.raises(ValueError)` for `TopPFilter(True)` and `TopPFilter("0.9")`.

---

### BUG-24: `train_contrastive` accepts `n_epochs=True` or `n_epochs=2.5` and fails late

- **Severity:** Low · **Category:** input validation · **Group:** G9
- **Where:** `src/nnx/embeddings/contrastive_trainer.py` `train_contrastive` (~L328–489; the checks are ~L402–408).
- **Problem:**
  - It checks only `if n_epochs <= 0` and `if batch_size < 2`. `n_epochs=True` runs one epoch.
  - `n_epochs=2.5` passes the check, moves the backbone and builds the optimizer, then fails at `range(2.5)`.
  - The same file's `embed_texts` already uses `require_count` (~L220).
- **Fix:**
  1. Replace both checks with `require_count(n_epochs, "n_epochs", owner="train_contrastive", minimum=1)` and `require_count(batch_size, "batch_size", owner="train_contrastive", minimum=2)`, keeping the NT-Xent explanation in the message.
  2. Replace `print` in this function with `logging.getLogger(__name__).info`.
- **Tests:** `pytest tests/test_embeddings_contrastive.py`. Add a test that `n_epochs=True`, `n_epochs=2.5` and `batch_size=1.0` raise **before** any parameter changes.
- **See also:** SMELL-53 (this is a second hand-written training loop).

---

### BUG-25: the frozen `TeacherRecord` holds a mutable `provenance` dict

- **Severity:** Low · **Category:** immutability broken · **Group:** G9
- **Where:** `src/nnx/paradigms/offline_distillation.py` ~L208 (`object.__setattr__(self, "provenance", dict(sorted(provenance.items())))`) and `__hash__` (~L223–224).
- **Problem:** `record.provenance["training_rights"] = "x"` silently changes equality and `state()` while the hash stays the same. Compare `decisions/benchmark.py` ~L417–429, which freezes its `execution` mapping.
- **Fix:** store `types.MappingProxyType(dict(sorted(provenance.items())))`, or the shared frozen mapping from SMELL-16. Make `state()` emit `dict(self.provenance)`.
- **Tests:** `pytest tests/test_offline_distillation.py`. Add a test that `record.provenance["x"] = "y"` raises `TypeError`.

---

### BUG-26: `NNTokenizerParams.save` uses a fixed `path + ".tmp"`, with no unique name and no fsync

- **Severity:** Low · **Category:** concurrency / durability · **Group:** G5
- **Where:** `src/nnx/nn/params/nn_tokenizer_params.py` ~L77–83.
- **Problem:** two concurrent saves to the same path collide on `tok.json.tmp`. A crash can also leave an unflushed file renamed into place. Every other NNx writer uses a unique temporary name and flushes before renaming.
  ```python
  tmp = path + ".tmp"
  try:
      cast(Any, tokenizer).save(tmp)
      os.replace(tmp, path)
  finally:
      if os.path.exists(tmp):
          os.remove(tmp)
  ```
- **Fix:**
  1. Use `partial = _reserve_partial(path)`, then `tokenizer.save(partial)`, then `_commit_partial(partial, path)`, and call `_discard_partial(partial)` on error (from `nnx.nn.params.nn_run`).
  2. After SMELL-01, use the shared `partial_file(path)` context manager instead.
- **Tests:** `pytest tests/test_tokenizer_params.py`. Add a test that no `*.tmp` file remains in the directory after a save.

---

### BUG-27: `export_conformance._weights_digest` would crash on bfloat16 weights

- **Severity:** Low (latent: today's profiles are FeedFwd fp32 only) · **Category:** robustness · **Group:** G9
- **Where:**
  - `src/nnx/export_conformance.py` `_weights_digest` (~L436–444)
  - the bfloat16-safe helper: `src/nnx/benchmarking.py` `_feed_digest` / `_weights_digest` (~L147)
- **Problem:** it calls `tensor...numpy()`, which raises for `torch.bfloat16`. `benchmarking.py` already has a digest helper that handles bfloat16 and quantized tensors.
- **Fix:** route `export_conformance._weights_digest` through the `benchmarking` helper. If the bytes differ, keep the current digest for fp32 so recorded evidence stays valid, and add a bf16 branch that hashes `tensor.view(torch.int16)`.
- **Tests:** `pytest tests/test_export_conformance*.py`. Add a unit test calling `_weights_digest` on a tiny bf16 `nn.Linear`.

---

### BUG-28: optional-dependency guards report a *broken* install as *missing*, and install hints differ by file

- **Severity:** Medium · **Category:** misleading errors + duplication · **Group:** G9
- **Where:**
  - `src/nnx/_optional.py` ~L19–48. `DOMAIN_EXTRAS` covers only graph, vision and plots, and `require()` already re-raises correctly when an *installed* package is broken.
  - Hand-rolled copies:
    - `quantize/ptq.py` ~L90–96
    - `quantize/qat.py` ~L93–99
    - `prune/semi_structured.py` ~L78–84
    - `interop/gguf/writer.py` ~L78–90
    - `decisions/jev.py` ~L105–110
    - `embeddings/faiss_export.py` ~L32–50
    - `viz/attribute.py` ~L52–63
    - `viz/summary.py` ~L72–77
    - `viz/netron.py` ~L115–123
    - `search.py` ~L104–111
    - `nn/nn_model.py` ~L4281 (`import onnxscript  # noqa: F401`)
  - The `hub`-extra message is written out six times: `nn_model.py` ~L165, ~L175, ~L4737, ~L4884 and `nn_checkpoint.py` ~L646, ~L1215.
- **Problem:** each copy catches *every* `ImportError` and says "install the extra". When, for example, torchao is installed but one of its own imports fails, the user is told to install what they already have. The install-hint wording also varies by file.
- **Fix:**
  1. Extend `DOMAIN_EXTRAS` with:
     - `torchao→quantize`
     - `gguf→gguf-write`
     - `typesafe_sdk→jev`
     - `faiss→embeddings` (pip name `faiss-cpu`)
     - `captum→viz`
     - `torchinfo→viz`
     - `netron→viz-interactive`
     - `optuna→optuna`
     - `safetensors→hub`
     - `huggingface_hub→hub`

     Check each extra name against `[project.optional-dependencies]` in `pyproject.toml`.
  2. Replace each `try/except ImportError` with `require("<module>", "<feature name>")`.
  3. Add `require_onnxscript(feature)` next to `require_onnx`.
  4. Remove the `# pragma: no cover` markers on those branches.
- **Tests:** run `pytest tests/test_optional*.py tests/test_core_install.py tests/test_quantize*.py tests/test_viz*.py tests/test_embeddings_faiss_export.py tests/test_search*.py tests/test_decision_jev.py tests/test_hub_mixin.py`. Many tests assert the hint text `pip install thekaveh-nnx[hub]`, so keep that format or update the tests.
## 5. Part B: CI, security and workflow risks (`CI-*`)

---

### CI-01: Dependabot PRs run untrusted dependency code on the maintainer's self-hosted Mac runner

- **Severity:** High · **Category:** security risk (live) · **Group:** G7
- **Where:**
  - `.github/workflows/mps.yml`: job `if:` conditions at ~L29 (`gpu-check`) and ~L98 (`mps-evidence`)
  - the same pattern, currently inactive, in `.github/workflows/docs.yml` ~L64
  - `.github/dependabot.yml` ~L22, 53, 62 (`target-branch: develop`)
- **Problem:**
  - `mps.yml` triggers on `pull_request`. Its only PR guard is `github.event.pull_request.head.repo.full_name == github.repository`, which blocks forks.
  - Dependabot branches live in the **same** repository, so they pass the guard.
  - The runner is live: `nnx-mps-studio` ran this workflow on 2026-11-27 (run 46065501837). Since the `NNX_MPS_RUNNER` variable is now set, the next Dependabot PR will run `uv sync --frozen --all-extras` (~L115) and `pytest` (~L133) with the *new, unreviewed* dependency versions on that machine.
  - The runner keeps state between runs: "Sync the extras" took 1 s, so `.venv` is reused.
  - The GPU-lock design (`scripts/with_mac_gpu_lock.py:3-6`, `~/.cache/mac-gpu.lock`) only works if the runner shares the maintainer's home directory, which means it runs as that account.
  - No repository secrets are referenced by these jobs.
- **Evidence:**
  ```yaml
  if: vars.NNX_MPS_RUNNER == 'true' && (github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository) && needs.gpu-check.outputs.busy == 'false'
  runs-on: [self-hosted, macos, arm64]
  ```
- **Fix:**
  1. In `mps.yml` ~L29, `mps.yml` ~L98 and `docs.yml` ~L64, replace
     `(github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository)`
     with
     `(github.event_name != 'pull_request' || (github.event.pull_request.head.repo.full_name == github.repository && github.event.pull_request.user.login != 'dependabot[bot]'))`.
  2. In `tests/test_ci_runner_image.py::test_the_self_hosted_jobs_exist_and_are_opt_in`, add `assert "dependabot[bot]" in jobs[key]["if"]` for each self-hosted job. The existing `startswith` / `in` asserts at ~L71–72 still hold.
  3. **Tell the maintainer (do not do this yourself):** run the runner as a dedicated non-admin macOS user with no access to the maintainer's home directory, keychain or SSH keys, and prefer `config.sh --ephemeral`.
- **Check:** `uv run pytest tests/test_ci_runner_image.py`. On GitHub, the next Dependabot PR should show "MPS tests (self-hosted Apple silicon)" as **skipped**.

---

### CI-02: the MPS GPU probe treats an HTTP error or timeout as "idle", and crashes on non-object JSON

- **Severity:** Medium · **Category:** CI bug (#857, commit d429011) · **Group:** G7
- **Where:** `.github/workflows/mps.yml` ~L41–54 (`service_busy` inside the `gpu-check` probe step heredoc).
- **Problem:**
  - `urllib.error.HTTPError` is a subclass of `URLError`, which is a subclass of `OSError`. So a ComfyUI or Ollama server that answers 4xx/5xx while rendering is counted as **not running**, and the MPS suite runs on a busy shared GPU.
  - A read timeout (`TimeoutError`, also an `OSError`) is likewise counted as idle.
  - A JSON list or string body makes `queue.get` raise `AttributeError`, which is not caught. Under `bash -e` the step fails red instead of emitting the "evidence not produced" warning.
  - All of this contradicts the comment's rule "state unknown → busy".
- **Evidence:**
  ```python
  try:
      with urllib.request.urlopen(url, timeout=5) as response:
          body = response.read()
  except OSError:
      return False
  try:
      return busy_for(json.loads(body))
  except ValueError:
      return True
  ```
- **Fix:** skip this item if SCOPE-02 deletes the `gpu-check` job.
  1. Move the heredoc into `scripts/probe_mac_gpu.py`, with a `main()` that prints `true` or `false`. The workflow step becomes `busy=$(python3 scripts/probe_mac_gpu.py)`. Add the script to `pyrightconfig.json` (see CI-06).
  2. Rewrite `service_busy`. Keep the except order: `HTTPError`, then `TimeoutError`, then `URLError`, then `OSError`.
     ```python
     try:
         with urllib.request.urlopen(url, timeout=5) as response:
             body = response.read()
     except urllib.error.HTTPError:
         return True                      # it answered, but not with its state
     except TimeoutError:
         return True                      # running, too slow to answer
     except urllib.error.URLError as error:
         return isinstance(error.reason, TimeoutError)  # refused/unreachable: not running
     except OSError:
         return False
     try:
         state = json.loads(body)
     except ValueError:
         return True
     return busy_for(state) if isinstance(state, dict) else True
     ```
- **Tests:** a new `tests/test_mac_gpu_probe.py`. Import `scripts.probe_mac_gpu` through the same `repo_file` guard other script tests use, monkeypatch `urllib.request.urlopen`, and assert:
  - `HTTPError(500)` → `True`
  - `URLError(ConnectionRefusedError())` → `False`
  - `TimeoutError` → `True`
  - body `b"[]"` → `True`
  - body `b'{"queue_running":[],"queue_pending":[]}'` → `False`

---

### CI-03: a busy GPU turns the MPS workflow green with no evidence

- **Severity:** Medium · **Category:** CI result misrepresents what happened · **Group:** G7
- **Where:** `.github/workflows/mps.yml` ~L25–26, ~L79–82, ~L95, ~L98; `CONTRIBUTING.md` ~L175–176.
- **Problem:**
  - The comments say a busy GPU gives "a visible skip, never a pass" and that the job "cannot pass by skipping". CONTRIBUTING says the job "fails on any skip".
  - In fact, when `busy == 'true'`, `gpu-check` succeeds and `mps-evidence` is skipped by its `if:`. GitHub reports a skipped job as **success**, for required checks too.
- **Fix:** skip this item if SCOPE-02 is applied.
  1. At ~L81, replace `echo "::warning title=MPS evidence not produced::…"` with `echo "::error title=MPS evidence not produced::the GPU is busy; rerun once it is free."` followed by `exit 1` on the next line.
  2. Change the comment at ~L25–26 to "A busy GPU fails this job, so the run is red and no evidence is produced; rerun it."
- **Check:** the maintainer holds the lock on the runner, dispatches MPS, and sees a red run.

---

### CI-04: the ComfyUI/Ollama check runs once, before the run, and never while holding the lock

- **Severity:** Low · **Category:** race · **Group:** G7
- **Where:** `.github/workflows/mps.yml` ~L61–75 vs `scripts/with_mac_gpu_lock.py` ~L31–55.
- **Problem:** the services are probed only in `gpu-check`. `with_mac_gpu_lock.py` takes only the flock, the lock is released between jobs, and `uv sync` runs in between. A ComfyUI job that starts after the probe overlaps the MPS tests.
- **Fix:** apply SCOPE-02, where the machine-local wrapper checks services inside its own wait loop. Otherwise, move the `comfy_busy` / `ollama_busy` checks into the `while True` loop of `with_mac_gpu_lock.py` (wait while the lock is held **or** a service is busy) and delete `gpu-check`.

---

### CI-05: `tests/test_with_mac_gpu_lock.py` imports `fcntl` at module level, so collection breaks on Windows

- **Severity:** Low · **Category:** portability · **Group:** G7
- **Where:** `tests/test_with_mac_gpu_lock.py:5`.
- **Problem:** `import fcntl` runs before the file's `repo_file(...)` guard (~L13), so test collection errors on Windows. The sdist ships `tests/`, and NNx targets Windows (see BUG-07 and BUG-10). The repo's own convention is `pytest.importorskip("resource")  # POSIX only` (`tests/test_benchmarking.py:918`).
- **Fix:** skip this item if SCOPE-02 deletes the file. Otherwise:
  1. Delete line 5.
  2. Add `import pytest` to the third-party import block.
  3. After `from tests._repo_files import repo_file`, add `fcntl = pytest.importorskip("fcntl")  # POSIX only`.
- **Check:** `uv run ruff check tests/test_with_mac_gpu_lock.py && uv run pytest tests/test_with_mac_gpu_lock.py`.

---

### CI-06: pyright does not check `scripts/with_mac_gpu_lock.py`

- **Severity:** Low · **Category:** config out of sync · **Group:** G7
- **Where:** `pyrightconfig.json` ~L2–12.
- **Problem:** every other top-level script is listed, but `with_mac_gpu_lock.py`, added in this range, is not.
- **Fix:** add `"scripts/with_mac_gpu_lock.py",` after `"scripts/smoke_jev_live.py",`, unless SCOPE-02 deletes the script. If CI-02 creates `scripts/probe_mac_gpu.py`, add that too.
- **Check:** `uv run pyright --warnings`.

---

### CI-07: the hosted `mps-gate-without-mps` job installs libcairo2 for nothing

- **Severity:** Low · **Category:** dead CI step · **Group:** G7
- **Where:** `.github/workflows/mps.yml` ~L166–171.
- **Problem:** the job runs only `pytest -m mps`. No `mps`-marked test imports cairosvg. The step looks copied from `ci.yml`.
- **Fix:** delete the "Install diagram rendering runtime" step. `test_every_apt_step_has_its_own_short_timeout` still finds apt steps in `ci.yml`.
- **Check:** `uv run pytest tests/test_ci_runner_image.py`. On GitHub, the "MPS gate without MPS" job stays green.
## 6. Part C: documentation issues (`DOC-*`)

The review checked these mechanically and found them **in sync**, so do not "fix" them:
- `docs/api.md` matches `scripts/docs/api_reference_catalog.txt` (484 entries).
- Every `from nnx… import` and every keyword argument in a doc code block resolves.
- The docs build passes (`mkdocs build --strict` on the generated config).
- `PYPI_README.md` equals `build_pypi_readme.render()`.
- The feature-composition ids are consistent.
- Numbered headings are sequential, except DOC-17.

**Workflow for docstring changes:** edit the docstring in `src/`, then run `uv run python -m scripts.docs.build_api_reference`, because `docs/api.md` is generated. For README changes, run `uv run python -m scripts.docs.build_pypi_readme`. Finish with `uv run python -m scripts.docs.build_docs --check`.

---

### DOC-01: the new DPO refusal of empty or all-pad responses is documented nowhere

- **Severity:** High: a breaking runtime refusal that fails training mid-epoch. · **Group:** G8
- **Where:** `CHANGELOG.md` `[Unreleased]` (~L7–39); `docs/dpo.md` ~L26–31 and ~L123; `src/nnx/paradigms/dpo.py` docstring Raises (~L110–117).
- **Evidence:**
  - `dpo.py` ~L151–152 calls `_require_response_tokens(chosen_ids, pad_token_id, "chosen")` and the same for `rejected`. It was added in 3ed0133, after 0.5.0.
  - It raises `ValueError("DPO {name} response of batch row {row} has no valid tokens (...) … Drop that row from the dataset.")` at every step.
  - `nn_preference_dataset.py` ~L76–81 still right-pads an empty response to all-pad without refusing it, so the dataset accepts rows that training then rejects.
  - `docs/dpo.md` ~L123 even says "the default keeps results unchanged".
  - `grep -n "no valid token" CHANGELOG.md docs/dpo.md` finds nothing.
- **Fix:**
  1. CHANGELOG `[Unreleased] ### Changed`, add: `- **DPO refuses a response with no valid tokens.** \`dpo_train_step_factory\` raises \`ValueError\` at the step whose batch holds a chosen or rejected response that is empty or every position \`pad_token_id\` (its log-prob would be 0 with no gradient yet still count as a pair). Drop such rows before building \`NNPreferenceDataset\`. With \`prompt_packing=True\` a prompt with no valid token is refused the same way.`
  2. In `docs/dpo.md`, after ~L31, add a paragraph saying the same thing. Reword ~L123 so it no longer claims results are unchanged.
  3. In the `dpo.py` docstring, append to the `ValueError` item in Raises: `At every step: a chosen or rejected response with no valid token (empty, or all \`\`pad_token_id\`\`) and, with \`\`prompt_packing=True\`\`, a prompt with no valid token.` Then regenerate `api.md`.
  4. Optional, better UX: make `NNPreferenceDataset` refuse or warn about such rows at construction time.
- **Check:** `grep -n "no valid token" CHANGELOG.md docs/dpo.md src/nnx/paradigms/dpo.py docs/api.md` returns 4 hits.

---

### DOC-02: `docs/hub.md` says the shape preflight protects against "a huge network". It does not.

- **Severity:** Medium: a safety claim about untrusted artifacts. · **Group:** G4
- **Where:** `docs/hub.md` ~L176.
- **Evidence:**
  - The doc says: "It also checks the declared shapes on the meta device, so a config asking for a huge network is refused before any memory is allocated."
  - `_refuse_shapes_before_build` (`nn_model.py` ~L1322–1357) only compares the meta-built module with the shapes in the `model.safetensors` header, and refuses only on a *mismatch*.
  - A config and header that agree on a huge network pass, and the network is then allocated.
  - `_declared_module` returns `None`, so no preflight runs at all, when the constructor cannot run on meta (~L1298–1320), and for registered factories after BUG-17.
  - CHANGELOG ~L13 is accurate.
- **Fix:** replace the sentence with: `It also builds the declared network on the meta device and compares it with the tensor shapes in the \`model.safetensors\` header, so a config whose shapes do not match its weights is refused before any memory is allocated. A config and header that agree are built as declared, however large, and a registered factory (or one that cannot run on the meta device) is checked only after the build. Inspect an untrusted \`config.json\` first.`
- **Check:** `grep -n "huge network" docs/hub.md` returns nothing.

---

### DOC-03: the benchmark docs miss the `evaluate()` fingerprint refusal and the saved-report incompatibility

- **Severity:** Medium · **Group:** G1 (document what BUG-01 and BUG-02 leave behind)
- **Where:** `docs/decisions.md` ~L553–554, ~L640–645 and ~L652–660; `CHANGELOG.md` ~L34; the `evaluate` docstring in `benchmark.py`.
- **Evidence:** see BUG-01 and BUG-02. `docs/decisions.md` tells readers only that the perturbation helpers need `id=` (~L645). The `evaluate` docstring has no Raises entry. CHANGELOG ~L34 does not warn that old reports stop comparing.
- **Fix:** write whichever text matches the code *after* BUG-01 and BUG-02.
  - **If BUG-01 is fixed:** in decisions.md §"Samples", after ~L554, add: `When a sample's input has no stable digest (text holding a lone surrogate, an object JSON cannot hold, a list holding NaN), \`evaluate\` warns and the report records no sample set, so \`compare_reports\` cannot check that two reports scored the same samples.`
  - **If BUG-02 is fixed:** under "Comparison" (~L660), add: `A report saved by NNx 0.5.0 or earlier compares with a newer one with a \`UserWarning\`: only its split and metric identity are checked.`
  - **If either is NOT fixed:** document the current refusal and the incompatibility instead (`re-run evaluate on the saved records`), and append the same note to CHANGELOG ~L34.

---

### DOC-04: `docs/gguf.md` omits the new refusal of state that no GGUF tensor carries

- **Severity:** Medium · **Group:** G8
- **Where:** `docs/gguf.md` ~L90–94.
- **Evidence:**
  - The doc says: "They also refuse PEFT adapters and factorized projections."
  - `interop/gguf/tensor_name_map.py` ~L292–298 computes `unread = _unread_state_keys(net, read)` and raises `TypeError("GGUF export has no tensor slot for …")`.
  - This is documented in the `write_gguf` docstring (~L206–209) and CHANGELOG ~L14.
  - It bites a model with a `logit_scale` attached, which the hub and jepa guides recommend.
- **Fix:** replace the sentence with: `They also refuse PEFT adapters, factorized projections and any \`state_dict\` entry no GGUF tensor carries (an auxiliary head, a \`logit_scale\`, a bias); the \`TypeError\` names every such key. Remove it from the network, or export a copy that holds only the \`TransformerNN\`'s own weights.`
- **Check:** `grep -n "logit_scale" docs/gguf.md`.

---

### DOC-05: the calibration and abstention guides miss the `test_split_id` deprecation and two new refusals

- **Severity:** Medium · **Group:** G6
- **Where:** `docs/concepts.md` ~L2526–2535 (§18 "Split separation") and ~L2693–2697 (§19 "Threshold selection").
- **Evidence:**
  - `_validation.py` ~L253–261 warns `"{function}() without test_split_id is deprecated: test_split_id will be required in the next minor release…"`. It is called from `fit_temperature` (`calibration.py` ~L941) and `select_threshold` (`abstention.py` ~L1377), and appears in CHANGELOG ~L18.
  - `select_threshold` also refuses a `CalibratedPrediction`'s own `calibration_split_id` (`abstention.py` ~L239–241, ~L1380) and zero rows (~L1393–1394), per CHANGELOG ~L19 and ~L33.
  - `grep -n "deprecat\|calibration_split_id" docs/concepts.md` finds nothing.
- **Fix:**
  1. Add to §18: `Omitting \`test_split_id\` (or passing \`None\`) is deprecated: \`fit_temperature\` and \`select_threshold\` emit a \`DeprecationWarning\`, and the argument becomes required in the next minor release.`
  2. Append to the §19 Threshold selection bullet: `So is a \`CalibratedPrediction\`'s own calibration split (\`calibration_split_id\`, where its temperature was fitted), and a source with no rows raises \`AbstentionError\`.`
  3. The example and test parts of this deprecation are SYNC-01 to SYNC-03.

---

### DOC-06: the KD docs say every record reports top-1 error and that the teacher is left in eval mode. Both changed.

- **Severity:** Medium · **Group:** G9
- **Where:** `docs/concepts.md` ~L898–899, ~L1601 and ~L1607; `src/nnx/objectives.py` `KDObjective` docstring (~L365–366).
- **Evidence:**
  - Line ~1607 says "EDP reports the combined loss and student top-1 error". At `alpha == 1` without a task, the record holds the loss only (`distillation.py` ~L210–211 and ~L396; `objectives.py` ~L452–455; CHANGELOG ~L16).
  - Line ~1601 says the teacher "puts its net in eval mode before its forward". The code now uses `pinned_eval` (`_step_helpers.py` ~L102–114), which restores every submodule's training flag afterwards (CHANGELOG ~L26).
  - `objectives.py` ~L365 still says "put back in eval mode before every teacher forward".
- **Fix:**
  1. Append to ~L1607: `At \`alpha=1\` without a declared task the labels are placeholders, so the record carries the loss only (no error, accuracy, F1, precision or recall, and \`extra_metrics\` are not called); \`kd_objective(alpha=1)\` and \`feature_kd_train_step_factory(alpha=1)\` record the same.` If BUG-15 is fixed, write "`extra_metrics` are refused" instead.
  2. At ~L1601, replace "puts its net in eval mode** before its forward" with "puts its net in eval mode** around its forward, then gives each submodule's training flag back".
  3. Append to ~L899: ", and records the loss only".
  4. Change the `objectives.py` docstring the same way, then regenerate `api.md`.

---

### DOC-07: CONTRIBUTING §5.2 does not describe the MPS `gpu-check` job, its skip, or `NNX_MAC_GPU_LOCK`

- **Severity:** Medium · **Group:** G7
- **Where:** `CONTRIBUTING.md` ~L172–186 vs `.github/workflows/mps.yml` ~L22–84 and ~L98; `scripts/with_mac_gpu_lock.py` ~L5 and ~L27.
- **Evidence:**
  - CONTRIBUTING says the job "fails on any skip" and that each GPU step "takes the shared Mac GPU lock … and gives up after 15 minutes".
  - In fact a non-waiting `gpu-check` job probes the lock and the ComfyUI (`127.0.0.1:8188`) and Ollama (`127.0.0.1:11434`) queues, and skips `mps-evidence` when any of them is busy.
  - The `NNX_MAC_GPU_LOCK` override is undocumented.
- **Fix:** if SCOPE-02 is applied, rewrite the section as SCOPE-02 says. Otherwise, after "gives up after 15 minutes.", insert: `A \`gpu-check\` job runs first. It probes the lock without waiting and asks the local ComfyUI (port 8188) and Ollama (port 11434) servers whether they are busy. When the lock is held or either is busy, \`mps-evidence\` is skipped with a "MPS evidence not produced" warning: a skip, never evidence; rerun once the GPU is free. \`NNX_MAC_GPU_LOCK\` overrides the lock path.` Also see CI-03.
- **Check:** `grep -n "gpu-check" CONTRIBUTING.md`.

---

### DOC-08: the Boolean response form in decisions.md and `validate_response` is out of date

- **Severity:** Low · **Group:** G1
- **Where:** `docs/decisions.md` ~L84–85; `src/nnx/decisions/schema.py` ~L449–450.
- **Evidence:** both say `response` is `p_true` or `{"true": p, "false": 1 - p}`. Since #857, any `q` with `|p + q - 1| <= PROBABILITY_TOLERANCE` is accepted and kept as `given_p_false` (`schema.py` ~L458–472). decisions.md ~L69–70 already says this.
- **Fix:** change both to: `` `p_true` or `{"true": p, "false": q}` with `p + q` within `PROBABILITY_TOLERANCE` of 1 (`q` is kept as `p_false`) ``. Regenerate `api.md`. If BUG-16 option B is chosen, write `1 - p` instead.
- **Check:** `grep -n '"false": 1 - p' docs/decisions.md src/nnx/decisions/schema.py docs/api.md` returns nothing.

---

### DOC-09: `BooleanResult.p_false` and `given_p_false` are undocumented, and 367 api.md entries say "No public description"

- **Severity:** Low · **Group:** G1, then G8
- **Where:** `src/nnx/decisions/schema.py` ~L347–372; `docs/api.md` ~L3199–3206.
- **Evidence:**
  - The class docstring still says "one ``p_true`` in ``[0, 1]``".
  - The `p_false` property has no docstring, so api.md ~L3206 shows "No public description is currently available."
  - 367 api.md entries carry that placeholder (e.g. `CoverageReport.coverage`, `CoverageReport.risk`, `ExperimentPlan.with_model`). `p_false` is the only one added in this range.
- **Fix:**
  1. Add to the class docstring: `` ``p_false`` is the provider's own false probability when it gave one (``given_p_false``, checked against ``p_true`` within :data:`PROBABILITY_TOLERANCE`), else ``1 - p_true``. ``given_p_false`` is kept out of equality. `` If BUG-16 option A removes `compare=False`, drop the last sentence.
  2. Give the `p_false` property a one-line docstring.
  3. Regenerate `api.md`.
  4. Separate, larger task: run `grep -c "No public description" docs/api.md` and write docstrings for the most-used undocumented public members, starting with those named in guides.

---

### DOC-10: "Stored as given" over-claims: benchmark records drop the given `p_false`

- **Severity:** Low · **Group:** G1
- **Where:** `CHANGELOG.md` ~L11; `docs/decisions.md` ~L69–70.
- **Evidence:** see BUG-16. `collect` records only `{"p_true": …}` (`benchmark.py` ~L647–648), and replay uses `1.0 - record.p_true` (~L1084).
- **Fix:** if BUG-16 option A is done, no doc change is needed. Otherwise append to both sentences: `on the \`BooleanResult\`; benchmark replay records (\`nnx.decision-record/1\`) keep \`p_true\` only, so replayed metrics use \`1 - p_true\`.`

---

### DOC-11: the new `StreamingMetrics.update` probability refusal is missing from its docstring and concepts; the CHANGELOG wording is incomplete

- **Severity:** Low · **Group:** G8
- **Where:** `src/nnx/streaming.py` ~L499–517 (docstring); `docs/concepts.md` ~L2962–2964; `CHANGELOG.md` ~L32.
- **Evidence:** 3ed0133 added `categorical_probabilities(_scored_rows(probabilities, mask), error=ValueError, allow_empty=True)` (~L545–547). It also refuses rows that do not sum to 1 (`_probability.py` ~L176), and it runs only for categorical semantics. CHANGELOG mentions "NaN, infinite and out-of-range" rows only.
- **Fix:**
  1. Add to the docstring and the concepts bullet: `Categorical probabilities of the scored rows must be finite, in [0, 1] and sum to 1 within the input dtype's rounding bound; any other row raises \`ValueError\` before anything is accumulated.`
  2. Change CHANGELOG ~L32 to "refuse NaN, infinite, out-of-range and unnormalized categorical probability rows".
  3. Regenerate `api.md`.

---

### DOC-12: `docs/hub.md` §2.1 lists the save layout without the new manifest and tokenizer

- **Severity:** Low · **Group:** G4
- **Where:** `docs/hub.md` ~L105–107.
- **Evidence:** it says "(`model.safetensors` + `config.json` + `README.md`)". The code also writes `nnx_hub_manifest.json` (`nn_model.py` ~L4783), and `tokenizer.json` for a `GenerativeNNModel` (`generative_nn_model.py` ~L61–73). §2.2 (~L158) was updated; §2.1 was not.
- **Fix:** replace the parenthetical with `(\`model.safetensors\` + \`config.json\` + \`nnx_hub_manifest.json\` + \`README.md\`, plus \`tokenizer.json\` for a \`GenerativeNNModel\`)`.

---

### DOC-13: the `_from_pretrained` docstring describes the old load path

- **Severity:** Low · **Group:** G4
- **Where:** `src/nnx/nn/nn_model.py` ~L4811–4818.
- **Evidence:**
  - The docstring says it reads `config.json` "via their public `from_state` constructors, then load[s] `model.safetensors`".
  - The code calls `NNParams.resolve_from_state` (~L4932), verifies the manifest first (`_verify_hub_manifest`, ~L4900) and preflights shapes (`_refuse_shapes_before_build`, ~L4955).
- **Fix:** rewrite the description as: `verify \`\`nnx_hub_manifest.json\`\` (a directory without one loads with a UserWarning), rebuild the net params with \`\`NNParams.resolve_from_state\`\` and \`\`NNModelParams\`\` with \`\`from_state\`\`, check the declared shapes against the weights' header on the meta device, then build and load \`\`model.safetensors\`\``.

---

### DOC-14: three behaviour changes from 3ed0133 have no CHANGELOG entry

- **Severity:** Low · **Group:** G8
- **Where:** `CHANGELOG.md` `[Unreleased] ### Fixed` (~L21–35).
- **Evidence:** the commit message lists these, and the code confirms each:
  - an atomic tokenizer write: a symlink is replaced, not written through (`generative_nn_model.py` ~L65–72)
  - an atomic GGUF write (`writer.py` ~L210–211)
  - `push_to_hub` refusing non-tensor state before it creates the repository (`nn_model.py` ~L4697–4699).
- **Fix:** add three Fixed bullets:
  - `` `GenerativeNNModel.save_pretrained` writes `tokenizer.json` atomically; a symlink at that path is replaced, never written through. ``
  - `` `write_gguf` / `export_ollama_modelfile` write atomically: a failed write leaves an existing file untouched. ``
  - `` `push_to_hub` refuses non-tensor state before it creates the repository. ``

---

### DOC-15: the `export_bundle` docs do not mention the new resume-point verification or the `"best"` refusal

- **Severity:** Low · **Group:** G8
- **Where:** `src/nnx/bundles.py` `export_bundle` docstring (~L1171–1200); `docs/concepts.md` ~L3072–3080 (§22 "Refused at export").
- **Evidence:** 3ed0133 added `NNCheckpoint.verify(run=run_id, type=label, root=root)` before loading. It raises `ResumePointError` for torn, mixed or corrupted points (`nn_checkpoint.py` ~L871–886; CHANGELOG ~L23). The `"best"` refusal (`BundleError`) is also missing from the docstring.
- **Fix:** add to both places: `a checkpoint whose resume point fails verification (torn, mixed or a digest mismatch) raises \`ResumePointError\` before anything is written; the run id \`"best"\` is refused (export the id \`NNRun.best_id()\` names).` Regenerate `api.md`.

---

### DOC-16: the `to_onnx` docstring omits the new warning, the atomic write and two refusals

- **Severity:** Low · **Group:** G2
- **Where:** `src/nnx/nn/nn_model.py` ~L4231–4272.
- **Evidence:**
  - The code warns with a `UserWarning` (~L4395–4403) and writes through `_commit_partial` (~L4404–4407; CHANGELOG ~L27).
  - It raises `NotImplementedError` for a net declaring `onnx_export_unsupported`, and `TypeError` for a mapping example input (~L4264–4272).
  - The docstring's Raises section lists only `ImportError`.
- **Fix:**
  1. Add: `Writes atomically (a failed export leaves an existing file untouched) and warns (\`\`UserWarning\`\`) when …`. Finish the sentence to match the code after BUG-04, e.g. "…when the net holds state tensors the forward pass never reads".
  2. Add Raises entries for `NotImplementedError` and `TypeError`.
  3. If BUG-03 is fixed, document that external data is written as `<name>.data` beside the file.
  4. Regenerate `api.md`.

---

### DOC-17: `docs/quickstart-recipes.md` numbering continues quickstart.md instead of restarting

- **Severity:** Low · **Group:** G8
- **Where:** `docs/quickstart-recipes.md` ~L5 and ~L7 (the whole page); `docs/quickstart.md` ~L65.
- **Evidence:**
  - The H1 is "# 3. Quickstart recipes", but the page starts with "## 2. Common variations" and "### 2.2. Precision…". There is no 2.1, and "## 2. Common variations" duplicates the heading at `quickstart.md:55`.
  - Every other page restarts at "## 1.".
  - Cross-references like "Quickstart recipes §3.10" read as page 3 §10.
- **Fix:** pick one.
  - **(a)** Add a one-line note under the H1: "Sections continue Quickstart's numbering: §2.2–§2.22 and §3."
  - **(b)** Renumber to "## 1." / "## 2." and update every `quickstart-recipes.md#2…` and `#3…` link (`README.md:281`, `capabilities.md:60`, `concepts.md:34` and others; find them all with `grep -rn "quickstart-recipes.md#" docs README.md`).
- **Check:** `uv run python -m scripts.docs.build_docs --check`.

---

### DOC-18: the README §6.1 doc list is stale after the quickstart split

- **Severity:** Low · **Group:** G8
- **Where:** `README.md` ~L328 and §6.1.
- **Evidence:**
  - README says "[Quickstart](docs/quickstart.md) — a runnable example with variations to copy and adapt". The variations moved to `docs/quickstart-recipes.md`.
  - README never links Quickstart recipes, Export conformance, Feature composition or the Decision-model pilot. `docs/index.md` lists them all.
- **Fix:**
  1. Change ~L328 to `- [Quickstart](docs/quickstart.md) — a runnable minimal example.`
  2. Add `- [Quickstart recipes](docs/quickstart-recipes.md) — variations and recipes beyond supervised classification.`, plus entries for Export conformance and Feature composition, copying the one-line descriptions from `docs/index.md`.
  3. Run `uv run python -m scripts.docs.build_pypi_readme`.
- **Check:** `uv run python -m scripts.docs.build_pypi_readme --check`.

---

### DOC-19: exported constants that guides cite are not in api.md, and nothing checks export coverage

- **Severity:** Low · **Group:** G8
- **Where:** `docs/gguf.md` ~L35; `docs/decisions.md` ~L276; `scripts/docs/api_reference_catalog.txt` ~L205 and ~L1057–1067.
- **Evidence:**
  - `grep -c SUPPORTED_QUANTIZATIONS docs/api.md` and `grep -c SDK_RANGE docs/api.md` both give 0, yet the guides name `nnx.interop.gguf.SUPPORTED_QUANTIZATIONS` and `nnx.decisions.jev.SDK_RANGE`.
  - Comparing every `__all__` with the catalog shows only module constants missing (`FORMAT`, `PROBABILITY_TOLERANCE`, `EXIT_CODES`, `PIPELINE_VERSION` and others). No test checks `__all__` coverage.
- **Fix:**
  1. Change catalog ~L205 to `@api nnx.decisions.jev members=SDK_RANGE`.
  2. Add `@api nnx.interop.gguf members=SUPPORTED_QUANTIZATIONS` in the GGUF section.
  3. Regenerate `api.md`.
  4. Optional: add a test that every `__all__` name is a catalog entry or member, or is on an explicit allow-list.

---

### DOC-20: `train(resume_from_run_id=...)` and `train(resume_mode=...)` imply keyword arguments that do not exist

- **Severity:** Low · **Group:** G8
- **Where:** `docs/hub.md` ~L257; `CHANGELOG.md` ~L168.
- **Evidence:** the signature is `NNModel.train(self, params, callbacks, train_step_fn, eval_step_fn, salt, components, objective, provenance, history, compile, distributed)` (`nn_model.py` ~L5028–5041). `resume_from_run_id` and `resume_mode` are `NNTrainParams` fields (`nn_train_params.py` ~L198 and ~L213).
- **Fix:** write `train(NNTrainParams(resume_from_run_id=...))` in hub.md, and "`NNModel.train` with `NNTrainParams(resume_mode="weights_only")`" in the CHANGELOG.
- **Check:** `grep -n "train(resume_" docs/*.md CHANGELOG.md` returns nothing.

---

### DOC-21: the Hub row of the external-contracts ledger omits the new manifest and preflight contract and their tests

- **Severity:** Low · **Group:** G4
- **Where:** `docs/external-contracts.md` ~L27 (its review rule at ~L140 requires this row to change with the contract).
- **Fix:**
  1. Add to the contract column: `the NNx \`nnx_hub_manifest.json\` (SHA-256 and size per file) and the safetensors JSON header read for shape preflight`.
  2. Add `tests/test_hub_manifest.py`, `tests/test_hub_load_preflight.py` and `tests/test_hub_export_atomic.py` to the verification column.

---

### DOC-22: `docs/hub.md` §1.2 omits the `to_file(format="safetensors")` refusals

- **Severity:** Low · **Group:** G4
- **Where:** `docs/hub.md` ~L66–71.
- **Evidence:** since 0.4.4, `to_file(format="safetensors")` refuses a recorded transform it cannot replay (`nn_checkpoint.py` ~L625–637), a runtime-only module (`MissingModelFactoryError`), and non-JSON metadata (`TypeError`). Its docstring (~L572–592) says so; §1.2 does not.
- **Fix:** append: `A runtime-only module (\`MissingModelFactoryError\`), a recorded transform NNx cannot replay from data (\`ValueError\`, as \`save_pretrained\` refuses it) and a record value JSON cannot hold (\`TypeError\`) are refused before anything is written; keep such a checkpoint as \`format="pickle"\`.`

---

### DOC-23: stale `mkdocs.yml` comments in `.gitignore` and the maintenance ledger

- **Severity:** Low · **Group:** G8
- **Where:** `.gitignore` ~L62–63; `.maintenance/overnight-2026-07-01.md` ~L36.
- **Evidence:**
  - `.gitignore` says "mkdocs.yml already excludes both [docs/specs, docs/superpowers] from the published site".
  - `mkdocs.yml` is now generated by `scripts/docs/build_docs.py:render_mkdocs` (~L301–328), ignored (`.gitignore:56`), and has no `exclude_docs`.
  - Those folders stay unpublished only because `docs_dir: generated/site` holds just the pages `docs/manifest.yaml` lists.
- **Fix:** replace `.gitignore` ~L62–63 with `# accidental \`git add -A\`. Belt-and-braces: the generated site holds only\n# the pages docs/manifest.yaml lists, so neither is published.` The ledger is handled by CLUT-01.
- **Check:** `grep -n "mkdocs.yml already" .gitignore` returns nothing.

---

### DOC-24: the README quickstart has drifted from `docs/quickstart.md`

- **Severity:** Low · **Group:** G8
- **Where:** `README.md` ~L129–169 vs `docs/quickstart.md` ~L9–51.
- **Evidence:**
  - quickstart.md has `seed=42` and `grad_clip_norm=1.0`, and calls `result = model.predict(X=X_val)`.
  - README omits both settings and calls `logits, classes = model.predict(X=X_val.numpy())`.
  - No test compares the two.
- **Fix:**
  1. Copy the quickstart.md block into README §3, so the two blocks are identical.
  2. Rebuild `PYPI_README.md`.
  3. Optional: add a test in `tests/test_docs_projection.py` asserting the two code blocks are equal.

---

### DOC-25: CHANGELOG 0.4.4 lists every entry twice, 0.4.3 has one duplicate, and the duplicate test misses `*` bullets

- **Severity:** Low · **Group:** G8
- **Where:** `CHANGELOG.md` ~L47–83 (0.4.4) and ~L87–92 (0.4.3); `scripts/release/finalize_changelog.py` `finalize` (~L118–148); `tests/test_release_finalize.py:237`.
- **Evidence:**
  - Every 0.4.4 bullet appears twice with different hashes (e.g. `31a4258` / `5c4bc22`), because each change landed once on develop and once on main.
  - release-please generated them (ee10a92, PR #803), and 15630ec (#833) restored them verbatim from the v0.4.4 tag.
  - 0.4.3 repeats "fix 0.4.2 audit findings in bundle, run, DDP and GGUF text" (00051b0 / 1a8dc31).
  - Both sections are still release-please's raw hash lists, although CONTRIBUTING §7 says the post-release sync replaces them with curated notes.
  - `test_no_changelog_section_lists_one_entry_twice` only matches `- **title**` bullets, so it misses release-please's `* ` bullets.
- **Fix:**
  1. In 0.4.4 and 0.4.3, delete the second line of each pair whose text is identical once the trailing `([hash](…))` and `, closes […]` are stripped. Optionally replace each section with curated notes, as CONTRIBUTING §7 describes.
  2. In `finalize()`, on the branch that keeps release-please's list (`else: body = lines[release:unreleased]`), add a `_dedupe_generated(lines)` helper. It strips each bullet with the regex `r"\s*\(\[[0-9a-f]{7}\]\([^)]*\)\).*$"` and drops bullets whose stripped text repeats an earlier one in the same `###` subsection.
  3. Extend the test to also collect `^\* (.+?) \(\[` subjects per section and assert there are no repeats.
  4. Add `test_finalize_dedupes_a_generated_list_without_curated_notes` with a two-line fixture.
- **Check:** `uv run pytest tests/test_release_finalize.py`.

---

### DOC-26: the `benchmark.py` module docstring understates what `compare_reports` refuses

- **Severity:** Low · **Group:** G1
- **Where:** `src/nnx/decisions/benchmark.py` ~L34–36.
- **Evidence:** it says `compare_reports` "refuses reports of another split or metric identity". It also refuses different revisions, prompt identities and sample sets (~L1616–1625). The function's own docstring (~L1604–1608) and decisions.md ~L652–661 are correct.
- **Fix:** reword it to match the code after BUG-02 and BUG-14, e.g. ``:func:`compare_reports` refuses reports of another split or metric identity, another sample set (when both record one), and, unless ``same_records=False``, other model revisions or prompt identities.`` Regenerate `api.md`.

---

### DOC-27: the contract ledger's frozen ipython version is stale

- **Severity:** Low · **Group:** G8
- **Where:** `docs/external-contracts.md` ~L23.
- **Evidence:** the ledger says "`ipython>=8.0` / `9.18.1`", but `uv.lock` ~L1486–1487 locks `9.19.1` (Dependabot 88fc6f2). `tests/test_docs_projection.py` ~L318–345 checks only the tool rows. Every other pair matches the lock.
- **Fix:**
  1. Replace `9.18.1` with `9.19.1`.
  2. Optional, weigh it first: a test comparing each "`pkg>=x` / `ver`" pair with `uv.lock` would force a ledger edit on every catch-all Dependabot PR. The alternative is to write "see uv.lock" for non-tool rows.
- **Check:** `grep -n '^name = "ipython"' -A1 uv.lock` and `grep -n ipython docs/external-contracts.md` agree.

---

### DOC-28: the `embeddings` extra pulls in sentence-transformers, which no code imports, and the docs overstate why

- **Severity:** Low · **Group:** G8
- **Where:** `pyproject.toml` ~L173–180 (the `embeddings` extra); `docs/embeddings.md` ~L23; `src/nnx/embeddings/__init__.py` ~L43–46.
- **Evidence:**
  - No file in `src/`, `examples/` or `scripts/` imports `sentence_transformers`; it appears only in docstrings.
  - The docs say "the `ImportError` is deferred to the call site that actually needs each one", but no call site needs it. Only `export_to_faiss` raises (`faiss_export.py` ~L40–45).
  - The package drags in `transformers`, which is heavy.
- **Fix:**
  1. Replace the sentence at `docs/embeddings.md` ~L23 with: `Pulls in \`faiss-cpu\` (used by \`export_to_faiss\`) and \`sentence-transformers\` (a convenience backbone source: NNx never imports it, and any \`nn.Module\` that maps \`list[str]\` to a tensor works). \`import nnx.embeddings\` works without either; \`export_to_faiss\` raises an \`ImportError\` naming the extra when \`faiss\` is missing.`
  2. Optional, in a minor release: drop `sentence-transformers` from the extra, run `uv lock`, and add a CHANGELOG entry.
- **Check:** `grep -rn "import sentence_transformers\|from sentence_transformers" src examples scripts` stays empty.

---

### DOC-29: CONTRIBUTING says CairoSVG is pinned only in the `docs-publish` group

- **Severity:** Low · **Group:** G8
- **Where:** `CONTRIBUTING.md` ~L17.
- **Evidence:** CairoSVG is also pinned in the `docs` extra (`pyproject.toml` ~L203–204), and that pin is what `uv sync --all-extras` installs.
- **Fix:** replace the sentence with "CairoSVG itself is pinned (`cairosvg==2.9.1`) in the `docs` extra and the `docs-publish` dependency group."

---

### DOC-30: the `tests/test_imports.py` docstring claims an exhaustive module list that covers about half the package

- **Severity:** Low · **Group:** G8
- **Where:** `tests/test_imports.py` ~L1–7.
- **Evidence:** it says "every module in nnx imports" and "The full list below is intentionally exhaustive". 82 of 152 public modules are not listed (e.g. `nnx.calibration`, `nnx.abstention`, `nnx.search`, `nnx.surgery.widen`).
- **Fix:** pick one.
  - **(a)** Replace ~L1–7 with "Smoke tests: each subpackage and a sample of modules import without error. Modules not listed here are imported by their own test modules."
  - **(b)** Make the test truly exhaustive by walking `pkgutil.walk_packages(nnx.__path__, "nnx.")` and skipping modules whose optional extra is missing.

---

### DOC-31: stale or wrong comments and docstrings in the feature packages

- **Severity:** Low · **Group:** G10
- **Where and fix:**
  - `src/nnx/quantize/qat.py` ~L55–57 says "Keeping this as a small dict (rather than an enum)", but `_SUPPORTED_CONFIGS` is a tuple. Change "dict" to "tuple".
  - `src/nnx/peft/_targets.py` ~L95–97: the docstring says the function raises only for empty patterns or aliasing. It also raises for weight-reading parents (~L144) and `refuse_inside` (~L151). Add both to the Raises text.
  - `src/nnx/viz/weight_histogram.py:67`, `activation.py:101`, `gradient_flow.py:51`, `summary.py:80` and `netron.py:72` each say "Local import to avoid a circular import at package init time". No module under `src/nnx/nn` imports `nnx.viz`, and `vis_utils.py:35` imports `NNModel` at top level without trouble.
    1. Move each `NNModel` import to module top.
    2. Delete the comments.
    3. Run `uv run pytest tests/test_imports.py tests/test_viz*.py`.
  - `src/nnx/diffusion/training.py` ~L68–70 says the classification metrics "stay zero". Fix it with BUG-20.
  - `src/nnx/interop/gguf/writer.py` ~L97 ("local import — … avoid top-level cycles in tests") and ~L229 ("…so import nnx.interop is cheap") are false, because `import nnx.interop` runs `nnx/__init__`, which imports torch. Delete both comments and hoist `import torch` and `from .tensor_name_map import map_tensors` to module top.
  - `src/nnx/interop/ollama.py` ~L164–170: a stale comment about `Path.write_text` cites an internal memo name (`feedback_utf8_explicit_text_opens`), while the code calls `_atomic_write_text`. Replace it with one line: `# written atomically, UTF-8, through _atomic_write_text`.

---

### DOC-32: a stale comment in `_step_helpers.finalize_step` contradicts `default_train_step`

- **Severity:** Low · **Group:** G10
- **Where:** `src/nnx/_step_helpers.py` ~L296–302.
- **Evidence:** the comment says "`default_train_step` checks AFTER for legacy reasons (its tests pin the existing post-step ordering)". But `_finite_training_loss_value` (`nn_model.py` ~L1632–1643, "Raise before backward/step") runs inside `_record_step_loss` (~L3236), *before* `backward()` (~L3254).
- **Fix:**
  1. Replace the comment with "Checked before backward, as default_train_step does".
  2. Optional: extract `_finite_loss(loss, what, hint)` into `_step_helpers` and call it from both places; the two `FloatingPointError` copies differ only in their hint text.

---

### DOC-33: a `_pickling.py` comment justifies itself with Python 3.10, which the project does not support

- **Severity:** Low · **Group:** G10
- **Where:** `src/nnx/nn/params/_pickling.py` ~L15–17. A second, unrelated module of the same name is `src/nnx/nn/net/_pickling.py`.
- **Evidence:** the comment says "on Python 3.10, dataclass(slots=True) replaces pickling hooks defined in the class body". `requires-python` is `>=3.11`. The review checked 3.11.18, 3.12.3 and 3.13.18: class-body hooks of a `frozen=True, slots=True` dataclass are kept.
- **Fix:**
  1. Change the comment to "early CPython releases replaced class-body hooks; attaching after the class exists works on every version". Keep the code as is.
  2. Optional: rename the two modules to `_param_pickling.py` and `_net_pickling.py`. Pickles do not reference these modules, but grep for imports first.

---

### DOC-34: comments on version pins name the wrong source of truth

- **Severity:** Low · **Group:** G10
- **Where:** `.pre-commit-config.yaml` ~L6–8; `pyproject.toml` ~L129.
- **Evidence:**
  - pre-commit says the "ruff rev should track the pin in pyproject.toml's [dev] extras", but `[dev]` has a range (`"ruff>=0.6,<0.18"`). What is enforced is the uv.lock version (`tests/test_pypi_publication.py` ~L136–147).
  - `pyproject.toml` ~L129 says `"pyright>=1.1.380", # … pin here so locals match`, but that is a floor, not a pin.
- **Fix:**
  1. Replace the pre-commit comment with `  # ruff rev must equal the ruff version locked in uv.lock (enforced by\n  # tests/test_pypi_publication.py::test_pre_commit_ruff_version_matches_frozen_ci_version);\n  # otherwise pre-commit lints under a different rule set than CI.`
  2. Change the pyproject comment to `# floor only; uv.lock pins the version CI runs`.
- **Check:** `uv lock --check && uv run pytest tests/test_pypi_publication.py`.

---

### DOC-35: the `nnx/__init__.py` version comment contradicts release-please

- **Severity:** Low · **Group:** G10 (do it together with DEAD-04)
- **Where:** `src/nnx/__init__.py` ~L28; `pyproject.toml` ~L20.
- **Evidence:** `__init__.py` says "Keep in sync with pyproject.toml … on every bump". pyproject says "Managed by release-please (CONTRIBUTING §7); never bump it by hand".
- **Fix:** reword it to "updated by release-please (x-release-please-version); never edit by hand".
## 7. Part D: out-of-sync examples, tests and config (`SYNC-*`)

The review checked these and found them clean, so do not "fix" them:
- All 57 example files are listed in `examples/README.md`. Numbering 01–26 has no gaps.
- No example uses a removed or renamed API or keyword argument.
- Version pins agree across `pyproject.toml`, `uv.lock`, `requirements-tools.txt` and pre-commit, and `uv lock --check` passes.
- Python 3.11–3.14 is stated consistently everywhere.
- No workflow references a missing path.

---

### SYNC-01: `examples/run_bundle.py` calls `fit_temperature` without `test_split_id`

- **Severity:** Medium · **Group:** G6
- **Where:** `examples/run_bundle.py` ~L130–136 (function `built_in_classifier`).
- **Problem:** it emits the new `DeprecationWarning` today and will raise once the argument becomes required in the next minor release. It is the **only** offending call in `examples/`, `src/` and the docs.
  ```python
  fit = fit_temperature(
      first.predict(X_CAL).logits, Y_CAL.numpy(),
      labels=LABELS, model_id=model_fingerprint(first), split_id="calibration",
  )
  ```
- **Fix:** add `train_split_id="train", test_split_id="test",` after `split_id="calibration",`, as `examples/calibration_offline.py` ~L59–67 does.
- **Check:** `uv run python -W error::DeprecationWarning examples/run_bundle.py` succeeds.

---

### SYNC-02: the example smoke tests ignore NNx's own `DeprecationWarning`, which is how SYNC-01 slipped through

- **Severity:** Medium · **Group:** G6
- **Where:** `tests/test_examples_smoke.py` ~L144–150 (the subprocess env) and ~L237–256 (`test_bounded_example_helpers_execute`).
- **Problem:** `_validation.py` ~L257–262 warns with `stacklevel=3`, so the warning is attributed to the example's own module (`__main__` in the subprocess, `__nnx_example_smoke__` under `runpy`). Neither test turns that into an error, and `pyproject.toml` ~L258–273 has no such filter.
- **Fix:**
  1. After `env["NNX_TQDM_DISABLE"] = "1"` (~L146), add `env["PYTHONWARNINGS"] = "error::DeprecationWarning:__main__"  # an example must not call a deprecated API`.
  2. Decorate `test_bounded_example_helpers_execute` with `@pytest.mark.filterwarnings("error::DeprecationWarning:__nnx_example_smoke__")`.
  3. If a torch `DeprecationWarning` attributed to example code also trips, narrow the filter to the message `.*without test_split_id is deprecated`.
- **Check:** with SYNC-01 unfixed, `uv run pytest tests/test_examples_smoke.py -k run_bundle` fails. With SYNC-01 fixed, it passes.

---

### SYNC-03: 39 test calls use the deprecated form and will break when `test_split_id` becomes required

- **Severity:** Medium · **Group:** G6
- **Where:**
  - `tests/test_abstention.py` (`select_threshold`):
    - unguarded calls at ~L296, 304, 347, 450, 498, 562, 594, 721, 750, 757, 773, 816, 823, 849, 853, 890, 898, 903, 950, 959, 1013, 1065, 1106, 1363
    - calls inside `pytest.raises` at ~L321, 323, 362, 449, 654, 714, 858, 1050, 1401
  - `tests/test_calibration.py` ~L173, 214, 882, 1025 (`fit_temperature`)
  - `tests/test_decision_benchmark.py` ~L1035
  - `tests/test_decision_jobs.py` ~L1089
- **Problem:** only `test_abstention.py` ~L488 and `test_calibration.py` ~L994 omit the argument on purpose, inside `pytest.warns`.
- **Fix:**
  1. Add `test_split_id="test"` to every call listed above. If a call already uses `"test"` for `split_id` or `train_split_id`, use `"holdout"` instead.
  2. Add `"error:.*without test_split_id is deprecated:DeprecationWarning",` to `[tool.pytest.ini_options].filterwarnings` in `pyproject.toml`. The two intentional `pytest.warns` tests still pass.
- **Check:** `uv run pytest tests/test_abstention.py tests/test_calibration.py tests/test_decision_benchmark.py tests/test_decision_jobs.py` passes with the new filter.

---

### SYNC-04: nine numbered examples are only imported by the tests, never run

- **Severity:** Medium · **Group:** G6
- **Where:** `tests/test_examples_smoke.py`: the end-to-end list (~L113–143) and `BOUNDED_EXAMPLE_HELPERS` (~L169–232); `examples/README.md` ~L37–39.
- **Problem:** these are import-only, so a runtime break in them is invisible: `06_finetune_with_layer_freezing.py`, `13_train_domain_embedder.py`, `14_moe_classifier.py`, `17_export_transformer_to_gguf.py`, `18_export_ollama_bundle.py`, `19_prune_synthetic_classifier.py`, `21_viz_attribute_xai.py`, `23_born_again_distillation.py` and `24_feature_kd.py`. The README says "Examples that are only imported are references" but never says which ones. SYNC-05 is a bug hidden by this.
- **Fix:**
  1. Append the core-only ones to the end-to-end list (~L115–142): `"06_finetune_with_layer_freezing.py", "14_moe_classifier.py", "19_prune_synthetic_classifier.py", "23_born_again_distillation.py", "24_feature_kd.py"`.
  2. Optionally also add `"13_…", "17_…", "18_…", "21_…"`. These skip when their extras are missing.
  3. If you do not add them, add after `examples/README.md` ~L39: "Import-only (no test executes them): 06, 13, 14, 17, 18, 19, 21, 23, 24."
- **Check:** `uv run pytest "tests/test_examples_smoke.py::test_representative_examples_run_end_to_end" -k "06_ or 14_ or 19_ or 23_ or 24_"`, each inside the 90 s timeout.

---

### SYNC-05: example 13 combines torch and FAISS without the macOS OpenMP guard the docs require

- **Severity:** Low · **Group:** G6
- **Where:** `examples/13_train_domain_embedder.py` ~L34–46 and ~L159–164.
- **Problem:**
  - The example imports torch (~L38), then calls `faiss.read_index` and `idx.search`, with no `KMP_DUPLICATE_LIB_OK` / `OMP_NUM_THREADS` setup.
  - `docs/embeddings.md` ~L25 says such a process can abort ("OMP: Error #15") and that you should set both variables first. `examples/ranking_offline.py` ~L48–54 does this.
  - Not reproduced here; the crash is macOS-only.
- **Fix:** add `import sys` next to `import os`, then insert before `import torch`:
  ```python
  # macOS: the faiss-cpu and torch wheels each bundle an OpenMP runtime, and
  # a process that loads both aborts ("OMP: Error #15") or crashes in the
  # FAISS search. Set before either library loads. Linux needs neither.
  if sys.platform == "darwin":
      os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
      os.environ.setdefault("OMP_NUM_THREADS", "1")
  ```
- **Check:** `uv run ruff check examples/13_train_domain_embedder.py`.

---

### SYNC-06: three tests skip without sentence-transformers, which they never use

- **Severity:** Low · **Group:** G6
- **Where:** `tests/test_paradigm_finite_settings.py` ~L58, ~L163, ~L199.
- **Problem:** each calls `pytest.importorskip("sentence_transformers")`, but the tests use `object()` or a local `Embedder` module. In the floor-dependency lane (no embeddings extra) they skip silently.
- **Fix:** delete the three `importorskip` lines.
- **Check:** in an env without sentence-transformers, `uv run pytest tests/test_paradigm_finite_settings.py -k text_contrastive -rs` shows passes, not skips.

---

### SYNC-07: tests `importorskip` the core dependencies numpy and torch

- **Severity:** Low · **Group:** G6
- **Where:** `tests/test_experiment_plan.py` ~L755; `tests/test_named_monitors.py` ~L1011; `tests/test_params_boundary_validation.py` ~L724.
- **Problem:** numpy and torch are core dependencies (`pyproject.toml` ~L99–107). The repo's own rule (`tests/test_examples_smoke.py` ~L48–53) is that a missing core dependency "fails, never skips".
- **Fix:**
  - `test_experiment_plan.py`: delete ~L755. numpy is already imported at ~L11.
  - `test_named_monitors.py`: replace ~L1011 with `import numpy as np`.
  - `test_params_boundary_validation.py`: replace ~L724 with `import torch`.
- **Check:** `uv run ruff check tests && uv run pytest tests/test_experiment_plan.py tests/test_named_monitors.py tests/test_params_boundary_validation.py`.

---

## 8. Part E: out-of-scope code and files (`SCOPE-*`)

These need a decision by the maintainer before anyone implements them. **An implementing agent should not do SCOPE-01 or SCOPE-03 without explicit approval**, because they move public API. SCOPE-02 changes CI and also needs approval.

---

### SCOPE-01: `nnx.decisions` is a separate LLM-decision and evaluation product bolted onto a PyTorch training library

- **Severity:** High (architecture) · **Group:** G11
- **Where:**
  - `src/nnx/decisions/`: 4,899 lines, about 7% of `src/nnx` (benchmark.py 1643, jobs.py 1149, jev.py 544, providers.py 496, schema.py 490, nli.py 449, `__init__` 128), plus about 230 tests in `tests/test_decision_*.py`
  - hooks into core: `src/nnx/__init__.py:43` (eager import), `src/nnx/result.py` ~L221–305, `src/nnx/abstention.py` ~L1451–1495 (`decide`), and the `jev` extra in `pyproject.toml`
- **Evidence:**
  - The package docstring says "inference only — nothing here trains, exports or executes actions."
  - `JevProvider` is a single-vendor client for a hosted LLM service: `TYPESAFE_API_KEY`, `base_url`, retries, per-event-loop async clients.
  - `jobs.py` is an async orchestrator: concurrency limits, cancellation, token budgets.
  - `benchmark.py` is an LLM evaluation harness: prompt perturbations, call budgets, bootstrap intervals.
  - Only `FixedHeadProvider` (`providers.py` ~L244–496) wraps an NNx model.
  - Nothing under `nn/`, `trainer/` or any training path imports `decisions`.
  - Every `import nnx` pays for ~5k lines, and the package forces the import cycles in SMELL-42.
- **Fix:**
  1. **Short term (safe):**
     - Remove `decisions` from the eager `from . import (...)` list in `src/nnx/__init__.py` (~L34–69) and serve it through the existing lazy `__getattr__` (~L288), adding a `"decisions"` entry.
     - Check `tests/test_imports.py`, `tests/test_public_api_exports.py` and `nnx.__all__`.
  2. **Medium:** move `abstention.decide` into `nnx/decisions/selective.py`, re-exported from `abstention` through a lazy wrapper. This also removes the cycle (SMELL-42).
  3. **Long term (maintainer decision):**
     - Extract `nnx.decisions` plus the decision boundaries in `result.py` into a separate distribution that depends on `thekaveh-nnx`.
     - Keep a ~50-line `FixedHeadProvider` bridge, or an `nnx.decisions` shim that raises `ImportError` naming the new package.
     - Move `docs/decisions.md`, `docs/decision-model-pilot.md` and the `examples/decision_*.py` files with it, and drop the `jev` extra.
- **Check:** `uv run pytest tests/test_decision_*.py tests/test_result_boundaries.py tests/test_abstention.py tests/test_imports.py tests/test_public_api_exports.py`.

---

### SCOPE-02: `mps.yml` and the GPU lock script encode one maintainer's workstation

- **Severity:** Medium · **Group:** G7
- **Where:**
  - `.github/workflows/mps.yml` ~L22–84, ~L100, ~L119, ~L133
  - `scripts/with_mac_gpu_lock.py`, `tests/test_with_mac_gpu_lock.py`
  - `CONTRIBUTING.md` ~L178–182
  - `docs/external-contracts.md` ~L66 ("ran green on the Mac Studio")
- **Evidence:** the workflow hard-codes `http://127.0.0.1:8188/queue` (ComfyUI), `http://127.0.0.1:11434/api/ps` (Ollama) and `~/.cache/mac-gpu.lock`. These are local services and conventions of one machine. The MPS test lane itself is legitimate; the GPU-sharing coordination is not library CI. No secrets are involved.
- **Fix:**
  1. Delete the `gpu-check` job (~L22–84). In `mps-evidence`, remove `needs: gpu-check` and `&& needs.gpu-check.outputs.busy == 'false'`.
  2. In both GPU steps, replace `python3 scripts/with_mac_gpu_lock.py -- uv run` with `$NNX_MPS_GPU_WRAPPER uv run`, and add `env: NNX_MPS_GPU_WRAPPER: ${{ vars.NNX_MPS_GPU_WRAPPER }}` (merge it into an existing `env:` block). An empty value runs the command directly.
  3. Delete `scripts/with_mac_gpu_lock.py` and `tests/test_with_mac_gpu_lock.py`. The lock and service probe move to a script on the runner machine, which the maintainer owns.
  4. In `tests/test_ci_runner_image.py`, drop the `("mps.yml", "gpu-check")` entries (~L28 and ~L33) and change ~L96 to `{"mps-evidence", "mps-gate-without-mps"}`.
  5. Replace `CONTRIBUTING.md` ~L178–182 with: "To share the Mac's GPU with other work, set the repository variable `NNX_MPS_GPU_WRAPPER` to a command on the runner machine that serialises GPU use; each GPU step runs under it."
  6. This makes CI-02 to CI-06 and DOC-07 unnecessary. Still do CI-01.
- **Check:** `uv run pytest tests/test_ci_runner_image.py`. `grep -rn "8188\|11434\|mac-gpu" .github scripts tests CONTRIBUTING.md` returns nothing.

---

### SCOPE-03: `export_conformance.py` is a 1,362-line CI harness shipped in the runtime package and imported by every `import nnx`

- **Severity:** Medium · **Group:** G11
- **Where:** `src/nnx/export_conformance.py`. Its `validate_record` (~L1105–1348) is about 231 lines with cyclomatic complexity 172. The profiles are FeedFwd-only (~L156–187).
- **Evidence:**
  - The docstring says the profiles are "the only targets NNx claims executed parity for: FeedFwdNN 4-8-2".
  - Its only consumers are `scripts/check_export_conformance.py`, `.github/workflows/ci.yml` and two examples.
  - The inline sha256-hex checks (~L1281, ~L1304) duplicate `bundles._SHA256` (~L110) and `nn_model._SHA256_HEX` (~L191).
- **Fix:**
  1. Drop `export_conformance` from the eager import list in `nnx/__init__.py`. It stays importable as `nnx.export_conformance`.
  2. Split `validate_record` along its existing blank-line sections into `_record_keys_problems`, `_stage_problems`, `_case_problems`, `_artifact_problems` and `_config_problems`, each taking `(record, problems)`.
  3. Add `_validation.is_sha256_hex(value)` and use it in all three places.
  4. Maintainer decision: in a later major release, move the module to `tools/` or the test tree.
- **Check:** `uv run pytest tests/test_export_conformance*.py` and `uv run python scripts/check_export_conformance.py --help`.

---

### SCOPE-04: the Ollama export targets an architecture that stock Ollama cannot load

- **Severity:** Low · **Group:** G11
- **Where:** `src/nnx/interop/ollama.py`; `docs/gguf.md` ~L9–12; the `"ollama"` keyword in `pyproject.toml`.
- **Evidence:**
  - The docstring and docs say stock Ollama, llama.cpp and LM Studio cannot load `nnx_transformer`.
  - The parameters are pinned to an "Ollama v0.32.2 tag snapshot contract".
  - `export_ollama_modelfile` only writes files; it never contacts a service.
- **Fix (maintainer decision):** either keep it and say clearly in the README feature list that a custom runtime is required, or deprecate `export_ollama_modelfile` and the `"ollama"` PyPI keyword until some runtime supports the architecture. The code smells in this module are SMELL-45.

---

## 9. Part F: dead code and clutter (`DEAD-*`, `CLUT-*`)

**All `DEAD-*` and `CLUT-*` items are Low severity.** Before deleting anything, re-run the grep in each item over `src tests examples scripts docs`. The review ran these greps at 9ad461e.

---

### DEAD-01: the `exact=` parameter of `_read_idps_csv` is never `False`

- **Where:** `src/nnx/nn/params/nn_run.py` ~L467 (signature) and ~L478 (`... if exact else pd.read_csv(csv_path)`).
- **Evidence:** `grep -rn "_read_idps_csv" src tests` finds calls at `nn_run.py:459` (no argument), `history.py:1457` (`exact=True`) and `tests/test_params_boundary_validation.py:307` (no argument).
- **Fix:**
  1. Drop the parameter, the conditional and the `exact=False` sentence in the docstring.
  2. Change `history.py:1457` to `_read_idps_csv(csv_path)`.
- **Check:** `uv run pytest tests/test_run_load_exact_history.py tests/test_params_boundary_validation.py tests/test_history_journal.py`.

### DEAD-02: `_DDPSession.validation_ids` is written but never read

- **Where:** `src/nnx/distributed.py:585` (`self.validation_ids: list[int] = []`) and `src/nnx/nn/nn_model.py:2519` (`ddp.validation_ids = ids`).
- **Evidence:** the tests read `validation.ids()` on the loader under a dict key named `"validation_ids"` (`tests/ddp_scenarios.py:545`, `test_ddp_adapter.py:191-193`), never the session attribute.
- **Fix:** delete both lines.
- **Check:** `uv run pytest tests/test_ddp_adapter.py`.

### DEAD-03: `nn_run._mean_field` is used only by a test

- **Where:** `src/nnx/nn/params/nn_run.py` ~L85–89; `tests/test_graph_classification.py` ~L284–292.
- **Fix:** rewrite the test to call `_field_total(records, name)` and divide, or move the helper into the test file. Then delete `_mean_field`.
- **Check:** `uv run pytest tests/test_graph_classification.py`.

### DEAD-04: an unreachable Python < 3.8 shim in `nnx/__init__.py`

- **Where:** `src/nnx/__init__.py` ~L18–32.
- **Evidence:** `except ImportError:  # pragma: no cover — Python <3.8.` cannot run with `requires-python >= 3.11`. It holds a second hard-coded `"0.5.0"` that release-please must keep updating.
- **Fix:**
  1. Remove the outer `try/except ImportError`. Keep `try: __version__ = _version("thekaveh-nnx") except _PackageNotFoundError: __version__ = "0.5.0"  # x-release-please-version`.
  2. Make sure exactly one `x-release-please-version` marker line remains. Check `release-please-config.json` for the files and markers it updates.
  3. Do DOC-35 in the same commit.
- **Check:** `uv run pytest tests/test_imports.py tests/test_release_finalize.py`.

### DEAD-05: compatibility guards for torch versions below the `torch>=2.4` floor

- **Where:**
  - `hasattr(torch.backends, "mps")` at `nn/enum/devices.py:29`, `bundles.py:1414` and `precision.py:374`
  - `hasattr(torch, "float8_e4m3fn")` at `bundles.py` ~L656–658
- **Evidence:** `torch.backends.mps` exists from torch 1.12, and both float8 dtypes exist in torch 2.4.
- **Fix:**
  1. Simplify the MPS checks to `torch.backends.mps.is_available()`.
  2. Put the float8 entries directly into `_CODES`.
  3. Leave the `getattr(nn.modules.conv, "_LazyConvXdMixin", None)` guards in `models.py` (~L411–456): they protect against *future* removal of private torch names. Add a one-line comment saying so.
- **Check:** `uv run pytest tests/test_mps_cpu_pins.py tests/test_run_bundles.py tests/test_external_module_runtime.py`.

### DEAD-06: the `ObjectiveFn` type alias is unused

- **Where:** `src/nnx/objectives.py:242`.
- **Evidence:** `grep -rn ObjectiveFn src tests examples docs` finds only the definition, and it is not in `__all__`.
- **Fix:** delete the line.

### DEAD-07: `PrefixTuner._original_attn_forwards` is written but never read

- **Where:** `src/nnx/peft/prefix.py:201` and `:205`.
- **Fix:** delete both lines. Before deleting, grep tests for `_original_attn_forwards`; the review found nothing.

### DEAD-08: a useless alias `selected = matched`

- **Where:** `src/nnx/peft/_targets.py:158`.
- **Fix:** use `matched` directly and delete the alias.

### DEAD-09: `try/except ImportError` around imports that cannot fail

- **Where:**
  - `src/nnx/decisions/benchmark.py` ~L614–620: `try: import torch ... except ImportError`. torch is a core dependency, and the module already imports `trainer.params`, which imports torch.
  - `src/nnx/surgery/deepen.py` ~L408–421: a same-package `Activations` import inside `try/except ImportError: pass`.
- **Fix:** replace each with a plain import at module top.
- **Check:** `uv run pytest tests/test_decision_benchmark.py tests/test_surgery*.py tests/test_imports.py`.

### DEAD-10: `bundles._save_tensors` has no caller in `src`

- **Where:** `src/nnx/bundles.py:715`, used only by `tests/test_run_bundles.py:457` and `tests/test_bundle_validation.py:969`.
- **Fix:** if BUG-21 is done, it gains a real caller and nothing needs to change. Otherwise add a comment `# used by tests`, or move it into a test helper.

### DEAD-11: `build_wiki._list_indent_after` survives only for one test

- **Where:** `scripts/docs/build_wiki.py` ~L787–791; `tests/test_wiki_projection.py` ~L833–834.
- **Evidence:** this range moved every caller to `_list_items_after`.
- **Fix:**
  1. Delete the function.
  2. Change the test body to `assert build_wiki._list_items_after("- Guides\n    - See [qs].\n", ())[-1] == (0, 6)`. The review confirmed this returns `((0, 2), (0, 6))`.
- **Check:** `uv run pytest tests/test_wiki_projection.py -k list_indent`.

### DEAD-12: the test helper `_only_run_dir` has never been used

- **Where:** `tests/test_compile_execution.py` ~L121–123 (added in 83c7c3e and never called).
- **Fix:** delete it.
- **Check:** `uv run ruff check tests/test_compile_execution.py`.

### DEAD-13: tombstone asserts for the removed `pip-audit-py310-ignores.txt`

- **Where:** `tests/test_pypi_publication.py` ~L113–114.
- **Fix (optional):** delete the two asserts, and keep `assert "--ignore-vuln" not in workflow`.

### DEAD-14: 11 `# noqa: BLE001` markers do nothing

- **Where:** `nn_model.py:655, 1506, 1518, 2644`; `nn_run.py:682`; `history.py:947, 1195, 1339`; `components.py:426`; `provenance.py:783`; `decisions/jev.py:229`.
- **Evidence:** `[tool.ruff.lint] select = ["E","F","W","B","I","UP"]` does not enable BLE, so these markers suppress nothing. Other broad `except Exception` clauses carry no marker, so the markers are not applied consistently either.
- **Fix:** pick one.
  - Delete all 11 markers.
  - **Or** add `"BLE"` to ruff `select`, run `uv run ruff check src`, and add a marker plus a one-line reason at every broad except that is deliberate. This is a bigger change.

---

### CLUT-01: the `.maintenance/` ledgers are stale agent run logs that nothing references

- **Where:** `.maintenance/overnight-2026-07-01.md` and `.maintenance/overnight-2026-07-22.md`.
- **Evidence:**
  - They are not shipped in the sdist.
  - Nothing references them except CHANGELOG prose.
  - The content is stale: it names a branch `codex/overnight-maintenance`, cites `mkdocs.yml` as a tracked file (it is now generated), says Dependabot covers "pip" (it is now `uv`), and calls pyright "advisory" (`ci.yml:82` says it gates).
- **Fix (maintainer decision):** `git rm -r .maintenance`, or move the files to the wiki or an issue. Do not add new reports there.

### CLUT-02: `tests/test_pass2_*_series.py` are named after an audit pass, not what they test

- **Where:** `tests/test_pass2_{d,f,n,ou,r,v}_series.py` (~4,077 lines, 163 tests); `CONTRIBUTING.md` ~L62 recommends them as "good models".
- **Fix:**
  1. `git mv` them:
     - `d` → `test_utils_function_aliases.py`
     - `f` → `test_training_features.py`
     - `n` → `test_evaluate_regressions.py`
     - `ou` → `test_predict_api_and_ux.py`
     - `r` → `test_training_reliability.py`
     - `v` → `test_seed_reproducibility.py`
  2. Update the references at `docs/concepts.md` ~L65–66, `docs/feature-composition.yaml` ~L53 (then run `uv run python scripts/check_feature_composition.py --render`), `tests/test_nn_dataset.py` ~L86 and `CONTRIBUTING.md` ~L82–83.
  3. Replace the `CONTRIBUTING.md` ~L62 sentence with "Put the test in the module named after the code it covers (for example `tests/test_calibration.py`)."
  4. BUG-07 edits `test_pass2_r_series.py`; do that first or rebase on it.
- **Check:** `uv run python scripts/check_feature_composition.py --check`; `git grep test_pass2_` is empty outside CHANGELOG.

### CLUT-03: the 2.4 MB README banner exceeds the repo's own 500 KB large-file limit

- **Where:** `docs/assets/nnx-poster.png` (2,445,656 bytes, 2304×921) vs `.pre-commit-config.yaml` ~L26–27 (`check-added-large-files --maxkb=500`). The hook checks only newly added files, so updates bypass it. Three revisions are already in history.
- **Fix:** recompress in place, keeping PNG with width ≥ 1200 and aspect ratio ≥ 2 (asserted in `tests/test_docs_projection.py` ~L197–206): `pngquant --quality=70-90 --strip --force --output docs/assets/nnx-poster.png docs/assets/nnx-poster.png`. If it is still over 500 KB, resize to 1600×640.
- **Check:** `test $(stat -c%s docs/assets/nnx-poster.png) -lt 512000 && uv run pytest tests/test_docs_projection.py -k banner`.

### CLUT-04: developer tooling is published as PyPI extras with exact pins

- **Where:** `pyproject.toml` ~L119–137 (`dev`, e.g. `"pip-audit==2.10.1"`) and ~L203–210 (`docs`, e.g. `"cairosvg==2.9.1"`).
- **Evidence:** the repo already uses PEP 735 `[dependency-groups]` for `release` and `docs-publish`. `pip install thekaveh-nnx[dev]` imposes exact tool pins on users.
- **Fix (optional, larger):**
  1. Move `dev` and `docs` into `[dependency-groups]`.
  2. In `ci.yml` ~L224, change `--extra dev` to `--group dev`.
  3. Add `--group docs` wherever `--all-extras` is relied on for mkdocs: `ci.yml` ~L63, `docs.yml` ~L128, `release.yml` ~L86.
  4. Run `uv lock`, then fix tests that assert the old strings (`grep -rn "all-extras\|extra dev" tests`).
- **Check:** `uv lock --check && uv run pytest tests/test_pypi_publication.py tests/test_docs_projection.py`.

### CLUT-05: `pyproject.toml` keyword comments narrate internal audit history

- **Where:** `pyproject.toml` ~L33–60, comments in the `keywords` list such as "Megamerge subsystems (PRs #29–#37) … rather than thekaveh/ml internals", "flagged by the post-#49 overnight-maintenance audit", "Post-#51 overnight-maintenance pass", "FEAT-029".
- **Problem:** these comments record how the list grew, not why a keyword is there. They are noise in a published manifest and refer to a private repo.
- **Fix:**
  1. Replace the whole comment block with one comment: `# One keyword per first-class nnx.* subsystem, so PyPI search finds it.`
  2. Keep the keywords themselves unchanged.
## 10. Part G: code smells, duplication and bloat (`SMELL-*`)

These are refactors. They should not change behaviour. Rules for every SMELL item:
- **Do one item per PR.** Run the listed tests before *and* after the change; both runs must pass.
- **Keep old private names importable** as aliases (`_old = new`) when moving a helper. Many tests monkeypatch module attributes such as `nn_model._scale_gradients`, `nn_checkpoint._full_fsync`, `nn_run._atomic_write_text` and `history._fsync_path`. After a move, a patch on the old module stops intercepting, so retarget those tests (`grep -rn "monkeypatch.setattr(.*<old_module>" tests`).
- **Keep error message text** unless the item says otherwise. Tests match on it: `grep -rn "<message fragment>" tests` before changing it.
- **Pickled class paths must keep resolving.** Never move or rename `NNModel`, `_CallbackContext` or any class that appears in a `__reduce__`.

Items SMELL-01 to SMELL-33 cover the `nn/` core and the top-level core modules. SMELL-34 to SMELL-61 cover the feature packages.

---

### SMELL-01: about 10 hand-rolled copies of atomic-write / fsync / partial-file logic, homed in the wrong module

- **Severity:** High (it caused BUG-10 and BUG-11) · **Group:** G5 (do it *after* the G5 bug fixes)
- **Where (every copy):**
  - `nn/params/nn_run.py`:
    - `_atomic_write_text` ~L210–276
    - `_reserve_partial`, `_fsync_file`, `_commit_partial`, `_discard_partial` ~L290–322
    - `_quietly` ~L325
    - `_CREATED_FILE_MODE` ~L279–287
  - `nn/params/nn_checkpoint.py`:
    - `_UNSUPPORTED_FLUSH`, `_full_fsync`, `_fsync_directory` ~L206–245
    - `_atomic_torch_save` ~L294
    - `_write_manifest` ~L361–371
    - `_to_safetensors_file` ~L673–682
    - name reservations in `save()` ~L765–768 and ~L831–835
  - `history.py`: `_fsync_path` ~L299–317 (the same job as `_fsync_file`, done differently) and `_write_chunk_file` ~L363–377
  - `nn/nn_model.py` `_save_pretrained` ~L4752–4762: a fixed-name `model.safetensors.partial`
  - `nn/params/nn_tokenizer_params.py` ~L77–82 (BUG-26)
  - `_artifacts.py` ~L64–67: `atomic_write`, a second name for `_atomic_write_text`
  - `bundles.py` ~L1136–1160 and ~L1386–1400
  - `interop/gguf/writer.py` ~L263
- **Problem:**
  - `nn_run` imports `history` at top level, so every `history → nn_run` import has to be function-local. That gives 16 local imports in `history.py`, 4 in `provenance.py`, and more in `_artifacts.py`, `data_splits.py` ~L323, `preprocessing.py` ~L469 and `interop/ollama.py` ~L171.
  - Callers are split between `atomic_write(path)` and `_atomic_write_text(os.fspath(path))`.
  - `_artifacts`' own docstring admits the inverted dependency: "the atomic writer is imported from nnx.nn.params.nn_run when saving".
- **Fix:**
  1. Create `src/nnx/_atomic_io.py`. It must be standard-library only, with `fcntl` behind its Windows fallback. Move into it:
     - `UNSUPPORTED_FLUSH`, `full_fsync`, `fsync_directory`
     - `fsync_path` (history's version, which opens `O_RDWR`)
     - `quietly`, `CREATED_FILE_MODE` (or its BUG-12 replacement)
     - `atomic_write_text(path, text, *, mode=None)` taking `str | PathLike`
     - `atomic_write_bytes(path, data, *, durable=True)`
     - `reserve_partial`, `commit_partial`, `discard_partial`
     - a `@contextmanager partial_file(path)` that yields the partial path, commits on success and discards on error.
  2. Rewrite the copies on top of these helpers:
     - `history._write_chunk_file` → `atomic_write_bytes(path, data, durable=False)`
     - `nn_checkpoint._write_manifest` → `atomic_write_text(path, json.dumps(...))`
     - `_to_safetensors_file`, `NNModel._save_pretrained`, `to_onnx`, `export_state_dict`, the gguf writer and `NNTokenizerParams.save` → `with partial_file(path) as tmp:`
  3. Keep the old names as re-exports in `nn_run` (`_atomic_write_text`, `_quietly`, `_CREATED_FILE_MODE`, `_reserve_partial`, `_commit_partial`, `_discard_partial`) and in `nn_checkpoint` (`_full_fsync`, `_fsync_directory`, `_UNSUPPORTED_FLUSH`).
  4. In `history.py`, `provenance.py`, `_artifacts.py`, `data_splits.py`, `preprocessing.py`, `generative_nn_model.py` and `interop/ollama.py`, import from `._atomic_io` at module top and delete the local imports.
  5. Make `_artifacts.atomic_write` a plain alias of `atomic_write_text`.
  6. Inside `_atomic_io`, call `full_fsync` through the module (`_atomic_io.full_fsync(...)`) so tests can patch it in one place, then retarget the tests that patch the old locations:
     - `nn_checkpoint._full_fsync` (6 tests)
     - `nn_run._full_fsync` (`test_hub_mixin.py:851`)
     - `nn_run._atomic_write_text` (3 tests)
     - `history._fsync_path`, `history._write_chunk_file`, `bundles._fsync_directory`
- **Check:** `uv run pytest tests/test_atomic_write_text.py tests/test_run_file_durability.py tests/test_resume_point_durability.py tests/test_resume_point_hashing.py tests/test_history_journal.py tests/test_hub_export_atomic.py tests/test_hub_mixin.py tests/test_hub_manifest.py tests/test_to_onnx_dropped_state.py tests/test_to_onnx_inputs.py tests/test_interop_gguf_writer.py tests/test_checkpoint_safetensors.py tests/test_tokenizer_params.py tests/test_run_bundles.py tests/test_provenance.py`. Use whichever of these files exist (`ls tests | grep -E "atomic|durab|hub|onnx|gguf|bundle|provenance|tokenizer"`).
- **Risk:** Medium. Each caller's file mode must stay exactly as it is today (see SMELL-31).

### SMELL-02: `nn_model.py` is a 6,826-line god module

- **Severity:** High · **Group:** G10 (large; one sub-step per PR)
- **Where:** `src/nnx/nn/nn_model.py`. It has 120 top-level functions, 17 classes and 36 constants; `class NNModel` alone spans ~L3839–6771 (2,932 lines). Other modules import 49 distinct private names from it; `trainer/trainer.py` ~L59–84 imports 24 of them.
- **Proposed split:** move each group as-is and leave re-exports in `nn_model` so `from nnx.nn.nn_model import X` keeps working.
  1. `nnx/nn/_resume.py`:
     - ~L203–1071, from `_component_type` to `_resume_fit`, plus `_ResumeWindow`, `_Optimization`, `_SavedCheckpoints` and `_Resumed`
     - `_check_resume_precision` (~L1889–1922)
     - ~L2797–2974, `_RESTORED_SCHEDULER_FIELDS` to `_saved_hyperparameters`
  2. Transform replay (~L1072–1260, `_topology_preflight` to `_refuse_unrecorded_recipe_state`) → `nnx/transforms.py`. See SMELL-11.
  3. `nnx/nn/_hub.py`:
     - ~L1261–1457: `_SAFETENSORS_MAX_HEADER_BYTES`, `_safetensors_shapes`, `_declared_module`, `_refuse_shapes_before_build`, `_file_digest`, `_manifest_*`, `_write_hub_manifest`, `_verify_hub_manifest`
     - the `_HUB_*` constants (~L183–193)
     - The method bodies `save_pretrained`, `_save_pretrained` and `_from_pretrained` (~L4592–4980) become thin delegations.
  4. `nnx/nn/_ddp_fit.py`: ~L2062–2215 and ~L2284–2568, which removes 12 local `from ..distributed import` lines.
  5. The lazy-module descriptor (~L3592–3838) → `nnx/models.py`, beside its lazy-layer helpers (~L320–615), or a new `nnx/_lazy_modules.py`.
- **Check after each step:** the full suite, or at least `uv run pytest tests -k "resume or hub or ddp or trainer or external_module_runtime or model_recipes or train_step_hook"`.
- **Risk:** Medium. Tests monkeypatch `nn_model` globals that functions look up at call time: `_scale_gradients` ×4, `_check_scaler_hook` ×4, `_check_resume_precision`, `_precision_key`, `_objective_engine`, `_evaluate_task`, `_check_finite_gradients`, `resolve_precision`, `NNRun`, `ResumeStatus`. A moved function must move together with every helper it looks up, and those tests must patch the new module.

### SMELL-03: giant functions and complexity hotspots

- **Severity:** Medium · **Group:** G10
- **Where:** line counts from the AST; CC is radon cyclomatic complexity.

  | Function | Location | Lines | CC |
  |---|---|---:|---:|
  | `NNModel._train_impl` | nn_model.py:5318 | 645 | 102 |
  | `Trainer._train_impl` | trainer.py:572 | 414 | 61 |
  | `NNTabularDataset.__post_init__` | nn_tabular_dataset.py:200 | 304 | 70 |
  | `NNModel.train` | :5028 | 266 | 26 |
  | `Trainer.train` | :316 | 255 | 28 |
  | `GenerativeNNModel.generate` | :89 | 207 | 27 |
  | `_resume_fit` | :872 | 198 | 37 |
  | `observations_from_runs` | comparison.py:1510 | 198 | 57 |
  | `to_onnx` | :4221 | 188 | 25 |
  | `_from_pretrained` | :4795 | 185 | 32 |
  | `default_train_step` | :3174 | 172 | 48 |
  | `NNCheckpoint.save` | :684 | 170 | – |
  | `_match_lazy_module` | :3700 | 94 | 28 |
  | `StreamingMetrics.update` | – | – | 29 |
  | `UpdateEngine.commit` | – | – | 26 |
- **Fix (pure extract-method):**
  - `_train_impl`: extract into private methods
    - the resume window and reporter setup (~L5468–5594), including the per-call nested class `_ReportUpdate` (~L5539), which becomes a module-level class taking `clock_report`, `resume_window`, `ctx` and `listeners`
    - the batch loop body (~L5675–5735)
    - the validation dispatch (~L5755–5785)
    - the epoch commit (~L5847–5879)
  - `observations_from_runs`: extract the per-run loop body (~L1613–1706) into `_observation(run_id, given, ...)`.
  - `to_onnx`: see SMELL-29. `_from_pretrained`: see SMELL-12.
- **Check:** `uv run pytest tests -k "train_integration or trainer or exact_resume or resume_parity or ddp or run_comparison"`.
- **Risk:** Low, if done as pure extraction. Do not add closures to the per-batch hot path.

### SMELL-04: the library-wide warning helper lives in `nn/callbacks.py`, which forces 29 function-local imports

- **Severity:** Medium · **Group:** G10
- **Where:**
  - `_warn_at_user_frame` is defined at `src/nnx/nn/callbacks.py:111`. It needs only `sys` and `warnings`.
  - It is imported inside functions 29 times: nn_model.py ×9 (648, 756, 944, 1055, 1500, 1512, 2637, 3581, 5801), history.py ×7, _metrics.py ×4, provenance.py ×2, trainer.py ×2, and once each in `_scheduler_clock`, `_update_engine`, `components`, `monitors` and `nn_run` (:731).
- **Fix:**
  1. Move the function, unchanged, to a new standard-library-only `src/nnx/_warnings.py`.
  2. Re-export it from `callbacks.py`: `from .._warnings import _warn_at_user_frame`.
  3. Replace each local import with a module-top `from .._warnings import _warn_at_user_frame`, adjusting the relative depth.
  4. Do BUG-18's `skip=` parameter in the same place.
- **Check:** `grep -rn "import.*_warn_at_user_frame" src/nnx` shows only module-level lines. Run `uv run pytest tests -k "callbacks or history_journal or provenance or metrics or component_resume or train_integration"`.
- **Risk:** Low. The frame walk skips frames by `__name__`, so it does not depend on where the function lives.

### SMELL-05: "warn, but never let a `-W error` filter replace the propagating error" is copy-pasted six times

- **Severity:** Medium · **Group:** G10
- **Where:** `nn_model.py` ~L647–656, ~L1497–1507, ~L1510–1519, ~L2634–2645; `components.py` ~L418–427; `provenance.py` ~L780–784. Each copy reads `try: _warn_at_user_frame(msg) / except Exception:  # noqa: BLE001 / pass`.
- **Fix:**
  1. Next to `_warn_at_user_frame` (SMELL-04), add:
     ```python
     def _warn_while_failing(message, category=RuntimeWarning):
         try:
             _warn_at_user_frame(message, category)
         except Exception:
             pass
     ```
  2. Replace the six blocks with one call each, keeping every message unchanged.
  3. This also removes 6 of the 11 inert markers in DEAD-14.
- **Check:** `uv run pytest tests -k "component_resume or attempt_provenance or callbacks or exact_resume"`.

### SMELL-06: the `NNModel` and `Trainer` training loops repeat the same epoch-end blocks

- **Severity:** Medium · **Group:** G10
- **Where:**
  - `nn_model.py` ~L5703–5715 vs `trainer.py` ~L850–863: the `NNIterationDataPoint` append
  - `nn_model.py` ~L5740–5751 vs `trainer.py` ~L866–876: the empty-epoch guard with an identical message, then `records.settle_lrs()`
  - `nn_model.py` ~L5785–5794 vs `trainer.py` ~L882–891: the `val_edp` attach, epoch summary and monitor `tracker.observe`
  - `nn_model.py` ~L5594 vs `trainer.py` ~L753: identical `NNX_TQDM_DISABLE` parsing and `tqdm(...)` construction

  The trainer even comments "Same guard as NNModel.train".
- **Fix:**
  1. In `nn_model.py`, add three helpers:
     - `_require_epoch_batches(records, idx_epoch)`
     - `_close_epoch_records(records, val_edp, train_edp, epoch_summary, tracker, monitor, idx_epoch) -> Optional[MonitorRecord]`
     - `_training_bar(n_iter)`
  2. Import them in `trainer.py` next to its existing private imports, and replace the copies.
  3. Keep `records.settle_lrs()` before validation, as both loops do now.
- **Check:** `uv run pytest tests -k "trainer or train_integration or named_monitors or history_journal or scheduler_clocks"`.

### SMELL-07: JSON parsing with a depth guard is implemented four times, each slightly differently

- **Severity:** Medium · **Group:** G10
- **Where:**
  - `_artifacts.parse_json` ~L82–114: strict, refuses NaN and duplicate keys, has a depth limit
  - `history._loads` ~L183–197: lax `json.loads` plus `too_deep`
  - `provenance._load_json` ~L589–597: raises `RecursionError` **as a sentinel**, which `load_provenance` (~L586) converts to `ValueError`
  - `bundles._parse_json` ~L420–437, with its own `_no_duplicates` (~L412); see BUG-22
- **Fix:**
  1. Add `allow_nan: bool = False` to `_artifacts.parse_json`. When it is true, skip the `constant` hook.
  2. Make `history._loads(data, what)` call `parse_json(data if isinstance(data, str) else data.decode(), what, HistoryCorruptionError, max_depth=MAX_JSON_DEPTH, allow_nan=True)`. Journal lines are written with `allow_nan=True`, so NaN must stay accepted.
  3. Make `provenance._load_json` call `parse_json(handle.read(), "provenance", ValueError, max_depth=MAX_JSON_DEPTH, allow_nan=True)`, and delete the `RecursionError` sentinel and its conversion.
  4. Do `bundles` as in BUG-22.
- **Risk:** `parse_json` also refuses duplicate keys and `1e999`. For history and provenance files that is stricter than today. If hand-edited files must keep loading, add a `strict_keys=False` flag. Tests match error text.
- **Check:** `uv run pytest tests -k "history_journal or provenance or run_bundles or bundle_validation or calibration or finite_config_validation"`.

### SMELL-08: three file-digest helpers, plus copies of the weight digest

- **Severity:** Medium · **Group:** G10
- **Where:**
  - `_artifacts.file_sha256` ~L170
  - `nn_model._file_digest` ~L1359, which returns `(hex, size)` and has its own `_HUB_DIGEST_CHUNK = 1 << 20` (~L192)
  - `provenance.hash_file` (~L210–225, public `chunk_size`)

  All three use the same 1 MiB `iter(lambda: handle.read(...), b"")` loop. The weight digests are BUG-27.
- **Fix:**
  1. Add `file_digest(path, chunk_size=1 << 20) -> tuple[str, int]` to `_artifacts.py`, and make `file_sha256` return `file_digest(path)[0]`.
  2. Replace `nn_model._file_digest` with it (keep an alias) and delete `_HUB_DIGEST_CHUNK`.
  3. Make `provenance.hash_file` call `file_digest(path, chunk_size)` after its `require_count` check.
- **Check:** `uv run pytest tests -k "hub_manifest or provenance or resume_point_hashing or export_conformance"`. The digests are byte-identical, so nothing persisted changes.

### SMELL-09: `NNTrainParams` and `NNTrainerParams` duplicate about 60 lines of validation and serialization

- **Severity:** Medium · **Group:** G10
- **Where:** `nn/params/nn_train_params.py` ~L220–268 and ~L279–329 vs `trainer/params.py` ~L200–243 and ~L260–312.
- **Duplicated logic:**
  - the `n_epochs` and `seed` checks
  - the `data_id` check
  - `plain_str` lineage normalization
  - the bool-flag loop
  - `_checkpoint_tag`
  - parent/resume exclusivity
  - `_validate_resume_mode` and `_validate_monitoring`
  - the omit-when-default keys in `state()`
  - the lineage restore in `from_state()`
- **Fix:**
  1. In `nn_train_params.py`, add `_validate_common(self, owner, flags)`, `_common_state(self, d)` and `_common_from_state_kwargs(state)`.
  2. Call them from both classes' `__post_init__`, `state` and `from_state`, keeping each class's specific checks.
- **Risk:** the omit-when-default rules feed run ids through `yaml.safe_dump(sort_keys=True)`, so they must stay **exactly** the same. Before refactoring, add a test that pins the `state()` of a few fixed params objects. Error messages embed the owner name; keep it.
- **Check:** `uv run pytest tests -k "trainer_params or params_round_trip or params_boundary_validation or integer_config_validation"`.

### SMELL-10: `NNCheckpoint.save` takes 30 keyword parameters, in singular/plural pairs, and boolean flags switch behaviour

- **Severity:** Medium · **Group:** G10 (larger; its own PR)
- **Where:**
  - `nn_checkpoint.py` ~L684–713: parallel families `optimizer_state`/`optimizers_state`, `optimizer_type`/`optimizer_types`, `…_topology/…_topologies`, `…_factory/…_factories`, `…_config/…_configs`, and the same for `scheduler_*`
  - `nn_model.py`:
    - `_source_hyperparameters(trainer=)` ~L2880
    - `_saved_hyperparameters(trainer=)` ~L2949
    - `_check_resume_hyperparameters(named=)` ~L2914
    - `_check_resume_horizons(named=)` ~L825
    - `_owner(named=)` ~L819
    - `_ONE_OPTIMIZER_KEYS` ~L838
- **Fix:**
  1. Add `@dataclass(frozen=True) class _TrainingStateParts`, with name-keyed mappings `optimizers`, `schedulers`, `types`, `topologies`, `factories` and `configs`.
  2. `save()` takes `training_state: Optional[_TrainingStateParts]` plus the scalar fields.
  3. The writer keeps the on-disk sidecar format: it writes the singular keys when `names == {""}`.
  4. Replace the `trainer=` / `named=` flags with `named = set(keys) != {""}`.
- **Risk:** the sidecar keys are an on-disk format, so the bytes written must stay identical. Pin them with a test first.
- **Check:** `uv run pytest tests -k "exact_resume or resume_parity or trainer or component_resume or objective_resume or resume_point"`.

### SMELL-11: transform-replay helpers live in `nn_model` and are imported privately elsewhere; the `"torchao_qat"` literal appears three times

- **Severity:** Medium · **Group:** G10
- **Where:**
  - `nn_model.py` ~L1089–1150: `_apply_checkpoint_transform`, `_unportable_transform`, `_replay_transforms`
  - the literal check `transform.name == "torchao_qat" and transform.version == 1` at ~L1098 and ~L1135; the producer is `quantize/qat.py` ~L383
  - private importers: `nn_checkpoint.py` ~L628 (lazily pulls in the whole 6.8k-line module) and `bundles.py` ~L1206
- **Fix:**
  1. In `nnx/transforms.py`, add `QAT_TRANSFORM = ("torchao_qat", 1)` and `def _is_qat_conversion(t): return (t.name, t.version) == QAT_TRANSFORM`. Use the constant in `qat.py`.
  2. Move `_unportable_transform` into `transforms.py` (it needs only `_replayable`, `_VERSIONS`, `TransformOp` and `RecipeError`) and re-export it from `nn_model`.
  3. Point `nn_checkpoint` and `bundles` at `transforms`.
- **Check:** `uv run pytest tests -k "model_recipes or checkpoint_safetensors or run_bundles or quantize_qat or hub_mixin"`.

### SMELL-12: the Hub strict-load check sequence is copy-pasted between the preflight and the real load

- **Severity:** Medium · **Group:** G4
- **Where:** `_refuse_shapes_before_build` (`nn_model.py` ~L1345–1356) and `_from_pretrained` (~L4967–4979).
- **Problem:** both run the same sequence:
  1. `_refuse_unrecorded_recipe_state`
  2. `check_state_schema`
  3. `load_state_dict(strict=strict)`
  4. `except (RuntimeError, ValueError)` → `_attached_submodules_error(...)`

  They differ only in how they re-raise.
- **Fix:**
  1. Extract `_load_hub_state(module, state, *, net_params, params, transforms, excluded, strict, call) -> None`. It raises the explained error `from error`, or re-raises.
  2. The preflight wraps the call: `except (RuntimeError, ValueError) as e: raise type(e)(f"refused before the network was built: {e}") from e`.
- **Check:** `uv run pytest tests -k "hub_load_preflight or hub_mixin or external_module_runtime"`.

### SMELL-13: lazy-module tracking (#562) spreads five ad-hoc attributes over `NNModel` and gives a property side effects

- **Severity:** Medium · **Group:** G10
- **Where:**
  - `nn_model.py` ~L3929–3955: `_lazy_wrap_keys`, `_lazy_layers`, `_lazy_places`, `_lazy_stamp`, `_lazy_wrapped`
  - the `params` property (~L3984–4010), which walks modules and rewrites `self.__dict__["params"]` on read
  - `_refresh_lazy_descriptor` (~L4012–4066)
- **Problem:**
  - Every access uses `getattr(self, "_lazy_*", default)` to cope with old pickles.
  - `__init__` fingerprints the module twice: `RuntimeModule.of(module)` at ~L3895, and again inside `_resolve_net_descriptor` (~L3615).
- **Fix:**
  1. Add `class _LazyTracker` with slots `keys`, `layers`, `places`, `stamp` and `wrapped`, plus a `refresh(model)` method. Store it as one attribute, `self._lazy`.
  2. In `__setstate__`, build the tracker from the old attributes when an old pickle carries `_lazy_wrap_keys`.
  3. Compute `descriptor = RuntimeModule.of(module)` once in `__init__` and pass it into `_resolve_net_descriptor` through a new optional `descriptor=` parameter.
- **Risk:** Medium. Pickle compatibility in both directions; the comment at ~L3995 states that intent.
- **Check:** `uv run pytest tests -k "external_module_runtime or legacy_net_pickles or quantize_ptq or model_factories"`.

### SMELL-14: the dry-run lazy-inference rules are written twice in `models.py`

- **Severity:** Low · **Group:** G10
- **Where:** `_inferred_attributes` (`models.py` ~L421–440) and `_left_by_a_loaded_forward` (~L493–520).
- **Problem:** both encode the same three rules:
  - `in_features = weight[-1]`
  - `in_channels = weight[0] if transposed else weight[1] * groups`
  - `num_features = (weight or running_mean)[0]`
- **Fix:** extract `_dry_run_value(kind, shape_of, layer) -> Optional[int]`. Call it with `shape_of=shapes.get` from `_inferred_attributes`, and with `shape_of=lambda n: _held_shape(layer, n)` from `_left_by_a_loaded_forward`.
- **Check:** `uv run pytest tests -k "external_module_runtime or finetune_loading"`.

### SMELL-15: `nn_model.py` has 34 function-local imports that no circular dependency requires

- **Severity:** Low · **Group:** G10 (after SMELL-02 step 4)
- **Where (in `nn_model.py`):**
  - local imports of modules already imported at top level:
    - `.params.nn_checkpoint` ~L477
    - `..transforms` ~L1094, ~L1124, ~L4144
    - `..seeding` ~L1301, ~L4117
    - `..models` ~L4938
    - `.._update_engine` ~L3152
    - `.._scheduler_clock` ~L6727
  - local imports with no cycle behind them:
    - `.callbacks` ×13 (callbacks reaches `nn_model` only under `TYPE_CHECKING`)
    - `..distributed` ×12
    - `..streaming`, `..optimizers`, `..objectives`
- **Fix:**
  1. Add the names to the existing top-level import lists and delete the local lines.
  2. Add top-level `from .callbacks import Callback, _LegacyCallback, _changes_topology, _WriterOwned` and the `..distributed` names.
- **Risk:** tests that patch `nnx.distributed.X` and rely on `nn_model`'s late lookup would stop intercepting. Run `grep -rn 'patch("nnx.distributed' tests` first.
- **Check:** `uv run pytest tests/test_imports.py tests/test_core_install.py`, then the full suite.

### SMELL-16: five read-only mapping classes and six freeze/thaw functions for "a frozen JSON-like value"

- **Severity:** Medium · **Group:** G10
- **Where:**
  - Mapping classes:
    - `_config._FrozenConfig` ~L24: insertion order, **unhashable**
    - `transforms._FrozenConfig` ~L100: same name, but sorted, hashable, O(n) `__getitem__`
    - `trainer.params._FrozenMapping` ~L44: ordered, order-free hash
    - `nn_evaluation_data_point._FrozenMetrics` ~L15: sorted, hashable
    - `decisions/benchmark._FrozenList` / `_FrozenDict` ~L365 and ~L378
  - Freeze/thaw functions:
    - `_config._freeze_config` / `_thaw_config` (~L49, ~L101)
    - `transforms._frozen` / `_plain` (~L131, ~L138)
    - `provenance._freeze` / `_thaw` (~L280, ~L291; imported by `data_splits.py` ~L62)
    - `benchmark._frozen` / `_plain` (~L388, ~L407)
    - `calibration._frozen` (~L444)
    - `_artifacts.frozen_json` (~L23)
  - Private cross-package imports of `trainer.params._FrozenMapping`: `decisions/benchmark.py` ~L65, `streaming.py` ~L774, `plans.py` ~L260 and ~L334.
- **Fix:**
  1. Create `src/nnx/_frozen.py` with `FrozenMapping` (a copy of `trainer.params._FrozenMapping`: insertion-ordered, order-free `__eq__`/`__hash__`, `__reduce__`), plus `freeze_json(value, *, lists=tuple)` and `thaw_json(value)`.
  2. Make `trainer.params._FrozenMapping = FrozenMapping` an alias, so the pickle path keeps resolving.
  3. Point `benchmark`, `streaming` and `plans` at `_frozen.FrozenMapping`.
  4. Replace the bodies of `provenance._thaw`, `transforms._plain`, `benchmark._plain` and `_config._thaw_config` with `thaw_json`, keeping the names.
  5. Keep `_config._FrozenConfig` and `transforms._FrozenConfig` as classes, because pickled run configs reference them. Rename `transforms._FrozenConfig` to `_FrozenOpConfig` and keep the old name as an alias.
- **Risk:** Medium. Hash and equality semantics differ between the copies, and pickle class paths must keep resolving.
- **Check:** `uv run pytest tests -k "transforms or provenance or decision_benchmark or trainer or plans or params_round_trip or legacy_record_pickles"`.

### SMELL-17: run-directory file names are scattered string literals, and `run.yaml` / `idps.csv` are read by four implementations

- **Severity:** Medium · **Group:** G10
- **Where:**
  - `"run.yaml"` reads: `nn_run.py` ~L1329, `history.py` ~L1171, `comparison.py` ~L1440 (`_read_run_state`), `nn_model.py` ~L2891 (`_source_hyperparameters`)
  - `"metadata.yaml"`: `nn_run` ~L769, ~L1192, ~L1375; `history` ~L1165; `comparison` ~L1030
  - `comparison.py` ~L1494–1503 re-implements `nn_run._read_idps_csv`, then de-duplicates "last record per epoch" twice (~L1499–1501 and ~L1625–1627)
- **Problem:** each copy does its own `yaml.safe_load` and "is a mapping" check, with a different error each time: `ValueError`, `ComparisonError`, or a silent `{}`.
- **Fix:**
  1. In `nn_run`, add `_RUN_YAML`, `_METADATA_YAML` and `_IDPS_CSV` constants, plus `_read_run_yaml(run_path) -> dict`, which raises `ValueError("malformed run.yaml at ...")`.
  2. Use it from `history.py`, `nn_model.py` (keeping its broad except) and `comparison.py` (wrapping `ValueError` into `ComparisonError`).
  3. Add `_read_idps_frame(csv_path) -> (records, columns)` to `nn_run`, and use it from `comparison._read_run`.
- **Check:** `uv run pytest tests -k "run_comparison or history_journal or exact_resume"`. Watch comparison's "malformed idps.csv" wording, which tests assert.

### SMELL-18: three copies of "parse one record out of `metadata.yaml`"

- **Severity:** Low · **Group:** G10
- **Where:** `nn_run.py`: `_load_resume_status` (~L753–762), `_load_compile` (~L773–782), `_load_precision` (~L785–794).
- **Fix:** add `_metadata_record(metadata, key, parse)` that returns `None` unless the value is a dict, and otherwise `try: return parse(record) except (KeyError, TypeError, ValueError): return None`. Make the three functions one-line wrappers around it.
- **Check:** `uv run pytest tests -k "run_load_exact_history or compile_execution"`.

### SMELL-19: `_read_checkpoint_path` validates through a discarded call, and four readers repeat the same guard

- **Severity:** Low · **Group:** G10
- **Where:**
  - `nn_checkpoint.py` ~L116–126: ~L124 calls `_checkpoint_path(...)` and discards the result, purely to validate
  - the guard is repeated at ~L888–890, ~L944–947, ~L966–969 and ~L994–997
- **Fix:**
  1. Split out `_validate_checkpoint_args(run, type)` and call it instead of discarding a path.
  2. Add `_existing_checkpoint(run, type, root) -> Optional[tuple[str, str]]` that also applies `_run_directory_exists`, and use it at the four sites.
- **Check:** `uv run pytest tests -k "best_pointer_readers or resume_point"`.

### SMELL-20: the same checks run twice on one save path

- **Severity:** Low · **Group:** G4
- **Where:**
  - `NNModel._require_portable` runs in `_refuse_unloadable_hub_artifact` (~L4652) and again in `_save_pretrained` (~L4733).
  - `NNCheckpoint._require_portable` runs in `to_file` (~L604) and again in `_to_safetensors_file` (~L641).
  - `_refuse_non_tensor_state` (~L4687) runs, and then `_tensor_state_dict` (~L4749) checks the same thing again.
  - `self.net.state_dict()` is built four times per `save_pretrained` (~L4657, ~L4687, ~L4695, ~L4749).
- **Fix:**
  1. Compute `state = self.net.state_dict()` once in `save_pretrained` and pass it down.
  2. Keep `_save_pretrained`'s own check: it is a huggingface_hub hook a user can call directly. Add a comment saying why it is repeated.
  3. Drop the second `_require_portable` inside `_to_safetensors_file`, which is private and reached only through `to_file`.
- **Check:** `uv run pytest tests -k "hub_mixin or checkpoint_safetensors"`.

### SMELL-21: the tokenizer file name is a duplicated literal, and the Hub stub has a wrong return annotation

- **Severity:** Low · **Group:** G4
- **Where:**
  - `generative_nn_model.py` ~L64 hard-codes `filename = "tokenizer.json"`, while `nn_model.py` ~L187 has `_HUB_TOKENIZER_FILENAME`, which the manifest writer (~L1409) and the stale-file cleanup (~L4625) depend on.
  - `nn_model.py` ~L164 declares `def _hub_unavailable(self) -> NNModel:`, but the function always raises.
- **Fix:**
  1. Import `_HUB_TOKENIZER_FILENAME` from `.nn_model` and use it in `generative_nn_model.py`.
  2. Annotate the stub `-> NoReturn` (from `typing`).
- **Check:** `uv run pytest tests -k "generative_nn_model or hub_manifest"`.

### SMELL-22: `_component_type` duplicates `models._qualified`

- **Severity:** Low · **Group:** G10
- **Where:** `nn_model.py` ~L203–205 and `models.py` ~L379–380.
- **Fix:** `def _component_type(value): return _qualified(type(value))`, importing `_qualified` from `..models`.
- **Check:** `uv run pytest tests -k component_resume`.

### SMELL-23: two batch-splitting helpers for paradigm steps, and the supervised one lives in `nn_model`

- **Severity:** Low · **Group:** G10
- **Where:**
  - `nn_model._single_input_batch` ~L2222, imported privately by `objectives.py` ~L412, `paradigms/distillation.py` ~L59, `paradigms/augmentation.py` ~L54 and `paradigms/moe.py` ~L47
  - `_step_helpers.first_input` ~L42, which re-implements the adapter, `unpack_batch` and tuple splitting
- **Problem:** the two have drifted apart: only `_single_input_batch` applies the graph `seed_count` slice.
- **Fix:**
  1. Move `_single_input_batch` to `_step_helpers.py` as `supervised_input`, and keep a re-export in `nn_model`.
  2. Update the four importers.
  3. Consider basing `first_input` on it.
- **Check:** `uv run pytest tests -k "paradigms or step_helpers or paradigm_step_slots"`.

### SMELL-24: `nn_run` duplicates provenance's file-name constants

- **Severity:** Low · **Group:** G10
- **Where:** `nn_run.py` ~L51–52 (`_PROVENANCE_MANIFEST_FILE = "provenance.json"  # nnx.provenance.MANIFEST_FILE`, plus an `ATTEMPT_FILE` copy), used at ~L341. The originals are `provenance.py` ~L83–84.
- **Fix:**
  1. Change the top-level import at ~L29 to `from ...provenance import ATTEMPT_FILE, MANIFEST_FILE, ProvenanceRecord, load_provenance`.
  2. Delete ~L51–52 and use the imported names.
- **Check:** `uv run pytest tests -k attempt_provenance`.

### SMELL-25: mutable module state indexed by magic numbers

- **Severity:** Low · **Group:** G10
- **Where:**
  - `models.py` ~L333–334: `_REGISTRATIONS: list[Any] = [object(), 0, False]`, indexed `[0]`, `[1]`, `[2]` at ~L346–370, plus `_REGISTRATION_HOOK: list` used as a once-flag
  - `history.py` ~L422: `_counters() -> tuple[int, int, int, str, int, int]`, read as `self._committed[2]` and `[5]` (~L463–464)
  - `distributed.py` ~L277–284: `global _exit_hook_registered`, defined *after* the function that uses it
- **Fix:**
  - `models.py`: `@dataclass class _RegistrationStamp: token: object; count: int; traced: bool`, with a single module-level instance. The torch.compile guard comment (~L340) needs a plain attribute read, which this keeps.
  - `history.py`: `class _Counters(NamedTuple)` with named fields.
  - `distributed.py`: move `_exit_hook_registered = False` above `init_process_group`.
- **Check:** `uv run pytest tests -k "external_module_runtime or history_journal or process_group"`.

### SMELL-26: the bfloat16 epsilon expression is duplicated

- **Severity:** Low · **Group:** G10
- **Where:** `_probability.py` ~L111 and ~L202: `source_eps = 2.0**-7 if is_tensor(x) and str(x.dtype) == "torch.bfloat16" else None`.
- **Fix:** add `BFLOAT16_EPS = 2.0**-7` and `def _source_eps(value): ...`, and use it in both places.
- **Check:** `uv run pytest tests -k "calibration or prediction_results"`.

### SMELL-27: raw `warnings.warn` with hand-picked stack levels, while everything else uses `_warn_at_user_frame`

- **Severity:** Low · **Group:** G10
- **Where:**
  - `nn_model.py` ~L4398: `stacklevel=2`, inside `to_onnx`
  - `nn_run.py` ~L996: `stacklevel=3`, inside `_epoch_series`. That function is reached through `_render_metric_chart_html` and then `_repr_html_`, so the attributed line depends on the call path.

  The Hub manifest warning is BUG-18.
- **Fix:** replace both with `_warn_at_user_frame(msg, UserWarning)` and `_warn_at_user_frame(msg)` (RuntimeWarning) respectively.
- **Check:** `uv run pytest tests -k "to_onnx_dropped_state or nn_run_repr_html"`.

### SMELL-28: small redundancies in `nn_run.py`

- **Severity:** Low · **Group:** G10
- **Where:**
  - `import shutil` inside `_point_best` (~L529), although `shutil` is already imported at top level (~L10)
  - `from collections import defaultdict` inside `_epoch_series` (~L1002)
  - the series-name tuple at ~L1008 duplicates `history._SERIES` (`history.py` ~L101, used as `("epochs", *_SERIES, "improved")` at ~L755)
- **Fix:**
  1. Delete the local `import shutil`.
  2. Hoist `defaultdict` to module top.
  3. Import `_SERIES` from `...history` (already imported at top level) and write `("epochs", *_SERIES, "improved")`.
- **Check:** `uv run pytest tests -k "nn_run_repr_html or best_pointer_readers"`.

### SMELL-29: `to_onnx` has three near-identical `torch.onnx.export` calls, and `viz/netron.py` has a fourth copy

- **Severity:** Low · **Group:** G2 (after BUG-03 and BUG-04)
- **Where:** `nn_model.py` ~L4352–4385; `viz/netron.py` ~L72–155. Both repeat the `"dynamo" in inspect.signature(torch.onnx.export).parameters` check and the branching export call.
- **Problem:** the branches differ only in `dynamic_shapes=` vs `dynamic_axes=` and in whether `dynamo=` is passed. The comment block at ~L4321–4331 mixes the training-mode comment with the partial-file comment.
- **Fix:**
  1. Add `src/nnx/_onnx.py` with `export_kwargs(*, input_names, output_names, opset_version, dynamic_axes=None, dynamic_shapes=None, dynamo=None) -> dict`. It adds `dynamo=` only when `torch.onnx.export` accepts it.
  2. Make one `torch.onnx.export(net, example_input, path, **kwargs)` call in `to_onnx` and one in `netron_export`.
  3. Move the shared floating-input placement helper there as well, and untangle the interleaved comments.
  4. Add to the `netron_export` docstring that `launch=True` starts a local HTTP server and opens a browser.
- **Check:** `uv run pytest tests -k "to_onnx or onnx_dynamo or viz_netron"`.

### SMELL-30: `_declared_module` builds a fake `NNModel` out of a `SimpleNamespace`

- **Severity:** Low · **Group:** G4
- **Where:** `nn_model.py` ~L1318: `_replay_transforms(cast(NNModel, SimpleNamespace(net=module, _topology_transforms=())), transforms)`.
- **Fix:** give `_replay_transforms` and `_apply_checkpoint_transform` a `net`-only entry point (e.g. `transforms._replay(net, transforms_so_far)`), and pass the module directly.
- **Check:** `uv run pytest tests -k "hub_load_preflight or model_recipes"`.

### SMELL-31: there is no stated policy for the permission bits of files NNx writes

- **Severity:** Low · **Group:** G5 (decide this together with BUG-12 and BUG-13)
- **Where:**
  - `mode=_CREATED_FILE_MODE` (≈0644) is passed only at `nn_model.py` ~L1414 and ~L4777, `generative_nn_model.py` ~L68 and `ollama.py` ~L173, and `_commit_partial` applies it.
  - Every other `_atomic_write_text` / `atomic_write` caller leaves `mkstemp`'s 0600. That includes user-requested exports: `export_history_csv` (`history.py` ~L1387), `JsonArtifact.save` (calibration and abstention artifacts), `ComparisonReport.save` (`comparison.py` ~L909), `data_splits` ~L325 and `preprocessing` ~L471.
  - No doc or CHANGELOG states which files get which mode.
- **Fix:**
  1. Pick a policy (maintainer decision). Suggested: files a user exports or shares get "what a plain `open(..., 'w')` gives", and run-internal files keep 0600.
  2. Apply it with an explicit `plain_mode=True` (from BUG-12) at each export call site.
  3. Document it in `docs/external-contracts.md`.
- **Check:** `uv run pytest tests -k "atomic_write_text or hub_export_atomic"`.

### SMELL-32: boilerplate copied across nets, and a duplicate `Devices` method

- **Severity:** Low · **Group:** G10
- **Where:**
  - the same `__setstate__` calling `restore_params_property` in `nn/net/feed_fwd_nn.py`, `conv_nn.py` and `feed_fwd_moe_nn.py`
  - `Devices.torch_device()` (`nn/enum/devices.py` ~L22–25), identical to `__call__` (~L19–20)
- **Fix:**
  1. In each class body, write `__setstate__ = restore_params_property`; the `(net, state)` signature already fits a method.
  2. Make `torch_device` return `self()`.
- **Check:** `uv run pytest tests -k legacy_net_pickles`.

### SMELL-33: two `_is_count` helpers with the same name and different rules

- **Severity:** Low · **Group:** G10
- **Where:** `_update_engine.py` ~L543 (strict int, no bool) and `nn/params/nn_evaluation_data_point.py` ~L44 (also accepts integral floats from a CSV reload).
- **Fix:** rename the second to `_csv_count`, and add a docstring line explaining why it differs from `_validation.require_count`.
- **Check:** `uv run pytest tests -k iteration_data_point`.
### SMELL-34: `JsonArtifact` is re-implemented by `SplitManifest` and `Standardizer`

- **Severity:** Medium · **Group:** G10
- **Where:**
  - Base class: `_artifacts.py` ~L137–172 (`JsonArtifact`: canonical bytes, `__eq__`/`__hash__`, `sha256:` digest, `to_json`, `save`).
  - Copies: `data_splits.py` ~L218–335 and `preprocessing.py` ~L439–480. Each has its own `_canonical`, `canonical_bytes`, `__eq__`, `__hash__`, `digest`, `to_json`, `from_json`, `save` and `load`.
- **Problem:** the copies have drifted. `SplitManifest.to_json` uses `ensure_ascii=False`; the other two use the default.
- **Fix:**
  1. **First** add a test that pins the current `digest()` of a fixed `SplitManifest` and a fixed `Standardizer`. Digests are persisted identities.
  2. Make both classes subclass `JsonArtifact`.
  3. Delete their `canonical_bytes`, `__eq__`, `__hash__` and `save`.
  4. Keep an overridden `_canonical` that calls `provenance.canonical_bytes`, so digests stay the same. Keep `SplitManifest.digest()`.
  5. Add `JsonArtifact.from_file(cls, path, what, error)` (`read_text` → `parse_json(max_depth=MAX_JSON_DEPTH)` → `cls.from_state`) and use it for `load`/`from_json`.
  6. Keep `ensure_ascii=False` for `SplitManifest` through an override.
- **Check:** `uv run pytest tests -k "data_splits or preprocessing"`. The pinned-digest test still passes.

### SMELL-35: knowledge distillation is implemented three times, and `KDObjective` sits in the wrong module

- **Severity:** Medium · **Group:** G9
- **Where:**
  - `paradigms/distillation.py` ~L169–219 (`kd_train_step_factory`)
  - `paradigms/distillation.py` ~L313–404 (`feature_kd_train_step_factory`: the same body plus a feature MSE)
  - `objectives.py` ~L351–465 (`KDObjective`, with lazy imports at ~L386, ~L402 and ~L409–416 of `_check_kd_weights`, `_check_live_teacher`, `_freeze_teacher`, `_single_input_batch`, `_classification_edp_for_loss`, …)
  - `paradigms/offline_distillation.py` ~L483–493 (`_check_alpha`)
- **Problem:**
  - The soft/hard mixing, the alpha 0/1 special cases, the task-adapter branch and the teacher-forward-under-`pinned_eval` block are copy-pasted three times.
  - Other paradigm objectives live with their paradigm (`diffusion/objective.py`, `paradigms/jepa_objective.py`); KD does not.
  - `_check_alpha` raises `TypeError` for `alpha=True`, while `_check_kd_weights` raises `ValueError` for the same input.
- **Fix:**
  1. Move `KDObjective` and `kd_objective` into a new `paradigms/kd_objective.py`, which can import the distillation helpers at top level.
  2. Keep them importable from `objectives.py` via a module-level `__getattr__` that lazily imports `.paradigms.kd_objective`. This avoids the `nn_model` cycle. Keep the `__all__` entries.
  3. Extract `_kd_loss(m, X, Y, student_logits, teacher_logits, alpha, temperature, extra_loss=0.0, ctx)` from the shared tail of the two step factories.
  4. Make `_check_alpha` use `checked(require_finite_real, alpha, "alpha", owner=..., error=TeacherRecordError, minimum=0, maximum=1)`.
  5. Do BUG-15 first.
- **Check:** `uv run pytest tests -k "distillation or objectives or offline_distillation or born_again"` and `python -c "import nnx.objectives"`.

### SMELL-36: Diffusion and JEPA each ship a step factory and an objective that duplicate loss and placement logic

- **Severity:** Medium · **Group:** G10
- **Where:**
  - `diffusion/training.py` ~L52–129 vs `diffusion/objective.py` ~L68–200: the forward noising `sqrt_a*x_0 + sqrt_1ma*eps`, the shape check and the MSE are repeated. The step uses the global RNG; the objective has its own generator.
  - `paradigms/jepa.py` ~L330–560 vs `paradigms/jepa_objective.py` ~L96–345: the component-state checks are repeated (`jepa.py` ~L539 vs `jepa_objective.py` ~L306).
  - `jepa_objective._check_target_placement` (~L72–94) duplicates the message logic of `_step_helpers.freeze_apart` (~L137–158).
  - Four ad-hoc "freeze a module" loops: `jepa.py` ~L58–61, `jepa_objective.py` ~L342–345, `born_again.py` ~L193–195, `_step_helpers.py` ~L157–158.
- **Fix:**
  1. Add `NoiseSchedule.q_sample(x_0, t, eps)` in `diffusion/schedules.py` and call it from both diffusion paths.
  2. Add `_step_helpers.check_apart(net, frozen, *, role, remedy, consequence)`. `freeze_apart` calls it and then freezes; `_check_target_placement` calls it with the JEPA wording.
  3. Add `_step_helpers.freeze(module, *, eval_mode=True)` and use it in the four loops.
- **Check:** `uv run pytest tests -k "diffusion or jepa or born_again"`. Tests assert the error messages, so pass the exact wording through parameters.

### SMELL-37: the decision benchmark re-implements accuracy, macro-F1, NLL and Brier, and three different NLL floors exist

- **Severity:** Medium · **Group:** G1
- **Where:**
  - `decisions/benchmark.py` ~L888–917: pure-Python `_accuracy`, an O(L·N) `_macro_f1`, `_nll`, `_brier`
  - `link_tasks.py` ~L663 (`np.clip(p, 1e-300, 1.0)`) and ~L680 (`np.clip(p, 1e-300, 1 - 1e-16)`)
  - Shared versions: `_probability.py` ~L43 (`NLL_EPSILON = 1e-12`), ~L127 `nll_terms`, ~L149 `brier_terms`; `_confusion.py` ~L123 `ConfusionCounts.scores("macro")`
- **Problem:**
  - The benchmark docstring claims NLL and Brier "follow the definitions of the named nll / brier metrics", but the code is a separate implementation.
  - The floor differs by module: the named metric uses 1e-12, `link_tasks` uses 1e-300, and the benchmark uses no floor.
- **Fix:**
  1. In `benchmark.py`, compute per-row terms with `nll_terms(np.array([row.truth]), np.array([row.probabilities]), epsilon)` and `brier_terms(...)`. Rows can have different widths, so loop over rows.
  2. Replace `_macro_f1` with `ConfusionCounts`, fed integer codes from a label→index dict built over `true ∪ predicted`.
  3. For `link_tasks`, choose one floor and document it. **Note:** changing it changes reported values, so add a CHANGELOG line.
- **Check:** `uv run pytest tests -k "decision_benchmark or link"`. Values must match to about 1e-12.

### SMELL-38: about 22 hand-rolled integer validators exist although `_validation.require_count` and `checked()` exist

- **Severity:** Medium · **Group:** G10
- **Where:** each line matches `isinstance(x, bool) or not isinstance(x, numbers.Integral) or x < N`:
  - `decisions/benchmark.py` ~L591, 709, 1024, 1260, 1390, 1392
  - `decisions/jobs.py` ~L250, 296, 299
  - `decisions/providers.py` ~L71
  - `decisions/nli.py` ~L140, 150
  - `diffusion/objective.py` ~L107
  - `generation/logits_processors.py` ~L100
  - `generation/pipeline.py` ~L101
  - `viz/weight_histogram.py` ~L62
  - `viz/activation.py` ~L97
  - `plans.py` ~L679
  - byte-identical `_count` helpers: `ranking.py` ~L86–89 and `link_tasks.py` ~L112–115
  - near-copies: `tasks.py` ~L97–102 and `lm_tasks.py` ~L85–88
- **Fix:**
  1. In each module, define `_count = functools.partial(checked, require_count, owner="nnx.<module>", error=<ModuleError>)`, or a two-line wrapper like `calibration._checked` (~L169).
  2. Replace each listed check with `_count(value, "name", minimum=1)`.
- **Check:** each module's tests. Error-message asserts may need updating, so run `grep -rn "<old message>" tests` first.

### SMELL-39: the "metric only the task's own evaluation computes" pattern is copied three times, and `_no_extra_metrics` three more

- **Severity:** Low · **Group:** G10
- **Where:**
  - `_Unavailable`: `ranking.py` ~L717–729, `link_tasks.py` ~L1474–1479, `offline_distillation.py` ~L816–830
  - `_no_config` / `_check_k_config`: `ranking.py` ~L732, `link_tasks.py` ~L1482, `offline_distillation.py` ~L833
  - `_register_metrics`: `ranking.py` ~L738, `link_tasks.py` ~L1487, `offline_distillation.py` ~L838
  - `_no_extra_metrics`: `ranking.py` ~L577, `link_tasks.py` ~L1199, `lm_tasks.py` ~L527
- **Fix:**
  1. In `nnx/monitors.py`, add `register_task_owned_metrics(specs: Mapping[str, tuple[str, str]], *, error, message, check_config=None)`, which builds the placeholder accumulator class internally.
  2. Also add `refuse_extra_metrics(ctx, error, message)`.
  3. Replace the nine helpers with calls to these two.
- **Check:** `uv run pytest tests -k "ranking or link or lm_tasks or offline_distillation or monitors"`.

### SMELL-40: `peft/_targets.py` is de facto shared module-walking code, imported privately by quantize, prune and transforms

- **Severity:** Low · **Group:** G10
- **Where:**
  - `peft/_targets.py` ~L33, ~L38–45 (`WEIGHT_READING_PARENTS`), ~L48 (`_registrations`), ~L73 (`_listed`)
  - importers: `quantize/qat.py` ~L48, `prune/magnitude.py` ~L136, `transforms.py` ~L474 and ~L484
  - `qat.py` ~L186–188 re-implements `_listed` inline although it imports it
  - other copies: `_listed` in `gguf/tensor_name_map.py` ~L89 and `comparison.py` ~L1390; a second walker, `surgery/_utils.registered_paths` (~L142)
- **Fix:**
  1. Create `src/nnx/_module_paths.py`. Move `_registrations` (as `registrations`), `WEIGHT_READING_PARENTS` and `_listed` (as `listed_paths`) into it, and leave aliases in `peft/_targets.py`.
  2. Point `qat`, `magnitude` and `transforms` at the new module.
  3. Delete the inline copy in `qat.py`.
  4. Make `registered_paths` use `registrations`.
- **Check:** `uv run pytest tests -k "peft or qat or prune or transforms or surgery"`.

### SMELL-41: `decisions/jobs.py` `_Runner.arun` and `decisions/benchmark.py` are oversized and mix responsibilities

- **Severity:** Low · **Group:** G11 (skip it if SCOPE-01's extraction is imminent)
- **Where:**
  - `jobs.py` ~L1029–1149: `arun`, 120 lines, cyclomatic complexity 42
  - `benchmark.py` sections:
    - samples and perturbations ~L129–345
    - records and JSONL ~L347–575
    - `collect` ~L679–780 (cyclomatic complexity 34)
    - metrics ~L831–935
    - evaluate and slices ~L937–1300
    - intervals ~L1339–1450
    - report and compare ~L1451–1643
- **Fix:**
  1. In `arun`, extract the nested closures into `_Runner` methods: `_launch(...)`, `_harvest(done, in_flight) -> failures` and `_drain(in_flight, threads, stop, started)`.
  2. Split `benchmark.py` into `decisions/_bench_samples.py`, `_bench_records.py` and `_bench_report.py`. `benchmark.py` re-exports everything.
- **Check:** `uv run pytest tests/test_decision_jobs.py tests/test_decision_benchmark.py`.

### SMELL-42: a lazy-import cycle runs decisions ↔ abstention ↔ result, and `result.py` imports private `jobs` helpers

- **Severity:** Low · **Group:** G11
- **Where:**
  - `abstention.py` ~L1462: `from .decisions.schema import ...` inside `decide`
  - `decisions/jobs.py` ~L429, 437, 537, 974 and `decisions/benchmark.py` ~L1117, 1263: `from ..abstention import ...`
  - `result.py` ~L223, 272, 283, 294, 296: `from .decisions.jobs import _check_results, _rows`
- **Fix:**
  1. Move `abstention.decide` into `decisions/selective.py` (SCOPE-01 step 2). After that, abstention never imports decisions, so decisions can import abstention at top level. Delete the six local imports.
  2. Rename `jobs._check_results` and `jobs._rows` to `check_results` and `input_rows`, keeping the old names as aliases, and import them at the top of `result.py`.
- **Check:** `uv run pytest tests -k "abstention or result_boundaries or decision or imports"`.

### SMELL-43: input-modality detection is implemented four times and disagrees

- **Severity:** Low · **Group:** G1
- **Where:**
  - `decisions/jobs.py` ~L575–611: `_first`, `_modality`, `_rows`
  - `decisions/benchmark.py` ~L781–791: `_modality`
  - `decisions/providers.py` ~L178–187: `_batch`
  - `decisions/nli.py` ~L155 and `decisions/jev.py` ~L210: `_texts`
- **How they disagree:**
  - Tuples of texts: `jobs._first` treats a tuple whose first item is a `str` as a batch of texts, while `benchmark._modality` takes `inputs[0]`, so the same tuple reads as unknown there.
  - Tensors: `jobs` requires `ndim >= 1` via `isinstance(torch.Tensor)`; `benchmark` duck-types `hasattr(first, "detach")` and accepts 0-d tensors.
  - Blank text: `jev._texts` accepts `""` and empty lists; `nli._texts` refuses blank strings.
- **Fix:**
  1. Keep `jobs._modality` as the single rule, moved to `decisions/providers.py` as `input_modality(state) -> Optional[tuple[str, int]]`.
  2. Make `benchmark._refusal` call it.
  3. Make `jev._texts` reject blank texts as `nli._texts` does, or document why it differs.
- **Check:** `uv run pytest tests -k decision`.

### SMELL-44: RNG snapshot/restore is re-implemented in `lr_finder`, `viz.summary` and `compilation`

- **Severity:** Low · **Group:** G10
- **Where:**
  - copies: `lr_finder.py` ~L159–185 and ~L297–310 (Python, NumPy, torch, CUDA/MPS and loader generators, all by hand); `viz/summary.py` ~L177–201; `compilation.py` ~L332
  - canonical helpers: `seeding.py` ~L101 `_capture_rng_state(train_loader, cuda=)`, ~L134 `_restore_rng_state(state, train_loader)`, ~L198 `_global_rng_kept()`
- **Fix:**
  1. In `lr_finder`, call `state = _capture_rng_state(train_loader, cuda=<cuda in use>)` before the sweep, and `_restore_rng_state(state, train_loader)` in its `finally`.
  2. In `viz/summary.py`, wrap the summary call in `with _global_rng_kept():`.
- **Risk:** `lr_finder` restores only the current CUDA device, while `seeding` restores all of them. That is a small behaviour change; mention it in the CHANGELOG.
- **Check:** `uv run pytest tests -k "lr_finder or viz_summary or seeding"`.

### SMELL-45: code smells in the GGUF writer and the Ollama export

- **Severity:** Low · **Group:** G10
- **Where:** `interop/gguf/writer.py` ~L94–110, ~L114–139, ~L229–230, ~L241, ~L263, ~L286, ~L331–345; `interop/ollama.py` ~L58–60, ~L164–173.
- **Problems:**
  - The quantization label is mapped twice by if-chains (`_quantization_torch_dtype`, `_gguf_file_type_for`).
  - A valid llama-quantize label that NNx does not write raises `ImportError` (~L119). It should be a value error.
  - `add_layer_norm_rms_eps(1e-6)` and `int(2*ffn_mult*d_model/3)` duplicate constants from `transformer_layers.py` (~L52, ~L150) instead of reading them from the model.
  - The false comments are DOC-31.
- **Fix:**
  1. Replace the two if-chains with `_QUANTIZATIONS = {"F32": (torch.float32, "ALL_F32"), "F16": (torch.float16, "MOSTLY_F16"), "BF16": (torch.bfloat16, "MOSTLY_BF16")}`.
  2. Read `feed_forward_length = transformer_nn.blocks[0].ffn.w1.out_features` and `eps = transformer_nn.norm_out.eps` from the model. Verify these attribute names against `TransformerNN` first.
  3. Raise `ValueError` for unsupported quantization labels. Because that breaks callers that catch `ImportError`, either keep `ImportError` for one release with a `FutureWarning`, or leave the type and only improve the message.
- **Check:** `uv run pytest tests -k "interop or gguf"`.

### SMELL-46: three overlapping ways to configure logits processing in `generate()`; `LogitsChain` is superseded

- **Severity:** Low · **Group:** G11
- **Where:**
  - `generation/logits_chain.py` ~L52–145: `LogitsChain`, `LogitsChainBuilder`
  - `generation/pipeline.py` ~L264–350: `OrderedLogitsPipeline`
  - `nn/generative_nn_model.py` ~L101–103 and ~L191–245: inline kwargs, `logits_chain=` and `logits_pipeline=`, plus mutual-exclusion checks
- **Problem:** `OrderedLogitsPipeline` is a strict superset: the same four stages, `CustomStage`, serialization and greedy handling. `LogitsChainBuilder.custom` already tells users to switch.
- **Fix:**
  1. Emit `DeprecationWarning` in `LogitsChain.__post_init__` and `LogitsChainBuilder.__init__`, pointing at `OrderedLogitsPipeline`.
  2. Add `OrderedLogitsPipeline.from_chain(chain)` for migration.
  3. Add a CHANGELOG entry, and remove both classes in the next minor release.
  4. Update any example or doc that uses `LogitsChain` (`grep -rn LogitsChain examples docs`).
- **Check:** `uv run pytest tests -k generation`.

### SMELL-47: state-dict source resolution and shape pre-checks exist in three places

- **Severity:** Low · **Group:** G10
- **Where:**
  - `peft/_source.py` ~L29–76: `_resolve_source_to_state_dict`, `_refuse_shape_mismatch`
  - `finetune/loading.py` ~L84–180, which imports the private `models._shape` (~L23)
  - `models.py` ~L613–641: `_shape`, `check_state_schema`
- **Problem:** peft's version ignores lazy parameters, and `.shape` on an `UninitializedParameter` raises.
- **Fix:**
  1. Rename `models._shape` to `models.state_shape`, keeping an alias.
  2. Move `_refuse_shape_mismatch` into `models` as `refuse_reshaped(module, sd, fn_name)`, using `state_shape`.
  3. Make `peft._source` and `finetune.loading` call it.
- **Check:** `uv run pytest tests -k "peft or finetune"`.

### SMELL-48: the four PEFT save/load pairs are copy-pasted

- **Severity:** Low · **Group:** G9 (do BUG-19 at the same time)
- **Where:** `peft/lora.py` ~L279–335, `peft/ia3.py` ~L161–209, `peft/prefix.py` ~L270–310, `peft/prompt.py` ~L182–217.
- **Problem:**
  - Every pair has the same body: `select → torch.save → return str(path)`, then `resolve → select → _refuse_shape_mismatch → load_state_dict(strict=False) → len(sd) - len(unexpected)`.
  - Adapter ownership is expressed three different ways: `owned_adapter_keys`, the `PrefixTuner._owned_keys()` method, and the `PromptTuner._OWNED_KEYS` constant.
- **Fix:**
  1. Add `_save_owned(state_dict, owned, path)` and `_load_owned(module, source, owned, fn_name)` to `peft/_source.py`.
  2. Make the eight public functions one-liners over them. This also gives `load_prompt_weights` the shape check (BUG-19).
- **Check:** `uv run pytest tests -k peft`.

### SMELL-49: JSONL record read/write is duplicated, and the key checks are hand-rolled

- **Severity:** Low · **Group:** G10
- **Where:**
  - read/write: `decisions/benchmark.py` ~L550–572 vs `paradigms/offline_distillation.py` ~L259–284
  - hand-rolled unknown/missing-key checks: `benchmark.py` ~L529–533 (with four `# type: ignore[arg-type]` at ~L536–539) and `offline_distillation.py` ~L250–256
  - `_artifacts.check_keys` already exists.
- **Fix:**
  1. Add `_artifacts.write_jsonl(path, states)` and `_artifacts.read_jsonl(path, what, error, from_state)`, with `path:line` prefixes in error messages.
  2. Replace the manual key checks with `check_keys(state, required=..., optional=..., what=..., error=...)`, and drop the `type: ignore`s.
- **Check:** `uv run pytest tests -k "decision_benchmark or offline_distillation"`.

### SMELL-50: probability-distribution validation and the 1e-6 tolerance are defined several times

- **Severity:** Low · **Group:** G10
- **Where:**
  - `decisions/schema.py` ~L33 (`PROBABILITY_TOLERANCE`) and ~L264–305 (`_probability`, `_check_distribution`)
  - `paradigms/offline_distillation.py` ~L105 (`PROBABILITY_TOLERANCE`) and ~L173–195
  - `decisions/benchmark.py` ~L347–362 (`_real`, `_finite_probability`)
  - `_probability.py` ~L44 (`SUM_TOLERANCE`)
  - `decisions/nli.py` ~L444–449: `_softmax` duplicates `_probability.softmax_` (~L93)
- **Fix:**
  1. Define both public `PROBABILITY_TOLERANCE` names as `= _probability.SUM_TOLERANCE`.
  2. Add `_probability.check_probabilities(values, *, error, what) -> tuple[float, ...]` (finite, in [0, 1], `fsum` within tolerance), and call it from `TeacherRecord.__post_init__` and `schema._check_distribution`.
  3. Make `nli._softmax` call `softmax_`.
- **Check:** `uv run pytest tests -k "decision_schema or offline_distillation or decision_nli"`.

### SMELL-51: `digest()` formats and canonical JSON differ across artifact types

- **Severity:** Low · **Group:** G10 (documentation only)
- **Where:**
  - prefixed `sha256:<hex>`: `calibration.py` ~L854, `data_splits.py` ~L298, `preprocessing.py` ~L452, `link_tasks.py` ~L435
  - bare hex: `decisions/schema.py` ~L145, `search.py` ~L276
  - canonical JSON forms:
    - `search.py` ~L277: `json.dumps(state, sort_keys=True)`, default separators, ASCII escaping
    - `link_tasks.py` ~L436 and `benchmark.py` ~L634: compact separators, ASCII escaping
    - `_config._canonical_json` (~L110): compact, no ASCII escaping
- **Fix:** **do not change existing digests**; they are persisted.
  1. Add a comment at each site naming its canonical form.
  2. Add a note in `_artifacts.py` that any new artifact must use `_config._canonical_json` and a `sha256:` prefix.

### SMELL-52: the same "weight recomputed by a hook" refusal raises different error types

- **Severity:** Low · **Group:** G10
- **Where:**
  - `TypeError`: `interop/gguf/tensor_name_map.py` ~L98–145 and `peft/dora.py` ~L94–153
  - `ValueError`: `surgery/_utils.py` ~L116–139 and `prune/magnitude.py` ~L117–130
  - the same "walk `named_modules` and qualify masked/normed names" loop: `tensor_name_map.py` ~L109–115 and `nn/nn_model.py` ~L1170–1186
- **Fix:**
  1. Add `_reparameterization.legacy_reparameterized_paths(net) -> tuple[list[str], list[str]]` and use it in both walkers.
  2. Write down the error-type rule in `_reparameterization.py`'s module docstring: `TypeError` when the object kind is unsupported, `ValueError` when its configuration is.
  3. Keep the existing types for compatibility; apply the rule to new code.

### SMELL-53: `embeddings.train_contrastive` is a second hand-written training loop

- **Severity:** Low · **Group:** G11 (BUG-24 is the short-term fix)
- **Where:** `embeddings/contrastive_trainer.py` ~L328–489.
- **Problem:** it owns its own optimizer, gradient clipping, non-finite check and `print` output, running in parallel with `NNModel.train` + `text_contrastive_train_step_factory` (~L257).
- **Fix (longer term):** re-implement it as a thin wrapper, `NNModel(module=backbone).train(params, train_step_fn=text_contrastive_train_step_factory(...))`, keeping the public signature. Add a test asserting the same loss trajectory for a fixed seed before and after the change.
- **Check:** `uv run pytest tests/test_embeddings_contrastive.py`.

### SMELL-54: viz repeats the `NNModel` unwrap six times and has doubled `TYPE_CHECKING` blocks

- **Severity:** Low · **Group:** G10
- **Where:**
  - the unwrap: `viz/weight_histogram.py:70`, `attribute.py:171`, `gradient_flow.py:53`, `activation.py:103`, `summary.py:83`, `netron.py:76`
  - back-to-back `if TYPE_CHECKING:` blocks: `weight_histogram.py` ~L23–29, `gradient_flow.py` ~L16–21, `activation.py`, `attribute.py`
- **Fix:**
  1. Add `viz/_common.py` with `def module_of(model) -> nn.Module`, and use it in the six places.
  2. Merge each pair of `TYPE_CHECKING` blocks.
  3. Do it together with the viz part of DOC-31 (hoisting the imports).
- **Check:** `uv run pytest tests -k viz`.

### SMELL-55: Mixup and CutMix duplicate the whole post-mix step

- **Severity:** Low · **Group:** G9 (together with BUG-20)
- **Where:** `paradigms/augmentation.py` ~L165–183 and ~L252–275. Both repeat the forward pass, `require_row_logits`, the task branch, the λ-mixed loss, the scorable check, `finalize_step`, `_weighted_acc` and the EDP.
- **Fix:** extract `_mixed_supervised_step(ctx, m, X_in, Y_a, Y_b, lam, *, paradigm)` next to `_mixed_task_step` (~L74), and call it from both.
- **Check:** `uv run pytest tests/test_paradigms_augmentation.py`.

### SMELL-56: DPO's `_require_response_tokens` and `_require_prompt_tokens` are near-identical

- **Severity:** Low · **Group:** G9
- **Where:** `paradigms/dpo.py` ~L252–268 and ~L271–287.
- **Fix:** write `_first_empty_row(ids, pad_token_id) -> Optional[tuple[int, str]]`, and have each function format its own message from its result.
- **Check:** `uv run pytest tests/test_paradigms_dpo.py`.

### SMELL-57: the pattern-targeting API is inconsistent across peft, finetune and prune

- **Severity:** Low · **Group:** G10
- **Where:**
  - `peft/lora.py` ~L176: `apply_lora_to(module, *name_patterns)` takes varargs, refuses an empty list and walks every registration path.
  - `finetune/freezing.py` ~L25 and ~L51: `freeze` / `unfreeze(module, *patterns)` are two copy-pasted loops.
  - `prune/magnitude.py` ~L37 and `prune/semi_structured.py` ~L31: take `layer_pattern: str = "*"` and match on `named_modules()`. That de-duplicates names, so a Linear aliased as `a.x` and `b.x` is missed by `layer_pattern="b.*"`.
  - `magnitude_prune` checks `sparsity` with a raw comparison (~L100).
- **Fix:**
  1. Add `_set_requires_grad(module, patterns, flag, fn)` in `freezing.py`, and use it from both functions.
  2. In both prune functions, accept `*layer_patterns` in addition to the `layer_pattern` keyword, and match with `_module_paths.registrations` (SMELL-40).
  3. Validate `sparsity` with `require_finite_real(..., minimum=0, maximum=1, exclusive_max=True)`.
- **Check:** `uv run pytest tests -k "prune or finetune"`.

### SMELL-58: non-empty-string validators are duplicated

- **Severity:** Low · **Group:** G10
- **Where:**
  - `decisions/benchmark.py` ~L129–134 and `comparison.py` ~L99–104 (identical except for the error class)
  - `paradigms/offline_distillation.py` ~L127–130
  - `decisions/providers.py` ~L76–78
  - `_validation.py` ~L246: `required_id` rejects only `""`, whereas the others also reject whitespace
- **Fix:** add `_validation.required_text(value, what, *, error, optional=False, strip=True)`, and make each `_text` / `_check_name` delegate to it.
- **Check:** `uv run pytest tests -k "decision or comparison or offline_distillation"`.

### SMELL-59: three "value, or unavailable with a reason" types, two of them public and both named `MetricValue`

- **Severity:** Low · **Group:** G10
- **Where:**
  - `link_tasks.py` ~L596: `MetricValue(value, reason)`, in `__all__`
  - `decisions/benchmark.py` ~L833: `MetricValue(name, value, denominator, reason, note)`
  - `paradigms/offline_distillation.py` ~L640: `Measure(value, denominator, reason)`
- **Fix:**
  1. Keep the public names.
  2. Add a note to each docstring pointing at the other two.
  3. Note in `_metrics.py` that future modules should share one type.

### SMELL-60: the half-precision widening check is inlined instead of using `_step_helpers.full_precision`

- **Severity:** Low · **Group:** G10
- **Where:** `lm_tasks.py` ~L94, `ranking.py` ~L251–252 and `peft/dora.py` ~L277. The helper is at `_step_helpers.py` ~L85–88.
- **Fix:**
  1. Use `full_precision` in `lm_tasks.py` and `ranking.py`.
  2. For `dora`, which also keeps fp64, add `promoted_dtype(t)` beside `full_precision`.
  3. `_step_helpers` imports `nn_model`. If that creates a cycle, put the helper in a new torch-only `_dtypes.py`.
- **Check:** `uv run pytest tests -k "lm_tasks or ranking or peft_dora"`.

### SMELL-61: needless function-local imports in the feature packages

- **Severity:** Low · **Group:** G10
- **Where:**
  - `decisions/benchmark.py`:
    - numpy and torch at ~L212, 246, 615, 621, 784, 923, 1386
    - `_artifacts`, which is standard-library only, at ~L552, 561, 1567, 1575
    - `.schema`, already imported at the top, at ~L1118
  - `decisions/jobs.py`: `.schema`, already imported, at ~L532; numpy and torch at ~L586–587 and ~L619–620
  - `paradigms/offline_distillation.py` ~L261–263 and ~L273
  - `interop/gguf/writer.py` ~L97, 212, 230
  - Local-import counts per file: `plans.py` 38, `transforms.py` 27, `bundles.py` 21, `export_conformance.py` 18, `decisions/benchmark.py` 17.
- **Fix:**
  1. Hoist imports of the standard library, numpy, torch, `_artifacts`, `_validation` and same-package modules to module top.
  2. Keep function-local imports only for optional extras and for real cycles (`nn.nn_model`, and `abstention` until SMELL-42 is done). Mark each one you keep with `# cycle: <module>` or `# optional extra: <name>`.
- **Check:** `uv run pytest tests/test_imports.py`, plus each touched module's tests.
## 11. Appendix

### 11.1. Mapping from the earlier `/code-review` run to this report

| Earlier # | New ID | Change after re-verification |
|---|---|---|
| 1 | BUG-01 | Confirmed. The existing test `test_evaluate_names_a_sample_whose_input_has_no_stable_digest` enshrines the bug and must be rewritten. |
| 2 | BUG-02 | Confirmed and widened: revisions and prompt identities have the same compatibility break. |
| 3 | BUG-03 | Confirmed and broader than "> 2 GB": any tensor ≥ 256 bytes is externalised by default. |
| 4 | BUG-04 | Confirmed. **Correction:** the BatchNorm fusion is the eval peephole pass, not constant folding, so "skip when constant folding is on" would not fix it. |
| 5 | BUG-07 | Confirmed; also covers the `best` pointer being silently demoted. |
| 6 | BUG-08 | Confirmed; also covers deleting a user-overwritten tokenizer. The failure window is narrower than first stated. |
| 7 | BUG-10 | Confirmed. |
| 8 | BUG-05 | Confirmed; fixed by BUG-04. |
| 9 | BUG-14, DOC-26 | Partly. The refusal is deliberate, documented and tested; only the module docstring is stale. The usability problem is worse than stated: two providers can never be compared. |
| 10 | BUG-15, DOC-06 | Partly. Dropping the classification fields is deliberate and tested; only the silent dropping of `extra_metrics` is a bug. |
| 11 | BUG-16 | Confirmed. The impact is bounded by 1e-6. |
| 12 | BUG-12 | Confirmed. |
| 13 | SYNC-01 | Confirmed; it is the only such call in the repo. |
| 14 | DOC-25 | Partly. The duplicates were deliberately copied from the v0.4.4 tag. 0.4.3 has one more, and the guarding test misses `*` bullets. |
| 15 | CI-02 | Confirmed. Timeouts also count as idle. |
| overflow: Hub manifest warning location | BUG-18 | Confirmed. |
| overflow: factory runs twice | BUG-17 | Confirmed. |

New since the earlier run:
- BUG-06, BUG-09, BUG-11, BUG-13, BUG-19 to BUG-28
- all CI items except CI-02
- all DOC items except DOC-25 and DOC-26
- SYNC-02 to SYNC-07, all SCOPE, DEAD, CLUT and SMELL items.

### 11.2. Checked and found clean (do not "fix")

- **Docs:**
  - `docs/api.md` and `scripts/docs/api_reference_catalog.txt` agree on the same 484 entries.
  - Every `@api` path resolves.
  - Every import and keyword argument in a doc code block resolves.
  - The docs build passes strict mode.
  - `PYPI_README.md` is current.
  - The feature-composition ids and test node ids are valid.
  - Heading numbering is sequential everywhere except DOC-17.
- **Examples:**
  - All 57 examples are listed in `examples/README.md`; numbering 01–26 has no gaps or duplicates.
  - No removed or renamed API, unknown keyword argument or missing enum attribute is used.
  - All 49 `-k` selectors resolve.
- **Config:**
  - Pins agree across `pyproject.toml`, `uv.lock`, `requirements-tools.txt`, external-contracts (except DOC-27) and pre-commit.
  - `uv lock --check` passes, and ruff lint and format pass.
  - Python 3.11–3.14 is stated consistently in the classifiers, `requires-python`, the CI matrices, README, pyright and ruff.
  - No workflow references a missing path.
  - The removed `pip-audit-py310-ignores.txt` is not referenced anywhere (DEAD-13 is only tombstone asserts).
- **Scripts:**
  - `check_export_conformance.py` exit codes match `ci.yml`.
  - `finalize_changelog.py --release-notes` strips the `v` prefix as `release.yml` needs.
  - No hard-coded secrets; `push_wiki` redacts its URL.
- **Tests:**
  - No unconditional skip or xfail, and no `breakpoint`/`pdb`.
  - No shadowed tests.
  - The private-import policy in `tests/README.md` is loose rather than violated.
- **Source:**
  - No TODO, FIXME, XXX or HACK markers anywhere in `src/`.
  - The `nnx/__init__.py` lazy `__getattr__` table matches `__all__`.
  - The 24 broad `except Exception` sites in the feature packages are documented, deliberate fallbacks, apart from BUG-28.
  - `compilation._compiled_writes_bump_versions` compares versions correctly.
  - vulture's other hits are public API, torch hooks or huggingface_hub mixin hooks.
- **Top level:**
  - `docs/architecture.html` is the SVG/PNG source, not a duplicate of `architecture.md`.
  - The large `uv.lock` churn is the Python 3.10 drop plus Dependabot.
  - `.maintenance/` is not shipped in the sdist (CLUT-01 is about repo clutter only).
