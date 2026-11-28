---
id: FEAT-121
title: "[FEAT-121] Split decisions/benchmark.py by responsibility"
labels: ["enhancement", "area: decisions", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-41"]
depends_on: ["DEC-017", "FEAT-113", "FEAT-114", "FEAT-126", "FEAT-127", "FEAT-134", "FEAT-136", "FIX-104", "FIX-105", "FIX-106", "DEC-016", "DEC-025", "FEAT-060", "FEAT-128", "FEAT-141", "FEAT-096"]
related_issues: [243, 558, 219]
status: READY
---

**Category:** Decisions · **Priority:** P3 · **Size:** L · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-121: `src/nnx/decisions/benchmark.py` is 1,643 lines that hold samples and perturbations, records and JSONL, the live collector, metrics, replay, intervals and the report. This ticket moves three self-contained layers into private modules (`_bench_samples.py`, `_bench_records.py`, `_bench_report.py`) and keeps `benchmark.py` as the public module: it keeps the collector, the metric computations, replay and `bootstrap_interval`, and re-exports every moved name. Nothing a caller imports or observes changes.

## Context

**What happens today.** One module, seven sections, marked by comments:

```python
# --- samples and perturbations ------------------------------------------------------------------
```

```python
# --- records ------------------------------------------------------------------------------------
```

followed by `# --- the live collector`, `# --- metrics`, `# --- evaluation`, `# --- intervals` and `# --- the report`. The report types (`MetricValue`, `Coverage`, `SliceReport`, `Resources`, `Interval`) sit in the metrics and intervals sections, while `BenchmarkReport` needs all of them.

**Why it matters.** Most of the open benchmark tickets edit different sections of this one file; a split by responsibility makes each layer reviewable alone and keeps their imports one-directional.

**History.** #243 (FEAT-021) introduced the benchmark as one module; #558 (FIX-090) froze its records and report mappings (the `_FrozenList` / `_FrozenDict` pickling must keep working). SMELL-41 was filed as an architecture item: this ticket runs only after DEC-017 settles where `nnx.decisions` lives, and after every other ticket that edits `benchmark.py`, so it moves final code.

**Background for the worker.**
- Re-export: `benchmark.py` imports every moved name, so `from nnx.decisions.benchmark import X` keeps working for every `X` it defines today, public or private.
- Redundant alias: `from ._bench_samples import _text as _text` tells ruff (F401) and pyright that the import is a deliberate re-export.
- Layering: `_bench_samples` imports none of the other three; `_bench_records` imports only `_bench_samples`; `_bench_report` imports only those two; `benchmark` imports all three. This order is what makes the split cycle-free.
- Pickles: a class keeps its `__qualname__`; its `__module__` becomes the new private module. Objects pickled before the split name `nnx.decisions.benchmark.<Class>` and still load through the re-export.
- `docs/api.md` documents these names under `nnx.decisions.benchmark.*` (`scripts/docs/api_reference_catalog.txt` L253-L297); those paths still resolve, and the signatures use string annotations, so the generated file does not change.

**Proposal.**
1. Create `src/nnx/decisions/_bench_samples.py` with the module docstring `"""Benchmark samples and their perturbations, and the constants and checks every benchmark layer shares. Internal: import from :mod:`nnx.decisions.benchmark`."""` and move into it, from `benchmark.py`, in this order and verbatim:
   - the constants `FORMAT`, `RECORD_FORMAT`, `STATUSES`, `METRICS`, `UNITS` and `Question` (L108-L121);
   - `BenchmarkError`, `_text`, `_labels` (L124-L138);
   - the whole `# --- samples and perturbations` section, its comment line included (L141-L341): `Sample`, `_fingerprint`, `_derived`, `_options`, `_with_options`, `permute_options`, `redescribe`, `add_distractors`, `add_none_of_the_above`, `add_context`, `rewrite_input`. (FEAT-136 deletes `_byte_dtype`.)
2. Create `src/nnx/decisions/_bench_records.py` with the docstring `"""Benchmark records and their JSONL files. Internal: import from :mod:`nnx.decisions.benchmark`."""` and move into it the whole `# --- records` section (L344-L572): `_real`, `_finite_probability`, `_FrozenList`, `_FrozenDict`, `_frozen`, `_check_depth`, `_plain`, `_snapshot`, `Record`, `write_records`, `read_records`.
3. Create `src/nnx/decisions/_bench_report.py` with the docstring `"""The benchmark report and its value types: metric values, coverage, slices, resources, intervals, and report comparison. Internal: import from :mod:`nnx.decisions.benchmark`."""` and move into it:
   - from the `# --- metrics` section: `MetricValue`, `_json_number` (L833-L866), `Coverage`, `SliceReport`, `Resources` (L937-L1040);
   - from the `# --- intervals` section: `Interval` (L1339-L1366);
   - the whole `# --- the report` section (L1447 to the end of the file): `BenchmarkReport`, `_csv_number`, `_report_state`, `compare_reports`.
4. Everything else stays in `benchmark.py`, in its current order: the module docstring, `__all__`, the `# --- the live collector` section (`Budget`, `Collection`, `_batch_input`, `_identity`, `_answer`, `_records`, `collect`, `_modality`, `_batch_cap`, `_refusal`), the computations of the `# --- metrics` section (at `9ad461e`: `_Row`, `_accuracy`, `_macro_f1`, `_nll`, `_brier`, `_bins`, `_STATISTICS`; FEAT-113 may have replaced some of them, and what it left stays), the `# --- evaluation` section (`_join` through `_sample_set`) and `bootstrap_interval`.
5. Line numbers above are at `9ad461e`; the dependencies move them. Place every top-level name by the rule they illustrate: a name keeps the section it sits in when this ticket starts; a name the earlier tickets added (for example FIX-105's report version, FIX-106's `same_records`, DEC-016's record field) goes with its section. A helper that two layers use goes to the lower layer (`_bench_samples` < `_bench_records` < `_bench_report` < `benchmark`).
6. Give each new module the module-top imports its moved code uses, and nothing else: the standard-library and third-party imports, `from .schema import (...)` with the names it uses, `from .._json_depth import ...`, `from .._frozen import FrozenMapping` (records; FEAT-096 moved the class there and benchmark.py already imports it under that name), and the lower layers' names. Example for `_bench_records.py`: `from ._bench_samples import RECORD_FORMAT, STATUSES, BenchmarkError, _text`.
7. In `benchmark.py`, replace the moved code with three import blocks right after the existing module-top imports, listing every name moved in steps 1-3 with a redundant alias, for example:

   ```python
   from ._bench_records import Record as Record
   from ._bench_records import _FrozenDict as _FrozenDict
   ```

   Use one `from ... import (...)` block per module with every name in the `name as name` form. Keep `__all__` exactly as it is.
8. Remove the imports `benchmark.py` no longer uses (ruff reports them), then run `uv run ruff check --fix src/nnx/decisions/` and `uv run ruff format src/nnx/decisions/`.
9. Regenerate nothing by hand: run `uv run python -m scripts.docs.build_api_reference --check`; it must pass with no change to `docs/api.md`.

No behaviour, message or public path changes, so no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The three modules exist and are layered. Check: `grep -n "^from \._bench\|^from \.benchmark" src/nnx/decisions/_bench_samples.py` returns 0 lines; `grep -n "^from \._bench_report\|^from \.benchmark" src/nnx/decisions/_bench_records.py` returns 0 lines; `grep -n "^from \.benchmark" src/nnx/decisions/_bench_report.py` returns 0 lines
- [ ] Every top-level name `benchmark.py` defined before the change still imports from it. Before editing, record the names: `uv run python -c "import ast; body=ast.parse(open('src/nnx/decisions/benchmark.py').read()).body; names={n.name for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef))} | {t.id for n in body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}; print(' '.join(sorted(names)))" > defined-before.txt`. After the change: `uv run python -c "import nnx.decisions.benchmark as b; print([n for n in open('defined-before.txt').read().split() if not hasattr(b, n)])"` prints `[]`. Delete `defined-before.txt` before committing. Check: the second command
- [ ] No name moved in steps 1–3 is still defined in `benchmark.py` (it only re-exports them). Check: `grep -nE "^(def|class) (Sample|_fingerprint|_derived|_options|_with_options|permute_options|redescribe|add_distractors|add_none_of_the_above|add_context|rewrite_input|_real|_finite_probability|_FrozenList|_FrozenDict|_frozen|_check_depth|_plain|_snapshot|Record|write_records|read_records|MetricValue|_json_number|Coverage|SliceReport|Resources|Interval|BenchmarkReport|_csv_number|_report_state|compare_reports|BenchmarkError|_text|_labels)\b\" src/nnx/decisions/benchmark.py` returns 0 lines
- [ ] The benchmark, pickling and decision suites pass unchanged. Check: `uv run pytest tests/test_decision_benchmark.py tests/test_decision_jobs.py tests/test_decision_jev.py tests/test_decision_nnx_adapter.py tests/test_decision_model_pilot.py tests/test_params_boundary_validation.py -k "decision or benchmark or Record"`
- [ ] The examples that import the benchmark still run. Check: `uv run pytest tests/test_examples_smoke.py -k "decision"`
- [ ] `docs/api.md` is unchanged. Check: `uv run python -m scripts.docs.build_api_reference --check && uv run python -m scripts.docs.build_docs --check`
- [ ] `uv run pytest tests/test_imports.py` passes. Check: `uv run pytest tests/test_imports.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [benchmark.py L108-L138](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L108-L138) | the shared constants, `BenchmarkError`, `_text`, `_labels` |
| [benchmark.py L141-L341](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L141-L341) | samples and perturbations |
| [benchmark.py L344-L572](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L344-L572) | records and JSONL |
| [benchmark.py L574-L826](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L574-L826) | the live collector (`collect`: complexity 34) |
| [benchmark.py L829-L1040](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L829-L1040) | metrics: the computations and the report value types interleaved |
| [benchmark.py L1043-L1332](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L1043-L1332) | evaluation and slices |
| [benchmark.py L1335-L1444](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L1335-L1444) | intervals |
| [benchmark.py L1447-L1643](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L1447-L1643) | the report and `compare_reports` |
| [api_reference_catalog.txt L253-L297](https://github.com/thekaveh/NNx/blob/9ad461e/scripts/docs/api_reference_catalog.txt#L253-L297) | the documented `nnx.decisions.benchmark.*` paths |
| #243 | the benchmark's origin |
| #558 | the frozen records and report mappings, and their pickling |

## Scope

**In:** moving code between `benchmark.py` and the three new private modules; the re-exports.

**Out:** any behaviour change; renaming public names; documenting the private modules in `docs/api.md`. Moving `nnx.decisions` into its own distribution (DEC-017 option B). Splitting `jobs.py` (FEAT-120 covers `_Runner.arun` only).

## Dependencies

**Blocked by:** DEC-017 (this ticket assumes `nnx.decisions` stays in NNx; if the owner picks option B, this issue is closed as not planned), and every ticket that edits `benchmark.py`: FEAT-060, FEAT-113, FEAT-114, FEAT-126, FEAT-127, FEAT-128, FEAT-134, FEAT-136, FEAT-141, FIX-104, FIX-105, FIX-106, DEC-016, DEC-025, FEAT-096 (it changes the `_FrozenMapping` import and edits `_FrozenDict`)
**Blocks:** none
**Related:** #243, #558, #219
