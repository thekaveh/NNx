---
id: FEAT-137
title: "[FEAT-137] Hoist needless function-local imports in offline distillation"
labels: ["enhancement", "area: training", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-61"]
depends_on: ["FEAT-118", "FEAT-126", "FEAT-127", "FEAT-134"]
related_issues: [249, 244]
status: READY
---

**Category:** Training · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-137: `src/nnx/paradigms/offline_distillation.py` has eight function-local imports, none for an optional extra or a real cycle. FEAT-126 and FEAT-118 remove four of them; this ticket moves the other four to module top, so that afterwards the module imports nothing inside a function.

## Context

**What happens today.** For example:

```python
    def __call__(self, ctx: ObjectiveContext) -> ObjectiveResult:
        from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
```

while the same name is imported under `TYPE_CHECKING` at module top (L82), and

```python
    from ..utils import _capture_training_modes, _restore_training_modes
```

inside `_score`, while the module already imports `from ..utils import _writable` at top (L78).

**Why it matters.** Function-local imports hide the module's dependencies and suggest a cycle that does not exist.

**History.** #249 (FEAT-031) set the rule: an import stays local only for an optional extra or a real cycle. #244 (FEAT-022) introduced the module. FEAT-126 replaces the JSONL pair (removing three local imports) and FEAT-118 replaces `_register_metrics`'s body (removing one); FEAT-127 and FEAT-134 edit the same import block, so they land first.

**Background for the worker.**
- Classes: **hoist** (move to module top), **keep-cycle**, **keep-optional-extra**.
- Cycle check done for this ticket: the module already imports `..objectives` at module top, which imports `..tasks`, which imports `..monitors` and `..nn.params.nn_evaluation_data_point` at module top; `..utils` is already imported at top. None of them imports `nnx.paradigms`, so none of the hoists can cycle.
- No test monkeypatches these names in `offline_distillation` (`git grep -n "offline_distillation, \"" tests/` returns 0 lines).

**Proposal.**
1. Every function-local import at `9ad461e`, with its class:

   | Line | Function | Import | Class | What to do |
   |---|---|---|---|---|
   | L261 | `write_teacher_records` | `import json` | — | removed by FEAT-126 |
   | L263 | `write_teacher_records` | `from .._artifacts import atomic_write` | — | removed by FEAT-126 |
   | L273 | `read_teacher_records` | `from .._artifacts import parse_json, read_text` | — | removed by FEAT-126 |
   | L477 | `TeacherDataset.metric_specs` | `from ..monitors import MetricSpec` | hoist | add `MetricSpec` to the module-top `from ..monitors import ...` line FEAT-118 added |
   | L577 | `OfflineDistillationObjective.__call__` | `from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint` | hoist | add a module-top `from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint` and delete the `TYPE_CHECKING` copy (L82) |
   | L740 | `_score` | `from ..utils import _capture_training_modes, _restore_training_modes` | hoist | merge into the top line: `from ..utils import _capture_training_modes, _restore_training_modes, _writable` |
   | L794 | `OfflineDistillationEval.__call__` | `from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint` | hoist | covered by the L577 top import |
   | L839 | `_register_metrics` | `from ..monitors import register_metric, registered_metrics` | — | removed by FEAT-118 |

2. Delete each hoisted local import line and any blank line it leaves at the start of a function body. Keep the `TYPE_CHECKING` import of `NNModel` (L81): it is an annotation-only import, not a function-local one.
3. Run `uv run ruff check --fix src/nnx/paradigms/offline_distillation.py` and `uv run ruff format src/nnx/paradigms/offline_distillation.py`.

No behaviour changes, no test changes, no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The module imports nothing inside a function. Check: `uv run python -c "import ast; p='src/nnx/paradigms/offline_distillation.py'; print([n.lineno for f in ast.walk(ast.parse(open(p).read())) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(f) if isinstance(n, (ast.Import, ast.ImportFrom))])"` prints `[]`
- [ ] The offline-distillation and import suites pass. Check: `uv run pytest tests/test_offline_distillation.py tests/test_imports.py tests/test_examples_smoke.py -k "offline or import or teacher"`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [offline_distillation.py L69-L83](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/offline_distillation.py#L69-L83) | the module-top imports and the `TYPE_CHECKING` block |
| [offline_distillation.py L258-L282](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/offline_distillation.py#L258-L282) | the JSONL pair's local imports (FEAT-126) |
| [offline_distillation.py L477](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/offline_distillation.py#L477) | `metric_specs`: local `MetricSpec` |
| [offline_distillation.py L577](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/offline_distillation.py#L577), [L794](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/offline_distillation.py#L794) | local `NNEvaluationDataPoint` |
| [offline_distillation.py L740](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/offline_distillation.py#L740) | `_score`: local `..utils` |
| [offline_distillation.py L839](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/offline_distillation.py#L839) | `_register_metrics`: local `..monitors` (FEAT-118) |
| #249 | the "local only for an optional extra or a cycle" rule |

## Scope

**In:** the function-local imports of `paradigms/offline_distillation.py`.

**Out:** the JSONL helpers (FEAT-126), the metric registration (FEAT-118), the other paradigm modules (not listed by SMELL-61; not ticketed).

## Dependencies

**Blocked by:** FEAT-118, FEAT-126, FEAT-127, FEAT-134 (each edits this module's imports or the functions that hold four of the local imports)
**Blocks:** none
**Related:** #249, #244
