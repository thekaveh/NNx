---
id: FEAT-127
title: "[FEAT-127] Define the 1e-6 probability tolerance once"
labels: ["enhancement", "area: numerics", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-50"]
depends_on: []
related_issues: [211, 219, 244]
status: READY
---

**Category:** Numerics · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-127: the "a distribution sums to 1 within 1e-6" tolerance is written as a literal three times: `_probability.SUM_TOLERANCE`, `decisions.schema.PROBABILITY_TOLERANCE` and `paradigms.offline_distillation.PROBABILITY_TOLERANCE`. This ticket makes the two public constants refer to `_probability.SUM_TOLERANCE`, so the value lives in one place. The public names and values do not change.

## Context

**What happens today.**

```python
SUM_TOLERANCE = 1e-6
"""Smallest row-sum tolerance of :func:`categorical_probabilities`."""
```

(`src/nnx/_probability.py` L44-L45)

```python
PROBABILITY_TOLERANCE = 1e-6
"""How far a distribution's sum may be from 1. Never renormalized."""
```

(`src/nnx/decisions/schema.py` L33-L34) and `PROBABILITY_TOLERANCE = 1e-6` (`src/nnx/paradigms/offline_distillation.py` L105).

**Why it matters.** The three are equal by specification (#211's probability results, #219's decision results, #244's teacher records all accept a sum within 1e-6 of 1); a literal per module lets one drift.

**History.** The three are equal by specification (#211, #219, #244); no issue decided they should differ.

**Background for the worker.**
- `_probability.py`: the shared probability validation of calibration, abstention, predictions and task adapters. `SUM_TOLERANCE` is the floor of `categorical_probabilities`'s row-sum bound (the bound grows with the dtype's precision and the class count, never below the floor).
- `PROBABILITY_TOLERANCE` is public in two places: `nnx.decisions.PROBABILITY_TOLERANCE` (exported from `nnx/decisions/__init__.py`) and `nnx.paradigms.offline_distillation.PROBABILITY_TOLERANCE` (in that module's `__all__`). Both keep their names and the value `1e-06`.
- `decisions/schema.py` imports only the standard library today; `_probability` imports NumPy, which `nnx.decisions` already loads (`providers.py` and `nli.py` import it at module top), and which imports nothing from `nnx.decisions`.

**Proposal.**
1. In `src/nnx/_probability.py`, replace the docstring of `SUM_TOLERANCE` (L45) with:

   ```python
   """How far a probability row's sum may be from 1: the smallest row-sum
   tolerance of :func:`categorical_probabilities`, and the fixed tolerance of a
   decision distribution (``nnx.decisions.PROBABILITY_TOLERANCE``) and a
   teacher record (``nnx.paradigms.offline_distillation.PROBABILITY_TOLERANCE``)."""
   ```

2. In `src/nnx/decisions/schema.py`, add `from .._probability import SUM_TOLERANCE` after the standard-library imports, and replace `PROBABILITY_TOLERANCE = 1e-6` (L33) with `PROBABILITY_TOLERANCE = SUM_TOLERANCE`. Keep its docstring.
3. In `src/nnx/paradigms/offline_distillation.py`, add `from .._probability import SUM_TOLERANCE` to the relative imports and replace `PROBABILITY_TOLERANCE = 1e-6` (L105) with `PROBABILITY_TOLERANCE = SUM_TOLERANCE`.
4. Add to `tests/test_decision_schema.py`:

   ```python
   def test_the_probability_tolerances_are_one_constant():
       from nnx import _probability
       from nnx.paradigms import offline_distillation

       assert PROBABILITY_TOLERANCE == offline_distillation.PROBABILITY_TOLERANCE == _probability.SUM_TOLERANCE == 1e-6
   ```

   (`PROBABILITY_TOLERANCE` is already imported at the top of that file.)
5. Run `uv run ruff check --fix src/nnx/` and `uv run ruff format src/nnx/ tests/`.

No value changes, so no CHANGELOG entry and `docs/api.md` stays as it is.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] No module writes the tolerance as a literal outside `_probability.py`. Check: `grep -n "TOLERANCE = 1e-6" src/nnx -r` returns 1 line (`_probability.py`)
- [ ] The new test passes. Check: `uv run pytest tests/test_decision_schema.py::test_the_probability_tolerances_are_one_constant`
- [ ] The decision, offline-distillation and probability suites pass. Check: `uv run pytest tests -k "decision_schema or offline_distillation or decision_nli or probability or calibration"`
- [ ] `uv run pytest tests/test_imports.py` passes (no import cycle). Check: that command
- [ ] `docs/api.md` is unchanged. Check: `uv run python -m scripts.docs.build_api_reference --check`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [_probability.py L43-L45](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_probability.py#L43-L45) | `SUM_TOLERANCE = 1e-6` |
| [schema.py L33-L34](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/schema.py#L33-L34) | `PROBABILITY_TOLERANCE = 1e-6` (decisions) |
| [offline_distillation.py L105](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/offline_distillation.py#L105) | `PROBABILITY_TOLERANCE = 1e-6` (teacher records) |
| #211, #219, #244 | the three specs that fix 1e-6 |

## Scope

**In:** the two constant definitions and `SUM_TOLERANCE`'s docstring.

**Out:** a shared `check_probabilities` validator for `TeacherRecord.__post_init__` and `schema._check_distribution` (SMELL-50 step 2: their messages differ; not ticketed), routing `nli._softmax` through `_probability.softmax_` (SMELL-50 step 3; not ticketed), and the benchmark's `_real` / `_finite_probability` (FEAT-121 moves them).

## Dependencies

**Blocked by:** none
**Blocks:** FEAT-121, FEAT-137
**Related:** #211, #219, #244
