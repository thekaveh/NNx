---
id: FEAT-060
title: "[FEAT-060] Move abstention.decide into nnx.decisions and break the import cycle"
labels: ["enhancement", "area: decisions", "priority: P3", "review: 2026-11-27"]
sources: ["SCOPE-01", "SMELL-42"]
depends_on: ["FEAT-059"]
related_issues: [219, 263, 245]
status: READY
---

**Category:** Decisions · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-060: `nnx.abstention` imports `nnx.decisions` inside `decide`, and `nnx.decisions` imports `nnx.abstention` inside six functions, so neither side can import the other at module level. Move the body of `abstention.decide` into a new `nnx/decisions/selective.py`, keep `abstention.decide` as a thin delegating wrapper, import `nnx.abstention` at the top of the decision modules, and give the two `jobs` helpers that `result.py` imports non-private names.

## Context

**What happens today.** `abstention.decide` reaches into `nnx.decisions` at call time:

```python
def decide(result: Any, policy: AbstentionPolicy, *, model_id: str) -> SelectiveDecision:
    """Apply ``policy`` to a typed decision (``nnx.decisions``).
```

```python
    from .decisions.schema import ChoiceResult, ScoreResult

    if not isinstance(result, (ChoiceResult, ScoreResult)):
```

and the decision modules reach back into `nnx.abstention` at call time, for example in `DecisionJob.ask`:

```python
        if policy is not None:
            from ..abstention import AbstentionPolicy

            if not isinstance(policy, AbstentionPolicy):
                raise InvalidJob(f"policy must be an nnx.abstention.AbstentionPolicy, got {type(policy).__name__}")
```

`result.decide_result` imports two private helpers of `jobs`:

```python
    context = {"question": question.digest(), "provider": type(provider).__name__}
    from .decisions.jobs import _check_results, _rows
```

**Why it matters.** The two-way lazy imports hide a cycle (decisions ↔ abstention ↔ result) that makes every import in these modules function-local, and `result.py` depends on names `jobs` marks private.

**History.** #219 put `nnx.decisions` inside NNx; #245 added `DecisionJob` with abstention policies; #263 added `decide_result` in `result.py`. FEAT-059 made `nnx.decisions` load on first access. That is why `result.py` must keep its decision imports inside its functions: `nnx.result` is imported by every `import nnx`, so a module-level import of `nnx.decisions.jobs` there would load `nnx.decisions` eagerly again and fail FEAT-059's test. This ticket therefore renames the helpers but leaves those imports where they are.

**Background for the worker.**
- *`abstention.decide`*: public API (in `nnx.abstention.__all__` and `docs/api.md` §`nnx.abstention.decide`). It applies an `AbstentionPolicy` to a `ChoiceResult` / `ScoreResult` and returns a `SelectiveDecision`.
- *`SelectiveDecision`*: the frozen dataclass `decide` returns. It stays in `nnx.abstention` so its class path (and any pickle of it) does not change.
- *Lazy wrapper*: a function that imports its implementation inside its body, at call time. `abstention.decide` becomes one, so `import nnx.abstention` never loads `nnx.decisions` (FEAT-059 needs that), while `nnx.decisions` may import `nnx.abstention` at module level.
- *`_check_results` / `_rows`*: `jobs` helpers that check a provider's answers against a question and count an input's rows; `DecisionJob` and `result.decide_result` both use them.

**Proposal.**

1. Create `src/nnx/decisions/selective.py`:
   ```python
   """Apply an abstention policy to a typed decision (``nnx.abstention.decide``).

   Lives in ``nnx.decisions`` so this package can import ``nnx.abstention`` at
   module level; ``nnx.abstention.decide`` delegates here. Internal; the public
   entry point is ``nnx.abstention.decide``.
   """

   from __future__ import annotations

   from typing import Any

   import numpy as np

   from ..abstention import AbstentionPolicy, Outcome, SelectiveDecision, _id, _labels, _scores
   from .schema import ChoiceResult, ScoreResult


   def decide(result: Any, policy: AbstentionPolicy, *, model_id: str) -> SelectiveDecision:
       """See :func:`nnx.abstention.decide`."""
   ```
   and give it, as its body, the body of `abstention.decide` from L1462–L1495 (`if not isinstance(result, (ChoiceResult, ScoreResult)):` through the final `return SelectiveDecision(...)`), unchanged except that the local `from .decisions.schema import ChoiceResult, ScoreResult` line is dropped (the module imports them at top). Every error message stays word for word.
2. In `src/nnx/abstention.py`, keep the signature and the docstring of `decide` (L1450–L1461) exactly as they are, and replace its body (L1462–L1495) with:
   ```python
       # nnx.decisions imports this module at module level, and `import nnx`
       # must not load nnx.decisions (FEAT-059): import the implementation here.
       from .decisions.selective import decide as _decide

       return _decide(result, policy, model_id=model_id)
   ```
3. In `src/nnx/decisions/jobs.py`:
   - add `from ..abstention import AbstentionPolicy, AbstentionSchemaError` and `from .selective import decide` to the module-level imports (beside `from .providers import _synchronous`, L87);
   - delete the four local imports at L429 (`from ..abstention import AbstentionPolicy`), L437 (`from ..abstention import AbstentionSchemaError`), L537 (`from ..abstention import AbstentionPolicy`) and L974 (`from ..abstention import decide`), keeping the code that used them;
   - rename `def _rows(state: Any)` (L602) to `def input_rows(state: Any)` and `def _check_results(...)` (L633) to `def check_results(...)`, update the two calls at L931 and L933, and add directly after `check_results`:
     ```python
     # Former private names, kept importable.
     _rows = input_rows
     _check_results = check_results
     ```
4. In `src/nnx/decisions/benchmark.py`:
   - add `from ..abstention import AbstentionPolicy, AbstentionSchemaError` and `from .selective import decide` to the module-level imports (beside `from .providers import _synchronous`, L66);
   - delete the local imports at L1117 (`from ..abstention import AbstentionSchemaError, decide`) and L1263 (`from ..abstention import AbstentionPolicy`), keeping the code that used them. Leave every other function-local import in this file alone (FEAT-136 hoists them).
5. In `src/nnx/result.py`, change L296 to `    from .decisions.jobs import check_results, input_rows` and the two uses at L298 and L302 to `input_rows(inputs)` and `check_results(...)`. Keep the import inside the function (see History).
6. Add this test at the end of `tests/test_imports.py`:
   ```python
   def test_nnx_decisions_imports_abstention_only_at_module_level():
       """FEAT-060: abstention.decide's body lives in nnx.decisions.selective, so
       nnx.decisions imports nnx.abstention at module level, never in a function."""
       import ast
       import pathlib

       import nnx.decisions

       root = pathlib.Path(nnx.decisions.__file__).parent
       local = []
       for path in sorted(root.glob("*.py")):
           tree = ast.parse(path.read_text(encoding="utf-8"))
           for function in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
               for node in ast.walk(function):
                   if isinstance(node, ast.ImportFrom) and node.level == 2 and node.module == "abstention":
                       local.append(f"{path.name}:{node.lineno}")
       assert local == []
   ```

No test monkeypatches `abstention.decide`, `jobs._rows` or `jobs._check_results` (`grep -rn "_check_results\|_rows\b\|setattr(.*abstention" tests` finds only `tests/test_decision_benchmark.py:458`, which patches `nnx.abstention.select_threshold` as a must-not-be-called sentinel; the benchmark never imports that name, so it keeps working). No behaviour changes, so no CHANGELOG entry. `abstention.decide`'s docstring is unchanged, so `docs/api.md` needs no rebuild.

**Worker setup.** Branch from `develop` after FEAT-059 has merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The new test `tests/test_imports.py::test_nnx_decisions_imports_abstention_only_at_module_level` fails on `develop` (it lists `jobs.py` and `benchmark.py` lines) and passes after the change. Check: `uv run pytest tests/test_imports.py::test_nnx_decisions_imports_abstention_only_at_module_level`
- [ ] FEAT-059's laziness still holds: `import nnx` does not load `nnx.decisions`. Check: `uv run pytest tests/test_imports.py::test_import_nnx_loads_decisions_and_export_conformance_on_first_access`
- [ ] `abstention.decide` keeps its signature and docstring, and the API reference is unchanged. Check: `uv run python -m scripts.docs.build_api_reference --check`
- [ ] `nnx.decisions.jobs._rows` and `nnx.decisions.jobs._check_results` still import. Check: `uv run python -c "from nnx.decisions.jobs import _rows, _check_results, input_rows, check_results; assert _rows is input_rows and _check_results is check_results"`
- [ ] The abstention, result, decision and import suites pass. Check: `uv run pytest tests -k "abstention or result_boundaries or decision or imports"`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [abstention.py L1450-L1495](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/abstention.py#L1450-L1495) | `decide` and its local `nnx.decisions.schema` import (L1462) |
| [abstention.py L1427-L1447](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/abstention.py#L1427-L1447) | `SelectiveDecision`, which stays in `nnx.abstention` |
| [jobs.py L429](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/jobs.py#L429), [L437](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/jobs.py#L437), [L537](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/jobs.py#L537), [L974](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/jobs.py#L974) | The four local `from ..abstention import` lines in `DecisionJob.ask`, `_from_state` and `_Runner._record` |
| [jobs.py L602-L640](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/jobs.py#L602-L640) | `_rows` and `_check_results` |
| [benchmark.py L1117](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L1117), [L1263](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L1263) | The two local `from ..abstention import` lines in `_selective` and `evaluate` |
| [result.py L294-L302](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/result.py#L294-L302) | `decide_result` imports `_check_results` and `_rows` |
| [test_decision_benchmark.py L458](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_decision_benchmark.py#L458) | The only test patch on `nnx.abstention` (a sentinel the change does not affect) |
| #219, #245, #263 | The decision API, `DecisionJob` policies and `decide_result` |

## Scope

**In:** `nnx/decisions/selective.py` (new), `abstention.decide`'s body, the six abstention imports in `jobs.py` and `benchmark.py`, the `jobs` helper names, `result.py`'s import line, one test.

**Out:**
- Making `decide` part of `nnx.decisions.__all__`.
- Hoisting the other function-local imports of `jobs.py` and `benchmark.py`: FEAT-136.
- Moving `result.py`'s decision imports to module level (blocked by FEAT-059's laziness).
- Extracting `nnx.decisions` into its own distribution: DEC-017.

## Dependencies

**Blocked by:** FEAT-059
**Blocks:** FEAT-136, DEC-017
**Related:** #219, #245, #263
