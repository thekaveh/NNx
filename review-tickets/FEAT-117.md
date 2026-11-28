---
id: FEAT-117
title: "[FEAT-117] Share one _count helper across task modules (part 4 of 4)"
labels: ["enhancement", "area: training", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-38"]
depends_on: []
related_issues: [544, 231]
status: READY
---

**Category:** Training · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-117: `ranking.py` and `link_tasks.py` carry byte-identical `_count` helpers, and `tasks.py` and `lm_tasks.py` carry near-copies. This ticket adds one shared validator, `_validation.require_integer`, and turns the four helpers into one-line wrappers around it. Every message stays word for word the same.

## Context

**What happens today.** `ranking.py` and `link_tasks.py` (identical except the error class):

```python
def _count(value: Any, what: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise RankingError(f"{what} must be an integer >= {minimum}, got {value!r}")
    return int(value)
```

`tasks.py` splits the type and range messages:

```python
def _count(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TaskValidationError(f"TaskSpec {name} must be an integer, got {value!r}")
    if int(value) < minimum:
        raise TaskValidationError(f"TaskSpec {name} must be >= {minimum}, got {value!r}")
    return int(value)
```

`lm_tasks.py` has the same rule with no minimum:

```python
def _integer(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise LMTaskError(f"{what} must be an integer, got {value!r}")
    return int(value)
```

**Why it matters.** Four copies of one rule (no `bool`, any `numbers.Integral`, returned as a plain `int`) can drift apart.

**History.** These are #544 leftovers (FIX-076 moved hand-written validators onto the shared rule). `require_count` (#231) is not used here because its wording (`"nnx.ranking requires k to be an integer >= 1, got 0 of type int"`) differs from these modules' wording, which `tests/test_ranking_task.py` L115-L116 pin (`"k must be an integer >= 1"`).

**Background for the worker.**
- `_validation.py` (`src/nnx/_validation.py`): standard-library-only validators shared across the package; `require_count` is its `owner requires field` count check.
- `{value!r}` is kept (not `shown(value)`), so even the text for a 100-digit integer is unchanged.

**Proposal.**
1. In `src/nnx/_validation.py`, add this function directly after `require_count` (after L179):

   ```python
   def require_integer(value: Any, what: str, *, error: type[Exception], minimum: Optional[int] = None) -> int:
       """``value`` as a plain ``int`` once it is a non-``bool``
       :class:`numbers.Integral` (NumPy integers included) of at least
       ``minimum``. Otherwise raises ``error("<what> must be an integer >=
       <minimum>, got <value!r>")``, or ``error("<what> must be an integer,
       got <value!r>")`` when ``minimum`` is ``None`` — the wording of the task
       modules (:func:`require_count` is the ``owner requires field`` form)."""
       if isinstance(value, bool) or not isinstance(value, numbers.Integral) or (minimum is not None and value < minimum):
           bound = "" if minimum is None else f" >= {minimum}"
           raise error(f"{what} must be an integer{bound}, got {value!r}")
       return int(value)
   ```

2. `src/nnx/ranking.py`: replace the body of `_count` (L86-L89) so the function reads:

   ```python
   def _count(value: Any, what: str, *, minimum: int) -> int:
       return require_integer(value, what, error=RankingError, minimum=minimum)
   ```

   Add `from ._validation import require_integer` to the relative imports.

3. `src/nnx/link_tasks.py`: the same change to `_count` (L112-L115), with `error=LinkTaskError`; add `from ._validation import require_integer`. Leave `link_tasks._integer` (L143, the `operator.index` normalizer) alone: it is a different function.

4. `src/nnx/tasks.py`: replace `_count` (L97-L102) with:

   ```python
   def _count(value: Any, name: str, *, minimum: int) -> int:
       count = require_integer(value, f"TaskSpec {name}", error=TaskValidationError)
       if count < minimum:
           raise TaskValidationError(f"TaskSpec {name} must be >= {minimum}, got {value!r}")
       return count
   ```

   Add `require_integer` to the existing `from ._validation import checked, plain_str, require_finite_real, shown` line (L64).

5. `src/nnx/lm_tasks.py`: replace the body of `_integer` (L85-L88) so it reads:

   ```python
   def _integer(value: Any, what: str) -> int:
       return require_integer(value, what, error=LMTaskError)
   ```

   Add `from ._validation import require_integer`.

6. Keep `import numbers` in all four modules: each still uses `numbers` elsewhere (ranking L156, link_tasks L547, tasks L148, lm_tasks L204).
7. Run `uv run ruff check --fix src/nnx/` (sorts the import blocks) and `uv run ruff format src/nnx/`.

All call sites keep calling `_count` / `_integer` with the same arguments. Messages, error classes and returned values are unchanged, so no test changes and no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] `src/nnx/_validation.py` defines `require_integer(value, what, *, error, minimum=None) -> int`. Check: `grep -n "def require_integer" src/nnx/_validation.py` returns 1 line
- [ ] No task module hand-rolls the rule any more. Check: `grep -n "not isinstance(value, numbers.Integral)" src/nnx/ranking.py src/nnx/link_tasks.py src/nnx/tasks.py src/nnx/lm_tasks.py` returns 0 lines
- [ ] A new test, `tests/test_integer_config_validation.py::test_require_integer_keeps_the_task_module_wording`, asserts: `require_integer(np.int64(3), "k", error=KeyError, minimum=1) == 3` and is a plain `int`; `require_integer(0, "k", error=KeyError, minimum=1)` raises `KeyError` matching `"k must be an integer >= 1, got 0"`; `require_integer(True, "k", error=KeyError)` raises matching `"k must be an integer, got True"`; `require_integer(2.0, "k", error=KeyError, minimum=1)` raises. The test imports `from nnx._validation import require_integer` inside its body (the file already imports `numpy as np`). Check: `uv run pytest tests/test_integer_config_validation.py::test_require_integer_keeps_the_task_module_wording`
- [ ] The pinned ranking messages still match. Check: `uv run pytest tests/test_ranking_task.py::test_the_task_declares_and_validates_its_configuration`
- [ ] The task suites pass. Check: `uv run pytest tests/test_ranking_task.py tests/test_link_tasks.py tests/test_causal_lm_task.py tests/test_task_adapters.py tests/test_integer_config_validation.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [ranking.py L86-L89](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/ranking.py#L86-L89) | `_count` with `RankingError` |
| [link_tasks.py L112-L115](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/link_tasks.py#L112-L115) | the byte-identical `_count` with `LinkTaskError` |
| [tasks.py L97-L102](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/tasks.py#L97-L102) | `_count` with split type / range messages |
| [lm_tasks.py L85-L88](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/lm_tasks.py#L85-L88) | `_integer`: the same rule, no minimum |
| [_validation.py L140-L179](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_validation.py#L140-L179) | `require_count`: the shared rule in the `owner requires field` wording |
| [test_ranking_task.py L115-L116](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_ranking_task.py#L115-L116) | pins `"k must be an integer >= 1"` |
| #544 | the hand-written-validator sweep these sites were left out of |
| #231 | introduced `require_count` |

## Scope

**In:** `require_integer` and the four task-module helpers.

**Out:** the viz checks (FEAT-116), the decisions checks (FEAT-114), the generation / diffusion / plans checks (FEAT-115). Switching these modules to the `require_count` wording (a message change).

## Dependencies

**Blocked by:** none
**Blocks:** none
**Related:** #544, #231
