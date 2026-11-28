---
id: FEAT-134
title: "[FEAT-134] Share one non-empty-text validator"
labels: ["enhancement", "area: configuration", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-58"]
depends_on: ["FEAT-114"]
related_issues: [558]
status: READY
---

**Category:** Configuration · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-134: four modules hand-write "a non-empty string" checks: `decisions/benchmark.py` and `comparison.py` (identical apart from the error class), `paradigms/offline_distillation.py` and `decisions/providers.py`. This ticket adds `_validation.required_text(value, what, *, error, optional=False, strip=True)` and makes the four delegate to it. Each keeps its rule (whether whitespace-only text is refused) and its message. `_validation.required_id` is not changed.

## Context

**What happens today.** `benchmark.py` L129-L134 (and `comparison.py` L99-L104 with `ComparisonError`):

```python
def _text(value: Any, what: str, *, optional: bool = False) -> Optional[str]:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{what} must be a non-empty string{' or None' if optional else ''}, got {value!r}")
    return value
```

`offline_distillation.py` L127-L130 and `providers.py` L76-L78 refuse only `""`:

```python
def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise TeacherRecordError(f"{what} must be a non-empty string, got {value!r}")
    return value
```

```python
def _check_name(name: Any) -> None:
    if not isinstance(name, str) or not name:
        raise InvalidDecisionRequest(f"a provider name must be a non-empty string, got {name!r}")
```

**Why it matters.** Four copies of one rule with two variants; a shared helper makes the variant an explicit argument.

**History.** #558 (FIX-090) added `add_context`'s blank-`kind` refusal through the benchmark's `_text` (`tests/test_decision_benchmark.py` L1096 pins `"kind must be a non-empty string"` for `None`, `""`, `"   "` and `3`). `_validation.required_id` (L246) refuses only `""` for ids; its callers rely on that, so it stays untouched.

**Background for the worker.**
- **`optional=` semantics:** with `optional=True`, `None` is accepted and returned as `None`, and the refusal says `"must be a non-empty string or None"`; with `optional=False` (the default), `None` is refused like any non-string and the message has no `" or None"`.
- **`strip=` semantics:** with `strip=True` (the default) a whitespace-only string (`"   "`) is refused; with `strip=False` only `""` is. The accepted value is returned as given, never stripped.
- **`required_id` does not change:** it keeps its own body, its `""`-only rule and its message; no caller moves to `required_text`.

**Proposal.**
1. In `src/nnx/_validation.py`, add `Literal` and `overload` to the `typing` import (L47), and after `required_id` add:

   ```python
   @overload
   def required_text(
       value: Any, what: str, *, error: type[Exception], optional: Literal[False] = False, strip: bool = True
   ) -> str: ...


   @overload
   def required_text(
       value: Any, what: str, *, error: type[Exception], optional: bool, strip: bool = True
   ) -> Optional[str]: ...


   def required_text(
       value: Any, what: str, *, error: type[Exception], optional: bool = False, strip: bool = True
   ) -> Optional[str]:
       """``value`` once it is a non-empty ``str``, else ``error("<what> must be
       a non-empty string, got <value!r>")``. ``optional=True`` also accepts
       ``None`` (returned as is; the message then says ``"or None"``);
       ``strip=True`` also refuses whitespace-only text, ``strip=False`` only
       ``""``. The text is returned as given. (:func:`required_id` is the ids'
       rule and stays separate.)"""
       if value is None and optional:
           return None
       if not isinstance(value, str) or not (value.strip() if strip else value):
           raise error(f"{what} must be a non-empty string{' or None' if optional else ''}, got {value!r}")
       return value
   ```

2. `src/nnx/decisions/benchmark.py`: add `from .._validation import required_text` to the module-top imports (`ruff check --fix` merges it with any other `from .._validation import` line), and replace the body of `_text` with `return required_text(value, what, error=BenchmarkError, optional=optional)`.
3. `src/nnx/comparison.py`: add `from ._validation import required_text`; replace the body of `_text` with `return required_text(value, what, error=ComparisonError, optional=optional)`.
4. `src/nnx/paradigms/offline_distillation.py`: add `required_text` to the existing `from .._validation import checked, require_count, shown` line (L75); replace the body of `_text` with `return required_text(value, what, error=TeacherRecordError, strip=False)`.
5. `src/nnx/decisions/providers.py`: add `from .._validation import required_text`; replace the body of `_check_name` with `required_text(name, "a provider name", error=InvalidDecisionRequest, strip=False)`.
6. Run `uv run ruff check --fix src/nnx/` and `uv run ruff format src/nnx/`.

Every message is the same string as before (`"<what> must be a non-empty string[ or None], got <repr>"`, with `what="a provider name"` for providers). No test changes, no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The four modules no longer hand-write the check. Check: `grep -n "not value.strip()\|or not value:\|or not name:" src/nnx/decisions/benchmark.py src/nnx/comparison.py src/nnx/paradigms/offline_distillation.py src/nnx/decisions/providers.py` returns 0 lines (4 today)
- [ ] `required_id` keeps its rule: the new test below also asserts `required_id("  ", "x", error=KeyError) == "  "` and that `required_id("", "x", error=KeyError)` raises matching `"x must be a non-empty string, got ''"`, and `git diff origin/develop -- src/nnx/_validation.py` shows no removed line inside `required_id`. Check: the test below
- [ ] A new test `tests/test_integer_config_validation.py::test_required_text_keeps_each_rule` asserts: `required_text("  ", "x", error=KeyError)` raises matching `"x must be a non-empty string, got '  '"`; `required_text("  ", "x", error=KeyError, strip=False) == "  "`; `required_text(None, "x", error=KeyError, optional=True) is None`; `required_text(3, "x", error=KeyError, optional=True)` raises matching `"x must be a non-empty string or None, got 3"`; `required_text("", "x", error=KeyError, strip=False)` raises (it imports `from nnx._validation import required_id, required_text` in its body). Check: `uv run pytest tests/test_integer_config_validation.py::test_required_text_keeps_each_rule`
- [ ] The pinned messages pass unchanged, including #558's blank-kind test. Check: `uv run pytest tests/test_decision_benchmark.py::test_add_context_refuses_a_missing_or_blank_kind tests -k "decision or comparison or offline_distillation"`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [benchmark.py L129-L134](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L129-L134) | `_text`: refuses blank text, `optional=` |
| [comparison.py L99-L104](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/comparison.py#L99-L104) | the identical copy with `ComparisonError` |
| [offline_distillation.py L127-L130](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/offline_distillation.py#L127-L130) | `_text`: refuses only `""` |
| [providers.py L76-L78](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/providers.py#L76-L78) | `_check_name`: refuses only `""` |
| [_validation.py L246-L250](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_validation.py#L246-L250) | `required_id`: unchanged |
| [test_decision_benchmark.py L1090-L1096](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_decision_benchmark.py#L1090-L1096) | #558's pin of the benchmark wording |
| #558 | the blank-`kind` refusal |

## Scope

**In:** `required_text` and the four delegating helpers.

**Out:** `required_id` and its callers (deliberately `""`-only). Moving offline distillation or providers to the blank-text rule (a behaviour change; not ticketed). Splitting `benchmark.py` (FEAT-121).

## Dependencies

**Blocked by:** FEAT-114 (it edits `providers.py` next to `_check_name` and the benchmark count checks)
**Blocks:** FEAT-121, FEAT-137
**Related:** #558
