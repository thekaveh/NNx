---
id: FEAT-106
title: "[FEAT-106] Replace magic-index module state with named fields"
labels: ["enhancement", "area: configuration", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-25"]
depends_on: []
related_issues: [562]
status: READY
---

**Category:** Configuration · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-106: Three pieces of module or object state are read by position or defined in an odd place: `models._REGISTRATIONS` is a list indexed `[0]`, `[1]`, `[2]`; the history journal writer's counters are a 6-tuple read as `[2]` and `[5]`; and `distributed._exit_hook_registered` is defined below the function that declares it `global`. This ticket gives the first two named fields (a small dataclass and a `NamedTuple`) and moves the flag above its user. Behaviour does not change.

## Context

**What happens today.** `models.py` L330–L334 and its use in `_module_registrations` (L367–L370):

```python
# _module_registrations' stamp: a token of this process's own (a stamp copied
# or unpickled with a model never equals one taken here) and the count. Then
# whether a registration was made while tracing, and the hook, once installed.
_REGISTRATIONS: list[Any] = [object(), 0, False]
_REGISTRATION_HOOK: list[Any] = []
```

```python
    if _REGISTRATIONS[2]:
        _REGISTRATIONS[2] = False
        _REGISTRATIONS[1] += 1
    return (_REGISTRATIONS[0], _REGISTRATIONS[1])
```

`history.py` `_JournalWriter` (L422–L423, L462–L465):

```python
    def _counters(self) -> tuple[int, int, int, str, int, int]:
        return (self.chunks, self.records, self.index_bytes, self.index_chain, self.epochs, self.epochs_bytes)
```

```python
        cuts = (
            (os.path.join(self.directory, _INDEX), self._committed[2]),
            (os.path.join(self.directory, _EPOCHS), self._committed[5]),
        )
```

`distributed.py` declares `global _exit_hook_registered` inside `init_process_group` (L277) and defines `_exit_hook_registered = False` after the function (L284).

**Why it matters.** `[2]` and `[5]` say nothing about what they read; reordering the tuple would silently cut the wrong file. A module global defined below its first user reads like a bug.

**History.** The registration stamp came with #562 (it tells `NNModel` when a pending lazy layer may have been registered elsewhere). Its comment in `_count_registration` (L337–L343) explains that while torch.compile traces, the hook only stores a constant, so a compiled forward gets no guard on the count. A dataclass attribute store keeps that property; `tests/test_compile_execution.py::test_a_lazy_model_in_the_process_adds_no_guard_or_recompile_to_a_compiled_fit` checks it.

**Background for the worker.**
- `_module_registrations()`: returns a `(token, count)` stamp that changes when a module bringing a pending lazy layer is registered; `_count_registration` is the torch module-registration hook that increments it.
- `_REGISTRATION_HOOK`: a one-element list used as a "hook installed" flag. Tests read it (`tests/test_external_module_runtime.py` L1261, L1264; `tests/test_compile_execution.py` L221), so it stays exactly as it is.
- `_JournalWriter._committed`: the writer's counters as of the last published manifest; `discard_epoch` rolls back to them.

**Proposal.**
1. In `src/nnx/models.py`, add `from dataclasses import dataclass` to the standard-library imports (after `import json`, L42), and replace L330–L333 (the comment and `_REGISTRATIONS: list[Any] = [object(), 0, False]`) with:

   ```python
   @dataclass
   class _RegistrationStamp:
       """_module_registrations' state: ``token``, this process's own (a stamp
       copied or unpickled with a model never equals one taken here); ``count``;
       and ``traced``, whether a registration was made while tracing."""

       token: object
       count: int
       traced: bool


   _REGISTRATIONS = _RegistrationStamp(token=object(), count=0, traced=False)
   # The registration hook, once installed.
   _REGISTRATION_HOOK: list[Any] = []
   ```

   Then in `_count_registration` replace `_REGISTRATIONS[2] = True` (L346) with `_REGISTRATIONS.traced = True` and `_REGISTRATIONS[1] += 1` (L353) with `_REGISTRATIONS.count += 1`; in `_module_registrations` replace L367–L370 with:

   ```python
       if _REGISTRATIONS.traced:
           _REGISTRATIONS.traced = False
           _REGISTRATIONS.count += 1
       return (_REGISTRATIONS.token, _REGISTRATIONS.count)
   ```

2. In `src/nnx/history.py`, add `NamedTuple` to `from typing import TYPE_CHECKING, Any, Optional, Union` (L75), and directly above `class _JournalWriter` (L399) add:

   ```python
   class _Counters(NamedTuple):
       """A journal writer's counters, as of its last published manifest."""

       chunks: int
       records: int
       index_bytes: int
       index_chain: str
       epochs: int
       epochs_bytes: int
   ```

   Change `_counters` (L422–L423) to:

   ```python
       def _counters(self) -> _Counters:
           return _Counters(self.chunks, self.records, self.index_bytes, self.index_chain, self.epochs, self.epochs_bytes)
   ```

   and in `discard_epoch` replace `self._committed[2]` (L463) with `self._committed.index_bytes` and `self._committed[5]` (L464) with `self._committed.epochs_bytes`. The tuple unpacking at L478–L480 stays (a `NamedTuple` unpacks in field order).
3. In `src/nnx/distributed.py`, cut `_exit_hook_registered = False` (L284, with the blank lines around it) and paste it directly above `def init_process_group` (L239), followed by two blank lines.
4. No test reads `_REGISTRATIONS` by index or `_committed` (`grep -rn "_REGISTRATIONS\b\|_committed\[" tests` returns nothing). No behaviour changes, so there is no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] No positional read remains. Check: `grep -n "_REGISTRATIONS\[" src/nnx/models.py` and `grep -n "_committed\[" src/nnx/history.py` both return 0 lines
- [ ] The flag is defined before its user. Check: `grep -n "^_exit_hook_registered = False\|^def init_process_group" src/nnx/distributed.py` lists `_exit_hook_registered` first
- [ ] The compiled-forward guard test and the lazy-module suite pass. Check: `uv run pytest tests/test_compile_execution.py::test_a_lazy_model_in_the_process_adds_no_guard_or_recompile_to_a_compiled_fit tests/test_external_module_runtime.py`
- [ ] The journal and process-group suites pass unchanged. Check: `uv run pytest tests/test_history_journal.py tests/test_process_group.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [models.py L330-L370](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/models.py#L330-L370) | `_REGISTRATIONS`, `_count_registration`, `_module_registrations` |
| [history.py L399-L423](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/history.py#L399-L423) | `_JournalWriter` and `_counters` |
| [history.py L452-L481](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/history.py#L452-L481) | `discard_epoch`: `[2]`, `[5]` and the unpacking |
| [distributed.py L239-L284](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/distributed.py#L239-L284) | `init_process_group` and the flag defined after it |
| [test_compile_execution.py L215-L226](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_compile_execution.py#L215-L226) | the no-guard, no-recompile test |
| [test_external_module_runtime.py L1255-L1264](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_external_module_runtime.py#L1255-L1264) | tests read `_REGISTRATION_HOOK` |
| #562 | the registration stamp's origin |

## Scope

**In:** `_RegistrationStamp`, `_Counters`, the moved flag.

**Out:** `_REGISTRATION_HOOK` (tests read it as a list). The lazy-module tracking in `NNModel` (FEAT-093).

## Dependencies

**Blocked by:** none
**Blocks:** none
**Related:** #562
