---
id: FEAT-099
title: "[FEAT-099] Share the metadata.yaml record reader"
labels: ["enhancement", "area: persistence", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-18"]
depends_on: ["FEAT-063"]
related_issues: []
status: READY
---

**Category:** Persistence · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-099: `_load_resume_status`, `_load_compile` and `_load_precision` in `nn_run.py` are the same eight lines with a different key and parser: read one record out of `metadata.yaml`, return `None` unless it is a dict, and return `None` when the parser raises. This ticket adds one helper, `_metadata_record`, and makes the three functions one-line wrappers. Behaviour does not change.

## Context

**What happens today.** `_load_compile` (`nn_run.py` L773–L782):

```python
def _load_compile(metadata: Optional[Mapping[str, Any]]) -> Optional[CompileRecord]:
    """The compile record in ``metadata.yaml`` (FEAT-029); ``None`` for an
    eager run, runs written before it, or when unreadable."""
    record = metadata.get("compile") if metadata is not None else None
    if not isinstance(record, dict):
        return None
    try:
        return CompileRecord.from_record(record)
    except (KeyError, TypeError, ValueError):
        return None
```

`_load_resume_status` (L753–L762) is the same with `"resume"` and `ResumeStatus.from_state`; `_load_precision` (L785–L794) with `"precision"` and `ResolvedPrecision.from_record`.

**Why it matters.** A fourth record (or a change to what "unreadable" means, a new exception type) must be copied three times.

**History.** No board issue covers this. `metadata.yaml` is provenance: it is never needed to reload a run, so every reader degrades to `None`.

**Background for the worker.**
- `metadata.yaml`: the run's sidecar of environment and execution records (`resume`, `compile`, `precision`, ...); `_read_metadata(path)` (L740) parses it once into a dict or `None`.
- `ResumeStatus.from_state`, `CompileRecord.from_record`, `ResolvedPrecision.from_record`: each rebuilds its record from a dict and raises `KeyError`, `TypeError` or `ValueError` for a malformed one.

**Proposal.**
1. In `src/nnx/nn/params/nn_run.py`, add `TypeVar` to `from typing import TYPE_CHECKING, Any, Optional, cast` (L18), and directly above `_load_resume_status` (L753) add:

   ```python
   _Record = TypeVar("_Record")


   def _metadata_record(
       metadata: Optional[Mapping[str, Any]], key: str, parse: Callable[[dict[str, Any]], _Record]
   ) -> Optional[_Record]:
       """The record ``metadata.yaml`` holds under ``key``, rebuilt by ``parse``;
       ``None`` when there is no metadata, no such record, a record that is not
       a mapping, or one ``parse`` refuses (``KeyError``, ``TypeError``,
       ``ValueError``): provenance, never needed to reload the run."""
       record = metadata.get(key) if metadata is not None else None
       if not isinstance(record, dict):
           return None
       try:
           return parse(record)
       except (KeyError, TypeError, ValueError):
           return None
   ```

   (`Callable` and `Mapping` are already imported from `collections.abc`, L13.)
2. Replace the bodies (after the docstrings, which stay) of:
   - `_load_resume_status` (L756–L762) with `return _metadata_record(metadata, "resume", ResumeStatus.from_state)`;
   - `_load_compile` (L776–L782) with `return _metadata_record(metadata, "compile", CompileRecord.from_record)`;
   - `_load_precision` (L788–L794) with `return _metadata_record(metadata, "precision", ResolvedPrecision.from_record)`.
3. The three names and signatures stay (`history.py` and `tests/test_history_journal.py` L1197 import `_load_resume_status`). No message or behaviour changes, so there is no CHANGELOG entry.

**Worker setup.** Branch from `develop` after FEAT-063 has merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The three readers are one-line wrappers. Check: `grep -c "_metadata_record(metadata, " src/nnx/nn/params/nn_run.py` prints `3`, and `grep -c "except (KeyError, TypeError, ValueError):" src/nnx/nn/params/nn_run.py` is 2 lower than on `develop`
- [ ] The metadata readers' suites pass unchanged. Check: `uv run pytest tests/test_run_load_exact_history.py tests/test_compile_execution.py tests/test_precision_policy.py tests/test_history_journal.py tests/test_component_resume.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [nn_run.py L740-L750](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L740-L750) | `_read_metadata`: parses the file once |
| [nn_run.py L753-L762](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L753-L762) | `_load_resume_status` |
| [nn_run.py L773-L782](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L773-L782) | `_load_compile` |
| [nn_run.py L785-L794](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L785-L794) | `_load_precision` |
| [test_history_journal.py L1197-L1203](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_history_journal.py#L1197-L1203) | imports `_load_resume_status` by name |

## Scope

**In:** `_metadata_record` and the three wrappers.

**Out:** the `run.yaml` and `idps.csv` readers (FEAT-098). `comparison._resume_mode`, which refuses an unreadable `metadata.yaml` instead of ignoring it (a different rule, kept).

## Dependencies

**Blocked by:** FEAT-063
**Blocks:** FEAT-098
**Related:** none
