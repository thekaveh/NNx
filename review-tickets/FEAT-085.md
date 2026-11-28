---
id: FEAT-085
title: "[FEAT-085] Share the JSON parse wrappers of history and provenance"
labels: ["enhancement", "area: persistence", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-07"]
depends_on: ["FIX-122"]
related_issues: [551, 762, 521]
status: READY
---

**Category:** Persistence · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-085: `history._loads` and `provenance._load_json` each wrap `json.loads` in the same depth guard (the parser's own `RecursionError`, then `too_deep` on the parsed value). Put that wrapper once in `nnx/_json_depth.py`, as `loads_within(data, deep, max_depth=MAX_JSON_DEPTH)`, and make both readers call it. Strictness, exception types and messages stay exactly as they are: both files keep loading `NaN`, `Infinity`, `1e999` and repeated keys as `json` does.

## Context

**What happens today.** `history.py` L183:

```python
def _loads(data: Union[str, bytes], what: str) -> Any:
    """``json.loads`` for a journal file, refusing nesting past
    ``MAX_JSON_DEPTH`` with :class:`HistoryCorruptionError`, the same on
    every Python version: the parser's own limit (a ``RecursionError``)
    varies by version and stack size, and below it a deep value parsed."""
```

```python
    deep = f"{what} is nested more than {MAX_JSON_DEPTH} levels deep"
    try:
        value = json.loads(data)
    except RecursionError:  # past the parser's own limit
        raise HistoryCorruptionError(deep) from None
    if too_deep(value, MAX_JSON_DEPTH):
        raise HistoryCorruptionError(deep)
    return value
```

and `provenance.py` L590, whose `RecursionError` sentinel `load_provenance` (L586–L587) turns into `ValueError("run …: provenance JSON nested too deeply to read")`:

```python
    value = json.load(handle)
    if too_deep(value, MAX_JSON_DEPTH):
        raise RecursionError("provenance JSON is nested too deeply")
    return value
```

**Why it matters.** Two copies of the version-independent depth rule; a third reader would copy it again.

**History.**
- #551 (FIX-083) introduced the depth bound on read and write; its tests (`tests/test_history_journal.py` L1481–L1560, `tests/test_provenance.py` L351–L357) pin both readers and must pass unchanged.
- #762 (DEC-008) decided how files from NNx 0.4.1 nested past the new bounds load; nothing here changes that.
- #521 made the artifact readers strict through `_artifacts.parse_json`. The finding proposed moving history and provenance onto `parse_json`, which also refuses `NaN`, `1e999` and repeated keys and rewraps malformed text in its own error; that would make journals and provenance files written today stricter to read and change their error messages. This ticket keeps their strictness and messages unchanged and shares only the depth wrapper.
- FIX-122 moves the bundle reader onto `parse_json` (with `bundles._MAX_DEPTH = 64` kept); it lands first and is not touched here.

**Background for the worker.**
- *Depth guard*: the parser's own nesting limit varies by Python version and stack size, so NNx refuses any value nested past `MAX_JSON_DEPTH` (256) itself: a `RecursionError` from the parser and a deep value that parsed both raise the reader's own error.
- *`too_deep(value, limit)`*: in `src/nnx/_json_depth.py`; measures nesting without recursion.
- *Lax JSON*: plain `json.loads`, which accepts `NaN`, `Infinity`, numbers that overflow to `inf`, and repeated keys (last wins). The history journal writes lines with `allow_nan=True`, so its reader must stay lax.

**Proposal.**

1. In `src/nnx/_json_depth.py`, add `import json`, `from collections.abc import Callable` and `from typing import Union` to the imports, and after `too_deep` add:
   ```python
   def loads_within(
       data: Union[str, bytes], deep: Callable[[], BaseException], max_depth: int = MAX_JSON_DEPTH
   ) -> Any:
       """``json.loads(data)``, lax as ``json`` is (``NaN``, ``Infinity``,
       overflowing numbers and repeated keys load), refusing a value nested
       past ``max_depth`` levels with the exception ``deep()`` returns, the same
       on every Python version: the parser's own ``RecursionError`` (raised
       ``from None``) and a deep value that parsed both raise it. Malformed text
       raises ``json.JSONDecodeError`` as ``json.loads`` does."""
       try:
           value = json.loads(data)
       except RecursionError:  # past the parser's own limit
           raise deep() from None
       if too_deep(value, max_depth):
           raise deep()
       return value
   ```
2. In `src/nnx/history.py`, change the import to `from ._json_depth import MAX_JSON_DEPTH, loads_within, too_deep` (drop `too_deep` if ruff reports it unused) and replace the body of `_loads` after its docstring (L188–L197) with:
   ```python
       # FIX-083: a deep journal.json raised a raw RecursionError, and a valid
       # manifest with a deep extra key was silently accepted.
       return loads_within(data, lambda: HistoryCorruptionError(f"{what} is nested more than {MAX_JSON_DEPTH} levels deep"))
   ```
   The function keeps its name and signature (three callers in `history.py`).
3. In `src/nnx/provenance.py`, import `loads_within` from `._json_depth` and replace the body of `_load_json` after its docstring (L594–L597) with:
   ```python
       # The sentinel load_provenance turns into its "nested too deeply" ValueError.
       return loads_within(handle.read(), lambda: RecursionError("provenance JSON is nested too deeply"))
   ```
   `load_provenance`'s `except RecursionError` (L586–L587) stays: it also covers recursion outside the parse.
4. Add `tests/test_json_depth.py`:
   ```python
   """FEAT-085: nnx._json_depth.loads_within, the shared depth-guarded JSON loader."""

   import json
   import math

   import pytest

   from nnx._json_depth import MAX_JSON_DEPTH, loads_within


   class Deep(Exception):
       pass


   def test_loads_within_is_as_lax_as_json():
       value = loads_within('{"a": NaN, "b": 1e999, "c": 1, "c": 2}', Deep)
       assert math.isnan(value["a"]) and value["b"] == math.inf and value["c"] == 2


   def test_loads_within_refuses_a_value_past_the_bound_and_past_the_parser():
       assert loads_within("[" * MAX_JSON_DEPTH + "]" * MAX_JSON_DEPTH, Deep) is not None
       with pytest.raises(Deep):
           loads_within("[" * (MAX_JSON_DEPTH + 1) + "]" * (MAX_JSON_DEPTH + 1), Deep)
       with pytest.raises(Deep) as refused:
           loads_within("[" * 200_000 + "]" * 200_000, Deep)
       assert refused.value.__cause__ is None and refused.value.__suppress_context__


   def test_loads_within_leaves_malformed_text_to_json():
       with pytest.raises(json.JSONDecodeError):
           loads_within("{", Deep)
   ```

No behaviour changes, so no CHANGELOG entry.

**Worker setup.** Branch from `develop` after FIX-122 has merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The new tests in `tests/test_json_depth.py` pass (they fail on `develop`, where `loads_within` does not exist). Check: `uv run pytest tests/test_json_depth.py`
- [ ] Neither reader keeps its own depth guard. Check: `grep -n "except RecursionError" src/nnx/history.py` prints nothing, and `grep -n "too_deep(value" src/nnx/history.py src/nnx/provenance.py` prints nothing
- [ ] #551's history and provenance tests pass unchanged, with the same messages. Check: `uv run pytest tests -k "history_journal or provenance or attempt_provenance or run_bundles or bundle_validation or calibration or finite_config_validation"`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [history.py L183-L197](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/history.py#L183-L197) | `_loads` and its depth guard |
| [history.py L618-L631](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/history.py#L618-L631) | The manifest reader's handling of `HistoryCorruptionError` and `ValueError`, which must see the same exceptions |
| [provenance.py L586-L597](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/provenance.py#L586-L597) | `load_provenance`'s conversion and `_load_json`'s sentinel |
| [_json_depth.py L28-L41](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_json_depth.py#L28-L41) | `MAX_JSON_DEPTH` and `too_deep` |
| [_artifacts.py L82-L114](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_artifacts.py#L82-L114) | The strict `parse_json`, left unchanged |
| [test_history_journal.py L1535-L1540](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_history_journal.py#L1535-L1540), [test_provenance.py L351-L357](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_provenance.py#L351-L357) | #551's message tests |
| #551, #762, #521 | The depth bound; the 0.4.1 decision; strict artifact readers |

## Scope

**In:** `loads_within` and the two readers; one test file.

**Out:**
- Making journal or provenance reads strict (refusing `NaN`, `1e999` or repeated keys): a behaviour change with no ticket.
- Removing `load_provenance`'s `RecursionError` conversion.
- The bundle reader: FIX-122. `_artifacts.parse_json`: unchanged.

## Dependencies

**Blocked by:** FIX-122
**Blocks:** none
**Related:** #551, #762, #521
