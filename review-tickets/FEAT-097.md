---
id: FEAT-097
title: "[FEAT-097] Share one thaw function for frozen JSON values (part 2 of 2)"
labels: ["enhancement", "area: configuration", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-16"]
depends_on: ["FEAT-096"]
related_issues: [551, 563]
status: READY
---

**Category:** Configuration · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-097: Three modules carry byte-identical functions that turn a frozen JSON-like value back into plain `dict`s and `list`s: `_config._thaw_config`, `provenance._thaw` and `transforms._plain`. This ticket adds one `thaw_json` to `nnx/_frozen.py` (created by FEAT-096) and makes the three names aliases of it. The two classes both named `_FrozenConfig` stay, because pickles reference them; their docstrings now say how they differ. Behaviour does not change. This is part 2 of SMELL-16.

## Context

**What happens today.** `_config._thaw_config` (`_config.py` L101–L107):

```python
def _thaw_config(value: Any) -> Any:
    """Plain YAML-safe copy (dicts / lists) of a frozen config value."""
    if isinstance(value, Mapping):
        return {key: _thaw_config(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_config(item) for item in value]
    return value
```

`provenance._thaw` (L291–L296) and `transforms._plain` (L139–L144) are the same three branches under other names. The benchmark's `_plain` (`decisions/benchmark.py` L407–L415) looks similar but is **not** the same function:

```python
    if isinstance(value, (dict, _FrozenMapping)):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, _FrozenList)):
        return [_plain(item) for item in value]
    return value
```

It converts only its own frozen types (and copies lists), and leaves a caller's plain tuple alone so that `Record.execution` can still refuse it. It stays.

The two `_FrozenConfig` classes: `_config._FrozenConfig` (L24) keeps insertion order and is unhashable; `transforms._FrozenConfig` (L100) sorts its keys, is hashable and looks keys up in O(n). Both return their own class from `__reduce__`, so both class paths are stored in pickles (run configs, recipes).

**Why it matters.** Three copies of one function can drift; two classes with one name mislead a reader who greps for it.

**History.** #551 bounded the nesting depth of these values (`MAX_JSON_DEPTH`), so the recursion of every thaw stays within Python's limit; `thaw_json` recurses exactly as the three copies do. #563 is the order-free hash of the shared mapping (FEAT-096).

**Background for the worker.**
- Frozen JSON-like value: a config or record made read-only by freezing every mapping into a read-only `Mapping` and every list into a `tuple`. Thawing reverses it for `state()`, YAML and JSON.
- Alias: `_thaw_config = thaw_json` makes the old name refer to the same function, so every `from ._config import _thaw_config` keeps working (`calibration.py`, `abstention.py`, `monitors.py`, `optimizers.py`, `models.py` import it), and `data_splits.py` L61 keeps importing `provenance._thaw`.
- A class that appears in its own `__reduce__` must not be renamed: a pickle stores its module and name, and new pickles would name the new class, which earlier releases cannot load.

**Proposal.**
1. In `src/nnx/_frozen.py` (FEAT-096), add `Any` to `from typing import ...` and append:

   ```python
   def thaw_json(value: Any) -> Any:
       """The plain JSON form of a frozen JSON-like value: every mapping as a
       ``dict`` and every tuple as a ``list``, at every depth; anything else as
       it is. Shared by ``_config._thaw_config``, ``provenance._thaw`` and
       ``transforms._plain``."""
       if isinstance(value, Mapping):
           return {key: thaw_json(item) for key, item in value.items()}
       if isinstance(value, tuple):
           return [thaw_json(item) for item in value]
       return value
   ```

2. In `src/nnx/_config.py`, add `from ._frozen import thaw_json` to the first-party imports (above `from ._json_depth import MAX_JSON_DEPTH, too_deep`, L16) and replace the whole `_thaw_config` function (L101–L107) with:

   ```python
   _thaw_config = thaw_json  # a plain YAML-safe copy (dicts / lists) of a frozen config value
   ```

3. In `src/nnx/provenance.py`, add `from ._frozen import thaw_json` to the first-party imports (after `from ._config import _canonical_json`, L54) and replace the whole `_thaw` function (L291–L296) with `_thaw = thaw_json`.
4. In `src/nnx/transforms.py`, add `from ._frozen import thaw_json` below `from torch import nn` (L56; `_frozen` is standard-library only, so `transforms` still has no import cycle), and replace the whole `_plain` function (L139–L144) with `_plain = thaw_json`.
5. Keep both `_FrozenConfig` classes and their names. Append one sentence to each docstring:
   - `_config._FrozenConfig` (L25–L26): "Not :class:`nnx.transforms._FrozenConfig`, which sorts its keys and is hashable (a recipe operation's config); both keep their names, which pickles store."
   - `transforms._FrozenConfig` (L101–L102): "Not :class:`nnx._config._FrozenConfig`, which keeps insertion order and is unhashable (a registered object's spec config); both keep their names, which pickles store."
6. Add `tests/test_frozen.py`:

   ```python
   """One thaw function for every frozen JSON-like value (nnx._frozen)."""

   from nnx._config import _freeze_config, _thaw_config
   from nnx._frozen import thaw_json
   from nnx.provenance import _thaw
   from nnx.transforms import _plain


   def test_the_thaw_functions_are_one_function():
       assert _thaw_config is thaw_json and _thaw is thaw_json and _plain is thaw_json


   def test_thaw_json_returns_dicts_and_lists_at_every_depth():
       frozen = _freeze_config({"b": [1, {"c": (2, 3)}], "a": None}, "")
       plain = thaw_json(frozen)
       assert plain == {"a": None, "b": [1, {"c": [2, 3]}]}
       assert type(plain) is dict and type(plain["b"]) is list and type(plain["b"][1]["c"]) is list
       assert thaw_json("text") == "text" and thaw_json([1, (2,)]) == [1, (2,)]  # a list is left as it is
   ```

7. No test patches the three functions (`grep -rn "_thaw_config\|provenance._thaw\|transforms._plain" tests` shows imports only), and nothing changes behaviour, so there is no CHANGELOG entry.

**Worker setup.** Branch from `develop` after FEAT-096 has merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The new tests pass. Check: `uv run pytest tests/test_frozen.py`
- [ ] One thaw body remains. Check: `grep -rnE "^def (_thaw_config|_thaw|_plain)\(" src/nnx/_config.py src/nnx/provenance.py src/nnx/transforms.py` returns 0 lines, and `grep -n "^def thaw_json" src/nnx/_frozen.py` returns 1 line
- [ ] The benchmark's own `_plain` is untouched. Check: `git diff --stat origin/develop -- src/nnx/decisions/benchmark.py` shows no change
- [ ] Both `_FrozenConfig` classes keep their names. Check: `grep -rn "^class _FrozenConfig" src/nnx` returns 2 lines
- [ ] The suites that thaw configs pass. Check: `uv run pytest tests/test_model_recipes.py tests/test_params_round_trip.py tests/test_model_factories.py tests/test_optimizer_factories.py tests/test_calibration.py tests/test_abstention.py tests/test_data_splits.py tests/test_attempt_provenance.py tests/test_legacy_record_pickles.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [_config.py L24-L46](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_config.py#L24-L46) | `_config._FrozenConfig`: insertion order, no hash, `__reduce__` |
| [_config.py L101-L107](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_config.py#L101-L107) | `_thaw_config` |
| [provenance.py L280-L296](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/provenance.py#L280-L296) | `_freeze` and `_thaw` |
| [transforms.py L100-L144](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/transforms.py#L100-L144) | `transforms._FrozenConfig` (sorted, hashable), `_frozen`, `_plain` |
| [benchmark.py L407-L415](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L407-L415) | the benchmark's `_plain`: different rules, kept |
| [data_splits.py L61](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/data_splits.py#L61) | imports `provenance._thaw` |
| #551 | the depth bound every thaw relies on |

## Scope

**In:** `thaw_json`, three aliases, two docstring sentences, `tests/test_frozen.py`.

**Out:** the freeze functions (`_config._freeze_config`, `transforms._frozen`, `provenance._freeze`, `benchmark._frozen`, `calibration._frozen`, `_artifacts.frozen_json`): they validate and order differently, so sharing them would change behaviour. Renaming either `_FrozenConfig` (pickle class paths). The benchmark's `_plain`. The shared mapping class (FEAT-096).

## Dependencies

**Blocked by:** FEAT-096
**Blocks:** none
**Related:** #551, #563
