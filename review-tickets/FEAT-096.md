---
id: FEAT-096
title: "[FEAT-096] Share one FrozenMapping class (part 1 of 2)"
labels: ["enhancement", "area: configuration", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-16"]
depends_on: []
related_issues: [563, 558, 209, 551]
status: READY
---

**Category:** Configuration · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-096: The read-only mapping that `NNTrainerParams`, `MetricSnapshot`, `ExperimentPlan` and the decision benchmark all use, `_FrozenMapping`, is defined in `nnx/trainer/params.py`. Three other packages import it privately from there, and `nnx.decisions.benchmark` imports the whole trainer package at module load for it. This ticket moves the class, unchanged, to a new standard-library-only module `nnx/_frozen.py` as `FrozenMapping`, keeps `nnx.trainer.params._FrozenMapping` as an alias so old pickles load, and points the three importers at the new home. Its non-recursive `__reduce__` stays as it is. This is part 1 of SMELL-16 (the mapping class); FEAT-097 is part 2 (the thaw functions).

## Context

**What happens today.** The class (`trainer/params.py` L44–L90) ends:

```python
    def __hash__(self) -> int:
        # Order-free, like __eq__ (#563); a frozenset needs no orderable keys,
        # and plans._frozen_mapping wraps any mapping.
        return hash(frozenset(self._items.items()))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._items!r})"

    def __reduce__(self):
        return (_FrozenMapping, (self._items,))
```

Its private importers: `decisions/benchmark.py` L65 (`from ..trainer.params import _FrozenMapping`, at module level, which runs `nnx/trainer/__init__.py`, `trainer.trainer` and `nn_model`), `streaming.py` L774 (inside `StreamingMetrics.finalize`), and `plans.py` L260 (`_frozen_mapping`) and L334 (`ExperimentPlan.__post_init__`). The benchmark subclasses it (L378):

```python
class _FrozenDict(_FrozenMapping[Any]):
    """A JSON object frozen by :func:`_frozen`: a :class:`_FrozenMapping`
    copied and pickled, like a :class:`_FrozenList`, as its plain form."""

    __slots__ = ()

    def __reduce__(self) -> tuple[Any, ...]:
        return (_frozen, (_plain(self),))
```

**Why it matters.** A shared type lives in a trainer-specific module, so the decisions toolkit and streaming metrics depend on the trainer package to build a read-only dict, and the class has no obvious owner when its semantics need to change.

**History.**
- #209 introduced the snapshot so that "A built `NNTrainerParams` kept references to the live `optims`, `schedulers` and `extra_metrics` dictionaries" could not happen again.
- #563 made "`_FrozenMapping` hash ignore insertion order, matching its equality" (`test_frozen_mapping_hash_needs_no_orderable_keys` pins it).
- #558 froze the benchmark `Record` and report mappings; `benchmark._FrozenDict` subclasses `_FrozenMapping` for that.
- #551 bounded JSON depth. The benchmark's `_FrozenDict` / `_FrozenList` pickle as their plain form so a value as deep as the bound allows does not overflow the stack. The shared class keeps `__reduce__` returning `(class, (self._items,))`: it never reduces its values one level at a time itself, and `_FrozenDict` keeps its own override.

**Background for the worker.**
- `_FrozenMapping`: an insertion-ordered, read-only `Mapping[str, V]`; equality and hash ignore order; `__reduce__` rebuilds it from its items dict (used by `copy.copy`, `copy.deepcopy` and `pickle`).
- Pickle class path: a pickle stores the defining module and name of each class (`nnx.trainer.params._FrozenMapping` today). Loading looks that name up, so the old name must stay importable. After the move, new pickles name `nnx._frozen.FrozenMapping`; NNx 0.5.0 and earlier cannot load those. No file NNx writes holds one: checkpoints pickle `NNCheckpoint` (`net_params`, `net_state`, `model_params`, `idp`, `transforms`), and runs, histories, bundles and benchmark reports are YAML, CSV or JSON.
- `nnx._frozen` imports only the standard library, so every module can import it at the top.

**Proposal.**
1. Create `src/nnx/_frozen.py`:

   ```python
   """Read-only snapshots of name-keyed mappings, shared by NNTrainerParams,
   MetricSnapshot, ExperimentPlan and the decision benchmark. Internal;
   standard library only."""

   from __future__ import annotations

   from collections.abc import Iterator, Mapping
   from typing import Generic, Optional, TypeVar

   _V = TypeVar("_V")


   class FrozenMapping(Mapping[str, _V], Generic[_V]):
   ```

   then paste the class body of `trainer.params._FrozenMapping` (its docstring and every method from `__slots__ = ("_items",)` to `__reduce__`, L45–L90) unchanged, except the last line of `__reduce__`, which becomes `return (FrozenMapping, (self._items,))`. Add one sentence at the end of the docstring: "Also the read-only values of ``MetricSnapshot``, a plan's frozen ``extra_metrics`` and the decision benchmark's report mappings."
2. In `src/nnx/trainer/params.py`:
   - delete the class (L44–L90) and `_V = TypeVar("_V")` (L41);
   - add `from .._frozen import FrozenMapping` to the first-party imports (above `from .._validation import plain_str, require_count`, L24), and directly below the imports add:

     ```python
     # The old name: pickles written by NNx 0.5.0 and earlier name nnx.trainer.params._FrozenMapping.
     _FrozenMapping = FrozenMapping
     ```

   - in `NNTrainerParams.__post_init__` (L191, L192, L198), replace `_FrozenMapping` with `FrozenMapping`;
   - run `uv run ruff check --fix src/nnx/trainer/params.py` to drop the imports that became unused (`Iterator`, `Generic`, `TypeVar`).
3. In `src/nnx/decisions/benchmark.py`, replace `from ..trainer.params import _FrozenMapping` (L65) with `from .._frozen import FrozenMapping`, and replace `_FrozenMapping` with `FrozenMapping` at L378 (`class _FrozenDict(FrozenMapping[Any]):`), in the `_FrozenDict` docstring (L379, ``a :class:`FrozenMapping` ``), at L410 (`isinstance(value, (dict, FrozenMapping))`) and at L425 (`return FrozenMapping(value)`). Leave `_FrozenDict.__reduce__` and `_FrozenList.__reduce__` unchanged.
4. In `src/nnx/streaming.py`, add `from ._frozen import FrozenMapping` to the top-level first-party imports (above `from ._probability import ...`, L62), delete the local import in `StreamingMetrics.finalize` (L774), and change `values=_FrozenMapping(values)` (L785) to `values=FrozenMapping(values)`.
5. In `src/nnx/plans.py`, add `from ._frozen import FrozenMapping` to the top-level first-party imports (above `from .nn.enum.checkpoints import Checkpoints`, L65), delete the local imports at L260 and L334, and replace `_FrozenMapping` with `FrozenMapping` at L262 and L336.
6. Tests in `tests/test_trainer_params.py`:
   - in `test_frozen_mapping_hash_needs_no_orderable_keys` (L431), change `from nnx.trainer.params import _FrozenMapping` to `from nnx._frozen import FrozenMapping` and the two constructor calls to `FrozenMapping(...)`;
   - add, after it:

     ```python
     def test_a_frozen_mapping_pickled_under_its_old_name_still_loads():
         """A pickle written by NNx 0.5.0 names nnx.trainer.params._FrozenMapping."""
         import pickle

         from nnx._frozen import FrozenMapping
         from nnx.trainer.params import _FrozenMapping

         legacy = (
             b"\x80\x02cnnx.trainer.params\n_FrozenMapping\nq\x00}q\x01(X\x02\x00\x00\x00lrq\x02G?\xe0\x00\x00\x00"
             b"\x00\x00\x00X\x04\x00\x00\x00nameq\x03X\x04\x00\x00\x00adamq\x04u\x85q\x05Rq\x06."
         )
         loaded = pickle.loads(legacy)
         assert type(loaded) is FrozenMapping and _FrozenMapping is FrozenMapping
         assert loaded == {"lr": 0.5, "name": "adam"} and list(loaded) == ["lr", "name"]
         assert pickle.loads(pickle.dumps(loaded)) == loaded
     ```

7. Add to `CHANGELOG.md` under `## [Unreleased]` → `### Changed`:

   ```
   - **Read-only parameter mappings moved to `nnx._frozen`.** `NNTrainerParams.optims`, `schedulers` and `extra_metrics`, `MetricSnapshot.values` and a plan's frozen `extra_metrics` are `FrozenMapping` snapshots from `nnx._frozen`, and `nnx.decisions` no longer imports the trainer package. Their `repr` names `FrozenMapping`. Pickles written by earlier releases still load; a pickle of one of these objects written by this release does not load in NNx 0.5.0 or earlier. No file NNx writes contains one.
   ```

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The legacy-pickle test passes after the change (the bytes are the pickle `develop` writes today). Check: `uv run pytest tests/test_trainer_params.py::test_a_frozen_mapping_pickled_under_its_old_name_still_loads`
- [ ] The class is defined once. Check: `grep -rn "class FrozenMapping\|class _FrozenMapping" src/nnx` returns 1 line, in `src/nnx/_frozen.py`
- [ ] No module imports it from `trainer.params`. Check: `grep -rn "trainer.params import _FrozenMapping" src/nnx` returns 0 lines
- [ ] `__reduce__` stays non-recursive. Check: `grep -n "return (FrozenMapping, (self._items,))" src/nnx/_frozen.py` returns 1 line, and `grep -c "return (_frozen, (_plain(self),))" src/nnx/decisions/benchmark.py` prints `2`
- [ ] `nnx._frozen` imports only the standard library. Check: `grep -nE "^(from|import) " src/nnx/_frozen.py` lists only `__future__`, `collections.abc` and `typing`
- [ ] The suites that use the mapping pass. Check: `uv run pytest tests/test_trainer_params.py tests/test_params_round_trip.py tests/test_decision_benchmark.py tests/test_experiment_plan.py tests/test_plan_execution.py tests/test_metric_accumulators.py tests/test_legacy_record_pickles.py`
- [ ] The CHANGELOG entry is under `[Unreleased]` → `### Changed`. Check: `grep -n "Read-only parameter mappings moved to" CHANGELOG.md`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [trainer/params.py L41-L90](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/trainer/params.py#L41-L90) | `_FrozenMapping` and its `__reduce__` |
| [trainer/params.py L189-L198](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/trainer/params.py#L189-L198) | `NNTrainerParams.__post_init__` wraps `optims`, `schedulers`, `extra_metrics` |
| [benchmark.py L65](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L65) | the module-level private import of the trainer package |
| [benchmark.py L365-L426](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L365-L426) | `_FrozenList`, `_FrozenDict` (non-recursive `__reduce__`), `_frozen`, `_plain`, `_snapshot` |
| [streaming.py L770-L785](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/streaming.py#L770-L785) | `StreamingMetrics.finalize`: the local import |
| [plans.py L256-L262](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/plans.py#L256-L262), [L333-L337](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/plans.py#L333-L337) | `_frozen_mapping` and `ExperimentPlan.__post_init__` |
| [test_trainer_params.py L431-L439](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_trainer_params.py#L431-L439) | #563's order-free hash test |
| #209, #563, #558, #551 | the snapshot's origin, its hash, the benchmark subclass, the depth bound |

## Scope

**In:** `nnx/_frozen.py`, the alias, the three importers, two tests, the CHANGELOG line.

**Out:** the thaw functions and the two `_FrozenConfig` classes (FEAT-097). `nn_evaluation_data_point._FrozenMetrics` (sorted keys, a different contract). The benchmark's `_frozen` / `_plain`, which keep telling a frozen list from a caller's tuple. Hoisting `benchmark.py`'s other local imports (FEAT-136).

## Dependencies

**Blocked by:** none
**Blocks:** FEAT-097
**Related:** #563, #558, #209, #551
