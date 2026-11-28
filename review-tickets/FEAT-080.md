---
id: FEAT-080
title: "[FEAT-080] Extract the per-run body of observations_from_runs"
labels: ["enhancement", "area: evaluation", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-03"]
depends_on: []
related_issues: []
status: READY
---

**Category:** Evaluation · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-080: `comparison.observations_from_runs` is 198 lines with cyclomatic complexity 57; 93 of them are the body of its loop over runs, which builds one `Observation`. Move that body, unchanged, into a private module-level function `_observation(...)` and call it once per run. No behaviour changes.

## Context

**What happens today.** After validating its arguments, `observations_from_runs` (L1510) runs one long loop (L1613–L1706) that shares two caches across runs:

```python
    observations = []
    identities: dict[str, tuple[str, int]] = {}  # run id -> (identity with its parents', ancestor count)
    files: dict[Any, Any] = {}  # every run.yaml / provenance read in this call, read once
    for given in run_ids:
        run_id = _observed_run_id(given, root)
        run_state, idps, recorded, provenance, committed, known, run_path = _read_run(run_id, root, files)
```

and ends each iteration with `observations.append(Observation(...))`, then `return observations` (L1707).

**Why it matters.** The argument checks, the per-run reading and the epoch selection are hard to tell apart in one function; a per-run function can be read and tested on its own.

**History.** No board issue covers this function's structure.

**Background for the worker.**
- *`Observation`*: one run's value of a metric plus its identity fields (status, attempt, configuration, data and split identities, replicate key, evaluation note), the input of `nnx.comparison`'s summaries.
- *Shared caches*: `identities` (run id → configuration identity with its ancestors') and `files` (every `run.yaml` and provenance file read in this call) are filled by the helpers the body calls; they must stay one dict each per call, shared by all runs.

**Proposal.**

1. In `src/nnx/comparison.py`, add directly before `def observations_from_runs` (L1510):
   ```python
   def _observation(
       given: str,
       *,
       metric: Metric,
       split: str,
       selection: str,
       root: Optional[str],
       replicate: Optional[str],
       config: Optional[Mapping[str, str]],
       identities: dict[str, tuple[str, int]],
       files: dict[Any, Any],
   ) -> Observation:
       """One run's observation for :func:`observations_from_runs` (``given`` is
       the id as the caller passed it). ``identities`` and ``files`` are the
       call's caches, shared by every run of one call."""
       from .history import has_journal
   ```
   followed by the loop body L1614–L1706, dedented one level, unchanged except that its last statement `observations.append(Observation(...))` becomes `return Observation(...)` with the same arguments.
2. In `observations_from_runs`, delete `from .history import has_journal` (L1608) and replace the loop L1613–L1706 with:
   ```python
       for given in run_ids:
           observations.append(
               _observation(
                   given,
                   metric=metric,
                   split=split,
                   selection=selection,
                   root=root,
                   replicate=replicate,
                   config=config,
                   identities=identities,
                   files=files,
               )
           )
   ```
   `observations = []`, the two cache dicts and `return observations` stay. The docstring is unchanged.
3. No test patch needs a new target: `tests/test_run_comparison.py` L1345–L1346 patch `comparison._provenance` and `comparison._read_run_state`, which `_observation` reaches through the same module.
4. Add this test to `tests/test_run_comparison.py`:
   ```python
   def test_observations_from_runs_builds_one_observation_per_run_through_one_helper(monkeypatch):
       """FEAT-080: the per-run body lives in _observation; the two caches are shared by every run of a call."""
       import nnx.comparison as module

       seen = []

       def fake(given, *, identities, files, **kwargs):
           seen.append((given, id(identities), id(files)))
           return given

       monkeypatch.setattr(module, "_observation", fake)
       out = module.observations_from_runs(["a", "b"], metric=module.Metric("loss", "minimize"))
       assert out == ["a", "b"] and seen[0][1:] == seen[1][1:]
   ```

No behaviour changes, so no CHANGELOG entry; `observations_from_runs`' docstring is unchanged, so `docs/api.md` needs no rebuild.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The new test fails on `develop` (no `_observation`) and passes after the change. Check: `uv run pytest tests/test_run_comparison.py::test_observations_from_runs_builds_one_observation_per_run_through_one_helper`
- [ ] `observations_from_runs` is at most 110 lines including its docstring. Check: `uv run python -c "import inspect, nnx.comparison as c; print(len(inspect.getsource(c.observations_from_runs).splitlines()))"` prints at most 110
- [ ] The comparison and best-pointer suites pass. Check: `uv run pytest tests/test_run_comparison.py tests/test_best_pointer_readers.py`
- [ ] The API reference is unchanged. Check: `uv run python -m scripts.docs.build_api_reference --check`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [comparison.py L1510-L1707](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/comparison.py#L1510-L1707) | `observations_from_runs` |
| [comparison.py L1608-L1613](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/comparison.py#L1608-L1613) | The local import, the shared caches and the loop head |
| [comparison.py L1614-L1706](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/comparison.py#L1614-L1706) | The per-run body that moves |
| [test_run_comparison.py L1345-L1346](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_run_comparison.py#L1345-L1346) | Module-level patches that keep working |

## Scope

**In:** `_observation` and the loop in `observations_from_runs`; one test.

**Out:** splitting `_observation` further; the other SMELL-03 hotspots (`NNModel._train_impl` is FEAT-079; the rest are not ticketed).

## Dependencies

**Blocked by:** none
**Blocks:** none
**Related:** none
