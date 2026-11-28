---
id: FEAT-100
title: "[FEAT-100] Share the checkpoint-path guard of the training-state readers"
labels: ["enhancement", "area: persistence", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-19"]
depends_on: []
related_issues: [206]
status: READY
---

**Category:** Persistence · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-100: `_read_checkpoint_path` validates its arguments by building a checkpoint path and throwing it away, and three `NNCheckpoint` readers repeat the same two-condition guard before taking the lock. This ticket splits the validation into `_validate_checkpoint_args`, adds `_existing_checkpoint` for the guard, and uses it in the four readers that need a run directory. Each reader keeps its own absent result, and no reader creates `runs/<id>/` (#206).

## Context

**What happens today.** `_read_checkpoint_path` (`nn_checkpoint.py` L116–L126):

```python
def _read_checkpoint_path(run: str, type: Checkpoints, root: Optional[str] = None) -> Optional[tuple[str, str]]:
    """The run and checkpoint path a reader opens: ``_checkpoint_path``
    (``run`` and ``type`` validated first) with the ``runs/best`` pointer
    resolved once (``nn_run._resolve_run_id``), or ``None`` when no run
    holds it, which each reader answers as for an absent run. Saves use
    ``_checkpoint_path``, which never resolves."""
    from .nn_run import _resolve_run_id

    _checkpoint_path(run, type, root=root)
    resolved = _resolve_run_id(run, root)
    return None if resolved is None else (resolved, _checkpoint_path(resolved, type, root=root))
```

`load_training_state` (L944–L946), `load_with_training_state` (L966–L968) and `_load_resume_point` (L994–L996) each open with:

```python
        target = _read_checkpoint_path(run, type, root=root)
        if target is None or not _run_directory_exists(target[1]):
            return None
```

(`return None, None` in the two that return a pair). `verify` (L887–L896) gets the target, returns `None` for `None`, and then returns `None` when the checkpoints directory does not exist.

**Why it matters.** The discarded call hides that it exists only to validate; the copied guard is the FIX-007 rule (#206), and a reader that forgot one half of it would reserve a run by probing it.

**History.** #206 (FIX-007) chose: "each reader returns its existing absent result (`None` or `(None, None)`)" and "no reader creates `runs/<id>/`", and required that reads still go "through `_checkpoint_path` first, so run-ID validation and traversal protection still apply". This ticket keeps all three.

**Background for the worker.**
- `_checkpoint_path(run, type, root)` (L85): validates `run` (`nn_run._validate_run_id`: no path traversal) and `type` (a `Checkpoints` member or a filename-safe tag, else `ValueError`), then returns `<root or ".">/runs/<run>/checkpoints/<type>.pt`. It never resolves `runs/best`.
- The tuple `_read_checkpoint_path` returns: `(resolved_run_id, checkpoint_path)`. `resolved_run_id` is `run` with the `runs/best` pointer resolved (`_resolve_run_id`), and `checkpoint_path` is `_checkpoint_path(resolved_run_id, type, root=root)`. `None` means `run` is `"best"` and no run holds the pointer.
- `root`: the directory that holds `runs/`; `None` means the current working directory.
- `_run_directory_exists(checkpoint_path)` (L129): whether `runs/<id>/` (two levels above the `.pt` path) is a directory, checked without touching the filesystem otherwise. The readers take a `FileLock` next, whose implementation creates `runs/<id>/checkpoints/`; the check comes first so a probe never creates a run.

**Proposal.**
1. In `src/nnx/nn/params/nn_checkpoint.py`, split `_checkpoint_path` (L85–L113). Move its validation (the docstring's threat-model paragraph, the comment, the local `from .nn_run import _validate_run_id`, `_validate_run_id(run)` and the `type` check with its `ValueError`) into a new function placed directly above it:

   ```python
   def _validate_checkpoint_args(run: str, type: Checkpoints) -> None:
       """Refuse a ``run`` that is not a safe run id (path traversal; see
       :func:`nnx.nn.params.nn_run._validate_run_id`) and a ``type`` that is
       neither a ``Checkpoints`` member nor a filename-safe tag."""
   ```

   followed by those lines unchanged. `_checkpoint_path` keeps its signature and docstring and becomes:

   ```python
       _validate_checkpoint_args(run, type)
       base = root if root is not None else "."
       return os.path.join(base, "runs", run, "checkpoints", str(type) + ".pt")
   ```

2. In `_read_checkpoint_path`, replace the discarded `_checkpoint_path(run, type, root=root)` (L124) with `_validate_checkpoint_args(run, type)`. Change "``_checkpoint_path`` (``run`` and ``type`` validated first)" in its docstring to "``_checkpoint_path`` (``run`` and ``type`` validated first, by :func:`_validate_checkpoint_args`)".
3. Directly below `_run_directory_exists` (after L143), add:

   ```python
   def _existing_checkpoint(run: str, type: Checkpoints, root: Optional[str] = None) -> Optional[tuple[str, str]]:
       """:func:`_read_checkpoint_path`, or ``None`` also when the run directory
       does not exist (FIX-007): the training-state readers then return their
       absent result before the lock, whose implementation would create
       ``runs/<id>/checkpoints/`` and reserve the run."""
       target = _read_checkpoint_path(run, type, root=root)
       return None if target is None or not _run_directory_exists(target[1]) else target
   ```

4. Use it in the four readers:
   - `verify` (L887): replace `target = _read_checkpoint_path(run, type, root=root)` with `target = _existing_checkpoint(run, type, root=root)`. Its `if target is None: return None` and its later `if not os.path.isdir(directory): return None` stay. (A missing run directory means a missing checkpoints directory, so `verify` returned `None` for it already.)
   - `load_training_state` (L944–L945), `load_with_training_state` (L966–L967) and `_load_resume_point` (L994–L995): replace the two lines `target = _read_checkpoint_path(run, type, root=root)` / `if target is None or not _run_directory_exists(target[1]):` with `target = _existing_checkpoint(run, type, root=root)` / `if target is None:`. The `return` line under each stays (`None` or `None, None`).
   - `resume_point` (L867) and `load` (L1271) keep `_read_checkpoint_path`: they take no lock.
5. No test patches these helpers (`grep -rn "_read_checkpoint_path\|_run_directory_exists\|_checkpoint_path" tests | grep "setattr\|patch"` returns nothing). No behaviour changes, so there is no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] No call builds a path only to validate. Check: `grep -n "^    _checkpoint_path(" src/nnx/nn/params/nn_checkpoint.py` returns 0 lines
- [ ] The guard is written once. Check: `grep -c "_run_directory_exists(target\[1\])" src/nnx/nn/params/nn_checkpoint.py` prints `1`, and `grep -c "_existing_checkpoint(run, type, root=root)" src/nnx/nn/params/nn_checkpoint.py` prints `4`
- [ ] FIX-007's tests pass: each reader returns its absent result and creates no `runs/<id>/`. Check: `uv run pytest tests -k "test_missing_state_read_does_not_reserve_run or test_missing_probe_then_first_train_keeps_overwrite_protection"`
- [ ] The checkpoint-reader suites pass unchanged. Check: `uv run pytest tests/test_best_pointer_readers.py tests/test_resume_point_durability.py tests/test_resume_point_hashing.py tests/test_exact_resume.py tests/test_run_bundles.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [nn_checkpoint.py L85-L113](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_checkpoint.py#L85-L113) | `_checkpoint_path`: validation, then the path |
| [nn_checkpoint.py L116-L126](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_checkpoint.py#L116-L126) | `_read_checkpoint_path`: the discarded call at L124 |
| [nn_checkpoint.py L129-L143](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_checkpoint.py#L129-L143) | `_run_directory_exists` (FIX-007) |
| [nn_checkpoint.py L887-L896](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_checkpoint.py#L887-L896) | `verify` |
| [nn_checkpoint.py L944-L946](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_checkpoint.py#L944-L946), [L966-L968](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_checkpoint.py#L966-L968), [L994-L996](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_checkpoint.py#L994-L996) | the copied guard |
| [test_train_integration.py L1782](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_train_integration.py#L1782), [test_pass2_r_series.py L396-L399](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_pass2_r_series.py#L396-L399) | FIX-007's tests |
| #206 | FIX-007: absent result, no phantom run, validation first |

## Scope

**In:** `_validate_checkpoint_args`, `_existing_checkpoint`, the four readers.

**Out:** `nn_model._verified_resume_point`, which also calls `_checkpoint_path` only to validate (L479): left as it is here, because FEAT-095 changes that function's imports. Renaming the `test_pass2_*` modules (FEAT-068).

## Dependencies

**Blocked by:** none
**Blocks:** none
**Related:** #206
