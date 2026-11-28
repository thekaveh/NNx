---
id: FEAT-083
title: "[FEAT-083] Share one warn-while-failing helper"
labels: ["enhancement", "area: training", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-05"]
depends_on: ["FEAT-082", "FEAT-067"]
related_issues: [507]
status: READY
---

**Category:** Training · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-083: six places warn while another error is already propagating, and each copies the same guard so a `-W error` filter cannot replace that error: `try: _warn_at_user_frame(msg) / except Exception: pass`. Add `_warn_while_failing(message, category=RuntimeWarning)` beside `_warn_at_user_frame` in `src/nnx/_warnings.py` and call it from the six places, every message unchanged.

## Context

**What happens today.** For example `nn_model._rollback_resume` (L647–L656 at `9ad461e`):

```python
    except Exception as error:
        from .callbacks import _warn_at_user_frame

        try:  # an error filter (-W error) must not replace the original error either (#507)
            _warn_at_user_frame(
                "the resume failed after its restore and the model could not be rolled back "
                f"({type(error).__name__}: {error})",
            )
        except Exception:  # noqa: BLE001
            pass
```

The six copies: `nn/nn_model.py` `_rollback_resume` (L647–L656), `_CallbackFinalizer.__exit__` (L1499–L1507 and L1511–L1519), `_with_attempt` (L2636–L2645); `components.py` `ComponentRegistry.restore` (L418–L427); `provenance.py` `_AttemptRecorder.fail` (L779–L784). After FEAT-082 their local imports are module-level, and after FEAT-067 the inert `# noqa: BLE001` markers are gone.

**Why it matters.** Six copies of one rule; a seventh written without the guard would let a warnings filter swap the user's real error for a warning.

**History.** #507 (FIX-040) introduced the rule ("an error filter (-W error) must not replace the original error") and its tests run with `-W error`; they must keep passing. Conflict rule 13: FEAT-067 deletes the inert `noqa: BLE001` markers first, and this ticket must not add any back.

**Background for the worker.**
- *`-W error`*: a warnings filter that turns every warning into an exception. Raised inside an `except` block, that exception would replace the error being handled.
- *`_warn_at_user_frame(message, category)`*: NNx's helper that warns at the first caller outside `nnx` (in `src/nnx/_warnings.py` since FEAT-082).

**Proposal.**

1. In `src/nnx/_warnings.py`, after `_warn_at_user_frame`, add:
   ```python
   def _warn_while_failing(message: str, category: type[Warning] = RuntimeWarning) -> None:
       """Warn as :func:`_warn_at_user_frame` does while another error is already
       propagating. If a warnings filter turns the warning into an exception
       (``-W error``), that exception is dropped, so it never replaces the error
       being raised (#507)."""
       try:
           _warn_at_user_frame(message, category)
       except Exception:
           pass
   ```
2. Import it at module top next to `_warn_at_user_frame` in `src/nnx/nn/nn_model.py` (`from .._warnings import _warn_at_user_frame, _warn_while_failing`), `src/nnx/components.py` and `src/nnx/provenance.py` (`from ._warnings import _warn_at_user_frame, _warn_while_failing`; drop `_warn_at_user_frame` from an import where ruff then reports it unused).
3. Replace each block's `try: _warn_at_user_frame(<message>) except Exception: pass` with one call, keeping the explanatory comment as a line above it and the message text exactly:
   - `_rollback_resume`:
     ```python
         except Exception as error:
             # an error filter (-W error) must not replace the original error either (#507)
             _warn_while_failing(
                 "the resume failed after its restore and the model could not be rolled back "
                 f"({type(error).__name__}: {error})",
             )
     ```
   - `_CallbackFinalizer.__exit__`, first loop: `_warn_while_failing(f"on_train_end cleanup failed while handling {type(exc).__name__}: {cleanup_error}")` under the comment `# an error filter (-W error) must not replace the training error (#507)`;
   - `_CallbackFinalizer.__exit__`, second loop: `_warn_while_failing(f"additional on_train_end cleanup failed: {cleanup_error}")` under `# nor the first on_train_end error`;
   - `_with_attempt`: `_warn_while_failing(f"run {run.id} trained and saved, but its attempt could not be recorded as completed " f"({record_error}); attempt.json still reads 'running'")` under `# an error filter (-W error) must not lose the saved run either`;
   - `ComponentRegistry.restore`: `_warn_while_failing(f"component restore failed and could not roll back {', '.join(failed)}; " "their state is undefined")` under `# an error filter (-W error) must not replace the original error either (#507)`;
   - `_AttemptRecorder.fail`: `_warn_while_failing(f"could not record the {status} attempt {self.attempt.attempt_id}: {record_error}")` under `# an error filter (-W error) must not replace the training error either`.
4. Add this test to `tests/test_callbacks.py`:
   ```python
   def test_warn_while_failing_warns_but_never_raises_under_an_error_filter():
       """FEAT-083 (#507): a warning given while an error propagates never replaces it."""
       import warnings

       from nnx._warnings import _warn_while_failing

       with warnings.catch_warnings(record=True) as caught:
           warnings.simplefilter("always")
           _warn_while_failing("cleanup failed")
       assert [(w.category, str(w.message)) for w in caught] == [(RuntimeWarning, "cleanup failed")]
       with warnings.catch_warnings():
           warnings.simplefilter("error")
           _warn_while_failing("cleanup failed")  # dropped, not raised
   ```

No behaviour changes, so no CHANGELOG entry.

**Worker setup.** Branch from `develop` after FEAT-082 and FEAT-067 have merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The new test `tests/test_callbacks.py::test_warn_while_failing_warns_but_never_raises_under_an_error_filter` passes (it fails on `develop`, where the helper does not exist). Check: `uv run pytest tests/test_callbacks.py::test_warn_while_failing_warns_but_never_raises_under_an_error_filter`
- [ ] No copy of the guard is left in the six places. Check: `grep -n -A1 "_warn_at_user_frame(" src/nnx/nn/nn_model.py src/nnx/components.py src/nnx/provenance.py | grep -c "except Exception:"` prints `0`, and `grep -c "_warn_while_failing(" src/nnx/nn/nn_model.py src/nnx/components.py src/nnx/provenance.py` totals 6
- [ ] No `noqa: BLE001` came back. Check: `grep -rn "BLE001" src/nnx` prints nothing
- [ ] #507's `-W error` tests and the resume suites pass. Check: `uv run pytest tests -k "component_resume or attempt_provenance or callbacks or exact_resume"`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [nn_model.py L645-L658](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L645-L658) | `_rollback_resume`'s copy |
| [nn_model.py L1498-L1519](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1498-L1519) | `_CallbackFinalizer.__exit__`'s two copies |
| [nn_model.py L2634-L2645](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L2634-L2645) | `_with_attempt`'s copy |
| [components.py L418-L427](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/components.py#L418-L427) | `ComponentRegistry.restore`'s copy |
| [provenance.py L778-L784](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/provenance.py#L778-L784) | `_AttemptRecorder.fail`'s copy |
| #507 | The rule and its `-W error` tests |

## Scope

**In:** `_warn_while_failing` and the six call sites; one test.

**Out:** enabling ruff's BLE rules; changing any message; other warning sites (FEAT-084 replaces two raw `warnings.warn` calls).

## Dependencies

**Blocked by:** FEAT-082, FEAT-067
**Blocks:** none
**Related:** #507
