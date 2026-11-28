---
id: FEAT-082
title: "[FEAT-082] Move _warn_at_user_frame to a standard-library-only module"
labels: ["enhancement", "area: packaging", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-04"]
depends_on: ["FIX-113"]
related_issues: [522]
status: READY
---

**Category:** Packaging · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-082: NNx's warning helper `_warn_at_user_frame` needs only `sys` and `warnings`, but it is defined in `nn/callbacks.py`, so 29 call sites import it inside functions. Move it, unchanged (FIX-113's `skip=` keyword included), to a new standard-library-only `src/nnx/_warnings.py`, keep it importable from `nnx.nn.callbacks`, and import it at module top everywhere.

## Context

**What happens today.** `src/nnx/nn/callbacks.py` L111:

```python
def _warn_at_user_frame(
    message: str, category: type[Warning] = RuntimeWarning, *, once_per_location: bool = False
) -> None:
    """Emit a warning (``RuntimeWarning`` by default) attributed to the first
    caller outside nnx.
```

and every user imports it at call time, for example `history.py` L1298:

```python
        from .nn.callbacks import _warn_at_user_frame
```

The 29 function-local imports at `9ad461e`: `nn/nn_model.py` L648, L756, L944, L1055, L1500, L1512, L2637, L3581, L5801; `history.py` L950, L1298, L1306, L1317, L1325, L1353, L1361; `_metrics.py` L114, L122, L129, L136; `trainer/trainer.py` L200, L911; `provenance.py` L698, L779; `nn/params/nn_run.py` L732; `_update_engine.py` L76; `components.py` L419; `monitors.py` L635; `_scheduler_clock.py` L273.

**Why it matters.** A library-wide helper in a feature module forces every caller into a function-local import and hides the real dependency.

**History.** #522 (FIX-055) made `NNRun.all()` and `runs/best` robust under `-W error`, which relies on this helper's attribution. FIX-113 (BUG-18) adds a `skip=` keyword to it first, in `nn/callbacks.py`; this ticket moves that version (conflict rule 5).

**Background for the worker.**
- *`_warn_at_user_frame`*: warns at the first stack frame whose module is outside `nnx` (it walks frames by `__name__`), with a fresh warning registry per call so user filters stay authoritative. Where the function is defined does not change which frame it picks.
- *Standard-library-only*: `_warnings.py` may import only `sys`, `warnings` and `typing`/`collections.abc`; that is what lets every module import it at module top without a cycle. (The stdlib's own `_warnings` is a built-in C module; `nnx._warnings`, imported relatively, does not clash with it.)

**Proposal.**

1. Create `src/nnx/_warnings.py`:
   ```python
   """NNx's warning helper (FEAT-082): warnings attributed to the first caller
   outside nnx. Standard library only, so any module imports it at module top.
   Internal; not part of the public API."""

   from __future__ import annotations

   import sys
   import warnings
   ```
   followed by `_warn_at_user_frame`, cut from `src/nnx/nn/callbacks.py` (L111–L144 at `9ad461e`, as FIX-113 left it) unchanged, plus any import its body needs from the standard library.
2. In `src/nnx/nn/callbacks.py`, add `from .._warnings import _warn_at_user_frame` to the module imports (callbacks uses it itself, and the name stays importable from `nnx.nn.callbacks`). Delete `import sys` and `import warnings` from `callbacks.py` only if ruff reports them unused.
3. Replace each of the 29 local imports listed above with one module-level import in that file (deleting the local line; the code that calls it stays):

   | File | Module-level import |
   |---|---|
   | `src/nnx/_metrics.py`, `src/nnx/_scheduler_clock.py`, `src/nnx/_update_engine.py`, `src/nnx/components.py`, `src/nnx/history.py`, `src/nnx/monitors.py`, `src/nnx/provenance.py` | `from ._warnings import _warn_at_user_frame` |
   | `src/nnx/nn/nn_model.py`, `src/nnx/trainer/trainer.py` | `from .._warnings import _warn_at_user_frame` |
   | `src/nnx/nn/params/nn_run.py` | `from ..._warnings import _warn_at_user_frame` |

   Keep each end-of-line comment that sat on a deleted import as a comment on the call it explained (`_update_engine.py` L76: "attributed to the caller's train(), not NNx"; `nn_model.py` L3581: "at the first caller outside nnx, whoever called predict").
4. No test patches `_warn_at_user_frame` (`grep -rn "_warn_at_user_frame" tests` finds only two subprocess snippets in `tests/test_params_boundary_validation.py`, L511 and L2028, that import it from `nnx.nn.callbacks`, which still works).
5. Add this test to `tests/test_imports.py`:
   ```python
   def test_the_warning_helper_is_standard_library_only_and_imported_at_module_level():
       """FEAT-082: nnx._warnings imports only the standard library, and no nnx
       module imports _warn_at_user_frame inside a function."""
       import ast
       import pathlib

       import nnx
       import nnx._warnings as user_warnings
       from nnx.nn.callbacks import _warn_at_user_frame

       assert _warn_at_user_frame is user_warnings._warn_at_user_frame
       root = pathlib.Path(nnx.__file__).parent
       tree = ast.parse((root / "_warnings.py").read_text(encoding="utf-8"))
       imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
       imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module and not n.level}
       assert imported <= set(sys.stdlib_module_names) and not any(isinstance(n, ast.ImportFrom) and n.level for n in ast.walk(tree))
       local = []
       for path in root.rglob("*.py"):
           for function in (n for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
               for node in ast.walk(function):
                   if isinstance(node, ast.ImportFrom) and any(a.name == "_warn_at_user_frame" for a in node.names):
                       local.append(f"{path.relative_to(root)}:{node.lineno}")
       assert local == []
   ```

No behaviour changes, so no CHANGELOG entry.

**Worker setup.** Branch from `develop` after FIX-113 has merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The new test `tests/test_imports.py::test_the_warning_helper_is_standard_library_only_and_imported_at_module_level` fails on `develop` and passes after the change. Check: `uv run pytest tests/test_imports.py`
- [ ] Every import of the helper is at module level. Check: `grep -rn "^\s\+from .*import _warn_at_user_frame" src/nnx` prints nothing
- [ ] Warnings are still attributed to the caller's line. Check: `uv run pytest tests -k "callbacks or history_journal or provenance or metrics or component_resume or train_integration or hub_mixin or best_pointer"`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [callbacks.py L111-L144](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/callbacks.py#L111-L144) | `_warn_at_user_frame`, needing only `sys` and `warnings` |
| [history.py L1298](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/history.py#L1298) | One of 29 function-local imports |
| [nn_model.py L648](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L648), [L3581](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L3581) | Two of `nn_model.py`'s nine |
| [_update_engine.py L76](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_update_engine.py#L76) | A local import with an explanatory comment |
| [test_params_boundary_validation.py L511](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_params_boundary_validation.py#L511) | A test importing it from `nnx.nn.callbacks` |
| #522 | Attribution under `-W error` |

## Scope

**In:** `nnx/_warnings.py`, the re-export in `callbacks`, the 29 imports, one test.

**Out:**
- The shared warn-while-failing helper: FEAT-083. Replacing raw `warnings.warn` calls: FEAT-084.
- Other function-local imports in `nn_model.py`: FEAT-095.

## Dependencies

**Blocked by:** FIX-113
**Blocks:** FEAT-083, FEAT-084, FEAT-095
**Related:** #522
