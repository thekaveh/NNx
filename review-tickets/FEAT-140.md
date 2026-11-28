---
id: FEAT-140
title: "[FEAT-140] Hoist needless function-local imports in bundles and export_conformance"
labels: ["enhancement", "area: persistence", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-61"]
depends_on: ["FEAT-061", "FEAT-072", "FEAT-091", "FIX-109", "FIX-122"]
related_issues: [249, 552, 255, 566]
status: READY
---

**Category:** Persistence · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-140: `src/nnx/bundles.py` imports 17 times inside functions and `src/nnx/export_conformance.py` 18 times. Two `bundles` imports are #552's flush test seams (owned by FEAT-072) and six `export_conformance` imports load the optional `onnx` / `onnxruntime` packages; the rest guard nothing. This ticket hoists the 15 + 12 needless ones and marks each kept optional import with `# optional extra: <extra>`.

## Context

**What happens today.** `export_bundle` (`bundles.py`):

```python
    from filelock import FileLock

    from .nn.nn_model import _resume_checkpoint_type, _unportable_transform
    from .nn.params.nn_checkpoint import NNCheckpoint
    from .nn.params.nn_run import _addresses_best_pointer
```

(`filelock` is a core dependency, `pyproject.toml` L102) and `build_model` (`export_conformance.py`) imports four enums, `NNModel`, `NNModelParams` and `NNParams` locally; `_versions` and `_source` each run `from . import __version__`.

**Why it matters.** The local imports hide the modules' dependencies and suggest cycles; only the optional `onnx` / `onnxruntime` imports are needed where they are.

**History.** #249 (FEAT-031) set the rule: keep an import local only for an optional extra or a real cycle. #552 made `bundles._write` and `bundles._fsync_directory` read `nn_checkpoint._full_fsync` / `_fsync_directory` at call time, so a test that replaces the flush primitive on `nn_checkpoint` sees every bundle flush (`tests/test_run_bundles.py` L703-L725, `tests/test_bundle_validation.py` L595); those two stay as FEAT-072 leaves them. #255 (FEAT-038) made `nnx.export_conformance` public with `import nnx` working without the `onnx` extra, which the kept imports preserve; #566 checks names before writing.

**Background for the worker.**
- Classes: **hoist** (move to module top), **keep-cycle**, **keep-optional-extra**.
- Cycle check done for this ticket: only `nnx/__init__.py` (and `nnx/result.py`, inside a function) imports `nnx.bundles` or `nnx.export_conformance`, and none of the hoisted targets (`nnx.nn.nn_model`, `nnx.nn.params.*`, `nnx.transforms`, `nnx.models`, `nnx.nn.enum.*`, `nnx._artifacts`) reaches either module through its module-top imports. No hoisted name collides with a module-level name.
- `from . import __version__` at module top is safe: `nnx/__init__.py` sets `__version__` (L28-L32) before it imports any submodule, and a lazily loaded module (FEAT-059) is imported after `nnx/__init__.py` finishes.
- Extras: `onnx` provides `onnx`; `onnx-runtime` provides `onnxruntime` (`pyproject.toml` `[project.optional-dependencies]`).
- Names changed by the dependencies: FIX-109 may rename the run-id validator `_check_state` imports; FEAT-091 moves `_unportable_transform` into `nnx.transforms` (keeping an alias in `nn_model`). Hoist the import as those tickets left it.

**Proposal.**
1. `src/nnx/bundles.py` — every function-local import at `9ad461e`:

   | Line | Function | Import | Class |
   |---|---|---|---|
   | L224 | `ReconstructedBundle.resume` | `from .nn.nn_model import _IN_MEMORY_RESUME` | hoist |
   | L244 | `ReconstructedBundle.resume` | `from .transforms import _replayable` | hoist |
   | L849 | `_check_state` | `from .nn.params.nn_run import _validate_run_id` | hoist |
   | L871 | `_check_state` | `from .nn.params.nn_checkpoint import _TRAINING_STATE_FORMAT_VERSION` | hoist |
   | L925 | `_checkpoint_parts` | `from .nn.params.nn_checkpoint import NNCheckpointTransform, _idp_from_nested_state` | hoist |
   | L926 | `_checkpoint_parts` | `from .nn.params.nn_model_params import NNModelParams` | hoist |
   | L927 | `_checkpoint_parts` | `from .nn.params.nn_params import NNParams` | hoist |
   | L1138 | `_write` | `from .nn.params.nn_checkpoint import _full_fsync` | — (#552 test seam; FEAT-072) |
   | L1158 | `_fsync_directory` | `from .nn.params.nn_checkpoint import _fsync_directory as fsync_directory` | — (#552 test seam; FEAT-072) |
   | L1204 | `export_bundle` | `from filelock import FileLock` | hoist (core dependency) |
   | L1206 | `export_bundle` | `from .nn.nn_model import _resume_checkpoint_type, _unportable_transform` | hoist |
   | L1207 | `export_bundle` | `from .nn.params.nn_checkpoint import NNCheckpoint` | hoist |
   | L1208 | `export_bundle` | `from .nn.params.nn_run import _addresses_best_pointer` | hoist |
   | L1459 | `reconstruct_bundle` | `from .nn.nn_model import NNModel, _excluded_names` | hoist |
   | L1460 | `reconstruct_bundle` | `from .nn.params.nn_checkpoint import NNCheckpoint` | hoist |
   | L1484 | `reconstruct_bundle` | `from .models import resolve_model_factory` | hoist (into the existing `.models` line, L78) |
   | L1514 | `reconstruct_bundle` | `from .models import _supplied_factories` | hoist (same line) |

   Delete `NNModel` and `NNCheckpoint` from the `TYPE_CHECKING` block (L81-L82; both become runtime imports); keep `NNRun` and `NNTrainParams` there. Do not touch `_write` or `_fsync_directory`.
2. `src/nnx/export_conformance.py` — every function-local import at `9ad461e`:

   | Line | Function | Import | Class |
   |---|---|---|---|
   | L237 | `_check_model` | `from .nn.enum.activations import Activations` | hoist |
   | L406 | `build_model` | `from .nn.enum.activations import Activations` | hoist |
   | L407 | `build_model` | `from .nn.enum.devices import Devices` | hoist |
   | L408 | `build_model` | `from .nn.enum.losses import Losses` | hoist |
   | L409 | `build_model` | `from .nn.enum.nets import Nets` | hoist |
   | L410 | `build_model` | `from .nn.nn_model import NNModel` | hoist |
   | L411 | `build_model` | `from .nn.params.nn_model_params import NNModelParams` | hoist |
   | L412 | `build_model` | `from .nn.params.nn_params import NNParams` | hoist |
   | L468 | `_version` | `from importlib import metadata` | hoist (standard library) |
   | L477 | `_versions` | `from . import __version__` | hoist |
   | L516 | `_source` | `from . import __version__` | hoist |
   | L530 | `_external_locations` | `import onnx` | keep-optional-extra: append `  # optional extra: onnx` |
   | L579 | `_manifest` | `import onnx` | keep-optional-extra: `# optional extra: onnx` |
   | L726 | `_signature` | `import onnx` | keep-optional-extra: `# optional extra: onnx` |
   | L742 | `_check` | `import onnx` | keep-optional-extra: `# optional extra: onnx` |
   | L771 | `_load` | `import onnxruntime as ort` | keep-optional-extra: `# optional extra: onnx-runtime` |
   | L796 | `_parity` | `from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument` | keep-optional-extra: `# optional extra: onnx-runtime` |
   | L1054 | `save_report` | `from ._artifacts import atomic_write` | hoist (into `from ._artifacts import file_sha256`, L25) |

3. Any function-local import that FEAT-061, FEAT-072, FEAT-091, FIX-109 or FIX-122 added to either file follows the same rule (hoist, unless it loads an optional extra; never touch `_write` / `_fsync_directory`).
4. Delete each hoisted local import line and any blank line it leaves at the start of a function body. Run `uv run ruff check --fix src/nnx/bundles.py src/nnx/export_conformance.py` and `uv run ruff format src/nnx/bundles.py src/nnx/export_conformance.py`.

No behaviour changes, no test changes, no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need (`onnx`, `onnxruntime`).

## Acceptance criteria

- [ ] In `bundles.py`, every function-local import left sits in `_write` or `_fsync_directory`. Check: `uv run python -c "import ast; t=ast.parse(open('src/nnx/bundles.py').read()); print([f.name for f in ast.walk(t) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) if any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(f))])"` prints a subset of `['_write', '_fsync_directory']`
- [ ] In `export_conformance.py`, every function-local import left loads `onnx` or `onnxruntime` and is marked. Check: `uv run python -c "import ast; p='src/nnx/export_conformance.py'; src=open(p).read(); lines=src.splitlines(); print([n.lineno for f in ast.walk(ast.parse(src)) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(f) if isinstance(n, (ast.Import, ast.ImportFrom)) and '# optional extra:' not in lines[n.lineno - 1]])"` prints `[]`
- [ ] `import nnx` still works without the onnx extra, and the flush seams still see every bundle flush. Check: `uv run pytest tests/test_imports.py tests/test_core_install.py tests/test_run_bundles.py::test_every_bundle_file_and_directory_is_flushed tests/test_bundle_validation.py`
- [ ] The bundle and conformance suites pass. Check: `uv run pytest tests/test_run_bundles.py tests/test_export_conformance.py tests/test_bundle_validation.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [bundles.py L74-L84](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/bundles.py#L74-L84) | the module-top imports and the `TYPE_CHECKING` block |
| [bundles.py L1136-L1162](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/bundles.py#L1136-L1162) | `_write`, `_fsync_directory`: #552's seams |
| [bundles.py L1204-L1208](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/bundles.py#L1204-L1208) | `export_bundle`'s local imports |
| [bundles.py L1459-L1514](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/bundles.py#L1459-L1514) | `reconstruct_bundle`'s local imports |
| [export_conformance.py L22-L25](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/export_conformance.py#L22-L25) | the module-top imports |
| [export_conformance.py L406-L412](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/export_conformance.py#L406-L412) | `build_model`'s seven local imports |
| [export_conformance.py L467-L516](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/export_conformance.py#L467-L516) | `_version`, `_versions`, `_source` |
| [export_conformance.py L530](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/export_conformance.py#L530), [L771](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/export_conformance.py#L771) | the optional `onnx` / `onnxruntime` imports that stay |
| [test_run_bundles.py L703-L725](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_run_bundles.py#L703-L725) | the flush seam test |
| #249, #552, #255, #566 | the lazy-extras rule; the flush seams; the conformance module's scope |

## Scope

**In:** the function-local imports of `bundles.py` and `export_conformance.py`, and the optional-extra markers.

**Out:** `_write` / `_fsync_directory` (FEAT-072). Loading `nnx.export_conformance` lazily on `import nnx` (FEAT-059). Splitting `validate_record` (FEAT-061). Routing the `onnx` imports through `nnx._optional.require` (not ticketed).

## Dependencies

**Blocked by:** FEAT-061, FEAT-072, FEAT-091, FIX-109, FIX-122 (each edits these modules' imports or the functions that hold them)
**Blocks:** none
**Related:** #249, #552, #255, #566
