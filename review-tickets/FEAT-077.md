---
id: FEAT-077
title: "[FEAT-077] Move the DDP fit helpers into nnx/nn/_ddp_fit.py (part 4 of 5)"
labels: ["enhancement", "area: execution", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-02", "SMELL-15"]
depends_on: ["FEAT-074"]
related_issues: [478, 539, 540]
status: READY
---

**Category:** Execution · **Priority:** P3 · **Size:** L · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-077: the DDP (`train(distributed=...)`) fit helpers sit in `nn/nn_model.py`, and `nn_model.py` plus the resume code imports `nnx.distributed` inside functions twelve times, although `nnx.distributed` imports nothing from NNx but `_validation`. Move the fit-level DDP helpers into a new `src/nnx/nn/_ddp_fit.py` with module-level imports, re-export them from `nn_model`, and turn every remaining function-local `from ..distributed import` in `nn_model.py` and `nn/_resume.py` into a module-level import. Part 4 of 5 of the `nn_model.py` split; no behaviour changes.

## Context

**What happens today.** The DDP preflight and session builder import `nnx.distributed` at call time:

```python
    """What DDP covers (FEAT-030), checked on each rank: the callbacks this
    rank runs, or a ValueError."""
    from ..distributed import ShardedLoader, classify_callbacks
    from .callbacks import _changes_topology
```

```python
    if spec is None:
        return None, callbacks  # writer_only(...) wrappers were built at the top of train()
    from ..distributed import DDP, _DDPSession, _world, agree
```

The twelve local imports of `nnx.distributed` at `9ad461e`: L476 and L514 (`DistributedFailure`) and L2546 (`rng_restore`), which FEAT-074 moved into `nn/_resume.py`; L2297 (`collectively`), L2326 (`ShardedLoader, classify_callbacks`), L2401 (`DDP, _DDPSession, _world, agree`) and L2437 (`_send`) in the helpers this ticket moves; L5134 (`_refuse_multi_rank`), L5254 (`collectively`), L5881, L5918 and L5957 (`agree`) in `NNModel` methods.

**Why it matters.** The local imports suggest a cycle that does not exist (SMELL-15) and scatter the DDP code through a 6,800-line file.

**History.** #478 and #539 made the DDP wrapper release on every exit path and #540 added the per-backend CUDA device advice; `tests/test_ddp_adapter.py` pins them and keeps passing unchanged. Conflict rule: FEAT-095 (SMELL-15) hoists every other needless local import in `nn_model.py` but leaves the `..distributed` ones to this ticket. FEAT-074 moved `_check_distributed_resume` into `nn/_resume.py` (it is resume code), so it does not move here.

**What stays in `nn_model.py`, and why.** The finding proposed moving L2062–L2215 and L2284–L2568 whole. Two groups stay, because moving them would need `_ddp_fit.py` to import `nn_model` at module top (a cycle):
- the step-level helpers `_agreed_step_checks`, `_global_window_check`, `_cpu`, `_global_records`, `_window_is_masked` (L2062–L2213): `default_train_step` calls them, and they use `_StepLossTerms`, `_record_step_loss`, `GradientAccumulationState` and `_DISTRIBUTED_STEP_SCORING`;
- `_Replay` and `_distributed_validation` (L2467–L2530): they score through `_evaluate`, `_inference_precision` and `_DISTRIBUTED_VALIDATION_SCORING`.
`_record_of` and `_state_of` (L2284–L2290) are compile helpers, not DDP, and stay too. None of these has a local `..distributed` import.

**Background for the worker.**
- *DDP fit helpers*: the code that checks what `distributed=` covers (`_distributed_scope`), builds the per-fit session collectively (`_distributed_session`), routes the train step through it (`_distributing`), wraps `nnx.distributed.collectively` (`_collectively`), builds writer-only callbacks (`_writer_owned_callbacks`) and records the DDP training state (`_distributed_state`).
- *Re-export*: `from ._ddp_fit import _distributed_session as _distributed_session` in `nn_model.py`; keeps the name importable (`trainer.py` imports `_writer_owned_callbacks`, `tests/test_ddp_adapter.py` imports `_distributed_scope`) and visible to `NNModel`'s call-time lookups.
- *Real cycle*: `_distributed_scope` compares the step with `default_train_step`, which lives in `nn_model`; that one name is imported inside the function, because `nn_model` imports `_ddp_fit` at module top.

**Proposal.**

1. Create `src/nnx/nn/_ddp_fit.py` with the docstring `"""DDP fit helpers for NNModel.train(distributed=...) (FEAT-030): the scope preflight, the collective session build, the step routing and the DDP training state. Moved from nn_model (FEAT-077). Internal."""`, `from __future__ import annotations`, and module-level imports:
   ```python
   import contextlib
   import os
   from collections.abc import Iterator
   from typing import Any, Optional

   import torch

   from ..distributed import DDP, ShardedLoader, _DDPSession, _send, _world, agree, classify_callbacks, collectively
   from ..precision import ResolvedPrecision
   from .callbacks import _changes_topology, _WriterOwned
   from .enum.nets import Nets
   ```
2. Move these definitions from `nn_model.py` into it, in this order, unchanged except for imports (at `9ad461e`):

   | Definition | Lines |
   |---|---|
   | `_GRAPH_NETS` | L2244 |
   | `_collectively` | L2292–L2299 |
   | `_writer_owned_callbacks` | L2302–L2308 |
   | `_distributed_scope` | L2311–L2380 |
   | `_distributed_session` | L2383–L2453 |
   | `_distributing` | L2456–L2464 |
   | `_distributed_state` | L2533–L2540 |

   Inside them, delete the local `from ..distributed import ...` and `from .callbacks import ...` lines (the names are module-level now), and add `    from .nn_model import default_train_step  # nn_model imports this module at top` as the first statement of `_distributed_scope`.
3. In `nn_model.py`, add the re-exports with the redundant alias (`from ._ddp_fit import _GRAPH_NETS as _GRAPH_NETS`, and likewise for the six functions). `_compile_session` keeps using `_GRAPH_NETS` through the re-export.
4. Hoist the remaining local imports of `nnx.distributed`:
   - `nn_model.py`: add `from ..distributed import _refuse_multi_rank, agree, collectively` to the module imports and delete the local lines at L5134, L5254, L5881, L5918 and L5957 (at `9ad461e`);
   - `src/nnx/nn/_resume.py` (from FEAT-074): add `from ..distributed import DistributedFailure, rng_restore` to its module imports and delete its three local `from ..distributed import` lines (in `_verified_resume_point`, `_resume_source_id` and `_check_distributed_resume`).
   `nnx.distributed` imports only the standard library, torch and `nnx._validation` at module top, so these imports add no cycle.
5. Run `uv run ruff check --fix src/nnx/nn/nn_model.py src/nnx/nn/_ddp_fit.py src/nnx/nn/_resume.py` to sort imports and drop the ones the remaining `nn_model.py` code no longer uses.
6. No test patch needs a new target: no test patches an attribute of `nnx.distributed` or a moved name (`grep -rn 'setattr(nnx_dist\|patch("nnx.distributed\|"_distributed_\|"_collectively"\|"_writer_owned_callbacks"' tests` prints nothing). `tests/ddp_scenarios.py` L1300 replaces `nn_model._scale_gradients`, which `default_train_step` (still in `nn_model`) reads; `tests/test_ddp_adapter.py` L756–L775 patches `torch.distributed.get_backend`, a library attribute.
7. Add this test to `tests/test_imports.py`:
   ```python
   def test_no_core_module_imports_nnx_distributed_inside_a_function():
       """FEAT-077: nnx.distributed has no cycle back into nnx.nn, so nn_model,
       _resume and _ddp_fit import it at module level only."""
       import ast
       import pathlib

       import nnx.nn as nn_package
       import nnx.nn._ddp_fit as ddp_fit
       import nnx.nn.nn_model as nn_model

       for name in ("_distributed_scope", "_distributed_session", "_distributing", "_distributed_state",
                    "_collectively", "_writer_owned_callbacks", "_GRAPH_NETS"):
           assert getattr(nn_model, name) is getattr(ddp_fit, name), name
       root = pathlib.Path(nn_package.__file__).parent
       local = []
       for module in ("nn_model.py", "_resume.py", "_ddp_fit.py"):
           tree = ast.parse((root / module).read_text(encoding="utf-8"))
           for function in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
               for node in ast.walk(function):
                   if isinstance(node, ast.ImportFrom) and node.level == 2 and node.module == "distributed":
                       local.append(f"{module}:{node.lineno}")
       assert local == []
   ```

No behaviour changes, so no CHANGELOG entry.

**Worker setup.** Branch from `develop` after FEAT-074 has merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need; the DDP tests run two Gloo ranks on CPU.

## Acceptance criteria

- [ ] The new test `tests/test_imports.py::test_no_core_module_imports_nnx_distributed_inside_a_function` fails before the change and passes after it. Check: `uv run pytest tests/test_imports.py`
- [ ] `nn_model.py` has no `..distributed` import inside a function. Check: `grep -n "^\s\+from \.\.distributed import" src/nnx/nn/nn_model.py src/nnx/nn/_resume.py src/nnx/nn/_ddp_fit.py` prints nothing
- [ ] `_ddp_fit.py` imports `nn_model` only inside `_distributed_scope`. Check: `grep -n "nn_model" src/nnx/nn/_ddp_fit.py` prints one line, the `default_train_step` import
- [ ] The DDP, compile, trainer and import suites pass. Check: `uv run pytest tests/test_ddp_adapter.py tests/test_compile_execution.py tests/test_trainer.py tests/test_core_install.py tests/test_imports.py`
- [ ] The full suite passes. Check: `uv run pytest -n logical --dist loadgroup`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [nn_model.py L2062-L2213](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L2062-L2213) | Step-level DDP helpers that stay (used by `default_train_step`) |
| [nn_model.py L2244](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L2244) | `_GRAPH_NETS` |
| [nn_model.py L2292-L2464](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L2292-L2464) | `_collectively` through `_distributing`, with four local `..distributed` imports |
| [nn_model.py L2467-L2540](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L2467-L2540) | `_Replay` and `_distributed_validation` (stay), `_distributed_state` (moves) |
| [nn_model.py L5134](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L5134), [L5254](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L5254), [L5881](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L5881), [L5918](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L5918), [L5957](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L5957) | The local imports in `NNModel` methods |
| [distributed.py L97-L113](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/distributed.py#L97-L113) | `nnx.distributed`'s module imports: no `nnx.nn` |
| [test_ddp_adapter.py L750-L780](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_ddp_adapter.py#L750-L780) | `_distributed_scope` imported from `nn_model` (#540) |
| #478, #539, #540 | Wrapper release and CUDA device advice pinned by the DDP tests |

## Scope

**In:** `_ddp_fit.py` with the seven definitions, their re-exports, the twelve `..distributed` imports hoisted, one test.

**Out:**
- The step-level DDP helpers, `_Replay`, `_distributed_validation`, `_record_of` and `_state_of` (see above).
- Other function-local imports in `nn_model.py`: FEAT-095.
- Any change to DDP behaviour or messages.

## Dependencies

**Blocked by:** FEAT-074
**Blocks:** none
**Related:** #478, #539, #540, FEAT-095
