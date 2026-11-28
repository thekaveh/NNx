---
id: FEAT-074
title: "[FEAT-074] Move the resume machinery out of nn_model.py (part 1 of 5)"
labels: ["enhancement", "area: training", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-02"]
depends_on: ["FEAT-081", "FEAT-079", "FEAT-088", "FEAT-089", "FEAT-090", "FEAT-103"]
related_issues: [536, 507, 206]
status: READY
---

**Category:** Training · **Priority:** P3 · **Size:** L · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-074: `src/nnx/nn/nn_model.py` is 6,826 lines; about 1,100 of them are the resume machinery `NNModel` and `Trainer` share (#536). Move that code, unchanged, into a new `src/nnx/nn/_resume.py`, re-export every moved name from `nn_model` so `from nnx.nn.nn_model import X` and `NNModel`'s own calls keep working, and retarget the three test patches whose target now lives in `_resume`. This is part 1 of 5 of the `nn_model.py` split; it changes no behaviour.

## Context

**What happens today.** `Trainer` imports the shared resume code from `nn_model`:

```python
from ..nn.nn_model import (
```

and `nn_model.py` holds, among `NNModel`'s other concerns, the whole resume path, from the training-state helpers at L203 to the end of `_resume_fit` at L1071:

```python
def _resume_fit(
```

plus `_check_resume_precision` (L1889), `_check_distributed_resume` (L2543) and the hyperparameter checks from `_check_plateau_resume` (L2797) to `_saved_hyperparameters` (L2949–L2974).

**Why it matters.** The file mixes training, evaluation, export, Hub I/O, DDP and resume; reviews and refactors of one concern drag in all the others. Other modules import 49 private names from it.

**History.** #536 (FEAT-049) made `NNModel` and `Trainer` share one resume, rollback and best-epoch path; it moves here as one unit. NNModel-only checks reach it as hooks and stay so. #507 (FIX-040) and #206 (FIX-007) added behaviour inside this code that the moved tests keep pinning. The tickets this one waits for (FEAT-079, FEAT-081, FEAT-088, FEAT-089, FEAT-090, FEAT-103) edit code in the moved ranges; moving after them avoids conflicts.

**Background for the worker.**
- *Re-export*: in `nn_model.py`, `from ._resume import _resume_fit as _resume_fit`. The redundant `as` alias tells ruff (F401) and pyright the import is deliberate; the name stays importable from `nn_model`, and `NNModel`'s methods, which look names up in `nn_model`'s globals at call time, find it there.
- *Call-time lookup and patches*: a function looks up module-level names in the globals of the module that defines it. After the move, the moved functions read `_resume`'s globals, so a test that patches a name *they* use must patch `nnx.nn._resume`. A test that patches a name `NNModel` or `Trainer` call (for example `_resume_fit`) keeps patching `nn_model` / `trainer`.
- *Cycle rule*: `nn_model` imports `_resume` at module top, so `_resume` must not import `nn_model` at module top. At `9ad461e` the moved code uses no `nn_model` name outside the moved set (checked by AST), so this holds.

**Proposal.**

1. Create `src/nnx/nn/_resume.py` with the docstring `"""The resume, rollback and resume-check machinery NNModel and Trainer share (#536). Moved from nn_model (FEAT-074). Internal."""`, `from __future__ import annotations`, and these top-level imports (the names the moved code uses, copied from `nn_model.py`'s header):
   ```python
   import contextvars
   import functools
   import json
   import os
   import re
   import uuid
   from collections import OrderedDict
   from collections.abc import Callable, Mapping, Sequence
   from dataclasses import dataclass, field
   from typing import Any, NamedTuple, Optional, cast

   import torch
   import yaml
   from torch.optim import lr_scheduler

   from .._scheduler_clock import HORIZON_KINDS, uses_update_clock
   from ..components import _BUNDLE_ATTEMPT_PREFIX, ComponentRegistry, ComponentRestoreError, ResumeStatus
   from ..monitors import MonitorSpec
   from ..precision import ResolvedPrecision
   from ..seeding import _capture_rng_state, _restore_rng_state
   from ..transforms import _recipe_transforms
   from .enum.checkpoints import Checkpoints
   from .params.nn_checkpoint import _MODEL_CHECKPOINT_TAG, NNCheckpoint, ResumePointError, _snapshot_state_dict
   from .params.nn_optim_params import NNOptimParams
   from .params.nn_run import _may_name_best_pointer, _recorded_attempt_token, _resolve_run_id, _runs_root, _validate_run_id
   from .params.nn_train_params import NNTrainParams
   ```
   After step 2, make this list exact mechanically, not by judgement:
   - Run `uv run ruff check --select F821 src/nnx/nn/_resume.py`. For each undefined name it reports, copy the import line `nn_model.py` uses for that name into `_resume.py`. If the name is defined in `nn_model.py` itself outside the moved blocks, stop and report it in the PR: importing it back would create a cycle.
   - Then run `uv run ruff check --select F401 --fix src/nnx/nn/_resume.py` to drop the imports the moved code does not use.
2. Cut these blocks from `nn_model.py` and paste them into `_resume.py` in the same order, unchanged (function-local imports inside them stay as they are; their relative paths are the same because both modules sit in `src/nnx/nn/`):

   | Block (anchors survive the dependency tickets) | At `9ad461e` |
   |---|---|
   | From the first definition after `CallbackLike = Union["Callback", LegacyCallback]` through the end of `_resume_fit` | L203–L1071 |
   | `_check_resume_precision` | L1889–L1911 |
   | `_check_distributed_resume` | L2543–L2566 |
   | From `def _check_plateau_resume` through the end of `_saved_hyperparameters` | L2797–L2974 |

   At `9ad461e` the first block holds `_component_type`, `_optimizer_topology`, `_named_training_state`, `_loader_num_workers`, `_resume_checkpoint_type`, `_check_resume_horizon`, `_manifest_lineage`, `_session_epochs`, `_ResumeSource`, `_new_attempt_token`, `_IN_MEMORY_RESUME`, `_RESUME_SOURCE_LISTENER`, `_attached_submodules_error`, `_excluded_names`, `_without_submodules`, `_verified_resume_point`, `_resume_source_id`, `_load_verified_resume_point`, `_load_resume_source`, `_rollback_resume`, `_ResumeWindow`, `_resume_training_state`, `_plan_component_restore`, `_Optimization`, `_owner`, `_check_resume_horizons`, `_ONE_OPTIMIZER_KEYS`, `_SavedCheckpoints`, `_Resumed` and `_resume_fit`; the last block holds `_check_plateau_resume`, `_RESTORED_SCHEDULER_FIELDS`, `_optimizer_config`, `_scheduler_config`, `_resume_hyperparameters`, `_source_hyperparameters`, `_check_resume_hyperparameters` and `_saved_hyperparameters`. Move exactly the code between the anchors in the table, whatever those ranges hold once the dependency tickets have landed. The list above is what they enclose at `9ad461e`. A definition a dependency ticket added inside an anchored range moves with it; one it added outside the ranges stays in `nn_model.py`. `_check_distributed_resume` moves here, not with the DDP helpers (FEAT-077), because `_resume_fit` calls it.
3. In `nn_model.py`, where the first block was, add one import of every moved name with the redundant alias, for example:
   ```python
   # The shared resume machinery (#536) lives in nnx.nn._resume (FEAT-074);
   # re-exported so `from nnx.nn.nn_model import <name>` keeps working.
   from ._resume import _check_distributed_resume as _check_distributed_resume
   from ._resume import _check_plateau_resume as _check_plateau_resume
   ```
   and so on for every moved name (one line each, or one parenthesized import with `name as name` entries). Then run `uv run ruff check --fix src/nnx/nn/nn_model.py` to sort the imports and drop the ones the remaining code no longer uses (at `9ad461e`: `uuid`, `yaml`, `HORIZON_KINDS`, `ResumeStatus`, `ComponentRestoreError`, `_BUNDLE_ATTEMPT_PREFIX`, `ResumePointError`, `_MODEL_CHECKPOINT_TAG` and the five `nn_run` run-id helpers). The `as`-aliased re-exports are not touched by `--fix`.
4. `src/nnx/trainer/trainer.py`, `plans.py`, `bundles.py`, `history.py`, `provenance.py` and the tests keep importing these names from `nnx.nn.nn_model`; change none of them.
5. Retarget the test patches whose target the moved code reads (capture and patch together):

   | Test (line at `9ad461e`) | Today | New target |
   |---|---|---|
   | `tests/test_component_resume.py` L434, L441 | `nn_model_module.ResumeStatus` | `resume_module.ResumeStatus`, with `import nnx.nn._resume as resume_module` beside the test's `import nnx.nn.nn_model as nn_model_module` |
   | `tests/test_component_resume.py` L507 | `nn_model_module._loader_num_workers` | `resume_module._loader_num_workers`, with the same import added in that test |
   | `tests/test_named_monitors.py` L775, L782 | `resume_module.ResumeStatus` (`resume_module` is `nnx.nn.nn_model`, L764) | `nnx.nn._resume`: change L764 and L813 to `import nnx.nn._resume as resume_module` |
   | `tests/test_named_monitors.py` L815 | `resume_module._loader_num_workers` | covered by the L813 change |
   | `tests/test_resume_parity.py` L400, L405 | `nn_model._optimizer_topology` | `resume_module._optimizer_topology`, with `import nnx.nn._resume as resume_module` added to the module imports |

   These stay as they are, because `NNModel` or `Trainer` call the patched name through their own module: `tests/test_resume_parity.py` L215–L216 (`_resume_fit` on `nn_model` and `trainer`), L343–L344 (`_best_err`), L436–L437 (`_topology_preflight`); `tests/test_component_resume.py` L486 (`warn_unowned_trainables`); `tests/test_precision_execution.py` L168–L169; `tests/test_precision_policy.py` L305; `tests/test_trainer.py` L2300; `tests/ddp_scenarios.py` L1300. Direct calls such as `nn_model._resume_source_id(...)` (`tests/test_best_pointer_readers.py`) and `module._load_resume_source(...)` (`tests/test_model_recipes.py`) reach the re-export and stay.
6. Add this test to `tests/test_imports.py`:
   ```python
   def test_the_resume_machinery_lives_in_nnx_nn_resume_and_stays_importable_from_nn_model():
       """FEAT-074: nn_model re-exports the moved names; nothing in _resume imports nn_model at module top."""
       import ast
       import pathlib

       import nnx.nn._resume as resume
       import nnx.nn.nn_model as nn_model

       for name in ("_resume_fit", "_ResumeWindow", "_Optimization", "_SavedCheckpoints", "_check_resume_horizons",
                    "_resume_hyperparameters", "_check_resume_precision", "_check_distributed_resume",
                    "_load_verified_resume_point", "_resume_checkpoint_type", "_IN_MEMORY_RESUME"):
           assert getattr(nn_model, name) is getattr(resume, name), name
       tree = ast.parse(pathlib.Path(resume.__file__).read_text(encoding="utf-8"))
       assert not [n for n in tree.body if isinstance(n, ast.ImportFrom) and (n.module or "").endswith("nn_model")]
   ```
   Drop from the tuple any name a dependency ticket removed.

No behaviour changes, so no CHANGELOG entry. Pickled class paths are unaffected: none of the moved classes is pickled (`NNModel` and `_CallbackContext` stay in `nn_model`).

**Worker setup.** Branch from `develop` after FEAT-079, FEAT-081, FEAT-088, FEAT-089, FEAT-090 and FEAT-103 have merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need. Run the listed tests before and after the move; both runs must pass.

## Acceptance criteria

- [ ] The new test `tests/test_imports.py::test_the_resume_machinery_lives_in_nnx_nn_resume_and_stays_importable_from_nn_model` fails before the move and passes after it. Check: `uv run pytest tests/test_imports.py`
- [ ] `nn_model.py` defines none of the moved names. Check: `grep -nE "^(def|class) (_resume_fit|_ResumeWindow|_Optimization|_check_resume_precision|_check_distributed_resume|_saved_hyperparameters|_load_resume_source)\b" src/nnx/nn/nn_model.py` prints nothing
- [ ] No test patches a moved name on `nn_model`. Check: `grep -rnE 'setattr\((nn_model|nn_model_module), "(ResumeStatus|_loader_num_workers|_optimizer_topology)"' tests` prints nothing, and `grep -rn "import nnx.nn.nn_model as resume_module" tests` prints nothing
- [ ] The resume, trainer and DDP suites pass. Check: `uv run pytest tests -k "resume or trainer or ddp or component or named_monitors or best_pointer or model_recipes or exact_resume or precision"`
- [ ] The full suite passes. Check: `uv run pytest -n logical --dist loadgroup`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [nn_model.py L199-L1071](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L199-L1071) | `CallbackLike`, then the resume machinery through `_resume_fit` |
| [nn_model.py L1889-L1911](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1889-L1911) | `_check_resume_precision` |
| [nn_model.py L2543-L2566](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L2543-L2566) | `_check_distributed_resume`, called by `_resume_fit` (L1036) |
| [nn_model.py L2797-L2974](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L2797-L2974) | `_check_plateau_resume` through `_saved_hyperparameters` |
| [trainer.py L59-L84](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/trainer/trainer.py#L59-L84) | `Trainer`'s private imports from `nn_model` |
| [test_component_resume.py L434-L441](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_component_resume.py#L434-L441), [L507](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_component_resume.py#L507) | Patches of `ResumeStatus` and `_loader_num_workers` |
| [test_named_monitors.py L764-L815](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_named_monitors.py#L764-L815) | The same patches through `resume_module` |
| [test_resume_parity.py L400-L405](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_resume_parity.py#L400-L405) | The `_optimizer_topology` patch |
| #536 | The shared resume path that moves as one unit |

## Scope

**In:** the four blocks above, their re-exports, five test retargets, one test.

**Out:**
- Transform replay: FEAT-075. Hub I/O: FEAT-076. DDP fit helpers: FEAT-077. Lazy-module descriptor: FEAT-078.
- Hoisting the moved code's function-local imports (the `..distributed` ones are FEAT-077's; the rest stay).
- Any change to resume behaviour, messages or the on-disk format.

## Dependencies

**Blocked by:** FEAT-079, FEAT-081, FEAT-088, FEAT-089, FEAT-090, FEAT-103
**Blocks:** FEAT-076, FEAT-077
**Related:** #536, #507, #206
