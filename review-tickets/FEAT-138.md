---
id: FEAT-138
title: "[FEAT-138] Hoist needless function-local imports in nnx.plans"
labels: ["enhancement", "area: fluent-api", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-61"]
depends_on: []
related_issues: [249]
status: READY
---

**Category:** Fluent API · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-138: `src/nnx/plans.py` (`ExperimentPlan`) imports 36 times inside functions, from 15 core modules, although none of them imports `nnx.plans` back and none is an optional extra. This ticket moves all 36 to module top. One of them, `_horizon_problem`, is a test seam, so it is reached through its module object to keep the seam working.

## Context

**What happens today.** For example, `ExperimentPlan._check_monitoring` begins:

```python
        from .monitors import MonitorSpec
        from .nn.nn_model import _monitor_problem

        broken = False
```

and `_check_callback_monitors` repeats the same pair; `_schedule_horizon_problem` begins:

```python
    from ._scheduler_clock import planned_updates, uses_update_clock
    from .nn.enum.schedulers import _horizon_problem
    from .nn.nn_model import default_train_step
```

`default_train_step` is imported locally three times. The module-top block imports only `nn.enum`, `nn.params`, `precision` and `tasks`.

**Why it matters.** 36 local imports hide what the plan validator depends on and suggest cycles that do not exist.

**History.** #249 (FEAT-031) set the rule: keep an import local only for an optional extra or a real cycle. No board issue made these deliberate.

**Background for the worker.**
- Classes: **hoist** (move to module top), **keep-cycle**, **keep-optional-extra**. Every import here is **hoist**.
- Cycle check done for this ticket: only `nnx/__init__.py` and `nnx/search.py` (inside functions) import `nnx.plans`, and none of the 15 target modules (`nnx.trainer.params`, `nnx.nn.nn_model`, `nnx.streaming`, `nnx.link_tasks`, `nnx.models`, `nnx.tasks`, `nnx.optimizers`, `nnx.components`, `nnx.objectives`, `nnx.monitors`, `nnx.seeding`, `nnx.distributed`, `nnx.nn.callbacks`, `nnx._scheduler_clock`, `nnx.nn.enum.schedulers`) reaches `nnx.plans` through its module-top imports. No hoisted name collides with a name `plans.py` defines, or is shadowed by a parameter or local variable.
- Test seam: `tests/test_experiment_plan.py::test_the_build_and_the_plan_share_one_horizon_check` (L997-L1006) monkeypatches `nnx.nn.enum.schedulers._horizon_problem` and expects `ExperimentPlan.validate()` to see the replacement. A local import reads the module attribute at call time; a module-top `from ... import _horizon_problem` would bind the original and break that test. So plans imports the module and calls `_schedulers._horizon_problem(...)`.
- No other test monkeypatches a name these imports bind (`git grep -nE "setattr\((nn_model|monitors|models|seeding|link_tasks|streaming|optimizers|objectives|components|distributed|callbacks|_scheduler_clock|params), \"(_is_streaming_eval_step|_streaming_problems|_link_eval_problems|_resolve_net_descriptor|resolve_model_factory|task_adapter|resolve_optimizer_factory|_check_provenance|_check_step_precision|_check_update_owner|_resume_checkpoint_type|_check_resume_horizon|_monitor_problem|_check_metric_inputs|_metric_context_of|default_train_step|_set_seed|_inference_precision|_step_loss_terms|_seed_scope|_refuse_multi_rank|uses_update_clock|planned_updates|_edp_metric_iter)\"" tests/` returns 0 lines; the `NNModel.__init__` patches at `tests/test_experiment_plan.py` L201 and L367 patch the class, which stays the same object).

**Proposal.**
1. Every function-local import in `src/nnx/plans.py` at `9ad461e`:

   | Line | Function | Import | Class |
   |---|---|---|---|
   | L260 | `_frozen_mapping` | `from .trainer.params import _FrozenMapping` | hoist |
   | L334 | `ExperimentPlan.__post_init__` | `from .trainer.params import _FrozenMapping` | hoist |
   | L516 | `ExperimentPlan._streaming_validation` | `from .nn.nn_model import _is_streaming_eval_step` | hoist |
   | L525 | `ExperimentPlan._check_eval_step` | `from .streaming import _streaming_problems` | hoist |
   | L530 | `ExperimentPlan._check_eval_step` | `from .link_tasks import FACTORY_ID, _link_eval_problems` | hoist |
   | L531 | `ExperimentPlan._check_eval_step` | `from .models import ModelSpec` | hoist |
   | L543 | `ExperimentPlan._check_model` | `from .models import MissingModelFactoryError` | hoist |
   | L544 | `ExperimentPlan._check_model` | `from .nn.nn_model import _resolve_net_descriptor` | hoist |
   | L560 | `ExperimentPlan._check_model_parts` | `from .models import BatchAdapter, MissingModelFactoryError, ModelSpec, resolve_model_factory` | hoist |
   | L561 | `ExperimentPlan._check_model_parts` | `from .tasks import task_adapter` | hoist |
   | L621 | `ExperimentPlan._check_train` | `from .optimizers import NNOptimFactoryParams, resolve_optimizer_factory` | hoist |
   | L715 | `ExperimentPlan._check_callbacks` | `from .components import StatefulComponent` | hoist |
   | L739 | `ExperimentPlan._check_steps` | `from .nn.nn_model import _check_provenance, _check_step_precision` | hoist |
   | L751 | `ExperimentPlan._check_steps` | `from .objectives import _check_update_owner` | hoist |
   | L783 | `ExperimentPlan._check_resume` | `from .nn.nn_model import _resume_checkpoint_type` | hoist |
   | L854 | `ExperimentPlan._check_resume_horizon` | `from .nn.nn_model import _check_resume_horizon` | hoist |
   | L873 | `ExperimentPlan._check_monitoring` | `from .monitors import MonitorSpec` | hoist |
   | L874 | `ExperimentPlan._check_monitoring` | `from .nn.nn_model import _monitor_problem` | hoist |
   | L925 | `ExperimentPlan._check_callback_monitors` | `from .monitors import MonitorSpec` | hoist |
   | L926 | `ExperimentPlan._check_callback_monitors` | `from .nn.nn_model import _monitor_problem` | hoist |
   | L953 | `ExperimentPlan._metric_input_problem` | `from .nn.nn_model import _check_metric_inputs, _metric_context_of` | hoist |
   | L954 | `ExperimentPlan._metric_input_problem` | `from .tasks import task_adapter` | hoist |
   | L1014 | `ExperimentPlan._default_step` | `from .nn.nn_model import default_train_step` | hoist |
   | L1026 | `ExperimentPlan._apply_seed` | `from .seeding import _set_seed` | hoist |
   | L1031 | `ExperimentPlan._build_model` | `from .nn.nn_model import NNModel` | hoist |
   | L1126 | `ExperimentPlan.probe` | `from .nn.nn_model import _inference_precision, _step_loss_terms` | hoist |
   | L1127 | `ExperimentPlan.probe` | `from .seeding import _seed_scope` | hoist |
   | L1208 | `ExperimentPlan.fit` | `from .distributed import _refuse_multi_rank` | hoist |
   | L1312 | `_is_callback` | `from .nn.callbacks import Callback` | hoist |
   | L1328 | `_update_budget_problem` | `from ._scheduler_clock import uses_update_clock` | hoist |
   | L1329 | `_update_budget_problem` | `from .nn.enum.schedulers import Schedulers` | hoist |
   | L1330 | `_update_budget_problem` | `from .nn.nn_model import default_train_step` | hoist |
   | L1362 | `_schedule_horizon_problem` | `from ._scheduler_clock import planned_updates, uses_update_clock` | hoist |
   | L1363 | `_schedule_horizon_problem` | `from .nn.enum.schedulers import _horizon_problem` | hoist (via the module object: see step 2) |
   | L1364 | `_schedule_horizon_problem` | `from .nn.nn_model import default_train_step` | hoist |
   | L1401 | `_values` | `from .nn.callbacks import _edp_metric_iter` | hoist |

2. Add to the module-top imports, merged with the existing lines (`from .tasks import ...` gains `task_adapter`):

   ```python
   from ._scheduler_clock import planned_updates, uses_update_clock
   from .components import StatefulComponent
   from .distributed import _refuse_multi_rank
   from .link_tasks import FACTORY_ID, _link_eval_problems
   from .models import BatchAdapter, MissingModelFactoryError, ModelSpec, resolve_model_factory
   from .monitors import MonitorSpec
   from .nn.callbacks import Callback, _edp_metric_iter
   from .nn.enum import schedulers as _schedulers  # called through the module: tests replace _horizon_problem there
   from .nn.enum.schedulers import Schedulers
   from .nn.nn_model import (
       NNModel,
       _check_metric_inputs,
       _check_provenance,
       _check_resume_horizon,
       _check_step_precision,
       _inference_precision,
       _is_streaming_eval_step,
       _metric_context_of,
       _monitor_problem,
       _resolve_net_descriptor,
       _resume_checkpoint_type,
       _step_loss_terms,
       default_train_step,
   )
   from .objectives import _check_update_owner
   from .optimizers import NNOptimFactoryParams, resolve_optimizer_factory
   from .seeding import _seed_scope, _set_seed
   from .streaming import _streaming_problems
   from .trainer.params import _FrozenMapping
   ```

   In `_schedule_horizon_problem`, change `return _horizon_problem(scheduler.kind, ...)` to `return _schedulers._horizon_problem(scheduler.kind, ...)` (same arguments).
3. Delete the 36 local import lines and any blank line a deletion leaves at the start of a function body. Delete `from .nn.nn_model import NNModel` from the `TYPE_CHECKING` block (it is now a runtime import); keep `from .nn.params.nn_run import NNRun` there.
4. Run `uv run ruff check --fix src/nnx/plans.py` (sorts and merges the import block) and `uv run ruff format src/nnx/plans.py`.

No behaviour changes, no test changes, no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] `plans.py` imports nothing inside a function. Check: `uv run python -c "import ast; p='src/nnx/plans.py'; print([n.lineno for f in ast.walk(ast.parse(open(p).read())) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(f) if isinstance(n, (ast.Import, ast.ImportFrom))])"` prints `[]` (36 entries today)
- [ ] The horizon seam still works. Check: `uv run pytest tests/test_experiment_plan.py::test_the_build_and_the_plan_share_one_horizon_check`
- [ ] The plan suites and the import tests pass. Check: `uv run pytest tests/test_experiment_plan.py tests/test_plan_execution.py tests/test_imports.py tests/test_public_api_exports.py tests/test_search.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [plans.py L1-L30](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/plans.py#L1-L30) | the module-top imports and the `TYPE_CHECKING` block |
| [plans.py L516-L561](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/plans.py#L516-L561) | the eval-step and model checks' local imports |
| [plans.py L873-L926](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/plans.py#L873-L926) | the same pair imported twice |
| [plans.py L1326-L1375](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/plans.py#L1326-L1375) | `_update_budget_problem`, `_schedule_horizon_problem`: `default_train_step` and the scheduler helpers |
| [test_experiment_plan.py L997-L1006](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_experiment_plan.py#L997-L1006) | the `_horizon_problem` seam |
| #249 | the "local only for an optional extra or a cycle" rule |

## Scope

**In:** the 36 function-local imports of `plans.py`.

**Out:** `search.py`'s local imports of `plans` (they are the cycle-safe direction; not listed by SMELL-61). Any other module's local imports (FEAT-095, FEAT-136, FEAT-137, FEAT-139, FEAT-140).

## Dependencies

**Blocked by:** none
**Blocks:** none
**Related:** #249
