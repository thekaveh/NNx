---
id: FEAT-118
title: "[FEAT-118] Share task-owned metric registration and the extra_metrics refusal"
labels: ["enhancement", "area: evaluation", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-39"]
depends_on: []
related_issues: [548]
status: READY
---

**Category:** Evaluation · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-118: three task modules each define a placeholder accumulator class (`_Unavailable`) and a `_register_metrics` loop for "a metric only the task's own evaluation computes", and three define the same `_no_extra_metrics` refusal. This ticket moves the mechanism into two `nnx.monitors` helpers, `_register_task_owned_metrics` and `_refuse_extra_metrics`. Each module keeps its own message and error class and passes them in.

## Context

**What happens today.** `ranking.py` (the `link_tasks.py` and `paradigms/offline_distillation.py` copies differ only in the metric names, the message and the error class):

```python
class _Unavailable:
    """The default evaluation path sees rows, not queries: a declared
    ranking metric is computed by :class:`RankingEval` only, and refuses
    anywhere else rather than reporting a per-batch value."""

    def update(self, target: Any, prediction: Any) -> None:
        raise RankingError(
            "ranking.mrr / ranking.recall / ranking.ndcg are query-grouped metrics: pass eval_step_fn=RankingTask.eval_step() "
            "(and objective=RankingTask.objective()) so they are computed per complete query"
        )
```

```python
def _register_metrics() -> None:
    from .monitors import register_metric, registered_metrics

    known = set(registered_metrics())
    for metric in ("mrr", "recall", "ndcg"):
        if (f"ranking.{metric}", 1) not in known:
            register_metric(
                f"ranking.{metric}",
                1,
                lambda config: _Unavailable(),
                input="continuous",
                mode="max",
                check_config=_check_k_config,
            )
```

And `ranking.py` / `link_tasks.py` / `lm_tasks.py` each have:

```python
def _no_extra_metrics(ctx: Any) -> None:
    if getattr(ctx, "extra_metrics", None):
        raise RankingError(
            "extra_metrics (y_true, y_pred callables over rows) do not apply to query-grouped ranking; its "
            "records carry MRR / Recall / NDCG per query — compute anything else in your own eval_step_fn"
        )
```

**Why it matters.** The mechanism (register once at version 1, a placeholder whose `update` refuses, `result()` is `None`; refuse `extra_metrics` first thing in the step) is the same in every module; only the texts differ.

**History.** #548 (FIX-080) made the link-task evaluation refuse misconfiguration "before the first training epoch". The refusal stays the first statement of each objective and evaluation step, so it still fires before any update.

**Background for the worker.**
- `register_metric(id, version, factory, *, input, mode, check_config)` (`src/nnx/monitors.py` L142): registers a named metric for this process. `factory(config)` returns a fresh accumulator with `update(target, prediction)` and `result()`.
- `check_config`: validates a metric spec's options when a run starts (ranking's `_check_k_config` takes one option, `k`; the link and offline `_no_config` take none). These stay in their modules; their messages differ.
- `extra_metrics`: the user's `{name: (y_true, y_pred) -> float}` callables on `NNTrainParams`, read from the step context (`ctx.extra_metrics`). These tasks' records carry their own metrics, so they refuse it.
- **How per-module messages and error classes are passed:** each module passes its error class as `error=` and its message as `message=`. The message strings move unchanged into the call (or a module constant); nothing in `monitors.py` names a task.
- Import order: `ranking`, `link_tasks`, `lm_tasks` and `offline_distillation` all import `objectives` at module top, which imports `tasks`, which imports `monitors` at module top; `monitors` imports none of them. A module-top `from .monitors import ...` is therefore cycle-free.

**Proposal.**
1. In `src/nnx/monitors.py`, directly after `registered_metrics` (after L183), add:

   ```python
   class _TaskOwnedMetric:
       """The accumulator of a metric only its task's own evaluation
       computes: anywhere else ``update`` raises ``error(message)`` rather
       than report another quantity under the metric's name."""

       def __init__(self, error: type[Exception], message: str) -> None:
           self._error = error
           self._message = message

       def update(self, target: Any, prediction: Any) -> None:
           raise self._error(self._message)

       def result(self) -> Optional[float]:
           return None


   def _register_task_owned_metrics(
       specs: Mapping[str, tuple[str, str]],
       *,
       error: type[Exception],
       message: str,
       check_config: Callable[[Mapping[str, Any]], None],
   ) -> None:
       """Register each ``metric id -> (input, mode)`` of ``specs`` at version
       1, unless that ``(id, 1)`` is already registered, as a
       :class:`_TaskOwnedMetric` that raises ``error(message)``;
       ``check_config`` validates a spec's options as the run starts."""
       known = set(registered_metrics())
       for metric_id, (kind, mode) in specs.items():
           if (metric_id, 1) not in known:
               register_metric(
                   metric_id,
                   1,
                   lambda config: _TaskOwnedMetric(error, message),
                   input=kind,
                   mode=mode,
                   check_config=check_config,
               )


   def _refuse_extra_metrics(ctx: Any, *, error: type[Exception], message: str) -> None:
       """Raise ``error(message)`` when ``ctx`` carries ``extra_metrics``: a
       task whose records carry its own metrics refuses the per-row
       callables, first thing in its objective and evaluation step."""
       if getattr(ctx, "extra_metrics", None):
           raise error(message)
   ```

   Leave `__all__` unchanged (the helpers are private).

2. `src/nnx/ranking.py`:
   - Add `from .monitors import _refuse_extra_metrics, _register_task_owned_metrics` to the relative imports.
   - Replace `_no_extra_metrics` (L577-L582) with this module constant:

     ```python
     _EXTRA_METRICS_REFUSAL = (
         "extra_metrics (y_true, y_pred callables over rows) do not apply to query-grouped ranking; its "
         "records carry MRR / Recall / NDCG per query — compute anything else in your own eval_step_fn"
     )
     ```

   - Replace each `_no_extra_metrics(ctx)` call (L614 in `RankingObjective.__call__`, L676 in `RankingEval.__call__`) with `_refuse_extra_metrics(ctx, error=RankingError, message=_EXTRA_METRICS_REFUSAL)`, on the same line, still the first statement.
   - Delete `class _Unavailable` (L717-L729). Keep `_check_k_config` unchanged.
   - Replace the body of `_register_metrics` (L739-L751) so the function reads:

     ```python
     def _register_metrics() -> None:
         # The default evaluation path sees rows, not queries: a declared ranking metric is computed by
         # RankingEval only, and refuses anywhere else rather than reporting a per-batch value.
         _register_task_owned_metrics(
             {f"ranking.{metric}": ("continuous", "max") for metric in ("mrr", "recall", "ndcg")},
             error=RankingError,
             message=(
                 "ranking.mrr / ranking.recall / ranking.ndcg are query-grouped metrics: pass "
                 "eval_step_fn=RankingTask.eval_step() (and objective=RankingTask.objective()) so they are "
                 "computed per complete query"
             ),
             check_config=_check_k_config,
         )
     ```

     The concatenated message is the same string as today.

3. `src/nnx/link_tasks.py`:
   - Add `from .monitors import _refuse_extra_metrics, _register_task_owned_metrics`.
   - Replace `_no_extra_metrics` (L1199-L1201) with `_EXTRA_METRICS_REFUSAL = "extra_metrics do not apply to link tasks; their records carry the link metrics"`, and each call (L1222 in `LinkObjective.__call__`, L1330 in `LinkEval.__call__`) with `_refuse_extra_metrics(ctx, error=LinkTaskError, message=_EXTRA_METRICS_REFUSAL)`.
   - Delete `class _Unavailable` (L1474-L1479). Keep `_no_config`.
   - Replace the body of `_register_metrics` (L1488-L1495) with:

     ```python
         _register_task_owned_metrics(
             {metric: ("probabilities", "max") for metric in ("link.auroc", "link.ap")},
             error=LinkTaskError,
             message="link.auroc / link.ap are computed by LinkTask.eval_step() over the whole candidate set",
             check_config=_no_config,
         )
     ```

     `_register()` keeps calling `_register_metrics()`.

4. `src/nnx/lm_tasks.py`:
   - Add `from .monitors import _refuse_extra_metrics`.
   - Replace `_no_extra_metrics` (L527-L532) with:

     ```python
     _EXTRA_METRICS_REFUSAL = (
         "extra_metrics (y_true, y_pred callables) do not apply to the causal-LM task; its records carry "
         "nll, perplexity and token_accuracy — compute anything else in your own eval_step_fn"
     )
     ```

   - Replace each call (L556 in `CausalLMObjective.__call__`, L612 in `CausalLMEval.__call__`) with `_refuse_extra_metrics(ctx, error=LMTaskError, message=_EXTRA_METRICS_REFUSAL)`.

5. `src/nnx/paradigms/offline_distillation.py`:
   - Add `from ..monitors import _register_task_owned_metrics` to the relative imports.
   - Delete `class _Unavailable` (L816-L830). Keep `_METRICS` and `_no_config`.
   - Replace the body of `_register_metrics` (L839-L846) with:

     ```python
         # The default paths see batches, not this evaluation's denominators: a declared offline-distillation
         # metric is computed by OfflineDistillationEval only, and refuses anywhere else.
         _register_task_owned_metrics(
             {f"offline.{name}": spec for name, spec in _METRICS.items()},
             error=TeacherRecordError,
             message=(
                 "offline.teacher_kl / offline.teacher_agreement / offline.label_accuracy / offline.label_nll are "
                 "computed by the offline evaluation: pass eval_step_fn=TeacherDataset.eval_step() (and "
                 "objective=TeacherDataset.objective()) over loaders of teacher batches"
             ),
             check_config=_no_config,
         )
     ```

6. No aliases: `_Unavailable` and `_no_extra_metrics` are referenced nowhere outside their own modules (`git grep -n "_Unavailable\|_no_extra_metrics" src tests` shows only the definitions and call sites removed here), no test monkeypatches them, and the registry's factories are lambdas, never pickled.
7. Run `uv run ruff check --fix src/nnx/` and `uv run ruff format src/nnx/`.

Registration order, metric ids, versions, inputs, modes, `check_config` functions and every message are unchanged, so no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] `nnx.monitors` defines `_TaskOwnedMetric`, `_register_task_owned_metrics` and `_refuse_extra_metrics`; `__all__` is unchanged. Check: `grep -n "^def _register_task_owned_metrics\|^def _refuse_extra_metrics\|^class _TaskOwnedMetric" src/nnx/monitors.py` returns 3 lines
- [ ] The nine per-module helpers are gone. Check: `git grep -n "class _Unavailable\|def _no_extra_metrics" src/` returns 0 lines
- [ ] A new test `tests/test_named_monitors.py::test_task_owned_metrics_refuse_outside_their_evaluation` registers `{"demo.owned": ("labels", "max")}` with `error=KeyError`, `message="owned"` and a no-op `check_config`, asserts the factory's accumulator raises `KeyError` on `update(np.zeros(1), np.zeros(1))` and returns `None` from `result()`, asserts a second call registers nothing new, and calls `unregister_metric("demo.owned", 1)` in a `finally`. It imports `numpy as np` and `from nnx.monitors import _REGISTRY, _register_task_owned_metrics, unregister_metric` inside its body, and reads the factory as `_REGISTRY[("demo.owned", 1)].factory({})`. Check: `uv run pytest tests/test_named_monitors.py::test_task_owned_metrics_refuse_outside_their_evaluation`
- [ ] The existing message and refusal tests pass unchanged: `tests/test_ranking_task.py::test_ranking_metrics_need_the_ranking_eval_step` ("query-grouped metrics", "exactly one option, k"), `tests/test_ranking_task.py::test_an_empty_validation_loader_and_extra_metrics_are_refused`, `tests/test_causal_lm_task.py::test_an_empty_validation_loader_and_extra_metrics_are_refused` and `tests/test_offline_distillation.py::test_the_metric_specs_declare_every_evaluation_metric` ("take no options"). Check: `uv run pytest tests/test_ranking_task.py tests/test_link_tasks.py tests/test_causal_lm_task.py tests/test_offline_distillation.py tests/test_named_monitors.py tests/test_metric_specs.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [ranking.py L577-L582](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/ranking.py#L577-L582) | `_no_extra_metrics` (ranking copy) |
| [ranking.py L717-L754](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/ranking.py#L717-L754) | `_Unavailable`, `_check_k_config`, `_register_metrics` |
| [link_tasks.py L1199-L1201](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/link_tasks.py#L1199-L1201) | `_no_extra_metrics` (link copy) |
| [link_tasks.py L1474-L1506](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/link_tasks.py#L1474-L1506) | `_Unavailable`, `_no_config`, `_register_metrics`, `_register` |
| [lm_tasks.py L527-L532](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/lm_tasks.py#L527-L532) | `_no_extra_metrics` (causal-LM copy) |
| [offline_distillation.py L807-L849](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/offline_distillation.py#L807-L849) | `_METRICS`, `_Unavailable`, `_no_config`, `_register_metrics` |
| [monitors.py L142-L183](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/monitors.py#L142-L183) | `register_metric`, `registered_metrics` |
| [test_ranking_task.py L469-L484](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_ranking_task.py#L469-L484) | pins "query-grouped metrics" and "exactly one option, k" |
| [test_offline_distillation.py L673](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_offline_distillation.py#L673) | pins "take no options" |
| #548 | link-task validation refuses before the first training epoch |

## Scope

**In:** the two monitors helpers and the four modules' registration and refusal code.

**Out:** sharing the `check_config` functions (their messages differ). The other function-local imports of `offline_distillation.py` (FEAT-137).

## Dependencies

**Blocked by:** none
**Blocks:** FEAT-137
**Related:** #548
