---
id: FEAT-110
title: "[FEAT-110] Share the knowledge-distillation loss and move KDObjective beside it"
labels: ["enhancement", "area: training", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-35"]
depends_on: ["FIX-118", "FEAT-105"]
related_issues: [214, 513, 542, 544]
status: READY
---

**Category:** Training · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-110: `kd_train_step_factory` and `feature_kd_train_step_factory` end with the same 30-line tail (soft/hard mixing, the alpha 0 and 1 special cases, the task branch, the update and the record), and `KDObjective` lives in `objectives.py` with seven function-local imports from the paradigm and model modules, while every other paradigm objective lives with its paradigm. This ticket extracts the tail into `distillation._kd_loss` and moves `KDObjective` and `kd_objective` into a new `paradigms/kd_objective.py`, still importable from `nnx.objectives`. `_check_alpha`'s `TypeError` stays as it is. Behaviour does not change.

## Context

**What happens today.** The tail of `kd_train_step_factory`'s step (`distillation.py` L191–L219), from the soft term to the record:

```python
        soft_loss = softened_kl(student_logits, teacher_logits, temperature) if alpha > 0 else 0.0
        adapter = m.task_adapter
        if adapter is not None:
            # A declared task scores the labels over its valid targets (#513), validated before any backward.
            hard = task_term(adapter, m.loss_fn, student_logits, Y)
            loss = alpha * soft_loss + (1.0 - alpha) * hard.loss if alpha < 1 else soft_loss
            assert isinstance(loss, torch.Tensor)
            trains = alpha > 0 or hard.trains  # every target masked at alpha=0: nothing to learn from
            loss_val = finalize_task_step(loss, ctx, trains=trains, paradigm="distillation")
            return task_record(adapter, [(1.0, hard)], loss=loss_val, extra_metrics=ctx.extra_metrics)
```

`feature_kd_train_step_factory` (L381–L404) repeats it with `+ beta * feature_loss` in both losses, `trains = alpha > 0 or beta > 0 or hard.trains`, `paradigm="feature_distillation"` and `_check_scorable_target(..., paradigm="feature distillation")`. `KDObjective.__init__`, `check_run` and `__call__` (`objectives.py` L351–L464) each import from `nnx.paradigms.distillation`, `nnx._step_helpers` or `nnx.nn.nn_model` inside the function, because `objectives` is imported by `nn_model`'s callers early and must not import `nn_model` at module load.

**Why it matters.** The tail has needed the same fix in both factories before (#513's task branch, BUG-15's alpha 1 rule); a third change would again have to be made twice. `KDObjective`'s placement forces lazy imports that `diffusion/objective.py` and `paradigms/jepa_objective.py` do not need.

**History.**
- #214 introduced the objective engine and `kd_objective`; #513 added the task branch to the KD steps and objective.
- #542 and #544 decided the validators' error types: `offline_distillation._check_alpha` raises `TypeError` for `alpha=True` on purpose, so it is **not** changed here.
- FIX-118 (BUG-15) lands first: it adds the `alpha == 1` `extra_metrics` refusal at the top of both steps and of `KDObjective.__call__`. Those lines move with the code; the tail this ticket extracts is unchanged by it.
- FEAT-105 lands first: the steps and `KDObjective.__call__` then read their batch with `_step_helpers.supervised_input`.

**Background for the worker.**
- Soft term: `softened_kl(student, teacher, T)`, the temperature-softened `KL(teacher ‖ student)` times `T²`. Hard term: the student's loss on the labels, through the task adapter (`task_term`) when the model declares a `TaskSpec`.
- `alpha` 0 and 1: at `alpha=0` there is no soft term (`soft_loss = 0.0`); at `alpha=1` no hard term (the labels are placeholders), and without a task the record carries the loss only.
- Arithmetic: the shared formula adds `extra_loss` (0.0 for plain KD) and, at `alpha=1`, multiplies the soft term by `1.0`; `x * 1.0` and `x + 0.0` equal `x` for every float except that `-0.0 + 0.0` is `+0.0`, so losses, gradients and updates are the same bits.
- Module `__getattr__` (PEP 562): a module-level `def __getattr__(name)` is called for names the module does not define; `from nnx.objectives import KDObjective` and pickles that name `nnx.objectives.KDObjective` then still work.
- `paradigms/kd_objective.py` may import `nnx.nn.nn_model`, `nnx._step_helpers` and `nnx.objectives` at the top: `nnx.paradigms` is imported after all of them.

**Proposal.**
1. In `src/nnx/paradigms/distillation.py`, add `from typing import Union` to the imports and, after `_freeze_teacher` (and after FIX-118's `_refuse_placeholder_metrics`), add:

   ```python
   def _kd_loss(
       ctx: TrainStepContext,
       m: NNModel,
       Y: torch.Tensor,
       student_logits: torch.Tensor,
       teacher_logits: torch.Tensor,
       *,
       alpha: float,
       temperature: float,
       extra_loss: Union[float, torch.Tensor] = 0.0,
       extra_trains: bool = False,
       paradigm: str = "distillation",
   ) -> NNEvaluationDataPoint:
       """The shared tail of the KD step factories: the softened KL (weight
       ``alpha``; none at ``alpha=0``), ``extra_loss`` (feature KD's
       ``beta``-weighted MSE) and the hard term (weight ``1 - alpha``; none at
       ``alpha=1``), scored through the task adapter when the model declares a
       task; then the update and the record. ``extra_trains``: ``extra_loss``
       gives something to learn from even when every target is masked (feature
       KD with ``beta > 0``). ``paradigm`` names the step in its refusals."""
       # KL(teacher || student) with both softened by T — the standard
       # Hinton direction (see softened_kl for the F.kl_div contract).
       # alpha=0 is plain supervised training: no soft term (a binary head has none).
       soft_loss = softened_kl(student_logits, teacher_logits, temperature) if alpha > 0 else 0.0
       adapter = m.task_adapter
       if adapter is not None:
           # A declared task scores the labels over its valid targets (#513), validated before any backward.
           hard = task_term(adapter, m.loss_fn, student_logits, Y)
           loss = alpha * soft_loss + extra_loss + ((1.0 - alpha) * hard.loss if alpha < 1 else 0.0)
           assert isinstance(loss, torch.Tensor)
           trains = alpha > 0 or extra_trains or hard.trains  # every target masked: nothing to learn from
           loss_val = finalize_task_step(loss, ctx, trains=trains, paradigm=paradigm)
           return task_record(adapter, [(1.0, hard)], loss=loss_val, extra_metrics=ctx.extra_metrics)
       # alpha=1 is pure distillation: no hard term (an all-ignored batch would make it NaN).
       hard_loss = m.loss_fn(_loss_input(m.loss_fn, student_logits), Y) if alpha < 1 else 0.0
       loss = alpha * soft_loss + extra_loss + (1.0 - alpha) * hard_loss
       assert isinstance(loss, torch.Tensor)  # 0 <= alpha <= 1: at least one term is a tensor
       # Before the update: a refused batch changes nothing.
       _check_scorable_target(m.loss_fn, Y, paradigm=paradigm.replace("_", " "))
       loss_val = finalize_step(loss, ctx, paradigm=paradigm)
       if alpha == 1:  # the labels are placeholders the loss never reads: no classification fields to score
           return NNEvaluationDataPoint(loss=loss_val)
       return _classification_edp_for_loss(
           loss_fn=m.loss_fn,
           target=Y,
           prediction=predicted_labels(m.loss_fn, student_logits),
           loss=loss_val,
           extra_metrics=ctx.extra_metrics,
       )
   ```

2. In `kd_train_step_factory`'s step, replace everything after the teacher forward (from the `# KL(teacher || student) ...` comment, L188, to the closing `)` of `_classification_edp_for_loss`, L218) with `return _kd_loss(ctx, m, Y, student_logits, teacher_logits, alpha=alpha, temperature=temperature)`.
3. In `feature_kd_train_step_factory`'s step, replace everything after `feature_loss = feature_loss / len(auxiliary_layers)` (from L380 to the closing `)` at L404) with:

   ```python
           return _kd_loss(
               ctx,
               m,
               Y,
               student_logits,
               teacher_logits,
               alpha=alpha,
               temperature=temperature,
               extra_loss=beta * feature_loss,
               extra_trains=beta > 0,
               paradigm="feature_distillation",
           )
   ```

4. Create `src/nnx/paradigms/kd_objective.py`:

   ```python
   """Hinton knowledge distillation as an objective (:class:`KDObjective`),
   beside the step factories whose checks it shares
   (:mod:`nnx.paradigms.distillation`). ``nnx.objectives`` re-exports both
   names."""

   from __future__ import annotations

   from collections.abc import Mapping, Sequence
   from typing import Any

   import torch

   from .._step_helpers import pinned_eval, predicted_labels, require_row_logits, softened_kl, supervised_input
   from ..nn.nn_model import _KD_OBJECTIVE_SCORING, NNModel, _check_scorable_target, _classification_edp_for_loss
   from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
   from ..objectives import (
       LossTerm,
       Objective,
       ObjectiveContext,
       ObjectiveResult,
       _loss_term,
       _own_precision,
       _supervised_terms,
   )
   from ..tasks import _StepTasks
   from .distillation import _check_kd_weights, _check_live_teacher, _freeze_teacher, _refuse_placeholder_metrics

   __all__ = ["KDObjective", "kd_objective"]
   ```

   then cut `class KDObjective` (`objectives.py` L351–L464, as FIX-118 and FEAT-105 left it) and `def kd_objective` (L484–L490) from `objectives.py` and paste them below, unchanged except that every function-local import inside `KDObjective` is deleted (the names are imported above). Docstrings and messages stay byte for byte. `_own_precision` stays in `objectives.py`.
5. In `src/nnx/objectives.py`:
   - add `import importlib` to the standard-library imports;
   - in the `if TYPE_CHECKING:` block (L63), add `from .paradigms.kd_objective import KDObjective, kd_objective` (for the type checker; `__all__` lists both, so ruff counts them as used);
   - keep `"KDObjective"` and `"kd_objective"` in `__all__` (L68–L78);
   - at the end of the module add:

     ```python
     def __getattr__(name: str) -> Any:
         """``KDObjective`` and ``kd_objective``, which live in
         :mod:`nnx.paradigms.kd_objective`: imported at first use, because that
         module imports ``nnx.nn.nn_model``, which this module must not import
         at load time."""
         if name in ("KDObjective", "kd_objective"):
             return getattr(importlib.import_module(".paradigms.kd_objective", __package__), name)
         raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
     ```

   - run `uv run ruff check --fix src/nnx/objectives.py` to drop imports that became unused.
6. Regenerate the API reference (`nnx.objectives.KDObjective` and `kd_objective` are catalogued, L135 and L139 of `scripts/docs/api_reference_catalog.txt`): `uv run python -m scripts.docs.build_api_reference`, and commit whatever it changes in `docs/api.md`.
7. Add to `tests/test_paradigms_distillation.py`:

   ```python
   def test_kd_objective_moved_but_keeps_its_import_paths():
       import pickle

       import nnx
       import nnx.objectives as objectives
       from nnx.paradigms import kd_objective as module

       assert objectives.KDObjective is module.KDObjective is nnx.KDObjective
       assert objectives.kd_objective is module.kd_objective is nnx.kd_objective
       assert pickle.loads(b"\x80\x02cnnx.objectives\nKDObjective\n.") is module.KDObjective  # an older pickle's class path
       with pytest.raises(AttributeError):
           objectives.no_such_name  # noqa: B018
   ```

8. No message changes and no behaviour changes, so there is no CHANGELOG entry. No test patches the moved names (`grep -rn "objectives.KDObjective\|objectives, \"KDObjective" tests` returns nothing).

**Worker setup.** Branch from `develop` after FIX-118 and FEAT-105 have merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The tail exists once. Check: `grep -c "loss_val = finalize_step(loss, ctx, paradigm=" src/nnx/paradigms/distillation.py` prints `1`, and `grep -c "_kd_loss(" src/nnx/paradigms/distillation.py` prints `3`
- [ ] `KDObjective` lives with its paradigm and has no function-local imports. Check: `grep -n "^class KDObjective" src/nnx/paradigms/kd_objective.py` returns 1 line, `grep -rn "^class KDObjective" src/nnx/objectives.py` returns 0 lines, and `awk '/^class KDObjective/,/^def kd_objective/' src/nnx/paradigms/kd_objective.py | grep -c "^\s\+from "` prints `0`
- [ ] The import paths and pickles still resolve. Check: `uv run python -c "import nnx.objectives"` and `uv run pytest tests/test_paradigms_distillation.py::test_kd_objective_moved_but_keeps_its_import_paths`
- [ ] `_check_alpha` is untouched. Check: `git diff origin/develop -- src/nnx/paradigms/offline_distillation.py` is empty
- [ ] The KD, objective and import suites pass unchanged, including the bit-identity tests. Check: `uv run pytest tests/test_paradigms_distillation.py tests/test_paradigms_feature_kd.py tests/test_offline_distillation.py tests/test_paradigms_born_again.py tests/test_objective_engine.py tests/test_task_adapters.py tests/test_paradigm_finite_settings.py tests/test_params_boundary_validation.py tests/test_imports.py tests/test_public_api_exports.py`
- [ ] The docs are in sync. Check: `uv run python -m scripts.docs.build_api_reference --check` and `uv run python -m scripts.docs.build_docs --check`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [distillation.py L170-L221](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/distillation.py#L170-L221) | `kd_train_step_factory`'s step and tail |
| [distillation.py L313-L409](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/distillation.py#L313-L409) | `feature_kd_train_step_factory`'s step and the same tail with `beta` |
| [objectives.py L59-L78](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/objectives.py#L59-L78) | imports, `TYPE_CHECKING` block, `__all__` |
| [objectives.py L351-L464](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/objectives.py#L351-L464) | `KDObjective` and its local imports (L386, L402, L409–L416) |
| [objectives.py L467-L490](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/objectives.py#L467-L490) | `_own_precision` (stays) and `kd_objective` |
| [offline_distillation.py L483-L493](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/offline_distillation.py#L483-L493) | `_check_alpha`: its `TypeError` is deliberate (#542, #544) |
| [api_reference_catalog.txt L135-L139](https://github.com/thekaveh/NNx/blob/9ad461e/scripts/docs/api_reference_catalog.txt#L135-L139) | the catalogued names |
| #214, #513, #542, #544 | the objective engine; the task branch; the validator error types |

## Scope

**In:** `_kd_loss` and its two callers, `paradigms/kd_objective.py`, the `nnx.objectives` re-export, one test.

**Out:** `_check_alpha` (its `TypeError` is a recorded decision, #542 and #544, so it is not changed). The `extra_metrics` refusal at `alpha=1` (FIX-118). The supervised batch reader (FEAT-105). Sharing `KDObjective`'s term construction with the steps (an objective returns weighted terms, not one loss). The decisions import cycle (FEAT-060).

## Dependencies

**Blocked by:** FIX-118, FEAT-105
**Blocks:** none
**Related:** #214, #513, #542, #544
