---
id: FEAT-131
title: "[FEAT-131] Share the post-mix step of Mixup and CutMix"
labels: ["enhancement", "area: training", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-55"]
depends_on: ["FIX-120"]
related_issues: [513, 542]
status: READY
---

**Category:** Training · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-131: once a batch is mixed, `mixup_train_step_factory` and `cutmix_train_step_factory` run the same 19 lines: the forward pass, the row-logits check, the declared-task branch, the λ-mixed loss, the scorable-target check, the update, the λ-weighted accuracy and the record. This ticket moves them into one helper, `_mixed_supervised_step`, beside `_mixed_task_step`. Each factory keeps its own mixing and its own input refusal.

## Context

**What happens today.** The tail of the Mixup step (`src/nnx/paradigms/augmentation.py` L165-L183); the CutMix step (L257-L275) is the same with `X_cut` and `"cutmix"`:

```python
        Y_hat_logits = m._net_forward((X_mixed,), {})
        require_row_logits(Y_hat_logits, "mixup")
        if m.task_adapter is not None:
            return _mixed_task_step(ctx, m, Y_hat_logits, Y_a, Y_b, lam, paradigm="mixup")
        loss_input = _loss_input(m.loss_fn, Y_hat_logits)
        loss = lam * m.loss_fn(loss_input, Y_a) + (1.0 - lam) * m.loss_fn(loss_input, Y_b)
        _check_scorable_target(m.loss_fn, Y, paradigm="mixup")  # before the update
        loss_val = finalize_step(loss, ctx, paradigm="mixup")
```

FIX-120 removes the fabricated `f1=0.0`, `recall=0.0`, `precision=0.0` from both records, so after it both end with `NNEvaluationDataPoint(accuracy=acc, loss=loss_val, error=float(1.0 - acc))`.

**Why it matters.** Two copies of the update path: a fix to one (as #513 and FIX-120 each needed) has to be made twice.

**History.** #513 (FIX-046) added the declared-task branch (`_mixed_task_step`) to both; #542 (FIX-074) added Mixup's refusal of non-floating input (token ids), which stays outside the shared helper: it must run before any random draw, and CutMix has its own rank refusal there.

**Background for the worker.**
- **Purpose of `paradigm=`:** it names the step (`"mixup"` or `"cutmix"`) in every refusal and warning the shared path can raise: `require_row_logits` ("the mixup step scores (rows, classes) logits …"), `_check_scorable_target` ("the mixup step scores classification only …"), `finalize_step` / `finalize_task_step` (unsupported mixed precision or gradient accumulation, non-finite loss). It changes no computation.
- `Y_a` is the batch's targets (`Y`), `Y_b` the permuted ones (`Y[perm]`), `lam` the mixing weight; CutMix's `lam` is the realized area ratio.
- `_check_scorable_target(m.loss_fn, Y, ...)` is called with `Y` today; `Y` is `Y_a` in both steps, so the helper passes `Y_a`.

**Proposal.**
1. In `src/nnx/paradigms/augmentation.py`, after `_weighted_acc` (L98-L112), add:

   ```python
   def _mixed_supervised_step(
       ctx: TrainStepContext,
       m: NNModel,
       X_in: torch.Tensor,
       Y_a: torch.Tensor,
       Y_b: torch.Tensor,
       lam: float,
       *,
       paradigm: str,
   ) -> NNEvaluationDataPoint:
       """The shared tail of a Mixup / CutMix step once the batch is mixed:
       the forward pass on ``X_in``, the ``lam``-mixed loss over ``Y_a`` and
       ``Y_b`` (through the declared task when there is one,
       :func:`_mixed_task_step`), the update and the record. ``paradigm``
       (``"mixup"`` / ``"cutmix"``) names the step in every refusal and
       warning."""
       Y_hat_logits = m._net_forward((X_in,), {})
       require_row_logits(Y_hat_logits, paradigm)
       if m.task_adapter is not None:
           return _mixed_task_step(ctx, m, Y_hat_logits, Y_a, Y_b, lam, paradigm=paradigm)
       loss_input = _loss_input(m.loss_fn, Y_hat_logits)
       loss = lam * m.loss_fn(loss_input, Y_a) + (1.0 - lam) * m.loss_fn(loss_input, Y_b)
       _check_scorable_target(m.loss_fn, Y_a, paradigm=paradigm)  # before the update
       loss_val = finalize_step(loss, ctx, paradigm=paradigm)

       Y_hat = predicted_labels(m.loss_fn, Y_hat_logits)
       acc = _weighted_acc(m.loss_fn, Y_hat, Y_a, Y_b, lam)
       return NNEvaluationDataPoint(accuracy=acc, loss=loss_val, error=float(1.0 - acc))
   ```

   The return statement is the one FIX-120 leaves in both steps; copy it from the current file so the two stay identical.
2. In `mixup_train_step_factory`'s `step`, replace everything from `Y_a = Y` (L162) to the end of the step's `return NNEvaluationDataPoint(...)` (L183) with:

   ```python
           return _mixed_supervised_step(ctx, m, X_mixed, Y, Y[perm], lam, paradigm="mixup")
   ```

   Keep the token-id refusal, `lam`, `perm` and `X_mixed` above it unchanged.
3. In `cutmix_train_step_factory`'s `step`, replace everything from `Y_a = Y` (L254) to the end of the step's return (L275) with:

   ```python
           return _mixed_supervised_step(ctx, m, X_cut, Y, Y[perm], lam, paradigm="cutmix")
   ```

   Keep the rank refusal, the box, `X_cut` and the realized `lam` above it unchanged.
4. Run `uv run ruff format src/nnx/paradigms/augmentation.py`.

The order of operations (forward, row check, task branch, loss, scorable check, update, accuracy, record) and every message are unchanged, so no test changes and no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] Both factories call the helper once. Check: `grep -n "_mixed_supervised_step(" src/nnx/paradigms/augmentation.py` returns 3 lines (the definition and two calls)
- [ ] The duplicated tail is gone. Check: `grep -c "m._net_forward(" src/nnx/paradigms/augmentation.py` returns 1
- [ ] Mixup's token-id refusal still runs before any draw, outside the helper. Check: `uv run pytest tests/test_paradigms_augmentation.py -k "token or integer or floating"`
- [ ] The augmentation suite passes, including FIX-120's `accuracy` / `f1 is None` assertions and the declared-task tests. Check: `uv run pytest tests/test_paradigms_augmentation.py tests/test_task_adapters.py -k "mixup or cutmix or augmentation"`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [augmentation.py L74-L112](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/augmentation.py#L74-L112) | `_mixed_task_step` and `_weighted_acc`, the neighbours of the new helper |
| [augmentation.py L150-L183](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/augmentation.py#L150-L183) | the Mixup step: its refusal, the mix, the shared tail |
| [augmentation.py L223-L275](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/augmentation.py#L223-L275) | the CutMix step: its refusal, the box, the same tail |
| #513 | the declared-task branch in both steps |
| #542 | Mixup's refusal of token ids, kept outside the helper |

## Scope

**In:** `_mixed_supervised_step` and the two call sites.

**Out:** the record's classification fields (FIX-120). The mixing itself and each step's input refusal.

## Dependencies

**Blocked by:** FIX-120 (edits both records)
**Blocks:** none
**Related:** #513, #542
