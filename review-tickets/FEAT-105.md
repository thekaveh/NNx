---
id: FEAT-105
title: "[FEAT-105] Share one supervised batch reader between nn_model and paradigm steps"
labels: ["enhancement", "area: training", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-23"]
depends_on: []
related_issues: [564, 561]
status: READY
---

**Category:** Training · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-105: The supervised batch reader of the paradigm steps, `_single_input_batch`, lives in `nn_model.py`, and four modules import it privately from there, while its self-supervised sibling `first_input` lives in `_step_helpers.py`. This ticket moves the reader into `_step_helpers.py` as `supervised_input`, points the four importers at it, and keeps `nn_model._single_input_batch` as a thin wrapper. The two readers stay separate functions; behaviour does not change.

## Context

**What happens today.** `nn_model._single_input_batch` (`nn_model.py` L2222–L2241):

```python
def _single_input_batch(model: Any, batch: Any, *, who: str) -> tuple[torch.Tensor, torch.Tensor]:
    """``(X, Y)`` of a supervised batch with exactly one positional input,
    split by the model's batch adapter (a built-in net's own
    ``unpack_batch``, FEAT-006) and moved to the model's device. Shared by
    the paradigm steps that transform ``X`` (Mixup / CutMix, KD, MoE)."""
    args, kwargs, target = model._split_batch(batch)
```

It is imported privately by `objectives.py` L412 (`KDObjective.__call__`), `paradigms/distillation.py` L59, `paradigms/augmentation.py` L54 and `paradigms/moe.py` L47, and by `tests/test_nn_graph_dataset_full_batch.py` L567. `_step_helpers.first_input` (L42–L71) reads a self-supervised batch's one input without a target; unlike `_single_input_batch`, it does not cut a graph split batch to its seed rows.

**Why it matters.** The supervised reader is step-helper code, not model code: its home forces four paradigm modules to depend on `nn_model` internals for it, beside the helpers they already take from `_step_helpers`.

**History.** #564 (FIX-096) fixed `first_input`'s bare-tensor reading and left an optional step: "Optionally reuse `NNModel._split_batch` for the adapter-less branch so the readers cannot drift." #561 applied `BatchAdapter.output()` in the diffusion and JEPA steps. Basing `first_input` on the supervised reader would also apply the graph seed-row cut to diffusion and JEPA batches, which is a behaviour change; it stays out of this ticket.

**Background for the worker.**
- `model._split_batch(batch)`: `NNModel`'s split of a batch into `(args, kwargs, target)` through the model's batch adapter (a built-in net's own `unpack_batch`, FEAT-006).
- Graph split batch: a PyG batch where only the first `net.seed_count(batch)` rows belong to the split; the rest are neighbours from other splits.
- Why a wrapper, not an alias: `_step_helpers.py` imports `nn_model` at the top (`from .nn.nn_model import TrainStepContext`, L29), so `nn_model` cannot import `_step_helpers` at the top; the wrapper imports it at call time.

**Proposal.**
1. In `src/nnx/_step_helpers.py`, directly after `first_input` (after L71), add `supervised_input`: the body of `nn_model._single_input_batch` (L2222–L2241) unchanged, with the new name and this docstring:

   ```python
   def supervised_input(model: Any, batch: Any, *, who: str) -> tuple[torch.Tensor, torch.Tensor]:
       """``(X, Y)`` of a supervised batch with exactly one positional input,
       split by the model's batch adapter (a built-in net's own
       ``unpack_batch``, FEAT-006) and moved to the model's device; a graph
       split batch read by a row-wise net keeps its seed rows only. Shared by
       the paradigm steps that transform ``X`` (Mixup / CutMix, KD, MoE) and
       ``KDObjective``. :func:`first_input` is the self-supervised reader (no
       target)."""
   ```

   followed by the body from `args, kwargs, target = model._split_batch(batch)` to `return X.to(model.device), Y.to(model.device)`, with its `ValueError` message unchanged.
2. In `src/nnx/nn/nn_model.py`, replace the body of `_single_input_batch` (keep its signature) with:

   ```python
       """The old home of :func:`nnx._step_helpers.supervised_input`, kept importable."""
       from .._step_helpers import supervised_input  # _step_helpers imports nn_model at module level

       return supervised_input(model, batch, who=who)
   ```

3. Point the importers at `_step_helpers`:
   - `src/nnx/objectives.py`, `KDObjective.__call__` (L410–L418): add `supervised_input` to the local `from ._step_helpers import pinned_eval, predicted_labels, require_row_logits, softened_kl`, remove `_single_input_batch` from the local `from .nn.nn_model import (...)`, and change `X, Y = _single_input_batch(model, ctx.batch, who="kd_objective")` to `X, Y = supervised_input(model, ctx.batch, who="kd_objective")`;
   - `src/nnx/paradigms/distillation.py`: remove `_single_input_batch` from `from ..nn.nn_model import (...)` (L52–L60), add `supervised_input` to `from .._step_helpers import (...)` (L38–L50), and rename the calls at L177 and L319;
   - `src/nnx/paradigms/augmentation.py`: the same for its imports (L36–L56) and the call at L71;
   - `src/nnx/paradigms/moe.py`: the same for its imports (L29–L49) and the call at L105.
4. In `tests/test_nn_graph_dataset_full_batch.py`, `test_paradigm_steps_on_a_graph_split_batch_read_only_the_seed_rows` (L561): change `from nnx.nn.nn_model import _single_input_batch` to `from nnx._step_helpers import supervised_input` plus `from nnx.nn.nn_model import _single_input_batch`, change the reader line to `X, Y = supervised_input(mlp, batch, who="kd")`, and add after the existing assertion:

   ```python
       old_X, old_Y = _single_input_batch(mlp, batch, who="kd")  # the old name still reads the same rows
       assert torch.equal(old_X, X) and torch.equal(old_Y, Y)
   ```

5. Run `uv run ruff check --fix` and `uv run ruff format` on the six touched source files. No message or behaviour changes, so there is no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The reader is defined once, in `_step_helpers.py`. Check: `grep -n "def supervised_input" src/nnx/_step_helpers.py` returns 1 line, and `grep -n "seed_count" src/nnx/nn/nn_model.py` returns no line inside `_single_input_batch`
- [ ] No source module imports `_single_input_batch`. Check: `grep -rn "_single_input_batch" src/nnx | grep -v "src/nnx/nn/nn_model.py"` returns 0 lines
- [ ] The graph seed-row test passes with both names. Check: `uv run pytest tests/test_nn_graph_dataset_full_batch.py::test_paradigm_steps_on_a_graph_split_batch_read_only_the_seed_rows`
- [ ] The paradigm and step-helper suites pass unchanged. Check: `uv run pytest tests/test_step_helpers.py tests/test_paradigms_distillation.py tests/test_paradigms_feature_kd.py tests/test_paradigms_augmentation.py tests/test_paradigms_moe.py tests/test_objective_engine.py tests/test_paradigm_step_slots.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [nn_model.py L2222-L2241](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L2222-L2241) | `_single_input_batch`, with the graph seed-row cut |
| [_step_helpers.py L29](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_step_helpers.py#L29), [L42-L71](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_step_helpers.py#L42-L71) | `_step_helpers` imports `nn_model`; `first_input` |
| [objectives.py L409-L418](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/objectives.py#L409-L418) | `KDObjective.__call__`'s private import |
| [distillation.py L52-L60](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/distillation.py#L52-L60), [L177](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/distillation.py#L177), [L319](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/distillation.py#L319) | KD and feature-KD |
| [augmentation.py L47-L56](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/augmentation.py#L47-L56), [L71](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/augmentation.py#L71) | Mixup / CutMix |
| [moe.py L40-L49](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/moe.py#L40-L49), [L105](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/moe.py#L105) | MoE |
| [test_nn_graph_dataset_full_batch.py L561-L578](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_nn_graph_dataset_full_batch.py#L561-L578) | the seed-row test |
| #564 | the optional "one reader" step, left out |
| #561 | `BatchAdapter.output()` in diffusion and JEPA |

## Scope

**In:** `supervised_input`, the wrapper, the four importers, the test update.

**Out:** basing `first_input` on `supervised_input` (it would cut diffusion and JEPA graph batches to their seed rows, a behaviour change with no ticket). Moving `KDObjective` (FEAT-110, which builds on this ticket). The DPO readers (FEAT-132).

## Dependencies

**Blocked by:** none
**Blocks:** FEAT-110
**Related:** #564, #561
