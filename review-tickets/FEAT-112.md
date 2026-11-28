---
id: FEAT-112
title: "[FEAT-112] Share one freeze helper for frozen modules"
labels: ["enhancement", "area: training", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-36"]
depends_on: []
related_issues: [782, 531]
status: READY
---

**Category:** Training · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-112: Four places freeze a module by hand (`requires_grad = False` on every parameter, and three of them `eval()`): the JEPA target builder, the JEPA objective's `_freeze`, born-again training's teacher snapshot and `_step_helpers.freeze_apart`. This ticket adds `_step_helpers.freeze_module(module, *, eval_mode=True)` and uses it in all four. Behaviour does not change. This is the freeze part of SMELL-36; `q_sample` is FEAT-111.

## Context

**What happens today.** `paradigms/jepa.py`, `build_target_encoder` (L58–L61):

```python
    target = copy.deepcopy(source)
    for p in target.parameters():
        p.requires_grad = False
    target.eval()
```

`paradigms/jepa_objective.py` L342–L345:

```python
def _freeze(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False
```

(called at L236 and L338, and imported by `jepa.py`'s step at L413). `paradigms/born_again.py` L192–L195:

```python
        teacher = copy.deepcopy(model)
        teacher.net.eval()
        for p in teacher.net.parameters():
            p.requires_grad = False
```

and the end of `_step_helpers.freeze_apart` (L157–L158), which freezes without changing the mode (its docstring: "Its mode is left to :func:`pinned_eval`, around each forward").

**Why it matters.** Four copies of one rule; the next frozen module (a DPO reference, a second teacher) would add a fifth.

**History.** #782 (FEAT-057) shared the placement check between the JEPA target and the KD/DPO teacher as `placement_in`, keeping each refusal's own wording ("Every refusal message stays word for word the same"); #531 is JEPA's placement rule. This ticket touches only the freeze after those checks, and no message.

**Background for the worker.**
- Frozen module: one whose parameters get no gradient (`requires_grad=False`); a JEPA EMA target, a KD teacher or a DPO reference.
- `eval()` mode: dropout off and BatchNorm on running statistics. Setting it and clearing `requires_grad` are independent, so their order does not matter.
- `nnx.finetune.freeze(module, *patterns)` is the public, pattern-based freeze; the new helper has another name to keep the two apart.

**Proposal.**
1. In `src/nnx/_step_helpers.py`, directly above `freeze_apart` (L137), add:

   ```python
   def freeze_module(module: torch.nn.Module, *, eval_mode: bool = True) -> None:
       """Set ``requires_grad=False`` on every parameter of ``module`` (a JEPA
       target encoder, a KD teacher, a DPO reference) and, with ``eval_mode``,
       put it in eval mode. :func:`freeze_apart` passes ``eval_mode=False``: its
       mode is left to :func:`pinned_eval`, around each forward."""
       if eval_mode:
           module.eval()
       for param in module.parameters():
           param.requires_grad = False
   ```

   and replace the last two lines of `freeze_apart` (L157–L158) with `freeze_module(frozen, eval_mode=False)`.
2. In `src/nnx/paradigms/jepa.py`:
   - add `freeze_module` to the top-level `from .._step_helpers import (...)` (L41);
   - in `build_target_encoder`, replace L59–L61 (the loop and `target.eval()`) with `freeze_module(target)`;
   - in `jepa_train_step_factory`'s step, change the local import (L413) to `from .jepa_objective import _check_masks, _check_target_placement` and the call `_freeze(target_encoder)` (L424) to `freeze_module(target_encoder)`.
3. In `src/nnx/paradigms/jepa_objective.py`, add `freeze_module` to `from .._step_helpers import ...` (L35), change the two calls (L236, L338) to `freeze_module(self.target_encoder)`, and replace the `_freeze` function (L342–L345) with `_freeze = freeze_module  # the old name, kept importable`.
4. In `src/nnx/paradigms/born_again.py`, add `from .._step_helpers import freeze_module` to the first-party imports (above `from .._validation import require_count`, L31), and replace L193–L195 (`teacher.net.eval()` and the loop) with `freeze_module(teacher.net)`.
5. No test patches these functions (`grep -rn "_freeze\b\|build_target_encoder\|freeze_apart" tests | grep "setattr\|patch"` returns nothing). No message or behaviour changes, so there is no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] No hand-written freeze loop remains in the four places. Check: `grep -rn "requires_grad = False" src/nnx/_step_helpers.py src/nnx/paradigms/jepa.py src/nnx/paradigms/jepa_objective.py src/nnx/paradigms/born_again.py` returns 1 line (`freeze_module`)
- [ ] The helper is used four ways. Check: `grep -rn "freeze_module(" src/nnx/_step_helpers.py src/nnx/paradigms` returns 6 call lines plus the definition
- [ ] The JEPA, KD, DPO and born-again suites pass unchanged, with every placement message. Check: `uv run pytest tests/test_paradigms_jepa.py tests/test_jepa_objective.py tests/test_jepa_encoder_rebuild.py tests/test_paradigms_born_again.py tests/test_paradigms_distillation.py tests/test_paradigms_dpo.py tests/test_paradigms_feature_kd.py tests/test_step_helpers.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [jepa.py L49-L62](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/jepa.py#L49-L62) | `build_target_encoder`'s loop |
| [jepa.py L412-L424](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/jepa.py#L412-L424) | the step imports and calls `jepa_objective._freeze` |
| [jepa_objective.py L236](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/jepa_objective.py#L236), [L336-L345](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/jepa_objective.py#L336-L345) | `_freeze` and its calls |
| [born_again.py L186-L195](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/born_again.py#L186-L195) | the teacher snapshot's loop |
| [_step_helpers.py L137-L158](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_step_helpers.py#L137-L158) | `freeze_apart`: freezes, leaves the mode |
| #782 | `placement_in` shared, messages kept distinct |
| #531 | JEPA's placement rule |

## Scope

**In:** `freeze_module` and its four users, the `_freeze` alias.

**Out:** the placement checks and their messages (#782; deliberately distinct). The JEPA component-state checks (different schemas). `nnx.finetune.freeze` / `unfreeze` (FEAT-133). Forward noising (FEAT-111).

## Dependencies

**Blocked by:** none
**Blocks:** none
**Related:** #782, #531
