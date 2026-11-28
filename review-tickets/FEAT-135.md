---
id: FEAT-135
title: "[FEAT-135] Use full_precision for half-precision widening"
labels: ["enhancement", "area: numerics", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-60"]
depends_on: []
related_issues: [516, 577, 766, 225]
status: READY
---

**Category:** Numerics · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-135: three places inline the "float16 / bfloat16 → float32, anything else unchanged" rule that `_step_helpers.full_precision` already names: the causal-LM token NLL, the ranking pairwise loss and DoRA's weight norm. This ticket moves `full_precision` into a new torch-only module, `src/nnx/_dtypes.py`, beside a `promoted_dtype` twin for DoRA, re-exports `full_precision` from `_step_helpers`, and uses them at the three sites. Results are bit-identical.

## Context

**What happens today.**

```python
    x = flat.float() if flat.dtype in (torch.float16, torch.bfloat16) else flat
```

(`_token_nll`, `src/nnx/lm_tasks.py` L94)

```python
    if scores.dtype in (torch.float16, torch.bfloat16):
        scores = scores.float()  # differentiable; a reduced-precision running sum stalls over many queries
```

(`pairwise_logistic_loss`, `src/nnx/ranking.py` L251-L252)

```python
        acc_dtype = torch.float32 if V.dtype in (torch.float16, torch.bfloat16) else V.dtype
```

(`DoRALinear.forward`, `src/nnx/peft/dora.py` L277), and the helper:

```python
def full_precision(tensor: torch.Tensor) -> torch.Tensor:
    """A mixed-precision (float16 / bfloat16) output as float32, so a loss
    sum accumulates in full precision; other dtypes unchanged."""
    return tensor.float() if tensor.dtype in (torch.float16, torch.bfloat16) else tensor
```

(`src/nnx/_step_helpers.py` L85-L88)

**Why it matters.** One widening rule, written four times.

**History.** #516 (FIX-049) widened ranking scores under bf16; #577 (FEAT-055) cut per-batch overhead in the causal-LM and ranking losses; #766 (DEC-012) kept the causal-LM loss's custom autograd `Function`, whose `_token_nll` this touches only by name; #225 (FIX-015) made DoRA's zero-row guard work in FP16 by computing the norm in at least float32 while keeping float64. None of their semantics change. The helper moves to a torch-only `_dtypes.py`: `_step_helpers` imports `nn.nn_model`, so `lm_tasks`, `ranking` and `peft.dora` must not import it.

**Background for the worker.**
- `tensor.float()` is `tensor.to(torch.float32)`, so `full_precision(t)` equals each inlined expression for every dtype.
- `promoted_dtype(t)` returns the dtype `full_precision(t)` would have, for code (DoRA) that casts several tensors to it.
- Re-export: `_step_helpers.full_precision` stays importable (`diffusion`, `jepa_objective` and `offline_distillation` import it from there); the redundant alias form `import x as x` marks it as deliberate for ruff.

**Proposal.**
1. Create `src/nnx/_dtypes.py`:

   ```python
   """Floating-point widening rules for losses, metrics and adapters.
   Internal; imports torch only, so any module can use it without a cycle."""

   from __future__ import annotations

   import torch

   HALF_PRECISION: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16)


   def full_precision(tensor: torch.Tensor) -> torch.Tensor:
       """A mixed-precision (float16 / bfloat16) output as float32, so a loss
       sum accumulates in full precision; other dtypes unchanged."""
       return tensor.float() if tensor.dtype in HALF_PRECISION else tensor


   def promoted_dtype(tensor: torch.Tensor) -> torch.dtype:
       """The dtype :func:`full_precision` gives ``tensor``: float32 for
       float16 / bfloat16, else ``tensor.dtype`` (float64 is kept)."""
       return torch.float32 if tensor.dtype in HALF_PRECISION else tensor.dtype
   ```

2. In `src/nnx/_step_helpers.py`, delete the `full_precision` definition (L85-L88) and add `from ._dtypes import full_precision as full_precision` to the module-top imports.
3. `src/nnx/lm_tasks.py`: add `from ._dtypes import full_precision`; in `_token_nll` replace L94 with `x = full_precision(flat)`.
4. `src/nnx/ranking.py`: add `from ._dtypes import full_precision`; in `pairwise_logistic_loss` replace L251-L252 with `scores = full_precision(scores)  # differentiable; a reduced-precision running sum stalls over many queries`.
5. `src/nnx/peft/dora.py`: add `from .._dtypes import promoted_dtype`; in `DoRALinear.forward` replace L277 with `acc_dtype = promoted_dtype(V)`. Keep the comment block above it (L265-L276).
6. Run `uv run ruff check --fix src/nnx/` and `uv run ruff format src/nnx/`.

No numeric, dtype or gradient change, so no test changes and no CHANGELOG entry. No test monkeypatches `full_precision` (`git grep -n "full_precision\"" tests/` returns 0 lines).

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The inlined rule is gone from the three sites. Check: `grep -n "(torch.float16, torch.bfloat16)" src/nnx/lm_tasks.py src/nnx/ranking.py src/nnx/peft/dora.py src/nnx/_step_helpers.py` returns 0 lines
- [ ] `_dtypes.py` imports nothing from `nnx`. Check: `grep -n "^from \.\|^import nnx" src/nnx/_dtypes.py` returns 0 lines
- [ ] The old import path still works. Check: `uv run python -c "from nnx._step_helpers import full_precision; from nnx._dtypes import full_precision as f; assert full_precision is f"`
- [ ] A new test `tests/test_precision_execution.py::test_the_widening_helpers_keep_float64` asserts `full_precision(torch.ones(1, dtype=torch.bfloat16)).dtype == torch.float32`, `full_precision(torch.ones(1, dtype=torch.float64)).dtype == torch.float64`, `promoted_dtype(torch.ones(1, dtype=torch.float16)) == torch.float32` and `promoted_dtype(torch.ones(1, dtype=torch.float64)) == torch.float64` (it imports `from nnx._dtypes import full_precision, promoted_dtype` in its body). Check: `uv run pytest tests/test_precision_execution.py::test_the_widening_helpers_keep_float64`
- [ ] The causal-LM, ranking, DoRA and precision suites pass. Check: `uv run pytest tests -k "lm_task or causal_lm or ranking or peft_dora or precision"`
- [ ] `uv run pytest tests/test_imports.py` passes. Check: that command
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [_step_helpers.py L85-L88](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_step_helpers.py#L85-L88) | `full_precision` |
| [_step_helpers.py L28-L29](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_step_helpers.py#L28-L29) | `_step_helpers` imports `models` and `nn.nn_model` |
| [lm_tasks.py L91-L96](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/lm_tasks.py#L91-L96) | `_token_nll`: inlined widening |
| [ranking.py L251-L252](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/ranking.py#L251-L252) | `pairwise_logistic_loss`: inlined widening |
| [dora.py L265-L280](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/peft/dora.py#L265-L280) | `DoRALinear.forward`: inlined promotion that keeps float64 |
| #516, #577, #766, #225 | the precision decisions at these sites |

## Scope

**In:** `_dtypes.py`, the `_step_helpers` re-export and the three sites.

**Out:** the other half-precision checks with different behaviour (`generation/logits_processors.py` L70, `precision.py` L245). Moving the other `_step_helpers.full_precision` importers to `_dtypes` (they keep working through the re-export).

## Dependencies

**Blocked by:** none
**Blocks:** none
**Related:** #516, #577, #766, #225
