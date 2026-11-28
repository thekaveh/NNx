---
id: FEAT-104
title: "[FEAT-104] Remove small duplicates in probability, nets and evaluation records"
labels: ["enhancement", "area: numerics", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-26", "SMELL-32", "SMELL-33"]
depends_on: []
related_issues: [532, 545, 544]
status: READY
---

**Category:** Numerics · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); `feed_fwd_nn.py` and `transformer_nn.py` re-read at `1402b49` (#859); not run: torch was unavailable in the review environment

## Summary

FEAT-104: Three small duplicates. `_probability.py` spells bfloat16's epsilon expression twice; three built-in nets define the same one-line `__setstate__`, and `Devices.torch_device()` copies `Devices.__call__`; two private helpers named `_is_count` follow different rules. This ticket names the bfloat16 epsilon once, assigns the shared `__setstate__` directly, makes `torch_device` call `self()`, and renames the evaluation-record helper to `_csv_count`. Behaviour does not change.

## Context

**What happens today.**
1. `_probability.py` L110–L111 (in `logit_probabilities`) and L201–L202 (in `categorical_probabilities_eps`):

   ```python
       # bfloat16 tensors arrive upcast to float32; their rounding is bfloat16's.
       source_eps = 2.0**-7 if is_tensor(logits) and str(logits.dtype) == "torch.bfloat16" else None
   ```

   (the second reads `value` instead of `logits`).
2. `nn/net/feed_fwd_nn.py` L52–L55 (L53–L56 at `1402b49`, where #859 added an import above it), and the same method in `conv_nn.py` L91–L95 and `feed_fwd_moe_nn.py` L88–L91:

   ```python
       def __setstate__(self, state: dict[str, Any]) -> None:
           # A whole-module pickle from 0.3.x stored params as a plain attribute
           # and no cached fields: set it through the property (FIX-065).
           restore_params_property(self, state)
   ```

   `nn/enum/devices.py` L19–L25:

   ```python
       def __call__(self) -> torch.device:
           return torch.device(self.value)

       def torch_device(self) -> torch.device:
           """Explicit alias for ``self()`` — more readable in code that mixes
           the enum and torch.device usage."""
           return torch.device(self.value)
   ```

3. `_update_engine._is_count` (L543) accepts only a non-negative `int` (no `bool`). `nn_evaluation_data_point._is_count` (L44) also accepts an integral float, because a CSV reload gives floats for integer columns; its docstring already says so.

**Why it matters.** Each copy is a place to fix twice; two helpers with one name and different rules invite calling the wrong one.

**History.**
- #532 (FIX-065) added `restore_params_property` so whole-module pickles from 0.3.x and 0.4.x restore; `tests/test_legacy_net_pickles.py` pins it and must keep passing.
- #545 made the evaluation record test a non-integral value for finiteness before `int()`; #544 covers the shared validators. The evaluation helper's docstring explains why it differs, so only the name changes here.

**Background for the worker.**
- bfloat16: a 16-bit float with 8 significand bits; its machine epsilon is `2**-7`. NumPy has no bfloat16 dtype, so a bfloat16 tensor arrives upcast to float32 and its epsilon must be supplied.
- `is_tensor(value)` (`_probability.py` L72): whether `value` is a `torch.Tensor`, checked by type name, without importing torch.
- `restore_params_property(net, state)` (`nn/net/_pickling.py` L8): an `nn.Module.__setstate__` for nets whose `params` is a property. Its signature `(net, state)` fits a method, so it can be assigned in the class body.

**Proposal.**
1. In `src/nnx/_probability.py`:
   - after `LOW_PRECISION_FACTOR`'s docstring (L49–L52), add:

     ```python
     BFLOAT16_EPS = 2.0**-7
     """Machine epsilon of bfloat16 (8 significand bits), which NumPy has no dtype for."""
     ```

   - after `is_tensor` (L72–L75), add:

     ```python
     def _source_eps(value: Any) -> Optional[float]:
         """bfloat16's epsilon for a bfloat16 tensor, which arrives upcast to
         float32 so its rounding is bfloat16's; ``None`` for anything else."""
         return BFLOAT16_EPS if is_tensor(value) and str(value.dtype) == "torch.bfloat16" else None
     ```

   - replace L110–L111 with `source_eps = _source_eps(logits)` and L201–L202 with `source_eps = _source_eps(value)` (the comment moves into the helper's docstring).
2. In `src/nnx/nn/net/feed_fwd_nn.py`, `conv_nn.py` and `feed_fwd_moe_nn.py`, replace each `def __setstate__(...)` method by a class attribute in the same place, keeping its comment:

   ```python
       # A whole-module pickle from 0.3.x stored params as a plain attribute
       # and no cached fields: set it through the property (FIX-065).
       __setstate__ = restore_params_property
   ```

   (use each file's own comment text). Then `Any` is unused in all three files: change `from typing import Any, Optional` to `from typing import Optional` in `feed_fwd_nn.py` (L5) and `conv_nn.py` (L19), and delete `from typing import Any` in `feed_fwd_moe_nn.py` (L23). `transformer_nn.py` keeps its own `__setstate__`, which does something else. #859's `from ..._artifacts import atomic_torch_save` in `feed_fwd_nn.py` (L11 at `1402b49`) stays: `to_file` uses it.
3. In `src/nnx/nn/enum/devices.py`, change the body of `torch_device` (L25) to `return self()`.
4. In `src/nnx/nn/params/nn_evaluation_data_point.py`, rename `_is_count` (L44) to `_csv_count`, update its one call (L119, `if not _csv_count(count):`), and below the function add `_is_count = _csv_count  # the old name, kept importable`. `_update_engine._is_count` keeps its name.
5. No test monkeypatches these names (`grep -rn "_is_count\|_source_eps\|torch_device\|__setstate__" tests | grep "setattr\|patch"` returns nothing), and nothing changes behaviour, so there is no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The bfloat16 epsilon is written once. Check: `grep -c "2.0\*\*-7" src/nnx/_probability.py` prints `1`, and `grep -c "_source_eps(" src/nnx/_probability.py` prints `3`
- [ ] The three nets assign the shared `__setstate__`. Check: `grep -rn "__setstate__ = restore_params_property" src/nnx/nn/net` returns 3 lines, and `grep -rn "def __setstate__" src/nnx/nn/net` returns 1 line (`transformer_nn.py`)
- [ ] `torch_device` delegates. Check: `grep -n "return self()" src/nnx/nn/enum/devices.py` returns 1 line
- [ ] The evaluation helper has its own name. Check: `grep -n "def _csv_count" src/nnx/nn/params/nn_evaluation_data_point.py` returns 1 line
- [ ] Legacy pickles, devices, bfloat16 probabilities and records pass unchanged. Check: `uv run pytest tests/test_legacy_net_pickles.py tests/test_calibration.py tests/test_prediction_results.py tests/test_iteration_data_point.py tests/test_pass2_ou_series.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [_probability.py L43-L52](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_probability.py#L43-L52) | the module's named constants |
| [_probability.py L72-L75](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_probability.py#L72-L75) | `is_tensor` |
| [_probability.py L110-L111](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_probability.py#L110-L111), [L201-L202](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_probability.py#L201-L202) | the two copies |
| [feed_fwd_nn.py L52-L55](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/net/feed_fwd_nn.py#L52-L55), [conv_nn.py L91-L95](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/net/conv_nn.py#L91-L95), [feed_fwd_moe_nn.py L88-L91](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/net/feed_fwd_moe_nn.py#L88-L91) | the three `__setstate__` methods |
| [_pickling.py L8-L29](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/net/_pickling.py#L8-L29) | `restore_params_property` (FIX-065) |
| [devices.py L19-L25](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/enum/devices.py#L19-L25) | `__call__` and its copy |
| [nn_evaluation_data_point.py L44-L62](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_evaluation_data_point.py#L44-L62), [L119](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_evaluation_data_point.py#L119) | the CSV-tolerant `_is_count` and its call |
| [_update_engine.py L543-L544](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_update_engine.py#L543-L544) | the strict `_is_count` |
| #532, #545, #544 | legacy net pickles; finite-before-`int()`; the shared validators |

## Scope

**In:** `BFLOAT16_EPS`, `_source_eps`, three `__setstate__` assignments, `torch_device`, the `_csv_count` rename.

**Out:** replacing either `_is_count` with `_validation.require_count`: they validate different inputs (a restored engine state; a CSV-reloaded record). The 1e-6 probability tolerance (FEAT-127).

## Dependencies

**Blocked by:** none
**Blocks:** none
**Related:** #532, #545, #544
