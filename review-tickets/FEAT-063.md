---
id: FEAT-063
title: "[FEAT-063] Remove dead code in nn_run, distributed, objectives, peft, benchmark and deepen"
labels: ["enhancement", "area: training", "priority: P3", "review: 2026-11-27"]
sources: ["DEAD-01", "DEAD-02", "DEAD-06", "DEAD-07", "DEAD-08", "DEAD-09"]
depends_on: []
related_issues: [570, 571]
status: READY
---

**Category:** Training · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-063: six pieces of code do nothing: a parameter no caller passes as `False`, two attributes written and never read, an unused type alias, a pointless alias variable, and two `try/except ImportError` blocks around imports that cannot fail. Delete them. No behaviour changes.

## Context

**What happens today.**

1. `_read_idps_csv` takes `exact=` that is never `False` (callers: `nn_run.py:459` with no argument, `history.py:1457` with `exact=True`, `tests/test_params_boundary_validation.py:307` with no argument):
   ```python
   def _read_idps_csv(csv_path: str, *, exact: bool = True) -> list[NNIterationDataPoint]:
   ```
   ```python
           frame = pd.read_csv(csv_path, float_precision=cast(Any, "round_trip")) if exact else pd.read_csv(csv_path)
   ```
2. `_DDPSession.validation_ids` is set (`distributed.py:585`, `self.validation_ids: list[int] = []`; `nn_model.py:2519`, `ddp.validation_ids = ids`) and never read. The tests read a report key named `"validation_ids"` that comes from `validation.ids()` on the loader (`tests/ddp_scenarios.py:545`, `examples/ddp_supervised.py:145`), not this attribute.
3. `objectives.py:242`: `ObjectiveFn = Callable[[ObjectiveContext], ObjectiveResult]` has no user and is not in `__all__`.
4. `PrefixTuner._original_attn_forwards` is filled (`prefix.py:201`, `:205`) and never read:
   ```python
           self._original_attn_forwards: list = []
           for i in range(n_layers):
               block = cast(Any, model.blocks[i])
               mha = block.attn
               self._original_attn_forwards.append(mha.forward)
   ```
5. `peft/_targets.py:158`: `selected = matched`, then `selected` is used twice (L161, L175).
6. Two imports guarded against an `ImportError` that cannot happen:
   ```python
       try:
           import torch

           if isinstance(first, torch.Tensor):
               return torch.stack(inputs)
       except ImportError:  # pragma: no cover - torch is a core dependency
           pass
   ```
   (`decisions/benchmark.py` `_batch_input`, L614–L620; torch is a core dependency and the module already imports `trainer.params`, which imports torch), and in `surgery/deepen.py` `_check_relu_activation` (L409–L421) a same-package `from ..nn.enum.activations import Activations as _A` inside `try/except ImportError: pass`, although `deepen.py` already imports `Activations` at L59.

**Why it matters.** Dead code misleads readers (a parser option, a teardown list, a guard against a missing torch) and has to be kept in sync for nothing.

**History.** #570 and #571 pinned the `"validation_ids"` key of the DDP scenario reports (`tests/ddp_scenarios.py:545`, `tests/test_ddp_adapter.py:191-193`, `tests/test_examples_smoke.py:545`). That key is built from the loader and stays; only the unread session attribute goes.

**Background for the worker.**
- *`idps.csv`*: a run's per-batch history file; `_read_idps_csv` reads it with pandas' round-trip float parser so reloaded floats equal the trained ones bit for bit.
- *`_DDPSession`*: the per-fit DDP state object in `nnx.distributed`.
- *`PrefixTuner`*: the prefix-tuning adapter in `nnx.peft.prefix`; it patches each attention block's `forward`.

**Proposal.**

1. `src/nnx/nn/params/nn_run.py`, `_read_idps_csv`:
   - change the signature (L467) to `def _read_idps_csv(csv_path: str) -> list[NNIterationDataPoint]:`;
   - in the docstring, replace `record when every value was NaN. ``exact=False`` is that default` + newline + `    parser."""` with `record when every value was NaN."""`;
   - change L478 to `        frame = pd.read_csv(csv_path, float_precision=cast(Any, "round_trip"))`.
2. `src/nnx/history.py` L1457: change `_read_idps_csv(csv_path, exact=True)` to `_read_idps_csv(csv_path)`.
3. Delete `src/nnx/distributed.py` L585 (`        self.validation_ids: list[int] = []`) and `src/nnx/nn/nn_model.py` L2519 (`    ddp.validation_ids = ids`). Keep the line above L2519 that builds `ids` (it feeds the "scored every row exactly once" check).
4. Delete `src/nnx/objectives.py` L242 (`ObjectiveFn = ...`) and the blank line pair around it, leaving two blank lines between the neighbouring definitions.
5. `src/nnx/peft/prefix.py`: delete L201 (`self._original_attn_forwards: list = []`) and L205 (`self._original_attn_forwards.append(mha.forward)`), and replace the comment L196–L200 with the single line `        # Monkey-patch each targeted block's MHA forward to inject the learned prefix.`.
6. `src/nnx/peft/_targets.py` (`select_linear_slots`): delete L158 (`    selected = matched`) and change `selected` to `matched` at L161 (`for slot in matched:`) and L175 (`return [(slots[slot][0], slot[1]) for slot in matched]`).
7. `src/nnx/decisions/benchmark.py`: add `import torch` as a third-party import block between `from typing import Any, Optional, Union` (L62) and `from .._json_depth import ...` (L64), and replace L614–L620 with:
   ```python
       if isinstance(first, torch.Tensor):
           return torch.stack(inputs)
   ```
   Leave the `import numpy as np` inside `_batch_input` alone (FEAT-136 hoists it).
8. `src/nnx/surgery/deepen.py` `_check_relu_activation`: replace L409–L421 (from `    try:` through `        pass`) with the same checks on the module-level import:
   ```python
       if isinstance(act, Activations):
           if act is Activations.RELU:
               return
           raise ValueError(
               f"deepen: the activation effective after hidden layer {idx} is {act.value!r}, "
               "but identity-init insertion is function-preserving only for ReLU. Sigmoid/tanh/"
               "GELU/etc. sites must be deepened by another method."
           )
   ```
   The message text stays word for word.

No test references `ObjectiveFn`, `_original_attn_forwards` or `validation_ids` as an attribute (`grep -rn "ObjectiveFn\|_original_attn_forwards\|\.validation_ids" src tests examples docs` lists only the lines deleted above). No behaviour changes, so no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] No dead name is left. Check: `grep -rn "ObjectiveFn\|_original_attn_forwards\|\.validation_ids\|exact=\|exact: bool\|selected = matched" src/nnx` prints nothing
- [ ] No `except ImportError` remains in `_batch_input` or `_check_relu_activation`. Check: `grep -n "except ImportError" src/nnx/decisions/benchmark.py src/nnx/surgery/deepen.py` prints nothing
- [ ] The `"validation_ids"` report key is untouched. Check: `git diff origin/develop -- tests/ddp_scenarios.py examples/ddp_supervised.py` is empty
- [ ] The affected suites pass. Check: `uv run pytest tests/test_run_load_exact_history.py tests/test_params_boundary_validation.py tests/test_history_journal.py tests/test_ddp_adapter.py tests/test_objective_engine.py tests/test_peft_prefix.py tests/test_peft_lora.py tests/test_peft_alias_preflight.py tests/test_decision_benchmark.py tests/test_surgery_deepen.py tests/test_imports.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [nn_run.py L459](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L459), [L467-L478](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L467-L478) | `_read_idps_csv`, its `exact=` and a caller without it |
| [history.py L1457](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/history.py#L1457) | The only caller passing `exact=True` |
| [distributed.py L585](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/distributed.py#L585), [nn_model.py L2519](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L2519) | `validation_ids` written, never read |
| [ddp_scenarios.py L545](https://github.com/thekaveh/NNx/blob/9ad461e/tests/ddp_scenarios.py#L545) | The report key comes from `validation.ids()` |
| [objectives.py L242](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/objectives.py#L242) | `ObjectiveFn` |
| [prefix.py L196-L205](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/peft/prefix.py#L196-L205) | `PrefixTuner._original_attn_forwards` |
| [_targets.py L158-L175](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/peft/_targets.py#L158-L175) | `selected = matched` in `select_linear_slots` |
| [benchmark.py L614-L620](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L614-L620) | `_batch_input`'s guarded torch import |
| [deepen.py L59](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/surgery/deepen.py#L59), [L409-L421](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/surgery/deepen.py#L409-L421) | `Activations` already imported; the guarded re-import |
| #570, #571 | The pinned `"validation_ids"` report key |

## Scope

**In:** the six deletions above.

**Out:**
- The `"validation_ids"` report key in tests and examples (#570, #571).
- Hoisting the other local imports of `benchmark.py`: FEAT-136.
- Reading `run.yaml` and `idps.csv` through one helper each: FEAT-098 (builds on this ticket).
- `bundles._save_tensors` (DEAD-10): resolved by FIX-121.

## Dependencies

**Blocked by:** none
**Blocks:** FEAT-099, FEAT-098
**Related:** #570, #571
