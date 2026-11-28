---
id: FEAT-095
title: "[FEAT-095] Hoist nn_model.py's needless function-local imports"
labels: ["enhancement", "area: packaging", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-15"]
depends_on: ["FEAT-082", "FEAT-091", "FEAT-108"]
related_issues: []
status: READY
---

**Category:** Packaging · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-095: `nn_model.py` imports about thirty names inside functions although no import cycle requires it: some come from modules it already imports at the top, the rest from modules that never import `nn_model` back. This ticket moves those imports to the module's top-level import lists. The `..distributed` imports stay (FEAT-077 removes them), and so do the local imports that a cycle, an optional dependency or a test seam needs. Behaviour does not change.

## Context

**What happens today.** Two examples. `_declared_module` (`nn_model.py` L1301) imports from a module already imported at the top (L100, `from ..seeding import _capture_rng_state, _restore_rng_state`):

```python
    from ..seeding import _global_rng_kept
```

`NNModel._normalize_callbacks` (L6760) imports from `nn/callbacks.py`, which reaches `nn_model` only under `TYPE_CHECKING`:

```python
        from .callbacks import Callback, _LegacyCallback
```

**Why it matters.** Local imports hide the module's real dependencies, cost a dictionary lookup on hot paths (`_step_scheduler` runs every update), and make readers suspect a cycle that does not exist.

**History.** No board issue covers this. SMELL-04 (FEAT-082) moves `_warn_at_user_frame` to `nnx/_warnings.py` and hoists its nine local imports in this file; this ticket does the rest.

**Background for the worker.**
- Function-local import: an `import` / `from … import` statement inside a function body; it runs at each call (cheaply, after the first) instead of once at module import.
- Import cycle: module A imports B at the top while B (directly or through other modules) imports A at the top; one of the two must then import lazily. A module "reaches" `nn_model` when its own top-level imports, followed transitively, import `nnx.nn.nn_model`. Checked at `9ad461e`: none of `nnx.streaming`, `nnx.optimizers`, `nnx.objectives`, `nnx.nn.callbacks`, `nnx.provenance`, `nnx.prediction`, `nnx.link_tasks`, `nnx.finetune` (package and `freezing`), `nnx._update_engine`, `nnx._scheduler_clock`, `nnx.seeding`, `nnx.transforms`, `nnx.models` or `nnx.nn.params.nn_checkpoint` reaches it. `nnx.quantize` (through `qat`) and `nnx.trainer` (through `trainer.trainer`) do.
- Test seam: a module attribute that a test monkeypatches; a top-level `from X import name` would bind the name in `nn_model` once, so the patch would stop intercepting.
- Line numbers below are at `9ad461e`. FEAT-082, FEAT-091 and FEAT-108 land first and shift them; find each line by its function and its import text.

**Proposal.**
1. Delete each local import below and add its names to `nn_model.py`'s top-level imports (merge into the existing `from <module> import (...)` statement where there is one; otherwise add a new statement in the first-party block):

   | L (9ad461e) | Function | Local import to remove |
   |---|---|---|
   | 477 | `_verified_resume_point` | `from .params.nn_checkpoint import _checkpoint_path` |
   | 953 | `_resume_fit` | `from ..optimizers import _canonical_factory_state` |
   | 1301 | `_declared_module` | `from ..seeding import _global_rng_kept` |
   | 2273 | `_compile_session` | `from .callbacks import _changes_topology` |
   | 2306 | `_writer_owned_callbacks` | `from .callbacks import _WriterOwned` |
   | 2327 | `_distributed_scope` | `from .callbacks import _changes_topology` (the `..distributed` line above it stays) |
   | 2610 | `_with_attempt` | `from ..provenance import _AttemptRecorder` |
   | 2792 | `_named_metric_set` | `from ..streaming import _bounded_accumulator` |
   | 3074 | `_objective_microbatch` | `from ..objectives import ObjectiveContext, ObjectiveResult` |
   | 3153 | `_objective_engine` | `from ..objectives import Objective` (the `UpdateEngine` line above it stays) |
   | 3567 | `_is_streaming_eval_step` | `from ..streaming import streaming_eval_step` |
   | 4117 | `NNModel._base_state` | `from ..seeding import _global_rng_kept` |
   | 4144 | `NNModel._topology_drift` | `from ..transforms import _topology_problems` |
   | 4938 | `NNModel._from_pretrained` | `from ..models import resolve_model_factory` |
   | 4990 | `NNModel.freeze` | `from ..finetune.freezing import freeze as _freeze` |
   | 4997 | `NNModel.unfreeze` | `from ..finetune.freezing import unfreeze as _unfreeze` |
   | 5129 | `NNModel.train` | `from ..objectives import _check_objective_run, _check_update_owner` |
   | 5165 | `NNModel.train` | `from ..streaming import _streaming_preflight` |
   | 5169 | `NNModel.train` | `from ..link_tasks import _link_eval_preflight` |
   | 5197 | `NNModel.train` | `from ..seeding import set_seed` |
   | 5376 | `NNModel._train_impl` | `from ..optimizers import optimizer_factory_state` |
   | 6133 | `NNModel.iter_predict` | `from ..prediction import _check_spec_fits, prediction_from_logits` |
   | 6134 | `NNModel.iter_predict` | `from ..streaming import PredictionBatch, PredictionStream, _as_probability_spec, _check_stream_source` |
   | 6212 | `NNModel.predict_proba` | `from ..prediction import _check_spec_fits, prediction_from_logits` |
   | 6727 | `NNModel._step_scheduler` | `from .._scheduler_clock import step_scheduler` |
   | 6760 | `NNModel._normalize_callbacks` | `from .callbacks import Callback, _LegacyCallback` |

   The top-level statements that result (before sorting):
   - `from .._scheduler_clock import (...)` (L37) gains `step_scheduler`;
   - `from ..models import (...)` (L58) gains `resolve_model_factory`;
   - `from ..seeding import ...` (L100) becomes `from ..seeding import _capture_rng_state, _global_rng_kept, _restore_rng_state, set_seed`;
   - `from ..transforms import ...` (L102) gains `_topology_problems`;
   - `from ..provenance import ExperimentManifest` (L99) gains `_AttemptRecorder`;
   - `from .params.nn_checkpoint import (...)` (L107) gains `_checkpoint_path`;
   - new: `from ..finetune.freezing import freeze as _freeze, unfreeze as _unfreeze`, `from ..link_tasks import _link_eval_preflight`, `from ..objectives import Objective, ObjectiveContext, ObjectiveResult, _check_objective_run, _check_update_owner`, `from ..optimizers import _canonical_factory_state, optimizer_factory_state`, `from ..prediction import PredictionResult, ProbabilitySpec, _check_spec_fits, prediction_from_logits`, `from ..streaming import PredictionBatch, PredictionStream, _as_probability_spec, _bounded_accumulator, _check_stream_source, _streaming_preflight, streaming_eval_step`, `from .callbacks import Callback, _changes_topology, _LegacyCallback, _WriterOwned`.
2. Delete the `if TYPE_CHECKING:` block (L139–L142): its three imports (`PredictionResult`, `ProbabilitySpec`; `PredictionStream`; `Callback`) are now runtime imports from step 1. Then remove `TYPE_CHECKING` from `from typing import ...` (L22); nothing else in the file uses it.
3. Keep these local imports, unchanged:
   - every `from ..distributed import …` (12 lines; FEAT-077 moves the DDP helpers out of this file);
   - `from ..quantize.qat import _build_quantizer` (cycle: `nnx.quantize.qat` imports `nn_model` at the top);
   - `from ..trainer.params import NNTrainerParams` in `_source_hyperparameters` (cycle: importing `nnx.trainer.params` runs `nnx/trainer/__init__.py`, which imports `trainer.trainer`, which imports `nn_model`);
   - `from .._update_engine import UpdateEngine` in `_objective_engine` (test seam: `tests/test_objective_engine.py` L1030 and L1089 patch `nnx._update_engine.UpdateEngine` and rely on this lookup). Add the comment `# imported here: tests patch nnx._update_engine.UpdateEngine` on the line above it;
   - `from ..optimizers import build_optimizer` in `NNModel.train` (test seam: `tests/test_optimizer_factories.py` L352 patches `nnx.optimizers.build_optimizer`). Add the comment `# imported here: tests patch nnx.optimizers.build_optimizer` on the line above it;
   - the optional or heavy third-party imports: `onnx`, `onnxscript`, `torch.export`, `safetensors.torch`, `huggingface_hub`;
   - `from .._step_helpers import supervised_input` in `_single_input_batch` (added by FEAT-105; cycle: `_step_helpers` imports `nn_model` at the top).
4. Run `uv run ruff check --fix src/nnx/nn/nn_model.py` (it sorts the import blocks) and `uv run ruff format src/nnx/nn/nn_model.py`.
5. Before step 1, confirm no test patches a hoisted name: `grep -rn "setattr\|patch" tests | grep -wE "_checkpoint_path|_canonical_factory_state|_global_rng_kept|_changes_topology|_WriterOwned|_bounded_accumulator|ObjectiveContext|ObjectiveResult|streaming_eval_step|_topology_problems|resolve_model_factory|_check_objective_run|_check_update_owner|_streaming_preflight|_link_eval_preflight|set_seed|optimizer_factory_state|_check_spec_fits|prediction_from_logits|PredictionBatch|_as_probability_spec|_check_stream_source|step_scheduler|_LegacyCallback"` returns nothing at `9ad461e`. (`tests/test_params_boundary_validation.py` L1894 patches a method of the `_AttemptRecorder` class, which the hoist does not affect.)
6. No behaviour changes, so there is no CHANGELOG entry.

**Worker setup.** Branch from `develop` after FEAT-082, FEAT-091 and FEAT-108 have merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] None of the hoisted modules is imported inside a function. Check: `grep -nE "^\s+from \.\.?(callbacks|streaming|objectives|seeding|models|_scheduler_clock|transforms|params\.nn_checkpoint|provenance|prediction|link_tasks|finetune\.freezing) import" src/nnx/nn/nn_model.py` returns 0 lines
- [ ] The two test seams stay local, each with its comment. Check: `grep -nE "^\s+from \.\.(optimizers|_update_engine) import" src/nnx/nn/nn_model.py` returns exactly 2 lines (`build_optimizer`, `UpdateEngine`), and `grep -c "# imported here: tests patch" src/nnx/nn/nn_model.py` prints `2`
- [ ] The `..distributed` imports are untouched. Check: `grep -c "from ..distributed import" src/nnx/nn/nn_model.py` prints `12`
- [ ] `nnx` and its submodules import cleanly in a fresh interpreter. Check: `uv run python -c "import nnx.nn.nn_model, nnx.trainer, nnx.quantize, nnx.objectives, nnx.streaming"` and `uv run pytest tests/test_imports.py tests/test_core_install.py`
- [ ] The two seams still intercept. Check: `uv run pytest tests/test_objective_engine.py -k bit_for_bit` and `uv run pytest tests/test_optimizer_factories.py::test_nn_model_and_trainer_share_one_build_hook`
- [ ] The full suite passes. Check: `uv run pytest -n logical --dist loadgroup`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [nn_model.py L22](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L22), [L139-L142](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L139-L142) | `TYPE_CHECKING` and the imports it guards |
| [nn_model.py L37-L46](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L37-L46), [L58-L82](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L58-L82), [L99-L102](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L99-L102), [L107-L115](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L107-L115) | the top-level imports of `_scheduler_clock`, `models`, `provenance`, `seeding`, `transforms`, `nn_checkpoint` |
| [nn_model.py L1301](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1301), [L6760](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L6760) | the two examples quoted |
| [nn_model.py L3152](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L3152), [L5201](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L5201) | the `UpdateEngine` and `build_optimizer` seams |
| [nn_model.py L1099](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1099), [L2894](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L2894) | the two cycle-bound imports kept |
| [test_objective_engine.py L1024-L1031](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_objective_engine.py#L1024-L1031), [L1089](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_objective_engine.py#L1089) | patches `nnx._update_engine.UpdateEngine` |
| [test_optimizer_factories.py L341-L358](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_optimizer_factories.py#L341-L358) | patches `nnx.optimizers.build_optimizer` |
| [callbacks.py L34](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/callbacks.py#L34), [L73](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/callbacks.py#L73), [L80](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/callbacks.py#L80), [L722](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/callbacks.py#L722) | `Callback`, `_changes_topology`, `_LegacyCallback`, `_WriterOwned` are module-level |

## Scope

**In:** the local imports listed in step 1 and the `TYPE_CHECKING` block.

**Out:** the `_warn_at_user_frame` imports (FEAT-082). The `..distributed` imports (FEAT-077, with the DDP move). The `nn_model` refactors that move code out (FEAT-074 to FEAT-078). Local imports in other modules (FEAT-136 to FEAT-140).

## Dependencies

**Blocked by:** FEAT-082, FEAT-091, FEAT-108
**Blocks:** none
**Related:** FEAT-077, FEAT-105
