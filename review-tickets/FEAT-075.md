---
id: FEAT-075
title: "[FEAT-075] Move the transform-replay helpers into nnx.transforms (part 2 of 5)"
labels: ["enhancement", "area: model-transforms", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-02"]
depends_on: ["FEAT-091", "FEAT-108"]
related_issues: [509, 767]
status: READY
---

**Category:** Model-transforms · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-075: the helpers that replay and check a model's recorded topology transforms (the FEAT-016 recipe and the QAT conversion) live in `nn/nn_model.py` (L1072–L1260), although `nnx.transforms` owns the recipe they replay. Move them, unchanged, into `src/nnx/transforms.py` and re-export them from `nn_model`. Part 2 of 5 of the `nn_model.py` split; no behaviour changes.

## Context

**What happens today.** `nn_model.py` replays recipe operations by calling back into `nnx.transforms` at call time:

```python
def _apply_checkpoint_transform(model: NNModel, transform: NNCheckpointTransform) -> None:
    if _replayable(transform):
        # FEAT-016: a recorded recipe operation — its topology is rebuilt
        # (low-rank factors allocated, never re-factorized) before the
        # saved tensors are loaded.
        from ..transforms import _replay

        _replay(model, transform)
        return
```

and holds the recipe-state checks (`_parametrized_prefix`, `_reparameterizing_hooks`, `_replaced_layers`, `_check_trained_recipe`, `_final_transforms`, `_refuse_unrecorded_recipe_state`) next to them.

**Why it matters.** The recipe logic is split across two modules, and `nnx.transforms`' readers do not see the replay rules.

**History.** FEAT-091 (SMELL-11) moves `_unportable_transform` into `nnx.transforms` and names the QAT transform once (`QAT_TRANSFORM`, `_is_qat_conversion`); FEAT-108 (SMELL-30) makes the Hub preflight replay transforms on a bare module. Both edit this code and land first (conflict order: FEAT-091 → FEAT-108 → this ticket). #767 (DEC-013) kept the refusal of an unreplayable transform before anything is written; #509 is the Hub save that calls these checks. Neither changes here.

**Background for the worker.**
- *Topology transform*: an `NNCheckpointTransform` (name, version, options) a checkpoint records so a reload can rebuild the network's shape (a recipe operation, or the torchao QAT conversion) before loading the weights.
- *Re-export*: `from ..transforms import _replay_transforms as _replay_transforms` in `nn_model.py`; the redundant alias keeps ruff and pyright quiet and keeps the name importable and visible to `NNModel`'s call-time lookups.
- *Cycle rule*: `nn_model` imports `nnx.transforms` at module top already; `nnx.transforms` imports `nn_model` only under `TYPE_CHECKING` and inside functions, which stays true.

**Proposal.**

1. Cut these top-level definitions from `src/nnx/nn/nn_model.py` (at `9ad461e` L1072–L1152 and L1163–L1260) and paste them at the end of `src/nnx/transforms.py`, in this order, as they stand after FEAT-091 and FEAT-108: `_topology_preflight`, `_collect_checkpoint_transforms`, `_apply_checkpoint_transform`, `_replay_transforms`, `_parametrized_prefix`, `_reparameterizing_hooks`, `_replaced_layers`, `_check_trained_recipe`, `_final_transforms`, `_refuse_unrecorded_recipe_state`. `_is_batch_list` (L1153–L1160, a graph-batch check) stays in `nn_model.py`; `_unportable_transform` is already in `nnx.transforms` (FEAT-091).
2. Adjust only the imports inside the moved code:
   - in `_apply_checkpoint_transform`, delete the local `from ..transforms import _replay` (`_replay` is now a module-level name) and change `from ..quantize.qat import _build_quantizer` to `from .quantize.qat import _build_quantizer` (it stays local: `nnx.quantize` imports `nnx.transforms`);
   - add to `transforms.py`'s module imports `import torch` and `from ._reparameterization import UNNAMED, legacy_reparameterizations`, and to its `if TYPE_CHECKING:` block `from .nn.callbacks import Callback` (used in annotations). `NNModel` and `NNCheckpointTransform` are already imported there under `TYPE_CHECKING`; `_recipe_transforms` and `_replayable` are defined in `transforms.py`.
3. In `nn_model.py`, where the block was, add the re-exports:
   ```python
   # Transform replay and the recipe-state checks live in nnx.transforms (FEAT-075).
   from ..transforms import (
       _apply_checkpoint_transform as _apply_checkpoint_transform,
       _check_trained_recipe as _check_trained_recipe,
       _collect_checkpoint_transforms as _collect_checkpoint_transforms,
       _final_transforms as _final_transforms,
       _parametrized_prefix as _parametrized_prefix,
       _refuse_unrecorded_recipe_state as _refuse_unrecorded_recipe_state,
       _replaced_layers as _replaced_layers,
       _replay_transforms as _replay_transforms,
       _reparameterizing_hooks as _reparameterizing_hooks,
       _topology_preflight as _topology_preflight,
   )
   ```
   then run `uv run ruff check --fix src/nnx/nn/nn_model.py` to move the import into the header and drop imports the remaining code no longer uses (at `9ad461e`: `UNNAMED` and `legacy_reparameterizations`).
4. No test patch needs a new target: `tests/test_resume_parity.py` L436–L437 patch `_topology_preflight` on `nn_model` and `trainer`, which `NNModel.train` and `Trainer.train` call through their own modules (the re-export and `trainer.py`'s import), and the `transforms._build` patches in `tests/test_model_recipes.py` (L172, L753, L991, L1042) already target `nnx.transforms`, where `_replay` now calls it from the same module.
5. Add this test to `tests/test_imports.py`:
   ```python
   def test_transform_replay_lives_in_nnx_transforms_and_stays_importable_from_nn_model():
       """FEAT-075: nn_model re-exports the replay helpers nnx.transforms now owns."""
       import nnx.nn.nn_model as nn_model
       import nnx.transforms as transforms

       for name in ("_topology_preflight", "_collect_checkpoint_transforms", "_apply_checkpoint_transform",
                    "_replay_transforms", "_reparameterizing_hooks", "_replaced_layers", "_check_trained_recipe",
                    "_final_transforms", "_refuse_unrecorded_recipe_state", "_parametrized_prefix"):
           assert getattr(nn_model, name) is getattr(transforms, name), name
       assert getattr(transforms._replay_transforms, "__module__") == "nnx.transforms"
   ```

No behaviour changes, so no CHANGELOG entry. None of these names is in `docs/api.md`.

**Worker setup.** Branch from `develop` after FEAT-091 and FEAT-108 have merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The new test `tests/test_imports.py::test_transform_replay_lives_in_nnx_transforms_and_stays_importable_from_nn_model` fails before the move and passes after it. Check: `uv run pytest tests/test_imports.py`
- [ ] `nn_model.py` defines none of the ten names. Check: `grep -nE "^def (_topology_preflight|_collect_checkpoint_transforms|_apply_checkpoint_transform|_replay_transforms|_parametrized_prefix|_reparameterizing_hooks|_replaced_layers|_check_trained_recipe|_final_transforms|_refuse_unrecorded_recipe_state)\b" src/nnx/nn/nn_model.py` prints nothing
- [ ] `nnx.transforms` no longer imports itself inside a function. Check: `grep -n "from \.\.transforms import\|from \.transforms import" src/nnx/transforms.py` prints nothing
- [ ] The recipe, checkpoint, bundle, QAT and Hub suites pass. Check: `uv run pytest tests -k "model_recipes or checkpoint_safetensors or run_bundles or quantize_qat or hub_mixin or hub_load_preflight or resume_parity"`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [nn_model.py L1072-L1152](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1072-L1152) | `_topology_preflight` through `_replay_transforms`, and the local `from ..transforms import _replay` |
| [nn_model.py L1153-L1160](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1153-L1160) | `_is_batch_list`, which stays |
| [nn_model.py L1163-L1260](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1163-L1260) | The recipe-state checks |
| [transforms.py L58-L62](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/transforms.py#L58-L62) | `transforms.py`'s `TYPE_CHECKING` imports (`NNModel`, `NNCheckpointTransform`) |
| [test_resume_parity.py L436-L437](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_resume_parity.py#L436-L437) | `_topology_preflight` patched on `nn_model` and `trainer` |
| #509, #767 | The Hub save and the pre-write refusal that use these checks |

## Scope

**In:** the ten definitions, their re-exports, one test.

**Out:**
- `_unportable_transform` and the QAT constant: FEAT-091. The bare-module preflight: FEAT-108.
- Hoisting `nnx.transforms`' other function-local imports: FEAT-139.
- Resume machinery (FEAT-074), Hub I/O (FEAT-076), DDP helpers (FEAT-077), lazy-module descriptor (FEAT-078).

## Dependencies

**Blocked by:** FEAT-091, FEAT-108
**Blocks:** FEAT-076, FEAT-139
**Related:** #509, #767
