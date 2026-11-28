---
id: FEAT-139
title: "[FEAT-139] Hoist needless function-local imports in nnx.transforms"
labels: ["enhancement", "area: model-transforms", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-61"]
depends_on: ["FEAT-075", "FEAT-119"]
related_issues: [249, 509, 767]
status: READY
---

**Category:** Model transforms · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-139: `src/nnx/transforms.py` imports 24 times inside functions. Two of them, `NNModel` from `nnx.nn.nn_model`, guard a real cycle; FEAT-119 replaces two more. This ticket moves the other 20 to module top and marks the two kept ones with `# cycle: nnx.nn.nn_model`.

## Context

**What happens today.** For example, `_fresh_base`:

```python
    from dataclasses import replace

    from .models import ModelSpec, _UnpackBatch
    from .nn.enum.devices import Devices
    from .nn.nn_model import NNModel
```

while `dataclasses` is already imported at module top, and `_build` imports `torch` locally twice although `torch.nn` is a module-top import.

**Why it matters.** Most of these imports suggest a cycle that does not exist; only the `NNModel` one is real.

**History.** #249 (FEAT-031) set the rule: keep an import local only for an optional extra or a real cycle. #509 and #767 shaped the recipe refusals in this module (refuse before anything is written); they are unchanged. FEAT-075 moves the transform-replay helpers from `nn_model.py` into this module, and FEAT-119 replaces the two `peft._targets` imports, so both land first.

**Background for the worker.**
- Classes: **hoist** (move to module top), **keep-cycle** (the target imports this module back), **keep-optional-extra**.
- The real cycle: `src/nnx/nn/nn_model.py` L102 and `src/nnx/nn/callbacks.py` L26 import `nnx.transforms` at module top, so `transforms.py` must not import `nnx.nn.nn_model` or `nnx.nn.callbacks` at module top.
- Cycle check done for this ticket: `nnx.nn.params.nn_checkpoint`, `nnx.seeding`, `nnx.surgery._utils`, `nnx.surgery.low_rank`, `nnx.utils`, `nnx.models`, `nnx.nn.enum.devices` and `nnx.peft.lora` (with their package `__init__` files) never reach `nnx.nn.nn_model`, `nnx.nn.callbacks` or `nnx.transforms` through module-top imports. No hoisted name collides with a name `transforms.py` defines.
- No test monkeypatches a name these imports bind for `transforms` (the `seeding._restore_rng_state` patches in `tests/test_plan_execution.py` L466, L525, L660 target `_seed_scope`, and `tests/test_mps_cpu_pins.py` L172 patches `low_rank._allocate_factors` for a direct `low_rank_factorize` call).

**Proposal.**
1. Every function-local import at `9ad461e`, with its class:

   | Line | Function | Import | Class |
   |---|---|---|---|
   | L185 | `TransformOp.checkpoint_transform` | `from .nn.params.nn_checkpoint import NNCheckpointTransform` | hoist (and delete the `TYPE_CHECKING` copy, L62) |
   | L317 | `TransformRecipe.validate` | `from .seeding import _global_rng_kept` | hoist |
   | L349 | `TransformRecipe.materialize` | `from .seeding import _global_rng_state, _restore_rng_state` | hoist |
   | L350 | `TransformRecipe.materialize` | `from .surgery._utils import get_module, set_module` | hoist |
   | L351 | `TransformRecipe.materialize` | `from .utils import _capture_training_modes, _restore_training_modes` | hoist |
   | L421 | `_check_source` | `from .models import RuntimeModule` | hoist |
   | L422 | `_check_source` | `from .nn.nn_model import NNModel` | keep-cycle |
   | L456 | `_fresh_base` | `from dataclasses import replace` | hoist (into `from dataclasses import dataclass, field, replace`) |
   | L458 | `_fresh_base` | `from .models import ModelSpec, _UnpackBatch` | hoist |
   | L459 | `_fresh_base` | `from .nn.enum.devices import Devices` | hoist |
   | L460 | `_fresh_base` | `from .nn.nn_model import NNModel` | keep-cycle |
   | L474 | `_registration_paths` | `from .peft._targets import _registrations` | — (FEAT-119) |
   | L484 | `_weight_reading_parents` | `from .peft._targets import WEIGHT_READING_PARENTS, _registrations` | — (FEAT-119) |
   | L548 | `_validate` | `from .surgery._utils import refuse_hooked, refuse_parametrized` | hoist |
   | L584 | `_build` | `from .peft.lora import LoRALinear` | hoist |
   | L591 | `_build` | `from .seeding import _global_rng_kept` | hoist |
   | L598 | `_build` | `from .surgery.low_rank import low_rank_factorize` | hoist |
   | L601 | `_build` | `import torch` | hoist (move `import torch` out of the `TYPE_CHECKING` block, L59) |
   | L603 | `_build` | `from .surgery.low_rank import _allocate_factors` | hoist |
   | L634 | `_replay` | `from .surgery._utils import get_module, set_module` | hoist |
   | L693 | `_state_shapes` | `import torch` | hoist |
   | L694 | `_state_shapes` | `from torch.nn.parameter import UninitializedBuffer, UninitializedParameter` | hoist |
   | L709 | `_topology_problems` | `from .peft.lora import LoRALinear` | hoist |
   | L758 | `check_optimizer` | `from .surgery._utils import get_module` | hoist |

2. Module top gains: `import torch`, `replace` in the `dataclasses` import, `from torch.nn.parameter import UninitializedBuffer, UninitializedParameter`, `from .models import ModelSpec, RuntimeModule, _UnpackBatch`, `from .nn.enum.devices import Devices`, `from .nn.params.nn_checkpoint import NNCheckpointTransform`, `from .peft.lora import LoRALinear`, `from .seeding import _global_rng_kept, _global_rng_state, _restore_rng_state`, `from .surgery._utils import get_module, refuse_hooked, refuse_parametrized, set_module`, `from .surgery.low_rank import _allocate_factors, low_rank_factorize`, `from .utils import _capture_training_modes, _restore_training_modes`. The `TYPE_CHECKING` block keeps only `from .nn.nn_model import NNModel`.
3. Keep the two `NNModel` imports where they are and append the marker: `from .nn.nn_model import NNModel  # cycle: nnx.nn.nn_model`.
4. The helpers FEAT-075 moved into this module follow the same rule: hoist each of their function-local imports, except one of `.nn.nn_model` or `.nn.callbacks`, which stays local with `# cycle: nnx.nn.nn_model` (or `# cycle: nnx.nn.callbacks`).
5. Delete each hoisted local import line and any blank line it leaves at the start of a function body. Run `uv run ruff check --fix src/nnx/transforms.py` and `uv run ruff format src/nnx/transforms.py`.

No behaviour changes, no test changes, no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] Only the cycle imports stay inside functions. Check: `uv run python -c "import ast; p='src/nnx/transforms.py'; print(sorted({n.module for f in ast.walk(ast.parse(open(p).read())) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(f) if isinstance(n, ast.ImportFrom)} | {a.name for f in ast.walk(ast.parse(open(p).read())) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(f) if isinstance(n, ast.Import) for a in n.names}))"` prints a subset of `['nn.callbacks', 'nn.nn_model']`
- [ ] Every kept import is marked. Check: `uv run python -c "import ast; p='src/nnx/transforms.py'; src=open(p).read(); lines=src.splitlines(); print([n.lineno for f in ast.walk(ast.parse(src)) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(f) if isinstance(n, (ast.Import, ast.ImportFrom)) and '# cycle:' not in lines[n.lineno - 1]])"` prints `[]`
- [ ] The recipe, transform, surgery and import suites pass. Check: `uv run pytest tests/test_model_recipes.py tests/test_imports.py` and `uv run pytest tests -k "transform or recipe or surgery or peft or materialize"`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [transforms.py L47-L62](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/transforms.py#L47-L62) | the module-top imports and the `TYPE_CHECKING` block |
| [transforms.py L418-L460](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/transforms.py#L418-L460) | `_check_source`, `_fresh_base`: the real `NNModel` cycle beside needless imports |
| [transforms.py L580-L605](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/transforms.py#L580-L605) | `_build`: five local imports |
| [transforms.py L690-L710](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/transforms.py#L690-L710) | `_state_shapes`, `_topology_problems` |
| [nn_model.py L102](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L102) | `nn_model` imports `transforms` at module top: the cycle |
| [callbacks.py L26](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/callbacks.py#L26) | `callbacks` imports `transforms` at module top |
| #249 | the "local only for an optional extra or a cycle" rule |

## Scope

**In:** the function-local imports of `transforms.py`.

**Out:** breaking the `nn_model` ↔ `transforms` cycle (FEAT-075 and the other SMELL-02 moves). The `peft._targets` imports (FEAT-119). `nn_model.py`'s own local imports of `transforms` (FEAT-095).

## Dependencies

**Blocked by:** FEAT-075 (moves code into this module), FEAT-119 (replaces two of its imports)
**Blocks:** none
**Related:** #249, #509, #767
