---
id: FEAT-078
title: "[FEAT-078] Move the lazy-module descriptor into nnx.models (part 5 of 5)"
labels: ["enhancement", "area: model-integration", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-02"]
depends_on: ["FEAT-093", "FEAT-094"]
related_issues: [562, 764, 528]
status: READY
---

**Category:** Model-integration · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-078: the code that resolves an `NNModel`'s network descriptor and matches a lazy (not yet materialized) module against a saved run (`nn_model.py` L3592–L3837) uses only helpers from `nnx.models`, where the lazy-layer rules live. Move it, unchanged, to the end of `src/nnx/models.py` and re-export it from `nn_model`. Part 5 of 5 of the `nn_model.py` split; no behaviour changes.

## Context

**What happens today.** `nn_model.py` imports a dozen lazy-layer helpers from `nnx.models` only to use them here:

```python
def _resolve_net_descriptor(
    net_params: Optional[NNParams],
    params: NNModelParams,
    module: Optional[torch.nn.Module],
    *,
    state: Optional[Callable[[], Mapping[str, Any]]] = None,
    reloading: bool = False,
    explain_state: Optional[Callable[[ValueError, torch.nn.Module], Optional[Exception]]] = None,
) -> NNModelParams:
```

```python
_LAZY_RELOAD: contextvars.ContextVar[Optional[_LazyReload]] = contextvars.ContextVar("nnx_lazy_reload", default=None)
```

**Why it matters.** The lazy-module rules are split between two modules, and `nn_model.py` carries 250 more lines than its own concern needs.

**History.** #562 (FIX-094) made a lazy module's `RuntimeModule` descriptor stable across its first forward; #764 (DEC-010) decided whether a fresh lazy module may take a materialized run's parameters; #528 (FIX-061) let `load_pretrained` load into uninitialized lazy modules. FEAT-093 (SMELL-13) bundles `NNModel`'s lazy tracking into one object and adds a `descriptor=` parameter to `_resolve_net_descriptor`; FEAT-094 (SMELL-14) shares the dry-run inference rules in `nnx.models`. Both edit this code and land first.

**Background for the worker.**
- *Lazy module*: a torch module whose parameter shapes are fixed only by its first forward (`nn.LazyLinear`, …). NNx describes a caller's module as a `RuntimeModule` and checks a reload's module against the saved descriptor.
- *`_LAZY_RELOAD`*: a context variable `NNModel.from_checkpoint` sets so the constructor it calls can check the caller's module against the saved weights.
- *Re-export*: `from ..models import _resolve_net_descriptor as _resolve_net_descriptor` in `nn_model.py`; keeps the name importable (`plans.py` imports it from `nn_model`) and visible to `NNModel`'s call-time lookups.
- *Cycle check*: `nnx.models` must not import `nn_model`. The moved code needs only `Nets` at run time from `nnx.nn`; `nnx.nn.enum.nets` and its imports never import `nnx.models` or `nn_model` (checked), so importing it at the top of `models.py` is safe. `NNModelParams` and `NNParams` appear only in annotations.

**Proposal.**

1. Cut these top-level definitions from `src/nnx/nn/nn_model.py` (at `9ad461e` L3592–L3837), as they stand after FEAT-093 and FEAT-094, and paste them at the end of `src/nnx/models.py`, in this order, unchanged: `_resolve_net_descriptor`, `_LAZY_RULE`, `_descriptor_mismatch`, `_match_lazy_module`, `_saved_dtypes`, `_fits_state`, `_LazyReload`, `_LAZY_RELOAD`. `_shuffles` and `_warn_positional_ids` (L3574–L3589) stay in `nn_model.py`.
2. Add to `models.py`'s module imports:
   ```python
   from dataclasses import dataclass, replace

   from .nn.enum.nets import Nets
   ```
   and a `TYPE_CHECKING` block with `from .nn.params.nn_model_params import NNModelParams` and `from .nn.params.nn_params import NNParams` (add `TYPE_CHECKING` to the `typing` import). `contextlib`, `contextvars`, `torch`, `Callable`, `Mapping` and every lazy-layer helper the code uses (`_has_pending_lazy`, `_lazy_layers_emulable`, `_materialized_class_name`, `_materialized_topology`, `_same_class_once_materialized`, `_unmaterialized_topology`, `check_state_schema`, `ModelSpec`, `RuntimeModule`, `MissingModelFactoryError`) are already in `models.py`. The parameter named `replace` of `register_model_factory` shadows `dataclasses.replace` only inside that function; leave it.
3. In `nn_model.py`, add the re-exports with the redundant alias:
   ```python
   # The lazy-module descriptor lives in nnx.models (FEAT-078).
   from ..models import (
       _LAZY_RELOAD as _LAZY_RELOAD,
       _LAZY_RULE as _LAZY_RULE,
       _descriptor_mismatch as _descriptor_mismatch,
       _fits_state as _fits_state,
       _LazyReload as _LazyReload,
       _match_lazy_module as _match_lazy_module,
       _resolve_net_descriptor as _resolve_net_descriptor,
       _saved_dtypes as _saved_dtypes,
   )
   ```
   then run `uv run ruff check --fix src/nnx/nn/nn_model.py src/nnx/models.py` to merge it into the existing `from ..models import (...)` block and drop the helpers `nn_model.py` no longer uses.
4. No test patch needs a new target: `tests/test_external_module_runtime.py` L1156 patches `nn_model._pending_lazy_modules`, which `NNModel.__init__` and `_refresh_lazy_descriptor` (both staying in `nn_model`) read, and the moved code does not use. No test patches a moved name.
5. Add this test to `tests/test_imports.py`:
   ```python
   def test_the_lazy_module_descriptor_lives_in_nnx_models_and_stays_importable_from_nn_model():
       """FEAT-078: nn_model re-exports the descriptor code nnx.models now owns."""
       import nnx.models as models
       import nnx.nn.nn_model as nn_model

       for name in ("_resolve_net_descriptor", "_descriptor_mismatch", "_match_lazy_module", "_saved_dtypes",
                    "_fits_state", "_LazyReload", "_LAZY_RELOAD", "_LAZY_RULE"):
           assert getattr(nn_model, name) is getattr(models, name), name
   ```

No behaviour changes, so no CHANGELOG entry. None of these names is in `docs/api.md`, and `_LazyReload` is never pickled (it lives in a context variable).

**Worker setup.** Branch from `develop` after FEAT-093 and FEAT-094 have merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The new test `tests/test_imports.py::test_the_lazy_module_descriptor_lives_in_nnx_models_and_stays_importable_from_nn_model` fails before the move and passes after it. Check: `uv run pytest tests/test_imports.py`
- [ ] `nn_model.py` defines none of the moved names. Check: `grep -nE "^(def|class) (_resolve_net_descriptor|_descriptor_mismatch|_match_lazy_module|_saved_dtypes|_fits_state|_LazyReload)\b|^_LAZY_(RELOAD|RULE) " src/nnx/nn/nn_model.py` prints nothing
- [ ] `nnx.models` imports no `nn_model` at module level. Check: `grep -n "^from \.nn\.nn_model\|^from \.nn import nn_model" src/nnx/models.py` prints nothing
- [ ] The lazy-module, finetune, pickle and plan suites pass. Check: `uv run pytest tests -k "external_module_runtime or finetune_loading or legacy_net_pickles or model_factories or quantize_ptq or experiment_plan or plan_execution"`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [nn_model.py L57-L80](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L57-L80) | The lazy-layer helpers `nn_model` imports from `nnx.models` |
| [nn_model.py L3574-L3589](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L3574-L3589) | `_shuffles` and `_warn_positional_ids`, which stay |
| [nn_model.py L3592-L3837](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L3592-L3837) | The descriptor block that moves |
| [models.py L37-L52](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/models.py#L37-L52) | `nnx.models`' imports (no `nn_model`) |
| [models.py L410-L412](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/models.py#L410-L412) | The lazy-layer families the moved code relies on |
| [test_external_module_runtime.py L1150-L1156](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_external_module_runtime.py#L1150-L1156) | The `_pending_lazy_modules` patch, unaffected |
| #562, #764, #528 | Stable lazy descriptors; fresh-module policy; lazy `load_pretrained` |

## Scope

**In:** the eight definitions, their re-exports, one test.

**Out:**
- `NNModel`'s lazy tracking object: FEAT-093. The shared dry-run rules: FEAT-094.
- Resume (FEAT-074), transform replay (FEAT-075), Hub I/O (FEAT-076), DDP (FEAT-077).

## Dependencies

**Blocked by:** FEAT-093, FEAT-094
**Blocks:** none
**Related:** #562, #764, #528
