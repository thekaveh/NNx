---
id: FEAT-129
title: "[FEAT-129] Share the legacy reparameterization walk and record per-API error types"
labels: ["enhancement", "area: model-transforms", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-52"]
depends_on: []
related_issues: [523, 524, 525, 554]
status: READY
---

**Category:** Model transforms · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-129: the GGUF export check (`tensor_name_map._refuse_reparameterized`) and the training preflight (`nn_model._reparameterizing_hooks`) both walk `named_modules()` and qualify the tensor names that unbaked prune masks and legacy `weight_norm` / `spectral_norm` hooks recompute. This ticket moves that walk into `nnx._reparameterization` as `legacy_reparameterized_paths` (plus `qualified_name`), and writes down in that module's docstring which error type each API raises for these refusals, as #523, #524, #525 and #554 decided. No error type or message changes.

## Context

**What happens today.** `src/nnx/interop/gguf/tensor_name_map.py`:

```python
    for path, module in net.named_modules():
        masks, norms = legacy_reparameterizations(module)
        for names, found in ((masks, masked), (norms, normed)):
            for name in names:
                # The tensor's qualified name, or the module's for a hook without one.
                found.append((path or "<root>") if name == UNNAMED else f"{path}.{name}" if path else name)
```

`src/nnx/nn/nn_model.py`:

```python
    for path, module in net.named_modules():
        masks, norms = legacy_reparameterizations(module)
        for names, found, own in ((masks, masked, ("_mask",)), (norms, normed, ("_g", "_orig"))):
            for name in names:
                if name == UNNAMED:
                    continue
                key = f"{path}.{name}" if path else name
```

**Why it matters.** The walk and the qualification rule are the same; only what each caller does with an unnamed hook and with the result differs.

**History.** The error types differ on purpose. They are deliberate: GGUF export raises `TypeError` (#525, FIX-058), a DoRA base `TypeError` (#523, FIX-056), surgery primitives `ValueError` (#524, FIX-057), `magnitude_prune` `ValueError` (#554, FIX-086). So SMELL-52's proposed rule ("`TypeError` for an unsupported kind, `ValueError` for an unsupported configuration") is dropped; this ticket only records the per-API types as they are.

**Background for the worker.**
- Legacy reparameterization: an unbaked `torch.nn.utils.prune` mask or the legacy `weight_norm` / `spectral_norm`, which recompute a tensor (`weight`) in a forward pre-hook; code that reads the tensor without calling the module sees a stale value.
- `legacy_reparameterizations(module)` (`src/nnx/_reparameterization.py` L27): the sorted tensor names `module`'s own hooks recompute, as `(masked, normed)`; `UNNAMED` (`"?"`) for a hook that records no name.
- `_reparameterizing_hooks` sits in `nn_model.py` at `9ad461e`; FEAT-078 may move nearby code into `nnx.models`. Find it by name: `git grep -n "def _reparameterizing_hooks" src/`.

**Proposal.**
1. In `src/nnx/_reparameterization.py`, after `legacy_reparameterizations`, add:

   ```python
   def qualified_name(path: str, name: str) -> str:
       """``name`` under the module at ``path`` (``"blocks.0.w1"`` + ``"weight"``);
       ``name`` itself for the root (``path == ""``)."""
       return f"{path}.{name}" if path else name


   def legacy_reparameterized_paths(net: nn.Module) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
       """``(masked, normed)``: a ``(module path, tensor name)`` pair for every
       tensor an unbaked prune mask or a legacy ``weight_norm`` /
       ``spectral_norm`` hook anywhere in ``net`` recomputes, in
       ``named_modules()`` order (sorted within a module), :data:`UNNAMED`
       for a hook without a name. Callers qualify the names
       (:func:`qualified_name`) and decide what an unnamed hook means."""
       masked: list[tuple[str, str]] = []
       normed: list[tuple[str, str]] = []
       for path, module in net.named_modules():
           masks, norms = legacy_reparameterizations(module)
           masked.extend((path, name) for name in masks)
           normed.extend((path, name) for name in norms)
       return masked, normed
   ```

2. Append this paragraph to the `_reparameterization.py` module docstring, before its closing `"""`:

   ```
   The refusals that read these helpers keep the error types their issues
   chose, one per API: GGUF export (``nnx.interop.gguf``) raises
   ``TypeError`` (#525), wrapping a DoRA base ``TypeError`` (#523), the
   surgery primitives ``ValueError`` (#524), ``magnitude_prune``
   ``ValueError`` (#554), and the training preflight (a recipe whose
   checkpoints could not rebuild the net) ``ValueError``. A new caller
   follows the API family it belongs to.
   ```

3. In `src/nnx/interop/gguf/tensor_name_map.py`, change the import (L33) to `from ..._reparameterization import UNNAMED, legacy_reparameterized_paths, qualified_name, weight_reparameterization`, and in `_refuse_reparameterized` replace L109-L115 (the two `list[str] = []` lines and the loop) with:

   ```python
       masked_at, normed_at = legacy_reparameterized_paths(net)
       # The tensor's qualified name, or the module's for a hook without one.
       masked = [(path or "<root>") if name == UNNAMED else qualified_name(path, name) for path, name in masked_at]
       normed = [(path or "<root>") if name == UNNAMED else qualified_name(path, name) for path, name in normed_at]
   ```

4. In `src/nnx/nn/nn_model.py`, change the import (L36) to `from .._reparameterization import UNNAMED, legacy_reparameterized_paths, qualified_name`, and replace the loop of `_reparameterizing_hooks` (L1177-L1186) with:

   ```python
       masked_at, normed_at = legacy_reparameterized_paths(net)
       for pairs, found, own in ((masked_at, masked, ("_mask",)), (normed_at, normed, ("_g", "_orig"))):
           for path, name in pairs:
               if name == UNNAMED:
                   continue
               key = qualified_name(path, name)
               if not any(f"{key}{suffix}" in base_keys for suffix in own):
                   found.add(key)
   ```

   Keep `masked: set[str] = set()`, `normed: set[str] = set()` and the `return masked, normed`.
5. Run `uv run ruff check --fix src/nnx/` and `uv run ruff format src/nnx/`.

The traversal order, the qualification and each caller's handling of unnamed hooks are unchanged, so every message is the same. No test changes, no CHANGELOG entry. `_reparameterization.py`'s docstring is internal (not in `docs/api.md`).

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need (`gguf` for the export tests).

## Acceptance criteria

- [ ] Both walkers call the shared helper. Check: `git grep -n "legacy_reparameterized_paths(" src/nnx` returns 3 lines (the definition and the 2 calls)
- [ ] Neither caller walks `named_modules()` for hooks any more. Check: `git grep -n "legacy_reparameterizations(" src/nnx/interop src/nnx/nn` returns 0 lines
- [ ] A new test `tests/test_surgery_low_rank.py::test_legacy_reparameterized_paths_lists_module_and_tensor` builds `nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))`, applies `torch.nn.utils.prune.l1_unstructured(net[0], "weight", 0.5)` and `torch.nn.utils.weight_norm(net[1])` inside `with warnings.catch_warnings(): warnings.simplefilter("ignore")` (the legacy norm's deprecation notice, as `tests/test_interop_gguf_writer.py` L637 does), and asserts `legacy_reparameterized_paths(net) == ([("0", "weight")], [("1", "weight")])`. Check: `uv run pytest tests/test_surgery_low_rank.py::test_legacy_reparameterized_paths_lists_module_and_tensor`
- [ ] The refusal tests pass unchanged, including the GGUF `_refuse_reparameterized` test (`tests/test_interop_gguf_writer.py` L819-L826). Check: `uv run pytest tests/test_interop_gguf_writer.py tests/test_peft_dora.py tests/test_prune_magnitude.py tests/test_model_recipes.py tests/test_surgery_low_rank.py tests/test_surgery_widen.py tests/test_surgery_deepen.py tests/test_surgery_embedding.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [tensor_name_map.py L98-L115](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/interop/gguf/tensor_name_map.py#L98-L115) | `_refuse_reparameterized`: walk one, `TypeError` |
| [nn_model.py L1170-L1187](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1170-L1187) | `_reparameterizing_hooks`: walk two |
| [_reparameterization.py L1-L38](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_reparameterization.py#L1-L38) | the module docstring and `legacy_reparameterizations` |
| [surgery/_utils.py L116-L138](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/surgery/_utils.py#L116-L138) | surgery's `ValueError` |
| [magnitude.py L117-L131](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/prune/magnitude.py#L117-L131) | `magnitude_prune`'s `ValueError` |
| [dora.py L112-L131](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/peft/dora.py#L112-L131), [L222-L250](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/peft/dora.py#L222-L250) | DoRA's reasons and its `TypeError` |
| #523, #524, #525, #554 | the four decisions on these refusals' types |

## Scope

**In:** the shared walk, `qualified_name`, the two callers and the docstring record.

**Out:** changing any error type (deliberate per the four issues). The single-module checks of surgery, DoRA and `magnitude_prune`, which read one module's hooks and need no walk.

## Dependencies

**Blocked by:** none
**Blocks:** none
**Related:** #523, #524, #525, #554
