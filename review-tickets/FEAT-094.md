---
id: FEAT-094
title: "[FEAT-094] Share the dry-run lazy-inference rules in nnx.models"
labels: ["enhancement", "area: model-integration", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-14"]
depends_on: []
related_issues: [562, 528]
status: READY
---

**Category:** Model integration · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-094: Two functions in `nnx/models.py` encode the same three rules for what a lazy layer's dry-run forward infers from its weights (`in_features`, `in_channels`, `num_features`). This ticket moves the rules into one helper, `_dry_run_value`, that both call. Results do not change.

## Context

**What happens today.** `_inferred_attributes` (`models.py` L421–L440), for a pending lazy layer and the shapes its saved weights give:

```python
    weight = shapes.get("weight")
    if isinstance(sub, nn.LazyLinear):
        return None if weight is None else {"in_features": weight[-1]}
    if _LAZY_CONV is not None and isinstance(sub, _LAZY_CONV):
        if weight is None:
            return None
        return {"in_channels": weight[0] if sub.transposed else weight[1] * cast(int, sub.groups)}
    if _LAZY_NORM is not None and isinstance(sub, _LAZY_NORM):
        held = weight if weight is not None else shapes.get("running_mean")
        # Without affine weights or running statistics nothing materializes:
        # the dry run leaves num_features at 0 as well.
        return {} if held is None else {"num_features": held[0]}
    return None
```

`_left_by_a_loaded_forward` (L493–L520), for a layer that materialized from loaded weights, repeats them:

```python
    if key == "num_features":
        shape = _held_shape(layer, "weight") or _held_shape(layer, "running_mean")
        value = shape[0] if shape else 0
    else:
        shape = _held_shape(layer, "weight")
        if shape is None:
            value = 0
        elif key == "in_features":
            value = shape[-1]
        else:
            value = shape[0] if bool(layer.transposed) else shape[1] * cast(int, layer.groups)
    return {key: value} if value else {}
```

**Why it matters.** A fix to one rule (a new conv layout, say) would have to be made twice; a missed copy makes a reloaded module describe itself differently from a dry-run one, which changes its run id.

**History.** Both functions came with #562 (a lazy module's descriptor stays stable across its first forward). #528 fixed `load_pretrained` for uninitialized lazy modules; it does not use these helpers.

**Background for the worker.**
- Lazy layer: `nn.LazyLinear`, `nn.LazyConv*d`, `nn.LazyBatchNorm*d` and similar, which infer their input size at the first forward ("dry run") and then become the plain layer class.
- `_LAZY_CONV` / `_LAZY_NORM` (L411–L412): torch's private lazy bases; `_CONV` / `_NORM` (L476–L477): the plain bases the layers become.
- `_held_shape(layer, name)` (L480): the shape of the layer's own tensor `name`, or `None` when it has none or it is uninitialized.
- The three rules: `in_features = weight.shape[-1]`; `in_channels = weight.shape[0]` for a transposed conv, else `weight.shape[1] * groups`; `num_features = weight.shape[0]`, or `running_mean.shape[0]` without affine weights.
- The only inputs on which the two copies differ today are 0-d tensors (shape `()`): `_left_by_a_loaded_forward` tests a shape with `or` / `if shape`, which treats `()` as absent. No torch Linear, Conv or Norm layer holds a 0-d weight or running mean (their weights are 1-D or more), so the shared helper uses `is None`, as `_inferred_attributes` does.

**Proposal.**
1. In `src/nnx/models.py`, directly above `_inferred_attributes` (L421), add:

   ```python
   def _dry_run_value(
       kind: str, shape_of: Callable[[str], Optional[tuple[int, ...]]], layer: nn.Module
   ) -> Optional[int]:
       """What a dry-run forward sets as ``kind`` (``"in_features"``,
       ``"in_channels"`` or ``"num_features"``) on a lazy ``layer``, read from
       its tensor shapes (``shape_of(name)``: ``None`` when the layer holds no
       such tensor): ``weight[-1]``; ``weight[0]`` for a transposed conv, else
       ``weight[1] * groups``; ``weight[0]``, else ``running_mean[0]``. ``None``
       when the tensor that gives it is absent."""
       weight = shape_of("weight")
       if kind == "num_features":
           held = weight if weight is not None else shape_of("running_mean")
           return None if held is None else held[0]
       if weight is None:
           return None
       if kind == "in_features":
           return weight[-1]
       return weight[0] if bool(layer.transposed) else weight[1] * cast(int, layer.groups)
   ```

   `Callable` is already imported from `collections.abc` (L43).
2. Replace the body of `_inferred_attributes` (L425–L440, after its docstring) with:

   ```python
       if isinstance(sub, nn.LazyLinear):
           kind = "in_features"
       elif _LAZY_CONV is not None and isinstance(sub, _LAZY_CONV):
           kind = "in_channels"
       elif _LAZY_NORM is not None and isinstance(sub, _LAZY_NORM):
           value = _dry_run_value("num_features", shapes.get, sub)
           # Without affine weights or running statistics nothing materializes:
           # the dry run leaves num_features at 0 as well.
           return {} if value is None else {"num_features": value}
       else:
           return None
       value = _dry_run_value(kind, shapes.get, sub)
       return None if value is None else {kind: value}
   ```

3. In `_left_by_a_loaded_forward`, replace the block from `if key == "num_features":` to `value = shape[0] if bool(layer.transposed) else shape[1] * cast(int, layer.groups)` (L509–L519) with:

   ```python
       value = _dry_run_value(key, lambda name: _held_shape(layer, name), layer)
   ```

   Keep the line after it, `return {key: value} if value else {}` (L520): a `None` value returns `{}`, as the old `value = 0` did.
4. No test patches either function (`grep -rn "_inferred_attributes\|_left_by_a_loaded_forward\|_dry_run_value" tests` returns nothing), and no message or behaviour changes, so there is no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] Each rule appears once in `models.py`. Check: `grep -c "groups)" src/nnx/models.py` prints `1`, and `grep -c '"running_mean"' src/nnx/models.py` prints `1`
- [ ] Both functions call the helper. Check: `grep -c "_dry_run_value(" src/nnx/models.py` prints `4` (the definition and three calls)
- [ ] The lazy-module and loading suites pass unchanged. Check: `uv run pytest tests/test_external_module_runtime.py tests/test_finetune_loading.py tests/test_model_factories.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [models.py L410-L418](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/models.py#L410-L418) | `_LAZY_CONV`, `_LAZY_NORM`, `_known_lazy` |
| [models.py L421-L440](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/models.py#L421-L440) | `_inferred_attributes`: the rules, for a pending layer |
| [models.py L476-L490](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/models.py#L476-L490) | `_CONV`, `_NORM`, `_held_shape` |
| [models.py L493-L520](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/models.py#L493-L520) | `_left_by_a_loaded_forward`: the same rules, for a materialized layer |
| [models.py L559-L569](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/models.py#L559-L569) | `_materialization_plan`, the caller of `_inferred_attributes` |
| #562 | the lazy-descriptor rules both functions serve |
| #528 | `load_pretrained` into uninitialized lazy modules |

## Scope

**In:** `_dry_run_value` and its two callers.

**Out:** the lazy-module tracking in `NNModel` (FEAT-093). Moving the lazy-module descriptor into `nnx.models` (FEAT-078). Custom lazy layers torch does not ship (still refused as non-emulable).

## Dependencies

**Blocked by:** none
**Blocks:** FEAT-078
**Related:** #562, #528
