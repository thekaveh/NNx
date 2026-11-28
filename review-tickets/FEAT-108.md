---
id: FEAT-108
title: "[FEAT-108] Replay transforms on a bare module in the Hub preflight"
labels: ["enhancement", "area: model-transforms", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-30"]
depends_on: ["FEAT-091"]
related_issues: []
status: READY
---

**Category:** Model transforms · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-108: The Hub shape preflight replays a checkpoint's recorded transforms on a meta-device module by dressing the module up as a fake `NNModel` (`SimpleNamespace(net=module, _topology_transforms=())`), because the replay helpers take a model. This ticket gives the replay a module-level entry point, `_replay_transforms_on(net, transforms)`, backed by `_replay_checkpoint_transform` in `nn_model.py` and `_replay_on` in `nnx.transforms`, and passes the module directly. `_replay_transforms(model, ...)` and `transforms._replay(model, ...)` keep their signatures. Behaviour does not change.

## Context

**What happens today.** `_declared_module` (`nn_model.py` L1316–L1318):

```python
    try:
        with torch.device("meta"), _global_rng_kept():
            module = build()
            _replay_transforms(cast(NNModel, SimpleNamespace(net=module, _topology_transforms=())), transforms)
```

The replay chain reads two attributes of the "model": `_replay_transforms` (L1140–L1150) calls `_apply_checkpoint_transform(model, transform)` (L1089), which calls `transforms._replay(model, transform)` (`transforms.py` L630–L640):

```python
    op = TransformOp.from_checkpoint_transform(transform)
    _validate(model.net, (op,), _recorded_operations(model), (), indexed=False, replay=True)
    for path in op.targets:
        set_module(model.net, path, _build(op, get_module(model.net, path), allocate_only=True))
    model._topology_transforms = (*model._topology_transforms, op.checkpoint_transform())
```

or prepares and converts `model.net` for the QAT conversion. Only `net` and `_topology_transforms` are ever read.

**Why it matters.** A `cast` to `NNModel` hides from the type checker that the object is not one; any future replay step that reads another model attribute would fail only in the preflight, where `_declared_module` swallows every exception and silently skips the check.

**History.** No board issue covers the stand-in; the preflight came with ed93d51 (#855). FEAT-091 adds `_is_qat_conversion` and moves `_unportable_transform`, which this ticket builds on.

**Background for the worker.**
- Recorded transforms: the `NNCheckpointTransform`s a checkpoint or Hub config lists; replaying them rebuilds the trained topology (LoRA wrappers, low-rank factors, a QAT conversion) on a fresh base before the weights load.
- `_recorded_operations(model)` (`transforms.py` L404): the recipe operations already recorded on a model (`model._topology_transforms`); a recorded non-recipe transform there makes a new recipe operation a `RecipeError`.
- Callers of the replay: `from_checkpoint` (L4537) and `_from_pretrained` (L4965) call `_replay_transforms(model, transforms)` and then overwrite `model._topology_transforms = _canonical_transforms(transforms)` on the next line; `_declared_module` discards its module on any exception. So whether `_topology_transforms` is updated after each transform or once at the end cannot be observed; this ticket updates it once, at the end.
- Tests call `transforms._replay(model, recorded)` directly (`tests/test_model_recipes.py` L532, L672); it keeps its signature.

**Proposal.**
1. In `src/nnx/transforms.py`:
   - split `_recorded_operations` (L404–L415): rename its body into a new function placed directly after it, and make `_recorded_operations` delegate:

     ```python
     def _recorded_operations(model: Any) -> tuple[TransformOp, ...]:
         """The recipe operations already recorded on a model; any other
         recorded transform (a train-end QAT conversion) refuses a recipe."""
         return _recorded_operations_in(getattr(model, "_topology_transforms", ()))


     def _recorded_operations_in(transforms: Iterable[NNCheckpointTransform]) -> tuple[TransformOp, ...]:
         """:func:`_recorded_operations` of the recorded ``transforms`` themselves."""
         recorded = tuple(transforms)
     ```

     followed by the rest of today's body unchanged (from `foreign = [...]` to the `return tuple(TransformOp.from_checkpoint_transform(t) for t in recorded)`);
   - replace `_replay` (L630–L640) with:

     ```python
     def _replay_on(
         net: nn.Module, transform: NNCheckpointTransform, recorded: Iterable[NNCheckpointTransform] = ()
     ) -> NNCheckpointTransform:
         """Rebuild one recorded operation's topology on ``net`` (a freshly built
         base, with the transforms ``recorded`` replayed on it already) before its
         saved tensors are loaded; return the operation's canonical transform for
         the caller to record."""
         from .surgery._utils import get_module, set_module

         op = TransformOp.from_checkpoint_transform(transform)
         _validate(net, (op,), _recorded_operations_in(recorded), (), indexed=False, replay=True)
         for path in op.targets:
             set_module(net, path, _build(op, get_module(net, path), allocate_only=True))
         return op.checkpoint_transform()


     def _replay(model: NNModel, transform: NNCheckpointTransform) -> None:
         """Rebuild one recorded operation's topology on a freshly built base
         before its saved tensors are loaded (``from_checkpoint`` /
         ``from_pretrained``)."""
         replayed = _replay_on(model.net, transform, getattr(model, "_topology_transforms", ()))
         model._topology_transforms = (*model._topology_transforms, replayed)
     ```

2. In `src/nnx/nn/nn_model.py`:
   - add `_replay_on` to the top-level `from ..transforms import ...` (L102);
   - replace `_apply_checkpoint_transform` (L1089–L1115) with `_replay_checkpoint_transform`, the same body on a module:

     ```python
     def _replay_checkpoint_transform(
         net: torch.nn.Module, transform: NNCheckpointTransform, recorded: Sequence[NNCheckpointTransform]
     ) -> Optional[NNCheckpointTransform]:
         """Rebuild one recorded transform's topology on ``net``, after the
         transforms ``recorded`` (already replayed on it): a recipe operation
         (FEAT-016), whose canonical form is returned for the caller to record,
         or the torchao QAT conversion, which records nothing here (``None``)."""
         if _replayable(transform):
             # FEAT-016: a recorded recipe operation — its topology is rebuilt
             # (low-rank factors allocated, never re-factorized) before the
             # saved tensors are loaded.
             return _replay_on(net, transform, recorded)
         if _is_qat_conversion(transform):
     ```

     followed by today's QAT branch with `model.net` replaced by `net` (`quantizer.prepare(net)`, `quantizer.convert(net)`) and `return None` instead of `return`, then today's final `raise ValueError(...)` unchanged;
   - replace `_replay_transforms` (L1140–L1150) with:

     ```python
     def _replay_transforms_on(
         net: torch.nn.Module,
         transforms: Sequence[NNCheckpointTransform],
         recorded: Sequence[NNCheckpointTransform] = (),
     ) -> tuple[NNCheckpointTransform, ...]:
         """Replay a checkpoint's recorded transforms on ``net`` in order —
         before any saved tensor is loaded — naming the transform that cannot be
         replayed; return ``recorded`` followed by the recipe operations
         replayed, in canonical form."""
         replayed = tuple(recorded)
         for index, transform in enumerate(transforms):
             try:
                 canonical = _replay_checkpoint_transform(net, transform, replayed)
             except (TypeError, ValueError) as error:
                 raise ValueError(
                     f"topology transform {index} ({transform.name!r} version {transform.version}) cannot be replayed: "
                     f"{error}"
                 ) from error
             if canonical is not None:
                 replayed = (*replayed, canonical)
         return replayed


     def _replay_transforms(model: NNModel, transforms: Sequence[NNCheckpointTransform]) -> None:
         """Replay a checkpoint's recorded transforms on ``model.net`` (see
         :func:`_replay_transforms_on`) and record the recipe operations."""
         model._topology_transforms = _replay_transforms_on(model.net, transforms, model._topology_transforms)
     ```

   - in `_declared_module`, change L1318 to `_replay_transforms_on(module, transforms)`, and change `from types import MappingProxyType, SimpleNamespace` (L21) to `from types import MappingProxyType` (the stand-in was its only use).
3. Add to `tests/test_model_recipes.py`, after `test_rebuilt_low_rank_factors_have_defined_values` (L671):

   ```python
   def test_recorded_transforms_replay_on_a_bare_module():
       from nnx.nn.nn_model import _replay_transforms_on
       from nnx.transforms import _canonical_transforms

       recorded = TransformRecipe([lora("layers.0", r=4, alpha=8.0)]).checkpoint_transforms()
       net = _model().net
       assert _replay_transforms_on(net, recorded) == _canonical_transforms(recorded)
       assert type(net.layers[0]).__name__ == "LoRALinear"
       with pytest.raises(ValueError, match=r"topology transform 0 \('prune' version 1\) cannot be replayed"):
           _replay_transforms_on(_model().net, (NNCheckpointTransform(name="prune"),))
   ```

4. No message changes. No test refers to `_apply_checkpoint_transform` (`grep -rn "_apply_checkpoint_transform" tests` returns nothing), so it is removed without an alias. No behaviour changes, so there is no CHANGELOG entry.

**Worker setup.** Branch from `develop` after FEAT-091 has merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The new test passes. Check: `uv run pytest tests/test_model_recipes.py::test_recorded_transforms_replay_on_a_bare_module`
- [ ] No stand-in model remains. Check: `grep -n "SimpleNamespace" src/nnx/nn/nn_model.py` returns 0 lines
- [ ] The QAT and recipe replays act on a module. Check: `grep -n "model.net" src/nnx/nn/nn_model.py | grep -n "quantizer\."` returns 0 lines, and `grep -c "_replay_on(" src/nnx/transforms.py` prints `2` (definition and `_replay`)
- [ ] The replay, preflight and QAT suites pass unchanged. Check: `uv run pytest tests/test_model_recipes.py tests/test_hub_load_preflight.py tests/test_hub_mixin.py tests/test_quantize_qat.py tests/test_checkpoint_safetensors.py tests/test_train_end_checkpoint.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [nn_model.py L1293-L1321](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1293-L1321) | `_declared_module`: the `SimpleNamespace` stand-in at L1318 |
| [nn_model.py L1089-L1115](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1089-L1115) | `_apply_checkpoint_transform`: reads only `model.net` |
| [nn_model.py L1140-L1150](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1140-L1150) | `_replay_transforms` |
| [nn_model.py L4537-L4538](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L4537-L4538), [L4965-L4966](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L4965-L4966) | both real callers overwrite `_topology_transforms` right after |
| [transforms.py L404-L415](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/transforms.py#L404-L415) | `_recorded_operations` |
| [transforms.py L630-L640](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/transforms.py#L630-L640) | `_replay`: reads `model.net` and `model._topology_transforms` |
| [test_model_recipes.py L531-L539](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_model_recipes.py#L531-L539), [L671-L677](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_model_recipes.py#L671-L677) | tests that call `transforms._replay(model, ...)` |
| ed93d51 (#855) | the commit that added the preflight |

## Scope

**In:** `_replay_on`, `_recorded_operations_in`, `_replay_checkpoint_transform`, `_replay_transforms_on`, the `_declared_module` call, one test.

**Out:** moving the replay helpers into `nnx.transforms` (FEAT-075, which moves them as this ticket leaves them). Building a registered factory once per `from_pretrained` (FIX-112). Hoisting `nn_model.py`'s other local imports (FEAT-095).

## Dependencies

**Blocked by:** FEAT-091
**Blocks:** FEAT-075, FEAT-095
**Related:** none
