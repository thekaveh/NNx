---
id: FEAT-091
title: "[FEAT-091] Move _unportable_transform into nnx.transforms and name the QAT transform once"
labels: ["enhancement", "area: model-transforms", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-11"]
depends_on: []
related_issues: [509, 767]
status: READY
---

**Category:** Model transforms · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-091: `_unportable_transform` lives in `nn_model.py`, so `nn_checkpoint` and `bundles` import the 6,800-line module privately just to call it, and the QAT transform's identity (`"torchao_qat"`, version 1) is written as a literal in three places. This ticket moves the function, unchanged, into `nnx/transforms.py` (keeping the old name importable from `nn_model`) and names the QAT transform once with `QAT_TRANSFORM` and `_is_qat_conversion`. Behaviour does not change.

## Context

**What happens today.** The QAT identity is tested twice in `nn_model.py`. In `_apply_checkpoint_transform` (L1098):

```python
    if transform.name == "torchao_qat" and transform.version == 1:
        from ..quantize.qat import _build_quantizer
```

and in `_unportable_transform` (L1124–L1136):

```python
    from ..transforms import _VERSIONS, RecipeError, TransformOp

    for index, transform in enumerate(transforms):
        where = f"topology transform {index} ({transform.name!r} version {transform.version})"
        if _replayable(transform):
            if transform.version not in _VERSIONS[transform.name]:
                return f"{where} has a version this NNx does not know"
            try:
                TransformOp.from_checkpoint_transform(transform)
            except RecipeError as error:
                return f"{where} has malformed options: {error}"
        elif not (transform.name == "torchao_qat" and transform.version == 1):
            return f"{where} is not an operation NNx can replay; only its producer can rebuild it"
```

The producer writes the same pair in `QATLifecycleCallback.checkpoint_transforms` (`quantize/qat.py` L382–L386):

```python
            NNCheckpointTransform(
                name="torchao_qat",
                version=1,
                options={"qat_config": self.qat_config, "groupsize": self.groupsize},
            ),
```

Two modules reach into `nn_model` for the function: `NNCheckpoint._require_replayable` (`nn_checkpoint.py` L628, `from ..nn_model import _unportable_transform`) and `export_bundle` (`bundles.py` L1206, `from .nn.nn_model import _resume_checkpoint_type, _unportable_transform`).

**Why it matters.** The function needs only names that `nnx.transforms` already owns (`_replayable`, `_VERSIONS`, `TransformOp`, `RecipeError`); its home in `nn_model` forces two lazy imports of the whole training module. Three copies of the QAT identity can drift: a QAT version 2 would have to be changed in three places.

**History.** #509 made `save_pretrained` and `push_to_hub` refuse unreplayable transforms through `_unportable_transform`. #767 (DEC-013) chose option C for `NNCheckpoint.to_file(format="safetensors")`: "Refuse before writing, as `save_pretrained` does, and advise `format="pickle"`." That refusal is `_require_replayable`, which runs before any directory is created (`to_file` L603–L605). It must stay before anything is written; this ticket only changes where the function is imported from.

**Background for the worker.**
- Checkpoint transform (`NNCheckpointTransform`): a recorded `(name, version, options)` step that rebuilds a checkpoint's module topology before its weights load.
- Recipe operation: a transform from `nnx.transforms` (`"lora"`, `"low_rank"`), replayable from data; `_replayable(t)` is `t.name in _VERSIONS`.
- QAT conversion: the torchao quantization-aware-training conversion `QATLifecycleCallback` records; NNx replays it too.
- Unportable transform: anything else (a callback's own train-end transform); safetensors checkpoints, Hub saves and bundles refuse it.
- `nnx/transforms.py` has no top-level NNx imports (only `torch.nn`), so any module can import it at top level without a cycle.

**Proposal.**
1. In `src/nnx/transforms.py`, after `_VERSIONS = {LORA: (1,), LOW_RANK: (1,)}` (L75), add:

   ```python
   # The torchao QAT conversion QATLifecycleCallback records (nnx.quantize.qat): not a recipe
   # operation, but replayed by NNModel.from_checkpoint / from_pretrained as well.
   QAT_TRANSFORM = ("torchao_qat", 1)
   ```

   and after `_replayable` (L612–L613), add:

   ```python
   def _is_qat_conversion(transform: NNCheckpointTransform) -> bool:
       """Whether ``transform`` is the torchao QAT conversion (:data:`QAT_TRANSFORM`)."""
       return (transform.name, transform.version) == QAT_TRANSFORM
   ```

   `QAT_TRANSFORM` is not added to `__all__`, like `LORA` and `LOW_RANK`.
2. Cut `_unportable_transform` (`nn_model.py` L1118–L1137) and paste it into `src/nnx/transforms.py` directly after `_is_qat_conversion`, with two edits: delete its first line `from ..transforms import _VERSIONS, RecipeError, TransformOp` (these names are in the same module now), and replace `elif not (transform.name == "torchao_qat" and transform.version == 1):` with `elif not _is_qat_conversion(transform):`. Keep the docstring and every message byte for byte.
3. In `src/nnx/nn/nn_model.py`:
   - Change the import at L102 to `from ..transforms import _canonical_transforms, _is_qat_conversion, _recipe_transforms, _replayable, _snapshot_transforms, _state_shapes, _unportable_transform`. This also keeps `nnx.nn.nn_model._unportable_transform` importable (the old name, for any caller outside the package).
   - In `_apply_checkpoint_transform`, replace `if transform.name == "torchao_qat" and transform.version == 1:` (L1098) with `if _is_qat_conversion(transform):`. Leave the error text `invalid torchao_qat checkpoint transform: ...` (L1105) as it is.
4. In `src/nnx/nn/params/nn_checkpoint.py`, `_require_replayable` (L625): replace `from ..nn_model import _unportable_transform` with `from ...transforms import _unportable_transform` (keep it a local import).
5. In `src/nnx/bundles.py`, `export_bundle` (L1206): replace `from .nn.nn_model import _resume_checkpoint_type, _unportable_transform` with the two lines `from .nn.nn_model import _resume_checkpoint_type` and `from .transforms import _unportable_transform`.
6. In `src/nnx/quantize/qat.py`, add `from ..transforms import QAT_TRANSFORM` to the first-party top-level imports (after `from ..peft.dora import DoRALinear`, L49), and in `checkpoint_transforms` replace `name="torchao_qat",` / `version=1,` with `name=QAT_TRANSFORM[0],` / `version=QAT_TRANSFORM[1],`.
7. Run `uv run ruff check --fix src/nnx/transforms.py src/nnx/nn/nn_model.py src/nnx/quantize/qat.py` so the import lists are sorted, then `uv run ruff format` on the same files.
8. No test monkeypatches `_unportable_transform` or `_apply_checkpoint_transform` (`grep -rn "_unportable_transform\|_apply_checkpoint_transform" tests` returns nothing), so no test is retargeted. No message changes, so there is no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] `_unportable_transform` is defined once, in `nnx/transforms.py`. Check: `grep -rn "def _unportable_transform" src/nnx` returns 1 line, in `src/nnx/transforms.py`
- [ ] The old import path still works. Check: `uv run python -c "from nnx.nn.nn_model import _unportable_transform; from nnx.transforms import _unportable_transform as t; assert _unportable_transform is t"`
- [ ] The QAT identity literal appears only in `QAT_TRANSFORM` (the error text in `_apply_checkpoint_transform` does not compare). Check: `grep -rn '"torchao_qat"' src/nnx` returns 1 line, in `src/nnx/transforms.py`
- [ ] Neither `nn_checkpoint` nor `bundles` imports `_unportable_transform` from `nn_model`. Check: `grep -rn "nn_model import.*_unportable_transform" src/nnx` returns 0 lines
- [ ] #767's refusal still runs before anything is written. Check: `uv run pytest tests/test_checkpoint_safetensors.py`
- [ ] The transform suites pass unchanged. Check: `uv run pytest tests/test_model_recipes.py tests/test_checkpoint_safetensors.py tests/test_run_bundles.py tests/test_bundle_validation.py tests/test_quantize_qat.py tests/test_hub_mixin.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [nn_model.py L1089-L1115](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1089-L1115) | `_apply_checkpoint_transform`: the QAT literal at L1098 |
| [nn_model.py L1118-L1137](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1118-L1137) | `_unportable_transform`: imports only `nnx.transforms` names; the QAT literal at L1135 |
| [nn_model.py L102](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L102) | the existing top-level import from `..transforms` |
| [transforms.py L73-L75](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/transforms.py#L73-L75), [L612-L613](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/transforms.py#L612-L613) | `LORA`, `LOW_RANK`, `_VERSIONS`; `_replayable` |
| [nn_checkpoint.py L625-L637](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_checkpoint.py#L625-L637) | `_require_replayable`: the private import of `nn_model` (#767's refusal) |
| [bundles.py L1206](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/bundles.py#L1206), [L1229](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/bundles.py#L1229) | `export_bundle`: the private import and the call |
| [qat.py L374-L387](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/quantize/qat.py#L374-L387) | `QATLifecycleCallback.checkpoint_transforms`: the producer's literal |
| #509 | Hub saves refuse unreplayable transforms through `_unportable_transform` |
| #767 | DEC-013 option C: `to_file(format="safetensors")` refuses before writing |

## Scope

**In:** the move of `_unportable_transform`, `QAT_TRANSFORM` and `_is_qat_conversion`, the four import sites, the QAT producer.

**Out:** a net-only replay entry point for the Hub preflight (FEAT-108). Moving `_apply_checkpoint_transform` and `_replay_transforms` into `nnx.transforms` (FEAT-075). Hoisting `nn_model.py`'s other local imports (FEAT-095).

## Dependencies

**Blocked by:** none
**Blocks:** FEAT-108, FEAT-075, FEAT-095
**Related:** #509, #767
