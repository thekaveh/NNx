---
id: FEAT-124
title: "[FEAT-124] Share one reshaped-key check between PEFT and finetune loading"
labels: ["enhancement", "area: peft", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-47"]
depends_on: []
related_issues: [528]
status: READY
---

**Category:** PEFT · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-124: the PEFT adapter loaders (`peft/_source._refuse_shape_mismatch`) and `nnx.finetune.load_pretrained` each find "keys both sides hold with different shapes" with their own loop, and `finetune/loading.py` imports the private `models._shape` to do it. This ticket adds `nnx.models.reshaped_keys` (the detection), renames `models._shape` to `models.state_shape` with an alias, and makes both loaders call the shared detection. Each keeps its own message.

## Context

**What happens today.** PEFT (`src/nnx/peft/_source.py`):

```python
    current = module.state_dict()
    mismatched = [
        f"{key}: {tuple(value.shape)} vs {tuple(current[key].shape)}"
        for key, value in sd.items()
        if key in current and hasattr(value, "shape") and tuple(value.shape) != tuple(current[key].shape)
    ]
```

finetune (`src/nnx/finetune/loading.py`):

```python
    mismatched = [
        f"{key} {source_shape} -> {target_shape}"
        for key in tensor_keys
        for source_shape, target_shape in ((_shape(remapped[key]), _shape(target_state[key])),)
        if source_shape is not None and target_shape is not None and source_shape != target_shape
    ]
```

**Why it matters.** The rule "a key both sides hold, both with a shape, shapes differ" is written twice, and a public loader depends on a private helper of another module.

**History.** The finding is narrowed: PEFT checks only the adapter keys it owns (`lora_A`, `scaling`, `prefix_keys.<i>`, …), which are never lazy, so the "peft ignores lazy parameters" problem is unreachable. Only the de-duplication is left. #528 (FIX-061) made `load_pretrained` load into uninitialized lazy modules again; its lazy rules (an uninitialized tensor matches any shape) stay exactly as they are.

**Background for the worker.**
- Lazy parameter: `nn.LazyLinear`'s weight before its first forward, an `UninitializedParameter` with no shape. `models._shape(value)` returns `None` for one, else `tuple(value.shape)`.
- `_refuse_shape_mismatch(module, sd, fn_name)`: runs before `load_state_dict(strict=False)`, which copies every matching tensor first and raises at the end, leaving a half-loaded module.
- Alias rule: a renamed private helper keeps its old name importable (`_shape = state_shape`).

**Proposal.**
1. In `src/nnx/models.py`:
   - Rename `def _shape(` (L613) to `def state_shape(`, keeping its body and docstring. Directly after the function add:

     ```python
     _shape = state_shape  # the old private name


     def reshaped_keys(
         source: Mapping[str, Any], target: Mapping[str, Any], keys: Iterable[str]
     ) -> list[tuple[str, tuple[int, ...], tuple[int, ...]]]:
         """``(key, source shape, target shape)`` for each of ``keys`` whose
         source and target values both have a shape (:func:`state_shape`: an
         uninitialized lazy tensor has none and matches any) and the shapes
         differ, in ``keys`` order. Loaders call it before ``load_state_dict``,
         which would copy every other tensor before raising."""
         found: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
         for key in keys:
             source_shape, target_shape = state_shape(source[key]), state_shape(target[key])
             if source_shape is not None and target_shape is not None and source_shape != target_shape:
                 found.append((key, source_shape, target_shape))
         return found
     ```

   - Replace the five remaining calls `_shape(` in `models.py` with `state_shape(` (L298, L449, L452 — two calls on that line count as one site, L544, L619).
   - Leave `__all__` unchanged.
2. In `src/nnx/peft/_source.py`, add `from ..models import reshaped_keys` after `import torch`, and replace the `mismatched = [...]` statement of `_refuse_shape_mismatch` (L70-L74) with:

   ```python
       keys = [key for key, value in sd.items() if key in current and hasattr(value, "shape")]
       mismatched = [f"{key}: {source} vs {target}" for key, source, target in reshaped_keys(sd, current, keys)]
   ```

   The key filter is today's (`key in current and hasattr(value, "shape")`), the order is `sd`'s, and the text per key is identical.
3. In `src/nnx/finetune/loading.py`:
   - Replace `from ..models import _shape` (L23) with `from ..models import reshaped_keys, state_shape`.
   - Replace the `mismatched = [...]` statement (L152-L157) with:

     ```python
         mismatched = [
             f"{key} {source_shape} -> {target_shape}"
             for key, source_shape, target_shape in reshaped_keys(remapped, target_state, tensor_keys)
         ]
     ```

   - Replace `_shape(` with `state_shape(` at L167 and L179.
4. Run `uv run ruff check --fix src/nnx/` and `uv run ruff format src/nnx/`.

`models` imports neither `peft` nor `finetune`, so the new imports are cycle-free. No message, order or refusal changes, so no test edits and no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] `nnx.models` defines `state_shape` and `reshaped_keys`, and `_shape is state_shape`. Check: `uv run python -c "from nnx import models; assert models._shape is models.state_shape and callable(models.reshaped_keys)"`
- [ ] Nothing calls or imports `_shape` any more (the alias stays for outside callers). Check: `grep -rnE "(^|[^a-z_])_shape\(|import _shape" src/nnx` returns 0 lines
- [ ] A new test `tests/test_finetune_loading.py::test_reshaped_keys_skips_lazy_and_keeps_key_order` asserts `reshaped_keys({"a": torch.zeros(2), "b": torch.zeros(3), "c": torch.zeros(1)}, {"a": torch.zeros(4), "b": torch.zeros(3), "c": nn.LazyLinear(2).weight}, ["c", "a", "b"]) == [("a", (2,), (4,))]` (it imports `from nnx.models import reshaped_keys` in its body). Check: `uv run pytest tests/test_finetune_loading.py::test_reshaped_keys_skips_lazy_and_keeps_key_order`
- [ ] The pinned refusals pass unchanged: `tests/test_peft_lora.py` L551 ("size mismatch") and `tests/test_finetune_loading.py` L196 and L284 ("different shapes"). Check: `uv run pytest tests -k "peft or finetune"`
- [ ] The lazy-module loading tests pass. Check: `uv run pytest tests/test_external_module_runtime.py tests/test_finetune_loading.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [_source.py L64-L76](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/peft/_source.py#L64-L76) | `_refuse_shape_mismatch`: PEFT's own reshaped-key loop |
| [loading.py L23](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/finetune/loading.py#L23) | `load_pretrained`'s module imports the private `models._shape` |
| [loading.py L146-L163](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/finetune/loading.py#L146-L163) | finetune's reshaped-key loop and its message |
| [models.py L613-L615](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/models.py#L613-L615) | `_shape` |
| [test_peft_lora.py L551](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_peft_lora.py#L551) | pins "size mismatch" |
| [test_finetune_loading.py L196](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_finetune_loading.py#L196), [L284](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_finetune_loading.py#L284) | pin "different shapes" |
| #528 | `load_pretrained` into uninitialized lazy modules |

## Scope

**In:** `state_shape`, `reshaped_keys`, and the two loaders' detection.

**Out:** one shared refusal message (the two loaders word it differently; changing either is a message change). `models.check_state_schema` (a different check: missing, unexpected and reshaped names together). The four PEFT save / load bodies (FEAT-125).

## Dependencies

**Blocked by:** none
**Blocks:** FEAT-125
**Related:** #528
