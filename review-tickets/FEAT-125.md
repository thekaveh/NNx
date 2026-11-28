---
id: FEAT-125
title: "[FEAT-125] Share save and load bodies across the four PEFT adapter families"
labels: ["enhancement", "area: peft", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-48"]
depends_on: ["FIX-119", "FEAT-124"]
related_issues: [201, 224]
status: READY
---

**Category:** PEFT · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); the four adapter modules and `_artifacts.py` re-read at `1402b49` (#859); not run: torch was unavailable in the review environment

## Summary

FEAT-125: the LoRA, IA3, prefix and prompt adapters each repeat the same save body (`atomic_torch_save` the owned subset, return the path) and the same load body (resolve the source, keep the owned keys, refuse a wrong shape, `load_state_dict(strict=False)`, count what landed). `load_prompt_weights` is the one loader that skips the shape refusal. This ticket adds `_save_owned` and `_load_owned` to `peft/_source.py`, makes the eight public functions call them, and thereby gives `load_prompt_weights` the same up-front refusal (a CHANGELOG line). Each family keeps its own ownership rule.

## Context

**What happens today.** `load_lora_weights` (`src/nnx/peft/lora.py`), and the IA3 and prefix copies:

```python
    sd = _resolve_source_to_state_dict(source, "load_lora_weights")
    sd = _lora_keys_only(module, sd)
    _refuse_shape_mismatch(module, sd, "load_lora_weights")
    result = module.load_state_dict(sd, strict=False)
    # strict=False silently drops keys that don't exist on the module
    # (e.g. loading into an un-adapted model) — subtract them so the
    # return value is the number of tensors that actually landed.
    return len(sd) - len(result.unexpected_keys)
```

`load_prompt_weights` (`src/nnx/peft/prompt.py`) has no refusal:

```python
    sd = {k: v for k, v in sd.items() if k in PromptTuner._OWNED_KEYS}
    result = tuner.load_state_dict(sd, strict=False)
```

Every save is `sd = <owned subset>; atomic_torch_save(sd, str(path)); return str(path)`: #859 (1402b49) replaced `torch.save` with `atomic_torch_save` in all four, each importing it from `.._artifacts` at module level (`lora.py` L47, `ia3.py` L36, `prefix.py` L49, `prompt.py` L44 at `1402b49`).

**Why it matters.** Four copies of the load contract (owned keys only, refuse before copying, count what landed) can drift, and one already did: the prompt loader lets torch raise its own size-mismatch error instead of NNx's.

**History.** #201 (FIX-002) made every family select adapter keys by registered ownership, never by key substring; #201's per-family ownership and errors stay, so `_lora_keys_only`, `_ia3_keys_only`, `PrefixTuner._owned_keys()` and `PromptTuner._OWNED_KEYS` stay where they are and are passed in. #224 added the alias refusals of `apply_*_to`, untouched here. FIX-119 adds a `UserWarning` to `save_lora_weights` for DoRA layers; that warning block stays first in the function.

**Background for the worker.**
- Owned keys: the `state_dict` keys an adapter family's wrappers register (`lora_A` / `lora_B`, `scaling`, `prefix_keys.<i>` / `prefix_values.<i>`, `soft_prompt`); the source is filtered by the destination's owned keys, so a full checkpoint never overwrites a frozen base tensor.
- `atomic_torch_save(obj, path)` (`src/nnx/_artifacts.py`, #859): `torch.save` through a temporary file beside `path` that is renamed onto it only once complete, so a failed save leaves the previous file intact. `tests/test_atomic_torch_saves.py::test_a_failed_lora_save_leaves_the_previous_file_intact` pins it for `save_lora_weights`. The shared save body keeps it.
- `_resolve_source_to_state_dict(source, fn_name)` (`src/nnx/peft/_source.py` L29): a path goes through `torch.load(weights_only=True)`, a dict passes through, anything else is a `TypeError` naming `fn_name`.
- `_refuse_shape_mismatch(module, sd, fn_name)` (L64): raises `RuntimeError("<fn_name>: size mismatch, nothing was loaded: <key>: <source shape> vs <target shape>")` before anything is copied.
- The prompt change: today a wrong-shaped `soft_prompt` raises torch's `RuntimeError` ("Error(s) in loading state_dict for PromptTuner: size mismatch for soft_prompt: …"); afterwards it raises `RuntimeError("load_prompt_weights: size mismatch, nothing was loaded: soft_prompt: (5, 16) vs (4, 16)")`. The type stays `RuntimeError`; with one owned tensor nothing is half-loaded either way, so only the message changes.

**Proposal.**
1. In `src/nnx/peft/_source.py`, add `from collections.abc import Mapping`, `from torch import nn` and `from .._artifacts import atomic_torch_save` to the imports and, after `_refuse_shape_mismatch`, add:

   ```python
   def _save_owned(owned_state: Mapping[str, Any], path: Union[str, Path]) -> str:
       """Save an adapter-only state dict (already restricted to the keys its
       family owns) to ``path`` atomically (:func:`nnx._artifacts.atomic_torch_save`:
       a failed save leaves the previous file intact); return ``str(path)`` so
       calls chain."""
       atomic_torch_save(owned_state, str(path))
       return str(path)


   def _load_owned(module: nn.Module, owned_state: dict, fn_name: str) -> int:
       """Load an adapter-only state dict (already restricted to the keys
       ``module``'s own adapters register) into ``module``: refuse a
       wrong-shaped key before anything is copied, then
       ``load_state_dict(strict=False)``, and return the number of tensors
       that landed."""
       _refuse_shape_mismatch(module, owned_state, fn_name)
       result = module.load_state_dict(owned_state, strict=False)
       # strict=False silently drops keys that don't exist on the module
       # (e.g. loading into an un-adapted model) — subtract them so the
       # return value is the number of tensors that actually landed.
       return len(owned_state) - len(result.unexpected_keys)
   ```

2. `src/nnx/peft/lora.py`: change the import to `from ._source import _load_owned, _resolve_source_to_state_dict, _save_owned` (`_refuse_shape_mismatch` leaves the import: no body in the file calls it any more), and delete `from .._artifacts import atomic_torch_save` (L47 at `1402b49`), which `_save_owned` now calls. Then:
   - `save_lora_weights`: keep FIX-119's DoRA warning block as the first statements; replace the last three lines (`sd = _lora_keys_only(...)` / `atomic_torch_save(...)` / `return str(path)`) with `return _save_owned(_lora_keys_only(module, module.state_dict()), path)`.
   - `load_lora_weights`: replace the body after the docstring (L329-L336 at `1402b49`) with:

     ```python
         sd = _lora_keys_only(module, _resolve_source_to_state_dict(source, "load_lora_weights"))
         return _load_owned(module, sd, "load_lora_weights")
     ```

3. `src/nnx/peft/ia3.py`: the same pattern. Import `_load_owned, _resolve_source_to_state_dict, _save_owned`, and delete `from .._artifacts import atomic_torch_save` (L36 at `1402b49`). `save_ia3_weights` (L179-L181 at `1402b49`) becomes `return _save_owned(_ia3_keys_only(module, module.state_dict()), path)`. In `load_ia3_weights` (L201-L210 at `1402b49`) the import is `from ._source import _load_owned, _resolve_source_to_state_dict, _save_owned`; keep the two-line allowlist comment and write:

   ```python
       sd = _resolve_source_to_state_dict(source, "load_ia3_weights")
       # Allowlist from the DESTINATION's registered wrappers: a full or
       # colliding source can never write a frozen base tensor.
       return _load_owned(module, _ia3_keys_only(module, sd), "load_ia3_weights")
   ```

4. `src/nnx/peft/prefix.py`: import `_load_owned, _resolve_source_to_state_dict, _save_owned`, and delete `from .._artifacts import atomic_torch_save` (L49 at `1402b49`). `save_prefix_weights` (L281-L283 at `1402b49`) becomes `return _save_owned(tuner.prefix_state_dict(), path)`. `load_prefix_weights` (L299-L309 at `1402b49`) becomes:

   ```python
       sd = _resolve_source_to_state_dict(source, "load_prefix_weights")
       # Allowlist from the DESTINATION tuner's registered lists so a
       # full-model state-dict (or colliding ``model.*`` keys) can never
       # overwrite the frozen base.
       owned = tuner._owned_keys()
       return _load_owned(tuner, {k: v for k, v in sd.items() if k in owned}, "load_prefix_weights")
   ```

   In its docstring replace `(same n_prefix, n_heads, head_dim, n_layers). Otherwise` / ```load_state_dict`` will surface the mismatch.` with `(same n_prefix, n_heads, head_dim, n_layers). A wrong-shaped` / `prefix tensor raises a size-mismatch ``RuntimeError`` before` / `anything is copied.` (it already does, through `_refuse_shape_mismatch`).
5. `src/nnx/peft/prompt.py`: change the import to `from ._source import _load_owned, _resolve_source_to_state_dict, _save_owned`, and delete `from .._artifacts import atomic_torch_save` (L44 at `1402b49`). `save_prompt_weights` (L193-L195 at `1402b49`) becomes `return _save_owned(tuner.prompt_state_dict(), path)`. `load_prompt_weights` (L211-L218 at `1402b49`) becomes:

   ```python
       sd = _resolve_source_to_state_dict(source, "load_prompt_weights")
       # Allowlist from the tuner's own registration: a full checkpoint (or
       # colliding ``model.*`` keys) can never overwrite the frozen base.
       sd = {k: v for k, v in sd.items() if k in PromptTuner._OWNED_KEYS}
       return _load_owned(tuner, sd, "load_prompt_weights")
   ```

   In its docstring replace `(same n_prompt_tokens, d_model). A shape mismatch is` / `surfaced by ``load_state_dict``.` with `(same n_prompt_tokens, d_model). A wrong-shaped soft` / `prompt raises a size-mismatch ``RuntimeError`` before anything` / `is copied.`
6. Keep `import torch` in all four files: each still uses `torch` outside the moved lines. `_source.py` keeps its `import torch` too (`torch.load`). Run `uv run ruff check --fix src/nnx/peft/` and `uv run ruff format src/nnx/peft/`.
7. Regenerate the API reference (the four loader docstrings are catalogued at `scripts/docs/api_reference_catalog.txt` L859-L891): `uv run python -m scripts.docs.build_api_reference`.
8. Add to `CHANGELOG.md` under `## [Unreleased]` → `### Changed`:

   ```
   - **`load_prompt_weights` refuses a wrong-shaped soft prompt before loading, like the other PEFT loaders.** The `RuntimeError` now reads `load_prompt_weights: size mismatch, nothing was loaded: soft_prompt: <source shape> vs <target shape>` instead of torch's `load_state_dict` message.
   ```

9. In `tests/test_peft_prompt.py`, after `test_prompt_tuner_save_load_round_trip`, add:

   ```python
   def test_load_prompt_weights_refuses_a_wrong_shape_before_loading(tmp_path):
       set_seed(0)
       path = tmp_path / "prompt.pt"
       save_prompt_weights(PromptTuner(_tiny_transformer(), n_prompt_tokens=8), path)
       tuner = PromptTuner(_tiny_transformer(), n_prompt_tokens=4)
       before = tuner.soft_prompt.detach().clone()
       with pytest.raises(RuntimeError, match=r"load_prompt_weights: size mismatch, nothing was loaded: soft_prompt"):
           load_prompt_weights(tuner, path)
       assert torch.equal(tuner.soft_prompt, before)
   ```

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] `test_load_prompt_weights_refuses_a_wrong_shape_before_loading` fails on `develop` before the change (torch's message does not match). Check: `uv run pytest tests/test_peft_prompt.py::test_load_prompt_weights_refuses_a_wrong_shape_before_loading`
- [ ] No PEFT adapter module saves or calls `load_state_dict(sd, strict=False)` itself any more. Check: `grep -nE "atomic_torch_save|torch\.save\(|load_state_dict\(sd, strict=False\)" src/nnx/peft/lora.py src/nnx/peft/ia3.py src/nnx/peft/prefix.py src/nnx/peft/prompt.py` returns 0 lines (12 at `1402b49`: four imports and four calls of `atomic_torch_save`, four `load_state_dict` calls)
- [ ] `_source.py` defines `_save_owned` and `_load_owned`, and the save stays atomic. Check: `grep -n "^def _save_owned\|^def _load_owned" src/nnx/peft/_source.py` returns 2 lines, `grep -c "atomic_torch_save(owned_state" src/nnx/peft/_source.py` returns 1, and `uv run pytest tests/test_atomic_torch_saves.py`
- [ ] Every PEFT test passes, including #201's ownership tests, FIX-119's DoRA warning tests and `tests/test_peft_lora.py` L551 ("size mismatch"). Check: `uv run pytest tests -k peft`
- [ ] `docs/api.md` shows the new loader docstrings. Check: `grep -c "before anything is copied" docs/api.md` returns at least 2, and `uv run python -m scripts.docs.build_api_reference --check && uv run python -m scripts.docs.build_docs --check`
- [ ] The CHANGELOG entry is under `[Unreleased]` → `### Changed`. Check: `grep -n "load_prompt_weights\` refuses a wrong-shaped soft prompt" CHANGELOG.md`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [lora.py L280-L336](https://github.com/thekaveh/NNx/blob/1402b49/src/nnx/peft/lora.py#L280-L336) | `save_lora_weights` / `load_lora_weights` |
| [ia3.py L162-L210](https://github.com/thekaveh/NNx/blob/1402b49/src/nnx/peft/ia3.py#L162-L210) | the IA3 copies |
| [prefix.py L271-L309](https://github.com/thekaveh/NNx/blob/1402b49/src/nnx/peft/prefix.py#L271-L309) | the prefix copies; the stale "load_state_dict will surface the mismatch" |
| [prompt.py L183-L218](https://github.com/thekaveh/NNx/blob/1402b49/src/nnx/peft/prompt.py#L183-L218) | the prompt copies, without `_refuse_shape_mismatch` |
| [_artifacts.py L191-L208](https://github.com/thekaveh/NNx/blob/1402b49/src/nnx/_artifacts.py#L191-L208) | `atomic_torch_save`, which every save calls since #859 |
| [_source.py L29-L76](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/peft/_source.py#L29-L76) | `_resolve_source_to_state_dict`, `_refuse_shape_mismatch` |
| [test_peft_lora.py L551](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_peft_lora.py#L551) | pins "size mismatch" |
| #201 | ownership by registration, per family |
| #224 | alias refusals of the `apply_*_to` helpers |

## Scope

**In:** `_save_owned`, `_load_owned`, the eight public bodies, two docstrings, the prompt refusal and its CHANGELOG line.

**Out:** one ownership representation for all families (#201's per-family rules stay). DoRA save / load helpers (DEC-020). The DoRA warning itself (FIX-119).

## Dependencies

**Blocked by:** FIX-119, FEAT-124 (both edit `peft/_source.py` or `save_lora_weights`)
**Blocks:** none
**Related:** #201, #224
