---
id: FEAT-119
title: "[FEAT-119] Move the module-path walkers out of peft/_targets.py"
labels: ["enhancement", "area: peft", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-40"]
depends_on: ["FIX-149"]
related_issues: [224, 523, 524, 526]
status: READY
---

**Category:** PEFT · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-119: `peft/_targets.py` holds the walk over every registration path (`_registrations`), the `WEIGHT_READING_PARENTS` tuple and the path lister (`_listed`), and QAT, magnitude pruning and the transform recipes import them privately from the PEFT package. `qat.py` also re-implements `_listed` inline. This ticket moves the three into a new torch-only module, `src/nnx/_module_paths.py`, leaves aliases in `_targets.py`, and points the importers at the new module. Every message stays word for word the same.

## Context

**What happens today.** The importers reach into PEFT:

```python
from ..peft._targets import _MAX_LISTED_PATHS, _listed, _registrations
```

(`quantize/qat.py` L48), `from ..peft._targets import WEIGHT_READING_PARENTS` inside `magnitude_prune` (`prune/magnitude.py` L136) and two function-local imports in `transforms.py` (L474, L484). `_refuse_bypassed_linears` in `qat.py` copies `_listed` without its parentheses:

```python
    if reasons:
        extra = len(read) - _MAX_LISTED_PATHS
        listed = ", ".join(read[:_MAX_LISTED_PATHS]) + (f", … and {extra} more" if extra > 0 else "")
        raise ValueError(f"QAT cannot prepare {listed}: {'; '.join(reasons)}. Nothing was modified.")
```

The original in `_targets.py`:

```python
def _listed(paths: list[str]) -> str:
    shown = ", ".join(paths[:_MAX_LISTED_PATHS])
    extra = len(paths) - _MAX_LISTED_PATHS
    return f"({shown}, … and {extra} more)" if extra > 0 else f"({shown})"
```

**Why it matters.** Module-walking code shared by four features lives in one feature's private module, so a PEFT refactor can break QAT or pruning, and the inline copy can drift.

**History.** #224 (FIX-014) introduced the every-path walk for PEFT aliases. #523 (DoRA bases), #524 (surgery) and #526 (QAT on DoRA and weight-reading parents) added the refusals that read it. Those issues' refusals depend on the members of `WEIGHT_READING_PARENTS`, so they do not change. The QAT messages are pinned without parentheses (`tests/test_quantize_qat.py` L535, L583, L655), so the inline copy is replaced by a parenthesis-free twin of `_listed`, not by `_listed` itself.

**Background for the worker.**
- Registration path: a dotted name under which a module object is registered. `named_modules()` reports a shared module once; `_registrations` reports every path, stopping only at a module already on the current ancestor chain (a cycle).
- `WEIGHT_READING_PARENTS`: torch parents that read a child `Linear`'s `.weight` directly (`nn.MultiheadAttention`, `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`), so a wrapped child would break them.
- Alias rule: tests and pickles refer to private names by module path, so the old names in `peft/_targets.py` stay importable.
- The new module imports only `collections.abc` and `torch.nn`, so it can be imported at module top anywhere without a cycle (`peft/_targets.py` itself imports `..seeding`).

**Proposal.**
1. Create `src/nnx/_module_paths.py`:

   ```python
   """Every registration path of a module tree, and how a refusal lists them.

   ``nn.Module.named_modules()`` reports a module object once, so a layer
   registered under two names is visible at its first path only.
   :func:`registrations` walks every path; PEFT target selection
   (``nnx.peft._targets``), QAT, magnitude pruning and the transform recipes
   read it. Internal; imports torch only; not part of the public API.
   """

   from __future__ import annotations

   from collections.abc import Iterator, Sequence

   from torch import nn

   MAX_LISTED_PATHS = 10
   """How many paths a refusal names before it says how many more there are."""


   # Parents that read a child Linear's weight directly (MultiheadAttention's
   # out_proj; the Transformer layers' eval fast path over linear1 / linear2):
   # a wrapped or factorized child breaks them.
   WEIGHT_READING_PARENTS: tuple[type[nn.Module], ...] = (
       nn.MultiheadAttention,
       nn.TransformerEncoderLayer,
       nn.TransformerDecoderLayer,
   )


   def registrations(root: nn.Module) -> Iterator[tuple[str, nn.Module, str, nn.Module]]:
       <the body and docstring of peft/_targets.py::_registrations (L49-L70), unchanged>


   def joined_paths(paths: Sequence[str]) -> str:
       """The first :data:`MAX_LISTED_PATHS` paths, comma-separated, then
       ``", … and <n> more"`` when some are left out."""
       extra = len(paths) - MAX_LISTED_PATHS
       shown = ", ".join(paths[:MAX_LISTED_PATHS])
       return f"{shown}, … and {extra} more" if extra > 0 else shown


   def listed_paths(paths: Sequence[str]) -> str:
       """:func:`joined_paths` in parentheses: ``"(a, b, … and 3 more)"``."""
       return f"({joined_paths(paths)})"
   ```

   Copy the `_registrations` body and docstring verbatim; only the function name changes.

2. In `src/nnx/peft/_targets.py`:
   - Delete `_MAX_LISTED_PATHS = 10` (L35), the `WEIGHT_READING_PARENTS` comment and tuple (L38-L45), `_registrations` (L48-L70) and `_listed` (L73-L76).
   - Add, after the `from ..seeding import ...` line:

     ```python
     from .._module_paths import MAX_LISTED_PATHS, WEIGHT_READING_PARENTS, listed_paths, registrations

     # The old private names, kept for importers and pickles (moved to nnx._module_paths).
     _MAX_LISTED_PATHS = MAX_LISTED_PATHS
     _registrations = registrations
     _listed = listed_paths
     ```

   - In `select_linear_slots`, replace `_registrations(module)` (L102) with `registrations(module)` and `_listed(` (L155, L167) with `listed_paths(`.
3. In `src/nnx/quantize/qat.py`:
   - Replace L48 with `from .._module_paths import joined_paths, listed_paths, registrations`.
   - Replace `_registrations(` with `registrations(` (L153, L326) and `_listed(dora_layers)` (L167) with `listed_paths(dora_layers)`.
   - Replace the two lines `extra = len(read) - _MAX_LISTED_PATHS` / `listed = ", ".join(...)` (L186-L187) with `listed = joined_paths(read)`.
   - In the `_refuse_bypassed_linears` docstring (L148), change ``the first ``_MAX_LISTED_PATHS`` of them listed`` to ``the first ``nnx._module_paths.MAX_LISTED_PATHS`` of them listed``.
4. In `src/nnx/prune/magnitude.py`: add `from .._module_paths import WEIGHT_READING_PARENTS` next to the `from .._reparameterization import weight_reparameterization` line (L34), and delete the function-local `from ..peft._targets import WEIGHT_READING_PARENTS` (L136). Keep the comment above it and the later `from ..peft.dora import DoRALinear` (L152) as they are.
5. In `src/nnx/transforms.py`: add `from ._module_paths import WEIGHT_READING_PARENTS, registrations` to the module-top imports; delete the function-local imports in `_registration_paths` (L474) and `_weight_reading_parents` (L484); replace `_registrations(net)` with `registrations(net)` at L477 and L488.
6. In `src/nnx/prune/_select.py` (added by FIX-149), replace `from ..peft._targets import _registrations` with `from .._module_paths import registrations as _registrations`.
7. Run `uv run ruff check --fix src/nnx/` and `uv run ruff format src/nnx/`.

`joined_paths(read)` equals the deleted inline expression and `listed_paths(p)` equals `_listed(p)` for every list, so every message is unchanged. No CHANGELOG entry. No test monkeypatches these names (`git grep -n "_registrations\|_listed\|WEIGHT_READING_PARENTS" tests/` returns 0 lines).

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need (torchao for the QAT tests).

## Acceptance criteria

- [ ] `src/nnx/_module_paths.py` defines `MAX_LISTED_PATHS`, `WEIGHT_READING_PARENTS`, `registrations`, `joined_paths` and `listed_paths`, and imports nothing from `nnx`. Check: `grep -n "^from \.\|^import nnx" src/nnx/_module_paths.py` returns 0 lines
- [ ] No module outside `nnx.peft` imports `peft._targets`. Check: `git grep -n "peft._targets\|from \.\._targets\|from \._targets" src/nnx -- ':!src/nnx/peft'` returns 0 lines
- [ ] The old names still import and are the new objects. A new test `tests/test_peft_lora.py::test_the_moved_path_helpers_keep_their_old_names` asserts `_targets._registrations is _module_paths.registrations`, `_targets._listed is _module_paths.listed_paths`, `_targets._MAX_LISTED_PATHS == 10` and `_targets.WEIGHT_READING_PARENTS is _module_paths.WEIGHT_READING_PARENTS`, and that `_module_paths.listed_paths([f"p{i}" for i in range(12)])` ends with `"p9, … and 2 more)"` while `joined_paths` gives the same text without the parentheses. Check: `uv run pytest tests/test_peft_lora.py::test_the_moved_path_helpers_keep_their_old_names`
- [ ] The QAT messages pinned without parentheses pass unchanged. Check: `uv run pytest tests/test_quantize_qat.py`
- [ ] The PEFT, pruning, transform and surgery suites pass. Check: `uv run pytest tests -k "peft or qat or prune or transforms or surgery"`
- [ ] `uv run pytest tests/test_imports.py` passes. Check: `uv run pytest tests/test_imports.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [_targets.py L32-L76](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/peft/_targets.py#L32-L76) | `_MAX_LISTED_PATHS`, `WEIGHT_READING_PARENTS`, `_registrations`, `_listed` |
| [_targets.py L102](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/peft/_targets.py#L102), [L155](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/peft/_targets.py#L155), [L167](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/peft/_targets.py#L167) | the internal uses in `select_linear_slots` |
| [qat.py L48](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/quantize/qat.py#L48) | QAT imports PEFT's private walker |
| [qat.py L184-L188](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/quantize/qat.py#L184-L188) | `_refuse_bypassed_linears`: the inline, parenthesis-free copy of `_listed` |
| [magnitude.py L133-L143](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/prune/magnitude.py#L133-L143) | `magnitude_prune`: the function-local import of `WEIGHT_READING_PARENTS` |
| [transforms.py L471-L490](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/transforms.py#L471-L490) | `_registration_paths`, `_weight_reading_parents`: two function-local imports |
| [test_quantize_qat.py L535](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_quantize_qat.py#L535), [L583](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_quantize_qat.py#L583), [L655](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_quantize_qat.py#L655) | the QAT wording without parentheses |
| #224, #523, #524, #526 | the refusals that read the walk |

## Scope

**In:** the new module, the aliases, and the QAT, pruning and transform importers.

**Out:** making `surgery/_utils.registered_paths` use `registrations`: it walks `named_modules(remove_duplicate=False)`, which recurses without end on a cyclic registration where `registrations` stops, and it includes the root as `""`, so the switch is a behaviour change, not a move. The other `_listed` helpers (`interop/gguf/tensor_name_map.py` L89, `comparison.py` L1390) format differently ("carry", "a and b") and stay. The `_targets.py` docstring (FIX-143) and `selected = matched` (FEAT-063). Matching prune patterns on every registration path (FIX-149). Any change to `WEIGHT_READING_PARENTS`'s members.

## Dependencies

**Blocked by:** FIX-149 (it adds `prune/_select.py`, whose import step 6 retargets)
**Blocks:** FEAT-139
**Related:** #224, #523, #524, #526
