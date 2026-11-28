---
id: FEAT-130
title: "[FEAT-130] Share the NNModel unwrap in nnx.viz and hoist its imports"
labels: ["enhancement", "area: inference", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-54", "DOC-31"]
depends_on: ["FEAT-116", "FEAT-122", "FIX-129"]
related_issues: [249, 472, 570]
status: READY
---

**Category:** Inference · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-130: six `nnx.viz` functions import `NNModel` inside the function, behind the comment "Local import to avoid a circular import at package init time", which is false; four of them then repeat the same `NNModel` → `.net` unwrap, and four modules carry two back-to-back `if TYPE_CHECKING:` blocks. This ticket adds `module_of` to `src/nnx/viz/_common.py`, uses it in the four plain unwraps, imports `NNModel` (and `viz.summary`'s other core helpers) at module top, merges the `TYPE_CHECKING` blocks and deletes the false comments.

## Context

**What happens today.** `weight_histogram` (the `activation_map`, `attribute` and `gradient_flow` copies are the same):

```python
    # Local import to avoid a circular import at package init time.
    from ..nn.nn_model import NNModel

    if isinstance(model, NNModel):
        model = model.net
```

and its module header:

```python
if TYPE_CHECKING:
    import plotly.graph_objects as go


if TYPE_CHECKING:
    from ..nn.nn_model import NNModel
```

`viz/summary.py` and `viz/netron.py` import `NNModel` the same way and also read the model's adapter or device, so they keep their own branch.

**Why it matters.** The comment is wrong — no module under `src/nnx/nn` imports `nnx.viz`, and `src/nnx/vis_utils.py` L35 imports `NNModel` at module top without trouble — so the local imports and the repeated unwrap are noise.

**History.** #249 (FEAT-031) made the visualization dependencies lazy: the optional `plotly`, `torchinfo`, `captum`, `netron` and `onnx` imports stay inside the functions (FIX-129 routes them through `nnx._optional.require`); `NNModel` is core and is not one of them. DOC-31's viz bullet (#472 / #570 doc sweeps) asks for this hoist. FEAT-116 created `viz/_common.py` with `positive_count`.

**Background for the worker.**
- Import-cycle check done for this ticket: `nnx.nn.nn_model`, `nnx.models`, `nnx.utils`, `nnx.peft.dora` and `nnx.seeding` never import `nnx.viz`, directly or through their own module-top imports, so importing them at the top of a viz module cannot cycle.
- `TYPE_CHECKING` imports exist for annotations only. After this ticket the four plain modules still need `NNModel` only in the `Union[nn.Module, NNModel]` annotation, so it stays under `TYPE_CHECKING` there; `summary` and `netron` use it at runtime, so they import it for real.

**Proposal.**
1. In `src/nnx/viz/_common.py` add `from typing import Union`, `from torch import nn` and `from ..nn.nn_model import NNModel` to the imports, and add:

   ```python
   def module_of(model: Union[nn.Module, NNModel]) -> nn.Module:
       """``model.net`` for an :class:`~nnx.NNModel`, else ``model`` itself."""
       return model.net if isinstance(model, NNModel) else model
   ```

2. `src/nnx/viz/weight_histogram.py`:
   - Merge the two `if TYPE_CHECKING:` blocks (L24-L29) into:

     ```python
     if TYPE_CHECKING:
         import plotly.graph_objects as go

         from ..nn.nn_model import NNModel
     ```

   - Replace the comment, the local import and the `isinstance` branch (L67-L71) with `model = module_of(model)`.
   - Import `module_of` next to `positive_count` (`from ._common import module_of, positive_count`).
3. `src/nnx/viz/activation.py`: the same three edits (blocks at L38-L43, unwrap at L100-L104).
4. `src/nnx/viz/attribute.py`: the same three edits (blocks at L25-L30; the unwrap at L167-L172 has a two-line comment, delete both lines); add `from ._common import module_of`.
5. `src/nnx/viz/gradient_flow.py`: the same three edits (blocks at L16-L21; the unwrap at L51-L54 has no comment); add `from ._common import module_of`.
6. `src/nnx/viz/summary.py`:
   - Module top: add `from ..models import KeywordInputs, default_batch_adapter`, `from ..nn.nn_model import NNModel, _to_device`, `from ..peft.dora import _tool_hooks` and `from ..utils import _capture_training_modes, _restore_training_modes`; remove `from ..nn.nn_model import NNModel` from the `TYPE_CHECKING` block (it keeps `from torchinfo import ModelStatistics`).
   - In `summary`, delete the line `# Local import to avoid a circular import at package init time.` and the local `from ..nn.nn_model import NNModel` (L79-L80), keeping `# Accept NNModel directly — unwrap to .net so callers don't have to.` and the `adapter` branch below it; delete the local imports at L87 and L112-L113.
   - In `_placed`, delete the local `from ..nn.nn_model import _to_device` (L128).
   - Leave the `torchinfo` import in `summary` as FIX-129 left it (optional extra).
7. `src/nnx/viz/netron.py`: add `from ..nn.nn_model import NNModel` at module top, delete the `if TYPE_CHECKING:` block (L32-L33) and `TYPE_CHECKING` from the `typing` import, and delete the comment and local import in `netron_export` (L73-L74). Keep its `isinstance` branch (it reads `model.device`) and the `netron` / `onnx` imports.
8. Run `uv run ruff check --fix src/nnx/viz/` and `uv run ruff format src/nnx/viz/`.

No behaviour or message changes, no test changes, no CHANGELOG entry. No test monkeypatches these names inside `nnx.viz` (`git grep -nE "setattr\((summary|netron|activation|attribute|gradient_flow|weight_histogram)" tests/` returns 0 lines).

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need (plotly, torchinfo, captum, onnx).

## Acceptance criteria

- [ ] The false comment is gone. Check: `git grep -n "Local import to avoid a circular import" src/nnx/viz` returns 0 lines
- [ ] No viz function imports from `nnx.nn.nn_model` any more. Check: `uv run python -c "import ast, pathlib; print([(p.name, n.lineno) for p in pathlib.Path('src/nnx/viz').glob('*.py') for f in ast.walk(ast.parse(p.read_text())) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(f) if isinstance(n, ast.ImportFrom) and n.module == 'nn.nn_model'])"` prints `[]` (7 entries today)
- [ ] One `TYPE_CHECKING` block per module. Check: `grep -c "^if TYPE_CHECKING:" src/nnx/viz/*.py` returns at most 1 for every file
- [ ] `module_of` is used four times. Check: `grep -rn "module_of(model)" src/nnx/viz` returns 4 lines
- [ ] The viz and import suites pass. Check: `uv run pytest tests/test_imports.py tests/test_core_install.py tests/test_viz_weight_histogram.py tests/test_viz_activation.py tests/test_viz_attribute.py tests/test_viz_gradient_flow.py tests/test_viz_summary.py tests/test_viz_netron.py tests/test_inference_helpers_preserve_training_mode.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [weight_histogram.py L24-L29](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/viz/weight_histogram.py#L24-L29), [L67-L71](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/viz/weight_histogram.py#L67-L71) | doubled `TYPE_CHECKING`; the local import, false comment and unwrap |
| [activation.py L38-L43](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/viz/activation.py#L38-L43), [L100-L104](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/viz/activation.py#L100-L104) | the same |
| [attribute.py L25-L30](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/viz/attribute.py#L25-L30), [L167-L172](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/viz/attribute.py#L167-L172) | the same |
| [gradient_flow.py L16-L21](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/viz/gradient_flow.py#L16-L21), [L51-L54](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/viz/gradient_flow.py#L51-L54) | the same |
| [summary.py L78-L87](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/viz/summary.py#L78-L87), [L112-L113](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/viz/summary.py#L112-L113), [L128](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/viz/summary.py#L128) | `summary`'s local core imports |
| [netron.py L32-L33](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/viz/netron.py#L32-L33), [L73-L80](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/viz/netron.py#L73-L80) | `netron_export`'s local import and its device-reading branch |
| [vis_utils.py L35](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/vis_utils.py#L35) | `NNModel` imported at module top elsewhere without a cycle |
| #249 | the optional viz dependencies stay lazy |

## Scope

**In:** `module_of`, the six `NNModel` imports, `summary`'s other core imports, the `TYPE_CHECKING` merges, the false comments.

**Out:** the optional-extra imports (`plotly`, `torchinfo`, `captum`, `netron`, `onnx`; FIX-129). The count validators (FEAT-116). The RNG snapshot in `summary` (FEAT-122). The other DOC-31 bullets (FIX-143, FEAT-123).

## Dependencies

**Blocked by:** FEAT-116 (creates `viz/_common.py`), FEAT-122 and FIX-129 (both edit `viz/summary.py`)
**Blocks:** none
**Related:** #249, #472, #570
