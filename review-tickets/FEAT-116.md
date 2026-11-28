---
id: FEAT-116
title: "[FEAT-116] Validate viz counts with require_count (part 3 of 4)"
labels: ["enhancement", "area: inference", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-38"]
depends_on: []
related_issues: [544, 231]
status: READY
---

**Category:** Inference · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-116: `weight_histogram` and `activation_map` each hand-roll the count rule (`isinstance(x, bool) or not isinstance(x, numbers.Integral) or x < 1`) that `nnx._validation.require_count` already owns. This ticket routes both through one helper, `positive_count`, in a new `src/nnx/viz/_common.py`, which calls `require_count` and keeps the exact refusal text the viz tests pin.

## Context

**What happens today.** `weight_histogram` (`src/nnx/viz/weight_histogram.py`):

```python
    # Any integer, NumPy integers included, passed on as a plain int: plotly's
    # make_subplots(cols=) and nbinsx refuse a numpy.int64.
    for name, value in (("bins", bins), ("cols", cols)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
            raise ValueError(f"weight_histogram: {name} must be a positive integer, got {shown(value)}")
    bins, cols = int(bins), int(cols)
```

`activation_map` (`src/nnx/viz/activation.py`) has the same loop over `max_channels` and `cols`, with the prefix `activation_map:`.

**Why it matters.** The count rule (no `bool`, any `numbers.Integral`, NumPy integers returned as plain `int`) is decided in `_validation.require_count`. Two private copies can drift from it.

**History.** #544 made these two sites accept NumPy integers and added the tests that pin the wording "positive integer" (`test_viz_weight_histogram.py` L84 and L110, `test_viz_activation.py` L149 and L164). Those tests also pin the wording for a `bool` and a float (`{"cols": True}`, `{"max_channels": 2.0}`). `require_count`'s `domain_message=` replaces only its out-of-range message, not its type message, so calling `require_count` directly would change the text for `True` and `2.0`. The helper below therefore re-raises with the existing message. #231 introduced `require_count`.

**Background for the worker.**
- `require_count(value, field, *, owner, minimum=...)` (`src/nnx/_validation.py` L140): returns `int(value)` for any non-`bool` `numbers.Integral` inside the domain; otherwise raises `ValueError` with its own wording (`"<owner> requires <field> to be an integer >= 1, got ..."`).
- `shown(value)` (`src/nnx/_validation.py` L99): the `repr` an error message quotes; an integer too long to print is described by its size.
- Part 1 of 4 (FEAT-114) covers `nnx.decisions`, part 2 (FEAT-115) generation / diffusion / plans, part 4 (FEAT-117) the task modules.

**Proposal.**
1. Create `src/nnx/viz/_common.py`:

   ```python
   """Helpers shared by the ``nnx.viz`` figures. Internal; not part of the public API."""

   from __future__ import annotations

   from .._validation import require_count, shown


   def positive_count(value: object, field: str, *, owner: str) -> int:
       """``value`` as a plain ``int`` once it is a positive integer (NumPy
       integers included: plotly refuses a ``numpy.int64``), by the shared
       count rule of :func:`nnx._validation.require_count`. Anything else
       raises ``ValueError("<owner>: <field> must be a positive integer, got
       <value>")``, the wording the viz tests pin (#544)."""
       try:
           return require_count(value, field, owner=owner, minimum=1)
       except ValueError:
           raise ValueError(f"{owner}: {field} must be a positive integer, got {shown(value)}") from None
   ```

2. In `src/nnx/viz/weight_histogram.py`, `weight_histogram`, replace the five lines from `for name, value in (("bins", bins), ("cols", cols)):` through `bins, cols = int(bins), int(cols)` (L61-L64) with:

   ```python
       bins = positive_count(bins, "bins", owner="weight_histogram")
       cols = positive_count(cols, "cols", owner="weight_histogram")
   ```

   Keep the two comment lines above them (L59-L60). Add `from ._common import positive_count` to the import block. Delete `import numbers` (L15) and `from .._validation import shown` (L22): the deleted loop was their only use in the file.

3. In `src/nnx/viz/activation.py`, `activation_map`, replace the four lines from `for name, value in (("max_channels", max_channels), ("cols", cols)):` through `max_channels, cols = int(max_channels), int(cols)` (L96-L99) with:

   ```python
       max_channels = positive_count(max_channels, "max_channels", owner="activation_map")
       cols = positive_count(cols, "cols", owner="activation_map")
   ```

   Keep the comment lines above them and the `require("plotly.graph_objects", ...)` call before them (the plots extra is still checked first). Add `from ._common import positive_count`. Delete `import numbers` (L27) and `from .._validation import shown` (L35): the deleted loop was their only use in the file.

4. Run `uv run ruff check --fix src/nnx/viz/` to sort the import blocks, then `uv run ruff format src/nnx/viz/`.

Every message stays word for word the same, the check order stays `bins` then `cols` and `max_channels` then `cols`, and the values passed on stay plain `int`. No CHANGELOG entry: nothing a user can observe changes.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] `src/nnx/viz/_common.py` defines `positive_count(value, field, *, owner) -> int` as in step 1. Check: `grep -n "def positive_count" src/nnx/viz/_common.py` returns 1 line
- [ ] Neither viz module hand-rolls the integer check any more. Check: `grep -n "numbers.Integral" src/nnx/viz/weight_histogram.py src/nnx/viz/activation.py` returns 0 lines
- [ ] The #544 tests pass unchanged, including the `True` and `2.0` cases. Check: `uv run pytest tests/test_viz_weight_histogram.py::test_weight_histogram_validates_its_grid_arguments tests/test_viz_weight_histogram.py::test_weight_histogram_passes_numpy_grid_arguments_on_as_ints tests/test_viz_activation.py::test_activation_map_validates_its_grid_arguments tests/test_viz_activation.py::test_activation_map_passes_numpy_grid_arguments_on_as_ints`
- [ ] The viz and import suites pass. Check: `uv run pytest tests/test_viz_weight_histogram.py tests/test_viz_activation.py tests/test_imports.py tests/test_params_boundary_validation.py -k "viz or activation or weight_histogram or import"`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [weight_histogram.py L59-L64](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/viz/weight_histogram.py#L59-L64) | `weight_histogram`: the hand-rolled count loop over `bins` / `cols` |
| [activation.py L94-L99](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/viz/activation.py#L94-L99) | `activation_map`: the same loop over `max_channels` / `cols` |
| [_validation.py L140-L179](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_validation.py#L140-L179) | `require_count`: the shared rule; `domain_message` covers only the out-of-range case |
| [test_viz_weight_histogram.py L82-L85](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_viz_weight_histogram.py#L82-L85) | pins "positive integer" for `cols=0`, `bins=0`, `cols=True` |
| [test_viz_activation.py L147-L165](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_viz_activation.py#L147-L165) | pins "positive integer", including `max_channels=2.0` and `cols=True` |
| #544 | made these sites accept NumPy integers (FIX-076) |
| #231 | introduced `require_count` (FIX-021) |

## Scope

**In:** the two viz count checks and the new `positive_count` helper.

**Out:** the `NNModel` unwrap and local imports in `nnx.viz` (FEAT-130, which adds `module_of` to the same `_common.py`). The decisions, generation, diffusion, plans and task-module count checks (FEAT-114, FEAT-115, FEAT-117).

## Dependencies

**Blocked by:** none
**Blocks:** FEAT-130
**Related:** #544, #231
