---
id: FEAT-115
title: "[FEAT-115] Validate generation, diffusion and plan counts with require_count (part 2 of 4)"
labels: ["enhancement", "area: configuration", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-38"]
depends_on: []
related_issues: [544, 231]
status: READY
---

**Category:** Configuration · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-115: Four more hand-written integer checks (`isinstance(x, bool) or not isinstance(x, numbers.Integral) or x < N`) remain outside `nnx.decisions`: `TopKFilter(top_k=)`, a `top_k` `LogitsStage`, `DiffusionObjective(seed=)` and `ExperimentPlan`'s seed check. This ticket routes them through `_validation.require_count`, which all four modules' neighbours already use. The error type stays `ValueError`; the messages change to the shared wording. Part 2 of SMELL-38 (FEAT-114 did the decisions sites).

## Context

**What happens today.** `TopKFilter.__init__` (`generation/logits_processors.py` L98–L102):

```python
    def __init__(self, top_k: int):
        # As LogitsStage.top_k: a float, bool or NaN failed mid-decode in torch.topk.
        if isinstance(top_k, bool) or not isinstance(top_k, numbers.Integral) or top_k < 1:
            raise ValueError(f"top_k must be an integer > 0, got {top_k!r}")
        self.top_k = int(top_k)
```

`LogitsStage.__post_init__` (`generation/pipeline.py` L100–L103):

```python
        if self.kind == "top_k":
            if isinstance(self.value, bool) or not isinstance(self.value, numbers.Integral) or self.value < 1:
                raise ValueError(f"a top_k stage needs an integer >= 1, got {self.value!r}")
            object.__setattr__(self, "value", int(self.value))
```

`DiffusionObjective.__init__` (`diffusion/objective.py` L106–L109) refuses a seed outside `[0, 2**64)` with `seed must be an integer in [0, 2**64) or None, got ...`, and `ExperimentPlan._check_seed` (`plans.py` L679–L680) reports `must be an integer in [0, 2**32), got ...` for the seed path.

**Why it matters.** Each copy words the same rule differently, and `require_count` is the rule the rest of NNx (for example `set_seed`, `seeding.py` L56) applies.

**History.** #231 introduced `require_count`; #544 moved most hand-written validators to it and left these. The tests that cover these sites match only the field name: `tests/test_params_boundary_validation.py` L554, L556 and L651 (`match="top_k"`), `tests/test_diffusion_objective.py` L157 (`match="seed"`), and `tests/test_experiment_plan.py` L295–L296 (the reported path `"seed"`). All keep passing.

**Background for the worker.**
- `require_count(value, field, *, owner, minimum=None, maximum=None)` (`_validation.py` L140): returns `int(value)` for a `numbers.Integral` that is not a `bool`; otherwise `ValueError` with `"{owner} requires {field} to be an integer{domain}, got {value!r} of type {type}"`, or `"{owner} requires {field}{domain}, got {value!r}"` when out of range (`{domain}`: ` >= 1`, or ` >= 0 and <= <max>`).
- `ExperimentPlan.validate()` collects problems through `report(path, message)` instead of raising; it already reports other validators' messages with `report(path, str(exc))` (`plans.py` L557, L577, L597).
- `_SEED_LIMIT = 2**32` (`plans.py` L95): NumPy's legacy seeding accepts `[0, 2**32)`.

**Proposal.**
1. `src/nnx/generation/logits_processors.py`: change `from .._validation import require_finite_real, shown` (L18) to `from .._validation import require_count, require_finite_real, shown`, and replace L100–L102 (the `if`, the `raise` and `self.top_k = int(top_k)`) with `self.top_k = require_count(top_k, "top_k", owner="TopKFilter", minimum=1)`. Keep the comment above.
2. `src/nnx/generation/pipeline.py`: add `from .._validation import require_count` to the first-party imports, and replace L101–L103 (the inner `if`, the `raise` and the `object.__setattr__`) with `object.__setattr__(self, "value", require_count(self.value, "top_k", owner="LogitsStage", minimum=1))`.
3. `src/nnx/diffusion/objective.py`: add `from .._validation import require_count` to the first-party imports (before `from ..components import ComponentSpec`, L49), and replace L106–L109 (the `if seed is not None and (...)` test and its `raise`) with:

   ```python
           if seed is not None:
               seed = require_count(seed, "seed", owner="DiffusionObjective", minimum=0, maximum=2**64 - 1)
   ```

   `self.seed = None if seed is None else int(seed)` (L113) stays.
4. `src/nnx/plans.py`: add `from ._validation import require_count` to the first-party imports, and in `_check_seed` replace L679–L680 with:

   ```python
           try:
               require_count(seed, path, owner="ExperimentPlan", minimum=0, maximum=_SEED_LIMIT - 1)
           except ValueError as exc:
               report(path, str(exc))
   ```

5. Run `uv run ruff check --fix src/nnx/generation/ src/nnx/diffusion/objective.py src/nnx/plans.py` (it drops `import numbers` where it became unused: `logits_processors.py` and `diffusion/objective.py`) and `uv run ruff format` on the same files.
6. Add to `tests/test_params_boundary_validation.py`, after `test_top_k_filter_refuses_what_its_pipeline_stage_refuses` (L647):

   ```python
   def test_generation_and_diffusion_counts_use_the_shared_count_message():
       import numpy as np

       from nnx import NoiseSchedulers
       from nnx.diffusion.objective import DiffusionObjective
       from nnx.generation.logits_processors import TopKFilter
       from nnx.generation.pipeline import LogitsStage

       with pytest.raises(ValueError, match=r"^TopKFilter requires top_k >= 1, got 0$"):
           TopKFilter(0)
       with pytest.raises(ValueError, match=r"^LogitsStage requires top_k to be an integer >= 1, got 2\.5 of type float$"):
           LogitsStage("top_k", 2.5)
       with pytest.raises(ValueError, match=r"^DiffusionObjective requires seed >= 0 and <= 18446744073709551615, got -1$"):
           DiffusionObjective(NoiseSchedulers.LINEAR(T=10), seed=-1)
       assert TopKFilter(np.int64(3)).top_k == 3 and LogitsStage("top_k", np.int64(3)).value == 3
       assert DiffusionObjective(NoiseSchedulers.LINEAR(T=10), seed=np.uint64(2**64 - 1)).seed == 2**64 - 1
   ```

7. Add to `CHANGELOG.md` under `## [Unreleased]` → `### Changed`:

   ```
   - **Generation, diffusion and plan counts use the shared count validator.** `TopKFilter(top_k=)`, a `top_k` `LogitsStage`, `diffusion_objective(seed=)` and an `ExperimentPlan` seed refuse a non-integer or out-of-range value with the shared message (`TopKFilter requires top_k >= 1, got 0`). The error type stays `ValueError`.
   ```

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The new test fails on `develop` before the change and passes after it. Check: `uv run pytest tests/test_params_boundary_validation.py::test_generation_and_diffusion_counts_use_the_shared_count_message`
- [ ] No hand-written count check remains at the four sites. Check: `grep -n "not isinstance(.*, numbers.Integral)" src/nnx/generation/logits_processors.py src/nnx/generation/pipeline.py src/nnx/diffusion/objective.py src/nnx/plans.py` returns 0 lines
- [ ] The existing field-name matches and the plan's seed paths pass. Check: `uv run pytest tests/test_params_boundary_validation.py tests/test_diffusion_objective.py tests/test_experiment_plan.py tests/test_ordered_logits_pipeline.py tests/test_logits_chain.py tests/test_generative_nn_model.py`
- [ ] The CHANGELOG entry is under `[Unreleased]` → `### Changed`. Check: `grep -n "Generation, diffusion and plan counts use the shared count validator" CHANGELOG.md`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [logits_processors.py L18](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/generation/logits_processors.py#L18), [L90-L102](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/generation/logits_processors.py#L90-L102) | `TopKFilter` |
| [pipeline.py L88-L104](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/generation/pipeline.py#L88-L104) | `LogitsStage`'s `top_k` branch |
| [objective.py L95-L113](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/diffusion/objective.py#L95-L113) | `DiffusionObjective`'s seed check |
| [plans.py L95](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/plans.py#L95), [L672-L680](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/plans.py#L672-L680) | `_SEED_LIMIT` and `_check_seed` |
| [seeding.py L56](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/seeding.py#L56) | `set_seed` already uses `require_count` |
| [test_params_boundary_validation.py L551-L556](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_params_boundary_validation.py#L551-L556), [L645-L651](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_params_boundary_validation.py#L645-L651) | the `top_k` tests (field name only) |
| [test_experiment_plan.py L290-L296](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_experiment_plan.py#L290-L296) | the plan seed paths |
| #231, #544 | the shared count validator and the leftovers |

## Scope

**In:** the four sites, one test, the CHANGELOG line.

**Out:** the decisions sites (FEAT-114), viz (FEAT-116), the task modules' `_count` helpers (FEAT-117). `MoELinear(top_k=)` (already validated). `top_p` (FIX-123).

## Dependencies

**Blocked by:** none
**Blocks:** none
**Related:** #544, #231
