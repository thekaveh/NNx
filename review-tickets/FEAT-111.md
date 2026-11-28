---
id: FEAT-111
title: "[FEAT-111] Share forward noising between the diffusion step and objective"
labels: ["enhancement", "area: training", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-36"]
depends_on: ["FIX-120"]
related_issues: [257, 561]
status: READY
---

**Category:** Training · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-111: `diffusion_train_step_factory` and `DiffusionObjective` both compute the DDPM forward noising `x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε` with two `_extract` calls each. This ticket adds `NoiseSchedule.q_sample(x_0, t, eps)` and moves `_extract` beside it into `diffusion/schedules.py`; both training paths call `q_sample`. Each path keeps its own random draws (the step the global RNG, the objective its generator), so training is bit-identical. This is the `q_sample` part of SMELL-36; the freeze helper is FEAT-112.

## Context

**What happens today.** The step (`diffusion/training.py` L103–L105):

```python
        sqrt_a = _extract(schedule.sqrt_alphas_cumprod, t, x_0.shape, x_0.dtype)
        sqrt_1ma = _extract(schedule.sqrt_one_minus_alphas_cumprod, t, x_0.shape, x_0.dtype)
        x_t = sqrt_a * x_0 + sqrt_1ma * eps
```

and the objective (`diffusion/objective.py` L180–L182), the same three lines on `self._schedule_on(x_0.device)`. `_extract` is defined in `training.py` (L37–L50) and imported by `objective.py` (L53).

**Why it matters.** The forward process is the core of both paths; two copies can drift (a dtype or broadcast fix applied to one).

**History.** #257 (FEAT-040) added the diffusion objective beside the imperative step; #561 applied `BatchAdapter.output()` in both. The rest of SMELL-36 is deliberate: #782 shared the placement check while keeping distinct messages, and the JEPA component-state checks validate different schemas, so only `q_sample` and the freeze helper are shared. FIX-120 (BUG-20) changes the step's returned record first; the noising lines are untouched by it.

**Background for the worker.**
- `NoiseSchedule` (`diffusion/schedules.py` L29): a frozen, slotted dataclass of 1-D length-`T` tensors (`sqrt_alphas_cumprod` is `√ᾱ_t`, `sqrt_one_minus_alphas_cumprod` is `√(1 − ᾱ_t)`). Adding a method does not change its fields or its pickles.
- `_extract(values, t, target_shape, dtype)`: gathers `values[t]` on `t`'s device, casts to `dtype` and adds trailing unit axes so it broadcasts against `x_0`.
- `q_sample` is the standard DDPM name for sampling `x_t ~ q(x_t | x_0)` given the noise.
- `schedules.py` cannot import `training.py` (`training.py` imports `NoiseSchedule` from it), so `_extract` moves into `schedules.py`.
- `q_sample` is public (`NoiseSchedule` is catalogued: `@api nnx.diffusion.schedules.NoiseSchedule`), so it appears in `docs/api.md` and gets an "Added" CHANGELOG line.

**Proposal.**
1. In `src/nnx/diffusion/schedules.py`:
   - add `from typing import Optional` to the imports;
   - cut `_extract` (`training.py` L37–L50, with its docstring) and paste it directly above `class NoiseSchedule` (L28), unchanged;
   - add to `NoiseSchedule`, after `to` (L58–L70):

     ```python
         def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
             """The forward-noised sample ``x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · eps``
             (q(x_t | x_0) given the noise ``eps``): the coefficients are read at
             the timesteps ``t`` (``(B,)``, on any device: moved to ``t``'s) in
             ``x_0``'s dtype and broadcast over its shape."""
             sqrt_a = _extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.dtype)
             sqrt_1ma = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape, x_0.dtype)
             return sqrt_a * x_0 + sqrt_1ma * eps
     ```

2. In `src/nnx/diffusion/training.py`:
   - change `from .schedules import NoiseSchedule` (L34) to `from .schedules import NoiseSchedule, _extract as _extract` (the redundant alias marks a re-export, so the old import path `nnx.diffusion.training._extract` keeps working and ruff does not flag it), and delete `from typing import Optional` (L25): `_extract`'s signature was its only use;
   - in the step, replace L103–L105 with `x_t = schedule.q_sample(x_0, t, eps)`.
3. In `src/nnx/diffusion/objective.py`, delete `from .training import _extract` (L53) and replace L180–L182 with `x_t = schedule.q_sample(x_0, t, eps)` (`schedule = self._schedule_on(x_0.device)` on the line above stays).
4. In `training.py`'s module docstring, change "via the ``_extract`` helper" (L14) to "via ``NoiseSchedule.q_sample``".
5. Regenerate the API reference: `uv run python -m scripts.docs.build_api_reference`.
6. Add to `tests/test_diffusion_schedules.py`:

   ```python
   def test_q_sample_is_the_forward_process():
       schedule = NoiseSchedulers.LINEAR(T=10)
       generator = torch.Generator().manual_seed(0)
       x_0 = torch.randn(3, 2, 4, generator=generator)
       eps = torch.randn(3, 2, 4, generator=generator)
       t = torch.tensor([0, 4, 9])
       expected = (
           schedule.sqrt_alphas_cumprod[t].view(3, 1, 1) * x_0
           + schedule.sqrt_one_minus_alphas_cumprod[t].view(3, 1, 1) * eps
       )
       assert torch.equal(schedule.q_sample(x_0, t, eps), expected)
       assert schedule.q_sample(x_0.half(), t, eps.half()).dtype == torch.float16
   ```

7. Add to `CHANGELOG.md` under `## [Unreleased]` → `### Added`:

   ```
   - **`NoiseSchedule.q_sample(x_0, t, eps)`.** The DDPM forward process, shared by `diffusion_train_step_factory` and `diffusion_objective`; custom diffusion steps can call it too.
   ```

**Worker setup.** Branch from `develop` after FIX-120 has merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The new test passes. Check: `uv run pytest tests/test_diffusion_schedules.py::test_q_sample_is_the_forward_process`
- [ ] The forward process is written once. Check: `grep -rn "sqrt_one_minus_alphas_cumprod, t" src/nnx/diffusion` returns 1 line (`schedules.py`)
- [ ] `_extract` is defined once and still importable from `training`. Check: `grep -rn "def _extract" src/nnx/diffusion` returns 1 line (`schedules.py`), and `uv run python -c "from nnx.diffusion.training import _extract"`
- [ ] Both diffusion paths train as before, seeded runs included. Check: `uv run pytest tests/test_diffusion_training.py tests/test_diffusion_objective.py tests/test_diffusion_sampling.py tests/test_diffusion_schedules.py`
- [ ] The CHANGELOG entry is under `[Unreleased]` → `### Added`. Check: `grep -n "NoiseSchedule.q_sample" CHANGELOG.md`
- [ ] The docs are in sync. Check: `uv run python -m scripts.docs.build_api_reference --check` and `uv run python -m scripts.docs.build_docs --check`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [schedules.py L28-L70](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/diffusion/schedules.py#L28-L70) | `NoiseSchedule` and `to` |
| [training.py L14](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/diffusion/training.py#L14), [L34-L50](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/diffusion/training.py#L34-L50) | the docstring mention, the import, `_extract` |
| [training.py L97-L105](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/diffusion/training.py#L97-L105) | the step's draws (global RNG) and noising |
| [objective.py L53](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/diffusion/objective.py#L53), [L172-L182](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/diffusion/objective.py#L172-L182) | the objective's import, draws (own generator) and noising |
| [api_reference_catalog.txt L939](https://github.com/thekaveh/NNx/blob/9ad461e/scripts/docs/api_reference_catalog.txt#L939) | `NoiseSchedule` is catalogued |
| #257, #561 | the diffusion objective; `output()` in both paths |
| #782 | the placement check shared with distinct messages (not touched) |

## Scope

**In:** `NoiseSchedule.q_sample`, the moved `_extract`, the two call sites, one test, the CHANGELOG line.

**Out:** the shared freeze helper (FEAT-112). The JEPA placement messages and component-state checks (deliberately separate, #782, #560). The step's and objective's random draws. The step's record fields (FIX-120).

## Dependencies

**Blocked by:** FIX-120
**Blocks:** none
**Related:** #257, #561, #782
