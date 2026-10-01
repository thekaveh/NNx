"""DDPM noise prediction as an objective (FEAT-040).

:func:`diffusion_train_step_factory` is an imperative step: it draws the
timesteps and noise from the global RNG, takes a mean ``F.mse_loss`` and
hands it to ``finalize_step``, which owns the optimizer step — so it cannot
accumulate gradients over uneven microbatches or share the update engine.
:func:`diffusion_objective` describes the same loss as an objective
(``nnx.objectives``) and leaves backward, accumulation, mixed precision,
clipping and the optimizer step to NNx's shared update engine::

    model.train(params=NNTrainParams(...), objective=diffusion_objective(schedule, seed=0))

For each microbatch the objective returns one ``"noise_mse"`` loss term:
the **sum** of squared errors between the predicted and the true noise
over every element, normalized by the element count. Over an update
window the engine divides the summed errors by the summed counts, so
microbatches ``[2, 1]`` take exactly the update of the full batch of 3 for
the same timesteps and noise.

The timesteps and noise come from the objective's **own** generator (a CPU
``torch.Generator``, seeded from ``seed`` or, when ``None``, from one draw
of the global RNG at first use). Its state is checkpointed component state
(``"diffusion.objective"``), so a stateful warm resume continues the
stream, and a :func:`~nnx.diffusion.sample` preview with its own generator
never perturbs it. Each run starts the stream afresh (a stateful resume
then restores the saved one), so an instance reused for a second, equally
seeded run draws the same noise. The draws happen on the CPU — the stream
is then the same on every device and survives a resume onto another — at
the cost of a host-to-device copy per microbatch; pass
``noise_fn(x_0, generator) -> (t, eps)`` to supply the timesteps and noise
yourself (fixed values in tests, another timestep distribution). The
``generator`` it receives is the objective's CPU generator: a ``noise_fn``
that draws on a CUDA device for large image batches must use its own CUDA
generator or the global RNG instead (which a stateful resume restores, but
which a ``sample`` preview without its own generator would advance). Its
timesteps must be int64 (or int32) values in ``[0, T)``.
"""

from __future__ import annotations

import hashlib
import numbers
from collections.abc import Callable, Mapping
from typing import Any, Optional

import torch

from .._step_helpers import first_input, full_precision
from ..components import ComponentSpec
from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from ..objectives import LossTerm, Objective, ObjectiveContext, ObjectiveResult
from .schedules import NoiseSchedule
from .training import _extract

__all__ = ["DiffusionObjective", "diffusion_objective"]

NoiseFn = Callable[[torch.Tensor, torch.Generator], "tuple[torch.Tensor, torch.Tensor]"]

_TERM = "noise_mse"


def _schedule_fingerprint(schedule: NoiseSchedule) -> str:
    """A stable identity for a schedule's betas (its other tensors derive
    from them)."""
    betas = schedule.betas.detach().to("cpu", torch.float64).contiguous()
    return hashlib.sha256(betas.numpy().tobytes()).hexdigest()


class DiffusionObjective(Objective):
    """DDPM noise prediction as an objective — see the module docstring.

    Callable as an objective and a checkpointable component (FEAT-005)
    named ``"diffusion.objective"``: its state is the objective spec (the
    schedule's length and fingerprint, the loss term) and the generator's
    state, so a stateful resume draws the timesteps and noise an
    uninterrupted run would have drawn. A resume with a different schedule
    is rejected before anything is restored.

    Args:
        schedule: the :class:`NoiseSchedule` (any device; the coefficients
            are indexed on the model's device).
        seed: seeds the objective's generator; ``None`` draws one seed from
            the global RNG at first use (reproducible under ``set_seed``).
        noise_fn: optional ``(x_0, generator) -> (t, eps)``: ``t`` a
            ``LongTensor[B]`` in ``[0, T)``, ``eps`` shaped like ``x_0``.
            Defaults to ``t ~ Uniform{0..T-1}``, ``eps ~ N(0, I)`` from the
            objective's generator.
        nonfinite: the engine's non-finite policy (``"fail"`` / ``"skip"``).
    """

    def __init__(
        self,
        schedule: NoiseSchedule,
        *,
        seed: Optional[int] = None,
        noise_fn: Optional[NoiseFn] = None,
        nonfinite: str = "fail",
    ) -> None:
        super().__init__(nonfinite=nonfinite)
        if not isinstance(schedule, NoiseSchedule):
            raise TypeError(f"schedule must be a NoiseSchedule, got {type(schedule).__name__}")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or not 0 <= int(seed) < 2**64
        ):
            raise ValueError(f"seed must be an integer in [0, 2**64) or None, got {seed!r}")
        if noise_fn is not None and not callable(noise_fn):
            raise TypeError(f"noise_fn must be callable, got {type(noise_fn).__name__}")
        self.schedule = schedule
        self.seed = None if seed is None else int(seed)
        self.noise_fn = noise_fn
        self._generator: Optional[torch.Generator] = None
        self._fingerprint = _schedule_fingerprint(schedule)

    # ---------- before each run ----------

    def check_run(self, model: Any, *, optimizers: Mapping[str, torch.optim.Optimizer], callbacks: Any) -> None:
        """A new run starts the noise stream afresh — from ``seed``, or from
        one draw of the (just seeded) global RNG — whatever an earlier run of
        this instance drew; a stateful resume then restores the saved stream."""
        self._generator = None

    # ---------- the objective ----------

    @property
    def generator(self) -> torch.Generator:
        """The objective's own CPU generator (created at first use)."""
        if self._generator is None:
            seed = self.seed
            if seed is None:
                seed = int(torch.randint(0, 2**62, (1,)).item())  # one draw: reproducible under set_seed
            self._generator = torch.Generator().manual_seed(seed)
        return self._generator

    def draw(self, x_0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """This microbatch's timesteps and noise, on ``x_0``'s device."""
        if self.noise_fn is not None:
            t, eps = self.noise_fn(x_0, self.generator)
        else:
            # Drawn on the CPU generator, then moved: the stream is the same on
            # every device, so a resume may change devices.
            t = torch.randint(0, self.schedule.T, (x_0.shape[0],), generator=self.generator)
            eps = torch.randn(x_0.shape, generator=self.generator, dtype=x_0.dtype)
        t, eps = t.to(x_0.device), eps.to(x_0.device)
        if t.shape != (x_0.shape[0],) or eps.shape != x_0.shape:
            raise ValueError(
                f"noise_fn must return t of shape ({x_0.shape[0]},) and eps of shape {tuple(x_0.shape)}; got "
                f"{tuple(t.shape)} and {tuple(eps.shape)}"
            )
        if self.noise_fn is not None:
            # Checked before t indexes the schedule (an out-of-range index is a
            # device-side assert on CUDA).
            # int32 / int64 only: uint8 would index as a boolean mask.
            if t.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"noise_fn must return int64 (or int32) timesteps, got {t.dtype}")
            if t.numel() and bool(((t < 0) | (t >= self.schedule.T)).any()):  # one sync
                raise ValueError(f"noise_fn returned timesteps outside [0, {self.schedule.T})")
            t = t.long()
        return t, eps

    def __call__(self, ctx: ObjectiveContext) -> ObjectiveResult:
        model = ctx.model
        model.net.train()
        x_0 = first_input(model, ctx.batch, who="diffusion_objective")
        if x_0.shape[0] == 0 or x_0.numel() == 0:
            raise ValueError("diffusion_objective got an empty batch: there is no noise to predict")
        t, eps = self.draw(x_0)
        sqrt_a = _extract(self.schedule.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_1ma = _extract(self.schedule.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        x_t = sqrt_a * x_0 + sqrt_1ma * eps
        eps_pred = model.net(x_t, t)
        error = full_precision(eps_pred) - full_precision(eps)
        term = LossTerm(_TERM, error.pow(2).sum(), error.numel(), "mean")
        value = term.value
        assert value is not None  # a non-empty batch always has elements
        # Detached metrics only: no classification fields for a generative loss.
        return ObjectiveResult((term,), NNEvaluationDataPoint(loss=value, metrics={_TERM: value}))

    # ---------- checkpointable component (FEAT-005) ----------

    def _spec(self) -> dict[str, Any]:
        return {"objective": "diffusion", "loss": _TERM, "T": int(self.schedule.T), "schedule": self._fingerprint}

    def component_spec(self) -> ComponentSpec:
        return ComponentSpec("diffusion.objective", version=1)

    def component_state(self) -> dict[str, Any]:
        state = self._generator.get_state() if self._generator is not None else None
        return {"spec": self._spec(), "generator": state}

    def check_component_state(self, state: Mapping[str, Any], *, version: int) -> list[str]:
        spec = state.get("spec")
        if spec != self._spec():
            return [
                f"the checkpoint's diffusion objective {spec!r} does not match this one {self._spec()!r}: resume "
                "with the same schedule"
            ]
        generator = state.get("generator")
        if generator is not None and not (isinstance(generator, torch.Tensor) and generator.dtype == torch.uint8):
            return [f"malformed diffusion objective generator state {type(generator).__name__}"]
        return []

    def load_component_state(self, state: Mapping[str, Any], *, version: int) -> None:
        generator = state["generator"]
        if generator is None:
            self._generator = None
            return
        restored = torch.Generator()
        restored.set_state(generator)
        self._generator = restored


def diffusion_objective(
    schedule: NoiseSchedule,
    *,
    seed: Optional[int] = None,
    noise_fn: Optional[NoiseFn] = None,
    nonfinite: str = "fail",
) -> DiffusionObjective:
    """DDPM noise prediction as an objective (see :class:`DiffusionObjective`)
    — the objective counterpart of :func:`diffusion_train_step_factory`,
    which stays available (and unchanged) as the imperative step."""
    return DiffusionObjective(schedule, seed=seed, noise_fn=noise_fn, nonfinite=nonfinite)
