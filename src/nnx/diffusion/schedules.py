"""Diffusion noise schedules — DDPM-style variance schedules pre-computed
once at construction.

Provides two standard choices via the :class:`NoiseSchedulers` enum:

  - **LINEAR**: betas interpolate linearly from ``beta_min`` to ``beta_max``
    across T steps. The original DDPM paper schedule.
  - **COSINE**: alphas_cumprod follows a shifted cosine curve. Smoother
    near t=T than linear; common in modern follow-ups (Improved DDPM).

The schedule object is a frozen dataclass holding 1D tensors of length T
that the training step and sampler index by per-batch timesteps. Tensors
live on CPU by default; call :meth:`NoiseSchedule.to(device)` to migrate.

There's no ``state()`` / ``from_state()`` round-trip — the tensors are
fully derived from ``kind`` + ``T`` + kind-specific knobs, so recovering
the schedule from an on-disk run is a matter of re-instantiating the
enum with the same arguments rather than serializing the tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


@dataclass(frozen=True, slots=True)
class NoiseSchedule:
    """Precomputed DDPM noise schedule.

    All tensors are 1D of length ``T`` and live on the same device. The
    factory constructs them on CPU; :meth:`to` returns a new schedule
    with every tensor migrated.

    Attributes:
        kind: which enum variant produced this schedule (for introspection).
        T: number of diffusion timesteps.
        betas: per-step variance, ``shape=(T,)``.
        alphas: ``1 - betas``.
        alphas_cumprod: cumulative product of alphas (``ᾱ_t`` in the paper).
        sqrt_alphas_cumprod: ``√ᾱ_t`` — the x_0 coefficient in q(x_t | x_0).
        sqrt_one_minus_alphas_cumprod: ``√(1 - ᾱ_t)`` — the noise coefficient.
        posterior_variance: variance of q(x_{t-1} | x_t, x_0), used by the
            reverse-step sampler.
    """

    kind: NoiseSchedulers
    T: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor

    def to(self, device) -> NoiseSchedule:
        """Return a copy with every tensor moved to ``device``. The kind
        and T fields are unchanged."""
        return NoiseSchedule(
            kind=self.kind,
            T=self.T,
            betas=self.betas.to(device),
            alphas=self.alphas.to(device),
            alphas_cumprod=self.alphas_cumprod.to(device),
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod.to(device),
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod.to(device),
            posterior_variance=self.posterior_variance.to(device),
        )


def _from_betas(kind: NoiseSchedulers, betas: torch.Tensor) -> NoiseSchedule:
    """Build a full :class:`NoiseSchedule` from a 1D `betas` tensor —
    every other field is derived. Centralized so LINEAR and COSINE
    factories share the same derivation logic."""
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    # ``alphas_cumprod_prev`` is ᾱ_{t-1} with ᾱ_{-1} := 1 (DDPM convention).
    # Used only for the posterior variance.
    alphas_cumprod_prev = torch.cat(
        [torch.ones(1, dtype=betas.dtype), alphas_cumprod[:-1]]
    )
    # posterior_variance = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t). Clamped to
    # avoid the t=0 division by (1 - 1) — DDPM's standard "fix".
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    # The first entry corresponds to t=0, where alphas_cumprod_prev is 1
    # and the formula above gives 0 — fine, but we clamp for downstream
    # sqrt() safety against floating-point negatives near zero.
    posterior_variance = posterior_variance.clamp(min=1e-20)

    return NoiseSchedule(
        kind=kind,
        T=int(betas.shape[0]),
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        sqrt_alphas_cumprod=alphas_cumprod.sqrt(),
        sqrt_one_minus_alphas_cumprod=(1.0 - alphas_cumprod).sqrt(),
        posterior_variance=posterior_variance,
    )


class NoiseSchedulers(Enum):
    """Diffusion noise-schedule factory. Enum-as-factory pattern (like
    :class:`nnx.Nets`, :class:`nnx.Optims`): each enum variant's
    ``__call__`` constructs the underlying :class:`NoiseSchedule`."""

    LINEAR = "linear"
    COSINE = "cosine"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return str(self)

    def __call__(
        self,
        T: int = 1000,
        *,
        beta_min: float = 1e-4,
        beta_max: float = 2e-2,
        s: float = 0.008,
    ) -> NoiseSchedule:
        """Build a :class:`NoiseSchedule`.

        Args:
            T: number of diffusion timesteps. Larger T means more steps
                during training (one t per batch) and at sampling time
                (T-many reverse passes); 1000 is the DDPM default.
            beta_min: LINEAR schedule lower endpoint; ignored for COSINE.
            beta_max: LINEAR schedule upper endpoint; ignored for COSINE.
            s: COSINE schedule offset (Improved DDPM eq. 17); ignored for LINEAR.

        Returns:
            A :class:`NoiseSchedule` on CPU. Call ``.to(device)`` after
            construction to migrate; the diffusion train step does this
            implicitly when indexing the schedule tensors with batch
            timesteps on the target device.
        """
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        match self:
            case NoiseSchedulers.LINEAR:
                if not (0 < beta_min < beta_max < 1):
                    raise ValueError(
                        f"LINEAR schedule needs 0 < beta_min ({beta_min}) "
                        f"< beta_max ({beta_max}) < 1"
                    )
                betas = torch.linspace(beta_min, beta_max, T, dtype=torch.float32)
                return _from_betas(self, betas)
            case NoiseSchedulers.COSINE:
                if s <= 0:
                    raise ValueError(f"COSINE schedule needs s > 0, got {s}")
                # Improved DDPM eq. 17. Build ᾱ_t first, then back out the
                # betas. Clamp betas to [1e-8, 0.999] so the sqrt() and
                # posterior_variance ops can't blow up near t=T-1.
                steps = torch.arange(T + 1, dtype=torch.float64) / T
                f_t = torch.cos((steps + s) / (1 + s) * torch.pi / 2) ** 2
                alphas_cumprod = (f_t / f_t[0]).to(torch.float32)
                betas = (1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]).clamp(
                    min=1e-8, max=0.999,
                )
                return _from_betas(self, betas)
