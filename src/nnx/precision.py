"""Explicit FP32 / FP16 / BF16 execution policy (FEAT-028).

``NNModelParams.precision`` takes a :class:`PrecisionPolicy`: the precision
training, evaluation and prediction run in. A policy is *resolved* against
the device it runs on — once per run, before the run is reserved — into a
:class:`ResolvedPrecision`, which says what is requested, what runs, and
why a fallback happened:

========  ===========================================  ==================
mode      runs as                                      where
========  ===========================================  ==================
``fp32``  full precision (no autocast, no scaler)      every device
``fp16``  float16 autocast + a ``GradScaler``          CUDA
``bf16``  bfloat16 autocast, no scaler                 CPU; CUDA devices
                                                       with bf16 support
========  ===========================================  ==================

A request the device cannot run fails before any run is reserved, unless
the policy names ``fallback="fp32"``: the run then trains in full precision
and records why. The model's parameters stay float32 in every mode
(autocast only — never ``model.half()``), the backward pass runs outside
autocast, and the update keeps its unscale → normalize → finite-check →
clip → step order.

The legacy ``NNModelParams(mixed_precision=True)`` keeps its meaning — FP16
autocast with a scaler on CUDA, silently full precision elsewhere — and its
run id; combining it with a policy other than ``fp16`` is refused.

The policy covers ``NNModel.train`` (the default step and objectives),
``Trainer.train(objective=...)``, ``evaluate()`` and prediction. It does
not cover :mod:`nnx.lr_finder`, :mod:`nnx.diffusion.sampling` or
generation, and TF32 is reported (never set) as separate metadata.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

import torch

__all__ = [
    "PRECISION_MODES",
    "REFERENCE_TOLERANCES",
    "PrecisionPolicy",
    "PrecisionUnsupportedError",
    "ReferenceTolerance",
    "ResolvedPrecision",
    "precision_support",
    "resolve_precision",
]

PrecisionMode = Literal["fp32", "fp16", "bf16"]
PrecisionFallback = Literal["error", "fp32"]
PRECISION_MODES: tuple[str, ...] = ("fp32", "fp16", "bf16")
PRECISION_FALLBACKS: tuple[str, ...] = ("error", "fp32")

_AUTOCAST_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
# What the policy applies to, and what it never reaches (run inspection).
COVERS: tuple[str, ...] = (
    "NNModel.train (default step and objectives)",
    "Trainer.train(objective=...)",
    "evaluate()",
    "predict() / predict_proba()",
)
NOT_COVERED: tuple[str, ...] = ("nnx.lr_finder", "nnx.diffusion.sampling", "generation")


class PrecisionUnsupportedError(ValueError):
    """A precision the device cannot run, with no fallback chosen — or a
    step function that cannot apply a reduced precision."""


# Marks a step function that runs in full precision only (the imperative
# paradigm steps built on ``finalize_step``): a reduced-precision run
# refuses it before any work is done.
FULL_PRECISION_ONLY = "__nnx_full_precision_only__"


def full_precision_only(step: Any) -> Any:
    """Mark ``step`` as running in full precision only and return it."""
    setattr(step, FULL_PRECISION_ONLY, True)
    return step


@dataclass(frozen=True, slots=True)
class ReferenceTolerance:
    """How far a reduced-precision run may drift from its FP32 reference
    on NNx's seeded fixtures: the largest absolute difference of a
    per-epoch training loss, and of any trained parameter."""

    loss: float
    weights: float


# Predeclared FP32-reference tolerances (FEAT-028). The CPU BF16 cell is
# verified against them by NNx's own test fixtures; a cell whose hardware
# is absent reports "unverified" (see precision_support), never support.
REFERENCE_TOLERANCES: Mapping[str, ReferenceTolerance] = {
    "fp16": ReferenceTolerance(loss=5e-3, weights=5e-4),
    "bf16": ReferenceTolerance(loss=5e-2, weights=5e-3),
}


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    """The requested execution precision.

    Args:
        mode: ``"fp32"`` (the default), ``"fp16"`` or ``"bf16"``.
        fallback: what happens when the device cannot run ``mode``:
            ``"error"`` (the default) fails before any run is reserved;
            ``"fp32"`` runs in full precision and records why.
    """

    mode: PrecisionMode = "fp32"
    fallback: PrecisionFallback = "error"

    def __post_init__(self) -> None:
        if self.mode not in PRECISION_MODES:
            raise ValueError(f"PrecisionPolicy.mode must be one of {list(PRECISION_MODES)}, got {self.mode!r}")
        if self.fallback not in PRECISION_FALLBACKS:
            raise ValueError(
                f"PrecisionPolicy.fallback must be one of {list(PRECISION_FALLBACKS)}, got {self.fallback!r}"
            )

    def state(self) -> dict[str, str]:
        """``{"mode": ...}``, plus ``"fallback"`` when it is not the
        default ``"error"``."""
        state = {"mode": str(self.mode)}
        if self.fallback != "error":
            state["fallback"] = str(self.fallback)
        return state

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> PrecisionPolicy:
        unknown = set(state) - {"mode", "fallback"}
        if unknown:
            raise ValueError(f"unknown PrecisionPolicy fields {sorted(unknown)}")
        return cls(mode=state["mode"], fallback=state.get("fallback", "error"))

    def resolve(self, device: Union[str, torch.device]) -> ResolvedPrecision:
        """What this policy runs as on ``device`` (see the module table)."""
        return _resolve(self.mode, self.fallback, device, source="policy")


def _tf32() -> dict[str, bool]:
    """TF32 as torch is configured — reported beside the policy, never set
    by it."""
    return {
        "cuda_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn": bool(torch.backends.cudnn.allow_tf32),
    }


@dataclass(frozen=True, slots=True)
class ResolvedPrecision:
    """A policy resolved against a device: what was requested, what runs,
    and why a fallback happened.

    ``autocast()`` is the forward context, ``build_scaler()`` the
    ``GradScaler`` (FP16 only) and ``record()`` the run-inspection
    mapping (``NNRun.precision``)."""

    requested: str
    effective: str
    device_type: str
    source: str = "default"
    fallback_reason: Optional[str] = None
    tf32: Mapping[str, bool] = field(default_factory=_tf32)

    @property
    def autocast_dtype(self) -> Optional[torch.dtype]:
        return _AUTOCAST_DTYPES.get(self.effective)

    @property
    def reduced(self) -> bool:
        """Whether anything runs below full precision."""
        return self.effective != "fp32"

    @property
    def uses_scaler(self) -> bool:
        return self.effective == "fp16"

    def autocast(self) -> contextlib.AbstractContextManager[Any]:
        """The forward-pass context: autocast to the effective dtype, or
        nothing in full precision."""
        dtype = self.autocast_dtype
        if dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=dtype)

    def build_scaler(self) -> Optional[Any]:
        """A fresh ``GradScaler`` for FP16, ``None`` otherwise."""
        return torch.amp.GradScaler(self.device_type) if self.uses_scaler else None

    def output(self, tensor: torch.Tensor) -> torch.Tensor:
        """``tensor`` as a caller sees it: a reduced-precision floating
        output of an autocast forward is returned as float32, so outputs
        keep their full-precision schema (and BF16 converts to a dtype
        NumPy can hold)."""
        if self.autocast_dtype is not None and tensor.dtype in (torch.float16, torch.bfloat16):
            return tensor.float()
        return tensor

    def record(self) -> dict[str, Any]:
        """The run-inspection mapping: requested and effective precision,
        the fallback reason, TF32 (separately), and what the policy covers
        — never :mod:`nnx.lr_finder` or :mod:`nnx.diffusion.sampling`."""
        return {
            "requested": self.requested,
            "effective": self.effective,
            "source": self.source,
            "device": self.device_type,
            "fallback_reason": self.fallback_reason,
            "autocast_dtype": None if self.autocast_dtype is None else str(self.autocast_dtype).replace("torch.", ""),
            "grad_scaler": self.uses_scaler,
            "tf32": dict(self.tf32),
            "covers": list(COVERS),
            "not_covered": list(NOT_COVERED),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ResolvedPrecision:
        return cls(
            requested=str(record["requested"]),
            effective=str(record["effective"]),
            device_type=str(record["device"]),
            source=str(record.get("source", "default")),
            fallback_reason=record.get("fallback_reason"),
            tf32=dict(record.get("tf32") or {}),
        )


def _device_type(device: Union[str, torch.device, Any]) -> str:
    if isinstance(device, torch.device):
        return device.type
    return torch.device(str(device)).type


def _unsupported(mode: str, device_type: str) -> Optional[str]:
    """Why ``device_type`` cannot run ``mode`` (``None`` when it can)."""
    if mode == "fp32":
        return None
    if mode == "fp16":
        if device_type == "cuda" and torch.cuda.is_available():
            return None
        return f"fp16 runs as float16 autocast with a GradScaler on CUDA only; this device is {device_type!r}"
    if device_type == "cpu":
        return None
    if device_type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return None
    return (
        f"bf16 runs on CPU and on CUDA devices with bf16 support (torch.cuda.is_bf16_supported()); this device "
        f"is {device_type!r}"
    )


def _resolve(mode: str, fallback: str, device: Any, *, source: str) -> ResolvedPrecision:
    device_type = _device_type(device)
    reason = _unsupported(mode, device_type)
    if reason is None:
        return ResolvedPrecision(requested=mode, effective=mode, device_type=device_type, source=source)
    if fallback == "fp32":
        return ResolvedPrecision(
            requested=mode, effective="fp32", device_type=device_type, source=source, fallback_reason=reason
        )
    raise PrecisionUnsupportedError(
        f"{reason}. Use PrecisionPolicy({mode!r}, fallback='fp32') to train in full precision instead, or another mode"
    )


def resolve_precision(params: Any, device: Union[str, torch.device, Any]) -> ResolvedPrecision:
    """The precision a model's parameters (``NNModelParams``) run in on
    ``device``: its policy, the legacy ``mixed_precision`` flag (FP16 on
    CUDA, full precision elsewhere — its historical meaning), or full
    precision."""
    policy = getattr(params, "precision", None)
    if policy is not None:
        return policy.resolve(device)
    if getattr(params, "mixed_precision", False):
        device_type = _device_type(device)
        if device_type == "cuda":
            return ResolvedPrecision(requested="fp16", effective="fp16", device_type=device_type, source="legacy")
        return ResolvedPrecision(
            requested="fp16",
            effective="fp32",
            device_type=device_type,
            source="legacy",
            fallback_reason=f"mixed_precision=True applies on CUDA only; this device is {device_type!r}",
        )
    return ResolvedPrecision(requested="fp32", effective="fp32", device_type=_device_type(device))


def precision_support(device: Union[str, torch.device, None] = None) -> dict[str, dict[str, str]]:
    """The precision matrix as this host sees it: for each device type
    (``"cpu"``, ``"cuda"``, ``"mps"``, or just ``device``'s), each mode's
    status —

    - ``"verified"``: NNx's seeded reference fixtures hold it within
      :data:`REFERENCE_TOLERANCES` (CPU FP32 and BF16);
    - ``"supported"``: the hardware is present and the mode resolves, but
      no NNx fixture vouches for it here;
    - ``"unverified"``: the hardware is absent, so nothing is claimed;
    - ``"unsupported"``: the device type cannot run the mode.
    """
    present = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    }
    device_types = [_device_type(device)] if device is not None else list(present)
    verified = {("cpu", "fp32"), ("cpu", "bf16")}
    matrix: dict[str, dict[str, str]] = {}
    for device_type in device_types:
        row: dict[str, str] = {}
        for mode in PRECISION_MODES:
            if (device_type, mode) in verified:
                row[mode] = "verified"
            elif device_type == "mps" and mode != "fp32" or device_type == "cpu" and mode == "fp16":
                row[mode] = "unsupported"
            elif not present.get(device_type, False):
                row[mode] = "unverified"
            else:
                row[mode] = "supported" if _unsupported(mode, device_type) is None else "unsupported"
        matrix[device_type] = row
    return matrix
