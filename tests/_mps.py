"""CPU stand-ins for an MPS tensor (FEAT-050), so a Linux lane pins each MPS fix.

MPS has no float64 and no NumPy view: NNx widens a tensor that may live on
the model's device on the CPU (``.cpu()`` first). On a CPU-only host that hop
is a no-op, so reverting it passes every test that does not run on MPS.
:class:`OffCPU` refuses what an MPS tensor refuses, and its ``.cpu()`` /
``.to("cpu")`` return a plain CPU tensor, so a pin fails without the hop.
:class:`ReportsMPS` also reports ``device`` ``mps``, for a branch that
checks ``tensor.device.type == "mps"``.

These cover only the float64 and NumPy refusals. Code that builds a new
tensor on ``device=tensor.device`` needs a real MPS device: those paths run
in the opt-in ``mps-evidence`` job (CONTRIBUTING §5.2).
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch

#: Set by the ``mps-evidence`` job: an ``mps``-marked test then fails, not skips, without MPS.
REQUIRE_MPS = "NNX_REQUIRE_MPS"
FLOAT64_ERROR = (
    "Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. "
    "Please use float32 instead."
)
NUMPY_ERROR = (
    "can't convert mps:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first."
)


def mps_skip_reason() -> str | None:
    """Why an ``mps``-marked test cannot run here, or ``None`` with a device."""
    if torch.backends.mps.is_available():
        return None
    return f"needs an MPS device (torch.backends.mps.is_built() is {torch.backends.mps.is_built()})"


def mps_or_skip() -> None:
    """Skip an ``mps``-marked test without an MPS device (``conftest.py``
    marks it at collection, so ``-rs`` names the test); under
    ``NNX_REQUIRE_MPS=1`` fail it instead, so the ``mps-evidence`` job
    cannot pass by skipping."""
    reason = mps_skip_reason()
    if reason is None:
        return
    if os.environ.get(REQUIRE_MPS) == "1":
        pytest.fail(f"{REQUIRE_MPS}=1 and {reason}", pytrace=False)
    pytest.skip(reason)


def _names_cpu(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    """Whether a ``Tensor.to`` call names the CPU as its target device."""
    for value in (*args, kwargs.get("device")):
        if isinstance(value, (str, torch.device)) and torch.device(value).type == "cpu":
            return True
    return False


def _has_float64(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.dtype == torch.float64
    if isinstance(value, (tuple, list)):
        return any(_has_float64(item) for item in value)
    return False


class OffCPU(torch.Tensor):
    """A CPU tensor that refuses float64 and NumPy, as an MPS tensor does.

    Every op on it returns an ``OffCPU``; ``.cpu()`` and ``.to("cpu")``
    return a plain tensor (the device hop the fixes make)."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.Tensor.cpu or (func is torch.Tensor.to and _names_cpu(args[1:], kwargs)):
            with torch._C.DisableTorchFunctionSubclass():
                return func(args[0].as_subclass(torch.Tensor), *args[1:], **kwargs)
        if func in (torch.Tensor.numpy, torch.Tensor.__array__):
            raise TypeError(NUMPY_ERROR)
        result = super().__torch_function__(func, types, args, kwargs)
        if _has_float64(result):
            raise TypeError(FLOAT64_ERROR)
        return result


class ReportsMPS(OffCPU):
    """An :class:`OffCPU` tensor whose ``device`` is ``mps``."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # ==, not is: each attribute access builds a new method-wrapper.
        if func == torch.Tensor.device.__get__:  # type: ignore[attr-defined]
            return torch.device("mps")
        return super().__torch_function__(func, types, args, kwargs)


def off_cpu(tensor: torch.Tensor) -> OffCPU:
    """``tensor`` as an :class:`OffCPU` (shares its storage)."""
    return tensor.as_subclass(OffCPU)  # type: ignore[return-value]


def reports_mps(tensor: torch.Tensor) -> ReportsMPS:
    """``tensor`` as a :class:`ReportsMPS` (shares its storage)."""
    return tensor.as_subclass(ReportsMPS)  # type: ignore[return-value]


class FakeMPSGenerator:
    """``torch.mps``'s generator as one fake state, for a host without MPS.

    :func:`fake_mps_generator` reports MPS available and routes
    ``torch.mps.get_rng_state`` / ``set_rng_state`` / ``manual_seed`` here;
    :meth:`draw` advances the state as an MPS draw (dropout on an MPS net)
    would."""

    def __init__(self) -> None:
        self.state = torch.tensor([0, 1, 2, 3], dtype=torch.uint8)
        self.seeds: list[int] = []

    def get_rng_state(self, device: Any = "mps") -> torch.Tensor:
        return self.state.clone()

    def set_rng_state(self, new_state: torch.Tensor, device: Any = "mps") -> None:
        self.state = new_state.clone()

    def manual_seed(self, seed: int) -> None:
        self.seeds.append(int(seed))
        self.state = torch.full((4,), int(seed) % 256, dtype=torch.uint8)

    def draw(self) -> None:
        self.state = self.state + 1


def fake_mps_generator(monkeypatch: Any) -> FakeMPSGenerator:
    """Install a :class:`FakeMPSGenerator` for one test (``monkeypatch``)."""
    # Dynamo reads torch.backends.mps.is_available.__wrapped__ when it is first
    # imported (torch >= 2.15): import it before the stand-in, which has none.
    import torch._dynamo  # noqa: F401

    generator = FakeMPSGenerator()
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    for name in ("get_rng_state", "set_rng_state", "manual_seed"):
        monkeypatch.setattr(torch.mps, name, getattr(generator, name))
    return generator
