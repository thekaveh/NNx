"""Helpers for the #561 tests: modules whose forward returns a mapping, not a
tensor, read through a batch adapter's ``output()``.

The diffusion step, objective and sampler and both JEPA modes (online net and
EMA target) call the net through the model's batch adapter, as the supervised
step does. Each wrapper keeps its inner module under one attribute, so a
copy of the inner module's state compares weight for weight.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from nnx.models import PositionalInputs

# The default BatchAdapter.output() refusal names this remedy.
REMEDY = r"not a tensor; use a BatchAdapter whose output\(\) extracts the output tensor"


class DictDenoiser(nn.Module):
    """``forward(x_t, t) -> {"sample": eps}``, like diffusers' ``UNet2DModel``
    (a ``UNet2DOutput`` by default)."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"sample": self.inner(x, t)}


class DictViT(nn.Module):
    """The ViT patch contract JEPA needs (``n_patches``, ``patch_positions()``,
    ``forward(x, mask=...)``), returning ``{"tokens": ...}``."""

    def __init__(self, vit: nn.Module) -> None:
        super().__init__()
        self.vit = vit
        self.n_patches = vit.n_patches

    def patch_positions(self) -> torch.Tensor:
        return self.vit.patch_positions()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        return {"tokens": self.vit(x, mask=mask)}


class Extract(PositionalInputs):
    """The documented remedy: ``output()`` extracts the tensor under ``key``.
    ``grad_modes`` records whether autograd was on at each call, so a test
    can tell the online forward (on) from the EMA target's (off)."""

    def __init__(self, key: str, n_inputs: int = 1) -> None:
        super().__init__(n_inputs)
        self.key = key
        self.grad_modes: list[bool] = []

    def output(self, raw: Any) -> torch.Tensor:
        self.grad_modes.append(torch.is_grad_enabled())
        return raw[self.key]
