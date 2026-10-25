"""Helpers for tests that apply surgery outside any recipe (#508)."""

from __future__ import annotations

from typing import Union

import torch

from nnx import low_rank_factorize
from nnx.prune import magnitude_prune

# Every unrecorded surgery form whose checkpoints NNModel.from_checkpoint could not rebuild.
SURGERY_FORMS = ["low_rank", "prune", "weight_norm", "spectral_norm", "legacy_weight_norm", "legacy_spectral_norm"]


def unrecorded_surgery(
    net: torch.nn.Module, layers: Union[torch.nn.Sequential, torch.nn.ModuleList], form: str
) -> None:
    """Change ``net`` in place, outside any recipe, so no checkpoint could rebuild it.

    ``layers`` is the indexable container in ``net`` whose first layer is
    replaced or reparameterized; ``prune`` masks every layer of ``net``.
    """
    if form == "low_rank":
        layers[0] = low_rank_factorize(layers[0], rank=2)
    elif form == "prune":
        magnitude_prune(net, 0.5, bake=False)
    elif form.startswith("legacy_"):
        getattr(torch.nn.utils, form.removeprefix("legacy_"))(layers[0])
    else:
        getattr(torch.nn.utils.parametrizations, form)(layers[0])
