"""Legacy forward-pre-hook reparameterizations on a module. Internal.

An unbaked ``torch.nn.utils.prune`` mask and the legacy ``weight_norm`` /
``spectral_norm`` recompute a tensor (``weight`` from ``weight_orig`` and
``weight_mask``, or from ``weight_g`` / ``weight_v`` / ``weight_orig``) in a
forward pre-hook, so code that reads the tensor without calling the module, or
rebuilds the module, sees a stale value and drops the hook. Surgery, DoRA and
the training preflight recognize them here, by hook type, never by parameter
names.
"""

from __future__ import annotations

from torch import nn
from torch.nn.utils.prune import BasePruningMethod
from torch.nn.utils.spectral_norm import SpectralNorm
from torch.nn.utils.weight_norm import WeightNorm

# The name listed for a hook that does not record its tensor's name (torch's
# own hooks always do).
UNNAMED = "?"


def legacy_reparameterizations(module: nn.Module) -> tuple[list[str], list[str]]:
    """The sorted names of the tensors that ``module``'s own unbaked prune
    masks and legacy ``weight_norm`` / ``spectral_norm`` hooks recompute, as
    ``(masked, normed)``; :data:`UNNAMED` for a hook without a name."""
    masked: list[str] = []
    normed: list[str] = []
    for hook in module._forward_pre_hooks.values():
        if isinstance(hook, BasePruningMethod):
            masked.append(getattr(hook, "_tensor_name", UNNAMED))
        elif isinstance(hook, (WeightNorm, SpectralNorm)):
            normed.append(getattr(hook, "name", UNNAMED))
    return sorted(masked), sorted(normed)
