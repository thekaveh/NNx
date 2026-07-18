"""Net2WiderNet — function-preserving width expansion for ``nn.Linear``.

Reference: Chen, Goodfellow, Shlens — *Net2Net: Accelerating Learning
via Knowledge Transfer* (ICLR 2016).

The idea: to grow a Linear's ``out_features`` from ``k`` to ``k + q``,
pick ``q`` of the existing output units (with replacement) and copy
them — so the new layer's output rows include duplicates of the
original ones. By itself this changes nothing about the layer's
forward, but it does double-count the duplicated units in the *next*
Linear. Net2Net's trick is to divide each downstream incoming weight
column by the number of times its source unit appears in the new
output, restoring the original forward exactly:

    y_new = W_down_new @ x_new == W_down @ x == y_old

That equality is testable via :func:`torch.allclose` and is the
correctness contract every test for this primitive should check first.
"""

from __future__ import annotations

import copy
from typing import Optional, cast

import torch
from torch import nn
from torch.nn.utils import skip_init

from ._utils import get_module, set_module


def widen(
    model: nn.Module,
    *,
    layer_name: str,
    new_width: int,
    rng_seed: Optional[int] = 0,
) -> nn.Module:
    """Net2WiderNet: grow a Linear's ``out_features`` to ``new_width``.

    Returns a deep copy of ``model`` with the named layer expanded and
    the downstream Linear's ``in_features`` adjusted so the overall
    forward output is preserved exactly (within FP rounding).

    Args:
        model: any :class:`nn.Module`. The function deep-copies it so
            the caller's reference survives.
        layer_name: dotted name (as produced by ``named_modules()``) of
            the :class:`nn.Linear` to widen. Must be a Linear, must
            have an immediately downstream Linear, otherwise raises.
        new_width: desired ``out_features``. Must be strictly greater
            than the current ``out_features``.
        rng_seed: seed for the unit-duplication choices. Pass an int
            for deterministic surgery; ``None`` to seed the local
            generator non-deterministically (fresh entropy — the global
            torch RNG is never read or advanced). Defaults to ``0`` so
            the primitive is deterministic by default.

    Returns:
        A new :class:`nn.Module` (same class as ``model``) with the
        widened Linear in place. Forward output equals the original's
        within ``atol=1e-5`` (typically much tighter).

    Raises:
        KeyError: if ``layer_name`` is not a submodule of ``model``.
        TypeError: if the named submodule is not :class:`nn.Linear`.
        ValueError: if ``new_width`` is not strictly greater than the
            current ``out_features``, or if no downstream Linear exists.
    """
    new_model = copy.deepcopy(model)
    layer = get_module(new_model, layer_name)
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"widen target {layer_name!r} is {type(layer).__name__}, expected nn.Linear")
    cur = layer.out_features
    if new_width <= cur:
        raise ValueError(f"new_width must be > current out_features ({cur}); got {new_width}")
    q = new_width - cur

    # Pick q indices (with replacement) of existing units to duplicate.
    # A local generator keeps the surgery reproducible without touching
    # global RNG state.
    g = torch.Generator()
    if rng_seed is not None:
        g.manual_seed(int(rng_seed))
    else:
        g.seed()
    duplicates = torch.randint(0, cur, (q,), generator=g)

    # Track replication count per original index so we can divide the
    # downstream weight columns. Each "1" is the original column; each
    # appearance in `duplicates` adds one more.
    replication_count = torch.ones(cur, dtype=layer.weight.dtype, device=layer.weight.device)
    for idx in duplicates.tolist():
        replication_count[idx] += 1.0

    # --- Expand the target layer (in-out: in → cur+q) -----------------
    new_weight = torch.cat([layer.weight.data, layer.weight.data[duplicates]], dim=0)
    # skip_init: every param is fully overwritten below, so meta-device
    # construction avoids burning ambient RNG draws on a discarded init
    # (a seeded caller pipeline would otherwise silently diverge).
    new_layer = cast(
        nn.Linear,
        skip_init(
            nn.Linear,
            layer.in_features,
            new_width,
            bias=layer.bias is not None,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        ),
    )
    new_layer.weight.data.copy_(new_weight)
    if layer.bias is not None:
        new_bias = torch.cat([layer.bias.data, layer.bias.data[duplicates]], dim=0)
        assert new_layer.bias is not None
        new_layer.bias.data.copy_(new_bias)
    set_module(new_model, layer_name, new_layer)

    # --- Adjust the downstream Linear so the forward is preserved -----
    down_name, down_layer = _find_next_linear(new_model, layer_name)
    if down_layer.in_features != cur:
        raise ValueError(
            f"downstream Linear {down_name!r} has in_features={down_layer.in_features}, "
            f"expected {cur} to match {layer_name!r}'s old out_features. "
            "The two layers are not directly connected."
        )

    # New columns mirror columns from `duplicates`; rescale every column
    # by 1 / replication_count to preserve the forward sum.
    new_down_weight = torch.cat(
        [down_layer.weight.data, down_layer.weight.data[:, duplicates]],
        dim=1,
    )
    extended_rep = torch.cat([replication_count, replication_count[duplicates]])
    new_down_weight = new_down_weight / extended_rep.unsqueeze(0)

    new_down_layer = cast(
        nn.Linear,
        skip_init(
            nn.Linear,
            new_width,
            down_layer.out_features,
            bias=down_layer.bias is not None,
            device=down_layer.weight.device,
            dtype=down_layer.weight.dtype,
        ),
    )
    new_down_layer.weight.data.copy_(new_down_weight)
    if down_layer.bias is not None:
        # Bias is *additive after* W·x, so it doesn't change with the
        # column rescaling — copy as-is.
        assert new_down_layer.bias is not None
        new_down_layer.bias.data.copy_(down_layer.bias.data)
    set_module(new_model, down_name, new_down_layer)

    return new_model


def _find_next_linear(model: nn.Module, after_name: str) -> tuple[str, nn.Linear]:
    """Return the (name, module) of the first :class:`nn.Linear` that
    appears after ``after_name`` in module-traversal order.

    Module-traversal order is what :meth:`nn.Module.named_modules` uses,
    which for the standard NNx nets (``nn.Sequential``, ``FeedFwdNN``'s
    ``nn.ModuleList``) coincides with forward-pass order. This is the
    contract Net2WiderNet relies on.
    """
    seen_after = False
    for name, mod in model.named_modules():
        if seen_after and isinstance(mod, nn.Linear):
            return name, mod
        if name == after_name:
            seen_after = True
    raise ValueError(
        f"no downstream nn.Linear found after {after_name!r} — widen() needs a directly-connected Linear to rescale."
    )
