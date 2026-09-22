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

from ..nn.enum.activations import Activations
from ..nn.net.feed_fwd_nn import FeedFwdNN
from ._utils import copy_param_roles, get_module, set_module


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
            the :class:`nn.Linear` to widen. Must be a Linear whose
            consumer can be *proven*: the target lives directly inside an
            ``nn.Sequential`` (the following siblings are walked) or in
            ``FeedFwdNN.layers`` (the effective per-layer activation /
            dropout is resolved), and every op between it and the next
            Linear is elementwise — the built-in activations except
            Softmax, ``Dropout``, ``Identity``. Anything else raises
            before any allocation (see ``Raises``).
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
        within ``atol=1e-5`` (typically much tighter) — in eval mode when
        a ``Dropout`` sits between the target and its consumer, since a
        stochastic training forward is not identical by construction.

    Raises:
        KeyError: if ``layer_name`` is not a submodule of ``model``.
        TypeError: if the named submodule is not :class:`nn.Linear`.
        ValueError: if ``new_width`` is not strictly greater than the
            current ``out_features``; if no downstream Linear exists; if
            a width-dependent op (Softmax, LayerNorm, BatchNorm, or any
            module outside the elementwise allowlist) sits between the
            target and its consumer; if the target lives in a container
            other than ``nn.Sequential`` / ``FeedFwdNN.layers`` (module
            registration order is not data flow); or if the target or
            its consumer is registered under more than one path (an
            alias used in several places). Validation runs on the source
            before anything is copied or allocated, so a rejected call
            leaves the model, its module identities and the RNG untouched.
    """
    source_layer = get_module(model, layer_name)
    if not isinstance(source_layer, nn.Linear):
        raise TypeError(f"widen target {layer_name!r} is {type(source_layer).__name__}, expected nn.Linear")
    cur = source_layer.out_features
    if new_width <= cur:
        raise ValueError(f"new_width must be > current out_features ({cur}); got {new_width}")
    # Prove the target→consumer path on the SOURCE before copying or
    # allocating anything (FIX-005): a rejected call leaves the model,
    # its module identities and the RNG untouched.
    down_name = _resolve_consumer(model, layer_name, source_layer)
    q = new_width - cur

    new_model = copy.deepcopy(model)
    layer = cast(nn.Linear, get_module(new_model, layer_name))

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
    # The replacement keeps the target's own trainability roles and mode
    # (skip_init defaults to trainable / train mode) — FIX-016.
    copy_param_roles(layer, new_layer)
    set_module(new_model, layer_name, new_layer)

    # --- Adjust the downstream Linear so the forward is preserved -----
    down_layer = cast(nn.Linear, get_module(new_model, down_name))
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
    # Independent of the target: the consumer keeps ITS OWN flags/mode.
    copy_param_roles(down_layer, new_down_layer)
    set_module(new_model, down_name, new_down_layer)

    return new_model


# Ops through which duplicating a unit and rescaling its consumer's
# incoming columns leaves the forward unchanged: they act per element and
# never mix units. Softmax (denominator over the width), LayerNorm and
# BatchNorm (statistics over the width) do mix units and are rejected.
_ELEMENTWISE_MODULES: tuple[type[nn.Module], ...] = (
    nn.ReLU,
    nn.LeakyReLU,
    nn.GELU,
    nn.Tanh,
    nn.Sigmoid,
    nn.ELU,
    nn.SELU,
    nn.Softplus,
    nn.SiLU,
    nn.Identity,
    nn.Dropout,
)
_WIDTH_DEPENDENT_ACTIVATIONS = frozenset({Activations.SOFTMAX})


def _registered_paths(model: nn.Module, target: nn.Module) -> list[str]:
    """Every qualified path under which ``target`` is registered
    (``named_modules`` deduplicates by default; an alias means the same
    module is used in more than one place)."""
    return [name for name, mod in model.named_modules(remove_duplicate=False) if mod is target]


def _resolve_consumer(model: nn.Module, layer_name: str, layer: nn.Linear) -> str:
    """Return the qualified name of the Linear that consumes ``layer``'s
    output, or raise ``ValueError`` when the path cannot be proven to be
    function-preserving (FIX-005).

    Only two containers prove execution order: an ``nn.Sequential`` (the
    siblings after the target run in registration order) and
    ``FeedFwdNN.layers`` (``forward`` applies ``activation_for(i)`` and
    ``dropout_for(i)`` between ``layers[i]`` and ``layers[i + 1]``). In a
    generic module, registration order is not data flow, so the
    consumer cannot be identified safely and the call is rejected.
    """
    aliases = _registered_paths(model, layer)
    if len(aliases) > 1:
        raise ValueError(
            f"widen target {layer_name!r} is registered under multiple paths {aliases}: a shared (aliased) "
            "module is used in more than one place, so widening one path cannot preserve the function"
        )
    parent_path, _, attr = layer_name.rpartition(".")
    parent = model if not parent_path else get_module(model, parent_path)
    allowed = ", ".join(t.__name__ for t in _ELEMENTWISE_MODULES)

    if isinstance(parent, nn.Sequential):
        keys = list(parent._modules)
        consumer_name: Optional[str] = None
        for key in keys[keys.index(attr) + 1 :]:
            mod = parent._modules[key]
            name = f"{parent_path}.{key}" if parent_path else key
            if isinstance(mod, nn.Linear):
                consumer_name = name
                break
            if not isinstance(mod, _ELEMENTWISE_MODULES):
                raise ValueError(
                    f"widen() cannot preserve the function through {type(mod).__name__} at {name!r} between "
                    f"{layer_name!r} and its consumer: only elementwise ops ({allowed}) are supported. "
                    "Width-dependent ops such as Softmax, LayerNorm and BatchNorm change their output when "
                    "units are duplicated."
                )
        if consumer_name is None:
            raise ValueError(
                f"no downstream nn.Linear found after {layer_name!r} — widen() needs a directly-connected "
                "Linear to rescale."
            )
    elif isinstance(parent, nn.ModuleList) and parent_path:
        owner_path, _, list_attr = parent_path.rpartition(".")
        owner = model if not owner_path else get_module(model, owner_path)
        if not (isinstance(owner, FeedFwdNN) and list_attr == "layers"):
            raise ValueError(
                f"widen() can only infer the consumer of a Linear inside nn.Sequential or FeedFwdNN.layers, "
                f"where the execution order is proven; {layer_name!r} lives in {type(owner).__name__}.{list_attr} "
                "and module registration order is not data flow."
            )
        idx = int(attr)
        if idx >= len(parent) - 1:
            raise ValueError(
                f"no downstream nn.Linear found after {layer_name!r} — widen() needs a directly-connected "
                "Linear to rescale."
            )
        activation = owner.params.activation_for(idx)
        if activation in _WIDTH_DEPENDENT_ACTIVATIONS:
            raise ValueError(
                f"widen() cannot preserve the function through the effective activation {activation.value!r} "
                f"of FeedFwdNN hidden layer {idx} (between {layer_name!r} and {parent_path}.{idx + 1!r}): "
                "it is width-dependent, so duplicating units changes its output."
            )
        consumer_name = f"{parent_path}.{idx + 1}"
    else:
        raise ValueError(
            f"widen() can only infer the consumer of a Linear inside nn.Sequential or FeedFwdNN.layers, "
            f"where the execution order is proven; {layer_name!r} lives in {type(parent).__name__} and "
            "module registration order is not data flow, so its consumer cannot be identified safely."
        )

    consumer = get_module(model, consumer_name)
    consumer_aliases = _registered_paths(model, consumer)
    if len(consumer_aliases) > 1:
        raise ValueError(
            f"the consumer {consumer_name!r} of {layer_name!r} is registered under multiple paths "
            f"{consumer_aliases}: a shared (aliased) module is used in more than one place, so rescaling "
            "one path cannot preserve the function"
        )
    return consumer_name
