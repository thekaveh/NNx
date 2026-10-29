"""Magnitude-based unstructured pruning.

Wraps :func:`torch.nn.utils.prune.l1_unstructured` with two
additions on top of the raw PyTorch API:

  1. **Pattern filter** — fnmatch glob against dotted submodule names
     (same convention as :func:`nnx.peft.apply_lora_to` /
     :func:`nnx.finetune.freeze`).
  2. **state_dict shape preservation** — by default, ``prune.remove`` is
     called immediately after each layer is pruned so the layer ends
     up with a plain ``weight`` tensor (with the zeroed entries baked
     in) instead of the reparameterization-time
     ``weight_orig`` + ``weight_mask`` pair. This keeps the network's
     ``state_dict`` schema identical to the unpruned network — pruned
     checkpoints load into unpruned code, and vice versa, with
     ``strict=True``. The trade-off is that the mask is gone: nothing
     enforces the zeros afterward, so any SUBSEQUENT training
     (fine-tuning included) immediately regrows the pruned entries and
     sparsity decays toward dense. Prune-then-finetune and iterative
     schedules must use ``bake=False`` (mask enforced through the
     reparameterization) and bake once at the very end. ``NNModel.train``
     refuses a registered net that still carries the reparameterization,
     because its checkpoints could not rebuild it: fine-tune such a net
     as ``NNModel(module=net)`` or with a custom loop.
"""

from __future__ import annotations

import fnmatch

import torch.nn.utils.prune as torch_prune
from torch import nn

from .._reparameterization import weight_reparameterization


def magnitude_prune(
    net: nn.Module,
    sparsity: float,
    *,
    layer_pattern: str = "*",
    bake: bool = True,
) -> int:
    """Zero the smallest-magnitude entries of every matched layer's weight.

    For each :class:`nn.Linear` submodule of ``net`` whose dotted name
    matches ``layer_pattern`` (fnmatch glob), call
    :func:`torch.nn.utils.prune.l1_unstructured` with
    ``amount=sparsity``. PyTorch's implementation zeros
    ``round(sparsity · weight.numel())`` entries — the ones with the
    smallest absolute value — per layer.

    Args:
        net: root module to walk. The function mutates ``net`` in place.
        sparsity: fraction of weights to zero, in ``[0, 1)``. ``0.0``
            is a valid no-op; ``1.0`` is rejected here — a fully-zeroed
            Linear is never useful (torch itself would accept
            ``amount=1.0`` and silently zero the whole weight).
        layer_pattern: fnmatch glob against dotted submodule name.
            ``"*"`` (the default) matches every :class:`nn.Linear`.
        bake: when ``True`` (default), call
            :func:`torch.nn.utils.prune.remove` immediately after each
            layer is pruned. The mask is baked into a plain ``weight``
            tensor, the reparameterization is dropped, and the
            ``state_dict`` keys stay identical to the pre-prune layout.
            When ``False``, the reparameterization stays in place — the
            ``state_dict`` carries ``<name>.weight_orig`` +
            ``<name>.weight_mask`` instead of ``<name>.weight``. Use
            ``False`` for iterative pruning schedules (e.g., 10% per
            epoch for N epochs) where successive ``magnitude_prune``
            calls need to compose with the existing mask. ``NNModel.train``
            refuses a registered net that still carries the masks: run
            such a schedule on ``NNModel(module=net)`` or a custom loop.

    Returns:
        The number of :class:`nn.Linear` submodules that were pruned.
        ``0`` if ``layer_pattern`` matched nothing.

    Raises:
        ValueError: if ``sparsity`` is outside ``[0, 1)``; if a matched
            Linear has a parametrized weight (a
            ``torch.nn.utils.parametrize`` parametrization such as
            ``parametrizations.weight_norm`` / ``spectral_norm`` /
            ``orthogonal``, or a legacy ``torch.nn.utils.weight_norm`` /
            ``spectral_norm`` hook), which torch's pruning cannot mask
            (the message lists every such Linear); or if ``bake=False``
            selects a Linear whose parent reads its weight directly
            (``nn.MultiheadAttention``'s ``out_proj``, or the ``base`` of
            a :class:`~nnx.peft.DoRALinear`). Every check runs before any
            layer is pruned, so a refused call modifies nothing.

    **Idempotency note:** with ``bake=True``, calling ``magnitude_prune``
    twice at the same ``sparsity`` is a no-op for the second call —
    l1_unstructured picks the smallest-magnitude entries, which after the
    first prune are exactly the already-zeroed positions. With
    ``bake=False`` the masks compose: a second call prunes ``sparsity`` of
    the remaining weights (0.5 twice leaves 0.75 zero).
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"magnitude_prune sparsity must be in [0, 1), got {sparsity}")

    # Two-phase: collect targets first, then mutate. Iterating
    # named_modules() while reassigning child attributes via
    # `prune.l1_unstructured` registers a forward_pre_hook that doesn't
    # alter traversal order today, but collecting up-front keeps the
    # function defensive against any future PyTorch internals change.
    targets: list[tuple[str, nn.Linear]] = []
    for name, mod in net.named_modules():
        if isinstance(mod, nn.Linear) and fnmatch.fnmatchcase(name, layer_pattern):
            targets.append((name, mod))
    # A parametrized weight is computed, not a registered Parameter, so torch's
    # pruning fails on it mid-loop, after earlier matches were already pruned
    # (FIX-086). Refuse every such target before touching any of them.
    reparameterized = [
        f"{name or 'the root module'} ({kind})"
        for name, mod in targets
        if (kind := weight_reparameterization(mod)) is not None
    ]
    if reparameterized:
        linears, have, them = ("Linears", "have", "them") if len(reparameterized) > 1 else ("Linear", "has", "it")
        raise ValueError(
            f"magnitude_prune: {linears} {', '.join(reparameterized)} {have} a parametrized weight, which torch's "
            "pruning cannot mask. Remove the parametrization first (torch.nn.utils.parametrize."
            "remove_parametrizations, or torch.nn.utils.remove_weight_norm / remove_spectral_norm for the legacy "
            f"hooks) or exclude {them} with layer_pattern. Nothing was modified."
        )
    if not bake:
        # The mask is applied by a forward pre-hook, which never runs when a
        # parent reads the Linear's weight directly (MultiheadAttention's
        # out_proj): its stale pruned weight breaks the second backward.
        from ..peft._targets import WEIGHT_READING_PARENTS

        read_directly = [
            f"{name}.{child_name}" if name else child_name
            for name, parent in net.named_modules()
            if isinstance(parent, WEIGHT_READING_PARENTS)
            for child_name, child in parent.named_children()
            if any(child is mod for _, mod in targets)
        ]
        if read_directly:
            raise ValueError(
                f"magnitude_prune(bake=False): {', '.join(read_directly)} are read by their parent's weight "
                "directly, so a mask could not apply; exclude them with layer_pattern or use bake=True. Nothing "
                "was modified."
            )
        # DoRALinear reads its base's weight directly as well (FIX-056); refused
        # here rather than through WEIGHT_READING_PARENTS, which also governs
        # PEFT slot selection and recipe validation.
        from ..peft.dora import DoRALinear

        dora_bases = [
            f"{name}.base" if name else "base"
            for name, parent in net.named_modules()
            if isinstance(parent, DoRALinear) and any(parent.base is mod for _, mod in targets)
        ]
        if dora_bases:
            raise ValueError(
                f"magnitude_prune(bake=False): {', '.join(dora_bases)} are the base of a DoRALinear, which reads "
                "the weight directly, so a mask could not apply; exclude them with layer_pattern, use bake=True, "
                "or adapt them with LoRA, which calls the base layer. Nothing was modified."
            )

    for _, mod in targets:
        # l1_unstructured at amount=0.0 is a valid no-op — torch still
        # registers the reparameterization but the mask is all-ones.
        # Calling remove afterwards (bake=True) restores the original
        # plain `weight` tensor with no change.
        torch_prune.l1_unstructured(mod, name="weight", amount=sparsity)
        if bake:
            # Bake the mask into a plain weight tensor. After this call
            # the layer's state_dict layout is identical to a fresh
            # nn.Linear's (weight + bias), so pruned checkpoints stay
            # shape-compatible with unpruned-network code.
            torch_prune.remove(mod, "weight")

    return len(targets)
