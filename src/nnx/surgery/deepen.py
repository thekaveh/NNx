"""Net2DeeperNet — function-preserving depth expansion.

Reference: Chen, Goodfellow, Shlens — *Net2Net* (ICLR 2016).

The construction: insert an *identity-initialized* :class:`nn.Linear`
(weight = I, bias = 0) immediately after a ReLU activation. Because
ReLU's output is non-negative, the identity Linear's output is also
non-negative, and a second ReLU applied to it is a no-op — the overall
forward is unchanged. Formally:

    ReLU(I · ReLU(x) + 0) == ReLU(x)   for all x.

That equality is what makes the surgery function-preserving. It breaks
for sigmoid / tanh / GELU / etc. because those activations are not
idempotent on the identity Linear's output range — so this primitive
**rejects any activation other than ReLU** at runtime, with an
explicit error message rather than silently producing a model whose
forward output drifts.

Two insertion modes are supported:

  - **nn.Sequential mode**: ``after_layer_name`` points at an
    :class:`nn.ReLU` module. The primitive returns a deep copy of
    the Sequential with ``[nn.Linear(I), nn.ReLU()]`` spliced in
    immediately after the named ReLU.
  - **FeedFwdNN / ModuleList mode**: ``after_layer_name`` points at a
    Linear inside a :class:`nn.ModuleList` whose parent applies an
    activation between consecutive layers (the FeedFwdNN forward
    contract). The primitive inserts a fresh identity-init Linear into
    the ModuleList right after the named one, and the parent's forward
    automatically applies the ReLU on either side. The parent's
    declared activation must be :class:`nn.ReLU`.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from ._utils import get_module


def deepen(
    model: nn.Module,
    *,
    after_layer_name: str,
) -> nn.Module:
    """Net2DeeperNet: insert an identity-initialized Linear after the
    named layer. Function-preserving on ReLU networks only.

    Args:
        model: any :class:`nn.Module`. Deep-copied so the caller's
            reference survives.
        after_layer_name: dotted name (as in ``named_modules()``) of
            the insertion site. Either:

              * an :class:`nn.ReLU` inside a parent :class:`nn.Sequential`
                — the primitive splices ``Linear(I) → ReLU`` in after it.
              * an :class:`nn.Linear` inside a parent :class:`nn.ModuleList`
                whose grandparent module declares ReLU as its activation
                (the FeedFwdNN contract) — the primitive inserts a new
                identity-init Linear into the ModuleList right after.

    Returns:
        A fresh :class:`nn.Module` whose forward output matches the
        original within ``atol=1e-5``.

    Raises:
        KeyError: if ``after_layer_name`` is not a submodule.
        TypeError: if the layer is neither a ReLU-in-Sequential nor a
            Linear-in-FeedFwdNN-like ModuleList.
        ValueError: if the parent's activation is anything other than
            ReLU. Sigmoid / tanh / GELU break function-preservation.
    """
    new_model = copy.deepcopy(model)
    target = get_module(new_model, after_layer_name)

    # Locate the parent container so we can splice.
    parent_path, _, attr = after_layer_name.rpartition(".")
    parent = new_model if not parent_path else new_model.get_submodule(parent_path)

    if isinstance(target, nn.ReLU) and isinstance(parent, nn.Sequential):
        return _insert_after_relu_in_sequential(new_model, parent_path, attr)

    if isinstance(target, nn.Linear) and isinstance(parent, nn.ModuleList):
        return _insert_after_linear_in_module_list(new_model, parent_path, attr, target)

    raise TypeError(
        f"deepen: cannot insert after {after_layer_name!r} "
        f"({type(target).__name__} in a {type(parent).__name__} parent). "
        "Supported sites: nn.ReLU inside nn.Sequential, or nn.Linear "
        "inside an nn.ModuleList whose parent applies an activation."
    )


def _insert_after_relu_in_sequential(
    new_model: nn.Module,
    parent_path: str,
    attr: str,
) -> nn.Module:
    """Splice [Linear(I), ReLU] into a Sequential right after the
    named ReLU. The Linear's dim comes from the previous Linear's
    out_features (walked from earlier siblings)."""
    parent: nn.Sequential = new_model if not parent_path else new_model.get_submodule(parent_path)
    idx = int(attr)

    # Find the most recent Linear earlier in the Sequential — its
    # out_features is the activation's hidden dimension.
    src_linear = None
    for j in range(idx - 1, -1, -1):
        if isinstance(parent[j], nn.Linear):
            src_linear = parent[j]
            break
    if src_linear is None:
        raise ValueError(
            "deepen: could not find an upstream nn.Linear before "
            f"the ReLU at position {idx} to source the hidden dim from."
        )

    # dtype/device come from the SAME Linear that sourced the dim — the
    # old probe peeked at parent[idx-1], which is wrong whenever a
    # non-Linear (Dropout, norm) sits between the Linear and the ReLU,
    # and it never threaded the device, splicing a CPU layer into a
    # CUDA-resident model.
    new_linear = _identity_linear(
        src_linear.out_features,
        dtype=src_linear.weight.dtype,
        device=src_linear.weight.device,
    )
    new_relu = nn.ReLU()

    # Rebuild the Sequential with the two new modules inserted right
    # after `idx`. nn.Sequential supports __setitem__ but not insert(),
    # so we build a new one. If the Sequential IS the root model, we
    # return the new Sequential directly; otherwise we splice it into
    # the original root in place of the old Sequential.
    children = list(parent.children())
    new_children = children[: idx + 1] + [new_linear, new_relu] + children[idx + 1 :]
    new_sequential = nn.Sequential(*new_children)
    if not parent_path:
        return new_sequential
    _replace_in_parent(new_model, parent_path, new_sequential)
    return new_model


def _insert_after_linear_in_module_list(
    new_model: nn.Module,
    parent_path: str,
    attr: str,
    target: nn.Linear,
) -> nn.Module:
    """Insert a fresh identity-init Linear into a ModuleList right
    after the named position. The grandparent's declared activation
    must be ReLU."""
    parent: nn.ModuleList = new_model if not parent_path else new_model.get_submodule(parent_path)
    idx = int(attr)

    # Inserting before the LAST layer (the output head) is fine, but
    # inserting AT the last position has no activation applied after it
    # — refusing to splice in that case keeps the function-preservation
    # contract honest.
    if idx == len(parent) - 1:
        raise ValueError(
            "deepen: cannot insert after the last Linear in a ModuleList "
            "— no activation is applied past it, so function-preservation "
            "cannot be guaranteed."
        )

    # The grandparent (e.g. FeedFwdNN) must declare ReLU as its activation.
    # We walk up one level and read either `.params.activation` (NNx's
    # FeedFwdNN contract) or refuse.
    grandparent_path, _, _ = parent_path.rpartition(".")
    grandparent = new_model if not grandparent_path else new_model.get_submodule(grandparent_path)
    _check_relu_activation(grandparent, parent_path)

    new_linear = _identity_linear(
        target.out_features,
        dtype=target.weight.dtype,
        device=target.weight.device,
    )

    # Build a fresh ModuleList with the insertion. nn.ModuleList has
    # .insert(idx, mod), but it mutates in place — to keep the
    # deep-copy semantics consistent with widen() we build a new one
    # and swap it in.
    children = list(parent)
    new_children = children[: idx + 1] + [new_linear] + children[idx + 1 :]
    new_module_list = nn.ModuleList(new_children)
    _replace_in_parent(new_model, parent_path, new_module_list)
    return new_model


def _check_relu_activation(grandparent: nn.Module, parent_path: str) -> None:
    """Inspect the FeedFwdNN-like parent for a ReLU activation choice."""
    # NNx's FeedFwdNN holds an Activations enum on `params.activation`
    # and calls it as a factory in the forward pass. We accept either
    # the enum value or a module / callable that produces nn.ReLU.
    params = getattr(grandparent, "params", None)
    if params is None or not hasattr(params, "activation"):
        raise ValueError(
            f"deepen: parent of {parent_path!r} has no `.params.activation` to "
            "inspect; cannot verify ReLU. Only FeedFwdNN-style modules are "
            "supported in ModuleList mode."
        )

    act = params.activation

    # Fast path: NNx Activations enum. The enum's `__call__` returns a
    # functional callable (e.g. F.relu), not an nn.Module — so we must
    # compare by enum identity, not by isinstance on the call result.
    try:
        from ..nn.enum.activations import Activations as _A

        if isinstance(act, _A):
            if act is _A.RELU:
                return
            raise ValueError(
                f"deepen: activation is {act.value!r}, but identity-init "
                "insertion is function-preserving only for ReLU. Sigmoid/tanh/"
                "GELU/etc. networks must be deepened by another method."
            )
    except ImportError:  # pragma: no cover — same package
        pass

    # Fallback: probe arbitrary module factories.
    try:
        probe = act() if callable(act) else None
    except Exception as e:  # pragma: no cover — defensive
        raise ValueError(f"deepen: could not instantiate activation {act!r} to check it; {e}") from e

    if not isinstance(probe, nn.ReLU):
        raise ValueError(
            f"deepen: activation is {type(probe).__name__ if probe else act!r}, "
            "but identity-init insertion is function-preserving only for ReLU. "
            "Sigmoid/tanh/GELU networks must be deepened by another method."
        )


def _identity_linear(dim: int, *, dtype=torch.float32, device=None) -> nn.Linear:
    """A fresh ``nn.Linear(dim, dim)`` with weight = I and bias = 0."""
    layer = nn.Linear(dim, dim, bias=True, dtype=dtype, device=device)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(dim, dtype=dtype, device=layer.weight.device))
        layer.bias.zero_()
    return layer


def _replace_in_parent(root: nn.Module, dotted: str, new_mod: nn.Module) -> None:
    """Replace the submodule at ``dotted`` with ``new_mod``. Handles the
    empty-path case (replacing root itself is not supported here — the
    caller never reaches that branch)."""
    if not dotted:
        raise ValueError("_replace_in_parent: refusing to replace root module")
    parent_path, _, attr = dotted.rpartition(".")
    parent = root if not parent_path else root.get_submodule(parent_path)
    if attr.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(attr)] = new_mod
    elif isinstance(parent, nn.ModuleDict):
        parent[attr] = new_mod
    else:
        setattr(parent, attr, new_mod)
