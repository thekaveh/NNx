"""Internal helpers for module-path resolution.

The surgery primitives all need to look up a submodule by its dotted
name (as produced by :meth:`torch.nn.Module.named_modules`) and write
a replacement module back in the same slot. Names can contain mixed
attribute / index segments (e.g., ``"layers.0"``) because
:class:`nn.ModuleList` exposes integer-string keys.
"""

from __future__ import annotations

from typing import Any, cast

from torch import nn


def get_module(root: nn.Module, dotted: str) -> nn.Module:
    """Return the submodule of ``root`` named by ``dotted``.

    Raises:
        KeyError: if no submodule is named ``dotted``. The message
            matches the wording in the surgery primitives'
            user-facing errors.
    """
    # `get_submodule` walks the dotted path correctly across both
    # named-attribute children (``layers``) and indexed children
    # (``layers.0``), so we delegate to it for the lookup itself.
    try:
        return root.get_submodule(dotted)
    except AttributeError as e:
        # PyTorch raises AttributeError on miss; surface as KeyError
        # so callers can distinguish "not found" from real type errors.
        raise KeyError(f"no module named {dotted!r} in model") from e


def set_module(root: nn.Module, dotted: str, new_mod: nn.Module) -> None:
    """Place ``new_mod`` at the slot named by ``dotted`` under ``root``.

    Handles both attribute-named children and index-named children of
    :class:`nn.Sequential` / :class:`nn.ModuleList` / :class:`nn.ModuleDict`.
    """
    if not dotted:
        raise ValueError("set_module requires a non-empty dotted path")
    parts = dotted.split(".")
    parent: nn.Module = root
    for p in parts[:-1]:
        parent = _step_into(parent, p)
    last = parts[-1]
    _assign(parent, last, new_mod)


def _step_into(parent: nn.Module, key: str) -> nn.Module:
    """Index one step down a dotted module path.

    A registered key is addressed *exactly* (the same slot
    ``get_submodule`` resolves) — after a named insertion (FIX-018) a
    retained numeric key such as ``"1"`` may no longer sit at position
    1, so positional ``parent[int(key)]`` is only a fallback for a
    numeric key that is not a registered name.
    """
    registered = parent._modules.get(key)
    if registered is not None:
        return registered
    if key.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        return parent[int(key)]
    return getattr(parent, key)


def _assign(parent: nn.Module, key: str, new_mod: nn.Module) -> None:
    """Write ``new_mod`` into ``parent`` at ``key`` — the registered slot
    when ``key`` is a registered name (``Module.__setattr__`` keeps the
    slot's position), else a positional index for a numeric key."""
    if key in parent._modules:
        setattr(parent, key, new_mod)
        return
    if key.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(key)] = new_mod
        return
    if isinstance(parent, nn.ModuleDict):
        parent[key] = new_mod
        return
    setattr(parent, key, new_mod)


def copy_param_roles(src: nn.Module, dst: nn.Module) -> None:
    """Give ``dst``'s parameters the ``requires_grad`` flags of ``src``'s
    same-named parameters and ``dst`` the train/eval mode of ``src``
    (FIX-016).

    Surgery rebuilds existing layers with freshly constructed modules
    whose parameters default to trainable and whose mode defaults to
    train — so a frozen layer silently became trainable after widening
    or factorization and an optimizer built afterwards updated it. Each
    replacement tensor inherits the flag of the *tensor it replaces*
    (weight and bias independently); parameters present on ``dst`` but
    absent on ``src`` (or vice versa) are left alone, so a factor
    without a bias, or a source without one, needs no special casing.
    Values are never touched: call this after copying the data.
    """
    src_flags = {name: p.requires_grad for name, p in src.named_parameters(recurse=False)}
    for name, p in dst.named_parameters(recurse=False):
        if name in src_flags:
            p.requires_grad_(src_flags[name])
    dst.train(src.training)


def stock_forward(module: nn.Module, *bases: type[nn.Module]) -> bool:
    """Whether ``module`` is one of ``bases`` and runs that base's own
    ``forward``: a rebuilt plain module would drop a subclass's forward
    (a fake-quantized Linear, a scaled Embedding) silently."""
    return any(isinstance(module, base) and type(module).forward is base.forward for base in bases)


def refuse_parametrized(module: nn.Module, what: str) -> None:
    """A ``weight_norm`` / ``spectral_norm`` (the ``torch.nn.utils.parametrize``
    forms or the legacy hooks) or an unbaked prune mask would be dropped
    silently by a rebuilt plain module, its frozen flags with it. The hooks
    are recognized by type, never by parameter names."""
    from torch.nn.utils import parametrize

    if parametrize.is_parametrized(module):
        raise ValueError(
            f"{what} carries a parametrization ({sorted(cast(Any, module).parametrizations)}); a surgery primitive "
            "would drop it silently. Remove it first (torch.nn.utils.parametrize.remove_parametrizations)"
        )
    from torch.nn.utils.prune import BasePruningMethod
    from torch.nn.utils.spectral_norm import SpectralNorm
    from torch.nn.utils.weight_norm import WeightNorm

    hooks = [hook for hook in module._forward_pre_hooks.values() if isinstance(hook, (WeightNorm, SpectralNorm))]
    if hooks:
        names = sorted(getattr(hook, "name", "?") for hook in hooks)
        raise ValueError(
            f"{what} carries a legacy weight_norm / spectral_norm on {names}; a surgery primitive would drop it "
            "silently. Remove it first (torch.nn.utils.remove_weight_norm / remove_spectral_norm)"
        )
    masks = [hook for hook in module._forward_pre_hooks.values() if isinstance(hook, BasePruningMethod)]
    if masks:
        names = sorted(getattr(hook, "_tensor_name", "?") for hook in masks)
        raise ValueError(
            f"{what} carries an unbaked prune mask on {names}; a surgery primitive would read a stale weight and "
            "drop the mask. Bake it first (torch.nn.utils.prune.remove, or magnitude_prune(bake=True))"
        )
