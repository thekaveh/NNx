"""Internal helpers for module-path resolution.

The surgery primitives all need to look up a submodule by its dotted
name (as produced by :meth:`torch.nn.Module.named_modules`) and write
a replacement module back in the same slot. Names can contain mixed
attribute / index segments (e.g., ``"layers.0"``) because
:class:`nn.ModuleList` exposes integer-string keys.
"""

from __future__ import annotations

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
    """Index one step down a dotted module path."""
    if key.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        return parent[int(key)]
    return getattr(parent, key)


def _assign(parent: nn.Module, key: str, new_mod: nn.Module) -> None:
    """Write ``new_mod`` into ``parent`` at ``key`` (index or attribute)."""
    if key.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(key)] = new_mod
        return
    if isinstance(parent, nn.ModuleDict):
        parent[key] = new_mod
        return
    setattr(parent, key, new_mod)
