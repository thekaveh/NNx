"""Target selection for the ``apply_*_to`` PEFT helpers (FIX-014).

``nn.Module.named_modules()`` deduplicates repeated module objects, so a
``Linear`` registered under two names was visible at its first path only:
a wildcard wrapped one registration and left the other pointing at the raw
layer (the aliases silently stopped being one layer), and selecting the
second name matched nothing. Selection here walks *every* registration
path and groups them by registration slot — the ``(parent module,
attribute)`` pair a wrapper replaces:

* a ``Linear`` inside a shared *container* has a single slot, reached by
  several paths; it is selected when any of its paths matches, wrapped
  once, and every path observes the same wrapper;
* a ``Linear`` registered in more than one slot is an alias. If any of its
  slots is selected, the helper raises ``ValueError`` naming every path —
  before a single wrapper is built, because building one already freezes
  the base's parameters. Nothing is modified.

Only several registrations of *one* ``Linear`` are covered; tied tensors
shared between distinct modules (e.g. tied embeddings) are not aliases.

Internal; not part of the public API.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable, Iterator

from torch import nn

_Slot = tuple[int, str]
_MAX_LISTED_PATHS = 10


def _registrations(root: nn.Module) -> Iterator[tuple[str, nn.Module, str, nn.Module]]:
    """Yield ``(path, parent, attr, child)`` for every registration path.

    Unlike ``named_modules()`` repeated objects are not skipped; a module
    already on the current ancestor chain is reported but not descended
    into, so a cyclic registration cannot recurse forever. The number of
    paths equals the number of distinct routes to each module, which is
    linear for ordinary sharing (e.g. one block repeated in a list).
    """
    ancestors = {id(root)}

    def walk(module: nn.Module, prefix: str) -> Iterator[tuple[str, nn.Module, str, nn.Module]]:
        for attr, child in module._modules.items():
            if child is None:
                continue
            path = f"{prefix}.{attr}" if prefix else attr
            yield path, module, attr, child
            if id(child) not in ancestors:
                ancestors.add(id(child))
                yield from walk(child, path)
                ancestors.discard(id(child))

    yield from walk(root, "")


def _listed(paths: list[str]) -> str:
    shown = ", ".join(paths[:_MAX_LISTED_PATHS])
    extra = len(paths) - _MAX_LISTED_PATHS
    return f"({shown}, … and {extra} more)" if extra > 0 else f"({shown})"


def select_linear_slots(
    module: nn.Module,
    name_patterns: tuple[str, ...],
    *,
    skip_inside: type[nn.Module] | tuple[type[nn.Module], ...],
    helper: str,
) -> list[tuple[nn.Module, str]]:
    """Return the ``(parent, attr)`` slots of the ``nn.Linear`` layers to wrap.

    A slot is selected when any of its dotted paths matches any fnmatch
    glob in ``name_patterns`` and its parent is not already a wrapper of
    type ``skip_inside`` (so re-applying never wraps an existing ``.base``).
    The root module itself is never a slot. Raises ``ValueError`` — before
    any mutation — when ``name_patterns`` is empty or a selected layer is
    registered in more than one slot.
    """
    if not name_patterns:
        raise ValueError(f"{helper} requires at least one name pattern")

    # slot -> (parent, every path reaching it); layer -> its slots.
    slots: dict[_Slot, tuple[nn.Module, list[str]]] = {}
    layer_slots: dict[int, list[_Slot]] = {}
    for path, parent, attr, child in _registrations(module):
        if not isinstance(child, nn.Linear):
            continue
        slot = (id(parent), attr)
        if slot not in slots:
            slots[slot] = (parent, [])
            layer_slots.setdefault(id(child), []).append(slot)
        slots[slot][1].append(path)

    selected = [
        slot
        for slot, (parent, paths) in slots.items()
        if not isinstance(parent, skip_inside)
        and any(fnmatch.fnmatchcase(path, pattern) for path in paths for pattern in name_patterns)
    ]

    aliased: dict[int, list[str]] = {}
    for slot in selected:
        parent, _ = slots[slot]
        layer_id = id(getattr(parent, slot[1]))
        if len(layer_slots[layer_id]) > 1:
            aliased[layer_id] = [path for other in layer_slots[layer_id] for path in slots[other][1]]
    if aliased:
        groups = "; ".join(_listed(paths) for paths in aliased.values())
        raise ValueError(
            f"{helper}: cannot wrap an nn.Linear registered under more than one name {groups}: wrapping "
            "one registration would split the shared layer into independent layers. Nothing was "
            "modified. Register the layer once (reuse the owning module from both call sites) or "
            "select only unaliased layers. Tied tensors between distinct modules are not covered "
            "by this check."
        )
    return [(slots[slot][0], slot[1]) for slot in selected]


def wrap_selected_linears(
    module: nn.Module,
    name_patterns: tuple[str, ...],
    *,
    skip_inside: type[nn.Module] | tuple[type[nn.Module], ...],
    helper: str,
    wrap: Callable[[nn.Linear], nn.Module],
) -> int:
    """Select and validate every target first, then replace each slot's
    ``Linear`` with ``wrap(linear)``; return the number of new wrappers.

    Building a wrapper already freezes its base, so nothing is constructed
    until the whole matched set has passed :func:`select_linear_slots`.
    """
    targets = select_linear_slots(module, name_patterns, skip_inside=skip_inside, helper=helper)
    for parent, attr in targets:
        setattr(parent, attr, wrap(getattr(parent, attr)))
    return len(targets)
