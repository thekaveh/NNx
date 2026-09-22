"""Adapter state ownership — which ``state_dict`` keys an adapter-only
artifact may contain (FIX-002).

Adapter-only save/load helpers used to select keys by *substring*
(``"lora_A" in key``, ``"scaling" in key``, ``"soft_prompt" in key``,
``"prefix_" in key``). A user-chosen module name that happens to contain
the marker — ``lora_A_projection``, ``scaling_projection`` — then leaked
the wrapped layer's frozen ``base.weight`` / ``base.bias`` into the
adapter file, and a full checkpoint passed to a loader could silently
overwrite those frozen base tensors (``load_state_dict`` writes frozen
parameters just fine).

The helpers here derive the allowed keys from *registered ownership*:
walk the target module tree, and for every wrapper of a recognized type
emit exactly its directly-owned parameter names under the wrapper's
qualified prefix (a root wrapper owns unprefixed keys). Save intersects
that allowlist with the module's own ``state_dict``; load intersects it
with the untrusted source, so the source can never gain authority over
tensors the destination does not register as adapter state.

Internal; not part of the public API.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from torch import nn


def owned_adapter_keys(module: nn.Module, wrapper_types: type | tuple[type, ...], owned: Iterable[str]) -> set[str]:
    """Qualified ``state_dict`` keys owned by every ``wrapper_types``
    instance in ``module``'s tree — ``"<wrapper path>.<name>"`` for each
    name in ``owned``, unprefixed when the wrapper is the root."""
    names = tuple(owned)
    keys: set[str] = set()
    for path, sub in module.named_modules():
        if isinstance(sub, wrapper_types):
            prefix = f"{path}." if path else ""
            keys.update(prefix + name for name in names)
    return keys


def select_owned(state_dict: Mapping[str, Any], allowed: set[str]) -> dict[str, Any]:
    """The subset of ``state_dict`` whose keys are in ``allowed`` — used
    both to build an adapter-only artifact (from the module's own state)
    and to sanitize an untrusted source before ``load_state_dict``."""
    return {k: v for k, v in state_dict.items() if k in allowed}
