"""Immutable JSON-like configuration values shared by registered-object
specs (``OptimizerFactorySpec``, ``MetricSpec``): frozen at construction,
thawed to plain YAML-safe dicts / lists for ``state()``. Internal."""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Iterator, Mapping
from typing import Any

# Identifier of a registered object (optimizer factory, metric) and of the
# names specs report under: letters, digits, '_', '.', '-'; starts with a
# letter or digit.
_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]*")


class _FrozenConfig(Mapping[str, Any]):
    """Read-only, picklable / deep-copyable string-keyed mapping used for
    a spec's ``config`` (``types.MappingProxyType`` cannot be copied)."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = dict(data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(self._data)

    def __reduce__(self):
        return (_FrozenConfig, (self._data,))


def _freeze_config(value: Any, path: str, owner: str = "OptimizerFactorySpec") -> Any:
    """Return an immutable, JSON-like copy of ``value`` or raise.

    Accepted: ``None``, ``bool``, ``int``, finite ``float``, ``str``,
    lists/tuples of those (frozen to tuples) and mappings with string keys
    (frozen to sorted read-only mappings). Callables, tensors, modules and
    every other object are rejected, so a spec always serializes to plain
    YAML and never smuggles executable state into ``run.yaml``.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    # NumPy / other numeric scalars are normalized to plain int / float so
    # state() stays YAML-portable (same convention as the params counts).
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        real = float(value)
        if not math.isfinite(real):
            raise ValueError(f"{owner} config{path} must be finite, got {value!r}")
        return real
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_config(item, f"{path}[{i}]", owner) for i, item in enumerate(value))
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key in sorted(value, key=lambda k: str(k)):
            if not isinstance(key, str):
                raise TypeError(f"{owner} config{path} keys must be strings, got {key!r}")
            frozen[key] = _freeze_config(value[key], f"{path}[{key!r}]", owner)
        return _FrozenConfig(frozen)
    kind = "a callable" if callable(value) else f"a {type(value).__name__}"
    raise TypeError(
        f"{owner} config{path} must be JSON-like (None, bool, int, finite float, str, "
        f"list, or a str-keyed mapping); got {kind}: {value!r}. Factories and other executable objects "
        "are registered (register_optimizer_factory / register_metric) and referenced by id/version, "
        "never stored in a run config."
    )


def _thaw_config(value: Any) -> Any:
    """Plain YAML-safe copy (dicts / lists) of a frozen config value."""
    if isinstance(value, Mapping):
        return {key: _thaw_config(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_config(item) for item in value]
    return value
