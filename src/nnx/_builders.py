"""Shared ``copy()`` / ``from_params()`` support for the mutable params
builders (FEAT-041).

The four builders (`NNOptimParamsBuilder`, `NNSchedulerParamsBuilder`,
`NNTransformerParamsBuilder`, `NNTrainerParamsBuilder`) hold the fields a
caller has set until ``build()``. Branching one copies only the
*configuration containers* each builder names — the `param_groups` list,
the per-layer lists, the `optims` / `schedulers` maps, the
`extra_metrics` mapping — one level deep, so neither branch can change
the other's settings. Every other value is shared by identity: params
dataclasses are immutable, and loaders (any iterable, lists and arrays
included), metric callables and registered-factory specs are runtime
objects that must never be copied, iterated or compared element-wise.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any, Optional


def copy_containers(fields: Mapping[str, Any], containers: Iterable[str]) -> dict[str, Any]:
    """Return a new field dict whose named configuration containers are
    shallow copies (a mapping becomes a ``dict``, a sequence a ``list``).

    Items inside a container are shared: they are immutable params rows
    or runtime callables. Fields not named in ``containers`` — scalars,
    enums, loaders — are carried as the very same objects.
    """
    copied = dict(fields)
    for name in containers:
        value = copied.get(name)
        if value is None:
            continue
        copied[name] = dict(value) if isinstance(value, Mapping) else list(value)
    return copied


def _is_default(field: dataclasses.Field, value: Any) -> bool:
    """Whether ``value`` equals ``field`` 's declared default.

    ``None`` defaults are tested by identity only, so a runtime value
    (e.g. an array used as a loader) is never compared element-wise.
    """
    if field.default is not dataclasses.MISSING:
        if field.default is None or value is None:
            return value is field.default
        return value is field.default or bool(value == field.default)
    if field.default_factory is not dataclasses.MISSING:
        return bool(value == field.default_factory())
    return False


def params_init_values(
    builder: str,
    params: object,
    expected: type,
    supported: tuple[str, ...],
    containers: tuple[str, ...] = (),
    hint: Optional[str] = None,
) -> dict[str, Any]:
    """Validate ``params`` for ``<builder>.from_params`` and return the
    init fields a builder must carry.

    ``params`` must be exactly ``expected`` — a subclass may add fields the
    builder cannot reproduce, so it is rejected rather than silently
    downgraded. Every init field must be one the builder knows
    (``supported``); an unknown one raises ``ValueError`` naming it.
    Derived ``init=False`` fields (``NNParams._dims``) are excluded.

    The result keeps dataclass field order and contains every field
    without a default plus every field whose value differs from its
    default, so a rebuilt builder forwards exactly what a hand-written
    chain would and ``state()`` omits the same defaults. The fields named
    in ``containers`` are shallow-copied into builder-owned containers.
    """
    if type(params) is not expected:
        kind = type(params).__name__
        detail = f" (a subclass of {expected.__name__})" if isinstance(params, expected) else ""
        message = f"{builder}.from_params expects {expected.__name__}, got {kind}{detail}"
        raise TypeError(message + (f"; {hint}" if hint else ""))
    init_fields = [f for f in dataclasses.fields(expected) if f.init]
    unknown = [f.name for f in init_fields if f.name not in supported]
    if unknown:
        raise ValueError(
            f"{builder}.from_params: unsupported field(s) {unknown} on {expected.__name__}; "
            "the builder cannot reproduce them, so rebuilding would silently drop configuration"
        )
    values = {f.name: getattr(params, f.name) for f in init_fields if not _is_default(f, getattr(params, f.name))}
    return copy_containers(values, containers)
