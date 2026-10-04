from __future__ import annotations

from dataclasses import MISSING, fields
from typing import Any


def pickle_with_trailing_defaults(cls: type) -> None:
    """Pickle the frozen ``slots`` dataclass `cls` by position, restoring a
    pickle written before a trailing field existed with that field's default.

    The dataclass-generated ``__setstate__`` zips the stored values onto the
    fields and leaves the rest unset, so a pickle from an earlier release
    restored a half-built object that failed on first use (``repr``, ``==``,
    re-pickling). The stored layout — a list of field values in declaration
    order — is unchanged, so new fields must only ever be appended. Attached
    after the class exists: on Python 3.10, ``dataclass(slots=True)``
    replaces pickling hooks defined in the class body with its own.
    """

    def getstate(self: Any) -> list[Any]:
        return [getattr(self, spec.name) for spec in fields(self)]

    def setstate(self: Any, state: Any) -> None:
        specs = fields(self)
        for spec, value in zip(specs, state, strict=False):
            object.__setattr__(self, spec.name, value)
        for spec in specs[len(state) :]:
            if spec.default is not MISSING:
                value = spec.default
            elif spec.default_factory is not MISSING:
                value = spec.default_factory()
            else:
                raise TypeError(f"cannot restore {type(self).__name__}: the pickle lacks {spec.name!r}")
            object.__setattr__(self, spec.name, value)

    cls.__getstate__ = getstate  # type: ignore[attr-defined]
    cls.__setstate__ = setstate  # type: ignore[attr-defined]
