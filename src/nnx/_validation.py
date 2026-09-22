"""Configuration-boundary validators shared across params dataclasses and
public constructors (FIX-020, FIX-021).

`require_finite_real` is the one place that decides what a *finite real*
hyperparameter is: an instance of :class:`numbers.Real` that is not a
``bool`` (``True == 1`` would otherwise pass every inequality) and that
``math.isfinite`` accepts. Callers check the domain *after* finiteness so a
NaN — which fails every comparison and therefore sails through ``<``/``>``
guards — is rejected with an error that names the field and its accepted
domain, at construction time, before any optimizer, run directory or
lookup table is created.

`require_count` is its integer twin: a *count* (a dimension, a layer or
head count, an epoch or step count, a patience) is any :class:`numbers.Integral`
that is not a ``bool`` — Python and NumPy integers alike — normalized to a
plain ``int`` so immutable lists and ``state()`` snapshots carry portable
values (a raw ``numpy.int64`` is not YAML-safe). Every float is rejected,
``2.0`` included — a count is never rounded — as are strings, ``None`` and
NaN / ±inf, before any ``range``, modulo, ``math.isqrt`` or layer
allocation consumes the value. Nothing here coerces numeric strings or
touches ``state()``.

Internal; not part of the public API.
"""

from __future__ import annotations

import math
import numbers
from typing import Optional


def require_finite_real(
    value: object,
    field: str,
    *,
    owner: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    exclusive_min: bool = False,
    exclusive_max: bool = False,
    domain_message: Optional[str] = None,
) -> float:
    """Return ``value`` once it is a finite real inside the domain.

    Raises ``ValueError`` naming ``owner.field`` and the accepted domain for
    a non-real (``bool``, ``str``, ``None``), a NaN / ±inf, or an
    out-of-range value. ``minimum`` / ``maximum`` bound the domain;
    ``exclusive_*`` turns a bound strict. ``domain_message`` replaces the
    generic out-of-range message so a boundary can keep its established
    user-facing wording (it must still name the field).
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(
            f"{owner} requires {field} to be a finite real number{_domain(minimum, maximum, exclusive_min, exclusive_max)}, "
            f"got {value!r} of type {type(value).__name__}"
        )
    real = float(value)
    if not math.isfinite(real):
        raise ValueError(
            f"{owner} requires {field} to be finite{_domain(minimum, maximum, exclusive_min, exclusive_max)}, got {value!r}"
        )
    below = minimum is not None and (real <= minimum if exclusive_min else real < minimum)
    above = maximum is not None and (real >= maximum if exclusive_max else real > maximum)
    if below or above:
        raise ValueError(
            domain_message
            or f"{owner} requires {field}{_domain(minimum, maximum, exclusive_min, exclusive_max)}, got {value!r}"
        )
    return real


def require_count(
    value: object,
    field: str,
    *,
    owner: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
    exclusive_min: bool = False,
    exclusive_max: bool = False,
    domain_message: Optional[str] = None,
) -> int:
    """Return ``value`` as a plain ``int`` once it is an integral count
    inside the domain (FIX-021).

    Accepts any :class:`numbers.Integral` except ``bool`` (NumPy integers
    included) and returns ``int(value)``. Raises ``ValueError`` naming
    ``owner.field`` and the accepted domain for a non-integral value — a
    ``bool``, *any* float (``2.0`` is not a count), a numeric string,
    ``None``, NaN or ±inf — or an out-of-range one. ``minimum`` /
    ``maximum`` bound the domain (inclusive unless ``exclusive_*``);
    ``domain_message`` keeps an established out-of-range wording (it must
    still name the field).
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(
            f"{owner} requires {field} to be an integer{_domain(minimum, maximum, exclusive_min, exclusive_max)}, "
            f"got {value!r} of type {type(value).__name__}"
        )
    count = int(value)
    below = minimum is not None and (count <= minimum if exclusive_min else count < minimum)
    above = maximum is not None and (count >= maximum if exclusive_max else count > maximum)
    if below or above:
        raise ValueError(
            domain_message
            or f"{owner} requires {field}{_domain(minimum, maximum, exclusive_min, exclusive_max)}, got {value!r}"
        )
    return count


def _domain(minimum, maximum, exclusive_min, exclusive_max) -> str:
    parts = []
    if minimum is not None:
        parts.append(f"{'>' if exclusive_min else '>='} {minimum:g}")
    if maximum is not None:
        parts.append(f"{'<' if exclusive_max else '<='} {maximum:g}")
    return f" {' and '.join(parts)}" if parts else ""
