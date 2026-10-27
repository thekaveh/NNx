"""Configuration-boundary validators shared across params dataclasses and
public constructors (FIX-020, FIX-021, FIX-022).

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
allocation consumes the value.

`require_batch_sizes` applies the count rule to a dataset wrapper's
``(train, val, test)`` request, where ``None`` is the *only* automatic
"one batch holding the complete split" sentinel: zero and ``False`` — which
truthiness resolution used to turn into a full-split request — are
rejected like every other non-positive or non-integral entry, and
anything but a 3-tuple fails with the tuple contract before a dataset
factory, tokenizer or split runs. Nothing here coerces numeric strings or
touches ``state()``.

`shown` is the value an error message quotes: its ``repr``, except that an
integer too long to print in full (``10**5000`` is past Python's 4300-digit
``str`` limit) is described by its size, so the refusal still names the field.

`plain_real` and `plain_str` turn an already-validated NumPy scalar or
``str`` subclass into the plain Python value it equals, keeping an exact
``int`` / ``float`` / ``str`` as given so existing run ids never shift.

Internal; not part of the public API.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Optional


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
            f"got {shown(value)} of type {type(value).__name__}"
        )
    try:
        real = float(value)
    except OverflowError as exc:
        raise ValueError(
            f"{owner} requires {field} to be a finite real number, got an integer too large for a float"
        ) from exc
    if not math.isfinite(real):
        raise ValueError(
            f"{owner} requires {field} to be finite{_domain(minimum, maximum, exclusive_min, exclusive_max)}, got {shown(value)}"
        )
    below = minimum is not None and (real <= minimum if exclusive_min else real < minimum)
    above = maximum is not None and (real >= maximum if exclusive_max else real > maximum)
    if below or above:
        raise ValueError(
            domain_message
            or f"{owner} requires {field}{_domain(minimum, maximum, exclusive_min, exclusive_max)}, got {shown(value)}"
        )
    return real


# An integer longer than this many bits (about 100 digits) is quoted by its size.
_SHOWN_BITS = 332


def shown(value: object, *, real: bool = False) -> str:
    """``repr(value)`` for an error message, or a description of an integer
    too long to print: ``10**5000`` would otherwise raise Python's own
    "Exceeds the limit (4300 digits)" ``ValueError`` while the message is
    built, so the refusal would never name its field. ``real=True`` (a
    float field) calls an integer that overflows a float "an integer too
    large for a float", as :func:`require_finite_real` does."""
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        integer = int(value)
        if real:
            try:
                float(integer)
            except OverflowError:
                return "an integer too large for a float"
        if integer.bit_length() > _SHOWN_BITS:
            return f"an integer of {integer.bit_length()} bits"
    return repr(value)


def plain_real(value: Any) -> Any:
    """``value`` as a plain Python number for ``state()``: a validated real of
    another type (a NumPy scalar, say) becomes ``int`` or ``float``, so ``run.yaml``
    can serialize it. Python ``int`` and ``float`` are kept as given, so
    existing run ids (an md5 of ``state()``) never shift."""
    if type(value) in (int, float):
        return value
    # A NumPy integer is the same setting as the Python int (and run id).
    return int(value) if isinstance(value, numbers.Integral) else float(value)


def plain_str(value: Any) -> Any:
    """``value`` as a plain Python ``str`` for ``state()``: a validated
    ``str`` subclass (a ``numpy.str_``, say) becomes the ``str`` it equals,
    so ``run.yaml`` can serialize it and the run id hashes the plain text.
    An exact ``str`` is kept as given; anything else is returned unchanged
    for the caller's own check to report."""
    if type(value) is str or not isinstance(value, str):
        return value
    return str.__str__(value)


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
            f"got {shown(value)} of type {type(value).__name__}"
        )
    count = int(value)
    below = minimum is not None and (count <= minimum if exclusive_min else count < minimum)
    above = maximum is not None and (count >= maximum if exclusive_max else count > maximum)
    if below or above:
        raise ValueError(
            domain_message
            or f"{owner} requires {field}{_domain(minimum, maximum, exclusive_min, exclusive_max)}, got {shown(value)}"
        )
    return count


_SPLITS = ("train", "val", "test")


def require_batch_sizes(value: object, *, owner: str) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Return a dataset wrapper's ``batch_sizes`` request as a
    ``(train, val, test)`` tuple of ``None`` or plain positive ``int``
    (FIX-022).

    ``None`` is the only automatic sentinel ("one batch holding the
    complete split"); a positive integral entry — NumPy integers
    normalized to ``int`` — is kept verbatim, never clipped to the split.
    Zero, ``False``, negatives, floats (``2.0`` included), strings and NaN
    raise ``ValueError`` naming ``batch_sizes[i] (split)``; a list, a bare
    value or a tuple of the wrong length raises with the 3-tuple contract.
    Call it *before* any dataset factory, tokenizer or RNG-consuming split.
    """
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(
            f"{owner} requires batch_sizes to be a 3-tuple (train, val, test) of None or positive integers, "
            f"got {shown(value)} of type {type(value).__name__}"
        )
    resolved: list[Optional[int]] = []
    for i, (split, entry) in enumerate(zip(_SPLITS, value, strict=True)):
        if entry is None:
            resolved.append(None)
            continue
        slot = f"batch_sizes[{i}] ({split})"
        resolved.append(
            require_count(
                entry,
                slot,
                owner=owner,
                minimum=0,
                exclusive_min=True,
                domain_message=f"{owner} requires {slot} to be None (one full-split batch) or a positive integer, got {shown(entry)}",
            )
        )
    return (resolved[0], resolved[1], resolved[2])


def _domain(minimum, maximum, exclusive_min, exclusive_max) -> str:
    parts = []

    def bound_text(bound):  # integral bounds exactly (2**32 - 1, not 4.29497e+09)
        return str(bound) if isinstance(bound, int) and not isinstance(bound, bool) else f"{bound:g}"

    if minimum is not None:
        parts.append(f"{'>' if exclusive_min else '>='} {bound_text(minimum)}")
    if maximum is not None:
        parts.append(f"{'<' if exclusive_max else '<='} {bound_text(maximum)}")
    return f" {' and '.join(parts)}" if parts else ""


def checked(check: Any, value: object, field: str, *, owner: str, error: type[Exception], **domain: Any) -> Any:
    """Run ``check`` (``require_finite_real`` / ``require_count``) and raise
    its ``ValueError`` — or the ``OverflowError`` of an integer too large for
    a float (``10**400``) — as ``error``, so a module keeps its own error type."""
    try:
        return check(value, field, owner=owner, **domain)
    except ValueError as exc:
        raise error(str(exc)) from exc
    except OverflowError as exc:
        raise error(
            f"{owner} requires {field} to be a finite real number, got an integer too large for a float"
        ) from exc


def required_id(value: Any, what: str, *, error: type[Exception]) -> str:
    """A non-empty string id, or ``error``."""
    if not isinstance(value, str) or not value:
        raise error(f"{what} must be a non-empty string, got {value!r}")
    return value
