"""An opt-in ``Result`` at fallible boundaries (FEAT-025).

NNx reports failures by raising. This module adds a small alternative for
the places where a caller would rather branch on a value than wrap a call in
``try`` / ``except``: :class:`Ok` holds a success, :class:`Err` a typed
error, and a few combinators compose them without ever catching anything::

    from nnx.result import inspect_bundle_result

    summary = (
        inspect_bundle_result("bundle/")
        .map(lambda info: info.capability)
        .recover(lambda error: Ok("unavailable") if error.code == "artifact_missing" else Err(error))
    )

- :meth:`Ok.map` changes only an ``Ok``; :meth:`Err.map_error` only an
  ``Err``. :meth:`bind` flattens one ``Result`` returned by its callback and
  :meth:`recover` does the same from an ``Err``; a callback that returns
  anything else raises ``TypeError``. Callbacks are never wrapped: their
  exceptions — ``RuntimeError``, ``KeyboardInterrupt``, a cancellation —
  propagate unchanged.
- :meth:`unwrap` returns an ``Ok``'s value and raises :class:`UnwrapError`
  (carrying the typed error) on an ``Err``. A ``Result`` has no truth value:
  ``Ok(0)``, ``Ok(None)`` and ``Ok(False)`` are successes, and ``if result:``
  raises ``TypeError``. Branch with ``isinstance(result, Ok)`` or ``match``
  — both narrow the type for a checker — or test :attr:`is_ok` /
  :attr:`is_err` when no narrowing is needed.
- **Boundaries.** Three wrappers turn exactly the exceptions they declare
  into an ``Err(BoundaryError)`` carrying a ``code``, the ``where`` (a field
  or a path), a ``context`` mapping and the original exception as ``cause``;
  every other exception propagates:

  - :func:`validate_decision_request_result` builds a typed question from
    plain data — :class:`~nnx.decisions.InvalidDecisionRequest` becomes
    ``"invalid_decision_request"``. No provider is involved.
  - :func:`decide_result` asks a provider — ``InvalidDecisionResponse`` (a
    malformed answer) becomes ``"invalid_decision_response"``, any other
    ``ProviderFailure`` ``"provider_failure"`` and ``UnsupportedCapability``
    ``"unsupported"``.
    With an abstention policy, an abstention is a **success**: the ``Ok``
    holds every row's ``SelectiveDecision``, probabilities and reasons
    included.
  - :func:`inspect_bundle_result` delegates to the safe
    :func:`nnx.bundles.inspect_bundle` (manifest and JSON only: it never
    unpickles, calls a factory or a registry) — a missing path becomes
    ``"artifact_missing"``, any other ``BundleError`` ``"bundle_invalid"``.

Every existing exception API is unchanged; nothing in NNx returns a
``Result`` unless one of these wrappers is called.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Generic, NoReturn, Optional, TypeVar, Union, cast

__all__ = [
    "BoundaryError",
    "Err",
    "Ok",
    "Result",
    "UnwrapError",
    "decide_result",
    "inspect_bundle_result",
    "validate_decision_request_result",
]

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")
F = TypeVar("F")


class UnwrapError(RuntimeError):
    """:meth:`Err.unwrap` was called; ``error`` is the typed error (and the
    ``__cause__`` when it is an exception)."""

    def __init__(self, error: Any) -> None:
        super().__init__(f"called unwrap() on an Err: {error}")
        self.error = error

    def __reduce__(self) -> tuple[Any, ...]:
        return (UnwrapError, (self.error,))


def _result(value: Any, what: str) -> Any:
    if not isinstance(value, (Ok, Err)):
        raise TypeError(f"{what} must return an Ok or an Err, got {type(value).__name__}")
    return value


class _NoTruth:
    def __bool__(self) -> NoReturn:
        raise TypeError(
            f"a {type(self).__name__} has no truth value (Ok(0) and Ok(None) are successes): "
            "branch with isinstance(result, Ok) or match, or test .is_ok / .is_err"
        )


@dataclass(frozen=True)
class Ok(_NoTruth, Generic[T]):
    """A success holding ``value`` (any value: ``0``, ``None`` and ``False``
    included)."""

    value: T

    @property
    def is_ok(self) -> bool:
        return True

    @property
    def is_err(self) -> bool:
        return False

    def map(self, fn: Callable[[T], U]) -> Ok[U]:
        """``Ok(fn(value))``."""
        return Ok(fn(self.value))

    def bind(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """``fn(value)``, which must be an ``Ok`` or an ``Err``."""
        return _result(fn(self.value), "bind's callback")

    def map_error(self, fn: Callable[[Any], Any]) -> Ok[T]:
        """Unchanged: there is no error to map."""
        return self

    def recover(self, fn: Callable[[Any], Any]) -> Ok[T]:
        """Unchanged: there is nothing to recover from."""
        return self

    def unwrap(self) -> T:
        return self.value


@dataclass(frozen=True)
class Err(_NoTruth, Generic[E]):
    """A failure holding a typed ``error``."""

    error: E

    @property
    def is_ok(self) -> bool:
        return False

    @property
    def is_err(self) -> bool:
        return True

    def map(self, fn: Callable[[Any], Any]) -> Err[E]:
        """Unchanged: there is no value to map."""
        return self

    def bind(self, fn: Callable[[Any], Any]) -> Err[E]:
        """Unchanged: there is no value to bind."""
        return self

    def map_error(self, fn: Callable[[E], F]) -> Err[F]:
        """``Err(fn(error))``."""
        return Err(fn(self.error))

    def recover(self, fn: Callable[[E], Result[T, F]]) -> Result[T, F]:
        """``fn(error)``, which must be an ``Ok`` or an ``Err``."""
        return _result(fn(self.error), "recover's callback")

    def unwrap(self) -> NoReturn:
        """Raise :class:`UnwrapError` carrying the error (and chaining it
        when it is an exception)."""
        cause = self.error if isinstance(self.error, BaseException) else None
        raise UnwrapError(self.error) from cause


Result = Union[Ok[T], Err[E]]
"""``Ok[T]`` or ``Err[E]``."""


@dataclass(frozen=True)
class BoundaryError:
    """A typed failure at a boundary: a stable ``code``, ``where`` it
    happened (a field name or a path), a read-only ``context`` mapping and
    the original exception as ``cause``.

    Equality compares ``code``, ``where`` and ``context`` (not ``cause``:
    exceptions compare by identity); the hash uses ``code`` and ``where``.
    It pickles and deep-copies (``context`` as a plain mapping); being
    read-only, ``context`` is not supported by ``dataclasses.asdict``."""

    code: str
    where: str
    context: Mapping[str, Any] = field(default_factory=dict, hash=False)
    cause: Optional[BaseException] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))

    def __reduce__(self) -> tuple[Any, ...]:
        return (BoundaryError, (self.code, self.where, dict(self.context), self.cause))

    def __str__(self) -> str:
        reason = f": {self.cause}" if self.cause is not None else ""
        return f"{self.code} at {self.where}{reason}"


# --- decision boundaries ----------------------------------------------------------------------


def validate_decision_request_result(
    kind: str, prompt: str, options: Optional[Sequence[Any]] = None
) -> Result[Any, BoundaryError]:
    """Build a typed decision question from plain data.

    ``kind`` is ``"choice"``, ``"boolean"`` or ``"score"``; ``options`` are
    ``(id, description)`` pairs (a Choice's options, a Score's levels, none
    for a Boolean). The question objects are built inside the boundary, so a
    malformed request — duplicate or empty ids, too few options, an empty
    prompt, an unknown kind — is ``Err(BoundaryError(code=
    "invalid_decision_request"))`` and nothing is ever sent to a provider.
    Only :class:`~nnx.decisions.InvalidDecisionRequest` is converted.
    """
    from .decisions.schema import InvalidDecisionRequest, question_from_state

    # Passed through as given: the question constructors judge every value.
    state: dict[str, Any] = {"kind": kind, "prompt": prompt}
    if kind == "choice":
        state["options"] = options
    elif kind == "score":
        state["levels"] = options
    try:
        return Ok(question_from_state(state))
    except InvalidDecisionRequest as error:
        return Err(
            BoundaryError(
                "invalid_decision_request",
                where="request",
                context={"kind": kind, "prompt": prompt},
                cause=error,
            )
        )


def decide_result(
    provider: Any,
    question: Any,
    inputs: Any,
    *,
    policy: Any = None,
    model_id: Optional[str] = None,
) -> Result[tuple[Any, ...], BoundaryError]:
    """Ask ``provider`` ``question`` for ``inputs``.

    ``Ok`` holds the results, one per input — or, with an abstention
    ``policy`` (and the ``model_id`` it was tuned for), each row's
    ``SelectiveDecision``: an abstention is a success whose probabilities
    and reason are kept. Only the provider's own failures are errors:
    ``InvalidDecisionResponse`` (an answer that does not fit the question)
    becomes ``"invalid_decision_response"``, any other ``ProviderFailure``
    (an outage, a timeout, a refused credential) ``"provider_failure"``, and
    ``UnsupportedCapability`` ``"unsupported"``; the first two keep the
    error's ``request_id`` in ``context``. The codes follow the exception
    type, not whether a retry would help. Anything else propagates.
    """
    from .decisions.schema import InvalidDecisionResponse, ProviderFailure, UnsupportedCapability

    if policy is not None and not isinstance(model_id, str):
        raise TypeError(
            f"decide_result(policy=...) needs the model_id (a str) the policy was tuned for, got {model_id!r}"
        )
    context = {"question": question.digest(), "provider": type(provider).__name__}
    try:
        results = tuple(provider.decide(question, inputs))
    except UnsupportedCapability as error:
        return Err(BoundaryError("unsupported", where="provider.decide", context=context, cause=error))
    except InvalidDecisionResponse as error:  # before ProviderFailure: a malformed Jev answer is both
        context = {**context, "request_id": getattr(error, "request_id", None)}
        return Err(BoundaryError("invalid_decision_response", where="provider.decide", context=context, cause=error))
    except ProviderFailure as error:
        context = {**context, "request_id": getattr(error, "request_id", None)}
        return Err(BoundaryError("provider_failure", where="provider.decide", context=context, cause=error))
    if policy is None:
        return Ok(results)
    from .abstention import decide

    tuned_for = cast(str, model_id)  # checked above: a policy comes with its model_id
    return Ok(tuple(decide(result, policy, model_id=tuned_for) for result in results))


# --- artifact boundaries ----------------------------------------------------------------------


def inspect_bundle_result(path: Union[str, os.PathLike[str]]) -> Result[Any, BoundaryError]:
    """Summarize the run bundle at ``path`` with
    :func:`nnx.bundles.inspect_bundle` — manifest and JSON records only:
    it never unpickles a checkpoint, loads a tensor, calls a factory or a
    registry.

    ``Ok`` holds the ``BundleInfo``. A path that does not exist is
    ``Err(BoundaryError(code="artifact_missing"))``; any other
    ``BundleError`` (not a bundle, a malformed or tampered manifest) is
    ``"bundle_invalid"``, its cause kept. Other exceptions propagate —
    a ``PermissionError`` on the path itself, or a ``TypeError`` for a
    ``path`` that is not a str or ``os.PathLike`` (``where`` is always a
    str).
    """
    from .bundles import BundleError, inspect_bundle

    where = os.fsdecode(path)
    try:
        os.lstat(path)
    except (FileNotFoundError, NotADirectoryError) as missing:  # no such path (a component may be a file)
        return Err(BoundaryError("artifact_missing", where=where, context={}, cause=missing))
    except PermissionError:
        raise
    except OSError:
        pass  # an unusable path (a symlink loop, a name too long): inspect_bundle reports it as a BundleError
    try:
        return Ok(inspect_bundle(path))
    except BundleError as error:
        return Err(BoundaryError("bundle_invalid", where=where, context={"error": type(error).__name__}, cause=error))
