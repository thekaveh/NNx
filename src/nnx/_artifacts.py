"""Frozen JSON artifacts: canonical bytes, equality, hashing, JSON text and
atomic files, all derived from ``state()``. Internal; standard library only
(the atomic writer is imported from ``nnx.nn.params.nn_run`` when saving).

Shared by the calibration and abstention artifacts, so their digests,
equality and file handling cannot drift apart.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping
from functools import cached_property
from typing import Any, Optional, Union, cast

from ._config import _canonical_json, _freeze_config, _FrozenConfig
from ._validation import required_id


def frozen_json(value: Any, what: str, error: type[Exception]) -> _FrozenConfig:
    """An immutable, picklable JSON-like copy of a mapping (nested mappings
    frozen, lists as tuples), so an artifact's state cannot drift from its
    digest; anything else raises ``error``."""
    if not isinstance(value, Mapping):
        raise error(f"{what} must be a mapping, got {value!r}")
    try:
        return _freeze_config(value, "", owner=what)
    except (TypeError, ValueError) as exc:
        raise error(
            f"{what} must be JSON-like with finite numbers: None, bool, int, float, str, lists and str-keyed mappings"
        ) from exc


OVERRIDE_FIELDS = ("labels", "model_id")


def override_record(value: Any, *, error: type[Exception], what: str = "an override record") -> Optional[Any]:
    """A calibration override record (``nnx.calibration``), frozen: ``None``,
    or a name plus the label / model-id mismatches it accepted, each
    ``{expected, actual}``; anything else raises ``error``."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"name", "mismatches"}:
        raise error(f"{what} has exactly 'name' and 'mismatches', got {value!r}")
    required_id(value["name"], "override name", error=error)
    mismatches = value["mismatches"]
    if not isinstance(mismatches, Mapping) or not mismatches or not set(mismatches) <= set(OVERRIDE_FIELDS):
        raise error(f"override mismatches must name some of {OVERRIDE_FIELDS}, got {mismatches!r}")
    for name, pair in mismatches.items():
        if not isinstance(pair, Mapping) or set(pair) != {"expected", "actual"}:
            raise error(f"override mismatch {name!r} must hold 'expected' and 'actual', got {pair!r}")
    return frozen_json(value, "override", error)


def atomic_write(path: Union[str, os.PathLike[str]], text: str) -> None:
    from .nn.params.nn_run import _atomic_write_text

    _atomic_write_text(os.fspath(path), text)


def read_text(path: Union[str, os.PathLike[str]], what: str, error: type[Exception]) -> str:
    """The UTF-8 text of ``path``; a non-UTF-8 file raises ``error``."""
    with open(path, "rb") as handle:
        data = handle.read()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise error(f"the {what} file is UTF-8 JSON; {os.fspath(path)!r} is not text: {exc}") from exc


def parse_json(text: str, what: str, error: type[Exception]) -> Any:
    """Strict JSON: ``NaN`` / ``Infinity`` (which ``json`` would accept), a
    number too large for a float (``1e999``) and malformed text raise
    ``error``."""

    def constant(name: str) -> Any:
        raise error(f"the {what} file is strict JSON; {name} is not a JSON number")

    def number(text: str) -> float:
        value = float(text)
        if not math.isfinite(value):
            raise error(f"the {what} file holds {text}, which overflows a float")
        return value

    try:
        return json.loads(text, parse_constant=constant, parse_float=number)
    except error:
        raise  # the constant hook's own message
    except (ValueError, RecursionError) as exc:  # malformed, an over-long integer, or nesting too deep
        raise error(f"the {what} file is JSON: {exc}") from exc


def check_keys(
    state: Mapping[str, Any],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    what: str,
    error: type[Exception],
) -> None:
    """Refuse a serialized ``state`` with unknown or missing keys, naming both."""
    need = set(required)  # read once: ``required`` may be a one-shot iterable
    allowed = need | set(optional)
    unknown, missing = sorted(set(state) - allowed, key=repr), sorted(need - set(state))  # keys of any type
    problems = [f"has unknown keys {unknown} (expected {sorted(allowed)})"] if unknown else []
    problems += [f"lacks {missing}"] if missing else []
    if problems:
        raise error(f"the {what} " + " and ".join(problems))


class JsonArtifact:
    """Canonical bytes, equality, hashing, ``sha256:`` digest, JSON text and
    atomic saving, all from ``state()``. Subclasses are frozen, so the
    canonical bytes and digest are computed once."""

    def state(self) -> dict[str, Any]:
        raise NotImplementedError

    @cached_property
    def _canonical(self) -> bytes:
        return _canonical_json(self.state())

    def canonical_bytes(self) -> bytes:
        return self._canonical

    @cached_property
    def _digest(self) -> str:
        return f"sha256:{hashlib.sha256(self.canonical_bytes()).hexdigest()}"

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.canonical_bytes() == cast(JsonArtifact, other).canonical_bytes()

    def __hash__(self) -> int:
        return hash(self.canonical_bytes())

    def to_json(self) -> str:
        return json.dumps(self.state(), indent=2, sort_keys=True, allow_nan=False) + "\n"

    def save(self, path: Union[str, os.PathLike[str]]) -> None:
        """Write :meth:`to_json` to ``path`` atomically."""
        atomic_write(path, self.to_json())
