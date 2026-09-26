"""Typed decision questions, validated results and the response validator
(FEAT-009). See :mod:`nnx.decisions` for the overview."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional, Union

__all__ = [
    "PROBABILITY_TOLERANCE",
    "Boolean",
    "BooleanResult",
    "Choice",
    "ChoiceResult",
    "DecisionError",
    "DecisionResult",
    "InvalidDecisionRequest",
    "InvalidDecisionResponse",
    "Option",
    "ProviderFailure",
    "Question",
    "Score",
    "ScoreResult",
    "UnsupportedCapability",
    "question_from_state",
    "validate_response",
]

PROBABILITY_TOLERANCE = 1e-6
"""How far a distribution's sum may be from 1. Never renormalized."""


# --- errors ---------------------------------------------------------------------------


class DecisionError(Exception):
    """Base class of every decision-API error."""


class InvalidDecisionRequest(DecisionError, ValueError):
    """A malformed question: too few options, duplicate or empty ids, empty
    text."""


class InvalidDecisionResponse(DecisionError, ValueError):
    """A provider response that does not fit its question: missing,
    duplicate, unknown or unlabeled ids, or a probability outside ``[0, 1]``
    or a distribution that does not sum to 1 (never renormalized)."""


class UnsupportedCapability(DecisionError):
    """A request the provider cannot serve — a primitive, modality, batch
    size or label space it does not declare. Raised before any model call
    or network I/O."""


class ProviderFailure(DecisionError, RuntimeError):
    """The provider's backend failed while answering a supported, valid
    request (the original error is the ``__cause__``)."""


# --- questions ------------------------------------------------------------------------


@dataclass(frozen=True)
class Option:
    """One answer of a :class:`Choice` or one level of a :class:`Score`.

    ``id`` is bookkeeping — how results are keyed, never shown to a model;
    ``description`` is the model-facing text.
    """

    id: str
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id or self.id != self.id.strip():
            raise InvalidDecisionRequest(
                f"option id must be a non-empty string without surrounding spaces, got {self.id!r}"
            )
        if not isinstance(self.description, str) or not self.description.strip():
            raise InvalidDecisionRequest(f"option {self.id!r} needs a non-empty description, got {self.description!r}")


def _option(value: Any, *, owner: str) -> Option:
    if isinstance(value, Option):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        return Option(value[0], value[1])
    raise InvalidDecisionRequest(
        f"{owner} options are Option(id, description) or (id, description) pairs, got {value!r}"
    )


def _options(values: Iterable[Any], *, owner: str, noun: str) -> tuple[Option, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidDecisionRequest(f"{owner} {noun}s must be a sequence of options, got {values!r}")
    options = tuple(_option(value, owner=owner) for value in values)
    if len(options) < 2:
        raise InvalidDecisionRequest(f"{owner} needs at least 2 {noun}s, got {len(options)}")
    ids = [option.id for option in options]
    duplicates = sorted({option_id for option_id in ids if ids.count(option_id) > 1})
    if duplicates:
        raise InvalidDecisionRequest(f"{owner} {noun} ids must be unique; duplicated: {duplicates}")
    return options


class Question:
    """Base of the decision primitives. ``kind`` is the discriminator;
    :meth:`digest` identifies the exact question (kind, prompt, and every
    option's id and description in order) so a result can never be matched
    to a reordered or reworded question."""

    kind: ClassVar[str]
    prompt: str

    def _check_prompt(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise InvalidDecisionRequest(f"{type(self).__name__} needs a non-empty prompt, got {self.prompt!r}")

    def state(self) -> dict[str, Any]:
        raise NotImplementedError  # pragma: no cover - abstract

    def digest(self) -> str:
        """SHA-256 of the canonical question state (computed once; questions
        are immutable)."""
        cached = self.__dict__.get("_digest")
        if cached is None:
            canonical = json.dumps(self.state(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            cached = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            object.__setattr__(self, "_digest", cached)
        return cached

    def model_view(self) -> dict[str, Any]:
        """What a model may see: the prompt and the option texts in order —
        never the bookkeeping ids."""
        raise NotImplementedError  # pragma: no cover - abstract

    @property
    def option_ids(self) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True)
class Choice(Question):
    """Pick one of 2+ options; answered by a distribution over the options."""

    kind: ClassVar[str] = "choice"
    prompt: str
    options: tuple[Option, ...]

    def __post_init__(self) -> None:
        self._check_prompt()
        object.__setattr__(self, "options", _options(self.options, owner="Choice", noun="option"))

    @property
    def option_ids(self) -> tuple[str, ...]:
        return tuple(option.id for option in self.options)

    def state(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "prompt": self.prompt,
            "options": [[option.id, option.description] for option in self.options],
        }

    def model_view(self) -> dict[str, Any]:
        return {"prompt": self.prompt, "options": [option.description for option in self.options]}


@dataclass(frozen=True)
class Boolean(Question):
    """A yes / no question; answered by one probability ``p_true``."""

    kind: ClassVar[str] = "boolean"
    prompt: str

    def __post_init__(self) -> None:
        self._check_prompt()

    def state(self) -> dict[str, Any]:
        return {"kind": self.kind, "prompt": self.prompt}

    def model_view(self) -> dict[str, Any]:
        return {"prompt": self.prompt}


@dataclass(frozen=True)
class Score(Question):
    """Place the input on 2+ ordered levels (lowest first); answered by a
    distribution over the levels. The levels are **ordinal**: their order is
    meaningful, their spacing is not."""

    kind: ClassVar[str] = "score"
    prompt: str
    levels: tuple[Option, ...]

    def __post_init__(self) -> None:
        self._check_prompt()
        object.__setattr__(self, "levels", _options(self.levels, owner="Score", noun="level"))

    @property
    def option_ids(self) -> tuple[str, ...]:
        return tuple(level.id for level in self.levels)

    def state(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "prompt": self.prompt,
            "levels": [[level.id, level.description] for level in self.levels],
        }

    def model_view(self) -> dict[str, Any]:
        return {"prompt": self.prompt, "levels": [level.description for level in self.levels]}


def question_from_state(state: Mapping[str, Any]) -> Union[Choice, Boolean, Score]:
    """Rebuild a question from :meth:`Question.state`, by its ``kind``;
    malformed state raises :class:`InvalidDecisionRequest`."""
    if not isinstance(state, Mapping):
        raise InvalidDecisionRequest(f"question state must be a mapping, got {type(state).__name__}")
    kind = state.get("kind")
    fields = {"choice": ("prompt", "options"), "boolean": ("prompt",), "score": ("prompt", "levels")}
    required = fields.get(kind) if isinstance(kind, str) else None
    if required is None:
        raise InvalidDecisionRequest(f"unknown decision kind {kind!r} (expected 'choice', 'boolean' or 'score')")
    missing = [key for key in required if key not in state]
    if missing:
        raise InvalidDecisionRequest(f"{kind} state is missing {missing}")
    if kind == Choice.kind:
        return Choice(state["prompt"], state["options"])
    if kind == Boolean.kind:
        return Boolean(state["prompt"])
    return Score(state["prompt"], state["levels"])


# --- results --------------------------------------------------------------------------


def _probability(value: Any, *, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise InvalidDecisionResponse(f"{what} must be a number in [0, 1], got {value!r}")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise InvalidDecisionResponse(f"{what} must be a finite number in [0, 1], got {value!r}")
    return probability


def _check_distribution(pairs: tuple[tuple[str, float], ...], *, owner: str) -> tuple[tuple[str, float], ...]:
    if len(pairs) < 2:
        raise InvalidDecisionResponse(f"{owner} needs a distribution over at least 2 ids, got {len(pairs)}")
    ids = [option_id for option_id, _ in pairs]
    if len(set(ids)) != len(ids):
        raise InvalidDecisionResponse(f"{owner} distribution repeats ids: {ids}")
    checked = tuple((option_id, _probability(p, what=f"{owner} p[{option_id!r}]")) for option_id, p in pairs)
    total = math.fsum(p for _, p in checked)
    if abs(total - 1.0) > PROBABILITY_TOLERANCE:
        raise InvalidDecisionResponse(
            f"{owner} distribution sums to {total!r}, not 1 within {PROBABILITY_TOLERANCE}; it is never renormalized"
        )
    return checked


@dataclass(frozen=True)
class ChoiceResult:
    """A validated answer to a :class:`Choice`: ``distribution`` in the
    question's option order. ``raw`` is the provider's own output, kept
    apart from the normalized fields (and from equality)."""

    question_digest: str
    distribution: tuple[tuple[str, float], ...]
    provider: Optional[str] = None
    raw: Any = field(default=None, compare=False, repr=False)
    kind: ClassVar[str] = "choice"

    def __post_init__(self) -> None:
        object.__setattr__(self, "distribution", _check_distribution(tuple(self.distribution), owner="ChoiceResult"))

    def probabilities(self) -> dict[str, float]:
        return dict(self.distribution)

    @property
    def top(self) -> str:
        """The most probable option id (the first one on ties)."""
        return max(self.distribution, key=lambda item: item[1])[0]


@dataclass(frozen=True)
class BooleanResult:
    """A validated answer to a :class:`Boolean`: one ``p_true`` in
    ``[0, 1]``."""

    question_digest: str
    p_true: float
    provider: Optional[str] = None
    raw: Any = field(default=None, compare=False, repr=False)
    kind: ClassVar[str] = "boolean"

    def __post_init__(self) -> None:
        object.__setattr__(self, "p_true", _probability(self.p_true, what="BooleanResult p_true"))

    @property
    def p_false(self) -> float:
        return 1.0 - self.p_true


@dataclass(frozen=True)
class ScoreResult:
    """A validated answer to a :class:`Score`: ``distribution`` over the
    levels in the question's order (lowest first).

    :attr:`expected_index` is ``sum(i * p_i)`` over zero-based levels — an
    **ordinal** summary (a position between levels), not an interval-scale
    score: the levels' spacing is undefined. A provider's own numeric
    score, if it reports one, stays in ``vendor_score`` and is never mixed
    into the distribution.
    """

    question_digest: str
    distribution: tuple[tuple[str, float], ...]
    provider: Optional[str] = None
    vendor_score: Optional[float] = None
    raw: Any = field(default=None, compare=False, repr=False)
    kind: ClassVar[str] = "score"

    def __post_init__(self) -> None:
        object.__setattr__(self, "distribution", _check_distribution(tuple(self.distribution), owner="ScoreResult"))

    def probabilities(self) -> dict[str, float]:
        return dict(self.distribution)

    @property
    def expected_index(self) -> float:
        return math.fsum(index * p for index, (_, p) in enumerate(self.distribution))


DecisionResult = Union[ChoiceResult, BooleanResult, ScoreResult]


# --- the validator -----------------------------------------------------------------------


def _keyed_pairs(response: Any) -> list[tuple[Any, Any]]:
    if isinstance(response, Mapping):
        return list(response.items())
    if isinstance(response, Sequence) and not isinstance(response, (str, bytes)):
        pairs = []
        for item in response:
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
                raise InvalidDecisionResponse(f"keyed output entries must be (id, probability) pairs, got {item!r}")
            pairs.append((item[0], item[1]))
        return pairs
    raise InvalidDecisionResponse(
        f"keyed output must be a mapping or a sequence of (id, probability) pairs, got {type(response).__name__}"
    )


def validate_response(
    question: Union[Choice, Boolean, Score],
    response: Any,
    *,
    provider: Optional[str] = None,
    raw: Any = None,
    vendor_score: Optional[float] = None,
) -> DecisionResult:
    """Validate a provider's keyed output against ``question`` and return the
    normalized result.

    For a :class:`Choice` or :class:`Score`, ``response`` maps option ids to
    probabilities (a mapping, or ``(id, p)`` pairs in any order). The result
    is reordered into the question's order; a missing, duplicate, unknown or
    unlabeled (empty / non-string) id, a probability outside ``[0, 1]`` and
    a distribution not summing to 1 within :data:`PROBABILITY_TOLERANCE`
    raise :class:`InvalidDecisionResponse` — a malformed distribution is
    never renormalized. For a :class:`Boolean`, ``response`` is ``p_true``
    or ``{"true": p, "false": 1 - p}``. ``raw`` (the provider's own output)
    and ``vendor_score`` (a :class:`Score` provider's own number) are kept
    apart from the normalized fields.
    """
    digest = question.digest()
    if isinstance(question, Boolean):
        if isinstance(response, Mapping):
            keys = set(response)
            if keys != {"true", "false"}:
                raise InvalidDecisionResponse(
                    f"a Boolean response keys 'true' and 'false', got {sorted(map(str, keys))}"
                )
            p_true = _probability(response["true"], what="p['true']")
            p_false = _probability(response["false"], what="p['false']")
            if abs(p_true + p_false - 1.0) > PROBABILITY_TOLERANCE:
                raise InvalidDecisionResponse(
                    f"p['true'] + p['false'] = {p_true + p_false!r}, not 1; never renormalized"
                )
        else:
            p_true = _probability(response, what="p_true")
        return BooleanResult(digest, p_true, provider=provider, raw=raw)
    if not isinstance(question, (Choice, Score)):
        raise InvalidDecisionRequest(f"not a decision question: {type(question).__name__}")

    expected = question.option_ids
    seen: dict[str, float] = {}
    for key, value in _keyed_pairs(response):
        if not isinstance(key, str) or not key.strip():
            raise InvalidDecisionResponse(f"unlabeled entry in the keyed output: id {key!r}")
        if key in seen:
            raise InvalidDecisionResponse(f"duplicate id {key!r} in the keyed output")
        if key not in expected:
            raise InvalidDecisionResponse(f"unknown id {key!r}; the question's ids are {list(expected)}")
        seen[key] = _probability(value, what=f"p[{key!r}]")
    missing = [option_id for option_id in expected if option_id not in seen]
    if missing:
        raise InvalidDecisionResponse(f"missing ids {missing} in the keyed output")
    ordered = tuple((option_id, seen[option_id]) for option_id in expected)
    if isinstance(question, Choice):
        return ChoiceResult(digest, ordered, provider=provider, raw=raw)
    if vendor_score is not None and (
        isinstance(vendor_score, bool) or not isinstance(vendor_score, numbers.Real) or not math.isfinite(vendor_score)
    ):
        raise InvalidDecisionResponse(f"vendor_score must be a finite number, got {vendor_score!r}")
    return ScoreResult(
        digest,
        ordered,
        provider=provider,
        vendor_score=None if vendor_score is None else float(vendor_score),
        raw=raw,
    )
