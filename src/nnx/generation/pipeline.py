"""OrderedLogitsPipeline — immutable decoding stages run in declared order (FEAT-037).

:class:`~nnx.generation.LogitsChain` (via its builder) sorts the standard
processors into NNx's canonical order. An :class:`OrderedLogitsPipeline`
does the opposite: it runs its stages **exactly in the order they were
declared**, duplicates and custom positions included, so ordered decoding
configuration can be written down as data::

    from nnx.generation import LogitsStage, OrderedLogitsPipeline

    nucleus_then_cool = OrderedLogitsPipeline.of(LogitsStage.top_p(0.8), LogitsStage.temperature(0.5))
    with_penalty = nucleus_then_cool.prepend(LogitsStage.repetition_penalty(1.2))  # a new pipeline
    model.generate(prompt, logits_pipeline=with_penalty)

- **Immutable.** A pipeline holds a tuple of stages; :meth:`append` and
  :meth:`prepend` return new pipelines. A built-in stage is a frozen spec
  (:class:`LogitsStage`) — a processor passed in is read once and copied
  into one, so mutating that processor later changes nothing — and every
  run compiles fresh processors from the specs.
- **Validated.** Every value is finite and in its domain when the stage is
  made. A zero-temperature stage is greedy: it turns the logits into
  ``±inf`` argmax markers, so it is **terminal** — a stage after it raises
  (nothing is ever reordered).
- **Custom stages.** Any ``(logits, token_history) -> logits`` callable can
  be a :class:`CustomStage` at any position (a subclass of a built-in
  processor stays custom, keeping its own ``__call__``; the terminal rule
  sees only built-in stages; a nested pipeline is flattened into its
  stages). Its state is the caller's (the
  pipeline keeps the reference, never a copy). It is runtime-only unless
  its ``tag`` names a :class:`LogitsStageCodec` registered in this process
  with :func:`register_logits_stage_codec`; a pipeline is never rebuilt by
  importing code named in data, and nothing about it enters a checkpoint
  or a run identity.
- **Versioned round trip.** :meth:`OrderedLogitsPipeline.state` writes
  ``{"version": 1, "stages": [...]}`` with one tagged entry per stage, in
  order; :meth:`OrderedLogitsPipeline.from_state` rebuilds it.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union, cast

import torch

from .logits_processors import (
    LogitsProcessor,
    RepetitionPenalty,
    TemperatureScaling,
    TopKFilter,
    TopPFilter,
    apply_chain,
)

__all__ = [
    "PIPELINE_VERSION",
    "CustomStage",
    "LogitsStage",
    "LogitsStageCodec",
    "OrderedLogitsPipeline",
    "register_logits_stage_codec",
    "registered_logits_stage_codecs",
    "unregister_logits_stage_codec",
]

PIPELINE_VERSION = 1
"""The version written by :meth:`OrderedLogitsPipeline.state`."""

_BUILTIN_KINDS = ("repetition_penalty", "top_k", "top_p", "temperature")


def _real(value: Any, kind: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"a {kind} stage needs a finite number, got {value!r}")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"a {kind} stage needs a finite number, got one too large for a float") from None
    if not math.isfinite(number):
        raise ValueError(f"a {kind} stage needs a finite number, got {value!r}")
    return number


@dataclass(frozen=True)
class LogitsStage:
    """One built-in decoding stage: ``kind`` is ``"repetition_penalty"``
    (``>= 1``), ``"top_k"`` (an integer ``>= 1``), ``"top_p"`` (in ``(0, 1]``)
    or ``"temperature"`` (``>= 0``; ``0`` is greedy and terminal). Values
    are validated here, before any generation."""

    kind: str
    value: Union[int, float]

    def __post_init__(self) -> None:
        if self.kind not in _BUILTIN_KINDS:
            raise ValueError(f"unknown stage kind {self.kind!r}; expected one of {list(_BUILTIN_KINDS)}")
        if self.kind == "top_k":
            if isinstance(self.value, bool) or not isinstance(self.value, numbers.Integral) or self.value < 1:
                raise ValueError(f"a top_k stage needs an integer >= 1, got {self.value!r}")
            object.__setattr__(self, "value", int(self.value))
            return
        value = _real(self.value, self.kind)
        if self.kind == "temperature" and value < 0:
            raise ValueError(f"a temperature stage needs a value >= 0, got {value!r}")
        if self.kind == "top_p" and not 0.0 < value <= 1.0:
            raise ValueError(f"a top_p stage needs a value in (0, 1], got {value!r}")
        if self.kind == "repetition_penalty" and value < 1.0:
            raise ValueError(f"a repetition_penalty stage needs a value >= 1, got {value!r}")
        object.__setattr__(self, "value", value)

    @classmethod
    def temperature(cls, value: float) -> LogitsStage:
        return cls("temperature", value)

    @classmethod
    def top_k(cls, value: int) -> LogitsStage:
        return cls("top_k", value)

    @classmethod
    def top_p(cls, value: float) -> LogitsStage:
        return cls("top_p", value)

    @classmethod
    def repetition_penalty(cls, value: float) -> LogitsStage:
        return cls("repetition_penalty", value)

    @property
    def terminal(self) -> bool:
        """Whether this is a greedy (zero-temperature) stage: it emits
        ``±inf`` argmax markers, so nothing may follow it."""
        return self.kind == "temperature" and self.value == 0.0

    def processor(self) -> LogitsProcessor:
        """A fresh processor for this stage."""
        if self.kind == "temperature":
            return TemperatureScaling(temperature=float(self.value))
        if self.kind == "top_k":
            return TopKFilter(top_k=int(self.value))
        if self.kind == "top_p":
            return TopPFilter(top_p=float(self.value))
        return RepetitionPenalty(penalty=float(self.value))

    def state(self) -> dict[str, Any]:
        return {"kind": self.kind, "value": self.value}


@dataclass(frozen=True, eq=False)
class CustomStage:
    """A user callable ``(logits, token_history) -> logits`` at a declared
    position. Its state belongs to the caller: the pipeline keeps this
    reference and never copies it. ``tag`` names the
    :class:`LogitsStageCodec` that can serialize it; without a registered
    codec the stage is runtime-only."""

    function: Callable[[torch.Tensor, list[int]], torch.Tensor]
    tag: Optional[str] = None

    # Two custom stages are equal only when they hold the same callable
    # object (and tag): callables need not be hashable or comparable.
    def __eq__(self, other: object) -> bool:
        return isinstance(other, CustomStage) and self.function is other.function and self.tag == other.tag

    def __hash__(self) -> int:
        return hash((id(self.function), self.tag))

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise ValueError(f"a custom stage needs a callable, got {self.function!r}")
        if self.tag is not None and (not isinstance(self.tag, str) or not self.tag.strip()):
            raise ValueError(f"a custom stage's tag must be a non-empty string or None, got {self.tag!r}")

    terminal = False

    def processor(self) -> LogitsProcessor:
        return cast(LogitsProcessor, self.function)


Stage = Union[LogitsStage, CustomStage]


@dataclass(frozen=True)
class LogitsStageCodec:
    """How to serialize custom stages tagged ``tag``: ``encode(function)``
    returns a JSON-like mapping, ``decode(mapping)`` returns the callable.
    Registered in-process with :func:`register_logits_stage_codec`; data
    never names code to import."""

    tag: str
    encode: Callable[[Callable[..., Any]], Mapping[str, Any]]
    decode: Callable[[Mapping[str, Any]], Callable[[torch.Tensor, list[int]], torch.Tensor]]

    def __post_init__(self) -> None:
        if not isinstance(self.tag, str) or not self.tag.strip():
            raise ValueError(f"a codec tag must be a non-empty string, got {self.tag!r}")
        if not callable(self.encode) or not callable(self.decode):
            raise ValueError("a codec needs callable encode and decode")


_CODECS: dict[str, LogitsStageCodec] = {}


def register_logits_stage_codec(codec: LogitsStageCodec) -> None:
    """Register ``codec`` for its tag (re-registering a tag replaces it)."""
    if not isinstance(codec, LogitsStageCodec):
        raise TypeError(f"expected a LogitsStageCodec, got {type(codec).__name__}")
    _CODECS[codec.tag] = codec


def unregister_logits_stage_codec(tag: str) -> None:
    """Remove the codec registered for ``tag`` (a missing tag is a no-op)."""
    _CODECS.pop(tag, None)


def registered_logits_stage_codecs() -> tuple[str, ...]:
    """The registered codec tags, in registration order."""
    return tuple(_CODECS)


def _as_stage(value: Any) -> Stage:
    """A stage from a stage, a built-in processor (copied by value) or a
    callable (a runtime-only custom stage)."""
    if isinstance(value, (LogitsStage, CustomStage)):
        return value
    # Exact types only: a subclass may override __call__, so it stays a
    # custom stage with its own behaviour.
    if type(value) is TemperatureScaling:
        return LogitsStage.temperature(value.temperature)
    if type(value) is TopKFilter:
        return LogitsStage.top_k(value.top_k)
    if type(value) is TopPFilter:
        return LogitsStage.top_p(value.top_p)
    if type(value) is RepetitionPenalty:
        return LogitsStage.repetition_penalty(value.penalty)
    if callable(value):
        return CustomStage(cast(Callable[[torch.Tensor, list[int]], torch.Tensor], value))
    raise TypeError(f"a pipeline stage is a LogitsStage, a CustomStage or a callable, got {type(value).__name__}")


def _flatten(values: Sequence[Any]) -> tuple[Stage, ...]:
    """Stages from ``values``, a nested pipeline contributing its own stages
    (so the terminal rule and serialization see them)."""
    stages: list[Stage] = []
    for value in values:
        if isinstance(value, OrderedLogitsPipeline):
            stages.extend(value.stages)
        else:
            stages.append(_as_stage(value))
    return tuple(stages)


def _check_placement(stages: tuple[Stage, ...]) -> None:
    for index, stage in enumerate(stages[:-1]):
        if stage.terminal:
            raise ValueError(
                f"stage {index} is a zero-temperature (greedy) stage and must be last: its ±inf argmax markers "
                f"cannot be processed further, but {len(stages) - index - 1} stage(s) follow it"
            )


@dataclass(frozen=True)
class OrderedLogitsPipeline:
    """Decoding stages run exactly in declared order. See the module
    docstring for immutability, validation, the terminal greedy rule,
    custom stages and serialization."""

    stages: tuple[Stage, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.stages, (str, bytes)) or not isinstance(self.stages, Sequence):
            raise TypeError(f"stages must be a sequence of stages, got {type(self.stages).__name__}")
        stages = _flatten(self.stages)
        _check_placement(stages)
        object.__setattr__(self, "stages", stages)

    @classmethod
    def of(cls, *stages: Any) -> OrderedLogitsPipeline:
        """A pipeline of ``stages`` in this order."""
        return cls(stages)

    def append(self, *stages: Any) -> OrderedLogitsPipeline:
        """A new pipeline with ``stages`` after this one's."""
        return OrderedLogitsPipeline(self.stages + tuple(stages))

    def prepend(self, *stages: Any) -> OrderedLogitsPipeline:
        """A new pipeline with ``stages`` before this one's."""
        return OrderedLogitsPipeline(tuple(stages) + self.stages)

    def __len__(self) -> int:
        return len(self.stages)

    def processors(self) -> list[LogitsProcessor]:
        """Fresh processors for every stage, in order (custom stages are
        the caller's callables)."""
        return [stage.processor() for stage in self.stages]

    def __call__(self, logits: torch.Tensor, token_history: list[int]) -> torch.Tensor:
        """Run every stage in declared order — :func:`apply_chain` over
        :meth:`processors`."""
        return apply_chain(logits, token_history=token_history, processors=self.processors())

    def state(self) -> dict[str, Any]:
        """``{"version": 1, "stages": [...]}``, one tagged entry per stage in
        order. A custom stage needs a registered codec for its tag."""
        entries: list[dict[str, Any]] = []
        for index, stage in enumerate(self.stages):
            if isinstance(stage, LogitsStage):
                entries.append(stage.state())
                continue
            codec = _CODECS.get(stage.tag) if stage.tag is not None else None
            if codec is None:
                raise ValueError(
                    f"stage {index} is a runtime-only custom stage (tag {stage.tag!r}): register a "
                    "LogitsStageCodec for its tag to serialize it"
                )
            entries.append({"kind": "custom", "tag": stage.tag, "config": dict(codec.encode(stage.function))})
        return {"version": PIPELINE_VERSION, "stages": entries}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> OrderedLogitsPipeline:
        """Rebuild a pipeline from :meth:`state`. A custom entry is decoded
        only by a codec registered in this process for its tag."""
        version = state.get("version") if isinstance(state, Mapping) else None
        if type(version) is not int or version != PIPELINE_VERSION:
            raise ValueError(f"unsupported logits pipeline state version {version!r}; expected {PIPELINE_VERSION}")
        entries = state.get("stages")
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ValueError("a logits pipeline state needs a list of stages")
        stages: list[Stage] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ValueError(f"stage {index} must be a mapping, got {entry!r}")
            if entry.get("kind") == "custom":
                unknown = sorted(set(entry) - {"kind", "tag", "config"})
                if unknown:
                    raise ValueError(f"stage {index} has unknown keys {unknown}")
                tag, config = entry.get("tag"), entry.get("config")
                if not isinstance(config, Mapping):
                    raise ValueError(f"stage {index}: a custom stage's config must be a mapping, got {config!r}")
                codec = _CODECS.get(tag) if isinstance(tag, str) else None
                if codec is None:
                    raise ValueError(f"stage {index}: no LogitsStageCodec is registered for tag {tag!r}")
                stages.append(CustomStage(codec.decode(config), tag=tag))
            else:
                unknown = sorted(set(entry) - {"kind", "value"})
                if unknown:
                    raise ValueError(f"stage {index} has unknown keys {unknown}")
                stages.append(LogitsStage(entry.get("kind"), entry.get("value")))  # type: ignore[arg-type]
        return cls(tuple(stages))
