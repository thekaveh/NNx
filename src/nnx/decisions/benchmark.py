"""A reproducible, offline-first decision-provider benchmark (FEAT-021).

Score decision providers (``nnx.decisions``) on **identical samples**, join
their outputs by sample id and question digest, and report probability
metrics — replayed from saved records, with no provider and no network.

- :class:`Sample` — one benchmark item: a sample ``id`` (the join key), the
  typed ``question`` (its :meth:`~nnx.decisions.Question.digest` is the
  schema digest), the ``input`` the provider reads, the true ``label``, its
  task ``family`` (``heldout`` families report apart), its grouping unit
  (``group``, for intervals) and the ``perturbation`` that produced it
  (:func:`permute_options`, :func:`redescribe`, :func:`add_distractors`,
  :func:`add_none_of_the_above`, :func:`add_context`, :func:`rewrite_input`).
- :class:`Record` — one provider output for one sample: the answer or the
  reason there is none (``"answered"``, ``"unsupported"``, ``"failed"``),
  the provider and model revision, the prompt identity and execution
  metadata. :func:`write_records` / :func:`read_records` keep every field
  across a JSONL round trip (``nnx.decision-record/1``).
- :func:`collect` — the **live** collector: it needs an explicit provider,
  provider id and :class:`Budget`, makes one attempt per batch (retries
  belong to the provider), records a batch the budget cut short as
  ``partial_batch`` and stops when the budget is spent.
- :func:`evaluate` — **replay**: joins samples and records (never by row
  position) and reports, per slice (in-family, held-out, each family and
  each perturbation), the coverage — eligible, missing, duplicate, extra,
  mismatched-digest, unsupported and failed counts, with reasons — and
  accuracy, macro-F1, NLL (exact by default), Brier, reliability bins with
  ECE and, given an abstention policy, selective coverage and risk. A
  metric that cannot be computed is reported unavailable with its reason
  and denominator. Replay fits nothing: no calibrator, no threshold.
- :func:`bootstrap_interval` — resamples the declared grouping units with a
  recorded seed and flags degenerate samples.
- :class:`BenchmarkReport` — JSON, CSV and text exports that agree on
  units, eligible and failure counts and unavailable states;
  :func:`compare_reports` refuses reports of another split or metric
  identity. :class:`Resources` declares how cost was measured — or leaves
  it ``None``.

Metrics are defined per row over each question's own options, so samples
with different label spaces share one report: NLL and Brier are the named
``nll`` / ``brier`` metrics' terms (``nnx.calibration``), a Boolean is the
two options ``("true", "false")`` and the top-1 prediction is the first
option of maximal probability.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import numbers
import os
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Union

from .schema import (
    PROBABILITY_TOLERANCE,
    Boolean,
    Choice,
    DecisionError,
    InvalidDecisionRequest,
    InvalidDecisionResponse,
    Option,
    Score,
    UnsupportedCapability,
    validate_response,
)

__all__ = [
    "FORMAT",
    "RECORD_FORMAT",
    "BenchmarkError",
    "BenchmarkReport",
    "Budget",
    "Collection",
    "Coverage",
    "Interval",
    "MetricValue",
    "Record",
    "Resources",
    "Sample",
    "SliceReport",
    "add_context",
    "add_distractors",
    "add_none_of_the_above",
    "bootstrap_interval",
    "collect",
    "compare_reports",
    "evaluate",
    "permute_options",
    "read_records",
    "redescribe",
    "rewrite_input",
    "write_records",
]

FORMAT = "nnx.decision-benchmark/1"
RECORD_FORMAT = "nnx.decision-record/1"
STATUSES = ("answered", "unsupported", "failed")
METRICS = ("accuracy", "macro_f1", "nll", "brier", "ece", "selective_coverage", "selective_risk")
UNITS = {
    "accuracy": "fraction of eligible rows",
    "macro_f1": "mean per-label F1",
    "nll": "nats per row",
    "brier": "class-summed squared error per row (0-2)",
    "ece": "fraction (count-weighted |accuracy - confidence|)",
    "selective_coverage": "fraction of eligible rows accepted",
    "selective_risk": "fraction of accepted rows incorrect",
}
Question = Union[Choice, Boolean, Score]


class BenchmarkError(DecisionError, ValueError):
    """A malformed sample, record, budget or report, or reports that cannot
    be compared."""


def _text(value: Any, what: str, *, optional: bool = False) -> Optional[str]:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{what} must be a non-empty string{' or None' if optional else ''}, got {value!r}")
    return value


def _labels(question: Question) -> tuple[str, ...]:
    return ("true", "false") if isinstance(question, Boolean) else question.option_ids


# --- samples and perturbations ------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """One benchmark item.

    Attributes:
        id: the sample id — how records join (never row position).
        question: the typed question; its digest is the schema identity.
        input: what the provider reads for this sample (a text, a tensor
            row, an array row).
        label: the true option id (Choice / Score) or ``True`` / ``False``
            (Boolean).
        family: the task family.
        heldout: whether the family is held out (reported apart from
            in-family rows).
        group: the grouping unit intervals resample (default: the id).
        perturbation: what produced this sample from another (``None``:
            an original).
    """

    id: str
    question: Question
    input: Any
    label: Union[str, bool]
    family: str = "default"
    heldout: bool = False
    group: Optional[str] = None
    perturbation: Optional[str] = None

    def __post_init__(self) -> None:
        _text(self.id, "Sample.id")
        if not isinstance(self.question, (Choice, Boolean, Score)):
            raise BenchmarkError(f"Sample.question must be a Choice, Boolean or Score, got {self.question!r}")
        if isinstance(self.question, Boolean):
            if not isinstance(self.label, bool):
                raise BenchmarkError(f"sample {self.id!r}: a Boolean's label is True or False, got {self.label!r}")
        elif self.label not in self.question.option_ids:
            raise BenchmarkError(
                f"sample {self.id!r}: label {self.label!r} is not one of {list(self.question.option_ids)}"
            )
        _text(self.family, "Sample.family")
        if not isinstance(self.heldout, bool):
            raise BenchmarkError(f"Sample.heldout must be a bool, got {self.heldout!r}")
        _text(self.group, "Sample.group", optional=True)
        _text(self.perturbation, "Sample.perturbation", optional=True)

    @property
    def digest(self) -> str:
        return self.question.digest()

    @property
    def unit(self) -> str:
        """The grouping unit (``group``, else the id)."""
        return self.group if self.group is not None else self.id

    @property
    def true_label(self) -> str:
        if isinstance(self.label, bool):
            return "true" if self.label else "false"
        return self.label


def _fingerprint(value: Any) -> str:
    """A digest of an input that is the same in every process: text as
    UTF-8, a numeric array or tensor as its dtype, shape and bytes, a tuple
    (several inputs) part by part, other JSON-able values as canonical JSON.
    Anything else has no stable digest: the caller must name the derived
    sample (``id=``)."""
    import numpy as np

    if isinstance(value, str):
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
    if isinstance(value, tuple):
        parts = ",".join(_fingerprint(part) for part in value)
        return hashlib.sha256(f"tuple:{parts}".encode()).hexdigest()
    if hasattr(value, "detach") and hasattr(value, "dtype"):  # a tensor
        tensor = value.detach().cpu().contiguous()
        header = f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode()
        raw = tensor.reshape(-1).view(_byte_dtype()).numpy().tobytes() if tensor.numel() else b""
        return hashlib.sha256(header + raw).hexdigest()
    if isinstance(value, np.ndarray) and value.dtype != object:
        array = np.ascontiguousarray(value)
        return hashlib.sha256(f"array:{array.dtype.str}:{array.shape}:".encode() + array.tobytes()).hexdigest()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise BenchmarkError(
            f"an input of type {type(value).__name__} has no stable digest to derive a sample id from; pass id="
        ) from error
    return hashlib.sha256(f"json:{text}".encode()).hexdigest()


def _byte_dtype() -> Any:
    import torch

    return torch.uint8


def _derived(sample: Sample, kind: str, id: Optional[str] = None, **changes: Any) -> Sample:
    """A perturbed copy: a new id (by default the original's, the kind and a
    digest of what changed, so two different perturbations of one kind never
    collide), the perturbation recorded and the original's grouping unit kept
    (so intervals resample them together)."""
    if id is None:
        question = changes.get("question", sample.question)
        change = hashlib.sha256(
            f"{question.digest()}|{_fingerprint(changes.get('input', sample.input))}|{changes.get('label', sample.label)}".encode()
        ).hexdigest()[:8]
        id = f"{sample.id}~{kind}~{change}"
    return replace(sample, id=id, perturbation=kind, group=sample.unit, **changes)


def _options(sample: Sample) -> tuple[Option, ...]:
    if isinstance(sample.question, Boolean):
        raise BenchmarkError(f"sample {sample.id!r}: a Boolean has no options to perturb")
    return sample.question.options if isinstance(sample.question, Choice) else sample.question.levels


def _with_options(sample: Sample, options: Sequence[Option]) -> Question:
    if isinstance(sample.question, Score):
        return Score(sample.question.prompt, tuple(options))
    return Choice(sample.question.prompt, tuple(options))


def permute_options(sample: Sample, order: Sequence[str], *, id: Optional[str] = None) -> Sample:
    """The same options in another order (``order``: every option id once).
    A Score's levels are ordered, so it is refused."""
    if isinstance(sample.question, Score):
        raise BenchmarkError("a Score's levels are ordered; permuting them changes the question")
    by_id = {option.id: option for option in _options(sample)}
    if sorted(order) != sorted(by_id):
        raise BenchmarkError(f"order must name every option id once: {sorted(by_id)}")
    return _derived(sample, "permutation", id, question=_with_options(sample, [by_id[i] for i in order]))


def redescribe(sample: Sample, descriptions: Mapping[str, str], *, id: Optional[str] = None) -> Sample:
    """New descriptions for some options (``id -> description``): the label
    space is the same, the text the provider sees is not."""
    options = [Option(o.id, descriptions.get(o.id, o.description)) for o in _options(sample)]
    unknown = sorted(set(descriptions) - {o.id for o in options})
    if unknown:
        raise BenchmarkError(f"no options {unknown} to redescribe")
    return _derived(sample, "new_descriptions", id, question=_with_options(sample, options))


def add_distractors(
    sample: Sample, distractors: Sequence[Union[Option, tuple[str, str]]], *, id: Optional[str] = None
) -> Sample:
    """Extra wrong options appended to a Choice."""
    if not isinstance(sample.question, Choice):
        raise BenchmarkError("distractors apply to a Choice")
    extra = [o if isinstance(o, Option) else Option(*o) for o in distractors]
    return _derived(sample, "distractors", id, question=_with_options(sample, [*_options(sample), *extra]))


def add_none_of_the_above(
    sample: Sample,
    option: Union[Option, tuple[str, str]] = ("none", "None of the above"),
    *,
    remove_label: bool = True,
    id: Optional[str] = None,
) -> Sample:
    """A none-of-the-above option; with ``remove_label`` the true option is
    removed, so none-of-the-above becomes the answer."""
    if not isinstance(sample.question, Choice):
        raise BenchmarkError("none-of-the-above applies to a Choice")
    none = option if isinstance(option, Option) else Option(*option)
    kept = [o for o in _options(sample) if not (remove_label and o.id == sample.label)]
    label = none.id if remove_label else sample.label
    return _derived(sample, "none_of_the_above", id, question=_with_options(sample, [*kept, none]), label=label)


def add_context(sample: Sample, context: str, *, kind: str = "long_context", id: Optional[str] = None) -> Sample:
    """Irrelevant context appended to a text input."""
    if not isinstance(sample.input, str):
        raise BenchmarkError("context applies to a text input")
    return _derived(sample, kind, id, input=f"{sample.input}\n\n{_text(context, 'context')}")


def rewrite_input(sample: Sample, text: str, *, kind: str, id: Optional[str] = None) -> Sample:
    """The same question over a rewritten text input — a translation
    (``kind="multilingual"``) or an adversarial rewrite
    (``kind="adversarial"``)."""
    if not isinstance(sample.input, str):
        raise BenchmarkError("a rewrite applies to a text input")
    _text(kind, "kind")
    return _derived(sample, kind, id, input=_text(text, "text"))


# --- records ------------------------------------------------------------------------------------


def _real(value: Any) -> Optional[float]:
    """``float(value)`` for a real number (never a bool), else ``None`` —
    including an integer too large for a float."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        return float(value)
    except (OverflowError, ValueError):
        return None


def _finite_probability(value: Any, what: str) -> float:
    number = _real(value)
    if number is None or not 0.0 <= number <= 1.0:
        raise BenchmarkError(f"{what} must be a probability in [0, 1], got {value!r}")
    return number


@dataclass(frozen=True)
class Record:
    """One provider output for one sample.

    ``status`` is ``"answered"`` (with ``distribution`` for a Choice or
    Score, ``p_true`` for a Boolean), ``"unsupported"`` (the provider
    declared it cannot serve the request) or ``"failed"`` (the call raised);
    the latter two carry a ``reason``. ``revision`` is the model revision,
    ``prompt_identity`` how the provider rendered the question (a digest of
    its settings), ``execution`` free JSON metadata (batch, timing, whether
    the budget cut the batch short)."""

    sample_id: str
    question_digest: str
    provider: str
    status: str
    distribution: Optional[tuple[tuple[str, float], ...]] = None
    p_true: Optional[float] = None
    reason: Optional[str] = None
    revision: Optional[str] = None
    prompt_identity: Optional[str] = None
    execution: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.sample_id, "Record.sample_id")
        _text(self.question_digest, "Record.question_digest")
        _text(self.provider, "Record.provider")
        if self.status not in STATUSES:
            raise BenchmarkError(f"Record.status must be one of {STATUSES}, got {self.status!r}")
        if self.status == "answered":
            if (self.distribution is None) == (self.p_true is None):
                raise BenchmarkError(f"record {self.sample_id!r}: an answer is a distribution or a p_true")
            if self.reason is not None:
                raise BenchmarkError(f"record {self.sample_id!r}: an answered record has no reason")
        else:
            _text(self.reason, f"record {self.sample_id!r}'s reason")
            if self.distribution is not None or self.p_true is not None:
                raise BenchmarkError(f"record {self.sample_id!r}: a {self.status} record has no answer")
        if self.distribution is not None:
            try:
                pairs = tuple((k, _finite_probability(p, "a record probability")) for k, p in self.distribution)
            except BenchmarkError:
                raise
            except (TypeError, ValueError) as error:
                raise BenchmarkError(
                    f"record {self.sample_id!r}: a distribution is (option id, probability) pairs, "
                    f"got {self.distribution!r}"
                ) from error
            if not all(isinstance(k, str) and k for k, _ in pairs):
                raise BenchmarkError(f"record {self.sample_id!r}: option ids are non-empty strings")
            if len({k for k, _ in pairs}) != len(pairs) or len(pairs) < 2:
                raise BenchmarkError(f"record {self.sample_id!r}: a distribution needs 2+ distinct option ids")
            total = math.fsum(p for _, p in pairs)
            if abs(total - 1.0) > PROBABILITY_TOLERANCE:
                raise BenchmarkError(f"record {self.sample_id!r}: the distribution sums to {total!r}, not 1")
            object.__setattr__(self, "distribution", pairs)
        if self.p_true is not None:
            object.__setattr__(self, "p_true", _finite_probability(self.p_true, "Record.p_true"))
        for name in ("revision", "prompt_identity"):
            _text(getattr(self, name), f"Record.{name}", optional=True)
        if not isinstance(self.execution, Mapping):
            raise BenchmarkError(f"Record.execution must be a mapping, got {self.execution!r}")
        object.__setattr__(self, "execution", dict(self.execution))

    def state(self) -> dict[str, Any]:
        return {
            "format": RECORD_FORMAT,
            "sample_id": self.sample_id,
            "question_digest": self.question_digest,
            "provider": self.provider,
            "status": self.status,
            "distribution": None if self.distribution is None else [list(pair) for pair in self.distribution],
            "p_true": self.p_true,
            "reason": self.reason,
            "revision": self.revision,
            "prompt_identity": self.prompt_identity,
            "execution": dict(self.execution),
        }

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> Record:
        if not isinstance(state, Mapping) or state.get("format") != RECORD_FORMAT:
            raise BenchmarkError(f"not a {RECORD_FORMAT} record: {state!r}")
        known = {"format", "sample_id", "question_digest", "provider", "status", "distribution", "p_true"}
        known |= {"reason", "revision", "prompt_identity", "execution"}
        unknown = sorted(set(state) - known)
        if unknown:
            raise BenchmarkError(f"a record has unknown keys {unknown}")
        distribution = state.get("distribution")
        return Record(
            sample_id=state.get("sample_id"),  # type: ignore[arg-type]
            question_digest=state.get("question_digest"),  # type: ignore[arg-type]
            provider=state.get("provider"),  # type: ignore[arg-type]
            status=state.get("status"),  # type: ignore[arg-type]
            distribution=distribution,
            p_true=state.get("p_true"),
            reason=state.get("reason"),
            revision=state.get("revision"),
            prompt_identity=state.get("prompt_identity"),
            execution={} if state.get("execution") is None else state["execution"],
        )


def write_records(path: Union[str, os.PathLike[str]], records: Iterable[Record]) -> None:
    """One JSON object per line (strict JSON, sorted keys)."""
    lines = [json.dumps(record.state(), sort_keys=True, allow_nan=False) for record in records]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("".join(line + "\n" for line in lines))


def read_records(path: Union[str, os.PathLike[str]]) -> list[Record]:
    """The records of a JSONL file written by :func:`write_records`."""

    from .._artifacts import parse_json, read_text

    records = []
    for number, line in enumerate(read_text(path, "records", BenchmarkError).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(Record.from_state(parse_json(line, "records", BenchmarkError)))
        except BenchmarkError as error:
            raise BenchmarkError(f"{os.fspath(path)}:{number}: {error}") from error
    return records


# --- the live collector -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Budget:
    """What a live collection may spend: ``max_calls`` provider calls and,
    optionally, ``max_samples`` samples. A batch the sample budget cuts
    short is sent shortened and recorded as ``partial_batch``."""

    max_calls: int
    max_samples: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("max_calls", "max_samples"):
            value = getattr(self, name)
            if value is None and name == "max_samples":
                continue
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise BenchmarkError(f"Budget.{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class Collection:
    """A live collection: the records, the provider calls made, whether every
    sample was attempted (``complete``) and, if not, why."""

    records: tuple[Record, ...]
    calls: int
    complete: bool
    stopped: Optional[str] = None


def _batch_input(inputs: list[Any]) -> Any:
    first = inputs[0]
    if isinstance(first, str):
        return list(inputs)
    if isinstance(first, tuple):  # several inputs per sample: batch each part
        if any(not isinstance(item, tuple) or len(item) != len(first) for item in inputs):
            raise BenchmarkError("samples with several inputs need tuples of one length")
        return tuple(_batch_input(list(part)) for part in zip(*inputs, strict=True))
    try:
        import torch

        if isinstance(first, torch.Tensor):
            return torch.stack(inputs)
    except ImportError:  # pragma: no cover - torch is a core dependency
        pass
    import numpy as np

    if isinstance(first, np.ndarray):
        return np.stack(inputs)
    return list(inputs)


def _identity(provider: Any) -> Optional[str]:
    record = getattr(provider, "record", None)
    if not callable(record):
        return None
    try:
        text = json.dumps(record(), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):  # not plain JSON: no stable identity
        return None
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _answer(sample: Sample, result: Any, provider_id: str) -> dict[str, Any]:
    """A provider result as a record's answer: it must answer this sample's
    question (its digest); the shared validator checks it and puts a
    distribution into the question's option order."""
    digest = getattr(result, "question_digest", None)
    if digest != sample.digest:
        raise InvalidDecisionResponse(
            f"sample {sample.id!r}: the result answers question {digest!r}, not {sample.digest!r}"
        )
    if isinstance(sample.question, Boolean):
        checked: Any = validate_response(sample.question, getattr(result, "p_true", None), provider=provider_id)
        return {"p_true": float(checked.p_true)}
    pairs = getattr(result, "distribution", None)
    if pairs is None:
        raise InvalidDecisionResponse(f"sample {sample.id!r}: the result carries no distribution")
    reordered: Any = validate_response(sample.question, tuple(pairs), provider=provider_id)  # duplicates are seen
    return {"distribution": tuple((option_id, float(p)) for option_id, p in reordered.distribution)}


def _records(
    batch: Sequence[Sample],
    status: str,
    reason: Optional[str],
    answers: Optional[Sequence[Mapping[str, Any]]],
    execution: Mapping[str, Any],
    common: Mapping[str, Any],
) -> list[Record]:
    return [
        Record(
            sample_id=sample.id,
            question_digest=sample.digest,
            status=status,
            reason=reason,
            execution=dict(execution),
            **common,
            **(answers[position] if answers is not None else {}),
        )
        for position, sample in enumerate(batch)
    ]


def collect(
    provider: Any,
    samples: Sequence[Sample],
    *,
    provider_id: str,
    budget: Budget,
    batch_size: int = 16,
    revision: Optional[str] = None,
    prompt_identity: Optional[str] = None,
) -> Collection:
    """Ask ``provider`` every sample's question, in batches of samples that
    share one question (at most ``batch_size`` inputs per call).

    The provider and its id and the :class:`Budget` are explicit. Each batch
    is attempted once: a provider that raises gives ``"failed"`` records
    (retries are the provider's own), a request the provider declares it
    cannot serve (its ``check(question, inputs)``, or an
    ``UnsupportedCapability`` from ``decide``) gives ``"unsupported"``
    records. Collection stops when the budget is spent; samples never
    attempted have no record (``missing`` when evaluated).
    ``prompt_identity`` defaults to a digest of the provider's ``record()``.
    """
    if provider is None or not callable(getattr(provider, "decide", None)):
        raise BenchmarkError("collect() needs an explicit provider with decide(question, inputs)")
    _text(provider_id, "provider_id")
    _text(revision, "revision", optional=True)
    _text(prompt_identity, "prompt_identity", optional=True)
    if not isinstance(budget, Budget):
        raise BenchmarkError(f"collect() needs an explicit Budget, got {budget!r}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, numbers.Integral) or batch_size < 1:
        raise BenchmarkError(f"batch_size must be a positive integer, got {batch_size!r}")
    samples = list(samples)  # iterated twice: a generator is read once
    identity = prompt_identity if prompt_identity is not None else _identity(provider)
    batches: dict[str, list[Sample]] = {}
    for sample in samples:
        if not isinstance(sample, Sample):
            raise BenchmarkError(f"collect() needs Samples, got {type(sample).__name__}")
        batches.setdefault(sample.digest, []).append(sample)
    repeated = sorted(i for i, n in Counter(sample.id for sample in samples).items() if n > 1)
    if repeated:  # refused before any call: such a collection could never be scored
        raise BenchmarkError(f"duplicate sample ids {repeated}")
    size = _batch_cap(provider, batch_size)
    plan = [group[i : i + size] for group in batches.values() for i in range(0, len(group), size)]
    records: list[Record] = []
    calls = sent = 0
    stopped = None
    common = {"provider": provider_id, "revision": revision, "prompt_identity": identity}
    for index, planned in enumerate(plan):
        if calls >= budget.max_calls:
            stopped = f"budget exhausted: max_calls={budget.max_calls}"
            break
        question = planned[0].question
        batch = planned
        if budget.max_samples is not None:
            room = budget.max_samples - sent
            if room <= 0:
                stopped = f"budget exhausted: max_samples={budget.max_samples}"
                break
            batch = planned[:room]
        partial = len(batch) < len(planned)
        execution = {"batch": index, "batch_size": len(planned), "partial_batch": False, "attempt": 1}
        try:
            inputs = _batch_input([sample.input for sample in batch])
        except Exception as error:  # inputs that cannot form one batch: no call is made
            records += _records(planned, "failed", f"{type(error).__name__}: {error}", None, execution, common)
            continue
        refusal = _refusal(provider, question, inputs, len(batch))
        if refusal is not None:  # before any call: the whole planned batch, no budget spent
            records += _records(planned, refusal[0], refusal[1], None, execution, common)
            continue
        execution = {**execution, "batch_size": len(batch), "partial_batch": partial}
        calls += 1
        sent += len(batch)
        started = time.perf_counter()
        try:
            results = list(provider.decide(question, inputs))
            if len(results) != len(batch):
                raise DecisionError(f"the provider returned {len(results)} results for {len(batch)} inputs")
        except UnsupportedCapability as error:
            timed = {**execution, "seconds": time.perf_counter() - started}
            records += _records(batch, "unsupported", f"UnsupportedCapability: {error}", None, timed, common)
        except Exception as error:  # one attempt per batch; retries are the provider's
            timed = {**execution, "seconds": time.perf_counter() - started}
            records += _records(batch, "failed", f"{type(error).__name__}: {error}", None, timed, common)
        else:
            timed = {**execution, "seconds": time.perf_counter() - started}
            for sample, result in zip(batch, results, strict=True):  # each answer checked on its own
                try:
                    answer = _answer(sample, result, provider_id)
                except Exception as error:
                    records += _records([sample], "failed", f"{type(error).__name__}: {error}", None, timed, common)
                else:
                    records += _records([sample], "answered", None, [answer], timed, common)
        if partial:  # the sample budget cut this batch short, whatever its outcome
            stopped = f"budget exhausted: max_samples={budget.max_samples}"
            break
    attempted = {record.sample_id for record in records}
    complete = stopped is None and len(attempted) == len({sample.id for sample in samples})
    return Collection(tuple(records), calls, complete, stopped)


def _batch_cap(provider: Any, batch_size: int) -> int:
    """``batch_size``, lowered to the provider's declared ``max_batch``."""
    capabilities = getattr(provider, "capabilities", None)
    if callable(capabilities):
        try:
            limit = getattr(capabilities(), "max_batch", None)
        except Exception:  # each batch's own pre-call check records the error
            return int(batch_size)
        if isinstance(limit, numbers.Integral) and not isinstance(limit, bool) and int(limit) >= 1:
            return min(int(batch_size), int(limit))
    return int(batch_size)


def _refusal(provider: Any, question: Question, inputs: Any, rows: int) -> Optional[tuple[str, str]]:
    """``(status, reason)`` when the provider refuses before any call — its
    ``check(question, inputs)`` when it has one, else its declared
    ``capabilities()``: ``"unsupported"`` for a declared refusal,
    ``"failed"`` when the check itself raises — or ``None``."""
    try:
        check = getattr(provider, "check", None)
        if callable(check):
            check(question, inputs)
            return None
        capabilities = getattr(provider, "capabilities", None)
        if callable(capabilities):
            first = inputs[0] if isinstance(inputs, tuple) and inputs else inputs  # several inputs: the first part
            modality = "text" if isinstance(first, list) and first and isinstance(first[0], str) else "tensor"
            declared: Any = capabilities()
            declared.check(question, modality=modality, batch_size=rows)
    except (UnsupportedCapability, InvalidDecisionRequest) as error:
        return "unsupported", f"{type(error).__name__}: {error}"
    except Exception as error:  # the check itself failed: no call, no budget spent
        return "failed", f"{type(error).__name__}: {error}"
    return None


# --- metrics ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricValue:
    """A metric's ``value`` over ``denominator`` rows, or ``None`` with the
    ``reason`` it is unavailable."""

    name: str
    value: Optional[float]
    denominator: int
    reason: Optional[str] = None

    @property
    def unit(self) -> str:
        return UNITS[self.name]

    def state(self) -> dict[str, Any]:
        return {
            "value": _json_number(self.value),
            "denominator": self.denominator,
            "reason": self.reason,
            "unit": self.unit,
        }


def _json_number(value: Optional[float]) -> Any:
    """A number for strict JSON (and the CSV): a non-finite value as the
    string ``"Infinity"`` / ``"-Infinity"`` / ``"NaN"``."""
    if value is None or math.isfinite(value):
        return value
    return "Infinity" if value > 0 else ("-Infinity" if value < 0 else "NaN")


@dataclass(frozen=True)
class _Row:
    sample: Sample
    labels: tuple[str, ...]
    probabilities: tuple[float, ...]

    @property
    def top(self) -> int:
        best = max(self.probabilities)
        return self.probabilities.index(best)  # the first option of maximal probability

    @property
    def truth(self) -> int:
        return self.labels.index(self.sample.true_label)

    @property
    def correct(self) -> bool:
        return self.top == self.truth


def _accuracy(rows: Sequence[_Row]) -> float:
    return sum(row.correct for row in rows) / len(rows)


def _macro_f1(rows: Sequence[_Row]) -> float:
    true = [row.labels[row.truth] for row in rows]
    predicted = [row.labels[row.top] for row in rows]
    scores = []
    for label in sorted(set(true) | set(predicted)):
        tp = sum(t == label and p == label for t, p in zip(true, predicted, strict=True))
        fp = sum(t != label and p == label for t, p in zip(true, predicted, strict=True))
        fn = sum(t == label and p != label for t, p in zip(true, predicted, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def _nll(rows: Sequence[_Row], epsilon: Optional[float]) -> float:
    total = 0.0
    for row in rows:
        p = row.probabilities[row.truth]
        if epsilon is not None:
            p = max(p, epsilon)
        total += math.inf if p == 0.0 else -math.log(p)
    return total / len(rows)


def _brier(rows: Sequence[_Row]) -> float:
    return sum(sum((p - (i == row.truth)) ** 2 for i, p in enumerate(row.probabilities)) for row in rows) / len(rows)


def _bins(rows: Sequence[_Row], n_bins: int) -> tuple[Any, ...]:
    """The calibration module's own reliability bins over each row's top-1
    confidence and correctness (rows may have different option counts)."""
    import numpy as np

    from ..calibration import _bins as calibration_bins

    confidence = np.array([row.probabilities[row.top] for row in rows], dtype=np.float64).reshape(-1, 1)
    p = np.concatenate([confidence, 1.0 - confidence], axis=1)
    correct = np.array([0 if row.correct else 1 for row in rows], dtype=np.int64)
    return calibration_bins(p, correct, n_bins, predicted=np.zeros(len(rows), dtype=np.int64))


_STATISTICS = {"accuracy": _accuracy, "macro_f1": _macro_f1, "brier": _brier}


@dataclass(frozen=True)
class Coverage:
    """How the samples of a slice joined their records: ``eligible`` (one
    answered record with the sample's digest), ``missing`` (no record),
    ``duplicate`` (several), ``mismatched`` (a record for another question
    digest), ``invalid`` (an answered record that does not fit its question:
    options in another order or set, a Boolean answer for a Choice),
    ``unsupported`` and ``failed`` (with their reasons). Records for no
    sample are the report's ``extra`` count."""

    samples: int
    eligible: int
    missing: int
    duplicate: int
    mismatched: int
    invalid: int
    unsupported: int
    failed: int
    reasons: Mapping[str, int] = field(default_factory=dict)

    def state(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "eligible": self.eligible,
            "missing": self.missing,
            "duplicate": self.duplicate,
            "mismatched": self.mismatched,
            "invalid": self.invalid,
            "unsupported": self.unsupported,
            "failed": self.failed,
            "reasons": dict(sorted(self.reasons.items())),
        }


@dataclass(frozen=True)
class SliceReport:
    """One slice: its coverage, every metric and the reliability bins."""

    name: str
    coverage: Coverage
    metrics: Mapping[str, MetricValue]
    bins: tuple[Any, ...]

    def state(self) -> dict[str, Any]:
        return {
            "coverage": self.coverage.state(),
            "metrics": {name: self.metrics[name].state() for name in METRICS},
            "bins": [b.state() for b in self.bins],
        }


@dataclass(frozen=True)
class Resources:
    """How cost was obtained — every field ``None`` unless declared.

    ``source`` is ``"measured"`` (timed here; ``hardware`` is then required)
    or ``"supplied"`` (reported by someone else); ``timing_boundary`` says
    what the time covers (for example ``"provider.decide calls only"``)."""

    warmup: Optional[int] = None
    hardware: Optional[str] = None
    timing_boundary: Optional[str] = None
    concurrency: Optional[int] = None
    batch_count: Optional[int] = None
    seconds: Optional[float] = None
    source: Optional[str] = None

    def __post_init__(self) -> None:
        if self.source not in (None, "measured", "supplied"):
            raise BenchmarkError(f"Resources.source must be 'measured', 'supplied' or None, got {self.source!r}")
        if self.source == "measured" and not self.hardware:
            raise BenchmarkError("measured resources need the hardware they were measured on")
        if self.seconds is not None:
            if self.source is None:
                raise BenchmarkError("a time needs its source: 'measured' or 'supplied'")
            seconds = _real(self.seconds)
            if seconds is None or not (math.isfinite(seconds) and seconds >= 0):
                raise BenchmarkError(f"Resources.seconds must be a finite number >= 0, got {self.seconds!r}")
            object.__setattr__(self, "seconds", seconds)  # a builtin float: the report stays JSON
        for name in ("warmup", "concurrency", "batch_count"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0):
                raise BenchmarkError(f"Resources.{name} must be a non-negative integer or None, got {value!r}")
            if value is not None:
                object.__setattr__(self, name, int(value))
        for name in ("hardware", "timing_boundary"):
            _text(getattr(self, name), f"Resources.{name}", optional=True)

    def state(self) -> dict[str, Any]:
        return {
            "warmup": self.warmup,
            "hardware": self.hardware,
            "timing_boundary": self.timing_boundary,
            "concurrency": self.concurrency,
            "batch_count": self.batch_count,
            "seconds": self.seconds,
            "source": self.source,
        }


# --- evaluation ------------------------------------------------------------------------------------------


def _join(samples: Sequence[Sample], records: Sequence[Record]) -> tuple[dict[str, tuple[str, Optional[Record]]], int]:
    """Each sample's join status and its record; the count of extra records."""
    by_id: dict[str, list[Record]] = {}
    for record in records:
        if not isinstance(record, Record):
            raise BenchmarkError(f"evaluate() needs Records, got {type(record).__name__}")
        by_id.setdefault(record.sample_id, []).append(record)
    ids = Counter(sample.id for sample in samples)
    repeated = sorted(i for i, n in ids.items() if n > 1)
    if repeated:
        raise BenchmarkError(f"duplicate sample ids {repeated}")
    joined: dict[str, tuple[str, Optional[Record]]] = {}
    for sample in samples:
        found = by_id.get(sample.id, [])
        if not found:
            joined[sample.id] = ("missing", None)
        elif len(found) > 1:
            joined[sample.id] = ("duplicate", None)
        elif found[0].question_digest != sample.digest:
            joined[sample.id] = ("mismatched", found[0])
        elif found[0].status == "answered":
            try:
                _row(sample, found[0])
            except BenchmarkError:
                joined[sample.id] = ("invalid", found[0])
            else:
                joined[sample.id] = ("answered", found[0])
        else:
            joined[sample.id] = (found[0].status, found[0])
    extra = sum(len(found) for sample_id, found in by_id.items() if sample_id not in ids)
    return joined, extra


def _row(sample: Sample, record: Record) -> _Row:
    labels = _labels(sample.question)
    if isinstance(sample.question, Boolean):
        if record.p_true is None:
            raise BenchmarkError(f"record {record.sample_id!r} answers a Boolean without p_true")
        return _Row(sample, labels, (record.p_true, 1.0 - record.p_true))
    if record.distribution is None or tuple(k for k, _ in record.distribution) != labels:
        raise BenchmarkError(f"record {record.sample_id!r}'s distribution does not follow its question's options")
    return _Row(sample, labels, tuple(p for _, p in record.distribution))


def _metrics(
    rows: Sequence[_Row], *, epsilon: Optional[float], n_bins: int, policy: Any, model_id: Optional[str]
) -> tuple[dict[str, MetricValue], tuple[Any, ...]]:
    n = len(rows)
    empty = "no eligible rows"
    metrics: dict[str, MetricValue] = {}
    for name, statistic in _STATISTICS.items():
        metrics[name] = MetricValue(name, statistic(rows), n) if n else MetricValue(name, None, 0, empty)
    metrics["nll"] = MetricValue("nll", _nll(rows, epsilon), n) if n else MetricValue("nll", None, 0, empty)
    bins = _bins(rows, n_bins)
    if n:
        from ..calibration import expected_calibration_error

        metrics["ece"] = MetricValue("ece", expected_calibration_error(bins), n)
    else:
        metrics["ece"] = MetricValue("ece", None, 0, empty)
    metrics.update(_selective(rows, policy, model_id))
    return metrics, bins


def _selective(rows: Sequence[_Row], policy: Any, model_id: Optional[str]) -> dict[str, MetricValue]:
    if policy is None:
        reason = "no abstention policy given"
        return {
            "selective_coverage": MetricValue("selective_coverage", None, len(rows), reason),
            "selective_risk": MetricValue("selective_risk", None, 0, reason),
        }
    from ..abstention import AbstentionSchemaError, decide
    from .schema import ChoiceResult

    eligible = [row for row in rows if not isinstance(row.sample.question, Boolean)]
    accepted = []
    skipped = 0
    for row in eligible:
        result = ChoiceResult(row.sample.digest, tuple(zip(row.labels, row.probabilities, strict=True)))
        try:
            decision = decide(result, policy, model_id=model_id or "")
        except AbstentionSchemaError:
            skipped += 1  # the policy was tuned for another label space or model
            continue
        if decision.outcome.status == "accepted":
            accepted.append(row)
    applied = len(eligible) - skipped
    if applied == 0:
        reason = "the policy fits no eligible row (labels or model_id differ, or only Boolean rows)"
        return {
            "selective_coverage": MetricValue("selective_coverage", None, 0, reason),
            "selective_risk": MetricValue("selective_risk", None, 0, reason),
        }
    coverage = MetricValue("selective_coverage", len(accepted) / applied, applied)
    if not accepted:
        return {
            "selective_coverage": coverage,
            "selective_risk": MetricValue("selective_risk", None, 0, "no accepted rows"),
        }
    risk = sum(not row.correct for row in accepted) / len(accepted)
    return {"selective_coverage": coverage, "selective_risk": MetricValue("selective_risk", risk, len(accepted))}


def _slice_members(samples: Sequence[Sample]) -> dict[str, list[Sample]]:
    slices: dict[str, list[Sample]] = {
        "in_family": [s for s in samples if not s.heldout],
        "heldout": [s for s in samples if s.heldout],
    }
    for sample in samples:
        slices.setdefault(f"family:{sample.family}", []).append(sample)
    for sample in samples:
        slices.setdefault(f"perturbation:{sample.perturbation or 'original'}", []).append(sample)
    return slices


def _slice(
    name: str,
    members: Sequence[Sample],
    joined: Mapping[str, tuple[str, Optional[Record]]],
    **settings: Any,
) -> SliceReport:
    statuses = Counter(joined[sample.id][0] for sample in members)
    reasons: Counter[str] = Counter()
    rows = []
    for sample in members:
        status, record = joined[sample.id]
        if status in ("unsupported", "failed") and record is not None and record.reason:
            reasons[f"{status}: {record.reason}"] += 1
        elif status == "invalid" and record is not None:
            try:
                _row(sample, record)
            except BenchmarkError as error:
                reasons[f"invalid: {error}"] += 1
        if status == "answered" and record is not None:
            rows.append(_row(sample, record))
    coverage = Coverage(
        samples=len(members),
        eligible=len(rows),
        missing=statuses["missing"],
        duplicate=statuses["duplicate"],
        mismatched=statuses["mismatched"],
        invalid=statuses["invalid"],
        unsupported=statuses["unsupported"],
        failed=statuses["failed"],
        reasons=dict(reasons),
    )
    metrics, bins = _metrics(rows, **settings)
    return SliceReport(name, coverage, metrics, bins)


def _for_provider(records: Sequence[Record], provider: Optional[str]) -> tuple[list[Record], list[str]]:
    """The records of ``provider`` (all, when ``None`` — then they must come
    from one provider) and the providers seen."""
    records = list(records)
    bad = [type(record).__name__ for record in records if not isinstance(record, Record)]
    if bad:
        raise BenchmarkError(f"records must be Records, got {bad}")
    if provider is not None:
        records = [record for record in records if record.provider == provider]
    providers = sorted({record.provider for record in records})
    if provider is None and len(providers) > 1:
        raise BenchmarkError(f"records come from several providers {providers}; pass provider=")
    return records, providers


def _check_epsilon(epsilon: Optional[float]) -> None:
    value = _real(epsilon)
    if epsilon is not None and (value is None or not 0 < value < 1):
        raise BenchmarkError(f"epsilon must be in (0, 1) or None, got {epsilon!r}")


def evaluate(
    samples: Sequence[Sample],
    records: Sequence[Record],
    *,
    split: str,
    provider: Optional[str] = None,
    epsilon: Optional[float] = None,
    n_bins: int = 10,
    policy: Any = None,
    model_id: Optional[str] = None,
    resources: Optional[Resources] = None,
) -> BenchmarkReport:
    """Replay: join ``records`` to ``samples`` by sample id and question
    digest and report every slice. Calls no provider and fits nothing.

    Args:
        split: the benchmark split's identity (reports of different splits
            are never compared).
        provider: which provider's records to score (``None``: all records,
            which must then come from one provider).
        epsilon: the NLL floor; ``None`` (default) is exact — ``+inf`` when a
            true label has probability 0.
        n_bins: reliability bins.
        policy / model_id: an ``nnx.abstention.AbstentionPolicy`` applied as
            given (never tuned here) for selective coverage and risk.
        resources: how cost was measured, if at all.
    """
    _text(split, "split")
    samples = list(samples)
    bad = [type(s).__name__ for s in samples if not isinstance(s, Sample)]
    if bad:
        raise BenchmarkError(f"evaluate() needs Samples, got {bad}")
    records, providers = _for_provider(records, provider)
    _check_epsilon(epsilon)
    if isinstance(n_bins, bool) or not isinstance(n_bins, numbers.Integral) or not 1 <= n_bins <= 10_000:
        raise BenchmarkError(f"n_bins must be an integer in [1, 10000], got {n_bins!r}")
    if policy is not None:
        from ..abstention import AbstentionPolicy

        if not isinstance(policy, AbstentionPolicy):
            raise BenchmarkError(f"policy must be an nnx.abstention.AbstentionPolicy, got {type(policy).__name__}")
        _text(model_id, "model_id")
    if resources is not None and not isinstance(resources, Resources):
        raise BenchmarkError(f"resources must be a Resources, got {type(resources).__name__}")
    joined, extra = _join(samples, records)
    settings = {"epsilon": epsilon, "n_bins": int(n_bins), "policy": policy, "model_id": model_id}
    slices = {name: _slice(name, members, joined, **settings) for name, members in _slice_members(samples).items()}
    identity = {
        "format": FORMAT,
        "metrics": list(METRICS),
        "epsilon": None if epsilon is None else float(epsilon),
        "n_bins": int(n_bins),
        "policy": None if policy is None else policy.id,
    }
    return BenchmarkReport(
        split=split,
        provider=provider if provider is not None else (providers[0] if providers else None),
        metric_identity=identity,
        slices=slices,
        resources=resources if resources is not None else Resources(),
        extra=extra,
        sample_set=_sample_set(samples),
    )


def _sample_set(samples: Sequence[Sample]) -> str:
    """Which samples a report scored — their ids, question digests, labels
    and slicing (family, held-out, perturbation, grouping unit) — order-free."""
    rows = sorted(
        json.dumps(
            [
                sample.id,
                sample.digest,
                sample.true_label,
                sample.family,
                sample.heldout,
                sample.perturbation,
                sample.unit,
            ]
        )
        for sample in samples
    )
    return "sha256:" + hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


# --- intervals ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """A percentile bootstrap interval of one metric, resampling the
    declared grouping units (``unit="group"``) with ``seed``. ``degenerate``
    flags a sample that cannot vary: fewer than two units, or every
    resample giving the same value (``low == high``)."""

    metric: str
    estimate: Optional[float]
    low: Optional[float]
    high: Optional[float]
    level: float
    seed: int
    resamples: int
    units: int
    degenerate: bool
    unit: str = "group"
    method: str = "percentile bootstrap, grouping units resampled with replacement"
    reason: Optional[str] = None

    def state(self) -> dict[str, Any]:
        """Strict JSON: a non-finite bound is written as MetricValue writes it."""
        state = {name: getattr(self, name) for name in self.__dataclass_fields__}
        for name in ("estimate", "low", "high"):
            state[name] = _json_number(state[name])
        return state


def bootstrap_interval(
    samples: Sequence[Sample],
    records: Sequence[Record],
    *,
    metric: str,
    seed: int,
    slice: str = "in_family",
    resamples: int = 1000,
    level: float = 0.95,
    epsilon: Optional[float] = None,
    provider: Optional[str] = None,
) -> Interval:
    """Bootstrap ``metric`` (accuracy, macro-F1, NLL or Brier) over the
    eligible rows of ``slice`` (of ``provider``'s records, as
    :func:`evaluate` selects them), resampling grouping units — every row of
    a drawn unit comes along — with ``numpy.random.default_rng(seed)``. The
    bounds are resampled values (no interpolation); a non-finite bound (an
    exact NLL of ``+inf``) flags the interval degenerate."""
    import numpy as np

    if metric not in (*_STATISTICS, "nll"):
        raise BenchmarkError(f"bootstrap_interval() covers accuracy, macro_f1, nll and brier, not {metric!r}")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise BenchmarkError(f"seed must be a non-negative integer, got {seed!r}")
    if isinstance(resamples, bool) or not isinstance(resamples, numbers.Integral) or resamples < 1:
        raise BenchmarkError(f"resamples must be a positive integer, got {resamples!r}")
    if isinstance(level, bool) or not isinstance(level, numbers.Real) or not 0 < level < 1:
        raise BenchmarkError(f"level must be in (0, 1), got {level!r}")
    _check_epsilon(epsilon)
    samples = list(samples)  # iterated twice: a generator is read once
    bad = [type(s).__name__ for s in samples if not isinstance(s, Sample)]
    if bad:
        raise BenchmarkError(f"bootstrap_interval() needs Samples, got {bad}")
    records, _ = _for_provider(records, provider)
    members = _slice_members(samples).get(slice)
    if members is None:
        raise BenchmarkError(f"no slice {slice!r}")
    joined, _ = _join(samples, records)
    rows = [_row(s, joined[s.id][1]) for s in members if joined[s.id][0] == "answered"]  # type: ignore[arg-type]

    def statistic(chosen: Sequence[_Row]) -> float:
        return _nll(chosen, epsilon) if metric == "nll" else _STATISTICS[metric](chosen)

    units: dict[str, list[_Row]] = {}
    for row in rows:
        units.setdefault(row.sample.unit, []).append(row)
    keys = sorted(units)
    base = {"metric": metric, "level": float(level), "seed": int(seed), "resamples": int(resamples), "units": len(keys)}
    if len(keys) < 2:
        estimate = statistic(rows) if rows else None
        return Interval(
            estimate=estimate, low=None, high=None, degenerate=True, reason="fewer than two grouping units", **base
        )
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(resamples)):
        drawn = rng.integers(0, len(keys), size=len(keys))
        values.append(statistic([row for index in drawn for row in units[keys[index]]]))
    tail = (1.0 - float(level)) / 2.0
    drawn_values = np.array(values, dtype=np.float64)
    low = float(np.quantile(drawn_values, tail, method="lower"))
    high = float(np.quantile(drawn_values, 1.0 - tail, method="higher"))
    reason = None
    if not (math.isfinite(low) and math.isfinite(high)):
        reason = "a resample gave a non-finite value"
    elif low == high:
        reason = "every resample gave the same value"
    return Interval(
        estimate=statistic(rows),
        low=low,
        high=high,
        degenerate=reason is not None,
        reason=reason,
        **base,
    )


# --- the report ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkReport:
    """A replayed benchmark: the split and metric identity, every slice's
    coverage, metrics and bins, and the declared :class:`Resources`."""

    split: str
    provider: Optional[str]
    metric_identity: Mapping[str, Any]
    slices: Mapping[str, SliceReport]
    resources: Resources = field(default_factory=Resources)
    extra: int = 0  # records for no sample (benchmark-wide)
    sample_set: Optional[str] = None  # a digest of the samples' ids, questions, labels and slicing

    def state(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "split": self.split,
            "provider": self.provider,
            "metric_identity": dict(self.metric_identity),
            "extra": self.extra,
            "sample_set": self.sample_set,
            "resources": self.resources.state(),
            "slices": {name: report.state() for name, report in self.slices.items()},
        }

    def to_json(self) -> str:
        return json.dumps(self.state(), sort_keys=True, indent=2, allow_nan=False) + "\n"

    def csv_rows(self) -> list[dict[str, Any]]:
        """One row per slice and metric: value, unit, denominator, the
        unavailable reason, every coverage count of the slice (samples,
        eligible, failed, unsupported, missing, duplicate, mismatched,
        invalid — they add up to the slice's samples) and the report's
        ``extra`` records. A non-finite value is written as in the JSON."""
        rows = []
        for name, report in self.slices.items():
            coverage = report.coverage
            for metric in METRICS:
                value = report.metrics[metric]
                rows.append(
                    {
                        "slice": name,
                        "metric": metric,
                        "value": "" if value.value is None else _csv_number(value.value),
                        "unit": value.unit,
                        "denominator": value.denominator,
                        "unavailable": value.reason or "",
                        "samples": coverage.samples,
                        "eligible": coverage.eligible,
                        "failed": coverage.failed,
                        "unsupported": coverage.unsupported,
                        "missing": coverage.missing,
                        "duplicate": coverage.duplicate,
                        "mismatched": coverage.mismatched,
                        "invalid": coverage.invalid,
                        "extra": self.extra,
                    }
                )
        return rows

    def to_csv(self) -> str:
        buffer = io.StringIO()
        fields = ["slice", "metric", "value", "unit", "denominator", "unavailable", "samples", "eligible", "failed"]
        fields += ["unsupported", "missing", "duplicate", "mismatched", "invalid", "extra"]
        writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(self.csv_rows())
        return buffer.getvalue()

    def text(self) -> str:
        lines = [f"{FORMAT} split={self.split} provider={self.provider or 'unknown'} extra_records={self.extra}"]
        for name, report in self.slices.items():
            c = report.coverage
            lines.append(
                f"[{name}] samples={c.samples} eligible={c.eligible} failed={c.failed} unsupported={c.unsupported}"
                f" missing={c.missing} duplicate={c.duplicate} mismatched={c.mismatched} invalid={c.invalid}"
            )
            for reason, count in sorted(c.reasons.items()):
                lines.append(f"    {count} x {reason}")
            for metric in METRICS:
                value = report.metrics[metric]
                if value.value is None:
                    lines.append(f"    {metric}: unavailable ({value.reason}; n={value.denominator}) [{value.unit}]")
                else:
                    lines.append(f"    {metric}: {value.value!r} (n={value.denominator}) [{value.unit}]")
        resources = {k: v for k, v in self.resources.state().items() if v is not None}
        lines.append(f"resources: {resources or 'not measured'}")
        return "\n".join(lines) + "\n"

    def save(self, path: Union[str, os.PathLike[str]]) -> None:
        """Write the JSON report atomically (serialized first: a report that
        cannot be written never replaces an existing file)."""
        from .._artifacts import atomic_write

        atomic_write(path, self.to_json())

    @staticmethod
    def load_state(path: Union[str, os.PathLike[str]]) -> dict[str, Any]:
        """A saved report's JSON state (for :func:`compare_reports`)."""

        from .._artifacts import parse_json, read_text

        try:
            state = parse_json(read_text(path, "report", BenchmarkError), "report", BenchmarkError)
        except BenchmarkError as error:
            raise BenchmarkError(f"{os.fspath(path)}: {error}") from error
        if not isinstance(state, Mapping) or state.get("format") != FORMAT:
            raise BenchmarkError(f"{os.fspath(path)} is not a {FORMAT} report")
        return dict(state)


def _csv_number(value: float) -> str:
    encoded = _json_number(value)
    return encoded if isinstance(encoded, str) else repr(float(value))


def _report_state(report: Any) -> dict[str, Any]:
    state = report.state() if isinstance(report, BenchmarkReport) else report
    if not isinstance(state, Mapping) or state.get("format") != FORMAT:
        raise BenchmarkError(f"not a {FORMAT} report")
    missing = sorted({"split", "metric_identity", "slices"} - set(state))
    if missing or not isinstance(state["slices"], Mapping):
        raise BenchmarkError(f"a {FORMAT} report needs split, metric_identity and slices; missing {missing}")
    return dict(state)


def compare_reports(
    a: Union[BenchmarkReport, Mapping[str, Any]], b: Union[BenchmarkReport, Mapping[str, Any]]
) -> dict[str, dict[str, Optional[float]]]:
    """``b - a`` per slice and metric, for two reports of the same split and
    metric identity and — when both record it — the same sample set (a
    :class:`BenchmarkReport` or a saved report's state); anything else
    raises :class:`BenchmarkError`. A metric unavailable on
    either side has no delta."""
    left, right = _report_state(a), _report_state(b)
    if left["split"] != right["split"]:
        raise BenchmarkError(f"the reports cover different splits: {left['split']!r} and {right['split']!r}")
    if left["metric_identity"] != right["metric_identity"]:
        raise BenchmarkError(
            f"the reports use different metric identities: {left['metric_identity']} and {right['metric_identity']}"
        )
    sets = (left.get("sample_set"), right.get("sample_set"))
    if None not in sets and sets[0] != sets[1]:
        raise BenchmarkError(
            f"the reports scored different samples of split {left['split']!r}: {sets[0]} and {sets[1]}"
        )
    deltas: dict[str, dict[str, Optional[float]]] = {}
    for name in sorted(set(left["slices"]) & set(right["slices"])):
        row = {}
        for metric in METRICS:
            try:
                x = left["slices"][name]["metrics"][metric]["value"]
                y = right["slices"][name]["metrics"][metric]["value"]
            except (KeyError, TypeError) as error:
                raise BenchmarkError(f"slice {name!r} has no {metric!r} value in one of the reports") from error
            finite = all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, y))
            row[metric] = float(y) - float(x) if finite else None
        deltas[name] = row
    return deltas
