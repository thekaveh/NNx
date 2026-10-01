"""A local label-conditioned baseline: decisions from a caller-supplied
natural-language-inference (NLI) model (FEAT-011).

A fixed-head classifier knows only the labels it was trained on. An NLI
cross-encoder scores whether a *premise* (the input text) entails a
*hypothesis* (a sentence built from a candidate's description), so the
candidates can be supplied at inference time — none of them needs to exist
when the provider is built::

    from nnx.decisions import Choice, NLIProvider

    provider = NLIProvider(model, tokenizer, entailment_id=2, contradiction_id=0,
                           hypothesis_template="This text is about {}.", revision="abc123")
    provider.decide(Choice("Topic?", (("t-sport", "sports"), ("t-econ", "the economy"))), texts)

**Scoring (explicit, never calibrated).**

- A :class:`Choice` scores every (input, candidate) pair and normalizes the
  **entailment logits across the candidates** with a softmax
  (``choice_scoring="entailment_softmax"``): logits ``log(3)`` and ``0``
  give ``(0.75, 0.25)``. Results are keyed by the request's option ids, in
  its order, whatever order the candidates arrive in.
- A :class:`Boolean` scores one pair per input — the prompt rendered by
  ``boolean_template`` — and takes the softmax over that pair's
  **contradiction and entailment** logits alone
  (``boolean_scoring="entailment_vs_contradiction"``): contradiction ``0``,
  entailment ``log(4)`` give ``p_true = 0.8``. Inputs are never normalized
  together.
- :class:`Score` is unsupported (an NLI model has no ordinal semantics) and
  is rejected before any model call.

These are zero-shot scores from a model trained for another task: they are
**not calibrated** probabilities, and how well they transfer to a decision
task stays an empirical question — measure it on labelled records (see
``examples/decision_nli.py``).

**The provider contract.** NNx imports no NLI library and downloads
nothing: the caller passes a tokenizer and a ``torch.nn.Module`` already
loaded (from a local path, a pinned ``revision`` or a test stub). The
tokenizer is called HuggingFace-style — ``tokenizer(premises, hypotheses,
truncation=..., max_length=..., padding=True, return_tensors="pt")``
returning a mapping of tensors — and the model as ``model(**encoded)``,
returning the logits ``(pairs, classes)`` or an object with ``.logits``.
Pairs are sent in chunks of ``pair_batch_size`` (the last chunk may be
shorter), moved to the model's device; the model runs in eval mode under
``no_grad`` and every submodule's training flag is restored afterwards, on
success and on failure.

**Truncation (explicit, reported).** Every pair is measured untruncated
first. ``truncation="only_first"`` cuts the premise to fit ``max_length``
and reports which pairs were cut in each result's ``raw["truncated"]``; a
hypothesis that does not fit even with the premise cut rejects the request
before any model call. ``truncation="error"`` rejects a request with an
over-long pair before any model call.

Each result's ``raw`` records the pair logits, the truncation report and
the provider's :meth:`NLIProvider.record` — the NLI template, label ids,
scoring methods, truncation policy and model ``revision`` — with
``"calibrated": False``, so a replayed or benchmarked result keeps how it
was scored.
"""

from __future__ import annotations

import math
import numbers
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, Union, cast

import numpy as np

from .providers import Capabilities, Question
from .schema import (
    Boolean,
    Choice,
    DecisionResult,
    InvalidDecisionRequest,
    ProviderFailure,
    Score,
    UnsupportedCapability,
    validate_response,
)

__all__ = ["NLIProvider"]

CHOICE_SCORING = ("entailment_softmax",)
BOOLEAN_SCORING = ("entailment_vs_contradiction",)
TRUNCATION = ("only_first", "error")


def _check_template(template: Any, what: str) -> str:
    """A template with exactly one bare ``{}`` placeholder (the description
    or the prompt): no other field, conversion or format spec, which could
    fail at request time or silently change the text."""
    if not isinstance(template, str) or not template.strip():
        raise InvalidDecisionRequest(f"{what} must be a non-empty string, got {template!r}")
    try:
        fields = [(name, spec, conversion) for _, name, spec, conversion in string.Formatter().parse(template)]
    except ValueError as error:
        raise InvalidDecisionRequest(f"{what} {template!r} is not a valid format string: {error}") from error
    if [field for field in fields if field[0] is not None] != [("", "", None)]:
        raise InvalidDecisionRequest(
            f"{what} must hold exactly one bare '{{}}' placeholder (no conversion or format spec), got {template!r}"
        )
    return template


def _label_space(model: Any) -> tuple[Optional[int], Mapping[str, int]]:
    """The model's class count and label names, when its config declares
    them (HuggingFace ``config.num_labels`` / ``label2id``)."""
    config = getattr(model, "config", None)
    label2id = getattr(config, "label2id", None)
    names = {str(k).lower(): int(v) for k, v in label2id.items()} if isinstance(label2id, Mapping) else {}
    n_labels = getattr(config, "num_labels", None)
    if isinstance(n_labels, numbers.Integral) and not isinstance(n_labels, bool):
        return int(n_labels), names
    return (len(names) or None), names


def _class_id(value: Any, what: str, n_labels: Optional[int], names: Mapping[str, int]) -> int:
    if isinstance(value, str):
        key = value.lower()
        if key not in names:
            raise InvalidDecisionRequest(
                f"unknown {what} label {value!r}: the model's config names {sorted(names) or 'no labels'}; pass the "
                "class id as an int"
            )
        return names[key]
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise InvalidDecisionRequest(f"{what} must be a class id (int) or a label name (str), got {value!r}")
    class_id = int(value)
    if class_id < 0 or (n_labels is not None and class_id >= n_labels):
        raise InvalidDecisionRequest(f"{what} {class_id} is out of range for a model with {n_labels} classes")
    return class_id


def _positive(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise InvalidDecisionRequest(f"{what} must be a positive integer, got {value!r}")
    return int(value)


def _texts(inputs: Any) -> list[str]:
    """The batch of input texts (the premises)."""
    if isinstance(inputs, str) or not isinstance(inputs, Sequence):
        raise UnsupportedCapability(
            f"an NLI provider reads a sequence of texts (a list or tuple of str), got {type(inputs).__name__}"
        )
    texts = list(inputs)
    bad = [i for i, text in enumerate(texts) if not isinstance(text, str) or not text.strip()]
    if bad:
        raise InvalidDecisionRequest(f"inputs {bad} are not non-empty strings")
    return texts


@dataclass
class NLIProvider:
    """A caller-supplied NLI model as a label-conditioned decision provider —
    see the module docstring for the scoring, the provider contract and the
    truncation policy.

    Args:
        model: the NLI cross-encoder (a ``torch.nn.Module``; ``model(**encoded)``
            returns logits ``(pairs, classes)`` or an object with ``.logits``).
            Its device is its parameters' device.
        tokenizer: the matching tokenizer, called HuggingFace-style.
        entailment_id / contradiction_id: the model's class ids for
            entailment and contradiction — ints, or label names resolved
            through ``model.config.label2id``. Distinct, and within the
            model's class count when its config declares one.
        hypothesis_template: renders a Choice candidate's description
            (exactly one ``{}``), e.g. ``"This example is {}."``.
        boolean_template: renders a Boolean's prompt (exactly one ``{}``).
        choice_scoring / boolean_scoring: the normalization methods (see the
            module docstring); explicit so a record says how it scored.
        max_length: the most tokens per pair.
        truncation: ``"only_first"`` (cut the premise, report it) or
            ``"error"`` (reject over-long pairs before any model call).
        pair_batch_size: pairs per model call.
        max_batch: the most inputs per request (``None``: unbounded).
        revision: the model revision the caller loaded (recorded, never
            fetched).
        name: the provider name recorded on results.
    """

    model: Any
    tokenizer: Any
    entailment_id: Union[int, str]
    contradiction_id: Union[int, str]
    hypothesis_template: str = "This example is {}."
    boolean_template: str = "{}"
    choice_scoring: str = "entailment_softmax"
    boolean_scoring: str = "entailment_vs_contradiction"
    max_length: int = 512
    truncation: str = "only_first"
    pair_batch_size: int = 16
    max_batch: Optional[int] = None
    revision: Optional[str] = None
    name: str = "nnx.nli"
    model_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        import torch

        if not isinstance(self.model, torch.nn.Module):
            raise InvalidDecisionRequest(f"model must be a torch.nn.Module, got {type(self.model).__name__}")
        if not callable(self.tokenizer):
            raise InvalidDecisionRequest(f"tokenizer must be callable, got {type(self.tokenizer).__name__}")
        n_labels, names = _label_space(self.model)
        self.entailment_id = _class_id(self.entailment_id, "entailment_id", n_labels, names)
        self.contradiction_id = _class_id(self.contradiction_id, "contradiction_id", n_labels, names)
        if self.entailment_id == self.contradiction_id:
            raise InvalidDecisionRequest(
                f"entailment_id and contradiction_id must differ, both are {self.entailment_id}"
            )
        self._n_labels = n_labels
        _check_template(self.hypothesis_template, "hypothesis_template")
        _check_template(self.boolean_template, "boolean_template")
        if self.choice_scoring not in CHOICE_SCORING:
            raise InvalidDecisionRequest(f"choice_scoring must be one of {CHOICE_SCORING}, got {self.choice_scoring!r}")
        if self.boolean_scoring not in BOOLEAN_SCORING:
            raise InvalidDecisionRequest(
                f"boolean_scoring must be one of {BOOLEAN_SCORING}, got {self.boolean_scoring!r}"
            )
        if self.truncation not in TRUNCATION:
            raise InvalidDecisionRequest(f"truncation must be one of {TRUNCATION}, got {self.truncation!r}")
        self.max_length = _positive(self.max_length, "max_length")
        self.pair_batch_size = _positive(self.pair_batch_size, "pair_batch_size")
        if self.max_batch is not None:
            self.max_batch = _positive(self.max_batch, "max_batch")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise InvalidDecisionRequest(f"revision must be a non-empty string or None, got {self.revision!r}")

    # ---------- what it declares ----------

    def capabilities(self) -> Capabilities:
        return Capabilities(
            primitives=frozenset({"choice", "boolean"}),
            modalities=frozenset({"text"}),
            dynamic_labels=True,  # candidates arrive with each request
            labels=None,
            max_batch=self.max_batch,
            inference=True,
            training=False,  # the caller's model is used as is
            export=False,
        )

    def record(self) -> dict[str, Any]:
        """How this provider scores: what a replay record or a benchmark
        keeps beside each result. The scores are never calibrated."""
        return {
            "provider": self.name,
            "method": "nli",
            "hypothesis_template": self.hypothesis_template,
            "boolean_template": self.boolean_template,
            "entailment_id": self.entailment_id,
            "contradiction_id": self.contradiction_id,
            "choice_scoring": self.choice_scoring,
            "boolean_scoring": self.boolean_scoring,
            "max_length": self.max_length,
            "truncation": self.truncation,
            "revision": self.revision,
            "calibrated": False,
        }

    # ---------- answering ----------

    def hypotheses(self, question: Question) -> list[str]:
        """The hypothesis per candidate (a Choice, in its option order) or
        the one hypothesis of a Boolean."""
        if isinstance(question, Choice):
            return [self.hypothesis_template.format(option.description) for option in question.options]
        if isinstance(question, Boolean):
            return [self.boolean_template.format(question.prompt)]
        raise UnsupportedCapability(f"an NLI provider answers choice and boolean questions, not {question.kind!r}")

    def decide(self, question: Question, inputs: Any) -> list[DecisionResult]:
        """Answer ``question`` for every text in ``inputs``."""
        if not isinstance(question, (Choice, Boolean, Score)):
            raise InvalidDecisionRequest(f"not a decision question: {type(question).__name__}")
        texts = _texts(inputs)
        self.capabilities().check(question, modality="text", batch_size=len(texts))
        hypotheses = self.hypotheses(question)
        if not texts:
            return []
        premises = [text for text in texts for _ in hypotheses]
        pair_hypotheses = hypotheses * len(texts)
        truncated = self._truncation_report(premises, pair_hypotheses)
        logits = self._pair_logits(premises, pair_hypotheses)
        per_input = len(hypotheses)
        record = self.record()
        entail, contra = cast(int, self.entailment_id), cast(int, self.contradiction_id)  # resolved at construction
        results: list[DecisionResult] = []
        for row in range(len(texts)):
            block = logits[row * per_input : (row + 1) * per_input]
            cut = truncated[row * per_input : (row + 1) * per_input]
            raw = {"logits": block.tolist(), "truncated": cut, **record}
            if isinstance(question, Choice):
                answer: Any = dict(zip(question.option_ids, _softmax(block[:, entail]), strict=True))
            else:
                pair = block[0, [contra, entail]]
                answer = float(_softmax(pair)[1])
            results.append(validate_response(question, answer, provider=self.name, raw=raw))
        return results

    def _truncation_report(self, premises: list[str], hypotheses: list[str]) -> list[bool]:
        """Which pairs exceed ``max_length`` untruncated, checked before the
        model is called: under ``truncation="error"`` any such pair rejects
        the request; under ``"only_first"`` each must fit once its premise
        is cut — a hypothesis that alone overflows ``max_length`` rejects it."""
        try:
            encoded = self.tokenizer(premises, hypotheses, truncation=False, padding=False)
            lengths = [len(ids) for ids in encoded["input_ids"]]
        except Exception as error:
            raise ProviderFailure(f"{self.name}: the tokenizer failed: {error}") from error
        truncated = [length > self.max_length for length in lengths]
        if not any(truncated):
            return truncated
        if self.truncation == "error":
            raise InvalidDecisionRequest(
                f"{sum(truncated)} premise/hypothesis pair(s) exceed max_length={self.max_length} tokens (the longest "
                f"has {max(lengths)}) and truncation='error'; shorten the inputs or use truncation='only_first'"
            )
        long = [i for i, cut in enumerate(truncated) if cut]
        overflow = (
            f"the hypothesis plus special tokens does not fit max_length={self.max_length} even with the premise "
            "cut; raise max_length or shorten the description or prompt"
        )
        try:
            cut = self.tokenizer(
                [premises[i] for i in long],
                [hypotheses[i] for i in long],
                truncation="only_first",
                max_length=self.max_length,
                padding=False,
            )
            cut_lengths = [len(ids) for ids in cut["input_ids"]]
        except Exception as error:  # a tokenizer that refuses to cut a too-short premise
            raise InvalidDecisionRequest(f"{overflow} ({error})") from error
        if any(length > self.max_length for length in cut_lengths):
            raise InvalidDecisionRequest(overflow)
        return truncated

    def _pair_logits(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        """Every pair's logits ``(pairs, classes)`` in float64, in chunks of
        ``pair_batch_size``; eval mode under no-grad, training flags restored."""
        import torch

        from ..utils import _capture_training_modes, _restore_training_modes

        device = next((p.device for p in self.model.parameters()), torch.device("cpu"))
        modes = _capture_training_modes(self.model)
        self.model_calls += 1
        chunks: list[np.ndarray] = []
        try:
            self.model.eval()
            with torch.no_grad():
                for start in range(0, len(premises), self.pair_batch_size):
                    stop = start + self.pair_batch_size
                    encoded = self.tokenizer(
                        premises[start:stop],
                        hypotheses[start:stop],
                        truncation="only_first",
                        max_length=self.max_length,
                        padding=True,
                        return_tensors="pt",
                    )
                    encoded = {key: value.to(device) for key, value in dict(encoded).items()}
                    output = self.model(**encoded)
                    logits = getattr(output, "logits", output)
                    chunks.append(np.asarray(logits.detach().to("cpu", torch.float64)))
        except Exception as error:  # the backend failed on a valid, supported request
            raise ProviderFailure(f"{self.name}: the NLI model failed: {error}") from error
        finally:
            _restore_training_modes(modes)
        try:
            logits = np.concatenate(chunks, axis=0)
        except ValueError as error:  # 0-d output, or a class count that differs between chunks
            raise ProviderFailure(
                f"{self.name}: the NLI model returned logits of inconsistent shapes {[c.shape for c in chunks]}"
            ) from error
        width = max(cast(int, self.entailment_id), cast(int, self.contradiction_id)) + 1
        if logits.ndim != 2 or logits.shape[0] != len(premises) or logits.shape[1] < width:
            raise ProviderFailure(
                f"{self.name}: the NLI model returned logits of shape {tuple(logits.shape)} for {len(premises)} pairs; "
                f"expected (pairs, classes) with at least {width} classes"
            )
        if not np.isfinite(logits).all():
            raise ProviderFailure(f"{self.name}: the NLI model returned non-finite logits")
        return logits


def _softmax(values: np.ndarray) -> list[float]:
    """A float64 softmax (so a distribution meets the 1e-6 tolerance)."""
    shifted = np.asarray(values, dtype=np.float64) - float(np.max(values))
    weights = np.exp(shifted)
    total = math.fsum(weights.tolist())
    return [float(w / total) for w in weights]
