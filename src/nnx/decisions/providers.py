"""Decision providers: declared capabilities, the provider protocol and the
fixed-head NNx adapter (FEAT-009)."""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol, Union, runtime_checkable

import numpy as np

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

if TYPE_CHECKING:
    from ..nn.nn_model import NNModel

__all__ = ["Capabilities", "DecisionProvider", "FixedHeadProvider"]

Question = Union[Choice, Boolean, Score]


@dataclass(frozen=True)
class Capabilities:
    """What a provider declares it can do. :meth:`check` rejects anything
    else with :class:`UnsupportedCapability` before any model call or
    network I/O.

    Attributes:
        primitives: the decision kinds it answers (``"choice"``,
            ``"boolean"``, ``"score"``).
        modalities: the input kinds it reads (e.g. ``"tensor"``,
            ``"text"``).
        dynamic_labels: whether a request may bring any option set; ``False``
            means the options must map exactly onto ``labels``.
        labels: the fixed label space when ``dynamic_labels`` is ``False``.
        max_batch: the most inputs per call (``None``: unbounded).
        inference / training / export: whether the backend can answer, be
            trained and be exported. A hosted provider declares inference
            only.
    """

    primitives: frozenset[str]
    modalities: frozenset[str]
    dynamic_labels: bool = False
    labels: Optional[tuple[str, ...]] = None
    max_batch: Optional[int] = None
    inference: bool = True
    training: bool = False
    export: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "primitives", frozenset(self.primitives))
        object.__setattr__(self, "modalities", frozenset(self.modalities))
        if self.labels is not None:
            object.__setattr__(self, "labels", tuple(self.labels))
        max_batch = self.max_batch
        if max_batch is not None and (
            isinstance(max_batch, bool) or not isinstance(max_batch, numbers.Integral) or max_batch < 1
        ):
            raise ValueError(f"max_batch must be a positive integer or None, got {max_batch!r}")

    def check(
        self,
        question: Question,
        *,
        modality: str,
        batch_size: int,
        label_ids: Optional[Sequence[str]] = None,
    ) -> None:
        """Raise :class:`UnsupportedCapability` unless this provider serves
        ``question`` for a batch of ``batch_size`` ``modality`` inputs.

        Without dynamic labels, a Choice's or Score's ids — or ``label_ids``,
        the provider's own mapping of them — must be exactly the declared
        ``labels``, in any order."""
        if not self.inference:
            raise UnsupportedCapability("this provider declares no inference capability")
        if question.kind not in self.primitives:
            raise UnsupportedCapability(
                f"this provider answers {sorted(self.primitives)} questions, not {question.kind!r}"
            )
        if modality not in self.modalities:
            raise UnsupportedCapability(f"this provider reads {sorted(self.modalities)} inputs, not {modality!r}")
        if self.max_batch is not None and batch_size > self.max_batch:
            raise UnsupportedCapability(f"batch of {batch_size} exceeds this provider's max_batch={self.max_batch}")
        if self.dynamic_labels or self.labels is None or not isinstance(question, (Choice, Score)):
            return
        ids = list(question.option_ids if label_ids is None else label_ids)
        unseen = [label for label in ids if label not in self.labels]
        if unseen:
            raise UnsupportedCapability(f"unseen labels {unseen}: this provider only knows {list(self.labels)}")
        missing = [label for label in self.labels if label not in ids]
        if missing:
            raise UnsupportedCapability(
                f"the request must cover the exact label space {list(self.labels)}; missing {missing}"
            )


@runtime_checkable
class DecisionProvider(Protocol):
    """Anything that answers typed decision questions: it declares
    :class:`Capabilities` and answers one question for a batch of inputs
    with validated results, one per input."""

    def capabilities(self) -> Capabilities: ...

    def decide(self, question: Question, inputs: Any) -> list[DecisionResult]: ...


def _batch(inputs: Any) -> tuple[str, int]:
    """Modality and batch size of a local model's inputs."""
    first = inputs[0] if isinstance(inputs, tuple) and inputs else inputs
    import torch

    if isinstance(first, (torch.Tensor, np.ndarray)) and getattr(first, "ndim", 0) >= 1:
        return "tensor", int(first.shape[0])
    raise UnsupportedCapability(
        f"a fixed-head provider reads tensor inputs (a tensor, an array or a tuple of them), got {type(inputs).__name__}"
    )


def _head_width(model: Any, spec: Any) -> Optional[int]:
    """The head's output width when it is known without running the model
    (a task's count, else a built-in net's ``output_dim``)."""
    width = getattr(spec, "num_outputs", None)
    if width is None:
        width = getattr(getattr(model, "net_params", None), "output_dim", None)
    return None if width is None else int(width)


@dataclass
class FixedHeadProvider:
    """A trained NNx classifier as a decision provider.

    The head fixes the label space and the primitives it justifies: a
    categorical head (softmax over its classes) answers :class:`Choice` —
    and :class:`Score` when constructed with ``ordinal=True``, whose levels
    must then follow the head's class order — and a single-logit Bernoulli
    head (``BCEWithLogitsLoss``, or a one-output multilabel task) answers
    :class:`Boolean`. Probabilities come from the model's ``predict_proba``
    (FEAT-001), which evaluates in eval mode and restores every submodule's
    training mode; they are computed in float64 from the returned logits,
    so half-precision models meet the ``1e-6`` tolerance.

    A request's option ids must be exactly the head's ``labels`` (in any
    order), or ``option_map`` must map them one-to-one onto the labels; an
    unseen or missing label raises :class:`UnsupportedCapability` before
    :attr:`model_calls` advances. A label count that does not fit the head
    is rejected at construction when the head's width is known (a task's
    count or a built-in net's ``output_dim``), otherwise with
    :class:`InvalidDecisionRequest` on the first call. Results come back in
    the request's option order with each row's logits (a copy) as ``raw``;
    an empty batch returns ``[]`` without calling the model.

    Args:
        model: the trained :class:`~nnx.NNModel`.
        labels: one name per output class, in column order; defaults to the
            model's ``TaskSpec`` labels, and must equal them when both exist.
        option_map: request option id → head label (a bijection).
        ordinal: whether the class order is meaningful (enables Score).
        max_batch: the most inputs per call.
        name: the provider name recorded on results.
    """

    model: NNModel
    labels: Optional[Sequence[str]] = None
    option_map: Optional[Mapping[str, str]] = None
    ordinal: bool = False
    max_batch: Optional[int] = None
    name: str = "nnx.fixed_head"
    model_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        import torch

        adapter = getattr(self.model, "task_adapter", None)
        spec = getattr(adapter, "spec", None)
        kind = getattr(spec, "kind", None)
        if kind == "regression":
            raise UnsupportedCapability("a regression head justifies no decision primitive")
        task_labels = getattr(spec, "labels", None)
        if self.labels is not None and task_labels is not None and tuple(self.labels) != tuple(task_labels):
            raise InvalidDecisionRequest(
                f"labels {tuple(self.labels)} disagree with the model's task labels {tuple(task_labels)}"
            )
        labels = self.labels if self.labels is not None else task_labels
        width = _head_width(self.model, spec)
        bernoulli = kind == "multilabel" or isinstance(self.model.loss_fn, torch.nn.BCEWithLogitsLoss)
        if bernoulli:
            if width not in (None, 1) or (labels is not None and len(tuple(labels)) != 1):
                raise UnsupportedCapability(
                    "a multi-output Bernoulli head justifies no single decision; use a one-output head for Boolean"
                )
            self._head = "bernoulli"
            self.labels = tuple(labels) if labels is not None else None
            return
        self._head = "categorical"
        if labels is None:
            raise InvalidDecisionRequest(
                "a categorical head needs its label space: pass labels=(...) (one per output class) or declare "
                "TaskSpec.categorical(..., labels=...)"
            )
        labels = tuple(labels)
        if len(labels) < 2 or len(set(labels)) != len(labels) or not all(isinstance(x, str) and x for x in labels):
            raise InvalidDecisionRequest(f"labels must be 2+ unique non-empty strings, got {labels!r}")
        if width is not None and width != len(labels):
            raise InvalidDecisionRequest(f"{len(labels)} labels for a head with {width} output classes")
        self.labels = labels
        if self.option_map is not None:
            mapped = list(self.option_map.values())
            if len(set(mapped)) != len(mapped) or set(mapped) != set(labels):
                raise InvalidDecisionRequest(
                    f"option_map must be a bijection onto the head's labels {list(labels)}, got {dict(self.option_map)}"
                )

    def capabilities(self) -> Capabilities:
        if self._head == "bernoulli":
            primitives = frozenset({"boolean"})
        else:
            primitives = frozenset({"choice", "score"} if self.ordinal else {"choice"})
        return Capabilities(
            primitives=primitives,
            modalities=frozenset({"tensor"}),
            dynamic_labels=False,
            labels=None if self.labels is None else tuple(self.labels),
            max_batch=self.max_batch,
            inference=True,
            training=True,  # the local NNModel trains (NNModel.train) ...
            export=True,  # ... and exports (to_onnx, save_pretrained)
        )

    def _head_labels(self, question: Question) -> Optional[list[str]]:
        """Each option's head label, in the request's order (``None`` for a
        Boolean); unmapped ids raise before any model call."""
        if not isinstance(question, (Choice, Score)):
            return None
        if self.option_map is None:
            return list(question.option_ids)
        unseen = [option_id for option_id in question.option_ids if option_id not in self.option_map]
        if unseen:
            raise UnsupportedCapability(f"unseen option ids {unseen}: option_map maps {sorted(self.option_map)}")
        return [self.option_map[option_id] for option_id in question.option_ids]

    def decide(self, question: Question, inputs: Any) -> list[DecisionResult]:
        """Answer ``question`` for every row of ``inputs`` (a tensor, an
        array or a tuple of them)."""
        if not isinstance(question, (Choice, Boolean, Score)):
            raise InvalidDecisionRequest(f"not a decision question: {type(question).__name__}")
        modality, batch_size = _batch(inputs)
        head_labels = self._head_labels(question)
        caps = self.capabilities()
        caps.check(question, modality=modality, batch_size=batch_size, label_ids=head_labels)
        if isinstance(question, Score) and head_labels != list(caps.labels or ()):
            raise UnsupportedCapability(
                f"a Score's levels must follow the head's class order {list(caps.labels or ())}; got {head_labels}"
            )
        if batch_size == 0:
            return []

        from ..prediction import PredictionValidationError, ProbabilitySpec, prediction_from_logits

        kind = "bernoulli" if self._head == "bernoulli" else "categorical"
        spec = ProbabilitySpec(kind=kind, class_axis=1, labels=caps.labels)
        self.model_calls += 1
        try:
            prediction = self.model.predict_proba(inputs, spec)
        except PredictionValidationError as error:
            raise InvalidDecisionRequest(f"{self.name}: the head's output does not fit its labels: {error}") from error
        except Exception as error:  # the backend failed on a valid, supported request
            raise ProviderFailure(f"{self.name}: the model failed to predict: {error}") from error
        logits = np.asarray(prediction.logits)
        if kind == "bernoulli" and (logits.ndim != 2 or logits.shape[1] != 1):
            raise UnsupportedCapability(
                f"a Boolean needs a one-output head; this head produced shape {tuple(logits.shape)}"
            )
        # Recomputed in float64 so half-precision logits still sum to 1 within 1e-6.
        probabilities = np.asarray(prediction_from_logits(logits.astype(np.float64), spec).probabilities)
        columns = None if head_labels is None else [list(caps.labels or ()).index(label) for label in head_labels]
        option_ids = question.option_ids
        results: list[DecisionResult] = []
        for row, raw in zip(probabilities, logits, strict=True):
            if columns is None:
                answer: Any = float(row[0])
            else:
                answer = {option_id: float(row[column]) for option_id, column in zip(option_ids, columns, strict=True)}
            results.append(validate_response(question, answer, provider=self.name, raw=raw.copy()))
        return results
