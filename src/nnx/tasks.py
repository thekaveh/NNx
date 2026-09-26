"""Task adapters: declare what a supervised model's outputs and targets mean.

By default ``NNModel`` assumes categorical classification: the default
training step, :meth:`NNModel.evaluate` and :meth:`NNModel.predict` decode
by argmax (or the ``BCEWithLogitsLoss`` threshold) and every evaluation
record carries accuracy / f1 / recall / precision. A :class:`TaskSpec` on
``NNModelParams(task=...)`` opts a model into an explicit task instead::

    params = NNModelParams(net=Nets.FEED_FWD, loss=Losses.MEAN_SQUARED_ERROR, task=TaskSpec.regression(2))

The task's adapter then owns, for the default step, ``evaluate()``,
``predict()`` and ``predict_proba()``:

* **validation** — each batch's outputs and targets are checked for the
  declared shape, dtype and value range *before* any backward pass or
  optimizer update (:class:`TaskValidationError`);
* **masking** — categorical targets equal to ``ignore_index``, and NaN
  multilabel / regression targets, are excluded from the loss and every
  metric alike;
* **loss units** — the loss is averaged over the valid targets, with the
  same denominators the default step already uses for gradient
  accumulation, so uneven batches and accumulated windows reduce exactly
  like one full batch;
* **decoding and metrics** — argmax and accuracy / macro f1 / recall /
  precision (``categorical``), independent sigmoid probabilities with a
  declared threshold plus subset and element accuracy (``multilabel``),
  continuous values with MSE / MAE and no classification fields
  (``regression``).

Three kinds exist; the kind is always declared, never inferred:

* ``"categorical"`` — mutually exclusive classes along axis 1 of the
  output (``(N, C)`` or ``(N, C, ...)``); integer targets shaped like the
  output without the class axis. Needs ``Losses.CROSS_ENTROPY`` or
  ``Losses.NEGATIVE_LOG_LIKELIHOOD``.
* ``"multilabel"`` — independent binary labels: ``(N, L)`` logits and
  0/1 targets of the same shape (NaN = masked). Needs
  ``Losses.BINARY_CROSS_ENTROPY`` (``BCEWithLogitsLoss``).
* ``"regression"`` — continuous targets shaped like the output (a
  ``(N,)`` target is accepted for an ``(N, 1)`` output; NaN = masked).
  Needs a regression loss (``Losses.MEAN_SQUARED_ERROR``, or any
  non-classification loss module assigned to ``model.loss_fn``).

Language-model, ranking, link-prediction and structured-prediction tasks
are out of scope; ``Nets.TRANSFORMER`` models are rejected. Models without
a task keep the legacy classification behaviour and serialization.
"""

from __future__ import annotations

import math
import numbers
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Optional

import numpy as np
import torch

from .nn.enum.losses import Losses
from .nn.enum.nets import Nets
from .nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from .prediction import PredictionResult, ProbabilitySpec, _check_spec_fits, prediction_from_logits

__all__ = [
    "TASK_SPEC_VERSION",
    "TaskAdapter",
    "TaskMetricAccumulator",
    "TaskSpec",
    "TaskValidationError",
    "task_adapter",
]

TASK_SPEC_VERSION = 1
TASK_KINDS = ("categorical", "multilabel", "regression")
_DEFAULT_THRESHOLD = 0.5
_STATE_KEYS = frozenset({"version", "kind", "num_outputs", "ignore_index", "threshold", "labels"})


class TaskValidationError(ValueError):
    """An invalid :class:`TaskSpec`, a model the task cannot drive, or a
    batch whose outputs / targets do not satisfy the declared task."""


def _count(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TaskValidationError(f"TaskSpec {name} must be an integer, got {value!r}")
    if int(value) < minimum:
        raise TaskValidationError(f"TaskSpec {name} must be >= {minimum}, got {value!r}")
    return int(value)


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Declaration of a supervised task (see the module docstring).

    Prefer the constructors :meth:`categorical`, :meth:`multilabel` and
    :meth:`regression`.

    Args:
        kind: ``"categorical"``, ``"multilabel"`` or ``"regression"``.
        num_outputs: classes / labels / targets along output axis 1, or
            ``None`` to accept the net's width. When set, a model whose
            ``output_dim`` differs is rejected at construction.
        ignore_index: categorical only — the target value excluded from
            the loss and every metric (e.g. ``-100`` for padding).
        threshold: multilabel only — probability at or above which a label
            is decoded as positive (default ``0.5``, i.e. ``logit >= 0``).
        labels: optional ordered names, one per output.

    Serializes with :meth:`state` / :meth:`from_state` as a versioned
    mapping (``version`` = :data:`TASK_SPEC_VERSION`).
    """

    kind: str
    num_outputs: Optional[int] = None
    ignore_index: Optional[int] = None
    threshold: float = _DEFAULT_THRESHOLD
    labels: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.kind not in TASK_KINDS:
            raise TaskValidationError(
                f"TaskSpec kind must be one of {', '.join(repr(k) for k in TASK_KINDS)}, got {self.kind!r}"
            )
        minimum = 2 if self.kind == "categorical" else 1
        if self.num_outputs is not None:
            object.__setattr__(self, "num_outputs", _count(self.num_outputs, "num_outputs", minimum=minimum))
        if self.ignore_index is not None:
            if self.kind != "categorical":
                raise TaskValidationError(f"TaskSpec ignore_index applies to categorical tasks only, not {self.kind}")
            if isinstance(self.ignore_index, bool) or not isinstance(self.ignore_index, numbers.Integral):
                raise TaskValidationError(f"TaskSpec ignore_index must be an integer, got {self.ignore_index!r}")
            object.__setattr__(self, "ignore_index", int(self.ignore_index))
        threshold = self.threshold
        if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real) or not math.isfinite(threshold):
            raise TaskValidationError(f"TaskSpec threshold must be a finite real number, got {threshold!r}")
        if self.kind == "multilabel":
            if not 0.0 < float(threshold) < 1.0:
                raise TaskValidationError(f"TaskSpec threshold must be in (0, 1), got {threshold!r}")
        elif float(threshold) != _DEFAULT_THRESHOLD:
            raise TaskValidationError(f"TaskSpec threshold applies to multilabel tasks only, not {self.kind}")
        object.__setattr__(self, "threshold", float(threshold))
        if self.labels is None:
            return
        if isinstance(self.labels, (str, bytes)) or not isinstance(self.labels, Sequence):
            raise TaskValidationError(
                f"TaskSpec labels must be a sequence of strings, got {type(self.labels).__name__}"
            )
        labels = tuple(self.labels)
        bad = [label for label in labels if not isinstance(label, str) or not label]
        if bad:
            raise TaskValidationError(f"TaskSpec labels must be non-empty strings, got {bad!r}")
        duplicates = sorted(label for label, count in Counter(labels).items() if count > 1)
        if duplicates:
            raise TaskValidationError(f"TaskSpec labels must be unique, duplicated: {duplicates}")
        if len(labels) < minimum:
            raise TaskValidationError(f"a {self.kind} TaskSpec needs at least {minimum} label(s), got {len(labels)}")
        if self.num_outputs is not None and len(labels) != self.num_outputs:
            raise TaskValidationError(f"TaskSpec has {len(labels)} label(s) but num_outputs={self.num_outputs}")
        object.__setattr__(self, "labels", labels)
        if self.num_outputs is None:
            object.__setattr__(self, "num_outputs", len(labels))

    # ---------- constructors ----------

    @classmethod
    def categorical(
        cls,
        num_classes: Optional[int] = None,
        *,
        ignore_index: Optional[int] = None,
        labels: Optional[Sequence[str]] = None,
    ) -> TaskSpec:
        """Mutually exclusive classes along output axis 1."""
        return cls("categorical", num_classes, ignore_index=ignore_index, labels=_labels(labels))

    @classmethod
    def multilabel(
        cls,
        num_labels: Optional[int] = None,
        *,
        threshold: float = _DEFAULT_THRESHOLD,
        labels: Optional[Sequence[str]] = None,
    ) -> TaskSpec:
        """Independent binary labels, decoded at ``probability >= threshold``."""
        return cls("multilabel", num_labels, threshold=threshold, labels=_labels(labels))

    @classmethod
    def regression(cls, num_targets: Optional[int] = None, *, labels: Optional[Sequence[str]] = None) -> TaskSpec:
        """Continuous targets shaped like the output."""
        return cls("regression", num_targets, labels=_labels(labels))

    # ---------- serialization ----------

    def state(self) -> dict[str, Any]:
        """Versioned, YAML-safe representation (defaults omitted)."""
        d: dict[str, Any] = {"version": TASK_SPEC_VERSION, "kind": self.kind}
        if self.num_outputs is not None:
            d["num_outputs"] = self.num_outputs
        if self.ignore_index is not None:
            d["ignore_index"] = self.ignore_index
        if self.threshold != _DEFAULT_THRESHOLD:
            d["threshold"] = self.threshold
        if self.labels is not None:
            d["labels"] = list(self.labels)
        return d

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> TaskSpec:
        """Rebuild a spec from :meth:`state`, rejecting an unknown version or key."""
        if not isinstance(state, Mapping):
            raise TaskValidationError(f"TaskSpec state must be a mapping, got {type(state).__name__}")
        version = state.get("version")
        if version != TASK_SPEC_VERSION:
            raise TaskValidationError(
                f"unsupported TaskSpec version {version!r}; this NNx reads version {TASK_SPEC_VERSION}"
            )
        unknown = sorted(set(state) - _STATE_KEYS)
        if unknown:
            raise TaskValidationError(f"unknown TaskSpec key(s) {unknown}")
        return TaskSpec(
            kind=state["kind"],
            num_outputs=state.get("num_outputs"),
            ignore_index=state.get("ignore_index"),
            threshold=state.get("threshold", _DEFAULT_THRESHOLD),
            labels=_labels(state.get("labels")),
        )

    def probability_spec(self) -> Optional[ProbabilitySpec]:
        """The :class:`~nnx.prediction.ProbabilitySpec` this task implies:
        categorical softmax or multilabel (bernoulli) sigmoid over axis 1,
        or ``None`` for regression (no probabilities)."""
        if self.kind == "regression":
            return None
        kind = "categorical" if self.kind == "categorical" else "bernoulli"
        return ProbabilitySpec(kind=kind, class_axis=1, labels=self.labels)

    def __str__(self) -> str:
        return self.kind if self.num_outputs is None else f"{self.kind}[{self.num_outputs}]"


def _labels(labels: Any) -> Any:
    """Freeze a label sequence into a tuple; anything else (None, a bare
    string, a non-sequence) passes through for TaskSpec to validate."""
    if labels is None or isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
        return labels
    return tuple(labels)


# ---------------------------------------------------------------- adapters


def _elementwise_losses() -> dict[type[torch.nn.Module], Callable[..., torch.Tensor]]:
    """Per-element forms of the stock elementwise losses (``reduction="none"``),
    keyed by exact type, honouring each module's own settings (``beta``,
    ``delta``, BCE ``weight`` / ``pos_weight``)."""
    functional = torch.nn.functional
    return {
        torch.nn.MSELoss: lambda fn, o, t: functional.mse_loss(o, t, reduction="none"),
        torch.nn.L1Loss: lambda fn, o, t: functional.l1_loss(o, t, reduction="none"),
        torch.nn.SmoothL1Loss: lambda fn, o, t: functional.smooth_l1_loss(o, t, reduction="none", beta=fn.beta),
        torch.nn.HuberLoss: lambda fn, o, t: functional.huber_loss(o, t, reduction="none", delta=fn.delta),
        torch.nn.BCEWithLogitsLoss: lambda fn, o, t: functional.binary_cross_entropy_with_logits(
            o, t, weight=fn.weight, pos_weight=fn.pos_weight, reduction="none"
        ),
    }


_ELEMENTWISE = _elementwise_losses()


def _elementwise_form(loss_fn: torch.nn.Module) -> Optional[Callable[..., torch.Tensor]]:
    """The per-element form of a stock loss whose ``forward`` is the stock one;
    ``None`` for any other module (a subclass may reduce differently)."""
    for base, form in _ELEMENTWISE.items():
        if type(loss_fn) is base or (isinstance(loss_fn, base) and type(loss_fn).forward is base.forward):
            return form
    return None


class TaskMetricAccumulator:
    """Mergeable per-task statistics over any number of batches.

    ``update`` takes one prepared batch; ``result`` builds the
    :class:`~nnx.NNEvaluationDataPoint` for everything seen so far.
    Metrics are computed over the whole accumulation (not averaged per
    batch), so uneven batches give the same numbers as one full batch.
    ``keep_arrays`` retains the valid targets and predictions for
    user ``extra_metrics``; it is off unless those are requested.
    """

    def __init__(self, adapter: TaskAdapter, *, keep_arrays: bool = False) -> None:
        self.adapter = adapter
        self.count = 0
        self.keep_arrays = keep_arrays
        self._targets: list[np.ndarray] = []
        self._predictions: list[np.ndarray] = []

    def update(self, output: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> None:
        raise NotImplementedError

    def result(
        self,
        *,
        loss: Optional[float],
        extra_metrics: Optional[Mapping[str, Callable]] = None,
    ) -> NNEvaluationDataPoint:
        raise NotImplementedError

    def _record(self, **fields: Any) -> NNEvaluationDataPoint:
        return NNEvaluationDataPoint(
            kind=self.adapter.spec.kind,
            count=self.count,
            status="ok" if self.count else "empty",
            **fields,
        )

    def _keep(self, target: np.ndarray, prediction: np.ndarray) -> None:
        if self.keep_arrays:
            self._targets.append(target)
            self._predictions.append(prediction)

    def _extras(self, extra_metrics: Optional[Mapping[str, Callable]]) -> dict[str, float]:
        """User metrics, called as ``fn(y_true, y_pred)`` on the valid
        (unmasked) targets only — never on masked positions."""
        if not extra_metrics:
            return {}
        if not self.keep_arrays:
            raise RuntimeError("extra_metrics need an accumulator built with keep_arrays=True")
        Y = np.concatenate(self._targets)
        Y_hat = np.concatenate(self._predictions)
        return {name: float(fn(Y, Y_hat)) for name, fn in extra_metrics.items()}


class TaskAdapter:
    """Validation, masking, loss units, decoding and metrics for one
    :class:`TaskSpec` kind. Obtain one with :func:`task_adapter`."""

    kind: ClassVar[str]
    _enum_losses: ClassVar[tuple[Losses, ...]] = ()

    def __init__(self, spec: TaskSpec) -> None:
        if spec.kind != self.kind:
            raise TaskValidationError(f"{type(self).__name__} drives {self.kind} tasks, got {spec.kind}")
        self.spec = spec

    # ---------- model preflight ----------

    def check_model(self, *, net: Nets, loss: Losses, output_dim: Optional[int]) -> None:
        """Reject a model configuration this task cannot drive — before
        any net is built or loader is iterated."""
        if net is Nets.TRANSFORMER:
            raise TaskValidationError(
                "task adapters drive (N, C) supervised outputs; language-model tasks on Nets.TRANSFORMER "
                "are out of scope — use GenerativeNNModel or a custom step"
            )
        if loss not in self._enum_losses:
            allowed = ", ".join(f"Losses.{item.name}" for item in self._enum_losses)
            raise TaskValidationError(f"a {self.kind} task needs {allowed}, got Losses.{loss.name}")
        if self.spec.num_outputs is not None and output_dim is not None and output_dim != self.spec.num_outputs:
            raise TaskValidationError(
                f"the {self.kind} task declares {self.spec.num_outputs} output(s) but the net's "
                f"output_dim is {output_dim}"
            )

    def check_loss_fn(self, loss_fn: torch.nn.Module) -> None:
        """Reject a runtime ``model.loss_fn`` that cannot score this task."""
        raise NotImplementedError

    # ---------- per batch ----------

    def prepare(self, output: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate one batch and return ``(output, target, valid)``:
        tensors shaped for the loss plus a boolean mask of the targets
        that count. Raises :class:`TaskValidationError` before any
        backward pass for a shape, dtype or value the task rejects."""
        raise NotImplementedError

    def loss_terms(
        self,
        loss_fn: torch.nn.Module,
        output: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[float]]:
        """``(display loss, additive numerator, normalization weight)`` over
        the valid targets only — the same contract as the default step's
        loss terms, so accumulation windows normalize once by their valid
        count. An all-masked batch contributes a differentiable zero with
        weight 0 and a NaN display loss."""
        from .nn.nn_model import _loss_terms

        n_valid = int(valid.sum())
        if n_valid == 0:
            zero = output.sum() * 0.0
            return zero + float("nan"), zero, 0.0
        if n_valid == valid.numel():
            # Nothing masked: the loss sees the batch exactly as without a task.
            return _loss_terms(loss_fn, output, target)
        return self._masked_loss_terms(loss_fn, output, target, valid, n_valid)

    def _masked_loss_terms(
        self,
        loss_fn: torch.nn.Module,
        output: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        n_valid: int,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[float]]:
        # Elementwise losses are evaluated at full shape (so per-label
        # weights still broadcast) and masked afterwards.
        form = _elementwise_form(loss_fn)
        if form is None:
            raise TaskValidationError(
                f"masked (NaN) {self.kind} targets need an elementwise loss (MSELoss, L1Loss, SmoothL1Loss, "
                f"HuberLoss, BCEWithLogitsLoss) to exclude them; {type(loss_fn).__name__} cannot be masked"
            )
        filled = torch.where(valid, target, torch.zeros_like(target))
        per_element = torch.where(
            valid, form(loss_fn, output, filled), torch.zeros((), dtype=output.dtype, device=output.device)
        )
        numerator = per_element.sum()
        if getattr(loss_fn, "reduction", "mean") == "sum":
            return numerator, numerator, None
        return numerator / n_valid, numerator, float(n_valid)

    def decode(self, output: torch.Tensor) -> torch.Tensor:
        """Decoded predictions for a prepared ``output``."""
        raise NotImplementedError

    def decode_array(self, logits: np.ndarray) -> np.ndarray:
        """Decoded predictions for raw ``predict()`` logits (numpy)."""
        raise NotImplementedError

    def accumulator(self, *, keep_arrays: bool = False) -> TaskMetricAccumulator:
        """A fresh :class:`TaskMetricAccumulator`; pass ``keep_arrays=True``
        when ``extra_metrics`` will be computed from it."""
        raise NotImplementedError

    def record(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        *,
        loss: Optional[float],
        extra_metrics: Optional[Mapping[str, Callable]] = None,
    ) -> NNEvaluationDataPoint:
        """Validate one batch and return its task record — for custom
        training steps that want the same record the default step writes."""
        output, target, valid = self.prepare(output, target)
        accumulator = self.accumulator(keep_arrays=bool(extra_metrics))
        accumulator.update(output, target, valid)
        return accumulator.result(loss=loss, extra_metrics=extra_metrics)

    def check_logits(self, logits: np.ndarray) -> None:
        """Shape check for raw prediction logits (a loader's first batch),
        so a mismatched net fails before the rest of inference runs."""
        if np.ndim(logits) < 2:
            raise TaskValidationError(
                f"{self.kind} task needs outputs with a sample axis and axis 1, got shape {np.shape(logits)}"
            )
        self._check_width(int(np.shape(logits)[1]), "outputs")
        spec = self.spec.probability_spec()
        if spec is not None:
            _check_spec_fits(logits, spec)

    def prediction(self, logits: np.ndarray, sample_ids: np.ndarray) -> PredictionResult:
        """The rich prediction this task implies (see ``predict_proba``)."""
        spec = self.spec.probability_spec()
        assert spec is not None
        return prediction_from_logits(logits, spec, sample_ids=sample_ids)

    def _check_width(self, width: int, what: str) -> None:
        if self.spec.num_outputs is not None and width != self.spec.num_outputs:
            raise TaskValidationError(
                f"{self.kind} task declares {self.spec.num_outputs} {what} but the output has {width} along axis 1"
            )


def _require_float_output(kind: str, output: torch.Tensor) -> None:
    if not output.is_floating_point():
        raise TaskValidationError(f"{kind} task needs floating-point outputs, got {output.dtype}")
    if output.ndim < 2:
        raise TaskValidationError(
            f"{kind} task needs outputs with a sample axis and axis 1, got shape {tuple(output.shape)}"
        )


class _CategoricalAdapter(TaskAdapter):
    kind = "categorical"
    _enum_losses = (Losses.CROSS_ENTROPY, Losses.NEGATIVE_LOG_LIKELIHOOD)

    def check_loss_fn(self, loss_fn):
        if not isinstance(loss_fn, (torch.nn.CrossEntropyLoss, torch.nn.NLLLoss)):
            raise TaskValidationError(
                f"a categorical task needs a CrossEntropyLoss or NLLLoss loss_fn, got {type(loss_fn).__name__}"
            )

    def prepare(self, output, target):
        _require_float_output(self.kind, output)
        n_classes = int(output.size(1))
        self._check_width(n_classes, "classes")
        if n_classes < 2:
            raise TaskValidationError(f"categorical task needs at least 2 classes along axis 1, got {n_classes}")
        if target.is_floating_point() or target.dtype == torch.bool or target.is_complex():
            raise TaskValidationError(f"categorical targets must be integer class indices, got {target.dtype}")
        expected = (output.size(0), *output.shape[2:])
        if tuple(target.shape) != tuple(expected):
            raise TaskValidationError(
                f"categorical targets must have shape {tuple(expected)} (the output without its class axis), "
                f"got {tuple(target.shape)}"
            )
        # (N, C, *rest) -> (M, C) rows, one per target position.
        flat_output = output.movedim(1, -1).reshape(-1, n_classes)
        flat_target = target.reshape(-1)
        ignore = self.spec.ignore_index
        valid = flat_target != ignore if ignore is not None else torch.ones_like(flat_target, dtype=torch.bool)
        valid_targets = flat_target[valid]
        if valid_targets.numel():
            low, high = torch.stack(torch.aminmax(valid_targets)).tolist()  # one device sync
            if low < 0 or high >= n_classes:
                raise TaskValidationError(
                    f"categorical targets must be class indices in [0, {n_classes})"
                    + (f" or ignore_index={ignore}" if ignore is not None else "")
                    + f", got values in [{low}, {high}]"
                )
        return flat_output, flat_target, valid

    def loss_terms(self, loss_fn, output, target, valid):
        # Rows are masked by selection: (M', C) keeps the class axis, so class
        # weights and CE/NLL's own normalization apply unchanged.
        from .nn.nn_model import _loss_terms

        n_valid = int(valid.sum())
        if n_valid == 0:
            zero = output.sum() * 0.0
            return zero + float("nan"), zero, 0.0
        if n_valid == valid.numel():
            return _loss_terms(loss_fn, output, target)
        return _loss_terms(loss_fn, output[valid], target[valid])

    def decode(self, output):
        return output.argmax(dim=1)

    def decode_array(self, logits):
        return logits.argmax(axis=1)

    def accumulator(self, *, keep_arrays=False):
        # Macro f1 / recall / precision need every (target, prediction) pair.
        return _CategoricalAccumulator(self, keep_arrays=True)


class _CategoricalAccumulator(TaskMetricAccumulator):
    def update(self, output, target, valid):
        n_valid = int(valid.sum())
        if not n_valid:
            return
        self._keep(target[valid].detach().cpu().numpy(), output[valid].argmax(dim=1).detach().cpu().numpy())
        self.count += n_valid

    def result(self, *, loss, extra_metrics=None):
        if not self.count:
            return self._record(loss=None)
        Y = np.concatenate(self._targets)
        Y_hat = np.concatenate(self._predictions)
        edp = NNEvaluationDataPoint.of(Y=Y, Y_hat=Y_hat, extra_metrics=extra_metrics)
        assert edp.accuracy is not None
        return replace(
            edp,
            loss=loss,
            error=float(1 - edp.accuracy),
            kind=self.adapter.spec.kind,
            count=self.count,
            status="ok",
        )


class _MultilabelAdapter(TaskAdapter):
    kind = "multilabel"
    _enum_losses = (Losses.BINARY_CROSS_ENTROPY,)

    def __init__(self, spec: TaskSpec) -> None:
        super().__init__(spec)
        # One decoding rule everywhere: sigmoid(logit) >= threshold  <=>
        # logit >= log(threshold / (1 - threshold)), with no exp overflow.
        self._logit_threshold = math.log(spec.threshold / (1.0 - spec.threshold))

    def check_loss_fn(self, loss_fn):
        if not isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
            raise TaskValidationError(
                f"a multilabel task needs a BCEWithLogitsLoss loss_fn, got {type(loss_fn).__name__}"
            )

    def prepare(self, output, target):
        _require_float_output(self.kind, output)
        self._check_width(int(output.size(1)), "labels")
        if tuple(target.shape) != tuple(output.shape):
            raise TaskValidationError(
                f"multilabel targets must match the output shape {tuple(output.shape)}, got {tuple(target.shape)}"
            )
        if target.is_complex():
            raise TaskValidationError(f"multilabel targets must be real 0/1 values, got {target.dtype}")
        if not target.is_floating_point():
            # Integer / bool 0-1 labels: BCE needs a floating target. 0 and 1
            # are exact in every float dtype, so the output's dtype is safe.
            target = target.to(dtype=output.dtype)
        valid = ~torch.isnan(target)
        values = target[valid]
        if values.numel() and not bool(((values == 0) | (values == 1)).all()):
            raise TaskValidationError("multilabel targets must be 0 or 1 (NaN marks a masked label)")
        return output, target, valid

    def decode(self, output):
        return (output >= self._logit_threshold).to(dtype=torch.long)

    def decode_array(self, logits):
        return (np.asarray(logits) >= self._logit_threshold).astype(np.int64)

    def accumulator(self, *, keep_arrays=False):
        return _MultilabelAccumulator(self, keep_arrays=keep_arrays)

    def prediction(self, logits, sample_ids):
        result = super().prediction(logits, sample_ids)
        if self.spec.threshold == _DEFAULT_THRESHOLD:
            return result
        return replace(result, decoded=self.decode_array(result.logits))


class _MultilabelAccumulator(TaskMetricAccumulator):
    def __init__(self, adapter, *, keep_arrays=False):
        super().__init__(adapter, keep_arrays=keep_arrays)
        self._tp: Optional[np.ndarray] = None
        self._fp: Optional[np.ndarray] = None
        self._fn: Optional[np.ndarray] = None
        self._label_valid: Optional[np.ndarray] = None
        self._element_correct = 0
        self._rows = 0
        self._rows_correct = 0

    def update(self, output, target, valid):
        prediction = self.adapter.decode(output).detach().cpu().numpy()
        truth = target.detach().cpu().numpy()
        mask = valid.detach().cpu().numpy()
        n_labels = truth.shape[1]
        if self._tp is None:
            self._tp = np.zeros(n_labels, dtype=np.int64)
            self._fp = np.zeros(n_labels, dtype=np.int64)
            self._fn = np.zeros(n_labels, dtype=np.int64)
            self._label_valid = np.zeros(n_labels, dtype=np.int64)
        assert self._fp is not None and self._fn is not None and self._label_valid is not None
        truth_1 = (truth == 1) & mask
        truth_0 = (truth == 0) & mask
        pred_1 = prediction == 1
        # Per-label counts over valid entries, reduced over every non-label axis.
        axes = tuple(i for i in range(truth.ndim) if i != 1)
        self._tp += (truth_1 & pred_1).sum(axis=axes)
        self._fp += (truth_0 & pred_1).sum(axis=axes)
        self._fn += (truth_1 & ~pred_1).sum(axis=axes)
        self._label_valid += mask.sum(axis=axes)
        hits = (truth == prediction) & mask
        self._element_correct += int(hits.sum())
        rows_with_valid = mask.reshape(mask.shape[0], -1).any(axis=1)
        rows_correct = (hits | ~mask).reshape(mask.shape[0], -1).all(axis=1) & rows_with_valid
        self._rows += int(rows_with_valid.sum())
        self._rows_correct += int(rows_correct.sum())
        self.count += int(mask.sum())
        self._keep(truth[mask], prediction[mask])

    def result(self, *, loss, extra_metrics=None):
        if not self.count:
            return self._record(loss=None)
        assert self._tp is not None and self._fp is not None and self._fn is not None and self._label_valid is not None
        present = self._label_valid > 0
        tp, fp, fn = self._tp[present], self._fp[present], self._fn[present]
        zeros = np.zeros(tp.shape, dtype=np.float64)
        precision = np.divide(tp, tp + fp, out=zeros.copy(), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=zeros.copy(), where=(tp + fn) > 0)
        f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=zeros.copy(), where=(2 * tp + fp + fn) > 0)
        subset_accuracy = self._rows_correct / self._rows
        element_accuracy = self._element_correct / self.count
        return self._record(
            accuracy=float(subset_accuracy),
            f1=float(f1.mean()),
            recall=float(recall.mean()),
            precision=float(precision.mean()),
            loss=loss,
            error=float(1 - subset_accuracy),
            metrics={"subset_accuracy": float(subset_accuracy), "element_accuracy": float(element_accuracy)},
            extra=self._extras(extra_metrics),
        )


class _RegressionAdapter(TaskAdapter):
    kind = "regression"
    _enum_losses = (Losses.MEAN_SQUARED_ERROR,)
    _classification_losses = (torch.nn.CrossEntropyLoss, torch.nn.NLLLoss, torch.nn.BCEWithLogitsLoss, torch.nn.BCELoss)

    def check_loss_fn(self, loss_fn):
        if isinstance(loss_fn, self._classification_losses):
            raise TaskValidationError(
                f"a regression task needs a regression loss (e.g. MSELoss, L1Loss, HuberLoss), "
                f"got {type(loss_fn).__name__}"
            )

    def prepare(self, output, target):
        _require_float_output(self.kind, output)
        self._check_width(int(output.size(1)), "targets")
        if not target.is_floating_point():
            raise TaskValidationError(f"regression targets must be floating point, got {target.dtype}")
        if target.ndim == 1 and output.ndim == 2 and output.size(1) == 1 and target.size(0) == output.size(0):
            target = target.unsqueeze(1)
        if tuple(target.shape) != tuple(output.shape):
            raise TaskValidationError(
                f"regression targets must match the output shape {tuple(output.shape)}, got {tuple(target.shape)} "
                "(no broadcasting: an (N,) target is only accepted for an (N, 1) output)"
            )
        # The target keeps its own dtype (no cast to a reduced-precision
        # autocast output): the loss promotes exactly as without a task.
        if bool(torch.isinf(target).any()):
            raise TaskValidationError("regression targets must be finite (NaN marks a masked target; ±inf is rejected)")
        return output, target, ~torch.isnan(target)

    def decode(self, output):
        return output

    def decode_array(self, logits):
        return np.array(logits, copy=True)

    def accumulator(self, *, keep_arrays=False):
        return _RegressionAccumulator(self, keep_arrays=keep_arrays)

    def prediction(self, logits, sample_ids):
        values = np.asarray(logits)
        ids = np.asarray(sample_ids, dtype=np.int64)
        return PredictionResult(logits=values, probabilities=None, decoded=values.copy(), sample_ids=ids, spec=None)


class _RegressionAccumulator(TaskMetricAccumulator):
    def __init__(self, adapter, *, keep_arrays=False):
        super().__init__(adapter, keep_arrays=keep_arrays)
        self._sse = 0.0
        self._sae = 0.0

    def update(self, output, target, valid):
        prediction = output.detach().to(dtype=torch.float64)
        truth = target.detach().to(dtype=torch.float64)
        diff = (prediction - truth)[valid]
        sums = torch.stack([(diff * diff).sum(), diff.abs().sum()]).tolist()  # one device sync
        self._sse += sums[0]
        self._sae += sums[1]
        self.count += int(diff.numel())
        if self.keep_arrays:
            self._keep(truth[valid].cpu().numpy(), prediction[valid].cpu().numpy())

    def result(self, *, loss, extra_metrics=None):
        if not self.count:
            return self._record(loss=None)
        return self._record(
            loss=loss,
            metrics={"mse": self._sse / self.count, "mae": self._sae / self.count},
            extra=self._extras(extra_metrics),
        )


_ADAPTERS: dict[str, type[TaskAdapter]] = {
    "categorical": _CategoricalAdapter,
    "multilabel": _MultilabelAdapter,
    "regression": _RegressionAdapter,
}


def task_adapter(spec: TaskSpec) -> TaskAdapter:
    """Return the adapter that drives ``spec``."""
    if not isinstance(spec, TaskSpec):
        raise TypeError(f"task_adapter() needs a TaskSpec, got {type(spec).__name__}")
    return _ADAPTERS[spec.kind](spec)
