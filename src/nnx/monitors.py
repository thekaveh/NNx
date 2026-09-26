"""Named metric and monitor contracts (FEAT-003).

Two declarations make metric-driven decisions explicit and shared.

**Metrics.** A :class:`MetricSpec` names a *registered* metric by
``(id, version)`` plus a JSON-like ``config`` and reports it under
``name`` (the id by default). Every registered metric declares the
prediction **input** it receives and its natural direction:

======== ================ =====================================================
id       input            value (over the full sample of an epoch)
======== ================ =====================================================
accuracy ``labels``       fraction of decoded predictions equal to the target
f1       ``labels``       F1 score (``config={"average": "macro"}`` default)
nll      ``probabilities`` mean negative log-likelihood of the target
brier    ``probabilities`` mean Brier score (summed over classes when categorical)
mae      ``continuous``   mean absolute error
mse      ``continuous``   mean squared error
======== ================ =====================================================

Values are accumulated over **every** sample the epoch saw — additive
metrics as running sums, non-additive ones (F1) over the full sample — so a
short last batch is weighted exactly like a full one, never averaged as a
batch mean. Categorical models feed one row per sample (labels are the
argmax, probabilities the softmax over the class axis); multilabel
(Bernoulli) and continuous models feed one entry per valid output, each a
binary decision (decoded with the task's threshold) or a value. Ignored /
masked targets are excluded. F1 on multilabel inputs is the F1 of the
positive decisions pooled over every output (``config={"average":
"binary"}`` — sklearn's multilabel "micro" F1); per-label averaging is not
available there. Custom metrics register with :func:`register_metric`; a spec
naming an unknown ``(id, version)`` is rejected before training starts,
without any metric code running.

**Monitors.** A :class:`MonitorSpec` names what drives selection: the split
(``"val"`` or ``"train"``), the metric (``"loss"``, ``"error"`` or a declared
metric's name), the direction, the minimum improvement and what to do when
the value is missing or non-finite. :meth:`MonitorSpec.improved` is the
**one** improvement rule: when ``NNTrainParams(monitor=...)`` is set, BEST
selection, ``ReduceLROnPlateau`` and every ``EarlyStopping`` built with the
same spec make identical decisions from the same epoch summary.
"""

from __future__ import annotations

import json
import math
import numbers
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol, cast

import numpy as np
import torch

from ._config import _SLUG, _freeze_config, _thaw_config
from ._validation import require_count

if TYPE_CHECKING:
    from .nn.params.nn_evaluation_data_point import NNEvaluationDataPoint

__all__ = [
    "METRIC_INPUTS",
    "MetricAccumulator",
    "MetricDefinition",
    "MetricSpec",
    "MonitorRecord",
    "MonitorSpec",
    "MonitorTracker",
    "MonitorUnavailableError",
    "register_metric",
    "registered_metrics",
    "unregister_metric",
]

METRIC_INPUTS = ("labels", "probabilities", "continuous")
_MODES = ("min", "max")
_POLICIES = ("skip", "error")
_SPLITS = ("val", "train")
_RECORD_STATUSES = ("ok", "missing", "nonfinite")
# The record's own fields a monitor may name directly (both lower-is-better).
_RECORD_FIELDS = {"loss": "min", "error": "min"}


class MonitorUnavailableError(ValueError):
    """A monitor with ``on_missing="error"`` / ``on_nonfinite="error"``
    found no finite value to decide on."""


class MetricAccumulator(Protocol):
    """Accumulates one metric over an epoch's full sample.

    ``update`` receives one batch's valid targets and the declared
    prediction input as NumPy arrays (sample axis first); ``result``
    returns the metric over everything seen, or ``None`` if nothing was."""

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None: ...

    def result(self) -> Optional[float]: ...


@dataclass(frozen=True)
class MetricDefinition:
    """A registered metric: what it receives and how it is computed."""

    id: str
    version: int
    input: str
    mode: str
    factory: Callable[[Mapping[str, Any]], MetricAccumulator]
    check_config: Optional[Callable[[Mapping[str, Any]], None]] = None


_REGISTRY: dict[tuple[str, int], MetricDefinition] = {}


def _require_slug(value: object, what: str) -> str:
    if not isinstance(value, str) or _SLUG.fullmatch(value) is None:
        raise ValueError(f"{what} must be a slug (letters, digits, '.', '_', '-'), got {value!r}")
    return value


def _require_version(value: object, owner: str) -> int:
    return require_count(value, "version", owner=owner, minimum=1)


def register_metric(
    id: str,
    version: int,
    factory: Callable[[Mapping[str, Any]], MetricAccumulator],
    *,
    input: str,
    mode: str,
    check_config: Optional[Callable[[Mapping[str, Any]], None]] = None,
    replace: bool = False,
) -> None:
    """Register a metric under ``(id, version)`` for this process.

    ``factory(config)`` returns a fresh :class:`MetricAccumulator` for one
    split of one epoch (``config`` is the spec's read-only config);
    ``input`` is one of :data:`METRIC_INPUTS`; ``mode`` is the natural
    direction (``"min"``: lower is better). ``check_config(config)``, when
    given, validates a spec's config as the run starts, before any batch.
    Registering an existing ``(id, version)`` raises unless
    ``replace=True``; bump the version when the metric's meaning changes.
    """
    key = (_require_slug(id, "metric id"), _require_version(version, "register_metric"))
    if input not in METRIC_INPUTS:
        raise ValueError(f"metric input must be one of {', '.join(repr(i) for i in METRIC_INPUTS)}, got {input!r}")
    if mode not in _MODES:
        raise ValueError(f"metric mode must be 'min' or 'max', got {mode!r}")
    if not callable(factory):
        raise TypeError(f"metric factory for {id}@v{version} must be callable")
    if check_config is not None and not callable(check_config):
        raise TypeError(f"check_config for {id}@v{version} must be callable")
    if key in _REGISTRY and not replace:
        raise ValueError(f"metric {id}@v{version} is already registered (pass replace=True to replace it)")
    _REGISTRY[key] = MetricDefinition(key[0], key[1], input, mode, factory, check_config)


def unregister_metric(id: str, version: int) -> bool:
    """Remove a registration; ``True`` when one existed."""
    return _REGISTRY.pop((id, version), None) is not None


def registered_metrics() -> tuple[tuple[str, int], ...]:
    """Every registered ``(id, version)``, sorted."""
    return tuple(sorted(_REGISTRY))


class MetricSpec:
    """Serializable declaration of a registered metric.

    Args:
        id / version: the registration to use.
        config: JSON-like options handed to the metric read-only
            (``None``, ``bool``, ``int``, finite ``float``, ``str``, lists
            and string-keyed mappings).
        name: the key the value is reported under (default: ``id``); a slug,
            unique among a run's metrics, and never ``"loss"`` / ``"error"``.

    The spec never holds code, so it round-trips through ``run.yaml`` and
    reloads without the metric being registered. It is resolved against
    the registry when training starts: an unknown ``(id, version)`` then
    fails before anything is trained, and no metric code has run.
    """

    __slots__ = ("id", "version", "config", "name")
    id: str
    version: int
    config: Mapping[str, Any]
    name: Optional[str]

    def __init__(
        self, id: str, version: int = 1, config: Optional[Mapping[str, Any]] = None, name: Optional[str] = None
    ) -> None:
        object.__setattr__(self, "id", _require_slug(id, "MetricSpec id"))
        object.__setattr__(self, "version", _require_version(version, "MetricSpec"))
        if config is None:
            config = {}
        if not isinstance(config, Mapping):
            raise TypeError(f"MetricSpec config must be a str-keyed mapping, got {type(config).__name__}")
        object.__setattr__(self, "config", _freeze_config(config, "", "MetricSpec"))
        if name is not None:
            _require_slug(name, "MetricSpec name")
        object.__setattr__(self, "name", name)
        if self.label in _RECORD_FIELDS:
            raise ValueError(f"MetricSpec name {self.label!r} is reserved for the record's own {self.label!r} field")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"MetricSpec is immutable; cannot set {name!r}")

    def __reduce__(self):
        return (MetricSpec, (self.id, self.version, _thaw_config(self.config), self.name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MetricSpec):
            return NotImplemented
        return json.dumps(self.state(), sort_keys=True) == json.dumps(other.state(), sort_keys=True)

    def __hash__(self) -> int:
        return hash(json.dumps(self.state(), sort_keys=True))

    def __repr__(self) -> str:
        parts = [f"{self.id!r}", f"version={self.version}"]
        if self.config:
            parts.append(f"config={_thaw_config(self.config)!r}")
        if self.name is not None:
            parts.append(f"name={self.name!r}")
        return f"MetricSpec({', '.join(parts)})"

    @property
    def label(self) -> str:
        """The key the metric is reported under."""
        return self.name or self.id

    @property
    def definition(self) -> MetricDefinition:
        """The registration this spec names (``ValueError`` when unknown)."""
        definition = _REGISTRY.get((self.id, self.version))
        if definition is None:
            known = ", ".join(f"{i}@v{v}" for i, v in registered_metrics())
            raise ValueError(
                f"unknown metric {self.id!r}@v{self.version} (reported as {self.label!r}); register it with "
                f"nnx.register_metric before training. Registered: {known}"
            )
        return definition

    @property
    def input(self) -> str:
        return self.definition.input

    @property
    def mode(self) -> str:
        return self.definition.mode

    def check(self) -> None:
        """Resolve the registration and validate the config, without
        computing anything."""
        definition = self.definition
        if definition.check_config is not None:
            definition.check_config(self.config)

    def accumulator(self) -> MetricAccumulator:
        return self.definition.factory(self.config)

    def state(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "version": self.version}
        if self.config:
            d["config"] = _thaw_config(self.config)
        if self.name is not None:
            d["name"] = self.name
        return d

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> MetricSpec:
        return MetricSpec(
            id=state["id"], version=state.get("version", 1), config=state.get("config") or {}, name=state.get("name")
        )


def _unique_metrics(metrics: Iterable[MetricSpec], owner: str) -> tuple[MetricSpec, ...]:
    out = tuple(metrics)
    for spec in out:
        if not isinstance(spec, MetricSpec):
            raise TypeError(f"{owner}.metrics entries must be MetricSpec, got {type(spec).__name__}")
    labels = [spec.label for spec in out]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"{owner}.metrics names must be unique; duplicated: {', '.join(duplicates)} (pass name=...)")
    return out


# --- built-in accumulators -------------------------------------------------


class _Mean:
    """Sample-additive metric: sum of per-sample terms over the sample count."""

    def __init__(self, terms: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
        self._terms = terms
        self._sum = 0.0
        self._count = 0

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None:
        values = np.asarray(self._terms(np.asarray(target), np.asarray(prediction)), dtype=np.float64).reshape(-1)
        self._sum += float(values.sum())
        self._count += int(values.size)

    def result(self) -> Optional[float]:
        return self._sum / self._count if self._count else None


_EPS = 1e-12


def _categorical(target: np.ndarray, probabilities: np.ndarray) -> bool:
    return probabilities.ndim == target.ndim + 1


def _accuracy_terms(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return (target == prediction).astype(np.float64)


def _nll_terms(target: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(probabilities.astype(np.float64), _EPS, 1.0)
    if _categorical(target, probabilities):
        return -np.log(np.take_along_axis(p, target.astype(np.int64)[..., None], axis=-1)[..., 0])
    t = target.astype(np.float64)
    q = np.clip(1.0 - probabilities.astype(np.float64), _EPS, 1.0)
    return -(t * np.log(p) + (1.0 - t) * np.log(q))


def _brier_terms(target: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    p = probabilities.astype(np.float64)
    if _categorical(target, probabilities):
        onehot = np.zeros_like(p)
        np.put_along_axis(onehot, target.astype(np.int64)[..., None], 1.0, axis=-1)
        return ((p - onehot) ** 2).sum(axis=-1)
    return (p - target.astype(np.float64)) ** 2


def _abs_terms(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.abs(prediction.astype(np.float64) - target.astype(np.float64))


def _squared_terms(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return (prediction.astype(np.float64) - target.astype(np.float64)) ** 2


_F1_AVERAGES = ("macro", "micro", "weighted", "binary")


def _check_f1(config: Mapping[str, Any]) -> None:
    unknown = sorted(set(config) - {"average"})
    if unknown:
        raise ValueError(f"f1 metric config has unknown keys {unknown}; accepted: 'average'")
    average = config.get("average", "macro")
    if average not in _F1_AVERAGES:
        raise ValueError(
            f"f1 metric average must be one of {', '.join(repr(a) for a in _F1_AVERAGES)}, got {average!r}"
        )


def _check_no_config(metric: str) -> Callable[[Mapping[str, Any]], None]:
    def check(config: Mapping[str, Any]) -> None:
        if config:
            raise ValueError(f"{metric} metric takes no config, got {sorted(config)}")

    return check


class _F1:
    """Non-additive: computed once over every label of the epoch."""

    def __init__(self, average: str = "macro") -> None:
        self._average = average
        self._targets: list[np.ndarray] = []
        self._predictions: list[np.ndarray] = []

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None:
        self._targets.append(np.asarray(target).reshape(-1))
        self._predictions.append(np.asarray(prediction).reshape(-1))

    def result(self) -> Optional[float]:
        if not self._targets:
            return None
        from sklearn.metrics import f1_score

        target = np.concatenate(self._targets)
        if target.size == 0:
            return None
        return float(
            f1_score(target, np.concatenate(self._predictions), average=self._average, zero_division=cast(Any, 0))
        )


for _id, _input, _mode, _factory, _check in (
    ("accuracy", "labels", "max", lambda config: _Mean(_accuracy_terms), _check_no_config("accuracy")),
    ("f1", "labels", "max", lambda config: _F1(**config), _check_f1),
    ("nll", "probabilities", "min", lambda config: _Mean(_nll_terms), _check_no_config("nll")),
    ("brier", "probabilities", "min", lambda config: _Mean(_brier_terms), _check_no_config("brier")),
    ("mae", "continuous", "min", lambda config: _Mean(_abs_terms), _check_no_config("mae")),
    ("mse", "continuous", "min", lambda config: _Mean(_squared_terms), _check_no_config("mse")),
):
    register_metric(_id, 1, _factory, input=_input, mode=_mode, check_config=_check)


# --- monitors -----------------------------------------------------------------


@dataclass(frozen=True)
class MonitorSpec:
    """What BEST selection, early stopping and plateau scheduling track.

    Args:
        metric: ``"loss"``, ``"error"`` or the name of a declared
            :class:`MetricSpec` (``NNTrainParams.metrics``).
        split: ``"val"`` (default) — the whole-validation-set record — or
            ``"train"`` — the whole-epoch training summary (full-epoch
            denominators, never the last batch).
        mode: ``"min"`` / ``"max"``; ``None`` (default) takes the metric's
            natural direction (``"min"`` for loss and error).
        min_delta: an improvement must beat the best by more than this
            (finite, ``>= 0``); ties never improve.
        on_missing: ``"skip"`` (default) — an epoch without a value makes no
            decision (not counted toward patience, no plateau step, no
            BEST) — or ``"error"``: raise :class:`MonitorUnavailableError`.
        on_nonfinite: ``"skip"`` (default) — a NaN / ±inf value is an epoch
            without improvement — or ``"error"``.
    """

    metric: str = "loss"
    split: str = "val"
    mode: Optional[str] = None
    min_delta: float = 0.0
    on_missing: str = "skip"
    on_nonfinite: str = "skip"

    def __post_init__(self) -> None:
        _require_slug(self.metric, "MonitorSpec metric")
        if self.split not in _SPLITS:
            raise ValueError(f"MonitorSpec split must be 'val' or 'train', got {self.split!r}")
        if self.mode is not None and self.mode not in _MODES:
            raise ValueError(f"MonitorSpec mode must be 'min', 'max' or None, got {self.mode!r}")
        if self.mode is None and self.metric in _RECORD_FIELDS:
            object.__setattr__(self, "mode", _RECORD_FIELDS[self.metric])
        delta = self.min_delta
        if isinstance(delta, bool) or not isinstance(delta, numbers.Real) or not math.isfinite(delta) or delta < 0:
            raise ValueError(f"MonitorSpec min_delta must be a finite number >= 0, got {delta!r}")
        object.__setattr__(self, "min_delta", float(delta))
        for name in ("on_missing", "on_nonfinite"):
            if getattr(self, name) not in _POLICIES:
                raise ValueError(f"MonitorSpec {name} must be 'skip' or 'error', got {getattr(self, name)!r}")

    @property
    def key(self) -> str:
        """``"<split>.<metric>"``, as shown by displays and loggers."""
        return f"{self.split}.{self.metric}"

    def check_names(self, metrics: Sequence[MetricSpec], owner: str = "monitor") -> None:
        """Fail when the metric is neither a record field nor declared."""
        if self.metric in _RECORD_FIELDS:
            return
        labels = {spec.label for spec in metrics}
        if self.metric not in labels:
            declared = ", ".join(["loss", "error", *sorted(labels)])
            raise ValueError(f"{owner} {self.key!r} names an undeclared metric; declared: {declared}")

    def resolve(self, metrics: Sequence[MetricSpec], owner: str = "monitor") -> MonitorSpec:
        """Check the name against the declared metrics and fill in the
        metric's natural direction when ``mode`` is ``None``."""
        self.check_names(metrics, owner)
        if self.mode is not None:
            return self
        spec = next(m for m in metrics if m.label == self.metric)
        return MonitorSpec(
            metric=self.metric,
            split=self.split,
            mode=spec.mode,
            min_delta=self.min_delta,
            on_missing=self.on_missing,
            on_nonfinite=self.on_nonfinite,
        )

    def improved(self, current: float, best: Optional[float]) -> bool:
        """The shared rule: the first finite value improves; later values
        must beat ``best`` by more than ``min_delta`` (ties never improve);
        a non-finite value never improves."""
        if self.mode is None:
            raise ValueError(f"monitor {self.key!r} has no direction; resolve it against the declared metrics")
        if not math.isfinite(current):
            return False
        if best is None:
            return True
        if self.mode == "max":
            return current > best + self.min_delta
        return current < best - self.min_delta

    def value(self, *, train: Optional[NNEvaluationDataPoint], val: Optional[NNEvaluationDataPoint]) -> Optional[float]:
        """The monitored value from an epoch's training summary / validation
        record (``None`` when absent)."""
        edp = val if self.split == "val" else train
        if edp is None:
            return None
        value = getattr(edp, self.metric) if self.metric in _RECORD_FIELDS else edp.metrics.get(self.metric)
        return None if value is None else float(value)

    def state(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "split": self.split,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "on_missing": self.on_missing,
            "on_nonfinite": self.on_nonfinite,
        }

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> MonitorSpec:
        fields = ("metric", "split", "mode", "min_delta", "on_missing", "on_nonfinite")
        return MonitorSpec(**{name: state[name] for name in fields if name in state})


@dataclass(frozen=True)
class MonitorRecord:
    """One epoch's monitor decision, persisted with the epoch's record.

    ``status`` is ``"ok"``, ``"missing"`` (no value; no decision) or
    ``"nonfinite"`` (an epoch without improvement); ``improved`` says
    whether this epoch became the best."""

    monitor: MonitorSpec
    value: Optional[float]
    status: str
    improved: bool

    def __post_init__(self) -> None:
        if self.status not in _RECORD_STATUSES:
            raise ValueError(f"MonitorRecord status must be one of {_RECORD_STATUSES}, got {self.status!r}")

    def state(self) -> dict[str, Any]:
        return {"monitor": self.monitor.state(), "value": self.value, "status": self.status, "improved": self.improved}

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> MonitorRecord:
        # A missing value reloads from CSV as NaN; a non-finite one stays.
        status = state["status"]
        value = state.get("value")
        return MonitorRecord(
            monitor=MonitorSpec.from_state(state["monitor"]),
            value=None if status == "missing" or value is None else float(value),
            status=status,
            improved=_as_bool(state["improved"]),
        )


def _as_bool(value: Any) -> bool:
    """CSV reloads may yield ``"True"`` / ``"False"`` strings or NumPy bools."""
    if isinstance(value, str):
        if value in ("True", "true", "1"):
            return True
        if value in ("False", "false", "0"):
            return False
        raise ValueError(f"not a boolean: {value!r}")
    return bool(value)


class MonitorTracker:
    """Applies one :class:`MonitorSpec` epoch by epoch: returns each
    epoch's :class:`MonitorRecord` and remembers the best value.

    The run's tracker is also an optional checkpointable component
    (``nnx.monitor``, FEAT-005): a stateful warm resume continues from the
    source run's best, so BEST keeps agreeing with a restored
    ``EarlyStopping`` and plateau scheduler after the split.
    """

    def __init__(self, spec: MonitorSpec, *, warn_missing: bool = False) -> None:
        if spec.mode is None:
            raise ValueError(f"monitor {spec.key!r} has no direction; resolve it against the declared metrics")
        self.spec = spec
        self.best: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self._warn_missing = warn_missing

    def observe(self, value: Optional[float], *, epoch: int) -> MonitorRecord:
        spec = self.spec
        if value is None:
            if spec.on_missing == "error":
                raise MonitorUnavailableError(f"epoch {epoch}: monitor {spec.key!r} has no value")
            if self._warn_missing:
                # Once per run: an epoch without a value makes no decision,
                # so a monitor that is never fed would silently never
                # select BEST.
                self._warn_missing = False
                warnings.warn(
                    f"epoch {epoch}: monitor {spec.key!r} has no value, so this epoch makes no BEST / plateau "
                    "decision; a custom step or evaluator must report the metric in its record's metrics",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return MonitorRecord(spec, None, "missing", False)
        current = float(value)
        if not math.isfinite(current):
            if spec.on_nonfinite == "error":
                raise MonitorUnavailableError(f"epoch {epoch}: monitor {spec.key!r} is non-finite ({current})")
            return MonitorRecord(spec, current, "nonfinite", False)
        improved = spec.improved(current, self.best)
        if improved:
            self.best, self.best_epoch = current, epoch
        return MonitorRecord(spec, current, "ok", improved)

    # ---------- checkpointable component (FEAT-005) ----------

    def component_spec(self) -> Any:
        from .components import ComponentSpec

        return ComponentSpec("nnx.monitor", version=1, required=False)

    def component_state(self) -> dict[str, Any]:
        return {"monitor": self.spec.state(), "best": self.best, "best_epoch": self.best_epoch}

    def check_component_state(self, state: Mapping[str, Any], *, version: int) -> list[str]:
        if state.get("monitor") != self.spec.state():
            return [f"the checkpoint's monitor {state.get('monitor')!r} differs from this run's {self.spec.state()!r}"]
        return []

    def load_component_state(self, state: Mapping[str, Any], *, version: int) -> None:
        best = state.get("best")
        self.best = None if best is None else float(best)
        self.best_epoch = state.get("best_epoch")


# --- metric inputs from model outputs (internal) -------------------------------

# The prediction inputs NNx can derive for each output domain.
_DOMAIN_INPUTS = {
    "categorical": ("labels", "probabilities"),
    "bernoulli": ("labels", "probabilities"),
    "continuous": ("continuous",),
}


def _metric_domain(loss_fn: torch.nn.Module, task: Any) -> Optional[str]:
    """How a model's outputs become metric inputs: ``"categorical"``
    (softmax over axis 1), ``"bernoulli"`` (element-wise sigmoid),
    ``"continuous"`` (raw outputs), or ``None`` when NNx cannot tell."""
    if task is not None:
        return {"categorical": "categorical", "multilabel": "bernoulli", "regression": "continuous"}.get(task.kind)
    if isinstance(loss_fn, (torch.nn.CrossEntropyLoss, torch.nn.NLLLoss)):
        return "categorical"
    if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
        return "bernoulli"
    if isinstance(loss_fn, (torch.nn.MSELoss, torch.nn.L1Loss, torch.nn.SmoothL1Loss, torch.nn.HuberLoss)):
        return "continuous"
    return None


def _check_metric_inputs(
    metrics: Sequence[MetricSpec], domain: Optional[str], *, where: str, n_classes: Optional[int] = None
) -> None:
    """Fail before training when NNx cannot derive a metric's input, or a
    built-in metric's config cannot apply to this model's outputs."""
    for spec in metrics:
        spec.check()
        available = _DOMAIN_INPUTS.get(domain or "", ())
        if spec.input not in available:
            what = f"a {domain} model" if domain else "a model whose loss NNx cannot map to a prediction domain"
            raise ValueError(
                f"metric {spec.label!r} needs {spec.input} inputs, which {where} cannot derive for {what} "
                f"(available: {', '.join(available) or 'none'}); declare a TaskSpec, choose a metric with a "
                "matching input, or compute it in a custom step and report it in the record's metrics"
            )
        if (spec.id, spec.version) == ("f1", 1):
            average = spec.config.get("average", "macro")
            if domain == "bernoulli" and average != "binary":
                raise ValueError(
                    f"metric {spec.label!r}: multilabel outputs are scored as pooled binary decisions, so f1 "
                    f"needs config={{'average': 'binary'}} (the positive-decision F1 pooled over every output, "
                    f"i.e. multilabel micro F1), not {average!r}"
                )
            if domain == "categorical" and average == "binary" and n_classes != 2:
                raise ValueError(
                    f"metric {spec.label!r}: f1 average='binary' needs exactly 2 classes"
                    f"{f', this model has {n_classes}' if n_classes is not None else ''}; use 'macro', 'micro' "
                    "or 'weighted'"
                )


def _batch_inputs(
    domain: str,
    target: torch.Tensor,
    output: torch.Tensor,
    valid: Optional[torch.Tensor],
    ignore_index: Optional[int],
    needed: frozenset[str],
    logit_threshold: float = 0.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One batch's valid targets and the ``needed`` prediction inputs as
    NumPy arrays (only those — no softmax or host copy for unused ones)."""
    output = output.detach()
    target = target.detach()
    if domain == "categorical":
        if target.is_floating_point() and target.shape == output.shape:
            target = target.argmax(dim=1)  # one-hot / soft class targets
        if output.ndim > 2:  # (N, C, ...) → one row per position
            n_classes = output.shape[1]
            output = output.movedim(1, -1).reshape(-1, n_classes)
            target = target.reshape(-1)
            valid = valid.reshape(-1) if valid is not None else None
        if valid is None and ignore_index is not None:
            valid = target != ignore_index
        if valid is not None:
            output, target = output[valid], target[valid]
        inputs: dict[str, np.ndarray] = {}
        if "labels" in needed:
            inputs["labels"] = output.argmax(dim=1).cpu().numpy()
        if "probabilities" in needed:
            inputs["probabilities"] = torch.softmax(output.float(), dim=1).cpu().numpy()
        return target.long().cpu().numpy(), inputs
    if valid is not None:
        output, target = output[valid], target[valid]
    output, target = output.reshape(-1), target.reshape(-1)
    if domain == "bernoulli":
        inputs = {}
        if "labels" in needed:
            # The task's decision threshold (logit space), like decode().
            inputs["labels"] = (output >= logit_threshold).long().cpu().numpy()
        if "probabilities" in needed:
            inputs["probabilities"] = torch.sigmoid(output.float()).cpu().numpy()
        return (target >= 0.5).long().cpu().numpy(), inputs
    return target.float().cpu().numpy(), {"continuous": output.float().cpu().numpy()}


class _MetricSet:
    """The declared metrics' accumulators for one split of one epoch."""

    def __init__(
        self,
        metrics: Sequence[MetricSpec],
        domain: Optional[str],
        ignore_index: Optional[int] = None,
        logit_threshold: float = 0.0,
    ):
        self._specs = tuple(metrics)
        self._domain = domain
        self._ignore_index = ignore_index
        self._logit_threshold = logit_threshold
        self._needed = frozenset(spec.input for spec in self._specs)
        self._accumulators = {spec.label: spec.accumulator() for spec in self._specs}

    def update(self, target: torch.Tensor, output: torch.Tensor, valid: Optional[torch.Tensor] = None) -> int:
        """Feed one batch; returns the number of samples scored."""
        if not self._specs or self._domain is None:
            return 0
        target_np, inputs = _batch_inputs(
            self._domain, target, output, valid, self._ignore_index, self._needed, self._logit_threshold
        )
        if target_np.shape[0] == 0:
            return 0
        for spec in self._specs:
            self._accumulators[spec.label].update(target_np, inputs[spec.input])
        return int(target_np.shape[0])

    def results(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for label, accumulator in self._accumulators.items():
            value = accumulator.result()
            if value is not None:
                out[label] = float(value)
        return out


def _ignore_index(loss_fn: torch.nn.Module) -> Optional[int]:
    if isinstance(loss_fn, (torch.nn.CrossEntropyLoss, torch.nn.NLLLoss)):
        return int(loss_fn.ignore_index)
    return None


class _WeightedMean:
    def __init__(self) -> None:
        self.total = 0.0
        self.weight = 0.0
        self.summed = False

    def add(self, value: Optional[float], weight: Optional[float]) -> None:
        if value is None:
            return
        if weight is None:  # a sum-reduction loss is already a total
            self.summed = True
            self.total += float(value)
            return
        self.total += float(value) * weight
        self.weight += weight

    def result(self) -> Optional[float]:
        if self.summed:
            return self.total
        return self.total / self.weight if self.weight else None


class _TrainEpochSummary:
    """Whole-epoch training record: the loss and error averaged with each
    batch's own denominator, and the declared metrics over the full sample.

    The default training step reports each batch's outputs and its loss and
    error denominators through :meth:`observe`; an objective run (FEAT-004)
    reports its loss terms through :meth:`observe_terms`, so the epoch's
    loss is the objective's own window rule applied to the whole epoch; for
    custom steps only the returned records are available, weighted by their
    ``count`` (task records) or by the batch's sample count."""

    def __init__(
        self,
        metrics: Sequence[MetricSpec],
        domain: Optional[str],
        ignore_index: Optional[int],
        logit_threshold: float = 0.0,
    ) -> None:
        self._metrics = _MetricSet(metrics, domain, ignore_index, logit_threshold)
        self._loss = _WeightedMean()
        self._error = _WeightedMean()
        self._pending: Optional[tuple[Optional[float], Optional[float]]] = None
        self._records = 0
        # Objective runs: {term name: [reduction, weight, Σ numerator, Σ denominator]}.
        self._terms: dict[str, list[Any]] = {}
        self._pending_terms = False

    def observe(
        self,
        target: torch.Tensor,
        output: torch.Tensor,
        valid: Optional[torch.Tensor],
        loss_weight: Optional[float],
        error_weight: Optional[float] = None,
    ) -> None:
        """Called by the default step: ``loss_weight`` is the batch loss's
        denominator (``None`` for a sum-reduction loss) and
        ``error_weight`` the number of targets its error scores (``None``:
        the record's ``count``)."""
        self._metrics.update(target, output, valid)
        self._pending = (loss_weight, error_weight)

    def observe_terms(self, terms: Sequence[Any]) -> None:
        """Called for each objective microbatch with its ``LossTerm``\\ s:
        normalized terms add their numerators and denominators, summed terms
        their totals."""
        for term in terms:
            entry = self._terms.setdefault(term.name, [term.reduction, term.weight, 0.0, 0.0])
            entry[2] += term._numerator_value
            if term.reduction == "mean":
                entry[3] += float(term.denominator)
        self._pending_terms = True

    def _terms_loss(self) -> Optional[float]:
        present = [
            (weight, numerator if reduction == "sum" else numerator / denominator)
            for reduction, weight, numerator, denominator in self._terms.values()
            if reduction == "sum" or denominator
        ]
        return sum(weight * value for weight, value in present) if present else None

    def add(self, edp: NNEvaluationDataPoint, batch_size: int) -> None:
        fallback = float(edp.count if edp.count is not None else batch_size)
        if self._pending_terms:
            # The loss comes from the observed terms; the error is weighted
            # like a custom step's.
            self._pending_terms = False
            self._records += 1
            self._error.add(edp.error, fallback)
            return
        if self._pending is not None:
            loss_weight, error_weight = self._pending
            if error_weight is None:
                error_weight = fallback
        else:
            loss_weight = error_weight = fallback
        self._pending = None
        self._records += 1
        self._loss.add(edp.loss, loss_weight)
        self._error.add(edp.error, error_weight)

    def result(self) -> Optional[NNEvaluationDataPoint]:
        if not self._records:
            return None
        from .nn.params.nn_evaluation_data_point import NNEvaluationDataPoint

        loss = self._terms_loss() if self._terms else self._loss.result()
        return NNEvaluationDataPoint(loss=loss, error=self._error.result(), metrics=self._metrics.results())
