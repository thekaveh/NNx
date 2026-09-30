"""Immutable fluent experiment plans (FEAT-012).

An :class:`ExperimentPlan` holds one experiment's configuration — the
network parameters (``NNParams``), the model parameters (``NNModelParams``),
the training parameters (``NNTrainParams``), the data, the seed and the
callbacks — and compiles it to the existing ``NNModel(...)`` +
``NNModel.train(...)`` loop. Every ``with_*`` method returns a **new** plan
and leaves the source unchanged, so branching a configuration is::

    base = ExperimentPlan().with_net(net).with_model(model).with_train(train).with_data(loader)
    small, large = base.with_epochs(2), base.with_epochs(8)  # base, small and large are all intact

Three boundaries are kept apart:

- :meth:`ExperimentPlan.validate` is pure. It aggregates every
  configuration problem as a field-path :class:`Diagnostic`, and consumes
  no loader, calls no factory, builds no model, reads no weights and writes
  no directory.
- :meth:`ExperimentPlan.probe` is effectful but leaves no trace (bar a
  CUDA model's first CUDA use, below): on one
  example batch it seeds, builds a temporary model and runs a forward pass
  (and the loss when the batch has a target) under ``torch.no_grad()``. It
  then restores the ambient RNG streams, the cuDNN / deterministic-algorithm
  settings and the seeding environment variables, even when the forward
  pass fails. It never touches a callback or a run directory.
- :meth:`ExperimentPlan.fit` trains. It seeds, calls the loader and callback
  factories once, builds a fresh model and calls ``NNModel.train``, the
  same loop an imperative script calls. Each fit is a distinct **attempt**:
  its id is folded into the run's ``salt``, so repeated fits of one plan
  get distinct run ids and directories, and a plan never sets
  ``overwrite_existing``. It returns a :class:`FitResult` ``(model, run,
  metrics, attempt_id)``.

**Ownership.** The plan's parameter objects are immutable and freely shared
between branches. Everything else is **borrowed**: shared by every branch
and every fit, never copied or cloned, and never reset by the plan.

- Borrowed: loader instances passed to :meth:`~ExperimentPlan.with_data`,
  callback instances passed to :meth:`~ExperimentPlan.with_callbacks`, the
  ``extra_metrics`` callables, step functions, the objective, components
  and the provenance manifest.
- A stateful borrowed object carries its state from one fit to the next.
  ``EarlyStopping`` resets itself; ``LRMonitor`` accumulates.
- For fresh objects per fit, pass zero-argument **factories**: a callable
  as ``with_data(train=...)`` / ``val=``, or
  :meth:`~ExperimentPlan.with_callback_factories`.
- Each fit builds a new model; ``FitResult.model`` is the trained one.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import inspect
import numbers
import re
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

from .nn.enum.checkpoints import Checkpoints
from .nn.enum.devices import Devices
from .nn.enum.losses import Losses
from .nn.enum.nets import Nets
from .nn.params.nn_model_params import NNModelParams
from .nn.params.nn_optim_params import NNOptimParams
from .nn.params.nn_params import NNParams
from .nn.params.nn_scheduler_params import NNSchedulerParams
from .nn.params.nn_train_params import NNTrainParams, _validate_resume_mode

if TYPE_CHECKING:
    from .nn.nn_model import NNModel
    from .nn.params.nn_run import NNRun

__all__ = [
    "ATTEMPT_SALT_PREFIX",
    "Diagnostic",
    "ExperimentPlan",
    "FitResult",
    "PlanError",
    "PlanValidation",
    "ProbeResult",
    "SplitMetrics",
]

ATTEMPT_SALT_PREFIX = "plan-attempt:"
"""A fit's ``NNRun.salt`` is this prefix followed by its attempt id."""

_SEED_LIMIT = 2**32  # NumPy's legacy seeding accepts [0, 2**32)


class PlanError(ValueError, TypeError):
    """A plan that cannot be fitted or probed. ``diagnostics`` lists every
    problem :meth:`ExperimentPlan.validate` found.

    It is a ``ValueError`` and a ``TypeError``, the two errors the params
    classes raise, so code catching either around a configuration still
    catches it."""

    def __init__(self, diagnostics: tuple[Diagnostic, ...]) -> None:
        self.diagnostics = tuple(diagnostics)
        lines = "\n".join(f"  {d.path}: {d.message}" for d in self.diagnostics)
        super().__init__(f"the experiment plan is invalid ({len(self.diagnostics)} problem(s)):\n{lines}")

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.diagnostics,))  # pickles and copies with its diagnostics


@dataclass(frozen=True)
class Diagnostic:
    """One configuration problem at a field ``path`` (``"data.train"``,
    ``"train.monitor"``, ``"callbacks[1]"``, ...)."""

    path: str
    message: str


@dataclass(frozen=True)
class PlanValidation:
    """The result of :meth:`ExperimentPlan.validate`: every
    :class:`Diagnostic` at once, in field order."""

    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.diagnostics

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(d.path for d in self.diagnostics)

    def raise_for_errors(self) -> None:
        """Raise :class:`PlanError` listing every diagnostic, if any."""
        if self.diagnostics:
            raise PlanError(self.diagnostics)


@dataclass(frozen=True)
class SplitMetrics:
    """One split's metrics from a fit's final epoch, read from the run's own
    history — never from a second pass over a loader.

    ``available`` says whether the split was measured: a plan without
    validation data reports ``val`` as unavailable, with the ``reason``,
    rather than inventing it. ``source`` is ``"epoch"`` for a whole-epoch
    record (validation, or a training summary when the run declares metrics
    or a monitor) and ``"last_batch"`` when only the final training batch's
    record exists. ``values`` maps metric names to floats: the recorded
    ``loss`` / ``error`` / classification fields, the declared ``metrics``
    and the ``extra_metrics`` as ``extra/<name>``, skipping any that were
    not recorded (a non-finite value is kept). A declared metric named like
    a classification field (``f1``, ``accuracy``, ...) holds that name — the
    value a monitor on it compares, and the one the logging callbacks write
    last; the record's own field stays on ``run.idps``. A whole-epoch training
    summary carries the declared metrics but not the extra metrics, which
    the loop computes per batch (``run.idps`` has them).
    """

    split: str
    available: bool
    epoch: Optional[int] = None
    source: Optional[str] = None
    values: Mapping[str, float] = field(default_factory=lambda: _frozen_mapping({}), hash=False)  # a mapping
    reason: Optional[str] = None


@dataclass(frozen=True, eq=False)
class FitResult:
    """One fit: the trained ``model``, its persisted ``run`` (the
    ``NNModel.train`` return value), the final epoch's ``metrics`` per split
    (``"train"`` and ``"val"``, see :class:`SplitMetrics`) and the
    ``attempt_id`` folded into the run's salt."""

    model: NNModel
    run: NNRun
    metrics: Mapping[str, SplitMetrics]
    attempt_id: str


@dataclass(frozen=True)
class ProbeResult:
    """A probe's forward pass on one example batch: the network's raw
    output shape and dtype (in eval mode, as ``predict`` sees it, before any
    reshaping the loss applies), the batch's target shape as given, the loss
    the default training step would compute on the batch (in train mode,
    under its mixed-precision autocast, without a gradient — so the batch
    is scored as a training batch: a BatchNorm net needs more than one row)
    when the batch has a target — ``None`` for a plan
    with its own ``train_step_fn`` or objective, whose loss a probe cannot
    know — and the temporary model's parameter counts."""

    output_shape: tuple[int, ...]
    output_dtype: str
    target_shape: Optional[tuple[int, ...]]
    loss: Optional[float]
    n_parameters: int
    n_trainable: int


@dataclass(frozen=True)
class _Resume:
    run_id: Any
    checkpoint: Any
    mode: Any


class _Marker:
    """A module-level marker with a stable repr, so rendered signatures read
    ``monitor=KEEP``; it copies and pickles as the one module-level
    instance."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name

    def __reduce__(self) -> str:
        return f"_{self._name}"


# Leave this field as it is (for methods that set several fields).
_KEEP: Any = _Marker("KEEP")

# Take the training parameters' own value (their ``val_loader`` / ``data_id``)
# when the plan compiles, whatever the order of the ``with_*`` calls.
_INHERIT: Any = _Marker("INHERIT")


def _given(**changes: Any) -> dict[str, Any]:
    """The changes that were given (not ``KEEP``)."""
    return {name: value for name, value in changes.items() if value is not _KEEP}


def _iterable(value: Any) -> bool:
    """Whether ``for batch in value`` works — ``__iter__`` or the sequence
    protocol (``__getitem__``), as the training loop accepts — without
    calling either."""
    return isinstance(value, Iterable) or callable(getattr(type(value), "__getitem__", None))


def _is_factory(value: Any) -> bool:
    """A zero-argument data or callback factory: a callable that is not
    itself something to iterate."""
    return callable(value) and not _iterable(value)


def _frozen_mapping(value: Any) -> Any:
    """A read-only copy of a mapping — a borrowed dict edited later cannot
    reach the plan, and the copy still copies, deep-copies and pickles —
    or anything else as is, for validate() to report."""
    from .trainer.params import _FrozenMapping

    return _FrozenMapping(value) if isinstance(value, Mapping) else value


@dataclass(frozen=True, eq=False)
class ExperimentPlan:
    """An immutable experiment configuration that compiles to
    ``NNModel(net, model).train(train, ...)``; see the module docstring for
    the boundaries and the ownership rules.

    Build it with the ``with_*`` methods, each of which returns a new plan:

    - ``with_net(NNParams)``, ``with_model(NNModelParams)``,
      ``with_train(NNTrainParams)``;
    - ``with_epochs``, ``with_optim``, ``with_scheduler``, ``with_metrics``
      and ``with_extra_metrics``, which edit the training parameters;
    - ``with_data(train, val, identity=...)`` (what is not given is kept,
      and validation data and identity otherwise come from the training
      parameters) and ``with_seed(seed)``;
    - ``with_callbacks`` (borrowed instances) and
      ``with_callback_factories`` (called once per fit);
    - ``with_step_fns``, ``with_objective``, ``with_components`` and
      ``with_provenance``, which are borrowed, and ``with_batch_adapter``
      for a registered module's inputs;
    - ``resuming(run_id, checkpoint="last", mode=None)`` (``mode`` defaults
      to the training parameters' ``resume_mode``).

    Plan-level arguments (data, seed, callbacks, factories, steps, resume)
    are recorded as given and checked by :meth:`validate`, so every problem
    is reported at once. The ``NNParams`` / ``NNModelParams`` /
    ``NNTrainParams`` objects validate themselves when constructed, so the
    methods that edit the training parameters (``with_epochs``,
    ``with_metrics``, ...) raise :class:`PlanError` naming the field at
    once.
    """

    net: Optional[NNParams] = None
    model: Optional[NNModelParams] = None
    train: Optional[NNTrainParams] = None
    train_data: Any = None
    val_data: Any = _INHERIT
    data_identity: Any = _INHERIT
    seed: Any = None
    callbacks: tuple[Any, ...] = ()
    callback_factories: tuple[Any, ...] = ()
    train_step_fn: Any = None
    eval_step_fn: Any = None
    objective: Any = None
    components: tuple[Any, ...] = ()
    provenance: Any = None
    resume: Optional[_Resume] = None
    batch_adapter: Any = None

    # --- branching ---------------------------------------------------------------------------

    def with_net(self, net: Optional[NNParams]) -> ExperimentPlan:
        """The network parameters (``None`` for a ``ModelSpec`` model)."""
        return replace(self, net=net)

    def with_model(self, model: NNModelParams) -> ExperimentPlan:
        return replace(self, model=model)

    def __post_init__(self) -> None:
        """Freeze the collections on every construction path (the
        constructor, ``dataclasses.replace`` and the ``with_*`` methods): a
        caller's list or dict edited later cannot reach the plan."""
        for name in ("callbacks", "callback_factories", "components"):
            value = getattr(self, name)
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
                object.__setattr__(self, name, tuple(value))  # generators too, read once here
        train = self.train
        if isinstance(train, NNTrainParams) and isinstance(train.extra_metrics, Mapping):
            from .trainer.params import _FrozenMapping

            if not isinstance(train.extra_metrics, _FrozenMapping):
                object.__setattr__(self, "train", replace(train, extra_metrics=_frozen_mapping(train.extra_metrics)))

    def with_train(self, train: NNTrainParams) -> ExperimentPlan:
        """The training parameters. Loaders inside them are borrowed, and
        :meth:`with_data` overrides them. ``extra_metrics`` is copied into a
        read-only mapping."""
        return replace(self, train=train)

    def _edit_train(self, what: str, **changes: Any) -> ExperimentPlan:
        """Rebuild the training parameters (which validate themselves); a
        bad value raises :class:`PlanError` naming the field."""
        if not isinstance(self.train, NNTrainParams):
            raise PlanError((Diagnostic("train", f"set NNTrainParams with with_train(...) before {what}"),))
        changes = _given(**changes)
        try:
            train = replace(self.train, **changes)
        except (TypeError, ValueError) as exc:
            # The changed field when the error is about it; else the params as a whole
            # (NNTrainParams may reject a kept field that the change contradicts).
            named = [name for name in changes if re.search(rf"\b{name}\b", str(exc))]
            path = f"train.{named[0]}" if len(named) == 1 else "train"
            raise PlanError((Diagnostic(path, str(exc)),)) from exc
        return replace(self, train=train)

    def with_epochs(self, n_epochs: int) -> ExperimentPlan:
        return self._edit_train("with_epochs", n_epochs=n_epochs)

    def with_optim(self, optim: Any) -> ExperimentPlan:
        return self._edit_train("with_optim", optim=optim)

    def with_scheduler(self, scheduler: Any) -> ExperimentPlan:
        return self._edit_train("with_scheduler", scheduler=scheduler)

    def with_metrics(self, metrics: Iterable[Any] = _KEEP, *, monitor: Any = _KEEP) -> ExperimentPlan:
        """Declared metrics (``MetricSpec``) and the monitor
        (``MonitorSpec``), as on ``NNTrainParams``; the one not given is
        kept (pass ``monitor=None`` to drop a monitor)."""
        if metrics is not _KEEP:
            if isinstance(metrics, (str, bytes)) or not isinstance(metrics, Iterable):
                raise PlanError((Diagnostic("train.metrics", f"must be a sequence of MetricSpec, got {metrics!r}"),))
            metrics = tuple(metrics)
        return self._edit_train("with_metrics", metrics=metrics, monitor=monitor)

    def with_extra_metrics(self, extra_metrics: Optional[Mapping[str, Callable[..., float]]]) -> ExperimentPlan:
        """Borrowed ``name -> callable(y_true, y_pred)`` metrics, copied into
        a read-only mapping."""
        return self._edit_train("with_extra_metrics", extra_metrics=extra_metrics)  # frozen by __post_init__

    def with_data(self, train: Any = _KEEP, val: Any = _KEEP, *, identity: Any = _KEEP) -> ExperimentPlan:
        """The training data and, optionally, validation data and identity.

        Each source is a re-iterable of batches (a ``DataLoader`` or a list,
        which is borrowed) or a zero-argument factory that returns one. A
        factory is called once per fit, never by :meth:`validate`.
        ``identity`` is the caller-supplied data identity; it becomes
        ``NNTrainParams.data_id``, which is part of the run id.

        As with the other ``with_*`` methods, what is not given is kept (so
        ``with_data(val=...)`` changes only the validation source): the
        current training and validation sources and identity — the plan's
        own, else the training parameters' ``val_loader`` / ``data_id``,
        resolved when the plan compiles, whatever the order of the calls.
        ``train=None`` keeps the training parameters' loader. Pass
        ``val=None`` to run train-only (a plan without validation data
        invents none) and ``identity=`` for new data — or ``identity=None``
        to record none. Without ``with_data`` the loaders and ``data_id``
        inside the training parameters are used (the loaders may be
        factories too).
        """
        return replace(self, **_given(train_data=train, val_data=val, data_identity=identity))

    def with_seed(self, seed: Optional[int]) -> ExperimentPlan:
        """Seed every RNG before the factories run and the model is built,
        and again (as ``NNTrainParams.seed``) when training starts. The
        plan's seed overrides the training parameters' own ``seed``, so
        branches can differ by seed; ``None`` falls back to it."""
        return replace(self, seed=seed)

    def with_callbacks(self, *callbacks: Any) -> ExperimentPlan:
        """Borrowed callback instances (or legacy ``fn(idps)`` callables),
        shared by every fit; they replace any earlier ones. A callable whose
        one parameter is required is a legacy callback, called with the
        history each epoch. One that can be called with no arguments reads
        as a factory and is reported: pass callback-making functions to
        :meth:`with_callback_factories`."""
        return replace(self, callbacks=tuple(callbacks))

    def with_callback_factories(self, *factories: Callable[[], Any]) -> ExperimentPlan:
        """Zero-argument callables, each called once per fit to make a fresh
        callback. Their callbacks run after the borrowed ones, in order."""
        return replace(self, callback_factories=tuple(factories))

    def with_step_fns(self, train_step_fn: Any = _KEEP, eval_step_fn: Any = _KEEP) -> ExperimentPlan:
        """Borrowed step functions; the one not given is kept (pass
        ``None`` to drop one)."""
        return replace(self, **_given(train_step_fn=train_step_fn, eval_step_fn=eval_step_fn))

    def with_objective(self, objective: Any) -> ExperimentPlan:
        return replace(self, objective=objective)

    def with_components(self, *components: Any) -> ExperimentPlan:
        return replace(self, components=tuple(components))

    def with_batch_adapter(self, adapter: Any) -> ExperimentPlan:
        """How a registered ``ModelSpec`` module reads a batch
        (``nnx.models.PositionalInputs`` / ``KeywordInputs``), passed to
        ``NNModel(..., batch_adapter=...)`` on every build."""
        return replace(self, batch_adapter=adapter)

    def with_provenance(self, manifest: Any) -> ExperimentPlan:
        return replace(self, provenance=manifest)

    def resuming(self, run_id: str, *, checkpoint: str = "last", mode: Optional[str] = None) -> ExperimentPlan:
        """Warm-resume from the run ``run_id``'s ``checkpoint``. The source
        run is recorded as the new run's parent lineage and is never
        overwritten: the resumed fit writes its own run directory. ``mode``
        (``"auto"`` / ``"stateful"`` / ``"weights_only"``) defaults to the
        training parameters' own ``resume_mode``."""
        if isinstance(checkpoint, Checkpoints):
            checkpoint = checkpoint.value  # the tag string NNTrainParams records as lineage
        return replace(self, resume=_Resume(run_id, checkpoint, mode))

    def without_resume(self) -> ExperimentPlan:
        return replace(self, resume=None)

    # --- validation (pure) -------------------------------------------------------------------

    def validate(self) -> PlanValidation:
        """Every configuration problem as a field-path :class:`Diagnostic`.

        Pure: it consumes no loader (iterables are never iterated), calls
        no data or callback factory, builds no model, reads no weights and
        writes no directory. Declared metrics and the monitor are resolved
        against their registries, as ``NNModel.train`` does before
        reserving a run. It constructs the built-in loss module, as
        ``NNModel`` does first, and resolves each borrowed callback's monitor
        on a shallow copy; with ``nnx.streaming.streaming_eval_step`` it also
        builds each declared metric's accumulator to check that it is
        bounded.
        """
        found: list[Diagnostic] = []

        def report(path: str, message: str) -> None:
            found.append(Diagnostic(path, message))

        self._check_model(report)
        train = self._check_train(report)
        has_val = self._check_data(report, train)
        self._check_seed(report, train)
        self._check_callbacks(report)
        self._check_steps(report)
        self._check_resume(report, train)
        if train is not None:
            self._check_monitoring(report, train, has_val)
            self._check_eval_step(report, train, has_val)
        return PlanValidation(tuple(found))

    def _streaming_validation(self, has_val: bool) -> bool:
        """Whether the fit validates with ``nnx.streaming.streaming_eval_step``
        (checked only with validation data, as ``NNModel.train`` checks it)."""
        from .nn.nn_model import _is_streaming_eval_step

        return has_val and _is_streaming_eval_step(self.eval_step_fn)

    def _check_eval_step(self, report: Callable[[str, str], None], train: NNTrainParams, has_val: bool) -> None:
        """What the streaming validation step reports it cannot compute, as
        ``NNModel.train`` checks before reserving a run."""
        if self._streaming_validation(has_val):
            from .streaming import _streaming_problems

            for path, message in _streaming_problems(train):
                report(path, message)

    def _check_model(self, report: Callable[[str, str], None]) -> None:
        from .models import MissingModelFactoryError
        from .nn.nn_model import _resolve_net_descriptor

        paths: list[str] = []

        def note(path: str, message: str) -> None:
            paths.append(path)
            report(path, message)

        self._check_model_parts(note)
        if isinstance(self.model, NNModelParams) and not {"net", "model.net"} & set(paths):
            try:  # NNModel's own descriptor rule, as a safety net for the checks above
                _resolve_net_descriptor(self.net, self.model, None)
            except (TypeError, ValueError, MissingModelFactoryError) as exc:
                report("model.net", str(exc))

    def _check_model_parts(self, report: Callable[[str, str], None]) -> None:
        from .models import BatchAdapter, MissingModelFactoryError, ModelSpec, resolve_model_factory
        from .tasks import task_adapter

        if self.model is None:
            report("model", "required: set NNModelParams with with_model(...)")
        elif not isinstance(self.model, NNModelParams):
            report("model", f"must be NNModelParams, got {type(self.model).__name__}")
        elif not self._model_usable:
            report(
                "model",
                "NNModelParams needs a device and a loss (a Devices and a Losses member), got "
                f"device={self.model.device!r}, loss={self.model.loss!r}",
            )
        else:
            try:
                self.model.loss()  # NNModel builds it first, whatever else the plan declares
            except Exception as exc:  # building the loss runs its own code: any failure is the model's diagnostic
                report("model.loss", str(exc))
        if self.batch_adapter is not None and not isinstance(self.batch_adapter, BatchAdapter):
            report("batch_adapter", f"must be an nnx.models.BatchAdapter, got {type(self.batch_adapter).__name__}")
        net_typed = self.net is None or isinstance(self.net, NNParams)
        if not net_typed:
            report("net", f"must be NNParams, got {type(self.net).__name__}")
        if not isinstance(self.model, NNModelParams):
            return
        descriptor = self.model.net
        if descriptor is None:
            report("model.net", "required: a built-in Nets member or a ModelSpec (nnx.models)")
        elif isinstance(descriptor, Nets):
            if self.net is None:
                report("net", f"{descriptor} needs NNParams: set them with with_net(...)")
        elif isinstance(descriptor, ModelSpec):
            if self.net is not None and net_typed:
                report("net", "a ModelSpec builds its own network from its config; pass with_net(None)")
            try:
                resolve_model_factory(descriptor)  # a registry lookup: nothing is imported or built
            except (MissingModelFactoryError, TypeError) as exc:
                report("model.net", str(exc))
        else:
            report(
                "model.net",
                "a plan builds a fresh model for every fit, so it needs a Nets member or a registered ModelSpec "
                f"(nnx.models), not a {type(descriptor).__name__} wrapping an existing module",
            )
            return
        if self.model.task is not None and net_typed and self._model_usable:
            try:  # NNModel's own check, run before any module is built
                task_adapter(self.model.task).check_model(
                    net=descriptor, loss=self.model.loss, output_dim=getattr(self.net, "output_dim", None)
                )
            except (TypeError, ValueError) as exc:
                report("model.task", str(exc))

    def _check_train(self, report: Callable[[str, str], None]) -> Optional[NNTrainParams]:
        if self.train is None:
            report("train", "required: set NNTrainParams with with_train(...)")
            return None
        if not isinstance(self.train, NNTrainParams):
            report("train", f"must be NNTrainParams, got {type(self.train).__name__}")
            return None
        train = self.train
        from .optimizers import NNOptimFactoryParams

        if not isinstance(train.optim, (NNOptimParams, NNOptimFactoryParams)) or not train.optim.is_valid():
            report(
                "train.optim",
                f"is not a valid optimizer configuration (NNOptimParams or NNOptimFactoryParams): {train.optim!r}",
            )
        if not isinstance(train.scheduler, NNSchedulerParams):
            report("train.scheduler", f"must be NNSchedulerParams, got {type(train.scheduler).__name__}")
        if train.overwrite_existing:
            report(
                "train.overwrite_existing",
                "a plan never overwrites a run: each fit is a distinct attempt with its own run directory",
            )
        if train.extra_metrics is not None and not isinstance(train.extra_metrics, Mapping):
            report("train.extra_metrics", f"must map names to callable(y_true, y_pred), got {train.extra_metrics!r}")
            return train
        for name, metric in (train.extra_metrics or {}).items():
            if not callable(metric):
                report(f"train.extra_metrics[{name!r}]", f"must be callable(y_true, y_pred), got {metric!r}")
        return train

    def _check_data(self, report: Callable[[str, str], None], train: Optional[NNTrainParams]) -> bool:
        """Check the data sources without iterating them; return whether
        the plan has validation data."""
        sources = tuple(zip(("data.train", "data.val"), self._sources(train), strict=True))
        for path, source in sources:
            if source is None:
                if path == "data.train":
                    report(path, "required: pass the training data with with_data(...)")
                continue
            if _is_factory(source):
                if not _is_zero_argument(source):
                    report(path, f"a data factory is called with no arguments; {source!r} needs some")
                continue
            problem = _loader_problem(source)
            if problem is not None:
                report(path, problem)
        identity = self._identity(train)
        if identity is not None and (not isinstance(identity, str) or not identity.strip()):
            report("data.identity", f"must be a non-empty string, got {identity!r}")
        return sources[1][1] is not None

    def _check_seed(self, report: Callable[[str, str], None], train: Optional[NNTrainParams]) -> None:
        """The seed that ``set_seed`` will receive — the plan's, else the
        training parameters' — must be one it accepts."""
        train_seed = train.seed if train is not None else None
        path, seed = ("seed", self.seed) if self.seed is not None else ("train.seed", train_seed)
        if seed is None:
            return
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or not 0 <= int(seed) < _SEED_LIMIT:
            report(path, f"must be an integer in [0, 2**32), got {seed!r}")

    def _check_callbacks(self, report: Callable[[str, str], None]) -> None:
        for name in ("callbacks", "callback_factories", "components"):
            if not isinstance(getattr(self, name), tuple):
                report(name, f"must be a sequence, got {getattr(self, name)!r}")

        def items(name: str) -> tuple[Any, ...]:
            value = getattr(self, name)
            return value if isinstance(value, tuple) else ()

        def spread(name: str, value: Any) -> str:
            # with_callbacks([a, b]) holds one list: the with_* methods take the items as arguments
            return f" (pass the items as separate arguments: with_{name}(*items))" if isinstance(value, list) else ""

        for index, callback in enumerate(items("callbacks")):
            if not _is_callback(callback):
                if callable(callback) and _takes_no_arguments(callback):
                    hint = (
                        " (it can be called with no arguments, so it reads as a factory: pass it with "
                        "with_callback_factories, or make a legacy callback's history parameter required)"
                    )
                elif callable(callback):
                    hint = " (pass a class or other factory with with_callback_factories)"
                else:
                    hint = spread("callbacks", callback)
                report(
                    f"callbacks[{index}]", f"must be a Callback instance or a callable(idps), got {callback!r}{hint}"
                )
        for index, factory in enumerate(items("callback_factories")):
            if not _is_zero_argument(factory):
                report(
                    f"callback_factories[{index}]",
                    f"must be a zero-argument callable, got {factory!r}{spread('callback_factories', factory)}",
                )
        from .components import StatefulComponent

        for index, component in enumerate(items("components")):
            if not isinstance(component, StatefulComponent):
                report(
                    f"components[{index}]",
                    f"must implement nnx.StatefulComponent, got {type(component).__name__}"
                    f"{spread('components', component)}",
                )

    def _check_steps(self, report: Callable[[str, str], None]) -> None:
        from .nn.nn_model import _check_provenance

        for path, value in (
            ("train_step_fn", self.train_step_fn),
            ("eval_step_fn", self.eval_step_fn),
            ("objective", self.objective),
        ):
            if value is not None and not callable(value):
                report(path, f"must be callable, got {value!r}")
        if self.objective is not None and self.train_step_fn is not None:
            report("objective", "pass train_step_fn or objective, not both: one owner per optimizer update")
        try:
            _check_provenance(self.provenance)  # train()'s own rule
        except TypeError as exc:
            report("provenance", str(exc))

    def _check_resume(self, report: Callable[[str, str], None], train: Optional[NNTrainParams]) -> None:
        resume = self.resume
        if resume is None:
            return
        if not isinstance(resume.run_id, str) or not resume.run_id.strip():
            report("resume.run_id", f"must be a non-empty run id, got {resume.run_id!r}")
        from .nn.nn_model import _resume_checkpoint_type

        try:
            _resume_checkpoint_type(resume.checkpoint)  # the loop's own rule: a tag or a ModelCheckpoint stem
        except ValueError as exc:
            report("resume.checkpoint", str(exc))
        if resume.mode is not None:
            try:
                _validate_resume_mode(resume.mode, "resuming()")
            except ValueError as exc:
                report("resume.mode", str(exc))
        if train is not None and train.parent_run_id is not None:
            report(
                "resume",
                f"conflicts with train.parent_run_id={train.parent_run_id!r}: a resume records its own source "
                "as the parent; set one",
            )
        if train is not None and train.resume_from_run_id is not None:
            report("resume", "train already resumes from a run; set the resume once, on the plan")

    def _check_monitoring(self, report: Callable[[str, str], None], train: NNTrainParams, has_val: bool) -> None:
        """The metric and monitor rules ``NNModel.train`` applies before
        reserving a run — for the training monitor and every borrowed
        callback's declared monitor, bound on a copy so the callback itself
        is never bound (factory-made callbacks are checked when the fit
        starts). With a broken metric declared, a monitor that fails to
        resolve is not reported: it would only restate that metric's
        problem."""
        from .monitors import MonitorSpec
        from .nn.nn_model import _monitor_problem

        broken = False
        for index, spec in enumerate(train.metrics):
            try:
                spec.check()
            except (KeyError, TypeError, ValueError) as exc:
                report(f"train.metrics[{index}]", str(exc))
                broken = True
        default_step = self._default_step
        # NNx derives the declared metrics' inputs itself on its default steps and
        # in its own eval step, which train() checks only when there is validation data.
        nnx_eval = self.eval_step_fn is None or self._streaming_validation(has_val)
        if train.metrics and not broken and (default_step or nnx_eval):
            where = (
                "the default training step"
                if default_step
                else "streaming_eval_step()"
                if self._streaming_validation(has_val)
                else "evaluate()"
            )
            problem = self._metric_input_problem(train, where=where)
            if problem is not None:
                report(*problem)
        if isinstance(train.monitor, MonitorSpec):
            try:
                resolved = train.monitor.resolve(train.metrics, owner="train.monitor")
            except (KeyError, TypeError, ValueError) as exc:
                if not broken:
                    report("train.monitor", str(exc))
            else:
                problem = _monitor_problem(resolved, has_val_loader=has_val, default_train_step=default_step)
                if problem is not None:
                    report("train.monitor", problem)
        borrowed = self.callbacks if isinstance(self.callbacks, tuple) else ()
        self._check_callback_monitors(report, train, "callbacks", borrowed, has_val=has_val, broken=broken)

    def _check_callback_monitors(
        self,
        report: Callable[[str, str], None],
        train: NNTrainParams,
        where: str,
        callbacks: Iterable[Any],
        *,
        has_val: bool,
        broken: bool = False,
    ) -> None:
        """Each callback's declared monitor, bound as ``train()`` binds it
        but on a shallow copy, so the callback itself is never bound — for
        the borrowed callbacks in :meth:`validate` and the factory-made ones
        when a fit starts."""
        from .monitors import MonitorSpec
        from .nn.nn_model import _monitor_problem

        for index, callback in enumerate(callbacks):
            if not _is_callback(callback) or not callable(getattr(callback, "_bind_metrics", None)):
                continue
            path = f"{where}[{index}].monitor"
            try:
                probe = copy.copy(callback)
            except Exception:  # an uncopyable callback: train() checks its monitor instead
                continue
            try:
                bound = probe._bind_metrics(train.metrics)  # train()'s own protocol
            except (KeyError, TypeError, ValueError) as exc:
                if not broken:
                    report(path, str(exc))
                continue
            except Exception:  # an error of the copy, not a monitor rule: train() binds the callback itself
                continue
            if isinstance(bound, MonitorSpec):
                problem = _monitor_problem(bound, has_val_loader=has_val, default_train_step=self._default_step)
                if problem is not None:
                    report(path, problem)

    def _metric_input_problem(self, train: NNTrainParams, *, where: str) -> Optional[tuple[str, str]]:
        """``train()``'s check that the model can provide every declared
        metric's input, run on the model's parameters alone: the loss, the
        task adapter and the output width — no model is built."""
        from .nn.nn_model import _check_metric_inputs, _metric_context_of
        from .tasks import task_adapter

        if not self._model_usable:
            return None  # the model's own diagnostic says why
        assert isinstance(self.model, NNModelParams)
        try:
            loss_fn = self.model.loss()
            adapter = task_adapter(self.model.task) if self.model.task is not None else None
        except Exception:
            return None  # the model's own diagnostic (model.loss / model.task) says why
        try:
            domain, _, _, n_classes = _metric_context_of(loss_fn, adapter, self.net)
            _check_metric_inputs(train.metrics, domain, where=where, n_classes=n_classes)
        except (KeyError, TypeError, ValueError) as exc:
            return "train.metrics", str(exc)
        return None

    # --- compiling ---------------------------------------------------------------------------

    @property
    def _model_usable(self) -> bool:
        """Whether the model parameters can be read further (a ``Devices``
        device and a ``Losses`` loss, which ``NNModel`` calls), so checks
        built on them do not pile onto the model's own diagnostic. The net
        is checked on its own path, so ``NNModelParams.is_valid`` (which
        also needs it) is not the test here."""
        return (
            isinstance(self.model, NNModelParams)
            and isinstance(self.model.device, Devices)
            and isinstance(self.model.loss, Losses)
        )

    @property
    def _default_step(self) -> bool:
        """Whether NNx's default training step runs (no step function or
        objective of the plan's own)."""
        return self.train_step_fn is None and self.objective is None

    def _seed(self) -> Optional[int]:
        if self.seed is not None:
            return int(self.seed)
        return self.train.seed if isinstance(self.train, NNTrainParams) else None

    def _apply_seed(self, *, scoped: bool = False) -> None:
        seed = self._seed()
        if seed is not None:
            from .seeding import _set_seed

            _set_seed(seed, scoped=scoped)

    def _build_model(self) -> NNModel:
        from .nn.nn_model import NNModel

        return NNModel(net_params=self.net, params=self.model, batch_adapter=self.batch_adapter)

    def _sources(self, train: Optional[NNTrainParams]) -> tuple[Any, Any]:
        """The train and val data sources: each the plan's own
        (``with_data``), else the loader inside the training parameters."""
        train_source = self.train_data if self.train_data is not None else getattr(train, "train_loader", None)
        val_source = getattr(train, "val_loader", None) if self.val_data is _INHERIT else self.val_data
        return train_source, val_source

    def _identity(self, train: Optional[NNTrainParams]) -> Any:
        """The data identity: the plan's own, else the training parameters'."""
        return getattr(train, "data_id", None) if self.data_identity is _INHERIT else self.data_identity

    def _compile_train(self, train_loader: Any, val_loader: Any) -> NNTrainParams:
        assert isinstance(self.train, NNTrainParams)
        changes: dict[str, Any] = {"seed": self._seed(), "train_loader": train_loader, "val_loader": val_loader}
        changes["data_id"] = self._identity(self.train)
        if self.resume is not None:
            changes.update(resume_from_run_id=self.resume.run_id, resume_from_checkpoint=self.resume.checkpoint)
            if self.resume.mode is not None:  # else the training parameters' own resume_mode
                changes["resume_mode"] = self.resume.mode
        return replace(self.train, **changes)

    # --- probing (effectful, restored) -------------------------------------------------------

    def probe(self, example_batch: Any) -> ProbeResult:
        """Build a temporary model and run one forward pass on
        ``example_batch`` under ``torch.no_grad()`` in eval mode, plus the
        loss when the batch has a target (``None`` for an all-masked task
        batch, which the default step records no loss for either).

        Effectful but restored: the plan's seed is applied first, and
        afterwards the ambient Python / NumPy / torch RNG streams, the
        cuDNN ``deterministic`` / ``benchmark`` flags, the
        deterministic-algorithms setting and the seeding environment
        variables are put back, whether or not the forward pass succeeds.
        The model is discarded. No callback, factory or loader is touched,
        and nothing is written. A CPU model's probe seeds no CUDA stream, so
        nothing waits for CUDA's first use. A CUDA model's probe starts CUDA
        first when it is not yet in use (running any seed queued for its
        first use), so its streams are restored like the others. A
        registered factory's module is built as ``NNModel`` builds it
        (:func:`nnx.models.build_module`), which saves and restores every
        RNG stream around the factory call — on a GPU host that reads, and
        so initializes, CUDA even for a CPU model; its streams are then put
        back as the build left them.

        A probe needs only the model, the network and the seed: when those
        are invalid it raises :class:`PlanError` first (the data, callbacks
        and training parameters are :meth:`validate`'s and :meth:`fit`'s
        concern); a failing forward pass raises its own error — including a
        batch the training step cannot take either, such as a single example
        for a ``BatchNorm`` net in train mode.
        """
        self._probe_validation().raise_for_errors()
        from .nn.nn_model import _step_loss_terms
        from .seeding import _seed_scope

        assert isinstance(self.model, NNModelParams)
        if self.model.device().type == "cuda" and torch.cuda.is_available() and not torch.cuda.is_initialized():
            torch.cuda.init()  # a CUDA model starts CUDA anyway: start it first, so its streams are captured too
        with _seed_scope() as cuda_started:
            # Only the streams the scope restores: an idle CUDA context is not
            # seeded (that would queue the seed for its first use, after the probe).
            self._apply_seed(scoped=True)
            model = self._build_model()
            cuda_started()  # CUDA the build started: its streams, as the build left them, are restored too
            model._check_task_preflight()  # as train() does, before any batch is read
            with torch.no_grad():
                model.net.eval()  # the output as predict() sees it
                output, target = model._inference_forward(example_batch)
                # With a target and the default step, the loss that step computes on this batch: in
                # train mode (dropout, batch statistics), with its reshaping, masking and task rules. A
                # custom step or objective defines its own loss, which a probe cannot know.
                loss = None
                if target is not None and self._default_step:
                    model.net.train()
                    amp = model._build_grad_scaler() is not None and model.device.type == "cuda"  # as train()
                    with torch.amp.autocast(device_type="cuda") if amp else contextlib.nullcontext():
                        terms = _step_loss_terms(model, example_batch, None, 1)
                    # An all-masked task batch has no loss of its own, as the default step records it.
                    loss = (
                        None if terms.valid is not None and terms.normalization_weight == 0 else float(terms.train_loss)
                    )
            parameters = list(model.net.parameters())
            target_shape = getattr(target, "shape", None)  # None, too, for a target that is not an array
            return ProbeResult(
                output_shape=tuple(output.shape),
                output_dtype=str(output.dtype).removeprefix("torch."),
                target_shape=None if target_shape is None else tuple(target_shape),
                loss=loss,
                n_parameters=sum(p.numel() for p in parameters),
                n_trainable=sum(p.numel() for p in parameters if p.requires_grad),
            )

    def _probe_validation(self) -> PlanValidation:
        """The diagnostics that stop a probe: the model, the network and the
        seed (a probe needs no data, callbacks or training loop)."""
        found: list[Diagnostic] = []

        def report(path: str, message: str) -> None:
            found.append(Diagnostic(path, message))

        self._check_model(report)
        self._check_seed(report, self.train if isinstance(self.train, NNTrainParams) else None)
        return PlanValidation(tuple(found))

    # --- fitting ------------------------------------------------------------------------------

    def fit(self, *, attempt: Optional[str] = None) -> FitResult:
        """Train one attempt and return its :class:`FitResult`.

        The steps are the ones an equally seeded imperative script takes:
        validate (an invalid plan raises :class:`PlanError` before
        anything runs); seed; call each data factory and callback factory
        exactly once (what they return is checked next, and a
        :class:`PlanError` then leaves the seed applied, as the script's
        ``set_seed`` would); build ``NNModel(net, model)``; and call
        ``model.train(train, callbacks, ...)``. The ``train`` given to
        ``model.train`` carries the plan's seed, data, data identity and
        resume. The run's ``salt`` is ``"plan-attempt:<attempt>"``, where
        ``attempt`` defaults to a fresh random id, so every fit gets its own
        run id and directory. ``overwrite_existing`` is never set; reusing
        an ``attempt`` id for the same configuration raises
        ``FileExistsError`` from ``NNModel.train``.
        """
        found = self.validate().diagnostics
        if attempt is not None and (not isinstance(attempt, str) or not attempt.strip()):
            found += (Diagnostic("attempt", f"must be a non-empty string, got {attempt!r}"),)
        PlanValidation(found).raise_for_errors()
        attempt_id = uuid.uuid4().hex if attempt is None else attempt
        self._apply_seed()
        sources = self._sources(self.train)
        loaders = [source() if _is_factory(source) else source for source in sources]
        callbacks = [*self.callbacks, *(factory() for factory in self.callback_factories)]
        made = callbacks[len(self.callbacks) :]
        _check_made(sources, loaders, made)
        made_found: list[Diagnostic] = []
        assert isinstance(self.train, NNTrainParams)
        self._check_callback_monitors(
            lambda path, message: made_found.append(Diagnostic(path, message)),
            self.train,
            "callback_factories",
            made,
            has_val=loaders[1] is not None,
        )
        PlanValidation(tuple(made_found)).raise_for_errors()
        params = self._compile_train(*loaders)
        model = self._build_model()
        run = model.train(
            params,
            callbacks=callbacks or None,
            train_step_fn=self.train_step_fn,
            eval_step_fn=self.eval_step_fn,
            salt=ATTEMPT_SALT_PREFIX + attempt_id,
            components=list(self.components) or None,
            objective=self.objective,
            provenance=self.provenance,
        )
        return FitResult(
            model=model,
            run=run,
            metrics=_fit_metrics(run, has_val=params.val_loader is not None),
            attempt_id=attempt_id,
        )


# --- helpers ------------------------------------------------------------------------------------


def _loader_problem(source: Any, *, made: bool = False) -> Optional[str]:
    """Why ``source`` cannot feed every epoch, or ``None``; never iterates
    it. ``made``: ``source`` is what a factory returned."""
    alternatives = "a DataLoader or a list" if made else "a DataLoader, a list, or a zero-argument factory"
    subject = "the factory returned" if made else "got"
    if isinstance(source, (str, bytes)) or not _iterable(source):
        return f"needs a re-iterable of batches ({alternatives}); {subject} {source!r}"
    if isinstance(source, Iterator):
        return f"{subject} a one-shot iterator, which cannot be replayed every epoch; use {alternatives}"
    if isinstance(source, Mapping):
        return f"{subject} a {type(source).__name__}, which iterates its keys, not batches; use {alternatives}"
    if isinstance(source, torch.utils.data.Dataset) and not isinstance(source, torch.utils.data.IterableDataset):
        return f"{subject} a {type(source).__name__}, which yields single samples, not batches; wrap it in a DataLoader"
    if isinstance(source, (torch.Tensor, np.ndarray)):
        return (
            f"{subject} a {type(source).__name__}, which yields single rows, not batches; wrap the data in a "
            "TensorDataset and a DataLoader"
        )
    return None


def _binds(value: Any, *args: Any, unknown: bool) -> bool:
    """Whether callable ``value`` can be called with ``args``; ``unknown``
    answers for one without an introspectable signature (some builtins)."""
    try:
        inspect.signature(value).bind(*args)
    except TypeError:
        return False
    except ValueError:
        return unknown
    return True


def _is_zero_argument(value: Any) -> bool:
    """A callable that can be called with no arguments (a factory); one
    without an introspectable signature is accepted as given."""
    return callable(value) and _binds(value, unknown=True)


def _takes_no_arguments(value: Any) -> bool:
    """Whether callable ``value`` is known to bind no arguments (every
    parameter optional)."""
    return _binds(value, unknown=False)


def _is_callback(value: Any) -> bool:
    """A callback instance or a legacy ``fn(idps)`` callable — never a
    class or a callable that can be called with no arguments (a factory),
    which would be called with the history instead of building a
    callback."""
    from .nn.callbacks import Callback

    if isinstance(value, Callback):
        return True
    if not callable(value) or isinstance(value, type):
        return False
    if isinstance(value, functools.partial) and isinstance(value.func, type):
        return False  # functools.partial(EarlyStopping, ...) builds a callback: a factory
    return _binds(value, object(), unknown=True) and not _takes_no_arguments(value)  # it needs the history


def _check_made(sources: tuple[Any, Any], loaders: list[Any], callbacks: list[Any]) -> None:
    """What the factories returned, before any model is built or run
    reserved: loaders that can feed every epoch (a val factory must return
    one too — ``None`` is only "no validation" when no val source was
    given), and callbacks."""
    found: list[Diagnostic] = []
    for path, source, loader in zip(("data.train", "data.val"), sources, loaders, strict=True):
        problem = None if source is None else _loader_problem(loader, made=_is_factory(source))
        if problem is not None:
            found.append(Diagnostic(path, problem))
    for index, callback in enumerate(callbacks):
        if not _is_callback(callback):
            found.append(Diagnostic(f"callback_factories[{index}]", f"returned {callback!r}, not a Callback"))
    PlanValidation(tuple(found)).raise_for_errors()


def _values(record: Any) -> Mapping[str, float]:
    """A record's recorded values under the names the callbacks log them:
    the standard fields, a task record's metrics and ``extra/<name>`` for
    the extra metrics (so an extra metric can never shadow a recorded
    field). Non-finite values are kept, so a diverged loss shows as
    ``nan`` rather than vanishing."""
    from .nn.callbacks import _edp_metric_iter

    values: dict[str, float] = {}
    for name, value in _edp_metric_iter(record):
        if isinstance(value, bool):
            continue
        if isinstance(value, numbers.Real):
            values[name] = float(value)
        elif callable(getattr(value, "item", None)) and getattr(value, "ndim", None) == 0:
            item = value.item()  # a 0-d tensor / array a custom step recorded without .item()
            if isinstance(item, numbers.Real) and not isinstance(item, bool):
                values[name] = float(item)
    return _frozen_mapping(values)


def _fit_metrics(run: NNRun, *, has_val: bool) -> Mapping[str, SplitMetrics]:
    """The final epoch's train and val metrics from the run's history."""
    no_val = "the plan has no validation data"
    if not run.idps:
        missing = "the run recorded no epochs"
        return _frozen_mapping(
            {
                "train": SplitMetrics("train", available=False, reason=missing),
                "val": SplitMetrics("val", available=False, reason=missing if has_val else no_val),
            }
        )
    last = run.idps[-1]
    epoch = last.epoch_idx
    train_values = _values(last.monitored_train_edp())  # the record monitors compare
    train = SplitMetrics(
        "train",
        available=bool(train_values),
        epoch=epoch,
        source="epoch" if last.train_summary is not None else "last_batch",
        values=train_values,
        reason=None if train_values else "the final training record holds no values (a custom step recorded none)",
    )
    if not has_val:
        val = SplitMetrics("val", available=False, reason=no_val)
    elif last.val_edp is None:
        val = SplitMetrics("val", available=False, epoch=epoch, reason="the final epoch recorded no validation")
    else:
        val_values = _values(last.val_edp)
        val = SplitMetrics(
            "val",
            available=bool(val_values),
            epoch=epoch,
            source="epoch",
            values=val_values,
            reason=None if val_values else "the final validation record holds no values",
        )
    return _frozen_mapping({"train": train, "val": val})
