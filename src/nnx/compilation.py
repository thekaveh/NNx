"""Opt-in ``torch.compile`` for the built-in FP32 training forward (FEAT-029).

``model.train(params, compile=CompileSpec())`` compiles the forward pass the
built-in step runs (and validation inside the fit), and nothing else:

- **The canonical module stays canonical.** The compiled wrapper is a
  per-run object beside ``model.net``, never assigned to it: optimizers,
  ``summary``, checkpoints (``net_state`` keys stay free of ``_orig_mod.``),
  Hub and ONNX export and weight tying all read the eager module. Every
  ``train()`` call (a resume included) builds a fresh wrapper, and the
  wrapper is dropped when the fit ends — prediction, evaluation and
  generation after training are eager (a ``predict`` / ``evaluate`` a
  callback runs *during* the fit goes through the wrapper).
- **Scope.** Built-in nets (not graph nets or custom modules), full
  precision (FP32: no autocast), the default train step (no
  ``train_step_fn`` or ``objective``), and no topology-changing callback
  (``QATLifecycleCallback`` and anything else declaring
  ``checkpoint_transforms``):
  each is refused before any run is reserved. Distributed (DDP) and
  quantized training are not covered, and **no speedup is guaranteed** —
  measure with :mod:`nnx.benchmarking`.
- **Failure policy.** ``on_failure="error"`` (default) raises
  :class:`CompileFailed` when a compiled forward fails to compile;
  ``on_failure="eager"`` re-runs *that forward* eagerly — from the RNG
  state and buffers it started with — and trains the rest of the run
  eager, recording the restart. A failure that the eager re-run also hits
  is the forward's own error: it propagates and the record is unchanged.
  Hitting dynamo's recompile limit is a failure too (where torch can
  report it), never a silent eager fallback; the limit is per code object
  and process-wide, so each fit gets it afresh on top of the entries
  earlier fits of the net class hold (``torch._dynamo.reset()`` clears
  them). Only the forward is retried:
  a failure after it — a backward that fails to compile (recorded
  ``failed``, stage ``backward``), clipping, the optimizer step —
  propagates unchanged, so a partly applied update is never replayed.
- **Recorded.** :class:`CompileRecord` holds what was requested (backend,
  options, policy, torch version) and what took effect (``pending`` until
  the first compiled forward, then ``compiled`` with full or partial graph
  capture, ``eager`` after a restart, ``failed`` when compilation failed
  under the ``error`` policy). The same record is on ``NNRun.compile``, in
  ``runs/<id>/metadata.yaml``, in every stateful checkpoint's training
  state (``"compile"``) and, with a provenance manifest, in
  ``attempt.json``; it is never part of the run id. The run and its
  checkpoints carry the record as of their last commit; ``attempt.json``
  as of the attempt's end, so a failure after the last commit shows there
  only. ``graph_breaks`` counts the breaks of the first compiled call when
  it compiled; a frame served from dynamo's cache (the same model, or
  class, compiled earlier in the process) leaves it ``None`` — capture
  unknown — and torch then reuses the cached graphs even under
  ``fullgraph=True``.

``fullgraph=False`` allows graph breaks (partial capture, counted in
``CompileRecord.graph_breaks``); ``fullgraph=True`` rejects them, which the
failure policy then handles.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Optional

import torch

__all__ = [
    "COMPILE_FAILURE_POLICIES",
    "COMPILE_MODES",
    "CompileFailed",
    "CompileRecord",
    "CompileSpec",
]

COMPILE_MODES: tuple[str, ...] = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
"""``torch.compile`` modes a :class:`CompileSpec` accepts (``None`` is torch's default)."""

COMPILE_FAILURE_POLICIES: tuple[str, ...] = ("error", "eager")
"""What a compile failure does: raise, or restart the forward eagerly."""

COMPILE_STATES: tuple[str, ...] = ("pending", "compiled", "eager", "failed")


class CompileFailed(RuntimeError):
    """A compiled forward failed to compile under ``on_failure="error"``;
    ``__cause__`` is torch's error and ``record`` the run's
    :class:`CompileRecord` (effective state ``failed``)."""

    def __init__(self, message: str, record: CompileRecord) -> None:
        super().__init__(message)
        self.record = record


@dataclass(frozen=True, kw_only=True, slots=True)
class CompileSpec:
    """What to compile with: a ``torch.compile`` ``backend`` (default
    ``"inductor"``), its ``mode`` (one of :data:`COMPILE_MODES`, or ``None``),
    ``fullgraph`` (reject graph breaks), ``dynamic`` (``None`` lets torch
    decide) and ``on_failure`` (``"error"`` or ``"eager"``)."""

    backend: str = "inductor"
    mode: Optional[str] = None
    fullgraph: bool = False
    dynamic: Optional[bool] = None
    on_failure: str = "error"

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError(f"CompileSpec.backend must be a non-empty backend name, got {self.backend!r}")
        if self.mode is not None and self.mode not in COMPILE_MODES:
            raise ValueError(f"CompileSpec.mode must be one of {COMPILE_MODES} or None, got {self.mode!r}")
        if not isinstance(self.fullgraph, bool):
            raise TypeError(f"CompileSpec.fullgraph must be a bool, got {self.fullgraph!r}")
        if self.dynamic is not None and not isinstance(self.dynamic, bool):
            raise TypeError(f"CompileSpec.dynamic must be a bool or None, got {self.dynamic!r}")
        if self.mode is not None and self.backend != "inductor":
            raise ValueError(f"CompileSpec.mode applies to the inductor backend only, not {self.backend!r}")
        if self.on_failure not in COMPILE_FAILURE_POLICIES:
            raise ValueError(
                f"CompileSpec.on_failure must be one of {COMPILE_FAILURE_POLICIES}, got {self.on_failure!r}"
            )

    def options(self) -> dict[str, Any]:
        """The keyword options passed to ``torch.compile``."""
        return {"mode": self.mode, "fullgraph": self.fullgraph, "dynamic": self.dynamic}

    def state(self) -> dict[str, Any]:
        """Fields that differ from their defaults (``{}`` for the defaults)."""
        state: dict[str, Any] = {}
        if self.backend != "inductor":
            state["backend"] = self.backend
        if self.mode is not None:
            state["mode"] = self.mode
        if self.fullgraph:
            state["fullgraph"] = True
        if self.dynamic is not None:
            state["dynamic"] = self.dynamic
        if self.on_failure != "error":
            state["on_failure"] = self.on_failure
        return state

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> CompileSpec:
        unknown = set(state) - {"backend", "mode", "fullgraph", "dynamic", "on_failure"}
        if unknown:
            raise ValueError(f"unknown CompileSpec state keys {sorted(unknown)}")
        return CompileSpec(**dict(state))


@dataclass(frozen=True, kw_only=True, slots=True)
class CompileRecord:
    """What a run asked to compile and what took effect.

    ``backend`` / ``options`` / ``on_failure`` are the request and
    ``torch_version`` the framework that served it. ``effective`` is
    ``pending`` (no compiled forward has run yet), ``compiled``, ``eager``
    (restarted eager under ``on_failure="eager"``) or ``failed``.
    ``graph_breaks`` counts the breaks of the first compiled forward
    (``capture`` is ``"full"`` or ``"partial"``); ``restart`` holds where
    and why an eager restart (or a failure) happened."""

    backend: str
    options: Mapping[str, Any] = field(hash=False)
    on_failure: str
    torch_version: str
    effective: str = "pending"
    graph_breaks: Optional[int] = None
    restart: Optional[Mapping[str, Any]] = field(default=None, hash=False)

    def __post_init__(self) -> None:
        if self.effective not in COMPILE_STATES:
            raise ValueError(f"CompileRecord.effective must be one of {COMPILE_STATES}, got {self.effective!r}")

    @property
    def requested(self) -> bool:
        """Always ``True``: a record exists only when compilation was requested."""
        return True

    @property
    def capture(self) -> Optional[str]:
        """``"full"`` / ``"partial"`` once a compiled forward ran, else ``None``."""
        if self.graph_breaks is None:
            return None
        return "full" if self.graph_breaks == 0 else "partial"

    def record(self) -> dict[str, Any]:
        """The plain mapping stored in metadata, checkpoints and attempts."""
        return {
            "requested": True,
            "backend": self.backend,
            "options": dict(self.options),
            "on_failure": self.on_failure,
            "torch_version": self.torch_version,
            "effective": self.effective,
            "graph_breaks": self.graph_breaks,
            "capture": self.capture,
            "restart": None if self.restart is None else dict(self.restart),
        }

    @staticmethod
    def from_record(record: Mapping[str, Any]) -> CompileRecord:
        return CompileRecord(
            backend=record["backend"],
            options=dict(record["options"]),
            on_failure=record["on_failure"],
            torch_version=record["torch_version"],
            effective=record["effective"],
            graph_breaks=record.get("graph_breaks"),
            restart=None if record.get("restart") is None else dict(record["restart"]),
        )


def _known_backends() -> list[str]:
    from torch._dynamo import list_backends

    return list_backends(exclude_tags=())


def _is_compile_failure(error: BaseException) -> bool:
    import torch._dynamo.exc as dynamo_errors

    kinds: tuple[type[BaseException], ...] = (dynamo_errors.TorchDynamoException,)
    limit_hit = getattr(dynamo_errors, "FailOnRecompileLimitHit", None)  # torch >= 2.6
    if isinstance(limit_hit, type):
        kinds = (*kinds, limit_hit)
    return isinstance(error, kinds)


def _counter(group: str) -> int:
    from torch._dynamo.utils import counters

    return sum(counters[group].values())


def _unique_graphs() -> int:
    from torch._dynamo.utils import counters

    return int(counters["stats"]["unique_graphs"])


def _message(error: BaseException) -> str:
    text = str(error).strip()
    return (text.splitlines()[0] if text else type(error).__name__)[:500]


_LIMIT_NAMES = ("recompile_limit", "cache_size_limit")  # torch >= 2.6 / older


def _cached_entries(net: torch.nn.Module) -> int:
    """Dynamo cache entries already held for the net class's forward (other
    fits of the same class, earlier in this process)."""
    try:
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list

        return len(_debug_get_cache_entry_list(type(net).forward.__code__))
    except Exception:  # a private helper: absent or changed in this torch
        return 0


def _strict_recompiles(allowance: int) -> Any:
    """Make dynamo's recompile limit an error, not a silent eager fallback
    (where torch supports it), so the failure policy decides. The limit is
    per code object and process-wide, so each fit gets it afresh: the
    configured limit plus the entries other fits of the class already hold."""
    import torch._dynamo.config as config

    if not hasattr(config, "fail_on_recompile_limit_hit"):
        return contextlib.nullcontext()
    changes: dict[str, Any] = {"fail_on_recompile_limit_hit": True}
    name = next((name for name in _LIMIT_NAMES if hasattr(config, name)), None)
    if name is not None and allowance:
        changes[name] = int(getattr(config, name)) + allowance
    return config.patch(**changes)


class _Snapshot:
    """The global RNG streams and the module's buffers before a compiled
    call, restored before an eager re-run so the retry draws and mutates
    exactly what an eager call would have."""

    def __init__(self, net: torch.nn.Module) -> None:
        self.cpu = torch.get_rng_state()
        self.cuda = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else None
        )
        self.buffers = [(buffer, buffer.detach().clone()) for buffer in net.buffers()]

    def restore(self) -> None:
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state_all(self.cuda)
        with torch.no_grad():
            for buffer, saved in self.buffers:
                buffer.copy_(saved)


class _CompileSession:
    """One fit's compiled forward: the wrapper (built around, never assigned
    to, the canonical module) and its record. Not thread-safe; one per
    ``train()`` call."""

    def __init__(self, net: torch.nn.Module, spec: CompileSpec) -> None:
        if spec.backend not in _known_backends():
            raise ValueError(
                f"CompileSpec.backend {spec.backend!r} is not a torch.compile backend here; "
                f"available: {sorted(_known_backends())}"
            )
        self.spec = spec
        self.record = CompileRecord(
            backend=spec.backend,
            options=spec.options(),
            on_failure=spec.on_failure,
            torch_version=str(torch.__version__),
        )
        self._net = net
        self._compiled: Optional[Callable[..., Any]] = torch.compile(net, backend=spec.backend, **spec.options())
        self._backward_proven = False
        self._allowance = _cached_entries(net)
        self.where: Mapping[str, Any] = {}

    def _fail(self, error: BaseException, **stage: Any) -> dict[str, Any]:
        restart = {**self.where, **stage, "error": type(error).__name__, "message": _message(error)}
        self._compiled = None
        return restart

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        compiled = self._compiled
        if compiled is None:  # restarted eager (or failed)
            return self._net(*args, **kwargs)
        snapshot = _Snapshot(self._net)
        first = self.record.effective == "pending"
        graphs_before, breaks_before = _unique_graphs(), _counter("graph_break")
        try:
            with _strict_recompiles(self._allowance):
                output = compiled(*args, **kwargs)
        except Exception as error:
            if not _is_compile_failure(error):
                raise
            # A real error in the user's forward also surfaces through
            # dynamo: an eager re-run (from the same RNG and buffers) tells
            # the two apart, and its own error is the one that propagates.
            snapshot.restore()
            eager_output = self._net(*args, **kwargs)
            restart = self._fail(error, stage="forward")
            if self.spec.on_failure == "error":
                snapshot.restore()  # leave the model as the failed call found it
                self.record = replace(self.record, effective="failed", restart=restart)
                raise CompileFailed(
                    f"torch.compile (backend {self.spec.backend!r}) failed: {restart['message']}; "
                    "pass CompileSpec(on_failure='eager') to restart this forward eagerly instead",
                    self.record,
                ) from error
            # Only the forward is re-run (nothing the optimizer owns has
            # changed), from the RNG and buffers it started with.
            self.record = replace(self.record, effective="eager", restart=restart)
            return eager_output
        compiled_now = _unique_graphs() > graphs_before
        if compiled_now and torch.is_grad_enabled():
            # A new training graph compiles its backward lazily, on its first backward.
            self._backward_proven = False
        if first:
            # Breaks are counted only when this call compiled: a frame served
            # from dynamo's cache compiles nothing, so its capture is unknown.
            breaks = _counter("graph_break") - breaks_before if compiled_now else None
            self.record = replace(self.record, effective="compiled", graph_breaks=breaks)
        return output

    @contextlib.contextmanager
    def backward(self) -> Iterator[None]:
        """Around the step's backward: AOTAutograd compiles a backward graph
        on its first use, so a failure there is a compile failure too. It is
        never retried (gradients may be partly accumulated) — the record
        says ``failed`` (stage ``backward``) and the error propagates
        unchanged. Conservatively, any error in the first backward after a
        new graph compiled counts (a backend's error need not be a dynamo
        exception), as does a dynamo error in any backward."""
        if self._compiled is None or self.record.effective != "compiled":
            yield
            return
        try:
            yield
        except Exception as error:
            if _is_compile_failure(error) or not self._backward_proven:
                self.record = replace(self.record, effective="failed", restart=self._fail(error, stage="backward"))
            raise
        self._backward_proven = True


@dataclass
class _CompileHolder:
    """The current fit's session, read by the attempt recorder whatever
    the fit's outcome."""

    session: Optional[_CompileSession] = field(default=None)

    def record(self) -> Optional[dict[str, Any]]:
        return None if self.session is None else self.session.record.record()
