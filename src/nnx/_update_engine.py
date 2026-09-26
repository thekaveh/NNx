"""The shared optimizer-update engine behind objective training (FEAT-004).

An *objective* (``nnx.objectives``) returns loss **terms** — a numerator
plus, for a normalized term, the explicit number of samples it sums over —
for each microbatch. This engine owns everything after that:

1. per microbatch, every term's numerator is back-propagated into its own
   gradient buffer (numerators are sums, so buffers add across
   microbatches). A window of one microbatch — the default — needs no
   buffers: its weighted, normalized loss is back-propagated once. Terms
   with weight ``0`` are never back-propagated, and only parameters that
   require gradients *now* are differentiated, so freezing or unfreezing
   parameters between microbatches (gradual unfreezing) just works;
2. at the end of an update window (``accumulate_grad_batches`` microbatches,
   or fewer at the epoch's end) each normalized term is divided by its
   window-total denominator, summed terms are left as totals, and the
   weighted combination becomes the parameters' gradients — exactly the
   gradient of the full-window loss, whatever the microbatch sizes or
   masks;
3. mixed precision: ``unscale`` → gradient clipping → ``step`` →
   ``update``, in that order (the objective's forward already ran under
   autocast);
4. a *committed-update* event fires once per successful optimizer update —
   never for a microbatch, an all-masked window or a skipped step.

Non-finite losses or gradients follow a declared policy: ``"fail"`` raises
``FloatingPointError`` before anything is stepped; ``"skip"`` drops the
window's gradients and takes no update. An AMP ``GradScaler`` skipping a
step (inf / NaN gradients while it calibrates its scale) is always a quiet
skip. The committed-update counters are checkpointed component state
(``nnx.update_engine``, FEAT-005), so they continue across a stateful
resume. Internal; ``nnx.objectives`` is the public surface.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Optional, cast

import torch

NONFINITE_POLICIES = ("fail", "skip")
REDUCTIONS = ("mean", "sum")


def check_nonfinite_policy(nonfinite: str) -> str:
    """Validate a non-finite policy (shared by the engine and objectives)."""
    if nonfinite not in NONFINITE_POLICIES:
        raise ValueError(f"nonfinite policy must be 'fail' or 'skip', got {nonfinite!r}")
    return nonfinite


@dataclass(frozen=True)
class UpdateEvent:
    """One committed optimizer update (detached values only).

    Attributes:
        optimizer: the optimizer that stepped (``"default"`` for
            ``NNModel.train``; the name for ``Trainer``).
        update_idx: this optimizer's committed updates so far (1-based).
        epoch_idx / batch_idx: where the window closed.
        microbatches: microbatches the update accumulated.
        losses: each term's window value (normalized terms divided by their
            window-total denominator).
        loss: the weighted total.
    """

    optimizer: str
    update_idx: int
    epoch_idx: int
    batch_idx: int
    microbatches: int
    losses: Mapping[str, float]
    loss: float


@dataclass
class _TermWindow:
    reduction: str
    weight: float
    numerator: float = 0.0
    denominator: float = 0.0
    grads: list[Optional[torch.Tensor]] = field(default_factory=list)  # per parameter, allocated on first use


class UpdateEngine:
    """Accumulates objective terms per microbatch and commits one update
    per window. See the module docstring for the contract."""

    def __init__(
        self,
        *,
        optimizers: Mapping[str, torch.optim.Optimizer],
        scaler: Any = None,
        clip_norms: Optional[Mapping[str, Optional[float]]] = None,
        nonfinite: str = "fail",
        autocast: Optional[Callable[[], AbstractContextManager[Any]]] = None,
        listeners: Iterable[Callable[[UpdateEvent], None]] = (),
    ) -> None:
        check_nonfinite_policy(nonfinite)
        if not optimizers:
            raise ValueError("UpdateEngine needs at least one optimizer")
        self.optimizers = dict(optimizers)
        self.scaler = scaler
        self.clip_norms = dict(clip_norms or {})
        self.nonfinite = nonfinite
        self._autocast = autocast
        self.listeners: list[Callable[[UpdateEvent], None]] = list(listeners)
        # Every optimizer's parameters, each once, in a stable order; which of
        # them require gradients is decided per microbatch (freezing).
        index: dict[int, int] = {}
        self._params: list[torch.nn.Parameter] = []
        self._owner: dict[str, list[int]] = {}
        for name, optimizer in self.optimizers.items():
            owned: list[int] = []
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if id(param) not in index:
                        index[id(param)] = len(self._params)
                        self._params.append(param)
                    owned.append(index[id(param)])
            self._owner[name] = owned
        self.update_counts: dict[str, int] = {name: 0 for name in self.optimizers}
        self.commits = 0
        self.skipped = 0
        self._reset_window()

    # ---------- window state ----------

    def _reset_window(self) -> None:
        self._terms: dict[str, _TermWindow] = {}
        self._microbatches = 0
        self._nonfinite_window = False
        # A one-microbatch window's combined gradients (one backward pass).
        self._fused: Optional[list[Optional[torch.Tensor]]] = None

    @property
    def pending_microbatches(self) -> int:
        return self._microbatches

    def autocast(self) -> AbstractContextManager[Any]:
        """The context the objective's forward runs in (mixed precision
        when enabled, a no-op otherwise)."""
        return self._autocast() if self._autocast is not None else contextlib.nullcontext()

    # ---------- per microbatch ----------

    def accumulate(self, terms: Sequence[Any], *, closes_window: bool = False) -> None:
        """Back-propagate one microbatch's terms into the window.
        ``closes_window`` announces that :meth:`commit` follows this
        microbatch: a window of just this microbatch then takes one backward
        pass through its combined loss instead of one per term."""
        if self._fused is not None:
            raise RuntimeError("accumulate(closes_window=True) must be followed by commit() before more microbatches")
        _check_terms(terms, self._terms)
        single = closes_window and not self._microbatches
        self._microbatches += 1
        live: list[Any] = []
        for term in terms:
            value = term._numerator_value  # one host sync per term, cached on the term
            if not math.isfinite(value):
                if self.nonfinite == "fail":
                    raise FloatingPointError(
                        f"non-finite loss term {term.name!r} ({value!r}); nothing was stepped. Check the learning "
                        "rate, the objective's inputs, or pass nonfinite='skip' to drop such windows"
                    )
                self._nonfinite_window = True
            window = self._terms.setdefault(term.name, _TermWindow(term.reduction, term.weight))
            window.numerator += value
            if term.reduction == "mean":
                window.denominator += float(term.denominator)
            if term.weight and term.numerator.requires_grad and (term.reduction == "sum" or term.denominator):
                live.append(term)
        if self._nonfinite_window:
            return  # the window is dropped at commit; no gradient work
        active = [j for j, param in enumerate(self._params) if param.requires_grad]
        if not live or not active:
            return
        if single:
            # The window's loss is known now: Σ weight · numerator / denominator.
            combined = sum(term.numerator * _term_scale(term.reduction, term.weight, term.denominator) for term in live)
            self._fused = self._gradients(cast(torch.Tensor, combined), active, retain_graph=False)
            return
        for i, term in enumerate(live):
            grads = self._gradients(term.numerator, active, retain_graph=i < len(live) - 1)
            window = self._terms[term.name]
            if not window.grads:
                window.grads = [None] * len(self._params)
            for j, grad in enumerate(grads):
                if grad is not None:
                    previous = window.grads[j]
                    window.grads[j] = grad if previous is None else previous + grad

    def _gradients(self, loss: torch.Tensor, active: list[int], *, retain_graph: bool) -> list[Optional[torch.Tensor]]:
        """Gradients of ``loss`` (scaled for AMP) for every parameter —
        ``None`` for a frozen or unused one."""
        if self.scaler is not None:
            loss = self.scaler.scale(loss)
        grads = torch.autograd.grad(
            loss, [self._params[j] for j in active], retain_graph=retain_graph, allow_unused=True
        )
        out: list[Optional[torch.Tensor]] = [None] * len(self._params)
        for j, grad in zip(active, grads, strict=True):
            out[j] = None if grad is None else grad.detach()
        return out

    # ---------- per window ----------

    def window_values(self) -> tuple[dict[str, float], float]:
        """Each term's window value and the weighted total (for records)."""
        values: dict[str, float] = {}
        total = 0.0
        for name, window in self._terms.items():
            if window.reduction == "sum":
                value = window.numerator
            elif window.denominator:
                value = window.numerator / window.denominator
            else:
                continue  # every sample of this term was masked
            values[name] = value
            total += window.weight * value
        return values, total

    def commit(self, *, epoch_idx: int, batch_idx: int) -> tuple[UpdateEvent, ...]:
        """Close the window: combine, (unscale), clip, step. Returns one
        event per optimizer that committed an update (none when the window
        was empty, all-masked, non-finite or skipped by the scaler)."""
        try:
            if not self._microbatches:
                return ()
            if self._nonfinite_window:
                self.skipped += 1
                return ()
            values, total = self.window_values()
            if not values:
                return ()  # every term fully masked: nothing to learn from
            self._assign_gradients()
            if self.scaler is not None:
                for optimizer in self.optimizers.values():
                    self.scaler.unscale_(optimizer)
            elif not self._gradients_finite():
                if self.nonfinite == "fail":
                    raise FloatingPointError(
                        "non-finite gradients in the update window; nothing was stepped. Pass nonfinite='skip' "
                        "to drop such windows"
                    )
                self.skipped += 1
                return ()
            for name in self.optimizers:
                norm = self.clip_norms.get(name)
                if norm is not None:
                    torch.nn.utils.clip_grad_norm_([self._params[i] for i in self._owner[name]], norm)
            if self.scaler is not None:
                scale_before = float(self.scaler.get_scale())
                for optimizer in self.optimizers.values():
                    self.scaler.step(optimizer)
                self.scaler.update()
                if float(self.scaler.get_scale()) < scale_before:
                    self.skipped += 1  # the scaler found inf/NaN gradients and skipped the step
                    return ()
            else:
                for optimizer in self.optimizers.values():
                    optimizer.step()
            self.commits += 1
            events = []
            for name in self.optimizers:
                self.update_counts[name] += 1
                events.append(
                    UpdateEvent(
                        optimizer=name,
                        update_idx=self.update_counts[name],
                        epoch_idx=epoch_idx,
                        batch_idx=batch_idx,
                        microbatches=self._microbatches,
                        losses=dict(values),
                        loss=total,
                    )
                )
            for event in events:
                for listener in self.listeners:
                    listener(event)
            return tuple(events)
        finally:
            for param in self._params:
                param.grad = None
            self._reset_window()

    def _assign_gradients(self) -> None:
        for j, param in enumerate(self._params):
            if not param.requires_grad:
                param.grad = None  # frozen since it was differentiated: not updated
                continue
            if self._fused is not None:
                param.grad = self._fused[j]
                continue
            combined: Optional[torch.Tensor] = None
            for window in self._terms.values():
                grad = window.grads[j] if window.grads else None
                if grad is None:
                    continue
                contribution = grad * _term_scale(window.reduction, window.weight, window.denominator)
                combined = contribution if combined is None else combined + contribution
            param.grad = combined

    def _gradients_finite(self) -> bool:
        return all(param.grad is None or bool(torch.isfinite(param.grad).all()) for param in self._params)

    # ---------- checkpointable component (FEAT-005) ----------

    def component_spec(self) -> Any:
        from .components import ComponentSpec

        return ComponentSpec("nnx.update_engine", version=1, required=False)

    def component_state(self) -> dict[str, Any]:
        return {"commits": self.commits, "skipped": self.skipped, "update_counts": dict(self.update_counts)}

    def check_component_state(self, state: Mapping[str, Any], *, version: int) -> list[str]:
        counts = state.get("update_counts")
        values = [state.get("commits"), state.get("skipped"), *(counts.values() if isinstance(counts, Mapping) else ())]
        if not isinstance(counts, Mapping) or not all(_is_count(value) for value in values):
            return [f"malformed update counters {dict(state)!r}"]
        return []

    def load_component_state(self, state: Mapping[str, Any], *, version: int) -> None:
        self.commits = int(state["commits"])
        self.skipped = int(state["skipped"])
        saved = state["update_counts"]
        self.update_counts = {name: int(saved.get(name, 0)) for name in self.optimizers}


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _term_scale(reduction: str, weight: float, denominator: Optional[float]) -> float:
    """A term's factor in the window loss: ``weight / denominator`` for a
    normalized term, ``weight`` for a summed one."""
    if reduction == "sum":
        return weight
    assert denominator, "a normalized term contributes only with a positive denominator"
    return weight / denominator


def _check_terms(terms: Sequence[Any], window: Mapping[str, _TermWindow]) -> None:
    """Names are unique per microbatch; a name keeps its reduction and
    weight for the whole window (summed and normalized terms never mix)."""
    names = [term.name for term in terms]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"objective returned duplicate loss terms: {', '.join(duplicates)}")
    if not terms:
        raise ValueError("objective returned no loss terms")
    for term in terms:
        existing = window.get(term.name)
        if existing is None:
            continue
        if existing.reduction != term.reduction:
            raise ValueError(
                f"loss term {term.name!r} mixes {existing.reduction!r} and {term.reduction!r} reductions within one "
                "update window; a term is either summed or normalized by an explicit denominator"
            )
        if existing.weight != term.weight:
            raise ValueError(f"loss term {term.name!r} changes its weight within one update window")
