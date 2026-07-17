from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional

from ..enum.activations import Activations


@dataclass(frozen=True, kw_only=True, slots=True)
class NNParams:
    dropout_prob: float
    n_heads: Optional[int] = field(default=None)
    activation: Optional[Activations] = field(default=Activations.LEAKY_RELU)
    # Per-layer overrides (#85): optional lists, len == number of hidden layers.
    # None → the net-wide scalar applies uniformly (unchanged default).
    activations: Optional[list[Activations]] = field(default=None)
    dropout_probs: Optional[list[float]] = field(default=None)

    input_dim: int = field(repr=False)
    output_dim: int = field(repr=False)
    hidden_dims: Optional[list[int]] = field(repr=False, default=None)
    _dims: Optional[list[int]] = field(repr=False, init=False, default=None)

    @property
    def dims(self) -> list[int]:
        # `_dims` is set unconditionally in __post_init__, so the
        # Optional type on the field is purely a dataclass artifact
        # (no init=True default for slotted frozen subclasses). The
        # assert documents that contract for both readers and the
        # type-checker (pyright can't model __post_init__-set fields
        # via `object.__setattr__`).
        assert self._dims is not None
        return self._dims

    def __post_init__(self):
        # Fail-fast on out-of-range numeric fields at construction time rather
        # than deep inside layer building. nn.Dropout / nn.Linear would raise
        # eventually for these, but far from the origin — surfacing the error
        # here keeps the [[params-boundary-validation]] contract consistent
        # with the rest of the params hierarchy. None of these touch state().
        if not 0.0 <= self.dropout_prob <= 1.0:
            raise ValueError(f"NNParams requires 0.0 <= dropout_prob <= 1.0, got {self.dropout_prob}")
        if self.input_dim <= 0:
            raise ValueError(f"NNParams requires input_dim > 0, got {self.input_dim}")
        if self.output_dim <= 0:
            raise ValueError(f"NNParams requires output_dim > 0, got {self.output_dim}")
        if self.hidden_dims is not None and not all(d > 0 for d in self.hidden_dims):
            raise ValueError(f"NNParams requires all hidden_dims > 0, got {self.hidden_dims}")

        # Per-layer overrides (#85): length must equal the number of hidden
        # layers; dropout entries must be valid probabilities.
        n_hidden = len(self.hidden_dims) if self.hidden_dims is not None else 0
        if self.activations is not None and len(self.activations) != n_hidden:
            raise ValueError(
                f"NNParams.activations must have one entry per hidden layer ({n_hidden}), got {len(self.activations)}"
            )
        if self.dropout_probs is not None:
            if len(self.dropout_probs) != n_hidden:
                raise ValueError(
                    f"NNParams.dropout_probs must have one entry per hidden layer "
                    f"({n_hidden}), got {len(self.dropout_probs)}"
                )
            if not all(0.0 <= q <= 1.0 for q in self.dropout_probs):
                raise ValueError(f"NNParams requires 0.0 <= dropout_probs[i] <= 1.0, got {self.dropout_probs}")

        dims = [self.input_dim]
        dims += self.hidden_dims if self.hidden_dims is not None else []
        dims += [self.output_dim]

        object.__setattr__(self, "_dims", dims)

    def activation_for(self, layer_idx: int):
        """The activation for hidden layer ``layer_idx`` — the per-layer entry
        when `activations` is set, else the net-wide scalar (#85)."""
        if self.activations is not None:
            return self.activations[layer_idx]
        return self.activation

    def dropout_for(self, layer_idx: int) -> float:
        """The dropout prob for hidden layer ``layer_idx`` — per-layer entry
        when `dropout_probs` is set, else the net-wide scalar (#85)."""
        if self.dropout_probs is not None:
            return self.dropout_probs[layer_idx]
        return self.dropout_prob

    def state(self) -> dict:
        ret = dict(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            dropout_prob=self.dropout_prob,
            hidden_dims=str(self.hidden_dims),
            # None stays None (yaml null / json null) rather than the
            # string "None", which from_state could never parse back
            # into the Activations enum.
            activation=str(self.activation) if self.activation is not None else None,
        )

        if self.n_heads is not None:
            ret["n_heads"] = self.n_heads

        # Per-layer overrides (#85): emitted ONLY when present AND not merely a
        # uniform repetition of the scalar — the omit-when-default invariant
        # that keeps every existing config hashing to the same NNRun.id.
        if self.activations is not None and any(a != self.activation for a in self.activations):
            ret["activations"] = str([str(a) for a in self.activations])
        if self.dropout_probs is not None and any(q != self.dropout_prob for q in self.dropout_probs):
            ret["dropout_probs"] = str(self.dropout_probs)

        return ret

    @staticmethod
    def from_state(state: dict) -> NNParams:
        raw_activation = state["activation"]
        raw_acts = state.get("activations")
        raw_drops = state.get("dropout_probs")
        return NNParams(
            input_dim=state["input_dim"],
            output_dim=state["output_dim"],
            dropout_prob=state["dropout_prob"],
            activation=Activations(raw_activation) if raw_activation is not None else None,
            hidden_dims=ast.literal_eval(state["hidden_dims"]),
            n_heads=state["n_heads"] if "n_heads" in state else None,
            activations=([Activations(a) for a in ast.literal_eval(raw_acts)] if raw_acts is not None else None),
            dropout_probs=ast.literal_eval(raw_drops) if raw_drops is not None else None,
        )

    @staticmethod
    def resolve_from_state(state: dict) -> NNParams:
        """Dispatch to the params subclass that wrote ``state``.

        ``NNTransformerParams.state()`` always emits its required
        architectural keys (``vocab_size`` among them); base
        ``NNParams.state()`` never does. Without this dispatch a
        transformer state is silently downgraded to base ``NNParams`` —
        the subclass keys are dropped, the reloaded run re-hashes to a
        different id, and net rebuilding crashes. Every loader
        (``NNRun.load``, the ``NNCheckpoint`` readers, hub
        ``from_pretrained``) resolves through here.
        """
        if "vocab_size" in state:
            # Local import: nn_transformer_params imports this module.
            from .nn_transformer_params import NNTransformerParams

            return NNTransformerParams.from_state(state)
        return NNParams.from_state(state)
