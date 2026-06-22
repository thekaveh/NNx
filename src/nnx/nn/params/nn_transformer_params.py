"""Transformer architecture params.

`NNTransformerParams` subclasses `NNParams` — the same lift-via-subclassing
pattern `GraphAttNN` uses for `n_heads`. The base-class fields
(`input_dim`, `output_dim`, `dropout_prob`, `n_heads`, `activation`,
`hidden_dims`) are accepted but the transformer-specific shape is
controlled by the subclass fields below. For an LM the convention is
``input_dim = output_dim = vocab_size``, which the example sets
explicitly so the NNParams contract still type-checks.

Every optional field omits itself from `state()` when at its default —
this is the **broken-three-times** invariant (param_groups,
mixed_precision, scheduler kind). Adding a TRANSFORMER variant without
preserving the invariant would force every existing run.id to shift
the next time someone instantiated the same config they had before.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..enum.activations import Activations
from .nn_params import NNParams

if TYPE_CHECKING:
    from .nn_transformer_params_builder import NNTransformerParamsBuilder


@dataclass(frozen=True, kw_only=True, slots=True)
class NNTransformerParams(NNParams):
    # Required architectural knobs. There's no meaningful default for
    # "vocabulary size" or "depth", so these are required.
    vocab_size: int
    n_layers: int
    d_model: int
    max_seq_len: int

    # `n_heads` is lifted from NNParams (where it was an Optional[int]
    # used only by GraphAttNN). For TransformerNN it's required; we
    # validate at __post_init__.

    # Defaulted knobs — all of these omit themselves from state() when
    # equal to the default value.
    ffn_mult: int = 4
    rope_base: float = 10000.0
    tie_embeddings: bool = True
    # Separate attention dropout vs. residual dropout — both default to
    # 0.0 (matching modern LLM training where regularization comes from
    # data scale, not dropout). Omitted from state() at default.
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0

    def __post_init__(self):
        # Initialize the base-class _dims slot first. We call the
        # parent's __post_init__ explicitly rather than via super()
        # because `slots=True` dataclass subclassing interacts oddly
        # with the cooperative super() chain (the explicit unbound call
        # is what the dataclasses docs recommend for slotted hierarchies).
        NNParams.__post_init__(self)
        # Transformer-specific validation. n_heads must be set and must
        # divide d_model evenly — checked here so an invalid config
        # fails loudly at params-construction time rather than during
        # the forward pass.
        if self.n_heads is None or self.n_heads <= 0:
            raise ValueError(f"NNTransformerParams requires n_heads > 0, got {self.n_heads!r}")
        # Required positive architectural dimensions. `d_model` must be
        # validated BEFORE the divisibility test below: `d_model=0` passes
        # `0 % n_heads == 0` but then yields `head_dim = d_model // n_heads = 0`
        # (a zero attention-scale divisor and zero-width FFN) downstream — a
        # silent-failure footgun, not a loud one. `n_layers<=0` likewise builds
        # a degenerate embed→norm→head model with no error at all.
        for name in ("vocab_size", "n_layers", "d_model", "max_seq_len", "ffn_mult"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"NNTransformerParams requires {name} > 0, got {value}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")

    def state(self) -> dict:
        # Start with the base-class state (input_dim/output_dim/etc.)
        # so an LM run is still legible the same way every other run
        # is — pretty-printed yaml carries the dims for grep-friendliness.
        # Explicit unbound call: same reason as __post_init__ above —
        # slots=True dataclass subclassing doesn't play nicely with
        # cooperative super().
        d = NNParams.state(self)
        d.update(
            n_layers=self.n_layers,
            d_model=self.d_model,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
        )
        # `n_heads` is always emitted by NNParams.state() — for the
        # transformer path it's required, so it's never None.

        # Omit-when-default invariants (see module docstring). These
        # exist so a "vanilla" TRANSFORMER config hashes to a stable
        # run.id even as we add more knobs over time.
        if self.ffn_mult != 4:
            d["ffn_mult"] = self.ffn_mult
        if self.rope_base != 10000.0:
            d["rope_base"] = self.rope_base
        if self.tie_embeddings is not True:
            d["tie_embeddings"] = self.tie_embeddings
        if self.attn_dropout != 0.0:
            d["attn_dropout"] = self.attn_dropout
        if self.resid_dropout != 0.0:
            d["resid_dropout"] = self.resid_dropout
        return d

    @staticmethod
    def from_state(state: dict) -> NNTransformerParams:
        # `hidden_dims` may be stringified (NNParams convention) or
        # absent for LM configs where it isn't meaningful.
        hidden = state.get("hidden_dims")
        if isinstance(hidden, str):
            hidden = ast.literal_eval(hidden)

        # Base-class fields — fall back to the LM convention where the
        # output dimension equals vocab_size so the NNParams.dims
        # invariant is satisfiable.
        vocab_size = state["vocab_size"]
        input_dim = state.get("input_dim", vocab_size)
        output_dim = state.get("output_dim", vocab_size)

        if "activation" in state and state["activation"] is not None:
            activation = Activations(state["activation"])
        elif "activation" in state:
            # Explicit null — the config was built with activation=None.
            activation = None
        else:
            # Key absent entirely — legacy LM configs omitted it.
            activation = Activations.LEAKY_RELU

        return NNTransformerParams(
            input_dim=input_dim,
            output_dim=output_dim,
            dropout_prob=state.get("dropout_prob", 0.0),
            hidden_dims=hidden,
            activation=activation,
            n_heads=state.get("n_heads"),
            vocab_size=vocab_size,
            n_layers=state["n_layers"],
            d_model=state["d_model"],
            max_seq_len=state["max_seq_len"],
            ffn_mult=state.get("ffn_mult", 4),
            rope_base=state.get("rope_base", 10000.0),
            tie_embeddings=state.get("tie_embeddings", True),
            attn_dropout=state.get("attn_dropout", 0.0),
            resid_dropout=state.get("resid_dropout", 0.0),
        )

    def __str__(self) -> str:
        return (
            f"[transformer, vocab={self.vocab_size}, layers={self.n_layers}, "
            f"d_model={self.d_model}, heads={self.n_heads}, seq={self.max_seq_len}]"
        )

    @classmethod
    def builder(cls) -> NNTransformerParamsBuilder:
        """Return a fluent LM-path builder. See `NNTransformerParamsBuilder`."""
        from .nn_transformer_params_builder import NNTransformerParamsBuilder

        return NNTransformerParamsBuilder()
