"""Builder for NNTransformerParams — LM-path config.

Six fluent methods present the TransformerNN-shaped knobs without
exposing the dead parent-NNParams fields (`hidden_dims`, `activation`,
`dropout_prob`) the dataclass inherits but the transformer path
doesn't use. `.layers(n, heads, d_model)` enforces
`d_model % heads == 0` at call-time so the user finds out at the
chain step that introduced the mismatch, not at .build() much later.

See `docs/superpowers/specs/2026-05-31-builder-pattern-investigation.md`
§3.2 for the rubric scoring and design rationale.
"""

from __future__ import annotations

from typing import Any, Optional

from ..enum.activations import Activations
from .nn_transformer_params import NNTransformerParams


class NNTransformerParamsBuilder:
    """Builder for `NNTransformerParams`.

    Reach via `NNTransformerParams.builder()`. The six methods can be
    chained in any order; `.build()` collects them, fills in the
    LM-path defaults for the dead parent-NNParams fields, and
    constructs the dataclass.
    """

    def __init__(self) -> None:
        self._fields: dict[str, Any] = {}

    def vocab(self, size: int) -> NNTransformerParamsBuilder:
        """Set the vocabulary size. Mirrors into both `input_dim` and
        `output_dim` on the parent NNParams (the LM convention)."""
        self._fields["vocab_size"] = size
        self._fields["input_dim"] = size
        self._fields["output_dim"] = size
        return self

    def layers(
        self,
        *,
        n: int,
        heads: int,
        d_model: int,
    ) -> NNTransformerParamsBuilder:
        """Set depth (`n_layers`), attention head count (`n_heads`),
        and hidden dimension (`d_model`). Enforces
        `d_model % heads == 0` immediately — this is the Builder's
        safety value-add over the direct-kwarg ctor, which only
        catches the mismatch at __post_init__ time after all kwargs
        have already been typed."""
        if d_model % heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by heads={heads} "
                "(transformer attention requires d_model / n_heads to be integral)"
            )
        self._fields["n_layers"] = n
        self._fields["n_heads"] = heads
        self._fields["d_model"] = d_model
        return self

    def ffn(self, *, mult: int) -> NNTransformerParamsBuilder:
        """FFN expansion ratio. Default is 4 (the SwiGLU-friendly
        ratio); only call this method to override."""
        self._fields["ffn_mult"] = mult
        return self

    def context(
        self,
        *,
        max_seq_len: int,
        rope_base: Optional[float] = None,
    ) -> NNTransformerParamsBuilder:
        """Context-length and RoPE base. `max_seq_len` is required;
        `rope_base` defaults to 10000.0 (the LLaMA / GPT convention)
        when None — passing it explicitly overrides the default."""
        self._fields["max_seq_len"] = max_seq_len
        if rope_base is not None:
            self._fields["rope_base"] = rope_base
        return self

    def dropout(
        self,
        *,
        attn: float = 0.0,
        resid: float = 0.0,
    ) -> NNTransformerParamsBuilder:
        """Attention and residual dropout rates. Defaults are both
        0.0 (modern LLM convention; regularization comes from data
        scale, not dropout). Calling this with the defaults is a no-op
        on state() — the omit-when-default invariant kicks in."""
        if attn != 0.0:
            self._fields["attn_dropout"] = attn
        if resid != 0.0:
            self._fields["resid_dropout"] = resid
        return self

    def tied_embeddings(self, value: bool) -> NNTransformerParamsBuilder:
        """Toggle weight-tying between input embeddings and LM head.
        Default is True; only call this to set False."""
        if value is not True:
            self._fields["tie_embeddings"] = value
        return self

    def build(self) -> NNTransformerParams:
        """Construct the dataclass.

        Fills in the dead parent-NNParams fields with their
        LM-path defaults — `hidden_dims=None`, `dropout_prob=0.0`,
        `activation=Activations.RELU`. These are required by the
        parent dataclass but never read by the TransformerNN net, so
        the Builder hides them. The user-visible API stays
        LM-path-shaped.
        """
        return NNTransformerParams(
            hidden_dims=None,
            dropout_prob=0.0,
            activation=Activations.RELU,
            **self._fields,
        )
