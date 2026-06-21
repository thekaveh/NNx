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
        `rope_base=None` is the sentinel for "use the dataclass default
        (10000.0, the LLaMA / GPT convention)". The fluent contract is
        "last call wins": `.context(rope_base=500000.0).context(max_seq_len=128)`
        resets `rope_base` to the default — the second call's implicit
        `rope_base=None` drops the prior override."""
        self._fields["max_seq_len"] = max_seq_len
        if rope_base is None:
            # Drop any prior override so the dataclass default governs
            # at build time; this is what makes last-call-wins work for
            # the None sentinel.
            self._fields.pop("rope_base", None)
        else:
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
        scale, not dropout).

        Like `.context()`, a `dropout()` call specifies BOTH rates
        together — each call fully replaces the pair, and a rate left
        at its 0.0 default is reset, not carried over from a prior
        call. So `.dropout(resid=0.3).dropout(attn=0.5)` yields
        `attn=0.5, resid=0.0` (the second call's implicit `resid=0.0`
        drops the prior override); call `.dropout(attn=0.5, resid=0.3)`
        once to set both. Same-field last-call-wins still holds:
        `.dropout(attn=0.5).dropout(attn=0.0)` resets to `attn=0.0`.
        The dataclass's omit-when-default `state()` then handles
        run.id stability automatically."""
        self._fields["attn_dropout"] = attn
        self._fields["resid_dropout"] = resid
        return self

    def tied_embeddings(self, value: bool) -> NNTransformerParamsBuilder:
        """Toggle weight-tying between input embeddings and LM head.
        Default is True. The fluent contract is "last call wins" — a
        prior `.tied_embeddings(False)` followed by `.tied_embeddings(True)`
        leaves the dataclass at the default (which `state()` then omits)."""
        self._fields["tie_embeddings"] = value
        return self

    def build(self) -> NNTransformerParams:
        """Construct the dataclass.

        Pre-empts the dataclass's missing-required-argument TypeError
        with an actionable Builder-level ValueError naming the setter
        methods that haven't been called yet — matches the
        [[builder-pattern-shape]] §11b convention that PR #52
        established on NNTrainerParamsBuilder.

        Fills in the dead parent-NNParams fields the TransformerNN
        net never reads but the parent dataclass requires at
        construction. `activation` mirrors the parent NNParams's
        default (`Activations.LEAKY_RELU`); a Builder-default
        mismatch here previously produced a different `state()` /
        `run.id` than the direct-kwarg ctor.

        Raises:
            ValueError: if `.vocab(size=...)`, `.layers(n=..., heads=...,
                d_model=...)`, or `.context(max_seq_len=...)` was not
                called before `.build()`. The message names the
                specific setter methods that are still missing so the
                user can complete the chain without consulting the
                dataclass schema.
        """
        missing: list[str] = []
        if "vocab_size" not in self._fields:
            missing.append(".vocab(size=...)")
        if "n_layers" not in self._fields:
            missing.append(".layers(n=..., heads=..., d_model=...)")
        if "max_seq_len" not in self._fields:
            missing.append(".context(max_seq_len=...)")
        if missing:
            raise ValueError(
                "NNTransformerParamsBuilder: call "
                + ", ".join(missing)
                + " before .build() — each setter fills the dataclass's "
                "required-no-default fields for the LM path."
            )
        return NNTransformerParams(
            hidden_dims=None,
            dropout_prob=0.0,
            activation=Activations.LEAKY_RELU,
            **self._fields,
        )
