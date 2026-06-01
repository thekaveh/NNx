"""Tests for NNTransformerParams.builder() — LM-path config without
the dead parent-NNParams kwargs.

The Builder hides `hidden_dims=None` / `activation=...` / `dropout_prob`
(required by the parent NNParams but unused for TRANSFORMER) and
enforces `d_model % n_heads == 0` at `.layers(...)` rather than waiting
for the dataclass's __post_init__.
"""

from __future__ import annotations

import pytest

from nnx.nn.params.nn_transformer_params import NNTransformerParams


def test_builder_minimal_happy_path():
    """Build a tiny LM config the way examples/11 does, but via the
    Builder. The Builder fills in the dead parent-NNParams fields
    (hidden_dims=None, activation=Activations.RELU, dropout_prob=0.0)
    so the user never sees them."""
    params = (
        NNTransformerParams.builder().vocab(1024).layers(n=4, heads=4, d_model=128).context(max_seq_len=128).build()
    )
    # Required fields set by the Builder.
    assert params.vocab_size == 1024
    assert params.input_dim == 1024  # `.vocab()` mirrors into input_dim
    assert params.output_dim == 1024  # and into output_dim
    assert params.n_layers == 4
    assert params.n_heads == 4
    assert params.d_model == 128
    assert params.max_seq_len == 128
    # Defaults preserved (omit-when-default invariant).
    assert params.ffn_mult == 4
    assert params.rope_base == 10000.0
    assert params.tie_embeddings is True
    assert params.attn_dropout == 0.0
    assert params.resid_dropout == 0.0


def test_builder_omit_when_default_invariant():
    """A Builder-produced TransformerParams with only the required
    knobs must emit the same state() as a direct-ctor TransformerParams
    with the same knobs. The 5 omittable LM-specific fields
    (ffn_mult, rope_base, tie_embeddings, attn_dropout, resid_dropout)
    must all be absent from state() when at defaults."""
    built = NNTransformerParams.builder().vocab(1024).layers(n=4, heads=4, d_model=128).context(max_seq_len=128).build()
    state = built.state()
    assert "ffn_mult" not in state
    assert "rope_base" not in state
    assert "tie_embeddings" not in state
    assert "attn_dropout" not in state
    assert "resid_dropout" not in state
    # Round-trip
    assert NNTransformerParams.from_state(state) == built


def test_builder_enforces_heads_divides_d_model_in_layers_call():
    """The `.layers()` method validates `d_model % heads == 0` at call
    time, not at .build() / __post_init__ time. The earlier validation
    is the Builder's safety value-add."""
    with pytest.raises(ValueError, match="must be divisible"):
        NNTransformerParams.builder().layers(n=4, heads=5, d_model=128)
