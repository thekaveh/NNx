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
    (hidden_dims=None, activation=Activations.LEAKY_RELU — the
    NNParams default, dropout_prob=0.0) so the user never sees them."""
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


def test_builder_ffn_override():
    params = (
        NNTransformerParams.builder()
        .vocab(1024)
        .layers(n=4, heads=4, d_model=128)
        .context(max_seq_len=128)
        .ffn(mult=8)
        .build()
    )
    assert params.ffn_mult == 8
    # ffn_mult != default → it MUST appear in state().
    assert params.state().get("ffn_mult") == 8


def test_builder_context_with_rope_base_override():
    params = (
        NNTransformerParams.builder()
        .vocab(1024)
        .layers(n=4, heads=4, d_model=128)
        .context(max_seq_len=2048, rope_base=500000.0)  # long-context recipe
        .build()
    )
    assert params.max_seq_len == 2048
    assert params.rope_base == 500000.0
    assert params.state().get("rope_base") == 500000.0


def test_builder_dropout_non_default():
    params = (
        NNTransformerParams.builder()
        .vocab(1024)
        .layers(n=4, heads=4, d_model=128)
        .context(max_seq_len=128)
        .dropout(attn=0.1, resid=0.05)
        .build()
    )
    assert params.attn_dropout == 0.1
    assert params.resid_dropout == 0.05
    state = params.state()
    assert state.get("attn_dropout") == 0.1
    assert state.get("resid_dropout") == 0.05


def test_builder_build_without_vocab_raises():
    """Calling .build() before .vocab() / .layers() / .context() raises
    an actionable Builder-level ValueError naming the missing setter
    methods — matches the [[builder-pattern-shape]] §11b convention PR
    #52 established on NNTrainerParamsBuilder. The error message must
    reference the Builder methods, not the dataclass fields, so the
    user knows what to call next.
    """
    with pytest.raises(ValueError, match=r"NNTransformerParamsBuilder.*\.vocab.*\.layers.*\.context"):
        NNTransformerParams.builder().build()


def test_builder_vocab_mirrors_into_both_input_and_output_dim():
    """Property test: for every vocab size, `.vocab(N)` writes the
    same N into all three of vocab_size, input_dim, output_dim. A
    regression where one of the three writes drifted (e.g. someone
    refactored .vocab() to skip input_dim) would be caught here."""
    for size in (512, 1024, 2048, 50257):
        params = (
            NNTransformerParams.builder().vocab(size).layers(n=2, heads=2, d_model=32).context(max_seq_len=32).build()
        )
        assert params.vocab_size == size
        assert params.input_dim == size
        assert params.output_dim == size


def test_builder_vocab_called_twice_last_wins():
    """`.vocab(512).vocab(1024)` keeps 1024 (the last call). The
    standard fluent contract. A regression that switched to "first
    wins" or "raises on second call" would be caught here."""
    params = (
        NNTransformerParams.builder()
        .vocab(512)
        .vocab(1024)
        .layers(n=2, heads=2, d_model=32)
        .context(max_seq_len=32)
        .build()
    )
    assert params.vocab_size == 1024
    assert params.input_dim == 1024
    assert params.output_dim == 1024


def test_builder_layers_called_twice_last_wins():
    """`.layers(...)` last-call-wins."""
    params = (
        NNTransformerParams.builder()
        .vocab(1024)
        .layers(n=2, heads=2, d_model=32)
        .layers(n=4, heads=4, d_model=128)
        .context(max_seq_len=128)
        .build()
    )
    assert params.n_layers == 4
    assert params.n_heads == 4
    assert params.d_model == 128


def test_builder_state_equals_direct_ctor_with_parent_defaults():
    """Regression test: the Builder's `.build()` hardcodes the dead
    parent-NNParams fields (`hidden_dims=None`, `dropout_prob=0.0`,
    `activation=...`). The activation MUST match the parent NNParams
    default (`Activations.LEAKY_RELU`), otherwise the Builder and the
    direct-kwarg ctor produce different `state()` dicts and different
    `run.id` hashes for what users would call "the same config".

    Pre-fix the Builder used `Activations.RELU` which differed from
    the parent `LEAKY_RELU` default — this test would have failed
    against the buggy version.
    """
    from nnx.nn.enum.activations import Activations

    built = NNTransformerParams.builder().vocab(1024).layers(n=4, heads=4, d_model=128).context(max_seq_len=128).build()
    direct = NNTransformerParams(
        vocab_size=1024,
        input_dim=1024,
        output_dim=1024,
        n_layers=4,
        n_heads=4,
        d_model=128,
        max_seq_len=128,
        hidden_dims=None,
        dropout_prob=0.0,
        activation=Activations.LEAKY_RELU,
    )
    assert built == direct
    assert built.state() == direct.state()


def test_builder_tied_embeddings_false_round_trips():
    params = (
        NNTransformerParams.builder()
        .vocab(1024)
        .layers(n=4, heads=4, d_model=128)
        .context(max_seq_len=128)
        .tied_embeddings(False)
        .build()
    )
    assert params.tie_embeddings is False
    state = params.state()
    assert state.get("tie_embeddings") is False
    # Round-trip
    assert NNTransformerParams.from_state(state) == params


def test_builder_tied_embeddings_true_after_false_overrides_to_true():
    """Regression: a prior `.tied_embeddings(False)` followed by
    `.tied_embeddings(True)` must leave the dataclass at the default
    (True). Pre-fix the True call was a silent no-op because the body
    skipped storing when `value is True`, so the prior False persisted.
    state() must omit `tie_embeddings` at the default."""
    params = (
        NNTransformerParams.builder()
        .vocab(1024)
        .layers(n=4, heads=4, d_model=128)
        .context(max_seq_len=128)
        .tied_embeddings(False)
        .tied_embeddings(True)
        .build()
    )
    assert params.tie_embeddings is True
    assert "tie_embeddings" not in params.state()


def test_builder_dropout_reset_to_default_after_non_default():
    """Regression: `.dropout(attn=0.5).dropout(attn=0.0)` must leave
    attn_dropout at 0.0 (the default), not at 0.5 from the prior call.
    Pre-fix the second call was a silent no-op because the body gated
    on `if attn != 0.0:`, so the prior 0.5 persisted — breaking the
    fluent "last call wins" contract that every other setter on this
    Builder honors (vocab/layers/ffn/tied_embeddings). state() must
    omit `attn_dropout` at the default."""
    params = (
        NNTransformerParams.builder()
        .vocab(1024)
        .layers(n=4, heads=4, d_model=128)
        .context(max_seq_len=128)
        .dropout(attn=0.5, resid=0.3)
        .dropout(attn=0.0, resid=0.0)
        .build()
    )
    assert params.attn_dropout == 0.0
    assert params.resid_dropout == 0.0
    state = params.state()
    assert "attn_dropout" not in state
    assert "resid_dropout" not in state


def test_builder_context_rope_base_reset_to_default_after_override():
    """Regression: `.context(max_seq_len=2048, rope_base=500000.0)`
    followed by `.context(max_seq_len=2048)` must leave rope_base at
    the dataclass default (10000.0). Pre-fix the second call (with the
    `rope_base=None` sentinel) was a silent no-op because the body
    gated on `if rope_base is not None:`, so the prior override
    persisted — breaking the fluent "last call wins" contract. state()
    must omit `rope_base` at the default."""
    params = (
        NNTransformerParams.builder()
        .vocab(1024)
        .layers(n=4, heads=4, d_model=128)
        .context(max_seq_len=2048, rope_base=500000.0)
        .context(max_seq_len=2048)
        .build()
    )
    assert params.rope_base == 10000.0
    assert "rope_base" not in params.state()
