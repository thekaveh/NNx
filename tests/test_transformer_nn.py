"""Tests for TransformerNN + NNTransformerParams.

Covers:
  * NNTransformerParams round-trip + omit-when-default invariants.
  * TransformerNN forward shape + tied-embedding invariant + param count.
  * Nets.TRANSFORMER enum dispatch through the standard NNModelParams
    construction path (parallels how FeedFwd / GraphConv / GraphAtt are
    discovered).
"""

from __future__ import annotations

import pytest
import torch

from nnx.nn.enum.devices import Devices
from nnx.nn.enum.losses import Losses
from nnx.nn.enum.nets import Nets
from nnx.nn.net.transformer_nn import TransformerNN
from nnx.nn.params.nn_model_params import NNModelParams
from nnx.nn.params.nn_transformer_params import NNTransformerParams


def _params(**overrides) -> NNTransformerParams:
    defaults = dict(
        input_dim=32,
        output_dim=32,
        dropout_prob=0.0,
        vocab_size=32,
        n_layers=2,
        n_heads=4,
        d_model=32,
        ffn_mult=4,
        max_seq_len=16,
    )
    defaults.update(overrides)
    return NNTransformerParams(**defaults)


# ---------------- NNTransformerParams round-trip ----------------


def test_nn_transformer_params_round_trip_minimal():
    """Round-trip with all optional fields at default — state() must
    omit them and the from_state path must restore the defaults."""
    obj = _params()
    rt = NNTransformerParams.from_state(obj.state())
    assert rt == obj


def test_nn_transformer_params_round_trip_full():
    """Round-trip with every optional field bumped off its default."""
    obj = _params(
        rope_base=50000.0,
        tie_embeddings=False,
    )
    rt = NNTransformerParams.from_state(obj.state())
    assert rt == obj


def test_nn_transformer_params_state_omits_rope_base_when_default():
    """Omit-when-default invariant: rope_base=10000.0 is the LLaMA
    default, omit it from state() so existing TRANSFORMER runs don't
    re-hash when we (eventually) add another optional field. Mirrors
    the param_groups / mixed_precision / kind omit-when-default
    pattern. This is the broken-three-times invariant — keep it covered."""
    obj = _params()
    state = obj.state()
    assert "rope_base" not in state, state


def test_nn_transformer_params_state_emits_rope_base_when_overridden():
    obj = _params(rope_base=50000.0)
    state = obj.state()
    assert state.get("rope_base") == 50000.0


def test_nn_transformer_params_state_omits_tie_embeddings_when_default():
    obj = _params()
    state = obj.state()
    assert "tie_embeddings" not in state


def test_nn_transformer_params_state_emits_tie_embeddings_when_overridden():
    obj = _params(tie_embeddings=False)
    state = obj.state()
    assert state.get("tie_embeddings") is False


def test_nn_transformer_params_state_omits_ffn_mult_when_default():
    obj = _params()
    state = obj.state()
    assert "ffn_mult" not in state


def test_nn_transformer_params_state_omits_attn_dropout_when_default():
    """Omit-when-default invariant: attn_dropout=0.0 is the default (modern
    LLM training favors data scale over dropout regularization). Keeping it
    out of state() means existing TRANSFORMER run.ids don't shift when a
    later subproject adds a different optional field. Same broken-three-times
    invariant as rope_base / tie_embeddings / ffn_mult."""
    obj = _params()
    state = obj.state()
    assert "attn_dropout" not in state, state


def test_nn_transformer_params_state_emits_attn_dropout_when_overridden():
    obj = _params(attn_dropout=0.1)
    state = obj.state()
    assert state.get("attn_dropout") == 0.1


def test_nn_transformer_params_state_omits_resid_dropout_when_default():
    """Omit-when-default invariant for resid_dropout — same rationale as
    attn_dropout above."""
    obj = _params()
    state = obj.state()
    assert "resid_dropout" not in state, state


def test_nn_transformer_params_state_emits_resid_dropout_when_overridden():
    obj = _params(resid_dropout=0.05)
    state = obj.state()
    assert state.get("resid_dropout") == 0.05


def test_nn_transformer_params_round_trip_with_dropouts():
    """Round-trip with both new dropout knobs bumped off their defaults —
    every from_state path must restore them. Pairs with the omit-when-default
    tests above to lock in the full invariant for the two transformer-
    dropout fields (attn_dropout, resid_dropout)."""
    obj = _params(attn_dropout=0.1, resid_dropout=0.05)
    rt = NNTransformerParams.from_state(obj.state())
    assert rt == obj


def test_nn_transformer_params_state_emits_required_arch_keys():
    obj = _params()
    state = obj.state()
    # n_layers / d_model / vocab_size / max_seq_len / n_heads are
    # architectural — they always show up in state() because there's
    # no meaningful default for "vocabulary size" or "depth."
    for k in ("n_layers", "d_model", "vocab_size", "max_seq_len", "n_heads"):
        assert k in state, (k, state)


def test_nn_transformer_params_subclasses_nnparams():
    """NNTransformerParams must be a subclass of NNParams so existing
    NNModel code paths (which type-annotate net_params: NNParams) accept
    it. Mirrors the lift-via-subclassing pattern GraphAttNN uses for
    n_heads."""
    from nnx.nn.params.nn_params import NNParams

    assert issubclass(NNTransformerParams, NNParams)


# ---------------- TransformerNN forward + invariants ----------------


def test_transformer_nn_forward_shape():
    net = TransformerNN(params=_params(vocab_size=50, max_seq_len=12, d_model=16, n_heads=2))
    # Token ids in (batch, seq); output is logits (batch, seq, vocab).
    tokens = torch.randint(0, 50, (2, 7))
    logits = net(tokens)
    assert logits.shape == (2, 7, 50), logits.shape


def test_transformer_nn_tie_embeddings_when_default():
    """When tie_embeddings=True (default), the LM head's weight must
    be the SAME tensor as the token embedding's weight (parameter-shared,
    not just equal-valued)."""
    net = TransformerNN(params=_params(vocab_size=50, d_model=16, n_heads=2))
    assert net.tok_embed.weight is net.lm_head.weight


def test_transformer_nn_separate_lm_head_when_untied():
    net = TransformerNN(params=_params(vocab_size=50, d_model=16, n_heads=2, tie_embeddings=False))
    assert net.tok_embed.weight is not net.lm_head.weight
    # Distinct nn.Parameter objects so they receive independent gradients.
    assert id(net.tok_embed.weight) != id(net.lm_head.weight)


def test_transformer_nn_param_count_finite_and_positive():
    """Sanity: a tiny config produces a non-trivial number of parameters
    and the count is finite (catches misconfigured dims that produce
    zero-size linear layers)."""
    net = TransformerNN(params=_params(vocab_size=100, d_model=32, n_heads=4, n_layers=2, max_seq_len=16))
    n_params = sum(p.numel() for p in net.parameters())
    assert n_params > 0
    # Tied embedding: vocab*d_model accounted for once. Two blocks at d=32
    # with attn(4*d*d) + ffn(2 * 4 * 32/3 ~ 84 hidden, 3 mats of 32*84 ~ 8064)
    # plus norms — order ~30k params. Loose bounds:
    assert 5_000 < n_params < 200_000, n_params


def test_transformer_nn_causality_changing_future_doesnt_change_past():
    """The whole-stack causality invariant: tokens at position 0 must
    not depend on tokens at later positions. This wires the per-layer
    causal mask through the full forward pass."""
    torch.manual_seed(0)
    net = TransformerNN(params=_params(vocab_size=20, d_model=16, n_heads=2, n_layers=2, max_seq_len=8))
    net.eval()
    tokens_a = torch.randint(0, 20, (1, 6))
    tokens_b = tokens_a.clone()
    tokens_b[0, 3:] = 0  # change tokens at positions 3, 4, 5
    with torch.no_grad():
        out_a = net(tokens_a)
        out_b = net(tokens_b)
    # Logits at positions 0,1,2 must be identical.
    assert torch.allclose(out_a[:, :3, :], out_b[:, :3, :], atol=1e-5)


def test_transformer_nn_rejects_sequence_longer_than_max_seq_len():
    net = TransformerNN(params=_params(vocab_size=20, max_seq_len=4, d_model=16, n_heads=2))
    tokens = torch.randint(0, 20, (1, 5))
    with pytest.raises(ValueError, match="max_seq_len"):
        net(tokens)


def test_transformer_nn_forward_with_cache_matches_full_forward():
    """Equivalence: a prompt fed through ``forward_with_cache`` in one
    shot must produce the same logits as the same prompt through plain
    ``forward``. This is the unit-level guarantee that the cache seam
    is wired correctly — token-level equivalence in ``generate`` falls
    out of this property at every step."""
    torch.manual_seed(0)
    net = TransformerNN(params=_params(vocab_size=20, d_model=16, n_heads=2, n_layers=3, max_seq_len=16))
    net.eval()
    tokens = torch.randint(0, 20, (1, 5))
    with torch.no_grad():
        logits_full = net(tokens)
        logits_cached, kvs = net.forward_with_cache(tokens, past_kvs=None)
    assert logits_cached.shape == logits_full.shape
    assert torch.allclose(logits_cached, logits_full, atol=1e-5)
    # One KV entry per layer; each is (k, v) with seq dimension == 5.
    assert len(kvs) == 3
    for kv in kvs:
        assert kv is not None
        k, v = kv
        assert k.size(-2) == 5
        assert v.size(-2) == 5


def test_transformer_nn_forward_with_cache_incremental_matches_full():
    """Incremental-decode equivalence: feeding tokens one at a time
    through the cache must produce the same last-position logits as
    feeding the whole sequence through plain ``forward`` in one shot.
    This is the test that catches off-by-one errors in the RoPE offset
    or mask slicing."""
    torch.manual_seed(0)
    net = TransformerNN(params=_params(vocab_size=20, d_model=16, n_heads=2, n_layers=2, max_seq_len=16))
    net.eval()
    tokens = torch.randint(0, 20, (1, 6))

    with torch.no_grad():
        logits_full = net(tokens)  # (1, 6, vocab)

        # Now feed the same tokens one at a time and accumulate the cache.
        past = None
        per_step_logits = []
        for t in range(tokens.size(1)):
            tok = tokens[:, t : t + 1]
            step_logits, past = net.forward_with_cache(tok, past_kvs=past)
            per_step_logits.append(step_logits[:, -1, :])
        cached_seq = torch.stack(per_step_logits, dim=1)  # (1, 6, vocab)

    assert torch.allclose(cached_seq, logits_full, atol=1e-5)


def test_transformer_nn_forward_with_cache_rejects_overflow():
    """Adding new tokens beyond ``max_seq_len`` raises (caller is
    responsible for sliding the cache before the next call)."""
    net = TransformerNN(params=_params(vocab_size=20, max_seq_len=4, d_model=16, n_heads=2, n_layers=1))
    net.eval()
    tokens = torch.randint(0, 20, (1, 4))
    with torch.no_grad():
        _, past = net.forward_with_cache(tokens, past_kvs=None)
        # past is full to max_seq_len; one more token overflows.
        with pytest.raises(ValueError, match="max_seq_len"):
            net.forward_with_cache(torch.randint(0, 20, (1, 1)), past_kvs=past)


def test_transformer_nn_unpack_batch_handles_xy_tuple():
    """TransformerNN must be compatible with the rest of the NNModel
    train loop, which calls .unpack_batch(batch) → ((X,), Y). For LM
    training the standard tuple is (tokens[:-1], tokens[1:])."""
    net = TransformerNN(params=_params(vocab_size=20))
    tokens = torch.randint(0, 20, (2, 5))
    targets = torch.randint(0, 20, (2, 5))
    X, Y = net.unpack_batch((tokens, targets))
    assert isinstance(X, tuple) and len(X) == 1
    assert torch.equal(X[0], tokens)
    assert torch.equal(Y, targets)


# ---------------- Nets.TRANSFORMER enum dispatch ----------------


def test_nets_transformer_enum_exists():
    assert Nets("transformer") is Nets.TRANSFORMER


def test_nets_transformer_enum_constructs_transformer_nn():
    """The enum-as-factory dispatch path that NNModel uses."""
    params = _params(vocab_size=20)
    net = Nets.TRANSFORMER(params=params)
    assert isinstance(net, TransformerNN)


def test_nn_model_params_with_transformer_net_round_trips():
    """Existing NNModelParams.from_state path must accept the new enum
    variant without modification."""
    obj = NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    rt = NNModelParams.from_state(obj.state())
    assert rt == obj


# ---------------- NNRun back-compat (pre-TRANSFORMER run loads) ----------------


def test_pre_transformer_run_yaml_still_loads(tmp_path, monkeypatch):
    """Back-compat: an NNRun saved before the TRANSFORMER variant existed
    must still load via NNRun.load. We simulate this by saving a normal
    FEED_FWD run (which uses no TRANSFORMER-specific keys) and then
    loading it back — the load path must not require any of the new
    NNTransformerParams keys.
    """
    from nnx.nn.enum.activations import Activations
    from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
    from nnx.nn.params.nn_iteration_data_point import NNIterationDataPoint
    from nnx.nn.params.nn_params import NNParams
    from nnx.nn.params.nn_run import NNRun
    from nnx.nn.params.nn_train_params import NNTrainParams

    monkeypatch.chdir(tmp_path)

    run = NNRun(
        net=NNParams(
            input_dim=4,
            output_dim=2,
            dropout_prob=0.0,
            activation=Activations.RELU,
            hidden_dims=[8],
        ),
        train=NNTrainParams(n_epochs=1),
        model=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        idps=[
            NNIterationDataPoint(
                iter_idx=0,
                epoch_idx=0,
                batch_idx=0,
                train_edp=NNEvaluationDataPoint(f1=0.0, recall=0.0, accuracy=0.0, precision=0.0, loss=0.5, error=0.2),
                val_edp=None,
                lr=1e-3,
            )
        ],
    )
    run.save()
    loaded = NNRun.load(id=run.id)
    assert loaded.model.net == Nets.FEED_FWD
    assert loaded.id == run.id  # same hash → no run.id shift
