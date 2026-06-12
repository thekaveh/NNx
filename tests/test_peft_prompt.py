"""Tests for nnx.peft.prompt — PromptTuner + save/load."""

from __future__ import annotations

import pytest
import torch

from nnx import (
    Activations,
    NNTransformerParams,
    PromptTuner,
    TransformerNN,
    load_prompt_weights,
    save_prompt_weights,
    set_seed,
)


def _tiny_transformer() -> TransformerNN:
    """Small TransformerNN fixture — kept tiny for test speed."""
    params = NNTransformerParams(
        input_dim=100,
        output_dim=100,
        dropout_prob=0.0,
        activation=Activations.RELU,
        n_heads=4,
        vocab_size=100,
        n_layers=2,
        d_model=32,
        max_seq_len=64,
    )
    return TransformerNN(params)


def test_prompt_tuner_freezes_base():
    """Every parameter of the wrapped TransformerNN must be frozen
    on construction — the PEFT contract."""
    set_seed(0)
    model = _tiny_transformer()
    tuner = PromptTuner(model, n_prompt_tokens=8)
    for name, p in tuner.model.named_parameters():
        assert not p.requires_grad, f"base parameter {name!r} not frozen"
    # The soft prompt itself IS trainable.
    assert tuner.soft_prompt.requires_grad


def test_prompt_tuner_only_prompt_trainable():
    """`trainable_parameters()` must yield exactly the soft-prompt
    tensor and nothing else."""
    set_seed(0)
    model = _tiny_transformer()
    tuner = PromptTuner(model, n_prompt_tokens=8)
    trainable = list(tuner.trainable_parameters())
    assert len(trainable) == 1
    assert trainable[0] is tuner.soft_prompt


def test_prompt_tuner_forward_shape_unchanged():
    """The wrapper returns logits over only the real input tokens —
    the soft-prompt positions are scaffolding and get trimmed off.
    Output shape must therefore match the base model's exactly."""
    set_seed(0)
    model = _tiny_transformer()
    tuner = PromptTuner(model, n_prompt_tokens=8)
    # n_prompt + t (16) = 24 <= max_seq_len (64).
    tokens = torch.randint(0, model.params.vocab_size, (2, 16))
    out = tuner(tokens)
    assert out.shape == (2, 16, model.params.vocab_size)


def test_prompt_tuner_save_load_round_trip(tmp_path):
    """Save the soft prompt, load into a fresh tuner, verify the
    loaded values match bit-exactly."""
    set_seed(0)
    model_a = _tiny_transformer()
    tuner_a = PromptTuner(model_a, n_prompt_tokens=8)
    with torch.no_grad():
        tuner_a.soft_prompt.fill_(0.31)

    path = save_prompt_weights(tuner_a, tmp_path / "prompt.pt")
    assert path.endswith("prompt.pt")

    set_seed(1)
    model_b = _tiny_transformer()
    tuner_b = PromptTuner(model_b, n_prompt_tokens=8)
    # Pre-load: tuner_b.soft_prompt is at its random init, NOT 0.31.
    assert not torch.allclose(tuner_a.soft_prompt, tuner_b.soft_prompt)

    n_loaded = load_prompt_weights(tuner_b, path)
    assert n_loaded > 0
    assert torch.equal(tuner_a.soft_prompt.detach(), tuner_b.soft_prompt.detach())


def test_prompt_tuner_validates_n_prompt_tokens():
    """n_prompt_tokens <= 0 must raise — a zero-token soft prompt is
    a no-op."""
    model = _tiny_transformer()
    with pytest.raises(ValueError, match="n_prompt_tokens"):
        PromptTuner(model, n_prompt_tokens=0)
    with pytest.raises(ValueError, match="n_prompt_tokens"):
        PromptTuner(model, n_prompt_tokens=-1)


def test_prompt_tuned_generate_survives_window_overflow(tmp_path):
    """generate() sized its sliding window from net_params.max_seq_len,
    but a PromptTuner consumes n_prompt_tokens of those slots — long
    generations crashed mid-stream once the window plus the soft prompt
    exceeded the wrapped model's max_seq_len. generate() now honors the
    wrapper's effective_max_seq_len."""
    pytest.importorskip("tokenizers")
    from nnx import GenerativeNNModel, NNModelParams
    from nnx.nn.enum.devices import Devices
    from nnx.nn.enum.losses import Losses
    from nnx.nn.enum.nets import Nets
    from nnx.nn.params.nn_tokenizer_params import NNTokenizerParams, train_bpe

    set_seed(0)
    params = NNTransformerParams(
        input_dim=64,
        output_dim=64,
        dropout_prob=0.0,
        vocab_size=64,
        n_layers=1,
        n_heads=2,
        d_model=16,
        ffn_mult=2,
        max_seq_len=32,
    )
    tk = train_bpe(
        files=None,
        texts=["the cat sat on the mat", "the dog ran in the park"],
        vocab_size=64,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
    )
    tokenizer = NNTokenizerParams.of(tokenizer=tk, path=str(tmp_path / "tok.json"))
    model = GenerativeNNModel(
        net_params=params,
        params=NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        tokenizer=tokenizer,
    )
    model.net = PromptTuner(model.net, n_prompt_tokens=4)
    assert model.net.effective_max_seq_len == 28
    # 40 new tokens forces the sliding window well past 28 — pre-fix
    # this raised ValueError mid-generation.
    out = model.generate(prompt="the", max_new_tokens=40, temperature=0.0)
    assert isinstance(out, str)
