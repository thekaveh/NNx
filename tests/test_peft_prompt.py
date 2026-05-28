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
