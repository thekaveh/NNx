"""Tests for nnx.paradigms.dpo — Direct Preference Optimization."""

from __future__ import annotations

import copy

import pytest
import torch

pytest.importorskip("tokenizers")

from nnx import (  # noqa: E402
    Devices,
    GenerativeNNModel,
    Losses,
    Nets,
    NNModelParams,
    NNOptimParams,
    NNPreferenceDataset,
    NNSchedulerParams,
    NNTokenizerParams,
    NNTrainParams,
    NNTransformerParams,
    Optims,
    dpo_train_step_factory,
    set_seed,
    train_bpe,
)

# ---------- helpers ----------


def _make_tokenizer(tmp_path):
    """Train a tiny BPE on a small synthetic corpus. Wide enough for
    'good' vs 'bad' continuations to share substrings with real BPE
    units."""
    corpus = [
        "the cat sat on the mat",
        "the cat is happy and warm",
        "the dog ran in the park",
        "the dog is loud and chaotic",
        "the world is round and gentle",
        "the world is full of pain",
        "hello there friend",
        "hello there enemy",
        "good morning sunshine",
        "bad morning thunderstorm",
    ]
    tk = train_bpe(
        files=None,
        texts=corpus,
        vocab_size=80,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
    )
    path = tmp_path / "tok.json"
    return NNTokenizerParams.of(tokenizer=tk, path=str(path))


def _make_model(tokenizer: NNTokenizerParams) -> GenerativeNNModel:
    net_params = NNTransformerParams(
        input_dim=tokenizer.vocab_size,
        output_dim=tokenizer.vocab_size,
        dropout_prob=0.0,
        vocab_size=tokenizer.vocab_size,
        n_layers=2,
        n_heads=2,
        d_model=16,
        ffn_mult=2,
        max_seq_len=64,
    )
    model_params = NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    return GenerativeNNModel(net_params=net_params, params=model_params, tokenizer=tokenizer)


def _preference_loader(tokenizer: NNTokenizerParams, n_pairs: int = 8, batch_size: int = 4):
    prompts = ["the cat", "the dog", "the world", "hello there", "good morning"] * (n_pairs // 5 + 1)
    chosen = ["is happy and warm", "is in the park", "is round and gentle", "friend", "sunshine"] * (n_pairs // 5 + 1)
    rejected = ["sat on the mat", "is loud and chaotic", "is full of pain", "enemy", "thunderstorm"] * (
        n_pairs // 5 + 1
    )
    prompts, chosen, rejected = prompts[:n_pairs], chosen[:n_pairs], rejected[:n_pairs]
    ds = NNPreferenceDataset(
        prompts=prompts,
        chosen=chosen,
        rejected=rejected,
        tokenizer=tokenizer,
        max_prompt_len=8,
        max_response_len=8,
        pad_token_id=1,  # "<pad>" lives at id 1 in our trainer's special token list
        batch_sizes=(batch_size, batch_size, batch_size),
        val_proportion=0.0,
        test_proportion=0.0,
        seed=0,
    )
    return ds.train_loader


# ---------- factory validation ----------


def test_dpo_factory_validates_beta(tmp_path):
    tokenizer = _make_tokenizer(tmp_path)
    ref_model = _make_model(tokenizer)
    with pytest.raises(ValueError, match="beta"):
        dpo_train_step_factory(ref_model, beta=0.0)
    with pytest.raises(ValueError, match="beta"):
        dpo_train_step_factory(ref_model, beta=-0.5)


def test_dpo_factory_freezes_reference_params(tmp_path):
    tokenizer = _make_tokenizer(tmp_path)
    ref_model = _make_model(tokenizer)
    # Reference params start trainable.
    assert all(p.requires_grad for p in ref_model.net.parameters())
    dpo_train_step_factory(ref_model, beta=0.1)
    assert all(not p.requires_grad for p in ref_model.net.parameters())


def test_dpo_factory_puts_reference_in_eval_mode(tmp_path):
    tokenizer = _make_tokenizer(tmp_path)
    ref_model = _make_model(tokenizer)
    ref_model.net.train()
    dpo_train_step_factory(ref_model, beta=0.1)
    assert not ref_model.net.training


# ---------- dataset shape ----------


def test_nn_preference_dataset_yields_correct_shape(tmp_path):
    tokenizer = _make_tokenizer(tmp_path)
    loader = _preference_loader(tokenizer, n_pairs=8, batch_size=4)
    batch = next(iter(loader))
    assert isinstance(batch, (list, tuple))
    assert len(batch) == 3
    prompt_ids, chosen_ids, rejected_ids = batch
    assert isinstance(prompt_ids, torch.Tensor)
    assert isinstance(chosen_ids, torch.Tensor)
    assert isinstance(rejected_ids, torch.Tensor)
    assert prompt_ids.dtype == torch.long
    assert chosen_ids.dtype == torch.long
    assert rejected_ids.dtype == torch.long
    # Batched 2D shape (B, T_*) — dataset padded/truncated to max_*_len.
    assert prompt_ids.dim() == 2
    assert chosen_ids.dim() == 2
    assert rejected_ids.dim() == 2
    assert prompt_ids.shape[0] == 4
    assert prompt_ids.shape[1] == 8  # max_prompt_len
    assert chosen_ids.shape[1] == 8
    assert rejected_ids.shape[1] == 8


def test_nn_preference_dataset_validates_inputs(tmp_path):
    tokenizer = _make_tokenizer(tmp_path)
    with pytest.raises(ValueError, match="align"):
        NNPreferenceDataset(
            prompts=["a", "b"],
            chosen=["c"],
            rejected=["d", "e"],
            tokenizer=tokenizer,
            val_proportion=0.0,
            test_proportion=0.0,
        )
    with pytest.raises(ValueError, match="non-empty"):
        NNPreferenceDataset(
            prompts=[],
            chosen=[],
            rejected=[],
            tokenizer=tokenizer,
            val_proportion=0.0,
            test_proportion=0.0,
        )


# ---------- end-to-end training ----------


def test_dpo_step_reduces_chosen_rejected_logprob_gap(tmp_path, monkeypatch):
    """After a few DPO steps, the policy should assign higher log-prob
    to chosen than to rejected responses across the training set,
    relative to the starting (= reference) gap."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    tokenizer = _make_tokenizer(tmp_path)

    # The reference policy is the SFT checkpoint. To exercise DPO end-
    # to-end we use the same architecture for policy and reference and
    # snapshot the reference's weights so policy and reference start
    # identical.
    ref_model = _make_model(tokenizer)
    policy = _make_model(tokenizer)
    policy.net.load_state_dict(ref_model.net.state_dict())

    loader = _preference_loader(tokenizer, n_pairs=8, batch_size=2)

    # Measure the gap (chosen − rejected log-prob) BEFORE training,
    # under the policy (which == reference at this point).
    def _logp(net, seq, prompt_len):
        logits = net(seq)
        log_probs = torch.log_softmax(logits, dim=-1)
        resp_logits = log_probs[:, prompt_len - 1 : -1, :]
        resp_targets = seq[:, prompt_len:]
        return resp_logits.gather(dim=-1, index=resp_targets.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

    def _compute_gap(net: torch.nn.Module) -> float:
        net.eval()
        gaps = []
        with torch.no_grad():
            for prompt_ids, chosen_ids, rejected_ids in loader:
                prompt_len = prompt_ids.shape[1]
                chosen_seq = torch.cat([prompt_ids, chosen_ids], dim=1)
                rejected_seq = torch.cat([prompt_ids, rejected_ids], dim=1)
                chosen_lp = _logp(net, chosen_seq, prompt_len)
                rejected_lp = _logp(net, rejected_seq, prompt_len)
                gaps.append((chosen_lp - rejected_lp).mean().item())
        return sum(gaps) / len(gaps)

    initial_gap = _compute_gap(policy.net)

    step_fn = dpo_train_step_factory(ref_model, beta=0.1)
    policy.train(
        params=NNTrainParams(
            n_epochs=8,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=5e-3,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7,
                factor=0.5,
                patience=10,
                cooldown=1,
                threshold=1e-3,
            ),
        ),
        train_step_fn=step_fn,
    )

    final_gap = _compute_gap(policy.net)
    # The chosen-minus-rejected gap must strictly increase under the
    # trained policy — that's the contract DPO trains toward.
    assert final_gap > initial_gap, (
        f"DPO did not increase the chosen−rejected log-prob gap: initial {initial_gap:.4f} vs final {final_gap:.4f}"
    )


def test_dpo_ref_model_stays_frozen(tmp_path, monkeypatch):
    """The reference model's parameters must NEVER receive a gradient
    update during policy training — snapshot weights before and check
    bit-for-bit equality after."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    tokenizer = _make_tokenizer(tmp_path)
    ref_model = _make_model(tokenizer)
    policy = _make_model(tokenizer)
    policy.net.load_state_dict(ref_model.net.state_dict())

    ref_snapshot = copy.deepcopy({k: v.clone() for k, v in ref_model.net.state_dict().items()})

    loader = _preference_loader(tokenizer, n_pairs=6, batch_size=2)
    step_fn = dpo_train_step_factory(ref_model, beta=0.1)
    policy.train(
        params=NNTrainParams(
            n_epochs=4,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=5e-3,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7,
                factor=0.5,
                patience=10,
                cooldown=1,
                threshold=1e-3,
            ),
        ),
        train_step_fn=step_fn,
    )
    # Reference weights are unchanged bit-for-bit.
    for k, v in ref_model.net.state_dict().items():
        assert torch.equal(v, ref_snapshot[k]), (
            f"reference param {k!r} drifted during policy training — "
            "dpo_train_step_factory must keep the reference frozen"
        )
    # And requires_grad stays cleared.
    assert all(not p.requires_grad for p in ref_model.net.parameters())


def test_response_logprob_excludes_pad_positions():
    """Right-padded responses must not be scored on their padding:
    extending a response with extra pad tokens leaves its log-prob
    unchanged when pad_token_id is passed. Pre-fix, every pad position
    was summed in — the terms don't cancel between policy/reference or
    chosen/rejected, biasing the DPO objective and training the policy
    to emit pads after short responses."""
    from torch import nn

    from nnx.paradigms.dpo import _response_logprob

    torch.manual_seed(0)
    vocab = 11
    net = nn.Embedding(vocab, vocab)  # (B, T) -> (B, T, vocab) logits stub
    prompt = torch.tensor([[3, 4]])
    resp = torch.tensor([[5, 6]])
    pads = torch.full((1, 3), 7)
    short = torch.cat([prompt, resp], dim=1)
    padded = torch.cat([prompt, resp, pads], dim=1)

    lp_short = _response_logprob(net, short, 2, pad_token_id=7)
    lp_padded = _response_logprob(net, padded, 2, pad_token_id=7)
    assert torch.allclose(lp_short, lp_padded, atol=1e-6)

    # Without the mask, the pads ARE scored and the totals differ.
    lp_unmasked = _response_logprob(net, padded, 2)
    assert not torch.allclose(lp_short, lp_unmasked, atol=1e-4)
