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

    # pad_token_id=1 matches _preference_loader's padding - exercises
    # the 4-site masking threading (policy/ref x chosen/rejected).
    step_fn = dpo_train_step_factory(ref_model, beta=0.1, pad_token_id=1)
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
    # pad_token_id=1 matches _preference_loader's padding - exercises
    # the 4-site masking threading (policy/ref x chosen/rejected).
    step_fn = dpo_train_step_factory(ref_model, beta=0.1, pad_token_id=1)
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


# ---------- reward-accuracy extras ----------


def test_dpo_step_emits_reward_metrics_in_extra(tmp_path, monkeypatch):
    """Every emitted NNEvaluationDataPoint.extra must carry the three
    implicit DPO reward diagnostics — reward_chosen, reward_rejected,
    and reward_accuracy — with reward_accuracy a valid fraction in
    [0, 1]."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    tokenizer = _make_tokenizer(tmp_path)
    ref_model = _make_model(tokenizer)
    policy = _make_model(tokenizer)
    policy.net.load_state_dict(ref_model.net.state_dict())

    loader = _preference_loader(tokenizer, n_pairs=8, batch_size=2)
    step_fn = dpo_train_step_factory(ref_model, beta=0.1, pad_token_id=1)
    run = policy.train(
        params=NNTrainParams(
            n_epochs=3,
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

    assert len(run.idps) > 0
    for idp in run.idps:
        extra = idp.train_edp.extra
        assert "reward_chosen" in extra
        assert "reward_rejected" in extra
        assert "reward_accuracy" in extra
        assert isinstance(extra["reward_chosen"], float)
        assert isinstance(extra["reward_rejected"], float)
        assert isinstance(extra["reward_accuracy"], float)
        assert 0.0 <= extra["reward_accuracy"] <= 1.0


def test_dpo_reward_accuracy_increases_with_training(tmp_path, monkeypatch):
    """DPO should learn to rank chosen above rejected: the mean
    reward_accuracy (fraction of the batch whose implicit chosen reward
    exceeds the rejected reward) of the last epoch must exceed that of
    the first epoch. reward_accuracy starts at 0.0 on the very first
    batch (policy==ref -> all implicit rewards are 0 -> strict > yields
    0.0), then climbs as DPO learns to rank chosen above rejected — so
    the first-EPOCH mean is already non-zero by the time epoch 0 ends.
    We assert the last-epoch mean exceeds the first-epoch mean."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    tokenizer = _make_tokenizer(tmp_path)
    ref_model = _make_model(tokenizer)
    policy = _make_model(tokenizer)
    policy.net.load_state_dict(ref_model.net.state_dict())

    # Larger pair count + finer batch granularity keeps the per-epoch
    # reward_accuracy mean stable enough for the increase assertion.
    loader = _preference_loader(tokenizer, n_pairs=16, batch_size=4)
    step_fn = dpo_train_step_factory(ref_model, beta=0.1, pad_token_id=1)
    run = policy.train(
        params=NNTrainParams(
            n_epochs=12,
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

    by_epoch: dict[int, list[float]] = {}
    for idp in run.idps:
        by_epoch.setdefault(idp.epoch_idx, []).append(idp.train_edp.extra["reward_accuracy"])

    first_epoch = min(by_epoch)
    last_epoch = max(by_epoch)
    first_mean = sum(by_epoch[first_epoch]) / len(by_epoch[first_epoch])
    last_mean = sum(by_epoch[last_epoch]) / len(by_epoch[last_epoch])
    assert last_mean > first_mean, (
        f"DPO reward_accuracy did not increase: epoch {first_epoch} mean {first_mean:.4f} "
        f"vs epoch {last_epoch} mean {last_mean:.4f}"
    )


def test_dpo_reward_chosen_exceeds_rejected_after_training(tmp_path, monkeypatch):
    """The core DPO semantic: after training, the policy assigns a
    higher mean implicit reward to chosen responses than to rejected
    ones. We train long enough for the chosen-minus-rejected margin to
    open up, then assert the mean reward_chosen across the last
    epoch's idps exceeds the mean reward_rejected."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    tokenizer = _make_tokenizer(tmp_path)
    ref_model = _make_model(tokenizer)
    policy = _make_model(tokenizer)
    policy.net.load_state_dict(ref_model.net.state_dict())

    loader = _preference_loader(tokenizer, n_pairs=16, batch_size=4)
    step_fn = dpo_train_step_factory(ref_model, beta=0.1, pad_token_id=1)
    run = policy.train(
        params=NNTrainParams(
            n_epochs=12,
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

    by_epoch: dict[int, list[tuple[float, float]]] = {}
    for idp in run.idps:
        extra = idp.train_edp.extra
        by_epoch.setdefault(idp.epoch_idx, []).append((extra["reward_chosen"], extra["reward_rejected"]))

    last_epoch = max(by_epoch)
    last_pairs = by_epoch[last_epoch]
    mean_chosen = sum(rc for rc, _ in last_pairs) / len(last_pairs)
    mean_rejected = sum(rr for _, rr in last_pairs) / len(last_pairs)
    assert mean_chosen > mean_rejected, (
        f"DPO did not learn to prefer chosen over rejected: last-epoch mean reward_chosen "
        f"{mean_chosen:.4f} vs mean reward_rejected {mean_rejected:.4f}"
    )


def test_dpo_reward_accuracy_matches_manual_computation(tmp_path, monkeypatch):
    """The reward_accuracy emitted in extra must equal a manual
    computation of β·(logπ_θ − logπ_ref) ranked chosen vs rejected,
    using the policy's PRE-step weights — the step computes its policy
    log-probs before finalize_step mutates the weights, so the emitted
    metric reflects the same pre-step state."""
    from nnx.nn.nn_model import TrainStepContext
    from nnx.paradigms.dpo import _response_logprob

    monkeypatch.chdir(tmp_path)
    set_seed(0)
    tokenizer = _make_tokenizer(tmp_path)
    ref_model = _make_model(tokenizer)
    policy = _make_model(tokenizer)
    policy.net.load_state_dict(ref_model.net.state_dict())

    beta = 0.1
    pad_token_id = 1
    loader = _preference_loader(tokenizer, n_pairs=4, batch_size=4)
    prompt_ids, chosen_ids, rejected_ids = next(iter(loader))

    prompt_len = prompt_ids.shape[1]
    chosen_seq = torch.cat([prompt_ids, chosen_ids], dim=1)
    rejected_seq = torch.cat([prompt_ids, rejected_ids], dim=1)

    # The factory freezes the reference and pins it to eval — call it
    # first so the manual ref log-probs are computed under the same
    # (eval, no-grad) state the step uses.
    step_fn = dpo_train_step_factory(ref_model, beta=beta, pad_token_id=pad_token_id)

    # Mirror the step's exact log-prob computation under the PRE-step
    # policy weights (policy.net.train() + zero_grad, matching step()).
    policy.net.train()
    policy.net.zero_grad()
    policy_chosen_logp = _response_logprob(policy.net, chosen_seq, prompt_len, pad_token_id=pad_token_id)
    policy_rejected_logp = _response_logprob(policy.net, rejected_seq, prompt_len, pad_token_id=pad_token_id)
    with torch.no_grad():
        ref_chosen_logp = _response_logprob(ref_model.net, chosen_seq, prompt_len, pad_token_id=pad_token_id)
        ref_rejected_logp = _response_logprob(ref_model.net, rejected_seq, prompt_len, pad_token_id=pad_token_id)

    reward_chosen = beta * (policy_chosen_logp.detach() - ref_chosen_logp)
    reward_rejected = beta * (policy_rejected_logp.detach() - ref_rejected_logp)
    expected_reward_accuracy = float((reward_chosen > reward_rejected).float().mean())

    # Run the actual step on the same batch. The step mutates the policy
    # via optimizer.step, but its emitted reward_accuracy was computed
    # from the PRE-step policy log-probs above.
    optimizer = torch.optim.Adam(policy.net.parameters(), lr=5e-3)
    ctx = TrainStepContext(
        model=policy,
        batch=(prompt_ids, chosen_ids, rejected_ids),
        optimizer=optimizer,
        scaler=None,
        grad_clip_norm=None,
        extra_metrics=None,
        accumulate_grad_batches=1,
        batch_idx=0,
        epoch_idx=0,
    )
    edp = step_fn(ctx)

    assert edp.extra["reward_accuracy"] == pytest.approx(expected_reward_accuracy)
    # reward_chosen/rejected are batch means of the same manual tensors.
    assert edp.extra["reward_chosen"] == pytest.approx(float(reward_chosen.mean()))
    assert edp.extra["reward_rejected"] == pytest.approx(float(reward_rejected.mean()))
