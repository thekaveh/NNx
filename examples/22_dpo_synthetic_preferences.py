"""DPO demo — fine-tune a tiny TransformerNN against synthetic preference pairs.

Pipeline:

  1. Build a tiny TransformerNN (2 layers, d_model=64) + BPE tokenizer.
  2. Synthesize 64 (prompt, chosen, rejected) triples where the
     "chosen" continuation contains a specific token sequence and the
     "rejected" one doesn't.
  3. Wrap the pretrained model as the frozen reference policy.
  4. Run `dpo_train_step_factory(ref_model=...)` for a handful of epochs.
  5. Check that the model's log-prob ratio for the preferred sequence
     vs the rejected sequence has increased.

Key API notes:
  - `dpo_train_step_factory` expects a frozen `NNModel` (or subclass)
    as the reference policy — not a raw `nn.Module`.
  - `NNPreferenceDataset` takes separate `prompts`, `chosen`, `rejected`
    keyword args; it builds its own train/val DataLoaders internally.
  - `train_bpe` must use the `texts=` keyword (not positional) when
    passing in-memory strings.

Requires the ``lm`` optional extra (for the HuggingFace ``tokenizers``
Rust BPE backing ``NNTokenizerParams`` and ``train_bpe``):

    pip install 'thekaveh-nnx[lm]'

Run:
    python examples/22_dpo_synthetic_preferences.py
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import torch

from nnx import (
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


def _avg_seq_logprob(net: torch.nn.Module, tokens: torch.Tensor) -> float:
    """Average per-token log-probability of the given token sequence.

    Args:
        net: A TransformerNN (or compatible) module that accepts ``(B, T)``
             long tensors and returns ``(B, T, vocab)`` logits.
        tokens: ``(1, T)`` long tensor of token ids.

    Returns:
        Mean log-probability over positions 1..T (next-token prediction).
    """
    net.eval()
    with torch.no_grad():
        logits = net(tokens[:, :-1])  # (1, T-1, V): next-token prediction
        log_probs = torch.log_softmax(logits, dim=-1)
        targets = tokens[:, 1:]  # (1, T-1)
        gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (1, T-1)
    return gathered.mean().item()


def _encode_to_tensor(tokenizer: NNTokenizerParams, text: str, max_len: int = 16) -> torch.Tensor:
    """Encode *text* to a ``(1, T)`` long tensor, truncated to *max_len*."""
    ids = tokenizer.encode(text)[:max_len]
    return torch.tensor([ids], dtype=torch.long)


def _build_model(tokenizer: NNTokenizerParams) -> GenerativeNNModel:
    """Tiny 2-layer TransformerNN; CPU-runnable in < 30 s."""
    vocab_size = tokenizer.vocab_size
    net_params = NNTransformerParams(
        input_dim=vocab_size,
        output_dim=vocab_size,
        vocab_size=vocab_size,
        n_layers=2,
        d_model=64,
        max_seq_len=32,
        n_heads=4,
        hidden_dims=None,
        dropout_prob=0.0,
    )
    model_params = NNModelParams(
        net=Nets.TRANSFORMER,
        device=Devices.CPU,
        loss=Losses.CROSS_ENTROPY,
    )
    return GenerativeNNModel(net_params=net_params, params=model_params, tokenizer=tokenizer)


def main() -> None:
    set_seed(42)

    # ─── Phase 1: tokenizer ───
    corpus = ["good story " * 20, "bad story " * 20, "neutral " * 40] * 5
    tk_raw = train_bpe(texts=corpus, vocab_size=64)
    with tempfile.TemporaryDirectory() as tmpdir:
        tok_path = str(Path(tmpdir) / "tok.json")
        tokenizer = NNTokenizerParams.of(tokenizer=tk_raw, path=tok_path)
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

        # ─── Phase 2: model ───
        model = _build_model(tokenizer)
        n_params = sum(p.numel() for p in model.net.parameters())
        print(f"TransformerNN parameters: {n_params:,}")

        # ─── Phase 3: preference dataset ───
        # Chosen: "good story end"; Rejected: "bad story end".
        prompts = ["story "] * 64
        chosen = ["good story end "] * 64
        rejected = ["bad story end "] * 64
        pref_ds = NNPreferenceDataset(
            prompts=prompts,
            chosen=chosen,
            rejected=rejected,
            tokenizer=tokenizer,
            max_prompt_len=8,
            max_response_len=8,
            # id 1 is "<pad>" in train_bpe's default specials — the
            # dataset default of 0 would pad (and the DPO step would
            # mask) with "<unk>", dropping genuine unknown tokens.
            pad_token_id=1,
            batch_sizes=(8, 4, 4),
            val_proportion=0.1,
            test_proportion=0.1,
            seed=42,
        )
        print(f"Preference dataset: {len(prompts)} triples total")

        # ─── Phase 4: frozen reference policy ───
        ref_model = copy.deepcopy(model)
        ref_model.net.eval()
        for p in ref_model.net.parameters():
            p.requires_grad = False

        # pad_token_id: exclude the dataset's right-padding from the
        # response log-prob sums (NNPreferenceDataset pads with id 0 by
        # default).
        dpo_step = dpo_train_step_factory(ref_model=ref_model, beta=0.1, pad_token_id=pref_ds.pad_token_id)

        # Snapshot log-probs BEFORE DPO (use a deep-copy so training
        # doesn't touch these weights).
        pre_net = copy.deepcopy(model.net)
        chosen_tokens = _encode_to_tensor(tokenizer, "good story end ")
        rejected_tokens = _encode_to_tensor(tokenizer, "bad story end ")
        pre_chosen_lp = _avg_seq_logprob(pre_net, chosen_tokens)
        pre_rejected_lp = _avg_seq_logprob(pre_net, rejected_tokens)

        # ─── Phase 5: DPO training ───
        model.train(
            params=NNTrainParams(
                n_epochs=3,
                train_loader=pref_ds.train_loader,
                optim=NNOptimParams(
                    name=Optims.ADAM,
                    max_lr=1e-4,
                    momentum=(0.9, 0.999),
                    weight_decay=0.0,
                ),
                scheduler=NNSchedulerParams(
                    min_lr=1e-7,
                    factor=0.5,
                    patience=1,
                    cooldown=1,
                    threshold=1e-3,
                ),
            ),
            train_step_fn=dpo_step,
        )

        # ─── Phase 5 verification: before-vs-after log-prob comparison ───
        post_chosen_lp = _avg_seq_logprob(model.net, chosen_tokens)
        post_rejected_lp = _avg_seq_logprob(model.net, rejected_tokens)

        pre_gap = pre_chosen_lp - pre_rejected_lp
        post_gap = post_chosen_lp - post_rejected_lp
        delta = post_gap - pre_gap

    frozen_count = sum(not p.requires_grad for p in ref_model.net.parameters())
    trained_count = sum(p.requires_grad for p in model.net.parameters())
    print("DPO training complete on 64 synthetic preference triples.")
    print(f"Reference policy: frozen ({frozen_count} param tensors).")
    print(f"Trained policy:   updated ({trained_count} param tensors).")
    print(f"Chosen   log-prob — before DPO: {pre_chosen_lp:+.3f},  after DPO: {post_chosen_lp:+.3f}")
    print(f"Rejected log-prob — before DPO: {pre_rejected_lp:+.3f},  after DPO: {post_rejected_lp:+.3f}")
    sign = "+" if delta >= 0 else ""
    print(f"Δ(chosen − rejected): {sign}{delta:.3f}   # positive = preference learned")


if __name__ == "__main__":
    main()
