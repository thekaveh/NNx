"""Train a tiny decoder-only LM on TinyStories — end-to-end demo of
the SP-4 TransformerNN path.

Pipeline:

  1. Load TinyStories (or a synthetic fallback corpus for offline runs).
  2. Train a tiny BPE tokenizer on a 1% subset.
  3. Build a ~10M-param TransformerNN + NNTokenizerParams.
  4. Train via the standard `NNModel.train()` loop with a custom
     next-token train_step.
  5. Call `model.generate("Once upon a time")` to confirm the model
     produces vaguely-story-like continuations.

Scope explicit: TinyStories-class CPU-friendly demo. A full TinyStories
epoch on a laptop is hours; this example trains on a 1% subset for a
sub-30-min CPU run. Production-scale training (multi-GPU, FlashAttention)
is out of scope for SP-4.

Run:
    python examples/11_tinystories_lm.py
    # or pass --use-hf to download TinyStories from HuggingFace:
    NNX_USE_HF=1 python examples/11_tinystories_lm.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from nnx import (
    Devices,
    Losses,
    Nets,
    NNModelParams,
    NNOptimParams,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    TrainStepContext,
    set_seed,
)
from nnx.nn.generative_nn_model import GenerativeNNModel
from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from nnx.nn.params.nn_tokenizer_params import NNTokenizerParams, train_bpe
from nnx.nn.params.nn_transformer_params import NNTransformerParams

# A small inline corpus — used when the user doesn't have datasets/HF
# access. Story-like sentences so the BPE tokenizer learns something
# vaguely English-like.
_FALLBACK_CORPUS = [
    "Once upon a time there was a little cat named tom.",
    "Tom lived in a small house with a big garden.",
    "Every day the cat would chase butterflies in the garden.",
    "One sunny morning tom met a friendly bird named lucy.",
    "Lucy could sing the most beautiful songs in the forest.",
    "The cat and the bird became the best of friends.",
    "They would play together every day under the big tree.",
    "Sometimes they would have small adventures in the woods.",
    "Tom loved chasing leaves while lucy sang sweet melodies.",
    "When evening came tom would go home to drink milk.",
    "Lucy would fly back to her warm cozy nest in the tree.",
    "And so the days passed in their happy little village.",
    "Once there was a brave knight who lived in a castle.",
    "The knight had a kind heart and a strong horse.",
    "Every day the knight helped people in the small village.",
    "The villagers loved the knight for being so brave and kind.",
    "One day a small dragon came to visit the village.",
    "The dragon was not scary it just wanted to play.",
    "The knight and the dragon became friends in the end.",
    "And everyone in the village lived happily ever after.",
] * 50  # repeat so BPE sees enough frequency to learn merges


def _load_corpus(use_hf: bool, n_lines: int) -> list[str]:
    """Try TinyStories via `datasets`; fall back to the inline corpus.

    The fallback is sufficient for the smoke-test path; the HF path is
    what you'd run for a real (but still tiny) training run.
    """
    if not use_hf:
        return list(_FALLBACK_CORPUS)
    try:
        from datasets import load_dataset

        ds = load_dataset("roneneldan/TinyStories", split=f"train[:{n_lines}]")
        return [str(row["text"]) for row in ds]
    except Exception as e:  # noqa: BLE001
        print(f"[tinystories] HF download failed ({e}); falling back to inline corpus")
        return list(_FALLBACK_CORPUS)


class _LMDataset(Dataset):
    """Slice a token stream into fixed-length context windows.

    Each sample is ``(x, y)`` where ``y = x.roll(-1)`` — next-token
    targets. We don't add BOS/EOS framing; the BPE training does that
    via the special tokens vocab.
    """

    def __init__(self, token_ids: list[int], seq_len: int):
        self.seq_len = seq_len
        # Drop the trailing remainder so every window is full-length.
        n_windows = (len(token_ids) - 1) // seq_len
        self.x = torch.tensor(token_ids[: n_windows * seq_len + 1], dtype=torch.long)

    def __len__(self) -> int:
        return (len(self.x) - 1) // self.seq_len

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        x = self.x[start : start + self.seq_len]
        y = self.x[start + 1 : start + self.seq_len + 1]
        return x, y


def _lm_train_step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
    """Next-token cross-entropy training step.

    Wraps the standard supervised step but flattens (B, T, V) → (B*T, V)
    so torch's cross_entropy can apply along the vocab dim.
    """
    model = ctx.model
    model.net.train()
    optimizer = ctx.optimizer
    if (ctx.batch_idx % ctx.accumulate_grad_batches) == 0:
        model.net.zero_grad()

    X, Y = ctx.batch
    X = X.to(model.device)
    Y = Y.to(model.device)
    logits = model.net(X)  # (B, T, V)
    b, t, v = logits.shape
    loss = torch.nn.functional.cross_entropy(logits.reshape(b * t, v), Y.reshape(b * t))
    (loss / ctx.accumulate_grad_batches).backward()
    if ((ctx.batch_idx + 1) % ctx.accumulate_grad_batches) == 0:
        if ctx.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), ctx.grad_clip_norm)
        optimizer.step()

    # We don't compute argmax-accuracy here (vocab is large, accuracy
    # is a weak signal for LM training). Return loss + a placeholder
    # error so the framework's NNEvaluationDataPoint stays uniform.
    return NNEvaluationDataPoint(
        loss=float(loss.detach()),
        error=float(loss.detach()),
        accuracy=0.0,
        f1=0.0,
        recall=0.0,
        precision=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-hf", action="store_true", help="download TinyStories from HuggingFace")
    parser.add_argument("--n-lines", type=int, default=2000, help="how many lines to slice off the dataset")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    use_hf = args.use_hf or os.environ.get("NNX_USE_HF") == "1"

    set_seed(0)

    # --- 1. Corpus ---
    corpus = _load_corpus(use_hf=use_hf, n_lines=args.n_lines)
    print(f"[tinystories] corpus: {len(corpus)} lines")

    # --- 2. BPE tokenizer ---
    artifacts_dir = Path("artifacts/tinystories_lm")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tk = train_bpe(
        files=None,
        texts=corpus,
        vocab_size=args.vocab_size,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
    )
    tokenizer = NNTokenizerParams.of(tokenizer=tk, path=str(artifacts_dir / "tokenizer.json"))
    print(f"[tinystories] tokenizer vocab_size={tokenizer.vocab_size}")

    # --- 3. Tokenize the corpus into a single id stream ---
    all_ids: list[int] = []
    for line in corpus:
        all_ids.extend(tokenizer.encode(line))
    print(f"[tinystories] total tokens: {len(all_ids):,}")

    ds = _LMDataset(token_ids=all_ids, seq_len=args.seq_len)
    train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # --- 4. Model ---
    net_params = NNTransformerParams(
        input_dim=tokenizer.vocab_size,
        output_dim=tokenizer.vocab_size,
        dropout_prob=0.0,
        vocab_size=tokenizer.vocab_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        ffn_mult=4,
        max_seq_len=args.seq_len,
    )
    model_params = NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    model = GenerativeNNModel(net_params=net_params, params=model_params, tokenizer=tokenizer)
    n_params = sum(p.numel() for p in model.net.parameters())
    print(f"[tinystories] model parameters: {n_params:,}")

    # --- 5. Train ---
    model.train(
        params=NNTrainParams(
            n_epochs=args.n_epochs,
            train_loader=train_loader,
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=args.lr,
                momentum=(0.9, 0.95),
                weight_decay=0.0,
                grad_clip_norm=1.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-6,
                factor=0.5,
                patience=1,
                cooldown=0,
                threshold=1e-3,
            ),
            seed=0,
        ),
        train_step_fn=_lm_train_step,
    )

    # --- 6. Generate ---
    print("\n[tinystories] sample generations:")
    for prompt in ["Once upon a time", "The knight", "The cat"]:
        out = model.generate(prompt=prompt, max_new_tokens=32, temperature=0.8, top_k=20, seed=42)
        print(f"  {prompt!r} → {out!r}")


if __name__ == "__main__":
    main()
