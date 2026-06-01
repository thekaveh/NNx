# Language modeling (decoder-only)

NNx ships a TinyStories-class decoder-only Transformer alongside the GNN and
FeedFwd architectures. Use it for small autoregressive experiments: byte-pair
encoded tokenization, RoPE positional encoding, SwiGLU FFN, RMSNorm,
tied input/output embeddings, and an autoregressive `generate()` with the
standard sampling knobs.

This is intentionally **not** a production LLM stack — there's no
FlashAttention v3, no tensor parallelism, no multi-GPU sharding. The path
is sized to "train end-to-end on a laptop, end up with a model whose
architecture matches what GGUF / HF Hub / Ollama expect." Production-scale
training is out of scope.

## Public surface

| Symbol | Notes |
|---|---|
| `nnx.Nets.TRANSFORMER` | Enum variant; dispatches to `TransformerNN` via `NNModelParams.net`. |
| `nnx.TransformerNN` | `nn.Module` — decoder-only stack: token embed + N blocks + final RMSNorm + tied LM head. |
| `nnx.NNTransformerParams` | Frozen dataclass (subclass of `NNParams`) — `vocab_size`, `n_layers`, `n_heads`, `d_model`, `ffn_mult`, `max_seq_len`, `rope_base`, `tie_embeddings`, `attn_dropout`, `resid_dropout`. Every optional field omits itself from `state()` when at default (the omit-when-default invariant). |
| `nnx.NNTokenizerParams` | Wraps `tokenizers.Tokenizer`; `state()` returns `{"path": "<tokenizer.json>"}`. Available when `nnx[lm]` is installed. |
| `nnx.train_bpe(...)` | Quick BPE training helper (Whitespace pre-tokenizer + BPE + BpeTrainer). |
| `nnx.GenerativeNNModel` | `NNModel` subclass adding `generate(prompt, max_new_tokens, temperature, top_k, top_p, repetition_penalty, stop, seed)`. |
| `nnx.{TemperatureScaling, TopKFilter, TopPFilter, RepetitionPenalty, apply_chain}` | LogitsProcessor chain — same shape as HF transformers' `LogitsProcessorList`. |

## Install

The LM path is opt-in:

```bash
pip install "nnx[lm]"        # adds tokenizers>=0.20, datasets>=2.20
```

`tokenizers` is the HF Rust BPE / WordPiece tokenizer. `datasets` is only
used by the example for downloading TinyStories — the rest of the LM path
runs without it.

## Quickstart

```python
import torch
from torch.utils.data import DataLoader

from nnx import (
    Devices, Losses, Nets, NNModelParams, NNOptimParams,
    NNSchedulerParams, NNTrainParams, NNTransformerParams,
    NNTokenizerParams, GenerativeNNModel, Optims, set_seed, train_bpe,
)

# 1. Train a tiny BPE tokenizer.
corpus = ["the quick brown fox", "the lazy dog", "once upon a time"]
tk = train_bpe(files=None, texts=corpus, vocab_size=128,
               special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"])
tokenizer = NNTokenizerParams.of(tokenizer=tk, path="artifacts/tok.json")

# 2. Build the model. NNTransformerParams subclasses NNParams so
#    the rest of the NNModel machinery accepts it unchanged.
net_params = NNTransformerParams(
    input_dim=tokenizer.vocab_size,
    output_dim=tokenizer.vocab_size,
    dropout_prob=0.0,
    vocab_size=tokenizer.vocab_size,
    n_layers=4, n_heads=4, d_model=128,
    ffn_mult=4, max_seq_len=64,
)
model_params = NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU,
                             loss=Losses.CROSS_ENTROPY)
model = GenerativeNNModel(net_params=net_params, params=model_params,
                          tokenizer=tokenizer)

# 3. Train (custom train_step_fn for next-token loss — see
#    examples/11_tinystories_lm.py for the full path).
# ...

# 4. Generate.
out = model.generate(
    prompt="once upon a time",
    max_new_tokens=32,
    temperature=0.8,
    top_k=20,
    seed=42,        # reproducible sampling
)
print(out)
```

**Alternative: variant-aware Builder**

Same config via `NNTransformerParams.builder()` — the dead parent
fields (`hidden_dims`, `activation`, `dropout_prob`) are hidden, and
`d_model % heads == 0` is enforced at `.layers(...)` call-time
rather than waiting for `__post_init__`:

```python
net_params = (
    NNTransformerParams.builder()
    .vocab(tokenizer.vocab_size)
    .layers(n=4, heads=4, d_model=128)
    .context(max_seq_len=128)
    .build()
)
```

Both paths produce identical `NNTransformerParams` instances and
round-trip through `state()` / `from_state()` interchangeably.

## When to use what

### Greedy decoding (deterministic)

```python
model.generate(prompt="...", temperature=0.0)
```

`temperature=0` short-circuits to argmax. Two calls with the same prompt
produce identical output — this is the regression-test contract.

### Sampling with top-k

```python
model.generate(prompt="...", temperature=0.8, top_k=40)
```

`top_k=40` is the original GPT-2 default. Smaller `top_k` (≤ 10) gives more
focused, less creative output.

### Nucleus (top-p) sampling

```python
model.generate(prompt="...", temperature=1.0, top_p=0.9)
```

Top-p adaptively shrinks the candidate set per token. Combine `top_k` +
`top_p` to layer both filters (top-k applied first, then top-p over what
remains).

### Repetition penalty

```python
model.generate(prompt="...", repetition_penalty=1.2)
```

Divides positive logits of seen tokens by 1.2 (HF semantics — for negative
logits, the penalty *multiplies* so the relative mass still drops). A
penalty of 1.0 (the default) is a no-op.

### Reproducible sampling

```python
out1 = model.generate(prompt="x", temperature=1.0, top_k=20, seed=42)
out2 = model.generate(prompt="x", temperature=1.0, top_k=20, seed=42)
assert out1 == out2
```

The `seed` kwarg constructs a `torch.Generator` pinned to the model's
device — same seed + same prompt + same model = same output.

## How it composes with the rest of NNx

- **Custom `train_step_fn`** — `GenerativeNNModel` doesn't ship a
  built-in LM training step. The convention (see `examples/11_tinystories_lm.py`)
  is to write a tiny next-token loss step and pass it via
  `NNModel.train(train_step_fn=...)`. Same pattern as diffusion / KD /
  SimCLR / Mixup / CutMix.
- **`NNRun` content-addressed persistence** — TRANSFORMER runs hash the
  same way as any other run. The `omit-when-default` invariant on
  `NNTransformerParams` is what keeps existing TRANSFORMER `run.id`
  values stable as we add knobs over time.
- **PEFT (LoRA)** — `nnx.apply_lora_to(model.net, ...)` works on the
  `TransformerNN`'s `nn.Linear` projections (the fused `w_qkv`, `w_o`,
  and the SwiGLU `w1`/`w2`/`w3`). Test before publishing weights:
  pattern-match against the actual `named_modules()` of your config.
- **Callbacks** — `EarlyStopping`, `ModelCheckpoint`,
  `TensorBoardCallback`, `WandbCallback` all work unchanged. Logging
  `train_loss` for an LM is the standard signal; perplexity = `exp(loss)`.
- **KV cache** — `TransformerBlock` exposes a `use_cache` kwarg whose
  off-path returns `None` for the new kv tuple. `GenerativeNNModel.generate`
  defaults `use_cache=True` and runs a single prefill pass through
  `forward_with_cache` followed by incremental token-by-token decoding,
  for ≈1.9× speedup at 128 tokens on CPU (gap widens on longer contexts
  and on GPU).

## Scope explicit

The decoder-only LM path covers:

- Decoder-only architecture (LLaMA / Mistral conventions: RMSNorm, RoPE,
  SwiGLU, tied embeddings).
- HF tokenizer integration via the `tokenizers` Rust library.
- Autoregressive `generate()` with greedy + sampling (`LogitsProcessor`
  chain: temperature / top-k / top-p / repetition-penalty).
- KV-cache acceleration on by default (≈1.9× speedup at 128 tokens on
  CPU; wider on longer contexts and GPU).
- CPU-friendly TinyStories-class training (sub-30-min runs).
- Onward integrations shipped post-LM: `Prefix-Tuner` / `Prompt-Tuner`
  PEFT for frozen `TransformerNN` (see [Concepts §11](concepts.md#11-parameter-efficient-fine-tuning-lora-dora-ia3-prefix-prompt-adapters)),
  `dpo_train_step_factory` preference fine-tuning (see [`docs/dpo.md`](dpo.md)),
  and GGUF / Ollama export for the llama.cpp ecosystem (see
  [`docs/gguf.md`](gguf.md)).

Out of scope:

- Multi-GPU / multi-node training.
- FlashAttention v3 (uses `torch.nn.functional.scaled_dot_product_attention`,
  which picks the best backend kernel automatically).
- Tensor parallelism / FSDP / ZeRO sharding.
- Production-scale RLHF / PPO (DPO is the lightweight preference path
  NNx ships; full PPO is intentionally out of scope).
