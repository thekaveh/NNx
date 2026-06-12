# GGUF export & Ollama bundles

NNx ships a writer for [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) —
the on-disk format consumed by llama.cpp, Ollama, LM Studio, and the
broader llama.cpp-derived inference ecosystem. The handoff in one line:

> **Train in NNx, serve via llama.cpp / Ollama** — export the
> `TransformerNN` to a `.gguf`, optionally bundle it with a Modelfile,
> and `ollama create` registers the model locally.

This is intentionally a one-way handoff (NNx -> GGUF). The reverse
direction (GGUF -> NNx) isn't covered here — every other Python tool
also writes GGUF; nothing reads it back into a training-shaped framework.

## 1. Install

The writer is opt-in — pull in the upstream `gguf` package via the
`gguf-write` extra:

```bash
pip install "thekaveh-nnx[gguf-write]"   # adds gguf>=0.19.0
```

`gguf` is the same Python writer every other GGUF producer uses, so
the artifact is byte-compatible with every GGUF reader in the ecosystem.

## 2. Public surface

| Symbol | Notes |
|---|---|
| `nnx.interop.write_gguf` | Write a `.gguf` from a `TransformerNN` + tokenizer. F16 / F32 / BF16 directly; sub-F16 needs `llama-quantize`. |
| `nnx.interop.export_ollama_modelfile` | Emit a directory containing `model.gguf` + `Modelfile` ready for `ollama create`. |
| `nnx.interop.gguf.SUPPORTED_QUANTIZATIONS` | The pure-Python quantizations — `("F32", "F16", "BF16")`. |

## 3. Quickstart — GGUF

```python
from nnx import NNTokenizerParams, NNTransformerParams, TransformerNN, train_bpe  # noqa: F401
from nnx.interop import write_gguf

# Assume `net` is a trained TransformerNN and `tokenizer` is the matching
# NNTokenizerParams. write_gguf packs both into a single .gguf.
write_gguf(net, tokenizer, "out/model.gguf")  # F16 by default
```

That's the whole API. Tensor naming (`token_embd.weight`,
`blk.{i}.attn_q.weight`, ...) and the GGUF metadata (`context_length`,
`block_count`, `embedding_length`, `feed_forward_length`,
`head_count`, `rope_freq_base`, ...) are emitted under llama.cpp's
canonical key namespace.

The fused-QKV projection in NNx's `TransformerNN.attn.w_qkv` is split
into three tensors (`attn_q`, `attn_k`, `attn_v`) on write so that
llama.cpp's reader sees the layout it expects.

## 4. Quantization

| Label | Path |
|---|---|
| `F32` / `F16` / `BF16` | Pure-Python via `gguf` directly. |
| `Q8_0` / `Q6_K` / `Q5_K_M` / `Q4_K_M` / `Q4_0` / `IQ4_XS` / ... | Two-step: write F16 here, then run the C++ `llama-quantize` binary. |

NNx deliberately doesn't shell out to `llama-quantize` automatically —
the binary's path varies by install layout (Homebrew, `pip install
llama-cpp-python`, source build) and silently swallowing a quantization
failure is worse than asking the user to run one command.

The recipe:

```bash
# 1. NNx writes F16
python -c "from nnx.interop import write_gguf; write_gguf(net, tok, 'out/model.gguf')"

# 2. llama-quantize is shipped by llama-cpp-python
pip install llama-cpp-python
llama-quantize out/model.gguf out/model.Q4_K_M.gguf Q4_K_M
```

If you ask `write_gguf(..., quantization="Q4_K_M")` directly it raises
`ImportError` with the recipe in the message.

## 5. Architecture tag

By default the writer stamps `general.architecture = "nnx_transformer"`.
This is readable by patched / forked llama.cpp builds and by any
inference stack that reflects on metadata rather than hard-coding the
arch name.

For stock llama.cpp readers, the NNx tensor layout (RMSNorm + RoPE +
SwiGLU + tied embeddings + per-head RoPE rotation) matches the LLaMA
family closely enough that you can pass `architecture="llama"`:

```python
write_gguf(net, tok, "out/model.gguf", architecture="llama")
```

Verify against your target reader version before deploying — small
divergences (e.g. RMS-norm epsilon) can show up as numerical drift
even when the tensor shapes match.

## 6. Ollama bundles

`export_ollama_modelfile` produces a directory ready for `ollama create`:

```python
from nnx.interop import export_ollama_modelfile

export_ollama_modelfile(
    net,
    tokenizer,
    out_dir="out/ollama_bundle",
    system="You are a tiny storytelling model.",
    parameters={
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.95,
        "stop": ["<eos>"],          # list -> repeated PARAMETER stop lines
    },
    template="{{ .Prompt }}",       # optional Go-template chat layout
)
```

Layout produced:

```
out/ollama_bundle/
  model.gguf
  Modelfile
  tokenizer.json     # (only if you saved it there via NNTokenizerParams.of(path=...), as examples/18 does)
```

The `Modelfile` looks like:

```
FROM ./model.gguf
PARAMETER temperature 0.8
PARAMETER top_k 40
PARAMETER top_p 0.95
PARAMETER stop <eos>
TEMPLATE """{{ .Prompt }}"""
SYSTEM """You are a tiny storytelling model."""
```

Then:

```bash
cd out/ollama_bundle
ollama create my-nnx-model -f Modelfile
ollama run my-nnx-model
```

## 7. End-to-end examples

| Example | Demonstrates |
|---|---|
| [`examples/17_export_transformer_to_gguf.py`](https://github.com/thekaveh/NNx/blob/main/examples/17_export_transformer_to_gguf.py) | Build TransformerNN -> write F16 GGUF -> round-trip via `gguf.GGUFReader`. |
| [`examples/18_publish_to_ollama.py`](https://github.com/thekaveh/NNx/blob/main/examples/18_publish_to_ollama.py) | Bundle GGUF + Modelfile + parameters for `ollama create`. |

## 8. Scope

The writer covers `TransformerNN` (NNx's decoder-only LM). Other
architectures (FeedFwd, the GNNs, diffusion nets) aren't applicable to
GGUF — GGUF is a llama.cpp-family format. The TransformerNN scope is
intentionally narrow: TinyStories-class single-GPU LM. Bigger models
work as long as the tensor layout stays the same (multi-head causal
attention, RMSNorm, SwiGLU, RoPE, tied embeddings).
