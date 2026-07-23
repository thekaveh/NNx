# Experimental GGUF export

NNx can serialize `TransformerNN` weights and tokenizer metadata into a
[GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) container.
The writer is useful for inspecting the model with GGUF tooling, archiving a
self-describing artifact, or handing it to a runtime explicitly patched for
NNx.

This support is experimental. NNx writes
`general.architecture = "nnx_transformer"`, which stock llama.cpp, Ollama, and
LM Studio do not implement. A structurally valid GGUF file is not automatically
an executable model for every GGUF consumer.

## 1. Install

Install the upstream Python GGUF writer with the optional extra:

```bash
pip install "thekaveh-nnx[gguf-write]"
```

The extra provides `gguf>=0.19.0`. Language-model examples also need the `lm`
extra for the Hugging Face `tokenizers` package:

```bash
pip install "thekaveh-nnx[gguf-write,lm]"
```

## 2. Public surface

| Symbol | Notes |
|---|---|
| `nnx.interop.write_gguf` | Write an NNx-tagged `.gguf` from a `TransformerNN` and tokenizer. F16, F32, and BF16 are written directly. |
| `nnx.interop.export_ollama_modelfile` | Generate `model.gguf` plus a syntactically valid Modelfile. This does not add stock Ollama runtime support. |
| `nnx.interop.gguf.SUPPORTED_QUANTIZATIONS` | Direct writer modes: `("F32", "F16", "BF16")`. |

## 3. Write and inspect a GGUF

```python
from nnx.interop import write_gguf

# `net` is a trained TransformerNN and `tokenizer` is its matching
# NNTokenizerParams.
write_gguf(net, tokenizer, "out/model.gguf")  # F16 by default
```

NNx records model dimensions, RoPE and normalization metadata, tokenizer
metadata, and stable tensor names. The fused QKV projection is split into
`attn_q`, `attn_k`, and `attn_v`; SwiGLU projections are similarly mapped to
separate gate, up, and down tensors.

The upstream Python reader can inspect the resulting container:

```python
import gguf

reader = gguf.GGUFReader("out/model.gguf")
print(reader.get_field("general.architecture").contents())
print(len(reader.tensors))
```

## 4. Quantization

| Label | Path |
|---|---|
| `F32` / `F16` / `BF16` | Written directly with the Python `gguf` package. |
| `Q8_0` / `Q6_K` / `Q5_K_M` / `Q4_K_M` / other llama.cpp modes | Write F16 first, then use a compatible `llama-quantize` binary. |

Build the official llama.cpp tools from source to obtain `llama-quantize`:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cmake -B llama.cpp/build -S llama.cpp -DLLAMA_BUILD_TESTS=OFF
cmake --build llama.cpp/build --config Release -j

llama.cpp/build/bin/llama-quantize \
  out/model.gguf out/model.Q4_K_M.gguf Q4_K_M
```

Quantization support for a custom architecture depends on the chosen
llama.cpp revision or patch set. `write_gguf(..., quantization="Q4_K_M")`
raises an `ImportError` containing the two-step guidance instead of invoking an
external binary implicitly.

## 5. Architecture compatibility

Do not relabel an NNx artifact as `architecture="llama"` merely to get past a
consumer's architecture check. `TransformerNN` uses an interleaved RoPE layout,
while stock LLaMA loaders expect the LLaMA split-half convention; metadata and
tensor-shape similarity do not make those computations equivalent. A relabeled
file can load yet produce incorrect output.

Use the default `nnx_transformer` tag and one of these supported purposes:

1. Inspect metadata and tensors with a generic GGUF parser.
2. Archive or exchange a self-describing NNx model artifact.
3. Load it with a runtime that explicitly implements the NNx architecture and
   tensor layout.

## 6. Modelfile bundle generation

`export_ollama_modelfile` generates a GGUF and adjacent Modelfile for testing,
inspection, or a patched Ollama/runtime integration:

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
        "stop": ["<eos>"],
    },
    template="{{ .Prompt }}",
)
```

The generated directory contains `model.gguf` and `Modelfile`; it may also
contain `tokenizer.json` when the tokenizer was saved there. The helper
validates the Ollama `v0.32.2` tagged-source parameter set: integer `num_ctx`, `repeat_last_n`,
`seed`, `num_predict`, `draft_num_predict`, and `top_k`; finite numeric
`repeat_penalty`, `temperature`, `top_p`, and `min_p`; and scalar or repeated
string `stop`. Unknown names and mismatched types fail before GGUF generation.
Parameter strings intentionally use a restricted safe subset that rejects
quotes and control characters rather than attempting broader Modelfile
escaping. The helper writes the standard `FROM`, `PARAMETER`, `TEMPLATE`, and
`SYSTEM` directives. Stock `ollama create` still rejects or cannot execute the
`nnx_transformer` architecture.

## 7. Examples

| Example | Demonstrates |
|---|---|
| [`examples/17_export_transformer_to_gguf.py`](https://github.com/thekaveh/NNx/blob/main/examples/17_export_transformer_to_gguf.py) | Write F16 GGUF and inspect it with `gguf.GGUFReader`. |
| [`examples/18_export_ollama_bundle.py`](https://github.com/thekaveh/NNx/blob/main/examples/18_export_ollama_bundle.py) | Generate and inspect a GGUF + Modelfile bundle while documenting the stock-runtime limitation. |

## 8. Scope

The writer covers NNx's decoder-only `TransformerNN`. Feed-forward, graph,
convolutional, diffusion, and other NNx networks are outside its scope. GGUF
import back into NNx is also not implemented.
