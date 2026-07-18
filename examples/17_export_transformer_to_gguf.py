"""Export a TransformerNN to an experimental GGUF artifact.

This script demonstrates NNx's GGUF container writer:

  1. Build a tiny TransformerNN + BPE tokenizer (or load one off disk).
  2. Call ``nnx.interop.gguf.write_gguf`` to produce ``model.gguf``.
  3. Optionally probe the file with ``gguf.GGUFReader`` to verify.

The output uses ``general.architecture=nnx_transformer``. Stock llama.cpp,
Ollama, and LM Studio do not implement that architecture; use the artifact for
inspection or with a runtime explicitly patched for NNx. Do not relabel it as
``llama`` because NNx and LLaMA use different RoPE layouts.

For Q4_K_M / Q5_K_M / etc., write F16 here and build the official
``llama-quantize`` tool from the llama.cpp source repository:

    python examples/17_export_transformer_to_gguf.py
    llama-quantize artifacts/lm_export/model.gguf \\
        artifacts/lm_export/model.Q4_K_M.gguf Q4_K_M

Requires the ``gguf-write`` optional extra (for the ``gguf`` PyPI
package that backs ``nnx.interop.write_gguf``) AND the ``lm`` extra
(for ``NNTokenizerParams`` / ``train_bpe`` via HuggingFace
``tokenizers``):

    pip install 'thekaveh-nnx[gguf-write,lm]'

Run:
    python examples/17_export_transformer_to_gguf.py
"""

from __future__ import annotations

from pathlib import Path

from nnx import NNTokenizerParams, NNTransformerParams, TransformerNN, train_bpe
from nnx.interop import write_gguf


def main() -> None:
    # --- 1. Build a tiny tokenizer + model ---
    # In a real flow you'd load these off the NNRun directory produced
    # by `examples/11_tinystories_lm.py`. We construct them inline so
    # the example is self-contained.
    out_dir = Path("artifacts/lm_export")
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = [
        "Once upon a time there was a small cat.",
        "The cat lived in a cozy house with a garden.",
        "Every day the cat chased butterflies.",
    ] * 50

    tk = train_bpe(texts=corpus, vocab_size=256, special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"])
    tokenizer = NNTokenizerParams.of(tokenizer=tk, path=str(out_dir / "tokenizer.json"))

    params = NNTransformerParams(
        input_dim=tokenizer.vocab_size,
        output_dim=tokenizer.vocab_size,
        dropout_prob=0.0,
        vocab_size=tokenizer.vocab_size,
        n_layers=2,
        n_heads=4,
        d_model=64,
        ffn_mult=4,
        max_seq_len=128,
    )
    net = TransformerNN(params=params)
    print(f"[gguf] model params: {sum(p.numel() for p in net.parameters()):,}")

    # --- 2. Write GGUF (F16 by default) ---
    gguf_path = out_dir / "model.gguf"
    write_gguf(net, tokenizer, gguf_path)
    print(f"[gguf] wrote {gguf_path} ({gguf_path.stat().st_size / 1024:.1f} KB)")

    # --- 3. Round-trip via gguf.GGUFReader (optional smoke check) ---
    import gguf

    reader = gguf.GGUFReader(str(gguf_path))
    n_tensors = len(reader.tensors)
    print(f"[gguf] reader sees {n_tensors} tensors")
    arch = reader.get_field("general.architecture").contents()
    n_layers = reader.get_field("nnx_transformer.block_count").contents()
    print(f"[gguf] architecture={arch!r}, block_count={n_layers}")

    # --- 4. Pointer at the sub-F16 quantization recipe ---
    print(
        "\n[gguf] For Q4_K_M / Q5_K_M / Q8_0 quantization, install the\n"
        "       official llama.cpp `llama-quantize` binary from a source build\n"
        f"       and run: `llama-quantize {gguf_path} {gguf_path.with_suffix('.Q4_K_M.gguf')} Q4_K_M`"
    )


if __name__ == "__main__":
    main()
