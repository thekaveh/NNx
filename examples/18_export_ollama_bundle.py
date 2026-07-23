"""Generate an experimental TransformerNN GGUF + Modelfile bundle.

The Ollama UX is: a directory containing ``model.gguf`` + a
``Modelfile`` that says ``FROM ./model.gguf`` plus optional
``SYSTEM`` / ``PARAMETER`` / ``TEMPLATE`` directives.

This script produces that directory as a bundle fixture. Stock Ollama does not
implement the ``nnx_transformer`` architecture and cannot run the generated
model. Use it for inspecting the bundle or developing a patched runtime; do not
relabel the GGUF as ``llama`` because the RoPE layouts differ.

Scope: the GGUF is written at F16 (the default). For smaller files use
the ``llama-quantize`` shell-out recipe from
``examples/17_export_transformer_to_gguf.py``, then point the Modelfile's
``FROM`` line at the quantized file.

Requires the ``gguf-write`` optional extra (for
``nnx.interop.export_ollama_modelfile``, which wraps
``nnx.interop.write_gguf``) AND the ``lm`` extra (for
``NNTokenizerParams`` / ``train_bpe`` via HuggingFace ``tokenizers``):

    pip install 'thekaveh-nnx[gguf-write,lm]'

Run:
    python examples/18_export_ollama_bundle.py
"""

from __future__ import annotations

from pathlib import Path

from nnx import NNTokenizerParams, NNTransformerParams, TransformerNN, train_bpe
from nnx.interop import export_ollama_modelfile


def main() -> None:
    out_dir = Path("artifacts/ollama_bundle")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Tokenizer + model (would be loaded from disk in real use). ---
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

    # --- 2. Bundle: GGUF + Modelfile. ---
    mf_path = export_ollama_modelfile(
        net,
        tokenizer,
        out_dir,
        system="You are a small storytelling model.",
        parameters={
            # Representative Modelfile parameters for bundle inspection.
            "temperature": 0.8,
            "top_k": 40,
            "top_p": 0.95,
            "stop": ["<eos>"],
        },
        # Optional Go-template carried by the generated Modelfile.
        # The TinyStories model has no chat structure, so we keep it
        # minimal here.
        template="{{ .Prompt }}",
    )

    print(f"[ollama] Modelfile: {mf_path}")
    print(f"[ollama] GGUF:      {out_dir / 'model.gguf'}")
    print("\n[ollama] Stock Ollama cannot run the nnx_transformer architecture.")
    print("         Inspect this bundle or use a runtime explicitly patched for NNx.")


if __name__ == "__main__":
    main()
