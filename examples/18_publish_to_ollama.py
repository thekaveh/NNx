"""Bundle a TransformerNN + Modelfile for ``ollama create``.

The Ollama UX is: a directory containing ``model.gguf`` + a
``Modelfile`` that says ``FROM ./model.gguf`` plus optional
``SYSTEM`` / ``PARAMETER`` / ``TEMPLATE`` directives.

This script produces that directory. To register the model:

    python examples/18_publish_to_ollama.py
    cd artifacts/ollama_bundle
    ollama create my-nnx-model -f Modelfile
    ollama run my-nnx-model

Scope: the GGUF is written at F16 (the default). For smaller files use
the ``llama-quantize`` shell-out recipe from
``examples/17_export_transformer_to_gguf.py``, then point the Modelfile's
``FROM`` line at the quantized file.

Run:
    python examples/18_publish_to_ollama.py
"""

from __future__ import annotations

from pathlib import Path

from nnx.interop import export_ollama_modelfile
from nnx.nn.net.transformer_nn import TransformerNN
from nnx.nn.params.nn_tokenizer_params import NNTokenizerParams, train_bpe
from nnx.nn.params.nn_transformer_params import NNTransformerParams


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
            # Tune these for the Ollama runtime; defaults are fine for a smoke test.
            "temperature": 0.8,
            "top_k": 40,
            "top_p": 0.95,
            "stop": ["<eos>"],
        },
        # Optional Go-template — Ollama renders it on every chat turn.
        # The TinyStories model has no chat structure, so we keep it
        # minimal here.
        template="{{ .Prompt }}",
    )

    print(f"[ollama] Modelfile: {mf_path}")
    print(f"[ollama] GGUF:      {out_dir / 'model.gguf'}")
    print(
        "\n[ollama] Register with:\n"
        f"  cd {out_dir}\n"
        "  ollama create my-nnx-model -f Modelfile\n"
        "  ollama run my-nnx-model"
    )


if __name__ == "__main__":
    main()
