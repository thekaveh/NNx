"""GGUF writer for ``TransformerNN``.

Single public entry point: :func:`write_gguf`. Writes a TransformerNN
plus its tokenizer to a ``.gguf`` file consumable by every llama.cpp
derivative (llama.cpp, ollama, LM Studio, GPT4All, ...).

Quantization scope:

* **F32 / F16 / BF16** — handled directly here via ``GGUFWriter``.
  The numpy array's dtype determines the on-disk storage; GGUFWriter
  records the matching ``GGMLQuantizationType`` in the tensor header.
* **Q8_0 and below** (Q4_K_M, Q5_K_M, Q6_K, Q8_0, IQ4_XS, ...) require
  the C++ ``llama-quantize`` binary because the k-quant codebook
  generation is not implemented in pure Python. We raise a clear
  ImportError pointing at ``pip install llama-cpp-python`` (which
  ships the binary) and the canonical shell-out path.

The writer is intentionally narrow: it knows how to emit one specific
architecture (NNx's decoder-only TransformerNN) under one architecture
tag (``"nnx_transformer"`` by default, overridable). Extending to
encoder-decoders or vision models would warrant a separate writer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    # Type-only imports so callers see meaningful annotations without
    # paying the cost (and circular-import risk) at runtime.
    from nnx.nn.net.transformer_nn import TransformerNN
    from nnx.nn.params.nn_tokenizer_params import NNTokenizerParams

# ``gguf`` is an optional dep — see pyproject.toml ``gguf-write`` extra.
# We do the import inside ``write_gguf`` so ``import nnx.interop`` works
# without the dep installed; the actual writer call raises a clear
# ImportError if the dep is missing.


# Quantizations we can emit directly without shelling out. Everything
# else requires the C++ llama-quantize binary.
SUPPORTED_QUANTIZATIONS: tuple[str, ...] = ("F32", "F16", "BF16")

# Quantizations that exist in the GGUF format but require the C++
# ``llama-quantize`` post-pass. We list them so the error message can
# tell the user which one they asked for and point at the install.
_NEEDS_LLAMA_QUANTIZE: frozenset[str] = frozenset(
    {
        "Q8_0",
        "Q6_K",
        "Q5_K_M",
        "Q5_K_S",
        "Q4_K_M",
        "Q4_K_S",
        "Q4_0",
        "Q4_1",
        "Q3_K_M",
        "Q3_K_S",
        "Q3_K_L",
        "Q2_K",
        "IQ4_XS",
        "IQ4_NL",
        "IQ3_XS",
        "IQ3_M",
        "IQ2_XS",
        "IQ2_M",
        "IQ1_S",
    }
)


def _require_gguf() -> Any:
    """Lazy import of the ``gguf`` Python package. Raise a clear
    ImportError pointing at the install path if it's missing.

    Returns the imported module so the caller can grab
    ``gguf.GGUFWriter`` and ``gguf.GGMLQuantizationType`` from it.
    """
    try:
        import gguf  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "The `gguf` package is required for nnx.interop.gguf.write_gguf. "
            "Install with `pip install 'thekaveh-nnx[gguf-write]'` or "
            "`pip install 'gguf>=0.19.0'` directly."
        ) from e
    return gguf


def _quantization_torch_dtype(quantization: str):
    """Map a quantization label to the torch dtype the tensor needs to
    be cast to before handing to ``GGUFWriter.add_tensor``.

    GGUFWriter picks the on-disk GGMLQuantizationType from the numpy
    array's dtype, so the right way to control the output precision
    is to cast the source tensor before converting to numpy.
    """
    import torch  # local import — torch is a hard dep, this is just to avoid top-level cycles in tests

    if quantization == "F32":
        return torch.float32
    if quantization == "F16":
        return torch.float16
    if quantization == "BF16":
        return torch.bfloat16
    raise ValueError(f"Unhandled quantization {quantization!r}")  # pragma: no cover — gated above


def _validate_quantization(quantization: str) -> None:
    """Raise ImportError (for things that need llama-quantize) or
    ValueError (for unknown labels).

    Distinct error types so callers can branch on "install something"
    vs. "fix your spelling".
    """
    if quantization in SUPPORTED_QUANTIZATIONS:
        return
    if quantization in _NEEDS_LLAMA_QUANTIZE:
        raise ImportError(
            f"Quantization {quantization!r} requires the C++ `llama-quantize` binary "
            "(NNx's GGUF writer is pure-Python and only supports F32 / F16 / BF16 "
            "directly). The canonical path is:\n"
            "  1. `pip install llama-cpp-python` — ships the `llama-quantize` CLI\n"
            "  2. Write F16 first with `nnx.interop.gguf.write_gguf(..., quantization='F16')`\n"
            "  3. Shell out:\n"
            f"     `llama-quantize <input.gguf> <output.gguf> {quantization}`\n"
            "We deliberately don't do the shell-out automatically — the binary's "
            "path varies by install layout and we don't want to silently swallow "
            "a quantization failure."
        )
    raise ValueError(
        f"Unknown quantization {quantization!r}. Supported pure-Python: "
        f"{SUPPORTED_QUANTIZATIONS}. Sub-F16 quantizations require the C++ "
        f"llama-quantize binary (see error message for `Q4_K_M` for the recipe)."
    )


def write_gguf(
    transformer_nn: TransformerNN,
    tokenizer: NNTokenizerParams,
    out_path: str | os.PathLike,
    *,
    architecture: str = "nnx_transformer",
    quantization: str = "F16",
    model_name: Optional[str] = None,
) -> str:
    """Write a TransformerNN + tokenizer to a single ``.gguf`` file.

    Args:
        transformer_nn: A ``nnx.TransformerNN`` instance. The forward
            path's tensors are exported under llama.cpp's tensor-naming
            convention (see ``tensor_name_map.map_tensors``).
        tokenizer: An ``nnx.NNTokenizerParams`` (or any object with a
            ``.tokenizer`` attribute exposing ``.get_vocab()`` and
            ``.get_vocab_size()``). Tokens + merges are emitted under
            the GGUF tokenizer keys.
        out_path: Destination ``.gguf`` path.
        architecture: ``general.architecture`` metadata value. Defaults
            to ``"nnx_transformer"`` — readable by patched llama.cpp
            forks. Pass ``"llama"`` to claim LLaMA-arch compatibility
            for stock llama.cpp readers (works because we match the
            LLaMA tensor naming + RMSNorm/SwiGLU/RoPE choices, but
            users should verify their target reader version).
        quantization: One of ``"F32"``, ``"F16"``, ``"BF16"``. Sub-F16
            quantizations require the C++ ``llama-quantize`` binary —
            see the ``ImportError`` message for the shell-out recipe.
        model_name: ``general.name`` metadata. Defaults to a
            ``"nnx_transformer_LxD"`` shape-derived name.

    Returns:
        The absolute path of the written file as a string.

    Raises:
        ImportError: when ``gguf`` is not installed, or when a
            quantization is requested that requires ``llama-quantize``.
        ValueError: when an unknown quantization label is passed.
    """
    import torch

    gguf = _require_gguf()
    _validate_quantization(quantization)

    # Local import — avoids importing torch / numpy at module top so
    # ``import nnx.interop`` is cheap even when the writer isn't used.
    from .tensor_name_map import map_tensors

    out_path = str(Path(out_path).expanduser().resolve())

    params = transformer_nn.params
    # NNParams declares ``n_heads`` as Optional[int]; NNTransformerParams
    # validates it's non-None in ``__post_init__``. Re-narrow here so
    # pyright / mypy can prove the division below is safe.
    assert params.n_heads is not None, "TransformerNN params must have n_heads set"
    n_heads = params.n_heads
    if model_name is None:
        model_name = f"nnx_transformer_L{params.n_layers}_D{params.d_model}"

    # Compute the SwiGLU hidden dim. The convention matches
    # ``SwiGLU.__init__`` exactly so a reader reconstructing the
    # architecture from metadata picks the same shape we shipped.
    feed_forward_length = int(2 * params.ffn_mult * params.d_model / 3)
    head_dim = params.d_model // n_heads

    writer = gguf.GGUFWriter(out_path, arch=architecture)

    # ---------- general metadata ----------
    # ``GGUFWriter(arch=...)`` already records ``general.architecture``;
    # don't call ``add_architecture()`` again (it overwrites with the
    # same value and emits a duplicate-key warning).
    writer.add_name(model_name)
    writer.add_file_type(_gguf_file_type_for(gguf, quantization))

    # ---------- architecture metadata (llama.cpp keys) ----------
    # These are the keys llama.cpp reads when reconstructing a model;
    # populating them lets a patched reader rebuild the right shapes
    # without consulting NNx-specific config.
    writer.add_context_length(params.max_seq_len)
    writer.add_block_count(params.n_layers)
    writer.add_embedding_length(params.d_model)
    writer.add_feed_forward_length(feed_forward_length)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_heads)  # NNx is MHA (not GQA) — kv heads == q heads
    writer.add_layer_norm_rms_eps(1e-6)  # matches RMSNorm default in transformer_layers.py
    writer.add_rope_freq_base(params.rope_base)
    # The RoPE dimension count is the per-head dim (rotary is applied
    # per head). llama.cpp expects this exact number.
    writer.add_rope_dimension_count(head_dim)

    # ---------- tokenizer ----------
    # ``model_name`` here is the *tokenizer* model name ("llama", "gpt2",
    # "bpe", ...). NNx ships a HF tokenizers.Tokenizer trained with the
    # BPE model — emit it as "gpt2" so llama.cpp's BPE path picks it up.
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("default")  # pre-tokenizer is Whitespace; "default" is the catch-all
    _add_tokenizer_vocab(writer, tokenizer)

    # ---------- tensors ----------
    target_dtype = _quantization_torch_dtype(quantization)
    tensors = map_tensors(transformer_nn)
    for name, arr in tensors.items():
        # The map returns numpy arrays already in F32. Cast via torch
        # so we get a single code path for F16 / BF16 / F32 (numpy
        # doesn't have a native bfloat16 dtype; torch + ml_dtypes is
        # how the upstream gguf package handles it).
        t = torch.from_numpy(arr).to(target_dtype)
        if target_dtype == torch.bfloat16:
            # GGUFWriter accepts bfloat16 via numpy when the user
            # passes raw_dtype=BF16; we route through that contract
            # because numpy proper has no bfloat16 scalar.
            np_arr = t.view(torch.uint16).numpy()
            writer.add_tensor(name, np_arr, raw_dtype=gguf.GGMLQuantizationType.BF16)
        else:
            writer.add_tensor(name, t.numpy())

    # ---------- flush ----------
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return out_path


def _gguf_file_type_for(gguf_mod, quantization: str) -> int:
    """Translate the user-facing quantization label to the integer
    ``general.file_type`` value GGUF uses for the same thing.

    The integer codes are stable in the GGUF spec — F32=0, F16=1,
    BF16=32 — but we read them off the upstream enum to avoid hard-coding."""
    # LlamaFileType (gguf >= 0.10) enumerates the canonical labels.
    ft = gguf_mod.LlamaFileType
    if quantization == "F32":
        return int(ft.ALL_F32)
    if quantization == "F16":
        return int(ft.MOSTLY_F16)
    if quantization == "BF16":
        return int(ft.MOSTLY_BF16)
    raise ValueError(f"Unhandled quantization {quantization!r}")  # pragma: no cover — gated above


def _add_tokenizer_vocab(writer, tokenizer) -> None:
    """Emit the tokenizer vocab + merges into the GGUF writer.

    NNx's ``NNTokenizerParams`` wraps a HF ``tokenizers.Tokenizer`` —
    we extract the vocab (token -> id) and order it by id, then push
    the BPE merges as a separate ``tokenizer.ggml.merges`` array. This
    matches what llama.cpp expects for a GPT-2-style BPE tokenizer."""
    # Unwrap NNTokenizerParams to the raw tokenizers.Tokenizer.
    tk = getattr(tokenizer, "tokenizer", tokenizer)

    vocab = tk.get_vocab()
    # Sort tokens by their integer id so the GGUF array index == token id.
    sorted_tokens = sorted(vocab.items(), key=lambda kv: kv[1])
    tokens = [t for t, _ in sorted_tokens]

    # GGUFWriter expects the list of tokens as either strings or bytes.
    # Strings are fine for BPE — llama.cpp's reader handles the bytes
    # vs. string normalization.
    writer.add_token_list(tokens)

    # Token types: 1 = NORMAL, 2 = UNKNOWN, 3 = CONTROL. We mark every
    # token NORMAL by default and bump the registered special tokens
    # to CONTROL so llama.cpp's sampler treats them as non-generative.
    n = len(tokens)
    token_types = [int(_NORMAL_TOKEN)] * n
    special_tokens = _collect_special_tokens(tk)
    for st in special_tokens:
        if st in vocab:
            token_types[vocab[st]] = int(_CONTROL_TOKEN)
    writer.add_token_types(token_types)

    # BPE merges live in tokenizers.Tokenizer's serialized JSON.
    # Extract them via to_str() rather than reaching into internals so
    # we stay forward-compatible across tokenizers versions.
    import json

    tk_json = json.loads(tk.to_str())
    merges = (tk_json.get("model") or {}).get("merges") or []
    # tokenizers stores merges as either ["a b", ...] or [["a","b"], ...]
    # depending on version; normalize to space-joined strings.
    normalized_merges = []
    for m in merges:
        if isinstance(m, (list, tuple)):
            normalized_merges.append(" ".join(m))
        else:
            normalized_merges.append(str(m))
    writer.add_token_merges(normalized_merges)

    # BOS / EOS / UNK / PAD ids — best-effort. NNx's train_bpe
    # registers `<unk>`, `<pad>`, `<bos>`, `<eos>` by convention.
    for special_tok, setter in (
        ("<bos>", writer.add_bos_token_id),
        ("<eos>", writer.add_eos_token_id),
        ("<unk>", writer.add_unk_token_id),
        ("<pad>", writer.add_pad_token_id),
    ):
        if special_tok in vocab:
            setter(vocab[special_tok])


# Token-type codes — pulled from the GGUF spec.
_NORMAL_TOKEN = 1
_CONTROL_TOKEN = 3


def _collect_special_tokens(tk) -> list[str]:
    """Pull the registered special tokens out of a HF tokenizers.Tokenizer.

    The serialized JSON has an ``added_tokens`` array — entries with
    ``special=True`` are the ones we want to mark CONTROL in GGUF."""
    import json

    try:
        tk_json = json.loads(tk.to_str())
    except Exception:  # pragma: no cover — paranoia; to_str shouldn't fail
        return []
    added = tk_json.get("added_tokens") or []
    return [a["content"] for a in added if isinstance(a, dict) and a.get("special")]
