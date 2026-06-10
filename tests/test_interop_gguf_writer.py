"""Round-trip tests for ``nnx.interop.gguf.write_gguf``.

Covers:
  * Writing a tiny TransformerNN at F16 produces a file readable by
    ``gguf.GGUFReader``, and the tensor shapes round-trip cleanly.
  * The tensor naming convention matches llama.cpp expectations
    (``token_embd.weight`` / ``blk.{i}.attn_q.weight`` / etc.).
  * The fused-QKV split produces three distinct tensors whose
    concatenation equals the original ``w_qkv`` weight (otherwise we
    silently broke attention on the reader side).
  * Sub-F16 quantization raises a clean error pointing at the
    llama-quantize install path.
  * The architecture metadata + model dims round-trip.
  * Tied embeddings cause the LM head to be omitted (file size win).
  * BF16 round-trips.
  * The hard ``gguf``-missing path raises ImportError with install text.

The gate: skipif when the optional ``gguf`` dep is missing, so this
test file is harmless on a minimal install.
"""

from __future__ import annotations

import importlib.util

import pytest

from nnx.nn.net.transformer_nn import TransformerNN
from nnx.nn.params.nn_transformer_params import NNTransformerParams

_HAS_GGUF = importlib.util.find_spec("gguf") is not None
_HAS_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None

pytestmark = pytest.mark.skipif(
    not _HAS_GGUF,
    reason="`gguf` package not installed (optional dep: nnx[gguf-write])",
)


def _tiny_transformer() -> TransformerNN:
    """A 2-layer, 32-dim, vocab=64 transformer — small enough to be
    instant, large enough to exercise every tensor in the name map."""
    params = NNTransformerParams(
        input_dim=64,
        output_dim=64,
        dropout_prob=0.0,
        vocab_size=64,
        n_layers=2,
        n_heads=4,
        d_model=32,
        ffn_mult=4,
        max_seq_len=16,
    )
    return TransformerNN(params=params)


def _tiny_tokenizer(tmp_path):
    """A trained BPE tokenizer on a tiny corpus. Real-enough that the
    GGUF write path exercises the merge-list extraction."""
    if not _HAS_TOKENIZERS:
        pytest.skip("`tokenizers` not installed")
    from nnx.nn.params.nn_tokenizer_params import NNTokenizerParams, train_bpe

    corpus = ["hello world", "the cat sat", "once upon a time"] * 20
    tk = train_bpe(texts=corpus, vocab_size=64, special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"])
    return NNTokenizerParams.of(tokenizer=tk, path=str(tmp_path / "tokenizer.json"))


# -------------------- tensor name map --------------------


def test_tensor_name_map_emits_expected_keys():
    """The tensor map's keyset is the contract with llama.cpp readers —
    pin it down."""
    from nnx.interop.gguf.tensor_name_map import map_tensors

    net = _tiny_transformer()
    tensors = map_tensors(net)
    # The full expected set for a 2-layer, tie_embeddings=True model.
    expected = {
        "token_embd.weight",
        "output_norm.weight",
        *{
            f"blk.{i}.{name}"
            for i in range(2)
            for name in (
                "attn_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            )
        },
    }
    assert set(tensors.keys()) == expected, set(tensors.keys()) - expected


def test_tensor_name_map_omits_output_weight_when_tied():
    """tie_embeddings=True (the NNx default) means llama.cpp re-uses
    token_embd.weight — no separate output.weight in the file."""
    from nnx.interop.gguf.tensor_name_map import map_tensors

    net = _tiny_transformer()
    assert net.params.tie_embeddings is True  # defensive
    tensors = map_tensors(net)
    assert "output.weight" not in tensors


def test_tensor_name_map_emits_output_weight_when_untied():
    from nnx.interop.gguf.tensor_name_map import map_tensors

    params = NNTransformerParams(
        input_dim=64,
        output_dim=64,
        dropout_prob=0.0,
        vocab_size=64,
        n_layers=1,
        n_heads=4,
        d_model=32,
        ffn_mult=4,
        max_seq_len=16,
        tie_embeddings=False,
    )
    net = TransformerNN(params=params)
    tensors = map_tensors(net)
    assert "output.weight" in tensors


def test_qkv_split_is_correct():
    """The fused QKV unpack must produce three tensors whose row-wise
    concatenation equals the original. Off-by-one here silently corrupts
    attention on the reader side — pin it down with a numerical check."""
    import numpy as np

    from nnx.interop.gguf.tensor_name_map import map_tensors

    net = _tiny_transformer()
    tensors = map_tensors(net)
    d_model = net.params.d_model
    q = tensors["blk.0.attn_q.weight"]
    k = tensors["blk.0.attn_k.weight"]
    v = tensors["blk.0.attn_v.weight"]
    assert q.shape == (d_model, d_model)
    assert k.shape == (d_model, d_model)
    assert v.shape == (d_model, d_model)
    original = net.blocks[0].attn.w_qkv.weight.detach().cpu().numpy()
    reconstructed = np.concatenate([q, k, v], axis=0)
    assert np.array_equal(reconstructed, original)


# -------------------- writer round-trip --------------------


def test_write_gguf_round_trip_f16(tmp_path):
    """Write a tiny TransformerNN at F16, read it back via GGUFReader,
    verify shapes match. The end-to-end pin: if this passes, the format
    is consumable by every llama.cpp-derived stack."""
    import gguf

    net = _tiny_transformer()
    tok = _tiny_tokenizer(tmp_path)
    out_path = tmp_path / "model.gguf"

    from nnx.interop.gguf import write_gguf

    write_gguf(net, tok, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    reader = gguf.GGUFReader(str(out_path))
    # All tensors round-trip with the same shapes.
    from nnx.interop.gguf.tensor_name_map import map_tensors

    expected = map_tensors(net)
    for t in reader.tensors:
        # GGUFReader stores shapes in (..., last-dim) row-major; the
        # NNx side built numpy arrays from torch tensors which also are
        # row-major. Match on shape directly.
        assert t.name in expected, t.name
        expected_shape = expected[t.name].shape
        # ``t.shape`` is a numpy array; convert and reverse — GGUF
        # stores dims in reverse order in the on-disk header.
        observed_shape = tuple(int(x) for x in reversed(list(t.shape)))
        assert observed_shape == expected_shape, (t.name, observed_shape, expected_shape)


def test_write_gguf_emits_architecture_metadata(tmp_path):
    """The architecture + dim metadata must be in the file; a reader
    needs every key to reconstruct the model shape without external
    config."""
    import gguf

    net = _tiny_transformer()
    tok = _tiny_tokenizer(tmp_path)
    out_path = tmp_path / "model.gguf"

    from nnx.interop.gguf import write_gguf

    write_gguf(net, tok, out_path, architecture="nnx_transformer")
    reader = gguf.GGUFReader(str(out_path))

    # ReaderField exposes a ``contents()`` helper that returns the
    # native-typed value (str for strings, int for uint32, ...). It's
    # the right knob for "give me the scalar a writer pushed in here."
    assert reader.get_field("general.architecture").contents() == "nnx_transformer"
    assert reader.get_field("nnx_transformer.context_length").contents() == net.params.max_seq_len
    assert reader.get_field("nnx_transformer.block_count").contents() == net.params.n_layers
    assert reader.get_field("nnx_transformer.embedding_length").contents() == net.params.d_model
    assert reader.get_field("nnx_transformer.attention.head_count").contents() == net.params.n_heads


def test_write_gguf_emits_tokenizer_vocab(tmp_path):
    """The vocab list + merges + special-token ids must round-trip."""
    import gguf

    net = _tiny_transformer()
    tok = _tiny_tokenizer(tmp_path)
    out_path = tmp_path / "model.gguf"

    from nnx.interop.gguf import write_gguf

    write_gguf(net, tok, out_path)
    reader = gguf.GGUFReader(str(out_path))

    # tokenizer.ggml.tokens — array field. The reader exposes it as a
    # ReaderField with one entry per token.
    tokens_field = reader.get_field("tokenizer.ggml.tokens")
    assert tokens_field is not None
    # Length matches the tokenizer's vocab.
    assert len(tokens_field.data) == tok.vocab_size, (len(tokens_field.data), tok.vocab_size)


def test_write_gguf_subq8_raises_with_install_hint(tmp_path):
    """A quantization below F16 must raise ImportError pointing at the
    llama-quantize binary. The error message is the user-facing
    contract — assert the install path is in there."""
    net = _tiny_transformer()
    tok = _tiny_tokenizer(tmp_path)
    out_path = tmp_path / "model.gguf"

    from nnx.interop.gguf import write_gguf

    with pytest.raises(ImportError, match="llama-quantize"):
        write_gguf(net, tok, out_path, quantization="Q4_K_M")


def test_write_gguf_unknown_quantization_raises(tmp_path):
    net = _tiny_transformer()
    tok = _tiny_tokenizer(tmp_path)
    out_path = tmp_path / "model.gguf"

    from nnx.interop.gguf import write_gguf

    with pytest.raises(ValueError, match="Unknown quantization"):
        write_gguf(net, tok, out_path, quantization="not_a_real_label")


def test_write_gguf_bf16_round_trip(tmp_path):
    """BF16 takes a different code path than F16 (numpy doesn't have
    bfloat16). Smoke-test it round-trips."""
    import gguf

    net = _tiny_transformer()
    tok = _tiny_tokenizer(tmp_path)
    out_path = tmp_path / "model.gguf"

    from nnx.interop.gguf import write_gguf

    write_gguf(net, tok, out_path, quantization="BF16")
    reader = gguf.GGUFReader(str(out_path))
    # Verify the file-type kv records BF16.
    ft_value = reader.get_field("general.file_type").contents()
    # MOSTLY_BF16 in the GGUF spec; cross-check via the upstream enum.
    assert ft_value == int(gguf.LlamaFileType.MOSTLY_BF16)


# -------------------- export_ollama_modelfile --------------------


def test_export_ollama_modelfile_emits_gguf_and_modelfile(tmp_path):
    from nnx.interop import export_ollama_modelfile

    net = _tiny_transformer()
    tok = _tiny_tokenizer(tmp_path)
    out_dir = tmp_path / "bundle"

    mf_path = export_ollama_modelfile(
        net,
        tok,
        out_dir,
        system="You are a helpful assistant.",
        parameters={"temperature": 0.8, "top_k": 40},
    )
    # Both artifacts present.
    assert (out_dir / "model.gguf").exists()
    assert (out_dir / "Modelfile").exists()
    # The returned path is the Modelfile.
    assert str(mf_path) == str(out_dir / "Modelfile")

    text = (out_dir / "Modelfile").read_text()
    # The FROM line is load-bearing — Ollama refuses without it.
    assert text.startswith("FROM ./model.gguf"), text
    assert "PARAMETER temperature 0.8" in text
    assert "PARAMETER top_k 40" in text
    assert 'SYSTEM """You are a helpful assistant."""' in text


def test_export_ollama_modelfile_with_list_parameter_emits_repeated_lines(tmp_path):
    """`stop` is a list parameter — Ollama expects one PARAMETER stop
    line per stop string."""
    from nnx.interop import export_ollama_modelfile

    net = _tiny_transformer()
    tok = _tiny_tokenizer(tmp_path)
    out_dir = tmp_path / "bundle"

    export_ollama_modelfile(
        net,
        tok,
        out_dir,
        parameters={"stop": ["<eos>", "<unk>"]},
    )
    text = (out_dir / "Modelfile").read_text()
    assert "PARAMETER stop <eos>" in text
    assert "PARAMETER stop <unk>" in text


def test_export_ollama_modelfile_with_template(tmp_path):
    from nnx.interop import export_ollama_modelfile

    net = _tiny_transformer()
    tok = _tiny_tokenizer(tmp_path)
    out_dir = tmp_path / "bundle"

    template = "{{ .System }}\nUser: {{ .Prompt }}\nAssistant:"
    export_ollama_modelfile(net, tok, out_dir, template=template)
    text = (out_dir / "Modelfile").read_text()
    assert f'TEMPLATE """{template}"""' in text


def test_export_ollama_modelfile_minimal_no_system_no_params(tmp_path):
    """Even with no extras, the FROM line is sufficient for `ollama create`."""
    from nnx.interop import export_ollama_modelfile

    net = _tiny_transformer()
    tok = _tiny_tokenizer(tmp_path)
    out_dir = tmp_path / "bundle"

    export_ollama_modelfile(net, tok, out_dir)
    text = (out_dir / "Modelfile").read_text()
    # Just the FROM line + trailing newline.
    assert text == "FROM ./model.gguf\n", repr(text)


def test_export_ollama_modelfile_writes_utf8_for_non_ascii_system(tmp_path):
    """Regression: `Path.write_text` was missing `encoding="utf-8"`, so
    a SYSTEM prompt with non-ASCII content (Asian-language fine-tune,
    emoji prompt) silently mojibake-encoded on Windows pre-PEP-686
    (locale-default = cp1252). Round-trip as bytes + utf-8 decode to
    catch any locale-default regression even on Linux/macOS runners."""
    from nnx.interop import export_ollama_modelfile

    net = _tiny_transformer()
    tok = _tiny_tokenizer(tmp_path)
    out_dir = tmp_path / "bundle"

    # Mix of Japanese, accented Latin, and an emoji — every one of these
    # would corrupt under cp1252 / latin-1 / ascii encoders.
    non_ascii_system = "あなたは親切なアシスタントです。Café ☕"

    export_ollama_modelfile(net, tok, out_dir, system=non_ascii_system)

    # Decode as utf-8 from raw bytes; if the file was written with the
    # locale-default encoder on a non-utf-8 platform, this would either
    # raise UnicodeDecodeError or read mojibake.
    raw = (out_dir / "Modelfile").read_bytes()
    text = raw.decode("utf-8")
    assert non_ascii_system in text, "non-ASCII SYSTEM content did not round-trip as utf-8"


def test_export_ollama_modelfile_rejects_injection_shaped_inputs(tmp_path):
    """Modelfiles are line/token-delimited: an embedded triple-quote
    terminates a SYSTEM/TEMPLATE block early and a newline in a
    parameter value injects whole directives — the boundary must reject
    these BEFORE the expensive GGUF write (so the model args are never
    touched on the failure path)."""
    import pytest

    from nnx.interop import export_ollama_modelfile

    with pytest.raises(ValueError, match="triple-quote"):
        export_ollama_modelfile(None, None, str(tmp_path / "a"), system='x"""y')
    with pytest.raises(ValueError, match="whitespace"):
        export_ollama_modelfile(None, None, str(tmp_path / "b"), parameters={"bad key": 1})
    with pytest.raises(ValueError, match="newlines"):
        export_ollama_modelfile(None, None, str(tmp_path / "c"), parameters={"stop": "a\nFROM /etc/x"})


def test_write_gguf_creates_parent_directories(tmp_path):
    """write_gguf("out/model.gguf") from a fresh cwd previously raised
    FileNotFoundError — the ollama exporter mkdirs, the raw writer
    didn't."""
    from nnx.interop.gguf import write_gguf

    net = _tiny_transformer()
    tokenizer = _tiny_tokenizer(tmp_path)
    nested = tmp_path / "deep" / "nested" / "model.gguf"
    out = write_gguf(net, tokenizer, nested)
    assert nested.exists()
    assert out == str(nested)
