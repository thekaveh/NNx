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
import re
from typing import Any

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


@pytest.mark.parametrize("adapter", ["lora", "ia3", "prefix"])
def test_tensor_name_map_refuses_a_peft_adapted_network(adapter):
    """GGUF stores plain weights: a wrapped projection has no weight of its
    own (it raised AttributeError), and a prefix tuner's prefix has no GGUF
    slot (it was silently dropped). Both are refused with a hint."""
    from nnx.interop.gguf.tensor_name_map import map_tensors
    from nnx.peft import PrefixTuner, apply_ia3_to, apply_lora_to

    net = _tiny_transformer()
    if adapter == "lora":
        apply_lora_to(net, "blocks.*.attn.w_qkv", r=2)
    elif adapter == "ia3":
        apply_ia3_to(net, "blocks.*.attn.w_qkv")
    else:
        PrefixTuner(net, n_prefix=2)
    with pytest.raises(TypeError, match="PEFT adapter"):
        map_tensors(net)


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
    verify shapes AND tensor values round-trip. This proves a generic
    GGUF parser can inspect the container and that the bytes are the
    model's actual weights, not runtime architecture compatibility."""
    import gguf
    import numpy as np

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
        # Values must survive the F16 write+read, not just the shapes. The
        # on-disk buffer is the same row-major order as the source numpy
        # array, so compare flattened with the F16-rounded reference
        # (exact, since write_gguf casts to F16 the same way).
        observed = t.data.astype(np.float32).flatten()
        expected_f16 = expected[t.name].astype(np.float16).astype(np.float32).flatten()
        assert np.array_equal(observed, expected_f16), t.name


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
    # The FROM line is required by Modelfile syntax. Runtime architecture
    # support is explicitly outside this bundle-structure test.
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
    """With no extras, the generated Modelfile contains only FROM."""
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


@pytest.mark.parametrize(
    "parameters",
    [
        {"unknown": 1},
        {"top_k": 1.5},
        {"top_k": True},
        {"temperature": True},
        {"temperature": float("nan")},
        {"top_p": float("inf")},
        {"num_ctx": [4096]},
        {"stop": 42},
        {"stop": ["valid", 42]},
        {"stop": "nul\0byte"},
        {"stop": "carriage\rreturn"},
    ],
)
def test_export_ollama_modelfile_rejects_invalid_parameter_contracts(tmp_path, parameters):
    from nnx.interop import export_ollama_modelfile

    with pytest.raises((TypeError, ValueError), match="parameter"):
        export_ollama_modelfile(None, None, str(tmp_path / "invalid"), parameters=parameters)


def test_export_ollama_modelfile_accepts_every_documented_parameter(tmp_path):
    from nnx.interop import export_ollama_modelfile

    net = _tiny_transformer()
    tok = _tiny_tokenizer(tmp_path)
    out_dir = tmp_path / "bundle"
    parameters = {
        "num_ctx": 4096,
        "repeat_last_n": 64,
        "seed": 7,
        "num_predict": 128,
        "draft_num_predict": 16,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "temperature": 0.8,
        "top_p": 0.9,
        "min_p": 0.05,
        "stop": ["<eos>", "<end>"],
    }

    export_ollama_modelfile(net, tok, out_dir, parameters=parameters)
    text = (out_dir / "Modelfile").read_text(encoding="utf-8")
    for name, value in parameters.items():
        values = value if isinstance(value, list) else [value]
        for item in values:
            assert f"PARAMETER {name} {item}" in text


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


def test_tensor_name_map_upcasts_bf16_weights_to_float32():
    """numpy has no bfloat16: a bf16-resident net maps to exact float32
    arrays instead of raising in .numpy()."""
    import numpy as np
    import torch

    from nnx.interop.gguf.tensor_name_map import map_tensors

    net = _tiny_transformer().to(torch.bfloat16)
    tensors = map_tensors(net)
    assert all(arr.dtype == np.float32 for arr in tensors.values())
    assert np.array_equal(tensors["token_embd.weight"], net.tok_embed.weight.detach().float().numpy())


def test_a_huge_integer_float_parameter_is_a_type_error():
    """10**400 cannot become a float: refused as a non-finite number, not
    an uncaught OverflowError."""
    from nnx.interop.ollama import _validate_modelfile_inputs

    with pytest.raises(TypeError, match="requires a finite number"):
        _validate_modelfile_inputs("", None, {"temperature": 10**400})


@pytest.mark.parametrize(
    ("label", "error", "match"), [("F8", ValueError, "F8"), ("Q4_K_M", ImportError, "llama-quantize")]
)
def test_the_quantization_label_is_judged_before_gguf_is_required(monkeypatch, label, error, match):
    """Without gguf installed, a misspelled label is still a ValueError and a
    sub-F16 one still gets the two-step llama-quantize guidance."""
    import sys

    from nnx.interop.gguf.writer import write_gguf

    monkeypatch.setitem(sys.modules, "gguf", None)
    with pytest.raises(error, match=match):
        write_gguf(_tiny_transformer(), None, "out.gguf", quantization=label)  # type: ignore[arg-type]


def test_a_factorized_projection_is_refused_with_a_type_error():
    """A low-rank factorized projection is a Sequential with no weight of
    its own: refused like an adapter, not an AttributeError mid-export."""
    from nnx.interop.gguf.tensor_name_map import _refuse_adapters
    from nnx.surgery import low_rank_factorize

    net = _tiny_transformer()
    ffn = net.blocks[0].ffn
    ffn.w1 = low_rank_factorize(ffn.w1, rank=4)
    with pytest.raises(TypeError, match="plain nn.Linear projections"):
        _refuse_adapters(net)


@pytest.mark.parametrize("case", ["not-a-transformer", "lora", "factorized", "tokenizer"])
def test_a_refused_gguf_export_creates_no_directory(tmp_path, case):
    """The network and the tokenizer are read (and refused) before the
    parent directory is created: a refused export leaves nothing on disk."""
    import torch

    from nnx.interop.gguf import write_gguf
    from nnx.peft import apply_lora_to
    from nnx.surgery import low_rank_factorize

    if case == "not-a-transformer":
        net = torch.nn.Linear(2, 2)
    else:
        net = _tiny_transformer()
        if case == "lora":
            apply_lora_to(net, "blocks.*.attn.w_qkv", r=2)
        elif case == "factorized":
            net.blocks[0].ffn.w1 = low_rank_factorize(net.blocks[0].ffn.w1, rank=4)
    path = tmp_path / "gguf" / "model.gguf"
    with (
        pytest.raises(AttributeError, match="get_vocab")
        if case == "tokenizer"
        else pytest.raises((AttributeError, TypeError))
    ):
        write_gguf(net, None, path)  # type: ignore[arg-type]
    assert not (tmp_path / "gguf").exists()


@pytest.mark.parametrize("case", ["quantization", "lora", "tokenizer"])
def test_a_refused_ollama_export_creates_no_directory(tmp_path, case):
    from nnx.interop.ollama import export_ollama_modelfile
    from nnx.peft import apply_lora_to

    out = tmp_path / "ollama" / "bundle"
    if case == "quantization":
        with pytest.raises(ValueError, match="Q9_BAD"):
            export_ollama_modelfile(None, None, out, quantization="Q9_BAD")  # type: ignore[arg-type]
    elif case == "lora":
        net = _tiny_transformer()
        apply_lora_to(net, "blocks.*.attn.w_qkv", r=2)
        with pytest.raises(TypeError, match="PEFT adapter"):
            export_ollama_modelfile(net, None, out)  # type: ignore[arg-type]
    else:
        with pytest.raises(AttributeError, match="get_vocab"):
            export_ollama_modelfile(_tiny_transformer(), None, out)  # type: ignore[arg-type]
    assert not (tmp_path / "ollama").exists()


# -------------------- unbaked reparameterizations (FIX-058) --------------------
#
# An unbaked prune mask and a legacy weight_norm / spectral_norm hook
# recompute ``.weight`` in a forward pre-hook, so the cached tensor the
# export reads goes stale on a load_state_dict or optimizer step, and a
# masked tied head has no GGUF slot of its own (token_embd holds the
# unmasked embedding). Export refuses them, by hook type.

_PRUNE_FIX = ("torch.nn.utils.prune.remove", "magnitude_prune(bake=True)")
_NORM_FIX = ("torch.nn.utils.remove_weight_norm", "remove_spectral_norm", "torch.nn.utils.parametrizations")


def _pruned(seed: int, pattern: str = "blocks.*") -> TransformerNN:
    import torch

    from nnx.prune import magnitude_prune

    torch.manual_seed(seed)
    net = _tiny_transformer()
    magnitude_prune(net, 0.5, layer_pattern=pattern, bake=False)
    return net


def _legacy_normed(seed: int, norm: str, path: str) -> TransformerNN:
    import warnings

    import torch

    torch.manual_seed(seed)
    net = _tiny_transformer()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # legacy weight_norm's deprecation notice
        getattr(torch.nn.utils, norm)(net.get_submodule(path))
    return net


def _refused_net(case: str) -> tuple[TransformerNN, str, tuple[str, ...]]:
    """The net, a tensor the refusal must name, and the fix it must give."""
    import torch

    if case == "reload":  # an unbaked checkpoint reloaded into a re-pruned net
        net = _pruned(1)
        net.load_state_dict(_pruned(0).state_dict())
        return net, "blocks.0.attn.w_qkv.weight", _PRUNE_FIX
    if case == "finetuned":  # a prune-then-finetune loop ending on opt.step()
        net = _pruned(0)
        opt = torch.optim.SGD(net.parameters(), lr=0.1)
        net(torch.randint(0, 64, (2, 8))).pow(2).mean().backward()
        opt.step()
        return net, "blocks.0.ffn.w1.weight", _PRUNE_FIX
    if case == "tied-head":  # no reload or step: the masked tied head has no GGUF slot
        return _pruned(0, pattern="*"), "lm_head.weight is tied", _PRUNE_FIX
    norm, path = {
        "weight_norm": ("weight_norm", "blocks.0.ffn.w1"),
        "spectral_norm": ("spectral_norm", "blocks.1.attn.w_o"),
        "embedding-weight_norm": ("weight_norm", "tok_embed"),  # not a projection
    }[case]
    net = _legacy_normed(1, norm, path)
    net.load_state_dict(_legacy_normed(0, norm, path).state_dict())
    return net, f"{path}.weight", _NORM_FIX


_REFUSED_CASES = ["reload", "finetuned", "tied-head", "weight_norm", "spectral_norm", "embedding-weight_norm"]


@pytest.mark.parametrize("exporter", ["map_tensors", "write_gguf", "export_ollama_modelfile"])
@pytest.mark.parametrize("case", _REFUSED_CASES)
def test_an_unbaked_reparameterization_is_refused_before_anything_is_written(tmp_path, case, exporter):
    """Every case wrote a stale or unmasked weight (up to 0.35 off per entry)
    with no error; each exporter now raises TypeError naming the tensor and
    the fix, and leaves no .gguf, model.gguf or Modelfile behind."""
    from nnx.interop.gguf import write_gguf
    from nnx.interop.gguf.tensor_name_map import map_tensors
    from nnx.interop.ollama import export_ollama_modelfile

    net, name, fixes = _refused_net(case)
    tokenizer = _tiny_tokenizer(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(TypeError, match="GGUF export reads") as info:
        if exporter == "map_tensors":
            map_tensors(net)
        elif exporter == "write_gguf":
            write_gguf(net, tokenizer, out / "model.gguf")
        else:
            export_ollama_modelfile(net, tokenizer, out)
    message = str(info.value)
    assert name in message
    assert all(fix in message for fix in fixes), message
    assert not out.exists()


def test_the_refusal_detects_the_hooks_by_type_not_by_parameter_names():
    """Plain parameters that merely share the reparameterizations' names
    export; a prune hook on a non-projection module is still refused."""
    import torch

    from nnx.interop.gguf.tensor_name_map import map_tensors

    net = _tiny_transformer()
    lin = net.blocks[0].ffn.w1
    for suffix in ("_orig", "_mask", "_g", "_v"):
        lin.register_parameter(f"weight{suffix}", torch.nn.Parameter(torch.zeros_like(lin.weight)))
    assert "blk.0.ffn_gate.weight" in map_tensors(net)

    net = _tiny_transformer()
    torch.nn.utils.prune.random_unstructured(net.norm_out, "weight", amount=0.5)
    with pytest.raises(TypeError, match=r"norm_out\.weight"):
        map_tensors(net)


@pytest.mark.parametrize(
    ("n", "listed"),
    [
        (2, "a.weight and b.weight carry"),
        (3, "a.weight, b.weight and c.weight carry"),
        (7, "a.weight, b.weight, c.weight, d.weight, e.weight … (7 in all) carry"),
    ],
)
def test_the_refusal_lists_the_affected_tensors_as_a_sentence(n, listed):
    """Two to five names join with "and"; more show the first five and a count."""
    import torch

    from nnx.interop.gguf.tensor_name_map import _refuse_reparameterized

    net = torch.nn.Module()
    for name in "abcdefg"[:n]:
        net.add_module(name, torch.nn.Linear(2, 2))
        torch.nn.utils.prune.random_unstructured(getattr(net, name), "weight", amount=0.5)
    with pytest.raises(TypeError, match=re.escape(f"{listed} an unbaked prune mask.")):
        _refuse_reparameterized(net)


def test_a_from_checkpoint_reload_into_an_unbaked_pruned_module_is_refused(tmp_path, monkeypatch):
    """NNModel(module=pruned).train, then from_checkpoint(module=a fresh
    unbaked net): its cached weight is the fresh net's init (about 0.35 off
    per entry), which write_gguf exported."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from nnx import Losses, NNCheckpoint, NNModel, NNModelParams, NNOptimParams, NNTrainParams
    from nnx.interop.gguf import write_gguf
    from nnx.lm_tasks import CausalLMTask
    from nnx.nn.enum.checkpoints import Checkpoints

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    ids = torch.randint(0, 64, (8, 6), generator=torch.Generator().manual_seed(0))
    model = NNModel(module=_pruned(0), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    run = model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=DataLoader(TensorDataset(ids, ids), batch_size=4),
            optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
            save_phase_checkpoints=False,
        ),
        objective=CausalLMTask(vocab_size=64, alignment="pre_shifted").objective(),
    )
    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert checkpoint is not None
    reloaded = NNModel.from_checkpoint(checkpoint, module=_pruned(1))
    out = tmp_path / "out" / "model.gguf"
    with pytest.raises(TypeError, match="unbaked prune mask"):
        write_gguf(reloaded.net, _tiny_tokenizer(tmp_path), out)  # type: ignore[arg-type]
    assert not out.parent.exists()


def _rebuilt_from_gguf(path, params: NNTransformerParams) -> TransformerNN:
    """A plain TransformerNN holding the file's tensors: its forward is what
    a GGUF reader of the file computes."""
    import gguf
    import numpy as np
    import torch

    stored = {t.name: torch.from_numpy(np.array(t.data)) for t in gguf.GGUFReader(str(path)).tensors}
    net = TransformerNN(params=params)

    def put(module: torch.nn.Module, name: str, rows: slice = slice(None)) -> None:
        weight = module.weight
        assert isinstance(weight, torch.Tensor)
        with torch.no_grad():
            weight[rows].copy_(stored[name].reshape(weight[rows].shape))

    d = params.d_model
    put(net.tok_embed, "token_embd.weight")  # also the tied head
    for i, block in enumerate(net.blocks):
        b: Any = block
        put(b.norm1, f"blk.{i}.attn_norm.weight")
        for j, part in enumerate(("q", "k", "v")):
            put(b.attn.w_qkv, f"blk.{i}.attn_{part}.weight", slice(j * d, (j + 1) * d))
        put(b.attn.w_o, f"blk.{i}.attn_output.weight")
        put(b.norm2, f"blk.{i}.ffn_norm.weight")
        put(b.ffn.w1, f"blk.{i}.ffn_gate.weight")
        put(b.ffn.w2, f"blk.{i}.ffn_down.weight")
        put(b.ffn.w3, f"blk.{i}.ffn_up.weight")
    put(net.norm_out, "output_norm.weight")
    return net.eval()


@pytest.mark.parametrize(
    "case", ["bake=True", "prune.remove", "parametrizations.weight_norm", "parametrizations.spectral_norm"]
)
def test_baked_and_parametrized_nets_still_export_the_weights_the_forward_uses(tmp_path, case):
    """A baked mask is a plain weight, and a torch.nn.utils.parametrize
    form computes ``.weight`` on every read, so the file holds the forward's
    weights: a plain net rebuilt from it gives the same logits exactly. The
    spectral_norm parametrization is compared in eval mode, where reading
    the weight does not step its power iteration."""
    import torch
    from torch.nn.utils import parametrizations
    from torch.nn.utils import prune as torch_prune

    from nnx.interop.gguf import write_gguf
    from nnx.prune import magnitude_prune

    torch.manual_seed(0)
    net = _tiny_transformer()
    if case == "bake=True":
        magnitude_prune(net, 0.5)  # every Linear, the tied head included
    elif case == "prune.remove":
        magnitude_prune(net, 0.5, layer_pattern="blocks.*", bake=False)
        net.load_state_dict(_pruned(1).state_dict())
        for module in net.modules():
            if hasattr(module, "weight_mask"):
                torch_prune.remove(module, "weight")
    elif case == "parametrizations.weight_norm":
        parametrizations.weight_norm(net.blocks[0].ffn.w1)
        parametrizations.weight_norm(net.blocks[1].attn.w_qkv)
    else:
        parametrizations.spectral_norm(net.blocks[1].attn.w_o)
    net.eval()
    path = write_gguf(net, _tiny_tokenizer(tmp_path), tmp_path / "model.gguf", quantization="F32")
    rebuilt = _rebuilt_from_gguf(path, net.params)
    x = torch.randint(0, 64, (2, 8), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(rebuilt(x), net(x))
