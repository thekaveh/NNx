"""safetensors round-trip for NNCheckpoint — write then read, weights bit-exact, params recovered.

These tests exercise the opt-in `format="safetensors"` path on
:meth:`NNCheckpoint.to_file` and the magic-byte sniff inside
:meth:`NNCheckpoint.from_file`. They require the `thekaveh-nnx[hub]` extra
(safetensors + huggingface_hub).
"""

from __future__ import annotations

import pytest
import torch

# Same optional-extra convention every other gated test file follows:
# skip gracefully when the required extras aren't installed (the
# shipped sdist's suite must not hard-fail without them).
pytest.importorskip("safetensors")

from nnx import (  # noqa: E402
    Activations,
    Devices,
    Losses,
    Nets,
    NNCheckpoint,
    NNEvaluationDataPoint,
    NNIterationDataPoint,
    NNModel,
    NNModelParams,
    NNParams,
)


def _tiny_model() -> NNModel:
    return NNModel(
        net_params=NNParams(
            input_dim=4,
            output_dim=2,
            hidden_dims=[8],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )


def _tiny_idp() -> NNIterationDataPoint:
    edp = NNEvaluationDataPoint(
        f1=0.5,
        recall=0.5,
        accuracy=0.5,
        precision=0.5,
        loss=0.4,
        error=0.5,
    )
    return NNIterationDataPoint(lr=1e-3, iter_idx=0, epoch_idx=0, batch_idx=0, train_edp=edp)


def _build_checkpoint(model: NNModel) -> NNCheckpoint:
    return NNCheckpoint(
        idp=_tiny_idp(),
        model_params=model.params,
        net_params=model.net_params,
        net_state=model.net.state_dict(),
    )


def test_checkpoint_safetensors_round_trip(tmp_path):
    """write → read recovers identical params dicts and bit-exact tensors."""
    m = _tiny_model()
    ckpt = _build_checkpoint(m)

    p = tmp_path / "ckpt.safetensors"
    ckpt.to_file(str(p), format="safetensors")
    assert p.exists()

    rt = NNCheckpoint.from_file(str(p))
    assert rt is not None
    assert rt.model_params.state() == m.params.state()
    assert rt.net_params.state() == m.net_params.state()
    for k in m.net.state_dict():
        assert torch.equal(m.net.state_dict()[k], rt.net_state[k])


def test_checkpoint_safetensors_round_trip_preserves_idp(tmp_path):
    """The NNIterationDataPoint is JSON-serialized into the safetensors metadata
    and reconstructed on load — its `.state()` must round-trip bit-exact.
    """
    m = _tiny_model()
    ckpt = _build_checkpoint(m)

    p = tmp_path / "ckpt.safetensors"
    ckpt.to_file(str(p), format="safetensors")

    rt = NNCheckpoint.from_file(str(p))
    assert rt is not None
    assert rt.idp.state() == ckpt.idp.state()


def test_checkpoint_pickle_default_unchanged(tmp_path):
    """`format` defaults to "pickle" — no kwarg given keeps existing behavior
    (a plain torch.save file readable by `from_file` via the pickle path).
    """
    m = _tiny_model()
    ckpt = _build_checkpoint(m)

    p = tmp_path / "ckpt.pt"
    ckpt.to_file(str(p))  # no format= kwarg
    assert p.exists()

    rt = NNCheckpoint.from_file(str(p))
    assert rt is not None
    # Pickle preserves the OrderedDict + dataclass identity exactly.
    assert rt.model_params == m.params
    assert rt.net_params == m.net_params


def test_checkpoint_from_file_magic_byte_sniff(tmp_path):
    """`from_file` must dispatch on the prefix: modern torch.save writes a
    ZIP container (``PK\\x03\\x04``); legacy / bare pickle starts with
    ``\\x80``; safetensors starts with neither (it begins with a u64
    header length followed by a JSON object). The same `from_file` call
    returns the right type for both files without an explicit format arg.
    """
    m = _tiny_model()
    ckpt = _build_checkpoint(m)

    p_pickle = tmp_path / "ckpt.pt"
    p_safe = tmp_path / "ckpt.safetensors"
    ckpt.to_file(str(p_pickle), format="pickle")
    ckpt.to_file(str(p_safe), format="safetensors")

    # Prefix sanity check — protects the magic-byte sniff against the
    # day torch changes its on-disk container.
    pickle_head = p_pickle.read_bytes()[:4]
    assert pickle_head == b"PK\x03\x04" or pickle_head[:1] == b"\x80"
    safe_head = p_safe.read_bytes()[:4]
    assert safe_head != b"PK\x03\x04"
    assert safe_head[:1] != b"\x80"

    rt_pickle = NNCheckpoint.from_file(str(p_pickle))
    rt_safe = NNCheckpoint.from_file(str(p_safe))
    assert rt_pickle is not None and rt_safe is not None
    assert rt_pickle.model_params.state() == rt_safe.model_params.state()


def test_checkpoint_to_file_rejects_unknown_format(tmp_path):
    m = _tiny_model()
    ckpt = _build_checkpoint(m)
    with pytest.raises(ValueError, match="unknown checkpoint format"):
        ckpt.to_file(str(tmp_path / "ckpt.bin"), format="hdf5")  # type: ignore[arg-type]


def test_checkpoint_safetensors_handles_tied_weights(tmp_path):
    """A default TransformerNN ties tok_embed/lm_head storage, and its
    state_dict carries BOTH keys pointing at one tensor — safetensors
    rejects shared storage, so to_file(format="safetensors") crashed on
    every tied-weight net pre-fix (.contiguous() is a no-op on an
    already-contiguous view; .clone() is what breaks the aliasing).
    The reload assigns both identical copies back into the tied
    parameter, so the round-trip preserves values and the tie."""
    pytest.importorskip("safetensors")
    from nnx.nn.net.transformer_nn import TransformerNN
    from nnx.nn.params.nn_transformer_params import NNTransformerParams

    torch.manual_seed(0)
    params = NNTransformerParams(
        input_dim=64,
        output_dim=64,
        dropout_prob=0.0,
        vocab_size=64,
        n_layers=1,
        n_heads=2,
        d_model=16,
        ffn_mult=2,
        max_seq_len=8,
    )
    net = TransformerNN(params)
    ckpt = NNCheckpoint(
        net_params=params,
        net_state=net.state_dict(),
        model_params=NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        idp=_tiny_idp(),
    )
    path = str(tmp_path / "tied.safetensors")
    ckpt.to_file(path, format="safetensors")

    loaded = NNCheckpoint.from_file(path)
    assert loaded is not None
    assert torch.equal(loaded.net_state["tok_embed.weight"], net.state_dict()["tok_embed.weight"])
    assert torch.equal(loaded.net_state["lm_head.weight"], net.state_dict()["lm_head.weight"])
    # Loading back into a fresh tied net keeps the tie intact.
    net2 = TransformerNN(params)
    net2.load_state_dict(loaded.net_state)
    assert net2.lm_head.weight is net2.tok_embed.weight


def test_checkpoint_sniff_handles_0x80_header_length(tmp_path):
    """A safetensors header length ≡ 128 mod 256 makes the file's FIRST
    byte 0x80 — the pickle PROTO opcode. Pre-fix the magic-byte sniff
    routed such files to torch.load, which died with a confusing
    UnpicklingError. Byte 8 (the JSON header's '{') now positively
    identifies safetensors before the pickle check."""
    pytest.importorskip("safetensors")
    import json

    from safetensors.torch import save_file

    model = _tiny_model()
    base_meta = {
        "nnx_format_version": "1",
        "model_params": json.dumps(model.params.state()),
        "net_params": json.dumps(model.net_params.state()),
        "idp": json.dumps(_tiny_idp().state()),
    }
    tensors = {k: v.detach().clone() for k, v in model.net.state_dict().items()}
    path = str(tmp_path / "padded.safetensors")
    for pad in range(256):
        save_file(tensors, path, metadata={**base_meta, "pad": "x" * pad})
        with open(path, "rb") as f:
            if f.read(1) == b"\x80":
                break
    else:  # pragma: no cover — alignment should always allow 0x80
        pytest.skip("could not coax a 0x80 leading byte out of the header alignment")

    loaded = NNCheckpoint.from_file(path)
    assert loaded is not None
    assert loaded.idp == _tiny_idp()
