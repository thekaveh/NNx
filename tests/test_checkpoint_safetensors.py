"""safetensors round-trip for NNCheckpoint — write then read, weights bit-exact, params recovered.

These tests exercise the opt-in `format="safetensors"` path on
:meth:`NNCheckpoint.to_file` and the magic-byte sniff inside
:meth:`NNCheckpoint.from_file`. They require the `thekaveh-nnx[hub]` extra
(safetensors + huggingface_hub).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

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
    NNCheckpointTransform,
    NNEvaluationDataPoint,
    NNIterationDataPoint,
    NNModel,
    NNModelParams,
    NNParams,
)


def test_pickle_checkpoint_load_defaults_to_cpu_map_location(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"PK\x03\x04" + b"0" * 8)
    observed = {}

    def fake_load(load_path, **kwargs):
        observed.update(kwargs)
        return None

    monkeypatch.setattr(torch, "load", fake_load)

    NNCheckpoint.from_file(str(path))

    assert observed["map_location"] == "cpu"


def test_from_checkpoint_device_override_rebuilds_cuda_metadata_on_cpu():
    model = _tiny_model()
    checkpoint = _build_checkpoint(model)
    cuda_checkpoint = replace(checkpoint, model_params=replace(checkpoint.model_params, device=Devices.CUDA))

    restored = NNModel.from_checkpoint(cuda_checkpoint, device=Devices.CPU)

    assert restored.device.type == "cpu"


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


@pytest.mark.parametrize("map_location", [{"cuda:0": "cpu"}, lambda storage, location: storage])
def test_checkpoint_safetensors_rejects_unsupported_map_location(tmp_path, map_location):
    checkpoint = _build_checkpoint(_tiny_model())
    path = tmp_path / "checkpoint.safetensors"
    checkpoint.to_file(str(path), format="safetensors")

    with pytest.raises(TypeError, match="safetensors.*map_location"):
        NNCheckpoint.from_file(str(path), map_location=map_location)


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


@pytest.mark.parametrize("format", ["pickle", "safetensors"])
def test_checkpoint_formats_round_trip_transform_metadata(tmp_path, format):
    model = _tiny_model()
    transform = NNCheckpointTransform(
        name="torchao_qat",
        version=1,
        options={"qat_config": "8da4w", "groupsize": 16},
    )
    checkpoint = NNCheckpoint(
        idp=_tiny_idp(),
        model_params=model.params,
        net_params=model.net_params,
        net_state=model.net.state_dict(),
        transforms=(transform,),
    )
    path = tmp_path / f"checkpoint.{format}"

    checkpoint.to_file(str(path), format=format)
    loaded = NNCheckpoint.from_file(str(path))

    assert loaded is not None
    assert loaded.transforms == (transform,)


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


def test_checkpoint_pickle_bare_filename_uses_destination_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    checkpoint = _build_checkpoint(_tiny_model())

    checkpoint.to_file("checkpoint.pt")

    assert (tmp_path / "checkpoint.pt").is_file()


def test_snapshot_state_dict_deep_copies_extra_state_and_metadata():
    from nnx.nn.params.nn_checkpoint import _snapshot_state_dict

    state = OrderedDict(
        [
            ("weight", torch.tensor([1.0])),
            ("_extra_state", {"labels": ["before"]}),
        ]
    )
    state._metadata = OrderedDict({"": {"version": 1}})  # type: ignore[attr-defined]

    snapshot = _snapshot_state_dict(state)
    state["weight"].add_(1)
    state["_extra_state"]["labels"].append("after")
    state._metadata[""]["version"] = 2  # type: ignore[attr-defined]

    assert torch.equal(snapshot["weight"], torch.tensor([1.0]))
    assert snapshot["_extra_state"] == {"labels": ["before"]}
    assert snapshot._metadata[""]["version"] == 1


def test_safetensors_checkpoint_rejects_non_tensor_extra_state_clearly(tmp_path):
    model = _tiny_model()
    state = model.net.state_dict()
    state["_extra_state"] = {"labels": ["class-a"]}
    checkpoint = NNCheckpoint(
        idp=_tiny_idp(),
        model_params=model.params,
        net_params=model.net_params,
        net_state=state,
    )

    with pytest.raises(TypeError, match="safetensors checkpoint export.*non-tensor state_dict"):
        checkpoint.to_file(str(tmp_path / "extra.safetensors"), format="safetensors")


def test_legacy_pickle_without_transform_slot_loads_as_empty(tmp_path, monkeypatch):
    """Pre-transform pickles serialized only the original four dataclass slots."""
    model = _tiny_model()
    checkpoint = _build_checkpoint(model)
    original_getstate = NNCheckpoint.__getstate__

    def legacy_getstate(self):
        return original_getstate(self)[:-1]

    monkeypatch.setattr(NNCheckpoint, "__getstate__", legacy_getstate)
    path = tmp_path / "legacy.pt"
    checkpoint.to_file(str(path))

    loaded = NNCheckpoint.from_file(str(path))
    assert loaded is not None
    assert loaded.transforms == ()
    reloaded_model = NNModel.from_checkpoint(loaded)
    assert set(reloaded_model.net.state_dict()) == set(model.net.state_dict())


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
    assert loaded.transforms == ()


def test_checkpoint_rejects_unsupported_safetensors_format_version(tmp_path):
    pytest.importorskip("safetensors")
    import json

    from safetensors.torch import save_file

    model = _tiny_model()
    path = str(tmp_path / "future.safetensors")
    save_file(
        {key: value.detach().clone() for key, value in model.net.state_dict().items()},
        path,
        metadata={
            "nnx_format_version": "999",
            "model_params": json.dumps(model.params.state()),
            "net_params": json.dumps(model.net_params.state()),
            "idp": json.dumps(_tiny_idp().state()),
        },
    )

    with pytest.raises(ValueError, match="unsupported safetensors checkpoint format version"):
        NNCheckpoint.from_file(path)


def test_checkpoint_safetensors_keeps_transformer_per_layer_lists(tmp_path):
    """FEAT-041: the safetensors metadata stores `net_params.state()`; the
    transformer's per-layer `activations` / `dropout_probs` must come back
    (NNTransformerParams.from_state used to drop them)."""
    from nnx import NNTransformerParams

    params = NNTransformerParams(
        input_dim=64,
        output_dim=64,
        dropout_prob=0.0,
        hidden_dims=[7, 5],
        activations=[Activations.RELU, Activations.TANH],
        dropout_probs=[0.1, 0.3],
        n_heads=2,
        vocab_size=64,
        n_layers=1,
        d_model=16,
        max_seq_len=8,
    )
    checkpoint = NNCheckpoint(
        idp=_tiny_idp(),
        model_params=NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        net_params=params,
        net_state={"w": torch.zeros(1)},
    )
    path = tmp_path / "transformer.safetensors"
    checkpoint.to_file(str(path), format="safetensors")
    restored = NNCheckpoint.from_file(str(path))
    assert restored is not None
    assert isinstance(restored.net_params, NNTransformerParams)
    assert restored.net_params == params


def test_checkpoint_safetensors_keeps_the_monitor_selection_and_training_summary(tmp_path):
    """FEAT-003: the selected checkpoint's metadata keeps the monitor's
    identity, direction, configuration and status, and the whole-epoch
    training summary, through the safetensors reader too."""
    import math

    from nnx import MonitorRecord, MonitorSpec

    monitor = MonitorSpec(metric="nll", mode="min", min_delta=0.01, on_missing="error")
    for record in (
        MonitorRecord(monitor, 0.7, "ok", True),
        MonitorRecord(monitor, math.inf, "nonfinite", False),
        MonitorRecord(monitor, None, "missing", False),
    ):
        idp = NNIterationDataPoint(
            lr=0.01,
            iter_idx=4,
            epoch_idx=1,
            batch_idx=2,
            train_edp=NNEvaluationDataPoint(loss=0.9, error=0.4),
            val_edp=NNEvaluationDataPoint(loss=0.8, error=0.3, metrics={"nll": 0.7}),
            train_summary=NNEvaluationDataPoint(loss=0.85, error=0.35, metrics={"nll": 0.72}),
            selection=record,
        )
        checkpoint = replace(_build_checkpoint(_tiny_model()), idp=idp)
        path = tmp_path / f"monitor_{record.status}.safetensors"
        checkpoint.to_file(str(path), format="safetensors")
        loaded = NNCheckpoint.from_file(str(path))
        assert loaded.idp.selection == record
        assert loaded.idp.train_summary == idp.train_summary
        assert loaded.idp.val_edp == idp.val_edp


# ---------- FEAT-006: registered and runtime-only modules ----------


class _ToyEncoder(torch.nn.Module):
    def __init__(self, width: int = 6) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(4, width), torch.nn.ReLU(), torch.nn.Linear(width, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def _fit_briefly(model: NNModel) -> str:
    from torch.utils.data import DataLoader, TensorDataset

    from nnx import NNOptimParams, NNTrainParams

    X = torch.randn(8, 4, generator=torch.Generator().manual_seed(0))
    loader = DataLoader(TensorDataset(X, (X[:, 0] > 0).long()), batch_size=4)
    run = model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=loader,
            optim=NNOptimParams.builder().sgd(max_lr=0.05).build(),
            save_phase_checkpoints=False,
        )
    )
    return run.id


def test_runtime_modules_write_non_reconstructible_weights_and_refuse_portable_saves(tmp_path, monkeypatch):
    from nnx import MissingModelFactoryError
    from nnx.nn.enum.checkpoints import Checkpoints

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    model = NNModel(module=_ToyEncoder(), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    run_id = _fit_briefly(model)
    checkpoint = NNCheckpoint.load(run=run_id, type=Checkpoints.LAST)
    assert checkpoint is not None and checkpoint.net_params is None
    assert checkpoint.reconstructible is False
    assert checkpoint.model_params.state()["net"]["reconstructible"] is False
    assert set(checkpoint.net_state) == set(model.net.state_dict())  # the weights themselves are kept
    target = tmp_path / "portable" / "encoder.safetensors"
    with pytest.raises(MissingModelFactoryError, match="runtime-only module"):
        checkpoint.to_file(str(target), format="safetensors")
    assert not target.parent.exists()  # rejected before any directory or file was written


def test_registered_modules_round_trip_through_safetensors(tmp_path, monkeypatch):
    from nnx import ModelSpec, register_model_factory, unregister_model_factory
    from nnx.nn.enum.checkpoints import Checkpoints

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    register_model_factory("tests.safetensors_encoder", 1, lambda config: _ToyEncoder(**config))
    try:
        spec = ModelSpec("tests.safetensors_encoder", 1, {"width": 6})
        model = NNModel(params=NNModelParams(net=spec, loss=Losses.CROSS_ENTROPY))
        checkpoint = NNCheckpoint.load(run=_fit_briefly(model), type=Checkpoints.LAST)
        assert checkpoint is not None
        path = str(tmp_path / "encoder.safetensors")
        checkpoint.to_file(path, format="safetensors")
        loaded = NNCheckpoint.from_file(path)
        assert loaded is not None and loaded.net_params is None and loaded.model_params.net == spec
        rebuilt = NNModel.from_checkpoint(loaded)
        X = torch.randn(3, 4)
        torch.testing.assert_close(torch.as_tensor(rebuilt.predict(X).logits), torch.as_tensor(model.predict(X).logits))
    finally:
        unregister_model_factory("tests.safetensors_encoder", 1)
