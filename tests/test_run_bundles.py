"""FEAT-015: portable, data-only run bundles (`nnx.bundles`).

A bundle holds a run checkpoint — weights, training state, calibrators — as
safetensors plus schema-validated JSON. A built-in classifier and a
registered module rebuilt from a bundle predict bit-for-bit like the source
model, and resuming one epoch from the bundle equals the continuous run.
"""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict

import numpy as np
import pytest
import torch

from nnx import (
    Activations,
    Devices,
    Losses,
    ModelSpec,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNSchedulerParams,
    NNTrainParams,
    Schedulers,
    TaskSpec,
    bundles,
    register_model_factory,
    unregister_model_factory,
)
from nnx.bundles import (
    BundleCapabilityError,
    BundleError,
    BundleInfo,
    BundleReconstructionError,
    _decode,
    _encode,
    _tensor_refs,
    export_bundle,
    inspect_bundle,
    reconstruct_bundle,
)
from nnx.calibration import TemperatureCalibrator, model_fingerprint
from nnx.nn.callbacks import EarlyStopping, ModelCheckpoint

LABELS = ("low", "mid", "high")


@pytest.fixture(autouse=True)
def _quiet(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")


def _data():
    generator = torch.Generator().manual_seed(3)
    X = torch.randn(24, 4, generator=generator)
    y = torch.randint(0, 3, (24,), generator=generator)
    return X, [(X[:8], y[:8]), (X[8:16], y[8:16]), (X[16:], y[16:])]


def _classifier() -> NNModel:
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=3, hidden_dims=[8], dropout_prob=0.1, activation=Activations.RELU),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
            task=TaskSpec.categorical(3, labels=LABELS),
        ),
    )


class _Encoder(torch.nn.Module):
    """A registered (non built-in) module with a buffer."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(4, width), torch.nn.ReLU(), torch.nn.Linear(width, 3))
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, x):
        return self.body(x) * self.scale


def _encoder_factory(config):
    return _Encoder(**config)


def _params(n_epochs: int, data_id: str, batches, **extra) -> NNTrainParams:
    return NNTrainParams(
        n_epochs=n_epochs,
        data_id=data_id,
        train_loader=batches,
        optim=NNOptimParams.builder().adam(max_lr=0.01).build(),
        scheduler=NNSchedulerParams(
            kind=Schedulers.STEP, step_size=1, factor=0.5, min_lr=0.0, patience=0, cooldown=0, threshold=0.0
        ),
        seed=0,
        **extra,
    )


def _assert_same_weights(left: NNModel, right: NNModel) -> None:
    right_state = right.net.state_dict()
    for name, tensor in left.net.state_dict().items():
        torch.testing.assert_close(right_state[name], tensor, rtol=0, atol=0, msg=name)


# --- encoding --------------------------------------------------------------------------------


def test_typed_json_round_trips_tuples_integer_keys_tensors_and_non_finite_floats():
    state = {
        "python": (3, tuple(range(5)), None),
        "optimizer": {
            "state": {0: {"step": torch.tensor(4.0), "exp_avg": torch.ones(2, 3)}},
            "param_groups": [{"params": [0], "betas": (0.9, 0.999)}],
        },
        "plateau": {"best": math.inf, "mode_worse": -math.inf, "loss": math.nan},
        "ordered": OrderedDict([("b", 1), ("a", [1.5, True, "x"])]),
        "numpy": np.float64(0.25),
        "rng": torch.ByteTensor([1, 2, 3]),
    }
    tensors: dict[str, torch.Tensor] = {}
    encoded = _encode(state, tensors, "state")
    text = json.dumps(encoded, allow_nan=False)  # plain, strict JSON
    parsed = json.loads(text)
    assert sorted(_tensor_refs(parsed, "state")) == sorted(tensors)
    decoded = _decode(parsed, tensors)
    assert decoded["python"] == (3, (0, 1, 2, 3, 4), None)
    assert list(decoded["optimizer"]["state"]) == [0] and torch.equal(
        decoded["optimizer"]["state"][0]["exp_avg"], torch.ones(2, 3)
    )
    assert decoded["optimizer"]["param_groups"][0]["betas"] == (0.9, 0.999)
    assert decoded["plateau"]["best"] == math.inf and decoded["plateau"]["mode_worse"] == -math.inf
    assert math.isnan(decoded["plateau"]["loss"])
    assert list(decoded["ordered"]) == ["b", "a"] and decoded["ordered"]["a"] == [1.5, True, "x"]
    assert decoded["numpy"] == 0.25 and type(decoded["numpy"]) is float
    assert torch.equal(decoded["rng"], torch.ByteTensor([1, 2, 3])) and decoded["rng"].dtype == torch.uint8


@pytest.mark.parametrize(
    "value, message",
    [
        ({"f": lambda: None}, "function"),
        ({1: "a", "b": 2}, "all strings or all integers"),
        ({"s": torch.zeros(2).to_sparse()}, "dense tensors only"),
        ({"obj": object()}, "no pickle fallback"),
    ],
)
def test_unsupported_state_is_refused_with_its_path(value, message):
    with pytest.raises(BundleError, match=message):
        _encode(value, {}, "training_state")


# --- a built-in classifier -------------------------------------------------------------------


def test_a_built_in_classifier_reproduces_predictions_and_resumes_like_the_continuous_run():
    X, batches = _data()
    torch.manual_seed(7)
    continuous = _classifier()
    continuous.train(params=_params(3, "full", batches))

    torch.manual_seed(7)
    first = _classifier()
    first_run = first.train(params=_params(2, "split", batches))
    info = export_bundle(first_run.id, "bundle")
    assert isinstance(info, BundleInfo) and info.verified and info.capability == "resume"
    assert (info.source_run_id, info.source_checkpoint, info.epoch) == (first_run.id, "last", 1)
    assert set(info.payloads) == {"state.json", "model.safetensors", "training.safetensors"}

    rebuilt = reconstruct_bundle("bundle")
    np.testing.assert_array_equal(rebuilt.model.predict(X).logits, first.predict(X).logits)  # bit for bit
    np.testing.assert_array_equal(
        rebuilt.model.predict_proba(X).probabilities,  # type: ignore[arg-type]
        first.predict_proba(X).probabilities,  # type: ignore[arg-type]
    )
    resumed_run = rebuilt.resume(_params(1, "split", batches))
    assert resumed_run.resume_status is not None and resumed_run.resume_status.mode == "stateful"
    assert resumed_run.idps[0].epoch_idx == 2  # epoch numbering continues
    assert resumed_run.train.resume_from_checkpoint == rebuilt.resume_checkpoint
    _assert_same_weights(continuous, rebuilt.model)


def test_calibrators_keep_their_label_and_model_linkage():
    X, batches = _data()
    torch.manual_seed(7)
    model = _classifier()
    run = model.train(params=_params(1, "cal", batches))
    fingerprint = model_fingerprint(model)
    calibrator = TemperatureCalibrator(temperature=1.7, labels=LABELS, model_id=fingerprint, split_id="calibration")
    info = export_bundle(run.id, "bundle", calibrators=[calibrator])
    assert [dict(summary) for summary in info.calibrators] == [
        {"payload": "calibrator-0.json", "id": calibrator.id, "labels": LABELS, "model_id": fingerprint}
    ]
    rebuilt = reconstruct_bundle("bundle")
    assert rebuilt.calibrators == (calibrator,)
    assert model_fingerprint(rebuilt.model) == rebuilt.calibrators[0].model_id  # the weights it was fitted on
    logits = rebuilt.model.predict(X).logits
    served = rebuilt.calibrators[0].transform(logits, labels=LABELS, model_id=model_fingerprint(rebuilt.model))
    np.testing.assert_array_equal(served.logits, logits)

    wrong_labels = TemperatureCalibrator(temperature=1.7, labels=("a", "b", "c"), model_id=fingerprint, split_id="c")
    with pytest.raises(BundleError, match="labels"):
        export_bundle(run.id, "other", calibrators=[wrong_labels])
    other_model = TemperatureCalibrator(temperature=1.7, labels=LABELS, model_id="sha256:" + "0" * 64, split_id="c")
    with pytest.raises(BundleError, match="not on the bundled weights"):
        export_bundle(run.id, "other", calibrators=[other_model])
    assert not os.path.exists("other/bundle.json")  # refused before anything was published


# --- a registered module -----------------------------------------------------------------------


def test_a_registered_module_rebuilds_from_supplied_factories_and_resumes_like_the_continuous_run():
    X, batches = _data()
    spec = ModelSpec("tests.bundle_encoder", 1, {"width": 6}, seed=5)
    model_params = NNModelParams(net=spec, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)

    def stopper() -> EarlyStopping:
        return EarlyStopping(monitor="train_edp.loss", patience=10)  # checkpointed component state

    register_model_factory("tests.bundle_encoder", 1, _encoder_factory)
    try:
        torch.manual_seed(7)
        continuous = NNModel(params=model_params)
        continuous.train(params=_params(3, "full", batches), callbacks=[stopper()])
        torch.manual_seed(7)
        first = NNModel(params=model_params)
        first_run = first.train(params=_params(2, "split", batches), callbacks=[stopper()])
        export_bundle(first_run.id, "bundle")
    finally:
        unregister_model_factory("tests.bundle_encoder", 1)

    info = inspect_bundle("bundle")
    assert info.model == str(spec) and dict(info.components) == {"early_stopping": {"version": 1, "required": False}}
    with pytest.raises(BundleReconstructionError, match="tests.bundle_encoder"):
        reconstruct_bundle("bundle")  # not registered in this process, and none supplied
    rebuilt = reconstruct_bundle(
        "bundle", factories={("tests.bundle_encoder", 1): _encoder_factory}, components=[stopper()]
    )
    np.testing.assert_array_equal(rebuilt.model.predict(X).logits, first.predict(X).logits)
    resumed_run = rebuilt.resume(_params(1, "split", batches), callbacks=[stopper()])
    assert resumed_run.resume_status is not None and "early_stopping" in resumed_run.resume_status.restored_components
    _assert_same_weights(continuous, rebuilt.model)


# --- weights-only snapshots ----------------------------------------------------------------------


def test_a_model_checkpoint_snapshot_is_inference_only_and_refuses_to_resume_before_changing_anything():
    X, batches = _data()
    torch.manual_seed(7)
    model = _classifier()
    run = model.train(params=_params(2, "snap", batches), callbacks=[ModelCheckpoint(epochs=[0], tag="snap")])
    info = export_bundle(run.id, "bundle", checkpoint="snap_e0")
    assert info.capability == "inference" and "training.safetensors" not in info.payloads
    rebuilt = reconstruct_bundle("bundle")
    before = {name: tensor.clone() for name, tensor in rebuilt.model.net.state_dict().items()}
    with pytest.raises(BundleCapabilityError, match="inference-only.*optimizer"):
        rebuilt.resume(_params(1, "snap", batches))
    for name, tensor in rebuilt.model.net.state_dict().items():
        assert torch.equal(tensor, before[name]), name
    assert not os.path.exists(os.path.join("runs", run.id, "checkpoints", f"{rebuilt.resume_checkpoint}.pt"))
    assert rebuilt.model.predict(X).logits.shape == (24, 3)  # still serves inference


def test_resume_refuses_a_second_resume_source():
    _, batches = _data()
    torch.manual_seed(7)
    run = _classifier().train(params=_params(1, "twice", batches))
    export_bundle(run.id, "bundle")
    rebuilt = reconstruct_bundle("bundle")
    with pytest.raises(ValueError, match="resume_from_run_id"):
        rebuilt.resume(_params(1, "twice", batches, resume_from_run_id=run.id))


def test_export_refuses_extra_state_and_non_bundle_destinations():
    _, batches = _data()
    torch.manual_seed(7)
    model = _classifier()
    run = model.train(params=_params(1, "extra", batches))
    os.makedirs("occupied")
    with open(os.path.join("occupied", "notes.txt"), "w") as handle:
        handle.write("mine")
    with pytest.raises(BundleError, match="neither empty nor a run bundle"):
        export_bundle(run.id, "occupied")
    with pytest.raises(ValueError, match="Checkpoints tag"):
        export_bundle(run.id, "bundle", checkpoint="best_typo")
    assert bundles.BUNDLE_VERSION == 1


def test_tensor_payloads_are_standard_safetensors():
    from nnx.bundles import _load_tensors, _save_tensors

    tensors = {
        "f64": torch.randn(2, 3, dtype=torch.float64),
        "f32": torch.randn(4),
        "f16": torch.randn(3).half(),
        "bf16": torch.randn(2, 2).bfloat16(),
        "i64": torch.arange(5),
        "i8": torch.tensor([-3, 4], dtype=torch.int8),
        "u8": torch.ByteTensor([0, 255]),
        "bool": torch.tensor([True, False, True]),
        "scalar": torch.tensor(2.5),
        "empty": torch.zeros(0, 3),
        "strided": torch.arange(12.0).reshape(3, 4).t(),  # a non-contiguous view
    }
    data = _save_tensors(tensors, {"nnx.bundle": "model"})
    assert int.from_bytes(data[:8], "little") % 8 == 0  # the header is padded to 8 bytes
    loaded = _load_tensors(data, "payload")
    for key, tensor in tensors.items():
        assert loaded[key].dtype == tensor.dtype and torch.equal(loaded[key], tensor), key
    try:  # the reference implementation, when the hub extra is installed, reads and writes the same format
        import safetensors.torch as reference
    except ImportError:
        reference = None
    if reference is not None:
        for key, tensor in reference.load(data).items():
            assert torch.equal(tensor, tensors[key].contiguous()), key
        round_trip = _load_tensors(reference.save({k: v.contiguous() for k, v in tensors.items()}), "reference")
        assert all(torch.equal(round_trip[key], tensor) for key, tensor in tensors.items())
