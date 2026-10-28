"""FEAT-015: portable, data-only run bundles (`nnx.bundles`).

A bundle holds a run checkpoint — weights, training state, calibrators — as
safetensors plus schema-validated JSON. A built-in classifier and a
registered module rebuilt from a bundle predict bit-for-bit like the source
model, and resuming one epoch from the bundle equals the continuous run.
"""

from __future__ import annotations

import errno
import json
import math
import os
import types
from collections import OrderedDict

import numpy as np
import pytest
import torch

import nnx.nn.params.nn_checkpoint as nn_checkpoint
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
    validate_bundle,
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


def test_a_bundle_resume_equals_the_continuous_run_when_on_train_end_draws():
    """#538: the bundle carries the final LAST's streams from before
    on_train_end, so a draw there (a final test score on a loader without a
    generator) leaves the bundle resume equal to the continuous run."""
    from nnx import Callback

    class TrainEndDraw(Callback):
        def on_train_end(self, ctx):
            torch.rand(1)

    _, batches = _data()
    torch.manual_seed(7)
    continuous = _classifier()
    continuous.train(params=_params(3, "full", batches), callbacks=[TrainEndDraw()])

    torch.manual_seed(7)
    first_run = _classifier().train(params=_params(2, "split", batches), callbacks=[TrainEndDraw()])
    export_bundle(first_run.id, "bundle")
    rebuilt = reconstruct_bundle("bundle")
    resumed_run = rebuilt.resume(_params(1, "split", batches), callbacks=[TrainEndDraw()])
    assert resumed_run.resume_status is not None and resumed_run.resume_status.mode == "stateful"
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
    assert info.net_params is None  # a registered module: its factory builds the architecture
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


def test_resume_refuses_changed_optimizer_hyperparameters_from_the_bundle_itself():
    """#502: ``RunBundle.resume`` is always stateful, so the bundle's param
    groups would replace a new lr / weight_decay / momentum while the run
    recorded them. The bundle carries what it trained with: the refusal
    needs no source run on disk."""
    import shutil
    from dataclasses import replace

    _, batches = _data()
    torch.manual_seed(7)
    parent = NNOptimParams.builder().sgd(max_lr=0.5).build()
    run = _classifier().train(params=replace(_params(1, "hparams", batches), optim=parent))
    export_bundle(run.id, "bundle")
    shutil.rmtree("runs")  # only the bundle remains
    rebuilt = reconstruct_bundle("bundle")
    before = {name: tensor.clone() for name, tensor in rebuilt.model.net.state_dict().items()}
    changed = replace(
        _params(1, "hparams", batches), optim=replace(parent, max_lr=1e-4, weight_decay=0.1, momentum=0.0)
    )
    with pytest.raises(ValueError, match="resume hyperparameter mismatch") as caught:
        rebuilt.resume(changed)
    for fragment in (
        "optimizer max_lr (checkpoint 0.5, configuration 0.0001)",
        "optimizer weight_decay (checkpoint 0.0, configuration 0.1)",
        "optimizer momentum (checkpoint 0.9, configuration 0.0)",
        "resume_mode='weights_only'",
    ):
        assert fragment in str(caught.value), str(caught.value)
    for name, tensor in rebuilt.model.net.state_dict().items():
        assert torch.equal(tensor, before[name]), name
    leftover = os.listdir("runs") if os.path.isdir("runs") else []
    assert not [name for name in leftover if not name.startswith(".")]  # no child run was written
    resumed = rebuilt.resume(replace(_params(1, "hparams", batches), optim=parent))  # unchanged: resumes
    assert resumed.resume_status is not None and resumed.resume_status.mode == "stateful"


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


def test_inspecting_a_built_in_bundle_shows_the_architecture_reconstruction_builds():
    _, batches = _data()
    run = _classifier().train(params=_params(1, "inspect-dims", batches))
    export_bundle(run.id, "bundle")
    info = inspect_bundle("bundle")
    assert info.net_params is not None and info.net_params["hidden_dims"] == "[8]"
    assert info.net_params["input_dim"] == 4 and info.net_params["output_dim"] == 3


# --- #509: a top-level tensor attached to the net ------------------------------------------------


@pytest.mark.parametrize("kind", ["parameter", "buffer"])
def test_a_run_with_a_top_level_tensor_rebuilds_without_it_through_every_rebuild_path(kind):
    """#509: train() accepts a top-level tensor registered on the net (a
    learnable temperature, a class prior); its checkpoints and its bundle
    rebuild without it through exclude_submodules, and each strict failure
    names the option of the entry point the caller used."""
    from nnx import Checkpoints, NNCheckpoint

    _, batches = _data()
    torch.manual_seed(7)
    model = _classifier()
    name = "logit_scale" if kind == "parameter" else "class_prior"
    if kind == "parameter":
        model.net.register_parameter(name, torch.nn.Parameter(torch.ones(1)))
        model.net.register_forward_hook(lambda module, args, output: output * module.logit_scale)
    else:
        model.net.register_buffer(name, torch.tensor([0.2, 0.3, 0.5]))
    run = model.train(params=_params(1, "top-level", batches))
    checkpoint = NNCheckpoint.load(run.id, Checkpoints.LAST)
    assert checkpoint is not None and name in checkpoint.net_state
    export_bundle(run.id, "bundle")
    base = {key: value for key, value in model.net.state_dict().items() if key != name}

    with pytest.raises(RuntimeError) as caught:
        NNModel.from_checkpoint(checkpoint)
    assert f"({name}): NNModel.from_checkpoint(checkpoint, exclude_submodules=('{name}',))" in str(caught.value)
    with pytest.raises(RuntimeError) as caught:
        reconstruct_bundle("bundle")
    message = str(caught.value)
    assert f"({name}): reconstruct_bundle(path, exclude_submodules=('{name}',))" in message
    assert "from_checkpoint" not in message

    for rebuilt in (
        NNModel.from_checkpoint(checkpoint, exclude_submodules=(name,)),
        reconstruct_bundle("bundle", exclude_submodules=(name,)).model,
    ):
        assert rebuilt.net.state_dict().keys() == base.keys()
        for key, value in base.items():
            assert torch.equal(rebuilt.net.state_dict()[key], value), key


# --- #553: a bundle child's history lineage -------------------------------------------------------


@pytest.mark.parametrize("journal", [False, True], ids=["idps.csv", "journal"])
@pytest.mark.parametrize("beside", ["an unrelated same-id run", "no same-id run", "its unchanged source run"])
def test_a_bundle_childs_lineage_starts_at_the_child_with_the_bundle_warning(beside, journal, tmp_path, monkeypatch):
    """The parent of a bundle-resumed run lives in the bundle: a same-id run
    in the child's runs root is never stitched in — not an unrelated run,
    and not the bundle's unchanged source either (the bundle does not prove
    which attempt it was exported from), and a root without one is not told
    the parent is "no longer saved"."""
    import filecmp
    import warnings

    from nnx.history import HistoryJournal, export_history_csv, iter_history

    history = HistoryJournal(retention=3, chunk_size=2) if journal else None
    _, batches = _data()
    os.makedirs("m1")
    monkeypatch.chdir("m1")
    torch.manual_seed(7)
    source = _classifier().train(params=_params(2, "split", batches), history=history)
    export_bundle(source.id, os.path.join(tmp_path, "bundle"))
    if beside != "its unchanged source run":
        os.makedirs(os.path.join(tmp_path, "m2"))
        monkeypatch.chdir(os.path.join(tmp_path, "m2"))
    if beside == "an unrelated same-id run":
        other = [(50 * features, labels) for features, labels in batches]  # other data, same configuration
        torch.manual_seed(7)
        assert _classifier().train(params=_params(2, "split", other), history=history).id == source.id
    child = reconstruct_bundle(os.path.join(tmp_path, "bundle")).resume(_params(1, "split", batches), history=history)
    assert child.resume_status is not None and child.resume_status.source_attempt_token is not None

    def recorded(read):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = read()
        return result, [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]

    own = [record.state() for record in iter_history(child.id)]
    assert own and {record["epoch_idx"] for record in own} == {2}
    lineage, messages = recorded(lambda: [record.state() for record in iter_history(child.id, lineage=True)])
    assert lineage == own
    assert len(messages) == 1 and "bundle" in messages[0] and source.id in messages[0]
    assert "no longer saved" not in messages[0]
    count, messages = recorded(lambda: export_history_csv(child.id, tmp_path / "lineage.csv", lineage=True))
    assert count == len(own) and len(messages) == 1 and "bundle" in messages[0]
    export_history_csv(child.id, tmp_path / "own.csv")
    assert filecmp.cmp(tmp_path / "lineage.csv", tmp_path / "own.csv", shallow=False)


# --- #552: a real flush I/O error fails the export ---------------------------------------------


def _fail_flushes(patch, code: int, applies=lambda fd: True) -> None:
    """Fail the flush primitives on the descriptors ``applies`` selects:
    ``F_FULLFSYNC`` (stood in for where the platform lacks it) and ``os.fsync``."""
    real_fsync = os.fsync

    def fail_or_flush(fd, *args):
        if applies(fd):
            raise OSError(code, os.strerror(code))
        return real_fsync(fd)

    patch.setattr(nn_checkpoint, "fcntl", types.SimpleNamespace(F_FULLFSYNC=51, fcntl=fail_or_flush))
    patch.setattr(os, "fsync", fail_or_flush)


def _generations(target: str) -> set[str]:
    return {entry[2:] for entry in os.listdir(target) if entry.startswith("g-")}


def _published(target: str) -> str:
    with open(os.path.join(target, "bundle.json"), encoding="utf-8") as handle:
        return json.load(handle)["generation"]


def _two_exports():
    _, batches = _data()
    torch.manual_seed(7)
    run = _classifier().train(params=_params(1, "flush", batches))
    first = export_bundle(run.id, "bundle").generation
    second = export_bundle(run.id, "bundle").generation
    return run, first, second


def test_an_io_error_on_every_flush_fails_the_export_and_keeps_the_published_generations(monkeypatch):
    run, first, second = _two_exports()
    with monkeypatch.context() as patch:
        _fail_flushes(patch, errno.EIO)
        with pytest.raises(OSError) as caught:
            export_bundle(run.id, "bundle")
    assert caught.value.errno == errno.EIO
    assert _published("bundle") == second
    assert _generations("bundle") == {first, second}  # nothing pruned, no new generation left behind
    assert validate_bundle("bundle").verified


def _new_entries(kind: str, published: set[str]) -> list[str]:
    """The paths of the export's own ``kind`` of entry: its payload files,
    its ``.bundle.json.*.tmp`` manifest or its generation directory."""
    if kind == "manifest":
        return [os.path.join("bundle", entry) for entry in os.listdir("bundle") if entry.startswith(".bundle.json.")]
    directories = [os.path.join("bundle", f"g-{generation}") for generation in _generations("bundle") - published]
    if kind == "directory":
        return directories
    return [os.path.join(directory, name) for directory in directories for name in os.listdir(directory)]


@pytest.mark.parametrize("kind", ["payload", "manifest", "directory"])
def test_an_io_error_flushing_one_new_entry_fails_the_export(monkeypatch, kind):
    """Only one kind of the export's entries fails (Linux reports a writeback
    error per inode): the others flush, so the export fails only if the
    helper that flushed that entry raises its error."""
    run, first, second = _two_exports()

    def one_new_entry(fd):
        stat = os.fstat(fd)
        return any(
            (stat.st_dev, stat.st_ino) == (entry.st_dev, entry.st_ino)
            for entry in map(os.stat, _new_entries(kind, {first, second}))
        )

    with monkeypatch.context() as patch:
        _fail_flushes(patch, errno.EIO, one_new_entry)
        with pytest.raises(OSError) as caught:
            export_bundle(run.id, "bundle")
    assert caught.value.errno == errno.EIO
    assert _published("bundle") == second
    assert _generations("bundle") == {first, second}
    assert validate_bundle("bundle").verified


@pytest.mark.parametrize("code", [errno.ENOTSUP, errno.EINVAL], ids=["ENOTSUP", "EINVAL"])
def test_an_unsupported_flush_does_not_fail_the_export(monkeypatch, code):
    run, _, second = _two_exports()
    with monkeypatch.context() as patch:
        _fail_flushes(patch, code)
        info = export_bundle(run.id, "bundle")
    assert info.verified and _published("bundle") == info.generation
    assert _generations("bundle") == {second, info.generation}


def test_an_io_error_after_the_manifest_rename_raises_without_pruning(monkeypatch):
    """The rename that made the new generation live cannot be undone, so the
    export raises before it prunes: the generations it replaced stay."""
    run, first, second = _two_exports()
    target = os.stat("bundle")

    def the_republished_root(fd):
        stat = os.fstat(fd)
        return (stat.st_dev, stat.st_ino) == (target.st_dev, target.st_ino) and _published("bundle") != second

    with monkeypatch.context() as patch:
        _fail_flushes(patch, errno.EIO, the_republished_root)
        with pytest.raises(OSError):
            export_bundle(run.id, "bundle")
    (third,) = _generations("bundle") - {first, second}
    assert _published("bundle") == third
    assert {first, second} < _generations("bundle")


def test_every_bundle_file_and_directory_is_flushed(monkeypatch):
    """Each payload, ``bundle.json``, the generation directory and the root
    (before and after the manifest rename) reach the flush primitive."""
    run, _, _ = _two_exports()
    flushed: list[tuple[int, int]] = []
    real = nn_checkpoint._full_fsync

    def record(fd):
        stat = os.fstat(fd)
        flushed.append((stat.st_dev, stat.st_ino))
        return real(fd)

    monkeypatch.setattr(nn_checkpoint, "_full_fsync", record)
    info = export_bundle(run.id, "bundle")

    def key(path):
        stat = os.stat(path)
        return stat.st_dev, stat.st_ino

    generation = os.path.join("bundle", f"g-{info.generation}")
    payloads = [os.path.join(generation, name) for name in os.listdir(generation)]
    assert payloads and all(key(path) in flushed for path in payloads)
    assert key(os.path.join("bundle", "bundle.json")) in flushed
    assert key(generation) in flushed
    assert flushed.count(key("bundle")) == 2
