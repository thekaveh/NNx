"""FEAT-015: run bundle validation and its trust boundary.

``inspect_bundle`` and ``validate_bundle`` never unpickle, read a tensor or
call a model factory — not even beside a legacy ``last.pt``. Validation
rejects removed, extra, swapped, corrupted, symlinked and out-of-root
payloads, duplicate JSON keys, generation mismatches and unknown versions
before any tensor is read, and an interrupted export leaves the previous
bundle usable. ``reconstruct_bundle`` names every missing factory and
component before any model is allocated.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil

import pytest
import torch

import nnx.bundles as bundles
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
    NNTrainParams,
    TaskSpec,
    register_model_factory,
    unregister_model_factory,
)
from nnx.bundles import (
    BundleError,
    BundleIntegrityError,
    BundleReconstructionError,
    export_bundle,
    inspect_bundle,
    reconstruct_bundle,
    validate_bundle,
)
from nnx.calibration import TemperatureCalibrator, model_fingerprint
from nnx.nn.callbacks import EarlyStopping

LABELS = ("a", "b", "c")


class _Encoder(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(4, width), torch.nn.Linear(width, 3))

    def forward(self, x):
        return self.body(x)


@pytest.fixture(autouse=True)
def _quiet(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")


def _batches():
    generator = torch.Generator().manual_seed(0)
    X = torch.randn(12, 4, generator=generator)
    y = torch.randint(0, 3, (12,), generator=generator)
    return [(X[:6], y[:6]), (X[6:], y[6:])]


def _train(model: NNModel, data_id: str, **train_kwargs):
    params = NNTrainParams(
        n_epochs=1, data_id=data_id, train_loader=_batches(), optim=NNOptimParams.builder().adam(max_lr=0.01).build()
    )
    return model.train(params=params, **train_kwargs)


def _classifier() -> NNModel:
    torch.manual_seed(0)
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=3, hidden_dims=[6], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
            task=TaskSpec.categorical(3, labels=LABELS),
        ),
    )


@pytest.fixture
def bundle() -> str:
    """A resume bundle with a calibrator, beside a legacy pickle ``last.pt``."""
    model = _classifier()
    run = _train(model, "trust")
    calibrator = TemperatureCalibrator(
        temperature=1.3, labels=LABELS, model_id=model_fingerprint(model), split_id="calibration"
    )
    export_bundle(run.id, "bundle", calibrators=[calibrator])
    shutil.copy(os.path.join("runs", run.id, "checkpoints", "last.pt"), os.path.join("bundle", "last.pt"))
    return "bundle"


@pytest.fixture
def traps(monkeypatch):
    """Fail the test on any unpickling or tensor read."""

    def refuse(what):
        def trap(*args, **kwargs):
            raise AssertionError(f"{what} was called")

        return trap

    monkeypatch.setattr(torch, "load", refuse("torch.load"))
    monkeypatch.setattr(pickle, "load", refuse("pickle.load"))
    monkeypatch.setattr(pickle, "loads", refuse("pickle.loads"))
    monkeypatch.setattr(bundles, "_load_tensors", refuse("a tensor read"))


def _generation_dir(path: str) -> str:
    with open(os.path.join(path, "bundle.json"), encoding="utf-8") as handle:
        return os.path.join(path, "g-" + json.load(handle)["generation"])


def _rewrite_manifest(path: str, edit) -> None:
    manifest_path = os.path.join(path, "bundle.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    edit(manifest)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)


def _rehash(path: str, name: str) -> None:
    """Make the manifest agree with a payload changed on purpose, so only the
    check under test can catch it."""
    import hashlib

    with open(os.path.join(_generation_dir(path), name), "rb") as handle:
        data = handle.read()

    def edit(manifest):
        manifest["payloads"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}

    _rewrite_manifest(path, edit)


# --- inspect / validate never unpickle, read tensors or call factories ------------------------------


def test_inspect_and_validate_read_no_pickle_and_no_tensor_beside_a_legacy_checkpoint(bundle, traps):
    info = inspect_bundle(bundle)
    assert info.capability == "resume" and not info.verified and len(info.calibrators) == 1
    checked = validate_bundle(bundle)
    assert checked.verified and checked.generation == info.generation
    with pytest.raises(BundleError, match="file inside the run bundle"):
        inspect_bundle(os.path.join(bundle, "last.pt"))  # a legacy pickle is never opened as a bundle


def test_inspect_and_validate_never_call_a_model_factory(traps):
    calls = []

    def factory(config):
        calls.append(config)
        return _Encoder(**config)

    register_model_factory("tests.trusted_encoder", 1, factory)
    try:
        model = NNModel(
            params=NNModelParams(
                net=ModelSpec("tests.trusted_encoder", 1, {"width": 5}), device=Devices.CPU, loss=Losses.CROSS_ENTROPY
            )
        )
        built = len(calls)
        with pytest.MonkeyPatch.context() as undo:  # training reads its own pickles
            undo.setattr(torch, "load", torch.serialization.load)
            run = _train(model, "factory")
            export_bundle(run.id, "bundle")
        inspect_bundle("bundle")
        validate_bundle("bundle")
        assert len(calls) == built  # neither called the factory
    finally:
        unregister_model_factory("tests.trusted_encoder", 1)


# --- reconstruct names what is missing before allocating a model ---------------------------------


def test_reconstruction_lists_every_missing_factory_and_component_before_building_anything(monkeypatch):
    register_model_factory("tests.missing_encoder", 1, lambda config: _Encoder(**config))
    try:
        model = NNModel(
            params=NNModelParams(
                net=ModelSpec("tests.missing_encoder", 1, {"width": 5}), device=Devices.CPU, loss=Losses.CROSS_ENTROPY
            )
        )
        run = _train(model, "missing", callbacks=[EarlyStopping(monitor="train_edp.loss", patience=5)])
        export_bundle(run.id, "bundle")
    finally:
        unregister_model_factory("tests.missing_encoder", 1)

    def never(*args, **kwargs):
        raise AssertionError("a model was allocated")

    monkeypatch.setattr(NNModel, "__init__", never)

    class _Required:
        def component_spec(self):
            from nnx import ComponentSpec

            return ComponentSpec("tests.required", version=1)

        def component_state(self):
            return {}

        def load_component_state(self, state, *, version):
            pass

    with pytest.raises(BundleReconstructionError) as refused:
        reconstruct_bundle("bundle", factories={}, components=[_Required()])
    problems = "\n".join(refused.value.problems)
    assert "tests.missing_encoder@v1" in problems  # the factory
    assert "tests.required" in problems  # a required component the bundle has no state for
    assert "early_stopping" not in problems  # optional: it would start fresh


# --- tampering is caught before any tensor is read -----------------------------------------------


def _remove(path):
    os.remove(os.path.join(_generation_dir(path), "model.safetensors"))


def _extra(path):
    with open(os.path.join(_generation_dir(path), "notes.txt"), "w") as handle:
        handle.write("unlisted")


def _legacy_in_generation(path):
    shutil.copy(os.path.join(path, "last.pt"), os.path.join(_generation_dir(path), "last.pt"))


def _swap(path):
    directory = _generation_dir(path)
    model, training = os.path.join(directory, "model.safetensors"), os.path.join(directory, "training.safetensors")
    os.rename(model, model + ".tmp")
    os.rename(training, model)
    os.rename(model + ".tmp", training)


def _corrupt(path):
    target = os.path.join(_generation_dir(path), "model.safetensors")
    with open(target, "r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        last = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([last[0] ^ 0xFF]))


def _out_of_root(path):
    with open(os.path.join(os.path.dirname(os.path.abspath(path)), "outside"), "w") as handle:
        handle.write("secret")
    _rewrite_manifest(
        path, lambda manifest: manifest["payloads"].update({"../outside": {"sha256": "0" * 64, "size": 6}})
    )


def _symlink_escape(path):
    directory = _generation_dir(path)
    target = os.path.join(directory, "model.safetensors")
    outside = os.path.join(os.path.dirname(os.path.abspath(path)), "model-copy.safetensors")
    shutil.copy(target, outside)  # identical bytes: the hash alone would pass
    os.remove(target)
    os.symlink(outside, target)


def _symlinked_generation(path):
    directory = _generation_dir(path)
    moved = os.path.join(os.path.dirname(os.path.abspath(path)), "moved-generation")
    shutil.move(directory, moved)
    os.symlink(moved, directory)


def _duplicate_manifest_key(path):
    manifest_path = os.path.join(path, "bundle.json")
    with open(manifest_path, encoding="utf-8") as handle:
        text = handle.read()
    with open(manifest_path, "w", encoding="utf-8") as handle:
        handle.write(text.replace('"version": 1,', '"version": 1, "version": 1,', 1))


def _duplicate_state_key(path):
    target = os.path.join(_generation_dir(path), "state.json")
    with open(target, encoding="utf-8") as handle:
        text = handle.read()
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(text.replace('"version": 1,', '"version": 1,\n "version": 1,', 1))
    _rehash(path, "state.json")


def _other_generation_state(path):
    other = _classifier()
    run = _train(other, "other")
    export_bundle(run.id, "other-bundle")
    shutil.copy(
        os.path.join(_generation_dir("other-bundle"), "state.json"), os.path.join(_generation_dir(path), "state.json")
    )
    _rehash(path, "state.json")


def _other_generation_weights(path):
    other = _classifier()
    run = _train(other, "weights")
    export_bundle(run.id, "other-bundle")
    target = os.path.join(_generation_dir(path), "model.safetensors")
    shutil.copy(os.path.join(_generation_dir("other-bundle"), "model.safetensors"), target)
    _rehash(path, "model.safetensors")


def _oversized(path):
    with open(os.path.join(_generation_dir(path), "model.safetensors"), "ab") as handle:
        handle.write(b"\0" * 4096)  # larger than the manifest says: refused before it is read


def _fifo(path):
    target = os.path.join(_generation_dir(path), "model.safetensors")
    os.remove(target)
    os.mkfifo(target)  # opening it for reading must not block


def _deep_state(path, depth=100_000):
    target = os.path.join(_generation_dir(path), "state.json")
    with open(target, "w", encoding="utf-8") as handle:
        handle.write("[" * depth + "]" * depth)
    _rehash(path, "state.json")


def _shallow_deep_state(path):
    # Deep enough to refuse, shallow enough that every Python's JSON decoder
    # parses it (3.14 parses the 100,000-level case too): the refusal must
    # not depend on the decoder's recursion limit.
    _deep_state(path, depth=200)


def _components_not_a_mapping(path):
    target = os.path.join(_generation_dir(path), "state.json")
    with open(target, encoding="utf-8") as handle:
        state = json.load(handle)
    state["training_state"]["$dict"]["components"] = [1, 2]
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(state, handle)
    _rehash(path, "state.json")


def _boolean_state_version(path):
    target = os.path.join(_generation_dir(path), "state.json")
    with open(target, encoding="utf-8") as handle:
        state = json.load(handle)
    state["version"] = True
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(state, handle)
    _rehash(path, "state.json")


def _negative_epoch(path):
    target = os.path.join(_generation_dir(path), "state.json")
    with open(target, encoding="utf-8") as handle:
        state = json.load(handle)
    state["checkpoint"]["$dict"]["idp"]["$dict"]["epoch_idx"] = -3
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(state, handle)
    _rehash(path, "state.json")


def _duplicate_calibrator_key(path):
    target = os.path.join(_generation_dir(path), "calibrator-0.json")
    with open(target, encoding="utf-8") as handle:
        text = handle.read()
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(text.replace("{", '{\n  "labels": ["c", "b", "a"],', 1))
    _rehash(path, "calibrator-0.json")


def _relabelled_calibrator(path):
    target = os.path.join(_generation_dir(path), "calibrator-0.json")
    with open(target, encoding="utf-8") as handle:
        record = json.load(handle)
    record["labels"] = ["c", "b", "a"]  # the task declares ("a", "b", "c")
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(record, handle)
    _rehash(path, "calibrator-0.json")


def _huge_dimension(path):
    target = os.path.join(_generation_dir(path), "model.safetensors")
    with open(target, "rb") as handle:
        data = handle.read()
    length = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + length])
    header["empty"] = {"dtype": "F32", "shape": [0, 2**70], "data_offsets": [0, 0]}
    text = json.dumps(header).encode("utf-8")
    text += b" " * (-len(text) % 8)
    with open(target, "wb") as handle:
        handle.write(len(text).to_bytes(8, "little") + text + data[8 + length :])
    _rehash(path, "model.safetensors")


def _oversized_manifest(path):
    with open(os.path.join(path, "bundle.json"), "r+b") as handle:
        handle.truncate(17 * 1024 * 1024)  # sparse: nothing is read before the size is refused


def _unknown_version(path):
    _rewrite_manifest(path, lambda manifest: manifest.update({"version": 2}))


def _truncated_manifest(path):
    manifest_path = os.path.join(path, "bundle.json")
    with open(manifest_path, "rb") as handle:
        data = handle.read()
    with open(manifest_path, "wb") as handle:
        handle.write(data[: len(data) // 2])


def _malformed_tag(path):
    target = os.path.join(_generation_dir(path), "state.json")
    with open(target, encoding="utf-8") as handle:
        state = json.load(handle)
    state["training_state"] = {"$pickle": "gASV"}
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(state, handle)
    _rehash(path, "state.json")


TAMPERING = {
    "removed payload": (_remove, "missing"),
    "unlisted file": (_extra, "unlisted"),
    "legacy pickle in the generation": (_legacy_in_generation, "unlisted"),
    "swapped payloads": (_swap, "SHA-256|bytes"),
    "corrupted payload": (_corrupt, "SHA-256"),
    "out-of-root path": (_out_of_root, "outside the bundle's payload names"),
    "symlink escape": (_symlink_escape, "symlink"),
    "symlinked generation": (_symlinked_generation, "no directory of its own"),
    "duplicate manifest key": (_duplicate_manifest_key, "duplicate JSON key"),
    "duplicate state key": (_duplicate_state_key, "duplicate JSON key"),
    "state from another generation": (_other_generation_state, "another bundle generation"),
    "weights from another generation": (_other_generation_weights, "another bundle generation"),
    "unknown version": (_unknown_version, "unsupported run bundle version 2"),
    "truncated manifest": (_truncated_manifest, "not valid UTF-8 JSON"),
    "malformed tag": (_malformed_tag, "tagged value"),
    "oversized payload": (_oversized, "the manifest says"),
    "fifo payload": (_fifo, "not a regular file"),
    "deeply nested state": (_deep_state, "nested too deeply"),
    "nested state the decoder parses": (_shallow_deep_state, "nested too deeply"),
    "components not a mapping": (_components_not_a_mapping, "components must map names"),
    "boolean state version": (_boolean_state_version, "another format or version"),
    "negative epoch": (_negative_epoch, "malformed checkpoint epoch"),
    "duplicate calibrator key": (_duplicate_calibrator_key, "duplicate JSON key"),
    "relabelled calibrator": (_relabelled_calibrator, "fitted for the labels"),
    "huge tensor dimension": (_huge_dimension, "malformed entry for tensor 'empty'"),
    "oversized manifest": (_oversized_manifest, "more than a bundle ever holds"),
}


@pytest.mark.parametrize("case", sorted(TAMPERING))
def test_validation_rejects_tampering_before_any_tensor_read(bundle, case, request):
    tamper, message = TAMPERING[case]
    tamper(bundle)
    request.getfixturevalue("traps")  # from here on, any tensor read or unpickling fails the test
    with pytest.raises(BundleIntegrityError, match=message):
        validate_bundle(bundle)
    with pytest.raises(BundleIntegrityError, match=message):
        reconstruct_bundle(bundle)


# --- publication -------------------------------------------------------------------------------


def test_an_interrupted_export_leaves_the_previous_bundle_usable(bundle):
    before = validate_bundle(bundle)
    model = _classifier()
    run = _train(model, "next")

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt  # killed before the manifest was replaced

    with pytest.MonkeyPatch.context() as patch, pytest.raises(KeyboardInterrupt):
        patch.setattr(bundles, "_publish", interrupted)
        export_bundle(run.id, bundle)
    assert validate_bundle(bundle).generation == before.generation
    assert reconstruct_bundle(bundle).info.generation == before.generation

    # A hard kill leaves its half-written generation and manifest behind: still the previous bundle.
    os.makedirs(os.path.join(bundle, "g-" + "f" * 32))
    with open(os.path.join(bundle, f".bundle.json.{'e' * 32}.tmp"), "w") as handle:
        handle.write("{")
    assert validate_bundle(bundle).generation == before.generation
    after = export_bundle(run.id, bundle)  # the next export publishes and clears the leftovers
    assert after.generation != before.generation
    left = sorted(entry for entry in os.listdir(bundle) if entry.startswith(("g-", ".bundle.json.")))
    assert left == sorted([f"g-{after.generation}", f"g-{before.generation}"])  # the replaced one stays for readers


def test_a_payload_swapped_for_a_symlink_after_the_check_is_never_followed(bundle, monkeypatch):
    target = os.path.join(_generation_dir(bundle), "model.safetensors")
    outside = os.path.abspath("model-copy.safetensors")
    shutil.copy(target, outside)
    os.remove(target)
    os.symlink(outside, target)
    real_lstat = os.lstat

    def racing_lstat(path, *args, **kwargs):  # the check sees a regular file; the open finds the symlink
        return os.stat(path) if os.fspath(path) == target else real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", racing_lstat)
    with pytest.raises(BundleIntegrityError, match="cannot be opened as a regular file"):
        validate_bundle(bundle)


def test_export_leaves_foreign_and_occupied_destinations_untouched():
    model = _classifier()
    run = _train(model, "foreign")
    os.makedirs("foreign")
    with open(os.path.join("foreign", "bundle.json"), "w", encoding="utf-8") as handle:
        handle.write('{"mine": true}')
    with pytest.raises(BundleError, match="not a readable NNx run bundle manifest"):
        export_bundle(run.id, "foreign")
    assert sorted(os.listdir("foreign")) == ["bundle.json"]  # no lock, no generation, nothing replaced
    with open(os.path.join("foreign", "bundle.json"), encoding="utf-8") as handle:
        assert handle.read() == '{"mine": true}'
    os.makedirs("occupied")
    with open(os.path.join("occupied", "notes.txt"), "w", encoding="utf-8") as handle:
        handle.write("mine")
    with pytest.raises(BundleError, match="neither empty nor a run bundle"):
        export_bundle(run.id, "occupied")
    assert os.listdir("occupied") == ["notes.txt"]


def test_the_replaced_generation_stays_for_readers_until_the_next_export():
    model = _classifier()
    run = _train(model, "generations")
    first = export_bundle(run.id, "bundle")
    second = export_bundle(run.id, "bundle")
    generations = sorted(entry for entry in os.listdir("bundle") if entry.startswith("g-"))
    assert generations == sorted([f"g-{first.generation}", f"g-{second.generation}"])  # a reader of `first` can finish
    third = export_bundle(run.id, "bundle")
    generations = sorted(entry for entry in os.listdir("bundle") if entry.startswith("g-"))
    assert generations == sorted([f"g-{second.generation}", f"g-{third.generation}"])


def test_a_trainer_bundle_refuses_to_resume_through_nnmodel_before_anything_changes():
    from dataclasses import replace

    model = _classifier()
    run = _train(model, "trainer")
    export_bundle(run.id, "bundle")
    rebuilt = reconstruct_bundle("bundle")
    trainer_state = {**rebuilt._training_state, "optimizer": None, "optimizers": {"g": {}}}  # a Trainer's sidecar
    trainer_bundle = replace(rebuilt, _training_state=trainer_state)
    before = {name: tensor.clone() for name, tensor in trainer_bundle.model.net.state_dict().items()}
    with pytest.raises(bundles.BundleCapabilityError, match="Trainer"):
        trainer_bundle.resume(
            NNTrainParams(
                n_epochs=1,
                data_id="trainer",
                train_loader=_batches(),
                optim=NNOptimParams.builder().adam(max_lr=0.01).build(),
            )
        )
    assert all(torch.equal(tensor, before[name]) for name, tensor in trainer_bundle.model.net.state_dict().items())


def test_an_interruption_after_the_manifest_rename_keeps_the_new_generation(bundle, monkeypatch):
    model = _classifier()
    run = _train(model, "late-interrupt")
    real = bundles._fsync_directory

    def interrupted_after_rename(path):
        if os.path.abspath(path) == os.path.abspath(bundle):  # the root: bundle.json is already replaced
            raise KeyboardInterrupt
        real(path)

    with pytest.MonkeyPatch.context() as patch, pytest.raises(KeyboardInterrupt):
        patch.setattr(bundles, "_fsync_directory", interrupted_after_rename)
        export_bundle(run.id, bundle)
    published = validate_bundle(bundle)  # the generation bundle.json names was not rolled back
    assert os.path.isdir(os.path.join(bundle, f"g-{published.generation}"))


def test_a_staged_generation_readers_would_reject_is_never_published(bundle):
    from nnx.nn.callbacks import Callback

    class _Deep(Callback):
        def component_spec(self):
            from nnx import ComponentSpec

            return ComponentSpec("tests.deep", version=1, required=False)

        def component_state(self):
            state = 0
            for _ in range(40):  # each tuple is two levels once encoded: past the readers' limit
                state = (state,)
            return {"nested": state}

        def load_component_state(self, state, *, version):
            pass

    before = validate_bundle(bundle)
    model = _classifier()
    run = _train(model, "deep", callbacks=[_Deep()])
    with pytest.raises(BundleIntegrityError, match="nested too deeply"):
        export_bundle(run.id, bundle)
    assert validate_bundle(bundle).generation == before.generation  # the previous bundle is still published
    assert sorted(entry for entry in os.listdir(bundle) if entry.startswith("g-")) == [f"g-{before.generation}"]


def test_the_manifest_is_as_readable_as_its_payloads(bundle):
    manifest_mode = os.stat(os.path.join(bundle, "bundle.json")).st_mode & 0o777
    payload_mode = os.stat(os.path.join(_generation_dir(bundle), "state.json")).st_mode & 0o777
    assert manifest_mode == payload_mode


def test_reconstruction_passes_a_runtime_batch_adapter_through(bundle):
    from nnx import PositionalInputs

    adapter = PositionalInputs()
    rebuilt = reconstruct_bundle(bundle, batch_adapter=adapter)
    assert rebuilt.model._batch_adapter is adapter  # runtime-only: never stored in the bundle


def _registered_bundle_with(module_factory, spec_id: str, path: str = "bundle") -> str:
    register_model_factory(spec_id, 1, module_factory)
    try:
        model = NNModel(params=NNModelParams(net=ModelSpec(spec_id, 1), device=Devices.CPU, loss=Losses.CROSS_ENTROPY))
        run = _train(model, spec_id)
        export_bundle(run.id, path)
    finally:
        unregister_model_factory(spec_id, 1)
    return path


def test_round_three_shapes_whose_strides_overflow_are_refused(bundle):
    target = os.path.join(_generation_dir(bundle), "model.safetensors")
    with open(target, "rb") as handle:
        data = handle.read()
    length = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + length])
    header["empty"] = {"dtype": "F32", "shape": [0, 2**62, 2**62], "data_offsets": [0, 0]}
    text = json.dumps(header).encode("utf-8")
    text += b" " * (-len(text) % 8)
    with open(target, "wb") as handle:
        handle.write(len(text).to_bytes(8, "little") + text + data[8 + length :])
    _rehash(bundle, "model.safetensors")
    with pytest.raises(BundleIntegrityError, match="malformed entry for tensor 'empty'"):
        validate_bundle(bundle)


def test_round_three_training_state_that_cannot_resume_is_refused(bundle):
    target = os.path.join(_generation_dir(bundle), "state.json")
    with open(target, encoding="utf-8") as handle:
        original = json.load(handle)
    for edit, message in (
        (lambda body: body.update({"completed_epoch": -7}), "completed_epoch"),
        (lambda body: body.update({"optimizer": None}), "no optimizer state"),
    ):
        state = json.loads(json.dumps(original))
        edit(state["training_state"]["$dict"])
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
        _rehash(bundle, "state.json")
        with pytest.raises(BundleIntegrityError, match=message):
            validate_bundle(bundle)


def test_round_three_a_calibrator_of_other_weights_is_refused_before_building(bundle, monkeypatch):
    target = os.path.join(_generation_dir(bundle), "calibrator-0.json")
    with open(target, encoding="utf-8") as handle:
        record = json.load(handle)
    record["model_id"] = "sha256:" + "1" * 64
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(record, handle)
    _rehash(bundle, "calibrator-0.json")

    def never(*args, **kwargs):
        raise AssertionError("a model was allocated")

    monkeypatch.setattr(NNModel, "__init__", never)
    with pytest.raises(BundleIntegrityError, match="not on the bundled weights"):
        reconstruct_bundle(bundle)


def test_round_three_calibrators_must_fit_the_model_class_count_and_task():
    torch.manual_seed(0)
    unlabelled = NNModel(
        net_params=NNParams(input_dim=4, output_dim=3, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    run = _train(unlabelled, "classes")
    two = TemperatureCalibrator(temperature=1.2, labels=("x", "y"), model_id="run:BEST", split_id="cal")
    with pytest.raises(BundleError, match="has 2 labels, the bundled model 3 classes"):
        export_bundle(run.id, "classes-bundle", calibrators=[two])


class _Masked(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
        self.register_buffer("mask", torch.tensor([True, False, True]))

    def forward(self, x):
        return self.linear(x) * self.mask


def test_round_three_non_canonical_booleans_are_refused(monkeypatch):
    path = _registered_bundle_with(lambda config: _Masked(), "tests.masked")
    target = os.path.join(_generation_dir(path), "model.safetensors")
    with open(target, "rb") as handle:
        data = bytearray(handle.read())
    length = int.from_bytes(data[:8], "little")
    begin, _ = json.loads(bytes(data[8 : 8 + length]))["mask"]["data_offsets"]
    data[8 + length + begin] = 2  # a byte a bool tensor never holds
    with open(target, "wb") as handle:
        handle.write(bytes(data))
    _rehash(path, "model.safetensors")
    with pytest.raises(BundleIntegrityError, match="bytes other than 0 and 1"):
        validate_bundle(path)  # the same verdict as reconstruction
    with pytest.raises(BundleIntegrityError, match="bytes other than 0 and 1"):
        reconstruct_bundle(path, factories={("tests.masked", 1): lambda config: _Masked()})


def test_round_three_a_failed_export_keeps_its_error_when_the_manifest_is_unreadable(bundle, monkeypatch):
    model = _classifier()
    run = _train(model, "enospc")

    def no_space(path, chunks):
        raise OSError(28, "No space left on device")

    real_read_manifest = bundles._read_manifest
    reads = []

    def unreadable(root):  # readable for the destination checks, unreadable during the rollback
        reads.append(root)
        if len(reads) > 2:
            raise PermissionError(13, "Permission denied")
        return real_read_manifest(root)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bundles, "_write", no_space)
        patch.setattr(bundles, "_read_manifest", unreadable)
        with pytest.raises(OSError, match="No space left"):  # the rollback's own error never replaces it
            export_bundle(run.id, bundle)
    assert len(reads) == 3


def test_round_three_provenance_links_a_bundle_resume_without_reading_a_pickle(bundle, monkeypatch):
    from nnx.provenance import ExperimentManifest, hash_bytes, load_provenance

    rebuilt = reconstruct_bundle(bundle)
    real_load = torch.load

    def no_parent_pickle(path, *args, **kwargs):
        if "bundle-" in os.fspath(path):
            raise AssertionError("the bundle's parent checkpoint was read from disk")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", no_parent_pickle)
    manifest = ExperimentManifest(data={"train": hash_bytes(b"bundle")}, config={"lr": 0.01})
    params = NNTrainParams(
        n_epochs=1, data_id="trust", train_loader=_batches(), optim=NNOptimParams.builder().adam(max_lr=0.01).build()
    )
    run = rebuilt.resume(params, provenance=manifest)
    record = load_provenance(run.id)
    assert record is not None and record.attempt is not None
    parent = record.attempt.parent
    assert parent is not None
    assert (parent["run_id"], parent["checkpoint"]) == (rebuilt.info.source_run_id, rebuilt.resume_checkpoint)
    assert (parent["epoch"], parent["generation"]) == (rebuilt.info.epoch, rebuilt.info.generation)


def test_round_four_a_categorical_task_without_a_class_count_uses_the_network_width():
    torch.manual_seed(0)
    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=3, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY, task=TaskSpec.categorical()
        ),
    )
    run = _train(model, "widths")
    five = TemperatureCalibrator(
        temperature=1.1, labels=("a", "b", "c", "d", "e"), model_id=model_fingerprint(model), split_id="cal"
    )
    with pytest.raises(BundleError, match="has 5 labels, the bundled model 3 classes"):
        export_bundle(run.id, "widths-bundle", calibrators=[five])


def test_round_four_a_transformed_topology_refuses_to_resume_before_anything_changes(bundle):
    from dataclasses import replace

    from nnx.nn.params.nn_checkpoint import NNCheckpointTransform

    rebuilt = reconstruct_bundle(bundle)
    transformed = replace(
        rebuilt, _checkpoint=replace(rebuilt._checkpoint, transforms=(NNCheckpointTransform(name="torchao_qat"),))
    )
    before = {name: tensor.clone() for name, tensor in transformed.model.net.state_dict().items()}
    with pytest.raises(bundles.BundleCapabilityError, match="transformed topology \\(torchao_qat\\)"):
        transformed.resume(
            NNTrainParams(
                n_epochs=1,
                data_id="qat",
                train_loader=_batches(),
                optim=NNOptimParams.builder().adam(max_lr=0.01).build(),
            )
        )
    assert all(torch.equal(tensor, before[name]) for name, tensor in transformed.model.net.state_dict().items())


class _Complex(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
        self.register_buffer("phase", torch.zeros(2, dtype=torch.complex64))

    def forward(self, x):
        return self.linear(x)


def test_round_four_a_dense_tensor_of_an_unsupported_dtype_is_named_as_such():
    register_model_factory("tests.complex", 1, lambda config: _Complex())
    try:
        model = NNModel(
            params=NNModelParams(net=ModelSpec("tests.complex", 1), device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
        )
        run = _train(model, "complex")
        with pytest.raises(BundleError, match="'phase': a torch.complex64 tensor has no safetensors dtype"):
            export_bundle(run.id, "complex-bundle")
    finally:
        unregister_model_factory("tests.complex", 1)


def test_round_five_a_device_this_host_lacks_is_named_before_building(bundle, monkeypatch):
    from nnx import Devices

    target = os.path.join(_generation_dir(bundle), "state.json")
    with open(target, encoding="utf-8") as handle:
        state = json.load(handle)
    state["checkpoint"]["$dict"]["model_params"]["$dict"]["device"] = "cuda"
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(state, handle)
    _rehash(bundle, "state.json")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # a CPU-only host
    with pytest.raises(BundleReconstructionError, match="runs on CUDA, which this host lacks"):
        reconstruct_bundle(bundle)
    assert reconstruct_bundle(bundle, device=Devices.CPU).model.params.device is Devices.CPU


class _Head(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from nnx.models import build_module

        self.backbone = build_module(ModelSpec("tests.backbone", 1))  # a nested, process-registered module
        self.head = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.head(self.backbone(x))


def test_round_five_supplied_factories_let_nested_builds_use_the_process_registry():
    register_model_factory("tests.backbone", 1, lambda config: torch.nn.Linear(4, 4))
    try:
        path = _registered_bundle_with(lambda config: _Head(), "tests.head")
        rebuilt = reconstruct_bundle(path, factories={("tests.head", 1): lambda config: _Head()})
        assert isinstance(rebuilt.model.net, _Head)
    finally:
        unregister_model_factory("tests.backbone", 1)


def test_round_five_validation_streams_tensor_payloads(bundle, monkeypatch):
    real = bundles._payload_bytes

    def whole(directory, name, entry):
        if name.endswith(".safetensors"):
            raise AssertionError(f"{name} was read whole")
        return real(directory, name, entry)

    monkeypatch.setattr(bundles, "_payload_bytes", whole)
    assert validate_bundle(bundle).verified


def test_round_five_a_bundle_resume_never_links_a_local_attempt_of_the_same_run_id():
    from nnx.provenance import ExperimentManifest, hash_bytes, load_provenance

    manifest = ExperimentManifest(data={"train": hash_bytes(b"bundle")}, config={"lr": 0.01})
    model = _classifier()
    source = _train(model, "local-attempt", provenance=manifest)  # this host holds the source run's attempt
    assert load_provenance(source.id).attempt is not None
    export_bundle(source.id, "bundle")
    rebuilt = reconstruct_bundle("bundle")
    params = NNTrainParams(
        n_epochs=1,
        data_id="local-attempt",
        train_loader=_batches(),
        optim=NNOptimParams.builder().adam(max_lr=0.01).build(),
    )
    parent = load_provenance(rebuilt.resume(params, provenance=manifest).id).attempt.parent
    assert parent["run_id"] == source.id and parent["attempt_id"] is None  # the bundle's parent, not a local attempt


def test_round_five_the_new_generation_entry_persists_before_the_manifest_names_it(monkeypatch):
    model = _classifier()
    run = _train(model, "durable")
    order = []
    real_fsync, real_publish = bundles._fsync_directory, bundles._publish
    monkeypatch.setattr(
        bundles, "_fsync_directory", lambda path: (order.append(("fsync", os.path.abspath(path))), real_fsync(path))
    )
    monkeypatch.setattr(
        bundles,
        "_publish",
        lambda root, manifest: (order.append(("publish", os.path.abspath(root))), real_publish(root, manifest)),
    )
    export_bundle(run.id, "bundle")
    root = os.path.abspath("bundle")
    assert order.index(("fsync", root)) < order.index(("publish", root))


def test_round_six_malformed_component_metadata_and_factories_are_refused(bundle):
    target = os.path.join(_generation_dir(bundle), "state.json")
    with open(target, encoding="utf-8") as handle:
        original = json.load(handle)
    for entry in (
        {"$dict": {"version": {"$float": "nan"}, "required": True, "state": None}},
        {"$dict": {"version": 0, "required": True, "state": None}},
        {"$dict": {"version": 1, "required": "yes", "state": None}},
        {"$intdict": [[1, 2]]},
    ):
        state = json.loads(json.dumps(original))
        state["training_state"]["$dict"]["components"] = {"$dict": {"x": entry}}
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
        _rehash(bundle, "state.json")
        with pytest.raises(BundleIntegrityError, match="training_state.components"):
            inspect_bundle(bundle)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(original, handle)
    _rehash(bundle, "state.json")
    with pytest.raises(TypeError, match="must be callable"):
        reconstruct_bundle(bundle, factories={("tests.any", 1): None})
    with pytest.raises(TypeError, match=r"\(id, version\) pairs"):
        reconstruct_bundle(bundle, factories={"tests.any": lambda config: None})


def test_round_seven_the_streamed_fingerprint_is_model_fingerprint():
    from nnx.bundles import _payload_fingerprint, _save_tensors

    tensors = {
        "w": torch.randn(3, 4),
        "b": torch.randn(4).bfloat16(),
        "mask": torch.tensor([True, False]),
        "count": torch.tensor(7),
        "empty": torch.zeros(0, 2, dtype=torch.float64),
    }
    os.makedirs("payloads")
    data = _save_tensors(tensors, {"nnx.bundle": "model"})
    with open(os.path.join("payloads", "model.safetensors"), "wb") as handle:
        handle.write(data)
    entry = {"sha256": "0" * 64, "size": len(data)}

    class _Holder(torch.nn.Module):
        def state_dict(self, *args, **kwargs):
            return dict(tensors)

    assert _payload_fingerprint("payloads", "model.safetensors", entry) == model_fingerprint(_Holder())


def test_round_seven_pre_transform_weights_without_a_transform_are_refused(bundle):
    target = os.path.join(_generation_dir(bundle), "state.json")
    with open(target, encoding="utf-8") as handle:
        state = json.load(handle)
    body = state["training_state"]["$dict"]
    body["model"] = {"$dict": {}}  # pre-transform weights (empty here): resume would load them
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(state, handle)
    _rehash(bundle, "state.json")
    with pytest.raises(BundleIntegrityError, match="pre-transform weights for no transform"):
        validate_bundle(bundle)


def test_round_seven_validation_checks_calibrator_fingerprints_without_reading_a_tensor(bundle, traps):
    target = os.path.join(_generation_dir(bundle), "model.safetensors")
    with open(target, "r+b") as handle:  # other weights, same names, a consistent manifest
        handle.seek(-1, os.SEEK_END)
        last = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([last[0] ^ 0x01]))
    _rehash(bundle, "model.safetensors")
    with pytest.raises(BundleIntegrityError, match="not on the bundled weights"):
        validate_bundle(bundle)
    with pytest.raises(BundleError, match="file inside the run bundle"):
        inspect_bundle(os.path.join(bundle, "bundle.json"))


def test_round_eight_unknown_training_state_versions_and_numpy_strings(bundle):
    import numpy as np

    from nnx.bundles import _decode, _encode

    target = os.path.join(_generation_dir(bundle), "state.json")
    with open(target, encoding="utf-8") as handle:
        state = json.load(handle)
    state["training_state"]["$dict"]["nnx_training_state_version"] = 99
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(state, handle)
    _rehash(bundle, "state.json")
    with pytest.raises(BundleIntegrityError, match="unsupported training-state version 99"):
        validate_bundle(bundle)
    encoded = _encode({"label": np.str_("cat")}, {}, "state")
    assert _decode(json.loads(json.dumps(encoded)), {}) == {"label": "cat"}
    with pytest.raises(BundleError, match="bytes_"):
        _encode({"raw": np.bytes_(b"x")}, {}, "state")
