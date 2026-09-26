"""FEAT-019: canonical provenance manifests, fingerprints and comparison.

Canonical bytes are UTF-8 JSON with sorted keys, preserved array order and
the format version inside the hashed document. Serialization rejects what
is not JSON-like instead of hashing its repr, and identity references are
declared, digest or unknown — a supplied id is never verified.
Fingerprinting and comparison read no loader, build no model and open no
socket; comparison reports stable field paths and separates missing,
unknown, declared and verified values.
"""

from __future__ import annotations

import math
import socket

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Losses,
    ModelSpec,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNRun,
    NNTrainParams,
    TaskSpec,
    provenance,
    register_model_factory,
    unregister_model_factory,
)
from nnx.provenance import (
    FORMAT,
    ExperimentManifest,
    IdentityRef,
    canonical_bytes,
    compare,
    hash_bytes,
    hash_file,
)

BASE = ExperimentManifest(
    task={"kind": "categorical", "num_outputs": 3},
    labels=("cat", "dog", "fox"),
    data={"train": "animals-v2"},
    splits={"val": hash_bytes(b"0,3,7")},
    objective={"id": "supervised", "version": 1},
    config={"lr": 0.1, "epochs": 3},
)


# --- canonical bytes ------------------------------------------------------------------------


def test_canonical_bytes_sort_keys_keep_array_order_and_encode_utf8():
    assert canonical_bytes({"b": [3, 1, 2], "a": "ü"}) == '{"a":"ü","b":[3,1,2]}'.encode()
    assert canonical_bytes({"a": 1, "b": 2}) == canonical_bytes({"b": 2, "a": 1})  # reordered keys
    assert canonical_bytes([1, 2]) != canonical_bytes([2, 1])  # arrays keep their order
    assert canonical_bytes((1, 2)) == canonical_bytes([1, 2])
    unicode = ExperimentManifest(labels=("ねこ", "chien", "Fuchs 🦊"))
    assert "ねこ".encode() in unicode.canonical_bytes() and b"\\u" not in unicode.canonical_bytes()
    assert ExperimentManifest.from_state(unicode.state()).fingerprint() == unicode.fingerprint()


def test_the_format_version_is_part_of_the_hashed_document(monkeypatch):
    assert BASE.document()["format"] == FORMAT == "nnx.provenance/1"
    assert BASE.fingerprint().startswith("sha256:")
    before = BASE.fingerprint()
    monkeypatch.setattr(provenance, "FORMAT", "nnx.provenance/2")
    assert BASE.fingerprint() != before


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("task", {"kind": "categorical", "num_outputs": 4}),
        ("labels", ("dog", "cat", "fox")),  # label order
        ("splits", {"val": hash_bytes(b"0,3,8")}),  # split digest
        ("objective", {"id": "supervised", "version": 2}),  # objective version
        ("data", {"train": "animals-v3"}),
        ("config", {"lr": 0.2, "epochs": 3}),
    ],
)
def test_changing_any_declared_part_changes_the_fingerprint(change, value):
    from dataclasses import replace

    assert replace(BASE, **{change: value}).fingerprint() != BASE.fingerprint()
    assert replace(BASE).fingerprint() == BASE.fingerprint()


# --- what serializes ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bad", "error", "message"),
    [
        (math.nan, ValueError, "non-finite"),
        (math.inf, ValueError, "non-finite"),
        ({1, 2}, TypeError, "a set is not JSON-like"),
        (b"raw", TypeError, "a bytes is not JSON-like"),
        (len, TypeError, "a callable is not JSON-like"),
        (lambda x: x, TypeError, "a callable is not JSON-like"),
        (object(), TypeError, "an? object is not JSON-like"),
        ({1: "int key"}, TypeError, "keys must be strings"),
    ],
)
def test_serialization_rejects_values_instead_of_hashing_their_repr(bad, error, message):
    with pytest.raises(error, match=message):
        canonical_bytes({"config": {"value": bad}})
    with pytest.raises(error, match=r"\$\.config\.value"):  # the path is named
        ExperimentManifest(config={"value": bad})


def test_identity_references_are_declared_digest_or_unknown(tmp_path):
    manifest = ExperimentManifest(data={"train": "imagenet-v2", "extra": None, "file": hash_file(_file(tmp_path))})
    assert manifest.data["train"] == IdentityRef.declared("imagenet-v2")
    assert not manifest.data["train"].verified  # a supplied id is never verified
    assert manifest.data["extra"] == IdentityRef.unknown() and not manifest.data["extra"].verified
    assert manifest.data["file"].verified and manifest.data["file"].value.startswith("sha256:")
    assert hash_file(_file(tmp_path)) == hash_bytes(b"row-1\nrow-2\n")
    assert manifest.state()["data"]["train"] == {"identity": "declared", "value": "imagenet-v2"}
    with pytest.raises(ValueError, match="kind"):
        IdentityRef("verified", "x")
    with pytest.raises(TypeError, match="IdentityRef, a declared id string or None"):
        ExperimentManifest(data={"train": 42})


def _file(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_bytes(b"row-1\nrow-2\n")
    return path


# --- no loader, no model, no socket ---------------------------------------------------------------


class _Unreadable:
    def __iter__(self):
        raise AssertionError("provenance must never iterate a loader")


def test_fingerprint_and_compare_read_no_loader_build_no_model_and_open_no_socket(monkeypatch):
    builds = []

    def factory(config):
        builds.append(config)
        return nn.Linear(4, 3)

    register_model_factory("tests.provenance_head", 1, factory)
    try:
        model = NNModel(
            params=NNModelParams(
                net=ModelSpec("tests.provenance_head", 1), loss=Losses.CROSS_ENTROPY, task=TaskSpec.categorical(3)
            )
        )
        assert len(builds) == 1

        def no_sockets(*args, **kwargs):
            raise AssertionError("provenance must never open a socket")

        monkeypatch.setattr(socket, "socket", no_sockets)
        train = NNTrainParams(n_epochs=1, train_loader=_Unreadable(), val_loader=_Unreadable())
        manifest = ExperimentManifest.for_model(model, train=train, data={"train": "animals-v2"})
        other = ExperimentManifest.for_model(model, train=train, data={"train": "animals-v3"})
        assert manifest.fingerprint() != other.fingerprint()
        result = compare(manifest, other)
        assert result.by_path()["data.train"].status == "different"
        assert manifest.model["net"]["kind"] == "registered" and "device" not in manifest.model
        assert manifest.task["kind"] == "categorical"
        assert len(builds) == 1  # nothing was rebuilt
    finally:
        unregister_model_factory("tests.provenance_head", 1)


# --- comparison -------------------------------------------------------------------------------------


def test_compare_separates_missing_unknown_declared_and_verified_values():
    left = ExperimentManifest(
        labels=("cat", "dog"),
        data={"train": "animals-v2", "val": None, "test": hash_bytes(b"t")},
        splits={"fold": hash_bytes(b"f")},
        config={"lr": 0.1, "only_left": True},
    )
    right = ExperimentManifest(
        labels=("cat", "dog"),
        data={"train": "animals-v2", "val": "animals-val", "test": hash_bytes(b"t")},
        splits={"fold": hash_bytes(b"g")},
        config={"lr": 0.1},
    )
    fields = compare(left, right).by_path()
    assert fields["data.train"].status == "declared"  # equal, but only declared
    assert fields["data.val"].status == "unknown"
    assert fields["data.test"].status == "verified"
    assert fields["splits.fold"].status == "different"
    assert fields["config.only_left"].status == "missing" and fields["config.only_left"].left is True
    assert fields["config.lr"].status == "equal" and fields["labels[1]"].status == "equal"
    assert not compare(left, right).verified_equal
    digests = ExperimentManifest(data={"train": hash_bytes(b"x")})
    assert compare(digests, ExperimentManifest(data={"train": hash_bytes(b"x")})).verified_equal
    assert not compare(ExperimentManifest(data={"train": "a"}), ExperimentManifest(data={"train": "a"})).verified_equal
    absent = compare(left, None)
    assert [(item.path, item.status) for item in absent.fields] == [("", "absent")] and not absent.verified_equal


def test_compare_reads_saved_runs_without_loading_models(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    X = torch.randn(8, 4, generator=torch.Generator().manual_seed(0))
    loader = DataLoader(TensorDataset(X, (X[:, 0] > 0).long()), batch_size=4)

    def fit(data_id: str, manifest):
        model = NNModel(module=nn.Linear(4, 2), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
        params = NNTrainParams(
            n_epochs=1, train_loader=loader, optim=NNOptimParams.builder().sgd(max_lr=0.1).build(), data_id=data_id
        )
        return model.train(params=params, provenance=manifest)

    a = fit("a", ExperimentManifest(data={"train": hash_bytes(b"a")}))
    b = fit("b", ExperimentManifest(data={"train": hash_bytes(b"b")}))
    plain = fit("plain", None)

    def no_model(*args, **kwargs):
        raise AssertionError("compare must not load a model")

    monkeypatch.setattr(NNModel, "from_checkpoint", no_model)
    by_id = compare(a.id, b.id)
    assert by_id.by_path()["data.train"].status == "different"
    assert compare(NNRun.load(a.id), a.id).verified_equal
    assert compare(a.id, plain.id).fields[0].status == "absent"


# --- review hardening -------------------------------------------------------------------------


def test_comparison_is_type_strict_like_the_fingerprint():
    left = ExperimentManifest(config={"lr": 1, "flag": True})
    right = ExperimentManifest(config={"lr": 1.0, "flag": 1})
    assert left.fingerprint() != right.fingerprint()
    fields = compare(left, right).by_path()
    assert fields["config.lr"].status == "different" and fields["config.flag"].status == "different"
    assert not compare(left, right).verified_equal and left != right


def test_manifests_are_immutable_hashable_and_equal_to_their_round_trip():
    config = {"lr": 1, "layers": (4, 2)}
    manifest = ExperimentManifest(config=config)
    fingerprint = manifest.fingerprint()
    config["lr"] = float("nan")  # the caller's dict can no longer change the plan
    assert manifest.fingerprint() == fingerprint
    assert ExperimentManifest.from_state(manifest.state()) == manifest
    assert hash(ExperimentManifest.from_state(manifest.state())) == hash(manifest)
    assert len({manifest, ExperimentManifest(config={"layers": [4, 2], "lr": 1})}) == 1


def test_user_config_shaped_like_an_identity_is_never_verified():
    lookalike = ExperimentManifest(config={"x": {"identity": "digest", "value": "a"}})
    fields = compare(lookalike, lookalike).by_path()
    assert fields["config.x.identity"].status == "equal" and fields["config.x.value"].status == "equal"
    assert all(item.status == "equal" for item in fields.values())


def test_attempt_only_training_fields_stay_out_of_the_plan():
    model = NNModel(module=nn.Linear(4, 2), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    parent = NNTrainParams(n_epochs=3)
    child = NNTrainParams(n_epochs=1, resume_from_run_id="0" * 32)
    assert ExperimentManifest.for_model(model, train=parent) == ExperimentManifest.for_model(model, train=child)
    assert "n_epochs" not in ExperimentManifest.for_model(model, train=parent).config["train"]


def test_run_ids_are_validated_before_any_path_is_built():
    with pytest.raises(ValueError):
        provenance.load_provenance("../outside")
    with pytest.raises(ValueError):
        compare("../outside", BASE)
