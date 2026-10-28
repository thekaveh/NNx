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

import json
import math
import socket

import numpy as np
import pandas as pd
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


@pytest.mark.parametrize(
    "chunk_size", [0, False, True, np.int64(0), -1, -5, None, 2.0, 2.5, "8"], ids=lambda value: repr(value)
)
def test_hash_file_refuses_a_chunk_size_that_is_not_an_integer_of_at_least_one(tmp_path, chunk_size):
    # FIX-053: read(0) returned b"" at once, so 0 / False hashed no bytes and every file got the
    # verified empty-input digest; the check runs before the file opens.
    with pytest.raises(ValueError, match=r"hash_file requires chunk_size"):
        hash_file(_file(tmp_path), chunk_size=chunk_size)
    with pytest.raises(ValueError, match=r"hash_file requires chunk_size"):
        hash_file(tmp_path / "missing.csv", chunk_size=chunk_size)


@pytest.mark.parametrize("chunk_size", [1, 7, np.int64(4), None], ids=["1", "7", "np.int64(4)", "default"])
@pytest.mark.parametrize("payload", [b"", bytes(range(256)) * 3], ids=["empty", "larger-than-chunk"])
def test_hash_file_digests_every_byte_at_any_valid_chunk_size(tmp_path, chunk_size, payload):
    path = tmp_path / "rows.bin"
    path.write_bytes(payload)
    ref = hash_file(path) if chunk_size is None else hash_file(path, chunk_size=chunk_size)
    assert ref == hash_bytes(payload) and ref.verified


def test_a_hash_file_source_still_catches_a_changed_source_in_split_replays(tmp_path):
    from nnx import NNTabularDataset
    from nnx.data_splits import SplitError, plan_split

    path = tmp_path / "rows.csv"
    path.write_bytes(b"id,x\n0,1\n1,2\n")
    groups = ["g0", "g1", "g2", "g3", "g4", "g5"]
    ids = [f"s{i}" for i in range(6)]
    positional = plan_split(
        None, strategy="group", groups=groups, proportions=(0.5, 0.25, 0.25), seed=7, source=hash_file(path)
    )
    by_id = plan_split(
        ids,
        strategy="group",
        groups=groups,
        proportions=(0.5, 0.25, 0.25),
        seed=7,
        source=hash_file(path, chunk_size=3),
    )
    path.write_bytes(b"id,x\n1,2\n0,1\n")  # the source changes after planning
    with pytest.raises(SplitError, match="source identity changed"):
        positional.resolve(range(len(groups)), source=hash_file(path))
    with pytest.raises(SplitError, match="source identity changed"):
        NNTabularDataset(
            df=pd.DataFrame({"id": ids, "x": [float(i) for i in range(6)], "y": [0, 1, 0, 1, 0, 1]}),
            feature_cols=["x"],
            target_col="y",
            split=by_id,
            id_col="id",
            source_identity=hash_file(path, chunk_size=3),
        )


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


@pytest.mark.parametrize(
    ("left", "right"),
    [({"a.b": 1}, {"a": {"b": 1}}), ({"x[0]": 1}, {"x": [1]})],
    ids=["dotted-key", "bracketed-key"],
)
def test_compare_never_equates_manifests_whose_paths_collide(left, right):
    """A key holding path syntax is quoted, so differently shaped configs
    never compare as verified equal."""
    assert not compare(ExperimentManifest(config=left), ExperimentManifest(config=right)).verified_equal


def test_a_deeply_nested_provenance_file_is_malformed_input(tmp_path):
    from nnx.provenance import MANIFEST_FILE, load_provenance

    run = tmp_path / "runs" / ("a" * 32)
    run.mkdir(parents=True)
    (run / MANIFEST_FILE).write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")
    with pytest.raises(ValueError, match="nested too deeply"):
        load_provenance("a" * 32, root=str(tmp_path))


def test_for_model_refuses_a_train_entry_twice():
    """train= beside config={"train": ...} was dropped silently."""
    with pytest.raises(ValueError, match="'train' entry"):
        model = NNModel(module=nn.Linear(4, 2), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
        ExperimentManifest.for_model(model, train=NNTrainParams(n_epochs=1), config={"train": "v1"})


def test_a_planned_resume_keeps_the_plan_fingerprint():
    """resume_epochs only says how n_epochs is read on a resume: like
    n_epochs it describes the attempt, not the plan."""
    model = NNModel(module=nn.Linear(4, 2), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    base = ExperimentManifest.for_model(model, train=NNTrainParams(n_epochs=2))
    planned = ExperimentManifest.for_model(
        model, train=NNTrainParams(n_epochs=5, resume_from_run_id="a" * 32, resume_epochs="planned")
    )
    assert base.fingerprint() == planned.fingerprint()


def test_a_manifest_with_config_none_reloads_with_its_fingerprint():
    """from_state read a missing config as {}: a None config reloaded with a
    different fingerprint and the run's provenance was dropped as corrupt."""
    from nnx.provenance import ExperimentManifest

    manifest = ExperimentManifest(config=None)
    assert ExperimentManifest.from_state(manifest.state()).fingerprint() == manifest.fingerprint()


# --- #551: manifests too deep to read back --------------------------------------------------------


def _nested(levels: int, leaf=None) -> dict:
    """``levels`` nested mappings (the innermost holds ``leaf`` when given),
    built without recursion."""
    value: dict = {} if leaf is None else {"k": leaf}
    for _ in range(levels - 1):
        value = {"k": value}
    return value


@pytest.mark.parametrize("field", ["task", "model", "objective", "config"])
def test_a_manifest_field_too_deep_to_read_back_is_refused_naming_its_path(field):
    """FIX-083: provenance.json wraps a field in two levels and reads 256,
    so 255 levels were written and then refused on load (``NNRun.load``
    dropped ``run.provenance``), and 1500 overflowed the recursive copies
    with a raw RecursionError."""
    for levels in (255, 511, 1500):
        with pytest.raises(ValueError, match=rf"^\$\.{field}: nested more than 254 levels"):
            ExperimentManifest(**{field: _nested(levels)})
    deepest = ExperimentManifest(**{field: _nested(254)})
    assert ExperimentManifest.from_state(deepest.state()) == deepest
    # An identity reference is an object once canonical: it counts as a level.
    ExperimentManifest(**{field: _nested(253, leaf=hash_bytes(b"x"))})
    with pytest.raises(ValueError, match=rf"^\$\.{field}: nested more than 254 levels"):
        ExperimentManifest(**{field: _nested(254, leaf=hash_bytes(b"x"))})


def test_a_self_containing_config_is_refused_not_a_recursion_error():
    config: dict = {}
    config["self"] = config
    with pytest.raises(ValueError, match=r"^\$\.config: nested more than 254 levels"):
        ExperimentManifest(config=config)


def test_a_provenance_file_deeper_than_the_manifest_bound_is_unreadable(tmp_path):
    """The reader's bound is the one construction applies: a 300-level
    config (which an older NNx wrote) parses, then is refused as too deep."""
    from nnx.provenance import FORMAT, MANIFEST_FILE, load_provenance

    run = tmp_path / "runs" / ("a" * 32)
    run.mkdir(parents=True)
    document = {"format": FORMAT, "manifest": {"config": "__DEEP__"}, "fingerprint": "sha256:0"}
    deep = '{"k":' * 299 + "{}" + "}" * 299
    (run / MANIFEST_FILE).write_text(json.dumps(document).replace('"__DEEP__"', deep), encoding="utf-8")
    with pytest.raises(ValueError, match="nested too deeply"):
        load_provenance("a" * 32, root=str(tmp_path))
