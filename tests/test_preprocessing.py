"""FEAT-018: train-fitted preprocessing and split transforms.

A ``Standardizer`` is fitted on an explicit training membership and never
reads a held-out row; validation, test and inference reuse its frozen
float64 statistics under a declared column / order / dtype schema. Split
views give training and evaluation their own transforms without touching the
base dataset. Statistics and schema serialize as primitive JSON and reload
exactly, so train → save → reload → predict needs no refit.

Fixtures: **S** = ``[[0,5],[2,5],[1000,5]]`` fitted on rows 0–1 → mean
``[1,5]``, scale ``[1,1]``, rows ``[[-1,0],[1,0]]``, held-out ``[999,0]``;
**V** = a ``_TinyVision``-style dataset whose training transform adds a
sampled offset and whose evaluation transform adds 1.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from nnx import Losses, NNModel, NNModelParams, NNOptimParams, NNTabularDataset, NNTrainParams
from nnx.data_splits import SplitManifest
from nnx.nn.dataset.nn_dataset import NNDataset
from nnx.preprocessing import FORMAT, PreprocessingError, SplitView, Standardizer
from nnx.provenance import ExperimentManifest, IdentityRef

S = [[0.0, 5.0], [2.0, 5.0], [1000.0, 5.0]]


def s_frame() -> pd.DataFrame:
    return pd.DataFrame({"id": ["r0", "r1", "r2"], "a": [0.0, 2.0, 1000.0], "b": [5.0, 5.0, 5.0], "y": [0, 1, 0]})


class _RowSpy:
    """A row-indexable source that records every row it hands out."""

    def __init__(self, rows):
        self.rows, self.read = rows, []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        self.read.append(index)
        return self.rows[index]


# --- fit on explicit membership -------------------------------------------------------------


def test_fit_reads_only_the_training_rows():
    spy = _RowSpy(S)
    fitted = Standardizer.fit(spy, rows=[0, 1])
    assert sorted(spy.read) == [0, 1]  # the held-out row is never read
    assert fitted.mean == (1.0, 5.0) and fitted.scale == (1.0, 1.0)  # constant column → scale 1
    out = fitted.transform(torch.tensor(S, dtype=torch.float64))
    assert out.tolist() == [[-1.0, 0.0], [1.0, 0.0], [999.0, 0.0]]
    assert fitted.fit_rows == 2 and fitted.fit_membership.verified  # a digest of the membership
    with pytest.raises(TypeError, match="rows"):
        Standardizer.fit(S)  # type: ignore[call-arg]  # membership is a required keyword
    with pytest.raises(PreprocessingError, match="empty"):
        Standardizer.fit(S, rows=[])
    with pytest.raises(PreprocessingError, match="out of range"):
        Standardizer.fit(S, rows=[0, 3])


def test_fit_on_frames_arrays_and_tensors_agrees():
    frame = s_frame()
    by_frame = Standardizer.fit(frame, rows=[0, 1], columns=["a", "b"])
    assert by_frame.columns == ("a", "b") and by_frame.mean == (1.0, 5.0)
    assert Standardizer.fit(np.array(S), rows=[0, 1]).mean == (1.0, 5.0)
    assert Standardizer.fit(torch.tensor(S), rows=[1, 0]).scale == (1.0, 1.0)
    with pytest.raises(PreprocessingError, match="non-finite"):
        Standardizer.fit([[0.0, float("nan")], [1.0, 2.0]], rows=[0, 1])


def test_dataset_and_plan_integration_fits_on_training_rows_only():
    frame = s_frame()
    plan = SplitManifest(strategy="explicit", train=("r0", "r1"), validation=(), test=("r2",))
    ds = NNTabularDataset(df=frame, feature_cols=["a", "b"], target_col="y", split=plan, id_col="id", standardize=True)
    fitted = ds.standardizer
    assert fitted is not None and fitted.mean == (1.0, 5.0) and fitted.fit_membership == plan.identity()
    train = torch.cat([x for x, _ in ds.train_loader])
    test = torch.cat([x for x, _ in ds.test_loader])
    assert sorted(train.tolist()) == [[-1.0, 0.0], [1.0, 0.0]] and test.tolist() == [[999.0, 0.0]]
    assert frame.equals(s_frame())  # the source is never modified
    assert ds.state()["standardizer"] == fitted.digest()


def test_random_split_membership_is_unchanged_by_standardizing():
    frame = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0) ** 2, "y": [0, 1] * 10})
    plain = NNTabularDataset(df=frame, feature_cols=["a", "b"], target_col="y", seed=3)
    scaled = NNTabularDataset(df=frame, feature_cols=["a", "b"], target_col="y", seed=3, standardize=True)
    assert plain.train_loader.dataset.indices == scaled.train_loader.dataset.indices
    rows = list(plain.train_loader.dataset.indices)
    assert scaled.standardizer.mean == tuple(frame[["a", "b"]].iloc[rows].mean().tolist())


# --- frozen statistics under a declared schema ------------------------------------------------


def test_inference_reuses_frozen_statistics_under_the_declared_schema():
    fitted = Standardizer.fit(s_frame(), rows=[0, 1], columns=["a", "b"])
    extra = s_frame().assign(note="ignored")  # extra columns are ignored
    assert fitted.transform(extra).tolist() == [[-1.0, 0.0], [1.0, 0.0], [999.0, 0.0]]
    with pytest.raises(PreprocessingError, match="order"):
        fitted.transform(s_frame()[["b", "a"]])
    with pytest.raises(PreprocessingError, match="missing"):
        fitted.transform(s_frame()[["a"]])
    with pytest.raises(PreprocessingError, match="width"):
        fitted.transform(torch.zeros(2, 3))
    with pytest.raises(PreprocessingError, match="non-finite"):
        fitted.transform(torch.tensor([[float("inf"), 5.0]]))
    with pytest.raises(PreprocessingError, match="numeric"):
        fitted.transform(s_frame().assign(a=["x", "y", "z"]))
    assert fitted.transform(torch.tensor([[1.0, 5.0]])).dtype == torch.float32  # the declared output dtype


# --- serialization -------------------------------------------------------------------------------


def test_statistics_round_trip_through_primitive_json(tmp_path):
    fitted = Standardizer.fit(np.array([[0.1, 1e-3], [0.7, 3.3], [0.2, -2.5]]), rows=[0, 1, 2])
    state = fitted.state()
    assert state["format"] == FORMAT == "nnx.preprocessing/1"
    assert json.loads(json.dumps(state)) == state  # primitive data only
    reloaded = Standardizer.from_json(fitted.to_json())
    assert reloaded == fitted and reloaded.mean == fitted.mean and reloaded.scale == fitted.scale  # float64-exact
    fitted.save(tmp_path / "standardizer.json")
    assert Standardizer.load(tmp_path / "standardizer.json").digest() == fitted.digest()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"mean": [0.0]}, "length"),
        ({"scale": [1.0, float("nan")]}, "finite"),
        ({"mean": [float("inf"), 0.0]}, "finite"),
        ({"scale": [1.0, 0.0]}, "positive"),
        ({"scale": [1.0, -2.0]}, "positive"),
        ({"columns": ["a"]}, "length"),
        ({"format": "nnx.preprocessing/9"}, "format"),
    ],
)
def test_load_rejects_bad_statistics(change, message):
    state = {**Standardizer.fit(S, rows=[0, 1], columns=None).state(), **change}
    with pytest.raises(PreprocessingError, match=message):
        Standardizer.from_state(state)


def test_train_save_reload_predict_needs_no_refit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    fits = []
    real_fit = Standardizer.fit.__func__

    def counting_fit(cls, *args, **kwargs):
        fits.append(1)
        return real_fit(cls, *args, **kwargs)

    monkeypatch.setattr(Standardizer, "fit", classmethod(counting_fit))
    gen = torch.Generator().manual_seed(0)
    frame = pd.DataFrame(
        {"a": torch.randn(24, generator=gen).mul(100).tolist(), "b": torch.randn(24, generator=gen).tolist()}
    )
    frame["y"] = (frame["a"] > 0).astype(int)
    ds = NNTabularDataset(
        df=frame, feature_cols=["a", "b"], target_col="y", seed=0, standardize=True, batch_sizes=(8, None, None)
    )
    torch.manual_seed(0)
    model = NNModel(module=torch.nn.Linear(2, 2), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    model.train(
        params=NNTrainParams(
            n_epochs=2,
            train_loader=ds.train_loader,
            optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
            data_id="preprocessing",
        )
    )
    assert fits == [1]
    ds.standardizer.save("standardizer.json")
    reloaded = Standardizer.load("standardizer.json")
    prepared = reloaded.transform(frame[["a", "b"]])  # applied once to raw rows
    rows = list(ds.train_loader.dataset.indices)
    assert torch.equal(prepared[rows], ds.train_loader.dataset.dataset.tensors[0][rows])  # no double transform
    assert np.array_equal(model.predict(prepared).logits, model.predict(ds.standardizer.transform(frame)).logits)
    assert fits == [1]  # reload never refits


def test_runtime_transforms_declare_that_reconstruction_needs_registration():
    view = SplitView(_TinyVision("."), [0, 1], transform=_AddOne())
    assert view.reconstructible is False
    assert view.state()["transform"] == {"kind": "runtime", "qualname": "_AddOne", "reconstructible": False}
    fitted = Standardizer.fit(S, rows=[0, 1])
    assert SplitView(_TinyVision("."), [0], transform=fitted).state()["transform"] == {
        "kind": "standardizer",
        "digest": fitted.digest(),
        "reconstructible": True,
    }


# --- split views ------------------------------------------------------------------------------------


class _TinyVision(VisionDataset):
    """V: 12 train / 6 test samples of shape (1, 2, 2), 3 classes."""

    classes = ["a", "b", "c"]

    def __init__(self, root, train=True, download=False, transform=None):
        super().__init__(root, transform=transform)
        n = 12 if train else 6
        self.data = torch.arange(n * 4, dtype=torch.float32).reshape(n, 1, 2, 2) + (0 if train else 100)
        self.targets = torch.arange(n) % 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.targets[idx])


class _AddOne:
    def __call__(self, x):
        return x + 1


class _AddSampledOffset:
    def __call__(self, x):
        return x + torch.rand(())


def _vision(tmp_path) -> NNDataset:
    return NNDataset(
        ds_class=_TinyVision,
        root_dir=str(tmp_path),
        download=False,
        val_proportion=0.25,
        seed=0,
        batch_sizes=(4, 4, 4),
        train_transform=_AddSampledOffset(),
        eval_transform=_AddOne(),
    )


def test_training_and_evaluation_transforms_use_independent_views(tmp_path):
    ds = _vision(tmp_path)
    train_view, val_view = ds.train_loader.dataset, ds.val_loader.dataset
    base = train_view.base
    assert base is val_view.base and base.transform is None
    val_ids = tuple(val_view.indices)
    first = [val_view[i][0] for i in range(len(val_view))]
    for _ in range(3):  # alternate training and evaluation reads
        for i in range(len(train_view)):
            x, y = train_view[i]
            raw = base.data[train_view.indices[i]]
            assert y == int(base.targets[train_view.indices[i]]) and 0 <= float((x - raw).mean()) < 1
        again = [val_view[i] for i in range(len(val_view))]
        assert all(torch.equal(a, b) for a, (b, _) in zip(first, again, strict=True))
        assert [y for _, y in again] == [int(base.targets[i]) for i in val_ids]
    assert base.transform is None and tuple(val_view.indices) == val_ids
    assert all(torch.equal(x, base.data[i] + 1) for x, i in zip(first, val_ids, strict=True))
    test_view = ds.test_loader.dataset
    assert torch.equal(test_view[0][0], test_view.base.data[0] + 1)


def test_evaluation_views_are_identical_under_multi_worker_loading(tmp_path):
    ds = _vision(tmp_path)
    single = [batch for batch in DataLoader(ds.val_loader.dataset, batch_size=2)]
    workers = [batch for batch in DataLoader(ds.val_loader.dataset, batch_size=2, num_workers=2)]
    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(single, workers, strict=True))
    assert ds.val_loader.dataset.base.transform is None


def test_existing_single_transform_calls_are_unchanged(tmp_path):
    plain = NNDataset(ds_class=_TinyVision, root_dir=str(tmp_path), download=False, seed=0)
    assert type(plain.train_loader.dataset).__name__ == "Subset"
    assert set(plain.state()) == {
        "name",
        "input_dim",
        "output_dim",
        "train_batch_size",
        "val_batch_size",
        "test_batch_size",
    }
    frame = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "y": [0, 1, 0, 1]})
    assert "standardizer" not in NNTabularDataset(df=frame, feature_cols=["a"], target_col="y", seed=0).state()
    views = _vision(tmp_path)
    assert views.state()["train_transform"]["qualname"] == "_AddSampledOffset"


def test_standardize_options_are_validated():
    frame = s_frame()
    fitted = Standardizer.fit(frame, rows=[0, 1], columns=["a", "b"])
    with pytest.raises(TypeError, match="standardize must be"):
        NNTabularDataset(df=frame, feature_cols=["a", "b"], target_col="y", standardize="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="floating"):
        NNTabularDataset(df=frame, feature_cols=["a", "b"], target_col="y", standardize=True, feature_dtype=torch.int64)
    reused = NNTabularDataset(df=frame, feature_cols=["a", "b"], target_col="y", standardize=fitted, seed=0)
    assert reused.standardizer is fitted  # a supplied standardizer is applied, never refitted
    rng = torch.get_rng_state()  # mismatches are caught before the split draws from the global RNG
    with pytest.raises(PreprocessingError, match="differ from feature_cols"):
        NNTabularDataset(df=frame, feature_cols=["b", "a"], target_col="y", standardize=fitted)
    with pytest.raises(PreprocessingError, match="2 features but feature_cols has 3"):
        NNTabularDataset(df=frame.assign(c=1.0), feature_cols=["a", "b", "c"], target_col="y", standardize=fitted)
    with pytest.raises(ValueError, match="dtype"):
        NNTabularDataset(
            df=frame, feature_cols=["a", "b"], target_col="y", standardize=fitted, feature_dtype=torch.float64
        )
    assert torch.equal(torch.get_rng_state(), rng)


# --- review hardening -------------------------------------------------------------------------


def test_fitting_ignores_physical_column_order_and_unrelated_duplicates():
    frame = s_frame()
    frame.insert(0, "note", "x")
    frame = pd.concat([frame, frame[["note"]]], axis=1)  # a duplicated, unrelated label
    ds = NNTabularDataset(df=frame, feature_cols=["b", "a"], target_col="y", seed=0, standardize=True)
    assert ds.standardizer.columns == ("b", "a")
    clone = __import__("dataclasses").replace(ds, batch_sizes=(1, None, None))  # rebuilds and refits the same
    assert clone.standardizer == ds.standardizer


def test_constant_columns_get_scale_one_and_their_exact_value():
    fitted = Standardizer.fit([[0.1, 0.7], [0.1, 0.7], [0.1, 0.7]], rows=[0, 1, 2])
    assert fitted.mean == (0.1, 0.7) and fitted.scale == (1.0, 1.0)
    assert fitted.transform([[0.1, 0.7], [0.2, 0.7]]).tolist() == [[0.0, 0.0], pytest.approx([0.1, 0.0])]


def test_integer_labels_float16_and_single_samples():
    frame = pd.DataFrame({0: [1e5, 3e5, 2e5, 4e5], 1: [0, 1, 0, 1]})
    ds = NNTabularDataset(
        df=frame, feature_cols=[0], target_col=1, standardize=True, feature_dtype=torch.float16, seed=0
    )
    assert ds.standardizer.columns == (0,) and Standardizer.from_json(ds.standardizer.to_json()) == ds.standardizer
    fitted = Standardizer.fit(S, rows=[0, 1])
    view = SplitView(torch.utils.data.TensorDataset(torch.tensor(S), torch.tensor([0, 1, 0])), [2], transform=fitted)
    x, y = view[0]
    assert x.tolist() == [999.0, 0.0] and int(y) == 0
    small = Standardizer(mean=(0.0,), scale=(1e-3,), dtype="float16")
    with pytest.raises(PreprocessingError, match="overflow"):
        small.transform([[100.0]])


@pytest.mark.parametrize(
    "state",
    [
        [1, 2],
        {"format": FORMAT, "kind": "standardize", "scale": [1.0]},
        {"format": FORMAT, "kind": "standardize", "mean": [0.0], "scale": [1.0], "fit": [1]},
        {"format": FORMAT, "kind": "standardize", "mean": [0.0], "scale": [1.0], "dtype": ["float32"]},
        {"format": FORMAT, "kind": "standardize", "mean": [0.0, 1.0], "scale": [1.0, 1.0], "columns": "ab"},
        {"format": FORMAT, "kind": "standardize", "mean": [0.0], "scale": [1.0], "fit": {"membership": 3}},
    ],
    ids=["not-a-mapping", "missing-mean", "fit-list", "dtype-list", "columns-string", "membership"],
)
def test_malformed_states_raise_preprocessing_errors(state):
    with pytest.raises(PreprocessingError):
        Standardizer.from_state(state)
    with pytest.raises(PreprocessingError):
        Standardizer.from_json("{not json")


def test_split_views_keep_sample_types_and_the_subset_alias():
    from collections import namedtuple

    Pair = namedtuple("Pair", "image label")

    class Named(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return Pair(torch.tensor([float(index)]), index)

    view = SplitView(Named(), [1], transform=_AddOne())
    sample = view[0]
    assert isinstance(sample, Pair) and sample.image.tolist() == [2.0] and sample.label == 1
    as_list = SplitView([[torch.tensor([1.0]), 5]], [0], transform=_AddOne())[0]
    assert isinstance(as_list, list) and as_list[1] == 5
    assert view.dataset is view.base


# --- FEAT-019 consumer -------------------------------------------------------------------------------


def test_provenance_carries_the_fitted_schema_and_membership_not_recomputed_statistics():
    fitted = Standardizer.fit(s_frame(), rows=[0, 1], columns=["a", "b"], membership="train-v1")
    model = NNModel(module=torch.nn.Linear(2, 2), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    plan = ExperimentManifest.for_model(model, preprocessing=fitted)
    recorded = plan.state()["config"]["preprocessing"]
    assert recorded == fitted.state() and recorded["fit"]["membership"] == IdentityRef.declared("train-v1").state()
    reloaded = Standardizer.from_json(fitted.to_json())
    assert ExperimentManifest.for_model(model, preprocessing=reloaded).fingerprint() == plan.fingerprint()


def test_single_rows_match_the_batch_path_exactly():
    fitted = Standardizer.fit([[1.7e9 + 37], [1.7e9 + 1e3], [1.7e9 - 5e2]], rows=[0, 1, 2], dtype="float64")
    batch = fitted.transform([[1.7e9 + 37]])[0]
    assert torch.equal(fitted([1.7e9 + 37]), batch) and torch.equal(fitted(np.array([1.7e9 + 37])), batch)
    named = Standardizer.fit(s_frame(), rows=[0, 1], columns=["a", "b"])
    assert named(s_frame().iloc[2]).tolist() == [999.0, 0.0]  # a Series row is checked by name
    with pytest.raises(PreprocessingError, match="order"):
        named(s_frame()[["b", "a"]].iloc[2])


def test_schema_problems_surface_before_the_split_draws():
    rng = torch.get_rng_state()
    text = pd.DataFrame({"a": ["x", "y", "z", "w"], "y": [0, 1, 0, 1]})
    with pytest.raises(PreprocessingError, match="numeric"):
        NNTabularDataset(df=text, feature_cols=["a"], target_col="y", standardize=True)
    tupled = pd.DataFrame({("x", "a"): [1.0, 2.0, 3.0, 4.0], ("x", "y"): [0, 1, 0, 1]})
    with pytest.raises(PreprocessingError, match="str or int"):
        NNTabularDataset(df=tupled, feature_cols=[("x", "a")], target_col=("x", "y"), standardize=True)
    assert torch.equal(torch.get_rng_state(), rng)
    numbered = pd.DataFrame(np.array([[1.0, 0.0], [3.0, 1.0], [2.0, 0.0], [4.0, 1.0]]))
    ds = NNTabularDataset(df=numbered.astype({1: int}), feature_cols=[np.int64(0)], target_col=1, standardize=True)
    assert ds.standardizer.columns == (0,) and type(ds.standardizer.columns[0]) is int


def test_statistics_ignore_membership_order_and_survive_huge_values():
    values = np.random.default_rng(0).normal(0.1, 1e3, size=(800, 2))
    rows = np.arange(800)
    shuffled = np.random.default_rng(1).permutation(rows)
    assert Standardizer.fit(values, rows=rows).digest() == Standardizer.fit(values, rows=shuffled).digest()
    huge = np.array([[1.7e308], [1.7e308], [-1e308], [1.0]] * 3)
    fitted = Standardizer.fit(huge, rows=range(12))
    assert np.isfinite(fitted.mean).all() and np.isfinite(fitted.scale).all()


def test_inputs_and_states_are_validated_strictly():
    mixed = pd.DataFrame({"a": [1.0, 2.0], "b": [True, False]})
    unnamed = Standardizer.fit(np.array([[1.0, 0.0], [2.0, 1.0]]), rows=[0, 1])
    assert unnamed.transform(mixed).shape == (2, 2)  # bool + float frames are numeric
    with pytest.raises(PreprocessingError, match="ordered list"):
        Standardizer.fit(s_frame(), rows=[0, 1], columns="ab")  # type: ignore[arg-type]
    with pytest.raises(PreprocessingError, match="ordered list"):
        Standardizer(mean=(0.0, 0.0), scale=(1.0, 1.0), columns={"a", "b"})  # type: ignore[arg-type]
    state = {**unnamed.state(), "mean": [10**400, 0.0]}
    with pytest.raises(PreprocessingError, match="finite"):
        Standardizer.from_state(state)
    with pytest.raises(PreprocessingError, match="membership"):
        Standardizer.fit(S, rows=[0, 1], membership="")
    spy = _RowSpy(S)
    with pytest.raises(TypeError, match="membership"):
        Standardizer.fit(spy, rows=[0, 1], membership=3)
    assert spy.read == []  # rejected before any row is read


def test_transform_never_mutates_its_input_and_views_forward_classes():
    fitted = Standardizer.fit(S, rows=[0, 1], dtype="float64")
    tensor = torch.tensor(S, dtype=torch.float64)
    frame = pd.DataFrame(S, columns=["a", "b"])
    before_tensor, before_frame = tensor.clone(), frame.copy()
    fitted.transform(tensor)
    fitted.transform(frame.to_numpy())
    assert torch.equal(tensor, before_tensor) and frame.equals(before_frame)
    view = SplitView(_TinyVision("."), [0], transform=_AddOne())
    assert view.classes == ["a", "b", "c"]
    with pytest.raises(TypeError, match="dict"):
        SplitView([{"x": torch.zeros(1), "y": 0}], [0], transform=_AddOne())[0]
