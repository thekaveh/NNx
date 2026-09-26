"""FEAT-017: reproducible group, time and stratified splits.

``plan_split`` returns a ``SplitManifest``: disjoint train / validation /
test memberships by sample id, the source identity, the parameters and the
seed. The plan depends only on the ids and their groups, times or labels,
never on row order or on any global RNG. Replaying a manifest maps ids back
to rows, and duplicate, missing or unexpected ids, a changed source identity
or a positional-only plan raise before any loader exists.

Fixtures: **G** = ids ``s0..s5``, groups ``a,a,b,b,c,c``, proportions
``(1/3, 1/3, 1/3)``, seed 7; **T** = the same ids, times ``[1,2,2,3,4,5]``,
cutoffs ``(3, 5)``.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd
import pytest
import torch

from nnx import NNTabularDataset, NNTrainParams
from nnx.data_splits import FORMAT, SplitError, SplitIndices, SplitManifest, plan_split
from nnx.provenance import ExperimentManifest, IdentityRef, hash_bytes

IDS = [f"s{i}" for i in range(6)]
GROUPS = ["a", "a", "b", "b", "c", "c"]
THIRDS = (1 / 3, 1 / 3, 1 / 3)
TIMES = [1, 2, 2, 3, 4, 5]


def plan_g(ids=IDS, groups=GROUPS, **kwargs) -> SplitManifest:
    return plan_split(ids, strategy="group", groups=groups, proportions=THIRDS, seed=7, **kwargs)


def plan_t(gap=0, ids=IDS, times=TIMES, **kwargs) -> SplitManifest:
    return plan_split(ids, strategy="chronological", times=times, cutoffs=(3, 5), gap=gap, **kwargs)


def plan_s(**kwargs) -> SplitManifest:
    ids = [f"r{i:02d}" for i in range(12)]
    labels = [0] * 6 + [1] * 6
    kwargs.setdefault("proportions", (0.5, 0.25, 0.25))
    return plan_split(ids, strategy="stratified", labels=labels, seed=3, **kwargs)


def members(manifest: SplitManifest) -> tuple[set, set, set]:
    return set(manifest.train), set(manifest.validation), set(manifest.test)


# --- every strategy emits a complete, serializable record ---------------------------------------


@pytest.mark.parametrize("make", [plan_g, plan_t, plan_s], ids=["group", "chronological", "stratified"])
def test_each_strategy_emits_disjoint_memberships_identity_parameters_and_seed(make, tmp_path):
    manifest = make(source="animals-v2")
    train, validation, test = members(manifest)
    excluded = set(manifest.excluded)
    assert not (train & validation or train & test or validation & test or excluded & (train | validation | test))
    assert manifest.source == IdentityRef.declared("animals-v2") and not manifest.source.verified
    assert manifest.parameters and manifest.strategy in ("group", "chronological", "stratified")
    state = manifest.state()
    assert state["format"] == FORMAT == "nnx.split/1"
    assert {"strategy", "parameters", "seed", "source", "train", "validation", "test", "excluded"} <= set(state)
    reloaded = SplitManifest.from_json(manifest.to_json())
    assert reloaded == manifest and reloaded.digest() == manifest.digest() and reloaded.state() == state
    json.loads(manifest.to_json())  # plain JSON
    path = tmp_path / "split.json"
    manifest.save(path)
    assert SplitManifest.load(path) == manifest
    assert manifest.digest().startswith("sha256:") and manifest.identity() == hash_bytes(manifest.canonical_bytes())


def test_seeded_strategies_record_their_seed_and_chronological_records_none():
    assert plan_g().seed == 7 and plan_s().seed == 3 and plan_t().seed is None
    drawn = plan_split(IDS, strategy="group", groups=GROUPS, proportions=THIRDS)
    assert isinstance(drawn.seed, int)  # a fresh seed is drawn and recorded, so the plan is replayable
    replanned = plan_split(IDS, strategy="group", groups=GROUPS, proportions=THIRDS, seed=drawn.seed)
    assert replanned == drawn


# --- group ------------------------------------------------------------------------------------------


def test_no_group_spans_two_splits():
    manifest = plan_g()
    by_id = dict(zip(IDS, GROUPS, strict=True))
    splits = [manifest.train, manifest.validation, manifest.test]
    assert [len(split) for split in splits] == [2, 2, 2]
    assert all(len({by_id[i] for i in split}) == 1 for split in splits)  # one group each
    assert len({by_id[split[0]] for split in splits}) == 3
    assert manifest.state()["parameters"] == {"proportions": list(THIRDS)} and manifest.excluded == ()


def test_group_plans_keep_every_active_split_non_empty_and_reject_too_few_groups():
    ids = [f"r{i}" for i in range(12)]
    groups = ["big"] * 10 + ["x", "y"]
    manifest = plan_split(ids, strategy="group", groups=groups, proportions=(0.8, 0.1, 0.1), seed=0)
    assert set(manifest.train) == set(ids[:10]) and len(manifest.validation) == len(manifest.test) == 1
    with pytest.raises(SplitError, match="2 groups .* 3 active splits"):
        plan_split(IDS[:4], strategy="group", groups=GROUPS[:4], proportions=THIRDS, seed=7)
    two = plan_split(IDS[:4], strategy="group", groups=GROUPS[:4], proportions=(0.5, 0.5, 0.0), seed=7)
    assert two.test == () and len(two.train) == len(two.validation) == 2  # an inactive split stays empty


# --- chronological ------------------------------------------------------------------------------------


def test_chronological_splits_honour_cutoffs_gap_and_timestamp_ties():
    zero = plan_t(gap=0)
    assert members(zero) == ({"s0", "s1", "s2"}, {"s3", "s4"}, {"s5"}) and zero.excluded == ()
    one = plan_t(gap=1)
    assert set(one.excluded) == {"s1", "s2", "s4"}
    assert members(one) == ({"s0"}, {"s3"}, {"s5"})
    assert one.state()["parameters"] == {"cutoffs": [3, 5], "gap": 1} and zero.parameters["gap"] == 0
    # A timestamp on a cutoff starts the later split; tied rows never separate.
    tie = plan_split(IDS, strategy="chronological", times=[1, 3, 3, 4, 5, 5], cutoffs=(3, 5))
    assert members(tie) == ({"s0"}, {"s1", "s2", "s3"}, {"s4", "s5"})


def test_chronological_optional_splits_and_rejections():
    no_test = plan_split(IDS, strategy="chronological", times=TIMES, cutoffs=(3, None))
    assert members(no_test) == ({"s0", "s1", "s2"}, {"s3", "s4", "s5"}, set())
    no_val = plan_split(IDS, strategy="chronological", times=TIMES, cutoffs=(None, 5))
    assert members(no_val) == ({"s0", "s1", "s2", "s3", "s4"}, set(), {"s5"})
    empty_window = plan_split(IDS, strategy="chronological", times=TIMES, cutoffs=(5, 5))
    assert empty_window.validation == ()  # an empty optional split is allowed
    with pytest.raises(SplitError, match="no training rows"):
        plan_split(IDS, strategy="chronological", times=[3, 3, 3, 5, 5, 5], cutoffs=(3, 5))
    with pytest.raises(ValueError, match="cutoffs"):
        plan_split(IDS, strategy="chronological", times=TIMES, cutoffs=(5, 3))
    with pytest.raises(ValueError, match="cutoffs"):
        plan_split(IDS, strategy="chronological", times=TIMES, cutoffs=(None, None))
    with pytest.raises(ValueError, match="gap"):
        plan_split(IDS, strategy="chronological", times=TIMES, cutoffs=(3, 5), gap=-1)
    with pytest.raises(ValueError, match="missing"):
        plan_split(IDS, strategy="chronological", times=[1, 2, float("nan"), 3, 4, 5], cutoffs=(3, 5))
    with pytest.raises(ValueError, match="seed"):
        plan_split(IDS, strategy="chronological", times=TIMES, cutoffs=(3, 5), seed=1)


def test_chronological_accepts_datetimes_with_a_timedelta_gap():
    from datetime import datetime, timedelta

    days = [datetime(2024, 1, day) for day in TIMES]
    manifest = plan_split(
        IDS,
        strategy="chronological",
        times=days,
        cutoffs=(datetime(2024, 1, 3), datetime(2024, 1, 5)),
        gap=timedelta(days=1),
    )
    assert members(manifest) == ({"s0"}, {"s3"}, {"s5"}) and set(manifest.excluded) == {"s1", "s2", "s4"}
    assert manifest.state()["parameters"] == {
        "cutoffs": ["2024-01-03T00:00:00", "2024-01-05T00:00:00"],
        "gap": {"seconds": 86400.0},
    }
    assert SplitManifest.from_json(manifest.to_json()) == manifest


# --- stratified -----------------------------------------------------------------------------------


def test_stratified_splits_keep_every_class_in_every_active_split():
    manifest = plan_s()
    labels = {f"r{i:02d}": 0 if i < 6 else 1 for i in range(12)}
    splits = [manifest.train, manifest.validation, manifest.test]
    assert [len(split) for split in splits] == [6, 3, 3]
    assert all({labels[i] for i in split} == {0, 1} for split in splits)
    assert manifest.state()["parameters"] == {"proportions": [0.5, 0.25, 0.25], "insufficient": "raise"}


def test_stratified_support_is_validated_before_any_dataset_exists():
    labels = [0, 0, 0, 1, 1, 2]  # class 1 has two examples, class 2 one; three active splits
    with pytest.raises(SplitError, match=r"class 1 has 2 rows .* 3 active splits"):
        plan_split(IDS, strategy="stratified", labels=labels, proportions=THIRDS, seed=0)
    relaxed = plan_split(IDS, strategy="stratified", labels=labels, proportions=THIRDS, seed=0, insufficient="train")
    assert {"s3", "s4", "s5"} <= set(relaxed.train)  # classes too small to stratify train only
    assert relaxed.parameters["insufficient"] == "train"
    with pytest.raises(ValueError, match="insufficient"):
        plan_split(IDS, strategy="stratified", labels=labels, proportions=THIRDS, insufficient="drop")


# --- source identity and replay ---------------------------------------------------------------------


def test_replay_on_reversed_rows_keeps_id_membership():
    manifest = plan_g()
    reversed_ids = IDS[::-1]
    indices = manifest.resolve(reversed_ids)
    assert isinstance(indices, SplitIndices)
    assert [reversed_ids[i] for i in indices.train] == list(manifest.train)
    assert [reversed_ids[i] for i in indices.validation] == list(manifest.validation)
    assert [reversed_ids[i] for i in indices.test] == list(manifest.test)
    assert plan_g(ids=IDS[::-1], groups=GROUPS[::-1]) == manifest  # planning ignores row order too
    assert plan_t(ids=IDS[::-1], times=TIMES[::-1]) == plan_t()


@pytest.mark.parametrize(
    ("ids", "kwargs", "message"),
    [
        (["s0", "s0", "s2", "s3", "s4", "s5"], {}, "duplicate"),
        (IDS[:5], {}, r"missing .*s5"),
        ([*IDS, "s6"], {}, r"not in the plan.*s6"),
        ([0, 1, 2, 3, 4, 5], {}, "sample ids"),
    ],
    ids=["duplicate", "missing", "unexpected", "wrong-type"],
)
def test_changed_sources_raise_on_replay(ids, kwargs, message):
    with pytest.raises(SplitError, match=message):
        plan_g().resolve(ids, **kwargs)


POSITIONAL = SplitManifest(strategy="explicit", ids="position", train=(0, 1), validation=(2, 3), test=(4, 5))


def test_source_identity_and_positional_only_plans_are_checked():
    recorded = plan_g(source=hash_bytes(b"v1"))
    assert recorded.resolve(IDS, source=hash_bytes(b"v1")).train
    with pytest.raises(SplitError, match="source identity"):
        recorded.resolve(IDS, source=hash_bytes(b"v2"))
    with pytest.raises(SplitError, match="source identity"):
        recorded.resolve(IDS)
    with pytest.raises(SplitError, match="needs source="):
        plan_split(strategy="group", groups=GROUPS, proportions=THIRDS, seed=7)  # would be unreplayable
    with pytest.raises(SplitError, match="positional-only"):
        POSITIONAL.resolve(range(6))
    with pytest.raises(SplitError, match="records no source identity"):
        plan_g().resolve(IDS, source=hash_bytes(b"v1"))  # nothing to check against: never silently ignored
    pinned = plan_split(strategy="group", groups=GROUPS, proportions=THIRDS, seed=7, source="rows-v1")
    assert pinned.ids == "position" and set(pinned.train) <= set(range(6))
    assert pinned.resolve(range(6), source="rows-v1").train == pinned.train
    with pytest.raises(SplitError, match="source identity"):
        pinned.resolve(range(6), source="rows-v2")


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"strategy": "group", "groups": GROUPS}, TypeError, "proportions"),
        ({"strategy": "group", "groups": GROUPS, "proportions": (0.5, 0.2, 0.2)}, ValueError, "sum to 1"),
        ({"strategy": "group", "groups": GROUPS, "proportions": (0.0, 0.5, 0.5)}, ValueError, "train"),
        ({"strategy": "group", "groups": GROUPS, "proportions": THIRDS, "times": TIMES}, TypeError, "times"),
        ({"strategy": "group", "groups": GROUPS[:5], "proportions": THIRDS}, ValueError, "length"),
        ({"strategy": "random", "groups": GROUPS, "proportions": THIRDS}, ValueError, "strategy"),
        ({"strategy": "stratified", "labels": [True] * 6, "proportions": THIRDS}, TypeError, "bool"),
    ],
)
def test_bad_plans_are_rejected(kwargs, error, message):
    with pytest.raises(error, match=message):
        plan_split(IDS, **kwargs)


def test_bad_ids_are_rejected():
    with pytest.raises(SplitError, match="duplicate"):
        plan_split(["s0", *IDS[1:5], "s0"], strategy="group", groups=GROUPS, proportions=THIRDS, seed=7)
    with pytest.raises(TypeError, match="str or int"):
        plan_split([0.5, *range(1, 6)], strategy="group", groups=GROUPS, proportions=THIRDS, seed=7)


def test_manifests_reject_overlaps_and_mixed_ids():
    with pytest.raises(ValueError, match="both train and test"):
        SplitManifest(strategy="explicit", train=("a", "b"), validation=(), test=("b",))
    with pytest.raises(TypeError, match="one id type"):
        SplitManifest(strategy="explicit", train=("a", 1), validation=(), test=())
    with pytest.raises(ValueError, match="format"):
        SplitManifest.from_state({**plan_g().state(), "format": "nnx.split/9"})
    explicit = SplitManifest(strategy="explicit", train=("b", "a"), validation=("c",), test=())
    assert explicit.train == ("a", "b")  # memberships are stored in canonical (sorted) order


# --- global RNG and legacy defaults ------------------------------------------------------------------


def _global_rng() -> tuple:
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    return torch.get_rng_state(), (kind, keys.tolist(), pos, has_gauss, cached), random.getstate()


def test_planning_never_touches_global_rng_state():
    torch_state, numpy_state, python_state = _global_rng()
    plan_g()
    plan_t(gap=1)
    plan_s()
    plan_split(IDS, strategy="group", groups=GROUPS, proportions=THIRDS)  # seed drawn from fresh entropy
    after = _global_rng()
    assert torch.equal(after[0], torch_state) and after[1] == numpy_state and after[2] == python_state


def test_legacy_serialized_defaults_are_unchanged():
    assert "split" not in NNTrainParams(n_epochs=1).state()
    assert not [key for key in NNTrainParams(n_epochs=1).state() if "split" in key]
    ds = NNTabularDataset(df=_frame(), feature_cols=["x"], target_col="y", seed=0)
    assert "split" not in ds.state()


# --- loader integration ------------------------------------------------------------------------------


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"id": IDS, "x": [float(i) for i in range(6)], "y": [0, 1, 0, 1, 0, 1]})


def _ids(loader) -> list[str]:
    if loader is None:
        return []
    return [f"s{int(x)}" for batch_x, _ in loader for x in batch_x[:, 0].tolist()]


def test_loaders_consume_a_manifest_without_random_split(monkeypatch):
    import nnx.nn.dataset.nn_tabular_dataset as tabular

    def no_random_split(*args, **kwargs):
        raise AssertionError("a manifest split must not call random_split")

    monkeypatch.setattr(tabular, "random_split", no_random_split)
    manifest = plan_g()
    shuffled = _frame().sample(frac=1.0, random_state=0).reset_index(drop=True)
    rng = torch.get_rng_state()
    ds = NNTabularDataset(
        df=shuffled, feature_cols=["x"], target_col="y", split=manifest, id_col="id", batch_sizes=(1, 1, 1)
    )
    assert torch.equal(torch.get_rng_state(), rng)
    for _ in range(2):  # shuffled epochs keep every id in its split
        assert sorted(_ids(ds.train_loader)) == sorted(manifest.train)
    assert _ids(ds.val_loader) == list(manifest.validation) and _ids(ds.test_loader) == list(manifest.test)
    assert ds.state()["split"] == manifest.digest()
    assert (ds.state()["n_train"], ds.state()["n_val"], ds.state()["n_test"]) == (2, 2, 2)


def test_empty_optional_splits_give_absent_loaders_and_an_empty_train_fails():
    no_test = plan_split(IDS, strategy="chronological", times=TIMES, cutoffs=(3, None))
    ds = NNTabularDataset(df=_frame(), feature_cols=["x"], target_col="y", split=no_test, id_col="id")
    assert ds.test_loader is None and ds.val_loader is not None and ds.batch_sizes == (3, 3, 1)
    gap = plan_t(gap=1)
    ds = NNTabularDataset(df=_frame(), feature_cols=["x"], target_col="y", split=gap, id_col="id")
    assert (_ids(ds.train_loader), _ids(ds.val_loader), _ids(ds.test_loader)) == (["s0"], ["s3"], ["s5"])
    empty_train = SplitManifest(strategy="explicit", train=(), validation=tuple(IDS[:3]), test=tuple(IDS[3:]))
    with pytest.raises(SplitError, match="no training rows"):
        NNTabularDataset(df=_frame(), feature_cols=["x"], target_col="y", split=empty_train, id_col="id")


def test_bad_replays_raise_before_any_loader_is_created(monkeypatch):
    import nnx.nn.dataset.nn_tabular_dataset as tabular

    loaders = []
    monkeypatch.setattr(tabular, "DataLoader", lambda *args, **kwargs: loaders.append(args))
    frame = _frame()
    duplicated = frame.assign(id=["s0", "s0", "s2", "s3", "s4", "s5"])
    with pytest.raises(SplitError, match="duplicate"):
        NNTabularDataset(df=duplicated, feature_cols=["x"], target_col="y", split=plan_g(), id_col="id")
    with pytest.raises(SplitError, match="missing"):
        NNTabularDataset(df=frame.iloc[:5], feature_cols=["x"], target_col="y", split=plan_g(), id_col="id")
    pinned = plan_g(source=hash_bytes(b"v1"))
    with pytest.raises(SplitError, match="source identity"):
        NNTabularDataset(
            df=frame,
            feature_cols=["x"],
            target_col="y",
            split=pinned,
            id_col="id",
            source_identity=hash_bytes(b"v2"),
        )
    with pytest.raises(SplitError, match="positional-only"):
        NNTabularDataset(df=frame, feature_cols=["x"], target_col="y", split=POSITIONAL)
    with pytest.raises(SplitError, match="str or int"):
        NNTabularDataset(
            df=frame.assign(id=[float(i) for i in range(6)]),
            feature_cols=["x"],
            target_col="y",
            split=plan_split(list(range(6)), strategy="group", groups=GROUPS, proportions=THIRDS, seed=7),
            id_col="id",
        )
    assert loaders == []


def test_split_options_are_validated():
    with pytest.raises(ValueError, match="split="):
        NNTabularDataset(df=_frame(), feature_cols=["x"], target_col="y", id_col="id")
    with pytest.raises(ValueError, match="seed"):
        NNTabularDataset(df=_frame(), feature_cols=["x"], target_col="y", split=plan_g(), id_col="id", seed=0)
    with pytest.raises(ValueError, match="val_proportion"):
        NNTabularDataset(
            df=_frame(), feature_cols=["x"], target_col="y", split=plan_g(), id_col="id", val_proportion=0.3
        )
    with pytest.raises(KeyError, match="nope"):
        NNTabularDataset(df=_frame(), feature_cols=["x"], target_col="y", split=plan_g(), id_col="nope")
    with pytest.raises(ValueError, match="needs id_col"):  # never inferred from a same-valued index
        NNTabularDataset(df=_frame().set_index("id"), feature_cols=["x"], target_col="y", split=plan_g())
    doubled = pd.concat([_frame(), _frame()[["id"]]], axis=1)
    with pytest.raises(ValueError, match="names 2 DataFrame columns"):
        NNTabularDataset(df=doubled, feature_cols=["x"], target_col="y", split=plan_g(), id_col="id")


# --- FEAT-019 consumer ------------------------------------------------------------------------------


def test_provenance_receives_the_same_split_identity():
    manifest = plan_g()
    plan = ExperimentManifest(splits={"plan": manifest})
    assert plan.splits["plan"] == manifest.identity() and plan.splits["plan"].verified
    replayed = SplitManifest.from_json(manifest.to_json())
    reordered = plan_g(ids=IDS[::-1], groups=GROUPS[::-1])
    assert ExperimentManifest(splits={"plan": replayed}).fingerprint() == plan.fingerprint()
    assert ExperimentManifest(splits={"plan": reordered}).fingerprint() == plan.fingerprint()
    other = plan_split(IDS, strategy="group", groups=GROUPS, proportions=THIRDS, seed=8)
    if other != manifest:
        assert ExperimentManifest(splits={"plan": other}).fingerprint() != plan.fingerprint()


def test_reversed_frames_convert_on_both_split_paths():
    # df.iloc[::-1] yields negative NumPy strides, which torch.tensor rejected
    # ("At least one stride ... is negative") on the legacy path as well.
    reversed_frame = _frame().iloc[::-1]
    legacy = NNTabularDataset(df=reversed_frame, feature_cols=["x"], target_col="y", seed=0)
    assert legacy.state()["n_train"] == 6
    planned = NNTabularDataset(df=reversed_frame, feature_cols=["x"], target_col="y", split=plan_g(), id_col="id")
    assert sorted(_ids(planned.train_loader)) == sorted(plan_g().train)


# --- review hardening -------------------------------------------------------------------------


def test_excluded_rows_are_never_admitted_counted_or_required():
    gap = plan_t(gap=1)  # excludes s1, s2, s4
    dirty = _frame().assign(x=[0.0, float("nan"), float("inf"), 3.0, float("nan"), 5.0], y=[0, 7, 7, 1, 7, 0])
    ds = NNTabularDataset(df=dirty, feature_cols=["x"], target_col="y", split=gap, id_col="id")
    assert ds.output_dim == 2  # label 7 lives only in excluded rows
    assert (_ids(ds.train_loader), _ids(ds.val_loader), _ids(ds.test_loader)) == (["s0"], ["s3"], ["s5"])
    dropped = _frame()[~_frame()["id"].isin(gap.excluded)]
    ds = NNTabularDataset(df=dropped, feature_cols=["x"], target_col="y", split=gap, id_col="id")
    assert gap.resolve(dropped["id"]).excluded == () and _ids(ds.test_loader) == ["s5"]


def test_stratified_carry_stays_bounded_for_many_small_classes():
    ids = list(range(400))
    labels = [i // 3 for i in range(300)] + [100] * 100  # 100 three-row classes, then one large class
    manifest = plan_split(ids, strategy="stratified", labels=labels, proportions=(0.5, 0.25, 0.25), seed=0)
    large = set(range(300, 400))
    counts = [len(large & set(split)) for split in (manifest.train, manifest.validation, manifest.test)]
    assert all(abs(count - target) <= 2 for count, target in zip(counts, (50, 25, 25), strict=True)), counts


def test_arrays_nullable_times_and_seeds_are_handled():
    by_array = plan_split(IDS, strategy="group", groups=GROUPS, proportions=np.array(THIRDS), seed=np.int64(7))
    assert by_array == plan_g()
    by_series = plan_split(IDS, strategy="chronological", times=pd.Series(TIMES), cutoffs=np.array([3, 5]))
    assert by_series == plan_t()
    with pytest.raises(ValueError, match="missing"):
        plan_split(IDS, strategy="chronological", times=pd.array([1, 2, None, 3, 4, 5], dtype="Int64"), cutoffs=(3, 5))
    with pytest.raises(ValueError, match="seed"):
        plan_split(IDS, strategy="group", groups=GROUPS, proportions=THIRDS, seed=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gap"):
        plan_split(IDS, strategy="chronological", times=TIMES, cutoffs=(3, 5), gap="1")


def test_manifest_bytes_are_computed_once():
    manifest = plan_g()
    assert manifest.canonical_bytes() is manifest.canonical_bytes()
    assert hash(manifest) == hash(SplitManifest.from_json(manifest.to_json()))


def test_positional_replays_need_every_position_and_a_real_identity():
    positional = plan_split(
        strategy="chronological", times=[2.5, 1, 1, 1, 4, 2.9], cutoffs=(3, None), gap=1, source="rows-v1"
    )
    assert positional.excluded == (0, 5)
    with pytest.raises(SplitError, match="missing"):  # dropping row 0 would shift every later position
        positional.resolve(range(5), source="rows-v1")
    with pytest.raises(SplitError, match="needs source="):
        plan_split(strategy="group", groups=GROUPS, proportions=THIRDS, seed=1, source=IdentityRef.unknown())


def test_unordered_collections_and_bad_calls_are_rejected():
    with pytest.raises(ValueError, match="triple"):
        plan_split(IDS, strategy="group", groups=GROUPS, proportions={0.5, 0.3, 0.2}, seed=1)
    with pytest.raises(ValueError, match="pair"):
        plan_split(IDS, strategy="chronological", times=TIMES, cutoffs={3, 5})
    with pytest.raises(TypeError, match="not iterable"):  # a caller bug is not a SplitError
        plan_g().resolve(None)  # type: ignore[arg-type]
    empty_train = SplitManifest(strategy="explicit", train=(), validation=("a",), test=("b",))
    with pytest.raises(SplitError, match="no training rows"):
        empty_train.resolve(["a", "b"])
    with pytest.raises(SplitError, match="source_identity"):
        plan_g(source="v1").resolve(IDS, source="v2")


def test_small_classes_never_skew_large_class_allocations():
    labels = [i // 4 for i in range(200)] + [50 + i // 100 for i in range(500)]  # 50 four-row, 5 hundred-row
    manifest = plan_split(list(range(700)), strategy="stratified", labels=labels, proportions=(0.8, 0.1, 0.1), seed=2)
    for label in range(50, 55):
        rows = {i for i, value in enumerate(labels) if value == label}
        counts = [len(rows & set(split)) for split in (manifest.train, manifest.validation, manifest.test)]
        assert all(abs(c - t) <= 1 for c, t in zip(counts, (80, 10, 10), strict=True)), (label, counts)


def test_numpy_datetimes_dates_and_timezones():
    from datetime import date, datetime, timedelta, timezone

    days = np.array([f"2024-01-0{d}" for d in TIMES], dtype="datetime64[ns]")
    cut = np.array(["2024-01-03", "2024-01-05"], dtype="datetime64[ns]")
    manifest = plan_split(IDS, strategy="chronological", times=days, cutoffs=cut, gap=np.timedelta64(1, "D"))
    assert members(manifest) == ({"s0"}, {"s3"}, {"s5"})
    plain = [date(2024, 1, d) for d in TIMES]
    with pytest.raises(ValueError, match="whole days"):
        plan_split(IDS, strategy="chronological", times=plain, cutoffs=(date(2024, 1, 3), None), gap=timedelta(hours=1))
    with pytest.raises(TypeError, match="cutoffs\\[0\\] is a date"):
        naive = [datetime(2024, 1, d) for d in TIMES]
        plan_split(IDS, strategy="chronological", times=naive, cutoffs=(date(2024, 1, 3), None))
    aware = [datetime(2024, 1, d, tzinfo=timezone.utc) for d in TIMES]
    with pytest.raises(TypeError, match="naive datetime but times are tz-aware"):
        plan_split(IDS, strategy="chronological", times=aware, cutoffs=(datetime(2024, 1, 3), None))


def test_repr_summarizes_instead_of_listing_every_id():
    big = plan_split(
        [f"r{i}" for i in range(5000)],
        strategy="group",
        groups=[i % 50 for i in range(5000)],
        proportions=(0.8, 0.1, 0.1),
        seed=0,
    )
    text = repr(big)
    assert len(text) < 250 and "train=" in text and "r4999" not in text
