"""FEAT-020: mergeable metric accumulators (`nnx.streaming.StreamingMetrics`).

`update` / `merge` / `finalize`: merges add sufficient statistics over
disjoint samples in any order and refuse different declarations; finalize is
a read-only, repeatable snapshot; a metric that needs every stored score is
refused in bounded mode unless storage is asked for explicitly.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn import metrics as sk

from nnx import MetricSpec, TaskSpec, register_metric, unregister_metric
from nnx.streaming import MetricMergeError, MetricSnapshot, StreamingMetrics

PROBABILITIES = np.array([[0.8, 0.2], [0.4, 0.6]])
TARGETS = np.array([0, 1])


def _probability_metrics(**kwargs) -> StreamingMetrics:
    return StreamingMetrics([MetricSpec("nll"), MetricSpec("brier")], "categorical", **kwargs)


def test_nll_and_brier_oracle():
    acc = _probability_metrics()
    assert acc.update(TARGETS, probabilities=PROBABILITIES) == 2
    snapshot = acc.finalize()
    assert snapshot.count == 2 and snapshot.available and snapshot.unavailable == ()
    assert snapshot.values["nll"] == pytest.approx(0.36698459, abs=1e-8)
    assert snapshot.values["brier"] == pytest.approx(0.20, abs=1e-12)


def test_merge_is_order_independent_and_matches_one_accumulation():
    rng = np.random.default_rng(0)
    target = rng.integers(0, 3, 40)
    logits = rng.normal(size=(40, 3))
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    declared = [MetricSpec("accuracy"), MetricSpec("f1"), MetricSpec("f1", config={"average": "weighted"}, name="wf1")]
    declared += [MetricSpec("nll"), MetricSpec("brier")]

    def accumulate(rows) -> StreamingMetrics:
        acc = StreamingMetrics(declared, "categorical")
        acc.update(target[rows], probabilities=probabilities[rows])
        return acc

    parts = [accumulate(rows) for rows in np.array_split(np.arange(40), 4)]
    whole = accumulate(np.arange(40)).finalize()
    forward = parts[0].merge(parts[1]).merge(parts[2]).merge(parts[3]).finalize()
    backward = parts[3].merge(parts[2]).merge(parts[1].merge(parts[0])).finalize()
    assert forward.count == backward.count == whole.count == 40
    for name in whole.values:
        assert forward.values[name] == pytest.approx(whole.values[name], rel=1e-12)
        assert backward.values[name] == pytest.approx(forward.values[name], rel=1e-12)
    predicted = probabilities.argmax(axis=1)
    assert whole.values["accuracy"] == sk.accuracy_score(target, predicted)
    assert whole.values["f1"] == pytest.approx(sk.f1_score(target, predicted, average="macro"), rel=1e-12)
    assert whole.values["wf1"] == pytest.approx(sk.f1_score(target, predicted, average="weighted"), rel=1e-12)


def test_merge_rejects_different_labels_metrics_and_semantics():
    base = _probability_metrics(labels=("neg", "pos"))
    with pytest.raises(MetricMergeError, match="task labels"):
        base.merge(_probability_metrics(labels=("no", "yes")))
    with pytest.raises(MetricMergeError, match="metric declarations"):
        base.merge(StreamingMetrics([MetricSpec("nll"), MetricSpec("accuracy")], "categorical", labels=("neg", "pos")))
    register_metric("tests.nll2", 2, lambda config: _Summing(), input="probabilities", mode="min")
    try:
        other_version = StreamingMetrics([MetricSpec("tests.nll2", 2)], "categorical")
        with pytest.raises(MetricMergeError, match="metric declarations"):
            StreamingMetrics([MetricSpec("nll", name="tests.nll2")], "categorical").merge(other_version)
    finally:
        unregister_metric("tests.nll2", 2)
    with pytest.raises(MetricMergeError, match="probability semantics"):
        StreamingMetrics([MetricSpec("brier")], "categorical").merge(
            StreamingMetrics([MetricSpec("brier")], "bernoulli")
        )
    with pytest.raises(MetricMergeError, match="thresholds"):
        StreamingMetrics([MetricSpec("accuracy")], "bernoulli", threshold=0.5).merge(
            StreamingMetrics([MetricSpec("accuracy")], "bernoulli", threshold=0.3)
        )
    with pytest.raises(TypeError, match="StreamingMetrics"):
        base.merge(object())  # type: ignore[arg-type]


def test_finalize_is_read_only_repeatable_and_snapshots_are_unaliased():
    acc = _probability_metrics()
    acc.update(TARGETS, probabilities=PROBABILITIES)
    first, again = acc.finalize(), acc.finalize()
    assert first == again and acc.count == 2  # repeatable, and finalize consumed nothing
    with pytest.raises(TypeError):
        first.values["nll"] = 0.0  # type: ignore[index]
    merged = acc.merge(_probability_metrics())
    acc.update(np.array([1]), probabilities=np.array([[0.9, 0.1]]))  # a later, very wrong prediction
    assert acc.finalize().values["nll"] > first.values["nll"]
    assert first.values == again.values and first.count == 2  # the earlier snapshot is unchanged
    assert merged.finalize().values == first.values  # the merge copied its inputs: no aliasing


def test_empty_merge_is_the_identity():
    acc = _probability_metrics()
    acc.update(TARGETS, probabilities=PROBABILITIES)
    empty = _probability_metrics()
    assert acc.merge(empty).finalize() == acc.finalize() == empty.merge(acc).finalize()
    assert empty.merge(empty).finalize() == empty.finalize()


def test_empty_and_all_masked_streams_finalize_unavailable_with_count_zero():
    empty = _probability_metrics().finalize()
    assert empty == MetricSnapshot(count=0, values={}, unavailable=("nll", "brier")) and not empty.available
    masked = StreamingMetrics([MetricSpec("mae")], "continuous")
    assert masked.update(np.array([np.nan, np.nan]), values=np.array([1.0, 2.0])) == 0  # NaN targets are masked
    assert masked.update(np.array([1.0, 2.0]), values=np.array([1.0, 2.0]), valid=np.array([False, False])) == 0
    snapshot = masked.finalize()
    assert snapshot.count == 0 and snapshot.values == {} and snapshot.unavailable == ("mae",)


def test_bernoulli_and_continuous_semantics_and_logit_updates():
    acc = StreamingMetrics([MetricSpec("accuracy"), MetricSpec("brier")], "bernoulli", threshold=0.7)
    acc.update(np.array([[1, 0], [1, np.nan]]), probabilities=np.array([[0.8, 0.6], [0.6, 0.5]]))
    snapshot = acc.finalize()
    assert snapshot.count == 3  # one masked entry
    assert snapshot.values["accuracy"] == pytest.approx(2 / 3)  # 0.6 < 0.7 decodes as 0
    assert snapshot.values["brier"] == pytest.approx(((0.2**2) + (0.6**2) + (0.4**2)) / 3)
    regression = StreamingMetrics([MetricSpec("mae"), MetricSpec("mse")], "continuous")
    regression.update_logits(torch.tensor([[1.0, 2.0], [3.0, float("nan")]]), torch.tensor([[2.0, 2.0], [1.0, 9.0]]))
    assert regression.finalize().values == {"mae": pytest.approx(1.0), "mse": pytest.approx(5 / 3)}
    task = StreamingMetrics.for_task([MetricSpec("nll")], TaskSpec.categorical(2, labels=("a", "b")))
    task.update_logits(torch.tensor([0, 1]), torch.log(torch.tensor(PROBABILITIES)))
    assert task.finalize().values["nll"] == pytest.approx(0.36698459, abs=1e-6)
    assert task.labels == ("a", "b") and task.semantics == "categorical"


def test_updates_are_validated():
    acc = _probability_metrics(labels=("a", "b"))
    with pytest.raises(ValueError, match="2 outputs along axis 1 but 3 are declared"):
        StreamingMetrics([MetricSpec("nll")], "categorical", labels=("a", "b", "c")).update(
            TARGETS, probabilities=PROBABILITIES
        )
    with pytest.raises(ValueError, match="needs probabilities"):
        acc.update(TARGETS, labels=np.array([0, 1]))
    with pytest.raises(ValueError, match=r"\(N, C\)"):
        acc.update(TARGETS, probabilities=np.array([0.2, 0.8]))
    with pytest.raises(ValueError, match="needs continuous inputs"):
        StreamingMetrics([MetricSpec("mae")], "continuous").update(np.array([1.0]))
    with pytest.raises(ValueError, match="cannot derive"):
        StreamingMetrics([MetricSpec("mae")], "categorical")  # a continuous metric on class probabilities
    with pytest.raises(ValueError, match="semantics"):
        StreamingMetrics([MetricSpec("nll")], "ordinal")


class _Summing:
    """A mergeable custom accumulator (sum of predictions)."""

    def __init__(self):
        self.total = 0.0

    def update(self, target, prediction):
        self.total += float(np.sum(prediction))

    def merge(self, other):
        self.total += other.total

    def result(self):
        return self.total


class _Scores:
    """A rank metric: needs every stored score (no merge)."""

    def __init__(self):
        self.targets, self.scores = [], []

    def update(self, target, prediction):
        self.targets.append(np.asarray(target))
        self.scores.append(np.asarray(prediction))

    def result(self):
        if not self.targets:
            return None
        return float(sk.roc_auc_score(np.concatenate(self.targets), np.concatenate(self.scores)))


@pytest.fixture
def auroc():
    register_metric("tests.auroc", 1, lambda config: _Scores(), input="probabilities", mode="max")
    register_metric("tests.sum", 1, lambda config: _Summing(), input="probabilities", mode="max")
    yield MetricSpec("tests.auroc")
    unregister_metric("tests.auroc", 1)
    unregister_metric("tests.sum", 1)


def test_a_stored_score_metric_is_refused_in_bounded_mode_or_materialized(auroc):
    with pytest.raises(ValueError, match="no bounded, mergeable form: its accumulator has no merge") as refused:
        StreamingMetrics([auroc], "bernoulli")  # refused before any update
    assert "materialize=True" in str(refused.value)
    stored = _Scores()
    stored.stores_scores = True  # type: ignore[attr-defined]
    stored.merge = lambda other: None  # type: ignore[attr-defined]
    register_metric("tests.flagged", 1, lambda config: stored, input="probabilities", mode="max")
    try:
        with pytest.raises(ValueError, match="stores_scores=True.*never additive"):
            StreamingMetrics([MetricSpec("tests.flagged")], "bernoulli")  # mergeable, but declares stored scores
    finally:
        unregister_metric("tests.flagged", 1)
    target = np.array([0, 0, 1, 1, 1, 0])
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.2])
    left = StreamingMetrics([auroc], "bernoulli", materialize=True)
    right = StreamingMetrics([auroc], "bernoulli", materialize=True)
    left.update(target[:3], probabilities=scores[:3])
    right.update(target[3:], probabilities=scores[3:])
    assert not left.bounded
    expected = sk.roc_auc_score(target, scores)
    assert left.merge(right).finalize().values["tests.auroc"] == pytest.approx(expected)
    assert right.merge(left).finalize().values["tests.auroc"] == pytest.approx(expected)
    mergeable = StreamingMetrics([MetricSpec("tests.sum")], "bernoulli")  # a custom metric with merge()
    mergeable.update(target, probabilities=scores)
    assert mergeable.bounded and mergeable.finalize().values["tests.sum"] == pytest.approx(scores.sum())


def test_review_round_one_label_rules_match_scikit_learn():
    binary = MetricSpec("f1", config={"average": "binary"}).accumulator()
    target, decided = np.array([-1, 1, 1, -1]), np.array([-1, 1, -1, -1])
    binary.update(target, decided)  # like scikit-learn: class 1 of any two-class label set
    assert binary.result() == pytest.approx(sk.f1_score(target, decided, average="binary"))
    with pytest.raises(ValueError, match=r"class indices in \[0, C\)"):  # categorical targets are class indices
        StreamingMetrics([MetricSpec("f1")], "categorical").update(target, labels=decided)
    with pytest.raises(ValueError, match="integer class"):
        StreamingMetrics([MetricSpec("f1")], "categorical").update(np.array([0.7, 1.0]), labels=np.array([0, 1]))
    with pytest.raises(ValueError, match="integer class"):
        StreamingMetrics([MetricSpec("accuracy")], "categorical").update(np.array([0, 1]), labels=np.array([0.5, 1]))


def test_review_round_one_update_logits_masks_nan_targets_even_with_a_valid_mask():
    acc = StreamingMetrics([MetricSpec("mae")], "continuous")
    acc.update_logits(
        torch.tensor([[1.0, float("nan")]]), torch.tensor([[2.0, 5.0]]), valid=torch.tensor([[True, True]])
    )
    assert acc.count == 1 and acc.finalize().values == {"mae": pytest.approx(1.0)}  # as update() would


def test_review_round_two_ignore_index_width_and_merge_contract():
    task = TaskSpec.categorical(3, ignore_index=-100)
    acc = StreamingMetrics.for_task([MetricSpec("accuracy"), MetricSpec("nll")], task)
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 5.0]])
    assert acc.update_logits(torch.tensor([0, 1, -100, 2]), logits) == 3  # the ignored target is not scored
    assert acc.finalize().values["accuracy"] == 1.0 and acc.ignore_index == -100
    probabilities = torch.softmax(logits, dim=1).numpy()
    assert acc.update(np.array([-100, 0]), probabilities=probabilities[:2]) == 1  # update() masks it too
    with pytest.raises(MetricMergeError, match="ignore indices"):
        acc.merge(StreamingMetrics([MetricSpec("accuracy"), MetricSpec("nll")], "categorical", num_outputs=3))
    with pytest.raises(ValueError, match="3 are declared"):
        StreamingMetrics([MetricSpec("nll")], "categorical", labels=("a", "b", "c")).update_logits(
            torch.tensor([0]), torch.zeros(1, 5)
        )


class _Returning(_Summing):
    """merge() that returns a new object instead of updating in place."""

    def merge(self, other):
        combined = _Returning()
        combined.total = self.total + other.total
        return combined


def test_review_round_two_merge_must_update_in_place(auroc):
    register_metric("tests.returning", 1, lambda config: _Returning(), input="probabilities", mode="max")
    try:
        left = StreamingMetrics([MetricSpec("tests.returning")], "bernoulli")
        with pytest.raises(TypeError, match="in place and return None"):
            left.merge(StreamingMetrics([MetricSpec("tests.returning")], "bernoulli"))
    finally:
        unregister_metric("tests.returning", 1)
    stored = StreamingMetrics([auroc], "bernoulli", materialize=True)
    other = StreamingMetrics([auroc], "bernoulli", materialize=True)
    stored.update(np.array([0, 1]), probabilities=np.array([0.2, 0.9]))
    other.update(np.array([1, 0]), probabilities=np.array([0.7, 0.4]))
    merged = stored.merge(other)
    before = merged.finalize()
    other.update(np.array([0, 1]), probabilities=np.array([0.99, 0.01]))  # would flip the AUROC if aliased
    stored.update(np.array([0, 1]), probabilities=np.array([0.99, 0.01]))
    assert merged.finalize() == before


def test_review_round_two_refusals_name_the_remedy_of_their_caller(auroc):
    with pytest.raises(ValueError, match="materialize=True"):
        StreamingMetrics([auroc], "bernoulli")
    from nnx.streaming import _streaming_problems

    params = SimpleNamespace(extra_metrics=None, metrics=(auroc,))
    [(path, message)] = _streaming_problems(params)  # type: ignore[arg-type]
    assert path == "train.metrics[0]" and "default validation step" in message and "materialize" not in message


def test_review_round_three_masks_ranges_widths_and_mixed_results():
    logits = torch.tensor([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]])
    acc = StreamingMetrics([MetricSpec("accuracy")], "categorical", ignore_index=-100)
    assert acc.update_logits(torch.tensor([0, 1, -100]), logits, valid=torch.tensor([True, True, True])) == 2
    assert acc.finalize().values["accuracy"] == 1.0  # the ignore index applies with an explicit mask too
    nll = StreamingMetrics([MetricSpec("nll")], "categorical")
    with pytest.raises(ValueError, match=r"class indices in \[0, 2\)"):
        nll.update(np.array([0, 5]), probabilities=PROBABILITIES)
    with pytest.raises(ValueError, match=r"class indices in \[0, 2\)"):
        nll.update(np.array([-1, 1]), probabilities=PROBABILITIES)  # never read as the last class
    with pytest.raises(ValueError, match=r"class indices in \[0, 2\)"):
        nll.update_logits(torch.tensor([0, 7, 1]), logits)
    with pytest.raises(ValueError, match="integer class"):
        nll.update(np.array([0.0, np.inf]), probabilities=PROBABILITIES)
    outputs = StreamingMetrics([MetricSpec("brier")], "bernoulli", labels=("a", "b", "c"))
    with pytest.raises(ValueError, match="3 are declared"):
        outputs.update(np.zeros((2, 5)), probabilities=np.full((2, 5), 0.5))
    assert nll.count == 0  # every refused batch left the accumulation untouched

    from nnx import PredictionResult
    from nnx.streaming import concatenate_predictions

    ids = np.arange(2)
    bare = PredictionResult(
        logits=np.zeros((2, 1)), probabilities=None, decoded=np.zeros((2, 1)), sample_ids=ids, spec=None
    )
    full = PredictionResult(
        logits=np.zeros((2, 1)), probabilities=np.zeros((2, 1)), decoded=np.zeros((2, 1)), sample_ids=ids, spec=None
    )
    for mixed in ([bare, full], [full, bare]):
        with pytest.raises(ValueError, match="with and without probabilities"):
            concatenate_predictions(mixed)


def test_review_round_four_both_update_paths_share_one_masking_rule():
    logits = torch.tensor([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0], [0.0, 5.0]])
    probabilities = torch.softmax(logits, dim=1).numpy()
    target = np.array([0.0, 1.0, -100.0, np.nan])  # float class indices: one ignored, one missing
    by_logits = StreamingMetrics([MetricSpec("accuracy"), MetricSpec("nll")], "categorical", ignore_index=-100)
    by_probabilities = StreamingMetrics([MetricSpec("accuracy"), MetricSpec("nll")], "categorical", ignore_index=-100)
    assert by_logits.update_logits(torch.tensor(target), logits, valid=torch.ones(4, dtype=torch.bool)) == 2
    assert by_probabilities.update(target, probabilities=probabilities) == 2
    assert dict(by_logits.finalize().values) == pytest.approx(dict(by_probabilities.finalize().values))
    with pytest.raises(ValueError, match="integer class"):
        by_logits.update_logits(torch.tensor([0.5, 1.0]), logits[:2])
    readonly = np.array([0, 1])
    readonly.setflags(write=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "non-writable array" warning from torch
        assert by_probabilities.update(readonly, probabilities=probabilities[:2]) == 2


def test_review_round_four_declared_classes_bound_label_only_updates():
    acc = StreamingMetrics([MetricSpec("accuracy")], "categorical", labels=("a", "b"))
    with pytest.raises(ValueError, match=r"categorical targets must be class indices in \[0, 2\)"):
        acc.update(np.array([5, 7]), labels=np.array([5, 7]))
    with pytest.raises(ValueError, match=r"decoded labels must be class indices in \[0, 2\)"):
        acc.update(np.array([0, 1]), labels=np.array([0, 9]))
    assert acc.update(np.array([0, 1]), labels=np.array([0, 1])) == 2


def test_review_round_five_soft_targets_bernoulli_labels_and_shapes():
    soft = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])
    logits = torch.log(soft)
    counts = []
    for valid in (None, torch.ones(3, dtype=torch.bool)):
        acc = StreamingMetrics([MetricSpec("accuracy")], "categorical", ignore_index=0)
        counts.append(acc.update_logits(soft, logits, valid=valid))
    assert counts == [3, 3]  # soft rows are never dropped by ignore_index, a class-index rule
    with pytest.raises(ValueError, match="target rows"):
        StreamingMetrics([MetricSpec("accuracy")], "categorical").update_logits(soft, logits, valid=torch.ones(3, 2))
    readonly = soft.numpy().copy()
    readonly.setflags(write=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        StreamingMetrics([MetricSpec("accuracy")], "categorical").update_logits(readonly, logits)
    bernoulli = StreamingMetrics([MetricSpec("accuracy")], "bernoulli")
    with pytest.raises(ValueError, match="0/1 decisions"):
        bernoulli.update(np.array([1, 0]), labels=np.array([0.8, 0.3]))
    with pytest.raises(ValueError, match="valid must be shaped like the targets"):
        bernoulli.update(np.array([1, 0]), labels=np.array([1, 0]), valid=np.array([True]))


class _Failing(_Summing):
    def update(self, target, prediction):
        raise RuntimeError("metric failed")


def test_review_round_seven_shapes_label_ranges_and_partial_failures():
    regression = StreamingMetrics([MetricSpec("mae")], "continuous")
    with pytest.raises(ValueError, match="shaped like the targets"):
        regression.update_logits(torch.tensor([1.0]), torch.tensor([[0.3, 1.0, 5.0]]))  # never broadcast
    with pytest.raises(ValueError, match="shaped like the targets"):
        regression.update_logits(torch.zeros(2, 1), torch.zeros(2))
    with pytest.raises(ValueError, match="class indices shaped like the logits"):
        StreamingMetrics([MetricSpec("nll")], "categorical").update_logits(torch.tensor([[0, 1]]), torch.zeros(2, 2))
    f1 = StreamingMetrics([MetricSpec("f1")], "categorical")
    with pytest.raises(ValueError, match=r"decoded labels must be class indices in \[0, 3\)"):
        f1.update(np.array([0, 1]), labels=np.array([0, 7]), probabilities=np.full((2, 3), 1 / 3))
    with pytest.raises(ValueError, match=r"decoded labels must be class indices in \[0, C\)"):
        f1.update(np.array([0, 1]), labels=np.array([0, -2]))
    assert f1.count == 0
    register_metric("tests.failing", 1, lambda config: _Failing(), input="probabilities", mode="max")
    try:
        acc = StreamingMetrics([MetricSpec("brier"), MetricSpec("tests.failing")], "bernoulli")
        with pytest.raises(RuntimeError, match="metric failed"):
            acc.update(np.array([1, 0]), probabilities=np.array([0.9, 0.2]))
        for use in (
            acc.finalize,
            lambda: acc.merge(acc),
            lambda: acc.update(np.array([1]), probabilities=np.array([0.5])),
        ):
            with pytest.raises(RuntimeError, match="inconsistent"):  # brier took the batch, the failing metric did not
                use()
    finally:
        unregister_metric("tests.failing", 1)


def test_review_round_eight_class_counts_and_unsigned_labels():
    task = TaskSpec.categorical(3)
    with pytest.raises(ValueError, match="exactly 2 classes"):
        StreamingMetrics.for_task([MetricSpec("f1", config={"average": "binary"})], task)
    acc = StreamingMetrics.for_task([MetricSpec("accuracy")], task)
    assert acc.num_outputs == 3
    with pytest.raises(ValueError, match=r"class indices in \[0, 3\)"):
        acc.update(np.array([5, 7]), labels=np.array([5, 7]))
    with pytest.raises(MetricMergeError, match="output counts"):
        acc.merge(StreamingMetrics([MetricSpec("accuracy")], "categorical"))
    with pytest.raises(ValueError, match="2\\*\\*63"):
        MetricSpec("f1").accumulator().update(np.array([2**63], dtype=np.uint64), np.array([0], dtype=np.uint64))


def test_review_round_nine_output_widths_strides_and_sparse_labels():
    multilabel = StreamingMetrics.for_task([MetricSpec("brier")], TaskSpec.multilabel(3))
    assert multilabel.num_outputs == 3
    with pytest.raises(ValueError, match="3 are declared"):
        multilabel.update_logits(torch.zeros(2, 5), torch.zeros(2, 5))
    with pytest.raises(ValueError, match="3 are declared"):
        multilabel.update(np.zeros((2, 5)), probabilities=np.full((2, 5), 0.5))
    with pytest.raises(MetricMergeError, match="output counts"):
        multilabel.merge(StreamingMetrics.for_task([MetricSpec("brier")], TaskSpec.multilabel(5)))
    target = np.array([0, 1, 1, 0])
    logits = np.array([[2.0, 0.0], [0.0, 2.0], [0.0, 2.0], [2.0, 0.0]])
    reversed_view = StreamingMetrics([MetricSpec("accuracy")], "categorical")
    assert reversed_view.update_logits(target[::-1], logits[::-1]) == 4  # negative strides are copied, not refused
    wide = MetricSpec("f1").accumulator()
    labels = np.array([5, 300_000, 300_000, -3, 5])
    decided = np.array([5, 300_000, 5, -3, 300_000])
    wide.update(labels[:2], decided[:2])
    wide.update(labels[2:], decided[2:])  # large and negative ids use the sparse table
    assert wide.result() == pytest.approx(sk.f1_score(labels, decided, average="macro"))


def test_review_round_eleven_thresholds_only_split_bernoulli_merges():
    a = StreamingMetrics([MetricSpec("accuracy")], "categorical", threshold=0.7)
    b = StreamingMetrics([MetricSpec("accuracy")], "categorical")
    assert a.merge(b).count == 0  # the threshold decides nothing for categorical labels


def test_review_round_twelve_string_labels_and_undeclared_widths():
    f1 = MetricSpec("f1").accumulator()
    target, decided = np.array(["a", "b", "a"]), np.array(["a", "a", "a"])
    f1.update(target, decided)  # labels that are not class indices: scored by scikit-learn, as before
    assert f1.result() == pytest.approx(sk.f1_score(target, decided, average="macro", zero_division=0))
    with pytest.raises(ValueError, match="one kind"):
        mixed = MetricSpec("f1").accumulator()
        mixed.update(np.array([0, 1]), np.array([0, 1]))
        mixed.update(target, decided)
    counted = MetricSpec("f1").accumulator()
    counted.update(np.array([0, 1]), np.array([0, 1]))
    with pytest.raises(ValueError, match="cannot merge"):
        counted.merge(f1)
    three, five = (StreamingMetrics([MetricSpec("nll")], "categorical") for _ in range(2))
    three.update(np.array([0, 2]), probabilities=np.full((2, 3), 1 / 3))
    five.update(np.array([0, 4]), probabilities=np.full((2, 5), 0.2))
    with pytest.raises(MetricMergeError, match="different widths"):
        three.merge(five)
    with pytest.raises(ValueError, match="came in earlier batches"):
        three.update(np.array([0]), probabilities=np.full((1, 5), 0.2))


def test_review_round_thirteen_masked_soft_rows_seen_widths_and_stored_batches():
    soft = torch.tensor([[0.9, 0.1], [float("nan"), float("nan")], [0.2, 0.8]])
    logits = torch.log(torch.tensor([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]]))
    nll = StreamingMetrics([MetricSpec("nll")], "categorical")
    assert nll.update_logits(soft, logits) == 2  # a NaN soft row is masked, never scored
    without = StreamingMetrics([MetricSpec("nll")], "categorical")
    without.update_logits(soft[[0, 2]], logits[[0, 2]])
    assert nll.finalize().values == without.finalize().values  # as if the row were never there

    accuracy = StreamingMetrics([MetricSpec("accuracy")], "categorical")  # no declared width
    accuracy.update(np.array([0, 2]), probabilities=np.full((2, 3), 1 / 3))
    with pytest.raises(ValueError, match=r"class indices in \[0, 3\)"):
        accuracy.update(np.array([5]), labels=np.array([5]))  # the width earlier batches fixed

    empty = MetricSpec("f1").accumulator()
    empty.update(np.array([], dtype=str), np.array([], dtype=str))
    assert empty.result() is None  # nothing scored, as for integer labels

    class _Finite(_Scores):
        def update(self, target, prediction):
            if not np.isfinite(prediction).all():
                raise ValueError("scores must be finite")
            super().update(target, prediction)

    register_metric("tests.finite", 1, lambda config: _Finite(), input="probabilities", mode="max")
    try:
        stored = StreamingMetrics([MetricSpec("tests.finite")], "bernoulli", materialize=True)
        stored.update(np.array([0, 1]), probabilities=np.array([0.2, 0.7]))
        with pytest.raises(ValueError, match="scores must be finite"):
            stored.update(np.array([0, 1]), probabilities=np.array([np.nan, 0.7]))  # fails on its own batch
    finally:
        unregister_metric("tests.finite", 1)
