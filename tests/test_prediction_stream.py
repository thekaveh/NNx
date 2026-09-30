"""FEAT-020: bounded prediction streams (`NNModel.iter_predict`).

A stream yields one batch at a time, in loader order; concatenated, the
batches are exactly the eager `predict()` / `predict_proba()` result. Each
batch restores every submodule's training mode, closing drops the stream's
references, a closed stream refuses reuse, and the stream never holds more
than the batch in flight.
"""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNParams,
    PredictionResult,
    ProbabilitySpec,
    TaskSpec,
)
from nnx.streaming import (
    PredictionBatch,
    PredictionStream,
    StreamClosedError,
    concatenate_predictions,
)


def _model(output_dim: int = 3, loss: Losses = Losses.CROSS_ENTROPY, task=None, dropout: float = 0.25) -> NNModel:
    torch.manual_seed(0)
    return NNModel(
        net_params=NNParams(
            input_dim=4, output_dim=output_dim, hidden_dims=[8], dropout_prob=dropout, activation=Activations.RELU
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=loss, task=task),
    )


def _loader(n: int = 5, batch_size: int = 2, target_dim=None) -> DataLoader:
    generator = torch.Generator().manual_seed(1)
    X = torch.randn(n, 4, generator=generator)
    Y = torch.zeros(n, dtype=torch.long) if target_dim is None else torch.zeros(n, *target_dim)
    return DataLoader(TensorDataset(X, Y), batch_size=batch_size)


def _collect(stream: PredictionStream) -> list:
    with stream:
        return list(stream)


# --- equivalence with the eager calls -----------------------------------------------------------------


@pytest.mark.parametrize(
    "task",
    [
        pytest.param(None, id="legacy-classification"),
        pytest.param(TaskSpec.categorical(3, labels=("a", "b", "c")), id="categorical"),
    ],
)
def test_streamed_batches_concatenate_to_eager_predict_in_loader_order(task):
    model, loader = _model(task=task), _loader()
    eager = model.predict(loader)
    batches = _collect(model.iter_predict(loader))
    assert [len(batch) for batch in batches] == [2, 2, 1]  # the loader's own batches, in order
    assert all(isinstance(batch, PredictionBatch) for batch in batches)
    whole = concatenate_predictions(batches)
    assert isinstance(whole, PredictionBatch)
    np.testing.assert_array_equal(whole.logits, eager.logits)
    np.testing.assert_array_equal(whole.classes, eager.classes)
    np.testing.assert_array_equal(whole.sample_ids, np.arange(5))


def test_rich_batches_match_predict_proba_for_categorical_multilabel_and_explicit_specs():
    categorical = _model(task=TaskSpec.categorical(3, labels=("a", "b", "c")))
    loader = _loader()
    rich = _collect(categorical.iter_predict(loader, rich=True))
    assert [len(r.logits) for r in rich] == [2, 2, 1] and all(isinstance(r, PredictionResult) for r in rich)
    whole, eager = concatenate_predictions(rich), categorical.predict_proba(loader)
    assert isinstance(whole, PredictionResult) and whole.spec == eager.spec
    for field in ("logits", "probabilities", "decoded", "sample_ids"):
        np.testing.assert_array_equal(getattr(whole, field), getattr(eager, field))

    multilabel = _model(loss=Losses.BINARY_CROSS_ENTROPY, task=TaskSpec.multilabel(3, threshold=0.3))
    ml_loader = _loader(target_dim=(3,))
    whole = concatenate_predictions(_collect(multilabel.iter_predict(ml_loader, rich=True)))
    eager = multilabel.predict_proba(ml_loader)
    for field in ("logits", "probabilities", "decoded", "sample_ids"):
        np.testing.assert_array_equal(getattr(whole, field), getattr(eager, field))
    np.testing.assert_array_equal(
        concatenate_predictions(_collect(multilabel.iter_predict(ml_loader))).classes,  # type: ignore[union-attr]
        multilabel.predict(ml_loader).classes,  # the task's threshold, not logit >= 0
    )

    spec = ProbabilitySpec(kind="categorical", class_axis=1)
    legacy = _model()
    whole = concatenate_predictions(_collect(legacy.iter_predict(loader, spec)))
    eager = legacy.predict_proba(loader, spec)
    np.testing.assert_array_equal(whole.probabilities, eager.probabilities)  # type: ignore[union-attr]


@pytest.mark.parametrize("width", [1, 2], ids=["scalar", "two-targets"])
def test_regression_streams_continuous_values(width):
    model = _model(output_dim=width, loss=Losses.MEAN_SQUARED_ERROR, task=TaskSpec.regression(width))
    loader = _loader(n=4, target_dim=(width,))
    batches = _collect(model.iter_predict(loader))
    assert [batch.classes.shape for batch in batches] == [(2, width), (2, width)]  # the values, no argmax
    whole = concatenate_predictions(batches)
    np.testing.assert_array_equal(whole.classes, model.predict(loader).classes)  # type: ignore[union-attr]
    rich = concatenate_predictions(_collect(model.iter_predict(loader, rich=True)))
    assert isinstance(rich, PredictionResult) and rich.probabilities is None and rich.spec is None
    np.testing.assert_array_equal(rich.decoded, model.predict_proba(loader).decoded)


class _GraphLoader(DataLoader):
    """NeighborLoader-shaped batches (seed rows first) without torch_geometric's loader."""

    def __init__(self, batches):
        self._batches = batches
        super().__init__(dataset=[0])

    def __iter__(self):
        return iter(self._batches)


def test_graph_streams_keep_seed_rows_and_global_node_ids():
    torch.manual_seed(0)
    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.GRAPH_CONV, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )

    def subgraph(n_nodes: int, seeds: list[int]) -> SimpleNamespace:
        return SimpleNamespace(
            x=torch.randn(n_nodes, 4),
            edge_index=torch.tensor([list(range(n_nodes)), [(i + 1) % n_nodes for i in range(n_nodes)]]),
            y=torch.zeros(n_nodes, dtype=torch.long),
            batch_size=len(seeds),
            input_id=torch.arange(len(seeds)),  # the NeighborLoader marker; its ids are subgraph-local here
            n_id=torch.tensor(seeds + list(range(100, 100 + n_nodes - len(seeds)))),  # global ids, seeds first
        )

    loader = _GraphLoader([subgraph(6, [7, 3]), subgraph(5, [9, 1, 4])])
    batches = _collect(model.iter_predict(loader))
    assert [len(batch) for batch in batches] == [2, 3]  # seed rows only
    assert np.concatenate([b.sample_ids for b in batches]).tolist() == [7, 3, 9, 1, 4]  # global node ids
    eager = model.predict(loader)
    np.testing.assert_array_equal(concatenate_predictions(batches).logits, eager.logits)  # type: ignore[union-attr]


# --- lifecycle ------------------------------------------------------------------------------------------


def _mixed_modes(model: NNModel) -> list[bool]:
    model.net.train()
    child = next(module for module in model.net.modules() if isinstance(module, torch.nn.Linear))
    child.eval()  # a child in its own mode
    modes = [module.training for module in model.net.modules()]
    assert True in modes and False in modes
    return modes


def test_modes_are_restored_between_batches_and_after_an_early_close():
    model = _model()
    before = _mixed_modes(model)
    stream = model.iter_predict(_loader(n=6))
    with stream:
        first = next(stream)
        assert [module.training for module in model.net.modules()] == before  # between batches
        assert len(first) == 2
    assert stream.closed and [module.training for module in model.net.modules()] == before
    with pytest.raises(StopIteration):
        next(stream)  # a closed stream is exhausted, like a closed generator
    with pytest.raises(StreamClosedError, match="closed"):
        iter(stream)  # but it cannot be iterated again
    with pytest.raises(StreamClosedError):
        stream.__enter__()


def test_raising_mid_forward_restores_modes_and_closes_the_stream():
    model = _model()
    before = _mixed_modes(model)
    calls = {"n": 0}

    def fail_on_second(module, inputs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("forward failed")
        assert not module.training  # every forward runs in eval mode

    handle = model.net.register_forward_pre_hook(fail_on_second)
    stream = model.iter_predict(_loader(n=6))
    try:
        assert len(next(stream)) == 2
        with pytest.raises(RuntimeError, match="forward failed"):
            next(stream)
    finally:
        handle.remove()
    assert [module.training for module in model.net.modules()] == before
    assert stream.closed
    with pytest.raises(StreamClosedError):
        iter(stream)


class _OwnedLoader:
    """A caller-owned re-iterable loader whose iterators can be tracked."""

    def __init__(self, n_batches: int, batch_size: int = 2):
        generator = torch.Generator().manual_seed(3)
        self.batches = [
            (torch.randn(batch_size, 4, generator=generator), torch.zeros(batch_size, dtype=torch.long))
            for _ in range(n_batches)
        ]
        self.iterators: list[weakref.ref] = []

    def __iter__(self):
        iterator = _TrackedIterator(self.batches)
        self.iterators.append(weakref.ref(iterator))
        return iterator


class _TrackedIterator:
    def __init__(self, batches):
        self._batches = iter(batches)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._batches)


def test_closing_drops_stream_owned_references_and_leaves_the_loader_to_the_caller():
    model = _model()
    loader = _OwnedLoader(n_batches=4)
    stream = model.iter_predict(loader)
    next(stream)
    assert loader.iterators[0]() is not None  # the stream's iterator over the loader is alive
    stream.close()
    gc.collect()
    assert loader.iterators[0]() is None  # dropped on close
    assert len(list(model.iter_predict(loader))) == 4  # the loader itself is untouched and re-iterable
    consumed = model.iter_predict(loader)
    assert len(list(consumed)) == 4 and consumed.closed
    with pytest.raises(StopIteration):
        next(consumed)  # the iterator protocol: an exhausted stream stays exhausted
    with pytest.raises(StreamClosedError, match="consumed"):
        iter(consumed)  # but it cannot be iterated again


def test_empty_loaders_stream_nothing_while_eager_predict_raises():
    model = _model()
    empty = DataLoader(TensorDataset(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)), batch_size=2)
    assert _collect(model.iter_predict(empty)) == []
    with pytest.raises(ValueError, match="zero batches"):
        model.predict(empty)


def test_in_memory_inputs_and_bad_specs_are_rejected_before_any_batch():
    model = _model()
    for value in (np.zeros((3, 4), dtype=np.float32), torch.zeros(3, 4), (np.zeros((3, 4)),)):
        with pytest.raises(TypeError, match="predict"):
            model.iter_predict(value)
    with pytest.raises(TypeError, match="ProbabilitySpec"):
        model.iter_predict(_loader(), "categorical")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="needs a ProbabilitySpec"):
        model.iter_predict(_loader(), rich=True)  # no task, no spec


# --- bounded memory -------------------------------------------------------------------------------------


def _peak_retained(model: NNModel, n_batches: int, keep_every: int = 0) -> tuple[int, int]:
    """Stream ``n_batches`` batches with a consumer that discards each one
    (keeping every ``keep_every``-th if set); return the peak number of live
    streamed batches the consumer does not hold, and how many it holds."""
    loader = _OwnedLoader(n_batches)
    alive: list[weakref.ref] = []
    held: list[PredictionBatch] = []
    peak = 0
    with model.iter_predict(loader) as stream:
        for index, batch in enumerate(stream):
            alive.append(weakref.ref(batch.logits))
            if keep_every and index % keep_every == 0:
                held.append(batch)
            del batch  # reference counting frees a discarded batch at once; no collector pass needed
            held_ids = {id(b.logits) for b in held}
            peak = max(peak, sum(1 for ref in alive if ref() is not None and id(ref()) not in held_ids))
    return peak, len(held)


def test_retained_batches_stay_bounded_when_the_dataset_grows_tenfold():
    model = _model()
    small, _ = _peak_retained(model, 5)
    large, _ = _peak_retained(model, 50)
    assert small == large <= 1  # at most the batch in flight, whatever the dataset size
    held_peak, held = _peak_retained(model, 50, keep_every=10)
    assert held == 5 and held_peak <= 1  # batches the consumer keeps are the consumer's, not the stream's


def test_review_round_two_every_stream_warns_about_shuffled_sample_ids():
    model = _model()
    shuffled = DataLoader(
        TensorDataset(torch.randn(4, 4), torch.zeros(4, dtype=torch.long)), batch_size=2, shuffle=True
    )
    with pytest.warns(UserWarning, match="shuffling DataLoader"):
        stream = model.iter_predict(shuffled)  # plain batches carry sample_ids too
    stream.close()


def test_review_round_five_one_graph_is_not_a_stream_of_batches():
    class Data:  # iterates its (name, value) fields, as torch_geometric.data.Data does
        def __iter__(self):
            return iter([("x", torch.zeros(3, 4))])

    Data.__module__ = "torch_geometric.data.data"
    with pytest.raises(TypeError, match="one graph"):
        _model().iter_predict(Data())


def test_review_round_six_closing_inside_a_loop_ends_it():
    model = _model()
    seen = []
    with model.iter_predict(_loader(n=6)) as stream:
        for batch in stream:
            seen.append(len(batch))
            stream.close()  # stop early from inside the loop
    assert seen == [2] and stream.closed


def test_review_round_seven_a_failing_cleanup_still_closes_the_stream():
    def batches():
        try:
            yield PredictionBatch(np.zeros((1, 2)), np.zeros(1), np.zeros(1, dtype=np.int64))
            yield PredictionBatch(np.zeros((1, 2)), np.zeros(1), np.zeros(1, dtype=np.int64))
        finally:
            raise OSError("the loader's iterator failed to shut down")

    stream = PredictionStream(batches())
    next(stream)
    with pytest.raises(OSError, match="shut down"):
        stream.close()
    assert stream.closed
    with pytest.raises(StopIteration):
        next(stream)
    with pytest.raises(TypeError, match="wrap it in a DataLoader"):
        _model().iter_predict(np.zeros((2, 4), dtype=np.float32))
