"""FIX-022: dataset ``batch_sizes`` — ``None`` is the *only* automatic
full-split sentinel. Zero / ``False`` / negative / fractional / boolean /
string entries and malformed tuples fail with the wrapper and the
``batch_sizes[i] (split)`` slot named, before any dataset factory,
tokenizer or RNG-consuming split runs. Positive sizes are kept verbatim
(never clipped to the split), NumPy integers normalize to ``int``, and the
empty-split conventions (placeholder 1 + ``None`` loader) are unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch
from torchvision.datasets import VisionDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNDataset,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNPreferenceDataset,
    NNSchedulerParams,
    NNTabularDataset,
    NNTrainParams,
    Optims,
)


class _CountingVision(VisionDataset):
    """30 train / 12 test samples of shape (1, 4, 4), 3 classes; counts
    constructor calls so a test can prove validation ran first."""

    classes = ["a", "b", "c"]
    calls = 0

    def __init__(self, root, train=True, download=False, transform=None):
        super().__init__(root, transform=transform)
        type(self).calls += 1
        n = 30 if train else 12
        self.data = torch.randn(n, 1, 4, 4, generator=torch.Generator().manual_seed(0 if train else 1))
        self.targets = torch.arange(n) % 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], int(self.targets[idx])


class _CountingTokenizer:
    vocab_size = 16

    def __init__(self):
        self.calls = 0

    def encode(self, text: str) -> list[int]:
        self.calls += 1
        return [2 + (ord(ch) % 13) for ch in text.strip()][:8] or [2]


class _CountingGraph:
    """8 nodes / 5 features / 3 classes split 4/2/2; counts constructor calls."""

    num_features = 5
    num_classes = 3
    calls = 0

    def __init__(self, root, transform=None):
        from torch_geometric.data import Data

        type(self).calls += 1
        n = 8
        masks = [torch.zeros(n, dtype=torch.bool) for _ in range(3)]
        masks[0][:4], masks[1][4:6], masks[2][6:] = True, True, True
        self._data = Data(
            x=torch.randn(n, 5, generator=torch.Generator().manual_seed(0)),
            edge_index=torch.tensor([[0, 1, 2, 3, 0, 2], [1, 2, 3, 0, 2, 0]], dtype=torch.long),
            y=torch.arange(n) % 3,
            train_mask=masks[0],
            val_mask=masks[1],
            test_mask=masks[2],
        )

    def __getitem__(self, idx):
        return self._data


_FRAME = pd.DataFrame({"x": [1.0, 2.0], "y": [0, 1]})
_TABULAR = dict(df=_FRAME, feature_cols=["x"], target_col="y", val_proportion=0.0, test_proportion=0.0)
_TRIPLES = 10
KINDS = ["vision", "tabular", "preference", "graph"]
SPLITS = ["train", "val", "test"]
BAD = [
    pytest.param(0, id="zero"),
    pytest.param(False, id="false"),
    pytest.param(-1, id="negative"),
    pytest.param(0.5, id="fraction"),
    pytest.param(2.0, id="integral-float"),
    pytest.param(True, id="true"),
    pytest.param("2", id="string"),
    pytest.param(float("nan"), id="nan"),
]
MALFORMED = [
    pytest.param((None, None), id="2-tuple"),
    pytest.param((None, None, None, None), id="4-tuple"),
    pytest.param([None, None, None], id="list"),
    pytest.param(None, id="bare-none"),
    pytest.param(8, id="bare-int"),
]


def _build(kind: str, batch_sizes, tmp_path, *, counter=None):
    """Construct one wrapper; returns (dataset, calls) where ``calls`` is the
    number of factory / tokenizer invocations the construction made."""
    if kind == "vision":
        _CountingVision.calls = 0
        ds = NNDataset(
            ds_class=_CountingVision,
            root_dir=str(tmp_path),
            download=False,
            batch_sizes=batch_sizes,
            val_proportion=0.1,
            seed=0,
        )
        return ds, _CountingVision.calls
    if kind == "tabular":
        return NNTabularDataset(**_TABULAR, batch_sizes=batch_sizes), 0
    if kind == "preference":
        tok = _CountingTokenizer()
        ds = NNPreferenceDataset(
            prompts=["p"] * _TRIPLES,
            chosen=["c"] * _TRIPLES,
            rejected=["r"] * _TRIPLES,
            tokenizer=tok,
            batch_sizes=batch_sizes,
            val_proportion=0.1,
            test_proportion=0.1,
            seed=0,
        )
        return ds, tok.calls
    pytest.importorskip("torch_geometric")
    from nnx import NNGraphDataset

    _CountingGraph.calls = 0
    ds = NNGraphDataset(
        ds_class=_CountingGraph, n_neighbors=[2], n_workers=0, batch_sizes=batch_sizes, root_dir=str(tmp_path)
    )
    return ds, _CountingGraph.calls


def _calls(kind: str, tmp_path) -> int:
    if kind == "vision":
        return _CountingVision.calls
    if kind == "graph":
        return _CountingGraph.calls
    return 0


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("value", BAD)
def test_invalid_entries_fail_with_split_context_before_any_dataset_work(kind, position, value, tmp_path):
    batch_sizes = [None, None, None]
    batch_sizes[position] = value
    tok = _CountingTokenizer()
    _CountingVision.calls = _CountingGraph.calls = 0
    with pytest.raises(ValueError, match=rf"batch_sizes\[{position}\] \({SPLITS[position]}\)"):
        if kind == "preference":
            NNPreferenceDataset(
                prompts=["p"] * _TRIPLES,
                chosen=["c"] * _TRIPLES,
                rejected=["r"] * _TRIPLES,
                tokenizer=tok,
                batch_sizes=tuple(batch_sizes),
            )
        else:
            _build(kind, tuple(batch_sizes), tmp_path)
    assert tok.calls == 0 and _CountingVision.calls == 0 and _CountingGraph.calls == 0


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("batch_sizes", MALFORMED)
def test_malformed_tuples_fail_with_the_tuple_contract_named(kind, batch_sizes, tmp_path):
    tok = _CountingTokenizer()
    _CountingVision.calls = _CountingGraph.calls = 0
    with pytest.raises(ValueError, match=r"batch_sizes.*3-tuple"):
        if kind == "preference":
            NNPreferenceDataset(prompts=["p"], chosen=["c"], rejected=["r"], tokenizer=tok, batch_sizes=batch_sizes)
        else:
            _build(kind, batch_sizes, tmp_path)
    assert tok.calls == 0 and _CountingVision.calls == 0 and _CountingGraph.calls == 0


def test_ticket_oracle_tabular():
    with pytest.raises(ValueError, match=r"batch_sizes\[0\]"):
        NNTabularDataset(**_TABULAR, batch_sizes=(0, None, None))
    ds = NNTabularDataset(**_TABULAR, batch_sizes=(None, None, None))
    assert ds.batch_sizes == (2, 1, 1) and ds.val_loader is None


@pytest.mark.parametrize("kind", KINDS)
def test_none_resolves_to_the_full_split_and_positive_sizes_are_kept_verbatim(kind, tmp_path):
    auto, _ = _build(kind, (None, None, None), tmp_path)
    expected = {"vision": (27, 3, 12), "tabular": (2, 1, 1), "preference": (8, 1, 1), "graph": (4, 2, 2)}[kind]
    assert auto.batch_sizes == expected
    assert all(type(size) is int for size in auto.batch_sizes)
    assert len(auto.train_loader) == 1  # the #188 contract: one full-split batch → one step per epoch

    explicit, _ = _build(kind, (7, np.int64(5), None), tmp_path)
    assert explicit.batch_sizes == (7, 5, expected[2]) and type(explicit.batch_sizes[1]) is int
    n_train = expected[0]
    assert len(explicit.train_loader) == math.ceil(n_train / 7)
    if kind != "graph":  # NeighborLoader iteration needs pyg-lib / torch-sparse; its length does not
        first = next(iter(explicit.train_loader))
        assert first[0].shape[0] == min(7, n_train)  # larger-than-split → one smaller batch; metadata not clipped


def test_tabular_empty_split_conventions_are_unchanged_with_explicit_sizes():
    """Placeholder 1 + ``None`` loader for an empty optional split; an
    explicit positive size on a populated split is kept."""
    ds = NNTabularDataset(**_TABULAR, batch_sizes=(1, 3, 3))
    assert ds.batch_sizes == (1, 3, 3) and ds.val_loader is None and ds.test_loader is None
    frame = pd.DataFrame({"x": np.arange(10, dtype=float), "y": np.arange(10) % 2})
    ds = NNTabularDataset(
        df=frame,
        feature_cols=["x"],
        target_col="y",
        val_proportion=0.2,
        test_proportion=0.0,
        batch_sizes=(3, None, None),
        seed=0,
    )
    assert ds.batch_sizes == (3, 2, 1) and ds.val_loader is not None and ds.test_loader is None
    assert len(ds.train_loader) == 3 and len(ds.val_loader) == 1


def test_full_graph_sampler_still_rejects_explicit_sizes_after_validation(tmp_path):
    pytest.importorskip("torch_geometric")
    from nnx import NNGraphDataset

    with pytest.raises(ValueError, match="not supported when sampler='full'"):
        NNGraphDataset(ds_class=_CountingGraph, sampler="full", batch_sizes=(4, None, None), root_dir=str(tmp_path))
    _CountingGraph.calls = 0
    with pytest.raises(ValueError, match=r"batch_sizes\[0\] \(train\)"):
        NNGraphDataset(ds_class=_CountingGraph, sampler="full", batch_sizes=(0, None, None), root_dir=str(tmp_path))
    assert _CountingGraph.calls == 0
    ds = NNGraphDataset(ds_class=_CountingGraph, sampler="full", root_dir=str(tmp_path))
    assert ds.batch_sizes == (4, 2, 2)


def test_training_cadence_follows_the_resolved_train_loader(tmp_path, monkeypatch):
    """One optimizer step per epoch under the default (one full-split
    batch); ``batch_sizes=(3, None, None)`` on ten rows → four steps."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    frame = pd.DataFrame({"x": np.arange(10, dtype=float), "y": np.arange(10) % 2})

    def _train(batch_sizes, data_id):
        ds = NNTabularDataset(
            df=frame,
            feature_cols=["x"],
            target_col="y",
            val_proportion=0.0,
            test_proportion=0.0,
            batch_sizes=batch_sizes,
        )
        model = NNModel(
            net_params=NNParams(
                input_dim=1, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU
            ),
            params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        )
        run = model.train(
            params=NNTrainParams(
                n_epochs=2,
                train_loader=ds.train_loader,
                data_id=data_id,
                optim=NNOptimParams(name=Optims.SGD, max_lr=0.1, momentum=0.0, weight_decay=0.0),
                scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
            )
        )
        return ds, run

    ds, run = _train((None, None, None), "full")
    assert ds.batch_sizes[0] == 10 and len(ds.train_loader) == 1 and len(run.idps) == 2
    ds, run = _train((3, None, None), "mini")
    assert ds.batch_sizes[0] == 3 and len(ds.train_loader) == 4 and len(run.idps) == 8
