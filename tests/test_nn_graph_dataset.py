"""Behavioral coverage for NNGraphDataset — the torch_geometric facade.

Before this file, NNGraphDataset was only touched by an `hasattr`
assertion in test_imports.py: its loader construction, dim/class
introspection, and (newly) the `seed` reproducibility contract had zero
behavioral exercise. It was the lone dataset whose loaders both shuffle
AND spawn worker processes (`n_workers=4` by default) yet threaded
neither a `generator` nor a `worker_init_fn`, so neighbor sampling was
non-deterministic even after `nnx.set_seed(...)`.

A tiny in-memory graph stands in for the PyG download datasets so the
tests stay offline and fast. We only *construct* the loaders (never
iterate them), so the heavy `pyg-lib` / `torch-sparse` sampling backend
is not required — that keeps this test runnable wherever torch_geometric
itself imports.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch_geometric")

import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from nnx.nn.dataset.nn_graph_dataset import NNGraphDataset  # noqa: E402
from nnx.seeding import dataloader_worker_init_fn  # noqa: E402


class _TinyGraph:
    """Minimal stand-in honoring the (root, transform) constructor and the
    `dataset[0]` / `num_features` / `num_classes` surface NNGraphDataset
    drives. One 8-node graph, 5 features, 3 classes, split 4/2/2."""

    num_features = 5
    num_classes = 3

    def __init__(self, root, transform=None):
        self.transform = transform
        n = 8
        x = torch.randn(n, self.num_features, generator=torch.Generator().manual_seed(0))
        edge_index = torch.tensor([[0, 1, 2, 3, 0, 2], [1, 2, 3, 0, 2, 0]], dtype=torch.long)
        train_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[:4] = True
        val_mask = torch.zeros(n, dtype=torch.bool)
        val_mask[4:6] = True
        test_mask = torch.zeros(n, dtype=torch.bool)
        test_mask[6:] = True
        self._data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.arange(n) % self.num_classes,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

    def __getitem__(self, idx):
        return self._data


def test_nn_graph_dataset_dims_and_batch_resolution():
    """batch_sizes=None resolves each split to its full mask size; dims and
    name come from the underlying dataset."""
    ds = NNGraphDataset(ds_class=_TinyGraph, n_neighbors=[2])

    assert ds.name == "_TinyGraph"
    assert ds.input_dim == 5
    assert ds.output_dim == 3
    # None batch sizes resolve to the per-split mask counts (4 / 2 / 2).
    assert ds.batch_sizes == (4, 2, 2)


def test_nn_graph_dataset_seeded_loaders_are_deterministic():
    """A set `seed` pins the shuffle RandomSampler's generator (train loader)
    and the worker_init_fn on every loader, so neighbor sampling reproduces."""
    ds = NNGraphDataset(ds_class=_TinyGraph, n_neighbors=[2], seed=42)

    # The shuffling train loader carries the seeded generator.
    assert ds.train_loader.generator.initial_seed() == 42
    # Every loader pins worker RNG via the shared helper.
    for loader in (ds.train_loader, ds.val_loader, ds.test_loader):
        assert loader.worker_init_fn is dataloader_worker_init_fn


def test_nn_graph_dataset_seed_none_follows_global_rng():
    """`seed=None` falls back to the *global* torch RNG (torch.default_generator)
    — the pre-fix behavior — rather than a fresh fixed-seed generator. The
    worker_init_fn is still threaded so worker numpy/python RNG tracks the
    propagated torch base seed."""
    ds = NNGraphDataset(ds_class=_TinyGraph, n_neighbors=[2])

    for loader in (ds.train_loader, ds.val_loader, ds.test_loader):
        assert loader.generator is torch.default_generator
        assert loader.worker_init_fn is dataloader_worker_init_fn


def test_nn_graph_dataset_seed_not_serialized_into_state():
    """`seed` is reproducibility plumbing, not identity — it must NOT appear in
    state() (mirrors the sibling datasets, which omit it too)."""
    ds = NNGraphDataset(ds_class=_TinyGraph, n_neighbors=[2], seed=42)
    assert "seed" not in ds.state()


def test_nn_graph_dataset_rejects_unknown_sampler():
    with pytest.raises(ValueError, match="sampler"):
        NNGraphDataset(ds_class=_TinyGraph, sampler="typo")  # type: ignore[arg-type]


def test_nn_graph_dataset_rejects_batch_sizes_in_full_mode():
    with pytest.raises(ValueError, match="batch_sizes"):
        NNGraphDataset(ds_class=_TinyGraph, sampler="full", batch_sizes=(2, 2, 2))


# ---------------------------------------------------------------------------
# FIX-019: empty optional masks → absent loaders; empty train mask rejected
# ---------------------------------------------------------------------------


def _graph_class(*, train: list[int], val: list[int], test: list[int]):
    """A dataset class over ONE shared 8-node graph with the given split
    node lists, so a test can also prove the caller's masks are untouched."""
    n = 8
    masks = []
    for nodes in (train, val, test):
        mask = torch.zeros(n, dtype=torch.bool)
        mask[nodes] = True
        masks.append(mask)
    shared = Data(
        x=torch.randn(n, 5, generator=torch.Generator().manual_seed(1)),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long),
        y=torch.arange(n) % 3,
        train_mask=masks[0],
        val_mask=masks[1],
        test_mask=masks[2],
    )

    class _SplitGraph:
        num_features = 5
        num_classes = 3
        data = shared

        def __init__(self, root, transform=None):
            pass

        def __getitem__(self, idx):
            return self.data

    return _SplitGraph


def _mode_kwargs(sampler: str) -> dict:
    return dict(n_neighbors=[2], n_workers=0) if sampler == "neighbor" else {}


@pytest.mark.parametrize("sampler", ["neighbor", "full"])
@pytest.mark.parametrize(
    ("val", "test"),
    [([], [6, 7]), ([4, 5], []), ([], [])],
    ids=["empty-val", "empty-test", "both-empty"],
)
def test_empty_optional_masks_return_none(sampler, val, test):
    """An empty val/test mask yields loader None, resolved size 0 and state
    "0" in both sampler modes; the nonempty splits are unchanged and the
    caller's masks are never mutated. Neighbor loaders are only constructed
    and inspected (no pyg-lib / torch-sparse iteration)."""
    ds_class = _graph_class(train=[0, 1, 2, 3], val=val, test=test)
    ds = NNGraphDataset(ds_class=ds_class, sampler=sampler, **_mode_kwargs(sampler))

    assert (ds.val_loader is None) == (not val)
    assert (ds.test_loader is None) == (not test)
    assert ds.batch_sizes == (4, len(val), len(test))
    assert ds.state()["val_batch_size"] == str(len(val)) and ds.state()["test_batch_size"] == str(len(test))

    def _size(loader):
        return loader[0].batch_size if sampler == "full" else loader.batch_size

    assert _size(ds.train_loader) == 4
    if val:
        assert _size(ds.val_loader) == len(val)
    if test:
        assert _size(ds.test_loader) == len(test)
    if sampler == "full" and test:
        assert set(ds.test_loader[0].input_id.tolist()) == set(test)

    # An explicit positive size on an EMPTY neighbor split still yields an
    # absent loader (and derived size 0); populated splits keep the request.
    if sampler == "neighbor":
        requested = NNGraphDataset(
            ds_class=ds_class, sampler="neighbor", n_neighbors=[2], n_workers=0, batch_sizes=(2, 3, 3)
        )
        assert requested.train_loader.batch_size == 2
        assert requested.batch_sizes == (2, 3 if val else 0, 3 if test else 0)
        assert (requested.val_loader is None) == (not val) and (requested.test_loader is None) == (not test)

    data = ds_class.data
    assert (
        int(data.train_mask.sum()) == 4
        and int(data.val_mask.sum()) == len(val)
        and int(data.test_mask.sum()) == len(test)
    )


@pytest.mark.parametrize("sampler", ["neighbor", "full"])
def test_empty_train_mask_rejected(sampler):
    ds_class = _graph_class(train=[], val=[4], test=[6])
    with pytest.raises(ValueError, match="train_mask"):
        NNGraphDataset(ds_class=ds_class, sampler=sampler, **_mode_kwargs(sampler))
