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
