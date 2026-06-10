"""Behavioral coverage for NNDataset — the torchvision facade.

Before this file, NNDataset was only touched by an `hasattr` assertion
in test_imports.py: its split arithmetic, loader construction,
dim/class introspection, and the `seed` contract (added in PR #54) had
zero behavioral exercise. A tiny in-memory VisionDataset stands in for
the torchvision download classes so the tests stay offline and fast.
"""

from __future__ import annotations

import torch
from torchvision.datasets import VisionDataset

from nnx.nn.dataset.nn_dataset import NNDataset


class _TinyVision(VisionDataset):
    """Minimal VisionDataset honoring the (root, train, download,
    transform) constructor contract NNDataset drives. 30 train / 12
    test samples of shape (1, 4, 4) across 3 classes."""

    classes = ["a", "b", "c"]

    def __init__(self, root, train=True, download=False, transform=None):
        super().__init__(root, transform=transform)
        n = 30 if train else 12
        gen = torch.Generator().manual_seed(0 if train else 1)
        self.data = torch.randn(n, 1, 4, 4, generator=gen)
        self.targets = torch.arange(n) % 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.targets[idx])


def _split_indices(ds: NNDataset) -> tuple[list[int], list[int]]:
    return (
        sorted(ds.train_loader.dataset.indices),
        sorted(ds.val_loader.dataset.indices),
    )


def test_nn_dataset_splits_dims_and_batch_resolution(tmp_path):
    """val_proportion carves val out of the train split (test split
    untouched); batch_sizes=None resolves to full-split batches;
    input_dim/output_dim come from the underlying dataset."""
    ds = NNDataset(ds_class=_TinyVision, root_dir=str(tmp_path), download=False)

    # val = int(30 * 0.1) = 3, train = 27, test untouched at 12.
    assert len(ds.train_loader.dataset) == 27
    assert len(ds.val_loader.dataset) == 3
    assert len(ds.test_loader.dataset) == 12

    # (None, None, None) → one full-split batch per loader.
    assert ds.batch_sizes == (27, 3, 12)

    assert ds.input_dim == 1 * 4 * 4
    assert ds.output_dim == 3
    assert ds.name == "_TinyVision"

    # The loaders actually yield batches of the resolved size.
    X, y = next(iter(ds.train_loader))
    assert X.shape == (27, 1, 4, 4)
    assert y.shape == (27,)


def test_nn_dataset_seeded_split_is_deterministic(tmp_path):
    """Same seed → identical train/val membership; different seed →
    different membership (P(collision) = 1/C(30,3) ≈ 0.025% — fine)."""
    a = NNDataset(ds_class=_TinyVision, root_dir=str(tmp_path), download=False, seed=42)
    b = NNDataset(ds_class=_TinyVision, root_dir=str(tmp_path), download=False, seed=42)
    c = NNDataset(ds_class=_TinyVision, root_dir=str(tmp_path), download=False, seed=7)
    assert _split_indices(a) == _split_indices(b)
    assert _split_indices(a) != _split_indices(c)


def test_nn_dataset_seed_none_follows_global_rng(tmp_path):
    """`seed=None` falls back to the *global* torch RNG, so
    torch.manual_seed controls the split — the same contract the
    tabular/preference datasets pin in test_pass2_f_series.py."""
    torch.manual_seed(123)
    a = NNDataset(ds_class=_TinyVision, root_dir=str(tmp_path), download=False)
    torch.manual_seed(123)
    b = NNDataset(ds_class=_TinyVision, root_dir=str(tmp_path), download=False)
    torch.manual_seed(456)
    c = NNDataset(ds_class=_TinyVision, root_dir=str(tmp_path), download=False)
    assert _split_indices(a) == _split_indices(b)
    assert _split_indices(a) != _split_indices(c)
