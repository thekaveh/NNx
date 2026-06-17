from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import NeighborLoader

from ...seeding import dataloader_worker_init_fn
from .nn_dataset_base import NNDatasetBase


@dataclass(frozen=True, kw_only=True, slots=True)
class NNGraphDataset(NNDatasetBase):
    ds_class: type[Dataset]
    n_neighbors: list[int]
    root_dir: str = "./data"
    transform: Optional[Callable] = None
    n_workers: int = 4
    # Per-split batch size. None for any entry means "use every node in the
    # split mask" (resolved in __post_init__ from the train/val/test masks).
    batch_sizes: tuple[Optional[int], Optional[int], Optional[int]] = (None, None, None)
    # Deterministic neighbor sampling when set — the train loader shuffles
    # (RandomSampler reads `generator`) and the default `n_workers=4` spawns
    # worker processes whose numpy/python RNG must be pinned via
    # `dataloader_worker_init_fn`. Default None falls back to the global torch
    # RNG (the pre-fix behavior). Mirrors the seed contract the other datasets
    # (NNDataset / NNTabularDataset / NNPreferenceDataset) already expose.
    seed: Optional[int] = None

    def __post_init__(self):
        dataset = self.ds_class(root=self.root_dir, transform=self.transform)
        # Single-graph datasets expose the underlying Data via dataset[0].
        # This replaces the historical private `dataset._data` access, which
        # was renamed/removed across PyG versions.
        data = dataset[0]

        # seed=None must genuinely fall back to the global torch RNG (the
        # documented contract): a fresh torch.Generator() always carries the
        # same fixed default seed, which would make every unseeded run
        # bit-identical and deaf to torch.manual_seed.
        gen = torch.Generator().manual_seed(int(self.seed)) if self.seed is not None else torch.default_generator

        object.__setattr__(self, "name", self.ds_class.__name__)

        train_batch_size = self.batch_sizes[0] or int(data.train_mask.sum())
        val_batch_size = self.batch_sizes[1] or int(data.val_mask.sum())
        test_batch_size = self.batch_sizes[2] or int(data.test_mask.sum())
        resolved_batch_sizes = (train_batch_size, val_batch_size, test_batch_size)

        object.__setattr__(self, "batch_sizes", resolved_batch_sizes)

        object.__setattr__(
            self,
            "train_loader",
            NeighborLoader(
                shuffle=True,
                data=data,
                num_workers=self.n_workers,
                num_neighbors=self.n_neighbors,
                batch_size=resolved_batch_sizes[0],
                input_nodes=data.train_mask,
                generator=gen,
                worker_init_fn=dataloader_worker_init_fn,
            ),
        )

        object.__setattr__(
            self,
            "val_loader",
            NeighborLoader(
                shuffle=False,
                data=data,
                num_workers=self.n_workers,
                num_neighbors=self.n_neighbors,
                batch_size=resolved_batch_sizes[1],
                input_nodes=data.val_mask,
                generator=gen,
                worker_init_fn=dataloader_worker_init_fn,
            ),
        )

        object.__setattr__(
            self,
            "test_loader",
            NeighborLoader(
                shuffle=False,
                data=data,
                num_workers=self.n_workers,
                num_neighbors=self.n_neighbors,
                batch_size=resolved_batch_sizes[2],
                input_nodes=data.test_mask,
                generator=gen,
                worker_init_fn=dataloader_worker_init_fn,
            ),
        )

        object.__setattr__(self, "input_dim", dataset.num_features)

        object.__setattr__(self, "output_dim", dataset.num_classes)

        state = dict(
            name=self.name,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            train_batch_size=f"{self.batch_sizes[0]:,}",
            val_batch_size=f"{self.batch_sizes[1]:,}",
            test_batch_size=f"{self.batch_sizes[2]:,}",
            n_workers=self.n_workers,
            n_neighbors=self.n_neighbors,
        )

        object.__setattr__(self, "_state", state)
