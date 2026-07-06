from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch_geometric.data import Data as PyGData
from torch_geometric.data import Dataset
from torch_geometric.loader import NeighborLoader

from ...seeding import dataloader_worker_init_fn
from .nn_dataset_base import NNDatasetBase


def _full_batch_loader(data: PyGData, split_mask: torch.Tensor) -> list[PyGData]:
    """Return a one-element list containing the full graph permuted so the
    split's seed nodes lead.

    The returned ``Data`` object mirrors the NeighborLoader batch contract
    that ``GraphNNBase.seed_count`` and ``NNModel._fwd_pass`` rely on:

    - ``.input_id``: original global indices of the split's nodes.
    - ``.batch_size``: number of split nodes.

    Only the leading ``batch_size`` rows are scored; the trailing rows
    supply message-passing context but are not counted in the loss or
    metrics.  GCN convolutions are permutation-invariant, so the output
    for each node is identical to the un-permuted graph.
    """
    num_nodes = int(data.num_nodes)
    idx = split_mask.nonzero(as_tuple=False).view(-1)
    rest = (~split_mask).nonzero(as_tuple=False).view(-1)
    perm = torch.cat([idx, rest])

    # inv[original_index] = position_in_permuted_graph
    inv = torch.empty(num_nodes, dtype=torch.long)
    inv[perm] = torch.arange(num_nodes, dtype=torch.long)

    edge_index = inv[data.edge_index]

    batch = PyGData(
        x=data.x[perm],
        edge_index=edge_index,
        y=data.y[perm],
    )
    batch.input_id = idx
    batch.batch_size = int(idx.numel())

    return [batch]


@dataclass(frozen=True, kw_only=True, slots=True)
class NNGraphDataset(NNDatasetBase):
    ds_class: type[Dataset]
    # n_neighbors is required for sampler="neighbor"; unused for "full".
    # Kept Optional so that full-batch callers need not supply a meaningless list.
    n_neighbors: Optional[list[int]] = None
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
    # Ignored for sampler="full" (no sampling randomness to control).
    seed: Optional[int] = None
    # "neighbor": NeighborLoader-based mini-batch sampling (default, today's
    #   behavior).  Requires n_neighbors.
    # "full": one-batch-per-epoch loader that carries the entire graph
    #   permuted so the split's seed nodes lead.  Does NOT require pyg-lib
    #   or torch-sparse — works on Apple Silicon without pre-built wheels.
    sampler: Literal["neighbor", "full"] = "neighbor"

    def __post_init__(self):
        if self.sampler == "neighbor" and self.n_neighbors is None:
            raise ValueError(
                "n_neighbors is required when sampler='neighbor'. "
                "Pass n_neighbors=[k1, k2, ...] or switch to sampler='full'."
            )

        dataset = self.ds_class(root=self.root_dir, transform=self.transform)
        # Single-graph datasets expose the underlying Data via dataset[0].
        # This replaces the historical private `dataset._data` access, which
        # was renamed/removed across PyG versions.
        data = dataset[0]

        object.__setattr__(self, "name", self.ds_class.__name__)

        train_batch_size = self.batch_sizes[0] or int(data.train_mask.sum())
        val_batch_size = self.batch_sizes[1] or int(data.val_mask.sum())
        test_batch_size = self.batch_sizes[2] or int(data.test_mask.sum())
        resolved_batch_sizes = (train_batch_size, val_batch_size, test_batch_size)

        object.__setattr__(self, "batch_sizes", resolved_batch_sizes)

        if self.sampler == "full":
            object.__setattr__(self, "train_loader", _full_batch_loader(data, data.train_mask))
            object.__setattr__(self, "val_loader", _full_batch_loader(data, data.val_mask))
            object.__setattr__(self, "test_loader", _full_batch_loader(data, data.test_mask))
        else:
            # seed=None must genuinely fall back to the global torch RNG (the
            # documented contract): a fresh torch.Generator() always carries the
            # same fixed default seed, which would make every unseeded run
            # bit-identical and deaf to torch.manual_seed.
            gen = torch.Generator().manual_seed(int(self.seed)) if self.seed is not None else torch.default_generator

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

        state: dict = dict(
            name=self.name,
            sampler=self.sampler,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            train_batch_size=f"{self.batch_sizes[0]:,}",
            val_batch_size=f"{self.batch_sizes[1]:,}",
            test_batch_size=f"{self.batch_sizes[2]:,}",
        )
        if self.sampler == "neighbor":
            state["n_workers"] = self.n_workers
            state["n_neighbors"] = self.n_neighbors

        object.__setattr__(self, "_state", state)
