from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Optional, cast

import torch
from torch_geometric.data import Data as PyGData
from torch_geometric.data import Dataset
from torch_geometric.loader import NeighborLoader

from ..._validation import require_batch_sizes
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
    if data.num_nodes is None or data.x is None or data.edge_index is None or data.y is None:
        raise ValueError("full-batch graph data requires num_nodes, x, edge_index, and y")
    num_nodes = int(data.num_nodes)
    idx = split_mask.nonzero(as_tuple=False).view(-1)
    rest = (~split_mask).nonzero(as_tuple=False).view(-1)
    perm = torch.cat([idx, rest])

    # inv[original_index] = position_in_permuted_graph
    inv = torch.empty(num_nodes, dtype=torch.long)
    inv[perm] = torch.arange(num_nodes, dtype=torch.long)

    edge_index = inv[cast(torch.Tensor, data.edge_index)]

    batch = PyGData(
        x=cast(torch.Tensor, data.x)[perm],
        edge_index=edge_index,
        y=cast(torch.Tensor, data.y)[perm],
    )
    batch.input_id = idx
    batch.batch_size = int(idx.numel())

    return [batch]


@dataclass(frozen=True, kw_only=True, slots=True)
class NNGraphDataset(NNDatasetBase):
    """Single-graph node-classification wrapper over a PyG dataset class.

    ``sampler="neighbor"`` (default) builds one ``NeighborLoader`` per split
    from the graph's ``train_mask`` / ``val_mask`` / ``test_mask``;
    ``sampler="full"`` yields the whole graph as one batch per split (no
    pyg-lib / torch-sparse needed).

    ``batch_sizes`` is a ``(train, val, test)`` tuple of *seed-node* counts
    for the neighbor sampler. ``None`` (the default for every slot) means
    *every node of the split mask in one batch* — one optimizer step per
    epoch for the default train loader; pass an explicit train size such as
    ``batch_sizes=(256, None, None)`` for mini-batches. A positive integer
    is kept verbatim; NumPy integers are normalized to ``int``. Zero,
    ``False``, negatives, floats, strings and anything but a 3-tuple are
    rejected with the ``batch_sizes[i] (split)`` slot named before
    ``ds_class`` is instantiated. ``sampler="full"`` rejects every explicit
    size (the complete split is always one batch).

    An empty ``val_mask`` / ``test_mask`` is an *absent* split in both
    sampler modes (FIX-019): that loader is ``None`` — the optional-loader
    contract the tabular / preference wrappers already follow, which
    ``NNModel.train`` / ``Trainer.train`` honour by skipping validation —
    its resolved size is ``0`` (derived from absence, never an accepted
    explicit zero) and ``state()`` reports ``"0"``. An empty ``train_mask``
    raises ``ValueError`` at construction. The caller's masks are only read.
    """

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
        if self.sampler not in ("neighbor", "full"):
            raise ValueError(f"sampler must be 'neighbor' or 'full', got {self.sampler!r}")
        # Validate the request before the dataset class is instantiated
        # (FIX-022): `None` is the only full-split sentinel, so an explicit
        # zero is rejected here rather than silently treated as "all nodes".
        requested = require_batch_sizes(self.batch_sizes, owner="NNGraphDataset")
        if self.sampler == "full" and any(size is not None for size in requested):
            raise ValueError("batch_sizes are not supported when sampler='full'; the complete split is one batch")
        if self.sampler == "neighbor" and self.n_neighbors is None:
            raise ValueError(
                "n_neighbors is required when sampler='neighbor'. "
                "Pass n_neighbors=[k1, k2, ...] or switch to sampler='full'."
            )

        dataset = self.ds_class(root=self.root_dir, transform=self.transform)
        # Single-graph datasets expose the underlying Data via dataset[0].
        # This replaces the historical private `dataset._data` access, which
        # was renamed/removed across PyG versions.
        data = cast(PyGData, dataset[0])

        object.__setattr__(self, "name", self.ds_class.__name__)

        # Split sizes are read from the masks once (the caller's graph is
        # never edited). An empty optional split is an ABSENT loader
        # (FIX-019) — not a zero-size batch that NeighborLoader rejects and
        # evaluate() cannot score — and an empty training split is an error.
        n_train, n_val, n_test = (
            int(cast(torch.Tensor, getattr(data, mask)).sum()) for mask in ("train_mask", "val_mask", "test_mask")
        )
        if n_train == 0:
            raise ValueError(
                f"{self.name}: train_mask selects no nodes — NNGraphDataset needs at least one training "
                "seed node (val_mask / test_mask may be empty, which yields an absent loader)"
            )
        # `None` → every node of the split mask in one batch; an explicit
        # positive size is kept verbatim — an `is None` test, not
        # truthiness (FIX-022). An empty optional split resolves to 0
        # whatever was requested: the zero is derived from absence (its
        # loader is None), never an accepted explicit zero.
        train_batch_size = n_train if requested[0] is None else requested[0]
        val_batch_size = 0 if n_val == 0 else (n_val if requested[1] is None else requested[1])
        test_batch_size = 0 if n_test == 0 else (n_test if requested[2] is None else requested[2])
        resolved_batch_sizes = (train_batch_size, val_batch_size, test_batch_size)

        object.__setattr__(self, "batch_sizes", resolved_batch_sizes)

        if self.sampler == "full":
            object.__setattr__(self, "train_loader", _full_batch_loader(data, data.train_mask))
            object.__setattr__(self, "val_loader", _full_batch_loader(data, data.val_mask) if n_val > 0 else None)
            object.__setattr__(self, "test_loader", _full_batch_loader(data, data.test_mask) if n_test > 0 else None)
        else:
            assert self.n_neighbors is not None
            n_neighbors = self.n_neighbors
            # seed=None must genuinely fall back to the global torch RNG (the
            # documented contract): a fresh torch.Generator() always carries the
            # same fixed default seed, which would make every unseeded run
            # bit-identical and deaf to torch.manual_seed.
            gen = torch.Generator().manual_seed(int(self.seed)) if self.seed is not None else torch.default_generator

            def neighbor_loader(mask: torch.Tensor, batch_size: int, *, shuffle: bool) -> NeighborLoader:
                return NeighborLoader(
                    shuffle=shuffle,
                    data=data,
                    num_workers=self.n_workers,
                    num_neighbors=n_neighbors,
                    batch_size=batch_size,
                    input_nodes=mask,
                    generator=gen,
                    worker_init_fn=dataloader_worker_init_fn,
                )

            object.__setattr__(self, "train_loader", neighbor_loader(data.train_mask, train_batch_size, shuffle=True))
            object.__setattr__(
                self, "val_loader", neighbor_loader(data.val_mask, val_batch_size, shuffle=False) if n_val > 0 else None
            )
            object.__setattr__(
                self,
                "test_loader",
                neighbor_loader(data.test_mask, test_batch_size, shuffle=False) if n_test > 0 else None,
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
