from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import VisionDataset

from .nn_dataset_base import NNDatasetBase


@dataclass(frozen=True, kw_only=True, slots=True)
class NNDataset(NNDatasetBase):
    """Vision dataset wrapper. `val_proportion` carves a validation slice
    out of the source `train=True` split (NOT out of the test split, which
    stays untouched for final evaluation)."""

    ds_class: type[VisionDataset]
    root_dir: str = "./data"
    download: bool = True
    transform: Optional[Callable] = None
    # Per-split batch size. None for any entry means "use the full split as
    # one batch" (resolved in __post_init__ once the split sizes are known).
    batch_sizes: tuple[Optional[int], Optional[int], Optional[int]] = (None, None, None)
    val_proportion: float = 0.1
    # Deterministic split when set — same `seed` + same `val_proportion`
    # round-trips to the same train/val ids across runs. Default None falls
    # back to the global torch RNG (the pre-fix behavior). Mirrors the
    # NNPreferenceDataset contract that the seeded-split family already used.
    seed: Optional[int] = None

    def __post_init__(self):
        if not 0.0 <= self.val_proportion < 1.0:
            raise ValueError(f"val_proportion must be in [0, 1), got {self.val_proportion}")
        full_train_dataset, test_dataset = (
            self.ds_class(root=self.root_dir, train=True, download=self.download, transform=self.transform),
            self.ds_class(root=self.root_dir, train=False, download=self.download, transform=self.transform),
        )

        # Carve val out of train so the test set stays held-out for final eval.
        # Compute val_size first, derive train_size as the remainder so the two
        # sum exactly to len(full_train_dataset) (int truncation safe).
        full_train_len = len(full_train_dataset)
        val_size = int(full_train_len * self.val_proportion)
        train_size = full_train_len - val_size
        # seed=None must genuinely fall back to the global torch RNG (the
        # documented contract): a fresh torch.Generator() is NOT that —
        # it always carries the same fixed default seed, which would make
        # every unseeded split bit-identical and deaf to torch.manual_seed.
        gen = torch.Generator().manual_seed(int(self.seed)) if self.seed is not None else torch.default_generator
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=gen)

        object.__setattr__(self, "name", self.ds_class.__name__)

        # Fail fast on the no-transform PIL case: torchvision datasets
        # yield PIL Images without `transform`, and everything downstream
        # (input_dim inference, batching) needs tensors.
        sample = full_train_dataset[0][0]
        if not hasattr(sample, "shape"):
            raise ValueError(
                f"{self.ds_class.__name__} samples have no .shape (got {type(sample).__name__}) — "
                "pass transform=torchvision.transforms.ToTensor() (or a pipeline ending in it)."
            )

        train_batch_size = self.batch_sizes[0] or len(train_dataset)
        # max(1, ...): an empty split would otherwise resolve to
        # DataLoader(batch_size=0), which raises.
        val_batch_size = self.batch_sizes[1] or max(1, len(val_dataset))
        test_batch_size = self.batch_sizes[2] or max(1, len(test_dataset))
        resolved_batch_sizes = (train_batch_size, val_batch_size, test_batch_size)

        object.__setattr__(self, "batch_sizes", resolved_batch_sizes)

        object.__setattr__(
            self, "train_loader", DataLoader(shuffle=True, dataset=train_dataset, batch_size=resolved_batch_sizes[0])
        )

        # val_proportion=0.0 → val_loader=None, matching the tabular /
        # preference siblings' documented empty-split contract (an empty
        # DataLoader would instead make train() run a zero-sample
        # validate pass and crash at the end of the first epoch).
        object.__setattr__(
            self,
            "val_loader",
            DataLoader(shuffle=False, dataset=val_dataset, batch_size=resolved_batch_sizes[1])
            if len(val_dataset) > 0
            else None,
        )

        object.__setattr__(
            self, "test_loader", DataLoader(shuffle=False, dataset=test_dataset, batch_size=resolved_batch_sizes[2])
        )

        # train_loader.dataset is now a Subset (from random_split); shape/classes
        # come from the underlying full_train_dataset instead.
        object.__setattr__(self, "input_dim", reduce(lambda x, y: x * y, sample.shape))

        object.__setattr__(self, "output_dim", len(full_train_dataset.classes))

        state = dict(
            name=self.name,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            train_batch_size=f"{self.batch_sizes[0]:,}",
            # 0 when the val split is empty (val_loader is None) — the
            # resolved batch_sizes carry a placeholder 1 there.
            val_batch_size=f"{self.batch_sizes[1] if len(val_dataset) > 0 else 0:,}",
            test_batch_size=f"{self.batch_sizes[2]:,}",
        )

        object.__setattr__(self, "_state", state)
