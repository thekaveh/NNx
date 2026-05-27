from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from typing import Optional

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

    def __post_init__(self):
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
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

        object.__setattr__(self, "name", self.ds_class.__name__)

        train_batch_size = self.batch_sizes[0] or len(train_dataset)
        val_batch_size = self.batch_sizes[1] or len(val_dataset)
        test_batch_size = self.batch_sizes[2] or len(test_dataset)
        resolved_batch_sizes = (train_batch_size, val_batch_size, test_batch_size)

        object.__setattr__(self, "batch_sizes", resolved_batch_sizes)

        object.__setattr__(
            self, "train_loader", DataLoader(shuffle=True, dataset=train_dataset, batch_size=resolved_batch_sizes[0])
        )

        object.__setattr__(
            self, "val_loader", DataLoader(shuffle=False, dataset=val_dataset, batch_size=resolved_batch_sizes[1])
        )

        object.__setattr__(
            self, "test_loader", DataLoader(shuffle=False, dataset=test_dataset, batch_size=resolved_batch_sizes[2])
        )

        # train_loader.dataset is now a Subset (from random_split); shape/classes
        # come from the underlying full_train_dataset instead.
        object.__setattr__(self, "input_dim", reduce(lambda x, y: x * y, full_train_dataset[0][0].shape))

        object.__setattr__(self, "output_dim", len(full_train_dataset.classes))

        state = dict(
            name=self.name,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            train_batch_size=f"{self.batch_sizes[0]:,}",
            val_batch_size=f"{self.batch_sizes[1]:,}",
            test_batch_size=f"{self.batch_sizes[2]:,}",
        )

        object.__setattr__(self, "_state", state)
