from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from typing import Any, Optional, cast

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import VisionDataset

from ..._validation import require_batch_sizes
from ...preprocessing import SplitView, describe_transform
from .nn_dataset_base import NNDatasetBase


@dataclass(frozen=True, kw_only=True, slots=True)
class NNDataset(NNDatasetBase):
    """Vision dataset wrapper. `val_proportion` carves a validation slice
    out of the source `train=True` split (NOT out of the test split, which
    stays untouched for final evaluation).

    ``batch_sizes`` is a ``(train, val, test)`` tuple. ``None`` — the
    default for every slot — means *one batch holding the complete split*:
    the default train loader therefore yields a single full-split batch per
    epoch, i.e. **one optimizer step per epoch**. That suits small
    full-batch fits but not stochastic training (diffusion, MoE routing,
    self-supervision); pass an explicit train size such as
    ``batch_sizes=(128, None, None)`` for mini-batches. A positive integer
    is used verbatim (a size larger than the split simply yields one smaller
    batch); NumPy integers are normalized to ``int``. Zero, ``False``,
    negatives, floats, strings and anything but a 3-tuple are rejected with
    the ``batch_sizes[i] (split)`` slot named before ``ds_class`` is
    instantiated — zero never disables a split; ``val_proportion=0.0`` does
    (``val_loader`` is then ``None`` and the resolved val size is a
    placeholder ``1``).

    ``transform`` is handed to both torchvision factories and applies to
    every split. Split transforms (FEAT-018): ``train_transform`` and
    ``eval_transform`` add a per-split input transform on top of it — e.g. a
    random augmentation for training and a fixed resize for validation and
    test. Each split then becomes a ``nnx.preprocessing.SplitView`` over
    the untouched base dataset: targets pass through, the base
    ``transform`` / labels / order never change, and views are safe to read
    alternately and from DataLoader workers. ``input_dim`` is probed through
    ``eval_transform`` when one is set. Both default to ``None`` (the
    existing single-transform behavior, plain ``Subset`` splits); ``state()``
    gains ``train_transform`` / ``eval_transform`` descriptions only when
    set — runtime callables are recorded by name and must be passed again
    to rebuild the dataset.
    """

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
    # FEAT-018: per-split input transforms on top of `transform` (None keeps
    # the single shared transform and plain Subset splits).
    train_transform: Optional[Callable] = None
    eval_transform: Optional[Callable] = None

    def __post_init__(self):
        if not 0.0 <= self.val_proportion < 1.0:
            raise ValueError(f"val_proportion must be in [0, 1), got {self.val_proportion}")
        # Validate the request before the factories run (FIX-022): `None`
        # is the only full-split sentinel; zero / False / malformed tuples
        # fail here, not after a download or deep inside DataLoader.
        requested = require_batch_sizes(self.batch_sizes, owner="NNDataset")
        dataset_factory = cast(Any, self.ds_class)
        full_train_dataset, test_dataset = (
            dataset_factory(root=self.root_dir, train=True, download=self.download, transform=self.transform),
            dataset_factory(root=self.root_dir, train=False, download=self.download, transform=self.transform),
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
        split_views = self.train_transform is not None or self.eval_transform is not None
        if split_views:
            # Independent views over the untouched base datasets; the random
            # split above (and its RNG draw) is unchanged.
            train_dataset = SplitView(full_train_dataset, train_dataset.indices, transform=self.train_transform)
            val_dataset = SplitView(full_train_dataset, val_dataset.indices, transform=self.eval_transform)
            test_dataset = SplitView(test_dataset, range(len(test_dataset)), transform=self.eval_transform)

        object.__setattr__(self, "name", self.ds_class.__name__)

        # Fail fast on the no-transform PIL case: torchvision datasets
        # yield PIL Images without `transform`, and everything downstream
        # (input_dim inference, batching) needs tensors.
        sample = full_train_dataset[0][0]
        if self.eval_transform is not None:
            sample = self.eval_transform(sample)  # what the model sees at evaluation
        if not hasattr(sample, "shape"):
            raise ValueError(
                f"{self.ds_class.__name__} samples have no .shape (got {type(sample).__name__}) — "
                "pass transform=torchvision.transforms.ToTensor() (or a pipeline ending in it)."
            )

        # `None` → one batch holding the complete split; an explicit
        # positive size is kept verbatim (an `is None` test, not
        # truthiness — FIX-022). max(1, ...): an empty split would
        # otherwise resolve to DataLoader(batch_size=0), which raises.
        train_batch_size = len(train_dataset) if requested[0] is None else requested[0]
        val_batch_size = max(1, len(val_dataset)) if requested[1] is None else requested[1]
        test_batch_size = max(1, len(test_dataset)) if requested[2] is None else requested[2]
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

        state: dict[str, Any] = dict(
            name=self.name,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            train_batch_size=f"{self.batch_sizes[0]:,}",
            # 0 when the val split is empty (val_loader is None) — the
            # resolved batch_sizes carry a placeholder 1 there.
            val_batch_size=f"{self.batch_sizes[1] if len(val_dataset) > 0 else 0:,}",
            test_batch_size=f"{self.batch_sizes[2]:,}",
        )

        if split_views:
            if self.train_transform is not None:
                state["train_transform"] = describe_transform(self.train_transform)
            if self.eval_transform is not None:
                state["eval_transform"] = describe_transform(self.eval_transform)

        object.__setattr__(self, "_state", state)
