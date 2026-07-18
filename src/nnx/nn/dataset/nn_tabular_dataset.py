"""Tabular dataset wrapper for pandas DataFrames.

Adapts a DataFrame of feature + target columns into the same
NNDatasetBase shape that NNDataset and NNGraphDataset already produce —
``train_loader / val_loader / test_loader``, ``input_dim``, ``output_dim``,
``name``, and a ``state()`` snapshot.

Usage:

    >>> df = pd.read_csv("data.csv")
    >>> ds = NNTabularDataset(
    ...     df=df,
    ...     feature_cols=["age", "income", "score"],
    ...     target_col="label",
    ...     batch_sizes=(64, 64, 64),
    ...     val_proportion=0.15,
    ...     test_proportion=0.15,
    ... )

Validation and test slices are random samples from the source DataFrame.
The remainder becomes train.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, cast

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from .nn_dataset_base import NNDatasetBase


@dataclass(frozen=True, kw_only=True, slots=True)
class NNTabularDataset(NNDatasetBase):
    """Wrap a pandas DataFrame as train/val/test DataLoaders.

    `feature_cols` columns are stacked into the input tensor; `target_col`
    is the integer label column. Targets are coerced to int64 (long), so
    use this for classification. For regression, prefer to construct the
    DataLoaders yourself and pass them through NNTrainParams.
    """

    df: pd.DataFrame
    feature_cols: list[str]
    target_col: str

    # Per-split batch size. None for any entry means "use the full split as
    # one batch" (resolved in __post_init__ once the split sizes are known).
    batch_sizes: tuple[Optional[int], Optional[int], Optional[int]] = (None, None, None)
    val_proportion: float = 0.15
    test_proportion: float = 0.15
    name_override: Optional[str] = None
    feature_dtype: torch.dtype = field(default=torch.float32)
    # Deterministic split when set — same `seed` + same `val_proportion` /
    # `test_proportion` round-trips to the same train/val/test ids across
    # runs. Default None falls back to the global torch RNG (the pre-fix
    # behavior). Mirrors NNPreferenceDataset's seeded-split contract.
    seed: Optional[int] = None

    def __post_init__(self):
        if not 0.0 <= self.val_proportion < 1.0:
            raise ValueError(f"val_proportion must be in [0, 1), got {self.val_proportion}")
        if not 0.0 <= self.test_proportion < 1.0:
            raise ValueError(f"test_proportion must be in [0, 1), got {self.test_proportion}")
        if self.val_proportion + self.test_proportion >= 1.0:
            raise ValueError(
                f"val_proportion + test_proportion must be < 1, got {self.val_proportion + self.test_proportion}"
            )

        # Validate columns up-front so missing-column errors point at user
        # input rather than failing deep inside torch.tensor with a KeyError.
        missing_features = [c for c in self.feature_cols if c not in self.df.columns]
        if missing_features:
            raise KeyError(f"NNTabularDataset feature_cols not in DataFrame: {missing_features}")
        if self.target_col not in self.df.columns:
            raise KeyError(f"NNTabularDataset target_col {self.target_col!r} not in DataFrame")
        if self.target_col in self.feature_cols:
            # Silent label leakage: the model would train on its own
            # target as an input feature and report near-perfect val
            # accuracy (classic feature_cols=list(df.columns) mistake).
            raise ValueError(
                f"target_col {self.target_col!r} must not appear in feature_cols — that trains on the label."
            )

        # NaN anywhere in the modeled columns is silent poison: NaN
        # features flow into NaN losses, and a NaN target's float→int64
        # cast is UNDEFINED (class 0 on ARM, INT64_MIN on x86 → CUDA
        # device assert) — and the contiguity check below can't see it
        # because pandas min/max/nunique skip NaN.
        # dict.fromkeys dedupes (order-preserving) duplicates WITHIN
        # feature_cols — target/feature overlap is rejected above.
        modeled = cast(pd.DataFrame, self.df[list(dict.fromkeys([*self.feature_cols, self.target_col]))])
        if bool(modeled.isna().to_numpy().any()):
            bad_cols = [c for c in modeled.columns if bool(modeled[c].isna().to_numpy().any())]
            raise ValueError(
                f"NaN values in columns {bad_cols} — drop or impute rows before constructing NNTabularDataset."
            )

        # Coerce features + target → tensors. Trust `dtype=` on torch.tensor
        # rather than going through an extra np.float32 intermediate copy.
        X = torch.tensor(
            self.df[self.feature_cols].to_numpy(),
            dtype=self.feature_dtype,
        )
        y = torch.tensor(
            self.df[self.target_col].to_numpy(),
            dtype=torch.long,
        )
        n_total = len(X)
        if n_total == 0:
            raise ValueError("NNTabularDataset requires a non-empty DataFrame")

        full_dataset = TensorDataset(X, y)

        # Sizes computed as (n_total - val - test, val, test) so the three
        # sum exactly even with int truncation.
        n_val = int(n_total * self.val_proportion)
        n_test = int(n_total * self.test_proportion)
        n_train = n_total - n_val - n_test
        # seed=None must genuinely fall back to the global torch RNG (the
        # documented contract): a fresh torch.Generator() is NOT that —
        # it always carries the same fixed default seed, which would make
        # every unseeded split bit-identical and deaf to torch.manual_seed.
        gen = torch.Generator().manual_seed(int(self.seed)) if self.seed is not None else torch.default_generator
        train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test], generator=gen)

        object.__setattr__(self, "name", self.name_override or "NNTabularDataset")

        train_batch_size = self.batch_sizes[0] or n_train
        val_batch_size = self.batch_sizes[1] or max(1, n_val)
        test_batch_size = self.batch_sizes[2] or max(1, n_test)
        resolved_batch_sizes = (train_batch_size, val_batch_size, test_batch_size)
        object.__setattr__(self, "batch_sizes", resolved_batch_sizes)

        object.__setattr__(
            self,
            "train_loader",
            DataLoader(train_ds, batch_size=resolved_batch_sizes[0], shuffle=True),
        )
        # Val/test loaders default to size 0 when the proportion is zero;
        # skip constructing them in that case and use None so callers can
        # check `ds.val_loader is None`.
        object.__setattr__(
            self,
            "val_loader",
            DataLoader(val_ds, batch_size=resolved_batch_sizes[1], shuffle=False) if n_val > 0 else None,
        )
        object.__setattr__(
            self,
            "test_loader",
            DataLoader(test_ds, batch_size=resolved_batch_sizes[2], shuffle=False) if n_test > 0 else None,
        )

        object.__setattr__(self, "input_dim", len(self.feature_cols))
        # Classification target dim = number of unique classes in the DF.
        # Labels must be contiguous 0..K-1: nunique() on e.g. {0, 5}
        # would size the model at 2 outputs and the mismatch only
        # surfaces much later inside cross-entropy as an opaque index /
        # device-side assert error. Fail fast with a fixable message.
        n_classes = int(self.df[self.target_col].nunique())
        target_min = int(self.df[self.target_col].min())
        target_max = int(self.df[self.target_col].max())
        if target_min != 0 or target_max != n_classes - 1:
            raise ValueError(
                f"target_col {self.target_col!r} labels must be contiguous integers 0..K-1; "
                f"got min={target_min}, max={target_max}, n_unique={n_classes}. "
                "Remap labels (e.g. pd.factorize) before constructing the dataset."
            )
        object.__setattr__(self, "output_dim", n_classes)

        object.__setattr__(
            self,
            "_state",
            dict(
                name=self.name,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                n_train=n_train,
                n_val=n_val,
                n_test=n_test,
                feature_cols=list(self.feature_cols),
                target_col=self.target_col,
            ),
        )
