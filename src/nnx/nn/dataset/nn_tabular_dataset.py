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
The remainder becomes train. Pass ``split=`` (a ``SplitManifest`` from
``nnx.data_splits.plan_split``) for explicit group, chronological or
stratified membership instead:

    >>> from nnx.data_splits import plan_split
    >>> plan = plan_split(df["row_id"], strategy="group", groups=df["patient"],
    ...                   proportions=(0.7, 0.15, 0.15), seed=7)
    >>> ds = NNTabularDataset(df=df, feature_cols=["age", "income"], target_col="label",
    ...                       split=plan, id_col="row_id")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union, cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

from ..._validation import require_batch_sizes
from ...data_splits import SplitIndices, SplitManifest
from ...preprocessing import PreprocessingError, Standardizer, check_frame, dtype_name
from ...provenance import IdentityRef
from .nn_dataset_base import NNDatasetBase


def _feature_problem(column, dtype: torch.dtype) -> Optional[str]:
    """Why a selected feature column cannot be admitted under ``dtype``.

    Judged in source precision, before any conversion: ``"non-finite"`` for
    ``±inf`` (NaN is rejected earlier), ``"out-of-range"`` for finite values
    an integer ``dtype`` cannot hold (the cast would wrap them), else
    ``None``. Boolean columns always fit; a column that cannot be read as
    numbers (e.g. strings) is left to the tensor conversion, as before —
    this check adds no feature coercion.
    """
    values = column.to_numpy()
    kind = values.dtype.kind
    if kind == "b":
        return None
    if kind not in "iufc":
        try:
            values = column.to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError):
            return None
        kind = "f"
    if kind in "fc" and bool(np.isinf(values).any()):
        return "non-finite"
    if dtype.is_floating_point or dtype.is_complex or dtype == torch.bool or kind == "c" or values.size == 0:
        return None
    info = torch.iinfo(dtype)
    if kind in "iu":
        return None if info.min <= int(values.min()) and int(values.max()) <= info.max else "out-of-range"
    # Float → integer truncates toward zero; compare the truncated extremes.
    # ``float(info.max) + 1`` rounds to the exclusive upper bound exactly.
    lo, hi = float(np.trunc(values.min())), float(np.trunc(values.max()))
    return None if lo >= float(info.min) and hi < float(info.max) + 1 else "out-of-range"


_DEFAULT_VAL_PROPORTION = 0.15
_DEFAULT_TEST_PROPORTION = 0.15


@dataclass(frozen=True, kw_only=True, slots=True)
class NNTabularDataset(NNDatasetBase):
    """Wrap a pandas DataFrame as train/val/test DataLoaders.

    `feature_cols` columns are stacked into the input tensor; `target_col`
    is the target column. By default, targets are coerced to int64 (long)
    and validated as contiguous integer classes 0..K-1 (classification);
    the loaders yield 1-D class-index targets `(batch,)` (the
    `CrossEntropyLoss` convention). Set `target_dtype` to a floating-point
    dtype (e.g. `torch.float32`) to skip the integer cast and contiguity
    check and fix `output_dim=1` for regression; the loaders then yield
    targets of shape `(batch, 1)` so they line up with a model whose
    final linear layer has one output. Integer dtypes are rejected —
    leave `target_dtype` unset (`None`) for classification.

    ``batch_sizes`` is a ``(train, val, test)`` tuple. ``None`` (the
    default for every slot) means *one batch holding the complete split* —
    the default train loader yields one full-split batch per epoch, i.e.
    one optimizer step per epoch; pass an explicit train size such as
    ``batch_sizes=(64, None, None)`` for mini-batches. A positive integer
    is kept verbatim (larger than the split → one smaller batch); NumPy
    integers are normalized to ``int``. Zero, ``False``, negatives, floats,
    strings and anything but a 3-tuple are rejected with the
    ``batch_sizes[i] (split)`` slot named before any tensor conversion or
    split. Zero never disables a split: `val_proportion=0.0` /
    `test_proportion=0.0` do, and that split's loader is then ``None`` with
    a placeholder ``1`` in the resolved ``batch_sizes``.

    Admission: only the selected feature and target columns are inspected.
    NaN in any of them, ``±inf`` in a feature (checked in source precision,
    before the ``feature_dtype`` cast that could absorb it), finite features
    outside an integer ``feature_dtype``'s range (the cast would wrap them)
    and a non-finite target raise ``ValueError`` naming the columns and
    dtypes; so does a finite value that overflows a narrower floating /
    complex ``feature_dtype`` or floating ``target_dtype`` during conversion.
    All checks, the classification label check included, run before the
    split, so a rejection leaves the DataFrame and the global RNG untouched
    (the one exception: with ``standardize``, a standardized feature that
    overflows ``feature_dtype`` can only be detected after the split, once
    the training statistics exist).

    Explicit membership (FEAT-017): ``split=`` takes a ``SplitManifest``
    (``nnx.data_splits``) instead of the seeded ``random_split``. Rows are
    matched to the plan by the sample ids in the ``id_col`` column (required;
    never inferred from the index), so reordered rows keep their split; a
    positional manifest matches row positions and needs its recorded
    ``source_identity``. Duplicate, missing or unexpected ids, a changed
    ``source_identity`` and an empty train split raise ``SplitError`` before
    any tensor or loader exists. Only the planned rows are admitted and
    counted (``output_dim`` included): rows the plan excluded (a
    chronological ``gap``) may be absent and are never inspected. An empty
    validation or test membership gives a ``None`` loader. ``seed``,
    ``val_proportion`` and ``test_proportion`` do not apply and must be left
    at their defaults; the train loader still shuffles, and validation /
    test follow the manifest's order. ``state()`` gains ``split`` (the
    manifest digest) only then.

    Train-only standardization (FEAT-018): ``standardize=True`` fits a
    ``nnx.preprocessing.Standardizer`` on the training split's rows of the
    feature columns only — after the split, so validation and test values
    never reach the statistics — and applies those frozen statistics to
    every split. ``standardize=<fitted Standardizer>`` reuses one without
    refitting; its columns must equal ``feature_cols`` (same order) and its
    dtype ``feature_dtype``, checked before the split. Either way the
    standardizer in use is ``ds.standardizer`` (save it to serve the model).
    Standardizing needs a floating ``feature_dtype`` and unique
    ``feature_cols``; the raw features are never converted, targets are never
    transformed, and the split membership (and RNG draw) is the same as
    without standardization. ``state()`` gains ``standardizer`` (its digest)
    only then.
    """

    df: pd.DataFrame
    feature_cols: list[str]
    target_col: str

    # Per-split batch size. None for any entry means "use the full split as
    # one batch" (resolved in __post_init__ once the split sizes are known).
    batch_sizes: tuple[Optional[int], Optional[int], Optional[int]] = (None, None, None)
    val_proportion: float = _DEFAULT_VAL_PROPORTION
    test_proportion: float = _DEFAULT_TEST_PROPORTION
    name_override: Optional[str] = None
    feature_dtype: torch.dtype = field(default=torch.float32)
    # None (default) = classification: target cast to int64 and validated
    # as contiguous 0..K-1 classes (the existing contract). When set to a
    # floating-point dtype (e.g. torch.float32) = regression: target cast
    # to this dtype, the integer and contiguity checks are skipped, and
    # output_dim is fixed at 1. Integer dtypes are rejected in
    # __post_init__ (torch.long is what classification already casts to
    # internally, so passing it explicitly would silently switch modes).
    # Replaces the previous "build the DataLoaders yourself" workaround
    # for regression. Not part of _state (matches feature_dtype / seed
    # precedent — the dataset is not part of run.id).
    target_dtype: Optional[torch.dtype] = None
    # Deterministic split when set — same `seed` + same `val_proportion` /
    # `test_proportion` round-trips to the same train/val/test ids across
    # runs. Default None falls back to the global torch RNG (the pre-fix
    # behavior). Mirrors NNPreferenceDataset's seeded-split contract.
    seed: Optional[int] = None
    # FEAT-017: explicit membership from a SplitManifest instead of
    # random_split. None (default) keeps the seeded random split above.
    split: Optional[SplitManifest] = None
    # The column holding the plan's sample ids (required with a sample-id plan).
    id_col: Optional[str] = None
    # Must equal the manifest's recorded source identity, when it has one.
    source_identity: Optional[Union[IdentityRef, str]] = None
    # FEAT-018: False (default) keeps raw features, the existing behavior;
    # True fits a Standardizer on the training split only; a fitted
    # Standardizer is applied as is (never refitted).
    standardize: Union[bool, Standardizer] = False
    # The standardizer in use after construction (None for raw features).
    standardizer: Optional[Standardizer] = field(init=False, default=None)

    def __post_init__(self):
        if not 0.0 <= self.val_proportion < 1.0:
            raise ValueError(f"val_proportion must be in [0, 1), got {self.val_proportion}")
        if not 0.0 <= self.test_proportion < 1.0:
            raise ValueError(f"test_proportion must be in [0, 1), got {self.test_proportion}")
        if self.val_proportion + self.test_proportion >= 1.0:
            raise ValueError(
                f"val_proportion + test_proportion must be < 1, got {self.val_proportion + self.test_proportion}"
            )
        # Validate the batch-size request before any tensor conversion or
        # split (FIX-022): `None` is the only full-split sentinel.
        requested = require_batch_sizes(self.batch_sizes, owner="NNTabularDataset")
        # FEAT-018: every standardization check that needs no statistics runs
        # here, before any conversion or split.
        if not isinstance(self.standardize, (bool, Standardizer)):
            raise TypeError(
                f"standardize must be True, False or a fitted nnx.preprocessing.Standardizer, got {self.standardize!r}"
            )
        scaling = self.standardize is not False
        if scaling:
            if not self.feature_dtype.is_floating_point:
                raise ValueError(f"standardized features need a floating feature_dtype, got {self.feature_dtype}")
            if len(set(self.feature_cols)) != len(self.feature_cols):
                raise ValueError("standardize needs unique feature_cols (one statistic per column)")
            output_dtype = dtype_name(self.feature_dtype)
            if isinstance(self.standardize, Standardizer):
                supplied = self.standardize
                if supplied.n_features != len(self.feature_cols):
                    raise PreprocessingError(
                        f"the standardizer has {supplied.n_features} features but feature_cols has "
                        f"{len(self.feature_cols)}"
                    )
                if supplied.columns is not None and list(supplied.columns) != list(self.feature_cols):
                    raise PreprocessingError(
                        f"the standardizer's columns {list(supplied.columns)} differ from feature_cols "
                        f"{list(self.feature_cols)}; pass feature_cols in the fitted order"
                    )
                if supplied.dtype != output_dtype:
                    raise ValueError(
                        f"the standardizer's dtype {supplied.dtype!r} differs from feature_dtype={self.feature_dtype}"
                    )
        # target_dtype is a tri-state: None = classification (the existing
        # contract), a floating-point dtype = regression. An integer dtype
        # is rejected because it's an unambiguous footgun: torch.long is
        # what classification already casts to internally, so passing it
        # explicitly would silently switch modes (skip the contiguity
        # check, force output_dim=1) — the exact opposite of what the
        # caller asked for. Fail fast with a fixable message.
        if self.target_dtype is not None and not self.target_dtype.is_floating_point:
            raise ValueError(
                f"target_dtype must be a floating-point dtype for regression "
                f"(e.g. torch.float32); got {self.target_dtype}. "
                "For classification, leave target_dtype unset (None)."
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
        # FEAT-017: resolve an explicit split before any conversion, so an id
        # or identity problem fails fast and no loader is ever built. Only the
        # planned rows are admitted, in split order: rows the plan excluded
        # (a chronological gap) are never inspected, converted or counted.
        indices = self._resolve_split()
        frame = self.df
        if indices is not None:
            frame = self.df.iloc[[*indices.train, *indices.validation, *indices.test]]

        # NaN anywhere in the modeled columns is silent poison: NaN
        # features flow into NaN losses, and a NaN target's float→int64
        # cast is UNDEFINED (class 0 on ARM, INT64_MIN on x86 → CUDA
        # device assert) — and the contiguity check below can't see it
        # because pandas min/max/nunique skip NaN.
        # dict.fromkeys dedupes (order-preserving) duplicates WITHIN
        # feature_cols — target/feature overlap is rejected above.
        modeled = cast(pd.DataFrame, frame[list(dict.fromkeys([*self.feature_cols, self.target_col]))])
        target_kind = self.target_dtype if self.target_dtype is not None else "int64 class labels"
        if bool(modeled.isna().to_numpy().any()):
            bad_cols = [c for c in modeled.columns if bool(modeled[c].isna().to_numpy().any())]
            raise ValueError(
                f"NaN values in columns {bad_cols} (feature_dtype={self.feature_dtype}, "
                f"target_dtype={target_kind}) — drop or impute rows before constructing NNTabularDataset."
            )
        # FIX-023: ``isna`` is False for ±inf, and the dtype cast below would
        # silently turn an infinite or out-of-range feature into garbage
        # (int64 → an INT64 extreme, bool → True, int8 wraps 300 to 44).
        # Check the selected feature columns in source precision, before
        # any conversion; unselected columns are never inspected.
        problems = {c: _feature_problem(frame[c], self.feature_dtype) for c in dict.fromkeys(self.feature_cols)}
        non_finite = [c for c, problem in problems.items() if problem == "non-finite"]
        if non_finite:
            raise ValueError(
                f"feature columns {non_finite} contain non-finite values (±inf); non-finite features are "
                f"never admitted (feature_dtype={self.feature_dtype}) — drop or impute rows before "
                "constructing NNTabularDataset."
            )
        out_of_range = [c for c, problem in problems.items() if problem == "out-of-range"]
        if out_of_range:
            info = torch.iinfo(self.feature_dtype)
            raise ValueError(
                f"feature columns {out_of_range} hold finite values outside the range of "
                f"feature_dtype={self.feature_dtype} ({info.min}..{info.max}); the conversion would wrap "
                "them — use a wider feature_dtype or rescale the columns."
            )

        # Coerce the target once, then use the same numeric values for
        # validation, tensor construction, and classification metadata.
        # Reading the raw column again after pd.to_numeric would accept
        # numeric strings here but fail later inside torch.tensor.
        target_series = cast(pd.Series, pd.to_numeric(frame[self.target_col], errors="coerce"))
        # ascontiguousarray: a reversed / strided frame (df.iloc[::-1]) hands
        # NumPy negative strides, which torch.tensor rejects.
        target_values = np.ascontiguousarray(target_series.to_numpy())
        # Finiteness is required for both classification and regression —
        # NaN/Inf targets are poison either way. The integer check is
        # classification-only: a regression target (float) is expected to
        # have non-integral values.
        if not bool(np.isfinite(target_values).all()):
            raise ValueError(
                f"target_col {self.target_col!r} contains non-finite values (NaN/Inf) "
                f"(target_dtype={target_kind}); drop or impute rows before constructing the dataset."
            )
        if self.target_dtype is None and not bool(np.equal(target_values, np.floor(target_values)).all()):
            raise ValueError(
                f"target_col {self.target_col!r} labels must be finite integers for classification; "
                "factorize categorical labels, or set target_dtype for regression."
            )

        # Classification: labels must be contiguous 0..K-1. nunique() on
        # e.g. {0, 5} would size the model at 2 outputs and the mismatch
        # only surfaces much later inside cross-entropy as an opaque index /
        # device-side assert. Fail fast with a fixable message — before the
        # split, so a rejection consumes no RNG.
        n_classes = int(np.unique(target_values).size)
        if self.target_dtype is None and target_values.size:
            target_min = int(target_values.min())
            target_max = int(target_values.max())
            if target_min != 0 or target_max != n_classes - 1:
                raise ValueError(
                    f"target_col {self.target_col!r} labels must be contiguous integers 0..K-1; "
                    f"got min={target_min}, max={target_max}, n_unique={n_classes}. "
                    "Remap labels (e.g. pd.factorize) before constructing the dataset."
                )

        features = cast(pd.DataFrame, frame[self.feature_cols])
        if scaling:
            check_frame(features, self.feature_cols)  # schema problems surface before the split
        # With standardization the raw features are never converted: X is
        # built from the training statistics after the split.
        X = None if scaling else torch.tensor(np.ascontiguousarray(features.to_numpy()), dtype=self.feature_dtype)
        y = torch.tensor(
            target_values,
            dtype=self.target_dtype if self.target_dtype is not None else torch.long,
        )
        # Regression: promote the 1-D target `(n,)` to `(n, 1)` so it lines
        # up with `output_dim=1` (a model whose final linear layer has one
        # output emits `(batch, 1)` predictions). Leaving `y` 1-D would
        # silently trigger right-aligned broadcasting inside MSELoss:
        # `(batch, 1)` vs `(batch,)` → `(batch, batch)` pairwise diffs
        # averaged into a scalar — a meaningless loss with no error.
        # Classification stays 1-D `(n,)` because that's the CrossEntropyLoss
        # convention (class indices, not one-hot).
        # A finite source can still overflow a narrower floating dtype (e.g.
        # 1e5 → float16 inf). Checked after conversion and before the split,
        # so a rejection consumes no RNG and leaves nothing half-built.
        if (
            X is not None
            and (X.is_floating_point() or X.is_complex())
            and X.numel()
            and not bool(torch.isfinite(X).all())
        ):
            finite_per_column = torch.isfinite(X).all(dim=0).tolist()
            # ``features.columns`` matches X column for column, duplicate labels included.
            overflowed = list(
                dict.fromkeys(c for c, ok in zip(features.columns, finite_per_column, strict=True) if not ok)
            )
            raise ValueError(
                f"feature columns {overflowed} overflow feature_dtype={self.feature_dtype}: finite source "
                "values become ±inf after conversion — use a wider feature_dtype or rescale the columns."
            )
        if y.is_floating_point() and y.numel() and not bool(torch.isfinite(y).all()):
            raise ValueError(
                f"target_col {self.target_col!r} overflows target_dtype={self.target_dtype}: finite source "
                "values become ±inf after conversion — use a wider target_dtype or rescale the target."
            )
        if self.target_dtype is not None:
            y = y.unsqueeze(-1)
        n_total = len(y)
        if n_total == 0:
            raise ValueError("NNTabularDataset requires a non-empty DataFrame")

        # Standardizing splits positions first (the same RNG draw), then
        # builds the features from the training statistics.
        full_dataset: Any = TensorDataset(X, y) if X is not None else range(n_total)

        if indices is not None:
            # The manifest fixes membership: no random_split, no RNG draw.
            # `frame` holds the train, validation and test rows in that order.
            n_train, n_val, n_test = len(indices.train), len(indices.validation), len(indices.test)
            train_ds, val_ds, test_ds = (
                Subset(full_dataset, range(0, n_train)),
                Subset(full_dataset, range(n_train, n_train + n_val)),
                Subset(full_dataset, range(n_train + n_val, n_total)),
            )
        else:
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

        if scaling:
            # FEAT-018: statistics come from the training rows of the feature
            # columns only; the same frozen standardizer then prepares every
            # split (transform raises if a value overflows feature_dtype).
            members = [np.asarray(train_ds.indices), np.asarray(val_ds.indices), np.asarray(test_ds.indices)]
            fitted = (
                self.standardize
                if isinstance(self.standardize, Standardizer)
                else Standardizer.fit(
                    features,
                    rows=members[0],
                    columns=list(self.feature_cols),
                    dtype=dtype_name(self.feature_dtype),
                    membership=self.split,
                )
            )
            full_dataset = TensorDataset(fitted.transform(features), y)
            train_ds, val_ds, test_ds = (Subset(full_dataset, rows.tolist()) for rows in members)
            object.__setattr__(self, "standardizer", fitted)

        object.__setattr__(self, "name", self.name_override or "NNTabularDataset")

        # `None` → one batch holding the complete split (placeholder 1 for
        # an empty optional split); an explicit positive size is kept
        # verbatim — an `is None` test, not truthiness (FIX-022).
        train_batch_size = n_train if requested[0] is None else requested[0]
        val_batch_size = max(1, n_val) if requested[1] is None else requested[1]
        test_batch_size = max(1, n_test) if requested[2] is None else requested[2]
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
        if self.target_dtype is None:
            # Classification: labels validated as contiguous before the split.
            object.__setattr__(self, "output_dim", n_classes)
        else:
            # Regression: single continuous output.
            object.__setattr__(self, "output_dim", 1)

        state = dict(
            name=self.name,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            feature_cols=list(self.feature_cols),
            target_col=self.target_col,
        )
        if self.split is not None:
            state["split"] = self.split.digest()  # omitted for the default random split
        if self.standardizer is not None:
            state["standardizer"] = self.standardizer.digest()  # omitted for raw features
        object.__setattr__(self, "_state", state)

    def _resolve_split(self) -> Optional[SplitIndices]:
        """Row positions of each split under ``split=``, or ``None`` for the
        default random split. Raises before any tensor or loader exists."""
        if self.split is None:
            if self.id_col is not None or self.source_identity is not None:
                raise ValueError(
                    "id_col / source_identity only apply with split= (a SplitManifest from nnx.data_splits)"
                )
            return None
        if not isinstance(self.split, SplitManifest):
            raise TypeError(f"split must be a nnx.data_splits.SplitManifest, got {type(self.split).__name__}")
        if self.seed is not None:
            raise ValueError("seed does not apply with split=: the manifest fixes membership, nothing is drawn")
        if (self.val_proportion, self.test_proportion) != (_DEFAULT_VAL_PROPORTION, _DEFAULT_TEST_PROPORTION):
            raise ValueError(
                "val_proportion / test_proportion do not apply with split=: the manifest fixes each split's rows"
            )
        if self.split.ids == "position":
            if self.id_col is not None:
                raise ValueError("a positional manifest matches row positions; id_col does not apply")
            ids: list = list(range(len(self.df)))
        else:
            # Required, never inferred from the index: a plan made on a column
            # would silently match the wrong rows through a same-valued index.
            if self.id_col is None:
                raise ValueError("split= needs id_col=: name the column holding the plan's sample ids")
            if self.id_col not in self.df.columns:
                raise KeyError(f"NNTabularDataset id_col {self.id_col!r} not in DataFrame")
            column = self.df[self.id_col]
            if isinstance(column, pd.DataFrame):
                raise ValueError(f"id_col {self.id_col!r} names {column.shape[1]} DataFrame columns; it must name one")
            ids = column.tolist()
        return self.split.resolve(ids, source=self.source_identity)  # raises on an empty train split
