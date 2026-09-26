"""Train-fitted preprocessing and split transforms (FEAT-018).

Two pieces keep held-out data out of what a model learns from:

- :class:`Standardizer` — per-column ``(x - mean) / scale``, fitted on an
  explicit training membership (``rows=``) and never reading another row.
  The statistics are float64 and frozen: validation, test and inference all
  reuse them under the declared schema (column names and order, a numeric
  input policy and an output dtype). A constant training column gets scale
  1. Statistics and schema serialize as primitive JSON (``to_json`` /
  ``save`` / ``load``) and reload exactly, so a model trained on prepared
  features is served by reloading the standardizer — never by refitting.
- :class:`SplitView` — a split of a base dataset with its own input
  transform (a random augmentation for training, a fixed one for
  evaluation). Targets pass through unchanged, and the base dataset — its
  own ``transform``, labels and order — is never modified, so views can be
  read alternately and from DataLoader workers.

``NNTabularDataset(standardize=True)`` fits on its training split (random or
``split=`` manifest) and ``NNDataset(train_transform=..., eval_transform=...)``
builds split views. ``NNModel.predict`` takes *prepared* features: apply the
reloaded standardizer exactly once to raw rows before calling it.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .provenance import IdentityRef, _ref, canonical_bytes

__all__ = ["FORMAT", "PreprocessingError", "SplitView", "Standardizer", "describe_transform"]

FORMAT = "nnx.preprocessing/1"
"""Format version of a serialized standardizer (part of its digest)."""

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}

ColumnLabel = Union[str, int]


class PreprocessingError(ValueError):
    """Preprocessing that cannot be fitted, loaded or applied as declared."""


def dtype_name(dtype: torch.dtype) -> str:
    """The standardizer ``dtype`` name of a floating torch dtype."""
    for name, value in _DTYPES.items():
        if value == dtype:
            return name
    raise PreprocessingError(f"standardized features need a floating dtype ({', '.join(_DTYPES)}), got {dtype}")


# --- reading rows ------------------------------------------------------------------------------


def _labels(columns: Any, what: str) -> tuple[ColumnLabel, ...]:
    """Schema labels: an ordered collection of unique ``str`` / ``int``
    labels (NumPy integers normalized) — what JSON round-trips exactly."""
    if isinstance(columns, (str, bytes, Mapping, Set)) or not isinstance(columns, Iterable):
        raise PreprocessingError(f"{what} must be an ordered list of column labels, got {columns!r}")
    out: list[ColumnLabel] = []
    for label in columns:
        if isinstance(label, str) and label:
            out.append(label)
        elif isinstance(label, numbers.Integral) and not isinstance(label, (bool, np.bool_)):
            out.append(int(label))
        else:
            raise PreprocessingError(
                f"{what} must be unique str or int labels (a JSON-serializable schema), got {label!r}; "
                "rename the columns"
            )
    if len(set(out)) != len(out):
        raise PreprocessingError(f"{what} must be unique, got {out}")
    return tuple(out)


def _column_positions(frame: Any, columns: Sequence[ColumnLabel], what: str) -> list[int]:
    """Positions of the schema columns, each present exactly once (other
    columns, duplicated or not, are ignored). Linear in the column count."""
    where: dict[Any, list[int]] = {}
    for position, label in enumerate(frame.columns):
        where.setdefault(label, []).append(position)
    missing = [column for column in columns if column not in where]
    if missing:
        raise PreprocessingError(f"{what} is missing columns {missing} of the fitted schema {list(columns)}")
    repeated = [column for column in columns if len(where[column]) > 1]
    if repeated:
        raise PreprocessingError(f"{what} holds the schema columns {repeated} more than once")
    return [where[column][0] for column in columns]


def _numeric_frame(frame: Any, columns: Sequence[Any], what: str) -> np.ndarray:
    """A float64 copy of an all-numeric frame (never a view of the caller's data)."""
    bad = [column for column, dtype in zip(columns, frame.dtypes, strict=True) if dtype.kind not in "biuf"]
    if bad:
        raise PreprocessingError(f"{what} columns {bad} must be numeric (bool, int or float)")
    return frame.to_numpy(dtype=np.float64, copy=True)


def check_frame(frame: Any, columns: Sequence[Any]) -> None:
    """Raise :class:`PreprocessingError` unless ``frame[columns]`` can be
    standardized: str / int labels, each present once, numeric dtypes. Lets
    a caller reject a frame before drawing a split."""
    labels = _labels(columns, "columns")
    positions = _column_positions(frame, labels, "the frame")
    bad = [
        label
        for label, position in zip(labels, positions, strict=True)
        if frame.dtypes.iloc[position].kind not in "biuf"
    ]
    if bad:
        raise PreprocessingError(f"the frame's columns {bad} must be numeric (bool, int or float)")


def _matrix(data: Any, what: str) -> np.ndarray:
    """A 2-D float64 copy of rows (tensor, array or a sequence of rows)."""
    if isinstance(data, torch.Tensor):
        if data.is_complex():
            raise PreprocessingError(f"{what} must be numeric (bool, int or float), got {data.dtype}")
        array = data.detach().to("cpu", torch.float64, copy=True).numpy()
    else:
        array = np.asarray(data)
        if array.dtype.kind not in "biuf":
            raise PreprocessingError(f"{what} must be numeric (bool, int or float), got dtype {array.dtype}")
        array = array.astype(np.float64)  # always a copy
    if array.ndim != 2:
        raise PreprocessingError(f"{what} must be 2-D (rows, features), got shape {tuple(array.shape)}")
    return array


def _rows(rows: Iterable[Any], size: int) -> np.ndarray:
    """Validated integer positions (vectorized: millions of rows stay cheap)."""
    if isinstance(rows, (torch.Tensor, np.ndarray)):
        array = rows.detach().cpu().numpy() if isinstance(rows, torch.Tensor) else rows
    else:
        items = list(rows)
        if any(isinstance(row, bool) for row in items):
            raise PreprocessingError("rows must be integer positions, not bools")
        array = np.asarray(items)
    if array.size == 0:
        raise PreprocessingError("rows is empty: a standardizer needs at least one training row")
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise PreprocessingError(f"rows must be a flat list of integer positions, got dtype {array.dtype}")
    array = array.astype(np.int64)
    if np.unique(array).size != array.size:
        raise PreprocessingError("rows lists a position more than once")
    if int(array.min()) < 0 or int(array.max()) >= size:
        outside = array[(array < 0) | (array >= size)][:5].tolist()
        raise PreprocessingError(f"rows {outside} are out of range for a source of {size} rows")
    return array


def _training_matrix(
    source: Any, rows: np.ndarray, columns: Optional[Sequence[ColumnLabel]]
) -> tuple[np.ndarray, Optional[list[ColumnLabel]]]:
    """Only the ``rows`` (and schema columns) of ``source``, as float64, read
    in sorted order so a membership's statistics never depend on its order."""
    import pandas as pd

    rows = np.sort(rows)
    if isinstance(source, pd.DataFrame):
        names = list(_labels(source.columns if columns is None else columns, "columns"))
        positions = _column_positions(source, names, "the source")
        return _numeric_frame(source.iloc[rows, positions], names, "the training rows'"), names
    if columns is not None:
        raise PreprocessingError("columns= names DataFrame columns; this source has no column names")
    if isinstance(source, (torch.Tensor, np.ndarray)):
        index = torch.as_tensor(rows) if isinstance(source, torch.Tensor) else rows
        return _matrix(source[index], "the training rows"), None
    return _matrix([source[int(row)] for row in rows], "the training rows"), None  # generic row source


def _membership(value: Any, rows: np.ndarray) -> IdentityRef:
    if value is None:  # a digest of the positions themselves
        payload = b"nnx.preprocessing.rows\0" + np.sort(rows).astype("<i8").tobytes()
        return IdentityRef("digest", f"sha256:{hashlib.sha256(payload).hexdigest()}")
    try:
        return _ref(value, "membership", allow_split=True)
    except ValueError as exc:  # e.g. an empty declared id
        raise PreprocessingError(f"membership: {exc}") from exc


def _moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column mean and population deviation in float64. A column whose sums
    overflow is recomputed after an exact power-of-two rescale, so any
    column of finite values has finite moments."""
    with np.errstate(over="ignore", invalid="ignore"):
        mean, deviation = values.mean(axis=0), values.std(axis=0)
        bad = ~(np.isfinite(mean) & np.isfinite(deviation))
        if bad.any():  # ldexp scales by 2**-e exactly without forming 2**e (which can overflow)
            _, exponent = np.frexp(np.abs(values[:, bad]).max(axis=0))
            scaled = np.ldexp(values[:, bad], -exponent)
            mean[bad] = np.ldexp(scaled.mean(axis=0), exponent)
            deviation[bad] = np.ldexp(scaled.std(axis=0), exponent)
    return mean, deviation


# --- the standardizer --------------------------------------------------------------------------


def _floats(values: Any, what: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Iterable):
        raise PreprocessingError(f"{what} must be a list of numbers, got {values!r}")
    out = []
    for value in values:
        try:
            number = float(value) if isinstance(value, numbers.Real) and not isinstance(value, bool) else math.nan
        except OverflowError:
            number = math.inf
        if not math.isfinite(number):
            raise PreprocessingError(f"{what} must hold finite numbers, got {value!r}")
        out.append(number)
    return tuple(out)


@dataclass(frozen=True, eq=False)
class Standardizer:
    """Frozen per-column ``(x - mean) / scale`` with a declared schema.

    Build one with :meth:`fit`; reload one with :meth:`load` /
    :meth:`from_state`. ``columns`` names the fitted DataFrame columns in
    order (``str`` or ``int`` labels; ``None`` for unnamed arrays);
    ``dtype`` is the output dtype of :meth:`transform`; ``fit_rows`` and
    ``fit_membership`` identify the training rows it was fitted on. Calling
    the standardizer on one sample (a 1-D row) standardizes that row, so it
    also works as a :class:`SplitView` transform. Equality and hashing
    follow :meth:`canonical_bytes`.
    """

    mean: tuple[float, ...]
    scale: tuple[float, ...]
    columns: Optional[tuple[ColumnLabel, ...]] = None
    dtype: str = "float32"
    fit_rows: int = 0
    fit_membership: IdentityRef = field(default_factory=IdentityRef.unknown)

    def __post_init__(self) -> None:
        mean, scale = _floats(self.mean, "mean"), _floats(self.scale, "scale")
        if not mean or len(mean) != len(scale):
            raise PreprocessingError(
                f"mean and scale need one entry per feature, got lengths {len(mean)} / {len(scale)}"
            )
        if any(value <= 0 for value in scale):
            raise PreprocessingError(f"scale must be positive, got {list(scale)}")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        if self.columns is not None:
            columns = _labels(self.columns, "columns")
            if len(columns) != len(mean):
                raise PreprocessingError(f"columns has length {len(columns)} but there are {len(mean)} features")
            object.__setattr__(self, "columns", columns)
        if not isinstance(self.dtype, str) or self.dtype not in _DTYPES:
            raise PreprocessingError(f"dtype must be one of {sorted(_DTYPES)}, got {self.dtype!r}")
        if isinstance(self.fit_rows, bool) or not isinstance(self.fit_rows, numbers.Integral) or self.fit_rows < 0:
            raise PreprocessingError(f"fit_rows must be a count >= 0, got {self.fit_rows!r}")
        object.__setattr__(self, "fit_rows", int(self.fit_rows))
        object.__setattr__(self, "fit_membership", _ref(self.fit_membership, "fit_membership"))

    @classmethod
    def fit(
        cls,
        source: Any,
        *,
        rows: Iterable[Any],
        columns: Optional[Sequence[ColumnLabel]] = None,
        dtype: Union[str, torch.dtype] = "float32",
        membership: Any = None,
    ) -> Standardizer:
        """Fit on ``source``'s training ``rows`` (integer positions) only.

        ``source`` is a DataFrame (``columns`` selects and orders the schema,
        default all columns; only the training rows of those columns are
        copied), a 2-D tensor or array, or any row-indexable sequence. No
        other row is read. The population mean and standard deviation are
        computed in float64; a constant column (all training values equal)
        gets its value as the mean and scale 1. ``membership`` identifies the
        rows (an ``IdentityRef``, a ``SplitManifest``, a declared string); by
        default it is a digest of the positions. Raises
        :class:`PreprocessingError` for an empty, repeated or out-of-range
        membership and for non-numeric or non-finite training values.
        """
        positions = _rows(rows, len(source))
        identity = _membership(membership, positions)  # validated before any row is read
        output = dtype if isinstance(dtype, str) else dtype_name(dtype)
        values, names = _training_matrix(source, positions, columns)
        if not np.isfinite(values).all():
            raise PreprocessingError("the training rows hold non-finite values; drop or impute them before fitting")
        constant = (values == values[0]).all(axis=0)  # no subtraction: cannot overflow
        moments = _moments(values)
        mean = np.where(constant, values[0], moments[0])
        scale = np.where(constant | (moments[1] == 0), 1.0, moments[1])
        return cls(
            mean=tuple(mean.tolist()),
            scale=tuple(scale.tolist()),
            columns=None if names is None else tuple(names),
            dtype=output,
            fit_rows=int(positions.size),
            fit_membership=identity,
        )

    @property
    def n_features(self) -> int:
        return len(self.mean)

    def transform(self, data: Any) -> torch.Tensor:
        """Standardize ``data`` with the frozen statistics.

        A DataFrame must hold the fitted ``columns`` once each and in the
        fitted order (other columns are ignored); a tensor, array or row
        sequence must be 2-D with ``n_features`` columns. Values must be
        numeric and finite, and so must the result in ``dtype``. Returns a
        new tensor; the input is never modified. Raises
        :class:`PreprocessingError` otherwise — before any model sees it.
        """
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            if self.columns is None:
                values = _numeric_frame(data, list(data.columns), "the input")
            else:
                positions = _column_positions(data, self.columns, "the input")
                if positions != sorted(positions):
                    present = [label for label in data.columns if label in set(self.columns)]
                    raise PreprocessingError(
                        f"the input column order {present} differs from the fitted schema {list(self.columns)}; "
                        "select the columns in the fitted order"
                    )
                values = _numeric_frame(data.iloc[:, positions], self.columns, "the input")
        else:
            values = _matrix(data, "the input")
        if values.shape[1] != self.n_features:
            raise PreprocessingError(
                f"the input width {values.shape[1]} does not match the fitted width {self.n_features}"
            )
        if not np.isfinite(values).all():
            raise PreprocessingError(f"the input holds non-finite values in columns {self._bad(values)}")
        values -= np.asarray(self.mean)  # in place: `values` is always this call's own copy
        values /= np.asarray(self.scale)
        out = torch.from_numpy(values).to(_DTYPES[self.dtype])
        if not bool(torch.isfinite(out).all()):
            raise PreprocessingError(
                f"standardized values overflow dtype={self.dtype} in columns {self._bad(out.double().numpy())}; "
                "use a wider dtype"
            )
        return out

    def __call__(self, sample: Any) -> torch.Tensor:
        """Standardize one sample (a 1-D row) or a batch (2-D) exactly like
        :meth:`transform`. A pandas ``Series`` row is checked against the
        fitted column names and order; any other 1-D row is positional."""
        import pandas as pd

        if isinstance(sample, pd.Series):
            return self.transform(sample.to_frame().T.infer_objects())[0]
        if isinstance(sample, torch.Tensor):
            return self.transform(sample.reshape(1, -1))[0] if sample.dim() == 1 else self.transform(sample)
        array = np.asarray(sample)
        return self.transform(array.reshape(1, -1))[0] if array.ndim == 1 else self.transform(array)

    def _bad(self, values: np.ndarray) -> list[Any]:
        columns = np.flatnonzero(~np.isfinite(values).all(axis=0)).tolist()
        return [self.columns[i] for i in columns] if self.columns is not None else columns

    # --- serialization -----------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "kind": "standardize",
            "columns": None if self.columns is None else list(self.columns),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "dtype": self.dtype,
            "fit": {"rows": self.fit_rows, "membership": self.fit_membership.state()},
        }

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> Standardizer:
        """Rebuild from :meth:`state`, rejecting anything malformed with
        :class:`PreprocessingError` (bad lengths, non-finite values,
        non-positive scales, missing or mistyped entries)."""
        if not isinstance(state, Mapping):
            raise PreprocessingError(f"a standardizer state is a mapping, got {type(state).__name__}")
        if state.get("format") != FORMAT:
            raise PreprocessingError(f"unsupported preprocessing format {state.get('format')!r}; expected {FORMAT!r}")
        if state.get("kind") != "standardize":
            raise PreprocessingError(f"unsupported preprocessing kind {state.get('kind')!r}")
        missing = [key for key in ("mean", "scale") if key not in state]
        if missing:
            raise PreprocessingError(f"the standardizer state lacks {missing}")
        fit = state.get("fit", {})
        if not isinstance(fit, Mapping):
            raise PreprocessingError(f"the standardizer state's 'fit' must be a mapping, got {fit!r}")
        membership = fit.get("membership")
        try:
            identity = IdentityRef.unknown() if membership is None else IdentityRef.from_state(membership)
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            raise PreprocessingError(f"the standardizer's fit membership is malformed: {membership!r}") from exc
        try:
            return Standardizer(
                mean=state["mean"],
                scale=state["scale"],
                columns=state.get("columns"),
                dtype=state.get("dtype", "float32"),
                fit_rows=fit.get("rows", 0),
                fit_membership=identity,
            )
        except TypeError as exc:
            raise PreprocessingError(str(exc)) from exc

    @cached_property
    def _canonical(self) -> bytes:
        return canonical_bytes(self.state())

    def canonical_bytes(self) -> bytes:
        return self._canonical

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Standardizer) and self.canonical_bytes() == other.canonical_bytes()

    def __hash__(self) -> int:
        return hash(self.canonical_bytes())

    def digest(self) -> str:
        """``sha256:<hex>`` of the canonical state: the schema identity."""
        return f"sha256:{hashlib.sha256(self.canonical_bytes()).hexdigest()}"

    def to_json(self) -> str:
        return json.dumps(self.state(), indent=2, sort_keys=True, allow_nan=False) + "\n"

    @staticmethod
    def from_json(text: str) -> Standardizer:
        try:
            state = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PreprocessingError(f"a standardizer file is JSON: {exc}") from exc
        return Standardizer.from_state(state)

    def save(self, path: Union[str, os.PathLike[str]]) -> None:
        """Write the statistics and schema as JSON, atomically."""
        from .nn.params.nn_run import _atomic_write_text

        _atomic_write_text(os.fspath(path), self.to_json())

    @staticmethod
    def load(path: Union[str, os.PathLike[str]]) -> Standardizer:
        with open(path, encoding="utf-8") as handle:
            return Standardizer.from_json(handle.read())


# --- split views -------------------------------------------------------------------------------


def describe_transform(transform: Optional[Callable[..., Any]]) -> Optional[dict[str, Any]]:
    """How a view's transform can be rebuilt: a ``Standardizer`` from its
    state (by digest); any other callable is runtime-only and must be
    registered again — passed to the view anew — to reconstruct it."""
    if transform is None:
        return None
    if isinstance(transform, Standardizer):
        return {"kind": "standardizer", "digest": transform.digest(), "reconstructible": True}
    name = getattr(transform, "__qualname__", None) or type(transform).__qualname__
    return {"kind": "runtime", "qualname": name, "reconstructible": False}


class SplitView(Dataset):
    """Rows ``indices`` of ``base``, with ``transform`` applied to each
    sample's input.

    A sample is read from ``base`` (with whatever transform ``base`` already
    applies) and only its first element is transformed; targets and any
    other elements pass through unchanged, and tuples, named tuples and
    lists keep their type. ``base`` itself — its ``transform``, labels and
    order — is never modified, so a training view with a random
    augmentation and an evaluation view with a fixed transform can share one
    base and be read alternately or from DataLoader workers. ``indices``
    are the view's sample ids in ``base``; ``dataset`` aliases ``base`` like
    ``torch.utils.data.Subset``, and ``classes`` forwards the base's class
    names. Dict samples are rejected: which entry is the input is ambiguous.
    """

    def __init__(self, base: Any, indices: Iterable[int], *, transform: Optional[Callable[..., Any]] = None) -> None:
        self.base = base
        self.indices = tuple(int(index) for index in indices)
        self.transform = transform

    @property
    def dataset(self) -> Any:
        return self.base

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        sample = self.base[self.indices[index]]
        if self.transform is None:
            return sample
        if isinstance(sample, tuple) and hasattr(sample, "_fields"):  # a named tuple
            named: Any = sample
            return named._replace(**{named._fields[0]: self.transform(named[0])})
        if isinstance(sample, tuple):
            return (self.transform(sample[0]), *sample[1:])
        if isinstance(sample, list):
            return [self.transform(sample[0]), *sample[1:]]
        if isinstance(sample, Mapping):
            raise TypeError(
                "SplitView transforms the first element of a (input, target, ...) sample or a bare input; "
                "for dict samples, wrap the base dataset to return a tuple"
            )
        return self.transform(sample)

    @property
    def classes(self) -> Any:
        """The base dataset's class names (dataset-wide, like its labels)."""
        return self.base.classes

    @property
    def reconstructible(self) -> bool:
        """Whether the transform can be rebuilt from :meth:`state` alone."""
        described = describe_transform(self.transform)
        return described is None or bool(described["reconstructible"])

    def state(self) -> dict[str, Any]:
        return {"rows": len(self.indices), "transform": describe_transform(self.transform)}
