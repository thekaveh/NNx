"""Bounded training history with an append journal (FEAT-036).

By default ``NNModel.train`` and ``Trainer.train`` keep every per-batch
``NNIterationDataPoint`` in memory and rewrite ``runs/<id>/idps.csv`` each
epoch, so memory and bytes written grow with the whole run. Passing
``history=HistoryJournal(retention=..., chunk_size=...)`` to either loop
switches that run to a bounded **history journal**:

- only the last ``retention`` records stay in memory; ``ctx.idps`` is that
  window and the returned ``NNRun.idps`` holds it;
- every record is appended once to an immutable chunk file, so the bytes
  written grow with the new records only (``idps.csv`` is not written);
- ``NNRun.load`` reads the committed tail (the last ``retention`` records),
  never the whole history, and the run's notebook chart reads per-epoch
  summary rows;
- :func:`iter_history` streams the committed history, :func:`export_history_csv`
  writes it in the legacy ``idps.csv`` layout, and :func:`migrate_history`
  moves a legacy run's CSV into a journal without changing its run id.

On disk, ``runs/<id>/history/`` holds::

    chunk-<seq>.jsonl   immutable: up to chunk_size records, one JSON line each
    index.jsonl         appended: one line per chunk (records, iteration and
                        epoch range, SHA-256), hash-chained
    epochs.jsonl        appended: one summary row per epoch (the chart's series)
    journal.json        the small manifest — counts, byte lengths and the
                        index chain — replaced atomically

**Commit protocol.** Each epoch's records are written (and, with the index
and summary rows, flushed to disk) and the manifest is published *before*
the LAST checkpoint, which stays the epoch's commit marker. Readers show
only records up to LAST's epoch: a crash while a chunk is written or before
the manifest is replaced leaves the previous manifest, and a crash before
LAST is replaced leaves an uncommitted tail that is ignored. A committed
chunk whose bytes no longer match its SHA-256 raises
:class:`HistoryCorruptionError`.

**Callbacks.** Built-in callbacks read ``ctx.idp`` (the epoch's last record),
so the window changes nothing for them; ``LRMonitor.history`` keeps the LRs
of the last ``retention`` epochs. A ``Callback`` that needs the whole history
sets ``history_access = "full"`` and gets the materialised history as
``ctx.idps`` at ``on_epoch_end`` and ``on_train_end`` (read back from the
journal: its cost grows with the run). A plain function callback —
``callbacks=[lambda idps: ...]``, which by contract receives the full list —
is refused before training, as is an unknown ``history_access``.

**Lineage.** A resumed run owns its own run directory and chunk files; its
parent is never written. ``iter_history(run_id, lineage=True)`` yields the
parent's committed records up to the checkpoint the child resumed from
(following ``resume_from_run_id`` / ``parent_run_id``, recursively), then
the child's — never the parent's later epochs, never an epoch twice.
"""

from __future__ import annotations

import collections
import hashlib
import itertools
import json
import math
import os
import tempfile
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from .nn.params.nn_iteration_data_point import NNIterationDataPoint
    from .nn.params.nn_run import NNRun

__all__ = [
    "JOURNAL_FORMAT",
    "JOURNAL_VERSION",
    "HistoryCorruptionError",
    "HistoryJournal",
    "export_history_csv",
    "iter_history",
    "migrate_history",
]

JOURNAL_FORMAT = "nnx.history-journal"
JOURNAL_VERSION = 1
HISTORY_DIR = "history"
_MANIFEST = "journal.json"
_INDEX = "index.jsonl"
_EPOCHS = "epochs.jsonl"
_GENESIS = "0" * 64
_ACCESS = ("window", "full")
_SERIES = ("train_loss", "train_err", "val_loss", "val_err", "monitor")


class HistoryCorruptionError(ValueError):
    """A history journal whose committed files do not match its manifest:
    a chunk whose bytes changed, a broken index chain, a missing file."""


@dataclass(frozen=True)
class HistoryJournal:
    """Opt-in bounded history for one training run.

    Args:
        retention: how many of the most recent records stay in memory (the
            ``ctx.idps`` window, ``NNRun.idps`` of the returned and of a
            loaded run).
        chunk_size: records per journal chunk file, at most ``retention``
            (so the records waiting for their chunk are always inside the
            window). ``None`` (the default) means ``min(250, retention)``;
            :attr:`chunk` is the resolved size.
    """

    retention: int = 1000
    chunk_size: Optional[int] = None

    def __post_init__(self) -> None:
        from ._validation import require_count

        object.__setattr__(
            self, "retention", require_count(self.retention, "retention", owner="HistoryJournal", minimum=1)
        )
        if self.chunk_size is not None:
            object.__setattr__(
                self, "chunk_size", require_count(self.chunk_size, "chunk_size", owner="HistoryJournal", minimum=1)
            )
            if self.chunk_size > self.retention:
                raise ValueError(
                    f"HistoryJournal.chunk_size ({self.chunk_size}) must not exceed retention ({self.retention}): "
                    "records waiting for their chunk stay in memory"
                )

    @property
    def chunk(self) -> int:
        """Records per chunk file: ``chunk_size``, or ``min(250, retention)``."""
        return self.chunk_size if self.chunk_size is not None else min(250, self.retention)


# --- encoding --------------------------------------------------------------------------------


def _scalar(value: Any) -> Any:
    """A NumPy or 0-d tensor scalar in a record (a custom step's metric,
    say) as the Python number it holds — ``idps.csv`` writes those too."""
    item = getattr(value, "item", None)
    if callable(item):
        try:
            result = item()
        except (TypeError, ValueError, RuntimeError):
            pass
        else:
            if result is None or isinstance(result, (bool, int, float, str)):
                return result
    raise TypeError(f"a history record holds a {type(value).__name__}, which the journal cannot store as JSON")


def _record_line(record: NNIterationDataPoint) -> bytes:
    text = json.dumps(record.state(), separators=(",", ":"), allow_nan=True, default=_scalar)
    return text.encode("utf-8") + b"\n"


def _record(line: bytes) -> NNIterationDataPoint:
    from .nn.params.nn_checkpoint import _idp_from_nested_state

    return _idp_from_nested_state(json.loads(line))


def _json_line(value: Any) -> bytes:
    text = json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=True, default=_scalar)
    return text.encode("utf-8") + b"\n"


def _chain(previous: str, line: bytes) -> str:
    return hashlib.sha256(previous.encode("ascii") + line).hexdigest()


def _chunk_name(seq: int) -> str:
    return f"chunk-{seq:08d}.jsonl"


def _nan_to_none(value: Optional[float]) -> Optional[float]:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


# --- the per-epoch summary row (the notebook chart's series) ------------------------------------


class _FieldStats:
    """One per-batch field's running sums over an epoch: plain and weighted
    by each record's ``count`` (FEAT-026). :meth:`total` picks between them
    as ``nn_run._field_total`` does for an eager run."""

    def __init__(self) -> None:
        self.sum = 0.0
        self.n = 0
        self.weighted = 0.0
        self.weight = 0
        self.counted = True  # every record with the field carries a count

    def add(self, value: Optional[float], count: Optional[int]) -> None:
        if value is None:
            return
        self.sum += value
        self.n += 1
        if count is None:
            self.counted = False
        elif self.counted:
            self.weighted += value * count
            self.weight += count

    def total(self) -> tuple[float, int]:
        if self.n and self.counted and self.weight:
            return self.weighted, self.weight
        return self.sum, self.n


class _EpochStats:
    """Running per-batch loss / error sums of the epoch being written — the
    same left-to-right ``+=`` ``NNRun._epoch_series`` applies
    (``nn_run._field_total``), so journal and eager charts agree to the bit."""

    def __init__(self) -> None:
        self._reset(None)

    def _reset(self, epoch: Optional[int]) -> None:
        self.epoch = epoch
        self.loss = _FieldStats()
        self.error = _FieldStats()

    def add(self, record: NNIterationDataPoint) -> None:
        if self.epoch != record.epoch_idx:
            self._reset(record.epoch_idx)
        train = record.train_edp
        if train is not None:
            self.loss.add(train.loss, train.count)
            self.error.add(train.error, train.count)

    def row(self, last: NNIterationDataPoint) -> dict[str, Any]:
        """The epoch's chart row — the rules ``NNRun._epoch_series`` applies
        to an eager run, from the epoch's last record (the only one that
        carries validation, the summary and the monitor record)."""
        from .nn.params.nn_run import _epoch_values

        values = _epoch_values(
            last.train_summary,
            self.loss.total(),
            self.error.total(),
            last.val_edp,
            last.selection,
        )
        return {
            "epoch": last.epoch_idx,
            **{name: _nan_to_none(values[name]) for name in _SERIES},
            "improved": values["improved"],
        }


# --- writing ---------------------------------------------------------------------------------


def _fsync_path(path: str) -> None:
    """Flush a written file (or a directory entry) to disk; like the run's
    other writers, a filesystem that cannot fsync is tolerated."""
    try:
        # A file is opened for writing: Windows' fsync needs a writable handle.
        fd = os.open(path, os.O_RDONLY if os.path.isdir(path) else os.O_RDWR)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _append_bytes(path: str, data: bytes) -> None:
    """Append to the index or the epoch rows. Not fsynced here: the epoch's
    files are flushed together before its manifest is published."""
    with open(path, "ab") as handle:
        handle.write(data)


def _write_chunk_file(path: str, data: bytes) -> None:
    """A chunk is written once: an owned temporary file, renamed into place
    (the temporary is removed if anything fails). Not fsynced here — see
    :func:`_append_bytes`."""
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise


def _empty_manifest(spec: HistoryJournal) -> dict[str, Any]:
    return {
        "format": JOURNAL_FORMAT,
        "version": JOURNAL_VERSION,
        "retention": spec.retention,
        "chunk_size": spec.chunk,
        "chunks": 0,
        "records": 0,
        "index_bytes": 0,
        "index_chain": _GENESIS,
        "epochs": 0,
        "epochs_bytes": 0,
    }


def _manifest_text(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, sort_keys=True, indent=1) + "\n"


class _JournalWriter:
    """Appends one run's records as immutable chunks and publishes the
    manifest; memory is bounded by ``chunk_size`` records (and one file
    name per chunk)."""

    def __init__(self, directory: str, spec: HistoryJournal) -> None:
        self.directory = directory
        self.spec = spec
        self.chunk_size = spec.chunk
        self.records = 0
        self.index_bytes = 0
        self.index_chain = _GENESIS
        self.epochs = 0
        self.epochs_bytes = 0
        self.chunks = 0  # chunk files written, named by sequence number
        self._synced = 0  # chunks already flushed to disk
        self._pending: list[NNIterationDataPoint] = []
        self._lines: list[bytes] = []  # the pending records, encoded as they arrive
        self._stats = _EpochStats()
        self._published: Optional[str] = None
        self._previous: Optional[str] = None

    def _ensure(self) -> None:
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory, exist_ok=True)
            self._new_directory = True  # its entry in the run directory is flushed with the epoch

    _new_directory = False

    def append(self, record: NNIterationDataPoint, line: Optional[bytes] = None) -> None:
        if line is None:
            line = _record_line(record)  # a value the journal cannot store fails on its first batch
        self._pending.append(record)
        self._lines.append(line)
        self._stats.add(record)
        if len(self._pending) > self.chunk_size:  # all but the newest record are final
            self._flush(self.chunk_size)

    def replace_last(self, record: NNIterationDataPoint) -> None:
        self._lines[-1] = _record_line(record)
        self._pending[-1] = record

    def discard_pending(self, count: int) -> None:
        """Forget up to ``count`` newest unwritten records (FEAT-033). Any
        already flushed to a chunk stay beyond the published manifest — an
        uncommitted tail every reader ignores."""
        keep = max(len(self._pending) - count, 0)
        del self._pending[keep:]
        del self._lines[keep:]

    def _flush(self, count: int) -> None:
        self._write_chunk(self._pending[:count], self._lines[:count])
        del self._pending[:count]
        del self._lines[:count]

    def _write_chunk(self, records: Sequence[NNIterationDataPoint], lines: Sequence[bytes]) -> None:
        self._ensure()
        data = b"".join(lines)
        seq = self.chunks
        name = _chunk_name(seq)
        _write_chunk_file(os.path.join(self.directory, name), data)
        line = _json_line(
            {
                "seq": seq,
                "file": name,
                "records": len(records),
                "first_iter": records[0].iter_idx,
                "last_iter": records[-1].iter_idx,
                "first_epoch": records[0].epoch_idx,
                "last_epoch": records[-1].epoch_idx,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        _append_bytes(os.path.join(self.directory, _INDEX), line)
        self.index_chain = _chain(self.index_chain, line)
        self.index_bytes += len(line)
        self.records += len(records)
        self.chunks += 1

    def end_epoch(self, *, sync: bool = True) -> None:
        """Write the epoch's remaining records (all final now) and its
        summary row, then (``sync``) flush every file written since the
        last flush; the manifest is published separately."""
        last = self._pending[-1] if self._pending else None
        if last is None:
            return
        row = self._stats.row(last)
        while self._pending:
            self._flush(self.chunk_size)
        line = _json_line(row)
        _append_bytes(os.path.join(self.directory, _EPOCHS), line)
        self.epochs += 1
        self.epochs_bytes += len(line)
        if sync:
            self.sync()

    def sync(self) -> None:
        """Flush the chunks written since the last flush, the index, the
        epoch rows and the directory entries to disk."""
        for name in (*map(_chunk_name, range(self._synced, self.chunks)), _INDEX, _EPOCHS):
            _fsync_path(os.path.join(self.directory, name))
        _fsync_path(self.directory)
        if self._new_directory:  # the run directory's new history/ entry
            _fsync_path(os.path.dirname(self.directory))
            self._new_directory = False
        self._synced = self.chunks

    def manifest(self) -> dict[str, Any]:
        return {
            **_empty_manifest(self.spec),
            "chunks": self.chunks,
            "records": self.records,
            "index_bytes": self.index_bytes,
            "index_chain": self.index_chain,
            "epochs": self.epochs,
            "epochs_bytes": self.epochs_bytes,
        }

    def publish(self) -> None:
        from .nn.params.nn_run import _atomic_write_text

        self._ensure()
        text = _manifest_text(self.manifest())
        _atomic_write_text(os.path.join(self.directory, _MANIFEST), text)
        _fsync_path(self.directory)  # the rename is durable before LAST, the commit marker
        self._previous, self._published = self._published, text

    def rollback(self) -> None:
        """Republish the manifest of the previous epoch (the current epoch
        could not commit) — an empty journal's before the first commit, as
        an eager run rolls back to an empty ``idps.csv``."""
        from .nn.params.nn_run import _atomic_write_text

        previous = self._previous if self._previous is not None else _manifest_text(_empty_manifest(self.spec))
        _atomic_write_text(os.path.join(self.directory, _MANIFEST), previous)

    def materialize(self) -> list[NNIterationDataPoint]:
        """Every record this writer has seen, read back from its chunks —
        each checked against the index, as every reader does — plus those
        still waiting for one."""
        if not self.chunks:
            return list(self._pending)
        written = _JournalReader(self.directory, manifest=self.manifest()).records(None)
        return [*written, *self._pending]


# --- reading ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Chunk:
    seq: int
    file: str
    records: int
    first_epoch: int
    last_epoch: int
    sha256: str


class _JournalReader:
    """A published journal: its manifest and index, checked against each
    other; chunks are read (and checked) on demand. ``manifest`` reads a
    journal not yet published (a migration checks it before publishing)."""

    def __init__(self, directory: str, manifest: Optional[dict[str, Any]] = None, *, index: bool = True) -> None:
        self.directory = directory
        path = os.path.join(directory, _MANIFEST)
        if manifest is None:
            try:
                with open(path, encoding="utf-8") as handle:
                    manifest = json.load(handle)
            except FileNotFoundError:
                raise HistoryCorruptionError(f"the history journal at {directory} has no {_MANIFEST}") from None
            except OSError as exc:
                raise HistoryCorruptionError(f"the history journal manifest at {path} is unreadable: {exc}") from None
            except ValueError as exc:
                raise HistoryCorruptionError(f"the history journal manifest at {path} is not JSON: {exc}") from None
        if not isinstance(manifest, dict) or manifest.get("format") != JOURNAL_FORMAT:
            raise HistoryCorruptionError(f"{path} is not an NNx history journal manifest")
        if manifest.get("version") != JOURNAL_VERSION:
            raise HistoryCorruptionError(
                f"unsupported history journal version {manifest.get('version')!r}; this NNx reads {JOURNAL_VERSION}"
            )
        counts = ("retention", "chunks", "records", "index_bytes", "epochs", "epochs_bytes")
        if not all(isinstance(manifest.get(key), int) and manifest[key] >= 0 for key in counts) or not isinstance(
            manifest.get("index_chain"), str
        ):
            raise HistoryCorruptionError(f"the history journal manifest at {path} is malformed")
        self.manifest = manifest
        self.retention = max(1, manifest["retention"])
        # index=False: the manifest alone (the epoch rows need nothing more).
        self.chunks = self._read_index() if index else []

    def _read_index(self) -> list[_Chunk]:
        path = os.path.join(self.directory, _INDEX)
        wanted = self.manifest["index_bytes"]
        data = b""
        if wanted:
            try:
                with open(path, "rb") as handle:
                    data = handle.read(wanted)
            except FileNotFoundError:
                raise HistoryCorruptionError(f"the history journal at {self.directory} has no {_INDEX}") from None
        if len(data) != wanted:
            raise HistoryCorruptionError(f"{path} is shorter than its committed {wanted} bytes")
        lines = data.splitlines(keepends=True)
        chain = _GENESIS
        for line in lines:
            chain = _chain(chain, line)
        if chain != self.manifest["index_chain"] or len(lines) != self.manifest["chunks"]:
            raise HistoryCorruptionError(f"the committed index {path} does not match its manifest (altered)")
        try:
            chunks = [
                _Chunk(
                    seq=entry["seq"],
                    file=entry["file"],
                    records=entry["records"],
                    first_epoch=entry["first_epoch"],
                    last_epoch=entry["last_epoch"],
                    sha256=entry["sha256"],
                )
                for entry in map(json.loads, lines)
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise HistoryCorruptionError(f"the committed index {path} is malformed: {exc}") from None
        if [(chunk.seq, chunk.file) for chunk in chunks] != [(seq, _chunk_name(seq)) for seq in range(len(chunks))]:
            raise HistoryCorruptionError(f"the committed index {path} does not number its chunks in order")
        if sum(chunk.records for chunk in chunks) != self.manifest["records"]:
            raise HistoryCorruptionError(f"the committed index {path} does not count the manifest's records")
        return chunks

    def read_chunk(self, chunk: _Chunk) -> list[NNIterationDataPoint]:
        path = os.path.join(self.directory, chunk.file)
        try:
            with open(path, "rb") as handle:
                data = handle.read()
        except FileNotFoundError:
            raise HistoryCorruptionError(f"committed history chunk {chunk.seq} ({chunk.file}) is missing") from None
        if hashlib.sha256(data).hexdigest() != chunk.sha256:
            raise HistoryCorruptionError(
                f"committed history chunk {chunk.seq} ({chunk.file}) is corrupt: its bytes do not match the index"
            )
        lines = [line for line in data.splitlines() if line]
        if len(lines) != chunk.records:
            raise HistoryCorruptionError(f"committed history chunk {chunk.seq} ({chunk.file}) has the wrong count")
        return [_record(line) for line in lines]

    def records(self, max_epoch: Optional[int]) -> Iterator[NNIterationDataPoint]:
        """Every published record up to ``max_epoch`` (``None``: all), in order."""
        for chunk in self.chunks:
            if max_epoch is not None and chunk.first_epoch > max_epoch:
                return
            for record in self.read_chunk(chunk):
                if max_epoch is None or record.epoch_idx <= max_epoch:
                    yield record

    def tail(self, count: int, max_epoch: Optional[int]) -> list[NNIterationDataPoint]:
        """The last ``count`` records up to ``max_epoch``, reading only the
        chunks that hold them."""
        visible = [chunk for chunk in self.chunks if max_epoch is None or chunk.first_epoch <= max_epoch]
        needed, start = 0, len(visible)
        while start > 0 and needed < count:
            start -= 1
            needed += visible[start].records
        window: collections.deque[NNIterationDataPoint] = collections.deque(maxlen=count)
        for chunk in visible[start:]:
            window.extend(
                record for record in self.read_chunk(chunk) if max_epoch is None or record.epoch_idx <= max_epoch
            )
        return list(window)

    def first_epoch(self, max_epoch: Optional[int]) -> Optional[int]:
        """The epoch of the first committed record (``None`` when there is none)."""
        if not self.chunks or (max_epoch is not None and self.chunks[0].first_epoch > max_epoch):
            return None
        return self.chunks[0].first_epoch

    def epoch_rows(self, max_epoch: Optional[int]) -> list[dict[str, Any]]:
        wanted = self.manifest["epochs_bytes"]
        path = os.path.join(self.directory, _EPOCHS)
        if not wanted:
            return []
        try:
            with open(path, "rb") as handle:
                data = handle.read(wanted)
            if len(data) != wanted:
                raise ValueError(f"shorter than its committed {wanted} bytes")
            rows = [json.loads(line) for line in data.splitlines() if line]
            return [row for row in rows if max_epoch is None or row["epoch"] <= max_epoch]
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise HistoryCorruptionError(f"the committed epoch rows {path} are unreadable: {exc}") from None


def _journal_epoch_series(directory: str, max_epoch: int) -> dict[str, list[Any]]:
    """``NNRun._epoch_series`` for a journal run: its per-epoch rows up to
    ``max_epoch``, with ``None`` read back as NaN."""
    nan = float("nan")
    series: dict[str, list[Any]] = {name: [] for name in ("epochs", *_SERIES, "improved")}
    for row in _JournalReader(directory, index=False).epoch_rows(max_epoch):
        try:
            values = [nan if row[name] is None else float(row[name]) for name in _SERIES]
            improved = bool(row["improved"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HistoryCorruptionError(f"a committed epoch row in {directory} is malformed: {exc!r}") from None
        series["epochs"].append(row["epoch"])
        for name, value in zip(_SERIES, values, strict=True):
            series[name].append(value)
        series["improved"].append(improved)
    return series


def has_journal(run_path: str) -> bool:
    """Whether a run directory keeps its history in a published journal."""
    return os.path.isfile(os.path.join(run_path, HISTORY_DIR, _MANIFEST))


# --- the training loop's history ---------------------------------------------------------------


class _EagerHistory:
    """Every record in one list, rewritten to ``idps.csv`` each epoch — the
    behaviour of every run without a :class:`HistoryJournal`."""

    def __init__(self) -> None:
        self.records: list[NNIterationDataPoint] = []
        self._epoch_start = 0

    def begin_epoch(self) -> None:
        self._epoch_start = len(self.records)

    def epoch_is_empty(self) -> bool:
        return len(self.records) == self._epoch_start

    def append(self, record: NNIterationDataPoint) -> None:
        self.records.append(record)

    @property
    def last(self) -> NNIterationDataPoint:
        return self.records[-1]

    def replace_last(self, record: NNIterationDataPoint) -> None:
        self.records[-1] = record

    def discard_epoch(self) -> None:
        """Drop the epoch in progress (a stop at an update boundary, FEAT-033)."""
        del self.records[self._epoch_start :]

    def __bool__(self) -> bool:
        return bool(self.records)

    def window(self) -> list[NNIterationDataPoint]:
        return self.records

    def lender(self, *, tolerant: bool = False) -> Any:
        """What to lend each callback as ``ctx.idps`` — nothing: an eager
        run's callbacks share the running list, as always."""
        return lambda callback: None

    def save_epoch(self, run: NNRun) -> None:
        run.with_idps(self.records).save(update_best=False)

    def rollback_epoch(self, run: NNRun) -> None:
        run.with_idps(self.records[: self._epoch_start]).save(update_best=False)

    def finish(self, run: NNRun) -> NNRun:
        return run.with_idps(self.records).save()


class _ReplicaHistory(_EagerHistory):
    """A DDP replica rank's records (FEAT-030): the same global records as
    the writer's, kept in memory only — the writer alone persists them."""

    def save_epoch(self, run: NNRun) -> None:
        pass

    def rollback_epoch(self, run: NNRun) -> None:
        pass

    def finish(self, run: NNRun) -> NNRun:
        return run.with_idps(self.records)


class _JournalHistory:
    """The last ``retention`` records in memory, every record in the run's
    journal."""

    def __init__(self, run: NNRun, spec: HistoryJournal) -> None:
        from .nn.params.nn_run import _runs_root

        self.spec = spec
        self.directory = os.path.realpath(os.path.join(_runs_root(None), run.id, HISTORY_DIR))
        self._window: collections.deque[NNIterationDataPoint] = collections.deque(maxlen=spec.retention)
        # ctx.idps: live like an eager run's list, but read-only — the journal
        # (and LAST, and NNRun.idps) keep the records as they were recorded.
        self._view = _WindowView(self._window)
        self._writer = _JournalWriter(self.directory, spec)
        self._epoch_records = 0

    def begin_epoch(self) -> None:
        self._epoch_records = 0

    def epoch_is_empty(self) -> bool:
        return self._epoch_records == 0

    def append(self, record: NNIterationDataPoint) -> None:
        self._writer.append(record)
        self._window.append(record)
        self._epoch_records += 1

    def discard_epoch(self) -> None:
        """Drop the epoch in progress (a stop at an update boundary, FEAT-033):
        the writer forgets its unwritten records, and the window is read back
        from the committed journal — the epoch's records may have evicted
        part of it, and keeping a copy would double the memory bound."""
        self._writer.discard_pending(self._epoch_records)
        self._epoch_records = 0
        self._window.clear()
        if os.path.exists(os.path.join(self.directory, _MANIFEST)):  # nothing committed yet otherwise
            self._window.extend(_JournalReader(self.directory).tail(self.spec.retention, None))

    @property
    def last(self) -> NNIterationDataPoint:
        return self._window[-1]

    def replace_last(self, record: NNIterationDataPoint) -> None:
        self._writer.replace_last(record)
        self._window[-1] = record

    def __bool__(self) -> bool:
        return bool(self._window)

    def window(self) -> Any:
        return self._view

    def full(self) -> list[NNIterationDataPoint]:
        return self._writer.materialize()

    def lender(self, *, tolerant: bool = False) -> Any:
        """What to lend each callback of one dispatch as ``ctx.idps``: the
        whole history — read back once for the dispatch and released with
        it — for a callback declaring ``history_access = "full"``, nothing
        for any other. With ``tolerant`` (``on_train_end``, where cleanup
        must run) a history that cannot be read back is warned about once
        and the window is lent instead."""
        held: list[Any] = []

        def view(callback: Any) -> Any:
            if not _wants_full(callback):
                return None
            if not held:
                try:
                    held.append(_WindowView(self.full()))  # read-only, like the window
                except Exception as exc:  # noqa: BLE001 — re-raised unless tolerant
                    if not tolerant:
                        raise
                    warnings.warn(
                        f"the history journal could not be read back for on_train_end ({type(exc).__name__}: "
                        f"{exc}); callbacks declaring history_access='full' get the last {self.spec.retention} "
                        "records",
                        RuntimeWarning,
                        stacklevel=4,
                    )
                    held.append(self._view)
            return held[0]

        return view

    def _run(self, run: NNRun) -> NNRun:
        return run.with_idps(list(self._window)).with_history(self.directory)

    def save_epoch(self, run: NNRun) -> None:
        self._writer.end_epoch()
        self._writer.publish()  # before the LAST checkpoint, the epoch's commit marker
        self._run(run).save(update_best=False)

    def rollback_epoch(self, run: NNRun) -> None:
        self._writer.rollback()

    def finish(self, run: NNRun) -> NNRun:
        return self._run(run).save()


TrainingHistory = Union[_EagerHistory, _JournalHistory]


def _training_history(run: NNRun, spec: Optional[HistoryJournal]) -> TrainingHistory:
    return _EagerHistory() if spec is None else _JournalHistory(run, spec)


def _check_history(history: Any, callbacks: Any) -> None:
    """Refuse, before any run is reserved, a history spec of the wrong type
    and — with a journal — the callbacks it cannot serve: a plain function
    callback (it receives the full list) or an unknown ``history_access``."""
    if history is None:
        return
    if not isinstance(history, HistoryJournal):
        raise TypeError(f"history must be an nnx.history.HistoryJournal, got {type(history).__name__}")
    from .nn.callbacks import Callback

    for callback in callbacks or []:
        if not isinstance(callback, Callback):
            raise ValueError(
                f"a function callback ({getattr(callback, '__name__', type(callback).__name__)}) receives the full "
                "idps list, which a history journal does not keep in memory: wrap it in a Callback that declares "
                "history_access = 'full' (the history is read back for it each epoch) or reads ctx.idps as the "
                f"last {history.retention} records"
            )
        access = getattr(callback, "history_access", "window")
        if access not in _ACCESS:
            raise ValueError(f"{type(callback).__name__}.history_access must be one of {_ACCESS}, got {access!r}")


def _wants_full(callback: Any) -> bool:
    return getattr(callback, "history_access", "window") == "full"


class _WindowView(Sequence):
    """``ctx.idps`` in a journal run: a live, read-only sequence over the
    window — it grows with each record (trimmed to ``retention``) as an
    eager run's list does — or over the whole history lent to a ``"full"``
    callback. It supports ``len``, indexing, slicing (a new list),
    iteration and ``reversed``, but no in-place change (item assignment
    raises ``TypeError``; it has no ``append`` or other list mutator): the
    journal, LAST and ``NNRun.idps`` keep the records exactly as they were
    recorded."""

    __slots__ = ("_records",)

    def __init__(self, records: Union[collections.deque[NNIterationDataPoint], list[NNIterationDataPoint]]) -> None:
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: Any) -> Any:
        if not isinstance(index, slice):
            return self._records[index]
        size = len(self._records)
        start, stop, step = index.indices(size)
        if step == 1 and start >= size // 2:  # a tail (ctx.idps[-k:]): walk from the right end
            tail = list(itertools.islice(reversed(self._records), size - stop, size - start))
            tail.reverse()
            return tail
        if step > 0:
            return list(itertools.islice(self._records, start, stop, step))
        return list(self._records)[index]

    def __iter__(self) -> Iterator[NNIterationDataPoint]:
        return iter(self._records)

    def __reversed__(self) -> Iterator[NNIterationDataPoint]:
        return reversed(self._records)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (_WindowView, list, tuple)):
            return list(self) == list(other)
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return repr(list(self._records))


def _lend_idps(ctx: Any, view: Optional[list[NNIterationDataPoint]], call: Any) -> None:
    """Run ``call()`` with ``ctx.idps`` lent as ``view`` (``None``: as it
    is), then put the previous list back unless the callback reassigned it
    — so a lent history never outlives its callback, and a reassignment
    reaches the next one, as in an eager run."""
    if view is None:
        call()
        return
    current, ctx.idps = ctx.idps, view
    try:
        call()
    finally:
        if ctx.idps is view:
            ctx.idps = current


def _dispatch_epoch_end(callbacks: Sequence[Any], ctx: Any, history: TrainingHistory) -> None:
    """``on_epoch_end`` for every callback. An eager run hands every callback
    the running list, exactly as before; a journal run hands each its window,
    or — for a callback declaring ``history_access = "full"`` — the whole
    history, read back once per epoch."""
    ctx.idps = history.window()
    view = history.lender()  # a full read-back lives only as long as this dispatch
    for callback in callbacks:
        _lend_idps(ctx, view(callback), lambda callback=callback: callback.on_epoch_end(ctx))


# --- reading a run's history -----------------------------------------------------------------


def _run_path(run_id: str, root: Optional[str]) -> str:
    from .nn.params.nn_run import _runs_root, _validate_run_id

    return os.path.join(_runs_root(root), _validate_run_id(run_id))


def _reader(run_id: str, root: Optional[str], readers: dict[str, Optional[_JournalReader]]) -> Optional[_JournalReader]:
    """A run's journal reader (``None`` for a CSV run), opened once per
    ``iter_history`` call."""
    if run_id not in readers:
        run_path = _run_path(run_id, root)
        readers[run_id] = _JournalReader(os.path.join(run_path, HISTORY_DIR)) if has_journal(run_path) else None
    return readers[run_id]


def _own_records(
    run_id: str, root: Optional[str], committed: Optional[int], readers: dict[str, Optional[_JournalReader]]
) -> Iterator[NNIterationDataPoint]:
    """A run's records up to its committed epoch (``None``: all) — lazily
    for a journal; a legacy CSV is read whole, floats round-trip (its export
    reproduces the file)."""
    from .nn.params.nn_run import _read_idps_csv

    reader = _reader(run_id, root, readers)
    if reader is not None:
        yield from reader.records(committed)
        return
    for record in _read_idps_csv(os.path.join(_run_path(run_id, root), "idps.csv"), exact=True):
        if committed is None or record.epoch_idx <= committed:
            yield record


def _first_epoch(
    run_id: str, root: Optional[str], committed: Optional[int], readers: dict[str, Optional[_JournalReader]]
) -> Optional[int]:
    """The epoch of a run's first committed record, without reading its
    history: a journal's index, or a CSV's first row."""
    import pandas as pd

    reader = _reader(run_id, root, readers)
    if reader is not None:
        return reader.first_epoch(committed)
    try:
        frame = pd.read_csv(os.path.join(_run_path(run_id, root), "idps.csv"), nrows=1)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None
    if frame.empty or "epoch_idx" not in frame.columns:
        return None
    first = int(frame["epoch_idx"].iloc[0])
    return None if committed is not None and first > committed else first


def _parent(run_id: str, root: Optional[str]) -> tuple[Optional[str], Any, Optional[int]]:
    """The run a run resumed from and the checkpoint it resumed from — from
    its recorded resume status (``metadata.yaml``); a run that started fresh
    has none, even when it names a parent (a born-again generation starts
    its own epochs). The status also records the epoch of the checkpoint
    resumed from (``None`` before it did). Runs written before resume
    status fall back to the lineage in their ``run.yaml``. No checkpoint
    and no history is read."""
    import yaml

    from .nn.params.nn_run import _load_resume_status, _read_metadata

    run_path = _run_path(run_id, root)
    status = _load_resume_status(_read_metadata(os.path.join(run_path, "metadata.yaml")))
    if status is not None:
        if status.mode == "fresh" or status.source_run_id is None:
            return None, None, None
        return status.source_run_id, status.source_checkpoint or "last", status.source_epoch
    path = os.path.join(run_path, "run.yaml")
    with open(path, encoding="utf-8") as handle:
        state = yaml.safe_load(handle)
    if not isinstance(state, dict):
        raise ValueError(f"malformed run.yaml at {path}: expected a mapping")
    for section in (state.get("trainer"), state.get("train")):
        if isinstance(section, dict) and section.get("parent_run_id") is not None:
            return str(section["parent_run_id"]), section.get("parent_checkpoint") or "last", None
    return None, None, None


def _resume_epoch(
    parent_id: str, checkpoint: Any, root: Optional[str], parent_committed: Optional[int]
) -> Optional[int]:
    """The epoch of the parent checkpoint a child resumed from, or ``None``
    when it can no longer be read. A resume from ``last`` reuses the
    parent's committed epoch (its LAST) instead of reading it again."""
    from .nn.nn_model import _resume_checkpoint_type
    from .nn.params.nn_checkpoint import NNCheckpoint

    if str(checkpoint) == "last" and parent_committed is not None:
        return parent_committed if parent_committed >= 0 else None
    try:
        loaded = NNCheckpoint.load(run=parent_id, type=_resume_checkpoint_type(checkpoint), root=root)
    except Exception:  # noqa: BLE001 — unreadable for any reason: the documented fallback applies
        return None
    return None if loaded is None else loaded.idp.epoch_idx


def iter_history(run_id: str, root: Optional[str] = None, *, lineage: bool = False) -> Iterator[NNIterationDataPoint]:
    """Stream a run's committed history in order: a journal chunk by chunk
    (each checked against its SHA-256), a legacy ``idps.csv`` whole (floats
    parsed round-trip, so its export reproduces the file). Records past the
    LAST checkpoint's epoch are never yielded.

    With ``lineage=True`` a resumed run is preceded by the committed records
    of the run it resumed from, up to and including the epoch it resumed
    from (recorded in its resume status; for runs recorded before that, the
    epoch of that checkpoint now) and never past the run's own first record,
    recursively: each epoch once, never the parent's later epochs. When
    that epoch cannot be known (the checkpoint, or the parent's LAST, is
    unreadable), the parent's records before the run's first record are
    used, with a ``RuntimeWarning``; when neither is known, or the parent
    run is gone, the lineage starts at the run, with a ``RuntimeWarning``.
    Lineage follows resumes: a run that started fresh (a born-again
    generation naming its teacher's run as ``parent_run_id``, say) starts
    its own epochs, and its lineage is its own history.
    """
    yield from _lineage(run_id, root, lineage, seen=set(), until=None, committed=_UNKNOWN, readers={})


_UNKNOWN: Any = object()


def _lineage(
    run_id: str,
    root: Optional[str],
    lineage: bool,
    *,
    seen: set[str],
    until: Optional[int],
    committed: Any,
    readers: dict[str, Optional[_JournalReader]],
) -> Iterator[NNIterationDataPoint]:
    from .nn.params.nn_run import _committed_epoch

    seen.add(run_id)
    if committed is _UNKNOWN:
        committed = _committed_epoch(run_id, _run_path(run_id, root), root)
    if lineage:
        parent, checkpoint, source_epoch = _parent(run_id, root)
        if parent is not None and parent not in seen:
            yield from _parent_prefix(
                run_id,
                parent,
                checkpoint,
                source_epoch,
                root,
                seen=seen,
                until=until,
                committed=committed,
                readers=readers,
            )
    for record in _own_records(run_id, root, committed, readers):  # read once the parent's prefix is done
        if until is not None and record.epoch_idx >= until:
            return
        yield record


def _parent_prefix(
    run_id: str,
    parent: str,
    checkpoint: Any,
    source_epoch: Optional[int],
    root: Optional[str],
    *,
    seen: set[str],
    until: Optional[int],
    committed: Optional[int],
    readers: dict[str, Optional[_JournalReader]],
) -> Iterator[NNIterationDataPoint]:
    """The parent's records that precede ``run_id``: up to the epoch it
    resumed from, never past its own first record or a descendant's cut."""
    from .nn.params.nn_run import _committed_epoch

    if parent == "best":
        warnings.warn(
            f"run {run_id} resumed from the runs/best pointer, which may name another run by now: its lineage "
            f"starts at {run_id}",
            RuntimeWarning,
            stacklevel=4,
        )
        return
    parent_path = _run_path(parent, root)
    if not os.path.isfile(os.path.join(parent_path, "run.yaml")):
        warnings.warn(
            f"run {run_id} resumed from run {parent}, which is no longer saved: its lineage starts at {run_id}",
            RuntimeWarning,
            stacklevel=4,
        )
        return
    try:
        parent_committed: Optional[int] = _committed_epoch(parent, parent_path, root)
    except Exception:  # noqa: BLE001 — a corrupt LAST: fall back to the child's first record
        parent_committed = -1
    if parent_committed == -1:
        # The parent's LAST is gone or unreadable, yet the child resumed from
        # a committed epoch: its records are read unfiltered and cut at the
        # recorded branch epoch, or else before the child's first record.
        parent_committed, checkpoint = None, None
    resumed = (
        source_epoch
        if source_epoch is not None
        else (None if checkpoint is None else _resume_epoch(parent, checkpoint, root, parent_committed))
    )
    first = _first_epoch(run_id, root, committed, readers)
    if resumed is None and first is None:
        warnings.warn(
            f"cannot tell where run {run_id} branched from run {parent} (the checkpoint it resumed from is "
            f"unreadable and {run_id} has no committed record): its lineage starts at {run_id}",
            RuntimeWarning,
            stacklevel=4,
        )
        return
    if resumed is None:
        warnings.warn(
            f"the checkpoint run {run_id} resumed from (or run {parent}'s LAST) is unreadable: its lineage uses "
            f"run {parent}'s records "
            f"before {run_id}'s first record (epoch {first})",
            RuntimeWarning,
            stacklevel=4,
        )
    # Each epoch comes from one run: the parent's own records stop where this
    # run — or a descendant below it — takes over.
    bounds = [bound for bound in (None if resumed is None else resumed + 1, first, until) if bound is not None]
    yield from _lineage(parent, root, True, seen=seen, until=min(bounds), committed=parent_committed, readers=readers)


def export_history_csv(
    run_id: str, path: Union[str, os.PathLike[str]], root: Optional[str] = None, *, lineage: bool = False
) -> int:
    """Write a run's committed history (see :func:`iter_history`) to ``path``
    in the legacy ``idps.csv`` layout — the same columns, order and index
    ``NNRun.save`` writes — atomically. Returns the number of records. The
    export holds the records in memory while it builds the table; stream
    :func:`iter_history` for histories that do not fit."""
    import pandas as pd

    from .nn.params.nn_run import _atomic_write_text

    records = [record.state() for record in iter_history(run_id, root, lineage=lineage)]
    _atomic_write_text(os.fspath(path), pd.json_normalize(data=records).to_csv())
    return len(records)


def migrate_history(
    run_id: str,
    root: Optional[str] = None,
    *,
    spec: Optional[HistoryJournal] = None,
    discard_uncommitted: bool = False,
) -> None:
    """Move a legacy run's ``idps.csv`` into a history journal, explicitly.

    The committed records are written as a journal (with per-epoch summary
    rows), published and read back, and only then is ``idps.csv`` removed;
    ``run.yaml`` — and so the run id — is untouched. A leftover journal from
    an interrupted migration (never published) is replaced. Refused: a run
    that already keeps a journal, a run that is being trained (its lease is
    held), and — unless ``discard_uncommitted=True`` — a CSV holding records
    past the LAST checkpoint's epoch, which readers hide but a migration
    would delete.
    """
    from filelock import FileLock, Timeout

    from .nn.params.nn_run import _lease_path, _runs_root

    spec = spec or HistoryJournal()
    if run_id == "best":
        raise ValueError("migrate a run by its own id, not through the runs/best pointer")
    run_path = _run_path(run_id, root)
    if not os.path.isfile(os.path.join(run_path, "run.yaml")):
        raise ValueError(f"no saved run {run_id} under {_runs_root(root)}")
    lease_path = _lease_path(_runs_root(root), run_id)  # the lock a training session holds
    os.makedirs(os.path.dirname(lease_path), exist_ok=True)
    lease = FileLock(lease_path, timeout=0)
    try:
        lease.acquire()
    except Timeout:
        raise RuntimeError(f"run {run_id} is being trained; migrate its history once training ends") from None
    try:
        with FileLock(os.path.join(run_path, ".run.lock")):
            _migrate_locked(run_id, run_path, root, spec, discard_uncommitted=discard_uncommitted)
    finally:
        lease.release()


def _reads_back(records: Iterator[NNIterationDataPoint], expected: Sequence[bytes]) -> bool:
    """Whether ``records`` encode, one by one, to exactly ``expected``."""
    count = 0
    for count, record in enumerate(records, start=1):
        if count > len(expected) or _record_line(record) != expected[count - 1]:
            return False
    return count == len(expected)


def _migrate_locked(
    run_id: str, run_path: str, root: Optional[str], spec: HistoryJournal, *, discard_uncommitted: bool
) -> None:
    import shutil

    from .nn.params.nn_run import _committed_epoch, _read_idps_csv

    csv_path = os.path.join(run_path, "idps.csv")
    directory = os.path.join(run_path, HISTORY_DIR)
    if has_journal(run_path) and not os.path.isfile(csv_path):
        raise ValueError(f"run {run_id} already keeps its history in a journal")
    if not os.path.isfile(csv_path):
        raise ValueError(f"run {run_id} has no idps.csv to migrate")
    # Floats parsed round-trip: the journal holds exactly the values the CSV
    # was written from.
    records = _read_idps_csv(csv_path, exact=True)
    committed = _committed_epoch(run_id, run_path, root)
    visible = [r for r in records if committed is None or r.epoch_idx <= committed]
    if len(visible) != len(records) and not discard_uncommitted:
        raise ValueError(
            f"idps.csv of run {run_id} holds {len(records) - len(visible)} records past the LAST checkpoint's epoch "
            f"({committed}); readers hide them, but migrating would delete them with the CSV — pass "
            "discard_uncommitted=True to migrate only the committed records"
        )
    records = visible
    expected = [_record_line(record) for record in records]  # NaN-safe: compared as the lines written
    if has_journal(run_path):
        # A migration interrupted after publishing: its journal was checked
        # before it was published — finish once it still matches the CSV.
        if not _reads_back(_JournalReader(directory).records(None), expected):
            raise ValueError(f"run {run_id} keeps both a history journal and an idps.csv that differ; remove one")
        os.remove(csv_path)
        return
    if os.path.lexists(directory):  # an unpublished leftover of an interrupted migration
        shutil.rmtree(directory)
    writer = _JournalWriter(directory, spec)
    writer._ensure()
    try:
        for position, record in enumerate(records):
            writer.append(record, expected[position])
            if position + 1 == len(records) or records[position + 1].epoch_idx != record.epoch_idx:
                writer.end_epoch(sync=False)
        writer.sync()  # once, before the journal is checked and published
        # The journal must read back as the CSV's records before it is
        # published and the CSV goes.
        if not _reads_back(_JournalReader(directory, manifest=writer.manifest()).records(None), expected):
            raise HistoryCorruptionError(f"the migrated journal of run {run_id} does not read back as its idps.csv")
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    writer.publish()
    os.remove(csv_path)
