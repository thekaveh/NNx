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

**Commit protocol.** Each epoch's records are written and the manifest is
published *before* the LAST checkpoint, which stays the epoch's commit
marker. Readers show only records up to LAST's epoch: a crash while a chunk
is written or before the manifest is replaced leaves the previous manifest,
and a crash before LAST is replaced leaves an uncommitted tail that is
ignored. A committed chunk whose bytes no longer match its SHA-256 raises
:class:`HistoryCorruptionError`.

**Callbacks.** Built-in callbacks read ``ctx.idp`` (the epoch's last record),
so the window changes nothing for them; ``LRMonitor.history`` keeps the same
window. A ``Callback`` that needs the whole history sets
``history_access = "full"`` and gets the materialised history as ``ctx.idps``
at ``on_epoch_end`` and ``on_train_end`` (read back from the journal: its cost
grows with the run). A plain function callback — ``callbacks=[lambda idps:
...]``, which by contract receives the full list — is refused before
training, as is an unknown ``history_access``.

**Lineage.** A resumed run owns its own run directory and chunk files; its
parent is never written. ``iter_history(run_id, lineage=True)`` yields the
parent's committed records that precede the child's first epoch (following
``resume_from_run_id`` / ``parent_run_id``, recursively), then the child's —
never the parent prefix twice.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import os
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
            window).
    """

    retention: int = 1000
    chunk_size: int = 250

    def __post_init__(self) -> None:
        for name in ("retention", "chunk_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"HistoryJournal.{name} must be a positive integer, got {value!r}")
        if self.chunk_size > self.retention:
            raise ValueError(
                f"HistoryJournal.chunk_size ({self.chunk_size}) must not exceed retention ({self.retention}): "
                "records waiting for their chunk stay in memory"
            )


# --- encoding --------------------------------------------------------------------------------


def _record_line(record: NNIterationDataPoint) -> bytes:
    return json.dumps(record.state(), separators=(",", ":"), allow_nan=True).encode("utf-8") + b"\n"


def _record(line: bytes) -> NNIterationDataPoint:
    from .nn.params.nn_checkpoint import _idp_from_nested_state

    return _idp_from_nested_state(json.loads(line))


def _json_line(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=True).encode("utf-8") + b"\n"


def _chain(previous: str, line: bytes) -> str:
    return hashlib.sha256(previous.encode("ascii") + line).hexdigest()


def _chunk_name(seq: int) -> str:
    return f"chunk-{seq:08d}.jsonl"


def _nan_to_none(value: Optional[float]) -> Optional[float]:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


# --- the per-epoch summary row (the notebook chart's series) ------------------------------------


class _EpochStats:
    """Running per-batch loss / error means of the epoch being written, in
    the order ``NNRun._epoch_series`` sums them."""

    def __init__(self) -> None:
        self._reset(None)

    def _reset(self, epoch: Optional[int]) -> None:
        self.epoch = epoch
        self.records = 0
        self.loss_sum = 0.0
        self.loss_n = 0
        self.err_sum = 0.0
        self.err_n = 0

    def add(self, record: NNIterationDataPoint) -> None:
        if self.epoch != record.epoch_idx:
            self._reset(record.epoch_idx)
        self.records += 1
        train = record.train_edp
        if train is not None and train.loss is not None:
            self.loss_sum += train.loss
            self.loss_n += 1
        if train is not None and train.error is not None:
            self.err_sum += train.error
            self.err_n += 1

    def row(self, last: NNIterationDataPoint) -> dict[str, Any]:
        nan = float("nan")
        summary = last.train_summary
        if summary is not None:
            train_loss = summary.loss if summary.loss is not None else nan
            train_err = summary.error if summary.error is not None else nan
        else:
            train_loss = self.loss_sum / self.loss_n if self.loss_n else nan
            train_err = self.err_sum / self.err_n if self.err_n else nan
        val = last.val_edp
        record = last.selection
        return {
            "epoch": last.epoch_idx,
            "records": self.records,
            "train_loss": _nan_to_none(train_loss),
            "train_err": _nan_to_none(train_err),
            "val_loss": _nan_to_none(val.loss) if val is not None else None,
            "val_err": _nan_to_none(val.error) if val is not None else None,
            "monitor": _nan_to_none(record.value) if record is not None else None,
            "monitor_key": record.monitor.key if record is not None else None,
            "improved": bool(record is not None and record.improved),
        }


# --- writing ---------------------------------------------------------------------------------


def _empty_manifest(spec: HistoryJournal) -> dict[str, Any]:
    return {
        "format": JOURNAL_FORMAT,
        "version": JOURNAL_VERSION,
        "retention": spec.retention,
        "chunk_size": spec.chunk_size,
        "chunks": 0,
        "records": 0,
        "index_bytes": 0,
        "index_chain": _GENESIS,
        "epochs": 0,
        "epochs_bytes": 0,
    }


def _manifest_text(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, sort_keys=True, indent=1) + "\n").encode("utf-8")


def _append_bytes(path: str, data: bytes) -> None:
    with open(path, "ab") as handle:
        handle.write(data)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass


def _write_chunk_file(path: str, data: bytes) -> None:
    """A chunk is written once: a temporary file, fsynced, renamed."""
    temporary = f"{path}.tmp"
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    os.replace(temporary, path)


class _JournalWriter:
    """Appends one run's records as immutable chunks and publishes the
    manifest; memory is bounded by ``chunk_size`` records (and one index
    line per chunk)."""

    def __init__(self, directory: str, spec: HistoryJournal) -> None:
        self.directory = directory
        self.spec = spec
        self.seq = 0
        self.records = 0
        self.index_bytes = 0
        self.index_chain = _GENESIS
        self.epochs = 0
        self.epochs_bytes = 0
        self.chunks: list[str] = []  # this writer's chunk files, for full-history callbacks
        self._pending: list[NNIterationDataPoint] = []
        self._stats = _EpochStats()
        self._published: Optional[bytes] = None
        self._previous: Optional[bytes] = None

    def _ensure(self) -> None:
        os.makedirs(self.directory, exist_ok=True)

    def append(self, record: NNIterationDataPoint) -> None:
        self._pending.append(record)
        self._stats.add(record)
        if len(self._pending) > self.spec.chunk_size:  # all but the newest record are final
            self._write_chunk(self._pending[: self.spec.chunk_size])
            del self._pending[: self.spec.chunk_size]

    def replace_last(self, record: NNIterationDataPoint) -> None:
        self._pending[-1] = record

    def _write_chunk(self, records: Sequence[NNIterationDataPoint]) -> None:
        self._ensure()
        data = b"".join(_record_line(record) for record in records)
        name = _chunk_name(self.seq)
        _write_chunk_file(os.path.join(self.directory, name), data)
        entry = {
            "seq": self.seq,
            "file": name,
            "records": len(records),
            "first_iter": records[0].iter_idx,
            "last_iter": records[-1].iter_idx,
            "first_epoch": records[0].epoch_idx,
            "last_epoch": records[-1].epoch_idx,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        line = _json_line(entry)
        _append_bytes(os.path.join(self.directory, _INDEX), line)
        self.index_chain = _chain(self.index_chain, line)
        self.index_bytes += len(line)
        self.records += len(records)
        self.chunks.append(name)
        self.seq += 1

    def end_epoch(self) -> None:
        """Write the epoch's remaining records (all final now) and its
        summary row; the manifest is published separately."""
        last = self._pending[-1] if self._pending else None
        if last is None:
            return
        row = self._stats.row(last)
        while self._pending:
            self._write_chunk(self._pending[: self.spec.chunk_size])
            del self._pending[: self.spec.chunk_size]
        line = _json_line(row)
        _append_bytes(os.path.join(self.directory, _EPOCHS), line)
        self.epochs += 1
        self.epochs_bytes += len(line)

    def manifest(self) -> dict[str, Any]:
        return {
            **_empty_manifest(self.spec),
            "chunks": len(self.chunks),
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
        _atomic_write_text(os.path.join(self.directory, _MANIFEST), text.decode("utf-8"))
        self._previous, self._published = self._published, text

    def rollback(self) -> None:
        """Republish the manifest of the previous epoch (the current epoch
        could not commit) — an empty journal's before the first commit, as
        an eager run rolls back to an empty ``idps.csv``."""
        from .nn.params.nn_run import _atomic_write_text

        previous = self._previous if self._previous is not None else _manifest_text(_empty_manifest(self.spec))
        _atomic_write_text(os.path.join(self.directory, _MANIFEST), previous.decode("utf-8"))

    def materialize(self) -> list[NNIterationDataPoint]:
        """Every record this writer has seen, read back from its chunks,
        plus those still waiting for one."""
        records: list[NNIterationDataPoint] = []
        for name in self.chunks:
            with open(os.path.join(self.directory, name), "rb") as handle:
                records.extend(_record(line) for line in handle if line.strip())
        return records + list(self._pending)


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
    other; chunks are read (and checked) on demand."""

    def __init__(self, directory: str) -> None:
        self.directory = directory
        path = os.path.join(directory, _MANIFEST)
        try:
            with open(path, encoding="utf-8") as handle:
                manifest = json.load(handle)
        except FileNotFoundError:
            raise HistoryCorruptionError(f"the history journal at {directory} has no {_MANIFEST}") from None
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
        self.chunks = self._read_index()

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
                    file=_chunk_name(entry["seq"]),
                    records=entry["records"],
                    first_epoch=entry["first_epoch"],
                    last_epoch=entry["last_epoch"],
                    sha256=entry["sha256"],
                )
                for entry in map(json.loads, lines)
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise HistoryCorruptionError(f"the committed index {path} is malformed: {exc}") from None
        if [chunk.seq for chunk in chunks] != list(range(len(chunks))):
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
        window: collections.deque[NNIterationDataPoint] = collections.deque(maxlen=count)
        needed, start = 0, len(self.chunks)
        for position in range(len(self.chunks) - 1, -1, -1):
            chunk = self.chunks[position]
            if max_epoch is not None and chunk.first_epoch > max_epoch:
                continue  # entirely uncommitted
            start = position
            needed += chunk.records
            if needed >= count:
                break
        for chunk in self.chunks[start:]:
            if max_epoch is not None and chunk.first_epoch > max_epoch:
                break
            window.extend(
                record for record in self.read_chunk(chunk) if max_epoch is None or record.epoch_idx <= max_epoch
            )
        return list(window)

    def first_epoch(self) -> Optional[int]:
        return self.chunks[0].first_epoch if self.chunks else None

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
    series: dict[str, list[Any]] = {
        name: [] for name in ("epochs", "train_loss", "train_err", "val_loss", "val_err", "monitor", "improved")
    }
    for row in _JournalReader(directory).epoch_rows(max_epoch):
        series["epochs"].append(row["epoch"])
        for name in ("train_loss", "train_err", "val_loss", "val_err", "monitor"):
            series[name].append(nan if row[name] is None else row[name])
        series["improved"].append(bool(row["improved"]))
    return series


def has_journal(run_path: str) -> bool:
    """Whether a run directory keeps its history in a published journal."""
    return os.path.isfile(os.path.join(run_path, HISTORY_DIR, _MANIFEST))


def _committed_epoch(run_id: str, run_path: str, root: Optional[str]) -> Optional[int]:
    """The LAST checkpoint's epoch for runs under the commit protocol (``-1``
    before the first commit); ``None`` (no filter) for runs written before it."""
    from .nn.enum.checkpoints import Checkpoints
    from .nn.params.nn_checkpoint import NNCheckpoint
    from .nn.params.nn_run import _HISTORY_PROTOCOL_FILE

    if not os.path.isfile(os.path.join(run_path, _HISTORY_PROTOCOL_FILE)):
        return None
    last = NNCheckpoint.load(run=run_id, type=Checkpoints.LAST, root=root)
    last_path = os.path.join(run_path, "checkpoints", f"{Checkpoints.LAST}.pt")
    if last is None and os.path.isfile(last_path):
        raise ValueError(f"malformed LAST checkpoint at {last_path}")  # as NNRun.load
    return -1 if last is None else last.idp.epoch_idx


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

    def __bool__(self) -> bool:
        return bool(self.records)

    def window(self) -> list[NNIterationDataPoint]:
        return self.records

    def full(self) -> list[NNIterationDataPoint]:
        return self.records

    def save_epoch(self, run: NNRun) -> None:
        run.with_idps(self.records).save(update_best=False)

    def rollback_epoch(self, run: NNRun) -> None:
        run.with_idps(self.records[: self._epoch_start]).save(update_best=False)

    def finish(self, run: NNRun) -> NNRun:
        return run.with_idps(self.records).save()


class _JournalHistory:
    """The last ``retention`` records in memory, every record in the run's
    journal."""

    def __init__(self, run: NNRun, spec: HistoryJournal, root: Optional[str] = None) -> None:
        from .nn.params.nn_run import _runs_root

        self.spec = spec
        self.directory = os.path.join(_runs_root(root), run.id, HISTORY_DIR)
        self._window: collections.deque[NNIterationDataPoint] = collections.deque(maxlen=spec.retention)
        self._writer = _JournalWriter(self.directory, spec)
        self._epoch_records = 0

    def begin_epoch(self) -> None:
        self._epoch_records = 0

    def epoch_is_empty(self) -> bool:
        return self._epoch_records == 0

    def append(self, record: NNIterationDataPoint) -> None:
        self._window.append(record)
        self._writer.append(record)
        self._epoch_records += 1

    @property
    def last(self) -> NNIterationDataPoint:
        return self._window[-1]

    def replace_last(self, record: NNIterationDataPoint) -> None:
        self._window[-1] = record
        self._writer.replace_last(record)

    def __bool__(self) -> bool:
        return bool(self._window)

    def window(self) -> list[NNIterationDataPoint]:
        return list(self._window)

    def full(self) -> list[NNIterationDataPoint]:
        return self._writer.materialize()

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


def _wants_full(callback: Any, history: Any) -> bool:
    return isinstance(history, _JournalHistory) and getattr(callback, "history_access", "window") == "full"


def _dispatch_epoch_end(callbacks: Sequence[Any], ctx: Any, history: TrainingHistory) -> None:
    """``on_epoch_end`` for every callback: ``ctx.idps`` is the history's
    window, or — for a callback declaring ``history_access = "full"`` —
    the whole history, read back once per epoch."""
    window = history.window()
    full: Optional[list[NNIterationDataPoint]] = None
    for callback in callbacks:
        if _wants_full(callback, history):
            if full is None:
                full = history.full()
            ctx.idps = full
        else:
            ctx.idps = window
        callback.on_epoch_end(ctx)
    ctx.idps = window


def _idps_view(callback: Any, history: Optional[TrainingHistory]) -> Optional[list[NNIterationDataPoint]]:
    """``ctx.idps`` for one callback at ``on_train_end`` in a journal run: the
    window, or the whole history for ``"full"``. ``None`` — leave ``ctx.idps``
    as the epoch loop left it — for an eager run, exactly as before."""
    if not isinstance(history, _JournalHistory):
        return None
    return history.full() if _wants_full(callback, history) else history.window()


# --- reading a run's history -----------------------------------------------------------------


def _run_path(run_id: str, root: Optional[str]) -> str:
    from .nn.params.nn_run import _runs_root, _validate_run_id

    return os.path.join(_runs_root(root), _validate_run_id(run_id))


def _own_records(run_id: str, root: Optional[str]) -> tuple[Iterator[NNIterationDataPoint], Optional[int]]:
    """A run's committed records (lazily for a journal) and its first epoch."""
    from .nn.params.nn_run import _read_idps_csv

    run_path = _run_path(run_id, root)
    committed = _committed_epoch(run_id, run_path, root)
    if has_journal(run_path):
        reader = _JournalReader(os.path.join(run_path, HISTORY_DIR))
        return reader.records(committed), reader.first_epoch()
    # A legacy CSV run is read whole, floats round-trip (its export
    # reproduces the file).
    records = [
        record
        for record in _read_idps_csv(os.path.join(run_path, "idps.csv"), exact=True)
        if committed is None or record.epoch_idx <= committed
    ]
    return iter(records), records[0].epoch_idx if records else None


def iter_history(run_id: str, root: Optional[str] = None, *, lineage: bool = False) -> Iterator[NNIterationDataPoint]:
    """Stream a run's committed history in order: a journal chunk by chunk
    (each checked against its SHA-256), a legacy ``idps.csv`` whole (floats
    parsed round-trip, so its export reproduces the file). Records past the
    LAST checkpoint's epoch are never yielded.

    With ``lineage=True`` a resumed run (``resume_from_run_id`` /
    ``parent_run_id``) is preceded by its parent's committed records up to
    — not including — its own first epoch, recursively: each record once.
    """
    yield from _lineage(run_id, root, lineage, seen=set(), until=None)


def _lineage(
    run_id: str, root: Optional[str], lineage: bool, *, seen: set[str], until: Optional[int]
) -> Iterator[NNIterationDataPoint]:
    from .nn.params.nn_run import NNRun

    seen.add(run_id)
    records, first = _own_records(run_id, root)
    if lineage:
        run = NNRun._load(run_id, root, records=False)
        source = run.trainer if run.trainer is not None else run.train
        parent = source.resume_from_run_id or source.parent_run_id
        if parent is not None and parent not in seen:
            # The parent contributes only what precedes this run (and what
            # its own child needs): each epoch comes from one run.
            bounds = [bound for bound in (first, until) if bound is not None]
            yield from _lineage(parent, root, lineage, seen=seen, until=min(bounds) if bounds else None)
    for record in records:
        if until is not None and record.epoch_idx >= until:
            return
        yield record


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


def migrate_history(run_id: str, root: Optional[str] = None, *, spec: Optional[HistoryJournal] = None) -> None:
    """Move a legacy run's ``idps.csv`` into a history journal, explicitly.

    The committed records are written as a journal (with per-epoch summary
    rows) and published, then ``idps.csv`` is removed; ``run.yaml`` — and so
    the run id — is untouched. A run that already keeps a journal, and one
    that is being trained (its lease is held), are refused.
    """
    from filelock import FileLock, Timeout

    from .nn.params.nn_run import _runs_root

    spec = spec or HistoryJournal()
    run_path = _run_path(run_id, root)
    if not os.path.isfile(os.path.join(run_path, "run.yaml")):
        raise ValueError(f"no saved run {run_id} under {_runs_root(root)}")
    leases = os.path.join(_runs_root(root), ".leases")
    os.makedirs(leases, exist_ok=True)
    lease = FileLock(os.path.join(leases, f"{run_id}.lock"), timeout=0)  # training holds it
    try:
        lease.acquire()
    except Timeout:
        raise RuntimeError(f"run {run_id} is being trained; migrate its history once training ends") from None
    try:
        _migrate_locked(run_id, run_path, root, spec)
    finally:
        lease.release()


def _migrate_locked(run_id: str, run_path: str, root: Optional[str], spec: HistoryJournal) -> None:
    from filelock import FileLock

    from .nn.params.nn_run import _read_idps_csv

    with FileLock(os.path.join(run_path, ".run.lock")):
        if has_journal(run_path):
            raise ValueError(f"run {run_id} already keeps its history in a journal")
        csv_path = os.path.join(run_path, "idps.csv")
        if not os.path.isfile(csv_path):
            raise ValueError(f"run {run_id} has no idps.csv to migrate")
        # The committed records, floats parsed round-trip: the journal holds
        # exactly the values the CSV was written from.
        committed = _committed_epoch(run_id, run_path, root)
        records = [r for r in _read_idps_csv(csv_path, exact=True) if committed is None or r.epoch_idx <= committed]
        writer = _JournalWriter(os.path.join(run_path, HISTORY_DIR), spec)
        writer._ensure()
        for position, record in enumerate(records):
            writer.append(record)
            ends_epoch = position + 1 == len(records) or records[position + 1].epoch_idx != record.epoch_idx
            if ends_epoch:
                writer.end_epoch()
        writer.publish()  # the journal is complete before the CSV goes
        os.remove(csv_path)
