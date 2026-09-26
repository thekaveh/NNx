"""Versioned experiment provenance manifests (FEAT-019).

A run has four distinct identities, and this module adds the two that were
missing:

- ``run.id`` — ``md5`` of the run's configuration (``NNRun.state()``):
  *which configuration* ran. Unchanged, and never affected by provenance.
- **experiment fingerprint** — SHA-256 of an :class:`ExperimentManifest`'s
  canonical bytes: the *declared intent* — task, label order, model
  descriptor, data and split identities, objective version and any other
  configuration. Two attempts at one plan share it.
- **attempt id** — a fresh id for every ``train()`` call: *this execution*.
  Its record links the parent attempt and checkpoint generation on resume
  and ends ``completed``, ``failed`` or ``cancelled`` with the last
  committed checkpoint.
- **environment snapshot** — ``metadata.yaml`` (library versions, GPU, OS,
  git commit): *where* it ran. Rewritten on every save; not provenance.

Provenance is opt-in: ``NNModel.train(..., provenance=manifest)`` (and
``Trainer.train``) writes ``runs/<id>/provenance.json`` (the manifest and
its fingerprint) and ``runs/<id>/attempt.json`` atomically; ``NNRun.load``
exposes them as ``run.provenance`` — ``None`` when a run has none, which is
*absent*, never equal.

**Canonical bytes.** UTF-8 JSON with sorted keys, arrays in order, no
insignificant whitespace, and the format version inside the hashed
document. Only JSON-like values serialize: non-finite floats, sets, bytes,
callables and other objects are rejected — never replaced by their ``repr``.

**Identity references.** A data or split identity is
:meth:`IdentityRef.declared` (a name you supply — recorded, never reported
as verified), a content digest from the explicit :func:`hash_file` /
:func:`hash_bytes` calls, or :meth:`IdentityRef.unknown`. Fingerprinting and
:func:`compare` iterate no loader, build no model, open no socket and hash
no file.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import uuid
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from .nn.params.nn_run import NNRun

__all__ = [
    "FORMAT",
    "Attempt",
    "ExperimentManifest",
    "FieldComparison",
    "IdentityRef",
    "ProvenanceComparison",
    "ProvenanceRecord",
    "canonical_bytes",
    "compare",
    "hash_bytes",
    "hash_file",
    "load_provenance",
]

FORMAT = "nnx.provenance/1"
"""Format version, part of every hashed document."""

# NNTrainParams.state() keys that describe one attempt, not the plan.
_ATTEMPT_ONLY_TRAIN_KEYS = ("n_epochs", "parent_run_id", "parent_checkpoint")

MANIFEST_FILE = "provenance.json"
ATTEMPT_FILE = "attempt.json"
ATTEMPT_STATUSES = ("running", "completed", "failed", "cancelled")


# --- canonical serialization ---------------------------------------------------------------


def _canonical(value: Any, path: str) -> Any:
    if isinstance(value, IdentityRef):
        return value.state()
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        real = float(value)
        if not math.isfinite(real):
            raise ValueError(f"{path}: non-finite number {value!r} cannot be serialized")
        return real
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path}: mapping keys must be strings, got {key!r}")
            out[key] = _canonical(item, f"{path}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [_canonical(item, f"{path}[{index}]") for index, item in enumerate(value)]
    kind = "a callable" if callable(value) else f"a {type(value).__name__}"
    raise TypeError(
        f"{path}: {kind} is not JSON-like (None, bool, int, finite float, str, list or str-keyed mapping); "
        "sets, bytes, callables and other objects are rejected, never serialized by their repr"
    )


def canonical_bytes(value: Any) -> bytes:
    """Canonical UTF-8 JSON of a JSON-like ``value``: sorted keys, arrays in
    order, compact separators, Unicode kept as UTF-8. Raises ``TypeError`` /
    ``ValueError`` (naming the path) for anything else."""
    return json.dumps(
        _canonical(value, "$"), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- identity references ---------------------------------------------------------------------


@dataclass(frozen=True)
class IdentityRef:
    """How a data or split identity is known.

    - ``declared`` — a name you supplied (a dataset id, a split name): it is
      recorded and compared, but **never reported as verified**;
    - ``digest`` — a content digest NNx computed (:func:`hash_file`,
      :func:`hash_bytes`): equal digests are verified equality;
    - ``unknown`` — not known.
    """

    kind: str
    value: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in ("declared", "digest", "unknown"):
            raise ValueError(f"IdentityRef kind must be 'declared', 'digest' or 'unknown', got {self.kind!r}")
        if self.kind == "unknown":
            if self.value is not None:
                raise ValueError("an unknown IdentityRef has no value")
        elif not isinstance(self.value, str) or not self.value:
            raise ValueError(f"a {self.kind} IdentityRef needs a non-empty string value, got {self.value!r}")

    @classmethod
    def declared(cls, value: str) -> IdentityRef:
        return cls("declared", value)

    @classmethod
    def unknown(cls) -> IdentityRef:
        return cls("unknown")

    @property
    def verified(self) -> bool:
        """Whether this identity is a computed content digest."""
        return self.kind == "digest"

    def state(self) -> dict[str, Any]:
        return {"identity": self.kind, "value": self.value}

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> IdentityRef:
        return IdentityRef(state["identity"], state.get("value"))


def hash_bytes(data: bytes) -> IdentityRef:
    """A verified ``sha256:<hex>`` identity of ``data``."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"hash_bytes expects bytes, got {type(data).__name__}")
    return IdentityRef("digest", f"sha256:{_sha256(bytes(data))}")


def hash_file(path: Union[str, os.PathLike[str]], *, chunk_size: int = 1 << 20) -> IdentityRef:
    """A verified ``sha256:<hex>`` identity of a file's bytes — the one call
    here that reads data, and only when you make it."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return IdentityRef("digest", f"sha256:{digest.hexdigest()}")


def _is_split_manifest(value: Any) -> bool:
    from .data_splits import SplitManifest

    return isinstance(value, SplitManifest)


def _ref(value: Any, what: str, *, allow_split: bool = False) -> IdentityRef:
    """One identity from an ``IdentityRef``, a declared id string (never
    verified), ``None`` (unknown) or — when allowed — a ``SplitManifest``
    (its digest). Shared by manifests and preprocessing."""
    if isinstance(value, IdentityRef):
        return value
    if value is None:
        return IdentityRef.unknown()
    if isinstance(value, str):
        return IdentityRef.declared(value)  # a supplied id is declared, never verified
    if allow_split and _is_split_manifest(value):
        return value.identity()  # FEAT-017: the plan's digest, the same after replay
    accepted = "an IdentityRef, a declared id string or None" + (
        ", or a nnx.data_splits.SplitManifest" if allow_split else ""
    )
    raise TypeError(f"{what} must be {accepted}, got {value!r}")


def _refs(values: Optional[Mapping[str, Any]], *, owner: str) -> dict[str, IdentityRef]:
    out: dict[str, IdentityRef] = {}
    for name, value in (values or {}).items():
        if not isinstance(name, str) or not name:
            raise TypeError(f"{owner} names must be non-empty strings, got {name!r}")
        out[name] = _ref(value, f"{owner}[{name!r}]", allow_split=owner == "splits")
    return dict(sorted(out.items()))


# --- the manifest --------------------------------------------------------------------------


def _freeze(value: Any) -> Any:
    """Immutable copy of a canonical JSON-like value."""
    from ._config import _FrozenConfig

    if isinstance(value, Mapping):
        return _FrozenConfig({key: _freeze(item) for key, item in sorted(value.items())})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, eq=False)
class ExperimentManifest:
    """The declared intent of an experiment. Everything is JSON-like and
    immutable; :meth:`fingerprint` is the SHA-256 of its canonical bytes
    (with :data:`FORMAT`), so changing the task, the label order, a split's
    digest or the objective's version changes it.

    Values are validated and copied into immutable form at construction, so
    later changes to the dicts you passed never change the manifest;
    equality and hashing follow the canonical bytes (``1`` and ``1.0``
    differ, as they do in the fingerprint).

    Args:
        task: the task declaration (e.g. ``TaskSpec.state()``).
        labels: the ordered output labels.
        model: the model descriptor (e.g. ``NNModelParams.state()``).
        data: named data identities (:class:`IdentityRef`, a declared id
            string, or ``None`` for unknown).
        splits: named split identities, likewise; a
            :class:`nnx.data_splits.SplitManifest` is recorded as its digest
            (:meth:`~nnx.data_splits.SplitManifest.identity`).
        objective: the objective's identity, e.g. ``{"id": "kd", "version": 1}``.
        config: any other JSON-like configuration.
    """

    task: Optional[Mapping[str, Any]] = None
    labels: Optional[Sequence[str]] = None
    model: Optional[Mapping[str, Any]] = None
    data: Mapping[str, Any] = field(default_factory=dict)
    splits: Mapping[str, Any] = field(default_factory=dict)
    objective: Optional[Mapping[str, Any]] = None
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from ._config import _FrozenConfig

        object.__setattr__(self, "data", _FrozenConfig(_refs(self.data, owner="data")))
        object.__setattr__(self, "splits", _FrozenConfig(_refs(self.splits, owner="splits")))
        if self.labels is not None:
            if isinstance(self.labels, str) or not all(isinstance(label, str) for label in self.labels):
                raise TypeError(f"labels must be a sequence of strings, got {self.labels!r}")
            object.__setattr__(self, "labels", tuple(self.labels))
        # Validate (rejecting anything not JSON-like, by path) and copy into
        # immutable form: the caller's dicts can no longer change the plan.
        for name in ("task", "model", "objective", "config"):
            object.__setattr__(self, name, _freeze(_canonical(getattr(self, name), f"$.{name}")))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExperimentManifest):
            return NotImplemented
        return self.canonical_bytes() == other.canonical_bytes()

    def __hash__(self) -> int:
        return hash(self.canonical_bytes())

    def state(self) -> dict[str, Any]:
        return {
            "task": _thaw(self.task),
            "labels": None if self.labels is None else list(self.labels),
            "model": _thaw(self.model),
            "data": {name: ref.state() for name, ref in self.data.items()},
            "splits": {name: ref.state() for name, ref in self.splits.items()},
            "objective": _thaw(self.objective),
            "config": _thaw(self.config),
        }

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> ExperimentManifest:
        return ExperimentManifest(
            task=state.get("task"),
            labels=state.get("labels"),
            model=state.get("model"),
            data={name: IdentityRef.from_state(ref) for name, ref in (state.get("data") or {}).items()},
            splits={name: IdentityRef.from_state(ref) for name, ref in (state.get("splits") or {}).items()},
            objective=state.get("objective"),
            config=state.get("config") or {},
        )

    def document(self) -> dict[str, Any]:
        """The hashed document: the format version plus the manifest."""
        return {"format": FORMAT, "manifest": self.state()}

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.document())

    def fingerprint(self) -> str:
        """``sha256:<hex>`` of the canonical bytes."""
        return f"sha256:{_sha256(self.canonical_bytes())}"

    @classmethod
    def for_model(
        cls,
        model: Any,
        *,
        train: Any = None,
        data: Optional[Mapping[str, Any]] = None,
        splits: Optional[Mapping[str, Any]] = None,
        objective: Optional[Mapping[str, Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
        preprocessing: Any = None,
    ) -> ExperimentManifest:
        """A manifest from an existing model's declarations: its task and
        label order (FEAT-002), its model descriptor and built-in net params
        (FEAT-006), and — when given — the training configuration
        (``NNTrainParams.state()`` without ``n_epochs`` and the resume
        lineage, which describe an attempt; loaders are never read) and a
        fitted ``nnx.preprocessing.Standardizer`` (FEAT-018), recorded as
        ``config["preprocessing"]``: its schema, frozen statistics and
        fit-membership identity, never recomputed. Builds no model and
        iterates no loader."""
        params = model.params
        task = getattr(params, "task", None)
        model_state = dict(params.state())
        model_state.pop("device", None)  # where it runs is environment, not intent
        net_params = getattr(model, "net_params", None)
        if net_params is not None:
            model_state["net_params"] = net_params.state()
        merged = dict(config or {})
        if train is not None:
            train_state = dict(train.state() if hasattr(train, "state") else train)
            # Resume lineage and run length belong to attempts, not to the
            # plan: a resume of the same plan keeps its fingerprint.
            for key in _ATTEMPT_ONLY_TRAIN_KEYS:
                train_state.pop(key, None)
            merged.setdefault("train", train_state)
        if preprocessing is not None:
            from .preprocessing import Standardizer

            if not isinstance(preprocessing, Standardizer):
                raise TypeError(f"preprocessing must be a fitted nnx.preprocessing.Standardizer, got {preprocessing!r}")
            if "preprocessing" in merged:
                raise ValueError("config already has a 'preprocessing' entry; pass it once")
            merged["preprocessing"] = preprocessing.state()
        return cls(
            task=None if task is None else task.state(),
            labels=None if task is None or task.labels is None else tuple(task.labels),
            model=model_state,
            data=data or {},
            splits=splits or {},
            objective=objective,
            config=merged,
        )


# --- attempts --------------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class Attempt:
    """One execution of a plan: a fresh ``attempt_id`` per ``train()``
    call, the plan's ``fingerprint`` and ``run_id``, its ``status``
    (``running`` / ``completed`` / ``failed`` / ``cancelled``), the
    ``parent`` it resumed from (run, attempt, checkpoint tag and
    generation), the ``last_committed`` checkpoint (tag, epoch, generation)
    and, when it did not complete, the ``error`` (type and message)."""

    attempt_id: str
    fingerprint: str
    run_id: str
    status: str
    started_at: str
    finished_at: Optional[str] = None
    parent: Optional[Mapping[str, Any]] = None
    last_committed: Optional[Mapping[str, Any]] = None
    error: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if self.status not in ATTEMPT_STATUSES:
            raise ValueError(f"attempt status must be one of {ATTEMPT_STATUSES}, got {self.status!r}")

    def state(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "attempt_id": self.attempt_id,
            "fingerprint": self.fingerprint,
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "parent": None if self.parent is None else dict(self.parent),
            "last_committed": None if self.last_committed is None else dict(self.last_committed),
            "error": None if self.error is None else dict(self.error),
        }

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> Attempt:
        return Attempt(
            attempt_id=state["attempt_id"],
            fingerprint=state["fingerprint"],
            run_id=state["run_id"],
            status=state["status"],
            started_at=state["started_at"],
            finished_at=state.get("finished_at"),
            parent=state.get("parent"),
            last_committed=state.get("last_committed"),
            error=state.get("error"),
        )


@dataclass(frozen=True)
class ProvenanceRecord:
    """A run's provenance: the declared manifest, its fingerprint and the
    attempt that produced the run (``NNRun.provenance``)."""

    manifest: ExperimentManifest
    fingerprint: str
    attempt: Optional[Attempt] = None


def _run_dir(run_id: str, root: Optional[str]) -> str:
    from .nn.params.nn_run import _runs_root, _validate_run_id

    return os.path.join(_runs_root(root), _validate_run_id(run_id))


def _write_json(path: str, value: Mapping[str, Any]) -> None:
    from .nn.params.nn_run import _atomic_write_text

    # Atomic: an interrupted write leaves the previous file intact.
    _atomic_write_text(path, canonical_bytes(value).decode("utf-8") + "\n")


def load_provenance(run_id: str, root: Optional[str] = None) -> Optional[ProvenanceRecord]:
    """Read a run's provenance files; ``None`` when it has none (absent).
    Raises ``ValueError`` for an unsupported format or a fingerprint that
    does not match its manifest."""
    run_path = _run_dir(run_id, root)
    manifest_path = os.path.join(run_path, MANIFEST_FILE)
    if not os.path.isfile(manifest_path):
        return None
    with open(manifest_path, encoding="utf-8") as handle:
        stored = json.load(handle)
    if stored.get("format") != FORMAT:
        raise ValueError(f"unsupported provenance format {stored.get('format')!r} in {manifest_path}")
    manifest = ExperimentManifest.from_state(stored["manifest"])
    if stored.get("fingerprint") != manifest.fingerprint():
        raise ValueError(f"{manifest_path}: the stored fingerprint does not match its manifest (edited or corrupt)")
    attempt = None
    attempt_path = os.path.join(run_path, ATTEMPT_FILE)
    if os.path.isfile(attempt_path):
        with open(attempt_path, encoding="utf-8") as handle:
            attempt = Attempt.from_state(json.load(handle))
    return ProvenanceRecord(manifest=manifest, fingerprint=stored["fingerprint"], attempt=attempt)


def _checkpoint_summary(run_id: str, checkpoint: Any, root: Optional[str]) -> Optional[dict[str, Any]]:
    """Tag, epoch and generation of a run's checkpoint, or ``None``."""
    import torch

    from .nn.nn_model import _resume_checkpoint_type
    from .nn.params.nn_checkpoint import NNCheckpoint, _checkpoint_path

    try:
        tag = _resume_checkpoint_type(checkpoint)
        path = _checkpoint_path(run_id, tag, root=root)
        if not os.path.isfile(path):
            return None
        # Memory-mapped: only the metadata is read, never the tensors.
        loaded = torch.load(path, weights_only=False, map_location="cpu", mmap=True)
    except Exception:  # an unreadable checkpoint is simply not known here
        return None
    if not isinstance(loaded, NNCheckpoint):
        return None
    generation = getattr(loaded, "training_state_id", None)
    return {"checkpoint": str(tag), "epoch": int(loaded.idp.epoch_idx), "generation": generation}


class _AttemptRecorder:
    """Writes one attempt's provenance files: the manifest and a
    ``running`` record at the start, the final status at the end. A failure
    to record a failed attempt is a warning, never a replacement for the
    training error."""

    def __init__(
        self,
        run: NNRun,
        manifest: ExperimentManifest,
        *,
        parent_run_id: Optional[str],
        parent_checkpoint: Any,
        root: Optional[str] = None,
    ) -> None:
        if not isinstance(manifest, ExperimentManifest):
            raise TypeError(f"provenance must be an nnx.provenance.ExperimentManifest, got {type(manifest).__name__}")
        self.run_id = run.id
        self.manifest = manifest
        self.fingerprint = manifest.fingerprint()
        self.root = root
        parent = None
        if parent_run_id is not None:
            try:
                prior = load_provenance(parent_run_id, root)
            except (OSError, ValueError, KeyError, TypeError) as error:  # linkage is optional; the fit is not
                warnings.warn(
                    f"parent run {parent_run_id}'s provenance is unreadable ({error}); its attempt is not linked",
                    RuntimeWarning,
                    stacklevel=4,
                )
                prior = None
            parent = {
                "run_id": parent_run_id,
                "attempt_id": None if prior is None or prior.attempt is None else prior.attempt.attempt_id,
                **(
                    _checkpoint_summary(parent_run_id, parent_checkpoint, root)
                    or {"checkpoint": str(parent_checkpoint), "epoch": None, "generation": None}
                ),
            }
        self.attempt = Attempt(
            attempt_id=uuid.uuid4().hex,
            fingerprint=self.fingerprint,
            run_id=run.id,
            status="running",
            started_at=_now(),
            parent=parent,
        )

    @property
    def record(self) -> ProvenanceRecord:
        return ProvenanceRecord(manifest=self.manifest, fingerprint=self.fingerprint, attempt=self.attempt)

    def _path(self, name: str) -> str:
        return os.path.join(_run_dir(self.run_id, self.root), name)

    def start(self) -> None:
        _write_json(self._path(MANIFEST_FILE), {**self.manifest.document(), "fingerprint": self.fingerprint})
        _write_json(self._path(ATTEMPT_FILE), self.attempt.state())

    def _finish(self, status: str, error: Optional[BaseException]) -> None:
        from dataclasses import replace

        self.attempt = replace(
            self.attempt,
            status=status,
            finished_at=_now(),
            last_committed=_checkpoint_summary(self.run_id, "last", self.root),
            error=None if error is None else {"type": type(error).__name__, "message": str(error)},
        )
        _write_json(self._path(ATTEMPT_FILE), self.attempt.state())

    def complete(self) -> None:
        self._finish("completed", None)

    def fail(self, error: BaseException) -> None:
        status = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        try:
            self._finish(status, error)
        except BaseException as record_error:  # never mask the training error
            warnings.warn(
                f"could not record the {status} attempt {self.attempt.attempt_id}: {record_error}",
                RuntimeWarning,
                stacklevel=3,
            )


# --- comparison ------------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldComparison:
    """One field of a comparison: a stable dotted ``path`` and a
    ``status`` — ``equal``, ``different``, ``missing`` (on one side only),
    ``unknown`` (an unknown identity), ``declared`` (equal *declared*
    identities: unverified), ``verified`` (equal digests) or ``absent``
    (no provenance at all)."""

    path: str
    status: str
    left: Any = None
    right: Any = None


@dataclass(frozen=True)
class ProvenanceComparison:
    """Every compared field. :attr:`verified_equal` is true only when both
    sides have provenance and every field is ``equal`` or ``verified`` —
    declared or unknown identities and absent provenance never count as
    verified equality."""

    fields: tuple[FieldComparison, ...]

    @property
    def differences(self) -> tuple[FieldComparison, ...]:
        return tuple(item for item in self.fields if item.status not in ("equal", "verified"))

    @property
    def verified_equal(self) -> bool:
        return bool(self.fields) and not self.differences

    def by_path(self) -> dict[str, FieldComparison]:
        return {item.path: item for item in self.fields}


_MISSING = object()


_REF_SECTIONS = ("data", "splits")


def _is_ref(value: Any) -> bool:
    return isinstance(value, Mapping) and set(value) == {"identity", "value"}


def _same(left: Any, right: Any) -> bool:
    """Type-strict equality by canonical bytes (``1`` != ``1.0`` != ``True``),
    exactly as the fingerprint sees it."""
    return canonical_bytes(left) == canonical_bytes(right)


def _flatten(value: Any, path: str, out: dict[str, Any]) -> None:
    if not isinstance(value, (Mapping, list)):
        out[path] = value
        return
    if isinstance(value, Mapping):
        if not value:
            out[path] = value
        for key in sorted(value):
            _flatten(value[key], f"{path}.{key}" if path else key, out)
        return
    if not value:
        out[path] = value
    for index, item in enumerate(value):
        _flatten(item, f"{path}[{index}]", out)


def _compare_values(path: str, left: Any, right: Any, *, identity: bool) -> FieldComparison:
    if left is _MISSING or right is _MISSING:
        return FieldComparison(
            path, "missing", None if left is _MISSING else left, None if right is _MISSING else right
        )
    if identity:
        if not (_is_ref(left) and _is_ref(right)):
            return FieldComparison(path, "different", left, right)
        if "unknown" in (left["identity"], right["identity"]):
            return FieldComparison(path, "unknown", left, right)
        if not _same(left, right):
            return FieldComparison(path, "different", left, right)
        return FieldComparison(path, "verified" if left["identity"] == "digest" else "declared", left, right)
    return FieldComparison(path, "equal" if _same(left, right) else "different", left, right)


def _fields(state: Mapping[str, Any]) -> dict[str, tuple[Any, bool]]:
    """Leaf values by stable path; identity references appear only as the
    direct children of ``data`` and ``splits``."""
    out: dict[str, tuple[Any, bool]] = {}
    for section in sorted(state):
        value = state[section]
        if section in _REF_SECTIONS and isinstance(value, Mapping):
            if not value:
                out[section] = (value, False)
            for name in sorted(value):
                out[f"{section}.{name}"] = (value[name], True)
            continue
        plain: dict[str, Any] = {}
        _flatten(value, section, plain)
        out.update((path, (leaf, False)) for path, leaf in plain.items())
    return out


def _manifest_of(value: Any, root: Optional[str]) -> Optional[ExperimentManifest]:
    if value is None or isinstance(value, ExperimentManifest):
        return value
    if isinstance(value, ProvenanceRecord):
        return value.manifest
    if isinstance(value, str):
        record = load_provenance(value, root)  # reads provenance files only; no model is built
        return None if record is None else record.manifest
    provenance: Any = getattr(value, "provenance", _MISSING)  # an NNRun
    if provenance is not _MISSING:
        return None if provenance is None else provenance.manifest
    raise TypeError(f"compare takes manifests, provenance records, NNRuns or run ids, got {type(value).__name__}")


def compare(left: Any, right: Any, *, root: Optional[str] = None) -> ProvenanceComparison:
    """Compare two plans field by field — manifests, provenance records,
    ``NNRun``\\ s or run ids (their provenance files are read; no model is
    loaded, no loader iterated, no file hashed)."""
    left_manifest, right_manifest = _manifest_of(left, root), _manifest_of(right, root)
    if left_manifest is None or right_manifest is None:
        return ProvenanceComparison(
            (FieldComparison("", "absent", left_manifest is not None, right_manifest is not None),)
        )
    left_fields = _fields(left_manifest.state())
    right_fields = _fields(right_manifest.state())
    entries = []
    for path in sorted(set(left_fields) | set(right_fields)):
        left_value, left_identity = left_fields.get(path, (_MISSING, False))
        right_value, right_identity = right_fields.get(path, (_MISSING, False))
        entries.append(_compare_values(path, left_value, right_value, identity=left_identity or right_identity))
    return ProvenanceComparison(tuple(entries))
