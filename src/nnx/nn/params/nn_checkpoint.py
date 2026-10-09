from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import tempfile
import threading
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, BinaryIO, Literal, Optional, cast

import torch
from filelock import FileLock

from ...monitors import MonitorRecord
from ..enum.checkpoints import Checkpoints
from ..params.nn_evaluation_data_point import NNEvaluationDataPoint
from ..params.nn_iteration_data_point import NNIterationDataPoint
from ..params.nn_model_params import NNModelParams
from ..params.nn_params import NNParams

# Bumped only when the safetensors metadata layout changes in a way that
# breaks older readers. Newer readers stay backwards-compatible by sniffing
# this version off the metadata dict.
_SAFETENSORS_FORMAT_VERSION = "1"
_TRAINING_STATE_FORMAT_VERSION = 4
# Filename-safe slug a ModelCheckpoint tag must match; its files are
# "<tag>_e<epoch>.pt", and a resume accepts that stem (FEAT-005).
_MODEL_CHECKPOINT_TAG = r"[A-Za-z0-9][A-Za-z0-9._-]*"


@dataclass(frozen=True, kw_only=True, slots=True)
class NNCheckpointTransform:
    """A versioned recipe for rebuilding a checkpoint's module topology."""

    name: str
    version: int = 1
    options: dict[str, Any] = field(default_factory=dict)

    def state(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "options": self.options}

    @staticmethod
    def from_state(state: dict[str, Any]) -> NNCheckpointTransform:
        return NNCheckpointTransform(
            name=state["name"],
            version=state.get("version", 1),
            options=state.get("options", {}),
        )


def _checkpoint_path(run: str, type: Checkpoints, root: Optional[str] = None) -> str:
    """Resolve the on-disk path for a checkpoint. Defaults to cwd-relative
    so existing notebook code stays untouched.

    Validates ``run`` to reject path-traversal identifiers — see
    :func:`nnx.nn.params.nn_run._validate_run_id` for the threat model.
    Internal callers pass the md5 hex of ``NNRun.state()`` (always safe),
    but ``NNCheckpoint.load`` / ``load_optimizer_state`` accept ``run``
    from the public API surface, so we validate here at the single
    path-construction site that every caller funnels through.
    """
    # Local import — _validate_run_id lives in the sibling module and
    # this module can't import it at module load time (nn_run imports
    # nn_checkpoint, so the reverse direction is a cycle). The function
    # call is one-shot per checkpoint save / load — overhead is negligible.
    from .nn_run import _validate_run_id

    _validate_run_id(run)
    base = root if root is not None else "."
    return os.path.join(base, "runs", run, "checkpoints", str(type) + ".pt")


def _run_directory_exists(checkpoint_path: str) -> bool:
    """Absence fast path for the training-state readers (FIX-007).

    ``checkpoint_path`` is ``<root>/runs/<id>/checkpoints/<type>.pt``; the
    run directory is two levels up. Readers used to take the in-tree
    ``FileLock`` before checking anything, and the lock implementation
    creates ``runs/<id>/checkpoints/`` as a side effect — so merely
    *probing* a prospective run ID reserved it and the first ``train()``
    then refused to start. Probing is observational: an absent run tree
    returns the absent result without touching the filesystem. A probe
    may race a concurrent first creation and legitimately observe
    absence; callers may retry. Existing run directories keep the
    original lock and generation validation (a missing or malformed
    sidecar inside an existing run is still an integrity error).
    """
    return os.path.isdir(os.path.dirname(os.path.dirname(checkpoint_path)))


def _generation_sidecar_path(checkpoint_path: str, generation: str) -> str:
    return f"{checkpoint_path}.opt.{generation}.pt"


def _generation_sidecar_paths(checkpoint_path: str) -> list[str]:
    """Enumerate generation sidecars without interpreting path metacharacters."""
    directory = os.path.dirname(checkpoint_path)
    prefix = f"{os.path.basename(checkpoint_path)}.opt."
    if not os.path.isdir(directory):
        return []
    paths = []
    for entry in os.scandir(directory):
        remainder = entry.name[len(prefix) :] if entry.name.startswith(prefix) else ""
        if entry.is_file() and remainder.endswith(".pt") and remainder != "pt":
            paths.append(entry.path)
    return paths


def _is_bundle_payload(path: str) -> bool:
    """Whether the safetensors file at ``path`` is a run-bundle payload
    (``nnx.bundles``), judged from its JSON header's metadata alone."""
    try:
        with open(path, "rb") as handle:
            length = int.from_bytes(handle.read(8), "little")
            if not 0 < length <= 100 * 1024 * 1024:
                return False
            header = json.loads(handle.read(length))
    except (OSError, ValueError):
        return False  # not ours to judge: the safetensors reader reports it
    metadata = header.get("__metadata__") if isinstance(header, dict) else None
    return isinstance(metadata, dict) and "nnx.bundle" in metadata


def _snapshot_state_dict(state: Any) -> Any:
    """Copy tensor and extra state while preserving OrderedDict metadata."""
    return deepcopy(state)


def _tensor_state_dict(state: Any, *, operation: str) -> dict[str, torch.Tensor]:
    """Clone a tensor-only state dict or explain the export limitation."""
    non_tensor_keys = [key for key, value in state.items() if not isinstance(value, torch.Tensor)]
    if non_tensor_keys:
        raise TypeError(
            f"{operation} does not support non-tensor state_dict entries {non_tensor_keys}; "
            "use NNCheckpoint pickle format to preserve module extra state"
        )
    return {key: value.detach().contiguous().clone() for key, value in state.items()}


def _fsync_directory(path: str) -> None:
    """Persist a rename: fsync the directory (a no-op where unsupported)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class _HashingWriter:
    """A binary file that SHA-256s every byte as it is written (FEAT-045):
    a resume point's digests come from the write itself, never from reading
    the file back. A chunk of ``_OVERLAP_BYTES`` or more is hashed on a helper
    thread while it is written (``hashlib`` and the write both release the
    GIL), so it costs about the larger of the two rather than their sum.

    torch's zip writer calls :meth:`write` from C++ and turns anything raised
    there into a ``RuntimeError``; the first exception is kept in ``error``
    for :func:`_atomic_torch_save` to re-raise as itself (a Ctrl-C stays a
    ``KeyboardInterrupt``, a full disk an ``OSError``)."""

    _OVERLAP_BYTES = 1 << 20  # smaller chunks are hashed inline

    def __init__(self, handle: Any, pool: ThreadPoolExecutor) -> None:
        self._handle = handle
        self._pool = pool
        self._digest = hashlib.sha256()
        self.error: Optional[BaseException] = None

    def write(self, data: Any) -> int:
        if self.error is not None:
            raise self.error
        try:
            view = memoryview(data).cast("B")
            if view.nbytes < self._OVERLAP_BYTES:
                self._digest.update(view)
                self._handle.write(view)  # a buffered file writes it all, or raises
                return view.nbytes
            hashed = self._pool.submit(self._digest.update, view)
            try:
                self._handle.write(view)
            finally:
                hashed.result()  # in order, and before torch may reuse the buffer
            return view.nbytes
        except BaseException as error:
            self.error = error
            raise

    def flush(self) -> None:
        self._handle.flush()

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _atomic_torch_save(obj, path: str) -> str:
    """torch.save(obj, path) wrapped with tmp + fsync + rename so a
    KeyboardInterrupt (or a crash) during the pickle never leaves a
    half-written .pt file at the destination. Returns the SHA-256 of the
    bytes written, hashed as they are written (FEAT-045)."""
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "wb", buffering=1 << 20) as handle, ThreadPoolExecutor(max_workers=1) as pool:
            writer = _HashingWriter(handle, pool)
            try:
                # torch.save calls write() and flush() on it; the writer has no
                # fileno(), which would let a legacy-format save bypass write().
                torch.save(obj, cast(BinaryIO, writer))
            except BaseException:
                if writer.error is not None:
                    raise writer.error from None  # as raised, not as torch's RuntimeError
                raise
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:  # best effort: some platforms refuse fsync on this handle
                pass
        os.replace(tmp, path)
        return writer.hexdigest()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# --- resume points (#394) ------------------------------------------------------------------------

_RESUME_POINT_FORMAT = "nnx.resume-point/1"


class ResumePointError(ValueError):
    """A resume point that cannot be trusted: torn (its manifest names
    another checkpoint), mixed (files from different generations) or
    corrupted (a file's digest differs). Raised before anything is
    restored; the message says which and why."""


def _manifest_path(checkpoint_path: str) -> str:
    return f"{checkpoint_path}.manifest.json"


def _staged_manifest_path(checkpoint_path: str) -> str:
    """The next generation's manifest, written before its checkpoint is
    published: a crash between publishing and the live manifest leaves it for
    :meth:`NNCheckpoint.verify` to read once its digests check out (the next
    save makes it live)."""
    return f"{checkpoint_path}.manifest.staged.json"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(checkpoint_path: str) -> Optional[dict[str, Any]]:
    path = _manifest_path(checkpoint_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as error:
        raise ResumePointError(f"unreadable resume-point manifest {os.path.basename(path)}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("format") != _RESUME_POINT_FORMAT:
        raise ResumePointError(f"{os.path.basename(path)} is not an NNx resume-point manifest")
    return manifest


def _write_manifest(checkpoint_path: str, manifest: dict[str, Any], *, staged: bool = False) -> None:
    path = _staged_manifest_path(checkpoint_path) if staged else _manifest_path(checkpoint_path)
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


@contextlib.contextmanager
def _deferred_interrupt():
    """Hold a Ctrl-C (SIGINT) until the block ends, so it cannot land between
    the paired replaces of one resume point; raise it right after. Only the
    main thread can install handlers; elsewhere the block runs as is."""
    if threading.current_thread() is not threading.main_thread() or signal.getsignal(signal.SIGINT) in (
        signal.SIG_IGN,
        None,
    ):
        yield
        return
    received: list[Any] = []

    def hold(signum: int, frame: Any) -> None:
        received.append((signum, frame))
        if len(received) > 1:  # insisting: stop now
            raise KeyboardInterrupt

    previous = signal.signal(signal.SIGINT, hold)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
    if received:
        if previous is signal.default_int_handler or not callable(previous):
            raise KeyboardInterrupt
        previous(*received[0])  # the caller's own handler, as if the signal had just arrived


def _publish(checkpoint_path: str, written: str, manifest: dict[str, Any]) -> None:
    """Publish one resume point (#394): stage its manifest, rename the written
    checkpoint into place, then write the live manifest — the window a Ctrl-C
    is held for. A crash inside it leaves the staged manifest, which
    :meth:`NNCheckpoint.verify` reads once every listed digest matches (the
    next save makes it live)."""
    _write_manifest(checkpoint_path, manifest, staged=True)
    with _deferred_interrupt():
        os.replace(written, checkpoint_path)
        _write_manifest(checkpoint_path, manifest)
        with contextlib.suppress(FileNotFoundError):
            os.remove(_staged_manifest_path(checkpoint_path))
    _fsync_directory(os.path.dirname(checkpoint_path))


def _manifest_problem(directory: str, checkpoint_name: str, manifest: Mapping[str, Any]) -> Optional[str]:
    """Why a manifest does not describe the files on disk, or ``None``."""
    files = manifest.get("files")
    if not isinstance(files, dict) or checkpoint_name not in files:
        return f"the manifest does not list {checkpoint_name}"
    for name in files:
        if os.path.basename(name) != name:
            return f"the manifest lists a path, not a file name: {name!r}"
        if not os.path.exists(os.path.join(directory, name)):
            return f"torn resume point — {name} (generation {manifest.get('generation')}) is missing"
    checkpoint_id = manifest.get("checkpoint_id")
    if checkpoint_id is not None:
        sidecar = f"{checkpoint_name}.opt.{checkpoint_id}.pt"
        if sidecar not in files:
            return (
                f"torn resume point — the manifest names checkpoint {checkpoint_id}, whose training state "
                f"{sidecar} it does not list"
            )
    mismatched = sorted(name for name, digest in files.items() if _sha256(os.path.join(directory, name)) != digest)
    if mismatched:
        return f"digest mismatch in {', '.join(mismatched)} — the file changed after it was written"
    return None


def _read_staged_manifest(checkpoint_path: str) -> Optional[dict[str, Any]]:
    """A staged manifest left by an interrupted publish, or ``None``."""
    try:
        with open(_staged_manifest_path(checkpoint_path), encoding="utf-8") as handle:
            staged = json.load(handle)
    except (OSError, ValueError):
        return None
    return staged if isinstance(staged, dict) and staged.get("format") == _RESUME_POINT_FORMAT else None


def _next_generation(checkpoint_path: str) -> int:
    """The tag's next generation ordinal: one more than the larger of its live
    manifest's and a staged one's (an interrupted publish), 1 for a first
    save or after an unreadable manifest."""
    try:
        live = _read_manifest(checkpoint_path)
    except ResumePointError:
        live = None
    ordinals = [
        value
        for value in (m.get("generation") for m in (live, _read_staged_manifest(checkpoint_path)) if m is not None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    ]
    return max(ordinals) + 1 if ordinals else 1


def _manifest(
    ordinal: int,
    checkpoint_id: Optional[str],
    completed_epoch: Any,
    counters: Optional[dict[str, Any]],
    files: dict[str, str],
) -> dict[str, Any]:
    counters = counters or {}
    return {
        "format": _RESUME_POINT_FORMAT,
        "generation": ordinal,
        "checkpoint_id": checkpoint_id,
        "completed_epoch": completed_epoch,
        "global_step": counters.get("global_step"),
        "committed_updates": counters.get("committed_updates"),
        "planned_n_epochs": counters.get("planned_n_epochs"),
        "files": files,
    }


def _net_params_from_json(raw: str) -> Optional[NNParams]:
    """The metadata's built-in net params; ``null`` for a registered module
    (FEAT-006). ``resolve_from_state``: transformer checkpoints round-trip as
    NNTransformerParams instead of degrading to base NNParams."""
    state = json.loads(raw)
    return None if state is None else NNParams.resolve_from_state(state)


def _idp_from_nested_state(state: dict) -> NNIterationDataPoint:
    """Reconstruct an NNIterationDataPoint from the nested form produced
    by :meth:`NNIterationDataPoint.state`.

    The public ``NNIterationDataPoint.from_state`` expects the *flattened*
    CSV-column form (``train_edp.loss`` etc.) that ``pd.json_normalize``
    emits during ``NNRun.save``. The safetensors metadata path stores the
    nested form directly via JSON, so we need a parallel reconstructor
    here. Keeping it local to this module avoids polluting the public
    NNIterationDataPoint surface with a second from_state variant.
    """
    train_edp = NNEvaluationDataPoint.from_state(state["train_edp"])
    val_edp_state = state.get("val_edp")
    val_edp = NNEvaluationDataPoint.from_state(val_edp_state) if val_edp_state is not None else None
    summary_state = state.get("train_summary")
    selection_state = state.get("selection")
    return NNIterationDataPoint(
        lr=state["lr"],
        iter_idx=state["iter_idx"],
        epoch_idx=state["epoch_idx"],
        batch_idx=state["batch_idx"],
        train_edp=train_edp,
        val_edp=val_edp,
        train_summary=NNEvaluationDataPoint.from_state(summary_state) if summary_state is not None else None,
        selection=MonitorRecord.from_state(selection_state) if selection_state is not None else None,
        update_count=state.get("update_count"),
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class NNCheckpoint:
    """Model state plus the recipes needed to rebuild its module topology.

    ``transforms`` is empty for ordinary and legacy checkpoints. Training
    callbacks that replace modules at ``on_train_end`` can persist ordered,
    versioned recipes here; :meth:`NNModel.from_checkpoint` replays recognized
    recipes before loading ``net_state``.
    """

    # Built-in nets only; ``None`` for a registered or runtime module
    # (FEAT-006), whose descriptor is ``model_params.net``.
    net_params: Optional[NNParams]
    net_state: dict[str, Any]
    model_params: NNModelParams
    idp: NNIterationDataPoint
    transforms: tuple[NNCheckpointTransform, ...] = ()
    training_state_id: Optional[str] = None
    training_state_present: Optional[bool] = None

    @property
    def reconstructible(self) -> bool:
        """Whether a model can be rebuilt from this checkpoint alone
        (FEAT-006): ``False`` for a runtime-only module, whose weights need
        the module again (``NNModel.from_checkpoint(ckpt, module=...)``)."""
        return bool(getattr(self.model_params.net, "reconstructible", True))

    def to_file(self, path: str, format: Literal["pickle", "safetensors"] = "pickle") -> None:
        """Atomically write this NNCheckpoint to ``path``.

        Args:
            path: destination path. Parent directory is created if missing.
            format: one of:

                - ``"pickle"`` (default): a ``torch.save`` of the whole
                  NNCheckpoint dataclass. Bit-exact round-trip including
                  the OrderedDict state and the dataclass identity. The
                  on-disk format NNx has always written; back-compat
                  default for existing callers.
                - ``"safetensors"``: a ``.safetensors`` file with the
                  net's tensors as the data section and
                  NNParams + NNModelParams + NNIterationDataPoint + transform
                  recipes
                  JSON-serialized into the metadata dict (str→str only,
                  per the safetensors spec). Safe to mmap, readable by
                  ComfyUI/vLLM/AutoGPTQ/HF tools, and proof against
                  arbitrary-code-execution on load. Requires the
                  ``thekaveh-nnx[hub]`` extra. Portable, so a runtime-only
                  module (``reconstructible=False``, FEAT-006) is rejected
                  with ``MissingModelFactoryError`` before anything is
                  written.

        Both formats write to ``<path>.tmp`` first and rename into place
        so a KeyboardInterrupt during the underlying save can never leave
        a half-written checkpoint at the destination — matching the
        atomicity guarantee NNRun.save offers for YAML/CSV.
        """
        if format == "safetensors":
            self._require_portable()  # before any directory or file is created
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if format == "pickle":
            _atomic_torch_save(self, path)
            return
        if format == "safetensors":
            self._to_safetensors_file(path)
            return
        raise ValueError(f"unknown checkpoint format: {format!r} (expected 'pickle' or 'safetensors')")

    def _require_portable(self) -> None:
        if not self.reconstructible:
            from ...models import MissingModelFactoryError

            raise MissingModelFactoryError(
                f"a safetensors checkpoint is portable, but {self.model_params.net} is a runtime-only module "
                "(reconstructible=False) with no model factory to rebuild it; register one "
                "(nnx.models.register_model_factory) and train from a ModelSpec, or keep format='pickle'"
            )

    def _to_safetensors_file(self, path: str) -> None:
        """Atomic safetensors write. Requires the ``thekaveh-nnx[hub]`` extra."""
        self._require_portable()
        try:
            from safetensors.torch import save_file
        except ImportError as e:  # pragma: no cover — gated by optional dep
            raise ImportError(
                "safetensors checkpoint format requires the `hub` extra: `pip install thekaveh-nnx[hub]`."
            ) from e

        # safetensors metadata is str→str only — every value must be a
        # string. We JSON-encode each subsection so the reader can parse
        # them back without ambiguity.
        metadata = {
            "nnx_format_version": _SAFETENSORS_FORMAT_VERSION,
            "model_params": json.dumps(self.model_params.state()),
            "net_params": json.dumps(self.net_params.state() if self.net_params is not None else None),
            "idp": json.dumps(self.idp.state()),
            "transforms": json.dumps([transform.state() for transform in self.transforms]),
            "training_state_id": self.training_state_id or "",
            "training_state_present": json.dumps(self.training_state_present),
        }

        # safetensors save_file doesn't accept OrderedDict (only dict). The
        # iteration order is preserved either way in Python 3.7+, so coerce.
        # Detach drops any autograd graph attached to live params; clone
        # breaks storage sharing — safetensors rejects tied tensors
        # (TransformerNN's tok_embed/lm_head share storage by default,
        # and .contiguous() is a no-op on already-contiguous views).
        # On reload, load_state_dict assigns both identical copies back
        # into the tied parameter, so the tie survives the round-trip.
        tensors = _tensor_state_dict(self.net_state, operation="safetensors checkpoint export")

        fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(os.path.abspath(path)))
        os.close(fd)
        try:
            save_file(tensors, tmp, metadata=metadata)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def save(
        self,
        run: str,
        type: Checkpoints,
        root: Optional[str] = None,
        optimizer_state: Optional[dict[str, Any]] = None,
        scheduler_state: Optional[dict[str, Any]] = None,
        scaler_state: Optional[dict[str, Any]] = None,
        rng_state: Optional[dict[str, Any]] = None,
        completed_epoch: Optional[int] = None,
        resume_net_state: Optional[dict[str, Any]] = None,
        optimizer_type: Optional[str] = None,
        scheduler_type: Optional[str] = None,
        optimizer_topology: Optional[list[list[dict[str, Any]]]] = None,
        optimizer_factory: Optional[dict[str, Any]] = None,
        components: Optional[dict[str, Any]] = None,
        optimizers_state: Optional[dict[str, Any]] = None,
        schedulers_state: Optional[dict[str, Any]] = None,
        optimizer_types: Optional[dict[str, str]] = None,
        scheduler_types: Optional[dict[str, str]] = None,
        optimizer_topologies: Optional[dict[str, list[list[dict[str, Any]]]]] = None,
        optimizer_factories: Optional[dict[str, Optional[dict[str, Any]]]] = None,
        precision: Optional[dict[str, Any]] = None,
        compile: Optional[dict[str, Any]] = None,
        distributed: Optional[dict[str, Any]] = None,
        counters: Optional[dict[str, Any]] = None,
    ) -> None:
        """Save the checkpoint to disk atomically.

        Every save is a resume point (#394): a manifest
        (``<tag>.pt.manifest.json``, written last) stamps it with the tag's
        generation ordinal, the checkpoint id and the SHA-256 of each file it
        consists of, so a resume refuses a torn, mixed or corrupted point
        (:meth:`verify`). Files are fsynced before they are renamed into
        place, and a Ctrl-C arriving meanwhile is held until the point is
        complete. ``counters`` (the run's logical position: ``global_step``,
        ``committed_updates``, ``planned_n_epochs``) rides in the training
        state.

        ``components`` (FEAT-005) is the ``ComponentRegistry.collect()``
        mapping of every registered component's versioned state; it lives
        in the same generation sidecar as the optimizer state, so a model
        and component state from different generations are never paired.
        ``optimizers_state`` / ``schedulers_state`` (with their
        ``*_types``, ``optimizer_topologies`` and ``optimizer_factories``)
        carry a multi-optimizer ``Trainer``'s name-keyed states; either
        ``optimizer_state`` or ``optimizers_state`` makes the checkpoint
        stateful.

        When `optimizer_state` is supplied, a generation-addressed sibling
        file holds the training state, plus a fixed-name compatibility copy.
        This sidecar is used by NNModel.train(resume_from=...) to warm-resume
        with the prior optimizer momentum / Adam state.

        The immutable generation sidecar is committed first and the checkpoint
        second. The checkpoint names the sidecar it owns, so interruption
        between replacements leaves the previous generation resumable.
        """
        ckpt_path = _checkpoint_path(run, type, root=root)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        sidecar_path = ckpt_path + ".opt.pt"
        with FileLock(ckpt_path + ".lock"):
            ordinal = _next_generation(ckpt_path)
            completed = self.idp.epoch_idx if completed_epoch is None else completed_epoch
            if optimizer_state is None and optimizers_state is None:
                fd, weights_tmp = tempfile.mkstemp(
                    prefix=f".{os.path.basename(ckpt_path)}.", dir=os.path.dirname(ckpt_path)
                )
                os.close(fd)  # the save replaces it: nothing is removed before the point is published
                try:
                    weights_only = replace(self, training_state_id=None, training_state_present=False)
                    digest = _atomic_torch_save(weights_only, weights_tmp)  # to_file's pickle, hashed as written
                    manifest = _manifest(ordinal, None, completed, counters, {os.path.basename(ckpt_path): digest})
                    _publish(ckpt_path, weights_tmp, manifest)
                finally:
                    if os.path.exists(weights_tmp):
                        os.remove(weights_tmp)
                if os.path.exists(sidecar_path):
                    os.remove(sidecar_path)
                for generation_sidecar in _generation_sidecar_paths(ckpt_path):
                    os.remove(generation_sidecar)
                return

            generation = uuid.uuid4().hex
            stamped = replace(self, training_state_id=generation, training_state_present=True)
            training_state = {
                "nnx_training_state_version": _TRAINING_STATE_FORMAT_VERSION,
                "checkpoint_id": generation,
                "optimizer": optimizer_state,
                "optimizer_type": optimizer_type,
                "optimizer_topology": optimizer_topology,
                # Registered-factory identity ({id, version, config}); absent
                # (None) for built-in optimizers and older sidecars.
                "optimizer_factory": optimizer_factory,
                "scheduler": scheduler_state,
                "scheduler_type": scheduler_type,
                "scaler": scaler_state,
                "rng": rng_state,
                "completed_epoch": completed,
                # #394: the resume point's generation ordinal and the run's
                # logical position (None when the caller keeps none).
                "generation": ordinal,
                "counters": counters,
                "model": resume_net_state,
                # FEAT-005: checkpointable components and Trainer's named
                # optimizers / schedulers (None when absent).
                "components": components,
                "optimizers": optimizers_state,
                "optimizer_types": optimizer_types,
                "optimizer_topologies": optimizer_topologies,
                "optimizer_factories": optimizer_factories,
                "schedulers": schedulers_state,
                "scheduler_types": scheduler_types,
                # FEAT-028: the run's resolved precision record (None when
                # absent; older sidecars lack the key).
                "precision": precision,
                # FEAT-029: the fit's compile record (None for an eager run;
                # older sidecars lack the key). Weights are always the eager
                # module's: no wrapper prefix ever reaches net_state.
                "compile": compile,
                # FEAT-030: a DDP run's world size, partitions and every
                # rank's RNG streams (None for a single-process run).
                "distributed": distributed,
            }
            fd, checkpoint_tmp = tempfile.mkstemp(
                prefix=f".{os.path.basename(ckpt_path)}.", dir=os.path.dirname(ckpt_path)
            )
            os.close(fd)
            os.remove(checkpoint_tmp)
            generation_sidecar_path = _generation_sidecar_path(ckpt_path, generation)
            try:
                # to_file's pickle and the sidecar, each hashed as it is written (FEAT-045)
                files = {
                    os.path.basename(ckpt_path): _atomic_torch_save(stamped, checkpoint_tmp),
                    os.path.basename(generation_sidecar_path): _atomic_torch_save(
                        training_state, generation_sidecar_path
                    ),
                }
                _publish(ckpt_path, checkpoint_tmp, _manifest(ordinal, generation, completed, counters, files))
                _atomic_torch_save(training_state, sidecar_path)
                for old_sidecar in _generation_sidecar_paths(ckpt_path):
                    if old_sidecar != generation_sidecar_path:
                        os.remove(old_sidecar)
            finally:
                if os.path.exists(checkpoint_tmp):
                    os.remove(checkpoint_tmp)

    @staticmethod
    def resume_point(run: str, type: Checkpoints, root: Optional[str] = None) -> Optional[dict[str, Any]]:
        """The manifest stamping this checkpoint's resume point (#394) —
        ``generation``, ``checkpoint_id``, ``completed_epoch``,
        ``global_step``, ``committed_updates``, ``planned_n_epochs`` and the
        ``files`` with their SHA-256 — or ``None`` for a checkpoint written
        before manifests existed. Read as is: after an interrupted publish
        the live manifest may name the previous checkpoint while a staged one
        describes the published files — :meth:`verify` checks both and
        returns the one that matches."""
        return _read_manifest(_checkpoint_path(run, type, root=root))

    @staticmethod
    def verify(run: str, type: Checkpoints, root: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Check a resume point before anything is restored from it and
        return its manifest (``None`` for a checkpoint without one, which
        loads as before). Raises :class:`ResumePointError` naming the reason:
        a listed file missing or a manifest naming another checkpoint
        (*torn*), files of different generations (*mixed*), or a file whose
        SHA-256 differs (*digest mismatch*). No file is unpickled to decide
        and nothing is written: a point whose writer stopped between
        publishing the checkpoint and its live manifest is recognized by its
        staged manifest — every digest matching — and that manifest is
        returned (the writer's next save makes it live)."""
        ckpt_path = _checkpoint_path(run, type, root=root)
        label = f"{run}/{type}"
        directory = os.path.dirname(ckpt_path)
        name = os.path.basename(ckpt_path)
        if not os.path.isdir(directory):
            return None
        with FileLock(ckpt_path + ".lock"):  # one consistent view against a concurrent writer
            manifest = _read_manifest(ckpt_path)
            problem = None if manifest is None else _manifest_problem(directory, name, manifest)
            if manifest is None or problem is not None:
                staged = _read_staged_manifest(ckpt_path)
                if staged is not None and _manifest_problem(directory, name, staged) is None:
                    manifest, problem = staged, None
            if manifest is None:
                return None
            if problem is not None:
                raise ResumePointError(f"{label}: {problem}")
            checkpoint_id = manifest.get("checkpoint_id")
            if checkpoint_id is not None:
                sidecar = os.path.join(directory, f"{name}.opt.{checkpoint_id}.pt")
                try:
                    state = torch.load(sidecar, map_location="cpu", weights_only=True, mmap=True)
                except FileNotFoundError as error:
                    raise ResumePointError(
                        f"{label}: torn resume point — {os.path.basename(sidecar)} is missing"
                    ) from error
                if (
                    isinstance(state, dict)
                    and "generation" in state
                    and state["generation"] != manifest.get("generation")
                ):
                    raise ResumePointError(
                        f"{label}: mixed resume point — the manifest is generation {manifest.get('generation')}, "
                        f"the training state generation {state['generation']}"
                    )
        return manifest

    @staticmethod
    def load_training_state(
        run: str,
        type: Checkpoints,
        root: Optional[str] = None,
        map_location: Any = "cpu",
    ) -> Optional[dict[str, Any]]:
        """Load and validate the resumable optimizer/scheduler/scaler bundle.

        Legacy optimizer-only sidecars are normalized into the new mapping so
        checkpoints written by older NNx versions remain resumable.

        Returns ``None`` — without creating ``runs/<run>/`` — when the run
        directory does not exist; the run ID is still validated first.
        """
        checkpoint_path = _checkpoint_path(run, type, root=root)
        if not _run_directory_exists(checkpoint_path):
            return None
        with FileLock(checkpoint_path + ".lock"):
            checkpoint = NNCheckpoint.from_file(checkpoint_path, map_location=map_location)
            return NNCheckpoint._load_training_state_unlocked(checkpoint_path, checkpoint, map_location)

    @staticmethod
    def load_with_training_state(
        run: str,
        type: Checkpoints,
        root: Optional[str] = None,
        map_location: Any = "cpu",
    ) -> tuple[Optional[NNCheckpoint], Optional[dict[str, Any]]]:
        """Atomically load a checkpoint and its matching training-state bundle.

        Returns ``(None, None)`` — without creating ``runs/<run>/`` — when
        the run directory does not exist; the run ID is still validated.
        """
        checkpoint_path = _checkpoint_path(run, type, root=root)
        if not _run_directory_exists(checkpoint_path):
            return None, None
        with FileLock(checkpoint_path + ".lock"):
            checkpoint = NNCheckpoint.from_file(checkpoint_path, map_location=map_location)
            state = NNCheckpoint._load_training_state_unlocked(checkpoint_path, checkpoint, map_location)
            return checkpoint, state

    @staticmethod
    def _load_training_state_unlocked(
        checkpoint_path: str,
        checkpoint: Optional[NNCheckpoint],
        map_location: Any = "cpu",
    ) -> Optional[dict[str, Any]]:
        sidecar_path = checkpoint_path + ".opt.pt"
        if checkpoint is not None and getattr(checkpoint, "training_state_present", None) is False:
            return None
        checkpoint_id = getattr(checkpoint, "training_state_id", None) if checkpoint is not None else None
        if checkpoint_id is not None:
            generation_path = _generation_sidecar_path(checkpoint_path, checkpoint_id)
            if os.path.exists(generation_path):
                sidecar_path = generation_path
        if not os.path.exists(sidecar_path):
            if checkpoint_id is not None:
                raise ValueError(
                    "checkpoint references a missing training-state sidecar; "
                    "resume from another checkpoint or restore the owned sidecar"
                )
            return None
        state = torch.load(sidecar_path, weights_only=True, map_location=map_location)
        if not isinstance(state, dict):
            raise ValueError(f"malformed training-state sidecar: expected a mapping, got {type(state).__name__}")
        if "nnx_training_state_version" not in state:
            if checkpoint_id is not None:
                raise ValueError(
                    "versioned checkpoint cannot use a legacy optimizer-only sidecar; "
                    "restore its matching generation-addressed training state"
                )
            return {
                "nnx_training_state_version": 0,
                "checkpoint_id": None,
                "optimizer": state,
                "optimizer_type": None,
                "optimizer_topology": None,
                "scheduler": None,
                "scheduler_type": None,
                "scaler": None,
                "rng": None,
                "completed_epoch": checkpoint.idp.epoch_idx if checkpoint is not None else None,
            }
        if checkpoint_id is None:
            # Cleanup can be interrupted after an optimizerless checkpoint
            # commits. Any remaining versioned sidecars are stale and unowned.
            return None
        version = state["nnx_training_state_version"]
        if type(version) is not int or not 1 <= version <= _TRAINING_STATE_FORMAT_VERSION:
            raise ValueError(
                f"unsupported training-state sidecar version {version!r}; "
                f"this NNx version supports 1..{_TRAINING_STATE_FORMAT_VERSION}"
            )
        if checkpoint_id != state.get("checkpoint_id"):
            raise ValueError(
                "checkpoint and training-state sidecar do not match; "
                "the previous save was interrupted, so resume from another checkpoint"
            )
        return state

    @staticmethod
    def load_optimizer_state(
        run: str,
        type: Checkpoints,
        root: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Load the optimizer state sidecar for a checkpoint. Returns None
        when no sidecar exists (e.g., checkpoints written before resume
        support was added).

        Loaded with ``weights_only=True`` — the optimizer state-dict
        contains only tensors and standard scalar/dict/list types, so the
        strict loader works AND it removes the arbitrary-code-execution
        risk that the main NNCheckpoint.from_file documents."""
        state = NNCheckpoint.load_training_state(run, type, root=root)
        return None if state is None else state["optimizer"]

    @staticmethod
    def from_file(path: str, map_location: Any = "cpu") -> Optional[NNCheckpoint]:
        """Load an NNCheckpoint from disk, auto-detecting pickle vs safetensors.

        Returns ``None`` if the path doesn't exist or the loaded pickle
        object isn't an NNCheckpoint instance.

        Dispatch is by magic bytes:

        - ``torch.save`` writes a ZIP archive in modern PyTorch
          (``_use_new_zipfile_serialization=True`` is the default since
          PyTorch 1.6), so the file starts with ``b"PK\\x03\\x04"``.
        - Legacy ``torch.save`` (with the zipfile serialization disabled)
          and bare pickle files begin with ``\\x80`` (the pickle PROTO
          opcode for protocol >= 2).
        - safetensors files begin with a little-endian u64 header length
          followed by a JSON object — byte 8 is always ``{``. The u64's
          LOW byte can legitimately be ``0x80`` (any header length
          ≡ 128 mod 256), which would collide with the pickle PROTO
          opcode — so safetensors is positively identified by byte 8
          BEFORE the ``\x80`` pickle check. The ZIP magic is checked
          first of all (a ZIP's byte 8 is the compression method, never
          ``{``; a torch-LEGACY pickle has the fixed magic byte ``0xf9``
          at offset 8, and a protocol ≥ 4 bare pickle has a frame-length
          byte there, ``0x00`` for any file under a terabyte. A
          protocol-2/3 *bare* pickle's byte 8 is content-dependent, but
          NNx never produces bare pickles and such a file failed under
          the old routing too).

        Anything matching none of the positive sniffs falls through to
        the safetensors loader, whose error on a genuinely corrupt file
        is clearer than a misleading unpickle attempt.

        SECURITY: the pickle branch calls ``torch.load(weights_only=False)``,
        which unpickles arbitrary Python objects. NEVER call this on a
        checkpoint file from an untrusted source — a malicious .pt file
        can execute arbitrary code at load time. The default
        ``./runs/<id>/checkpoints/`` layout assumes the files were
        produced locally by NNCheckpoint.save. For untrusted sources,
        use the safetensors path on save and load: safetensors has no
        arbitrary-code path.
        """
        if not os.path.exists(path):
            return None
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "bundle.json")):
            raise ValueError(
                f"{path!r} is an NNx run bundle, not a checkpoint file; read it with nnx.bundles "
                "(inspect_bundle / validate_bundle / reconstruct_bundle), which never unpickles"
            )
        parent = os.path.dirname(os.path.abspath(path))
        grandparent = os.path.dirname(parent)
        if os.path.isfile(os.path.join(parent, "bundle.json")) or (
            os.path.basename(parent).startswith("g-") and os.path.isfile(os.path.join(grandparent, "bundle.json"))
        ):
            raise ValueError(
                f"{path!r} is a file of an NNx run bundle, not a checkpoint; read the bundle directory with "
                "nnx.bundles.reconstruct_bundle"
            )

        with open(path, "rb") as f:
            head = f.read(9)
        if not head:
            return None

        # Modern torch-save ZIP container.
        if head[:4] == b"PK\x03\x04":
            return NNCheckpoint._from_pickle_file(path, map_location=map_location)
        # safetensors: byte 8 is the opening brace of the JSON header.
        # Checked BEFORE the \x80 pickle sniff — see the docstring.
        if len(head) == 9 and head[8:9] == b"{":
            return NNCheckpoint._from_safetensors_file(path, map_location=map_location)
        # Legacy / bare-pickle protocol prefix.
        if head[:1] == b"\x80":
            return NNCheckpoint._from_pickle_file(path, map_location=map_location)
        return NNCheckpoint._from_safetensors_file(path, map_location=map_location)

    @staticmethod
    def _from_pickle_file(path: str, map_location: Any = "cpu") -> Optional[NNCheckpoint]:
        """Load the legacy pickle format (a torch.save'd NNCheckpoint)."""
        # NNCheckpoint files are pickled Python objects (not bare state dicts),
        # so the weights_only=True default introduced in torch>=2.6 would raise
        # UnpicklingError. See the security note in `from_file` before
        # widening this to externally-sourced files.
        ret = torch.load(path, weights_only=False, map_location=map_location)

        if not isinstance(ret, NNCheckpoint):
            return None

        # Checkpoints pickled before transform metadata was introduced have
        # no value for the new slot. Normalize them to the empty legacy form.
        if not hasattr(ret, "transforms"):
            object.__setattr__(ret, "transforms", ())
        if not hasattr(ret, "training_state_id"):
            object.__setattr__(ret, "training_state_id", None)
        if not hasattr(ret, "training_state_present"):
            object.__setattr__(ret, "training_state_present", None)

        return ret

    @staticmethod
    def _from_safetensors_file(path: str, map_location: Any = "cpu") -> NNCheckpoint:
        """Load a safetensors-format NNCheckpoint. Requires `thekaveh-nnx[hub]`."""
        if _is_bundle_payload(path):  # from the header alone, before any tensor or optional import
            raise ValueError(
                f"{path!r} is a payload of an NNx run bundle, not a checkpoint; read the bundle directory "
                "with nnx.bundles.reconstruct_bundle"
            )
        try:
            from safetensors import safe_open
        except ImportError as e:  # pragma: no cover — gated by optional dep
            raise ImportError(
                "loading a safetensors checkpoint requires the `hub` extra: `pip install thekaveh-nnx[hub]`."
            ) from e

        net_state: OrderedDict[str, torch.Tensor] = OrderedDict()
        if not isinstance(map_location, (str, torch.device)):
            raise TypeError(
                "safetensors checkpoint map_location must be a device string or torch.device; "
                f"got {type(map_location).__name__}"
            )
        device = str(map_location)
        with safe_open(path, framework="pt", device=device) as f:
            meta = f.metadata() or {}
            # Preserve key order — safetensors guarantees insertion-order on
            # iteration in v0.5+; we rebuild the OrderedDict for parity with
            # the pickle path.
            for k in f.keys():
                net_state[k] = f.get_tensor(k)

        version = meta.get("nnx_format_version")
        if version != _SAFETENSORS_FORMAT_VERSION:
            raise ValueError(
                f"unsupported safetensors checkpoint format version {version!r}; "
                f"expected {_SAFETENSORS_FORMAT_VERSION!r}"
            )
        return NNCheckpoint(
            idp=_idp_from_nested_state(json.loads(meta["idp"])),
            model_params=NNModelParams.from_state(json.loads(meta["model_params"])),
            net_params=_net_params_from_json(meta["net_params"]),
            net_state=net_state,
            transforms=tuple(
                NNCheckpointTransform.from_state(state) for state in json.loads(meta.get("transforms", "[]"))
            ),
            training_state_id=meta.get("training_state_id") or None,
            training_state_present=json.loads(meta.get("training_state_present", "null")),
        )

    @staticmethod
    def load(
        run: str,
        type: Checkpoints,
        root: Optional[str] = None,
        map_location: Any = "cpu",
    ) -> Optional[NNCheckpoint]:
        return NNCheckpoint.from_file(path=_checkpoint_path(run, type, root=root), map_location=map_location)
