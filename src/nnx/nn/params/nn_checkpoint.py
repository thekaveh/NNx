from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Optional

import torch
from filelock import FileLock

from ..enum.checkpoints import Checkpoints
from ..params.nn_evaluation_data_point import NNEvaluationDataPoint
from ..params.nn_iteration_data_point import NNIterationDataPoint
from ..params.nn_model_params import NNModelParams
from ..params.nn_params import NNParams

# Bumped only when the safetensors metadata layout changes in a way that
# breaks older readers. Newer readers stay backwards-compatible by sniffing
# this version off the metadata dict.
_SAFETENSORS_FORMAT_VERSION = "1"
_TRAINING_STATE_FORMAT_VERSION = 3


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


def _atomic_torch_save(obj, path: str) -> None:
    """torch.save(obj, path) wrapped with tmp + rename so a
    KeyboardInterrupt during the pickle never leaves a half-written
    .pt file at the destination."""
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(os.path.abspath(path)))
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


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
    return NNIterationDataPoint(
        lr=state["lr"],
        iter_idx=state["iter_idx"],
        epoch_idx=state["epoch_idx"],
        batch_idx=state["batch_idx"],
        train_edp=train_edp,
        val_edp=val_edp,
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class NNCheckpoint:
    """Model state plus the recipes needed to rebuild its module topology.

    ``transforms`` is empty for ordinary and legacy checkpoints. Training
    callbacks that replace modules at ``on_train_end`` can persist ordered,
    versioned recipes here; :meth:`NNModel.from_checkpoint` replays recognized
    recipes before loading ``net_state``.
    """

    net_params: NNParams
    net_state: dict[str, Any]
    model_params: NNModelParams
    idp: NNIterationDataPoint
    transforms: tuple[NNCheckpointTransform, ...] = ()
    training_state_id: Optional[str] = None
    training_state_present: Optional[bool] = None

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
                  ``thekaveh-nnx[hub]`` extra.

        Both formats write to ``<path>.tmp`` first and rename into place
        so a KeyboardInterrupt during the underlying save can never leave
        a half-written checkpoint at the destination — matching the
        atomicity guarantee NNRun.save offers for YAML/CSV.
        """
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

    def _to_safetensors_file(self, path: str) -> None:
        """Atomic safetensors write. Requires the ``thekaveh-nnx[hub]`` extra."""
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
            "net_params": json.dumps(self.net_params.state()),
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
    ) -> None:
        """Save the checkpoint to disk atomically.

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
            if optimizer_state is None:
                replace(self, training_state_id=None, training_state_present=False).to_file(path=ckpt_path)
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
                "scheduler": scheduler_state,
                "scheduler_type": scheduler_type,
                "scaler": scaler_state,
                "rng": rng_state,
                "completed_epoch": self.idp.epoch_idx if completed_epoch is None else completed_epoch,
                "model": resume_net_state,
            }
            fd, checkpoint_tmp = tempfile.mkstemp(
                prefix=f".{os.path.basename(ckpt_path)}.", dir=os.path.dirname(ckpt_path)
            )
            os.close(fd)
            os.remove(checkpoint_tmp)
            generation_sidecar_path = _generation_sidecar_path(ckpt_path, generation)
            try:
                stamped.to_file(checkpoint_tmp)
                _atomic_torch_save(training_state, generation_sidecar_path)
                os.replace(checkpoint_tmp, ckpt_path)
                _atomic_torch_save(training_state, sidecar_path)
                for old_sidecar in _generation_sidecar_paths(ckpt_path):
                    if old_sidecar != generation_sidecar_path:
                        os.remove(old_sidecar)
            finally:
                if os.path.exists(checkpoint_tmp):
                    os.remove(checkpoint_tmp)

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
        """
        checkpoint_path = _checkpoint_path(run, type, root=root)
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
        """Atomically load a checkpoint and its matching training-state bundle."""
        checkpoint_path = _checkpoint_path(run, type, root=root)
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
            # resolve_from_state: transformer checkpoints round-trip as
            # NNTransformerParams instead of degrading to base NNParams.
            net_params=NNParams.resolve_from_state(json.loads(meta["net_params"])),
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
