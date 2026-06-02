from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal, Optional

import torch

from ..enum.checkpoints import Checkpoints
from ..params.nn_evaluation_data_point import NNEvaluationDataPoint
from ..params.nn_iteration_data_point import NNIterationDataPoint
from ..params.nn_model_params import NNModelParams
from ..params.nn_params import NNParams

# Bumped only when the safetensors metadata layout changes in a way that
# breaks older readers. Newer readers stay backwards-compatible by sniffing
# this version off the metadata dict.
_SAFETENSORS_FORMAT_VERSION = "1"


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


def _atomic_torch_save(obj, path: str) -> None:
    """torch.save(obj, path) wrapped with tmp + rename so a
    KeyboardInterrupt during the pickle never leaves a half-written
    .pt file at the destination."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


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
    net_params: NNParams
    net_state: OrderedDict
    model_params: NNModelParams
    idp: NNIterationDataPoint

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
                  NNParams + NNModelParams + NNIterationDataPoint
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
        }

        # safetensors save_file doesn't accept OrderedDict (only dict). The
        # iteration order is preserved either way in Python 3.7+, so coerce.
        # We also detach to drop any autograd graph attached to live params.
        tensors = {k: v.detach().contiguous() for k, v in self.net_state.items()}

        tmp = path + ".tmp"
        save_file(tensors, tmp, metadata=metadata)
        os.replace(tmp, path)

    def save(
        self,
        run: str,
        type: Checkpoints,
        root: Optional[str] = None,
        optimizer_state: Optional[OrderedDict] = None,
    ) -> None:
        """Save the checkpoint to disk atomically.

        When `optimizer_state` is supplied, a sibling file is written at
        ``<id>/checkpoints/<type>.opt.pt`` holding the optimizer state dict.
        This sidecar is used by NNModel.train(resume_from=...) to warm-resume
        with the prior optimizer momentum / Adam state.
        """
        ckpt_path = _checkpoint_path(run, type, root=root)
        self.to_file(path=ckpt_path)
        if optimizer_state is not None:
            _atomic_torch_save(optimizer_state, ckpt_path + ".opt.pt")

    @staticmethod
    def load_optimizer_state(
        run: str,
        type: Checkpoints,
        root: Optional[str] = None,
    ) -> Optional[OrderedDict]:
        """Load the optimizer state sidecar for a checkpoint. Returns None
        when no sidecar exists (e.g., checkpoints written before resume
        support was added).

        Loaded with ``weights_only=True`` — the optimizer state-dict
        contains only tensors and standard scalar/dict/list types, so the
        strict loader works AND it removes the arbitrary-code-execution
        risk that the main NNCheckpoint.from_file documents."""
        path = _checkpoint_path(run, type, root=root) + ".opt.pt"
        if not os.path.exists(path):
            return None
        return torch.load(path, weights_only=True)

    @staticmethod
    def from_file(path: str) -> Optional[NNCheckpoint]:
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
          followed by a JSON object. Neither of the pickle prefixes can
          appear there (the JSON section always starts with ``{``).

        We positively-identify pickle via either prefix and otherwise
        fall through to the safetensors path. A genuinely corrupt file
        will surface as a clear error from the underlying loader rather
        than a misleading sniff.

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
            head = f.read(4)
        if not head:
            return None

        # Pickle: either the modern torch-save ZIP container (``PK\x03\x04``)
        # or the legacy / bare-pickle ``\x80`` protocol prefix.
        if head[:4] == b"PK\x03\x04" or head[:1] == b"\x80":
            return NNCheckpoint._from_pickle_file(path)
        return NNCheckpoint._from_safetensors_file(path)

    @staticmethod
    def _from_pickle_file(path: str) -> Optional[NNCheckpoint]:
        """Load the legacy pickle format (a torch.save'd NNCheckpoint)."""
        # NNCheckpoint files are pickled Python objects (not bare state dicts),
        # so the weights_only=True default introduced in torch>=2.6 would raise
        # UnpicklingError. See the security note in `from_file` before
        # widening this to externally-sourced files.
        ret = torch.load(path, weights_only=False)

        if not isinstance(ret, NNCheckpoint):
            return None

        return ret

    @staticmethod
    def _from_safetensors_file(path: str) -> NNCheckpoint:
        """Load a safetensors-format NNCheckpoint. Requires `thekaveh-nnx[hub]`."""
        try:
            from safetensors import safe_open
        except ImportError as e:  # pragma: no cover — gated by optional dep
            raise ImportError(
                "loading a safetensors checkpoint requires the `hub` extra: `pip install thekaveh-nnx[hub]`."
            ) from e

        net_state: OrderedDict[str, torch.Tensor] = OrderedDict()
        with safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
            # Preserve key order — safetensors guarantees insertion-order on
            # iteration in v0.5+; we rebuild the OrderedDict for parity with
            # the pickle path.
            for k in f.keys():
                net_state[k] = f.get_tensor(k)

        # Future-proof: today only version "1" exists. If a future writer
        # bumps the version, we'd add a dispatch here on `meta["nnx_format_version"]`.
        return NNCheckpoint(
            idp=_idp_from_nested_state(json.loads(meta["idp"])),
            model_params=NNModelParams.from_state(json.loads(meta["model_params"])),
            net_params=NNParams.from_state(json.loads(meta["net_params"])),
            net_state=net_state,
        )

    @staticmethod
    def load(run: str, type: Checkpoints, root: Optional[str] = None) -> Optional[NNCheckpoint]:
        return NNCheckpoint.from_file(path=_checkpoint_path(run, type, root=root))
