"""Portable, data-only run bundles (FEAT-015).

A run checkpoint (``runs/<id>/checkpoints/<tag>.pt``) and its training-state
sidecar are pickles: reading them runs ``torch.load(weights_only=False)``,
which is only safe for files you produced. A safetensors checkpoint
(``NNCheckpoint.to_file(format="safetensors")``) is data only, but holds one
weights dict and no training state. A **run bundle** is a directory that
holds a run checkpoint — its weights, its training state and optional
calibrators — as data only, readable without unpickling anything::

    <bundle>/bundle.json           the manifest: format, version, generation id,
                                   and every payload's SHA-256 and size
    <bundle>/g-<generation>/
        state.json                 primitive state, schema-validated JSON
        model.safetensors          the network's tensors
        training.safetensors       the training state's tensors (resume bundles)
        calibrator-<n>.json        TemperatureCalibrator records (nnx.calibration)

Four separate operations:

- :func:`export_bundle` reads one of **your own** runs' checkpoints (the
  legacy pickle trust boundary: files NNx wrote locally) and publishes it as
  a new bundle generation. A checkpoint with training state gives a
  ``"resume"`` bundle; a weights-only one (a ``ModelCheckpoint`` snapshot)
  an ``"inference"`` bundle.
- :func:`inspect_bundle` reads the manifest and ``state.json`` — never a
  tensor payload — and summarizes the bundle.
- :func:`validate_bundle` checks every payload against the manifest (size,
  SHA-256, generation id, no missing, extra, symlinked or out-of-root file,
  no duplicate JSON keys, tensor headers matching ``state.json``) before any
  tensor is read.
- :func:`reconstruct_bundle` validates, then rebuilds the model from
  caller-supplied model factories (a ``ModelSpec`` needs one) and, when given
  the run's components, checks them — naming everything missing before any
  model is allocated. :meth:`ReconstructedBundle.resume` continues training
  from the bundle's training state through ``NNModel.train``'s own stateful
  resume.

None of them unpickles, imports code named by the bundle, or downloads
anything; ``inspect_bundle`` and ``validate_bundle`` call no factory either.
A bundle is published by writing a fresh generation directory and then
replacing ``bundle.json`` atomically, so an interrupted export leaves the
previous bundle usable. Primitive state is JSON with typed encodings for
what JSON cannot say (``$tuple``, integer-keyed ``$intdict``, non-finite
``$float``, ``$tensor`` references into a safetensors payload); anything else
— module extra state that is not a tensor, a custom object in optimizer or
component state — is refused at export, with no pickle fallback.

NNx writes and reads the safetensors payloads itself — the standard format,
which any safetensors reader opens — so bundles need no optional
dependency.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import uuid
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, BinaryIO, Optional, Union

import numpy as np
import torch

from ._config import _freeze_config
from .calibration import _FINGERPRINT_SEED, TemperatureCalibrator, _fingerprint_header, _state_fingerprint
from .components import ComponentRegistry, ComponentRestoreError
from .models import BatchAdapter, MissingModelFactoryError, ModelFactory, ModelSpec, RuntimeModule

if TYPE_CHECKING:
    from .nn.nn_model import NNModel
    from .nn.params.nn_checkpoint import NNCheckpoint
    from .nn.params.nn_run import NNRun
    from .nn.params.nn_train_params import NNTrainParams

__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_VERSION",
    "BundleCapabilityError",
    "BundleError",
    "BundleInfo",
    "BundleIntegrityError",
    "BundleReconstructionError",
    "ReconstructedBundle",
    "export_bundle",
    "inspect_bundle",
    "reconstruct_bundle",
    "validate_bundle",
]

BUNDLE_FORMAT = "nnx.run-bundle"
BUNDLE_VERSION = 1
_MANIFEST = "bundle.json"
_LOCK = ".bundle.lock"
_STATE = "state.json"
_MODEL = "model.safetensors"
_TRAINING = "training.safetensors"
_GENERATION = re.compile(r"[0-9a-f]{32}")
_PAYLOAD = re.compile(r"(state\.json|model\.safetensors|training\.safetensors|calibrator-[0-9]+\.json)")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TEMPORARY = re.compile(r"\.bundle\.json\..+")  # a manifest an interrupted export never renamed into place
_MAX_DEPTH = 64  # nesting deeper than any NNx state: refused before a recursive walk
_CAPABILITIES = ("inference", "resume")
_TAGS = ("$tensor", "$tuple", "$dict", "$intdict", "$float")
_MAX_HEADER = 100 * 1024 * 1024  # a safetensors header larger than this is malformed
_MAX_MANIFEST = 16 * 1024 * 1024  # far above any real manifest; a larger one is refused unread
_MAX_JSON = 256 * 1024 * 1024  # state.json and calibrator records: primitives only, never this large
_MAX_DIM = 2**63 - 1


class BundleError(ValueError):
    """A run bundle that cannot be written, read or used."""


class BundleIntegrityError(BundleError):
    """A bundle that fails validation: a missing, extra, altered, symlinked
    or out-of-root payload, a generation mismatch, duplicate JSON keys, an
    unknown format or version, or malformed state."""


class BundleCapabilityError(BundleError):
    """An operation the bundle's capability does not support — resuming an
    ``"inference"`` bundle, which has no training state."""


class BundleReconstructionError(BundleError):
    """What a reconstruction needs and was not given (a model factory, a
    component), every problem listed in ``problems``; raised before any
    model is allocated."""

    def __init__(self, problems: Iterable[str]) -> None:
        self.problems = tuple(problems)
        lines = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"cannot reconstruct the run bundle:\n{lines}")


@dataclass(frozen=True)
class BundleInfo:
    """A bundle's summary, from :func:`inspect_bundle` (``verified=False``:
    only the manifest and ``state.json`` were checked) or
    :func:`validate_bundle` / :func:`reconstruct_bundle` (``verified=True``).

    Attributes:
        path: the bundle directory.
        version: the bundle format version.
        generation: the published generation id.
        capability: ``"resume"`` (weights and training state) or
            ``"inference"`` (weights only).
        source_run_id / source_checkpoint: the run and checkpoint it came from.
        epoch: the checkpoint's epoch.
        model: the network descriptor (a built-in ``Nets`` name or a
            registered ``ModelSpec``).
        model_params: ``NNModelParams.state()`` of the checkpoint.
        components: saved component metadata, ``{name: {"version", "required"}}``.
        calibrators: each calibrator's ``{"payload", "id", "labels", "model_id"}``.
        payloads: ``{name: {"sha256", "size"}}`` from the manifest.
        verified: whether every payload was checked.
    """

    path: str
    version: int
    generation: str
    capability: str
    source_run_id: str
    source_checkpoint: str
    epoch: int
    model: str
    model_params: Mapping[str, Any]
    components: Mapping[str, Any] = field(default_factory=dict)
    calibrators: tuple[Mapping[str, Any], ...] = ()
    payloads: Mapping[str, Any] = field(default_factory=dict)
    verified: bool = False


@dataclass(frozen=True, eq=False)
class ReconstructedBundle:
    """A model rebuilt from a validated bundle, its calibrators and — for a
    ``"resume"`` bundle — the training state :meth:`resume` continues from."""

    model: NNModel
    info: BundleInfo
    calibrators: tuple[TemperatureCalibrator, ...]
    _checkpoint: NNCheckpoint = field(repr=False)
    _training_state: Optional[dict[str, Any]] = field(repr=False)

    @property
    def capability(self) -> str:
        return self.info.capability

    @property
    def resume_checkpoint(self) -> str:
        """The checkpoint label a resumed run records as its parent
        checkpoint (``"bundle-<generation>_e<epoch>"``)."""
        return f"bundle-{self.info.generation[:16]}_e{self.info.epoch}"

    def resume(self, params: NNTrainParams, **train_kwargs: Any) -> NNRun:
        """Continue training :attr:`model` from the bundle's training state:
        ``model.train(params, **train_kwargs)`` as a stateful resume of the
        bundle's run and checkpoint — the optimizer, scheduler, GradScaler,
        RNG and component state are restored and epoch numbering continues,
        exactly as resuming the original run on disk would. ``params`` must
        not name a resume source of its own.

        An ``"inference"`` bundle raises :class:`BundleCapabilityError`
        before anything changes, and so does a ``Trainer`` run's bundle (named
        optimizers): this continues ``NNModel.train`` runs. ``resume_mode`` is
        always ``"stateful"`` here."""
        from .nn.nn_model import _IN_MEMORY_RESUME

        if self._training_state is None:
            raise BundleCapabilityError(
                f"the run bundle {self.info.path!r} is inference-only: it holds no training state (optimizer, "
                "scheduler, GradScaler, RNG and component state), so it cannot resume; export a checkpoint "
                "written by train() (e.g. 'last'), or train the reconstructed model from its weights"
            )
        if self._training_state.get("optimizers") is not None and self._training_state.get("optimizer") is None:
            raise BundleCapabilityError(
                f"the run bundle {self.info.path!r} holds a Trainer's training state (named optimizers); resume() "
                "continues NNModel.train runs only — the model and its weights are still yours to use"
            )
        if self._checkpoint.transforms:
            recipes = ", ".join(transform.name for transform in self._checkpoint.transforms)
            raise BundleCapabilityError(
                f"the run bundle {self.info.path!r} holds a transformed topology ({recipes}): the rebuilt model "
                "serves inference, but cannot take the pre-transform training state; resume the original run"
            )
        if params.resume_from_run_id is not None or params.parent_run_id is not None:
            raise ValueError(
                "resume() resumes from the bundle; params must not set resume_from_run_id or parent_run_id"
            )
        resumed = replace(
            params,
            resume_from_run_id=self.info.source_run_id,
            resume_from_checkpoint=self.resume_checkpoint,
            resume_mode="stateful",
        )
        source = (
            self.info.source_run_id,
            self.resume_checkpoint,
            self._checkpoint,
            copy.deepcopy(self._training_state),
        )
        token = _IN_MEMORY_RESUME.set(source)
        try:
            return self.model.train(params=resumed, **train_kwargs)
        finally:
            _IN_MEMORY_RESUME.reset(token)


# --- typed JSON encoding ---------------------------------------------------------------------


def _encode(value: Any, tensors: Optional[dict[str, torch.Tensor]], path: str) -> Any:
    """``value`` as JSON: tensors become ``{"$tensor": key}`` references into
    ``tensors`` (``None``: tensors are refused), tuples ``$tuple``,
    string-keyed mappings ``$dict``, integer-keyed ones ``$intdict`` and
    non-finite floats ``$float``. Anything else is refused."""
    if isinstance(value, str):
        return str(value)  # a str subclass (NumPy's str_ included) is stored as the plain string
    if isinstance(value, np.generic) and not isinstance(value, (np.bool_, np.integer, np.floating)):
        raise BundleError(f"unsupported state at {path}: {type(value).__name__}")
    if isinstance(value, np.generic):
        value = value.item()  # NumPy scalars are stored as the Python number they hold
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"$float": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
    if isinstance(value, torch.Tensor):
        if tensors is None:
            raise BundleError(f"unsupported state at {path}: a tensor where the bundle holds primitive state only")
        if value.layout is not torch.strided or value.is_quantized:
            raise BundleError(
                f"unsupported state at {path}: a {value.layout} tensor; a run bundle holds dense tensors only"
            )
        if value.dtype not in _CODES:
            raise BundleError(f"unsupported state at {path}: a {value.dtype} tensor has no safetensors dtype")
        key = f"t{len(tensors)}"
        tensors[key] = value.detach().to("cpu").contiguous().clone()  # no shared storage in safetensors
        return {"$tensor": key}
    if isinstance(value, tuple):
        return {"$tuple": [_encode(item, tensors, f"{path}[{index}]") for index, item in enumerate(value)]}
    if isinstance(value, list):
        return [_encode(item, tensors, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        keys = list(value)
        if all(type(key) is str for key in keys):
            return {"$dict": {key: _encode(value[key], tensors, f"{path}.{key}") for key in keys}}
        if all(type(key) is int for key in keys):
            return {"$intdict": [[key, _encode(value[key], tensors, f"{path}[{key}]")] for key in keys]}
        raise BundleError(
            f"unsupported state at {path}: mapping keys must be all strings or all integers, got "
            f"{sorted({type(key).__name__ for key in keys})}"
        )
    raise BundleError(
        f"unsupported state at {path}: {type(value).__name__}; a run bundle holds tensors and JSON primitives "
        "only (there is no pickle fallback)"
    )


def _tensor_refs(value: Any, path: str) -> list[str]:
    """Check an encoded value's tags and return its tensor keys, in order."""
    refs: list[str] = []

    def walk(item: Any, where: str, depth: int = 0) -> None:
        if depth > _MAX_DEPTH:
            raise BundleIntegrityError(f"malformed state at {path}: nested more than {_MAX_DEPTH} levels")
        if item is None or isinstance(item, (bool, str)):
            return
        if isinstance(item, (int, float)):
            if isinstance(item, float) and not math.isfinite(item):
                raise BundleIntegrityError(f"malformed state at {where}: a bare non-finite number")
            return
        if isinstance(item, list):
            for index, element in enumerate(item):
                walk(element, f"{where}[{index}]", depth + 1)
            return
        if not isinstance(item, dict) or len(item) != 1 or next(iter(item)) not in _TAGS:
            raise BundleIntegrityError(f"malformed state at {where}: expected a tagged value ({', '.join(_TAGS)})")
        ((tag, body),) = item.items()
        if tag == "$tensor":
            if not isinstance(body, str) or not re.fullmatch(r"t[0-9]+", body):
                raise BundleIntegrityError(f"malformed state at {where}: a tensor reference must be 't<n>'")
            refs.append(body)
        elif tag == "$float":
            if body not in ("nan", "inf", "-inf"):
                raise BundleIntegrityError(f"malformed state at {where}: $float must be 'nan', 'inf' or '-inf'")
        elif tag == "$tuple":
            if not isinstance(body, list):
                raise BundleIntegrityError(f"malformed state at {where}: $tuple must hold a list")
            walk(body, where, depth + 1)
        elif tag == "$dict":
            if not isinstance(body, dict):
                raise BundleIntegrityError(f"malformed state at {where}: $dict must hold an object")
            for key, element in body.items():
                walk(element, f"{where}.{key}", depth + 1)
        else:  # $intdict
            if not isinstance(body, list) or not all(
                isinstance(pair, list) and len(pair) == 2 and type(pair[0]) is int for pair in body
            ):
                raise BundleIntegrityError(f"malformed state at {where}: $intdict must hold [int, value] pairs")
            if len({pair[0] for pair in body}) != len(body):
                raise BundleIntegrityError(f"malformed state at {where}: duplicate $intdict keys")
            for key, element in body:
                walk(element, f"{where}[{key}]", depth + 1)

    walk(value, path)
    if len(set(refs)) != len(refs):
        raise BundleIntegrityError(f"malformed state at {path}: a tensor is referenced twice")
    return refs


def _decode(value: Any, tensors: Optional[Mapping[str, torch.Tensor]]) -> Any:
    """The inverse of :func:`_encode` for a value :func:`_tensor_refs` accepted."""
    if isinstance(value, list):
        return [_decode(item, tensors) for item in value]
    if not isinstance(value, dict):
        return value
    ((tag, body),) = value.items()
    if tag == "$tensor":
        assert tensors is not None
        return tensors[body]
    if tag == "$float":
        return float(body)
    if tag == "$tuple":
        return tuple(_decode(item, tensors) for item in body)
    if tag == "$dict":
        return {key: _decode(item, tensors) for key, item in body.items()}
    return {key: _decode(item, tensors) for key, item in body}  # $intdict


# --- files -----------------------------------------------------------------------------------


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise BundleIntegrityError(f"duplicate JSON key {key!r}")
        seen[key] = value
    return seen


def _too_deep(value: Any) -> bool:
    """Whether a parsed JSON value nests more than ``_MAX_DEPTH`` levels,
    measured without recursion."""
    stack = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, (dict, list)):
            if depth > _MAX_DEPTH:
                return True
            stack.extend((child, depth + 1) for child in (item.values() if isinstance(item, dict) else item))
    return False


def _parse_json(data: bytes, what: str) -> Any:
    def constant(name: str) -> Any:
        raise BundleIntegrityError(f"{what} is strict JSON; {name} is not a JSON number")

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_no_duplicates, parse_constant=constant)
    except BundleIntegrityError as exc:
        raise BundleIntegrityError(f"{what}: {exc}") from None
    except RecursionError:
        raise BundleIntegrityError(f"{what} is nested too deeply") from None
    except (UnicodeDecodeError, ValueError) as exc:
        raise BundleIntegrityError(f"{what} is not valid UTF-8 JSON: {exc}") from None
    # The decoder's own limit varies by Python version (3.14 parses 100,000
    # levels), so the depth is refused here, the same on every version.
    if _too_deep(value):
        raise BundleIntegrityError(f"{what} is nested too deeply")
    return value


def _dumps(value: Any) -> bytes:
    return (json.dumps(value, indent=1, allow_nan=False, ensure_ascii=False) + "\n").encode("utf-8")


def _open_regular(path: str, what: str, size: Optional[int] = None, *, limit: Optional[int] = None) -> BinaryIO:
    """The regular file at ``path``, opened for reading — never through a
    symlink (refused before and, where the OS can, at open), never a FIFO or
    device (checked on the open descriptor), with the manifest's ``size``
    (or at most ``limit`` bytes) checked before anything is read."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        raise BundleIntegrityError(f"{what} is missing: {path}") from None
    if stat.S_ISLNK(info.st_mode):
        raise BundleIntegrityError(f"{what} is a symlink, which a bundle never holds: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:  # swapped for a symlink after the check (ELOOP), or gone
        raise BundleIntegrityError(f"{what} cannot be opened as a regular file: {exc}") from None
    handle = os.fdopen(descriptor, "rb")
    try:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise BundleIntegrityError(f"{what} is not a regular file: {path}")
        if size is not None and opened.st_size != size:
            raise BundleIntegrityError(f"{what} has {opened.st_size} bytes, the manifest says {size}")
        if limit is not None and opened.st_size > limit:
            raise BundleIntegrityError(f"{what} has {opened.st_size} bytes, more than a bundle ever holds there")
    except BaseException:
        handle.close()
        raise
    return handle


def _read_regular(path: str, what: str, size: Optional[int] = None, *, limit: Optional[int] = None) -> bytes:
    """The whole regular file at ``path`` (see :func:`_open_regular`)."""
    with _open_regular(path, what, size, limit=limit) as handle:
        data = handle.read((size if size is not None else limit if limit is not None else -2) + 1)
    if size is not None and len(data) != size:
        raise BundleIntegrityError(f"{what} has {len(data)} bytes, the manifest says {size}")
    return data


def _bundle_root(path: Union[str, os.PathLike[str]]) -> str:
    """The bundle directory, or an error that says what ``path`` is instead
    (a checkpoint file, a Hub distribution, nothing). Never downloads."""
    root = os.fspath(path)
    if not os.path.lexists(root):
        raise BundleError(
            f"no run bundle at {root!r}: bundles are local directories and nothing is downloaded "
            "(a Hugging Face Hub model loads with NNModel.from_pretrained)"
        )
    if not os.path.isdir(root):
        enclosing = _enclosing_bundle(root)
        if enclosing is not None:
            raise BundleError(f"{root!r} is a file inside the run bundle {enclosing!r}; pass the bundle directory")
        raise BundleError(
            f"{root!r} is not a run bundle directory: a bundle is the directory that holds {_MANIFEST} (an NNx "
            "pickle checkpoint loads with NNCheckpoint.from_file, and only when you produced it)"
        )
    if not os.path.lexists(os.path.join(root, _MANIFEST)):
        if os.path.exists(os.path.join(root, "config.json")):
            raise BundleError(
                f"{root!r} is not a run bundle: it has no {_MANIFEST} but a config.json — a Hugging Face Hub "
                "distribution, which loads with NNModel.from_pretrained"
            )
        raise BundleError(f"{root!r} is not a run bundle: it has no {_MANIFEST}")
    return root


def _enclosing_bundle(path: str) -> Optional[str]:
    """The bundle directory ``path`` lies in (its manifest, or a payload in a
    generation directory), if any."""
    parent = os.path.dirname(os.path.abspath(path))
    for candidate in (parent, os.path.dirname(parent)):
        if os.path.isfile(os.path.join(candidate, _MANIFEST)):
            return candidate
    return None


def _read_manifest(root: str) -> dict[str, Any]:
    path = os.path.join(root, _MANIFEST)
    return _check_manifest(_parse_json(_read_regular(path, _MANIFEST, limit=_MAX_MANIFEST), _MANIFEST))


def _check_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise BundleIntegrityError(f"{_MANIFEST} must hold an object")
    if manifest.get("format") != BUNDLE_FORMAT:
        raise BundleIntegrityError(f"{_MANIFEST} is not an NNx run bundle manifest (format {manifest.get('format')!r})")
    version = manifest.get("version")
    if version != BUNDLE_VERSION or type(version) is not int:
        raise BundleIntegrityError(
            f"unsupported run bundle version {version!r}; this NNx reads version {BUNDLE_VERSION} "
            "(there is no fallback to another format)"
        )
    expected = {"format", "version", "generation", "payloads"}
    if set(manifest) != expected:
        raise BundleIntegrityError(f"{_MANIFEST} must hold exactly {sorted(expected)}, got {sorted(manifest)}")
    generation = manifest["generation"]
    if not isinstance(generation, str) or not _GENERATION.fullmatch(generation):
        raise BundleIntegrityError(f"{_MANIFEST} has a malformed generation id {generation!r}")
    payloads = manifest["payloads"]
    if not isinstance(payloads, dict) or _STATE not in payloads or _MODEL not in payloads:
        raise BundleIntegrityError(f"{_MANIFEST} must list the payloads {_STATE} and {_MODEL}")
    for name, entry in payloads.items():
        if not _PAYLOAD.fullmatch(name):
            raise BundleIntegrityError(
                f"{_MANIFEST} names the payload {name!r}, outside the bundle's payload names "
                "(paths, '..' and other directories are never read)"
            )
        if (
            not isinstance(entry, dict)
            or set(entry) != {"sha256", "size"}
            or not isinstance(entry["sha256"], str)
            or not _SHA256.fullmatch(entry["sha256"])
            or type(entry["size"]) is not int
            or entry["size"] < 0
        ):
            raise BundleIntegrityError(f"{_MANIFEST} has a malformed entry for {name!r}: {{'sha256', 'size'}}")
    return manifest


def _generation_dir(root: str, manifest: Mapping[str, Any]) -> str:
    directory = os.path.join(root, f"g-{manifest['generation']}")
    if os.path.islink(directory) or not os.path.isdir(directory):
        raise BundleIntegrityError(
            f"the published generation {manifest['generation']} has no directory of its own at {directory}"
        )
    return directory


def _payload_bytes(directory: str, name: str, entry: Mapping[str, Any]) -> bytes:
    """A payload's bytes, after checking it is a regular file inside the
    generation directory with the manifest's size and SHA-256."""
    if name.endswith(".json") and entry["size"] > _MAX_JSON:
        raise BundleIntegrityError(
            f"payload {name!r} declares {entry['size']} bytes, more than a bundle ever holds there"
        )
    data = _read_regular(os.path.join(directory, name), f"payload {name!r}", entry["size"])
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise BundleIntegrityError(f"payload {name!r} does not match its manifest SHA-256 (altered or swapped)")
    return data


def _safetensors_header(data: bytes, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """A safetensors payload's ``(tensor entries, metadata)``, parsed from its
    JSON header alone — no tensor is read."""
    length = _header_length(data[:8], len(data), name)
    return _parse_header(data[8 : 8 + length], len(data) - 8 - length, name)


def _header_length(prefix: bytes, total: int, name: str) -> int:
    if len(prefix) < 8 or total < 8:
        raise BundleIntegrityError(f"payload {name!r} is not a safetensors file (too short)")
    length = int.from_bytes(prefix[:8], "little")
    if length > min(total - 8, _MAX_HEADER):
        raise BundleIntegrityError(f"payload {name!r} has a malformed safetensors header length")
    return length


def _parse_header(text: bytes, body: int, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The ``(tensor entries, metadata)`` of a safetensors JSON header over a
    data section of ``body`` bytes, checked as the format requires."""
    header = _parse_json(text, f"payload {name!r} header")
    if not isinstance(header, dict):
        raise BundleIntegrityError(f"payload {name!r} has a malformed safetensors header")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict):
        raise BundleIntegrityError(f"payload {name!r} has malformed safetensors metadata")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()):
        raise BundleIntegrityError(f"payload {name!r} has malformed safetensors metadata")
    spans = []
    for key, info in header.items():
        offsets = info.get("data_offsets") if isinstance(info, dict) else None
        shape = info.get("shape") if isinstance(info, dict) else None
        dtype = _BY_CODE.get(str(info.get("dtype"))) if isinstance(info, dict) else None
        if (
            dtype is None
            or set(info) != {"dtype", "shape", "data_offsets"}
            or not isinstance(shape, list)
            or len(shape) > _MAX_DEPTH
            or not all(type(size) is int and 0 <= size <= _MAX_DIM for size in shape)
            or not _strides_fit(shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(type(offset) is int for offset in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= body
            or offsets[1] - offsets[0] != math.prod(shape) * _itemsize(dtype)
        ):
            raise BundleIntegrityError(f"payload {name!r} has a malformed entry for tensor {key!r}")
        spans.append((offsets[0], offsets[1]))
    position = 0  # the tensors tile the data section exactly, as the safetensors format requires
    for start, end in sorted(spans):
        if start != position:
            raise BundleIntegrityError(f"payload {name!r} has overlapping or gapped tensor data")
        position = end
    if position != body:
        raise BundleIntegrityError(f"payload {name!r} has data no tensor describes")
    return header, metadata


# The safetensors dtype codes NNx writes and reads (the format's own names).
_CODES = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}
for _name, _code in (("float8_e4m3fn", "F8_E4M3"), ("float8_e5m2", "F8_E5M2")):
    if hasattr(torch, _name):
        _CODES[getattr(torch, _name)] = _code
_BY_CODE = {code: dtype for dtype, code in _CODES.items()}


def _strides_fit(shape: list[int]) -> bool:
    """Whether a tensor of ``shape`` has strides that fit 64 bits — its
    non-zero dimensions' product (a zero dimension empties the tensor, not
    its strides)."""
    product = 1
    for size in shape:
        product *= max(size, 1)
        if product > _MAX_DIM:
            return False
    return True


def _itemsize(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _tensor_bytes(tensor: torch.Tensor, where: str) -> bytes:
    """A dense tensor's elements as little-endian bytes, in C order."""
    if tensor.dtype not in _CODES:
        raise BundleError(f"unsupported state at {where}: a {tensor.dtype} tensor has no safetensors dtype")
    flat = tensor.detach().to("cpu").resolve_conj().resolve_neg().contiguous().reshape(-1)
    raw = flat.view(torch.uint8)
    if sys.byteorder == "big" and flat.element_size() > 1:
        raw = raw.reshape(-1, flat.element_size()).flip(-1).reshape(-1)
    return raw.numpy().tobytes()


def _tensor_chunks(tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str]) -> Iterator[bytes]:
    """``tensors`` in the safetensors format, one piece at a time — an
    8-byte little-endian header length, a JSON header (``dtype``, ``shape``,
    ``data_offsets`` per tensor, plus ``__metadata__``) padded to 8 bytes,
    then each tensor's bytes — so any safetensors reader opens it. Written by
    NNx itself: no optional dependency."""
    header: dict[str, Any] = {"__metadata__": dict(metadata)}
    offset = 0
    for key in sorted(tensors):
        tensor = tensors[key]
        if tensor.dtype not in _CODES:
            raise BundleError(f"unsupported state at {key}: a {tensor.dtype} tensor has no safetensors dtype")
        size = tensor.numel() * tensor.element_size()
        header[key] = {
            "dtype": _CODES[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    text = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    text += b" " * (-len(text) % 8)
    yield len(text).to_bytes(8, "little") + text
    for key in sorted(tensors):
        yield _tensor_bytes(tensors[key], key)


def _save_tensors(tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str]) -> bytes:
    """:func:`_tensor_chunks` as one ``bytes`` object."""
    return b"".join(_tensor_chunks(tensors, metadata))


def _verify_tensor_payload(directory: str, name: str, entry: Mapping[str, Any]) -> tuple[dict, dict]:
    """A tensor payload's size and SHA-256, hashed as it streams by (memory
    bounded by its header, not its tensors), and its checked header."""
    what = f"payload {name!r}"
    with _open_regular(os.path.join(directory, name), what, entry["size"]) as handle:
        prefix = handle.read(8)
        length = _header_length(prefix, entry["size"], name)
        text = handle.read(length)
        digest = hashlib.sha256(prefix + text)
        header, metadata = _parse_header(text, entry["size"] - 8 - length, name)
        booleans = sorted(
            (info["data_offsets"][0], info["data_offsets"][1], key)
            for key, info in header.items()
            if info["dtype"] == "BOOL"
        )
        position = 0  # within the data section
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
            for begin, end, key in booleans:  # every byte of a bool tensor is 0 or 1
                low, high = max(begin, position), min(end, position + len(chunk))
                if low < high and chunk[low - position : high - position].translate(None, b"\x00\x01"):
                    raise BundleIntegrityError(f"{what}: the boolean tensor {key!r} holds bytes other than 0 and 1")
            position += len(chunk)
    if 8 + length + position != entry["size"] or digest.hexdigest() != entry["sha256"]:
        raise BundleIntegrityError(f"{what} does not match its manifest SHA-256 (altered or swapped)")
    return header, metadata


def _payload_fingerprint(directory: str, name: str, entry: Mapping[str, Any]) -> str:
    """``nnx.calibration.model_fingerprint`` of a tensor payload's weights,
    computed from the file — each tensor's little-endian bytes streamed in
    name order — without building a tensor."""
    with _open_regular(os.path.join(directory, name), f"payload {name!r}", entry["size"]) as handle:
        prefix = handle.read(8)
        length = _header_length(prefix, entry["size"], name)
        header, _ = _parse_header(handle.read(length), entry["size"] - 8 - length, name)
        digest = hashlib.sha256(_FINGERPRINT_SEED)
        for key in sorted(header):
            info = header[key]
            begin, end = info["data_offsets"]
            dtype, shape = _BY_CODE[info["dtype"]], tuple(info["shape"])
            digest.update(_fingerprint_header(key, "tensor", dtype, shape, math.prod(shape)))
            handle.seek(8 + length + begin)
            remaining = end - begin
            while remaining:
                chunk = handle.read(min(remaining, 1 << 20))
                if not chunk:
                    raise BundleIntegrityError(f"payload {name!r} ended inside tensor {key!r}")
                digest.update(chunk)
                remaining -= len(chunk)
    return f"sha256:{digest.hexdigest()}"


def _load_tensors(data: bytes, name: str) -> dict[str, torch.Tensor]:
    """The tensors of a safetensors payload whose header
    :func:`_safetensors_header` accepts, as CPU tensors."""
    header, _ = _safetensors_header(data, name)
    start = 8 + int.from_bytes(data[:8], "little")
    tensors: dict[str, torch.Tensor] = {}
    for key, info in header.items():
        dtype, shape = _BY_CODE[info["dtype"]], tuple(info["shape"])
        begin, end = info["data_offsets"]
        if begin == end:
            tensors[key] = torch.empty(shape, dtype=dtype)
            continue
        raw = torch.frombuffer(bytearray(memoryview(data)[start + begin : start + end]), dtype=torch.uint8)
        if dtype is torch.bool and bool((raw > 1).any()):
            raise BundleIntegrityError(f"payload {name!r}: the boolean tensor {key!r} holds bytes other than 0 and 1")
        size = _itemsize(dtype)
        if sys.byteorder == "big" and size > 1:
            raw = raw.reshape(-1, size).flip(-1).reshape(-1).contiguous()
        tensors[key] = raw.view(dtype).reshape(shape)
    return tensors


# --- inspect / validate ----------------------------------------------------------------------


@dataclass(frozen=True)
class _Read:
    root: str
    directory: str
    manifest: dict[str, Any]
    state: dict[str, Any]
    info: BundleInfo
    payloads: dict[str, bytes]
    parts: dict[str, Any]
    calibrators: tuple[TemperatureCalibrator, ...]


def _check_state(state: Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise BundleIntegrityError(f"{_STATE} must hold an object")
    expected = {
        "format",
        "version",
        "generation",
        "source",
        "capability",
        "checkpoint",
        "net_state",
        "training_state",
        "calibrators",
    }
    if set(state) != expected:
        raise BundleIntegrityError(f"{_STATE} must hold exactly {sorted(expected)}, got {sorted(state)}")
    if state["format"] != BUNDLE_FORMAT or type(state["version"]) is not int or state["version"] != BUNDLE_VERSION:
        raise BundleIntegrityError(f"{_STATE} names another format or version")
    if state["generation"] != manifest["generation"]:
        raise BundleIntegrityError(
            f"{_STATE} belongs to generation {state['generation']!r}, the manifest publishes "
            f"{manifest['generation']!r} (a payload from another bundle generation)"
        )
    source = state["source"]
    if (
        not isinstance(source, dict)
        or set(source) != {"run_id", "checkpoint"}
        or not all(isinstance(source[key], str) and source[key] for key in source)
    ):
        raise BundleIntegrityError(f"{_STATE} has a malformed source {{'run_id', 'checkpoint'}}")
    capability = state["capability"]
    if capability not in _CAPABILITIES:
        raise BundleIntegrityError(f"{_STATE} has an unknown capability {capability!r}")
    if (capability == "resume") != (state["training_state"] is not None):
        raise BundleIntegrityError(
            f"{_STATE}: a {capability!r} bundle must {'' if capability == 'resume' else 'not '}hold training state"
        )
    if (capability == "resume") != (_TRAINING in manifest["payloads"]):
        raise BundleIntegrityError(f"the manifest's {_TRAINING} payload contradicts the {capability!r} capability")
    from .nn.params.nn_run import _validate_run_id

    try:
        _validate_run_id(source["run_id"])
    except ValueError as exc:
        raise BundleIntegrityError(f"{_STATE} has a malformed source run id: {exc}") from None
    checkpoint = state["checkpoint"]
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != {"$dict"}
        or not isinstance(checkpoint["$dict"], dict)
        or set(checkpoint["$dict"]) != {"model_params", "net_params", "idp", "transforms"}
    ):
        raise BundleIntegrityError(f"{_STATE} has a malformed checkpoint section")
    if _tensor_refs(checkpoint, "checkpoint"):
        raise BundleIntegrityError(f"{_STATE}: the checkpoint section holds primitive state only, not tensors")
    training = state["training_state"]
    if training is not None:
        _tensor_refs(training, "training_state")  # every tag checked, even when only inspecting
        body = training.get("$dict") if isinstance(training, dict) else None
        if not isinstance(body, dict):
            raise BundleIntegrityError(f"{_STATE}: training_state must be a mapping")
        from .nn.params.nn_checkpoint import _TRAINING_STATE_FORMAT_VERSION

        version = body.get("nnx_training_state_version")
        if type(version) is not int or not 0 <= version <= _TRAINING_STATE_FORMAT_VERSION:
            raise BundleIntegrityError(
                f"{_STATE}: unsupported training-state version {version!r}; this NNx reads 0..{_TRAINING_STATE_FORMAT_VERSION} "
                "(there is no fallback)"
            )
        completed = body.get("completed_epoch")
        if completed is not None and (type(completed) is not int or completed < 0):
            raise BundleIntegrityError(f"{_STATE}: training_state.completed_epoch must be a non-negative integer")
        if body.get("optimizer") is None and body.get("optimizers") is None:
            raise BundleIntegrityError(f"{_STATE}: training_state holds no optimizer state")
        if body.get("model") is not None and not checkpoint["$dict"]["transforms"]:
            # train() records pre-transform weights only for a transformed topology;
            # otherwise a resume would train from weights model.safetensors never showed.
            raise BundleIntegrityError(f"{_STATE}: training_state holds pre-transform weights for no transform")
        components = body.get("components")
        if components is not None and not (
            isinstance(components, dict)
            and isinstance(components.get("$dict"), dict)
            and all(_component_entry(entry) for entry in components["$dict"].values())
        ):
            raise BundleIntegrityError(
                f"{_STATE}: training_state.components must map names to {{version: int >= 1, required: bool, state}}"
            )
    net_state = state["net_state"]
    if not isinstance(net_state, list) or not all(isinstance(key, str) for key in net_state):
        raise BundleIntegrityError(f"{_STATE} must list the network's tensor names in net_state")
    if len(set(net_state)) != len(net_state):
        raise BundleIntegrityError(f"{_STATE} lists a network tensor twice")
    calibrators = state["calibrators"]
    if not isinstance(calibrators, list) or not all(
        isinstance(name, str) and name.startswith("calibrator-") for name in calibrators
    ):
        raise BundleIntegrityError(f"{_STATE} has a malformed calibrator list")
    listed = sorted(name for name in manifest["payloads"] if name.startswith("calibrator-"))
    if sorted(calibrators) != listed:
        raise BundleIntegrityError(f"{_STATE} names the calibrators {calibrators}, the manifest {listed}")
    return state


def _component_entry(entry: Any) -> bool:
    """Whether an encoded component entry is ``{version, required, state}``
    as ``ComponentRegistry.collect`` writes it."""
    body = entry.get("$dict") if isinstance(entry, dict) else None
    if not isinstance(body, dict) or set(body) != {"version", "required", "state"}:
        return False
    version, required = body["version"], body["required"]
    return type(version) is int and version >= 1 and type(required) is bool


def _checkpoint_parts(state: Mapping[str, Any]) -> dict[str, Any]:
    """The checkpoint section as NNx objects (no tensors, no factory)."""
    from .nn.params.nn_checkpoint import NNCheckpointTransform, _idp_from_nested_state
    from .nn.params.nn_model_params import NNModelParams
    from .nn.params.nn_params import NNParams

    section = _decode(state["checkpoint"], None)
    try:
        model_params = NNModelParams.from_state(section["model_params"])
        net_params = None if section["net_params"] is None else NNParams.resolve_from_state(section["net_params"])
        idp = _idp_from_nested_state(section["idp"])
        transforms = tuple(NNCheckpointTransform.from_state(transform) for transform in section["transforms"])
    except BundleError:
        raise
    except Exception as exc:  # malformed primitive state: the from_state readers' own errors
        raise BundleIntegrityError(f"{_STATE} has a malformed checkpoint: {type(exc).__name__}: {exc}") from None
    if isinstance(model_params.net, RuntimeModule):
        raise BundleIntegrityError(f"{_STATE} describes a runtime-only module, which a bundle never holds")
    if type(idp.epoch_idx) is not int or idp.epoch_idx < 0:
        raise BundleIntegrityError(f"{_STATE} has a malformed checkpoint epoch {idp.epoch_idx!r}")
    return {"model_params": model_params, "net_params": net_params, "idp": idp, "transforms": transforms}


def _info(
    root: str,
    manifest: Mapping[str, Any],
    state: Mapping[str, Any],
    parts: Mapping[str, Any],
    calibrators: Iterable[Mapping[str, Any]],
    *,
    verified: bool,
) -> BundleInfo:
    training = state["training_state"]
    components: dict[str, Any] = {}
    body = training.get("$dict") if isinstance(training, dict) else None
    saved = body.get("components") if isinstance(body, dict) else None
    if saved is not None:  # names and versions only: tensor references read as None
        for name, entry in (_decode(_strip_tensors(saved), None) or {}).items():
            if isinstance(entry, Mapping):
                components[name] = {"version": entry.get("version"), "required": bool(entry.get("required", True))}
    net = parts["model_params"].net
    return BundleInfo(
        path=root,
        version=manifest["version"],
        generation=manifest["generation"],
        capability=state["capability"],
        source_run_id=state["source"]["run_id"],
        source_checkpoint=state["source"]["checkpoint"],
        epoch=parts["idp"].epoch_idx,
        model=str(net),
        model_params=_freeze_config(parts["model_params"].state(), "", owner="BundleInfo"),
        components=_freeze_config(components, "", owner="BundleInfo"),
        calibrators=tuple(_freeze_config(summary, "", owner="BundleInfo") for summary in calibrators),
        payloads=_freeze_config(manifest["payloads"], "", owner="BundleInfo"),
        verified=verified,
    )


def _strip_tensors(value: Any) -> Any:
    """An encoded value with every tensor reference replaced by ``None`` —
    enough to read component names and versions without the tensors."""
    if isinstance(value, list):
        return [_strip_tensors(item) for item in value]
    if isinstance(value, dict):
        ((tag, body),) = value.items()
        if tag == "$tensor":
            return None
        if tag == "$dict":
            return {"$dict": {key: _strip_tensors(item) for key, item in body.items()}}
        if tag == "$intdict":
            return {"$intdict": [[key, _strip_tensors(item)] for key, item in body]}
        if tag == "$tuple":
            return {"$tuple": [_strip_tensors(item) for item in body]}
    return value


def _calibrator_summary(name: str, calibrator: TemperatureCalibrator) -> dict[str, Any]:
    return {"payload": name, "id": calibrator.id, "labels": list(calibrator.labels), "model_id": calibrator.model_id}


def _read(path: Union[str, os.PathLike[str]], *, full: bool) -> _Read:
    root = _bundle_root(path)
    manifest = _read_manifest(root)
    return _read_generation(root, manifest, _generation_dir(root, manifest), full=full)


def _read_generation(root: str, manifest: dict[str, Any], directory: str, *, full: bool) -> _Read:
    """Check (``full``: every payload; otherwise ``state.json`` and the
    calibrator records) and summarize the generation ``manifest`` publishes
    in ``directory`` — also run by an export on its staged generation."""
    payloads: dict[str, bytes] = {}
    names = manifest["payloads"]
    if full:
        present = set(os.listdir(directory))
        extra, missing = sorted(present - set(names)), sorted(set(names) - present)
        if missing:
            raise BundleIntegrityError(f"the bundle is missing the payloads {missing}")
        if extra:
            raise BundleIntegrityError(f"the bundle's generation directory holds unlisted files {extra}")
        for name in sorted(names):
            if name.endswith(".json"):
                payloads[name] = _payload_bytes(directory, name, names[name])
    else:
        payloads[_STATE] = _payload_bytes(directory, _STATE, names[_STATE])
    state = _check_state(_parse_json(payloads[_STATE], _STATE), manifest)
    parts = _checkpoint_parts(state)
    calibrators = []
    for name in state["calibrators"]:
        data = payloads.get(name)
        if data is None:
            data = _payload_bytes(directory, name, names[name])
        record = _parse_json(data, f"payload {name!r}")  # strict: no duplicate keys, no NaN
        try:
            calibrator = TemperatureCalibrator.from_state(record)
        except Exception as exc:
            raise BundleIntegrityError(f"payload {name!r} is not a calibrator record: {exc}") from None
        problem = _label_problem(calibrator, parts["model_params"], parts["net_params"])
        if problem is not None:
            raise BundleIntegrityError(f"payload {name!r}: {problem}")
        calibrators.append((name, calibrator))
    if full:
        refs = [] if state["training_state"] is None else _tensor_refs(state["training_state"], "training_state")
        for name, expected, kind in ((_MODEL, state["net_state"], "model"), (_TRAINING, refs, "training")):
            if name not in names:
                continue
            header, metadata = _verify_tensor_payload(directory, name, names[name])
            if metadata != {"nnx.bundle": kind, "generation": manifest["generation"]}:
                raise BundleIntegrityError(f"payload {name!r} belongs to another bundle generation or payload kind")
            if set(header) != set(expected):
                raise BundleIntegrityError(
                    f"payload {name!r} holds the tensors {sorted(header)}, {_STATE} expects {sorted(expected)}"
                )
        fingerprinted = [calibrator for _, calibrator in calibrators if calibrator.model_id.startswith("sha256:")]
        if fingerprinted:  # every integrity check passed: hash the weights as model_fingerprint would
            fingerprint = _payload_fingerprint(directory, _MODEL, names[_MODEL])
            mismatched = [
                f"calibrator {calibrator.id} was fitted on the model {calibrator.model_id}, not on the bundled "
                f"weights ({fingerprint})"
                for calibrator in fingerprinted
                if calibrator.model_id != fingerprint
            ]
            if mismatched:
                raise BundleIntegrityError("; ".join(mismatched))
    summaries = [_calibrator_summary(name, calibrator) for name, calibrator in calibrators]
    info = _info(root, manifest, state, parts, summaries, verified=full)
    return _Read(root, directory, manifest, state, info, payloads, parts, tuple(c for _, c in calibrators))


def inspect_bundle(path: Union[str, os.PathLike[str]]) -> BundleInfo:
    """Summarize the bundle at ``path`` from its manifest, ``state.json`` and
    calibrator records (each checked against its manifest size and SHA-256).
    Tensor payloads are not read or checked — :func:`validate_bundle` checks
    them. Never unpickles, calls a factory or downloads."""
    return _read(path, full=False).info


def validate_bundle(path: Union[str, os.PathLike[str]]) -> BundleInfo:
    """Check the whole bundle at ``path`` and summarize it: every listed
    payload present, a regular file inside the published generation
    directory with the manifest's size and SHA-256, nothing unlisted, one
    generation id throughout, strict JSON without duplicate keys, and
    safetensors headers holding exactly the tensors ``state.json``
    references. No tensor is read before all of that passes (a calibrator
    with a fingerprint ``model_id`` is then checked against the weights);
    nothing is unpickled, and no factory is called. Raises :class:`BundleIntegrityError` (or :class:`BundleError`
    for a path that is not a bundle)."""
    return _read(path, full=True).info


# --- export ----------------------------------------------------------------------------------


def _label_problem(calibrator: TemperatureCalibrator, model_params: Any, net_params: Any) -> Optional[str]:
    """Why ``calibrator`` does not fit the model's classes, if it does not:
    a task that is not categorical, other labels than the task's, or
    another class count than the model's outputs."""
    task = model_params.task
    if task is not None and task.kind != "categorical":
        return f"calibrator {calibrator.id} calibrates categorical logits, the bundled model's task is {task.kind!r}"
    labels = getattr(task, "labels", None) if task is not None else None
    if labels is not None and tuple(labels) != calibrator.labels:
        return (
            f"calibrator {calibrator.id} was fitted for the labels {list(calibrator.labels)}, the bundled model's "
            f"task declares {list(labels)}"
        )
    classes = task.num_outputs if task is not None else None
    if classes is None:  # a categorical task may leave its class count to the network
        classes = getattr(net_params, "output_dim", None)
    if classes is not None and classes != len(calibrator.labels):
        return f"calibrator {calibrator.id} has {len(calibrator.labels)} labels, the bundled model {classes} classes"
    return None


def _fingerprint_problems(
    calibrators: Iterable[TemperatureCalibrator], net_state: Mapping[str, torch.Tensor]
) -> list[str]:
    """Each fingerprinted calibrator (``model_id`` ``sha256:...``) fitted on
    other weights than ``net_state`` — the weights hashed once."""
    fingerprinted = [calibrator for calibrator in calibrators if calibrator.model_id.startswith("sha256:")]
    if not fingerprinted:
        return []
    fingerprint = _state_fingerprint(net_state)
    return [
        f"calibrator {calibrator.id} was fitted on the model {calibrator.model_id}, not on the bundled weights "
        f"({fingerprint}); bundle the checkpoint it was fitted on"
        for calibrator in fingerprinted
        if calibrator.model_id != fingerprint
    ]


def _write(path: str, chunks: Iterable[bytes]) -> dict[str, Any]:
    """Write ``chunks`` to a new file at ``path``; its manifest entry."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "xb") as handle:
        for chunk in chunks:
            handle.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:  # not every filesystem supports it (some network mounts)
            pass
    return {"sha256": digest.hexdigest(), "size": size}


def _fsync_directory(path: str) -> None:
    """Persist ``path``'s entries where the OS allows it (not on Windows)."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def export_bundle(
    run_id: str,
    destination: Union[str, os.PathLike[str]],
    *,
    checkpoint: str = "last",
    root: Optional[str] = None,
    calibrators: Iterable[TemperatureCalibrator] = (),
) -> BundleInfo:
    """Publish ``run_id``'s ``checkpoint`` as a run bundle at ``destination``.

    Args:
        run_id: a run under ``<root>/runs`` (the current directory by default)
            — your own run: its pickle checkpoint is read locally.
        destination: an empty or missing directory, or an existing bundle,
            whose generation this export replaces.
        checkpoint: a ``Checkpoints`` tag (``"last"``, ``"best"``, …) or a
            ``ModelCheckpoint`` file stem ``"<tag>_e<epoch>"``.
        calibrators: ``TemperatureCalibrator`` records to ship with the
            model. Their labels must match the task's, and a fingerprint
            ``model_id`` (``nnx.calibration.model_fingerprint``) must be the
            bundled weights'.

    Returns the published bundle's :class:`BundleInfo` (``verified=True``).
    A checkpoint written by ``train()`` gives a ``"resume"`` bundle; a
    weights-only one an ``"inference"`` bundle. A runtime-only module, a
    recorded topology transform NNx cannot replay from data (anything but an
    ``nnx.transforms`` recipe operation of a known version or a torchao QAT
    conversion — the error names its index and id), module extra state that
    is not a tensor, and training state that is not tensors and JSON
    primitives are refused before anything is written.
    """
    from filelock import FileLock

    from .nn.nn_model import _resume_checkpoint_type, _unportable_transform
    from .nn.params.nn_checkpoint import NNCheckpoint

    label = str(_resume_checkpoint_type(checkpoint))
    ckpt, training_state = NNCheckpoint.load_with_training_state(run=run_id, type=label, root=root)  # type: ignore[arg-type]
    if ckpt is None:
        raise BundleError(f"run {run_id!r} has no {label!r} checkpoint under {os.path.join(root or '.', 'runs')}")
    if not ckpt.reconstructible:
        raise MissingModelFactoryError(
            f"a run bundle is portable, but {ckpt.model_params.net} is a runtime-only module (reconstructible=False) "
            "with no model factory to rebuild it; register one (nnx.models.register_model_factory) and train from a "
            "ModelSpec"
        )
    unportable = _unportable_transform(ckpt.transforms)
    if unportable is not None:
        raise BundleError(
            f"a run bundle is portable, but {unportable}: a bundle carries only operations NNx replays from data "
            "(nnx.transforms recipe operations, a torchao QAT conversion); keep this run in its pickle checkpoints"
        )
    non_tensors = [key for key, value in ckpt.net_state.items() if not isinstance(value, torch.Tensor)]
    if non_tensors:
        raise BundleError(
            f"unsupported extra state {non_tensors}: a run bundle holds tensors and JSON primitives only "
            "(there is no pickle fallback); keep this run in its pickle checkpoints"
        )
    net_tensors: dict[str, torch.Tensor] = {}
    for key, value in ckpt.net_state.items():
        if value.layout is not torch.strided or value.is_quantized:
            kind = "quantized" if value.is_quantized else str(value.layout)
            raise BundleError(f"unsupported extra state {key!r}: a {kind} tensor; bundles hold dense tensors")
        if value.dtype not in _CODES:
            raise BundleError(f"unsupported extra state {key!r}: a {value.dtype} tensor has no safetensors dtype")
        net_tensors[key] = value.detach().to("cpu").contiguous().clone()
    calibrators = tuple(calibrators)
    for calibrator in calibrators:
        if not isinstance(calibrator, TemperatureCalibrator):
            raise TypeError(
                f"calibrators must be nnx.calibration.TemperatureCalibrator, got {type(calibrator).__name__}"
            )
    problems = [
        problem
        for problem in (_label_problem(calibrator, ckpt.model_params, ckpt.net_params) for calibrator in calibrators)
        if problem is not None
    ]
    problems += _fingerprint_problems(calibrators, net_tensors)
    if problems:
        raise BundleError("; ".join(problems))
    training_tensors: dict[str, torch.Tensor] = {}
    encoded_training = None if training_state is None else _encode(training_state, training_tensors, "training_state")
    generation = uuid.uuid4().hex
    state = {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "generation": generation,
        "source": {"run_id": run_id, "checkpoint": label},
        "capability": "inference" if training_state is None else "resume",
        "checkpoint": _encode(
            {
                "model_params": ckpt.model_params.state(),
                "net_params": None if ckpt.net_params is None else ckpt.net_params.state(),
                "idp": ckpt.idp.state(),
                "transforms": [transform.state() for transform in ckpt.transforms],
            },
            None,
            "checkpoint",
        ),
        "net_state": list(net_tensors),
        "training_state": encoded_training,
        "calibrators": [f"calibrator-{index}.json" for index in range(len(calibrators))],
    }
    json_payloads = {_STATE: _dumps(state)}
    for index, calibrator in enumerate(calibrators):
        json_payloads[f"calibrator-{index}.json"] = calibrator.to_json().encode("utf-8")
    tensor_payloads = {_MODEL: (net_tensors, "model")}
    if training_state is not None:
        tensor_payloads[_TRAINING] = (training_tensors, "training")

    target = os.fspath(destination)
    _previous_generation(target)  # refuse an unusable destination before creating anything
    os.makedirs(target, exist_ok=True)
    with FileLock(os.path.join(target, _LOCK)):
        previous = _previous_generation(target)  # again, now that no other export runs
        for entry in os.listdir(target):
            if _TEMPORARY.fullmatch(entry):
                os.remove(os.path.join(target, entry))
        directory = os.path.join(target, f"g-{generation}")
        os.mkdir(directory)
        published = False
        try:
            entries = {}
            for name, data in json_payloads.items():
                entries[name] = _write(os.path.join(directory, name), [data])
            for name, (tensors, kind) in tensor_payloads.items():
                chunks = _tensor_chunks(tensors, {"nnx.bundle": kind, "generation": generation})
                entries[name] = _write(os.path.join(directory, name), chunks)
            _fsync_directory(directory)
            # Every copy of the tensors goes before the staged read loads the payloads back.
            del tensors, tensor_payloads, net_tensors, training_tensors, training_state, ckpt
            manifest = _check_manifest(
                {
                    "format": BUNDLE_FORMAT,
                    "version": BUNDLE_VERSION,
                    "generation": generation,
                    "payloads": dict(sorted(entries.items())),
                }
            )
            staged = _read_generation(target, manifest, directory, full=True)  # what readers will see, checked first
            _fsync_directory(target)  # the new generation's entry persists before the manifest names it
            _publish(target, _dumps(manifest).decode("utf-8"))
            published = True
        except BaseException:
            if not published and not _publishes(target, generation):  # never the generation bundle.json names
                shutil.rmtree(directory, ignore_errors=True)
            raise
        # The generation just replaced stays until the next export, so a reader
        # that read the old manifest can finish; older ones (and any an
        # interrupted export left behind) go.
        keep = {f"g-{generation}", f"g-{previous}"}
        for entry in os.listdir(target):
            if entry not in keep and _is_generation(target, entry):
                shutil.rmtree(os.path.join(target, entry), ignore_errors=True)
        return staged.info


def _publishes(target: str, generation: str) -> bool:
    """Whether ``bundle.json`` at ``target`` may already name ``generation``
    (an interruption after the rename). Unknown — the manifest unreadable
    for another reason than being malformed — counts as yes: the directory
    is kept, and the next export clears it if it is not published."""
    try:
        return _read_manifest(target)["generation"] == generation
    except BundleError:
        return False
    except Exception:
        return True


def _previous_generation(target: str) -> Optional[str]:
    """The generation an existing bundle at ``target`` publishes (``None``
    for a missing or empty directory); anything else is refused: a
    directory with other contents, or a ``bundle.json`` that is not a
    readable NNx run bundle manifest of this version."""
    if not os.path.lexists(target):
        return None
    if not os.path.isdir(target):
        raise BundleError(f"{target!r} is not a directory; export a bundle to an empty or missing directory")
    entries = [entry for entry in os.listdir(target) if entry != _LOCK and not _TEMPORARY.fullmatch(entry)]
    if _MANIFEST not in entries:
        if any(not _is_generation(target, entry) for entry in entries):
            raise BundleError(f"{target!r} is neither empty nor a run bundle; export to an empty directory")
        return None
    try:
        return _read_manifest(target)["generation"]
    except BundleError as exc:
        raise BundleError(
            f"{target!r} holds a bundle.json that is not a readable NNx run bundle manifest ({exc}); "
            "it is left untouched: export to an empty directory"
        ) from None


def _is_generation(root: str, entry: str) -> bool:
    path = os.path.join(root, entry)
    return (
        entry.startswith("g-")
        and _GENERATION.fullmatch(entry[2:]) is not None
        and os.path.isdir(path)
        and not os.path.islink(path)
    )


def _publish(root: str, manifest: str) -> None:
    """Replace ``bundle.json`` atomically: a temporary ``.bundle.json.*``
    file, created like the payloads (so it is as readable as they are),
    fsynced and renamed over it — then persist the rename."""
    temporary = os.path.join(root, f".{_MANIFEST}.{uuid.uuid4().hex}.tmp")
    try:
        _write(temporary, [manifest.encode("utf-8")])
        os.replace(temporary, os.path.join(root, _MANIFEST))
    except BaseException:
        if os.path.lexists(temporary):
            try:
                os.remove(temporary)
            except OSError:
                pass  # the next export clears it; the original error wins
        raise
    _fsync_directory(root)


# --- reconstruct -----------------------------------------------------------------------------


def _missing_device(device: Any) -> Optional[str]:
    """Why this host cannot build on ``device`` (a ``Devices`` member), if it cannot."""
    value = getattr(device, "value", device)
    if value == "cuda" and not torch.cuda.is_available():
        return "the bundle's model runs on CUDA, which this host lacks; pass device=Devices.CPU"
    if value == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return "the bundle's model runs on MPS, which this host lacks; pass device=Devices.CPU"
    return None


def reconstruct_bundle(
    path: Union[str, os.PathLike[str]],
    *,
    factories: Optional[Mapping[tuple[str, int], ModelFactory]] = None,
    components: Optional[Iterable[Any]] = None,
    device: Any = None,
    batch_adapter: Optional[BatchAdapter] = None,
) -> ReconstructedBundle:
    """Validate the bundle at ``path`` and rebuild its model.

    Args:
        factories: the model factories to rebuild a registered ``ModelSpec``
            from, ``{(id, version): factory}`` — used instead of the process
            registry. ``None`` uses the process registry
            (``nnx.models.register_model_factory``). A built-in net needs none.
        components: for a ``"resume"`` bundle, the stateful components the
            resumed run will register (e.g. the same callbacks): their saved
            state is checked against them. ``None`` skips the check (for
            inference use; :meth:`ReconstructedBundle.resume` checks again).
        device: a ``Devices`` member to build the model on instead of the
            saved one.
        batch_adapter: how a registered module sees a batch
            (``nnx.models.BatchAdapter``), as passed to ``NNModel`` — it is
            runtime-only, never stored.

    Everything missing — the model factory, a required or incompatible
    component — is reported in one :class:`BundleReconstructionError` before
    any model is allocated. Calibrators with a fingerprint ``model_id`` are
    checked against the rebuilt weights.
    """
    from .nn.nn_model import NNModel
    from .nn.params.nn_checkpoint import NNCheckpoint

    read = _read(path, full=True)
    parts = read.parts
    model_params = parts["model_params"]
    if device is not None:
        model_params = replace(model_params, device=device)
    problems: list[str] = []
    missing_device = _missing_device(model_params.device)
    if missing_device is not None:
        problems.append(missing_device)
    net = model_params.net
    if factories is not None:
        factories = dict(factories)
        for key, factory in factories.items():
            if not (isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], str) and type(key[1]) is int):
                raise TypeError(f"factories keys must be (id, version) pairs, got {key!r}")
            if not callable(factory):
                raise TypeError(f"the factory for {key[0]}@v{key[1]} must be callable, got {type(factory).__name__}")
    if isinstance(net, ModelSpec):
        if factories is not None:
            if (net.id, net.version) not in factories:
                problems.append(f"model factory {net} is not among the supplied factories")
        else:
            from .models import resolve_model_factory

            try:
                resolve_model_factory(net)
            except MissingModelFactoryError as exc:
                problems.append(str(exc))
    training_state = None
    if read.state["training_state"] is not None:
        # Read again, and hashed again: these very bytes are the validated ones.
        tensors = _load_tensors(
            _payload_bytes(read.directory, _TRAINING, read.manifest["payloads"][_TRAINING]), _TRAINING
        )
        training_state = _decode(read.state["training_state"], tensors)
        if components is not None and training_state.get("components") is not None:  # None: written before FEAT-005
            try:
                ComponentRegistry.discover(list(components)).plan(training_state.get("components"))
            except ComponentRestoreError as exc:
                problems.extend(exc.problems)
    if problems:
        raise BundleReconstructionError(problems)
    net_state = _load_tensors(_payload_bytes(read.directory, _MODEL, read.manifest["payloads"][_MODEL]), _MODEL)
    checkpoint = NNCheckpoint(
        net_params=parts["net_params"],
        net_state={key: net_state[key] for key in read.state["net_state"]},
        model_params=model_params,
        idp=parts["idp"],
        transforms=parts["transforms"],
        training_state_id=None if training_state is None else read.info.generation,  # recorded as the parent's
        training_state_present=training_state is not None,
    )
    from .models import _supplied_factories

    with _supplied_factories(factories):
        model = NNModel.from_checkpoint(checkpoint, batch_adapter=batch_adapter)
    return ReconstructedBundle(
        model=model,
        info=read.info,
        calibrators=read.calibrators,
        _checkpoint=checkpoint,
        _training_state=training_state,
    )
