"""Executable ONNX export conformance profiles (FEAT-038).

``onnx.checker`` proves an exported file is *well formed*; it says nothing
about what a runtime computes from it. A **profile** fixes everything a
parity claim depends on — the model and its weights, the exporter and its
options, the runtime, provider and dtype, the input cases and the
tolerances — runs it, and records each stage's outcome separately::

    dependencies ─► export ─► check ─► load ─► parity

- **dependencies** — ``onnx`` and ``onnxruntime`` (and ``onnxscript`` for
  the dynamo exporter) are importable. A missing one is a
  ``missing-dependency`` failure, never a skip.
- **export** — ``NNModel.to_onnx`` writes the graph (and any external data)
  into the profile's own directory; the model's parameters, gradients and
  (mixed) train / eval modes must be unchanged afterwards.
- **check** — ``onnx.checker.check_model(path, full_check=True)``; records
  the opset, IR version and input / output signature. Every file the
  exporter wrote is hashed (SHA-256), external tensors included.
- **load** — the hashes are verified first (a mismatch is rejected before
  anything is loaded), then an ``onnxruntime.InferenceSession`` is created
  on the declared provider.
- **parity** — every valid input case's raw logits must match the native
  model in eval mode within ``rtol`` / ``atol``, and an input of the wrong
  feature width must be refused by the runtime (``input-contract``).

Failures are classified (:data:`FAILURES`); a record's ``level`` is
``"executed"`` only when the runtime ran the graph and every case passed,
``"structural"`` when only the checker did, and ``"none"`` otherwise.

Tested profiles (:data:`PROFILES`) — the only targets NNx claims executed
parity for: ``FeedFwdNN`` 4-8-2 (ReLU, dropout 0, deterministic nontrivial
FP32 weights), exported at batch 2 with a dynamic batch dimension and run
on ONNX Runtime's CPU provider at batches 1, 2, 3 and 7, under the legacy
TorchScript exporter (``feedfwd-fp32-torchscript``) and the
``torch.export``-based one (``feedfwd-fp32-dynamo``). Everything else —
other architectures, quantized exports, Netron files, GGUF / Ollama
artifacts — stays structural or container-only: a checker pass or a
parser round trip never promotes a target to executed parity.

Run both profiles with ``python scripts/check_export_conformance.py
--output conformance.json`` (needs ``thekaveh-nnx[onnx-runtime,onnx-dynamo]``).
Nothing here imports ``onnx`` or ``onnxruntime`` until a profile runs.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
import platform
import posixpath
import re
import subprocess
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import torch

__all__ = [
    "EXIT_CODES",
    "FAILURES",
    "FORMAT",
    "LEVELS",
    "MODEL",
    "PROFILES",
    "STAGES",
    "ConformanceError",
    "Profile",
    "build_model",
    "execute",
    "exit_code",
    "run_profile",
    "run_profiles",
    "save_report",
    "summary_lines",
    "validate_record",
    "verify_artifacts",
]

FORMAT = "nnx.export-conformance/1"
STAGES = ("dependencies", "export", "check", "load", "parity")
STATUSES = ("passed", "failed", "not_run")
LEVELS = ("executed", "structural", "none")
# Each failure class and the CLI exit code that reports it: 0 is success,
# and 1 / 2 stay an error / a usage error (argparse), never a failure class.
EXIT_CODES = {
    "missing-dependency": 10,
    "export-error": 11,
    "state-changed": 12,
    "checker-error": 13,
    "hash-mismatch": 14,
    "load-error": 15,
    "mismatch": 16,
    "input-contract": 17,
    "runtime-error": 18,
}
FAILURES = tuple(EXIT_CODES)
_STAGE_FAILURES = {
    "dependencies": ("missing-dependency",),
    "export": ("export-error", "state-changed"),
    "check": ("checker-error",),
    "load": ("hash-mismatch", "load-error"),
    "parity": ("mismatch", "input-contract", "runtime-error"),
}
MODEL_FILE = "model.onnx"
_PLAIN = re.compile(r"[A-Za-z0-9._-]+")


class ConformanceError(ValueError):
    """A malformed conformance record, or artifacts that do not match it."""


# The model every tested profile exports: FeedFwdNN 4-8-2, deterministic
# nontrivial weights drawn from a local generator.
MODEL: Mapping[str, Any] = {
    "net": "feed_fwd",
    "input_dim": 4,
    "hidden_dims": [8],
    "output_dim": 2,
    "activation": "relu",
    "dropout": 0.0,
    "weights": {"generator": "torch.Generator", "seed": 0, "distribution": "normal", "std": 0.5},
}


@dataclass(frozen=True)
class Profile:
    """A conformance profile: one model, exporter configuration, runtime,
    provider, dtype, set of input cases and tolerances."""

    name: str
    exporter: str  # "torchscript" (legacy torch.onnx.export) or "dynamo" (torch.export-based)
    opset_version: int = 17
    export_batch: int = 2
    dynamic_batch: bool = True
    batches: tuple[int, ...] = (1, 2, 3, 7)
    rtol: float = 1e-4
    atol: float = 1e-5
    runtime: str = "onnxruntime"
    provider: str = "CPUExecutionProvider"
    dtype: str = "float32"
    model: Mapping[str, Any] = field(default_factory=lambda: copy.deepcopy(dict(MODEL)))

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _PLAIN.fullmatch(self.name) or self.name in (".", ".."):
            raise ConformanceError(f"a profile name is a plain directory name ([A-Za-z0-9._-]), got {self.name!r}")
        if self.exporter not in ("torchscript", "dynamo"):
            raise ConformanceError(f"exporter must be 'torchscript' or 'dynamo', got {self.exporter!r}")
        # The record states what ran, so a profile may only declare what this module runs.
        if self.runtime != "onnxruntime" or self.dtype != "float32":
            raise ConformanceError(
                f"profiles run FP32 inputs in ONNX Runtime; got runtime={self.runtime!r}, dtype={self.dtype!r}"
            )
        try:
            batches = tuple(self.batches)
        except TypeError:
            batches = ()
        if not 1 <= len(batches) <= MAX_CASES or not all(
            isinstance(b, int) and not isinstance(b, bool) and 1 <= b <= MAX_DIM for b in batches
        ):
            raise ConformanceError(
                f"batches must list 1..{MAX_CASES} batch sizes in 1..{MAX_DIM}, got {self.batches!r}"
            )
        object.__setattr__(self, "batches", batches)
        for name in ("rtol", "atol"):
            value = getattr(self, name)
            if not _finite(value) or value < 0:
                raise ConformanceError(f"{name} must be finite and non-negative, got {value!r}")
        for name in ("export_batch", "opset_version"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_DIM:
                raise ConformanceError(f"{name} must be an integer in 1..{MAX_DIM}, got {value!r}")
        if not isinstance(self.dynamic_batch, bool):
            raise ConformanceError(f"dynamic_batch must be a bool, got {self.dynamic_batch!r}")
        if not isinstance(self.provider, str) or not self.provider:
            raise ConformanceError(f"provider must name an ONNX Runtime execution provider, got {self.provider!r}")
        _check_model(self.model)
        object.__setattr__(self, "model", _canonical_model(self.model))  # a private copy, not the caller's
        _check_size(batches, self.export_batch, self.model)

    @property
    def requires(self) -> tuple[str, ...]:
        return ("onnx", "onnxruntime", "onnxscript") if self.exporter == "dynamo" else ("onnx", "onnxruntime")

    def options(self) -> dict[str, Any]:
        return {
            "exporter": self.exporter,
            "dynamo": self.exporter == "dynamo",
            "opset_version": self.opset_version,
            "dynamic_batch": self.dynamic_batch,
            "export_batch": self.export_batch,
            "input_names": ["features"],
            "output_names": ["logits"],
        }

    def cases(self) -> list[dict[str, Any]]:
        width = int(self.model["input_dim"])
        cases = [{"name": f"batch-{b}", "shape": [b, width], "seed": 1000 + b, "expect": "match"} for b in self.batches]
        cases.append({"name": f"width-{width + 1}", "shape": [3, width + 1], "seed": 2000, "expect": "rejected"})
        return cases


# A conformance profile is a small model: anything larger is refused before it is built or run.
MAX_DIM = 1 << 16  # units per layer, and the largest batch
MAX_LAYERS = 64  # hidden layers
MAX_ELEMENTS = 1 << 24  # weights in the model, and values one batch holds in any layer
MAX_CASES = 64  # valid input cases


def _finite(value: Any) -> bool:
    """A real, finite number (a huge integer that overflows a float is not)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _check_model(config: Any) -> None:
    """Refuse a model config :func:`build_model` cannot build exactly."""
    from .nn.enum.activations import Activations

    def positive(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= MAX_DIM

    problems = []
    if not isinstance(config, Mapping) or set(config) != set(MODEL):
        raise ConformanceError(f"a profile's model holds exactly {sorted(MODEL)}, got {config!r}")
    if config["net"] != "feed_fwd":
        problems.append(f"net {config['net']!r} (only feed_fwd)")
    if not all(positive(config[key]) for key in ("input_dim", "output_dim")):
        problems.append(f"input_dim / output_dim must be integers in 1..{MAX_DIM}")
    hidden = config["hidden_dims"]
    if not isinstance(hidden, (list, tuple)) or not all(positive(h) for h in hidden):
        problems.append(f"hidden_dims must be integers in 1..{MAX_DIM}")
    if not isinstance(config["activation"], str) or config["activation"] not in {a.value for a in Activations}:
        problems.append(f"unknown activation {config['activation']!r}")
    dropout = config["dropout"]
    if not _finite(dropout) or not 0 <= dropout < 1:
        problems.append(f"dropout {dropout!r}")
    weights = config["weights"]
    if not (
        isinstance(weights, Mapping)
        and set(weights) == set(MODEL["weights"])
        and isinstance(weights["seed"], int)
        and not isinstance(weights["seed"], bool)
        and 0 <= weights["seed"] < 2**63
        and _finite(weights["std"])
        and weights["std"] > 0
        and {k: weights[k] for k in ("generator", "distribution")}
        == {k: MODEL["weights"][k] for k in ("generator", "distribution")}
    ):
        problems.append(f"weights {weights!r}")
    if not problems:
        widths = [config["input_dim"], *hidden, config["output_dim"]]
        if len(hidden) > MAX_LAYERS or sum(a * b + b for a, b in zip(widths, widths[1:], strict=False)) > MAX_ELEMENTS:
            problems.append(f"the model exceeds {MAX_LAYERS} hidden layers or {MAX_ELEMENTS} weights")
    if problems:
        raise ConformanceError("malformed model config: " + "; ".join(problems))


def _check_size(batches: Sequence[int], export_batch: int, model: Mapping[str, Any]) -> None:
    """Refuse a profile whose largest batch, through its widest layer (input,
    hidden activations or logits), holds more than :data:`MAX_ELEMENTS` values."""
    batch = max((*batches, export_batch))
    widest = max(model["input_dim"], *model["hidden_dims"], model["output_dim"])
    if batch * widest > MAX_ELEMENTS:
        raise ConformanceError(
            f"a batch of {batch} through a layer of {widest} units holds more than {MAX_ELEMENTS} values"
        )


def _settings(
    *, tolerances: Any, cases: Any, dtype: Any, runtime: Any, provider: Any, options: Any, model: Any, directory: Any
) -> str:
    """The canonical text of everything a profile's name stands for."""
    try:
        return json.dumps(
            {
                "tolerances": tolerances,
                "input cases": cases,
                "dtype": dtype,
                "runtime": [runtime, provider],
                "exporter options": options,
                "model config": model,
                "artifact directory": directory,
            },
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ConformanceError(f"the profile settings are not JSON: {error}") from error


def _canonical_model(model: Mapping[str, Any]) -> dict[str, Any]:
    """A model config with its real-valued fields as floats (0 and 0.0 are one setting)."""
    canonical = copy.deepcopy(dict(model))
    canonical["dropout"] = float(canonical["dropout"])
    canonical["weights"] = {**canonical["weights"], "std": float(canonical["weights"]["std"])}
    canonical["hidden_dims"] = list(canonical["hidden_dims"])
    return canonical


def _profile_settings(profile: Profile) -> str:
    return _settings(
        tolerances={"rtol": float(profile.rtol), "atol": float(profile.atol)},
        cases=[{k: c[k] for k in ("name", "shape", "seed", "expect")} for c in profile.cases()],
        dtype=profile.dtype,
        runtime=profile.runtime,
        provider=profile.provider,
        options=profile.options(),
        model=_canonical_model(profile.model),
        directory=profile.name,
    )


def _record_profile(record: Mapping[str, Any]) -> Profile:
    """The :class:`Profile` a record claims to have run (its settings, rebuilt
    and checked as a ``Profile`` is), or a refusal."""
    try:
        options = record["exporter"]["options"]
        runtime = record["runtime"]
        return Profile(
            name=record["profile"],
            exporter=options["exporter"],
            opset_version=options["opset_version"],
            export_batch=options["export_batch"],
            dynamic_batch=options["dynamic_batch"],
            batches=tuple(c["shape"][0] for c in record["input_cases"] if c.get("expect") == "match"),
            rtol=record["tolerances"]["rtol"],
            atol=record["tolerances"]["atol"],
            runtime=runtime["name"],
            provider=runtime["provider"],
            dtype=record["dtype"],
            model={k: v for k, v in record["config"].items() if k != "weights_sha256"},
        )
    except ConformanceError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, OverflowError) as error:
        raise ConformanceError(f"the record's settings are not a profile: {type(error).__name__}: {error}") from error


def _record_settings(record: Mapping[str, Any]) -> str:
    """The canonical settings text a record states (refusals as text)."""
    try:
        config = record["config"]
        return _settings(
            tolerances={"rtol": float(record["tolerances"]["rtol"]), "atol": float(record["tolerances"]["atol"])},
            cases=[{k: c.get(k) for k in ("name", "shape", "seed", "expect")} for c in record["input_cases"]],
            dtype=record["dtype"],
            runtime=record["runtime"]["name"],
            provider=record["runtime"]["provider"],
            options=record["exporter"]["options"],
            model=_canonical_model({k: v for k, v in config.items() if k != "weights_sha256"}),
            directory=record["artifacts"]["directory"],
        )
    except ConformanceError as error:
        return str(error)
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as error:
        return f"{type(error).__name__}: {error}"


def _blank(case: Mapping[str, Any]) -> dict[str, Any]:
    """An input case before parity ran: its spec, and no outcome."""
    return {
        **{key: case[key] for key in ("name", "shape", "seed", "expect")},
        "outcome": None,
        "passed": False,
        "failure": None,
        "max_abs_error": None,
        "max_rel_error": None,
        "detail": None,
    }


PROFILES: Mapping[str, Profile] = {
    "feedfwd-fp32-torchscript": Profile("feedfwd-fp32-torchscript", "torchscript"),
    "feedfwd-fp32-dynamo": Profile("feedfwd-fp32-dynamo", "dynamo"),
}
# What each tested name stands for, fixed at import: a profile changed in place later is not it.
_TESTED = {name: _profile_settings(profile) for name, profile in PROFILES.items()}


# --- the model ----------------------------------------------------------------------------------------


def build_model(config: Mapping[str, Any] = MODEL) -> Any:
    """The profile's ``NNModel`` (CPU, FP32) with deterministic weights; the
    caller's global RNG streams are left untouched."""
    from .nn.enum.activations import Activations
    from .nn.enum.devices import Devices
    from .nn.enum.losses import Losses
    from .nn.enum.nets import Nets
    from .nn.nn_model import NNModel
    from .nn.params.nn_model_params import NNModelParams
    from .nn.params.nn_params import NNParams

    if config.get("net") != "feed_fwd":
        raise ConformanceError(f"conformance profiles export a feed_fwd net, got {config.get('net')!r}")
    weights = config["weights"]
    with torch.random.fork_rng(devices=[]):
        model = NNModel(
            net_params=NNParams(
                input_dim=int(config["input_dim"]),
                output_dim=int(config["output_dim"]),
                hidden_dims=[int(h) for h in config["hidden_dims"]],
                dropout_prob=float(config["dropout"]),
                activation=Activations(config["activation"]),
            ),
            params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        )
    generator = torch.Generator().manual_seed(int(weights["seed"]))
    with torch.no_grad():
        for parameter in model.net.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * float(weights["std"]))
    return model


def _weights_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(f"{name}|{array.dtype}|{list(array.shape)}|".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _inputs(case: Mapping[str, Any]) -> torch.Tensor:
    return torch.randn(*case["shape"], generator=torch.Generator().manual_seed(int(case["seed"])))


# --- provenance ---------------------------------------------------------------------------------------------


def _import_problem(name: str) -> Optional[str]:
    """Why ``name`` cannot be imported (``None`` when it can): a broken
    install is a missing dependency, with its cause."""
    if importlib.util.find_spec(name) is None:
        return f"{name}: not installed"
    try:
        importlib.import_module(name)
    except Exception as error:  # any failure to import is the dependency's, not the export's
        return f"{name}: {type(error).__name__}: {(str(error).splitlines() or [''])[0][:200]}"
    return None


def _version(distribution: str) -> Optional[str]:
    from importlib import metadata

    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _versions() -> dict[str, Any]:
    from . import __version__

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "nnx": __version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "onnx": _version("onnx"),
        "onnxruntime": _version("onnxruntime"),
        "onnxscript": _version("onnxscript"),
    }


def _git_revision() -> tuple[Optional[str], Optional[bool], str]:
    """``(HEAD, dirty, "git")`` when this very file is tracked in a git
    checkout (an editable install); an installed copy that merely sits
    inside some other repository is not that repository's revision."""
    here = os.path.dirname(os.path.abspath(__file__))

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", here, *args], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()

    try:
        top = git("rev-parse", "--show-toplevel")
        mine = os.path.join(top, "src", "nnx", os.path.basename(__file__))
        if not (os.path.isfile(mine) and os.path.samefile(mine, __file__)):
            return None, None, "unknown"
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no")), "git"
    except (OSError, subprocess.SubprocessError):
        return None, None, "unknown"


def _source(revision: Optional[str]) -> dict[str, Any]:
    """The source revision: given, or from the git checkout this very
    package is imported from (``None`` when there is none — an installed
    copy, whatever repository or CI it runs in)."""
    from . import __version__

    dirty: Optional[bool] = None
    origin = "given"
    if revision is None:
        revision, dirty, origin = _git_revision()
    return {"revision": revision or None, "origin": origin, "dirty": dirty, "nnx_version": __version__}


# --- artifacts -------------------------------------------------------------------------------------------------


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _external_locations(model: Any) -> set[str]:
    """Files the model's tensors live in (``external_data`` locations)."""
    import onnx

    locations: set[str] = set()

    def tensors(graph: Any) -> Iterable[Any]:
        yield from graph.initializer
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("t"):
                    yield attribute.t
                yield from attribute.tensors
                if attribute.HasField("g"):
                    yield from tensors(attribute.g)
                for subgraph in attribute.graphs:
                    yield from tensors(subgraph)

    for tensor in tensors(model.graph):
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            for entry in tensor.external_data:
                if entry.key == "location":
                    locations.add(entry.value)
    return locations


def _files(directory: str) -> list[str]:
    """Every file under ``directory``, as sorted ``/``-separated relative paths."""
    found = []

    def unreadable(error: OSError) -> None:
        raise error

    for folder, folders, names in os.walk(directory, onerror=unreadable):
        linked = [name for name in folders if os.path.islink(os.path.join(folder, name))]
        for name in [*names, *linked]:  # a symlinked directory is an entry, never followed
            found.append(os.path.relpath(os.path.join(folder, name), directory).replace(os.sep, "/"))
    return sorted(found)


def _relative(path: Any) -> str:
    """A safe ``/``-separated path inside an artifact directory, or a refusal."""
    if not isinstance(path, str) or not path or "\\" in path or path.startswith("/"):
        raise ConformanceError(f"artifact paths are relative '/'-separated paths, got {path!r}")
    normal = posixpath.normpath(path)
    if normal in (".", "..") or normal.startswith("../") or os.path.isabs(normal) or ":" in normal:
        raise ConformanceError(f"artifact path {path!r} leaves its directory")
    return normal


def _manifest(directory: str) -> list[dict[str, Any]]:
    import onnx

    model = onnx.load(os.path.join(directory, MODEL_FILE), load_external_data=False)
    external = {_relative(location) for location in _external_locations(model)}
    files = []
    for name in _files(directory):
        path = os.path.join(directory, *name.split("/"))
        if os.path.islink(path) or not os.path.isfile(path):
            raise ConformanceError(f"the exporter wrote {name!r}, which is not a regular file")
        role = "model" if name == MODEL_FILE else "external-data" if name in external else "sidecar"
        files.append({"path": name, "role": role, "bytes": os.path.getsize(path), "sha256": _sha256(path)})
    missing = sorted(external - {entry["path"] for entry in files})
    if missing:
        raise ConformanceError(f"external data {missing} is referenced but was not written")
    return files


def _artifact_dir(record: Mapping[str, Any], directory: Union[str, os.PathLike[str]]) -> str:
    name = record["artifacts"]["directory"]
    if not isinstance(name, str) or not _PLAIN.fullmatch(name) or name in (".", ".."):
        raise ConformanceError(f"artifact directory must be a plain name, got {name!r}")
    return os.path.join(os.fspath(directory), name)


def verify_artifacts(record: Mapping[str, Any], directory: Union[str, os.PathLike[str]]) -> None:
    """Refuse artifacts that differ from the record: a file missing, a
    SHA-256 or size mismatch (external tensors included), or a file the
    record does not list. ``directory`` is the root the record's
    ``artifacts.directory`` lives under. An unreadable file or folder is a
    refusal too."""
    try:
        _verify_artifacts(record, directory)
    except OSError as error:
        raise ConformanceError(f"artifacts cannot be read: {type(error).__name__}: {error}") from error
    except (KeyError, TypeError, AttributeError) as error:
        raise ConformanceError(f"the record or directory is malformed: {type(error).__name__}: {error}") from error


def _verify_artifacts(record: Mapping[str, Any], directory: Union[str, os.PathLike[str]]) -> None:
    root = _artifact_dir(record, directory)
    if os.path.islink(root):
        raise ConformanceError(f"the artifact directory {root!r} is a symlink, not the profile's own folder")
    listed = record["artifacts"]["files"]
    names = {_relative(entry["path"]) for entry in listed}
    problems = []
    for entry in listed:
        name = _relative(entry["path"])
        path = os.path.join(root, *name.split("/"))
        if os.path.islink(path):
            problems.append(f"{name}: a symlink, not a written file")
        elif not os.path.isfile(path):
            problems.append(f"{name}: missing")
        elif os.path.getsize(path) != entry["bytes"] or _sha256(path) != entry["sha256"]:
            problems.append(f"{name}: hash mismatch (recorded sha256 {entry['sha256'][:12]}…)")
    if os.path.isdir(root):
        extra = sorted(set(_files(root)) - names)
        problems += [f"{name}: not in the record" for name in extra]
    if problems:
        raise ConformanceError("artifacts do not match the record: " + "; ".join(problems))


# --- stages ----------------------------------------------------------------------------------------------------


def _stage(status: str, failure: Optional[str] = None, detail: Any = None) -> dict[str, Any]:
    return {"status": status, "failure": failure, "detail": detail}


def _not_run() -> dict[str, Any]:
    return _stage("not_run")


def _model_state(module: torch.nn.Module) -> tuple[dict, dict, list]:
    return (
        {k: v.detach().clone() for k, v in module.state_dict().items()},
        {k: (None if p.grad is None else p.grad.detach().clone()) for k, p in module.named_parameters()},
        [(name, part.training) for name, part in module.named_modules()],
    )


def _same(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    if before.keys() != after.keys():
        return False
    for key, value in before.items():
        other = after[key]
        if (value is None) != (other is None):
            return False
        if value is not None and not torch.equal(value, other):
            return False
    return True


def _export(profile: Profile, model: Any, root: str) -> tuple[dict[str, Any], Optional[dict]]:
    """Export into ``root`` with the model in a mixed train / eval mode and
    holding gradients, then check neither changed."""
    net = model.net
    net.train()
    first = next(module for module in net.modules() if module is not net)
    first.eval()  # a mixed mode the export must restore exactly
    x = _inputs({"shape": [profile.export_batch, int(profile.model["input_dim"])], "seed": 7})
    torch.nn.functional.cross_entropy(net(x), torch.zeros(profile.export_batch, dtype=torch.long)).backward()
    params, grads, modes = _model_state(net)
    options = profile.options()
    caught: list[str] = []
    printed = io.StringIO()
    try:
        with warnings.catch_warnings(record=True) as seen, contextlib.redirect_stdout(printed):
            warnings.simplefilter("always")
            model.to_onnx(
                os.path.join(root, MODEL_FILE),
                example_input=x,
                input_names=options["input_names"],
                output_names=options["output_names"],
                dynamic_batch=profile.dynamic_batch,
                opset_version=profile.opset_version,
                dynamo=options["dynamo"],
            )
        caught = sorted({f"{w.category.__name__}: {(str(w.message).splitlines() or [''])[0][:200]}" for w in seen})
    except Exception as error:  # every exporter failure is a classified outcome
        tail = printed.getvalue().strip().splitlines()[-5:]
        return _stage("failed", "export-error", {"error": f"{type(error).__name__}: {error}", "log": tail}), None
    after_params, after_grads, after_modes = _model_state(net)
    unchanged = {
        "parameters": _same(params, after_params),
        "gradients": _same(grads, after_grads),
        "modes": modes == after_modes,
    }
    if not all(unchanged.values()):
        changed = sorted(name for name, same in unchanged.items() if not same)
        return _stage("failed", "state-changed", {"changed": changed, "warnings": caught}), unchanged
    if not os.path.isfile(os.path.join(root, MODEL_FILE)):
        return _stage("failed", "export-error", {"error": f"the exporter wrote no {MODEL_FILE}"}), unchanged
    return _stage("passed", detail={"warnings": caught}), unchanged


def _signature(values: Iterable[Any]) -> list[dict[str, Any]]:
    import onnx

    out = []
    for value in values:
        tensor = value.type.tensor_type
        out.append(
            {
                "name": value.name,
                "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type).lower(),
                "shape": [d.dim_param if d.dim_param else d.dim_value for d in tensor.shape.dim],
            }
        )
    return out


def _check(root: str) -> tuple[dict[str, Any], Optional[dict], Optional[list]]:
    import onnx

    path = os.path.join(root, MODEL_FILE)
    try:
        onnx.checker.check_model(path, full_check=True)
        model = onnx.load(path, load_external_data=False)
        files = _manifest(root)
    except Exception as error:
        return _stage("failed", "checker-error", f"{type(error).__name__}: {error}"), None, None
    info = {
        "opset": {(entry.domain or "ai.onnx"): int(entry.version) for entry in model.opset_import},
        "ir_version": int(model.ir_version),
        "producer": f"{model.producer_name} {model.producer_version}".strip(),
        "inputs": _signature(model.graph.input),
        "outputs": _signature(model.graph.output),
    }
    return _stage("passed", detail=info), info, files


def _session_options() -> dict[str, Any]:
    return {"intra_op_num_threads": 1, "inter_op_num_threads": 1, "graph_optimization_level": "ORT_ENABLE_ALL"}


def _load(record: Mapping[str, Any], directory: str, provider: str) -> tuple[dict[str, Any], Any]:
    try:
        verify_artifacts(record, directory)
    except ConformanceError as error:  # changed, missing, unlisted or unreadable: unverified
        return _stage("failed", "hash-mismatch", f"{type(error).__name__}: {error}"), None
    try:
        import onnxruntime as ort

        options = ort.SessionOptions()
        settings = _session_options()
        options.intra_op_num_threads = settings["intra_op_num_threads"]
        options.inter_op_num_threads = settings["inter_op_num_threads"]
        options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, settings["graph_optimization_level"])
        path = os.path.join(_artifact_dir(record, directory), MODEL_FILE)
        session = ort.InferenceSession(path, sess_options=options, providers=[provider])
        used = session.get_providers()
    except Exception as error:
        return _stage("failed", "load-error", f"{type(error).__name__}: {error}"), None
    if not used or used[0] != provider:
        return _stage("failed", "load-error", f"the session runs on {used}, not {provider}"), None
    return _stage("passed", detail={"providers": list(used)}), session


def _first_line(error: BaseException) -> str:
    return f"{type(error).__name__}: {(str(error).splitlines() or [''])[0][:300]}"


def _parity(profile_cases: Sequence[Mapping[str, Any]], session: Any, model: Any, tolerances: Mapping[str, float]):
    """Run every case: a valid batch must give the native raw logits within
    tolerance; an invalid one must be refused by the runtime as an invalid
    argument. Any other runtime failure is a ``runtime-error``."""
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

    rtol, atol = float(tolerances["rtol"]), float(tolerances["atol"])
    feed = session.get_inputs()[0].name
    results = []
    for case in profile_cases:
        x = _inputs(case)
        result = {key: case[key] for key in ("name", "shape", "seed", "expect")}
        result.update(outcome=None, passed=False, failure=None, max_abs_error=None, max_rel_error=None, detail=None)
        results.append(result)
        try:
            outputs = session.run(None, {feed: x.numpy()})
        except InvalidArgument as error:  # the runtime refused the input
            refused = case["expect"] == "rejected"
            result.update(outcome="rejected", passed=refused, detail=_first_line(error))
            result["failure"] = None if refused else "input-contract"
            continue
        except Exception as error:  # the runtime itself failed
            result.update(outcome="error", failure="runtime-error", detail=_first_line(error))
            continue
        if case["expect"] == "rejected":
            result.update(outcome="accepted", failure="input-contract", detail=f"ran and gave {len(outputs)} output(s)")
            continue
        reference = np.asarray(model.predict(x).logits, dtype=np.float64)
        out = np.asarray(outputs[0]) if len(outputs) == 1 else None
        if out is None or out.dtype != np.float32 or out.shape != reference.shape:
            got = f"{len(outputs)} outputs" if out is None else f"{out.dtype} {list(out.shape)}"
            result.update(
                outcome="mismatch", failure="mismatch", detail=f"{got} for native float32 {list(reference.shape)}"
            )
            continue
        if not np.all(np.isfinite(out)) or not np.all(np.isfinite(reference)):
            which = "the runtime" if not np.all(np.isfinite(out)) else "the native model"
            result.update(outcome="mismatch", failure="mismatch", detail=f"{which} gave non-finite logits")
            continue
        error = np.abs(out.astype(np.float64) - reference)
        result["max_abs_error"] = float(error.max()) if error.size else 0.0
        result["max_rel_error"] = float((error / np.maximum(np.abs(reference), 1e-30)).max()) if error.size else 0.0
        within = bool(np.all(error <= atol + rtol * np.abs(reference)))
        result.update(outcome="match" if within else "mismatch", passed=within, failure=None if within else "mismatch")
    failed = [r for r in results if not r["passed"]]
    compared = sum(1 for r in results if r["outcome"] == "match")
    if not failed:  # a profile always has a valid case, so one was compared
        return _stage("passed", detail={"cases": len(results), "compared": compared}), results
    kinds = {r["failure"] for r in failed}
    failure = next(kind for kind in ("runtime-error", "mismatch", "input-contract") if kind in kinds)
    return _stage("failed", failure, {"failed_cases": [r["name"] for r in failed]}), results


def _finish(record: dict[str, Any]) -> dict[str, Any]:
    stages = record["stages"]
    failure = next((stages[s]["failure"] for s in STAGES if stages[s]["status"] == "failed"), None)
    record["status"] = "passed" if all(stages[s]["status"] == "passed" for s in STAGES) else "failed"
    record["failure"] = failure
    if stages["parity"]["status"] == "passed":
        record["level"] = "executed"
    elif stages["check"]["status"] == "passed":
        record["level"] = "structural"
    else:
        record["level"] = "none"
    return record


# --- profiles ----------------------------------------------------------------------------------------------------


def run_profile(
    profile: Union[str, Profile],
    directory: Union[str, os.PathLike[str]],
    *,
    source_revision: Optional[str] = None,
) -> dict[str, Any]:
    """Run one profile, writing its artifacts into ``directory/<name>/``
    (which must not exist or be empty), and return its record (see the
    module docstring). Never raises for a failing stage: every outcome is
    classified in the record."""
    if isinstance(profile, str):
        if profile not in PROFILES:
            raise ConformanceError(f"unknown profile {profile!r}; tested profiles are {sorted(PROFILES)}")
        profile = PROFILES[profile]
    _check_model(profile.model)  # again: the model mapping may have changed since the profile was made
    _check_size(profile.batches, profile.export_batch, profile.model)
    if profile.name in _TESTED and _profile_settings(profile) != _TESTED[profile.name]:
        raise ConformanceError(
            f"{profile.name!r} names a tested profile; a profile with other settings needs its own name"
        )
    root = os.path.join(os.fspath(directory), profile.name)
    if os.path.exists(root) and (not os.path.isdir(root) or os.listdir(root)):
        raise ConformanceError(f"{root!r} must be a new or empty directory: a profile owns its artifacts")
    os.makedirs(root, exist_ok=True)
    model = build_model(profile.model)
    config = {**copy.deepcopy(dict(profile.model)), "weights_sha256": _weights_digest(model.net)}
    record: dict[str, Any] = {
        "format": FORMAT,
        "profile": profile.name,
        "status": "failed",
        "failure": None,
        "level": "none",
        "source": _source(source_revision),
        "config": config,
        "exporter": {"name": profile.exporter, "options": profile.options()},
        "opset": {"requested": profile.opset_version, "model": None},
        "artifacts": {"directory": profile.name, "model": MODEL_FILE, "files": []},
        "versions": _versions(),
        "runtime": {
            "name": profile.runtime,
            "version": _version("onnxruntime"),
            "provider": profile.provider,
            "session": _session_options(),
        },
        "dtype": profile.dtype,
        "input_cases": [_blank(case) for case in profile.cases()],
        "tolerances": {"rtol": profile.rtol, "atol": profile.atol},
        "model_state": None,
        "stages": {stage: _not_run() for stage in STAGES},
    }
    missing = [problem for problem in map(_import_problem, profile.requires) if problem is not None]
    if missing:
        record["stages"]["dependencies"] = _stage(
            "failed",
            "missing-dependency",
            f"not importable: {'; '.join(missing)} — install thekaveh-nnx[onnx-runtime"
            + (",onnx-dynamo]" if profile.exporter == "dynamo" else "]"),
        )
        return _finish(record)
    record["stages"]["dependencies"] = _stage("passed", detail={name: _version(name) for name in profile.requires})
    record["stages"]["export"], record["model_state"] = _export(profile, model, root)
    if record["stages"]["export"]["status"] != "passed":
        return _finish(record)
    record["stages"]["check"], info, files = _check(root)
    if info is None or files is None:
        return _finish(record)
    record["opset"]["model"] = info["opset"]
    record["artifacts"]["files"] = files
    return _execute(record, os.fspath(directory), model)


def _execute(record: dict[str, Any], directory: str, model: Any) -> dict[str, Any]:
    record["input_cases"] = [_blank(case) for case in record["input_cases"]]  # no outcome survives a re-run
    record["stages"]["load"], session = _load(record, directory, record["runtime"]["provider"])
    if session is None:
        record["stages"]["parity"] = _not_run()
        return _finish(record)
    cases = [{key: case[key] for key in ("name", "shape", "seed", "expect")} for case in record["input_cases"]]
    record["stages"]["parity"], record["input_cases"] = _parity(cases, session, model, record["tolerances"])
    return _finish(record)


def execute(record: Mapping[str, Any], directory: Union[str, os.PathLike[str]]) -> dict[str, Any]:
    """Re-run the load and parity stages of a saved record against its
    artifacts under ``directory``: the hashes are verified before anything
    is loaded (a mismatch is a ``hash-mismatch`` failure), and the native
    reference is rebuilt from the recorded config, whose weights must hash
    to the recorded digest. Returns the new record."""
    validate_record(record)
    replay: dict[str, Any] = json.loads(json.dumps(record))
    if replay["stages"]["check"]["status"] != "passed":
        raise ConformanceError("only a record whose export and check passed can be executed again")
    problem = _import_problem("onnxruntime")
    if problem is not None:
        raise ConformanceError(
            f"execute() runs the recorded artifacts in ONNX Runtime, which cannot be imported ({problem}): "
            "install thekaveh-nnx[onnx-runtime]"
        )
    config = {key: value for key, value in replay["config"].items() if key != "weights_sha256"}
    try:
        model = build_model(config)
    except ConformanceError:
        raise
    except (KeyError, TypeError, ValueError, RuntimeError, OverflowError, MemoryError) as error:
        raise ConformanceError(
            f"the recorded config cannot rebuild the model: {type(error).__name__}: {error}"
        ) from error
    if _weights_digest(model.net) != replay["config"]["weights_sha256"]:
        raise ConformanceError("the recorded config no longer rebuilds the recorded weights")
    replay["versions"] = _versions()
    replay["runtime"]["version"] = _version("onnxruntime")
    return _execute(replay, os.fspath(directory), model)


def run_profiles(
    directory: Union[str, os.PathLike[str]],
    names: Optional[Sequence[str]] = None,
    *,
    source_revision: Optional[str] = None,
) -> dict[str, Any]:
    """Run ``names`` (every tested profile by default) under ``directory``;
    the report's ``status`` is ``passed`` only when every profile passed."""
    chosen = list(PROFILES) if names is None else list(dict.fromkeys(names))  # each profile once
    records = [run_profile(name, directory, source_revision=source_revision) for name in chosen]
    return {
        "format": FORMAT,
        "status": "passed" if records and all(r["status"] == "passed" for r in records) else "failed",
        "profiles": records,
    }


def exit_code(report: Mapping[str, Any]) -> int:
    """0 when every profile passed, else the exit code of the first failing
    profile's failure class (:data:`EXIT_CODES`)."""
    for record in report["profiles"]:
        if record["status"] != "passed":
            return EXIT_CODES.get(record["failure"], 1)
    return 0 if report["profiles"] else 1


def save_report(report: Mapping[str, Any], path: Union[str, os.PathLike[str]]) -> None:
    """Write the report as strict, sorted JSON (atomically)."""
    from ._artifacts import atomic_write

    for record in report["profiles"]:
        validate_record(record)
    atomic_write(path, json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


# --- the record schema -----------------------------------------------------------------------------------------

_CASE_KEYS = {
    "name",
    "shape",
    "seed",
    "expect",
    "outcome",
    "passed",
    "failure",
    "max_abs_error",
    "max_rel_error",
    "detail",
}
# Each outcome a run of an input case gives, by what the case expects, and the
# failure it names (``None``: the case passed).
_OUTCOMES: Mapping[str, Mapping[str, Optional[str]]] = {
    "match": {"match": None},
    "mismatch": {"match": "mismatch"},
    "rejected": {"match": "input-contract", "rejected": None},
    "accepted": {"rejected": "input-contract"},
    "error": {"match": "runtime-error", "rejected": "runtime-error"},
}
_KEYS = {
    "format",
    "profile",
    "status",
    "failure",
    "level",
    "source",
    "config",
    "exporter",
    "opset",
    "artifacts",
    "versions",
    "runtime",
    "dtype",
    "input_cases",
    "tolerances",
    "model_state",
    "stages",
}


def validate_record(record: Any) -> None:
    """Refuse a record that does not follow ``nnx.export-conformance/1``:
    missing or unknown keys (in the record and in its ``runtime``,
    ``exporter`` and ``opset``), anything but strict JSON, unknown statuses
    or failure classes, a failure in the wrong stage (an input case's
    included), an input case outcome no run gives, a stage that ran after a
    failed one, inconsistent ``status`` / ``failure`` / ``level``, a passed
    stage its own evidence contradicts (parity without a compared valid
    case or with a failed case, an export that changed the model, a check
    without an opset or a listed model file), malformed hashes or paths,
    non-finite tolerances, or settings no profile produces (a model or
    batch larger than a profile allows included)."""
    if not isinstance(record, Mapping):
        raise ConformanceError(f"a conformance record is a mapping, got {type(record).__name__}")
    problems: list[str] = []
    try:
        if json.loads(json.dumps(record, sort_keys=True, allow_nan=False)) != record:
            problems.append("a conformance record is strict JSON: it does not survive a round trip unchanged")
    except (TypeError, ValueError, RecursionError) as error:
        problems.append(f"a conformance record is strict JSON: {type(error).__name__}: {error}")
    unknown, missing = sorted(set(record) - _KEYS), sorted(_KEYS - set(record))
    if unknown:
        problems.append(f"unknown keys {unknown}")
    if missing:
        raise ConformanceError(f"the record lacks {missing}" + (f"; and has unknown keys {unknown}" if unknown else ""))
    if record["format"] != FORMAT:
        problems.append(f"format {record['format']!r} is not {FORMAT!r}")
    stages = record["stages"]
    if not isinstance(stages, Mapping) or set(stages) != set(STAGES):
        raise ConformanceError(f"stages must be exactly {list(STAGES)}, got {stages!r}")

    def status(name: str) -> Any:
        stage = stages[name]
        return stage.get("status") if isinstance(stage, Mapping) else None

    halted = False
    for name in STAGES:
        stage = stages[name]
        if not isinstance(stage, Mapping) or set(stage) != {"status", "failure", "detail"}:
            problems.append(f"stage {name!r} must hold status, failure and detail")
            halted = True
            continue
        current, failure = stage["status"], stage["failure"]
        if current not in STATUSES:
            problems.append(f"stage {name!r} has status {current!r}, not one of {STATUSES}")
        if current == "failed" and failure not in _STAGE_FAILURES[name]:
            problems.append(f"stage {name!r} failed with {failure!r}, not one of {_STAGE_FAILURES[name]}")
        if current != "failed" and failure is not None:
            problems.append(f"stage {name!r} is {current} but names failure {failure!r}")
        if halted and current != "not_run":
            problems.append(f"stage {name!r} ran after an earlier stage did not pass")
        halted = halted or current != "passed"
    expected_status = "passed" if all(status(s) == "passed" for s in STAGES) else "failed"
    if record["status"] != expected_status:
        problems.append(f"status {record['status']!r} contradicts the stages ({expected_status})")
    first = next(
        (stages[s].get("failure") for s in STAGES if status(s) == "failed" and isinstance(stages[s], Mapping)), None
    )
    if record["failure"] != first:
        problems.append(f"failure {record['failure']!r} is not the first failing stage's ({first!r})")
    level = "executed" if status("parity") == "passed" else "structural" if status("check") == "passed" else "none"
    if record["level"] != level:
        problems.append(f"level {record['level']!r} contradicts the stages ({level!r})")

    cases = record["input_cases"]
    if not isinstance(cases, list) or not cases or not all(isinstance(case, Mapping) for case in cases):
        problems.append("input_cases must be a non-empty list of cases")
        cases = []
    elif not any(case.get("expect") == "match" for case in cases):
        problems.append("input_cases need at least one valid case to compare")
    if any(case.get("expect") not in ("match", "rejected") for case in cases):
        problems.append("every input case expects 'match' or 'rejected'")
    for case in cases:
        if set(case) != _CASE_KEYS:
            problems.append(f"input case {case.get('name')!r} must hold exactly {sorted(_CASE_KEYS)}")
        elif not (
            isinstance(case["name"], str)
            and isinstance(case["seed"], int)
            and not isinstance(case["seed"], bool)
            and 0 <= case["seed"] < 2**63
            and isinstance(case["shape"], list)
            and len(case["shape"]) == 2
            and all(isinstance(d, int) and not isinstance(d, bool) and 1 <= d <= MAX_DIM + 1 for d in case["shape"])
            and case["shape"][0] * case["shape"][1] <= MAX_ELEMENTS
        ):
            problems.append(f"input case {case.get('name')!r} has a malformed name, seed or shape")
        elif not (case["detail"] is None or isinstance(case["detail"], str)) or not all(
            case[key] is None or (_finite(case[key]) and case[key] >= 0) for key in ("max_abs_error", "max_rel_error")
        ):
            problems.append(f"input case {case['name']!r} has a malformed detail or error")
    if status("parity") not in ("passed", "failed"):
        if any(set(case) == _CASE_KEYS and dict(case) != _blank(case) for case in cases):
            problems.append("parity did not run, so no input case may carry an outcome")
    elif status("parity") == "failed":
        # The stage names the first class its failing cases name, in the order _parity ranks them.
        named = [case.get("failure") for case in cases if case.get("passed") is False]
        ranked = next((kind for kind in ("runtime-error", "mismatch", "input-contract") if kind in named), None)
        if ranked is None or stages["parity"].get("failure") != ranked:
            problems.append(
                f"parity failed with {stages['parity'].get('failure')!r}; its failing cases name {ranked!r}"
            )
    for case in cases:
        if type(case.get("passed")) is not bool:
            problems.append(f"input case {case.get('name')!r} must say passed as a bool")
        elif status("parity") in ("passed", "failed"):
            # Parity ran every case: its outcome, the failure it names and whether it passed agree.
            outcome, failure = case.get("outcome"), case.get("failure")
            given = _OUTCOMES.get(outcome) if isinstance(outcome, str) else None
            expect = case.get("expect")
            if given is None or not isinstance(expect, str) or expect not in given:
                problems.append(f"input case {case.get('name')!r} has an outcome {outcome!r} no run of it gives")
            elif failure != given[expect] or case["passed"] is not (given[expect] is None):
                problems.append(
                    f"input case {case.get('name')!r} {outcome!r} names failure {failure!r} and "
                    f"passed={case['passed']}, not {given[expect]!r} and passed={given[expect] is None}"
                )
            else:
                # A compared case carries its errors and no detail; any other outcome, a detail and no errors.
                compared = all(_finite(case.get(key)) for key in ("max_abs_error", "max_rel_error"))
                uncompared = case.get("max_abs_error") is None and case.get("max_rel_error") is None
                described = isinstance(case.get("detail"), str)
                if not (
                    (compared and case.get("detail") is None and outcome in ("match", "mismatch"))
                    or (uncompared and described and outcome != "match")
                ):
                    problems.append(f"input case {case.get('name')!r} {outcome!r} carries the wrong errors or detail")
    if status("parity") == "passed" and not all(
        case.get("passed") is True and case.get("outcome") == ("match" if case.get("expect") == "match" else "rejected")
        for case in cases
    ):
        problems.append("parity passed but an input case did not")
    state = record["model_state"]
    kept = isinstance(state, Mapping) and set(state) == {"parameters", "gradients", "modes"}
    if kept and not all(type(value) is bool for value in state.values()):
        problems.append(f"model_state holds bools, got {state!r}")
    elif status("export") == "passed" and not (kept and all(state.values())):
        problems.append(f"the export passed but the model state was not kept: {state!r}")
    elif status("export") == "failed" and stages["export"].get("failure") == "state-changed":
        if not kept or all(state.values()):
            problems.append(f"the export changed the model state, but model_state says otherwise: {state!r}")
    elif status("export") == "failed" and kept and not all(state.values()):
        problems.append(f"the model state changed, so the export failed as state-changed: {state!r}")
    elif status("export") not in ("passed", "failed") and state is not None:
        problems.append("the export did not run, so there is no model_state")
    elif state is not None and not kept:
        problems.append(f"model_state is None or holds parameters, gradients and modes, got {state!r}")

    artifacts = record["artifacts"]
    files = artifacts.get("files") if isinstance(artifacts, Mapping) else None
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts) != {"directory", "model", "files"}
        or not isinstance(files, list)
    ):
        problems.append("artifacts must hold directory, model and a list of files")
        files = []
    normalised = []
    for entry in files:
        try:
            normalised.append(_relative(entry.get("path")) if isinstance(entry, Mapping) else None)
        except ConformanceError:
            normalised.append(None)
    named = [path for path in normalised if path is not None]
    if len(named) != len(set(named)):
        problems.append("artifacts list a file more than once")
    for entry in files:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "role", "bytes", "sha256"}:
            problems.append(f"artifact entry {entry!r} must hold path, role, bytes and sha256")
            continue
        try:
            _relative(entry["path"])
        except ConformanceError as error:
            problems.append(str(error))
        if not (
            isinstance(entry["sha256"], str)
            and len(entry["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in entry["sha256"])
        ):
            problems.append(f"artifact {entry['path']!r} has a malformed sha256")
        if entry["role"] not in ("model", "external-data", "sidecar"):
            problems.append(f"artifact {entry['path']!r} has an unknown role {entry['role']!r}")
        if isinstance(entry["bytes"], bool) or not isinstance(entry["bytes"], int) or entry["bytes"] < 0:
            problems.append(f"artifact {entry['path']!r} has a malformed size {entry['bytes']!r}")
    if status("check") == "passed":
        models = [e for e in files if isinstance(e, Mapping) and e.get("role") == "model"]
        if len(models) != 1 or models[0].get("path") != artifacts.get("model"):
            problems.append("a checked record lists its model file, once, as artifacts.model")
        opset = record["opset"].get("model") if isinstance(record["opset"], Mapping) else None
        if not isinstance(opset, Mapping) or not opset:
            problems.append("a checked record carries the model's opset")
    tolerances = record["tolerances"]
    if not isinstance(tolerances, Mapping) or set(tolerances) != {"rtol", "atol"}:
        problems.append("tolerances must hold rtol and atol")
    elif not all(_finite(v) and v >= 0 for v in tolerances.values()):
        problems.append(f"tolerances must be finite and non-negative, got {dict(tolerances)}")
    if not isinstance(record["config"], Mapping) or "weights_sha256" not in record["config"]:
        problems.append("config must carry weights_sha256")
    for key in ("source", "exporter", "opset", "versions", "runtime"):
        if not isinstance(record[key], Mapping):
            problems.append(f"{key} must be a mapping")
    for key, keys in (
        ("runtime", {"name", "version", "provider", "session"}),
        ("exporter", {"name", "options"}),
        ("opset", {"requested", "model"}),
    ):
        if isinstance(record[key], Mapping) and set(record[key]) != keys:
            problems.append(f"{key} must hold exactly {sorted(keys)}, got {sorted(record[key])}")
    # Every record states a profile this module runs, exactly as run_profile writes it:
    # its settings are rebuilt as a Profile (checked like one) and compared as canonical text,
    # against the tested profile of that name or against the rebuilt profile itself.
    runtime, exporter, opset = record["runtime"], record["exporter"], record["opset"]
    try:
        claimed = _record_profile(record)
    except ConformanceError as error:
        problems.append(str(error))
    else:
        if _record_settings(record) != _TESTED.get(claimed.name, _profile_settings(claimed)):
            what = (
                "names a tested profile, but its settings differ from it"
                if claimed.name in _TESTED
                else "states settings no profile produces"
            )
            problems.append(f"{record['profile']!r} {what}")
    if not isinstance(runtime, Mapping) or runtime.get("session") != _session_options():
        problems.append("runtime must carry the session options this module runs")
    options = exporter.get("options") if isinstance(exporter, Mapping) else None
    if not isinstance(options, Mapping) or exporter.get("name") != options.get("exporter"):
        problems.append("exporter.name must be the exporter its options name")
    elif (
        not isinstance(opset, Mapping)
        or type(opset.get("requested")) is not int
        or opset.get("requested") != options.get("opset_version")
    ):
        problems.append("opset.requested must be the exporter options' opset_version")
    if isinstance(artifacts, Mapping) and artifacts.get("model") != MODEL_FILE:
        problems.append(f"artifacts.model must be {MODEL_FILE!r}, the file a profile loads")
    if problems:
        raise ConformanceError("malformed conformance record: " + "; ".join(problems))


def summary_lines(report: Mapping[str, Any]) -> list[str]:
    """One line per profile: its verdict, level, each stage's status and
    how many input cases passed."""
    lines = []
    for record in report["profiles"]:
        stages = ", ".join(f"{name} {record['stages'][name]['status']}" for name in STAGES)
        cases = sum(1 for case in record["input_cases"] if case.get("passed"))
        verdict = record["status"] + (f" ({record['failure']})" if record["failure"] else "")
        lines.append(
            f"{record['profile']}: {verdict}; level={record['level']}; {stages}; "
            f"{cases}/{len(record['input_cases'])} input cases passed"
        )
    return lines
