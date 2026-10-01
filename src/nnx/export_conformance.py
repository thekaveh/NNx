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
        batches = tuple(self.batches)
        if not batches or not all(isinstance(b, int) and not isinstance(b, bool) and b >= 1 for b in batches):
            raise ConformanceError(f"batches must be a non-empty list of positive batch sizes, got {self.batches!r}")
        object.__setattr__(self, "batches", batches)
        for name in ("rtol", "atol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ConformanceError(f"{name} must be finite and non-negative, got {value!r}")

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


def _importable(name: str) -> bool:
    """Present and importable: a broken install is a missing dependency."""
    if importlib.util.find_spec(name) is None:
        return False
    try:
        importlib.import_module(name)
    except Exception:  # any failure to import is the dependency's, not the export's
        return False
    return True


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
    for folder, folders, names in os.walk(directory):
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
    ``artifacts.directory`` lives under."""
    root = _artifact_dir(record, directory)
    listed = record["artifacts"]["files"]
    names = {_relative(entry["path"]) for entry in listed}
    problems = []
    for entry in listed:
        name = _relative(entry["path"])
        path = os.path.join(root, *name.split("/"))
        if not os.path.isfile(path):
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
    except (ConformanceError, OSError) as error:  # unreadable is unverified
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
        if not np.all(np.isfinite(out)):
            result.update(outcome="mismatch", failure="mismatch", detail="the runtime gave non-finite logits")
            continue
        error = np.abs(out.astype(np.float64) - reference)
        result["max_abs_error"] = float(error.max()) if error.size else 0.0
        result["max_rel_error"] = float((error / np.maximum(np.abs(reference), 1e-30)).max()) if error.size else 0.0
        within = bool(np.all(error <= atol + rtol * np.abs(reference)))
        result.update(outcome="match" if within else "mismatch", passed=within, failure=None if within else "mismatch")
    failed = [r for r in results if not r["passed"]]
    compared = sum(1 for r in results if r["outcome"] == "match")
    if not failed and compared:
        return _stage("passed", detail={"cases": len(results), "compared": compared}), results
    if not failed:
        return _stage("failed", "input-contract", {"failed_cases": [], "reason": "no valid input case"}), results
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
    elif profile.name in PROFILES and profile != PROFILES[profile.name]:
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
    missing = [name for name in profile.requires if not _importable(name)]
    if missing:
        record["stages"]["dependencies"] = _stage(
            "failed",
            "missing-dependency",
            f"not importable: {', '.join(missing)} — install thekaveh-nnx[onnx-runtime"
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
    if importlib.util.find_spec("onnxruntime") is None:
        raise ConformanceError(
            "execute() runs the recorded artifacts in ONNX Runtime, which is not installed: "
            "install thekaveh-nnx[onnx-runtime]"
        )
    config = {key: value for key, value in replay["config"].items() if key != "weights_sha256"}
    try:
        model = build_model(config)
    except ConformanceError:
        raise
    except (KeyError, TypeError, ValueError) as error:
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
    missing or unknown keys, unknown statuses or failure classes, a failure
    in the wrong stage, a stage that ran after a failed one, inconsistent
    ``status`` / ``failure`` / ``level``, a passed stage its own evidence
    contradicts (parity without a compared valid case or with a failed
    case, an export that changed the model, a check without an opset or a
    listed model file), malformed hashes or paths, or non-finite
    tolerances. Every problem is named."""
    if not isinstance(record, Mapping):
        raise ConformanceError(f"a conformance record is a mapping, got {type(record).__name__}")
    problems: list[str] = []
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
            and isinstance(case["shape"], list)
            and len(case["shape"]) == 2
            and all(isinstance(d, int) and not isinstance(d, bool) and d >= 1 for d in case["shape"])
        ):
            problems.append(f"input case {case.get('name')!r} has a malformed name, seed or shape")
    if status("parity") != "passed" and status("parity") != "failed":
        if any(case.get("passed") is not False or case.get("outcome") is not None for case in cases):
            problems.append("parity did not run, so no input case may carry an outcome")
    if status("parity") == "passed" and not all(
        case.get("passed") is True and case.get("outcome") == ("match" if case.get("expect") == "match" else "rejected")
        for case in cases
    ):
        problems.append("parity passed but an input case did not")
    state = record["model_state"]
    if status("export") == "passed" and not (
        isinstance(state, Mapping) and set(state) == {"parameters", "gradients", "modes"} and all(state.values())
    ):
        problems.append(f"the export passed but the model state was not kept: {state!r}")

    artifacts = record["artifacts"]
    files = artifacts.get("files") if isinstance(artifacts, Mapping) else None
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts) != {"directory", "model", "files"}
        or not isinstance(files, list)
    ):
        problems.append("artifacts must hold directory, model and a list of files")
        files = []
    paths = [entry.get("path") for entry in files if isinstance(entry, Mapping)]
    if len(paths) != len(set(map(str, paths))):
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
    elif not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
        for v in tolerances.values()
    ):
        problems.append(f"tolerances must be finite and non-negative, got {dict(tolerances)}")
    if not isinstance(record["config"], Mapping) or "weights_sha256" not in record["config"]:
        problems.append("config must carry weights_sha256")
    for key in ("source", "exporter", "opset", "versions", "runtime"):
        if not isinstance(record[key], Mapping):
            problems.append(f"{key} must be a mapping")
    if isinstance(record["runtime"], Mapping) and not {"name", "version", "provider"} <= set(record["runtime"]):
        problems.append("runtime must name its name, version and provider")
    tested = PROFILES.get(record["profile"]) if isinstance(record["profile"], str) else None
    if tested is not None:  # a tested profile's name stands for its exact settings
        config = record["config"]
        expected = {
            "tolerances": {"rtol": tested.rtol, "atol": tested.atol},
            "input cases": [{k: c[k] for k in ("name", "shape", "seed", "expect")} for c in tested.cases()],
            "dtype": tested.dtype,
            "runtime": (tested.runtime, tested.provider),
            "exporter options": tested.options(),
            "model config": copy.deepcopy(dict(tested.model)),
        }
        actual = {
            "tolerances": record["tolerances"],
            "input cases": [{k: c.get(k) for k in ("name", "shape", "seed", "expect")} for c in cases],
            "dtype": record["dtype"],
            "runtime": (
                (record["runtime"].get("name"), record["runtime"].get("provider"))
                if isinstance(record["runtime"], Mapping)
                else None
            ),
            "exporter options": record["exporter"].get("options") if isinstance(record["exporter"], Mapping) else None,
            "model config": (
                {k: v for k, v in config.items() if k != "weights_sha256"} if isinstance(config, Mapping) else None
            ),
        }
        changed = sorted(
            key
            for key in expected
            if json.dumps(expected[key], sort_keys=True) != json.dumps(actual[key], sort_keys=True)
        )
        if changed:
            problems.append(f"{record['profile']!r} is a tested profile, but its {', '.join(changed)} differ from it")
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
