"""Run a GitHub Actions job locally, as far as the release workflows need (FIX-068).

A job runs the way the runner runs it: a job's and each step's ``if:`` is
evaluated with the implicit ``success() &&`` that GitHub adds when the
expression calls no status function; ``run:`` blocks are interpolated and run
in bash with the runner's default flags (``--noprofile --norc -eo pipefail``),
the step's ``env:`` and a ``GITHUB_OUTPUT`` file; a step's outputs are kept
when it fails, as the runner keeps what it wrote to ``GITHUB_OUTPUT``. A
``uses:`` step runs the callable the caller gives for that action (keyed by
the action's name without its ``@`` pin); any other ``uses:`` step, and a
``run:`` step outside ``execute``, is a no-op that succeeds.

Expressions cover the subset these workflows use: contexts, string and
boolean literals, ``==``, ``!=``, ``!``, ``&&``, ``||``, parentheses and the
four status functions. Workflow-dispatch inputs are typed as the workflow
declares them (a ``boolean`` input is a boolean, as in the ``inputs``
context), with the declared defaults."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# (rendered ``with:`` mapping, working directory, job name) -> (outcome, outputs)
Action = Callable[[dict[str, str], Path, str], tuple[str, dict[str, str]]]

_TOKEN = re.compile(
    r"\s*(?:(?P<string>'(?:[^']|'')*')|(?P<op>==|!=|&&|\|\||!|\(|\))"
    r"|(?P<status>(?:success|failure|cancelled|always)\(\))|(?P<name>[A-Za-z_][\w.-]*))"
)
_STATUS_FUNCTIONS = ("success()", "failure()", "cancelled()", "always()")
_INTERPOLATION = re.compile(r"\$\{\{(.*?)\}\}")


def _lookup(contexts: Mapping[str, Any], path: str) -> Any:
    value: Any = contexts
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None  # a missing context property is null
        value = value[part]
    return value


def evaluate(expression: str, contexts: Mapping[str, Any], *, failed: bool = False) -> Any:
    """The value of a workflow expression (``${{ }}`` optional)."""
    text = expression.strip()
    if text.startswith("${{") and text.endswith("}}"):
        text = text[3:-2]
    text = text.strip()
    source, names, pos = [], {}, 0
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if match is None or match.end() == pos:
            raise ValueError(f"unsupported expression: {expression!r}")
        pos = match.end()
        if match["string"] is not None:
            source.append(repr(match["string"][1:-1].replace("''", "'")))
        elif match["op"] is not None:
            source.append({"&&": " and ", "||": " or ", "!": " not "}.get(match["op"], match["op"]))
        elif match["status"] is not None:
            status = {"success()": not failed, "failure()": failed, "cancelled()": False, "always()": True}
            source.append(repr(status[match["status"]]))
        elif match["name"] in ("true", "false", "null"):
            source.append({"true": "True", "false": "False", "null": "None"}[match["name"]])
        else:
            key = f"_v{len(names)}"
            names[key] = _lookup(contexts, match["name"])
            source.append(key)
    return eval("".join(source), {"__builtins__": {}}, names)  # built from the tokens above only


def guard(condition: Any, contexts: Mapping[str, Any], *, failed: bool) -> bool:
    """Whether a job or step with this ``if:`` runs after (no) earlier failure."""
    if condition is None:
        return not failed
    condition = str(condition)
    value = bool(evaluate(condition, contexts, failed=failed))
    if not any(status in condition for status in _STATUS_FUNCTIONS):
        return not failed and value  # the implicit success() &&
    return value


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def render(text: str, contexts: Mapping[str, Any]) -> str:
    """``text`` with each ``${{ }}`` interpolated."""
    return _INTERPOLATION.sub(lambda match: _text(evaluate(match[1], contexts)), text)


def _read_outputs(text: str) -> dict[str, str]:
    outputs, lines = {}, iter(text.splitlines())
    for line in lines:
        if "<<" in line and ("=" not in line or line.index("<<") < line.index("=")):
            key, delimiter = line.split("<<", 1)
            body = []
            for value in lines:
                if value == delimiter:
                    break
                body.append(value)
            outputs[key] = "\n".join(body)
        elif "=" in line:
            key, value = line.split("=", 1)
            outputs[key] = value
    return outputs


def workflow(path: Path) -> dict[str, Any]:
    """The parsed workflow, with its ``on:`` key as the string (PyYAML reads it as ``True``)."""
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    if True in parsed:
        parsed["on"] = parsed.pop(True)
    return parsed


def dispatch_inputs(path: Path, given: Mapping[str, str]) -> dict[str, Any]:
    """The ``inputs`` context of a workflow_dispatch run given ``-f key=value`` strings."""
    declared = workflow(path)["on"]["workflow_dispatch"]["inputs"]
    unknown = set(given) - set(declared)
    if unknown:
        raise ValueError(f"{path.name} declares no input {sorted(unknown)}")  # GitHub refuses such a dispatch
    inputs: dict[str, Any] = {}
    for name, spec in declared.items():
        if name not in given:
            if spec.get("required"):
                raise ValueError(f"{path.name} requires input {name!r}")
            value: Any = spec.get("default", "")
        else:
            value = given[name]
        if spec.get("type") == "boolean":
            value = value if isinstance(value, bool) else value == "true"
        inputs[name] = value
    return inputs


@dataclass
class StepRun:
    outcome: str
    outputs: dict[str, str] = field(default_factory=dict)
    log: str = ""


@dataclass
class JobRun:
    steps: dict[str, StepRun]
    succeeded: bool

    def outcome(self, step: str) -> str:
        return self.steps[step].outcome


def run_job(
    path: Path,
    job: str,
    *,
    contexts: Mapping[str, Any],
    workdir: Path,
    env: Mapping[str, str],
    actions: Mapping[str, Action],
    execute: Collection[str] | None = None,
) -> JobRun | None:
    """Run ``job`` of the workflow at ``path``; ``None`` when its ``if:`` skips it.
    ``execute`` names the ``run:`` steps to run (default: all)."""
    spec = workflow(path)["jobs"][job]
    if not guard(spec.get("if"), contexts, failed=False):
        return None
    workdir.mkdir(parents=True, exist_ok=True)
    scope: dict[str, Any] = {**contexts, "steps": {}}
    results: dict[str, StepRun] = {}
    failed = False
    for step in spec["steps"]:
        name = step["name"]
        if not guard(step.get("if"), scope, failed=failed):
            result = StepRun("skipped")
        elif "run" in step and (execute is None or name in execute):
            result = _run_block(step, scope, workdir, env)
        elif "uses" in step and step["uses"].split("@", 1)[0] in actions:
            given = {key: render(str(value), scope) for key, value in (step.get("with") or {}).items()}
            outcome, outputs = actions[step["uses"].split("@", 1)[0]](given, workdir, job)
            result = StepRun(outcome, outputs)
        else:
            result = StepRun("success")  # a step the simulation does not need
        results[name] = result
        if "id" in step:
            scope["steps"][step["id"]] = {
                "outcome": result.outcome,
                "conclusion": result.outcome,
                "outputs": result.outputs,
            }
        failed = failed or result.outcome == "failure"
    return JobRun(results, not failed)


def _run_block(step: Mapping[str, Any], scope: Mapping[str, Any], workdir: Path, env: Mapping[str, str]) -> StepRun:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "output"
        output.write_text("", encoding="utf-8")
        script = Path(tmp) / "step.sh"
        script.write_text(render(step["run"], scope), encoding="utf-8")
        step_env = {
            **env,
            **{key: render(str(value), scope) for key, value in (step.get("env") or {}).items()},
            "GITHUB_OUTPUT": str(output),
        }
        done = subprocess.run(
            ["bash", "--noprofile", "--norc", "-eo", "pipefail", str(script)],
            cwd=workdir,
            env=step_env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        outputs = _read_outputs(output.read_text(encoding="utf-8"))
    return StepRun("success" if done.returncode == 0 else "failure", outputs, done.stdout + done.stderr)


# A stand-in for the GitHub CLI: it records every call, fails a call that
# matches a pending one-off failure, and keeps the releases and the workflow
# dispatches in the JSON state file named by FAKE_GH_STATE.
FAKE_GH = r"""
import json
import os
import subprocess
import sys
from pathlib import Path

state_path = Path(os.environ["FAKE_GH_STATE"])
state = json.loads(state_path.read_text(encoding="utf-8"))
args = sys.argv[1:]
command = " ".join(args)
state["calls"].append(command)


def save():
    state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")


def option(name):
    return args[args.index(name) + 1] if name in args else None


for pattern in list(state["fail_once"]):
    if pattern in command:
        state["fail_once"].remove(pattern)
        save()
        sys.exit("HTTP 502: Bad Gateway (a one-off failure)")

if args[:2] == ["workflow", "run"]:
    inputs = dict(args[i + 1].split("=", 1) for i, arg in enumerate(args) if arg == "-f")
    state["dispatches"].append({"workflow": args[2], "ref": option("--ref"), "inputs": inputs})
elif args[:2] == ["pr", "list"]:
    print("\n".join(state["release_branches"]))
elif args[0] == "api" and "/commits/" in args[1]:
    ref = args[1].rsplit("/commits/", 1)[1]
    found = subprocess.run(
        ["git", "-C", state["origin"], "rev-parse", "--verify", "--quiet", ref + "^{commit}"],
        capture_output=True, text=True,
    )
    if found.returncode:
        save()
        sys.exit("HTTP 422: No commit found for SHA: " + ref)
    print(found.stdout.strip())
elif args[:2] == ["release", "view"]:
    release = state["releases"][args[2]]
    print({"databaseId": 1, "isDraft": json.dumps(release["draft"]), "assets": "\n".join(release["assets"])}[option("--json")])
elif args[:2] == ["release", "upload"]:
    state["releases"][args[2]]["assets"] = sorted(Path(arg).name for arg in args[3:] if arg.startswith("dist/"))
elif args[:2] == ["release", "edit"]:
    release = state["releases"][args[2]]
    if option("--notes-file"):
        release["body"] = Path(option("--notes-file")).read_text(encoding="utf-8")
    if "--draft=false" in args:
        release["draft"] = False
elif args[:2] == ["release", "verify"]:
    pass
else:
    save()
    sys.exit("fake gh: unhandled command " + repr(command))
save()
"""


def os_path(*directories: Path) -> str:
    """``PATH`` with ``directories`` first."""
    return os.pathsep.join([*(str(directory) for directory in directories), os.environ.get("PATH", "")])
