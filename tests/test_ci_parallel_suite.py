"""FIX-031: the required test jobs run the suite on every core.

The serial suite took ~18.5 min on the slowest required leg (3.11, with
coverage) against a cap raised from 20 to 30 minutes; with pytest-xdist on a
4-core runner it takes a fraction of that. These checks keep the parallel
invocation and its dependency in place."""

from __future__ import annotations

from pathlib import Path

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def _pytest_runs(workflow: str, job: str) -> list[str]:
    steps = yaml.safe_load((ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8"))["jobs"][job][
        "steps"
    ]
    lines = [line.strip() for step in steps for line in str(step.get("run", "")).splitlines()]
    return [line for line in lines if line.startswith("uv run --frozen pytest")]


def test_the_full_suite_runs_in_parallel_in_the_required_job_and_the_release_gate():
    for workflow, job in (("ci.yml", "lint-and-test"), ("release.yml", "test")):
        runs = _pytest_runs(workflow, job)
        assert runs, (workflow, job)
        for run in runs:
            assert " -n auto" in run, (workflow, job, run)


def test_pytest_xdist_is_a_dev_dependency():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert any(spec.startswith("pytest-xdist") for spec in project["optional-dependencies"]["dev"])
