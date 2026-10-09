"""FIX-031: the required test jobs run the suite on every core.

The serial suite took ~18.5 min on the slowest required leg (3.11, with
coverage) against a cap raised from 20 to 30 minutes; with pytest-xdist on a
4-vCPU runner it takes a fraction of that. These checks keep the parallel
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
            # `-n auto` counts physical cores: 2 workers on a 4-vCPU runner.
            assert " -n logical " in f"{run} ", (workflow, job, run)
            # Each torchrun module's shared launch runs once, on one worker.
            assert " --dist loadgroup" in run, (workflow, job, run)


def test_pytest_xdist_is_a_dev_dependency():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert any(spec.startswith("pytest-xdist") for spec in project["optional-dependencies"]["dev"])


def test_every_torchrun_launch_shares_one_worker():
    """A module-scoped torchrun fixture runs once per worker that gets one of
    its tests, and two concurrent 2-rank launches oversubscribe a 4-vCPU
    runner (a launch timed out at 240 s on #453): every test that launches
    torchrun is in the one ``torchrun`` group."""
    for name in ("test_ddp_adapter.py", "test_ddp_resume.py"):
        assert 'pytestmark = pytest.mark.xdist_group("torchrun")' in (ROOT / "tests" / name).read_text(encoding="utf-8")
    smoke = (ROOT / "tests" / "test_examples_smoke.py").read_text(encoding="utf-8")
    for test in (
        "test_the_ddp_example_runs_under_torchrun_with_one_artifact_owner",
        "test_the_ddp_example_terminates_after_an_injected_failure",
    ):
        assert f'@pytest.mark.xdist_group("torchrun")  # one torchrun launch at a time (FIX-031)\ndef {test}(' in smoke
    launchers = sorted(
        path.name
        for path in (ROOT / "tests").glob("test_*.py")
        if "torch.distributed.run" in path.read_text(encoding="utf-8")
    )
    assert launchers == ["test_ci_parallel_suite.py", "test_ddp_adapter.py", "test_examples_smoke.py"], launchers
