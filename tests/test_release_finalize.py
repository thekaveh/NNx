"""FIX-029: a release-please PR passes its checks with no manual finalize commit.

Two steps used to need a hand-made commit on every release PR (v0.2.3,
v0.3.0, v0.4.0): ``docs/api.md`` embedded the *installed* version, so the
API-reference check failed once release-please bumped ``pyproject.toml``; and
the curated ``[Unreleased]`` notes had to replace release-please's generated
commit list under the version heading. The version line now carries the
``x-release-please-version`` marker (release-please bumps it), and
``scripts/release/finalize_changelog.py`` — run by the release workflow —
does the CHANGELOG step. The fixtures are release PR #399's CHANGELOG head
before (430a0af) and after (4eacc12) the manual finalize."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from tests import _actions_sim
from tests._repo_files import repo_file

# Repository-only: the sdist ships tests/ but not scripts/, examples/ or docs/,
# so a packager running its suite skips this module instead of failing collection.
repo_file("scripts", "release", "finalize_changelog.py")

from scripts.release.finalize_changelog import finalize, release_notes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "release_finalize"
# release-please's generic updater (17.6.0, src/updaters/generic.ts): the first
# semantic version on every line that carries the marker.
_MARKED_VERSION = re.compile(
    r"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(-(?P<pre>[\w.]+))?(\+(?P<build>[-\w.]+))?"
)


def _release_please_bump(text: str, version: str) -> str:
    return "\n".join(
        _MARKED_VERSION.sub(version, line, count=1) if "x-release-please-version" in line else line
        for line in text.split("\n")
    )


# --- CHANGELOG ---------------------------------------------------------------------------------------


def test_the_curated_notes_replace_the_generated_list_as_the_manual_finalize_did():
    before = (FIXTURES / "before.md").read_text(encoding="utf-8")
    after = (FIXTURES / "after.md").read_text(encoding="utf-8")
    assert finalize(before) == after


def test_finalizing_is_idempotent_and_the_repository_changelog_is_finalizable():
    after = (FIXTURES / "after.md").read_text(encoding="utf-8")
    assert finalize(after) == after
    # On develop it is already final; on a release PR (FIX-033: not finalized
    # there) it holds exactly one release section above [Unreleased]. Either
    # way finalizing succeeds and is a fixed point.
    current = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert finalize(finalize(current)) == finalize(current)


def test_the_release_notes_are_the_curated_notes_of_an_unfinalized_release_commit():
    """FIX-033: the release commit is release-please's own, not finalized; the
    published notes are the curated [Unreleased] notes it carries."""
    before = (FIXTURES / "before.md").read_text(encoding="utf-8")
    after = (FIXTURES / "after.md").read_text(encoding="utf-8")
    version = after.split("## [", 2)[2].split("]", 1)[0]  # the first release section of the finalized fixture
    assert release_notes(before, version) == release_notes(after, version)


def test_without_curated_notes_the_generated_list_stays():
    text = (
        "# 22. Changelog\n\n## [0.5.0](https://example/compare) (2026-11-01)\n\n\n### Features\n\n* **x:** y\n\n"
        "## [Unreleased]\n\n## [0.4.0](https://example/compare) (2026-10-06)\n"
    )
    assert finalize(text) == (
        "# 22. Changelog\n\n## [Unreleased]\n\n## [0.5.0](https://example/compare) (2026-11-01)\n\n\n"
        "### Features\n\n* **x:** y\n\n## [0.4.0](https://example/compare) (2026-10-06)\n"
    )


def test_two_unfinalized_releases_are_refused_rather_than_one_dropped():
    text = "# C\n\n## [0.6.0](x) (d)\n\n* new\n\n## [0.5.0](x) (d)\n\n* older\n\n## [Unreleased]\n\n- curated\n"
    with pytest.raises(ValueError, match="more than one release section"):
        finalize(text)


def test_a_heading_inside_a_fenced_block_is_content():
    text = "# C\n\n## [0.5.0](x) (d)\n\n* gen\n\n## [Unreleased]\n\n```md\n## not a heading\n```\n"
    assert finalize(text) == "# C\n\n## [Unreleased]\n\n## [0.5.0](x) (d)\n\n```md\n## not a heading\n```\n"


def test_the_release_notes_are_the_curated_section():
    after = (FIXTURES / "after.md").read_text(encoding="utf-8")
    notes = release_notes(after, "0.4.0")
    assert notes.startswith("### Added\n") and "Ordered logits pipelines" in notes and "## [" not in notes
    assert "([486ad73]" not in notes  # never the generated commit list
    with pytest.raises(ValueError, match="no section for 9.9.9"):
        release_notes(after, "9.9.9")


def test_a_changelog_without_an_unreleased_section_is_refused():
    with pytest.raises(ValueError, match=r"\[Unreleased\]"):
        finalize("# Changelog\n\n## [0.5.0](x) (2026-11-01)\n\n* a\n")


# --- docs/api.md --------------------------------------------------------------------------------------


def test_the_api_reference_version_line_is_bumped_by_release_please():
    import nnx
    from scripts.docs import build_api_reference

    api = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    marked = [line for line in api.split("\n") if "x-release-please-version" in line]
    assert marked == [f"nnx.__version__ = {nnx.__version__!r}  # x-release-please-version"]
    config = json.loads((ROOT / "release-please-config.json").read_text(encoding="utf-8"))
    extra = config["packages"]["."]["extra-files"]
    assert {"type": "generic", "path": "docs/api.md"} in extra
    # What release-please writes on a release branch equals the reference
    # rendered there (the installed metadata then reports the new version).
    bumped = _release_please_bump(api, "9.8.7")
    assert bumped == api.replace(f"{nnx.__version__!r}  # x-release", "'9.8.7'  # x-release")
    assert build_api_reference._signature("nnx.__version__", "9.8.7") == (
        "nnx.__version__ = '9.8.7'  # x-release-please-version"
    )


def test_the_release_workflow_publishes_the_curated_notes_and_never_commits():
    """FIX-033: a bot push to a release PR makes the main ruleset ask for an
    approval, verified or not (#475), so the workflows commit nothing; the
    release notes are folded from the release commit at publish time
    (FIX-068: by release.yml, from the commit the tag names)."""
    release_please = (ROOT / ".github" / "workflows" / "release-please.yml").read_text(encoding="utf-8")
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert '--release-notes "$RELEASE_TAG"' in release and "--notes-file" in release
    for workflow in (release_please, release):
        for push in ("git push", "git commit", "api_commit", "git/commits"):
            assert push not in workflow, push
    assert "uv lock" not in release_please  # release.yml runs `uv lock --check`
    assert 'gh workflow run ci.yml --repo "$GITHUB_REPOSITORY" --ref "$branch"' in release_please


def test_release_please_bumps_the_lockfile_version_itself():
    """FIX-033: release-please's toml updater bumps the package's version in
    uv.lock (one line), so the release PR's lock stays consistent with
    pyproject.toml without a lockfile refresh commit."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib

    config = json.loads((ROOT / "release-please-config.json").read_text(encoding="utf-8"))
    updaters = config["packages"]["."]["extra-files"]
    assert {
        "type": "toml",
        "path": "uv.lock",
        "jsonpath": "$.package[?(@.name.value=='thekaveh-nnx')].version",
    } in updaters
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    ours = [entry for entry in lock["package"] if entry["name"] == "thekaveh-nnx"]
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert [entry["version"] for entry in ours] == [project["version"]]


def test_release_please_bumps_the_readme_version_line_and_its_pypi_copy_alike():
    """#472: the README names the version it documents on a marked line that
    release-please bumps in README.md and PYPI_README.md (both extra-files);
    the PyPI copy carries the line unchanged, so the bumped pair stays in
    sync and `build_pypi_readme --check` passes on the release PR."""
    marker = "x-release-please-version"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    pypi = (ROOT / "PYPI_README.md").read_text(encoding="utf-8")
    (line,) = [entry for entry in readme.split("\n") if marker in entry]
    assert [entry for entry in pypi.split("\n") if marker in entry] == [line]
    bumped = _release_please_bump(readme, "9.8.7")
    assert "NNx 9.8.7 and its unreleased changes" in bumped
    assert _release_please_bump(pypi, "9.8.7").count("NNx 9.8.7 and") == 1


_SHIPPED = """# Changelog

## [Unreleased]

### Fixed

- A shipped fix.
"""

_SYNCED = """# Changelog

## [0.9.0](https://example.invalid) (2030-01-01)

* generated A

## [Unreleased]

### Added

- A develop-only feature.

### Fixed

- A shipped fix.
- A develop-only fix.

## [0.8.0]

- old
"""


def test_the_post_release_sync_files_only_what_the_release_shipped():
    """Entries develop added after the release branched stay in
    [Unreleased]; only the shipped ones move under the version."""
    finalized = finalize(_SYNCED, _SHIPPED)
    unreleased, rest = finalized.split("## [0.9.0]", 1)
    assert "A develop-only feature." in unreleased and "A develop-only fix." in unreleased
    assert "A shipped fix." not in unreleased
    released = rest.split("## [0.8.0]", 1)[0]
    assert "A shipped fix." in released and "develop-only" not in released and "generated A" not in released
    assert "### Added" in unreleased and unreleased.index("### Added") < unreleased.index("### Fixed")
    assert finalize(finalized) == finalized


def test_an_entry_edited_after_the_release_is_refused_rather_than_misfiled():
    edited = _SYNCED.replace("- A shipped fix.", "- A shipped fix, reworded.")
    with pytest.raises(ValueError, match="fix the changelog by hand"):
        finalize(edited, _SHIPPED)


def test_no_changelog_section_lists_one_entry_twice():
    """merge=union keeps both sides of an edited line: two entries with one
    bold title in a section are a union-merge duplicate, not two changes."""
    import collections

    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    for section in re.split(r"\n## ", text):
        titles = re.findall(r"^- \*\*(.+?)\*\*", section, flags=re.M)
        # A title ending in ":" is a category label ("Critical:"), repeated by design.
        repeated = [t for t, count in collections.Counter(titles).items() if count > 1 and not t.endswith(":")]
        assert not repeated, (section.splitlines()[0], repeated)


def test_a_later_entry_keeps_its_blank_lines_and_fenced_code():
    synced = _SYNCED.replace(
        "- A develop-only fix.\n", "- A develop-only fix.\n\n  ```python\n  x = 1\n\n  y = 2\n  ```\n"
    )
    unreleased = finalize(synced, _SHIPPED).split("## [0.9.0]", 1)[0]
    assert "  ```python\n  x = 1\n\n  y = 2\n  ```" in unreleased


# --- FIX-068: retryable post-release steps, notes from the tagged commit -----------------------------
#
# The workflows run in the simulation of tests/_actions_sim.py against a
# local git history and a fake `gh`. History (Context part 2 of #535):
# main: P <- M (the release PR's merge, tagged v0.4.2) <- N (a later push
# whose [Unreleased] notes hold "A later fix"). The release-please action's
# outputs are those of release-please-action 45996ed (v5.0.0): it writes its
# release outputs before its PR phase, and a re-run after the release exists
# outputs no release (the PR is "autorelease: tagged" by then).

_WORKFLOWS = ROOT / ".github" / "workflows"
_RELEASE_PLEASE = _WORKFLOWS / "release-please.yml"
_RELEASE = _WORKFLOWS / "release.yml"
_TAG = "v0.4.2"
_GENERATED = "* a fix ([1a2b3c4](https://github.com/thekaveh/NNx/commit/1a2b3c4))\n"
_ENTRY = (
    "## [0.4.2](https://github.com/thekaveh/NNx/compare/v0.4.1...v0.4.2) (2026-10-22)\n\n\n### Bug Fixes\n\n"
    + _GENERATED
)
_STALE = "## [0.4.1](https://github.com/thekaveh/NNx/compare/v0.4.0...v0.4.1) (2026-10-01)\n\n\n* an older fix\n"
_DEVELOP = (
    "# Changelog\n\n## [Unreleased]\n\n### Fixed\n\n- **A shipped fix.** In 0.4.2.\n\n"
    "## [0.4.0](https://github.com/thekaveh/NNx/compare/v0.3.0...v0.4.0) (2026-09-01)\n\n- old\n"
)
# The release-workflow steps the simulation runs; the others (tests, builds,
# PyPI) succeed as no-ops.
_BUILD_STEPS = {"Assert the tag identifies the release commit", "Fold the curated release notes from the tagged commit"}
_PUBLISH_STEPS = {
    "Verify tag still identifies the release commit",
    "Attach distributions or verify an already-published release",
    "Verify existing immutable release attestation",
    "Publish the curated release notes",
    "Publish Release Please draft",
    "Verify new immutable release attestation",
}


class _ReleaseWorld:
    """An origin repository with a tagged release commit and a draft release."""

    def __init__(self, tmp_path: Path, *, stale_section: bool = False, fail_once: tuple[str, ...] = ()):
        self.tmp = tmp_path
        self.origin = tmp_path / "origin"
        self.runs = 0
        self.failing_checkouts: set[str] = set()
        self.artifacts: dict[str, dict[str, bytes]] = {}
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_gh = tmp_path / "fake_gh.py"
        fake_gh.write_text(_actions_sim.FAKE_GH, encoding="utf-8")
        python = shlex.quote(sys.executable)
        for name, command in (("gh", f"{python} -I {shlex.quote(str(fake_gh))}"), ("python3", python)):
            (bin_dir / name).write_text(f'#!/bin/sh\nexec {command} "$@"\n', encoding="utf-8")
            (bin_dir / name).chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": _actions_sim.os_path(bin_dir),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GITHUB_REPOSITORY": "thekaveh/NNx",
            "FAKE_GH_STATE": str(tmp_path / "gh.json"),
        }
        self.env.pop("GH_TOKEN", None)
        self.origin.mkdir()
        self.git("init", "-q", "-b", "main")
        script = self.origin / "scripts" / "release" / "finalize_changelog.py"
        script.parent.mkdir(parents=True)
        script.write_bytes((ROOT / "scripts" / "release" / "finalize_changelog.py").read_bytes())
        log = self.origin / "CHANGELOG.md"
        develop = _DEVELOP.replace("## [Unreleased]", _STALE + "\n## [Unreleased]") if stale_section else _DEVELOP
        log.write_text(develop, encoding="utf-8")
        self.commit("P")
        # release-please 17.6.0 puts the new entry above the first section.
        log.write_text(develop.replace("# Changelog\n\n", "# Changelog\n\n" + _ENTRY + "\n", 1), encoding="utf-8")
        self.release_ref = self.commit("chore(main): release 0.4.2")
        self.git("tag", _TAG)
        text = log.read_text(encoding="utf-8")
        at = text.index("### Fixed\n\n", text.index("## [Unreleased]")) + len("### Fixed\n\n")
        log.write_text(text[:at] + "- **A later fix.** Not in 0.4.2.\n" + text[at:], encoding="utf-8")
        self.head = self.commit("N")
        self.save_state(
            {
                "calls": [],
                "fail_once": list(fail_once),
                "dispatches": [],
                "release_branches": [],
                "origin": str(self.origin),
                "releases": {_TAG: {"draft": True, "body": _GENERATED, "assets": []}},
            }
        )

    def git(self, *args: str) -> str:
        done = subprocess.run(
            ["git", "-C", str(self.origin), "-c", "user.email=x@example.invalid", "-c", "user.name=x", *args],
            check=True,
            capture_output=True,
            text=True,
            env=self.env,
        )
        return done.stdout.strip()

    def commit(self, message: str) -> str:
        self.git("add", "-A")
        self.git("commit", "-qm", message)
        return self.git("rev-parse", "HEAD")

    def changelog_at(self, rev: str) -> str:
        return self.git("show", f"{rev}:CHANGELOG.md") + "\n"

    def state(self) -> dict:
        return json.loads(Path(self.env["FAKE_GH_STATE"]).read_text(encoding="utf-8"))

    def save_state(self, state: dict) -> None:
        Path(self.env["FAKE_GH_STATE"]).write_text(json.dumps(state), encoding="utf-8")

    def dispatches(self, name: str) -> list[dict]:
        return [dispatch for dispatch in self.state()["dispatches"] if dispatch["workflow"] == name]

    def gh(self, *args: str) -> None:
        """A maintainer's own ``gh`` call."""
        subprocess.run(["gh", *args], check=True, capture_output=True, text=True, env=self.env)

    # --- the uses: steps -------------------------------------------------------------------------

    def checkout(self, given: dict[str, str], workdir: Path, job: str) -> tuple[str, dict[str, str]]:
        if job in self.failing_checkouts:
            self.failing_checkouts.discard(job)
            return "failure", {}
        subprocess.run(["git", "clone", "-q", str(self.origin), str(workdir)], check=True, env=self.env)
        ref = given.get("ref") or self.head  # no ref: the run's github.sha
        subprocess.run(["git", "-C", str(workdir), "checkout", "-q", "--detach", ref], check=True, env=self.env)
        return "success", {}

    def upload(self, given: dict[str, str], workdir: Path, job: str) -> tuple[str, dict[str, str]]:
        path = workdir / given["path"]
        files = [path] if path.is_file() else sorted(path.glob("*")) if path.is_dir() else []
        self.artifacts[given["name"]] = {file.name: file.read_bytes() for file in files}  # overwrite: true
        return "success", {}

    def download(self, given: dict[str, str], workdir: Path, job: str) -> tuple[str, dict[str, str]]:
        if given["name"] not in self.artifacts:
            return "failure", {}
        target = workdir / given["path"]
        target.mkdir(parents=True, exist_ok=True)
        for name, data in self.artifacts[given["name"]].items():
            (target / name).write_bytes(data)
        return "success", {}

    def _run(self, path: Path, job: str, contexts: dict, actions: dict, execute=None):
        self.runs += 1
        runner_temp = self.tmp / f"runner-{self.runs}"
        runner_temp.mkdir()
        contexts = {**contexts, "runner": {"temp": str(runner_temp)}}
        return _actions_sim.run_job(
            path,
            job,
            contexts=contexts,
            workdir=self.tmp / f"work-{self.runs}",
            env={**self.env, "RUNNER_TEMP": str(runner_temp)},
            actions={
                "actions/checkout": self.checkout,
                "actions/upload-artifact": self.upload,
                "actions/download-artifact": self.download,
                **actions,
            },
            execute=execute,
        )

    # --- the workflows ---------------------------------------------------------------------------

    def release_please_push(self, outputs: dict[str, str], *, action_fails: bool = False) -> dict:
        """One run of release-please.yml for the push of N (all its jobs)."""

        def action(given, workdir, job):
            return ("failure" if action_fails else "success"), outputs

        github = {"event_name": "push", "ref": "refs/heads/main", "sha": self.head, "token": "x"}
        return {
            job: self._run(
                _RELEASE_PLEASE, job, {"github": github, "inputs": {}}, {"googleapis/release-please-action": action}
            )
            for job in _actions_sim.workflow(_RELEASE_PLEASE)["jobs"]
        }

    def release_please_dispatch(self, **given: str) -> dict:
        """A maintainer's ``gh workflow run release-please.yml --ref main -f ...``."""
        github = {"event_name": "workflow_dispatch", "ref": "refs/heads/main", "sha": self.head, "token": "x"}
        inputs = _actions_sim.dispatch_inputs(_RELEASE_PLEASE, given)
        return {
            job: self._run(_RELEASE_PLEASE, job, {"github": github, "inputs": inputs}, {})
            for job in _actions_sim.workflow(_RELEASE_PLEASE)["jobs"]
        }

    def release(self, dispatch: dict) -> list[tuple[str, bool]]:
        """release.yml for one dispatch: the notes and GitHub-release jobs, each
        re-run once after a failure ("Re-run failed jobs": the same inputs and
        the same run's artifacts). Returns ``(job, succeeded)`` per attempt; a
        failed build stops the run before anything is published."""
        assert dispatch["workflow"] == "release.yml" and dispatch["ref"] == "main"
        github = {"event_name": "workflow_dispatch", "ref": "refs/heads/main", "sha": self.head, "token": "x"}
        contexts = {"github": github, "inputs": _actions_sim.dispatch_inputs(_RELEASE, dispatch["inputs"])}
        attempts = []
        for job, steps in (("build-distributions", _BUILD_STEPS), ("publish-github-release", _PUBLISH_STEPS)):
            for _ in range(2):
                run = self._run(_RELEASE, job, contexts, {}, execute=steps)
                assert run is not None, job
                attempts.append((job, run.succeeded))
                if run.succeeded:
                    break
            else:
                return attempts
        return attempts


def _released(world: _ReleaseWorld, *, prs_created: bool = False) -> dict[str, str]:
    return {
        "releases_created": "true",
        "release_created": "true",
        "tag_name": _TAG,
        "sha": world.release_ref,  # the release PR's merge commit, as the action reports it
        "paths_released": '["."]',
        "prs_created": "true" if prs_created else "false",
    }


_RERUN = {"releases_created": "false", "paths_released": "[]", "prs_created": "false"}

# One-off failures after release-please created the release: (fail_once
# patterns for gh, failing checkouts by job, the action's PR phase fails,
# a release PR is open).
_FAILURES = {
    "none": ((), (), False, False),
    "the action's PR phase": ((), (), True, False),
    "the required-checks dispatch": (("workflow run ci.yml",), (), False, True),
    "the release dispatch": (("workflow run release.yml",), (), False, False),
    "the notes publication": (("--notes-file",), (), False, False),
    "the notes job's checkout": ((), ("build-distributions",), False, False),
    "the publish job's checkout": ((), ("publish-github-release",), False, False),
}


@pytest.mark.parametrize("failure", list(_FAILURES))
def test_a_one_off_failure_after_the_release_exists_still_publishes_it_with_the_tagged_notes(tmp_path, failure):
    """#535: each failure ends with the curated notes of the tagged commit
    published and release.yml run for the existing tag, through a re-run of
    the failed job or the documented recovery entry point."""
    fail_once, checkouts, action_fails, prs_created = _FAILURES[failure]
    world = _ReleaseWorld(tmp_path, fail_once=fail_once)
    world.failing_checkouts.update(checkouts)
    if prs_created:
        state = world.state()
        state["release_branches"] = ["release-please--branches--main--components--thekaveh-nnx"]
        world.save_state(state)
    world.release_please_push(_released(world, prs_created=prs_created), action_fails=action_fails)
    # Only a failed dispatch itself leaves release.yml undispatched: a failed
    # step before it no longer skips it.
    assert bool(world.dispatches("release.yml")) == (failure != "the release dispatch")
    if not world.dispatches("release.yml"):
        # "Re-run failed jobs": the release exists, so the action outputs none
        # and the re-run dispatches nothing...
        world.release_please_push(_RERUN)
        assert not world.dispatches("release.yml")
        # ...so the recovery entry point dispatches release.yml for the tag.
        world.release_please_dispatch(tag_name=_TAG)
    (dispatch,) = world.dispatches("release.yml")
    assert dispatch["inputs"]["release_ref"] == world.release_ref and dispatch["inputs"]["tag_name"] == _TAG
    attempts = world.release(dispatch)
    assert attempts[-1] == ("publish-github-release", True), attempts
    release = world.state()["releases"][_TAG]
    assert release["draft"] is False
    assert release["body"] == release_notes(world.changelog_at(_TAG), "0.4.2")
    assert "A shipped fix." in release["body"] and _GENERATED not in release["body"]
    # Context part 2: github.sha's notes would hold the later fix.
    assert "A later fix" not in release["body"]
    assert "A later fix" in release_notes(world.changelog_at("HEAD"), "0.4.2")


def test_a_refused_fold_stops_the_release_before_anything_is_published_and_hand_fixed_notes_ship(tmp_path):
    """#535: two unfinalized release sections are still refused; the release
    stops before PyPI, and after a manual fix of the notes the recovery entry
    point ships the draft with them (keep_draft_notes)."""
    world = _ReleaseWorld(tmp_path, stale_section=True)
    world.release_please_push(_released(world))
    (dispatch,) = world.dispatches("release.yml")
    # A re-run cannot fix the tagged commit: the build job fails both times
    # and nothing is published (publish-pypi needs it).
    assert world.release(dispatch) == [("build-distributions", False)] * 2
    assert world.state()["releases"][_TAG] == {"draft": True, "body": _GENERATED, "assets": []}
    tagged = world.changelog_at(_TAG)
    with pytest.raises(ValueError, match="more than one release section"):
        release_notes(tagged, "0.4.2")
    # The manual fix: fold the notes from a fixed copy of the tag's CHANGELOG
    # and publish them on the draft.
    notes = tmp_path / "notes.md"
    notes.write_text(release_notes(tagged.replace(_STALE + "\n", ""), "0.4.2"), encoding="utf-8")
    world.gh("release", "edit", _TAG, "--repo", "thekaveh/NNx", "--notes-file", str(notes))
    world.release_please_dispatch(tag_name=_TAG, keep_draft_notes="true")
    recovery = world.dispatches("release.yml")[-1]
    assert recovery["inputs"] == {"release_ref": world.release_ref, "tag_name": _TAG, "keep_draft_notes": "true"}
    assert world.release(recovery) == [("build-distributions", True), ("publish-github-release", True)]
    release = world.state()["releases"][_TAG]
    assert release["draft"] is False and release["body"] == notes.read_text(encoding="utf-8")
    assert "A shipped fix." in release["body"] and "older fix" not in release["body"]


def test_the_recovery_entry_point_refuses_a_name_that_is_not_a_release_tag(tmp_path):
    world = _ReleaseWorld(tmp_path)
    runs = world.release_please_dispatch(tag_name="main")
    assert runs["release-please"] is None  # the push job never runs on a dispatch
    assert not runs["dispatch-release"].succeeded
    assert not world.dispatches("release.yml")


def test_the_release_please_workflow_only_dispatches_and_checks_nothing_out():
    """#535: release-please.yml checks out no code (the CodeQL alert closed by
    5f7ebe4), publishes no notes, and dispatches the release for the tag's
    commit even after a failed step; a dispatch of the workflow is the
    recovery entry point."""
    text = _RELEASE_PLEASE.read_text(encoding="utf-8")
    for absent in ("actions/checkout", "finalize_changelog", "gh release edit"):
        assert absent not in text, absent
    parsed = _actions_sim.workflow(_RELEASE_PLEASE)
    assert set(parsed["on"]) == {"push", "workflow_dispatch"}
    inputs = parsed["on"]["workflow_dispatch"]["inputs"]
    assert inputs["tag_name"]["required"] is True and inputs["tag_name"]["type"] == "string"
    assert inputs["keep_draft_notes"] == {
        "description": inputs["keep_draft_notes"]["description"],
        "required": False,
        "default": False,
        "type": "boolean",
    }
    jobs = parsed["jobs"]
    assert jobs["release-please"]["if"] == "github.event_name == 'push'"
    steps = {step["name"]: step for step in jobs["release-please"]["steps"]}
    dispatch = steps["Dispatch verified release workflow"]
    assert list(steps)[-1] == "Dispatch verified release workflow"
    assert dispatch["if"] == "${{ !cancelled() && steps.release.outputs.release_created == 'true' }}"
    assert dispatch["env"]["RELEASE_REF"] == "${{ steps.release.outputs.sha }}"
    assert dispatch["env"]["RELEASE_TAG"] == "${{ steps.release.outputs.tag_name }}"
    recovery = jobs["dispatch-release"]
    assert recovery["if"] == "github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main'"
    assert recovery["permissions"] == {"actions": "write", "contents": "read"}
    (step,) = recovery["steps"]
    assert step["env"]["RELEASE_TAG"] == "${{ inputs.tag_name }}"
    assert 'gh api "repos/$GITHUB_REPOSITORY/commits/$RELEASE_TAG" --jq .sha' in step["run"]
    assert "${{" not in step["run"] and "${{" not in dispatch["run"]  # inputs reach the shell through env only


def test_release_yml_folds_the_notes_from_the_tagged_commit_and_publishes_them_before_the_draft():
    """#535: the notes are folded in the build job, from the checkout of
    inputs.release_ref after the tag is asserted to name it, before anything
    is published, and are published before the draft is."""
    parsed = _actions_sim.workflow(_RELEASE)
    keep = parsed["on"]["workflow_dispatch"]["inputs"]["keep_draft_notes"]
    assert (keep["type"], keep["default"], keep["required"]) == ("boolean", False, False)
    jobs = parsed["jobs"]
    build = {step["name"]: step for step in jobs["build-distributions"]["steps"]}
    names = list(build)
    assert build["Checkout"]["with"]["ref"] == "${{ inputs.release_ref || github.sha }}"
    fold = build["Fold the curated release notes from the tagged commit"]
    assert (
        names.index("Checkout")
        < names.index("Assert the tag identifies the release commit")
        < names.index("Fold the curated release notes from the tagged commit")
        < names.index("Upload the curated release notes")
    )
    assert fold["if"] == "inputs.tag_name != '' && !inputs.keep_draft_notes"
    assert 'finalize_changelog.py CHANGELOG.md --release-notes "$RELEASE_TAG"' in fold["run"]
    assert build["Upload the curated release notes"]["if"] == "steps.notes.outcome == 'success'"
    assert "build-distributions" in jobs["publish-pypi"]["needs"]
    publish = {step["name"]: step for step in jobs["publish-github-release"]["steps"]}
    names = list(publish)
    assert publish["Checkout release commit"]["with"]["ref"] == "${{ inputs.release_ref }}"
    notes = publish["Publish the curated release notes"]
    assert (
        names.index("Download the curated release notes")
        < names.index("Publish the curated release notes")
        < names.index("Publish Release Please draft")
    )
    assert notes["if"] == "steps.release_state.outputs.already_published != 'true' && !inputs.keep_draft_notes"
    assert '--notes-file "$RUNNER_TEMP/release-notes/notes.md"' in notes["run"]
    assert publish["Download the curated release notes"]["if"] == "${{ !inputs.keep_draft_notes }}"


def test_the_workflow_simulation_applies_the_implicit_success_check():
    """The simulation's own premise, from GitHub's expression docs: an ``if:``
    without a status function gets ``success() &&``."""
    contexts = {"steps": {"release": {"outputs": {"release_created": "true"}}}}
    assert _actions_sim.guard("steps.release.outputs.release_created == 'true'", contexts, failed=False)
    assert not _actions_sim.guard("steps.release.outputs.release_created == 'true'", contexts, failed=True)
    assert _actions_sim.guard(
        "${{ !cancelled() && steps.release.outputs.release_created == 'true' }}", contexts, failed=True
    )
    assert not _actions_sim.guard(
        "${{ !cancelled() && steps.release.outputs.release_created == 'true' }}", {}, failed=True
    )
