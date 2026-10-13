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
import re
from pathlib import Path

import pytest

# Repository-only: the sdist ships tests/ but not scripts/, examples/ or docs/,
# so a packager running its suite skips this module instead of failing collection.
pytest.importorskip("scripts", reason="needs the repository's scripts/ (not shipped in the sdist)")

from scripts.release.finalize_changelog import finalize, release_notes

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
    approval, verified or not (#475), so the workflow commits nothing; the
    release notes are folded from the release commit at publish time."""
    workflow = (ROOT / ".github" / "workflows" / "release-please.yml").read_text(encoding="utf-8")
    assert '--release-notes "$RELEASE_TAG"' in workflow and "gh release edit" in workflow
    for push in ("git push", "git commit", "api_commit", "git/commits", "uv lock"):
        assert push not in workflow, push
    assert 'gh workflow run ci.yml --repo "$GITHUB_REPOSITORY" --ref "$branch"' in workflow


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
