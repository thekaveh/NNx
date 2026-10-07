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

from scripts.release.finalize_changelog import finalize

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "release_finalize"
# release-please's generic updater: the semantic version on every line that
# carries the marker.
_MARKED_VERSION = re.compile(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?")


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


def test_finalizing_is_idempotent_and_leaves_a_develop_changelog_alone():
    after = (FIXTURES / "after.md").read_text(encoding="utf-8")
    assert finalize(after) == after
    develop = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")  # [Unreleased] first: nothing to finalize
    assert finalize(develop) == develop


def test_without_curated_notes_the_generated_list_stays():
    text = (
        "# 22. Changelog\n\n## [0.5.0](https://example/compare) (2026-11-01)\n\n\n### Features\n\n* **x:** y\n\n"
        "## [Unreleased]\n\n## [0.4.0](https://example/compare) (2026-10-06)\n"
    )
    assert finalize(text) == (
        "# 22. Changelog\n\n## [Unreleased]\n\n## [0.5.0](https://example/compare) (2026-11-01)\n\n\n"
        "### Features\n\n* **x:** y\n\n## [0.4.0](https://example/compare) (2026-10-06)\n"
    )


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


def test_the_release_workflow_finalizes_the_release_branch():
    workflow = (ROOT / ".github" / "workflows" / "release-please.yml").read_text(encoding="utf-8")
    assert "python scripts/release/finalize_changelog.py" in workflow
    assert "git add uv.lock CHANGELOG.md" in workflow
