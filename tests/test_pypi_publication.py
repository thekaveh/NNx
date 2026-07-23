"""Tests that the installed package + PyPI publication are in sync.

Four layers:

1. Local version consistency: pyproject.toml [project] version equals
   nnx.__version__.
2. The editable root package in uv.lock carries that same version.
3. importlib.metadata lookup uses the current distribution name (catches
   the rename regression where src/nnx/__init__.py's _version() argument
   stops matching pyproject's [project] name).
4. PyPI availability (network-gated, skip-on-404): the distribution exists
   on PyPI under the current name.

The post-publish CI smoke job in .github/workflows/release.yml asserts a
stricter contract — that the exact-just-tagged version lands on PyPI
within the propagation window. That's the right place for the strict
assertion; this pytest module is the local feedback loop.

Skip controls:
    NNX_SKIP_PYPI_TESTS=1   force-skip the network test (offline contrib)
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import pathlib
import re
import urllib.error
import urllib.request

import pytest

import nnx

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

_PYPROJECT = pathlib.Path(__file__).parent.parent / "pyproject.toml"
_LOCKFILE = _PYPROJECT.parent / "uv.lock"
_RELEASE_WORKFLOW = _PYPROJECT.parent / ".github" / "workflows" / "release.yml"
_WORKFLOW_DIRECTORY = _RELEASE_WORKFLOW.parent
_RELEASE_PLEASE_WORKFLOW = _PYPROJECT.parent / ".github" / "workflows" / "release-please.yml"
_RELEASE_PLEASE_CONFIG = _PYPROJECT.parent / "release-please-config.json"
_PYPI_README = _PYPROJECT.parent / "PYPI_README.md"
_PYPI_TIMEOUT_SEC = 5.0
_STUDIO_REQUIRED_APIS = ("NNMoEParams", "NNConvParams", "FeedFwdMoENN", "ConvNN")


def test_draft_release_please_forces_tag_creation():
    config = json.loads(_RELEASE_PLEASE_CONFIG.read_text(encoding="utf-8"))
    assert config["draft"] is True
    assert config["force-tag-creation"] is True


def test_privileged_release_please_job_installs_only_pinned_uv_action():
    workflow = _RELEASE_PLEASE_WORKFLOW.read_text(encoding="utf-8")
    assert "astral-sh/setup-uv@37802adc94f370d6bfd71619e3f0bf239e1f3b78" in workflow
    assert "pip install -r requirements-tools.txt" not in workflow


def test_all_workflows_use_pinned_uv_action_and_no_unlocked_tool_install():
    workflows = list(_WORKFLOW_DIRECTORY.glob("*.yml"))
    assert workflows
    for path in workflows:
        workflow = path.read_text(encoding="utf-8")
        assert "pip install -r requirements-tools.txt" not in workflow, path
        if "uv " in workflow:
            assert "astral-sh/setup-uv@37802adc94f370d6bfd71619e3f0bf239e1f3b78" in workflow, path


def test_release_build_and_audit_tools_are_resolved_from_uv_lock():
    workflow = _RELEASE_WORKFLOW.read_text(encoding="utf-8")
    pyproject = _PYPROJECT.read_text(encoding="utf-8")

    assert "uv run --frozen --no-sync twine check" in workflow
    assert "uv run --frozen python -m pip_audit" in workflow
    assert '"pip-audit==2.10.1"' in pyproject
    assert '"twine==6.2.0"' in pyproject
    assert workflow.count("uv build --no-build-isolation --python .venv/bin/python") == 2
    assert "uv export --frozen --all-extras --all-groups" in workflow
    assert "uv sync --frozen --only-group release --no-install-project --no-build" in workflow
    assert "uv pip sync --python" in workflow and "--require-hashes smoke-requirements.txt" in workflow
    assert "uv pip install --python" in workflow and "--no-deps --no-build-isolation" in workflow


def test_package_uses_fresh_absolute_link_pypi_readme():
    from scripts.docs.build_pypi_readme import render

    pyproject = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    assert pyproject["project"]["readme"] == "PYPI_README.md"
    text = _PYPI_README.read_text(encoding="utf-8")
    assert text == render()
    assert "https://raw.githubusercontent.com/thekaveh/NNx/main/" in text
    assert "https://github.com/thekaveh/NNx/blob/main/" in text


def test_privileged_wiki_publication_uses_locked_nonbuilding_tool_group():
    workflow = (_WORKFLOW_DIRECTORY / "docs.yml").read_text(encoding="utf-8")

    assert "uv sync --frozen --only-group docs-publish --no-install-project --no-build" in workflow
    assert workflow.count("uv run --frozen --no-sync python -m scripts.docs") >= 2


def _read_pyproject_field(field: str) -> str:
    """Grep a top-level [project] string field from pyproject.toml.

    Hand-rolled instead of importing tomllib to keep this test cheap on
    Python 3.10 (tomllib is stdlib only on 3.11+).
    """
    text = _PYPROJECT.read_text(encoding="utf-8")
    match = re.search(rf'^{re.escape(field)}\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if not match:
        raise AssertionError(f"pyproject.toml: [project] {field!r} field not found")
    return match.group(1)


def test_pyproject_version_matches_dunder_version():
    """pyproject.toml [project] version == nnx.__version__.

    Mismatch usually means stale egg-info (re-run `pip install -e .`) OR
    src/nnx/__init__.py is looking up the wrong distribution name via
    importlib.metadata.version() and falling through to the
    `0.1.0+local` fallback string. After a distribution-name rename,
    BOTH the [project] name in pyproject.toml AND the lookup key in
    src/nnx/__init__.py must move in lock-step.
    """
    pyproject_version = _read_pyproject_field("version")
    assert nnx.__version__ == pyproject_version, (
        f"nnx.__version__ ({nnx.__version__!r}) doesn't match "
        f"pyproject.toml version ({pyproject_version!r}). Either the venv "
        f"has stale egg-info (re-run `pip install -e .`) or "
        f"src/nnx/__init__.py is looking up the wrong distribution name."
    )


def test_uv_lock_root_package_version_matches_pyproject():
    """Release version bumps must refresh the editable root lock entry."""
    locked = tomllib.loads(_LOCKFILE.read_text(encoding="utf-8"))
    dist_name = _read_pyproject_field("name")
    root_packages = [
        package
        for package in locked["package"]
        if package["name"] == dist_name and package["source"] == {"editable": "."}
    ]

    assert len(root_packages) == 1
    assert root_packages[0]["version"] == _read_pyproject_field("version")


def test_importlib_metadata_lookup_uses_current_dist_name():
    """importlib.metadata.version(<pyproject [project] name>) resolves
    and equals nnx.__version__.

    Regression test: the PR #47 + PR #49 distribution-name renames each
    left src/nnx/__init__.py's _version() argument pointing at the wrong
    name (originally "nnx", then "nnx-pytorch", now "thekaveh-nnx").
    This test ensures the lookup key and the pyproject [project] name
    field stay in lock-step on future renames.
    """
    dist_name = _read_pyproject_field("name")
    try:
        from_metadata = importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError as e:
        raise AssertionError(
            f"importlib.metadata.version({dist_name!r}) raised "
            f"PackageNotFoundError. The venv likely has stale egg-info "
            f"from an earlier install under a different distribution name. "
            f"Run `pip install -e .` to refresh and re-run."
        ) from e
    assert from_metadata == nnx.__version__, (
        f"importlib.metadata.version({dist_name!r}) returned "
        f"{from_metadata!r}, but nnx.__version__ is {nnx.__version__!r}. "
        f"The lookup is hitting a different distribution than pyproject "
        f"claims."
    )


def test_post_publish_smoke_imports_studio_required_apis():
    """The clean-environment PyPI smoke must cover Studio's minimum API set."""
    workflow = _RELEASE_WORKFLOW.read_text(encoding="utf-8")

    for symbol in _STUDIO_REQUIRED_APIS:
        assert f"from nnx import {symbol}" in workflow, (
            f"release.yml post-publish smoke does not import {symbol}; "
            "a release could satisfy the version check while missing Studio's required API"
        )


def test_reusable_release_inputs_enable_the_publish_path():
    """Release Please calls must publish even though their event remains ``push``."""
    workflow = _RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "github.event_name == 'workflow_call'" not in workflow
    assert workflow.count("inputs.tag_name != ''") == 4, (
        "release.yml must use the supplied tag_name input to enable tag/version "
        "validation, PyPI publication, post-publish verification, and GitHub release publication"
    )


def test_release_publication_has_one_managed_entry_point():
    workflow = _RELEASE_WORKFLOW.read_text(encoding="utf-8")
    release_please = _RELEASE_PLEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "tags:" not in workflow
    assert "startsWith(github.ref, 'refs/tags/v')" not in workflow
    assert "actions: write" in release_please
    assert "gh workflow run ci.yml" in release_please
    assert "gh workflow run security.yml" in release_please
    assert "Verify tag still identifies the release commit" in workflow
    assert "gh release upload" in workflow
    assert "actions/download-artifact@" in workflow
    assert "overwrite: true" in workflow
    assert "skip-existing: true" in workflow
    assert "Verify PyPI artifact hashes" in workflow
    assert "published == local" in workflow
    assert "gh release verify" in workflow
    assert "already_published=true" in workflow
    assert "Verify exact GitHub release asset hashes" in workflow
    assert "remote == local" in workflow
    assert "Verify new immutable release attestation" in workflow
    assert "uv lock" in release_please
    assert "chore: refresh release lockfile" in release_please


def test_pypi_lists_the_current_distribution_name():
    """The distribution exists on PyPI under the current name.

    Skipped when:
      - ``NNX_SKIP_PYPI_TESTS=1`` is set
      - PyPI returns 404 (package never published — expected pre-release)
      - network is unreachable (timeout / DNS)

    A 200 response with a name mismatch is a genuine failure (we'd be
    looking at the wrong project on PyPI).
    """
    if os.environ.get("NNX_SKIP_PYPI_TESTS") == "1":
        pytest.skip("NNX_SKIP_PYPI_TESTS=1 set")

    dist_name = _read_pyproject_field("name")
    url = f"https://pypi.org/pypi/{dist_name}/json"

    try:
        with urllib.request.urlopen(url, timeout=_PYPI_TIMEOUT_SEC) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            pytest.skip(
                f"{dist_name!r} not yet published to PyPI (404 at {url}). "
                f"This test will start passing once the first release "
                f"lands via .github/workflows/release.yml."
            )
        raise
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        pytest.skip(f"network unreachable for PyPI verification: {e}")

    canonical_name = payload["info"]["name"]
    assert canonical_name.lower() == dist_name.lower(), (
        f"PyPI returned project {canonical_name!r} for the URL {url}, "
        f"but pyproject [project] name is {dist_name!r}. URL was built "
        f"correctly but PyPI is returning a different project."
    )
