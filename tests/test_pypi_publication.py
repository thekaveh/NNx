"""Tests that the installed package + PyPI publication are in sync.

Three layers:

1. Local version consistency: pyproject.toml [project] version equals
   nnx.__version__.
2. importlib.metadata lookup uses the current distribution name (catches
   the rename regression where src/nnx/__init__.py's _version() argument
   stops matching pyproject's [project] name).
3. PyPI availability (network-gated, skip-on-404): the distribution exists
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

_PYPROJECT = pathlib.Path(__file__).parent.parent / "pyproject.toml"
_PYPI_TIMEOUT_SEC = 5.0


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
