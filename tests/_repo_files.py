"""Locate repository-only files, skipping where the sdist does not ship them.

The sdist ships ``tests/`` (see MANIFEST.in) but not ``examples/``, ``docs/``,
``scripts/``, ``.github/``, ``uv.lock`` or ``.pre-commit-config.yaml``. A test
that reads one of those calls :func:`repo_file`: in a repository checkout it
returns the path, and in an extracted sdist it skips with a reason naming the
missing file. It checks the path itself, never an import of a generic name
such as ``scripts`` that some other installed package could provide.

Call it inside a test to skip that test, or at module level to skip the
whole module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def repo_file(*parts: str) -> Path:
    """Return ``ROOT / *parts``, or skip when that file or directory is absent."""
    path = ROOT.joinpath(*parts)
    if not path.exists():
        relative = "/".join(parts)
        pytest.skip(f"needs the repository's {relative} (not shipped in the sdist)", allow_module_level=True)
    return path
