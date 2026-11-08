"""Locate repository-only files: required in a checkout, skipped in the sdist.

The sdist ships ``tests/`` (see MANIFEST.in) but not ``examples/``, ``docs/``,
``scripts/``, ``.github/``, ``uv.lock`` or ``.pre-commit-config.yaml``. A test
that reads one of those calls :func:`repo_file`, which returns the path when it
exists. A missing path follows one rule for every caller: an extracted sdist,
which carries ``PKG-INFO`` at its root, skips with a reason naming the file; a
repository checkout fails, so a renamed or deleted file cannot turn its tests
into silent skips. It checks the path itself, never an import of a generic
name such as ``scripts`` that some other installed package could provide.

Call it inside a test to skip or fail that test, or at module level to skip
the whole module or fail its collection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def in_sdist() -> bool:
    """Whether ``ROOT`` is an extracted sdist: an sdist carries ``PKG-INFO``
    at its root, and a repository checkout does not."""
    return (ROOT / "PKG-INFO").is_file()


def repo_file(*parts: str) -> Path:
    """Return ``ROOT / *parts``; when that file or directory is absent, skip
    in an extracted sdist and fail in a repository checkout."""
    path = ROOT.joinpath(*parts)
    if not path.exists():
        relative = "/".join(parts)
        if in_sdist():
            pytest.skip(f"needs the repository's {relative} (not shipped in the sdist)", allow_module_level=True)
        pytest.fail(
            f"the repository checkout has no {relative}: a test reads this repository-only file "
            "(renamed or deleted?); only an extracted sdist, with PKG-INFO at its root, skips it",
            pytrace=False,
        )
    return path
