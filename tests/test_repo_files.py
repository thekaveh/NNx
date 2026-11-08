"""#571: ``repo_file`` skips a missing repository-only file only in an
extracted sdist; in a checkout the test that reads it fails."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests import _repo_files
from tests._repo_files import repo_file

_HELPER = Path(_repo_files.__file__)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A stand-in repository root holding one repository-only script."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "gate.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(_repo_files, "ROOT", tmp_path)
    return tmp_path


def test_a_present_file_is_returned_in_either_layout(tree):
    assert repo_file("scripts", "gate.py") == tree / "scripts" / "gate.py"
    (tree / "PKG-INFO").write_text("Name: thekaveh-nnx\n", encoding="utf-8")
    assert repo_file("scripts") == tree / "scripts"


def _outcome(*parts: str) -> tuple[str, str]:
    """How ``repo_file(*parts)`` ends: ``("failed" | "skipped", message)``.
    Both outcomes are caught, so a skip where a failure is due fails the
    calling test instead of skipping it."""
    try:
        repo_file(*parts)
    except pytest.fail.Exception as failed:
        return "failed", failed.msg
    except pytest.skip.Exception as skipped:
        return "skipped", skipped.msg
    raise AssertionError(f"repo_file{parts} returned for a missing path")


def test_a_missing_file_fails_in_a_checkout(tree):
    kind, message = _outcome("scripts", "renamed.py")
    assert kind == "failed" and message.startswith("the repository checkout has no scripts/renamed.py: ")
    kind, message = _outcome("examples")  # a whole directory follows the same rule
    assert kind == "failed" and message.startswith("the repository checkout has no examples: ")


def test_a_missing_file_skips_in_an_extracted_sdist(tree):
    (tree / "PKG-INFO").write_text("Name: thekaveh-nnx\n", encoding="utf-8")
    assert _outcome("examples") == ("skipped", "needs the repository's examples (not shipped in the sdist)")
    assert _outcome("scripts", "renamed.py") == (
        "skipped",
        "needs the repository's scripts/renamed.py (not shipped in the sdist)",
    )


def test_this_checkout_is_not_taken_for_an_sdist():
    if not (_repo_files.ROOT / "pyproject.toml").is_file() or (_repo_files.ROOT / "PKG-INFO").exists():
        pytest.skip("runs from a repository checkout")
    assert _repo_files.in_sdist() is False


@pytest.mark.parametrize("sdist", [False, True], ids=["checkout", "sdist"])
def test_a_module_level_guard_fails_collection_in_a_checkout_and_skips_in_an_sdist(tmp_path, sdist):
    """The whole-suite effect of a renamed script: pytest exits non-zero in a
    checkout and 0, with the sdist reason, in an extracted sdist."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy(_HELPER, tmp_path / "tests" / "_repo_files.py")
    (tmp_path / "tests" / "test_gate.py").write_text(
        'from tests._repo_files import repo_file\n\nrepo_file("scripts", "check_gate.py")\n\n\n'
        "def test_gate():\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_core.py").write_text("def test_core():\n    pass\n", encoding="utf-8")
    if sdist:
        (tmp_path / "PKG-INFO").write_text("Name: thekaveh-nnx\n", encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", "-o", "addopts=", "tests"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = done.stdout + done.stderr
    if sdist:
        assert done.returncode == 0, output
        assert "needs the repository's scripts/check_gate.py (not shipped in the sdist)" in output
    else:
        assert done.returncode != 0, output
        assert "ERROR collecting tests/test_gate.py" in output and "has no scripts/check_gate.py" in output


def test_every_repository_only_guard_goes_through_repo_file():
    """One checkout and sdist rule: no test skips inline for a repository
    path the sdist omits (an ``exists()`` check before ``pytest.skip``, or an
    import of a generic name such as ``scripts``)."""
    import ast

    offenders = []
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        if path.name == "_repo_files.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or ast.unparse(node.func) not in ("pytest.skip", "pytest.importorskip"):
                continue
            words = [arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
            if any("sdist" in word or word in ("scripts", "examples") for word in words):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []
