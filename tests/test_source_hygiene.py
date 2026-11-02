"""#575: runtime strings cite no internal tracking id.

A ``FEAT-nnn`` or ``FIX-nnn`` id is a maintainer's cross-reference. Code
comments and private docstrings may cite one, but a string a user reads at
runtime (an error message, a warning, an ``--help`` line) names the problem
and the remedy instead. The scan parses every module and checks each ``str``
constant that is not a docstring; comments are not AST nodes, so the
cross-reference comments stay out of it."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import nnx
from tests._repo_files import repo_file

_TRACKING_ID = re.compile(r"(?:FEAT|FIX)-\d+")


def _runtime_strings_citing_an_id(root: Path) -> list[str]:
    """``path:line: id`` for each non-docstring ``str`` constant under ``root``
    that cites a tracking id. A string that is a whole expression statement
    (a module, class, function or attribute docstring) is never a runtime
    value, so it is skipped."""
    found = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        docstrings = {
            id(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                for match in _TRACKING_ID.findall(node.value):
                    found.append(f"{path.relative_to(root.parent)}:{node.lineno}: {match}")
    return found


def test_package_runtime_strings_cite_no_tracking_id():
    root = Path(nnx.__file__).parent
    assert len(list(root.rglob("*.py"))) > 100, f"package scan looks broken under {root}"
    assert _runtime_strings_citing_an_id(root) == []


def test_example_runtime_strings_cite_no_tracking_id():
    root = repo_file("examples")
    assert len(list(root.glob("*.py"))) > 20, f"examples scan looks broken under {root}"
    assert _runtime_strings_citing_an_id(root) == []
