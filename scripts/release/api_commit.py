"""Commit files to a branch through the GitHub API (FIX-033).

The release workflow's own commit on a release-please branch (the lockfile
refresh and the CHANGELOG finalize) is made here rather than with
``git push``: a commit created through the REST API with ``GITHUB_TOKEN`` and
no author or committer is signed by GitHub — *verified*, like release-please's
own — rather than an unverified pushed commit, which the ``main`` ruleset's
``require_extra_approval_for_unattributed_changes`` held for an approving
review on release PR #399.

It creates a blob per file, a tree over the branch head's (every other file
unchanged), a commit whose only parent is that head, then fast-forwards the
branch ref to it: a branch that moved meanwhile is refused, never overwritten.

Usage::

    python3 scripts/release/api_commit.py --repo OWNER/REPO --branch BRANCH \\
        --parent SHA --message MESSAGE FILE [FILE ...]

Prints the new commit's sha. Requires ``gh`` authenticated with a token that
can write the repository's contents (``GH_TOKEN``).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional

GitHub = Callable[..., Any]


def _gh(args: list[str], body: Optional[dict[str, Any]] = None) -> Any:
    """``gh <args>``, with ``body`` as the JSON request body on stdin."""
    completed = subprocess.run(
        ["gh", *args],
        input=None if body is None else json.dumps(body),
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {completed.stderr.strip()}")
    return json.loads(completed.stdout)


def commit_files(
    repo: str,
    branch: str,
    parent: str,
    paths: Sequence[str],
    message: str,
    *,
    root: Path = Path("."),
    gh: GitHub = _gh,
) -> str:
    """Commit ``paths`` (relative to ``root``, as they are on disk) onto
    ``parent``, the head of ``branch``, and fast-forward the branch to it.
    Returns the new commit's sha."""
    entries = []
    for path in paths:
        local = root / path
        content = base64.b64encode(local.read_bytes()).decode("ascii")
        blob = gh(["api", f"repos/{repo}/git/blobs", "--input", "-"], {"content": content, "encoding": "base64"})
        mode = "100755" if os.access(local, os.X_OK) else "100644"
        entries.append({"path": path, "mode": mode, "type": "blob", "sha": blob["sha"]})
    base_tree = gh(["api", f"repos/{repo}/git/commits/{parent}"])["tree"]["sha"]
    tree = gh(["api", f"repos/{repo}/git/trees", "--input", "-"], {"base_tree": base_tree, "tree": entries})
    # No author or committer: GitHub signs the commit as the token's app.
    commit = gh(
        ["api", f"repos/{repo}/git/commits", "--input", "-"],
        {"message": message, "tree": tree["sha"], "parents": [parent]},
    )
    gh(
        ["api", "--method", "PATCH", f"repos/{repo}/git/refs/heads/{branch}", "--input", "-"],
        {"sha": commit["sha"], "force": False},
    )
    return commit["sha"]


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Commit files to a branch through the GitHub API (FIX-033).")
    parser.add_argument("--repo", required=True, help="OWNER/REPO")
    parser.add_argument("--branch", required=True, help="the branch to fast-forward")
    parser.add_argument("--parent", required=True, help="the branch head the commit goes on")
    parser.add_argument("--message", required=True)
    parser.add_argument("paths", nargs="+", help="files to commit, relative to the repository root")
    args = parser.parse_args(argv)
    print(commit_files(args.repo, args.branch, args.parent, args.paths, args.message, gh=_gh))


if __name__ == "__main__":
    main()
