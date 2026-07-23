"""Replace the GitHub wiki master branch with generated pages."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True)


def push(source: Path, repository: str) -> None:
    env_defaults = {
        "GIT_AUTHOR_NAME": "github-actions[bot]",
        "GIT_AUTHOR_EMAIL": "41898282+github-actions[bot]@users.noreply.github.com",
        "GIT_COMMITTER_NAME": "github-actions[bot]",
        "GIT_COMMITTER_EMAIL": "41898282+github-actions[bot]@users.noreply.github.com",
    }
    for key, value in env_defaults.items():
        os.environ.setdefault(key, value)

    with tempfile.TemporaryDirectory() as tmp:
        checkout = Path(tmp) / "wiki"
        _run("git", "clone", repository, str(checkout))
        if _run("git", "branch", "--show-current", cwd=checkout).stdout.strip() != "master":
            has_head = (
                subprocess.run(
                    ("git", "rev-parse", "--verify", "HEAD"),
                    cwd=checkout,
                    text=True,
                    capture_output=True,
                ).returncode
                == 0
            )
            if has_head:
                _run("git", "checkout", "master", cwd=checkout)
            else:
                _run("git", "checkout", "--orphan", "master", cwd=checkout)
        for path in checkout.iterdir():
            if path.name == ".git":
                continue
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        shutil.copytree(source, checkout, dirs_exist_ok=True)
        _run("git", "add", "--all", cwd=checkout)
        if not _run("git", "status", "--porcelain", cwd=checkout).stdout:
            return
        _run("git", "commit", "-m", "docs: sync wiki from canonical documentation", cwd=checkout)
        _run("git", "push", "origin", "HEAD:master", cwd=checkout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("repository")
    args = parser.parse_args()
    push(args.source, args.repository)


if __name__ == "__main__":
    main()
