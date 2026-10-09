"""FIX-033: the release workflow commits through the GitHub API.

The workflow's own commit on a release-please branch (the lockfile refresh and
the CHANGELOG finalize) used to be made with ``git push``: GitHub marked it
*unverified*, and the ``main`` ruleset's
``require_extra_approval_for_unattributed_changes`` then held every release PR
for an approving review (#399). A commit created through the REST API with
``GITHUB_TOKEN`` and no author or committer is signed by GitHub, like
release-please's own, so a green release PR merges without one."""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.release.api_commit import commit_files

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "release-please.yml"


def _dispatch_step() -> str:
    steps = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["release-please"]["steps"]
    (step,) = [step for step in steps if step.get("name") == "Dispatch required checks for managed release PRs"]
    return step["run"]


def test_the_workflow_commits_through_the_api_and_never_pushes_the_release_branch():
    run = _dispatch_step()
    assert "git push" not in run and "git commit" not in run
    assert "python3 scripts/release/api_commit.py" in run
    # still skipped when nothing changed, and the checks dispatched after the new head exists
    assert run.index("git diff --quiet -- uv.lock CHANGELOG.md") < run.index("scripts/release/api_commit.py")
    assert run.index("scripts/release/api_commit.py") < run.index("gh workflow run ci.yml")
    assert run.index("scripts/release/api_commit.py") < run.index("gh workflow run security.yml")
    assert '--ref "$branch"' in run


class FakeGitHub:
    """``gh api`` as the Git database endpoints answer it."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict | None]] = []

    def __call__(self, args: list[str], body: dict | None = None) -> dict:
        self.calls.append((args, body))
        endpoint = args[-1] if body is None else args[args.index("--input") - 1]
        if endpoint.endswith("/git/commits/parent-sha"):
            return {"sha": "parent-sha", "tree": {"sha": "parent-tree"}}
        if endpoint.endswith("/git/blobs"):
            return {"sha": f"blob-{len(self.calls)}"}
        if endpoint.endswith("/git/trees"):
            return {"sha": "new-tree"}
        if endpoint.endswith("/git/commits"):
            return {"sha": "new-commit", "verification": {"verified": True}}
        if "/git/refs/heads/" in endpoint:
            return {"object": {"sha": body["sha"]}}
        raise AssertionError(f"unexpected call {args}")


def test_the_commit_is_built_from_blobs_a_tree_on_the_branch_head_and_a_fast_forward(tmp_path):
    (tmp_path / "uv.lock").write_bytes(b"lock \x00 bytes\n" * 50_000)  # larger than one argv entry may be
    (tmp_path / "CHANGELOG.md").write_text("# notes\n", encoding="utf-8")
    gh = FakeGitHub()
    sha = commit_files(
        "owner/repo",
        "release-please--branches--main",
        "parent-sha",
        ["uv.lock", "CHANGELOG.md"],
        "chore: refresh release lockfile and finalize the release notes",
        root=tmp_path,
        gh=gh,
    )
    assert sha == "new-commit"
    blobs = [body for args, body in gh.calls if args[-3:] == ["repos/owner/repo/git/blobs", "--input", "-"]]
    assert [base64.b64decode(blob["content"]) for blob in blobs] == [
        (tmp_path / "uv.lock").read_bytes(),
        (tmp_path / "CHANGELOG.md").read_bytes(),
    ]
    assert all(blob["encoding"] == "base64" for blob in blobs)
    (tree,) = [body for args, body in gh.calls if "repos/owner/repo/git/trees" in args]
    assert tree["base_tree"] == "parent-tree"  # every other file as on the branch head
    assert [(entry["path"], entry["mode"], entry["type"]) for entry in tree["tree"]] == [
        ("uv.lock", "100644", "blob"),
        ("CHANGELOG.md", "100644", "blob"),
    ]
    (commit,) = [body for args, body in gh.calls if "repos/owner/repo/git/commits" in args]
    assert commit == {
        "message": "chore: refresh release lockfile and finalize the release notes",
        "tree": "new-tree",
        "parents": ["parent-sha"],
    }  # no author or committer: GitHub signs it as the token's app
    args, ref = gh.calls[-1]
    assert args[:3] == ["api", "--method", "PATCH"]
    assert "repos/owner/repo/git/refs/heads/release-please--branches--main" in args
    assert ref == {"sha": "new-commit", "force": False}  # a fast-forward only: a moved branch is refused


def test_gh_receives_each_body_on_stdin(tmp_path, monkeypatch):
    from scripts.release import api_commit

    seen = []

    def fake_run(args, *, input, capture_output, check, text):
        seen.append((args, json.loads(input) if input else None))
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps({"sha": "x", "tree": {"sha": "t"}}), stderr="")

    monkeypatch.setattr(api_commit.subprocess, "run", fake_run)
    assert api_commit._gh(["api", "repos/o/r/git/blobs", "--input", "-"], {"content": "", "encoding": "base64"}) == {
        "sha": "x",
        "tree": {"sha": "t"},
    }
    assert seen == [(["gh", "api", "repos/o/r/git/blobs", "--input", "-"], {"content": "", "encoding": "base64"})]


def test_the_cli_commits_the_named_files_and_prints_the_new_head(tmp_path, monkeypatch, capsys):
    from scripts.release import api_commit

    (tmp_path / "CHANGELOG.md").write_text("x\n", encoding="utf-8")
    gh = FakeGitHub()
    monkeypatch.setattr(api_commit, "_gh", gh)
    monkeypatch.chdir(tmp_path)
    api_commit.main(
        ["--repo", "owner/repo", "--branch", "b", "--parent", "parent-sha", "--message", "m", "CHANGELOG.md"]
    )
    assert capsys.readouterr().out.strip() == "new-commit"
    with pytest.raises(SystemExit):
        api_commit.main(["--repo", "owner/repo", "--branch", "b", "--parent", "parent-sha", "--message", "m"])
