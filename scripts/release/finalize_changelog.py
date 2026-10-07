"""Finalize a release-please branch's CHANGELOG (FIX-029).

release-please opens its release PR with a generated commit list under the
new version heading, above the curated ``## [Unreleased]`` notes. NNx
publishes the curated notes: this moves them under the version heading in
place of the generated list and leaves ``## [Unreleased]`` empty above it —
the finalize that used to be a hand-made commit on every release PR (v0.2.3,
v0.3.0, v0.4.0). Without curated notes the generated list stays. A
changelog whose first section is already ``[Unreleased]`` (develop, or a
release branch finalized before) is left unchanged, so running it again is a
no-op.

Usage::

    python3 scripts/release/finalize_changelog.py [CHANGELOG.md] [--check | --release-notes VERSION]

``--check`` writes nothing and exits 1 when the file would change;
``--release-notes VERSION`` prints that version's section (the GitHub
release body the workflow publishes on release).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

UNRELEASED = "## [Unreleased]"


def _strip_blank(lines: list[str]) -> list[str]:
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def _headings(lines: list[str]) -> list[int]:
    """Indices of ``## `` section headings, outside fenced code blocks."""
    found, fenced = [], False
    for index, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and line.startswith("## "):
            found.append(index)
    return found


def finalize(text: str) -> str:
    """The changelog with the curated ``[Unreleased]`` notes moved under the
    release heading release-please put above them (see the module docstring)."""
    lines = text.split("\n")
    headings = _headings(lines)
    unreleased = next((i for i in headings if lines[i].rstrip() == UNRELEASED), None)
    if unreleased is None:
        raise ValueError(f"the changelog has no {UNRELEASED!r} section")
    if headings[0] == unreleased:  # nothing above the curated notes: nothing to finalize
        return text
    release = headings[0]
    above = [lines[i] for i in headings if release < i < unreleased]
    if above:
        raise ValueError(
            f"more than one release section lies above {UNRELEASED!r} ({lines[release]!r}, {above[0]!r}, ...): "
            "a release merged without its finalize — fix the changelog by hand rather than drop a section"
        )
    following = next((i for i in headings if i > unreleased), len(lines))
    curated = _strip_blank(lines[unreleased + 1 : following])
    if curated:
        body = [lines[release], "", *curated, ""]
    else:  # nothing curated: keep release-please's list
        body = lines[release:unreleased]
    return "\n".join([*lines[:release], UNRELEASED, "", *body, *lines[following:]])


def release_notes(text: str, version: str) -> str:
    """The body of the ``## [<version>]`` section — the curated notes the
    GitHub release publishes (release-please would use its generated list)."""
    lines = text.split("\n")
    headings = _headings(lines)
    start = next((i for i in headings if lines[i].startswith(f"## [{version}]")), None)
    if start is None:
        raise ValueError(f"the changelog has no section for {version}")
    end = next((i for i in headings if i > start), len(lines))
    return "\n".join(_strip_blank(lines[start + 1 : end])) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("path", nargs="?", default="CHANGELOG.md")
    parser.add_argument("--check", action="store_true", help="exit 1 when the file would change; write nothing")
    parser.add_argument("--release-notes", metavar="VERSION", help="print that version's section and exit")
    args = parser.parse_args(argv)
    path = Path(args.path)
    text = path.read_text(encoding="utf-8")
    if args.release_notes:
        sys.stdout.write(release_notes(text, args.release_notes.removeprefix("v")))
        return 0
    finalized = finalize(text)
    if finalized == text:
        print(f"{path}: already final")
        return 0
    if args.check:
        print(f"{path}: the curated [Unreleased] notes are not yet under the release heading")
        return 1
    path.write_text(finalized, encoding="utf-8")
    print(f"{path}: finalized")
    return 0


if __name__ == "__main__":
    sys.exit(main())
