"""Finalize the CHANGELOG after a release (FIX-029, FIX-033).

release-please's release commit puts a generated commit list under the new
version heading, above the curated ``## [Unreleased]`` notes. NNx publishes
the curated notes: this moves them under the version heading in place of the
generated list and leaves ``## [Unreleased]`` empty above it. Without curated
notes the generated list stays. A changelog whose first section is already
``[Unreleased]`` is left unchanged, so running it again is a no-op.

Nothing commits to a release PR but release-please itself (FIX-033), so the
release workflow publishes the folded notes (``--release-notes``) and the
post-release sync of ``main`` into ``develop`` runs this to fold them into
the file.

On ``develop`` the ``[Unreleased]`` notes can hold entries added after the
release branched. ``--released FILE`` names the changelog the release
commit shipped (``git show vX.Y.Z:CHANGELOG.md``): only its curated entries
move under the version heading, and the later ones stay in
``[Unreleased]`` unchanged. A shipped entry that this branch edited after the
release (an item appended to a grouped entry, say) is refused, to be fixed
by hand, rather than filed under the wrong version.

Usage::

    python3 scripts/release/finalize_changelog.py [CHANGELOG.md] [--released FILE] [--check | --release-notes VERSION]

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


def _unreleased(lines: list[str]) -> list[str]:
    """The curated lines of the ``[Unreleased]`` section, blank-trimmed."""
    headings = _headings(lines)
    unreleased = next((i for i in headings if lines[i].rstrip() == UNRELEASED), None)
    if unreleased is None:
        raise ValueError(f"the changelog has no {UNRELEASED!r} section")
    following = next((i for i in headings if i > unreleased), len(lines))
    return _strip_blank(lines[unreleased + 1 : following])


def _entries(lines: list[str]) -> list[tuple[str, str, list[str]]]:
    """``(subsection heading, key, raw lines)`` per top-level entry: a line
    that is not indented, with the indented (and blank) lines that follow
    it. The key ignores blank lines; the raw lines keep them, so a fenced
    block or a second paragraph is emitted unchanged."""
    entries: list[tuple[str, list[str]]] = []
    heading = ""
    for line in lines:
        if line.startswith("### "):
            heading = line.rstrip()
        elif line.strip() and not line[0].isspace():
            entries.append((heading, [line]))
        elif entries and (line.strip() or line == ""):
            entries[-1][1].append(line)
    out = []
    for head, body in entries:
        raw = _strip_blank(body)
        out.append((head, "\n".join(line for line in raw if line.strip()), raw))
    return out


def _split_unreleased(curated: list[str], published: list[str]) -> tuple[list[str], list[str]]:
    """``(released, still unreleased)``: the published entries, and develop's
    entries the release did not ship, under their subsection headings."""
    if curated == published:
        return published, []
    shipped = {key for _, key, _ in _entries(published)}
    present = {key for _, key, _ in _entries(curated)}
    edited = sorted(shipped - present)
    if edited:
        raise ValueError(
            f"{len(edited)} released [Unreleased] entries were edited on this branch after the release (first: "
            f"{edited[0].splitlines()[0]!r}), e.g. an item appended to a shipped grouped entry: fix the changelog "
            "by hand (start a new entry for later changes) rather than file it under the wrong version"
        )
    later: list[str] = []
    current = None
    for heading, key, raw in _entries(curated):
        if key in shipped:
            continue
        if heading and heading != current:
            later += ["", heading, ""] if later else [heading, ""]
            current = heading
        later.extend(raw)
    return published, later


def finalize(text: str, released: str | None = None) -> str:
    """The changelog with the curated ``[Unreleased]`` notes moved under the
    release heading release-please put above them (see the module docstring).
    With ``released`` (the changelog the release commit shipped), only its
    curated entries move; later ones stay in ``[Unreleased]``."""
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
    later: list[str] = []
    if released is not None:
        curated, later = _split_unreleased(curated, _unreleased(released.split("\n")))
    if curated:
        body = [lines[release], "", *curated, ""]
    else:  # nothing curated: keep release-please's list
        body = lines[release:unreleased]
    head = [UNRELEASED, "", *later, ""] if later else [UNRELEASED, ""]
    return "\n".join([*lines[:release], *head, *body, *lines[following:]])


def release_notes(text: str, version: str) -> str:
    """The body of the ``## [<version>]`` section once finalized — the curated
    notes the GitHub release publishes (release-please would use its generated
    list). The release commit is not finalized yet (FIX-033: no commit but
    release-please's own lands on a release PR), so the curated
    ``[Unreleased]`` notes are folded in here as :func:`finalize` would."""
    lines = finalize(text).split("\n")
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
    parser.add_argument(
        "--released",
        metavar="FILE",
        help="the changelog the release commit shipped: only its [Unreleased] entries are filed under the release",
    )
    args = parser.parse_args(argv)
    path = Path(args.path)
    text = path.read_text(encoding="utf-8")
    released = Path(args.released).read_text(encoding="utf-8") if args.released else None
    if args.release_notes:
        sys.stdout.write(release_notes(text, args.release_notes.removeprefix("v")))
        return 0
    finalized = finalize(text, released)
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
