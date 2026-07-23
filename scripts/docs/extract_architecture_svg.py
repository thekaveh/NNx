"""Derive the published architecture SVG from its standalone HTML master."""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "docs" / "architecture.html"
OUTPUT = ROOT / "docs" / "assets" / "architecture.svg"
SVG_RE = re.compile(r"(?s)(<svg\b.*?</svg>)")


def render() -> str:
    match = SVG_RE.search(MASTER.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"no inline SVG found in {MASTER}")
    return textwrap.dedent(match.group(1)).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"{OUTPUT} is stale; run python -m scripts.docs.extract_architecture_svg")
        return
    OUTPUT.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
