"""Build the package long description with absolute repository links."""

from __future__ import annotations

import argparse
from pathlib import Path

from .build_wiki import ROOT, rewrite_markdown

OUTPUT = ROOT / "PYPI_README.md"
REPOSITORY_BLOB = "https://github.com/thekaveh/NNx/blob/main/"
REPOSITORY_RAW = "https://raw.githubusercontent.com/thekaveh/NNx/main/"


def render() -> str:
    source = ROOT / "README.md"
    source_map: dict[Path, str] = {}
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or ".venv" in path.parts or "generated" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.is_dir():
            source_map[path.resolve()] = "https://github.com/thekaveh/NNx/tree/main/" + relative
            continue
        prefix = (
            REPOSITORY_RAW
            if path.suffix.lower() in {".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"}
            else REPOSITORY_BLOB
        )
        source_map[path.resolve()] = prefix + relative
    return rewrite_markdown(source, source.read_text(encoding="utf-8"), source_map, surface="site")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != rendered:
            raise SystemExit("PYPI_README.md is stale; run python -m scripts.docs.build_pypi_readme")
        return
    OUTPUT.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
