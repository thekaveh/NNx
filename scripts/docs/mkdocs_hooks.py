"""Project canonical root documentation into the MkDocs source tree."""

from __future__ import annotations

import shutil
from pathlib import Path

from build_wiki import EXTRA_PAGES, ROOT, _nav_pages, rewrite_markdown

OUTPUT = ROOT / "docs" / "_project"


def _source_map() -> dict[Path, str]:
    mapping = {source.resolve(): f"{slug}.md" for _, source, slug in EXTRA_PAGES}
    for _, source, _ in _nav_pages():
        mapping[source.resolve()] = f"../{source.relative_to(ROOT / 'docs')}"
    mapping[(ROOT / "docs" / "architecture.html").resolve()] = "../architecture.md"
    mapping[(ROOT / "docs" / "assets" / "architecture.svg").resolve()] = "../assets/architecture.svg"
    mapping[(ROOT / "examples").resolve()] = "Examples.md"
    mapping[(ROOT / "tests").resolve()] = "Test-Import-Boundaries.md"
    for example in (ROOT / "examples").glob("*.py"):
        mapping[example.resolve()] = "Examples.md"
    mapping[(ROOT / "src" / "nnx" / "_step_helpers.py").resolve()] = "../api.md"
    return mapping


def build_site_pages() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    mapping = _source_map()
    for _, source, slug in EXTRA_PAGES:
        text = source.read_text(encoding="utf-8")
        if source.name == "LICENSE":
            text = f"# License\n\n```text\n{text.rstrip()}\n```\n"
        (OUTPUT / f"{slug}.md").write_text(rewrite_markdown(source, text, mapping), encoding="utf-8")


def on_config(config):  # noqa: ANN001 - MkDocs hook protocol
    build_site_pages()
    return config
