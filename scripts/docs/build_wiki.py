"""Project canonical repository docs into a self-contained GitHub wiki."""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "generated" / "wiki"
GITHUB_BLOB_PREFIX = "https://github.com/thekaveh/NNx/blob/"

EXTRA_PAGES = (
    ("Repository Overview", ROOT / "README.md", "Repository-Overview"),
    ("Examples", ROOT / "examples" / "README.md", "Examples"),
    ("Contributing", ROOT / "CONTRIBUTING.md", "Contributing"),
    ("Changelog", ROOT / "CHANGELOG.md", "Changelog"),
    ("Test Import Boundaries", ROOT / "tests" / "README.md", "Test-Import-Boundaries"),
)

LINK_RE = re.compile(r"(?P<image>!)?\[(?P<label>[^]]*)\]\((?P<target>[^)]+)\)")
ATTR_RE = re.compile(r"\{target=\"_blank\" rel=\"noopener\"\}")


def _slug(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-")


def _nav_pages() -> list[tuple[str, Path, str]]:
    config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    pages: list[tuple[str, Path, str]] = []

    def visit(items: list[object]) -> None:
        for item in items:
            if not isinstance(item, dict):
                continue
            for title, value in item.items():
                if isinstance(value, list):
                    visit(value)
                elif isinstance(value, str):
                    slug = "Home" if value == "index.md" else _slug(str(title))
                    pages.append((str(title), ROOT / "docs" / value, slug))

    visit(config["nav"])
    return pages


def _split_target(target: str) -> tuple[str, str]:
    path, marker, fragment = target.partition("#")
    return path, f"#{fragment}" if marker else ""


def _resolve_repo_target(source: Path, target: str) -> Path | None:
    path, _ = _split_target(target)
    if target.startswith(GITHUB_BLOB_PREFIX):
        parts = urlparse(target).path.split("/")
        if len(parts) >= 6:
            return (ROOT / Path(*parts[5:])).resolve()
    if re.match(r"^[a-z]+://", path) or path.startswith(("mailto:", "/")):
        return None
    return (source.parent / path).resolve()


def rewrite_markdown(source: Path, text: str, source_map: dict[Path, str]) -> str:
    """Rewrite local Markdown links for a self-contained wiki page."""

    def replace(match: re.Match[str]) -> str:
        label = match.group("label")
        target = match.group("target").strip()
        is_image = bool(match.group("image"))
        path_part, fragment = _split_target(target)

        if path_part.startswith(("http://", "https://", "mailto:")) and not target.startswith(GITHUB_BLOB_PREFIX):
            return match.group(0)

        resolved = _resolve_repo_target(source, target)
        if resolved is not None and resolved in source_map:
            return f"[{label}]({source_map[resolved]}{fragment})"

        if is_image and resolved is not None and resolved.is_file():
            return f"![{label}](images/{resolved.name})"

        if path_part == "" and fragment:
            return f"[{label}]({fragment})"

        return label

    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        lines.append(line if in_fence else LINK_RE.sub(replace, ATTR_RE.sub("", line)))
    return "\n".join(lines).rstrip() + "\n"


def build(output: Path = DEFAULT_OUTPUT) -> None:
    pages = _nav_pages() + list(EXTRA_PAGES)
    source_map = {source.resolve(): f"{slug}.md" for _, source, slug in pages}

    if output.exists():
        shutil.rmtree(output)
    (output / "images").mkdir(parents=True)

    for _, source, slug in pages:
        rendered = rewrite_markdown(source, source.read_text(encoding="utf-8"), source_map)
        (output / f"{slug}.md").write_text(rendered, encoding="utf-8")

    shutil.copy2(ROOT / "docs" / "assets" / "architecture.svg", output / "images" / "architecture.svg")
    sidebar = "# NNx\n\n" + "\n".join(f"- [{title}]({slug})" for title, _, slug in pages) + "\n"
    (output / "_Sidebar.md").write_text(sidebar, encoding="utf-8")
    (output / "_Footer.md").write_text("Apache-2.0 licensed.\n", encoding="utf-8")


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def check() -> None:
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        first_path, second_path = Path(first), Path(second)
        build(first_path)
        build(second_path)
        if _snapshot(first_path) != _snapshot(second_path):
            raise SystemExit("wiki projection is not deterministic")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
    else:
        build(args.output)


if __name__ == "__main__":
    main()
