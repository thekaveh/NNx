"""Generate self-contained MkDocs and GitHub wiki projections."""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html import unescape as html_unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

import yaml
from markdown import Markdown

from .build_wiki import (
    _is_absolute_web_path,
    _lookup_path,
    _markdown_unescape,
    _parse_target,
    _split_destination,
    html_target_destinations,
    iter_html_targets,
    iter_inline_links,
    iter_reference_definitions,
    iter_reference_usages,
    markdown_blocks,
    rewrite_markdown,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "manifest.yaml"
GENERATED = ROOT / "generated"
MKDOCS_MARKDOWN_EXTENSIONS = [
    "admonition",
    "attr_list",
    "pymdownx.details",
    "pymdownx.superfences",
    "pymdownx.highlight",
    "pymdownx.inlinehilite",
    {"toc": {"permalink": True}},
]


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate key in documentation manifest: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


@dataclass(frozen=True)
class Page:
    number: str
    title: str
    source: Path
    slug: str


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-")


def _validate_output_names(pages: list[Page]) -> None:
    if sum(page.slug == "Home" for page in pages) != 1:
        raise ValueError("manifest must define exactly one page with slug Home")

    names_by_surface = {
        "site": ["index.md" if page.slug == "Home" else f"{page.slug}.md" for page in pages],
        "wiki": [*(f"{page.slug}.md" for page in pages), "_Sidebar.md", "_Footer.md"],
    }
    for surface, names in names_by_surface.items():
        normalized = [name.casefold() for name in names]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"manifest {surface} output paths must be unique and must not use reserved names")


def load_sections() -> list[tuple[str | None, list[Page]]]:
    try:
        data = yaml.load(MANIFEST.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        if not isinstance(data, dict):
            raise ValueError("manifest root must be a mapping")
        unknown_root = data.keys() - {"surfaces", "numbering", "sections"}
        if unknown_root:
            raise ValueError(f"manifest root has unknown keys: {', '.join(sorted(unknown_root))}")
        if data.get("surfaces") != ["repo", "site", "wiki"] or data.get("numbering") != "navigation":
            raise ValueError("manifest must declare all three surfaces and navigation numbering")
        raw_sections = data["sections"]
        if not isinstance(raw_sections, list):
            raise ValueError("manifest sections must be a list")
        sections: list[tuple[str | None, list[Page]]] = []
        for section_index, item in enumerate(raw_sections, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"section {section_index} must be a mapping")
            grouped = "children" in item
            allowed_section_keys = {"title", "children"} if grouped else {"number", "title", "source", "slug"}
            unknown_section = item.keys() - allowed_section_keys
            if unknown_section:
                raise ValueError(f"section {section_index} has unknown keys: {', '.join(sorted(unknown_section))}")
            group = item.get("title") if grouped else None
            if grouped and (not isinstance(group, str) or not group.strip()):
                raise ValueError(f"section {section_index} title must be a non-empty string")
            leaves = item["children"] if grouped else [item]
            if not isinstance(leaves, list):
                raise ValueError(f"section {section_index} children must be a list")
            if grouped and not leaves:
                raise ValueError(f"section {section_index} must have non-empty children")

            pages: list[Page] = []
            for leaf_index, leaf in enumerate(leaves, start=1):
                if not isinstance(leaf, dict):
                    raise ValueError(f"section {section_index} page {leaf_index} must be a mapping")
                unknown_page = leaf.keys() - {"number", "title", "source", "slug"}
                if unknown_page:
                    raise ValueError(
                        f"section {section_index} page {leaf_index} has unknown keys: {', '.join(sorted(unknown_page))}"
                    )
                missing = {"number", "title", "source"} - leaf.keys()
                if missing:
                    raise ValueError(
                        f"section {section_index} page {leaf_index} is missing {', '.join(sorted(missing))}"
                    )
                title = leaf["title"]
                source_value = leaf["source"]
                if not isinstance(title, str) or not title.strip():
                    raise ValueError(f"section {section_index} page {leaf_index} title must be a non-empty string")
                if not isinstance(source_value, str) or not source_value.strip():
                    raise ValueError(f"section {section_index} page {leaf_index} source must be a non-empty string")
                source = (ROOT / source_value).resolve()
                if not source.is_relative_to(ROOT) or not source.is_file():
                    raise ValueError(f"manifest source does not exist inside the repository: {source}")
                slug = leaf.get("slug", _slug(title))
                if not isinstance(slug, str) or re.fullmatch(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", slug) is None:
                    raise ValueError(f"section {section_index} page {leaf_index} must have a filename-safe slug")
                pages.append(Page(number=str(leaf["number"]), title=title, source=source, slug=slug))
            sections.append((group, pages))
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid documentation manifest: {exc}") from exc
    all_pages = [page for _, pages in sections for page in pages]
    for label, values in (
        ("number", [page.number for page in all_pages]),
        ("slug", [page.slug for page in all_pages]),
        ("source", [page.source.resolve() for page in all_pages]),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"manifest page {label}s must be unique")
    expected_numbers = [str(index) for index in range(1, len(all_pages) + 1)]
    actual_numbers = [page.number for page in all_pages]
    if actual_numbers != expected_numbers:
        raise ValueError(f"manifest page numbers must be the ordered sequence 1..{len(all_pages)}")
    canonical_docs = {path.resolve() for path in (ROOT / "docs").glob("*.md")}
    manifested_docs = {page.source.resolve() for page in all_pages if page.source.parent == ROOT / "docs"}
    if manifested_docs != canonical_docs:
        missing = sorted(str(path.relative_to(ROOT)) for path in canonical_docs - manifested_docs)
        extra = sorted(str(path.relative_to(ROOT)) for path in manifested_docs - canonical_docs)
        raise ValueError(f"manifest docs coverage mismatch: missing={missing}, extra={extra}")
    _validate_output_names(all_pages)
    return sections


def _pages() -> list[Page]:
    return [page for _, pages in load_sections() for page in pages]


def _source_map(surface: str) -> dict[Path, str]:
    mapping: dict[Path, str] = {}
    for page in _pages():
        if surface == "site":
            name = "index.md" if page.slug == "Home" else f"{page.slug}.md"
        else:
            name = page.slug
        mapping[page.source.resolve()] = name
        project_alias = ROOT / "docs" / "_project" / f"{page.slug}.md"
        mapping[project_alias.resolve()] = name
    mapping[(ROOT / "docs" / "architecture.html").resolve()] = (
        "architecture.html" if surface == "site" else "Architecture"
    )
    for asset in (ROOT / "docs" / "assets").rglob("*"):
        if asset.is_symlink():
            raise ValueError(f"asset symlink must not escape docs/assets: {asset}")
        if asset.is_file():
            if not asset.resolve().is_relative_to((ROOT / "docs" / "assets").resolve()):
                raise ValueError(f"asset symlink must not escape docs/assets: {asset}")
            relative = asset.relative_to(ROOT / "docs" / "assets").as_posix()
            mapping[asset.resolve()] = f"assets/{relative}" if surface == "site" else f"images/{relative}"
    mapping[(ROOT / "examples").resolve()] = "Examples.md" if surface == "site" else "Examples"
    mapping[(ROOT / "tests").resolve()] = "Test-Import-Boundaries.md" if surface == "site" else "Test-Import-Boundaries"
    for example in (ROOT / "examples").glob("*.py"):
        mapping[example.resolve()] = "Examples.md" if surface == "site" else "Examples"
    mapping[(ROOT / "src" / "nnx" / "_step_helpers.py").resolve()] = (
        "API-reference.md" if surface == "site" else "API-reference"
    )
    return mapping


def _render_page(page: Page, surface: str, source_map: dict[Path, str]) -> str:
    text = page.source.read_text(encoding="utf-8")
    if page.source.name == "LICENSE":
        text = f"# License\n\n```text\n{text.rstrip()}\n```\n"
    return rewrite_markdown(page.source, text, source_map, surface=surface)


def _prepare_output(output: Path) -> None:
    absolute = output.absolute()
    for component in (absolute, *absolute.parents):
        system_tmp_alias = (component, component.resolve()) in {
            (Path("/tmp"), Path("/private/tmp")),
            (Path("/var"), Path("/private/var")),
        }
        if component.is_symlink() and not system_tmp_alias:
            raise ValueError(f"documentation output path cannot contain a symlink: {output}")

    resolved = absolute.resolve(strict=False)
    repository = ROOT.resolve()
    if resolved == repository or repository.is_relative_to(resolved):
        raise ValueError(f"documentation output cannot be the repository or one of its ancestors: {output}")

    managed = {(GENERATED / "site").absolute(), (GENERATED / "wiki").absolute()}
    if absolute.exists():
        if not absolute.is_dir():
            raise ValueError(f"documentation output must be a directory: {output}")
        if any(absolute.iterdir()) and absolute not in managed:
            raise ValueError(f"refusing to replace nonempty unmanaged documentation output: {output}")
        shutil.rmtree(absolute)
    absolute.mkdir(parents=True)


def render_site(output: Path) -> None:
    _prepare_output(output)
    (output / "assets").mkdir(parents=True)
    (output / "stylesheets").mkdir(parents=True)
    mapping = _source_map("site")
    for page in _pages():
        name = "index.md" if page.slug == "Home" else f"{page.slug}.md"
        (output / name).write_text(_render_page(page, "site", mapping), encoding="utf-8")
    shutil.copytree(ROOT / "docs" / "assets", output / "assets", dirs_exist_ok=True)
    shutil.copy2(ROOT / "docs" / "stylesheets" / "extra.css", output / "stylesheets" / "extra.css")
    shutil.copy2(ROOT / "docs" / "architecture.html", output / "architecture.html")


def render_wiki(output: Path) -> None:
    _prepare_output(output)
    (output / "images").mkdir(parents=True)
    mapping = _source_map("wiki")
    sections = load_sections()
    for page in _pages():
        (output / f"{page.slug}.md").write_text(_render_page(page, "wiki", mapping), encoding="utf-8")
    shutil.copytree(ROOT / "docs" / "assets", output / "images", dirs_exist_ok=True)
    lines = ["# NNx", ""]
    for group, pages in sections:
        if group:
            lines.extend((f"## {group}", ""))
        lines.extend(f"- [{page.number}. {page.title}]({page.slug})" for page in pages)
        lines.append("")
    (output / "_Sidebar.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    (output / "_Footer.md").write_text("Apache-2.0 licensed.\n", encoding="utf-8")


def render_mkdocs(output: Path = ROOT / "mkdocs.yml") -> None:
    nav: list[dict[str, object]] = []
    for group, pages in load_sections():
        entries = [
            {f"{page.number}. {page.title}": "index.md" if page.slug == "Home" else f"{page.slug}.md"} for page in pages
        ]
        entry: dict[str, object] = {group: entries} if group else dict(entries[0])
        nav.append(entry)
    config = {
        "site_name": "NNx",
        "site_description": "Lightweight PyTorch training, evaluation, and visualization toolkit",
        "site_url": "https://thekaveh.github.io/NNx/",
        "docs_dir": "generated/site",
        "site_dir": "site",
        "theme": {
            "name": "material",
            "font": {"text": "Inter", "code": "JetBrains Mono"},
            "features": ["navigation.sections", "navigation.expand", "content.code.copy", "search.suggest"],
            "palette": [
                {"scheme": "slate", "toggle": {"icon": "material/weather-sunny", "name": "Switch to light mode"}},
                {"scheme": "default", "toggle": {"icon": "material/weather-night", "name": "Switch to dark mode"}},
            ],
        },
        "extra_css": ["stylesheets/extra.css"],
        "plugins": [
            "search",
            {
                "mkdocstrings": {
                    "handlers": {
                        "python": {
                            "options": {
                                "docstring_style": "google",
                                "show_source": True,
                                "show_root_heading": True,
                                "show_signature_annotations": True,
                            }
                        }
                    }
                }
            },
        ],
        "markdown_extensions": MKDOCS_MARKDOWN_EXTENSIONS,
        "nav": nav,
    }
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def validate_links(root: Path, surface: str) -> None:
    prohibited_origins = (
        "https://thekaveh.github.io/NNx",
        "https://github.com/thekaveh/NNx/wiki",
        "https://github.com/thekaveh/NNx/blob/",
    )

    def validate_destination(source: Path, destination: str) -> None:
        _, _, raw_fragment = _split_destination(destination)
        path = _lookup_path(destination)
        fragment = _markdown_unescape(raw_fragment.removeprefix("#"))
        if path.startswith(prohibited_origins):
            raise ValueError(f"cross-surface {surface} link in {source.name}: {destination}")
        if path.startswith("/") and not path.startswith("//"):
            raise ValueError(f"root-relative {surface} link in {source.name}: {destination}")
        if _is_absolute_web_path(path):
            return
        candidate = source.resolve() if not path else (source.parent / path).resolve()
        if not candidate.is_relative_to(root.resolve()):
            raise ValueError(f"local link points outside the projection in {source.name}: {destination}")
        if surface == "wiki" and not candidate.exists() and not Path(path).suffix:
            candidate = (root / f"{path}.md").resolve()
            if not candidate.is_relative_to(root.resolve()):
                raise ValueError(f"local link points outside the projection in {source.name}: {destination}")
        if not candidate.exists():
            raise ValueError(f"broken {surface} link in {source.name}: {destination}")
        if fragment and candidate.suffix == ".md" and unquote(fragment) not in _anchors(candidate, surface):
            raise ValueError(f"broken {surface} anchor in {source.name}: {destination}")

    def validate_text(source: Path, text: str) -> None:
        for link in iter_inline_links(text):
            destination, _ = _parse_target(link.target)
            validate_destination(source, destination)

    def validate_html(source: Path, text: str) -> None:
        for target in iter_html_targets(text):
            for destination in html_target_destinations(target):
                validate_destination(source, destination)
        css_parser = _EmbeddedCssParser()
        css_parser.feed(text)
        for css in css_parser.snippets:
            for destination in _css_destinations(css):
                validate_destination(source, destination)

    for source in root.glob("*.md"):
        source_text = source.read_text(encoding="utf-8")
        blocks = markdown_blocks(source_text)
        definitions: set[str] = set()
        definitions_by_block: dict[int, list] = {}
        for block_index, block in enumerate(blocks):
            if block.html_processable:
                validate_html(source, block.text)
            if block.suppressed:
                continue
            block_definitions = list(iter_reference_definitions(block.text))
            definitions_by_block[block_index] = block_definitions
            for definition in block_definitions:
                if not definition.target:
                    continue
                definitions.add(_normalize_reference_label(definition.label))
                destination, _ = _parse_target(definition.target)
                validate_destination(source, destination)

        for block_index, block in enumerate(blocks):
            if block.suppressed:
                continue
            block_definitions = definitions_by_block.get(block_index, [])
            chunks: list[str] = []
            cursor = 0
            for definition in block_definitions:
                chunks.append(block.text[cursor : definition.start])
                cursor = definition.end
            chunks.append(block.text[cursor:])
            prose = "".join(chunks)
            for usage in iter_reference_usages(prose):
                label = usage.reference or usage.label
                if _normalize_reference_label(label) not in definitions:
                    raise ValueError(f"undefined {surface} reference in {source.name}: {label}")
            validate_text(source, prose)

    for source in root.rglob("*.html"):
        validate_html(source, source.read_text(encoding="utf-8"))

    for source in root.rglob("*.svg"):
        svg_text = source.read_text(encoding="utf-8")
        validate_html(source, svg_text)
        for destination in _svg_stylesheet_destinations(svg_text):
            validate_destination(source, destination)

    for source in root.rglob("*.css"):
        for destination in _css_destinations(source.read_text(encoding="utf-8")):
            validate_destination(source, destination)


class _EmbeddedCssParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.snippets: list[str] = []
        self._style_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name.lower() == "style" and value is not None:
                self.snippets.append(value)
        if tag.lower() == "style":
            self._style_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "style" and self._style_depth:
            self._style_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._style_depth:
            self.snippets.append(data)


def _css_destinations(text: str) -> list[str]:
    css = _strip_css_comments(text)
    destinations: list[str] = []
    cursor = 0
    while cursor < len(css):
        if css[cursor] in {'"', "'"}:
            _, cursor = _css_string(css, cursor)
            continue
        import_match = re.match(
            r"@((?:[A-Za-z_-]|\\(?:[0-9A-Fa-f]{1,6}[ \t\r\n\f]?|.))+)[ \t\r\n\f]+",
            css[cursor:],
        )
        if import_match is not None and _css_unescape(import_match.group(1)).lower() == "import":
            value_start = cursor + import_match.end()
            if value_start < len(css) and css[value_start] in {'"', "'"}:
                value, cursor = _css_string(css, value_start)
                destinations.append(value)
                continue
        identifier_match = re.match(r"(?:[A-Za-z_-]|\\(?:[0-9A-Fa-f]{1,6}[ \t\r\n\f]?|.))+", css[cursor:])
        if identifier_match is None:
            cursor += 1
            continue
        name = _css_unescape(identifier_match.group()).lower()
        name_end = cursor + identifier_match.end()
        opening = name_end
        while opening < len(css) and css[opening].isspace():
            opening += 1
        if name not in {"url", "image", "image-set", "-webkit-image-set"} or opening >= len(css) or css[opening] != "(":
            cursor = name_end
            continue
        content_start = opening + 1
        content_end = _css_function_end(css, content_start)
        if content_end is None:
            break
        content = css[content_start:content_end]
        if name == "url":
            value = content.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            value = _css_unescape(value)
            if value:
                destinations.append(value)
        else:
            destinations.extend(_css_destinations(content))
            for candidate in _css_top_level_candidates(content):
                candidate = candidate.lstrip()
                candidate = re.sub(r"^(?:ltr|rtl)[ \t\r\n\f]+", "", candidate, flags=re.IGNORECASE)
                if candidate.startswith(('"', "'")):
                    value, _ = _css_string(candidate, 0)
                    destinations.append(value)
        cursor = content_end + 1
    return destinations


def _css_string(text: str, start: int) -> tuple[str, int]:
    quote = text[start]
    characters: list[str] = []
    cursor = start + 1
    while cursor < len(text):
        if text[cursor] == "\\" and cursor + 1 < len(text):
            character, cursor = _css_escape(text, cursor)
            characters.append(character)
            continue
        elif text[cursor] == quote:
            return "".join(characters), cursor + 1
        else:
            characters.append(text[cursor])
        cursor += 1
    return "".join(characters), cursor


def _css_escape(text: str, start: int) -> tuple[str, int]:
    cursor = start + 1
    if cursor < len(text) and text[cursor] in {"\r", "\n", "\f"}:
        if text[cursor] == "\r" and cursor + 1 < len(text) and text[cursor + 1] == "\n":
            cursor += 1
        return "", cursor + 1
    match = re.match(r"[0-9A-Fa-f]{1,6}", text[cursor:])
    if match is not None:
        cursor += match.end()
        codepoint = int(match.group(), 16)
        character = "�" if codepoint == 0 or 0xD800 <= codepoint <= 0xDFFF or codepoint > 0x10FFFF else chr(codepoint)
        if cursor < len(text) and text[cursor].isspace():
            cursor += 1
        return character, cursor
    if cursor < len(text):
        return text[cursor], cursor + 1
    return "", cursor


def _css_unescape(text: str) -> str:
    characters: list[str] = []
    cursor = 0
    while cursor < len(text):
        if text[cursor] == "\\":
            character, cursor = _css_escape(text, cursor)
            characters.append(character)
        else:
            characters.append(text[cursor])
            cursor += 1
    return "".join(characters)


def _css_function_end(text: str, start: int) -> int | None:
    depth = 1
    cursor = start
    while cursor < len(text):
        if text[cursor] == "\\":
            _, cursor = _css_escape(text, cursor)
            continue
        if text[cursor] in {'"', "'"}:
            _, cursor = _css_string(text, cursor)
            continue
        if text[cursor] == "(":
            depth += 1
        elif text[cursor] == ")":
            depth -= 1
            if depth == 0:
                return cursor
        cursor += 1
    return None


def _css_top_level_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    start = 0
    depth = 0
    cursor = 0
    while cursor < len(text):
        if text[cursor] == "\\":
            _, cursor = _css_escape(text, cursor)
            continue
        if text[cursor] in {'"', "'"}:
            _, cursor = _css_string(text, cursor)
            continue
        if text[cursor] == "(":
            depth += 1
        elif text[cursor] == ")":
            depth = max(0, depth - 1)
        elif text[cursor] == "," and depth == 0:
            candidates.append(text[start:cursor])
            start = cursor + 1
        cursor += 1
    candidates.append(text[start:])
    return candidates


def _svg_stylesheet_destinations(text: str) -> list[str]:
    destinations: list[str] = []
    for _event, instruction in ET.iterparse(io.StringIO(text), events=("pi",)):
        instruction_text = instruction.text or ""
        if not instruction_text.lower().startswith("xml-stylesheet"):
            continue
        attributes = _processing_instruction_attributes(instruction_text[len("xml-stylesheet") :])
        if "href" in attributes:
            destinations.append(html_unescape(attributes["href"]))
    return destinations


def _processing_instruction_attributes(text: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    cursor = 0
    while cursor < len(text):
        match = re.match(r"\s*([A-Za-z_:][A-Za-z0-9_.:-]*)\s*=\s*(['\"])", text[cursor:])
        if match is None:
            break
        name, quote = match.group(1).lower(), match.group(2)
        value_start = cursor + match.end()
        value_end = text.find(quote, value_start)
        if value_end < 0:
            raise ValueError("malformed processing-instruction attribute")
        if name in attributes:
            raise ValueError(f"duplicate processing-instruction attribute: {name}")
        attributes[name] = text[value_start:value_end]
        cursor = value_end + 1
    if text[cursor:].strip():
        raise ValueError("malformed processing-instruction attributes")
    return attributes


def _strip_css_comments(text: str) -> str:
    """Remove CSS comments without treating comment markers in strings as syntax."""
    chunks: list[str] = []
    cursor = 0
    quote: str | None = None
    while cursor < len(text):
        character = text[cursor]
        if quote is not None:
            chunks.append(character)
            if character == "\\" and cursor + 1 < len(text):
                cursor += 1
                chunks.append(text[cursor])
            elif character == quote:
                quote = None
            cursor += 1
            continue
        if character in {'"', "'"}:
            quote = character
            chunks.append(character)
            cursor += 1
            continue
        if text.startswith("/*", cursor):
            closing = text.find("*/", cursor + 2)
            if closing < 0:
                return "".join(chunks)
            chunks.append(" ")
            cursor = closing + 2
            continue
        chunks.append(character)
        cursor += 1
    return "".join(chunks)


class _AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.anchors: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.anchors.add(values["id"] or "")
        if tag == "a" and (values.get("id") or values.get("name")):
            self.anchors.add(values.get("id") or values.get("name") or "")


class _WikiAnchorParser(_AnchorParser):
    def __init__(self) -> None:
        super().__init__()
        self.headings: list[str] = []
        self._heading_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        super().handle_starttag(tag, attrs)
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading_parts = []

    def handle_data(self, data: str) -> None:
        if self._heading_parts is not None:
            self._heading_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self._heading_parts is not None:
            self.headings.append("".join(self._heading_parts))
            self._heading_parts = None


def _github_heading_slug(heading: str) -> str:
    visible = html_unescape(heading).strip().lower()
    kept = "".join(
        character for character in visible if character.isalnum() or character in "_- " or character.isspace()
    )
    return re.sub(r"\s+", "-", kept)


def _anchors(source: Path, surface: str) -> set[str]:
    text = source.read_text(encoding="utf-8")
    if surface == "site":
        parser = _AnchorParser()
        extension_names = [item if isinstance(item, str) else next(iter(item)) for item in MKDOCS_MARKDOWN_EXTENSIONS]
        parser.feed(Markdown(extensions=extension_names, extension_configs={"toc": {"permalink": True}}).convert(text))
        return parser.anchors

    rendered_parser = _WikiAnchorParser()
    rendered_parser.feed(Markdown(extensions=["pymdownx.superfences"]).convert(text))
    anchors = set(rendered_parser.anchors)
    generated_anchors: set[str] = set()
    counts: dict[str, int] = {}

    def add_heading(heading: str) -> None:
        slug = _github_heading_slug(heading)
        duplicate_separator = "-" if surface == "wiki" else "_"
        suffix = counts.get(slug, 0)
        candidate = slug if suffix == 0 else f"{slug}{duplicate_separator}{suffix}"
        while candidate in generated_anchors:
            suffix += 1
            candidate = f"{slug}{duplicate_separator}{suffix}"
        counts[slug] = suffix + 1
        generated_anchors.add(candidate)
        anchors.add(candidate)

    for heading in rendered_parser.headings:
        add_heading(heading)
    return anchors


def _normalize_reference_label(label: str) -> str:
    return " ".join(html_unescape(_markdown_unescape(label)).split()).casefold()


def _snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _render_bundle(site: Path, wiki: Path, config: Path) -> None:
    render_site(site)
    render_wiki(wiki)
    render_mkdocs(config)
    validate_links(site, "site")
    validate_links(wiki, "wiki")


def _assert_bundles_match(first: Path, second: Path) -> None:
    if _snapshot(first / "site") != _snapshot(second / "site") or _snapshot(first / "wiki") != _snapshot(
        second / "wiki"
    ):
        raise SystemExit("documentation projections are not deterministic")
    if (first / "mkdocs.yml").read_bytes() != (second / "mkdocs.yml").read_bytes():
        raise SystemExit("mkdocs.yml generation is not deterministic")


def build(*, check: bool = False) -> None:
    with tempfile.TemporaryDirectory() as temp:
        reference = Path(temp).resolve()
        _render_bundle(reference / "site", reference / "wiki", reference / "mkdocs.yml")
        if check:
            with tempfile.TemporaryDirectory() as second_temp:
                second = Path(second_temp).resolve()
                _render_bundle(second / "site", second / "wiki", second / "mkdocs.yml")
                _assert_bundles_match(reference, second)
            return

        _render_bundle(GENERATED / "site", GENERATED / "wiki", ROOT / "mkdocs.yml")
        current = Path(temp) / "current"
        current.mkdir()
        shutil.copytree(GENERATED / "site", current / "site")
        shutil.copytree(GENERATED / "wiki", current / "wiki")
        shutil.copy2(ROOT / "mkdocs.yml", current / "mkdocs.yml")
        _assert_bundles_match(reference, current)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    build(check=args.check)


if __name__ == "__main__":
    main()
