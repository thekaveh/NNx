"""Project canonical repository docs into a self-contained GitHub wiki."""

from __future__ import annotations

import argparse
import re
import tempfile
from dataclasses import dataclass
from html import unescape as html_unescape
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "generated" / "wiki"
GITHUB_BLOB_PREFIX = "https://github.com/thekaveh/NNx/blob/"

ATTR_RE = re.compile(r"\{target=\"_blank\" rel=\"noopener\"\}")
FENCE_RE = re.compile(r"^ {0,3}(?P<marker>`{3,}|~{3,})(?P<rest>.*)$")
ATTR_LIST_RE = re.compile(r"[ \t]+\{(?P<attrs>[^{}]*)\}[ \t]*$")
ATTR_ID_RE = re.compile(r"(?:^|[ \t])#(?P<id>[A-Za-z0-9_.:-]+)(?=$|[ \t])")
ATX_HEADING_RE = re.compile(r"^ {0,3}#{1,6}[ \t]+")
SETEXT_UNDERLINE_RE = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")
MARKDOWN_ESCAPE_RE = re.compile(r"""\\([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])""")
LIST_ITEM_RE = re.compile(r"^(?P<prefix> {0,3}(?:[-+*]|\d+[.)])[ \t]+)(?P<content>.*)$")
NESTED_LIST_ITEM_RE = re.compile(r"^(?P<prefix>[ \t]*(?:(?:[-+*]|\d+[.)])[ \t]+)+)(?P<content>.*)$")
RAW_HTML_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "base",
    "basefont",
    "blockquote",
    "body",
    "caption",
    "center",
    "col",
    "colgroup",
    "dd",
    "details",
    "dialog",
    "dir",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "frame",
    "frameset",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "head",
    "header",
    "hr",
    "html",
    "iframe",
    "legend",
    "li",
    "link",
    "main",
    "menu",
    "menuitem",
    "nav",
    "noframes",
    "ol",
    "optgroup",
    "option",
    "p",
    "param",
    "search",
    "section",
    "summary",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "title",
    "tr",
    "track",
    "ul",
    "script",
    "style",
    "textarea",
}
HTML_RESOURCE_ATTRIBUTES = {
    "a": {"href"},
    "area": {"href"},
    "audio": {"src"},
    "animate": {"href", "xlink:href"},
    "animatemotion": {"href", "xlink:href"},
    "animatetransform": {"href", "xlink:href"},
    "base": {"href"},
    "button": {"formaction"},
    "cursor": {"href", "xlink:href"},
    "embed": {"src"},
    "feimage": {"href", "xlink:href"},
    "filter": {"href", "xlink:href"},
    "iframe": {"src"},
    "image": {"href", "xlink:href"},
    "img": {"src", "srcset"},
    "link": {"href"},
    "lineargradient": {"href", "xlink:href"},
    "mpath": {"href", "xlink:href"},
    "object": {"data"},
    "pattern": {"href", "xlink:href"},
    "radialgradient": {"href", "xlink:href"},
    "form": {"action"},
    "input": {"formaction", "src"},
    "script": {"src", "href", "xlink:href"},
    "source": {"src", "srcset"},
    "set": {"href", "xlink:href"},
    "track": {"src"},
    "textpath": {"href", "xlink:href"},
    "video": {"poster", "src"},
    "use": {"href", "xlink:href"},
}


@dataclass(frozen=True)
class InlineLink:
    start: int
    end: int
    image: bool
    label: str
    target: str


@dataclass(frozen=True)
class ReferenceDefinition:
    start: int
    end: int
    indent: str
    label: str
    target: str | None


@dataclass(frozen=True)
class MarkdownBlock:
    text: str
    suppressed: bool
    html_processable: bool


@dataclass(frozen=True)
class HtmlTarget:
    start: int
    end: int
    tag: str
    attr: str
    target: str
    target_start: int
    target_end: int


@dataclass(frozen=True)
class ReferenceUsage:
    start: int
    end: int
    image: bool
    label: str
    reference: str


def _append_block(blocks: list[MarkdownBlock], text: str, suppressed: bool, html_processable: bool) -> None:
    if not text:
        return
    if blocks and (blocks[-1].suppressed, blocks[-1].html_processable) == (suppressed, html_processable):
        previous = blocks[-1]
        blocks[-1] = MarkdownBlock(previous.text + text, suppressed, html_processable)
    else:
        blocks.append(MarkdownBlock(text, suppressed, html_processable))


def _html_tag_spans(text: str):
    cursor = 0
    while cursor < len(text):
        start = text.find("<", cursor)
        if start < 0:
            return
        if text.startswith("<!--", start):
            end = text.find("-->", start + 4)
            cursor = len(text) if end < 0 else end + 3
            continue
        name_start = start + 1 + (text[start + 1 : start + 2] == "/")
        if name_start >= len(text) or not text[name_start].isalpha():
            cursor = start + 1
            continue
        position = name_start + 1
        while position < len(text) and (text[position].isalnum() or text[position] in "-:"):
            position += 1
        tag = text[name_start:position].lower()
        quote_char: str | None = None
        end = position
        while end < len(text):
            character = text[end]
            if quote_char is not None:
                if character == quote_char:
                    quote_char = None
            elif character in {'"', "'"}:
                quote_char = character
            elif character == ">":
                end += 1
                break
            end += 1
        else:
            return
        yield start, end, tag, position
        cursor = end


def _html_block_depth_delta(text: str, tag: str) -> int:
    void_tags = {"base", "basefont", "col", "frame", "hr", "link", "param", "track"}
    depth = 0
    for start, end, candidate, _attrs in _html_tag_spans(text):
        if candidate != tag:
            continue
        raw = text[start:end]
        if raw.startswith("</"):
            depth -= 1
        elif tag not in void_tags and not raw.rstrip().endswith("/>"):
            depth += 1
    return depth


def iter_html_targets(text: str):
    """Yield exact href/src attributes with source offsets."""
    for start, end, tag, attrs_start in _html_tag_spans(text):
        target_attributes = HTML_RESOURCE_ATTRIBUTES.get(tag)
        if target_attributes is None:
            continue
        cursor = attrs_start
        targets: list[HtmlTarget] = []
        while cursor < end - 1:
            while cursor < end - 1 and (text[cursor].isspace() or text[cursor] == "/"):
                cursor += 1
            name_start = cursor
            while cursor < end - 1 and (text[cursor].isalnum() or text[cursor] in "-:_"):
                cursor += 1
            if cursor == name_start:
                cursor += 1
                continue
            attr = text[name_start:cursor].lower()
            while cursor < end - 1 and text[cursor].isspace():
                cursor += 1
            if cursor >= end - 1 or text[cursor] != "=":
                continue
            cursor += 1
            while cursor < end - 1 and text[cursor].isspace():
                cursor += 1
            if cursor >= end - 1:
                break
            if text[cursor] in {'"', "'"}:
                quote_char = text[cursor]
                value_start = cursor + 1
                value_end = text.find(quote_char, value_start, end)
                if value_end < 0:
                    break
                cursor = value_end + 1
            else:
                value_start = cursor
                while cursor < end - 1 and not text[cursor].isspace() and text[cursor] != ">":
                    cursor += 1
                value_end = cursor
            if attr in target_attributes:
                targets.append(HtmlTarget(start, end, tag, attr, text[value_start:value_end], value_start, value_end))
        seen: set[str] = set()
        for target in targets:
            if target.attr in seen:
                raise ValueError(f"duplicate HTML attribute {target.attr!r} on <{tag}>")
            seen.add(target.attr)
        yield from targets


def html_target_destinations(target: HtmlTarget) -> list[str]:
    if target.attr != "srcset":
        return [html_unescape(target.target)]
    return [html_unescape(url) for url, _descriptor in _srcset_candidates(target.target)]


def _srcset_candidates(value: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    cursor = 0
    while cursor < len(value):
        while cursor < len(value) and (value[cursor].isspace() or value[cursor] == ","):
            cursor += 1
        if cursor >= len(value):
            break
        start = cursor
        while (
            cursor < len(value)
            and not value[cursor].isspace()
            and (value[start:].startswith("data:") or value[cursor] != ",")
        ):
            cursor += 1
        url = value[start:cursor]
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1
        descriptor_start = cursor
        while cursor < len(value) and value[cursor] != ",":
            cursor += 1
        descriptor = value[descriptor_start:cursor].strip()
        valid_descriptor = re.fullmatch(r"(?:\d+w(?:\s+\d+h)?|\d+(?:\.\d+)?x)", descriptor)
        if url.startswith("data:") and url.endswith(",") and descriptor and valid_descriptor is None:
            candidates.append((url[:-1], ""))
            cursor = descriptor_start
            continue
        candidates.append((url, descriptor))
        if cursor < len(value):
            cursor += 1
    return candidates


def _code_span_ranges(text: str) -> list[tuple[int, int]]:
    link_ranges = [(link.start, link.end) for link in iter_inline_links(text)]
    link_ranges.extend((usage.start, usage.end) for usage in iter_reference_usages(text))
    link_ranges.extend((definition.start, definition.end) for definition in iter_reference_definitions(text))
    ranges: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(text):
        opening = text.find("`", cursor)
        if opening < 0:
            break
        run_end = opening
        while run_end < len(text) and text[run_end] == "`":
            run_end += 1
        width = run_end - opening
        closing_end = -1
        search = run_end
        while search < len(text):
            closing = text.find("`", search)
            if closing < 0:
                break
            candidate_end = closing
            while candidate_end < len(text) and text[candidate_end] == "`":
                candidate_end += 1
            if candidate_end - closing == width:
                closing_end = candidate_end
                break
            search = candidate_end
        if closing_end < 0:
            cursor = run_end
            continue
        if not any(start <= opening and closing_end <= end for start, end in link_ranges):
            ranges.append((opening, closing_end))
        cursor = closing_end
    return ranges


def _inline_suppressed_blocks(text: str) -> list[MarkdownBlock]:
    code_ranges = _code_span_ranges(text)
    suppressed_ranges = list(code_ranges)
    cursor = 0
    while cursor < len(text):
        code_range = next((item for item in code_ranges if item[0] <= cursor < item[1]), None)
        if code_range is not None:
            cursor = code_range[1]
            continue
        opening = text.find("<!--", cursor)
        if opening < 0:
            break
        containing_code = next((item for item in code_ranges if item[0] <= opening < item[1]), None)
        if containing_code is not None:
            cursor = containing_code[1]
            continue
        closing = text.find("-->", opening + 4)
        end = len(text) if closing < 0 else closing + 3
        suppressed_ranges.append((opening, end))
        cursor = end

    merged: list[tuple[int, int]] = []
    for start, end in sorted(suppressed_ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    blocks: list[MarkdownBlock] = []
    cursor = 0
    for start, end in merged:
        _append_block(blocks, text[cursor:start], False, True)
        _append_block(blocks, text[start:end], True, False)
        cursor = end
    _append_block(blocks, text[cursor:], False, True)
    return blocks


def markdown_blocks(text: str) -> list[MarkdownBlock]:
    """Tokenize Markdown into processable prose and suppressed code/comment blocks."""
    blocks: list[MarkdownBlock] = []
    prose: list[str] = []
    active_fence: tuple[str, int, int, int] | None = None
    in_indented_code = False
    in_pre_block: int | None = None
    in_html_block: tuple[str, int, bool, int] | None = None
    active_list_indent = 0
    literal_html_state: tuple[str, int, int] | None = None
    previous_blank = True

    def flush_prose() -> None:
        if prose:
            for block in _inline_suppressed_blocks("".join(prose)):
                _append_block(blocks, block.text, block.suppressed, block.html_processable)
            prose.clear()

    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        quote_prefix, container_content = _blockquote_parts(content)
        quote_depth = quote_prefix.count(">")
        blank = not container_content.strip()
        previous_list_indent = active_list_indent
        list_item = NESTED_LIST_ITEM_RE.match(container_content)
        if list_item is not None:
            list_indent = _prefix_columns(list_item.group("prefix"))
            active_list_indent = list_indent
            parser_content = list_item.group("content")
        else:
            leading = _indent_columns(container_content)
            if active_list_indent and (blank or leading >= active_list_indent):
                list_indent = active_list_indent
                parser_content = _strip_indent_columns(container_content, active_list_indent) if not blank else ""
            else:
                if not blank:
                    active_list_indent = 0
                list_indent = 0
                parser_content = container_content
        if previous_list_indent and list_indent < previous_list_indent and not blank:
            flush_prose()
            in_html_block = None
        if literal_html_state is not None:
            _, literal_quote_depth, literal_list_indent = literal_html_state
            if quote_depth != literal_quote_depth or (
                literal_list_indent and list_indent < literal_list_indent and not blank
            ):
                literal_html_state = None
        match = FENCE_RE.match(parser_content)
        marker = match.group("marker") if match else None

        if active_fence is not None and active_fence[2] > 0 and quote_depth != active_fence[2]:
            active_fence = None
        if in_pre_block is not None and in_pre_block > 0 and quote_depth != in_pre_block:
            in_pre_block = None
        if in_html_block is not None and in_html_block[1] > 0 and quote_depth != in_html_block[1]:
            in_html_block = None
        if in_html_block is not None and blank and in_html_block[0] not in {"pre", "script", "style", "textarea"}:
            in_html_block = None

        if literal_html_state is not None:
            flush_prose()
            _append_block(blocks, line, True, False)
            if literal_html_state[0] in parser_content:
                literal_html_state = None
            previous_blank = blank
            continue

        stripped_container = parser_content.lstrip()
        special_literal = None
        if stripped_container.startswith("<![CDATA["):
            special_literal = ("<![CDATA[", "]]>")
        elif stripped_container.startswith("<?"):
            special_literal = ("<?", "?>")
        elif re.match(r"<![A-Z]", stripped_container):
            special_literal = ("<!", ">")
        if special_literal is not None and not container_content.lstrip().startswith("<!--"):
            flush_prose()
            _append_block(blocks, line, True, False)
            remainder = container_content[container_content.find(special_literal[0]) + len(special_literal[0]) :]
            if special_literal[1] not in remainder:
                literal_html_state = (special_literal[1], quote_depth, list_indent)
            previous_blank = blank
            continue

        if active_fence is not None:
            fence_content = container_content
            if active_fence[3]:
                leading = _indent_columns(container_content)
                if not blank and leading < active_fence[3]:
                    closing = re.fullmatch(
                        rf" {{0,3}}{re.escape(active_fence[0])}{{{active_fence[1]},}}[ \t]*",
                        container_content,
                    )
                    if closing is not None:
                        flush_prose()
                        _append_block(blocks, line, True, False)
                        active_fence = None
                        previous_blank = blank
                        continue
                    active_fence = None
                else:
                    fence_content = _strip_indent_columns(container_content, active_fence[3]) if not blank else ""
            if active_fence is None:
                match = FENCE_RE.match(parser_content)
                marker = match.group("marker") if match else None
            else:
                flush_prose()
                _append_block(blocks, line, True, False)
                closing = re.fullmatch(
                    rf" {{0,3}}{re.escape(active_fence[0])}{{{active_fence[1]},}}[ \t]*",
                    fence_content,
                )
                if closing is not None and quote_depth == active_fence[2]:
                    active_fence = None
                previous_blank = blank
                continue

        if in_pre_block is not None:
            flush_prose()
            _append_block(blocks, line, True, False)
            if re.search(r"</pre\s*>", container_content, re.IGNORECASE):
                in_pre_block = None
            previous_blank = blank
            continue

        if in_html_block is not None:
            flush_prose()
            tag, depth_quote, process_html, depth = in_html_block
            _append_block(blocks, line, True, process_html)
            if tag in {"script", "style", "textarea"}:
                depth += _html_block_depth_delta(container_content, tag)
                if depth <= 0:
                    in_html_block = None
                else:
                    in_html_block = (tag, depth_quote, process_html, depth)
            else:
                in_html_block = (tag, depth_quote, process_html, depth)
            previous_blank = blank
            continue

        if re.match(r"^ {0,3}<pre(?:\s|>)", parser_content, re.IGNORECASE):
            flush_prose()
            _append_block(blocks, line, True, False)
            if re.search(r"</pre\s*>", container_content, re.IGNORECASE) is None:
                in_pre_block = quote_depth
            previous_blank = blank
            continue
        html_open = re.match(r"^ {0,3}<(?P<tag>[A-Za-z][A-Za-z0-9-]*)(?:\s|>)", parser_content)
        if html_open is not None and html_open.group("tag").lower() in RAW_HTML_BLOCK_TAGS:
            flush_prose()
            tag = html_open.group("tag").lower()
            process_html = tag not in {"script", "textarea"}
            process_opening = process_html or tag in HTML_RESOURCE_ATTRIBUTES
            _append_block(blocks, line, True, process_opening)
            depth = _html_block_depth_delta(parser_content, tag)
            if tag not in {"script", "style", "textarea"} or depth > 0:
                in_html_block = (tag, quote_depth, process_html, depth)
            previous_blank = blank
            continue

        if in_indented_code:
            if blank or parser_content.startswith(("    ", "\t")):
                flush_prose()
                _append_block(blocks, line, True, False)
                previous_blank = blank
                continue
            in_indented_code = False

        if marker is not None:
            flush_prose()
            _append_block(blocks, line, True, False)
            active_fence = (marker[0], len(marker), quote_depth, list_indent)
        elif parser_content.startswith(("    ", "\t")) and previous_blank:
            flush_prose()
            _append_block(blocks, line, True, False)
            in_indented_code = True
        else:
            prose.append(line)
        previous_blank = blank

    flush_prose()
    return blocks


def _scan_balanced(text: str, start: int, opening: str, closing: str) -> int | None:
    depth = 1
    cursor = start + 1
    while cursor < len(text):
        character = text[cursor]
        if character == "\\" and cursor + 1 < len(text):
            cursor += 2
            continue
        if character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return cursor
        cursor += 1
    return None


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _scan_link_target(text: str, start: int) -> int | None:
    depth = 1
    cursor = start + 1
    angle_destination = False
    quoted_title: str | None = None
    destination_started = False
    destination_finished = False
    while cursor < len(text):
        character = text[cursor]
        if character == "\\" and cursor + 1 < len(text):
            cursor += 2
            continue
        if angle_destination:
            if character == ">":
                angle_destination = False
                destination_finished = True
            cursor += 1
            continue
        if quoted_title is not None:
            if character == quoted_title:
                quoted_title = None
            cursor += 1
            continue
        if depth == 1 and not destination_started and character.isspace():
            cursor += 1
            continue
        if depth == 1 and not destination_started and character == "<":
            destination_started = True
            angle_destination = True
            cursor += 1
            continue
        if depth == 1 and destination_started and character.isspace():
            destination_finished = True
            cursor += 1
            continue
        if depth == 1 and destination_finished and character in {'"', "'"}:
            quoted_title = character
            cursor += 1
            continue
        destination_started = True
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return cursor
        cursor += 1
    return None


def iter_inline_links(text: str):
    """Yield balanced inline Markdown links and images."""
    html_ranges = [(start, end) for start, end, _tag, _attrs in _html_tag_spans(text)]
    cursor = 0
    while cursor < len(text):
        label_start = text.find("[", cursor)
        if label_start < 0:
            return
        if _is_escaped(text, label_start):
            cursor = label_start + 1
            continue
        containing_html = next((end for start, end in html_ranges if start <= label_start < end), None)
        if containing_html is not None:
            cursor = containing_html
            continue
        image = label_start > 0 and text[label_start - 1] == "!" and (not _is_escaped(text, label_start - 1))
        start = label_start - 1 if image else label_start
        label_end = _scan_balanced(text, label_start, "[", "]")
        if label_end is None or label_end + 1 >= len(text) or text[label_end + 1] != "(":
            cursor = label_start + 1
            continue
        target_end = _scan_link_target(text, label_end + 1)
        if target_end is None:
            cursor = label_start + 1
            continue
        yield InlineLink(
            start=start,
            end=target_end + 1,
            image=image,
            label=text[label_start + 1 : label_end],
            target=text[label_end + 2 : target_end],
        )
        cursor = target_end + 1


def parse_reference_definition(line: str) -> ReferenceDefinition | None:
    quote_prefix, content = _blockquote_parts(line)
    indent_length = len(content) - len(content.lstrip(" "))
    if indent_length > 3:
        return None
    cursor = indent_length
    if cursor >= len(content) or content[cursor] != "[":
        return None
    label_end = _scan_balanced(content, cursor, "[", "]")
    if label_end is None or label_end + 1 >= len(content) or content[label_end + 1] != ":":
        return None
    target = content[label_end + 2 :].strip() or None
    return ReferenceDefinition(
        0,
        len(line),
        quote_prefix + content[:indent_length],
        content[cursor + 1 : label_end],
        target,
    )


def iter_reference_definitions(text: str):
    """Yield single- and continuation-line reference definitions."""
    lines = text.splitlines(keepends=True)
    offset = 0
    index = 0
    active_list_indent = 0
    while index < len(lines):
        line = lines[index]
        content = line.rstrip("\r\n")
        quote_prefix, body = _blockquote_parts(content)
        list_item = LIST_ITEM_RE.match(body)
        if list_item is not None:
            active_list_indent = _prefix_columns(list_item.group("prefix"))
            parse_body = list_item.group("content")
        else:
            leading = _indent_columns(body)
            if active_list_indent and leading >= active_list_indent:
                parse_body = _strip_indent_columns(body, active_list_indent)
            else:
                if body.strip():
                    active_list_indent = 0
                parse_body = body
        definition = parse_reference_definition(quote_prefix + parse_body)
        if definition is None:
            offset += len(line)
            index += 1
            continue

        end = offset + len(line)
        target = definition.target
        if target is None and index + 1 < len(lines):
            continuation = lines[index + 1]
            continuation_content = continuation.rstrip("\r\n")
            continuation_prefix, continuation_body = _blockquote_parts(continuation_content)
            if (
                continuation_prefix.count(">") == definition.indent.count(">")
                and continuation_body.startswith((" ", "\t"))
                and continuation_body.strip()
            ):
                target = continuation_body.strip()
                end += len(continuation)
                index += 1

        yield ReferenceDefinition(
            start=offset,
            end=end,
            indent=content[: content.find("[")],
            label=definition.label,
            target=target,
        )
        offset = end
        index += 1


def iter_reference_usages(text: str):
    """Yield balanced full and collapsed reference-link usages."""
    html_ranges = [(start, end) for start, end, _tag, _attrs in _html_tag_spans(text)]
    cursor = 0
    while cursor < len(text):
        label_start = text.find("[", cursor)
        if label_start < 0:
            return
        if _is_escaped(text, label_start):
            cursor = label_start + 1
            continue
        containing_html = next((end for start, end in html_ranges if start <= label_start < end), None)
        if containing_html is not None:
            cursor = containing_html
            continue
        image = label_start > 0 and text[label_start - 1] == "!" and (not _is_escaped(text, label_start - 1))
        start = label_start - 1 if image else label_start
        label_end = _scan_balanced(text, label_start, "[", "]")
        if label_end is None or label_end + 1 >= len(text) or text[label_end + 1] != "[":
            cursor = label_start + 1
            continue
        reference_end = _scan_balanced(text, label_end + 1, "[", "]")
        if reference_end is None:
            cursor = label_start + 1
            continue
        yield ReferenceUsage(
            start=start,
            end=reference_end + 1,
            image=image,
            label=text[label_start + 1 : label_end],
            reference=text[label_end + 2 : reference_end],
        )
        cursor = reference_end + 1


def replace_inline_links(text: str, transform) -> str:
    chunks: list[str] = []
    cursor = 0
    for link in iter_inline_links(text):
        chunks.extend((text[cursor : link.start], transform(link)))
        cursor = link.end
    chunks.append(text[cursor:])
    return "".join(chunks)


def explicit_heading_id(line: str) -> tuple[str, int] | None:
    attr_list = ATTR_LIST_RE.search(line)
    if attr_list is None:
        return None
    identifier = ATTR_ID_RE.search(attr_list.group("attrs"))
    if identifier is None:
        return None
    return identifier.group("id"), attr_list.start()


def _parse_target(target: str) -> tuple[str, str]:
    value = target.strip()
    if value.startswith("<"):
        closing = value.find(">")
        if closing < 0:
            raise ValueError(f"invalid documentation link target: {target}")
        destination = value[1:closing]
        remainder = value[closing + 1 :].strip()
    else:
        parts = value.split(maxsplit=1)
        destination = parts[0]
        remainder = parts[1].strip() if len(parts) == 2 else ""

    if remainder and not (
        len(remainder) >= 2
        and (
            (remainder[0] == remainder[-1] and remainder[0] in {'"', "'"})
            or (remainder[0] == "(" and remainder[-1] == ")")
        )
    ):
        raise ValueError(f"invalid documentation link title: {target}")
    return destination, f" {remainder}" if remainder else ""


def _find_unescaped(value: str, character: str) -> int:
    cursor = 0
    while cursor < len(value):
        if value[cursor] == "\\" and cursor + 1 < len(value):
            cursor += 2
            continue
        if value[cursor] == character:
            return cursor
        cursor += 1
    return -1


def _split_destination(destination: str) -> tuple[str, str, str]:
    fragment_at = _find_unescaped(destination, "#")
    before_fragment = destination if fragment_at < 0 else destination[:fragment_at]
    query_at = _find_unescaped(before_fragment, "?")
    path_end = query_at if query_at >= 0 else len(before_fragment)
    path = destination[:path_end]
    suffix = destination[path_end:]
    fragment = destination[fragment_at:] if fragment_at >= 0 else ""
    return path, suffix, fragment


def _markdown_unescape(value: str) -> str:
    return MARKDOWN_ESCAPE_RE.sub(r"\1", value)


def _lookup_path(destination: str) -> str:
    path, _, _ = _split_destination(destination)
    return unquote(_markdown_unescape(path))


def _is_absolute_web_path(path: str) -> bool:
    return path.startswith(("/", "//")) or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", path) is not None


def _resolve_repo_target(source: Path, destination: str) -> Path | None:
    path = _lookup_path(destination)
    if destination.startswith(GITHUB_BLOB_PREFIX):
        parts = unquote(urlparse(destination).path).split("/")
        if len(parts) >= 6:
            return (ROOT / Path(*parts[5:])).resolve()
    if _is_absolute_web_path(path):
        return None
    return (source.parent / path).resolve()


def _rewrite_outside_code_spans(line: str, transform) -> str:
    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(line):
        opening = line.find("`", cursor)
        if opening < 0:
            break
        run_end = opening
        while run_end < len(line) and line[run_end] == "`":
            run_end += 1
        width = run_end - opening
        search = run_end
        closing_end = -1
        while search < len(line):
            closing = line.find("`", search)
            if closing < 0:
                break
            candidate_end = closing
            while candidate_end < len(line) and line[candidate_end] == "`":
                candidate_end += 1
            if candidate_end - closing == width:
                closing_end = candidate_end
                break
            search = candidate_end
        if closing_end < 0:
            cursor = run_end
            continue
        spans.append((opening, closing_end))
        cursor = closing_end

    link_spans = [(link.start, link.end) for link in iter_inline_links(line)]
    link_spans.extend((usage.start, usage.end) for usage in iter_reference_usages(line))
    spans = [
        (start, end)
        for start, end in spans
        if not any(link_start <= start and end <= link_end for link_start, link_end in link_spans)
    ]
    if not spans:
        return transform(line)
    chunks: list[str] = []
    cursor = 0
    for start, end in spans:
        chunks.extend((transform(line[cursor:start]), line[start:end]))
        cursor = end
    chunks.append(transform(line[cursor:]))
    return "".join(chunks)


def _line_ending(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    return ""


def _blockquote_parts(line: str) -> tuple[str, str]:
    match = re.match(r"^(?P<prefix>(?: {0,3}>[ \t]?)+)(?P<content>.*)$", line)
    if match is None:
        return "", line
    return match.group("prefix"), match.group("content")


def _indent_columns(value: str) -> int:
    columns = 0
    for character in value:
        if character == " ":
            columns += 1
        elif character == "\t":
            columns += 4 - (columns % 4)
        else:
            break
    return columns


def _prefix_columns(value: str) -> int:
    columns = 0
    for character in value:
        columns += 4 - (columns % 4) if character == "\t" else 1
    return columns


def _strip_indent_columns(value: str, required: int) -> str:
    columns = 0
    cursor = 0
    while cursor < len(value) and columns < required and value[cursor] in {" ", "\t"}:
        width = 1 if value[cursor] == " " else 4 - (columns % 4)
        columns += width
        cursor += 1
        if columns > required:
            return " " * (columns - required) + value[cursor:]
    if cursor < len(value) and value[cursor] == "\t":
        return " " * (4 - (columns % 4)) + value[cursor + 1 :]
    return value[cursor:]


def _project_wiki_heading_ids(text: str) -> str:
    suppressed_ranges: list[tuple[int, int]] = []
    offset = 0
    for block in markdown_blocks(text):
        if block.suppressed:
            suppressed_ranges.append((offset, offset + len(block.text)))
        offset += len(block.text)

    def is_suppressed(position: int) -> bool:
        return any(start <= position < end for start, end in suppressed_ranges)

    lines = text.splitlines(keepends=True)
    projected: list[str] = []
    line_offset = 0
    for index, line in enumerate(lines):
        ending = _line_ending(line)
        content_with_prefix = line[: -len(ending)] if ending else line
        prefix, content = _blockquote_parts(content_with_prefix)
        list_item = NESTED_LIST_ITEM_RE.match(content)
        list_prefix = list_item.group("prefix") if list_item is not None else ""
        heading_content = list_item.group("content") if list_item is not None else content
        explicit_id = explicit_heading_id(heading_content)
        next_is_setext = False
        if index + 1 < len(lines):
            next_ending = _line_ending(lines[index + 1])
            next_line = lines[index + 1][: -len(next_ending)] if next_ending else lines[index + 1]
            next_prefix, next_content = _blockquote_parts(next_line)
            normalized_next_content = next_content
            if list_prefix:
                next_leading = len(next_content) - len(next_content.lstrip(" "))
                if next_leading >= len(list_prefix):
                    normalized_next_content = next_content[len(list_prefix) :]
            next_position = line_offset + len(line) + len(next_prefix)
            next_is_setext = (
                next_prefix.count(">") == prefix.count(">")
                and SETEXT_UNDERLINE_RE.fullmatch(normalized_next_content) is not None
                and not is_suppressed(next_position)
            )
        heading_position = line_offset + len(prefix) + len(list_prefix)
        if (
            explicit_id is not None
            and not is_suppressed(heading_position + explicit_id[1])
            and not is_suppressed(heading_position)
            and (ATX_HEADING_RE.match(heading_content) is not None or next_is_setext)
        ):
            anchor_ending = ending or "\n"
            projected.append(f'{prefix}<a id="{explicit_id[0]}"></a>{anchor_ending}')
            heading_content = heading_content[: explicit_id[1]].rstrip()
            line = f"{prefix}{list_prefix}{heading_content}{ending}"
        projected.append(line)
        line_offset += len(lines[index])
    return "".join(projected)


def _rewrite_reference_definitions(text: str, rewrite_target) -> str:
    chunks: list[str] = []
    cursor = 0
    for definition in iter_reference_definitions(text):
        chunks.append(text[cursor : definition.start])
        original = text[definition.start : definition.end]
        rewritten = rewrite_target(definition.target, is_image=False) if definition.target else None
        if rewritten is None:
            chunks.append(original)
        else:
            chunks.append(f"{definition.indent}[{definition.label}]: {rewritten}{_line_ending(original)}")
        cursor = definition.end
    chunks.append(text[cursor:])
    return "".join(chunks)


def rewrite_markdown(source: Path, text: str, source_map: dict[Path, str], surface: str = "wiki") -> str:
    """Rewrite local Markdown links for a self-contained wiki page."""

    def rewrite_target(target: str, *, is_image: bool) -> str | None:
        target = html_unescape(target)
        destination, title = _parse_target(target)
        path_part, suffix, _ = _split_destination(destination)
        lookup_path = _lookup_path(destination)

        if lookup_path.startswith("/") and not lookup_path.startswith("//"):
            raise ValueError(f"root-relative documentation link is not portable across surfaces in {source}: {target}")

        if _is_absolute_web_path(lookup_path) and not destination.startswith(GITHUB_BLOB_PREFIX):
            return None

        if path_part == "" and suffix:
            return f"{suffix}{title}"

        resolved = _resolve_repo_target(source, destination)
        if resolved is not None and resolved in source_map:
            return f"{source_map[resolved]}{suffix}{title}"

        repository_root = ROOT if source.resolve().is_relative_to(ROOT) else source.resolve().parent
        if is_image and resolved is not None and not resolved.is_relative_to(repository_root):
            raise ValueError(f"local image target is outside the repository in {source}: {target}")

        if is_image and resolved is not None and resolved.is_file():
            if source.resolve().is_relative_to(ROOT) and not resolved.is_relative_to(ROOT / "docs" / "assets"):
                raise ValueError(f"local image target is not published to the wiki in {source}: {target}")
            relative = (
                resolved.relative_to(ROOT / "docs" / "assets")
                if source.resolve().is_relative_to(ROOT)
                else Path(resolved.name)
            )
            prefix = "assets" if surface == "site" else "images"
            return f"{prefix}/{quote(relative.as_posix())}{suffix}{title}"

        if resolved is not None and resolved.exists():
            raise ValueError(f"unmapped local documentation link in {source}: {target}")

        if resolved is not None and not resolved.exists():
            raise ValueError(f"broken local documentation link in {source}: {target}")

        return None

    def replace(link: InlineLink) -> str:
        rewritten = rewrite_target(link.target.strip(), is_image=link.image)
        prefix = "!" if link.image else ""
        if rewritten is None:
            return f"{prefix}[{link.label}]({link.target})"
        return f"{prefix}[{link.label}]({rewritten})"

    def replace_html_value(target: HtmlTarget) -> str:
        if target.attr != "srcset":
            return rewrite_target(target.target, is_image=target.tag != "a") or target.target
        rewritten_candidates = []
        for url, descriptor in _srcset_candidates(target.target):
            rewritten = rewrite_target(url, is_image=True) or url
            rewritten_candidates.append(f"{rewritten} {descriptor}" if descriptor else rewritten)
        return ", ".join(rewritten_candidates)

    def rewrite_html(text_value: str) -> str:
        chunks: list[str] = []
        cursor = 0
        for target in iter_html_targets(text_value):
            chunks.extend((text_value[cursor : target.target_start], replace_html_value(target)))
            cursor = target.target_end
        chunks.append(text_value[cursor:])
        return "".join(chunks)

    rendered: list[str] = []
    projected_text = _project_wiki_heading_ids(text) if surface == "wiki" else text
    for block in markdown_blocks(projected_text):
        chunk = rewrite_html(block.text) if block.html_processable else block.text
        if block.suppressed:
            rendered.append(chunk)
            continue
        chunk = _rewrite_reference_definitions(chunk, rewrite_target)
        rendered.append(replace_inline_links(ATTR_RE.sub("", chunk), replace))
    return "".join(rendered).rstrip() + "\n"


def build(output: Path = DEFAULT_OUTPUT) -> None:
    from .build_docs import render_wiki

    render_wiki(output)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def check() -> None:
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        first_path, second_path = Path(first).resolve(), Path(second).resolve()
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
