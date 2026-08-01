"""Derive published SVG and PNG assets from standalone HTML masters."""

from __future__ import annotations

import argparse
import hashlib
import struct
import textwrap
import zlib
from html.parser import HTMLParser
from pathlib import Path

import cairosvg

ROOT = Path(__file__).resolve().parents[2]
DIAGRAMS = {
    ROOT / "docs" / "architecture.html": ROOT / "docs" / "assets" / "architecture.svg",
    ROOT / "docs" / "diagrams" / "training-lifecycle.html": ROOT / "docs" / "assets" / "training-lifecycle.svg",
    ROOT / "docs" / "diagrams" / "docs-projection.html": ROOT / "docs" / "assets" / "docs-projection.svg",
}


class _SvgBoundsParser(HTMLParser):
    def __init__(self, text: str) -> None:
        super().__init__(convert_charrefs=False)
        self.text = text
        self.line_offsets = [0]
        for line in text.splitlines(keepends=True):
            self.line_offsets.append(self.line_offsets[-1] + len(line))
        self.depth = 0
        self.start: int | None = None
        self.end: int | None = None

    def _offset(self) -> int:
        line, column = self.getpos()
        return self.line_offsets[line - 1] + column

    def handle_starttag(self, tag: str, _attrs) -> None:
        if tag.lower() != "svg" or self.end is not None:
            return
        if self.depth == 0:
            self.start = self._offset()
        self.depth += 1

    def handle_startendtag(self, tag: str, _attrs) -> None:
        if tag.lower() == "svg" and self.depth == 0 and self.end is None:
            self.start = self._offset()
            raw_tag = self.get_starttag_text()
            if raw_tag is not None:
                self.end = self.start + len(raw_tag)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "svg" or self.depth == 0 or self.end is not None:
            return
        self.depth -= 1
        if self.depth == 0:
            closing = self.text.find(">", self._offset())
            self.end = len(self.text) if closing < 0 else closing + 1


def render(master: Path) -> str:
    text = master.read_text(encoding="utf-8")
    parser = _SvgBoundsParser(text)
    parser.feed(text)
    if parser.start is None or parser.end is None:
        raise ValueError(f"no inline SVG found in {master}")
    return textwrap.dedent(text[parser.start : parser.end]).strip() + "\n"


def render_png(svg: str) -> bytes:
    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=1200)
    if png is None:
        raise RuntimeError("CairoSVG did not return PNG bytes")
    source_hash = hashlib.sha256(svg.encode("utf-8")).hexdigest().encode("ascii")
    data = b"nnx-svg-sha256\x00" + source_hash
    chunk_type = b"tEXt"
    chunk = struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", zlib.crc32(chunk_type + data))
    return png[:33] + chunk + png[33:]


def png_matches_source(png: bytes, svg: str) -> bool:
    if not png.startswith(b"\x89PNG\r\n\x1a\n") or len(png) < 33 or png[12:16] != b"IHDR":
        return False
    width, height = struct.unpack(">II", png[16:24])
    source_hash = hashlib.sha256(svg.encode("utf-8")).hexdigest().encode("ascii")
    return width == 1200 and height > 0 and b"nnx-svg-sha256\x00" + source_hash in png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale: list[Path] = []
    for master, output in DIAGRAMS.items():
        rendered = render(master)
        png_output = output.with_suffix(".png")
        if args.check:
            if not output.exists() or output.read_text(encoding="utf-8") != rendered:
                stale.append(output)
            if not png_output.exists() or not png_matches_source(png_output.read_bytes(), rendered):
                stale.append(png_output)
        else:
            output.write_text(rendered, encoding="utf-8")
            png_output.write_bytes(render_png(rendered))
    if stale:
        raise SystemExit(f"stale diagram assets: {', '.join(map(str, stale))}")


if __name__ == "__main__":
    main()
