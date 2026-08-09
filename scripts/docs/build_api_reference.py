"""Render a portable Markdown API catalog from runtime signatures and docstrings."""

from __future__ import annotations

import argparse
import importlib
import inspect
import re
from enum import Enum
from pathlib import Path
from typing import Any, get_origin

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "scripts" / "docs" / "api_reference_catalog.txt"
OUTPUT = ROOT / "docs" / "api.md"
ENTRY = re.compile(r"^@api\s+(\S+?)(?:\s+members=([A-Za-z0-9_,]+))?\s*$")


def _documentation(value: Any) -> tuple[str, str]:
    if get_origin(value) is not None:
        return "Public type alias.", ""
    if isinstance(value, Enum):
        return f"Enum value `{value.value}`.", ""
    if inspect.isclass(value) and issubclass(value, Enum):
        own_doc = value.__dict__.get("__doc__")
        if not own_doc or own_doc == "An enumeration.":
            names = ", ".join(f"`{name}`" for name in value.__members__)
            return f"Enum values: {names}.", ""
        doc = inspect.cleandoc(own_doc)
        paragraphs = re.split(r"\n\s*\n", doc, maxsplit=1)
        summary = " ".join(line.strip() for line in paragraphs[0].splitlines())
        details = paragraphs[1].strip() if len(paragraphs) == 2 else ""
        return summary, details
    if not callable(value) and not inspect.ismodule(value) and not isinstance(value, property):
        return "Exported value.", ""
    doc = inspect.getdoc(value) or "No public description is currently available."
    paragraphs = re.split(r"\n\s*\n", doc, maxsplit=1)
    summary = " ".join(line.strip() for line in paragraphs[0].splitlines())
    details = paragraphs[1].strip() if len(paragraphs) == 2 else ""
    return summary, details


def _stable_annotation(annotation: Any) -> Any:
    if annotation is inspect.Signature.empty:
        return annotation
    if isinstance(annotation, str):
        return annotation
    forward_arg = getattr(annotation, "__forward_arg__", None)
    if isinstance(forward_arg, str):
        return forward_arg
    module = getattr(annotation, "__module__", None)
    qualname = getattr(annotation, "__qualname__", None)
    if isinstance(module, str) and isinstance(qualname, str):
        return f"{module}.{qualname}"
    return str(annotation).removeprefix("typing.")


def _signature(path: str, value: Any) -> str:
    if get_origin(value) is not None:
        return f"type alias {path}"
    if isinstance(value, Enum):
        return f"{path} = {value.value!r}"
    if inspect.isclass(value) and issubclass(value, Enum):
        return f"class {path}(Enum)"
    if inspect.ismodule(value):
        return f"module {path}"
    if isinstance(value, property):
        return f"property {path}"
    if not callable(value):
        return f"{path} = {value!r}"
    try:
        raw_signature = inspect.signature(value)
        parameters = [
            parameter.replace(annotation=_stable_annotation(parameter.annotation))
            for parameter in raw_signature.parameters.values()
        ]
        normalized = raw_signature.replace(
            parameters=parameters,
            return_annotation=_stable_annotation(raw_signature.return_annotation),
        )
        signature = re.sub(r"<class '([^']+)'>", r"\1", str(normalized))
    except (TypeError, ValueError):
        signature = ""
    prefix = "class " if inspect.isclass(value) else ""
    return f"{prefix}{path}{signature}"


def _locate(path: str) -> Any | None:
    parts = path.split(".")
    for module_end in range(len(parts), 0, -1):
        try:
            value: Any = importlib.import_module(".".join(parts[:module_end]))
        except ModuleNotFoundError:
            continue
        for attribute in parts[module_end:]:
            if not hasattr(value, attribute):
                break
            value = getattr(value, attribute)
        else:
            return value
    return None


def _class_members(value: type) -> list[tuple[str, Any]]:
    members: list[tuple[str, Any]] = []
    enum_member_names: set[str] = set()
    if issubclass(value, Enum):
        enum_member_names = set(value.__members__)
        members.extend((name, member) for name, member in value.__members__.items())
    for name, descriptor in value.__dict__.items():
        if name.startswith("_") or name in enum_member_names:
            continue
        member = getattr(value, name)
        if callable(member) or isinstance(descriptor, property):
            members.append((name, descriptor if isinstance(descriptor, property) else member))
    return members


def _render_entry(path: str, value: Any, *, heading_level: int = 4) -> list[str]:
    summary, details = _documentation(value)
    rendered = [
        f"{'#' * heading_level} `{path}`",
        "",
        "```python",
        _signature(path, value),
        "```",
        "",
        summary,
        "",
    ]
    if details:
        rendered.extend(("**Details**", "", "```text", details, "```", ""))
    return rendered


def render() -> str:
    rendered: list[str] = []
    for line in CATALOG.read_text(encoding="utf-8").splitlines():
        match = ENTRY.fullmatch(line)
        if match is None:
            rendered.append(line)
            continue
        path = match.group(1)
        explicit_members = match.group(2).split(",") if match.group(2) else None
        value = _locate(path)
        if value is None:
            raise ValueError(f"cannot resolve API catalog entry: {path}")
        rendered.extend(_render_entry(path, value))
        if explicit_members is not None:
            members = [(name, getattr(value, name)) for name in explicit_members]
        elif inspect.isclass(value):
            members = _class_members(value)
        else:
            members = []
        for name, member in members:
            rendered.extend(_render_entry(f"{path}.{name}", member, heading_level=5))
    return "\n".join(rendered).rstrip() + "\n"


def build(*, check: bool = False) -> None:
    expected = render()
    if check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != expected:
            raise SystemExit("docs/api.md is stale; run python -m scripts.docs.build_api_reference")
        return
    OUTPUT.write_text(expected, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    build(check=args.check)


if __name__ == "__main__":
    main()
