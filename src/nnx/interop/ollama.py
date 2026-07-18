"""Experimental Modelfile + GGUF bundle exporter.

The Ollama bundle shape is a directory containing ``model.gguf`` and a
``Modelfile`` that points at it via ``FROM ./model.gguf``. The emitted
Modelfile is syntactically valid, but stock Ollama cannot
load NNx's ``nnx_transformer`` GGUF architecture. The bundle is useful
for patched runtimes and for developing a future compatible conversion.

This module's job is the bundle assembly. Quantization / tokenizer
bookkeeping all live in :mod:`nnx.interop.gguf`; this file is the
thin Modelfile-emission layer on top.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nnx.nn.net.transformer_nn import TransformerNN
    from nnx.nn.params.nn_tokenizer_params import NNTokenizerParams


def _validate_modelfile_inputs(system: str, template: Optional[str], parameters: Optional[dict]) -> None:
    """Validate-at-boundary: Modelfiles are line/token-delimited, so
    unescaped user strings can terminate a triple-quoted block early or
    inject whole directives (FROM/ADAPTER/...) via embedded newlines —
    fail fast (and BEFORE the expensive GGUF write) instead of emitting
    a corrupted Modelfile."""
    for label, text in (("system", system), ("template", template)):
        # An embedded triple-quote terminates the block early; content
        # merely ENDING in quotes merges with the closing delimiter into
        # an early terminator too (e.g. 'x""' renders SYSTEM """x""""",
        # whose first """ scan ends the block at the wrong spot).
        if text and ('"""' in text or text.endswith('"')):
            raise ValueError(
                f"{label} must not contain a triple-quote or end with a double-quote — "
                "it would terminate the Modelfile block early."
            )
    if parameters:
        for key, val in parameters.items():
            if any(ch.isspace() for ch in str(key)):
                raise ValueError(f"parameter key {key!r} must not contain whitespace.")
            vals = val if isinstance(val, (list, tuple)) else [val]
            for item in vals:
                if isinstance(item, str) and ("\n" in item or '"' in item):
                    raise ValueError(f"parameter {key!r} value {item!r} must not contain newlines or double quotes.")


def export_ollama_modelfile(
    transformer_nn: TransformerNN,
    tokenizer: NNTokenizerParams,
    out_dir: str | os.PathLike,
    *,
    system: str = "",
    parameters: Optional[dict] = None,
    template: Optional[str] = None,
    quantization: str = "F16",
    model_name: Optional[str] = None,
) -> str:
    """Emit an experimental ``model.gguf`` + ``Modelfile`` bundle.

    Stock Ollama does not implement the ``nnx_transformer`` GGUF
    architecture. Emission verifies bundle structure only; it does not
    establish runtime compatibility.

    Args:
        transformer_nn: An NNx ``TransformerNN`` instance — the model
            to export.
        tokenizer: Corresponding ``NNTokenizerParams``.
        out_dir: Output directory. Created if it doesn't exist.
        system: Optional system prompt; emitted as a ``SYSTEM ...``
            block (triple-quoted) when non-empty. Must not contain a
            triple-quote or end with a double-quote (Modelfile block
            delimiters — validated, raises ``ValueError``); same
            constraint applies to ``template``.
        parameters: Optional dict of Ollama runtime parameters
            (``{"temperature": 0.8, "top_k": 40}`` etc.). Each entry
            becomes a ``PARAMETER <key> <value>`` line. Lists become
            multiple ``PARAMETER <key> <item>`` lines (Ollama's
            convention for things like ``stop``).
        template: Optional chat template. Emitted as a
            ``TEMPLATE ...`` block (triple-quoted) when set.
        quantization: Forwarded to :func:`write_gguf`. Defaults to F16.
        model_name: Forwarded to :func:`write_gguf` as ``model_name``.

    Returns:
        Absolute path to the emitted ``Modelfile``.
    """
    _validate_modelfile_inputs(system, template, parameters)

    # Local import — keeps the ``nnx.interop`` boot path light when
    # only one of the two exporters is used, and matches the lazy-import
    # style in writer.py.
    from .gguf import write_gguf

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gguf_path = out_dir / "model.gguf"
    write_gguf(
        transformer_nn,
        tokenizer,
        out_path=gguf_path,
        quantization=quantization,
        model_name=model_name,
    )

    modelfile_path = out_dir / "Modelfile"
    # encoding="utf-8" explicit so a non-ASCII SYSTEM prompt / TEMPLATE
    # (Asian-language fine-tunes, emoji prompts) round-trips correctly
    # on Windows pre-PEP-686, where Path.write_text would otherwise fall
    # back to the locale code page (cp1252) and silently mojibake.
    # Matches the convention every other text-mode write in src/nnx
    # carries (`feedback_utf8_explicit_text_opens`).
    modelfile_path.write_text(
        _render_modelfile(system=system, parameters=parameters, template=template),
        encoding="utf-8",
    )
    return str(modelfile_path)


def _render_modelfile(
    *,
    system: str,
    parameters: Optional[dict],
    template: Optional[str],
) -> str:
    """Render the Modelfile text. Separated for testability — the
    emit path is just file I/O; the formatting is the load-bearing bit.

    Modelfile syntax (per the Ollama Modelfile docs): each directive on
    its own line, multi-line values (SYSTEM, TEMPLATE) wrapped in
    triple double-quotes.
    """
    lines: list[str] = []
    lines.append("FROM ./model.gguf")

    if parameters:
        for key, val in parameters.items():
            if isinstance(val, (list, tuple)):
                # Convention: lists like `stop` become repeated PARAMETER lines.
                for item in val:
                    lines.append(f"PARAMETER {key} {_format_parameter_value(item)}")
            else:
                lines.append(f"PARAMETER {key} {_format_parameter_value(val)}")

    if template:
        lines.append(f'TEMPLATE """{template}"""')

    if system:
        lines.append(f'SYSTEM """{system}"""')

    # Trailing newline — many Modelfile parsers (including stock ollama)
    # are forgiving here, but adding one keeps `git diff` clean.
    return "\n".join(lines) + "\n"


def _format_parameter_value(v) -> str:
    """Format a Modelfile PARAMETER value. Strings with whitespace
    get quoted; everything else passes through ``str()``."""
    if isinstance(v, str):
        # Ollama PARAMETER values are space-delimited; quote anything
        # containing whitespace so the parser sees one token.
        if any(ch.isspace() for ch in v):
            return f'"{v}"'
        return v
    if isinstance(v, bool):
        # JSON-style lowercase. ollama's parser accepts both but
        # lowercase is the convention in their examples.
        return "true" if v else "false"
    return str(v)
