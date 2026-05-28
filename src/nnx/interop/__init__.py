"""nnx.interop — exporters and bridges to external runtime stacks.

Currently houses:

* ``nnx.interop.gguf`` — GGUF (llama.cpp / Ollama) writer for
  ``TransformerNN``. Optional dep: ``pip install 'nnx[gguf-write]'``.
* ``nnx.interop.ollama`` — Modelfile + GGUF bundle emitter for
  ``ollama create``.

Both submodules import lazily so that ``import nnx.interop`` works even
when the optional ``gguf`` package isn't installed. The actual import
of ``gguf`` happens inside ``write_gguf`` / ``export_ollama_modelfile``
and raises a clear ImportError with install instructions.
"""

from __future__ import annotations

# Re-export the high-level writer at the subpackage root so callers
# can do ``from nnx.interop import write_gguf``. The function itself
# does the lazy ``import gguf`` and raises ImportError with install
# instructions when the dep is missing.
from .gguf import write_gguf
from .ollama import export_ollama_modelfile

__all__ = ["write_gguf", "export_ollama_modelfile"]
