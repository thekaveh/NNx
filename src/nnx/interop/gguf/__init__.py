"""GGUF write path for NNx TransformerNN.

GGUF is a container format used by llama.cpp-derived tools. NNx writes
``TransformerNN`` under the custom ``nnx_transformer`` architecture for
inspection, archival exchange, or runtimes explicitly patched for NNx.
Stock llama.cpp, Ollama, and LM Studio do not implement that architecture.

Public surface:

* :func:`write_gguf` — write a single GGUF file from a TransformerNN +
  tokenizer, at F32 / F16 / BF16. Sub-F16 quantization (Q4_K_M etc.)
  requires the ``llama-quantize`` C++ binary; we raise a clear error
  pointing at the official llama.cpp build path rather than silently
  producing F16.
* :data:`SUPPORTED_QUANTIZATIONS` — the tuple of quantization strings
  ``write_gguf`` writes natively: ``('F32', 'F16', 'BF16')``. Sub-F16
  types route through ``llama-quantize`` instead.

Optional dependency: install with ``pip install 'thekaveh-nnx[gguf-write]'``
(adds ``gguf>=0.19.0``). The ``gguf`` package is the upstream
``llama-cpp``-maintained Python writer. Generic parsers can inspect the
container, but executable compatibility remains architecture-specific.
"""

from __future__ import annotations

from .writer import SUPPORTED_QUANTIZATIONS, write_gguf

__all__ = ["write_gguf", "SUPPORTED_QUANTIZATIONS"]
