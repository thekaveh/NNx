"""GGUF write path for NNx TransformerNN.

GGUF is the on-disk format consumed by llama.cpp / Ollama / LM Studio.
Writing GGUF lets a model trained via NNx (``TransformerNN``) be loaded
by every llama.cpp-derived inference stack — the canonical handoff
format for "trained in NNx, served via Ollama."

Public surface:

* :func:`write_gguf` — write a single GGUF file from a TransformerNN +
  tokenizer, at F32 / F16 / BF16. Sub-F16 quantization (Q4_K_M etc.)
  requires the ``llama-quantize`` C++ binary; we raise a clear error
  pointing at the install path rather than silently producing F16.
* :data:`SUPPORTED_QUANTIZATIONS` — the tuple of quantization strings
  ``write_gguf`` writes natively: ``('F32', 'F16', 'BF16')``. Sub-F16
  types route through ``llama-quantize`` instead.

Optional dependency: install with ``pip install 'thekaveh-nnx[gguf-write]'``
(adds ``gguf>=0.19.0``). The ``gguf`` package is the upstream
``llama-cpp``-maintained Python writer — same library every other
GGUF producer uses, so the artifact is byte-compatible with every
GGUF reader in the ecosystem.
"""

from __future__ import annotations

from .writer import SUPPORTED_QUANTIZATIONS, write_gguf

__all__ = ["write_gguf", "SUPPORTED_QUANTIZATIONS"]
