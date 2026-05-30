"""Domain-specific text embedder training + FAISS index export.

The only RAG-adjacent piece NNx ships. Users train a domain-specific
text embedder via the existing SimCLR / NT-Xent machinery, then export
to FAISS for ANY retrieval-augmented-generation stack (LangChain,
LlamaIndex, Haystack, raw FAISS) to consume.

NNx does NOT host the RAG stack itself — chunkers, rerankers, prompt
orchestration, vector-DB clients are inference-time concerns and live
downstream. The job ends at the FAISS index on disk.

Public surface — re-exported from the top-level ``nnx`` package as
``nnx.embeddings``:

  - :class:`ContrastiveTextDataset` — wraps a list of
    ``(anchor, positive)`` string pairs into a :class:`torch.utils.data.Dataset`.
  - :func:`pair_collate` — :class:`torch.utils.data.DataLoader`
    ``collate_fn`` that splits a batch of ``(anchor, positive)`` tuples
    into ``(anchors: list[str], positives: list[str])``. Pair with
    :class:`ContrastiveTextDataset` when you build your own loader
    instead of going through :func:`train_contrastive`.
  - :func:`train_contrastive` — high-level training loop. Takes a
    text-encoder backbone, runs NT-Xent contrastive fine-tuning, returns
    the trained backbone.
  - :func:`text_contrastive_train_step_factory` — lower-level
    :class:`nnx.TrainStepFn` factory, for users who want to drive
    contrastive text training through :meth:`NNModel.train` directly
    instead of the high-level helper.
  - :func:`export_to_faiss` — embed a corpus and write a FAISS index
    file. Reloadable by any FAISS-aware retriever.
  - :func:`export_to_safetensors` — persist the backbone's state for
    HuggingFace Hub / sentence-transformers interop. Falls back to
    plain :func:`torch.save` when ``safetensors`` isn't installed.
  - :func:`embed_texts` — encode a list of strings into a normalized
    ``(N, D)`` tensor using the supplied backbone. Shared between the
    trainer's similarity probes and :func:`export_to_faiss`.

Optional dependencies — install via the ``embeddings`` extra:

    pip install "nnx[embeddings]"

The extra pulls ``faiss-cpu`` (for the FAISS index export) and
``sentence-transformers`` (the canonical backbone source). Both are
optional at import time; this module imports cleanly without them and
raises a clear :class:`ImportError` only at the call site that actually
needs them.
"""

from __future__ import annotations

from .contrastive_trainer import (
    ContrastiveTextDataset,
    embed_texts,
    pair_collate,
    text_contrastive_train_step_factory,
    train_contrastive,
)
from .faiss_export import export_to_faiss, export_to_safetensors

__all__ = [
    "ContrastiveTextDataset",
    "embed_texts",
    "export_to_faiss",
    "export_to_safetensors",
    "pair_collate",
    "text_contrastive_train_step_factory",
    "train_contrastive",
]
