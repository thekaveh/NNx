"""Domain-specific text embedder training (contrastive).

The training-time half of NNx's RAG-adjacent surface. Users train a
domain-specific text embedder via the existing SimCLR / NT-Xent
machinery exposed by ``nnx.paradigms.contrastive``. The companion
FAISS-export surface lands separately in this same module.

NNx does NOT host the RAG stack itself — chunkers, rerankers, prompt
orchestration, vector-DB clients are inference-time concerns and live
downstream.

Public surface — re-exported from the top-level ``nnx`` package as
``nnx.embeddings``:

  - :class:`ContrastiveTextDataset` — wraps a list of
    ``(anchor, positive)`` string pairs into a :class:`torch.utils.data.Dataset`.
  - :func:`train_contrastive` — high-level training loop. Takes a
    text-encoder backbone, runs NT-Xent contrastive fine-tuning, returns
    the trained backbone.
  - :func:`text_contrastive_train_step_factory` — lower-level
    :class:`nnx.TrainStepFn` factory, for users who want to drive
    contrastive text training through :meth:`NNModel.train` directly
    instead of the high-level helper.
  - :func:`embed_texts` — encode a list of strings into a normalized
    ``(N, D)`` tensor using the supplied backbone. Inference helper
    shared with the (forthcoming) FAISS-export surface.

Optional dependencies — install via the ``embeddings`` extra:

    pip install "nnx[embeddings]"

The extra pulls ``sentence-transformers`` (the canonical backbone
source) and ``faiss-cpu`` (used by the FAISS-export surface in the
same module). Both are optional at import time; this module imports
cleanly without them and raises a clear :class:`ImportError` only at
the call site that actually needs them.
"""

from __future__ import annotations

from .contrastive_trainer import (
    ContrastiveTextDataset,
    embed_texts,
    text_contrastive_train_step_factory,
    train_contrastive,
)

__all__ = [
    "ContrastiveTextDataset",
    "embed_texts",
    "text_contrastive_train_step_factory",
    "train_contrastive",
]
