"""Contrastive text-embedder trainer.

Trains a text encoder on ``(anchor, positive)`` pairs via NT-Xent
loss (the SimCLR objective, already exposed as :func:`nnx.nt_xent_loss`).
In-batch negatives — every other sample in the batch is treated as a
negative — so a batch of 32 pairs gives 31 negatives per anchor with
no extra mining work.

The "backbone" is whatever encodes a ``list[str]`` into a ``(B, D)``
tensor:

  - A :class:`sentence_transformers.SentenceTransformer` (the dominant
    case in practice). We call its ``preprocess`` / ``tokenize`` method
    to materialize input ids, then forward through the model and pull
    ``sentence_embedding`` out of the returned dict.
  - Any plain :class:`torch.nn.Module` whose ``forward(list[str]) ->
    Tensor[B, D]`` matches that signature directly. Useful for
    end-to-end-trained custom embedders and for hermetic tests that
    avoid an HF Hub download.

The two public entry points are:

  - :func:`train_contrastive` — high-level: takes a backbone + a
    :class:`ContrastiveTextDataset` (or raw list of pairs) and runs N
    epochs of NT-Xent. Returns the (in-place-trained) backbone for
    chaining into :func:`nnx.embeddings.export_to_faiss`.
  - :func:`text_contrastive_train_step_factory` — lower-level: returns
    a :class:`nnx.TrainStepFn` that can be passed to
    :meth:`NNModel.train(train_step_fn=...)`. Useful if you want
    NNx's full callback / checkpoint / run-tracking machinery wrapped
    around the contrastive step.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .._step_helpers import finalize_step
from ..nn.nn_model import TrainStepContext, TrainStepFn
from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from ..paradigms.contrastive import nt_xent_loss


class ContrastiveTextDataset(Dataset):
    """Wraps ``(anchor, positive)`` string pairs as a torch ``Dataset``.

    Each ``__getitem__`` returns a 2-tuple of strings (``anchor``,
    ``positive``). The default collate from :class:`torch.utils.data.DataLoader`
    would attempt to stack these into tensors and crash; pair this
    dataset with :func:`pair_collate` (or pass it directly to
    :func:`train_contrastive` which wires the collate for you).

    Args:
        pairs: list of ``(anchor, positive)`` string tuples. Empty
            input raises :class:`ValueError`. Note that
            :func:`train_contrastive` additionally requires >= 2 pairs
            (NT-Xent needs a negative); a 1-pair dataset is accepted
            here only for embedding/inference-style uses.

    Raises:
        ValueError: if ``pairs`` is empty or any entry isn't a 2-tuple
            of strings.
    """

    def __init__(self, pairs: list[tuple[str, str]]):
        if not pairs:
            raise ValueError("ContrastiveTextDataset requires at least one pair")
        for i, p in enumerate(pairs):
            if not isinstance(p, (tuple, list)) or len(p) != 2:
                raise ValueError(f"pair {i} is not a 2-tuple: {p!r}")
            a, b = p
            if not isinstance(a, str) or not isinstance(b, str):
                raise ValueError(f"pair {i} contains non-string entries: ({type(a).__name__}, {type(b).__name__})")
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.pairs[idx]


def pair_collate(batch: Iterable[tuple[str, str]]) -> tuple[list[str], list[str]]:
    """Collate a batch of ``(anchor, positive)`` string tuples into two
    parallel lists ``(anchors, positives)``. Pair this with
    :class:`ContrastiveTextDataset` when building your own
    :class:`DataLoader`.
    """
    anchors: list[str] = []
    positives: list[str] = []
    for item in batch:
        a, p = item
        anchors.append(a)
        positives.append(p)
    return anchors, positives


def _is_sentence_transformer(backbone: Any) -> bool:
    """Duck-type detection: does ``backbone`` look like a
    :class:`sentence_transformers.SentenceTransformer`?

    We avoid a hard ``isinstance`` against ``SentenceTransformer``
    so users without ``sentence-transformers`` installed can still
    import ``nnx.embeddings`` and feed in their own ``nn.Module``
    encoder. The detection is "has preprocess (new API) or tokenize
    (legacy API) AND is an nn.Module" — both behaviors that an SBERT
    model exposes.
    """
    if not isinstance(backbone, torch.nn.Module):
        return False
    return hasattr(backbone, "preprocess") or hasattr(backbone, "tokenize")


def _sbert_preprocess(backbone: Any, texts: list[str]) -> dict:
    """Tokenize ``texts`` via the backbone's preprocess/tokenize call.

    Sentence-transformers ≥3 renamed ``tokenize`` to ``preprocess``;
    older releases only have ``tokenize``. Newer releases keep
    ``tokenize`` as a deprecation shim. We prefer ``preprocess`` when
    available to avoid the DeprecationWarning that fires per batch.
    """
    if hasattr(backbone, "preprocess"):
        return backbone.preprocess(texts)
    return backbone.tokenize(texts)


def _resolve_device(backbone: Any, device: Optional[Union[str, torch.device]]) -> torch.device:
    """Materialize a concrete :class:`torch.device` for ``backbone``.

    Priority: explicit ``device`` arg → ``backbone.device`` (the SBERT
    convention) → first parameter's device → CPU. Centralized here so
    both :func:`embed_texts` and :func:`train_contrastive` pick the
    same device with the same fallback chain (and so pyright's
    None-tracking through the chain stays inside one function).
    """
    if device is not None:
        return torch.device(device)
    sb_device = getattr(backbone, "device", None)
    if isinstance(sb_device, torch.device):
        return sb_device
    if isinstance(sb_device, str):
        return torch.device(sb_device)
    try:
        return next(backbone.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def _encode(backbone: Any, texts: list[str], device: torch.device) -> torch.Tensor:
    """Run ``texts`` through ``backbone`` and return a ``(B, D)`` tensor
    of embeddings on ``device``.

    Sentence-transformers backbones get the ``preprocess`` → ``forward``
    → ``['sentence_embedding']`` dance. Plain ``nn.Module`` backbones
    are invoked directly on the string list — the forward is the
    user's responsibility to make device-aware.

    Gradients flow through this call: the caller (training step) keeps
    autograd enabled. Inference-time callers should wrap with
    ``torch.no_grad()``.
    """
    if _is_sentence_transformer(backbone):
        features = _sbert_preprocess(backbone, texts)
        # Move every input tensor onto the target device. SBERT's
        # forward expects all entries on the same device as the model.
        features = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in features.items()}
        out = backbone(features)
        # The "sentence_embedding" key is the standard SBERT output;
        # this is the contract every SBERT pooling head writes to.
        return out["sentence_embedding"]
    # Plain nn.Module path. Caller's forward defines device handling.
    return backbone(texts)


def embed_texts(
    backbone: Any,
    texts: list[str],
    *,
    batch_size: int = 64,
    device: Optional[Union[str, torch.device]] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Encode ``texts`` with ``backbone`` and return a ``(N, D)`` tensor.

    Runs in ``torch.no_grad()`` + ``eval()`` mode — this is the
    inference helper, not the training one. The trainer drives
    :func:`_encode` directly so gradients flow.

    Args:
        backbone: text encoder — a sentence-transformers model or any
            ``nn.Module`` whose ``forward(list[str]) -> Tensor[B, D]``.
        texts: input strings. May be empty (returns a ``(0, ?)``
            placeholder — the embedding dim isn't known until the
            first forward).
        batch_size: how many texts per forward pass. Default 64.
        device: target device. ``None`` uses the backbone's device
            (sentence-transformers exposes one; plain Modules don't, in
            which case we fall back to the first parameter's device,
            or CPU when the backbone has no parameters).
        normalize: if True, L2-normalize each row so dot products with
            the result are cosine similarities. Default True because
            FAISS's ``IndexFlatIP`` interprets the inner product as a
            similarity score and the standard cosine-by-IP trick is
            normalize-then-IP.

    Returns:
        A ``(N, D)`` ``torch.Tensor`` on ``device``. Detached from
        any autograd graph.
    """
    device = _resolve_device(backbone, device)

    if not texts:
        # No way to know D without a forward; return an empty 2D tensor
        # so callers don't get a confusing 1D shape.
        return torch.empty((0, 0), device=device)

    # Snapshot training-mode for non-destructive restore (matches the
    # convention used by NNModel.predict / evaluate,
    # GenerativeNNModel.generate, diffusion.sample,
    # nnx.viz.activation_map, and nnx.lr_finder).
    was_training = backbone.training
    backbone.eval()
    chunks: list[torch.Tensor] = []
    try:
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                emb = _encode(backbone, chunk, device)
                if normalize:
                    emb = F.normalize(emb, dim=-1)
                chunks.append(emb.detach())
    finally:
        if was_training:
            backbone.train()
    return torch.cat(chunks, dim=0)


def text_contrastive_train_step_factory(*, temperature: float = 0.5) -> TrainStepFn:
    """Build a :class:`TrainStepFn` for text-pair contrastive training.

    This is the text-aware sibling of
    :func:`nnx.simclr_train_step_factory`. The training loader must
    yield ``(anchors: list[str], positives: list[str])`` batches —
    typically by pairing :class:`ContrastiveTextDataset` with
    :func:`pair_collate`.

    The step runs:

      1. Encode anchors through ``model.net`` → ``z1``.
      2. Encode positives through ``model.net`` → ``z2``.
      3. NT-Xent loss across the ``(2B, 2B)`` similarity matrix.
      4. Standard :func:`finalize_step` tail (NaN guard, grad clip,
         optimizer step).

    Args:
        temperature: NT-Xent temperature. Lower sharpens; 0.5 is the
            SimCLR default. Must be > 0.

    Returns:
        A ``TrainStepFn`` suitable for ``NNModel.train(..., train_step_fn=...)``.

    Raises:
        ValueError: at factory-build time if ``temperature`` ≤ 0.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    def step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        m = ctx.model
        m.net.train()
        m.net.zero_grad()

        batch = ctx.batch
        if not (isinstance(batch, (tuple, list)) and len(batch) == 2):
            raise ValueError(
                "text contrastive step expects a batch of "
                "(anchors: list[str], positives: list[str]). Got "
                f"{type(batch).__name__} with {len(batch) if hasattr(batch, '__len__') else '?'} entries."
            )
        anchors, positives = batch
        if not (isinstance(anchors, list) and isinstance(positives, list)):
            raise ValueError(
                "text contrastive step expects list[str] views — pair "
                "ContrastiveTextDataset with pair_collate when constructing "
                "the DataLoader. Got "
                f"anchors={type(anchors).__name__}, positives={type(positives).__name__}."
            )

        z1 = _encode(m.net, anchors, m.device)
        z2 = _encode(m.net, positives, m.device)

        loss = nt_xent_loss(z1, z2, temperature=temperature)
        loss_val = finalize_step(loss, ctx, paradigm="embeddings.contrastive")

        return NNEvaluationDataPoint(
            f1=0.0,
            recall=0.0,
            accuracy=0.0,
            precision=0.0,
            loss=loss_val,
            error=loss_val,
        )

    return step


def train_contrastive(
    backbone: Any,
    dataset: Union[ContrastiveTextDataset, list[tuple[str, str]]],
    *,
    n_epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    temperature: float = 0.05,
    device: Optional[Union[str, torch.device]] = None,
    shuffle: bool = True,
    grad_clip_norm: Optional[float] = 1.0,
    weight_decay: float = 0.0,
    optimizer_cls: type = torch.optim.AdamW,
    verbose: bool = False,
) -> Any:
    """Train ``backbone`` on ``(anchor, positive)`` pairs via NT-Xent.

    High-level wrapper around :func:`nt_xent_loss`. Builds a
    :class:`DataLoader` with :func:`pair_collate`, instantiates an
    optimizer over the backbone's trainable parameters, and runs
    ``n_epochs`` of contrastive updates. The backbone is updated
    in-place AND returned for chaining (e.g., directly into
    :func:`nnx.embeddings.export_to_faiss`).

    For more elaborate setups — callbacks, custom schedulers, multi-
    optimizer training, run.id persistence under ``runs/<id>/`` — use
    :func:`text_contrastive_train_step_factory` with the standard
    :meth:`NNModel.train` driver instead.

    Args:
        backbone: text encoder. Either a
            :class:`sentence_transformers.SentenceTransformer` or any
            ``nn.Module`` whose ``forward(list[str]) -> Tensor[B, D]``.
            Parameters with ``requires_grad=False`` are excluded from
            the optimizer (so :func:`nnx.freeze` composes cleanly).
        dataset: a :class:`ContrastiveTextDataset` or a plain list of
            ``(anchor, positive)`` string tuples (we'll wrap it).
        n_epochs: number of full passes. Default 3 — contrastive
            fine-tuning of a pretrained encoder typically needs few.
        batch_size: pairs per batch. NT-Xent's in-batch-negatives
            scaling means bigger is usually better; 16-64 is typical
            for CPU sanity runs, hundreds for GPU.
        lr: optimizer learning rate. Default 2e-5 (the canonical SBERT
            fine-tune LR).
        temperature: NT-Xent temperature. Default 0.05 (sharper than
            SimCLR's image default — text embedders work in a much
            higher-dim cosine space where small temperature helps).
        device: target device. ``None`` infers from the backbone (its
            ``.device`` if present, else its first parameter's device,
            else CPU).
        shuffle: shuffle the dataset each epoch. Default True.
        grad_clip_norm: global L2 grad-clip norm. ``None`` to disable;
            must be positive otherwise (a non-positive norm zeros every
            gradient). Default 1.0 — text encoders are sensitive to
            gradient spikes early in fine-tuning.
        weight_decay: AdamW weight decay. Default 0.0.
        optimizer_cls: optimizer constructor. Default
            :class:`torch.optim.AdamW`. Receives
            ``(trainable_params, lr=lr, weight_decay=weight_decay)``.
        verbose: print per-epoch mean loss. Default False.

    Returns:
        The (in-place-mutated) ``backbone``.

    Raises:
        ValueError: on a dataset of fewer than 2 pairs, batch_size < 2,
            non-positive epochs, non-positive temperature, or a
            non-positive ``grad_clip_norm`` — NT-Xent needs at least one
            negative, so both the dataset and every batch must carry
            >= 2 pairs.
        FloatingPointError: when the contrastive loss goes non-finite
            mid-training (check lr / temperature / input normalization).
    """
    if n_epochs <= 0:
        raise ValueError(f"n_epochs must be positive, got {n_epochs}")
    if batch_size < 2:
        # NT-Xent needs at least one negative per batch (nt_xent_loss
        # itself raises on B < 2 as the backstop; rejecting here gives
        # the error before any training work happens).
        raise ValueError(f"batch_size must be >= 2 for NT-Xent (got {batch_size}) — each batch needs a negative.")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        # `None` disables clipping; a non-positive norm would otherwise slip
        # past the `is not None` guard below and clip_grad_norm_(..., 0.0)
        # scales every gradient by 0.0/total_norm == 0, silently zeroing all
        # grads so the backbone never learns. Same footgun NNOptimParams
        # guards against; fail fast here before any training work happens.
        raise ValueError(f"grad_clip_norm must be positive or None to disable, got {grad_clip_norm}")

    if isinstance(dataset, list):
        dataset = ContrastiveTextDataset(dataset)
    if len(dataset) < 2:
        # 0 pairs: nothing to train. 1 pair: the only possible batch has
        # no negative, which nt_xent_loss rejects — fail here with the
        # dataset-level message instead.
        raise ValueError(f"train_contrastive needs >= 2 pairs (got {len(dataset)}) — NT-Xent requires negatives.")

    device = _resolve_device(backbone, device)
    backbone.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=pair_collate,
        # Drop the trailing batch ONLY when it would have size 1 — a
        # single pair has no negative (nt_xent_loss raises on B < 2).
        # Larger partial batches carry real contrastive signal and are
        # kept. Together with the batch_size >= 2 and len(dataset) >= 2
        # guards above, this makes size-1 batches unreachable; if a
        # future loader change reintroduced one, nt_xent_loss fails
        # loudly rather than silently no-op'ing.
        drop_last=(len(dataset) > batch_size and len(dataset) % batch_size == 1),
    )

    trainable = [p for p in backbone.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError(
            "backbone has no trainable parameters. Did you accidentally freeze "
            "everything? Use nnx.unfreeze(backbone, '*') to thaw it."
        )
    optimizer = optimizer_cls(trainable, lr=lr, weight_decay=weight_decay)

    backbone.train()
    for epoch in range(n_epochs):
        epoch_losses: list[float] = []
        for anchors, positives in loader:
            optimizer.zero_grad()
            z1 = _encode(backbone, anchors, device)
            z2 = _encode(backbone, positives, device)
            loss = nt_xent_loss(z1, z2, temperature=temperature)
            loss_val = float(loss.detach())
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"non-finite contrastive loss ({loss_val!r}) at epoch "
                    f"{epoch}. Check lr / temperature / input normalization."
                )
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
            optimizer.step()
            epoch_losses.append(loss_val)
        if verbose:
            mean = sum(epoch_losses) / max(1, len(epoch_losses))
            print(f"[epoch {epoch + 1}/{n_epochs}] mean NT-Xent loss = {mean:.4f}")

    return backbone
