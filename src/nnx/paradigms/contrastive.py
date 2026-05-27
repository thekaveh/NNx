"""SimCLR-style contrastive training.

The standard SimCLR recipe: produce two augmented views of each input,
encode both through a shared backbone, project to an embedding space,
and pull positive pairs (same source) together while pushing apart
negative pairs (different sources) via the NT-Xent loss.

This module ships the **loss** (:func:`nt_xent_loss`) and the **step**
factory (:func:`simclr_train_step_factory`). The augmentation pipeline
that produces the two views is the caller's responsibility — typically
via a paired-view :class:`Dataset` whose ``__getitem__`` returns
``(view1, view2)`` for the same source sample.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .._step_helpers import finalize_step
from ..nn.nn_model import TrainStepContext, TrainStepFn
from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    temperature: float = 0.5,
) -> torch.Tensor:
    """SimCLR's Normalized Temperature-scaled cross-entropy loss.

    Args:
        z1: ``(B, D)`` embeddings of the first view of each sample.
        z2: ``(B, D)`` embeddings of the second view.
        temperature: divisor on the cosine similarity. Lower T sharpens
            the distribution; 0.5 is the SimCLR default. Must be > 0.

    Returns:
        Scalar loss tensor (mean across the 2B positions in the batch).

    Raises:
        ValueError: if shapes mismatch or ``temperature`` ≤ 0.
    """
    if z1.shape != z2.shape:
        raise ValueError(f"z1 / z2 shape mismatch: {z1.shape} vs {z2.shape}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    B = z1.shape[0]
    # L2-normalize so the matmul gives cosine similarity directly.
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    # Stack into (2B, D) and form the (2B, 2B) similarity matrix.
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t()) / temperature

    # Mask out self-similarity (the diagonal) — every row's logits
    # otherwise have a +1/T entry for the row's own embedding, which
    # would always be the most-similar position and crush the loss.
    self_mask = torch.eye(2 * B, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(self_mask, float("-inf"))

    # Positives: row i in [0, B) has positive at column i+B; row j in
    # [B, 2B) has positive at column j-B. Cross-entropy of the
    # softmax over rejected self-positions, targeting those columns.
    targets = torch.cat(
        [torch.arange(B, 2 * B, device=sim.device),
         torch.arange(0, B, device=sim.device)]
    )
    return F.cross_entropy(sim, targets)


def _unpack_views(batch):
    """Pull (view1, view2) out of a batch. Accepts either a plain
    ``(x1, x2)`` tuple or the conventional ``((x1, x2), y)`` shape
    where ``y`` is unused — SimCLR is label-free."""
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2 and isinstance(batch[0], (list, tuple)) and len(batch[0]) == 2:
            return batch[0][0], batch[0][1]
        if len(batch) >= 2 and torch.is_tensor(batch[0]) and torch.is_tensor(batch[1]):
            return batch[0], batch[1]
    raise ValueError(
        "SimCLR step expects a batch of (view1, view2) tensors. Got "
        f"{type(batch).__name__} with element types "
        f"{[type(b).__name__ for b in batch] if isinstance(batch, (list, tuple)) else 'scalar'}."
    )


def simclr_train_step_factory(*, temperature: float = 0.5) -> TrainStepFn:
    """Build a SimCLR :class:`TrainStepFn`.

    Args:
        temperature: temperature in :func:`nt_xent_loss`. 0.5 default.

    Returns:
        A ``TrainStepFn`` for ``NNModel.train(..., train_step_fn=...)``.
        The training loader MUST yield batches of two augmented views
        per source sample — typically ``(view1, view2)`` tensors, or
        ``((view1, view2), y_unused)`` when reusing a labelled dataset.
        ``model.net`` is invoked once per view (no batch-doubling) so
        BatchNorm statistics see one view at a time; users who want
        all-at-once normalization can stack the views and forward once.

        **Sharp edge:** a labeled ``(X, Y)`` batch from a standard
        ``TensorDataset`` will silently be interpreted as
        ``(view1=X, view2=Y)`` and produce a shape-mismatch in
        :func:`nt_xent_loss`. Use a paired-view dataset whose
        ``__getitem__`` returns ``(view1, view2)`` instead.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    def step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        m = ctx.model
        m.net.train()
        m.net.zero_grad()

        x1, x2 = _unpack_views(ctx.batch)
        x1 = x1.to(m.device)
        x2 = x2.to(m.device)

        z1 = m.net(x1)
        z2 = m.net(x2)

        loss = nt_xent_loss(z1, z2, temperature=temperature)
        loss_val = finalize_step(loss, ctx, paradigm="simclr")

        # No classification metric for a contrastive paradigm — report
        # the loss in both slots so BEST tracking and ReduceLROnPlateau
        # have a single signal to lock onto.
        return NNEvaluationDataPoint(
            f1=0.0, recall=0.0, accuracy=0.0, precision=0.0,
            loss=loss_val,
            error=loss_val,
        )

    return step
