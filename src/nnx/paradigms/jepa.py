"""I-JEPA: Joint Embedding Predictive Architecture (Meta CVPR 2023).

Predict in latent space, not pixel space. No decoder.  Avoids
representation collapse via an EMA target encoder (gradients never
flow through it). Two backbones cooperate:

  * **Context encoder** = ``model.net`` (a ViT). Trained by gradient
    descent. Sees only the unmasked patches of each image.
  * **Target encoder** = an exponentially-moving-average copy of the
    context encoder. Sees the full image (no mask) and produces
    target embeddings under ``no_grad``. Updated once per step by
    :func:`update_ema`, never by the optimizer.

The **predictor** is a small ViT-like network that maps
``(context_embeds, target_positions) -> predicted_target_embeds``.
Loss is the L2 distance between predicted and actual target
embeddings on the held-out positions.

The factory in :func:`jepa_train_step_factory` returns a
:class:`nnx.TrainStepFn` that plugs into :meth:`NNModel.train` via
the ``train_step_fn=`` hook.

The mask helper :func:`random_block_mask` ships with the module —
it's the canonical I-JEPA mask sampler (a single rectangular block
in the patch grid, dropped from the context view, kept as the
prediction target).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .._step_helpers import finalize_step
from ..nn.nn_model import TrainStepContext, TrainStepFn
from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint


def build_target_encoder(source: nn.Module) -> nn.Module:
    """Deep-copy ``source``, freeze every parameter, return the copy.

    The target encoder is updated **only** via :func:`update_ema` after
    each optimizer step. Freezing here is belt-and-braces — even if a
    user accidentally hands the target into an optimizer that scans
    ``parameters()``, ``requires_grad=False`` keeps the gradients off
    and the optimizer's state empty for those tensors.
    """
    target = copy.deepcopy(source)
    for p in target.parameters():
        p.requires_grad = False
    target.eval()
    return target


def update_ema(source: nn.Module, target: nn.Module, momentum: float) -> None:
    """In-place EMA update: ``target ← momentum * target + (1 - momentum) * source``.

    Called once per training step from inside the JEPA train_step_fn.
    Runs under ``torch.no_grad`` so the EMA tensors do not become part
    of the autograd graph — the target encoder is supposed to be a
    detached snapshot.

    Args:
        source: the trainable module (i.e., ``model.net``).
        target: the EMA copy returned by :func:`build_target_encoder`.
            Mutated in place.
        momentum: EMA decay in ``[0, 1)``. Higher = slower target
            tracking. I-JEPA's reference recipe uses 0.996 with a
            cosine schedule up to 1.0 over training; the factory's
            default matches.

    Raises:
        ValueError: when ``momentum`` is outside ``[0, 1)``.
    """
    if not (0.0 <= momentum < 1.0):
        raise ValueError(f"EMA momentum must be in [0, 1), got {momentum}")
    # Walk by named-parameter so the EMA continues to work when the
    # *source* has been augmented with extra submodules (typical
    # idiom: register the JEPA predictor under ``model.net`` so a
    # single optimizer picks it up). The target encoder is a frozen
    # snapshot of the original encoder; any source param whose name
    # is absent from the target is something the EMA was never
    # responsible for and is silently skipped.
    src_by_name = dict(source.named_parameters())
    with torch.no_grad():
        for name, pt in target.named_parameters():
            ps = src_by_name.get(name)
            if ps is None:
                raise KeyError(
                    f"EMA target param {name!r} has no counterpart in the source; "
                    "did you swap the source network's structure after building "
                    "the target encoder?"
                )
            pt.mul_(momentum).add_(ps, alpha=1.0 - momentum)


def random_block_mask(
    *,
    n_patches: int,
    grid_size: int,
    block_scale: tuple[float, float] = (0.15, 0.2),
    block_aspect: tuple[float, float] = (0.75, 1.5),
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one I-JEPA-style rectangular block mask on a patch grid.

    Returns ``(context_mask, target_mask)`` where:

      * ``context_mask: BoolTensor[n_patches]`` — True at positions
        kept by the context encoder (i.e., NOT in the target block).
      * ``target_mask: BoolTensor[n_patches]`` — True at positions
        the predictor is asked to predict (i.e., inside the target
        block, exactly the complement of context_mask).

    The block is a single rectangle of randomly-sampled width/height
    drawn from ``block_scale`` × n_patches with an aspect ratio in
    ``block_aspect``. Reference I-JEPA samples 4 target blocks per
    image; this helper samples 1 — enough for the verify-the-plumbing
    example we ship. Users can compose multiple calls if they want
    the 4-block recipe.

    Args:
        n_patches: total number of patch tokens. Must equal
            ``grid_size**2``.
        grid_size: width (= height) of the patch grid. The
            rectangular block is sampled in this coordinate system.
        block_scale: ``(min, max)`` fraction of ``n_patches`` covered
            by the block. Default ``(0.15, 0.2)`` mirrors I-JEPA.
        block_aspect: ``(min, max)`` width/height ratio.
        generator: optional ``torch.Generator`` for reproducibility.
        device: device on which the masks are placed. ``None`` →
            default tensor device (CPU).

    Returns:
        A pair of ``BoolTensor``s, both 1-D length ``n_patches``.

    Raises:
        ValueError: when ``grid_size**2 != n_patches``, or when the
            sampled block would be empty / larger than the grid.
    """
    if grid_size * grid_size != n_patches:
        raise ValueError(f"grid_size**2 ({grid_size**2}) != n_patches ({n_patches})")

    # Sample block scale and aspect. ``torch.rand`` accepts a
    # generator directly; ``Tensor.uniform_`` does not, which is why
    # we go through rand here instead of uniform_.
    u_scale = torch.rand(1, generator=generator).item()
    u_aspect = torch.rand(1, generator=generator).item()
    scale = block_scale[0] + (block_scale[1] - block_scale[0]) * u_scale
    aspect = block_aspect[0] + (block_aspect[1] - block_aspect[0]) * u_aspect
    block_area = scale * n_patches
    h = max(1, int(round(math.sqrt(block_area / aspect))))
    w = max(1, int(round(math.sqrt(block_area * aspect))))
    h = min(h, grid_size)
    w = min(w, grid_size)
    # Sample top-left corner. The block must fit inside the grid.
    top = int(torch.randint(0, grid_size - h + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, grid_size - w + 1, (1,), generator=generator).item())

    grid = torch.zeros(grid_size, grid_size, dtype=torch.bool, device=device)
    grid[top : top + h, left : left + w] = True
    target_mask = grid.flatten()  # (n_patches,)
    context_mask = ~target_mask
    return context_mask, target_mask


def _broadcast_mask(mask_1d: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Broadcast a single-image mask across a batch. ``mask_1d`` is
    ``BoolTensor[n_patches]``; result is ``BoolTensor[B, n_patches]``.

    JEPA's plain recipe uses one mask per *step* (shared across the
    batch). Sampling a fresh mask per sample would be more diverse but
    also forces the ViT to handle ragged kept-counts, which the encoder
    doesn't support. Shared-per-step is the configuration I-JEPA's
    public reference uses too.
    """
    return mask_1d.unsqueeze(0).expand(batch_size, -1).contiguous()


class JEPAPredictor(nn.Module):
    """Tiny ViT-like predictor: ``(context_embeds, target_positions)
    -> predicted_target_embeds``.

    Architecture: project context_embeds to ``predictor_dim``,
    concatenate learnable mask tokens (one per target position) plus
    that position's positional embedding, run a few ViT blocks, project
    back to ``embed_dim``, return the predictions at the target
    positions only.

    Kept deliberately small — the reference I-JEPA predictor is also
    much narrower than the encoder. For our CIFAR-shape demo, two
    blocks at ``predictor_dim = embed_dim // 2`` is enough plumbing
    to verify the loss decreases without dominating wall-clock time.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        n_patches: int,
        predictor_dim: Optional[int] = None,
        n_layers: int = 2,
        n_heads: int = 2,
        ffn_mult: int = 4,
    ):
        super().__init__()
        # Local import keeps the paradigm module independent of the
        # vit_nn module ordering (the only reason ViTBlock lives where
        # it does is its use of RMSNorm + SwiGLU from the transformer
        # building blocks).
        from ..nn.net.vit_nn import ViTBlock

        if predictor_dim is None:
            predictor_dim = max(8, embed_dim // 2)
        self.embed_dim = embed_dim
        self.predictor_dim = predictor_dim
        self.n_patches = n_patches

        # In/out projections so the predictor can run at a smaller
        # internal width than the encoder.
        self.in_proj = nn.Linear(embed_dim, predictor_dim, bias=False)
        self.out_proj = nn.Linear(predictor_dim, embed_dim, bias=False)
        # Per-patch positional embedding shared with the encoder *in
        # spirit* but learned independently — the predictor doesn't
        # have access to the encoder's tied weights.
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, predictor_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # Learnable mask token — same value at every target position.
        # The predictor disambiguates "which patch are we predicting?"
        # via the additive positional embedding.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.blocks = nn.ModuleList(
            [ViTBlock(d_model=predictor_dim, n_heads=n_heads, ffn_mult=ffn_mult) for _ in range(n_layers)]
        )

    def forward(
        self,
        context_embeds: torch.Tensor,
        context_positions: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict embeddings at ``target_positions`` from
        ``context_embeds``.

        Args:
            context_embeds: ``(B, T_ctx, embed_dim)``. The CLS token
                produced by the encoder is included as the first entry
                (position 0).
            context_positions: ``LongTensor[T_ctx]`` — positions of
                the kept context tokens *including* CLS at index 0.
            target_positions: ``LongTensor[T_tgt]`` — positions of the
                target patches to predict (1..n_patches).

        Returns:
            ``(B, T_tgt, embed_dim)`` predicted target embeddings.
        """
        B = context_embeds.shape[0]
        x_ctx = self.in_proj(context_embeds)  # (B, T_ctx, predictor_dim)
        # Add the predictor's own positional embedding to context tokens.
        x_ctx = x_ctx + self.pos_embed[:, context_positions, :]
        # Build mask tokens at target positions.
        T_tgt = target_positions.shape[0]
        x_tgt = self.mask_token.expand(B, T_tgt, -1) + self.pos_embed[:, target_positions, :]
        # Concatenate and pass through the small transformer.
        x = torch.cat([x_ctx, x_tgt], dim=1)
        for block in self.blocks:
            x = block(x)
        # Slice out only the target-position predictions and project back.
        T_ctx = x_ctx.shape[1]
        x_tgt_out = x[:, T_ctx:, :]
        return self.out_proj(x_tgt_out)


def jepa_train_step_factory(
    target_encoder: nn.Module,
    predictor: nn.Module,
    mask_fn: Callable[[int, torch.device], tuple[torch.Tensor, torch.Tensor]],
    *,
    ema_momentum: float = 0.996,
) -> TrainStepFn:
    """Build an I-JEPA :class:`TrainStepFn`.

    Per step:

      1. Sample ``(context_mask, target_mask)`` for the batch via
         ``mask_fn(n_patches, device)``. Both are 1-D
         ``BoolTensor[n_patches]`` and **complementary** — every
         patch is either context or target.
      2. Forward each input image through ``model.net`` with the
         context mask, producing ``(B, T_ctx + 1, d_model)`` context
         embeddings (CLS at index 0).
      3. Forward the full image (no mask) through ``target_encoder``
         under ``no_grad`` to produce target embeddings. Slice out
         the positions in ``target_mask`` only.
      4. Predict ``(B, T_tgt, d_model)`` from context via
         ``predictor``.
      5. MSE loss against the target embeddings.
      6. :func:`finalize_step` — NaN guard, optimizer step, grad clip.
      7. :func:`update_ema` — EMA-update the target encoder from
         ``model.net``.

    Args:
        target_encoder: an EMA copy of ``model.net``. Build via
            :func:`build_target_encoder`. The factory **freezes** it
            again on call and pins to ``eval()`` mode.
        predictor: a :class:`JEPAPredictor` (or any module with the
            same ``forward(context_embeds, context_positions,
            target_positions)`` contract). The predictor's parameters
            are *not* frozen — the optimizer's ``param_groups`` need
            to include them; the simplest path is to register the
            predictor as a submodule of ``model.net`` (the ViTNN)
            before constructing the optimizer.
        mask_fn: callable ``(n_patches, device) -> (context_mask,
            target_mask)`` where both are 1-D ``BoolTensor[n_patches]``.
            Sampled freshly **once per step** and shared across the
            batch. The bundled :func:`random_block_mask` helper is the
            common choice; passing a fixed mask is fine for tests.
        ema_momentum: EMA decay used by :func:`update_ema`. Default
            0.996 (reference I-JEPA).

    Returns:
        A ``TrainStepFn`` for ``NNModel.train(..., train_step_fn=...)``.

    Raises:
        ValueError: when ``ema_momentum`` is outside ``[0, 1)``.
    """
    if not (0.0 <= ema_momentum < 1.0):
        raise ValueError(f"ema_momentum must be in [0, 1), got {ema_momentum}")

    # Defensive freeze in case the caller built the target encoder by
    # hand without going through ``build_target_encoder``.
    target_encoder.eval()
    for p in target_encoder.parameters():
        p.requires_grad = False

    def step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        m = ctx.model
        m.net.train()
        predictor.train()
        m.net.zero_grad()

        # Standard dataloader contract: (X, Y) or single tensor. Y is
        # ignored — JEPA is self-supervised.
        if hasattr(m.net, "unpack_batch"):
            (x,), _ = m.net.unpack_batch(ctx.batch)
        elif isinstance(ctx.batch, (list, tuple)):
            x = ctx.batch[0]
        else:
            x = ctx.batch
        x = x.to(m.device)

        # Sample the per-step mask. Both masks are 1-D length n_patches.
        n_patches = m.net.n_patches
        context_mask_1d, target_mask_1d = mask_fn(n_patches, m.device)
        if context_mask_1d.shape != (n_patches,) or target_mask_1d.shape != (n_patches,):
            raise ValueError(
                f"mask_fn must return two BoolTensors of shape ({n_patches},); "
                f"got {tuple(context_mask_1d.shape)} and {tuple(target_mask_1d.shape)}"
            )
        if not torch.equal(context_mask_1d, ~target_mask_1d):
            raise ValueError(
                "context_mask and target_mask must be complementary (every patch is either context or target)."
            )
        B = x.shape[0]
        context_mask = _broadcast_mask(context_mask_1d, B)

        # Forward context through trainable encoder.
        context_embeds = m.net(x, mask=context_mask)  # (B, T_ctx + 1, d_model)

        # Build position indices the predictor needs. CLS is position 0;
        # ViTNN.patch_positions() owns the 1..n_patches CLS shift, so
        # boolean-masking it yields the kept/target position indices.
        patch_positions = m.net.patch_positions()
        kept_patch_positions = patch_positions[context_mask_1d]
        context_positions = torch.cat([torch.zeros(1, dtype=torch.long, device=m.device), kept_patch_positions])
        target_positions = patch_positions[target_mask_1d]

        # Forward FULL image through EMA target (no grad), then slice
        # out the target-position embeddings only. Skip CLS at index 0.
        with torch.no_grad():
            target_full = target_encoder(x)  # (B, n_patches + 1, d_model)
            # target_positions are in the [1, n_patches] range.
            target_embeds = target_full[:, target_positions, :]

        # Predict target embeddings from context.
        predicted_embeds = predictor(context_embeds, context_positions, target_positions)

        loss = F.mse_loss(predicted_embeds, target_embeds)
        loss_val = finalize_step(loss, ctx, paradigm="jepa")
        # EMA-update target encoder AFTER the optimizer has stepped
        # — finalize_step ran the optimizer, so model.net now reflects
        # the post-step weights and the EMA tracks the freshest signal.
        update_ema(m.net, target_encoder, ema_momentum)

        # No classification metric for a self-supervised paradigm; report
        # the loss in both slots so BEST tracking + ReduceLROnPlateau
        # have a signal.
        return NNEvaluationDataPoint(
            f1=0.0,
            recall=0.0,
            accuracy=0.0,
            precision=0.0,
            loss=loss_val,
            error=loss_val,
        )

    return step
