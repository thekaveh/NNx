"""Tests for nnx.paradigms.jepa — I-JEPA (Joint Embedding Predictive
Architecture).

Covers four contracts:

  1. :func:`build_target_encoder` deep-copies, freezes, eval-mode.
  2. :func:`update_ema` is the convex combination it advertises and
     rejects out-of-range momentum.
  3. :func:`random_block_mask` is well-formed (complementary masks,
     right shapes) and reproducible under a fixed generator.
  4. End-to-end: a small ViT + predictor + JEPA step trains for 2
     epochs on synthetic 32x32 images. Loss stays finite, target
     encoder is EMA-tracked (changed from init) but never receives
     gradients (every target param has ``requires_grad=False`` and
     ``.grad is None`` after the run).
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    JEPAPredictor,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    ViTBlock,
    ViTNN,
    build_target_encoder,
    jepa_train_step_factory,
    random_block_mask,
    set_seed,
    update_ema,
)

# -------------------------------------------------------------------------
# build_target_encoder
# -------------------------------------------------------------------------


def test_build_target_encoder_is_deep_copy_frozen_eval():
    src = ViTNN(image_size=16, patch_size=4, in_channels=3, d_model=32, n_layers=2, n_heads=4)
    tgt = build_target_encoder(src)
    # Deep copy: same shapes/values, distinct storage.
    for (n_s, p_s), (n_t, p_t) in zip(src.named_parameters(), tgt.named_parameters(), strict=True):
        assert n_s == n_t
        assert torch.equal(p_s, p_t)
        assert p_s.data_ptr() != p_t.data_ptr(), f"{n_s} shares storage — not a deep copy"
    # Frozen.
    assert all(not p.requires_grad for p in tgt.parameters())
    # Eval mode.
    assert not tgt.training


# -------------------------------------------------------------------------
# update_ema
# -------------------------------------------------------------------------


def test_update_ema_is_convex_combination():
    src = ViTNN(image_size=16, patch_size=4, in_channels=3, d_model=32, n_layers=1, n_heads=4)
    tgt = build_target_encoder(src)
    # Perturb source so EMA target should drift toward the new values.
    with torch.no_grad():
        for p in src.parameters():
            p.add_(torch.ones_like(p))

    # Capture pre-EMA target params and source params.
    pre_tgt = {n: p.clone() for n, p in tgt.named_parameters()}
    src_now = {n: p.clone() for n, p in src.named_parameters()}
    update_ema(src, tgt, momentum=0.9)
    for n, p_tgt in tgt.named_parameters():
        expected = 0.9 * pre_tgt[n] + 0.1 * src_now[n]
        assert torch.allclose(p_tgt, expected, atol=1e-6), f"EMA mismatch on {n}"


def test_update_ema_rejects_bad_momentum():
    src = ViTNN(image_size=16, patch_size=4, in_channels=3, d_model=16, n_layers=1, n_heads=4)
    tgt = build_target_encoder(src)
    with pytest.raises(ValueError, match="momentum"):
        update_ema(src, tgt, momentum=1.0)
    with pytest.raises(ValueError, match="momentum"):
        update_ema(src, tgt, momentum=-0.1)


# -------------------------------------------------------------------------
# random_block_mask
# -------------------------------------------------------------------------


def test_random_block_mask_is_complementary_and_well_shaped():
    g = torch.Generator().manual_seed(0)
    ctx, tgt = random_block_mask(n_patches=64, grid_size=8, generator=g)
    assert ctx.shape == (64,)
    assert tgt.shape == (64,)
    assert ctx.dtype == torch.bool
    assert tgt.dtype == torch.bool
    # Complementary partition of the grid.
    assert torch.equal(ctx, ~tgt)
    # The target block must be non-empty (we never sample an empty block).
    assert int(tgt.sum().item()) > 0
    # And smaller than the full grid.
    assert int(tgt.sum().item()) < 64


def test_random_block_mask_is_reproducible():
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    c1, t1 = random_block_mask(n_patches=64, grid_size=8, generator=g1)
    c2, t2 = random_block_mask(n_patches=64, grid_size=8, generator=g2)
    assert torch.equal(c1, c2)
    assert torch.equal(t1, t2)


def test_random_block_mask_rejects_mismatched_grid():
    with pytest.raises(ValueError, match="grid_size"):
        random_block_mask(n_patches=64, grid_size=7)


# -------------------------------------------------------------------------
# jepa_train_step_factory — end-to-end smoke
# -------------------------------------------------------------------------


def _build_tiny_vit_model() -> NNModel:
    """Build an NNModel whose net is a small ViTNN. The base NNParams
    is a placeholder — the real net is swapped onto ``model.net``.
    """
    model = NNModel(
        net_params=NNParams(
            input_dim=3 * 32 * 32,
            output_dim=64,
            hidden_dims=[64],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )
    # Swap in the real ViT-S as the trained encoder.
    model.net = ViTNN(
        image_size=32,
        patch_size=8,
        in_channels=3,
        d_model=32,
        n_layers=2,
        n_heads=4,
    ).to(model.device)
    return model


def _build_predictor(model: NNModel) -> JEPAPredictor:
    return JEPAPredictor(
        embed_dim=model.net.d_model,
        n_patches=model.net.n_patches,
        predictor_dim=16,
        n_layers=2,
        n_heads=2,
    ).to(model.device)


def _attach_predictor_for_optim(model: NNModel, predictor: JEPAPredictor) -> None:
    """Register the predictor under model.net so the optimizer that
    walks ``model.net.parameters()`` picks up its weights too.

    This is the same idiom used by paradigms that compose a second
    module into the optimization (the predictor is trained jointly
    with the encoder, not separately).
    """
    model.net.add_module("_jepa_predictor", predictor)


def _image_loader(n: int = 8, batch_size: int = 4) -> DataLoader:
    torch.manual_seed(0)
    x = torch.randn(n, 3, 32, 32)
    y = torch.zeros(n, dtype=torch.long)  # unused by JEPA
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _fixed_mask_fn(model_n_patches: int, model_grid: int):
    """Returns a deterministic mask_fn — masks the top-left 4 patches
    as the prediction target. Deterministic-by-design so the test
    exercises only the step plumbing and not the random sampler.
    """

    def fn(n_patches: int, device: torch.device):
        if n_patches != model_n_patches:
            raise AssertionError(f"unexpected n_patches={n_patches}")
        grid = torch.zeros(model_grid, model_grid, dtype=torch.bool, device=device)
        grid[:2, :2] = True
        target_mask = grid.flatten()
        return ~target_mask, target_mask

    return fn


def test_jepa_step_runs_and_target_encoder_stays_frozen(tmp_path, monkeypatch):
    """End-to-end: 2 epochs of JEPA on synthetic 32x32 images.

    Verifies all four invariants the plan calls out:
      (a) loss is finite throughout,
      (b) target encoder params changed from their init (EMA tracked),
      (c) target encoder params still ``requires_grad=False`` after
          training,
      (d) target encoder params have no ``.grad`` attached (gradients
          never flow into them).
    """
    monkeypatch.chdir(tmp_path)
    set_seed(0)

    model = _build_tiny_vit_model()
    target_encoder = build_target_encoder(model.net)
    predictor = _build_predictor(model)
    _attach_predictor_for_optim(model, predictor)
    target_initial = {n: p.clone() for n, p in target_encoder.named_parameters()}

    mask_fn = _fixed_mask_fn(model_n_patches=model.net.n_patches, model_grid=32 // 8)
    step_fn = jepa_train_step_factory(
        target_encoder=target_encoder,
        predictor=predictor,
        mask_fn=mask_fn,
        ema_momentum=0.5,  # aggressive EMA so the test sees movement quickly
    )
    run = model.train(
        params=NNTrainParams(
            n_epochs=2,
            train_loader=_image_loader(),
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=1e-3,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7,
                factor=0.5,
                patience=1,
                cooldown=1,
                threshold=1e-3,
            ),
        ),
        train_step_fn=step_fn,
    )

    # (a) Finite loss throughout.
    losses = [idp.train_edp.loss for idp in run.idps]
    assert len(losses) > 0
    assert all(lo is not None and torch.isfinite(torch.tensor(lo)).item() for lo in losses), (
        f"non-finite loss in {losses}"
    )

    target_after = dict(target_encoder.named_parameters())
    # (b) EMA-tracked: at least one param visibly moved.
    moved = any(not torch.equal(target_initial[n], p.detach()) for n, p in target_after.items())
    assert moved, "target encoder did NOT change — EMA update broken"
    # (c) Still frozen — requires_grad=False everywhere.
    assert all(not p.requires_grad for p in target_after.values()), (
        "target encoder has a param with requires_grad=True after training"
    )
    # (d) No gradients attached — gradients never flowed in.
    assert all(p.grad is None for p in target_after.values()), (
        "target encoder has a .grad — gradients leaked into the EMA copy"
    )


def test_jepa_factory_rejects_bad_momentum():
    src = ViTNN(image_size=16, patch_size=4, in_channels=3, d_model=16, n_layers=1, n_heads=4)
    tgt = build_target_encoder(src)
    pred = JEPAPredictor(embed_dim=16, n_patches=16, predictor_dim=8, n_layers=1, n_heads=2)
    with pytest.raises(ValueError, match="momentum"):
        jepa_train_step_factory(
            target_encoder=tgt,
            predictor=pred,
            mask_fn=lambda n, d: (
                torch.ones(n, dtype=torch.bool, device=d),
                torch.zeros(n, dtype=torch.bool, device=d),
            ),
            ema_momentum=1.0,
        )


def test_jepa_predictor_forward_shapes():
    """Predictor maps (B, T_ctx, embed_dim) -> (B, T_tgt, embed_dim)
    correctly."""
    pred = JEPAPredictor(embed_dim=16, n_patches=16, predictor_dim=8, n_layers=1, n_heads=2)
    ctx = torch.randn(3, 11, 16)  # 10 context patches + 1 CLS
    ctx_pos = torch.cat([torch.tensor([0]), torch.arange(1, 11)])
    tgt_pos = torch.arange(11, 17)  # 6 target patches
    out = pred(ctx, ctx_pos, tgt_pos)
    assert out.shape == (3, 6, 16)


# --- Boundary validation: net constructors that bypass the params dataclasses ---
# These take raw numeric kwargs and would otherwise build silently degenerate
# models (n_layers=0 -> attention-free; ffn_mult=0 -> zero-width FFN; d_model=0
# -> head_dim=0 divisor-masking) — the same [[params-boundary-validation]] class
# guarded on NNTransformerParams. All are public top-level nnx.* exports.


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"d_model": 0}, "d_model > 0"),  # masks itself through `0 % n_heads == 0`
        ({"n_layers": 0}, "n_layers > 0"),
        ({"n_layers": -1}, "n_layers > 0"),
        ({"ffn_mult": 0}, "ffn_mult > 0"),
        ({"image_size": 0}, "image_size > 0"),
        ({"patch_size": 0}, "patch_size > 0"),
        ({"in_channels": 0}, "in_channels > 0"),
        ({"n_heads": 0}, "n_heads > 0"),
        ({"attn_dropout": 1.5}, "0.0 <= attn_dropout <= 1.0"),
        ({"resid_dropout": -0.1}, "0.0 <= resid_dropout <= 1.0"),
    ],
)
def test_vit_nn_rejects_non_positive_dims(overrides, match):
    kwargs = dict(image_size=16, patch_size=4, in_channels=3, d_model=32, n_layers=2, n_heads=4)
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        ViTNN(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"d_model": 0}, "d_model > 0"),
        ({"n_heads": -1}, "n_heads > 0"),
        ({"ffn_mult": 0}, "ffn_mult > 0"),
        ({"attn_dropout": 1.5}, "0.0 <= attn_dropout <= 1.0"),
        ({"resid_dropout": -0.1}, "0.0 <= resid_dropout <= 1.0"),
    ],
)
def test_vit_block_rejects_non_positive_dims(overrides, match):
    kwargs = dict(d_model=32, n_heads=4, ffn_mult=4)
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        ViTBlock(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"embed_dim": 0}, "embed_dim > 0"),
        ({"n_patches": 0}, "n_patches > 0"),
        ({"n_layers": 0}, "n_layers > 0"),
        ({"n_heads": -2}, "n_heads > 0"),
        ({"ffn_mult": 0}, "ffn_mult > 0"),
        ({"predictor_dim": 0}, "predictor_dim > 0 when set"),
    ],
)
def test_jepa_predictor_rejects_non_positive_dims(overrides, match):
    kwargs = dict(embed_dim=16, n_patches=16, n_layers=2, n_heads=2, ffn_mult=4)
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        JEPAPredictor(**kwargs)


def test_jepa_predictor_accepts_none_predictor_dim():
    """predictor_dim=None is the documented default (resolves to embed_dim//2);
    the guard must not reject it."""
    pred = JEPAPredictor(embed_dim=16, n_patches=16, predictor_dim=None, n_layers=1, n_heads=2)
    assert pred.predictor_dim == 8
