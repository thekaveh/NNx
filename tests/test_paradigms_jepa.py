"""Tests for nnx.paradigms.jepa — I-JEPA (Joint Embedding Predictive
Architecture).

Covers five contracts:

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
  5. Target placement: the step refuses a target inside ``model.net``
     or sharing its parameters (with the objective's check and wording),
     never freezes the online net on construction or on a resume's
     restore, and runs every target forward in eval mode, across a
     stateful resume too.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Callback,
    Devices,
    JEPAPredictor,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNRun,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    TrainStepFn,
    ViTBlock,
    ViTNN,
    build_target_encoder,
    jepa_objective,
    jepa_train_step_factory,
    random_block_mask,
    set_seed,
    update_ema,
)
from tests._adapter_outputs import REMEDY, DictViT, Extract

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


def test_update_ema_takes_a_one_element_tensor_momentum():
    # A momentum scheduled with torch (``torch.linspace(0.996, 1.0, n)[i]``)
    # updates like the same float.
    src = torch.nn.Linear(3, 3)
    by_float, by_tensor = build_target_encoder(src), build_target_encoder(src)
    with torch.no_grad():
        for p in src.parameters():
            p.add_(torch.ones_like(p))
    update_ema(src, by_float, momentum=0.9)
    update_ema(src, by_tensor, momentum=torch.tensor(0.9))
    for a, b in zip(by_float.parameters(), by_tensor.parameters(), strict=True):
        assert torch.allclose(a, b)


@pytest.mark.parametrize(
    "momentum",
    [torch.tensor(True), torch.tensor([0.9, 0.9]), torch.tensor(1.0), torch.tensor(float("nan"))],
    ids=["bool", "two-element", "one", "nan"],
)
def test_update_ema_refuses_a_bad_tensor_momentum(momentum):
    src = torch.nn.Linear(3, 3)
    tgt = build_target_encoder(src)
    before = [p.clone() for p in tgt.parameters()]
    with pytest.raises(ValueError, match="update_ema requires momentum"):
        update_ema(src, tgt, momentum=momentum)
    assert all(torch.equal(a, b) for a, b in zip(before, tgt.parameters(), strict=True))


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


def _build_tiny_vit_model(resid_dropout: float = 0.0) -> NNModel:
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
        resid_dropout=resid_dropout,
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


def test_vit_jepa_float64_forward_backward_keeps_rmsnorm_precision(rmsnorm_fp64_oracle):
    """FIX-026: a double ViTNN context encoder, its EMA target copy, and a
    double JEPAPredictor run every shared RMSNorm (ViTBlock ``norm1``/
    ``norm2`` plus ``ViTNN.norm_out``) in FP64 with no internal downcast;
    one JEPA-shaped step backpropagates finite FP64 gradients and the EMA
    update keeps the target encoder FP64."""
    torch.manual_seed(0)
    encoder = ViTNN(image_size=16, patch_size=4, in_channels=3, d_model=16, n_layers=2, n_heads=2).double()
    target_encoder = build_target_encoder(encoder)
    predictor = JEPAPredictor(
        embed_dim=16, n_patches=encoder.n_patches, predictor_dim=8, n_layers=1, n_heads=2
    ).double()
    images = torch.randn(2, 3, 16, 16, dtype=torch.float64)
    context_mask, target_mask = random_block_mask(n_patches=encoder.n_patches, grid_size=4)
    positions = encoder.patch_positions()
    context_positions = torch.cat([torch.zeros(1, dtype=torch.long), positions[context_mask]])
    target_positions = positions[target_mask]

    with rmsnorm_fp64_oracle(torch.nn.ModuleList([encoder, target_encoder, predictor])) as norms:
        context = encoder(images, mask=context_mask.expand(2, -1))
        with torch.no_grad():
            targets = target_encoder(images)[:, target_positions]
        predicted = predictor(context, context_positions, target_positions)
    assert len(norms) == 2 * (2 * 2 + 1) + 2 * 1

    assert predicted.dtype == targets.dtype == torch.float64
    torch.nn.functional.mse_loss(predicted, targets).backward()
    for name, param in [*encoder.named_parameters(), *predictor.named_parameters()]:
        assert param.grad is not None, name
        assert param.grad.dtype == torch.float64 and torch.isfinite(param.grad).all(), name

    update_ema(encoder, target_encoder, momentum=0.5)
    for name, param in target_encoder.named_parameters():
        assert param.dtype == torch.float64 and torch.isfinite(param).all(), name


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


def test_an_empty_target_mask_is_refused_as_a_mask_problem(tmp_path, monkeypatch):
    """The imperative step checks the objective's mask contract: an
    all-context mask is refused by name, not as a NaN "training diverged"."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    model = _build_tiny_vit_model()
    predictor = _build_predictor(model)
    _attach_predictor_for_optim(model, predictor)
    step_fn = jepa_train_step_factory(
        target_encoder=build_target_encoder(model.net),
        predictor=predictor,
        mask_fn=lambda n, d: (torch.ones(n, dtype=torch.bool, device=d), torch.zeros(n, dtype=torch.bool, device=d)),
    )
    with pytest.raises(ValueError, match="target"):
        model.train(
            params=NNTrainParams(
                n_epochs=1,
                train_loader=_image_loader(),
                optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0),
            ),
            train_step_fn=step_fn,
        )


def test_a_predictor_outside_the_net_is_refused(tmp_path, monkeypatch):
    """NNModel.train builds the optimizer from model.net: an unregistered
    predictor got gradients but was never updated (no error)."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    model = _build_tiny_vit_model()
    predictor = _build_predictor(model)  # deliberately not attached
    step_fn = jepa_train_step_factory(
        target_encoder=build_target_encoder(model.net),
        predictor=predictor,
        mask_fn=_fixed_mask_fn(model_n_patches=model.net.n_patches, model_grid=32 // 8),
    )
    with pytest.raises(ValueError, match="not a submodule of model.net"):
        model.train(
            params=NNTrainParams(
                n_epochs=1,
                train_loader=_image_loader(),
                optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0),
            ),
            train_step_fn=step_fn,
        )


# -------------------------------------------------------------------------
# Target placement and mode (FIX-064, #531)
# -------------------------------------------------------------------------


def _sgd_params(**fields) -> NNTrainParams:
    return NNTrainParams(
        n_epochs=1,
        train_loader=_image_loader(),
        optim=NNOptimParams(name=Optims.SGD, max_lr=0.05, momentum=0.0, weight_decay=0.0),
        **fields,
    )


def _misplaced_target(model: NNModel, predictor: JEPAPredictor, layout: str) -> torch.nn.Module:
    """A target the step must refuse, with the predictor attached: under
    model.net, model.net itself, or an EMA copy tied to one online parameter."""
    if layout == "is_net":
        _attach_predictor_for_optim(model, predictor)
        return model.net
    target = build_target_encoder(model.net)
    _attach_predictor_for_optim(model, predictor)
    if layout == "inside":  # e.g. so .to() / .double() / checkpoints carry it
        model.net.add_module("_jepa_target", target)
    else:  # "shares"
        target.cls_token = model.net.cls_token
    return target


_PLACEMENT = {  # each names where the target sits and what that placement would cost
    "inside": r"must not be part of model\.net \(it is model\.net\._jepa_target\).*the optimizer would train it",
    "is_net": r"must not be part of model\.net \(it is model\.net itself\).*freezing it would freeze the online",
    "shares": "shares parameters with model.net",
}


@pytest.mark.parametrize("layout", sorted(_PLACEMENT))
def test_a_misplaced_target_is_refused_alike_on_both_paths_before_any_change(tmp_path, monkeypatch, layout):
    """Each step's model.net.train() put a target under model.net in train
    mode, so its dropout changed the targets; the imperative step now refuses
    the placement at its first step (before any target forward, freeze, mode
    change or update) with the objective's check and wording."""
    monkeypatch.chdir(tmp_path)
    messages = []
    forwards: list[bool] = []
    for path in ("imperative", "objective"):
        set_seed(0)
        model = _build_tiny_vit_model(resid_dropout=0.5)
        predictor = _build_predictor(model)
        target = _misplaced_target(model, predictor, layout)
        model.net.eval()  # so a mode change before the refusal shows
        target.register_forward_hook(lambda module, inputs, output: forwards.append(module.training))
        mask_fn = _fixed_mask_fn(model_n_patches=model.net.n_patches, model_grid=32 // 8)
        trainable = [n for n, p in model.net.named_parameters() if p.requires_grad]
        before = copy.deepcopy(model.net.state_dict())
        if path == "imperative":
            kwargs = {"train_step_fn": jepa_train_step_factory(target, predictor, mask_fn)}
        else:
            kwargs = {"objective": jepa_objective(target, predictor, mask_fn)}
        with pytest.raises(ValueError, match=_PLACEMENT[layout]) as raised:
            model.train(params=_sgd_params(overwrite_existing=True), **kwargs)
        messages.append(str(raised.value))
        assert forwards == []
        assert not any(module.training for module in model.net.modules())
        assert [n for n, p in model.net.named_parameters() if p.requires_grad] == trainable
        assert all(torch.equal(model.net.state_dict()[k], v) for k, v in before.items())
    assert messages[0] == messages[1]


@pytest.mark.parametrize("layout", ["is_net", "shares"])
def test_building_the_step_on_an_online_target_leaves_the_net_alone(tmp_path, monkeypatch, layout):
    """The factory froze and eval()'d its target on construction: model.net
    as the target froze the online network, and train() then failed with a
    misleading freeze('*') hint instead of naming the placement."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    model = _build_tiny_vit_model()
    predictor = _build_predictor(model)
    target = _misplaced_target(model, predictor, layout)
    modes = {name: module.training for name, module in model.net.named_modules()}
    step_fn = jepa_train_step_factory(
        target, predictor, _fixed_mask_fn(model_n_patches=model.net.n_patches, model_grid=32 // 8)
    )
    assert all(p.requires_grad for p in model.net.parameters())
    assert {name: module.training for name, module in model.net.named_modules()} == modes
    with pytest.raises(ValueError, match=_PLACEMENT[layout]) as raised:
        model.train(params=_sgd_params(), train_step_fn=step_fn)
    assert "freeze('*')" not in str(raised.value)
    assert all(p.requires_grad for p in model.net.parameters())


class _TargetToTrainMode(Callback):
    """Code outside the step that switches the target to train mode."""

    def __init__(self, target: torch.nn.Module) -> None:
        self.target = target

    def on_epoch_begin(self, ctx) -> None:
        self.target.train()


@pytest.mark.parametrize("layout", ["built", "by_hand"])
def test_every_target_forward_runs_in_eval_mode_across_a_stateful_resume(tmp_path, monkeypatch, layout):
    """The factory pinned eval() once, at construction: a target switched to
    train mode afterwards ran its dropout on the targets. The step now re-pins
    it before each target forward, in every accepted layout (a
    build_target_encoder copy, or a hand-made trainable copy in train mode
    that the step freezes) and after a stateful resume restores it. The
    target is back in train mode at every forward's start (a callback at each
    epoch, the hook after each forward), so each step must re-pin it."""
    monkeypatch.chdir(tmp_path)
    modes: list[bool] = []

    def record(module: torch.nn.Module, inputs: object, output: object) -> None:
        modes.append(module.training)
        module.train()  # as other code might between steps

    def session(**resume) -> NNRun:
        set_seed(0)
        model = _build_tiny_vit_model(resid_dropout=0.5)
        target = build_target_encoder(model.net) if layout == "built" else copy.deepcopy(model.net)
        predictor = _build_predictor(model)
        _attach_predictor_for_optim(model, predictor)
        target.register_forward_hook(record)
        step_fn = jepa_train_step_factory(
            target,
            predictor,
            _fixed_mask_fn(model_n_patches=model.net.n_patches, model_grid=32 // 8),
            ema_momentum=0.5,
        )
        run = model.train(params=_sgd_params(**resume), train_step_fn=step_fn, callbacks=[_TargetToTrainMode(target)])
        assert all(not p.requires_grad and p.grad is None for p in target.parameters())
        return run

    first = session()
    second = session(resume_from_run_id=first.id)
    assert second.resume_status is not None and second.resume_status.mode == "stateful"
    assert "jepa.target_encoder" in second.resume_status.restored_components
    assert modes == [False] * 4  # two steps per session


_RESUME_REFUSAL = {
    "inside": (ValueError, "parameter topology does not match the checkpoint"),
    "is_net": (RuntimeError, "Missing key"),
    "shares": (ValueError, _PLACEMENT["shares"]),
}


@pytest.mark.parametrize("layout", sorted(_RESUME_REFUSAL))
def test_a_resumed_misplaced_target_is_refused_before_freezing_the_net(tmp_path, monkeypatch, layout):
    """A stateful resume restored the target and froze it before the first
    step's placement check, so a target sharing an online parameter froze it
    and the refusal left model.net partly frozen. The restore now loads the
    weights only (the resume window puts the net's back on a failure), so a
    resumed misplaced target is refused with the net as it was: at the first
    step, or in the restore when the layout no longer matches the checkpoint."""
    monkeypatch.chdir(tmp_path)

    def session(layout: str) -> tuple[NNModel, TrainStepFn]:
        set_seed(0)
        model = _build_tiny_vit_model()
        predictor = _build_predictor(model)
        if layout in _RESUME_REFUSAL:
            target = _misplaced_target(model, predictor, layout)
        else:
            target = build_target_encoder(model.net)
            _attach_predictor_for_optim(model, predictor)
        mask_fn = _fixed_mask_fn(model_n_patches=model.net.n_patches, model_grid=32 // 8)
        return model, jepa_train_step_factory(target, predictor, mask_fn, ema_momentum=0.5)

    model, step_fn = session("built")
    first = model.train(params=_sgd_params(), train_step_fn=step_fn)
    model, step_fn = session(layout)
    model.net.eval()  # so a mode change before the refusal shows
    trainable = [n for n, p in model.net.named_parameters() if p.requires_grad]
    before = copy.deepcopy(model.net.state_dict())
    error, match = _RESUME_REFUSAL[layout]
    with pytest.raises(error, match=match):
        model.train(params=_sgd_params(resume_from_run_id=first.id), train_step_fn=step_fn)
    assert [n for n, p in model.net.named_parameters() if p.requires_grad] == trainable
    assert not any(module.training for module in model.net.modules())
    assert all(torch.equal(model.net.state_dict()[k], v) for k, v in before.items())


@pytest.mark.parametrize("which", ["target_encoder", "predictor"])
def test_the_factory_refuses_a_non_module_target_or_predictor_when_built(which):
    """It no longer touches the target when built (#531), but still refuses a
    non-Module up front, as jepa_objective does, before any run is reserved."""
    net = torch.nn.Linear(4, 4)
    parts = {"target_encoder": torch.nn.Linear(4, 4), "predictor": torch.nn.Linear(4, 4)}
    parts[which] = net.state_dict()  # a state_dict passed by mistake
    with pytest.raises(TypeError, match="target_encoder and predictor must be torch.nn.Module instances"):
        jepa_train_step_factory(parts["target_encoder"], parts["predictor"], lambda n, device: (None, None))


# -------------------------------------------------------------------------
# #561: the online net's and the EMA target's return values through the
# model's batch adapter
# -------------------------------------------------------------------------


def _three_jepa_models(tmp_path, monkeypatch):
    """One ViT's and predictor's weights three ways, each with its own EMA
    target and predictor: the ViT swapped onto a built-in model (no batch
    adapter), wrapped with its default adapter, and inside a mapping-returning
    module read through ``Extract("tokens")``."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    vit = ViTNN(image_size=32, patch_size=8, in_channels=3, d_model=32, n_layers=2, n_heads=4)
    predictor = JEPAPredictor(embed_dim=32, n_patches=vit.n_patches, predictor_dim=16, n_layers=2, n_heads=2)
    swapped = _build_tiny_vit_model()
    swapped.net = copy.deepcopy(vit)
    wrapped = NNModel(module=copy.deepcopy(vit), params=NNModelParams(loss=Losses.MEAN_SQUARED_ERROR))
    extract = Extract("tokens")
    mapped = NNModel(
        module=DictViT(copy.deepcopy(vit)),
        params=NNModelParams(loss=Losses.MEAN_SQUARED_ERROR),
        batch_adapter=extract,
    )
    parts = []
    for model in (swapped, wrapped, mapped):
        target = build_target_encoder(model.net)
        own = copy.deepcopy(predictor)
        _attach_predictor_for_optim(model, own)
        parts.append((model, target, own))
    return parts, extract


def _unwrapped(state):
    """A DictViT state keyed like the bare ViT's (its ``vit.`` prefix dropped)."""
    return {k.removeprefix("vit."): v for k, v in state.items()}


def test_a_dict_returning_net_trains_through_the_adapters_output(tmp_path, monkeypatch):
    """The step called model.net(x, mask=...) and the EMA target raw, so a
    net returning a mapping failed slicing it (#561). Read through
    Extract("tokens") — the target's return value too — it trains exactly as
    the same weights returning a tensor do, with the default adapter or none:
    loss for loss, online and EMA-target weight for weight."""
    parts, extract = _three_jepa_models(tmp_path, monkeypatch)
    mask_fn = _fixed_mask_fn(model_n_patches=16, model_grid=4)
    losses = []
    for model, target, predictor in parts:
        run = model.train(
            params=NNTrainParams(
                n_epochs=2,
                train_loader=_image_loader(),
                optim=NNOptimParams(name=Optims.SGD, max_lr=0.05, momentum=0.0, weight_decay=0.0),
                seed=1,
            ),
            train_step_fn=jepa_train_step_factory(target, predictor, mask_fn, ema_momentum=0.5),
        )
        losses.append([idp.train_edp.loss for idp in run.idps])

    assert all(loss is not None and torch.isfinite(torch.tensor(loss)) for loss in losses[2])
    assert losses[0] == losses[1] == losses[2]
    (swapped, swapped_target, _), *others = parts
    for model, target, _ in others:
        for state, reference in (
            (model.net.state_dict(), swapped.net.state_dict()),
            (target.state_dict(), swapped_target.state_dict()),
        ):
            state = _unwrapped(state)
            assert state.keys() == reference.keys()
            assert all(torch.equal(state[k], reference[k]) for k in reference)
    # Per step (2 epochs x 2 batches): the online forward, then the EMA target's under no_grad.
    assert extract.grad_modes == [True, False] * 4


def test_a_non_tensor_net_output_names_the_adapter_remedy(tmp_path, monkeypatch):
    """With the default adapter, a mapping return raises BatchAdapter.output()'s
    TypeError, naming the remedy, not a KeyError from slicing it."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    vit = ViTNN(image_size=32, patch_size=8, in_channels=3, d_model=32, n_layers=2, n_heads=4)
    model = NNModel(module=DictViT(vit), params=NNModelParams(loss=Losses.MEAN_SQUARED_ERROR))
    target = build_target_encoder(model.net)
    predictor = JEPAPredictor(embed_dim=32, n_patches=vit.n_patches, predictor_dim=16, n_layers=2, n_heads=2)
    _attach_predictor_for_optim(model, predictor)
    with pytest.raises(TypeError, match=REMEDY):
        model.train(
            params=_sgd_params(),
            train_step_fn=jepa_train_step_factory(target, predictor, _fixed_mask_fn(16, 4)),
        )


# -------------------------------------------------------------------------
# #564: a swapped net whose unpack_batch returns a bare tensor
# -------------------------------------------------------------------------


class _BareViT(ViTNN):
    """A ViTNN whose ``unpack_batch`` returns its one input as a bare tensor,
    as NNModel's supervised path reads it."""

    def unpack_batch(self, batch):
        return batch[0], None


@pytest.mark.parametrize("batch_size", [8, 1])
def test_a_swapped_net_with_a_bare_tensor_unpack_batch_trains_as_one_input(batch_size, tmp_path, monkeypatch):
    """The step read each row of the bare tensor as an input: 8-row batches
    were refused ("gave 8") and 1-row batches lost their batch axis (#564).
    Swapped onto a built-in model, the net trains exactly as the same weights
    wrapped with module= do, loss for loss."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    swapped = _build_tiny_vit_model()  # a built-in model: no batch adapter
    swapped.net = _BareViT(image_size=32, patch_size=8, in_channels=3, d_model=32, n_layers=2, n_heads=4)
    wrapped = NNModel(module=copy.deepcopy(swapped.net), params=NNModelParams(loss=Losses.MEAN_SQUARED_ERROR))
    predictor = _build_predictor(swapped)
    loader = _image_loader(n=8, batch_size=batch_size)
    losses = []
    for model in (swapped, wrapped):
        target = build_target_encoder(model.net)
        own = copy.deepcopy(predictor)
        _attach_predictor_for_optim(model, own)
        run = model.train(
            params=NNTrainParams(
                n_epochs=1,
                train_loader=loader,
                optim=NNOptimParams(name=Optims.SGD, max_lr=0.05, momentum=0.0, weight_decay=0.0),
                seed=1,
            ),
            train_step_fn=jepa_train_step_factory(target, own, _fixed_mask_fn(16, 4), ema_momentum=0.5),
        )
        losses.append([idp.train_edp.loss for idp in run.idps])

    assert all(loss is not None and torch.isfinite(torch.tensor(loss)) for loss in losses[0])
    assert losses[0] == losses[1]
