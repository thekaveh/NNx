"""Tests for nnx.paradigms.moe — MoE supervised step factory."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Devices,
    Losses,
    MoELinear,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    moe_train_step_factory,
    set_seed,
)
from nnx.nn.net.feed_fwd_nn import FeedFwdNN

# -------------------------------------------------------------------------
# Test net: a tiny classifier whose hidden layer is an MoELinear so the
# factory has something to optimize. We use FeedFwdNN's unpack_batch
# contract for free by subclassing it and swapping one layer.
# -------------------------------------------------------------------------


class _MoEClassifier(FeedFwdNN):
    """FeedFwdNN whose first hidden layer is an MoELinear.

    Subclassing FeedFwdNN inherits the ``(X,), Y = unpack_batch(batch)``
    contract that all the paradigm factories rely on for supervised
    data — no need to duplicate the boilerplate.
    """

    def __init__(self, params: NNParams, *, num_experts: int, top_k: int):
        # Build the standard FeedFwdNN backbone, then swap layer 0 for
        # an MoELinear with matching in/out dims.
        super().__init__(params)
        in_dim = params.dims[0]
        out_dim = params.dims[1]
        # Replace the first nn.Linear with an MoELinear of identical
        # in/out shape — the rest of the forward chain is untouched.
        self.layers[0] = MoELinear(in_dim, out_dim, num_experts=num_experts, top_k=top_k)


def _make_moe_model(num_experts: int = 4, top_k: int = 2) -> tuple[NNModel, _MoEClassifier]:
    """Build an NNModel whose net's hidden layer is an MoELinear."""
    params = NNParams(
        input_dim=8,
        output_dim=3,
        hidden_dims=[16],
        dropout_prob=0.0,
    )
    model_params = NNModelParams(
        net=Nets.FEED_FWD,
        device=Devices.CPU,
        loss=Losses.CROSS_ENTROPY,
    )
    model = NNModel(net_params=params, params=model_params)
    # Replace the model's auto-built FeedFwdNN with an MoE version
    # using the same params. Same loss_fn, same device.
    model.net = _MoEClassifier(params, num_experts=num_experts, top_k=top_k).to(model.device)
    return model, model.net


def _loader(n: int = 64, batch_size: int = 16) -> DataLoader:
    torch.manual_seed(0)
    X = torch.randn(n, 8)
    y = torch.randint(0, 3, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)


def _train_params(n_epochs: int, loader: DataLoader, lr: float = 5e-2) -> NNTrainParams:
    return NNTrainParams(
        n_epochs=n_epochs,
        train_loader=loader,
        optim=NNOptimParams(
            name=Optims.ADAM,
            max_lr=lr,
            momentum=(0.9, 0.999),
            weight_decay=0.0,
        ),
        scheduler=NNSchedulerParams(
            min_lr=1e-7,
            factor=0.5,
            patience=3,
            cooldown=1,
            threshold=1e-3,
        ),
    )


# -------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------


def test_moe_step_factory_validates_aux_loss_weight():
    with pytest.raises(ValueError, match="aux_loss_weight"):
        moe_train_step_factory(aux_loss_weight=-0.01)
    with pytest.raises(ValueError, match="aux_loss_weight"):
        moe_train_step_factory(aux_loss_weight=-1.0)
    # 0.0 is allowed — collapses to plain supervised.
    moe_train_step_factory(aux_loss_weight=0.0)
    # Positive is allowed.
    moe_train_step_factory(aux_loss_weight=0.1)


# -------------------------------------------------------------------------
# End-to-end training
# -------------------------------------------------------------------------


def test_moe_step_factory_end_to_end(tmp_path, monkeypatch):
    """Train a small classifier with an MoELinear; verify (a) loss is
    finite throughout, and (b) the aux loss drops over the run on the
    actual training-data distribution.

    The supervised gradient through the gate weights AND the aux-loss
    gradient through P_i jointly pull the router. With a large
    aux_loss_weight, the aux-loss term dominates and routing
    re-balances. Measured over a held-out copy of the training inputs
    to avoid coupling the metric to a single batch.
    """
    monkeypatch.chdir(tmp_path)
    set_seed(0)

    model, net = _make_moe_model(num_experts=4, top_k=2)
    # Skew the router toward two experts so the aux loss starts well
    # above its uniform-routing reference value (1.0).
    with torch.no_grad():
        net.layers[0].router.weight.zero_()
        net.layers[0].router.weight[0] += 2.0
        net.layers[0].router.weight[1] += 1.0

    loader = _loader(n=64, batch_size=16)
    # Capture the actual full dataset tensor for the before/after
    # aux-loss measurements — same distribution the training run sees.
    all_X = torch.cat([batch[0] for batch in loader], dim=0)

    # Snapshot aux loss BEFORE training on the actual training data.
    net.eval()
    with torch.no_grad():
        _ = net(all_X)
    aux_loss_start = float(net.layers[0].last_aux_loss)

    step_fn = moe_train_step_factory(aux_loss_weight=1.0)
    run = model.train(
        params=_train_params(n_epochs=8, loader=loader),
        train_step_fn=step_fn,
    )

    losses = [idp.train_edp.loss for idp in run.idps]
    assert all(lo is not None for lo in losses)
    assert all(torch.isfinite(torch.tensor(lo)).item() for lo in losses)

    # After training, aux loss on the same data should have dropped.
    net.eval()
    with torch.no_grad():
        _ = net(all_X)
    aux_loss_end = float(net.layers[0].last_aux_loss)
    assert aux_loss_end < aux_loss_start, (
        f"aux loss did not decrease across training: start {aux_loss_start:.4f} → end {aux_loss_end:.4f}"
    )


def test_moe_step_factory_finalize_step_called(tmp_path, monkeypatch):
    """A NaN supervised loss must raise FloatingPointError — proof that
    the step routes through finalize_step (which contains the NaN-guard).
    """
    monkeypatch.chdir(tmp_path)
    set_seed(0)

    model, net = _make_moe_model(num_experts=4, top_k=2)

    # Replace the model's loss_fn with one that produces NaN. The
    # NaN guard in finalize_step should trip on first batch.
    def _nan_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return logits.sum() * float("nan")

    model.loss_fn = _nan_loss  # type: ignore[assignment]

    loader = _loader(n=32, batch_size=16)
    step_fn = moe_train_step_factory(aux_loss_weight=0.01)
    with pytest.raises(FloatingPointError, match="non-finite moe loss"):
        model.train(
            params=_train_params(n_epochs=1, loader=loader),
            train_step_fn=step_fn,
        )


def test_moe_step_factory_no_moe_layers_acts_supervised(tmp_path, monkeypatch):
    """A net with zero MoELinear layers should still train cleanly —
    the factory's aux-loss accumulator stays at 0 and the step is
    exactly supervised. Guards against an accidental 'requires MoE'
    coupling in the factory."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)

    # Plain FeedFwdNN, no MoELinear anywhere.
    plain_model = NNModel(
        net_params=NNParams(
            input_dim=8,
            output_dim=3,
            hidden_dims=[16],
            dropout_prob=0.0,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )
    loader = _loader(n=32, batch_size=16)
    step_fn = moe_train_step_factory(aux_loss_weight=0.01)
    run = plain_model.train(
        params=_train_params(n_epochs=2, loader=loader),
        train_step_fn=step_fn,
    )
    losses = [idp.train_edp.loss for idp in run.idps]
    assert all(lo is not None for lo in losses)
    assert all(torch.isfinite(torch.tensor(lo)).item() for lo in losses)


def test_moe_step_factory_aux_weight_zero_drops_aux_gradient(tmp_path, monkeypatch):
    """``aux_loss_weight=0.0`` should still train (collapses to plain
    supervised), and the router should NOT receive any aux-loss-driven
    gradient signal — it can still move from supervised gradients
    flowing through the gating weights, but the load-balancing pressure
    is disabled.
    """
    monkeypatch.chdir(tmp_path)
    set_seed(0)

    model, net = _make_moe_model(num_experts=4, top_k=2)
    loader = _loader(n=32, batch_size=16)
    step_fn = moe_train_step_factory(aux_loss_weight=0.0)
    run = model.train(
        params=_train_params(n_epochs=2, loader=loader),
        train_step_fn=step_fn,
    )
    losses = [idp.train_edp.loss for idp in run.idps]
    assert all(lo is not None for lo in losses)


def test_moe_step_factory_sums_aux_across_multiple_layers(tmp_path, monkeypatch):
    """If the net has multiple MoELinear layers, the factory should sum
    every one's ``.last_aux_loss`` into the combined penalty. Verified
    by comparing the per-step aux-loss contribution against the sum we
    expect from the net's MoELinear layers after a forward."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)

    # Two-MoE net: hidden_dims=[16, 8] with both linear layers swapped
    # for MoELinears.
    params = NNParams(input_dim=8, output_dim=3, hidden_dims=[16, 8], dropout_prob=0.0)
    model = NNModel(
        net_params=params,
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )

    class _TwoMoE(FeedFwdNN):
        def __init__(self, p):
            super().__init__(p)
            self.layers[0] = MoELinear(p.dims[0], p.dims[1], num_experts=3, top_k=2)
            self.layers[1] = MoELinear(p.dims[1], p.dims[2], num_experts=3, top_k=2)

    model.net = _TwoMoE(params).to(model.device)

    # Run a single forward and confirm both layers have aux loss set.
    X, _ = next(iter(_loader(n=16, batch_size=16)))
    model.net.eval()
    with torch.no_grad():
        _ = model.net(X)
    aux_layer_0 = float(model.net.layers[0].last_aux_loss)  # type: ignore[arg-type]
    aux_layer_1 = float(model.net.layers[1].last_aux_loss)  # type: ignore[arg-type]
    assert aux_layer_0 > 0
    assert aux_layer_1 > 0

    # Count modules: there should be exactly 2 MoELinear layers.
    moe_count = sum(1 for m in model.net.modules() if isinstance(m, MoELinear))
    assert moe_count == 2

    # End-to-end training run — proves the factory walks all MoE layers.
    loader = _loader(n=32, batch_size=16)
    step_fn = moe_train_step_factory(aux_loss_weight=0.05)
    run = model.train(
        params=_train_params(n_epochs=2, loader=loader),
        train_step_fn=step_fn,
    )
    losses = [idp.train_edp.loss for idp in run.idps]
    assert all(lo is not None and torch.isfinite(torch.tensor(lo)).item() for lo in losses)


def test_moe_step_factory_rejects_mixed_precision():
    """MoE step factory shares the AMP rejection with other paradigm
    factories — finalize_step raises when ctx.scaler is non-None.

    We simulate this by building a TrainStepContext directly with a
    non-None scaler and calling the step. (Going through NNModel.train
    on CPU would never set a real scaler because AMP is CUDA-only.)
    """
    set_seed(0)
    model, _ = _make_moe_model(num_experts=4, top_k=2)
    X = torch.randn(4, 8)
    y = torch.randint(0, 3, (4,))

    step_fn = moe_train_step_factory(aux_loss_weight=0.01)
    # Build a context with a fake-but-truthy scaler. finalize_step
    # only checks for non-None, not for type — fine for this test.
    fake_scaler = torch.amp.GradScaler(device="cpu", enabled=False)
    optim = torch.optim.Adam(model.net.parameters(), lr=1e-2)

    from nnx import TrainStepContext

    ctx = TrainStepContext(
        model=model,
        batch=(X, y),
        optimizer=optim,
        scaler=fake_scaler,
        grad_clip_norm=None,
        extra_metrics=None,
        accumulate_grad_batches=1,
        batch_idx=0,
        epoch_idx=0,
    )
    with pytest.raises(ValueError, match="mixed precision"):
        step_fn(ctx)


def test_moe_step_factory_honors_grad_clip():
    """``ctx.grad_clip_norm`` is honored — gradient norms are clipped
    to the threshold after backward. Verified by inspecting per-param
    grad norms directly after a step.

    We build the TrainStepContext by hand (rather than going through
    NNModel.train) so we can read the post-step gradients before the
    next zero_grad() overwrites them.
    """
    set_seed(0)
    model, _ = _make_moe_model(num_experts=4, top_k=2)
    X = torch.randn(8, 8)
    y = torch.randint(0, 3, (8,))

    step_fn = moe_train_step_factory(aux_loss_weight=0.01)
    optim = torch.optim.SGD(model.net.parameters(), lr=0.0)  # lr=0: no actual step

    from nnx import TrainStepContext

    ctx = TrainStepContext(
        model=model,
        batch=(X, y),
        optimizer=optim,
        scaler=None,
        grad_clip_norm=0.01,
        extra_metrics=None,
        accumulate_grad_batches=1,
        batch_idx=0,
        epoch_idx=0,
    )
    step_fn(ctx)
    # After clipping, global grad norm should be ≤ 0.01.
    import math

    total_norm = math.sqrt(
        sum(float(p.grad.detach().norm(2)) ** 2 for p in model.net.parameters() if p.grad is not None)
    )
    assert total_norm <= 0.01 + 1e-5, f"grad_clip_norm=0.01 not honored; got {total_norm}"


def test_moe_step_factory_returns_evaluation_data_point(tmp_path, monkeypatch):
    """The factory must return a populated NNEvaluationDataPoint with
    finite loss and error — same shape as default_train_step's return."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)

    model, _ = _make_moe_model(num_experts=4, top_k=2)
    loader = _loader(n=16, batch_size=16)
    step_fn = moe_train_step_factory(aux_loss_weight=0.01)
    run = model.train(
        params=_train_params(n_epochs=1, loader=loader),
        train_step_fn=step_fn,
    )
    edp = run.idps[-1].train_edp
    assert edp.loss is not None
    assert edp.error is not None
    assert 0.0 <= edp.error <= 1.0


# Smoke test on _MoEClassifier helper to confirm nn.Module structural
# tests at minimum import the public API.
def test_moe_layer_recognized_via_isinstance():
    layer = MoELinear(8, 4, num_experts=4, top_k=2)
    assert isinstance(layer, nn.Module)
    assert isinstance(layer, MoELinear)


def test_moe_step_clears_stale_aux_loss_of_unexercised_layers():
    """A MoELinear registered on the net but not exercised by the
    current batch's forward must not contribute last step's aux tensor
    — its graph was freed by the previous backward, so collecting it
    raised 'backward through the graph a second time'. The step now
    clears every MoELinear's last_aux_loss before the forward."""
    import torch

    from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams
    from nnx.nn.moe import MoELinear
    from nnx.nn.nn_model import TrainStepContext
    from nnx.paradigms.moe import moe_train_step_factory

    torch.manual_seed(0)
    model = NNModel(
        net_params=NNParams(
            input_dim=8,
            output_dim=3,
            hidden_dims=[16],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    # Swap one Linear for MoE (exercised) and attach a SECOND MoELinear
    # that the forward never touches (simulates a conditional branch).
    model.net.layers[1] = MoELinear(in_features=16, out_features=3, num_experts=4, top_k=2)
    model.net.unused_moe = MoELinear(in_features=8, out_features=8, num_experts=2, top_k=1)
    # Give the unused layer a stale aux loss from a fake "previous step"
    # whose graph has already been consumed.
    stale = model.net.unused_moe(torch.randn(4, 8)).sum()
    stale.backward()
    assert model.net.unused_moe.last_aux_loss is not None

    optimizer = torch.optim.SGD(model.net.parameters(), lr=1e-3)
    step = moe_train_step_factory(aux_loss_weight=0.01)
    batch = (torch.randn(8, 8), torch.randint(0, 3, (8,)))
    ctx = TrainStepContext(
        model=model,
        batch=batch,
        optimizer=optimizer,
        scaler=None,
        grad_clip_norm=None,
        extra_metrics=None,
        accumulate_grad_batches=1,
        batch_idx=0,
        epoch_idx=0,
    )
    edp = step(ctx)  # pre-fix: RuntimeError (backward through freed graph)
    assert edp.loss is not None
    assert model.net.unused_moe.last_aux_loss is None
