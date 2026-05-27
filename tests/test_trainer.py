"""Integration tests for nnx.trainer.Trainer.

Covers:
- end-to-end run on a supervised step (Trainer can substitute for NNModel.train
  when given a single optim — proves the orchestration layer alone works)
- multi-optim e2e on a GAN-style composite (the actual target use case)
- per-optim param_groups partition the model's parameters
- validation errors (None step_fn, None params, invalid optim)
- NNRun.trainer block is populated + round-trips through save/load
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNEvaluationDataPoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParamGroupSpec,
    NNParams,
    NNRun,
    NNTrainerParams,
    Optims,
    Trainer,
    TrainerStepContext,
    TrainerStepFn,
)


def _supervised_model() -> NNModel:
    return NNModel(
        net_params=NNParams(
            input_dim=4, output_dim=2, hidden_dims=[8],
            dropout_prob=0.0, activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
        ),
    )


def _supervised_loader(n: int = 32) -> DataLoader:
    torch.manual_seed(0)
    X = torch.randn(n, 4)
    y = torch.randint(0, 2, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False)


_supervised_step: TrainerStepFn  # name-binding annotation — exercises the public type alias


def _supervised_step(ctx: TrainerStepContext) -> NNEvaluationDataPoint:
    """Standard supervised step using Trainer's single 'main' optimizer."""
    m = ctx.model
    opt = ctx.optimizers["main"]
    m.net.train()
    opt.zero_grad()

    X, Y = m.net.unpack_batch(ctx.batch)
    X = tuple(x.to(m.device) for x in X)
    Y = Y.to(m.device)

    Y_hat_log = m.net(*X)
    loss = m.loss_fn(Y_hat_log, Y)
    loss.backward()
    opt.step()

    Y_hat = Y_hat_log.argmax(dim=1)
    loss_val = float(loss.detach())
    return NNEvaluationDataPoint(
        f1=0.0, recall=0.0, accuracy=0.0, precision=0.0,
        loss=loss_val,
        error=float(1 - (Y_hat == Y).sum().item() / Y.size(0)),
    )


# -------------------------------------------------------------------------
# Construction + validation
# -------------------------------------------------------------------------

def test_trainer_constructor_rejects_none_model():
    with pytest.raises(ValueError, match="non-None model"):
        Trainer(model=None)


def test_trainer_train_rejects_none_params():
    trainer = Trainer(model=_supervised_model())
    with pytest.raises(ValueError, match="params must not be None"):
        trainer.train(params=None, trainer_step_fn=_supervised_step)


def test_trainer_train_rejects_none_step_fn():
    trainer = Trainer(model=_supervised_model())
    params = NNTrainerParams(
        n_epochs=1,
        train_loader=_supervised_loader(),
        optims={"main": NNOptimParams(
            name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0,
        )},
    )
    with pytest.raises(ValueError, match="trainer_step_fn is required"):
        trainer.train(params=params, trainer_step_fn=None)


def test_trainer_train_rejects_invalid_optim():
    trainer = Trainer(model=_supervised_model())
    # Adam with a scalar momentum is invalid (Adam wants (beta1, beta2)).
    bad = NNOptimParams(
        name=Optims.ADAM, max_lr=1e-3, momentum=0.9, weight_decay=0.0,
    )
    params = NNTrainerParams(
        n_epochs=1,
        train_loader=_supervised_loader(),
        optims={"main": bad},
    )
    with pytest.raises(ValueError, match="invalid config"):
        trainer.train(params=params, trainer_step_fn=_supervised_step)


# -------------------------------------------------------------------------
# End-to-end: supervised single-optim
# -------------------------------------------------------------------------

def test_trainer_train_runs_end_to_end_single_optim(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = Trainer(model=_supervised_model())
    params = NNTrainerParams(
        n_epochs=2,
        train_loader=_supervised_loader(),
        optims={"main": NNOptimParams(
            name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0,
        )},
    )
    run = trainer.train(params=params, trainer_step_fn=_supervised_step)

    assert run.idps is not None
    assert len(run.idps) == 2 * 4  # 2 epochs * 4 batches
    assert all(idp.train_edp is not None for idp in run.idps)
    # Run dir + idps.csv exist.
    runs_dir = tmp_path / "runs" / run.id
    assert runs_dir.is_dir()
    assert (runs_dir / "run.yaml").is_file()
    assert (runs_dir / "idps.csv").is_file()


def test_trainer_run_yaml_carries_trainer_block(tmp_path, monkeypatch):
    """The on-disk run.yaml must include a `trainer` section so
    NNRun.load can reconstruct the multi-optim config."""
    monkeypatch.chdir(tmp_path)
    trainer = Trainer(model=_supervised_model())
    params = NNTrainerParams(
        n_epochs=1,
        train_loader=_supervised_loader(),
        optims={"main": NNOptimParams(
            name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0,
        )},
        seed=7,
    )
    run = trainer.train(params=params, trainer_step_fn=_supervised_step)
    # Round-trip via NNRun.load
    loaded = NNRun.load(id=run.id)
    assert loaded.trainer is not None
    assert "main" in loaded.trainer.optims
    assert loaded.trainer.seed == 7
    # idps must round-trip too — without this, the run.yaml load
    # would silently drop training history.
    assert loaded.idps is not None
    assert len(loaded.idps) == len(run.idps)
    for orig, rt in zip(run.idps, loaded.idps, strict=True):
        assert orig.iter_idx == rt.iter_idx
        assert orig.epoch_idx == rt.epoch_idx
        assert orig.train_edp.loss == rt.train_edp.loss


def test_trainer_invokes_callbacks(tmp_path, monkeypatch):
    """Trainer.train must dispatch the same Callback lifecycle hooks
    NNModel.train does — on_train_begin / on_epoch_begin / on_epoch_end /
    on_train_end. Without this dispatch, the callback parameter is dead code."""
    monkeypatch.chdir(tmp_path)

    from nnx import Callback

    class _RecordingCallback(Callback):
        def __init__(self):
            self.events: list[str] = []

        def on_train_begin(self, ctx):
            self.events.append("train_begin")

        def on_epoch_begin(self, ctx):
            self.events.append(f"epoch_begin_{ctx.epoch}")

        def on_epoch_end(self, ctx):
            self.events.append(f"epoch_end_{ctx.epoch}")
            # Trainer-mode callbacks should see ctx.optimizers (dict) +
            # ctx.trainer in addition to the legacy ctx.optimizer (primary).
            assert hasattr(ctx, "optimizers"), "Trainer should set ctx.optimizers"
            assert hasattr(ctx, "trainer"), "Trainer should set ctx.trainer"

        def on_train_end(self, ctx):
            self.events.append("train_end")

    cb = _RecordingCallback()
    trainer = Trainer(model=_supervised_model())
    trainer.train(
        params=NNTrainerParams(
            n_epochs=2,
            train_loader=_supervised_loader(),
            optims={"main": NNOptimParams(
                name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0,
            )},
        ),
        trainer_step_fn=_supervised_step,
        callbacks=[cb],
    )
    # Exact sequence the lifecycle must produce.
    assert cb.events == [
        "train_begin",
        "epoch_begin_0", "epoch_end_0",
        "epoch_begin_1", "epoch_end_1",
        "train_end",
    ]


def test_trainer_early_stop_via_callback(tmp_path, monkeypatch):
    """A callback setting ctx.should_stop = True must terminate the
    Trainer loop early — same contract as NNModel.train."""
    monkeypatch.chdir(tmp_path)

    from nnx import Callback

    class _StopAfter(Callback):
        def __init__(self, after_epoch: int):
            self.after_epoch = after_epoch

        def on_epoch_end(self, ctx):
            if ctx.epoch >= self.after_epoch:
                ctx.should_stop = True

    trainer = Trainer(model=_supervised_model())
    run = trainer.train(
        params=NNTrainerParams(
            n_epochs=10,                # would run 10 if not stopped
            train_loader=_supervised_loader(),
            optims={"main": NNOptimParams(
                name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0,
            )},
        ),
        trainer_step_fn=_supervised_step,
        callbacks=[_StopAfter(after_epoch=1)],
    )
    # 4 batches/epoch × 2 epochs (stopped after epoch 1) = 8 idps.
    assert len(run.idps) == 2 * 4


def test_trainer_with_val_loader_evaluates(tmp_path, monkeypatch):
    """When val_loader is set, Trainer must call model.evaluate() at
    the end of each epoch and populate val_edp on the last idp."""
    monkeypatch.chdir(tmp_path)

    trainer = Trainer(model=_supervised_model())
    val_loader = _supervised_loader(n=16)
    run = trainer.train(
        params=NNTrainerParams(
            n_epochs=1,
            train_loader=_supervised_loader(),
            val_loader=val_loader,
            optims={"main": NNOptimParams(
                name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0,
            )},
        ),
        trainer_step_fn=_supervised_step,
    )
    # Last idp of each epoch carries val_edp; earlier idps don't.
    assert run.idps[-1].val_edp is not None
    assert run.idps[-1].val_edp.loss is not None


# -------------------------------------------------------------------------
# Multi-optim — the GAN-style use case
# -------------------------------------------------------------------------

class _MiniGAN(nn.Module):
    """G + D inside one nn.Module so a single NNModel can hold both.
    Mirrors the example file's pattern, kept tiny for fast tests."""

    def __init__(self):
        super().__init__()
        self.G = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        self.D = nn.Sequential(nn.Linear(1, 8), nn.LeakyReLU(0.2), nn.Linear(8, 1))

    def forward(self, x):
        return self.G(x)


def _make_gan_model() -> NNModel:
    m = _supervised_model()
    m.net = _MiniGAN().to(m.device)
    return m


def _gan_step(ctx: TrainerStepContext) -> NNEvaluationDataPoint:
    net: _MiniGAN = ctx.model.net  # type: ignore[assignment]
    opt_G = ctx.optimizers["G"]
    opt_D = ctx.optimizers["D"]
    device = ctx.model.device

    X_real, _ = ctx.batch
    X_real = X_real.to(device)
    n = X_real.size(0)

    # D step
    opt_D.zero_grad()
    z = torch.randn(n, 4, device=device)
    X_fake = net.G(z).detach()
    d_real = net.D(X_real)
    d_fake = net.D(X_fake)
    d_loss = (
        F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
        + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
    )
    d_loss.backward()
    opt_D.step()

    # G step
    opt_G.zero_grad()
    z = torch.randn(n, 4, device=device)
    g_logits = net.D(net.G(z))
    g_loss = F.binary_cross_entropy_with_logits(g_logits, torch.ones_like(g_logits))
    g_loss.backward()
    opt_G.step()

    return NNEvaluationDataPoint(
        f1=0.0, recall=0.0, accuracy=0.0, precision=0.0,
        loss=float((d_loss + g_loss).detach()) / 2,
        error=float(g_loss.detach()),
    )


def _gan_loader(n: int = 64) -> DataLoader:
    # "real" 1D samples — mixture of N(-3, 0.5) and N(3, 0.5).
    torch.manual_seed(0)
    mix = torch.randint(0, 2, (n, 1)).float()
    means = mix * 3 - (1 - mix) * 3
    X = means + 0.5 * torch.randn(n, 1)
    y = torch.zeros(n, dtype=torch.long)
    return DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)


def test_trainer_multi_optim_gan_e2e(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    model = _make_gan_model()
    # Snapshot pre-train weights for BOTH sub-nets — the test would otherwise
    # pass even if _gan_step did nothing (constant EDP, no updates).
    g_pre = {k: v.clone() for k, v in model.net.G.state_dict().items()}
    d_pre = {k: v.clone() for k, v in model.net.D.state_dict().items()}

    trainer = Trainer(model=model)

    g_optim = NNOptimParams(
        name=Optims.ADAM, max_lr=2e-4, momentum=(0.5, 0.999), weight_decay=0.0,
        param_groups=[NNParamGroupSpec(name_pattern="G.*", lr=2e-4)],
    )
    d_optim = NNOptimParams(
        name=Optims.ADAM, max_lr=2e-4, momentum=(0.5, 0.999), weight_decay=0.0,
        param_groups=[NNParamGroupSpec(name_pattern="D.*", lr=2e-4)],
    )

    run = trainer.train(
        params=NNTrainerParams(
            n_epochs=2,
            train_loader=_gan_loader(),
            optims={"G": g_optim, "D": d_optim},
        ),
        trainer_step_fn=_gan_step,
    )

    assert run.idps is not None
    assert len(run.idps) == 2 * 4  # 2 epochs * 4 batches

    # Both G and D weights must have actually changed — otherwise the
    # test only verifies idp accounting, not that optimizers ran.
    g_post = model.net.G.state_dict()
    d_post = model.net.D.state_dict()
    g_moved = any(not torch.equal(g_pre[k], g_post[k]) for k in g_pre)
    d_moved = any(not torch.equal(d_pre[k], d_post[k]) for k in d_pre)
    assert g_moved, "G's parameters did not update during multi-optim training"
    assert d_moved, "D's parameters did not update during multi-optim training"


def test_trainer_per_optim_param_groups_partition_params(tmp_path, monkeypatch):
    """When each optim's param_groups scopes it to a sub-net, the two
    optimizers should own disjoint sets of parameters. This is the
    invariant that makes GAN-style training work — backprop through D
    must not update G via the wrong optimizer.

    Driven through the Trainer (which passes strict_param_groups=True)
    so we exercise the actual code path, not a hand-rolled imitation.
    """
    monkeypatch.chdir(tmp_path)
    model = _make_gan_model()

    g_optim = NNOptimParams(
        name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0,
        param_groups=[NNParamGroupSpec(name_pattern="G.*", lr=1e-3)],
    )
    d_optim = NNOptimParams(
        name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0,
        param_groups=[NNParamGroupSpec(name_pattern="D.*", lr=1e-3)],
    )

    captured: dict = {}

    def capture_step(ctx: TrainerStepContext) -> NNEvaluationDataPoint:
        # Snapshot the optimizers on the first batch, then do a no-op step
        # so the loop terminates without divergence.
        captured.setdefault("optimizers", ctx.optimizers)
        return NNEvaluationDataPoint(
            f1=0.0, recall=0.0, accuracy=0.0, precision=0.0,
            loss=0.0, error=0.0,
        )

    trainer = Trainer(model=model)
    trainer.train(
        params=NNTrainerParams(
            n_epochs=1,
            train_loader=_gan_loader(n=16),
            optims={"G": g_optim, "D": d_optim},
        ),
        trainer_step_fn=capture_step,
    )

    g_opt = captured["optimizers"]["G"]
    d_opt = captured["optimizers"]["D"]
    g_param_ids = {id(p) for grp in g_opt.param_groups for p in grp['params']}
    d_param_ids = {id(p) for grp in d_opt.param_groups for p in grp['params']}

    assert g_param_ids.isdisjoint(d_param_ids), (
        "G and D optimizers must own disjoint params"
    )
    assert len(g_param_ids) > 0
    assert len(d_param_ids) > 0
    # Every param in G's optimizer should correspond to a named G.* param
    # on the underlying module (sanity check on the partition).
    g_names = {n for n, p in model.net.named_parameters() if id(p) in g_param_ids}
    assert all(n.startswith("G.") for n in g_names)
    d_names = {n for n, p in model.net.named_parameters() if id(p) in d_param_ids}
    assert all(n.startswith("D.") for n in d_names)
