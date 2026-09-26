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
            input_dim=4,
            output_dim=2,
            hidden_dims=[8],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )


def _supervised_loader(n: int = 32) -> DataLoader:
    torch.manual_seed(0)
    X = torch.randn(n, 4)
    y = torch.randint(0, 2, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False)


def test_trainer_records_completed_topology_transforms_on_live_model(tmp_path, monkeypatch):
    from nnx import Callback
    from nnx.nn.params.nn_checkpoint import NNCheckpointTransform

    class TransformCallback(Callback):
        def checkpoint_transforms(self):
            return (NNCheckpointTransform(name="test-transform"),)

    monkeypatch.chdir(tmp_path)
    model = _supervised_model()
    model._topology_transforms = (NNCheckpointTransform(name="preexisting"),)
    params = NNTrainerParams(
        n_epochs=1,
        train_loader=_supervised_loader(8),
        optims={"main": NNOptimParams(name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0)},
    )
    Trainer(model).train(params=params, trainer_step_fn=_supervised_step, callbacks=[TransformCallback()])

    assert [item.name for item in model._topology_transforms] == ["preexisting", "test-transform"]


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

    Y_hat_logits = m.net(*X)
    loss = m.loss_fn(Y_hat_logits, Y)
    loss.backward()
    opt.step()

    Y_hat = Y_hat_logits.argmax(dim=1)
    loss_val = float(loss.detach())
    return NNEvaluationDataPoint(
        f1=0.0,
        recall=0.0,
        accuracy=0.0,
        precision=0.0,
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
        optims={
            "main": NNOptimParams(
                name=Optims.ADAM,
                max_lr=1e-3,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            )
        },
    )
    with pytest.raises(ValueError, match="trainer_step_fn is required"):
        trainer.train(params=params, trainer_step_fn=None)


def test_trainer_train_rejects_none_train_loader():
    """train_loader=None must fail fast with an actionable ValueError
    instead of the raw TypeError the epoch loop used to throw after
    printing the run-details table."""
    trainer = Trainer(model=_supervised_model())
    params = NNTrainerParams(
        n_epochs=1,
        optims={
            "main": NNOptimParams(
                name=Optims.ADAM,
                max_lr=1e-3,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            )
        },
    )
    with pytest.raises(ValueError, match="train_loader is required"):
        trainer.train(params=params, trainer_step_fn=_supervised_step)


def test_trainer_train_rejects_invalid_optim():
    trainer = Trainer(model=_supervised_model())
    # Adam with a scalar momentum is invalid (Adam wants (beta1, beta2)).
    bad = NNOptimParams(
        name=Optims.ADAM,
        max_lr=1e-3,
        momentum=0.9,
        weight_decay=0.0,
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
        optims={
            "main": NNOptimParams(
                name=Optims.ADAM,
                max_lr=1e-2,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            )
        },
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
        optims={
            "main": NNOptimParams(
                name=Optims.ADAM,
                max_lr=1e-3,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            )
        },
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
            optims={
                "main": NNOptimParams(
                    name=Optims.ADAM,
                    max_lr=1e-3,
                    momentum=(0.9, 0.999),
                    weight_decay=0.0,
                )
            },
        ),
        trainer_step_fn=_supervised_step,
        callbacks=[cb],
    )
    # Exact sequence the lifecycle must produce.
    assert cb.events == [
        "train_begin",
        "epoch_begin_0",
        "epoch_end_0",
        "epoch_begin_1",
        "epoch_end_1",
        "train_end",
    ]


def test_trainer_last_checkpoint_contains_on_train_end_mutation(tmp_path, monkeypatch):
    """Trainer LAST must reflect net mutations made during on_train_end."""
    from nnx import Callback
    from nnx.nn.enum.checkpoints import Checkpoints
    from nnx.nn.params.nn_checkpoint import NNCheckpoint, NNCheckpointTransform

    class _MutateNetOnTrainEnd(Callback):
        completed = False

        def on_train_end(self, ctx):
            ctx.model.net.register_buffer("post_train_end_marker", torch.tensor([42.0]))
            self.completed = True

        def checkpoint_transforms(self):
            if not self.completed:
                return ()
            return (NNCheckpointTransform(name="test-marker"),)

    monkeypatch.chdir(tmp_path)
    model = _supervised_model()
    run = Trainer(model=model).train(
        params=NNTrainerParams(
            n_epochs=1,
            train_loader=_supervised_loader(n=8),
            optims={
                "main": NNOptimParams(
                    name=Optims.ADAM,
                    max_lr=1e-3,
                    momentum=(0.9, 0.999),
                    weight_decay=0.0,
                )
            },
        ),
        trainer_step_fn=_supervised_step,
        callbacks=[_MutateNetOnTrainEnd()],
    )

    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert checkpoint is not None
    assert torch.equal(checkpoint.net_state["post_train_end_marker"], torch.tensor([42.0]))
    assert [transform.name for transform in checkpoint.transforms] == ["test-marker"]


def test_trainer_step_exception_still_dispatches_train_end(tmp_path, monkeypatch):
    """Trainer callbacks must clean up even when the step function aborts."""
    monkeypatch.chdir(tmp_path)

    from nnx import Callback

    class _RecordingCallback(Callback):
        ended = False

        def on_train_end(self, ctx):
            self.ended = True

    def boom_step(ctx: TrainerStepContext) -> NNEvaluationDataPoint:  # noqa: ARG001
        raise RuntimeError("trainer boom")

    cb = _RecordingCallback()
    trainer = Trainer(model=_supervised_model())
    with pytest.raises(RuntimeError, match="trainer boom"):
        trainer.train(
            params=NNTrainerParams(
                n_epochs=1,
                train_loader=_supervised_loader(n=8),
                optims={
                    "main": NNOptimParams(
                        name=Optims.ADAM,
                        max_lr=1e-3,
                        momentum=(0.9, 0.999),
                        weight_decay=0.0,
                    )
                },
            ),
            trainer_step_fn=boom_step,
            callbacks=[cb],
        )

    assert cb.ended is True


def test_trainer_flushes_deferred_model_checkpoint(tmp_path, monkeypatch):
    from nnx import ModelCheckpoint

    monkeypatch.chdir(tmp_path)
    run = Trainer(model=_supervised_model()).train(
        params=NNTrainerParams(
            n_epochs=1,
            train_loader=_supervised_loader(n=8),
            optims={
                "main": NNOptimParams(
                    name=Optims.ADAM,
                    max_lr=1e-3,
                    momentum=(0.9, 0.999),
                    weight_decay=0.0,
                )
            },
        ),
        trainer_step_fn=_supervised_step,
        callbacks=[ModelCheckpoint(epochs=[0], tag="trainer")],
    )

    assert (tmp_path / "runs" / run.id / "checkpoints" / "trainer_e0.pt").is_file()


def test_trainer_does_not_flush_deferred_checkpoint_when_later_callback_fails(tmp_path, monkeypatch):
    from nnx import Callback, ModelCheckpoint

    class FailAfterQueue(Callback):
        def on_epoch_end(self, ctx):
            raise RuntimeError("later callback failed")

    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="later callback failed"):
        Trainer(model=_supervised_model()).train(
            params=NNTrainerParams(
                n_epochs=1,
                train_loader=_supervised_loader(n=8),
                optims={
                    "main": NNOptimParams(
                        name=Optims.ADAM,
                        max_lr=1e-3,
                        momentum=(0.9, 0.999),
                        weight_decay=0.0,
                    )
                },
            ),
            trainer_step_fn=_supervised_step,
            callbacks=[ModelCheckpoint(epochs=[0], tag="queued"), FailAfterQueue()],
        )

    assert not list(tmp_path.glob("runs/*/checkpoints/queued_e0.pt"))


def test_trainer_checkpoint_failure_rolls_history_back(tmp_path, monkeypatch):
    from nnx.nn.params.nn_checkpoint import NNCheckpoint

    monkeypatch.chdir(tmp_path)

    def fail_checkpoint(*args, **kwargs):
        raise OSError("trainer checkpoint failure")

    monkeypatch.setattr(Trainer, "_save_checkpoint", fail_checkpoint)
    with pytest.raises(OSError, match="trainer checkpoint failure"):
        Trainer(model=_supervised_model()).train(
            params=NNTrainerParams(
                n_epochs=1,
                train_loader=_supervised_loader(n=8),
                optims={
                    "main": NNOptimParams(
                        name=Optims.ADAM,
                        max_lr=1e-3,
                        momentum=(0.9, 0.999),
                        weight_decay=0.0,
                    )
                },
            ),
            trainer_step_fn=_supervised_step,
        )

    run_dirs = [
        path for path in (tmp_path / "runs").iterdir() if path.is_dir() and path.name not in {"best", ".leases"}
    ]
    assert len(run_dirs) == 1
    loaded = NNRun.load(run_dirs[0].name)
    assert loaded.idps == []
    assert NNCheckpoint.from_file(str(run_dirs[0] / "checkpoints" / "last.pt")) is None


def test_trainer_deferred_failure_does_not_update_global_best(tmp_path, monkeypatch):
    from nnx import Callback

    class DeferredFailure(Callback):
        def on_epoch_end(self, ctx):
            ctx.deferred_checkpoint_writes.append(lambda: (_ for _ in ()).throw(OSError("trainer deferred failure")))

    monkeypatch.chdir(tmp_path)
    with pytest.raises(OSError, match="trainer deferred failure"):
        Trainer(model=_supervised_model()).train(
            params=NNTrainerParams(
                n_epochs=1,
                train_loader=_supervised_loader(n=8),
                optims={
                    "main": NNOptimParams(
                        name=Optims.ADAM,
                        max_lr=1e-3,
                        momentum=(0.9, 0.999),
                        weight_decay=0.0,
                    )
                },
            ),
            trainer_step_fn=_supervised_step,
            callbacks=[DeferredFailure()],
        )

    assert not (tmp_path / "runs" / "best").exists()


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
            n_epochs=10,  # would run 10 if not stopped
            train_loader=_supervised_loader(),
            optims={
                "main": NNOptimParams(
                    name=Optims.ADAM,
                    max_lr=1e-3,
                    momentum=(0.9, 0.999),
                    weight_decay=0.0,
                )
            },
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
            optims={
                "main": NNOptimParams(
                    name=Optims.ADAM,
                    max_lr=1e-3,
                    momentum=(0.9, 0.999),
                    weight_decay=0.0,
                )
            },
        ),
        trainer_step_fn=_supervised_step,
    )
    # Last idp of each epoch carries val_edp; earlier idps don't.
    assert run.idps[-1].val_edp is not None
    assert run.idps[-1].val_edp.loss is not None


def test_trainer_dispatches_non_plateau_scheduler(tmp_path, monkeypatch):
    """Trainer._build_scheduler must honor `NNSchedulerParams.kind` and
    dispatch through the Schedulers enum factory — not silently fall back
    to ReduceLROnPlateau. NNModel's identical path is tested in
    test_schedulers.py; this is the symmetric Trainer-side cover.

    The two `_build_scheduler` implementations are intentionally duplicated
    (see the inline comment in trainer.py). Without an explicit non-plateau
    test on the Trainer side, a future change to the NNModel path alone
    would silently miss the Trainer's non-plateau dispatch.

    The scheduler instance is reachable via `TrainerStepContext.schedulers`
    inside the step fn — that's where Trainer exposes the per-name dict."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

    from nnx import NNSchedulerParams, Schedulers

    monkeypatch.chdir(tmp_path)

    captured: dict = {}

    def _capture_step(ctx: TrainerStepContext) -> NNEvaluationDataPoint:
        # First call grabs the scheduler dict; subsequent calls are no-op
        # captures so we exit cleanly with the same EDP shape Trainer expects.
        if not captured:
            captured.update(ctx.schedulers)
        return _supervised_step(ctx)

    trainer = Trainer(model=_supervised_model())
    trainer.train(
        params=NNTrainerParams(
            n_epochs=1,
            train_loader=_supervised_loader(),
            optims={
                "main": NNOptimParams(
                    name=Optims.ADAM,
                    max_lr=1e-3,
                    momentum=(0.9, 0.999),
                    weight_decay=0.0,
                )
            },
            schedulers={
                "main": NNSchedulerParams(
                    min_lr=1e-7,
                    factor=0.5,
                    patience=1,
                    cooldown=1,
                    threshold=1e-3,
                    kind=Schedulers.COSINE_ANNEALING,
                    T_max=10,
                )
            },
        ),
        trainer_step_fn=_capture_step,
    )
    sched = captured["main"]
    assert isinstance(sched, CosineAnnealingLR), (
        f"Trainer should dispatch through Schedulers.COSINE_ANNEALING; "
        f"got {type(sched).__name__} instead. Likely a regression in "
        f"trainer.py:_build_scheduler's `kind` dispatch."
    )
    assert not isinstance(sched, ReduceLROnPlateau)


def test_trainer_does_not_double_step_user_owned_scheduler(tmp_path, monkeypatch):
    from nnx import NNSchedulerParams, Schedulers

    monkeypatch.chdir(tmp_path)
    captured: dict = {}

    def _manual_scheduler_step(ctx: TrainerStepContext) -> NNEvaluationDataPoint:
        result = _supervised_step(ctx)
        if ctx.batch_idx == 0:
            ctx.schedulers["main"].step()
        captured["scheduler"] = ctx.schedulers["main"]
        return result

    Trainer(model=_supervised_model()).train(
        params=NNTrainerParams(
            n_epochs=2,
            train_loader=_supervised_loader(),
            optims={
                "main": NNOptimParams(
                    name=Optims.ADAM,
                    max_lr=1e-3,
                    momentum=(0.9, 0.999),
                    weight_decay=0.0,
                )
            },
            schedulers={
                "main": NNSchedulerParams(
                    kind=Schedulers.COSINE_ANNEALING,
                    T_max=10,
                    min_lr=1e-7,
                    factor=0.5,
                    patience=1,
                    cooldown=1,
                    threshold=1e-3,
                )
            },
            auto_step_schedulers=False,
        ),
        trainer_step_fn=_manual_scheduler_step,
    )
    assert captured["scheduler"].last_epoch == 2


def test_trainer_steps_default_owned_scheduler_once_per_epoch(tmp_path, monkeypatch):
    from nnx import NNSchedulerParams, Schedulers

    monkeypatch.chdir(tmp_path)
    captured: dict = {}

    def _capture(ctx: TrainerStepContext) -> NNEvaluationDataPoint:
        captured["scheduler"] = ctx.schedulers["main"]
        return _supervised_step(ctx)

    Trainer(model=_supervised_model()).train(
        params=NNTrainerParams(
            n_epochs=2,
            train_loader=_supervised_loader(),
            optims={
                "main": NNOptimParams(
                    name=Optims.ADAM,
                    max_lr=1e-3,
                    momentum=(0.9, 0.999),
                    weight_decay=0.0,
                )
            },
            schedulers={
                "main": NNSchedulerParams(
                    kind=Schedulers.COSINE_ANNEALING,
                    T_max=10,
                    min_lr=1e-7,
                    factor=0.5,
                    patience=1,
                    cooldown=1,
                    threshold=1e-3,
                )
            },
        ),
        trainer_step_fn=_capture,
    )
    assert captured["scheduler"].last_epoch == 2


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
    d_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + F.binary_cross_entropy_with_logits(
        d_fake, torch.zeros_like(d_fake)
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
        f1=0.0,
        recall=0.0,
        accuracy=0.0,
        precision=0.0,
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
        name=Optims.ADAM,
        max_lr=2e-4,
        momentum=(0.5, 0.999),
        weight_decay=0.0,
        param_groups=[NNParamGroupSpec(name_pattern="G.*", lr=2e-4)],
    )
    d_optim = NNOptimParams(
        name=Optims.ADAM,
        max_lr=2e-4,
        momentum=(0.5, 0.999),
        weight_decay=0.0,
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
        name=Optims.ADAM,
        max_lr=1e-3,
        momentum=(0.9, 0.999),
        weight_decay=0.0,
        param_groups=[NNParamGroupSpec(name_pattern="G.*", lr=1e-3)],
    )
    d_optim = NNOptimParams(
        name=Optims.ADAM,
        max_lr=1e-3,
        momentum=(0.9, 0.999),
        weight_decay=0.0,
        param_groups=[NNParamGroupSpec(name_pattern="D.*", lr=1e-3)],
    )

    captured: dict = {}

    def capture_step(ctx: TrainerStepContext) -> NNEvaluationDataPoint:
        # Snapshot the optimizers on the first batch, then do a no-op step
        # so the loop terminates without divergence.
        captured.setdefault("optimizers", ctx.optimizers)
        return NNEvaluationDataPoint(
            f1=0.0,
            recall=0.0,
            accuracy=0.0,
            precision=0.0,
            loss=0.0,
            error=0.0,
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
    g_param_ids = {id(p) for grp in g_opt.param_groups for p in grp["params"]}
    d_param_ids = {id(p) for grp in d_opt.param_groups for p in grp["params"]}

    assert g_param_ids.isdisjoint(d_param_ids), "G and D optimizers must own disjoint params"
    assert len(g_param_ids) > 0
    assert len(d_param_ids) > 0
    # Every param in G's optimizer should correspond to a named G.* param
    # on the underlying module (sanity check on the partition).
    g_names = {n for n, p in model.net.named_parameters() if id(p) in g_param_ids}
    assert all(n.startswith("G.") for n in g_names)
    d_names = {n for n, p in model.net.named_parameters() if id(p) in d_param_ids}
    assert all(n.startswith("D.") for n in d_names)


def test_trainer_multi_optim_without_param_groups_raises(tmp_path, monkeypatch):
    """Two optimizers with no param_groups would each grab all net parameters and
    silently double-step them. The Trainer fails fast at train(); the params
    object itself stays constructible (serialization / builder round-trips)."""
    monkeypatch.chdir(tmp_path)
    model = _make_gan_model()
    unscoped = NNOptimParams(name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0)
    params = NNTrainerParams(
        n_epochs=1,
        train_loader=_gan_loader(n=16),
        optims={"G": unscoped, "D": unscoped},
    )  # constructs fine — the guard lives at train(), not __post_init__
    with pytest.raises(ValueError, match="scope its parameters via"):
        Trainer(model=model).train(params=params, trainer_step_fn=lambda ctx: None)


def test_trainer_plateau_never_receives_nonfinite_metric(tmp_path, monkeypatch):
    """FIX-009 on the Trainer path: Trainer has no eval_step_fn, so the
    validation signal is scripted through ``model.evaluate`` and the
    training signal through the step fn. Plateau schedulers only ever
    receive finite values, skip entirely when nothing finite exists, and
    the warnings distinguish rejected non-finite metrics from absent ones."""
    import warnings
    from dataclasses import replace

    from torch.optim.lr_scheduler import ReduceLROnPlateau

    from nnx import NNSchedulerParams

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    steps: list[float] = []
    original_step = ReduceLROnPlateau.step

    def spy_step(self, metrics, epoch=None):
        steps.append(metrics)
        return original_step(self, metrics, epoch)

    monkeypatch.setattr(ReduceLROnPlateau, "step", spy_step)

    model = _supervised_model()
    val_script = iter(
        [
            (float("nan"), 1.0),
            (float("inf"), float("-inf")),
            (None, None),
            (0.1, 0.5),
        ]
    )

    def scripted_evaluate(loader, extra_metrics=None):
        error, loss = next(val_script)
        return NNEvaluationDataPoint(loss=loss, error=error, accuracy=0.5, f1=0.5, precision=0.5, recall=0.5)

    monkeypatch.setattr(model, "evaluate", scripted_evaluate)

    last_train_error: dict[int, float] = {}

    def step_fn(ctx: TrainerStepContext) -> NNEvaluationDataPoint:
        edp = _supervised_step(ctx)
        if ctx.epoch_idx == 2:
            return replace(edp, loss=None, error=None)
        assert edp.error is not None
        last_train_error[ctx.epoch_idx] = edp.error
        return edp

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Trainer(model=model).train(
            params=NNTrainerParams(
                n_epochs=4,
                train_loader=_supervised_loader(),
                val_loader=_supervised_loader(8),
                optims={"main": NNOptimParams(name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0)},
                schedulers={"main": NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3)},
            ),
            trainer_step_fn=step_fn,
        )

    assert steps == [1.0, last_train_error[1], 0.1]
    messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    rejected = [m for m in messages if "non-finite" in m]
    absent = [m for m in messages if "no metric available" in m]
    assert any("epoch 0" in m and "val_edp.error=nan" in m for m in rejected), messages
    assert any("epoch 1" in m and "val_edp.loss=-inf" in m for m in rejected), messages
    assert len(absent) == 1 and "epoch 2" in absent[0] and "non-finite" not in absent[0], messages


def test_trainer_runs_captured_config_after_builder_mutation(tmp_path, monkeypatch):
    """FIX-010 end to end: a configuration built before the builder is
    reused must create only its own optimizers/schedulers, keep scoped
    parameter ownership, expose the original sorted-first primary to
    callbacks, and persist a descriptor that matches what executed."""
    from nnx import Callback, NNSchedulerParams

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")

    def _adam(pattern: str) -> NNOptimParams:
        return NNOptimParams(
            name=Optims.ADAM,
            max_lr=1e-3,
            momentum=(0.9, 0.999),
            weight_decay=0.0,
            param_groups=[NNParamGroupSpec(name_pattern=pattern, lr=1e-3)],
        )

    plateau = NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3)
    builder = (
        NNTrainerParams.builder()
        .n_epochs(1)
        .train_loader(_supervised_loader(16))
        .optimizer("body", _adam("layers.0.*"))
        .optimizer("head", _adam("layers.1.*"))
        .scheduler("head", plateau)
    )
    captured = builder.build()
    builder.optimizer("aaa_extra", _adam("layers.*")).scheduler("aaa_extra", plateau)

    seen: dict = {}

    class Probe(Callback):
        def on_train_begin(self, ctx):
            seen["primary"] = ctx.optimizer
            seen["names"] = sorted(ctx.optimizers)

    def step(ctx: TrainerStepContext) -> NNEvaluationDataPoint:
        seen.setdefault("optimizers", ctx.optimizers)
        seen.setdefault("schedulers", sorted(ctx.schedulers))
        m = ctx.model
        for opt in ctx.optimizers.values():
            opt.zero_grad()
        X, Y = m.net.unpack_batch(ctx.batch)
        loss = m.loss_fn(m.net(*X), Y)
        loss.backward()
        for opt in ctx.optimizers.values():
            opt.step()
        return NNEvaluationDataPoint(
            f1=0.0, recall=0.0, accuracy=0.0, precision=0.0, loss=float(loss.detach()), error=0.5
        )

    model = _supervised_model()
    run = Trainer(model=model).train(params=captured, trainer_step_fn=step, callbacks=[Probe()])

    assert seen["names"] == ["body", "head"]
    assert seen["schedulers"] == ["body", "head"]  # missing entries default to plateau
    assert seen["primary"] is seen["optimizers"]["body"]
    owned = {
        name: {id(p) for group in opt.param_groups for p in group["params"]} for name, opt in seen["optimizers"].items()
    }
    named = dict(model.net.named_parameters())
    assert owned["body"] == {id(p) for n, p in named.items() if n.startswith("layers.0.")}
    assert owned["head"] == {id(p) for n, p in named.items() if n.startswith("layers.1.")}

    assert run.trainer is not None and sorted(run.trainer.optims) == ["body", "head"]
    reloaded = NNRun.load(run.id)
    assert reloaded.trainer is not None
    assert sorted(reloaded.trainer.optims) == ["body", "head"]
    assert sorted(reloaded.trainer.schedulers) == ["head"]
    assert reloaded.id == run.id
    assert reloaded.state()["trainer"] == captured.state()


def test_trainer_step_fn_owns_optimizer_updates(tmp_path, monkeypatch):
    """FIX-024 documented contract: Trainer never steps an optimizer itself —
    a step function that computes gradients but does not call ``step()``
    leaves every parameter unchanged."""
    monkeypatch.chdir(tmp_path)
    model = _supervised_model()
    before = {name: p.detach().clone() for name, p in model.net.named_parameters()}

    def _no_update(ctx: TrainerStepContext) -> NNEvaluationDataPoint:
        X, Y = ctx.model.net.unpack_batch(ctx.batch)
        loss = ctx.model.loss_fn(ctx.model.net(*X), Y)
        loss.backward()  # gradients exist, but the step fn chooses not to step
        return NNEvaluationDataPoint(f1=0.0, recall=0.0, accuracy=0.0, precision=0.0, loss=float(loss.detach()))

    Trainer(model=model).train(
        params=NNTrainerParams(
            n_epochs=2,
            train_loader=_supervised_loader(),
            optims={"main": NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0)},
        ),
        trainer_step_fn=_no_update,
    )
    for name, p in model.net.named_parameters():
        assert torch.equal(p.detach(), before[name]), name


# --- FEAT-002: Trainer with a task-adapter model ---------------------------


def _regression_task_model() -> NNModel:
    from nnx import TaskSpec

    torch.manual_seed(0)
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=1, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, loss=Losses.MEAN_SQUARED_ERROR, task=TaskSpec.regression(1)),
    )


def _regression_loader(n: int = 16) -> DataLoader:
    torch.manual_seed(1)
    X = torch.randn(n, 4)
    return DataLoader(TensorDataset(X, X.sum(dim=1, keepdim=True)), batch_size=8, shuffle=False)


def _task_step(ctx: TrainerStepContext) -> NNEvaluationDataPoint:
    """A custom hook that writes the same task record the default step does."""
    model = ctx.model
    adapter = model.task_adapter
    assert adapter is not None
    model.net.train()
    optimizer = ctx.optimizers["main"]
    optimizer.zero_grad()
    (x,), y = model.net.unpack_batch(ctx.batch)
    output = model.net(x)
    loss = model.loss_fn(output, y)
    loss.backward()
    optimizer.step()
    return adapter.record(output.detach(), y, loss=float(loss.detach()))


def test_trainer_custom_hook_and_task_evaluation_share_record_semantics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    run = Trainer(model=_regression_task_model()).train(
        params=NNTrainerParams(
            n_epochs=2,
            train_loader=_regression_loader(),
            val_loader=_regression_loader(8),
            optims={"main": NNOptimParams.builder().sgd(max_lr=0.01).build()},
        ),
        trainer_step_fn=_task_step,
    )
    for idp in run.idps:
        assert idp.train_edp.kind == "regression" and idp.train_edp.f1 is None
    val = run.idps[-1].val_edp
    assert val is not None and val.kind == "regression" and val.count == 8 and set(val.metrics) == {"mse", "mae"}
    assert val.error is None and val.accuracy is None
    _assert_same_records(NNRun.load(run.id).idps, run.idps)


class _ExplodingLoader:
    def __iter__(self):
        raise AssertionError("preflight must fail before any loader is consumed")


def test_trainer_task_preflight_fails_before_consuming_the_loader(tmp_path, monkeypatch):
    from nnx import TaskValidationError

    monkeypatch.chdir(tmp_path)
    model = _regression_task_model()
    model.loss_fn = nn.CrossEntropyLoss()  # cannot score a regression task
    with pytest.raises(TaskValidationError, match="regression loss"):
        Trainer(model=model).train(
            params=NNTrainerParams(
                n_epochs=1,
                train_loader=_ExplodingLoader(),  # type: ignore[arg-type]
                val_loader=_ExplodingLoader(),  # type: ignore[arg-type]
                optims={"main": NNOptimParams.builder().sgd(max_lr=0.01).build()},
            ),
            trainer_step_fn=_task_step,
        )
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())


def _assert_same_records(loaded, expected):
    """Reloaded idps.csv records equal the originals up to pandas' float
    parsing (the CSV reader is not bit-exact for every double)."""
    assert len(loaded) == len(expected)
    for got, want in zip(loaded, expected, strict=True):
        for edp_got, edp_want in ((got.train_edp, want.train_edp), (got.val_edp, want.val_edp)):
            assert (edp_got is None) == (edp_want is None)
            if edp_got is None:
                continue
            state_got, state_want = edp_got.state(), edp_want.state()
            assert state_got.keys() == state_want.keys()
            for key, value in state_want.items():
                if isinstance(value, float):
                    assert state_got[key] == pytest.approx(value, rel=1e-12), key
                elif isinstance(value, dict):
                    assert state_got[key] == pytest.approx(value, rel=1e-12), key
                else:
                    assert state_got[key] == value, key


# --- FEAT-005: two-optimizer Trainer warm resume ---------------------------


def _gan_like_model() -> NNModel:
    torch.manual_seed(0)
    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    return model


def _two_optimizer_params(n_epochs: int, **resume) -> NNTrainerParams:
    from nnx import NNSchedulerParams

    step_decay = (
        NNSchedulerParams.builder()
        .step(step_size=1, min_lr=0.0, factor=0.5, patience=0, cooldown=0, threshold=0.0)
        .build()
    )
    builder = (
        NNTrainerParams.builder()
        .n_epochs(n_epochs)
        .train_loader(_supervised_loader(16))
        .optimizer(
            "body",
            NNOptimParams.builder()
            .adam(max_lr=1e-2)
            .param_groups([NNParamGroupSpec(name_pattern="layers.0.*")])
            .build(),
        )
        .optimizer(
            "head",
            NNOptimParams.builder()
            .sgd(max_lr=0.05)
            .param_groups([NNParamGroupSpec(name_pattern="layers.1.*")])
            .build(),
        )
        .scheduler("head", step_decay)
    )
    if resume:
        builder.resume_from(**resume)
    return builder.build()


def _two_step(ctx: TrainerStepContext) -> NNEvaluationDataPoint:
    model = ctx.model
    model.net.train()
    for optimizer in ctx.optimizers.values():
        optimizer.zero_grad()
    (x,), y = model.net.unpack_batch(ctx.batch)
    loss = model.loss_fn(model.net(x), y)
    loss.backward()
    for optimizer in ctx.optimizers.values():
        optimizer.step()
    return NNEvaluationDataPoint(loss=float(loss.detach()), error=float(loss.detach()))


def test_two_optimizer_trainer_resume_matches_the_continuous_run(tmp_path, monkeypatch):
    from nnx import Checkpoints, NNCheckpoint

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    continuous_model = _gan_like_model()
    continuous = Trainer(continuous_model).train(params=_two_optimizer_params(4), trainer_step_fn=_two_step)

    first = Trainer(_gan_like_model()).train(params=_two_optimizer_params(2), trainer_step_fn=_two_step)
    stored = NNCheckpoint.load_training_state(run=first.id, type=Checkpoints.LAST)
    assert (
        stored is not None
        and set(stored["optimizers"]) == {"body", "head"}
        and set(stored["schedulers"])
        == {
            "body",
            "head",
        }
    )
    resumed_model = _gan_like_model()
    resumed = Trainer(resumed_model).train(params=_two_optimizer_params(2, run_id=first.id), trainer_step_fn=_two_step)
    assert resumed.resume_status is not None and resumed.resume_status.mode == "stateful"
    assert resumed.trainer is not None and resumed.trainer.state()["parent_run_id"] == first.id
    assert [idp.epoch_idx for idp in resumed.idps][0] == 2

    for key, value in continuous_model.net.state_dict().items():
        torch.testing.assert_close(resumed_model.net.state_dict()[key], value, msg=key)
    state_a = NNCheckpoint.load_training_state(run=continuous.id, type=Checkpoints.LAST)
    state_b = NNCheckpoint.load_training_state(run=resumed.id, type=Checkpoints.LAST)
    assert state_a is not None and state_b is not None
    assert state_a["schedulers"] == state_b["schedulers"]
    for name in ("body", "head"):
        assert state_a["optimizers"][name]["param_groups"] == state_b["optimizers"][name]["param_groups"]


def test_trainer_resume_rejects_a_mismatched_optimizer_set_before_mutating(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    first = Trainer(_gan_like_model()).train(params=_two_optimizer_params(1), trainer_step_fn=_two_step)
    model = _gan_like_model()
    before = {k: v.clone() for k, v in model.net.state_dict().items()}
    single = (
        NNTrainerParams.builder()
        .n_epochs(1)
        .train_loader(_supervised_loader(16))
        .optimizer("main", NNOptimParams.builder().sgd(max_lr=0.05).build())
        .resume_from(run_id=first.id)
        .build()
    )
    with pytest.raises(ValueError, match="optimizer names"):
        Trainer(model).train(params=single, trainer_step_fn=_two_step)
    for key, value in model.net.state_dict().items():
        assert torch.equal(value, before[key]), key


def test_trainer_resume_controls_round_trip_without_changing_default_serialization():
    base = _two_optimizer_params(1)
    assert "parent_run_id" not in base.state() and "resume_from_run_id" not in base.state()
    resumed = _two_optimizer_params(1, run_id="a" * 32, checkpoint="best", mode="stateful")
    state = resumed.state()
    assert "resume_from_run_id" not in state and "resume_mode" not in state
    assert state["parent_run_id"] == "a" * 32 and state["parent_checkpoint"] == "best"
    reloaded = NNTrainerParams.from_state(state)
    assert reloaded.parent_run_id == "a" * 32 and reloaded.resume_from_run_id is None
    assert reloaded.state() == state
    copied = resumed.with_train_loader(_supervised_loader(8)).with_val_loader(_supervised_loader(8))
    assert copied.resume_from_run_id == "a" * 32 and copied.resume_mode == "stateful"
    from nnx import NNTrainerParamsBuilder

    rebuilt = NNTrainerParamsBuilder.from_params(resumed).build()
    assert rebuilt.resume_from_run_id == "a" * 32 and rebuilt.state() == state


def _one_cycle(total_steps):
    from nnx import NNSchedulerParams, Schedulers

    return NNSchedulerParams(
        kind=Schedulers.ONE_CYCLE,
        max_lr=0.05,
        total_steps=total_steps,
        min_lr=0.0,
        factor=0.5,
        patience=0,
        cooldown=0,
        threshold=0.0,
    )


def _with_head_scheduler(params: NNTrainerParams, scheduler, **resume) -> NNTrainerParams:
    from nnx import NNTrainerParamsBuilder

    builder = NNTrainerParamsBuilder.from_params(params).scheduler("head", scheduler)
    if resume:
        builder.resume_from(**resume)
    return builder.build()


def test_trainer_resume_checks_one_cycle_horizons_before_mutating(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    first = Trainer(_gan_like_model()).train(
        params=_with_head_scheduler(_two_optimizer_params(2), _one_cycle(3)), trainer_step_fn=_two_step
    )
    model = _gan_like_model()
    before = {k: v.clone() for k, v in model.net.state_dict().items()}
    # 2 completed + 2 resumed epochs overrun the shared 3-step horizon: the
    # OneCycleLR would raise mid-run, so the resume is refused up front.
    with pytest.raises(ValueError, match="resumed one_cycle for 'head' would reach epoch 4"):
        Trainer(model).train(
            params=_with_head_scheduler(_two_optimizer_params(2), _one_cycle(3), run_id=first.id),
            trainer_step_fn=_two_step,
        )
    with pytest.raises(ValueError, match="resuming one_cycle for 'head' requires scheduler.total_steps"):
        Trainer(model).train(
            params=_with_head_scheduler(_two_optimizer_params(2), _one_cycle(None), run_id=first.id),
            trainer_step_fn=_two_step,
        )
    for key, value in model.net.state_dict().items():
        assert torch.equal(value, before[key]), key
    resumed = Trainer(model).train(
        params=_with_head_scheduler(_two_optimizer_params(1), _one_cycle(3), run_id=first.id),
        trainer_step_fn=_two_step,
    )
    assert resumed.idps[-1].epoch_idx == 2
    warm = Trainer(_gan_like_model()).train(
        params=_with_head_scheduler(_two_optimizer_params(2), _one_cycle(None), run_id=first.id, mode="weights_only"),
        trainer_step_fn=_two_step,
    )
    assert warm.resume_status is not None and warm.resume_status.mode == "weights_only"


def test_trainer_resume_rejects_a_changed_parameter_topology(tmp_path, monkeypatch):
    from nnx import Checkpoints, NNCheckpoint

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    first = Trainer(_gan_like_model()).train(params=_two_optimizer_params(1), trainer_step_fn=_two_step)
    stored = NNCheckpoint.load_training_state(run=first.id, type=Checkpoints.LAST)
    assert stored is not None and set(stored["optimizer_topologies"]) == {"body", "head"}
    # Same names, types and group sizes, different parameters: torch would
    # load the saved moments into the wrong tensors without complaint.
    from nnx import NNTrainerParamsBuilder

    swapped = (
        NNTrainerParamsBuilder.from_params(_two_optimizer_params(1))
        .optimizer(
            "body",
            NNOptimParams.builder()
            .adam(max_lr=1e-2)
            .param_groups([NNParamGroupSpec(name_pattern="layers.1.*")])
            .build(),
        )
        .optimizer(
            "head",
            NNOptimParams.builder()
            .sgd(max_lr=0.05)
            .param_groups([NNParamGroupSpec(name_pattern="layers.0.*")])
            .build(),
        )
        .resume_from(run_id=first.id)
        .build()
    )
    model = _gan_like_model()
    before = {k: v.clone() for k, v in model.net.state_dict().items()}
    with pytest.raises(ValueError, match="topology for 'body' does not match"):
        Trainer(model).train(params=swapped, trainer_step_fn=_two_step)
    for key, value in model.net.state_dict().items():
        assert torch.equal(value, before[key]), key


def test_trainer_resume_rejects_a_changed_optimizer_factory(tmp_path, monkeypatch):
    from nnx import (
        NNOptimFactoryParams,
        OptimizerFactorySpec,
        register_optimizer_factory,
        unregister_optimizer_factory,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    register_optimizer_factory("tests.trainer_resume_sgd", 1, lambda groups, config: torch.optim.SGD(groups))
    try:
        factory_head = NNOptimFactoryParams(
            factory=OptimizerFactorySpec(id="tests.trainer_resume_sgd", version=1, config={}),
            max_lr=0.05,
            param_groups=[NNParamGroupSpec(name_pattern="layers.1.*")],
        )

        def params(head, **resume):
            builder = NNTrainerParams.builder().n_epochs(1).train_loader(_supervised_loader(16))
            builder.optimizer(
                "body",
                NNOptimParams.builder()
                .adam(max_lr=1e-2)
                .param_groups([NNParamGroupSpec(name_pattern="layers.0.*")])
                .build(),
            ).optimizer("head", head)
            if resume:
                builder.resume_from(**resume)
            return builder.build()

        first = Trainer(_gan_like_model()).train(params=params(factory_head), trainer_step_fn=_two_step)
        built_in_head = (
            NNOptimParams.builder().sgd(max_lr=0.05).param_groups([NNParamGroupSpec(name_pattern="layers.1.*")]).build()
        )
        # Both build torch.optim.SGD, so only the recorded factory identity differs.
        with pytest.raises(ValueError, match="factory mismatch for 'head'"):
            Trainer(_gan_like_model()).train(params=params(built_in_head, run_id=first.id), trainer_step_fn=_two_step)
        resumed = Trainer(_gan_like_model()).train(
            params=params(factory_head, run_id=first.id), trainer_step_fn=_two_step
        )
        assert resumed.resume_status is not None and resumed.resume_status.mode == "stateful"
    finally:
        unregister_optimizer_factory("tests.trainer_resume_sgd", 1)
