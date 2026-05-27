"""Tests for the train_step_fn hook on NNModel.train().

Covers:
- Default-None path is byte-identical to the existing supervised loop.
- Custom train_step_fn is dispatched per batch with correct context.
- Custom hook can populate NNEvaluationDataPoint.extra; it survives the
  NNRun.save / NNRun.load round-trip.
- batch_idx / epoch_idx in the context sequence correctly across the
  epoch boundary.
- An autoencoder-style step (no labels, MSE reconstruction loss) trains
  end-to-end and the mean loss in the last epoch is below the mean loss
  in the first (real, measurable training).
"""

from __future__ import annotations

import torch
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
    NNParams,
    NNRun,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    TrainStepContext,
    default_train_step,
)


def _make_model() -> NNModel:
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


def _make_train_params(loader, **kw):
    return NNTrainParams(
        n_epochs=kw.pop("n_epochs", 2),
        train_loader=loader,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
        **kw,
    )


def _make_loader(n=16, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n, 4)
    y = torch.randint(0, 2, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False)


def test_default_none_path_equals_explicit_default_train_step(tmp_path, monkeypatch):
    """The default-None call path must produce byte-identical (within FP
    noise) results to passing default_train_step explicitly. Guards the
    invariant that swapping the dispatcher didn't change behavior."""
    # Run A: no train_step_fn (existing call signature).
    (tmp_path / "a").mkdir()
    monkeypatch.chdir(tmp_path / "a")
    torch.manual_seed(42)
    model_a = _make_model()
    run_a = model_a.train(params=_make_train_params(_make_loader()))

    # Run B: explicit train_step_fn=default_train_step. Same seed, same data.
    (tmp_path / "b").mkdir()
    monkeypatch.chdir(tmp_path / "b")
    torch.manual_seed(42)
    model_b = _make_model()
    run_b = model_b.train(
        params=_make_train_params(_make_loader()),
        train_step_fn=default_train_step,
    )

    assert run_a.id == run_b.id  # state() identical → same hash
    assert len(run_a.idps) == len(run_b.idps)
    for a, b in zip(run_a.idps, run_b.idps, strict=True):
        assert abs(a.train_edp.loss - b.train_edp.loss) < 1e-9
        assert abs(a.train_edp.error - b.train_edp.error) < 1e-9
        assert a.iter_idx == b.iter_idx
        assert a.epoch_idx == b.epoch_idx


def test_custom_train_step_fn_is_invoked(tmp_path, monkeypatch):
    """Spy hook counts invocations; assert equals batches × epochs."""
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)

    calls = {"n": 0}

    def spy_step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        calls["n"] += 1
        return default_train_step(ctx)

    model = _make_model()
    run = model.train(
        params=_make_train_params(_make_loader(n=16), n_epochs=2),  # 16/8 = 2 batches × 2 epochs = 4
        train_step_fn=spy_step,
    )
    assert calls["n"] == 4
    assert len(run.idps) == 4


def test_train_step_context_carries_batch_and_epoch_idx(tmp_path, monkeypatch):
    """The context's (batch_idx, epoch_idx) must sequence correctly:
    (0,0), (1,0), (0,1), (1,1) for a 2-batch × 2-epoch run."""
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)

    seen: list[tuple[int, int]] = []

    def recording_step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        seen.append((ctx.batch_idx, ctx.epoch_idx))
        return default_train_step(ctx)

    model = _make_model()
    model.train(
        params=_make_train_params(_make_loader(n=16), n_epochs=2),
        train_step_fn=recording_step,
    )
    assert seen == [(0, 0), (1, 0), (0, 1), (1, 1)]


def test_custom_step_without_error_field_doesnt_crash_best_compare(tmp_path, monkeypatch):
    """A custom train_step_fn that returns an EDP with error=None must not
    crash the BEST-checkpoint comparison. The shared `_best_err` helper
    (imported by both NNModel._save_checkpoints and the inline
    best_checkpoint tracking) falls through val_edp → train_edp → +inf
    so missing .error fields are tolerated. Custom hooks shouldn't be
    required to populate the error field (the supervised proxy doesn't
    apply to all paradigms)."""
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)

    def no_error_step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        edp = default_train_step(ctx)
        # Explicitly drop the error field; loss is still set.
        from dataclasses import replace

        return replace(edp, error=None)

    model = _make_model()
    # 2 epochs ensures _save_checkpoints runs the BEST comparison at
    # least once with a non-None best_checkpoint already on hand.
    run = model.train(
        params=_make_train_params(_make_loader(n=16), n_epochs=2),
        train_step_fn=no_error_step,
    )
    assert all(idp.train_edp.error is None for idp in run.idps)
    # Run completed without TypeError.
    assert (tmp_path / "runs" / run.id / "checkpoints" / "best.pt").exists()


def test_custom_step_extra_survives_run_save_load(tmp_path, monkeypatch):
    """A custom hook that populates EDP.extra must have those values
    round-trip through NNRun.save → idps.csv → NNRun.load. Validates
    that the train_step_fn dispatch path doesn't break the EDP-extra
    persistence machinery."""
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)

    def extra_step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        edp = default_train_step(ctx)
        return edp.with_extra("custom_key", 42.0 + ctx.batch_idx)

    model = _make_model()
    run = model.train(
        params=_make_train_params(_make_loader(n=16), n_epochs=1),
        train_step_fn=extra_step,
    )
    for idp in run.idps:
        assert idp.train_edp.extra.get("custom_key") == 42.0 + idp.batch_idx

    reloaded = NNRun.load(id=run.id)
    for idp in reloaded.idps:
        assert idp.train_edp.extra.get("custom_key") == 42.0 + idp.batch_idx


def test_back_compat_train_step_wrapper_still_callable(tmp_path, monkeypatch):
    """NNModel._train_step kept as a one-line wrapper around default_train_step
    so any hypothetical subclass that overrode the old _train_step keeps
    working. Train() no longer dispatches through it; this test exists to
    prove the wrapper itself still returns a valid EDP for callers that
    invoke it directly (e.g., a `super()._train_step(...)` call from a
    subclass override)."""
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)

    model = _make_model()
    # Need an optimizer on hand; build the same way train() does.
    from nnx.nn.enum.optims import Optims

    optimizer = Optims.ADAM(
        net=model.net,
        lr_start=1e-2,
        momentum=(0.9, 0.999),
        weight_decay=0.0,
    )

    loader = _make_loader(n=8)
    batch = next(iter(loader))

    edp = model._train_step(
        batch=batch,
        optimizer=optimizer,
        scaler=None,
    )
    assert isinstance(edp, NNEvaluationDataPoint)
    assert edp.loss is not None
    assert edp.error is not None


def test_autoencoder_style_step_trains_end_to_end(tmp_path, monkeypatch):
    """A reconstruction-loss step (no labels) trains end-to-end and the
    loss decreases meaningfully across the run. Mirrors example 05:
    FeedFwdNN with input_dim == output_dim is structurally an
    autoencoder (d → bottleneck → d), no separate decoder needed."""
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)

    d = 8
    n = 128
    X = torch.randn(n, d)
    y_dummy = torch.zeros(n, dtype=torch.long)  # satisfies the (X, Y) loader contract
    loader = DataLoader(TensorDataset(X, y_dummy), batch_size=16, shuffle=False)

    model = NNModel(
        net_params=NNParams(
            input_dim=d,
            output_dim=d,
            hidden_dims=[3],  # bottleneck at 3 < d
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )

    def autoencoder_step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        m = ctx.model
        m.net.train()
        m.net.zero_grad()

        X_batch, _ = m.net.unpack_batch(ctx.batch)
        X_batch = tuple(x.to(m.device) for x in X_batch)
        reconstructed = m.net(*X_batch)
        loss = torch.nn.functional.mse_loss(reconstructed, X_batch[0])
        loss.backward()
        ctx.optimizer.step()

        loss_val = float(loss.detach())
        return NNEvaluationDataPoint(
            f1=0.0,
            recall=0.0,
            accuracy=0.0,
            precision=0.0,
            loss=loss_val,
            error=loss_val,  # loss serves as the proxy error for BEST tracking
        )

    train_params = _make_train_params(loader, n_epochs=5)
    run = model.train(params=train_params, train_step_fn=autoencoder_step)

    # 8 batches per epoch × 5 epochs = 40 idps.
    assert len(run.idps) == 8 * 5
    assert all(torch.isfinite(torch.tensor(idp.train_edp.loss)) for idp in run.idps)

    # Loss should actually decrease: the autoencoder is real, its params
    # are in the optimizer (because input_dim == output_dim collapses
    # encoder+decoder into one FeedFwdNN), so training is meaningful.
    # Compare the mean loss of the first vs last epoch — bottleneck=3
    # on Gaussian inputs in 8d won't reach zero but should drop clearly.
    first_epoch_mean = sum(idp.train_edp.loss for idp in run.idps if idp.epoch_idx == 0) / 8
    last_epoch_mean = sum(idp.train_edp.loss for idp in run.idps if idp.epoch_idx == 4) / 8
    assert last_epoch_mean < first_epoch_mean, (
        f"autoencoder loss should decrease across epochs; got first={first_epoch_mean:.4f}, last={last_epoch_mean:.4f}"
    )

    # On-disk artifacts exist.
    assert (tmp_path / "runs" / run.id / "run.yaml").exists()
    assert (tmp_path / "runs" / run.id / "idps.csv").exists()
