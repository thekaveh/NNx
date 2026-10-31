"""End-to-end test for the diffusion training step factory.

Exercises the full path: build a schedule, build the network, build the
step fn, drive it through NNModel.train(), and verify the loss is finite
and decreases as training proceeds. Saves into a tmp runs dir so the
test doesn't pollute the repo.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    DiffusionMLP,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNSchedulerParams,
    NNTrainParams,
    NoiseSchedulers,
    Optims,
    diffusion_train_step_factory,
    set_seed,
)
from tests._adapter_outputs import REMEDY, DictDenoiser, Extract


def _make_model() -> NNModel:
    """An NNModel with placeholder NNParams, whose .net is swapped for
    a DiffusionMLP. The placeholder mirrors DiffusionMLP's input dim so
    the run.yaml stays interpretable."""
    m = NNModel(
        net_params=NNParams(
            input_dim=2,
            output_dim=2,
            hidden_dims=[16],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )
    m.net = DiffusionMLP(input_dim=2, hidden_dims=[32, 32], time_embed_dim=16).to(m.device)
    return m


def _gaussian_mixture_loader(n: int = 256, batch_size: int = 64) -> DataLoader:
    """2D mixture of 4 isotropic Gaussians at the corners of a square.
    Small + multimodal — diffusion has to learn the structure, not just
    fit a single mean."""
    torch.manual_seed(0)
    centers = torch.tensor([[-2, -2], [-2, 2], [2, -2], [2, 2]], dtype=torch.float32)
    idx = torch.randint(0, 4, (n,))
    means = centers[idx]
    X = means + 0.3 * torch.randn(n, 2)
    y_dummy = torch.zeros(n, dtype=torch.long)
    return DataLoader(TensorDataset(X, y_dummy), batch_size=batch_size, shuffle=True)


def test_diffusion_train_step_runs_and_loss_decreases(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    set_seed(0)

    model = _make_model()
    loader = _gaussian_mixture_loader()
    schedule = NoiseSchedulers.LINEAR(T=100)

    step_fn = diffusion_train_step_factory(schedule)
    run = model.train(
        params=NNTrainParams(
            n_epochs=4,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=1e-3,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7,
                factor=0.5,
                patience=2,
                cooldown=1,
                threshold=1e-3,
            ),
        ),
        train_step_fn=step_fn,
    )

    assert run.idps is not None and len(run.idps) > 0
    losses = [idp.train_edp.loss for idp in run.idps]
    # All losses finite — no NaN/Inf from a misbroadcast schedule.
    for lo in losses:
        assert lo is not None
        assert lo == lo  # NaN-safe finiteness check
        assert abs(lo) < 1e9
    # Loss should decrease in expectation; we average the first vs last
    # quarter of iterations because per-step loss is noisy (each batch
    # samples a fresh random timestep).
    n = len(losses)
    early = sum(losses[: n // 4]) / max(1, n // 4)
    late = sum(losses[3 * n // 4 :]) / max(1, n - 3 * n // 4)
    assert late < early, (
        f"diffusion loss did not decrease across training: "
        f"early-quarter mean {early:.4f} vs late-quarter mean {late:.4f}"
    )


def test_diffusion_train_step_cosine_schedule(tmp_path, monkeypatch):
    """COSINE schedule should produce a working training step just like
    LINEAR — the factory is schedule-agnostic. Mirrors LINEAR's
    loss-decreases check rather than just asserting "ran without error"."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)

    model = _make_model()
    loader = _gaussian_mixture_loader()
    schedule = NoiseSchedulers.COSINE(T=100)

    run = model.train(
        params=NNTrainParams(
            n_epochs=4,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=1e-3,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7,
                factor=0.5,
                patience=2,
                cooldown=1,
                threshold=1e-3,
            ),
        ),
        train_step_fn=diffusion_train_step_factory(schedule),
    )
    losses = [idp.train_edp.loss for idp in run.idps]
    assert len(losses) > 0
    for lo in losses:
        assert lo is not None and torch.isfinite(torch.tensor(lo)).item()
    n = len(losses)
    early = sum(losses[: n // 4]) / max(1, n // 4)
    late = sum(losses[3 * n // 4 :]) / max(1, n - 3 * n // 4)
    assert late < early, f"diffusion COSINE loss did not decrease: early {early:.4f} vs late {late:.4f}"


def test_diffusion_step_fallback_unpacks_plain_tensor_batches(tmp_path, monkeypatch):
    """When the user supplies a custom nn.Module with NO `unpack_batch`
    method, the diffusion step falls through to the tuple-batch handling
    at training.py:75-77 (this loader yields ``(X, Y)`` tuples).
    The bare-tensor branch at training.py:78 is exercised by
    :func:`test_diffusion_step_fallback_bare_tensor_batch` below."""
    import torch.nn as nn

    monkeypatch.chdir(tmp_path)
    set_seed(0)

    class _PlainDiffusionNet(nn.Module):
        """Same shape as DiffusionMLP but without `unpack_batch` — forces
        the fallback path in diffusion_train_step_factory."""

        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2 + 8, 2)

        def forward(self, x, t):
            from nnx.diffusion import sinusoidal_time_embed

            t_emb = sinusoidal_time_embed(t, 8)
            return self.proj(torch.cat([x, t_emb], dim=-1))

    model = _make_model()
    model.net = _PlainDiffusionNet().to(model.device)
    assert not hasattr(model.net, "unpack_batch")

    schedule = NoiseSchedulers.LINEAR(T=50)
    run = model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=_gaussian_mixture_loader(n=32, batch_size=8),
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
        train_step_fn=diffusion_train_step_factory(schedule),
    )
    # Loop ran without AttributeError; fallback path is covered.
    assert len(run.idps) > 0
    assert all(idp.train_edp.loss is not None for idp in run.idps)


def test_diffusion_step_fallback_bare_tensor_batch(monkeypatch):
    """Drive the diffusion step against the bare-tensor branch at
    training.py:78 — the `else: x_0 = ctx.batch` case. We invoke the
    step function directly with a hand-built TrainStepContext to bypass
    DataLoader (which would always yield a list/tuple)."""
    import torch.nn as nn

    from nnx.nn.nn_model import TrainStepContext

    set_seed(0)

    class _PlainDiffusionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2 + 8, 2)

        def forward(self, x, t):
            from nnx.diffusion import sinusoidal_time_embed

            t_emb = sinusoidal_time_embed(t, 8)
            return self.proj(torch.cat([x, t_emb], dim=-1))

    model = _make_model()
    model.net = _PlainDiffusionNet().to(model.device)
    assert not hasattr(model.net, "unpack_batch")

    schedule = NoiseSchedulers.LINEAR(T=20)
    step_fn = diffusion_train_step_factory(schedule)
    optimizer = Optims.ADAM(
        net=model.net,
        lr_start=1e-3,
        momentum=(0.9, 0.999),
        weight_decay=0.0,
    )

    # Bare tensor — NOT wrapped in (X, Y). Hits the else-branch at line 78.
    bare_batch = torch.randn(8, 2)
    ctx = TrainStepContext(
        model=model,
        batch=bare_batch,
        optimizer=optimizer,
        scaler=None,
        grad_clip_norm=None,
        extra_metrics=None,
        accumulate_grad_batches=1,
        batch_idx=0,
        epoch_idx=0,
    )
    edp = step_fn(ctx)
    assert edp.loss is not None
    assert torch.isfinite(torch.tensor(edp.loss)).item()


def test_diffusion_step_reports_loss_and_error_equal(tmp_path, monkeypatch):
    """Documented invariant in diffusion/training.py: train_edp.loss and
    train_edp.error must be set to the same value so BEST tracking and
    ReduceLROnPlateau have a metric. Without this, the scheduler falls
    back through edp.error → edp.loss and works anyway, but the contract
    deserves a regression test — a future refactor could break it silently."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    model = _make_model()
    schedule = NoiseSchedulers.LINEAR(T=50)
    run = model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=_gaussian_mixture_loader(n=32, batch_size=8),
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=1e-3,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7,
                factor=0.5,
                patience=2,
                cooldown=1,
                threshold=1e-3,
            ),
        ),
        train_step_fn=diffusion_train_step_factory(schedule),
    )
    for idp in run.idps:
        assert idp.train_edp.loss == idp.train_edp.error, (
            f"diffusion step contract violated: loss={idp.train_edp.loss} != error={idp.train_edp.error}"
        )


def test_the_step_factory_refuses_a_schedule_enum_member_before_training():
    import pytest

    from nnx.diffusion import NoiseSchedulers, diffusion_train_step_factory

    with pytest.raises(TypeError, match="NoiseSchedule"):
        diffusion_train_step_factory(NoiseSchedulers.COSINE)  # type: ignore[arg-type]


# --- #561: the denoiser's return value through the model's batch adapter ---------------------------------------


def _adapter_params(*, seed: int = 1) -> NNTrainParams:
    """Two epochs of two uneven batches of 2-D points (labels unused)."""
    generator = torch.Generator().manual_seed(5)
    loader = [(torch.randn(n, 2, generator=generator), torch.zeros(n)) for n in (6, 4)]
    return NNTrainParams(
        n_epochs=2,
        train_loader=loader,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
        scheduler=NNSchedulerParams(min_lr=0.0, factor=0.5, patience=10, cooldown=0, threshold=0.0),
        seed=seed,
        save_phase_checkpoints=False,
        overwrite_existing=True,
    )


def test_a_dict_returning_denoiser_trains_through_the_adapters_output(tmp_path, monkeypatch):
    """The step called model.net(x_t, t) raw, so a denoiser returning a
    mapping (diffusers' UNet2DOutput) failed on ``eps_pred.shape`` (#561).
    Read through Extract("sample"), it trains exactly as the same weights
    returning a tensor do, with the default adapter or none at all."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    mlp = DiffusionMLP(input_dim=2, hidden_dims=[16], time_embed_dim=8)
    swapped = _make_model()  # a built-in model: no batch adapter
    swapped.net = copy.deepcopy(mlp)
    wrapped = NNModel(module=copy.deepcopy(mlp), params=NNModelParams(loss=Losses.MEAN_SQUARED_ERROR))
    extract = Extract("sample")
    mapped = NNModel(
        module=DictDenoiser(copy.deepcopy(mlp)),
        params=NNModelParams(loss=Losses.MEAN_SQUARED_ERROR),
        batch_adapter=extract,
    )
    schedule = NoiseSchedulers.LINEAR(T=20)
    losses = []
    for model in (swapped, wrapped, mapped):
        run = model.train(params=_adapter_params(), train_step_fn=diffusion_train_step_factory(schedule))
        losses.append([idp.train_edp.loss for idp in run.idps])

    assert all(loss is not None and math.isfinite(loss) for loss in losses[2])
    assert losses[0] == losses[1] == losses[2]
    reference = swapped.net.state_dict()
    for state in (wrapped.net.state_dict(), mapped.net.inner.state_dict()):
        assert state.keys() == reference.keys()
        assert all(torch.equal(state[k], reference[k]) for k in reference)
    assert extract.grad_modes == [True] * 4  # one denoiser forward per step: 2 epochs x 2 batches


def test_a_non_tensor_denoiser_output_names_the_adapter_remedy(tmp_path, monkeypatch):
    """With the default adapter, a mapping return raises BatchAdapter.output()'s
    TypeError, naming the remedy, not an AttributeError on ``.shape``."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    model = NNModel(
        module=DictDenoiser(DiffusionMLP(input_dim=2, hidden_dims=[16], time_embed_dim=8)),
        params=NNModelParams(loss=Losses.MEAN_SQUARED_ERROR),
    )
    with pytest.raises(TypeError, match=REMEDY):
        model.train(params=_adapter_params(), train_step_fn=diffusion_train_step_factory(NoiseSchedulers.LINEAR(T=20)))
