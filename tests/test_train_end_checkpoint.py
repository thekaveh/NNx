"""``NNModel.train`` must persist net mutations made by ``on_train_end`` callbacks.

Regression tests for #87: the epoch loop saves the LAST checkpoint *inside* the
loop, but ``on_train_end`` fires in ``_CallbackFinalizer.__exit__`` — after the
final in-loop save. A callback that mutates ``model.net`` in ``on_train_end``
(the QAT ``convert()`` being the flagship case) therefore had its mutation
silently lost on disk: the in-memory net was converted, the persisted LAST was
pre-convert.

The fix re-writes the LAST checkpoint from the live net after the finalizer
exits, so the on-disk artifact always matches the post-``on_train_end`` net.
BEST semantics are deliberately untouched (BEST tracks the best *training-time*
state).
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNEvaluationDataPoint,
    NNIterationDataPoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
)
from nnx.nn.callbacks import Callback
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.nn.params.nn_checkpoint import NNCheckpoint, NNCheckpointTransform
from nnx.nn.params.nn_params import NNParams


def _tiny_model() -> NNModel:
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


def _tiny_train_params(loader: DataLoader, n_epochs: int = 2) -> NNTrainParams:
    return NNTrainParams(
        n_epochs=n_epochs,
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
            patience=1,
            cooldown=1,
            threshold=1e-3,
        ),
    )


def _loader() -> DataLoader:
    torch.manual_seed(0)
    X = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    return DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False)


class _MutateNetOnTrainEnd(Callback):
    """Structurally mutates the net in ``on_train_end`` — a minimal stand-in for
    QAT's ``convert()`` (which swaps Linear modules, changing state-dict keys)."""

    def on_train_end(self, ctx) -> None:  # noqa: ANN001 - Callback context
        ctx.model.net.register_buffer("post_train_end_marker", torch.tensor([42.0]))


class _DescribeTransform(Callback):
    def __init__(self, name: str):
        self.name = name
        self.completed = False

    def on_train_end(self, ctx) -> None:  # noqa: ANN001 - Callback context
        self.completed = True

    def checkpoint_transforms(self) -> tuple[NNCheckpointTransform, ...]:
        if not self.completed:
            return ()
        return (NNCheckpointTransform(name=self.name),)


def test_last_checkpoint_contains_on_train_end_mutation(tmp_path, monkeypatch):
    """A net mutation made in ``on_train_end`` must be present in the persisted
    LAST checkpoint (previously the final save preceded the finalizer)."""
    monkeypatch.chdir(tmp_path)
    model = _tiny_model()
    run = model.train(params=_tiny_train_params(_loader()), callbacks=[_MutateNetOnTrainEnd()])

    # In-memory net carries the mutation (sanity — this always held).
    assert "post_train_end_marker" in model.net.state_dict()

    # THE FIX: the on-disk LAST must carry it too.
    ckpt = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert ckpt is not None
    assert ckpt.transforms == ()
    assert "post_train_end_marker" in ckpt.net_state, (
        "LAST checkpoint was saved before on_train_end fired — the callback's net mutation was lost on disk (#87)"
    )
    assert torch.equal(ckpt.net_state["post_train_end_marker"], torch.tensor([42.0]))


def test_last_checkpoint_unchanged_without_mutating_callbacks(tmp_path, monkeypatch):
    """No-mutation regression guard: a plain train's LAST checkpoint still
    matches the live net exactly (the post-finalizer re-save is a content no-op)."""
    monkeypatch.chdir(tmp_path)
    model = _tiny_model()
    run = model.train(params=_tiny_train_params(_loader()))

    ckpt = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert ckpt is not None
    live = model.net.state_dict()
    assert set(ckpt.net_state.keys()) == set(live.keys())
    for k in live:
        assert torch.equal(ckpt.net_state[k], live[k])


def test_last_checkpoint_keeps_optimizer_sidecar(tmp_path, monkeypatch):
    """The re-saved LAST must not lose the optimizer sidecar (warm-resume path)."""
    monkeypatch.chdir(tmp_path)
    model = _tiny_model()
    run = model.train(params=_tiny_train_params(_loader()), callbacks=[_MutateNetOnTrainEnd()])

    opt_state = NNCheckpoint.load_optimizer_state(run=run.id, type=Checkpoints.LAST)
    assert opt_state is not None, "LAST .opt.pt sidecar missing after the post-train_end re-save"


def test_best_checkpoint_stays_pre_mutation(tmp_path, monkeypatch):
    """BEST semantics untouched: BEST tracks the best training-time state and
    must NOT absorb the on_train_end mutation."""
    monkeypatch.chdir(tmp_path)
    model = _tiny_model()
    run = model.train(params=_tiny_train_params(_loader()), callbacks=[_MutateNetOnTrainEnd()])

    best = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert best is not None
    assert "post_train_end_marker" not in best.net_state


def test_checkpoint_transform_order_matches_train_end_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = _DescribeTransform("first")
    second = _DescribeTransform("second")
    model = _tiny_model()
    run = model.train(params=_tiny_train_params(_loader()), callbacks=[first, second])

    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert checkpoint is not None
    assert [transform.name for transform in checkpoint.transforms] == ["second", "first"]


def test_transformed_last_sidecar_keeps_pre_transform_model_and_rng(tmp_path, monkeypatch):
    class MutateModelAndRng(Callback):
        completed = False
        rng_before = None
        model_before = None

        def on_train_end(self, ctx):
            self.rng_before = torch.get_rng_state().clone()
            self.model_before = {name: tensor.detach().clone() for name, tensor in ctx.model.net.state_dict().items()}
            torch.rand(1)
            with torch.no_grad():
                next(ctx.model.net.parameters()).add_(5.0)
            self.completed = True

        def checkpoint_transforms(self):
            return (NNCheckpointTransform(name="test-transform"),) if self.completed else ()

    monkeypatch.chdir(tmp_path)
    callback = MutateModelAndRng()
    model = _tiny_model()
    run = model.train(params=_tiny_train_params(_loader(), n_epochs=1), callbacks=[callback])

    checkpoint = NNCheckpoint.load(run.id, Checkpoints.LAST)
    state = NNCheckpoint.load_training_state(run.id, Checkpoints.LAST)
    assert checkpoint is not None and state is not None
    assert callback.rng_before is not None and callback.model_before is not None
    assert torch.equal(state["rng"]["torch"], callback.rng_before)
    for name, tensor in callback.model_before.items():
        assert torch.equal(state["model"][name], tensor)
    assert any(
        not torch.equal(checkpoint.net_state[name], callback.model_before[name]) for name in callback.model_before
    )


def test_qat_convert_state_persists_in_last(tmp_path, monkeypatch):
    """Flagship #87 case: QAT ``convert()`` (on_train_end) swaps Linear modules;
    the persisted LAST must carry the quantized keys, and reloading it into a
    prepared+converted net must round-trip. Skipped without the torchao extra."""
    pytest.importorskip("torchao")
    from nnx import QATLifecycleCallback, qat_train_step_factory

    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    # Hidden widths divide the default int4 groupsize (32).
    model = NNModel(
        net_params=NNParams(
            input_dim=32,
            output_dim=3,
            hidden_dims=[64, 64],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    X = torch.randn(32, 32)
    y = torch.randint(0, 3, (32,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)
    qat_cb = QATLifecycleCallback()
    run = model.train(
        params=_tiny_train_params(loader, n_epochs=1),
        train_step_fn=qat_train_step_factory(),
        callbacks=[qat_cb],
    )

    assert qat_cb.is_converted is True
    live_keys = set(model.net.state_dict().keys())
    ckpt = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert ckpt is not None
    saved_keys = set(ckpt.net_state.keys())
    # The persisted LAST mirrors the converted net exactly — including the
    # quantized parametrization keys convert() introduced.
    assert saved_keys == live_keys
    assert any(("scales" in k) or ("zeros" in k) for k in saved_keys), (
        f"no quantized keys persisted; saved keys: {sorted(saved_keys)[:8]}"
    )
    assert ckpt.transforms == (
        NNCheckpointTransform(
            name="torchao_qat",
            version=1,
            options={"qat_config": "8da4w", "groupsize": 32},
        ),
    )

    # Public round-trip: metadata rebuilds the converted topology before
    # load_state_dict, with no consumer-side prepare/convert workaround.
    fresh = NNModel.from_checkpoint(ckpt)
    fresh.net.eval()
    output = fresh.net(torch.randn(2, 32))
    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()
    assert any("Int8" in type(module).__name__ and "Int4" in type(module).__name__ for module in fresh.net.modules())

    # Warm resume uses the pre-conversion model state stored in the sidecar,
    # then applies QAT afresh before producing another converted LAST.
    resumed = NNModel(net_params=model.net_params, params=model.params)
    resumed_callback = QATLifecycleCallback()
    resumed_run = resumed.train(
        params=replace(
            _tiny_train_params(loader, n_epochs=1),
            data_id="qat-resume",
            resume_from_run_id=run.id,
        ),
        train_step_fn=qat_train_step_factory(),
        callbacks=[resumed_callback],
    )
    assert resumed_callback.is_converted is True
    assert NNCheckpoint.load(resumed_run.id, Checkpoints.LAST) is not None


def test_unknown_checkpoint_transform_fails_clearly():
    model = _tiny_model()
    checkpoint = NNCheckpoint(
        idp=NNIterationDataPoint(
            lr=0.0,
            iter_idx=0,
            epoch_idx=0,
            batch_idx=0,
            train_edp=NNEvaluationDataPoint(
                f1=0.0,
                recall=0.0,
                accuracy=0.0,
                precision=0.0,
                loss=0.0,
            ),
        ),
        model_params=model.params,
        net_params=model.net_params,
        net_state=model.net.state_dict(),
        transforms=(NNCheckpointTransform(name="future_transform", version=1),),
    )

    with pytest.raises(ValueError, match="unsupported checkpoint transform.*future_transform"):
        NNModel.from_checkpoint(checkpoint)


def test_legacy_converted_qat_checkpoint_has_targeted_error(tmp_path, monkeypatch):
    pytest.importorskip("torchao")
    from nnx import QATLifecycleCallback, qat_train_step_factory

    monkeypatch.chdir(tmp_path)
    model = NNModel(
        net_params=NNParams(
            input_dim=32,
            output_dim=3,
            hidden_dims=[64],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    loader = DataLoader(
        TensorDataset(torch.randn(16, 32), torch.randint(0, 3, (16,))),
        batch_size=8,
        shuffle=False,
    )
    run = model.train(
        params=_tiny_train_params(loader, n_epochs=1),
        train_step_fn=qat_train_step_factory(),
        callbacks=[QATLifecycleCallback()],
    )
    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert checkpoint is not None
    legacy = replace(checkpoint, transforms=())

    with pytest.raises(ValueError, match="converted QAT.*lacks reconstruction metadata"):
        NNModel.from_checkpoint(legacy)
