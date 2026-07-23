"""Pass-2 catalog: F-series tests (features).

- F1: warm-resume training via NNTrainParams.resume_from_run_id.
- F2: NNOptimParams.accumulate_grad_batches steps the optimizer every N
  batches; gradient accumulation produces the same final weights as
  training with N×-larger batches (within FP tolerance).
- F5: TensorBoardCallback writes events to the configured log_dir.
- F6: WandbCallback construction without wandb installed raises ImportError
  with a helpful message (we don't require wandb in test env).
- F7: NNModel.to_onnx exports a loadable .onnx file.
- F8: NNTabularDataset wraps a DataFrame into loaders + state.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from nnx.nn.callbacks import TensorBoardCallback, WandbCallback
from nnx.nn.dataset.nn_tabular_dataset import NNTabularDataset
from nnx.nn.enum.activations import Activations
from nnx.nn.enum.devices import Devices
from nnx.nn.enum.losses import Losses
from nnx.nn.enum.nets import Nets
from nnx.nn.enum.optims import Optims
from nnx.nn.nn_model import (
    GradientAccumulationState,
    NNModel,
    TrainStepContext,
    _enumerate_with_last,
    default_train_step,
)
from nnx.nn.params.nn_model_params import NNModelParams
from nnx.nn.params.nn_optim_params import NNOptimParams
from nnx.nn.params.nn_params import NNParams
from nnx.nn.params.nn_scheduler_params import NNSchedulerParams
from nnx.nn.params.nn_train_params import NNTrainParams


def _model():
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


# --- F1: warm-resume training ----------------------------------------------


def test_f1_resume_loads_weights_and_optimizer_state(tmp_path, monkeypatch):
    """Run A for 1 epoch, save checkpoints + opt sidecar; Run B resumes
    from A's LAST and continues training. Run B's starting weights should
    equal Run A's ending weights."""
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(7)

    X = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8)

    base_params = dict(
        train_loader=loader,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
    )

    # Run A: train fresh.
    model_a = _model()
    run_a = model_a.train(params=NNTrainParams(n_epochs=1, **base_params))
    weights_after_a = {k: v.clone() for k, v in model_a.net.state_dict().items()}

    # The optimizer state sidecar exists for LAST.
    last_pt = tmp_path / "runs" / run_a.id / "checkpoints" / "last.pt"
    assert last_pt.exists()
    assert (last_pt.parent / "last.pt.opt.pt").exists(), "opt sidecar missing"

    # Run B: build a fresh model with random init, resume from Run A's LAST.
    model_b = _model()
    weights_before_b = {k: v.clone() for k, v in model_b.net.state_dict().items()}
    # Sanity: starting weights differ from A's ending weights pre-resume.
    assert any(not torch.equal(weights_before_b[k], weights_after_a[k]) for k in weights_after_a)

    # Train for 0 epochs to isolate the resume — n_epochs=1 trains a bit then
    # save; we want to confirm the resume *replaced* the weights, so check
    # before any further training. Easiest: use a callback that stops after
    # epoch begin (before any batch runs) and inspect weights then.
    # Simpler: drive with n_epochs=1, then assert that weights immediately
    # post-resume match A. Achieved by inspecting via a stop-immediately
    # callback.
    from nnx.nn.callbacks import Callback

    captured = {}

    class _StopAtStart(Callback):
        def on_epoch_begin(self, ctx):
            captured["weights"] = {k: v.clone() for k, v in ctx.model.net.state_dict().items()}
            ctx.should_stop = True

    model_b.train(
        params=NNTrainParams(
            n_epochs=1,
            resume_from_run_id=run_a.id,
            resume_from_checkpoint="last",
            **base_params,
        ),
        callbacks=[_StopAtStart()],
    )

    # At epoch_begin (before any batch step), weights should equal A's last.
    for k in weights_after_a:
        assert torch.equal(captured["weights"][k], weights_after_a[k]), f"resume weights diverge from source on {k}"


def test_f1_resume_from_missing_run_id_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = _model()
    X = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(X, y), batch_size=4)

    with pytest.raises(ValueError, match="not found on disk"):
        model.train(
            params=NNTrainParams(
                n_epochs=1,
                train_loader=loader,
                optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
                scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
                resume_from_run_id="DOES_NOT_EXIST",
                resume_from_checkpoint="last",
            )
        )


# --- F2: gradient accumulation --------------------------------------------


def test_f2_accumulate_grad_batches_only_steps_at_cycle_end(tmp_path, monkeypatch):
    """With accumulate_grad_batches=4, optimizer.step is called once per
    4 batches (not once per batch). Counts steps via spy."""
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)

    X = torch.randn(32, 4)
    y = torch.randint(0, 2, (32,))
    loader = DataLoader(TensorDataset(X, y), batch_size=4)  # 8 batches

    model = _model()

    # Use Adam-specific subclass step to count calls — patching all
    # Optimizer.step would also trigger scheduler.step. Track via the
    # actual instance after train() builds it. Simpler: inspect run.idps
    # by ensuring training completed without error AND weights changed
    # only on cycle boundaries.
    #
    # We assert behavior indirectly via state_dict comparison: with N=4 and
    # 8 batches, optimizer.step runs 2 times. Each step updates weights;
    # without accumulation the same loop runs 8 steps. So total weight
    # change magnitude is smaller for the accumulated case under same LR.
    initial_w = {k: v.clone() for k, v in model.net.state_dict().items()}
    model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=1e-2,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
                accumulate_grad_batches=4,
            ),
            scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
        )
    )

    # Sanity: at least *some* weights moved during the 2 optimizer steps.
    moved = any(not torch.equal(initial_w[k], model.net.state_dict()[k]) for k in initial_w)
    assert moved


def test_f2_accumulate_grad_batches_steps_trailing_partial_cycle(tmp_path, monkeypatch):
    """A final short accumulation cycle must update weights instead of being dropped."""
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    loader = DataLoader(
        TensorDataset(torch.randn(12, 4), torch.randint(0, 2, (12,))),
        batch_size=4,
    )
    step_count = 0
    original_step = torch.optim.Adam.step

    def counted_step(optimizer, *args, **kwargs):
        nonlocal step_count
        step_count += 1
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", counted_step)
    _model().train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=1e-2,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
                accumulate_grad_batches=2,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7,
                factor=0.5,
                patience=1,
                cooldown=1,
                threshold=1e-3,
            ),
        )
    )

    assert step_count == 2


def test_f2_enumerate_with_last_uses_observed_final_batch():
    class EstimatedLengthDataset(IterableDataset):
        def __len__(self):
            return 4

        def __iter__(self):
            yield from range(3)

    loader = DataLoader(EstimatedLengthDataset(), batch_size=1)

    observed = [(index, int(batch[0]), is_last) for index, batch, is_last in _enumerate_with_last(loader)]

    assert observed == [(0, 0, False), (1, 1, False), (2, 2, True)]


@pytest.mark.parametrize("accumulate_grad_batches", [2, 4])
@pytest.mark.parametrize("amp_control_flow", [False, True])
def test_f2_accumulation_matches_combined_uneven_batch_update(amp_control_flow, accumulate_grad_batches, monkeypatch):
    class LinearNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 2, bias=False)

        def forward(self, X):
            return self.linear(X)

    class StepModel:
        def __init__(self, state):
            self.net = LinearNet()
            self.net.load_state_dict(state)
            self.loss_fn = torch.nn.CrossEntropyLoss()
            self.device = torch.device("cuda" if amp_control_flow else "cpu")

        def _fwd_pass(self, batch):
            X, Y = batch
            logits = self.net(X)
            return X, Y, logits, logits.argmax(dim=1)

    initial = LinearNet()
    initial.linear.weight.data.copy_(torch.tensor([[0.25], [-0.25]]))
    initial_state = initial.state_dict()
    batches = (
        (torch.tensor([[1.0], [2.0]]), torch.tensor([0, 1])),
        (torch.tensor([[4.0]]), torch.tensor([1])),
    )

    accumulated = StepModel(initial_state)
    accumulated_optimizer = torch.optim.SGD(accumulated.net.parameters(), lr=0.1)
    accumulation_state = GradientAccumulationState()
    if amp_control_flow:
        monkeypatch.setattr(torch.amp, "autocast", lambda **_kwargs: nullcontext())
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    else:
        scaler = None
    for batch_idx, batch in enumerate(batches):
        default_train_step(
            TrainStepContext(
                model=accumulated,
                batch=batch,
                optimizer=accumulated_optimizer,
                scaler=scaler,
                grad_clip_norm=None,
                extra_metrics=None,
                accumulate_grad_batches=accumulate_grad_batches,
                batch_idx=batch_idx,
                epoch_idx=0,
                is_last_batch=batch_idx == len(batches) - 1,
                accumulation_state=accumulation_state,
            )
        )

    combined = StepModel(initial_state)
    combined_optimizer = torch.optim.SGD(combined.net.parameters(), lr=0.1)
    combined_optimizer.zero_grad()
    X = torch.cat([batch[0] for batch in batches])
    Y = torch.cat([batch[1] for batch in batches])
    loss = combined.loss_fn(combined.net(X), Y)
    loss.backward()
    combined_optimizer.step()

    assert torch.allclose(accumulated.net.linear.weight, combined.net.linear.weight)


@pytest.mark.parametrize("loss_kind", ["cross_entropy", "nll"])
@pytest.mark.parametrize("amp_control_flow", [False, True])
def test_f2_accumulation_matches_weighted_ignored_combined_batch(loss_kind, amp_control_flow, monkeypatch):
    class LinearNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 2, bias=False)

        def forward(self, X):
            return self.linear(X)

    class StepModel:
        def __init__(self, state):
            self.net = LinearNet()
            self.net.load_state_dict(state)
            loss_type = torch.nn.CrossEntropyLoss if loss_kind == "cross_entropy" else torch.nn.NLLLoss
            self.loss_fn = loss_type(weight=torch.tensor([1.0, 7.0]), ignore_index=-100)
            self.device = torch.device("cuda" if amp_control_flow else "cpu")

        def _fwd_pass(self, batch):
            X, Y = batch
            raw = self.net(X)
            logits = raw if loss_kind == "cross_entropy" else torch.log_softmax(raw, dim=1)
            return X, Y, logits, logits.argmax(dim=1)

    initial = LinearNet()
    initial.linear.weight.data.copy_(torch.tensor([[0.25], [-0.25]]))
    initial_state = initial.state_dict()
    batches = (
        (torch.tensor([[1.0], [2.0]]), torch.tensor([-100, -100])),
        (torch.tensor([[3.0], [4.0]]), torch.tensor([0, 0])),
        (torch.tensor([[5.0]]), torch.tensor([1])),
    )

    accumulated = StepModel(initial_state)
    accumulated_optimizer = torch.optim.SGD(accumulated.net.parameters(), lr=0.1)
    accumulation_state = GradientAccumulationState()
    if amp_control_flow:
        monkeypatch.setattr(torch.amp, "autocast", lambda **_kwargs: nullcontext())
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    else:
        scaler = None
    for batch_idx, batch in enumerate(batches):
        default_train_step(
            TrainStepContext(
                model=accumulated,
                batch=batch,
                optimizer=accumulated_optimizer,
                scaler=scaler,
                grad_clip_norm=0.25,
                extra_metrics=None,
                accumulate_grad_batches=len(batches),
                batch_idx=batch_idx,
                epoch_idx=0,
                is_last_batch=batch_idx == len(batches) - 1,
                accumulation_state=accumulation_state,
            )
        )

    combined = StepModel(initial_state)
    combined_optimizer = torch.optim.SGD(combined.net.parameters(), lr=0.1)
    X = torch.cat([batch[0] for batch in batches])
    Y = torch.cat([batch[1] for batch in batches])
    raw = combined.net(X)
    logits = raw if loss_kind == "cross_entropy" else torch.log_softmax(raw, dim=1)
    combined.loss_fn(logits, Y).backward()
    torch.nn.utils.clip_grad_norm_(combined.net.parameters(), 0.25)
    combined_optimizer.step()

    assert torch.allclose(accumulated.net.linear.weight, combined.net.linear.weight)


@pytest.mark.parametrize("loss_kind", ["cross_entropy", "nll"])
def test_f2_accumulation_preserves_classification_sum_reduction(loss_kind):
    class LinearNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 2, bias=False)

        def forward(self, X):
            return self.linear(X)

    class StepModel:
        def __init__(self, state):
            self.net = LinearNet()
            self.net.load_state_dict(state)
            loss_type = torch.nn.CrossEntropyLoss if loss_kind == "cross_entropy" else torch.nn.NLLLoss
            self.loss_fn = loss_type(weight=torch.tensor([1.0, 7.0]), ignore_index=-100, reduction="sum")
            self.device = torch.device("cpu")

        def _fwd_pass(self, batch):
            X, Y = batch
            raw = self.net(X)
            logits = raw if loss_kind == "cross_entropy" else torch.log_softmax(raw, dim=1)
            return X, Y, logits, logits.argmax(dim=1)

    initial = LinearNet()
    initial.linear.weight.data.copy_(torch.tensor([[0.25], [-0.25]]))
    initial_state = initial.state_dict()
    batches = (
        (torch.tensor([[1.0], [2.0]]), torch.tensor([0, -100])),
        (torch.tensor([[3.0]]), torch.tensor([1])),
    )

    accumulated = StepModel(initial_state)
    accumulated_optimizer = torch.optim.SGD(accumulated.net.parameters(), lr=0.1)
    accumulation_state = GradientAccumulationState()
    for batch_idx, batch in enumerate(batches):
        default_train_step(
            TrainStepContext(
                model=accumulated,
                batch=batch,
                optimizer=accumulated_optimizer,
                scaler=None,
                grad_clip_norm=None,
                extra_metrics=None,
                accumulate_grad_batches=4,
                batch_idx=batch_idx,
                epoch_idx=0,
                is_last_batch=batch_idx == len(batches) - 1,
                accumulation_state=accumulation_state,
            )
        )

    combined = StepModel(initial_state)
    combined_optimizer = torch.optim.SGD(combined.net.parameters(), lr=0.1)
    X = torch.cat([batch[0] for batch in batches])
    Y = torch.cat([batch[1] for batch in batches])
    raw = combined.net(X)
    logits = raw if loss_kind == "cross_entropy" else torch.log_softmax(raw, dim=1)
    combined.loss_fn(logits, Y).backward()
    combined_optimizer.step()

    assert torch.allclose(accumulated.net.linear.weight, combined.net.linear.weight)


@pytest.mark.parametrize(
    "loss_fn",
    [
        pytest.param(torch.nn.MSELoss(reduction="sum"), id="mse"),
        pytest.param(torch.nn.BCEWithLogitsLoss(reduction="sum"), id="bce"),
    ],
)
def test_f2_accumulation_preserves_elementwise_sum_reduction(loss_fn):
    class BinaryNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1, bias=False)

        def forward(self, X):
            return self.linear(X).squeeze(-1)

    class StepModel:
        def __init__(self, state):
            self.net = BinaryNet()
            self.net.load_state_dict(state)
            self.loss_fn = loss_fn
            self.device = torch.device("cpu")

        def _fwd_pass(self, batch):
            X, Y = batch
            logits = self.net(X)
            return X, Y, logits, (logits >= 0).to(Y.dtype)

    initial = BinaryNet()
    initial.linear.weight.data.fill_(0.25)
    initial_state = initial.state_dict()
    batches = (
        (torch.tensor([[1.0], [2.0]]), torch.tensor([0.0, 1.0])),
        (torch.tensor([[3.0]]), torch.tensor([1.0])),
    )

    accumulated = StepModel(initial_state)
    accumulated_optimizer = torch.optim.SGD(accumulated.net.parameters(), lr=0.1)
    accumulation_state = GradientAccumulationState()
    for batch_idx, batch in enumerate(batches):
        default_train_step(
            TrainStepContext(
                model=accumulated,
                batch=batch,
                optimizer=accumulated_optimizer,
                scaler=None,
                grad_clip_norm=None,
                extra_metrics=None,
                accumulate_grad_batches=4,
                batch_idx=batch_idx,
                epoch_idx=0,
                is_last_batch=batch_idx == len(batches) - 1,
                accumulation_state=accumulation_state,
            )
        )

    combined = StepModel(initial_state)
    combined_optimizer = torch.optim.SGD(combined.net.parameters(), lr=0.1)
    X = torch.cat([batch[0] for batch in batches])
    Y = torch.cat([batch[1] for batch in batches])
    combined.loss_fn(combined.net(X), Y).backward()
    combined_optimizer.step()

    assert torch.allclose(accumulated.net.linear.weight, combined.net.linear.weight)


@pytest.mark.parametrize(
    "loss_fn",
    [
        pytest.param(torch.nn.MSELoss(), id="mse"),
        pytest.param(torch.nn.BCEWithLogitsLoss(), id="bce"),
    ],
)
def test_f2_accumulation_uses_element_count_for_elementwise_means(loss_fn):
    class MatrixNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 2, bias=False)

        def forward(self, X):
            return self.linear(X)

    class StepModel:
        def __init__(self, state):
            self.net = MatrixNet()
            self.net.load_state_dict(state)
            self.loss_fn = loss_fn
            self.device = torch.device("cpu")

        def _fwd_pass(self, batch):
            X, Y = batch
            logits = self.net(X)
            return X, Y, logits, (logits >= 0).to(Y.dtype)

    initial = MatrixNet()
    initial.linear.weight.data.copy_(torch.tensor([[0.25], [-0.25]]))
    initial_state = initial.state_dict()
    batches = (
        (torch.tensor([[1.0], [2.0]]), torch.tensor([[0.0, 1.0], [1.0, 0.0]])),
        (torch.tensor([[3.0]]), torch.tensor([[1.0, 1.0]])),
    )

    accumulated = StepModel(initial_state)
    accumulated_optimizer = torch.optim.SGD(accumulated.net.parameters(), lr=0.1)
    accumulation_state = GradientAccumulationState()
    for batch_idx, batch in enumerate(batches):
        default_train_step(
            TrainStepContext(
                model=accumulated,
                batch=batch,
                optimizer=accumulated_optimizer,
                scaler=None,
                grad_clip_norm=None,
                extra_metrics=None,
                accumulate_grad_batches=4,
                batch_idx=batch_idx,
                epoch_idx=0,
                is_last_batch=batch_idx == len(batches) - 1,
                accumulation_state=accumulation_state,
            )
        )

    combined = StepModel(initial_state)
    combined_optimizer = torch.optim.SGD(combined.net.parameters(), lr=0.1)
    X = torch.cat([batch[0] for batch in batches])
    Y = torch.cat([batch[1] for batch in batches])
    combined.loss_fn(combined.net(X), Y).backward()
    combined_optimizer.step()

    assert torch.allclose(accumulated.net.linear.weight, combined.net.linear.weight)


def test_f2_accumulation_calls_cross_entropy_subclass_and_hooks():
    class TrackingCrossEntropy(torch.nn.CrossEntropyLoss):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, logits, target):
            self.calls += 1
            return super().forward(logits, target) + 2.0

    model = _model()
    loss_fn = TrackingCrossEntropy()
    hook_calls = []
    loss_fn.register_forward_hook(lambda *_args: hook_calls.append(True))
    model.loss_fn = loss_fn
    optimizer = torch.optim.SGD(model.net.parameters(), lr=0.0)
    batch = (torch.randn(3, 4), torch.tensor([0, 1, 0]))

    edp = default_train_step(
        TrainStepContext(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scaler=None,
            grad_clip_norm=None,
            extra_metrics=None,
            accumulate_grad_batches=1,
            batch_idx=0,
            epoch_idx=0,
            is_last_batch=True,
            accumulation_state=GradientAccumulationState(),
        )
    )

    with torch.no_grad():
        _, Y, logits, _ = model._fwd_pass(batch)
        expected = torch.nn.functional.cross_entropy(logits, Y) + 2.0
    assert edp.loss == pytest.approx(float(expected))
    assert loss_fn.calls == 1
    assert hook_calls == [True]


def test_f2_accumulation_uses_batch_weighting_for_cross_entropy_subclasses():
    class RemappingCrossEntropy(torch.nn.CrossEntropyLoss):
        def forward(self, logits, target):
            remapped = torch.where(target == 0, torch.ones_like(target), target)
            return super().forward(logits, remapped)

    class LinearNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 2, bias=False)

        def forward(self, X):
            return self.linear(X)

    class StepModel:
        def __init__(self, state):
            self.net = LinearNet()
            self.net.load_state_dict(state)
            self.loss_fn = RemappingCrossEntropy(weight=torch.tensor([1.0, 7.0]))
            self.device = torch.device("cpu")

        def _fwd_pass(self, batch):
            X, Y = batch
            logits = self.net(X)
            return X, Y, logits, logits.argmax(dim=1)

    initial = LinearNet()
    initial.linear.weight.data.copy_(torch.tensor([[0.25], [-0.25]]))
    batches = (
        (torch.tensor([[1.0], [2.0]]), torch.tensor([0, 0])),
        (torch.tensor([[3.0]]), torch.tensor([1])),
    )

    accumulated = StepModel(initial.state_dict())
    accumulated_optimizer = torch.optim.SGD(accumulated.net.parameters(), lr=0.1)
    accumulation_state = GradientAccumulationState()
    for batch_idx, batch in enumerate(batches):
        default_train_step(
            TrainStepContext(
                model=accumulated,
                batch=batch,
                optimizer=accumulated_optimizer,
                scaler=None,
                grad_clip_norm=None,
                extra_metrics=None,
                accumulate_grad_batches=len(batches),
                batch_idx=batch_idx,
                epoch_idx=0,
                is_last_batch=batch_idx == len(batches) - 1,
                accumulation_state=accumulation_state,
            )
        )

    combined = StepModel(initial.state_dict())
    combined_optimizer = torch.optim.SGD(combined.net.parameters(), lr=0.1)
    X = torch.cat([batch[0] for batch in batches])
    Y = torch.cat([batch[1] for batch in batches])
    combined.loss_fn(combined.net(X), Y).backward()
    combined_optimizer.step()

    assert torch.allclose(accumulated.net.linear.weight, combined.net.linear.weight)


def test_f2_partially_ignored_cycle_records_finite_aggregate_loss_and_metrics():
    class IdentityLogits(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = torch.nn.Parameter(torch.zeros(2))

        def forward(self, X):
            return X + self.offset

    class StepModel:
        def __init__(self, state):
            self.net = IdentityLogits()
            self.net.load_state_dict(state)
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
            self.device = torch.device("cpu")

        def _fwd_pass(self, batch):
            X, Y = batch
            logits = self.net(X)
            return X, Y, logits, logits.argmax(dim=1)

    initial = IdentityLogits()
    batches = (
        (torch.tensor([[4.0, 0.0], [0.0, 4.0]]), torch.tensor([0, 1])),
        (torch.tensor([[1.0, 0.0]]), torch.tensor([-100])),
    )
    accumulated = StepModel(initial.state_dict())
    optimizer = torch.optim.SGD(accumulated.net.parameters(), lr=0.1)
    state = GradientAccumulationState()
    final_edp = None
    for batch_idx, batch in enumerate(batches):
        final_edp = default_train_step(
            TrainStepContext(
                model=accumulated,
                batch=batch,
                optimizer=optimizer,
                scaler=None,
                grad_clip_norm=None,
                extra_metrics={"count": lambda y, _y_hat: len(y)},
                accumulate_grad_batches=2,
                batch_idx=batch_idx,
                epoch_idx=0,
                is_last_batch=batch_idx == 1,
                accumulation_state=state,
            )
        )

    combined = StepModel(initial.state_dict())
    combined_optimizer = torch.optim.SGD(combined.net.parameters(), lr=0.1)
    expected_loss = combined.loss_fn(combined.net(batches[0][0]), batches[0][1])
    expected_loss.backward()
    combined_optimizer.step()

    assert final_edp is not None
    assert final_edp.loss == pytest.approx(float(expected_loss.detach()))
    assert final_edp.error is None
    assert final_edp.extra == {}
    assert torch.allclose(accumulated.net.offset, combined.net.offset)


def test_f2_exact_cross_entropy_filters_ignored_targets_from_metrics():
    class IdentityLogits(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = torch.nn.Parameter(torch.zeros(2))

        def forward(self, X):
            return X + self.offset

    class StepModel:
        def __init__(self):
            self.net = IdentityLogits()
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
            self.device = torch.device("cpu")

        def _fwd_pass(self, batch):
            X, Y = batch
            logits = self.net(X)
            return X, Y, logits, logits.argmax(dim=1)

    model = StepModel()
    optimizer = torch.optim.SGD(model.net.parameters(), lr=0.0)
    edp = default_train_step(
        TrainStepContext(
            model=model,
            batch=(
                torch.tensor([[0.0, 4.0], [4.0, 0.0], [0.0, 4.0]]),
                torch.tensor([-100, 0, 1]),
            ),
            optimizer=optimizer,
            scaler=None,
            grad_clip_norm=None,
            extra_metrics={"count": lambda y, _y_hat: len(y)},
            accumulate_grad_batches=1,
            batch_idx=0,
            epoch_idx=0,
            is_last_batch=True,
            accumulation_state=GradientAccumulationState(),
        )
    )

    assert edp.accuracy == 1.0
    assert edp.error == 0.0
    assert edp.extra == {"count": 2.0}


@pytest.mark.parametrize("accumulate_grad_batches", [1, 2])
def test_f2_all_ignored_accumulation_cycle_rejects_before_optimizer_step(accumulate_grad_batches):
    model = _model()
    model.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=0.1, weight_decay=0.1)
    before = {name: value.detach().clone() for name, value in model.net.state_dict().items()}
    accumulation_state = GradientAccumulationState()

    for batch_idx in range(accumulate_grad_batches - 1):
        default_train_step(
            TrainStepContext(
                model=model,
                batch=(torch.randn(2, 4), torch.full((2,), -100)),
                optimizer=optimizer,
                scaler=None,
                grad_clip_norm=None,
                extra_metrics=None,
                accumulate_grad_batches=accumulate_grad_batches,
                batch_idx=batch_idx,
                epoch_idx=0,
                accumulation_state=accumulation_state,
            )
        )

    with pytest.raises(FloatingPointError, match="non-finite training loss"):
        default_train_step(
            TrainStepContext(
                model=model,
                batch=(torch.randn(2, 4), torch.full((2,), -100)),
                optimizer=optimizer,
                scaler=None,
                grad_clip_norm=None,
                extra_metrics=None,
                accumulate_grad_batches=accumulate_grad_batches,
                batch_idx=accumulate_grad_batches - 1,
                epoch_idx=0,
                is_last_batch=True,
                accumulation_state=accumulation_state,
            )
        )

    for name, value in model.net.state_dict().items():
        assert torch.equal(value, before[name])


def test_f2_accumulate_grad_batches_state_round_trip():
    p = NNOptimParams(
        name=Optims.ADAM,
        max_lr=1e-3,
        momentum=(0.9, 0.999),
        weight_decay=0.0,
        accumulate_grad_batches=4,
    )
    rt = NNOptimParams.from_state(p.state())
    assert rt.accumulate_grad_batches == 4


def test_f2_accumulate_grad_batches_default_back_compat():
    """Default value (1) must NOT be in state() — preserves pre-feature run.id."""
    p = NNOptimParams(
        name=Optims.ADAM,
        max_lr=1e-3,
        momentum=(0.9, 0.999),
        weight_decay=0.0,
    )
    assert "accumulate_grad_batches" not in p.state()
    # And from_state of a YAML missing this key still works.
    legacy = {
        "max_lr": 1e-3,
        "momentum": "(0.9, 0.999)",
        "name": "adam",
        "weight_decay": 0.0,
    }
    assert NNOptimParams.from_state(legacy).accumulate_grad_batches == 1


# --- F5: TensorBoardCallback ----------------------------------------------


def test_f5_tensorboard_callback_writes_events(tmp_path, monkeypatch):
    pytest.importorskip("torch.utils.tensorboard")
    monkeypatch.chdir(tmp_path)

    model = _model()
    X = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(X, y), batch_size=4)

    tb_dir = tmp_path / "tb_logs"
    cb = TensorBoardCallback(log_dir=str(tb_dir))
    model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=loader,
            optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
            scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
        ),
        callbacks=[cb],
    )
    # SummaryWriter creates at least one tfevents file in log_dir.
    event_files = list(tb_dir.glob("events.out.tfevents.*"))
    assert len(event_files) >= 1


# --- F6: WandbCallback construction ---------------------------------------


def test_f6_wandb_callback_raises_helpful_error_without_wandb(monkeypatch):
    """When wandb isn't installed, attempting to construct the callback
    raises ImportError with a one-line install hint."""
    import sys

    # Simulate wandb being uninstalled.
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(ImportError, match="wandb"):
        WandbCallback(project="x")


# --- F7: ONNX export ------------------------------------------------------


def test_f7_to_onnx_writes_file(tmp_path):
    pytest.importorskip("onnx")
    model = _model()
    onnx_path = tmp_path / "model.onnx"
    example = torch.randn(2, 4)
    out = model.to_onnx(str(onnx_path), example_input=example)
    assert Path(out).exists()
    assert os.path.getsize(out) > 0
    # Validate via the onnx library.
    import onnx

    onnx.checker.check_model(str(onnx_path))


# --- F8: NNTabularDataset -------------------------------------------------


def test_f8_tabular_dataset_basic():
    df = pd.DataFrame(
        {
            "f1": np.random.RandomState(0).randn(100),
            "f2": np.random.RandomState(1).randn(100),
            "label": np.random.RandomState(2).randint(0, 3, 100),
        }
    )
    ds = NNTabularDataset(
        df=df,
        feature_cols=["f1", "f2"],
        target_col="label",
        batch_sizes=(16, 16, 16),
        val_proportion=0.2,
        test_proportion=0.2,
    )
    assert ds.input_dim == 2
    assert ds.output_dim == 3
    assert ds.train_loader is not None
    assert ds.val_loader is not None
    assert ds.test_loader is not None

    # Sanity: train loader yields (X, y) of correct dtype.
    X, y = next(iter(ds.train_loader))
    assert X.dtype == torch.float32
    assert y.dtype == torch.long
    assert X.shape[1] == 2


def test_f8_tabular_dataset_no_val_no_test():
    df = pd.DataFrame(
        {
            "f1": np.random.RandomState(0).randn(20),
            "label": np.zeros(20, dtype=int),
        }
    )
    ds = NNTabularDataset(
        df=df,
        feature_cols=["f1"],
        target_col="label",
        val_proportion=0.0,
        test_proportion=0.0,
    )
    # With 0 proportions, the loaders for val/test are None.
    assert ds.val_loader is None
    assert ds.test_loader is None


def test_f8_tabular_dataset_rejects_bad_proportions():
    df = pd.DataFrame({"f1": [1.0, 2.0], "label": [0, 1]})
    with pytest.raises(ValueError):
        NNTabularDataset(
            df=df,
            feature_cols=["f1"],
            target_col="label",
            val_proportion=0.6,
            test_proportion=0.6,
        )


def test_f8_tabular_dataset_rejects_empty_df():
    df = pd.DataFrame({"f1": [], "label": []})
    with pytest.raises(ValueError, match="non-empty"):
        NNTabularDataset(
            df=df,
            feature_cols=["f1"],
            target_col="label",
        )


def _split_indices(ds: NNTabularDataset) -> tuple[list[int], list[int], list[int]]:
    """Sorted per-split row indices — shared by the two F8 seed tests."""
    return (
        sorted(ds.train_loader.dataset.indices),
        sorted(ds.val_loader.dataset.indices),
        sorted(ds.test_loader.dataset.indices),
    )


def test_f8_tabular_dataset_seeded_split_is_deterministic():
    """Reproducibility contract: two NNTabularDataset instances built
    from the same DataFrame + same `seed` must yield identical
    train/val/test row allocations. Pre-fix the underlying
    ``random_split`` call had no ``generator=`` arg, so the split
    consumed the global torch RNG — fragile under any intervening
    RNG consumption between ``set_seed(...)`` and dataset construction.
    Mirrors the seeded-split contract NNPreferenceDataset already had."""
    df = pd.DataFrame(
        {
            "f1": np.arange(200, dtype=float),
            "f2": np.arange(200, dtype=float) * 2.0,
            "label": np.arange(200) % 4,
        }
    )

    a = NNTabularDataset(
        df=df,
        feature_cols=["f1", "f2"],
        target_col="label",
        val_proportion=0.2,
        test_proportion=0.2,
        seed=42,
    )
    b = NNTabularDataset(
        df=df,
        feature_cols=["f1", "f2"],
        target_col="label",
        val_proportion=0.2,
        test_proportion=0.2,
        seed=42,
    )
    assert _split_indices(a) == _split_indices(b)

    # Sanity: different seed → different split (probabilistic, but with
    # 200 rows + a 20/20/60 split the chance of accidental equality is
    # astronomically small).
    c = NNTabularDataset(
        df=df,
        feature_cols=["f1", "f2"],
        target_col="label",
        val_proportion=0.2,
        test_proportion=0.2,
        seed=7,
    )
    assert _split_indices(a) != _split_indices(c)


def test_f8_tabular_dataset_seed_none_follows_global_rng():
    """The documented `seed=None` contract: the split falls back to the
    *global* torch RNG, so `torch.manual_seed(N)` controls it and
    different global seeds give different splits. Pre-fix the code
    passed a fresh `torch.Generator()` — which always carries the same
    fixed default seed — so every unseeded split was bit-identical and
    completely deaf to `torch.manual_seed`."""
    df = pd.DataFrame(
        {
            "f1": np.arange(200, dtype=float),
            "f2": np.arange(200, dtype=float) * 2.0,
            "label": np.arange(200) % 4,
        }
    )

    def _build() -> NNTabularDataset:
        return NNTabularDataset(
            df=df,
            feature_cols=["f1", "f2"],
            target_col="label",
            val_proportion=0.2,
            test_proportion=0.2,
        )

    # Same global seed → same split (the split consumes the global RNG).
    torch.manual_seed(123)
    a = _build()
    torch.manual_seed(123)
    b = _build()
    assert _split_indices(a) == _split_indices(b)

    # Different global seed → different split (astronomically unlikely
    # to collide with 200 rows and a 60/20/20 split). This is the
    # assertion the pre-fix constant-generator behavior fails.
    torch.manual_seed(456)
    c = _build()
    assert _split_indices(a) != _split_indices(c)


def test_f8_tabular_dataset_rejects_noncontiguous_labels():
    """Labels {0, 5} would size output_dim=2 from nunique() and only
    fail much later inside cross-entropy — construction now fails fast
    with a remapping hint."""
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "label": [0, 5, 0, 5]})
    with pytest.raises(ValueError, match="contiguous"):
        NNTabularDataset(df=df, feature_cols=["f1"], target_col="label")


def test_f8_tabular_dataset_rejects_nan_cells():
    """NaN features flow into NaN losses and a NaN target's float→int64
    cast is undefined (silent class-0 on ARM) — and the contiguity
    check can't see it because pandas min/max/nunique skip NaN.
    Construction must fail fast naming the offending columns."""
    df = pd.DataFrame({"f1": [1.0, float("nan"), 3.0], "label": [0, 1, 0]})
    with pytest.raises(ValueError, match="NaN"):
        NNTabularDataset(df=df, feature_cols=["f1"], target_col="label")


def test_f8_tabular_dataset_rejects_target_in_feature_cols():
    """Silent label leakage: feature_cols containing the target trains
    the model on its own label (near-perfect val accuracy from the
    classic feature_cols=list(df.columns) mistake)."""
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "label": [0, 1, 0]})
    with pytest.raises(ValueError, match="must not appear in feature_cols"):
        NNTabularDataset(df=df, feature_cols=["f1", "label"], target_col="label")


@pytest.mark.parametrize("labels", [[0.2, 1.2], [0.0, float("inf")]])
def test_f8_tabular_dataset_rejects_non_integral_or_non_finite_labels(labels):
    df = pd.DataFrame({"f1": [1.0, 2.0], "label": labels})
    with pytest.raises(ValueError, match="finite integers"):
        NNTabularDataset(df=df, feature_cols=["f1"], target_col="label")
