"""The training completion message prints a portable display path (#190).

`NNModel.train()` and `Trainer.train()` end with ``Run saved to <path>``.
Captured notebook output must not embed the executing machine's working
directory (temporary dirs, worktrees, container mounts), so the default
root is shown as ``runs/<id>`` relative to the working directory. The
message is a display path, not artifact provenance: persistence paths and
saved artifacts are unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

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
    NNTrainerParams,
    NNTrainParams,
    Optims,
    Trainer,
    TrainerStepContext,
)

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
_OPTIM = NNOptimParams(name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0)


def _model() -> NNModel:
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def _loader() -> DataLoader:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(16, 4, generator=generator)
    y = torch.randint(0, 2, (16,), generator=generator)
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)


def _trainer_step(ctx: TrainerStepContext) -> NNEvaluationDataPoint:
    model, optimizer = ctx.model, ctx.optimizers["main"]
    model.net.train()
    optimizer.zero_grad()
    (x,), y = model.net.unpack_batch(ctx.batch)
    loss = model.loss_fn(model.net(x), y)
    loss.backward()
    optimizer.step()
    return NNEvaluationDataPoint(f1=0.0, recall=0.0, accuracy=0.0, precision=0.0, loss=float(loss.detach()), error=0.0)


def _train_nn_model() -> NNRun:
    return _model().train(NNTrainParams(n_epochs=1, train_loader=_loader(), optim=_OPTIM))


def _train_trainer() -> NNRun:
    params = NNTrainerParams(n_epochs=1, train_loader=_loader(), optims={"main": _OPTIM})
    return Trainer(_model()).train(params=params, trainer_step_fn=_trainer_step)


_ENTRY_POINTS = {"NNModel.train": _train_nn_model, "Trainer.train": _train_trainer}


def _assert_portable_and_persisted(train, cwd: Path, capsys) -> str:
    torch.manual_seed(0)
    run = train()
    out = capsys.readouterr().out

    assert f"Run saved to runs/{run.id} (relative to the working directory)\n" in out
    for prefix in {str(cwd), os.path.realpath(cwd), os.getcwd(), _REPO_ROOT}:
        assert prefix not in out, f"absolute prefix {prefix!r} leaked into the completion output"
    # Persistence is unchanged: the run lives under <cwd>/runs/<id> and loads back.
    assert (cwd / "runs" / run.id / "run.yaml").is_file()
    loaded = NNRun.load(run.id)
    assert loaded.id == run.id and loaded.state() == run.state()
    return out.replace(run.id, "<run-id>")


def test_nn_model_train_prints_portable_run_path(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _assert_portable_and_persisted(_train_nn_model, tmp_path, capsys)


def test_trainer_train_prints_portable_run_path(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _assert_portable_and_persisted(_train_trainer, tmp_path, capsys)


def test_default_root_output_is_identical_across_working_directories(tmp_path, monkeypatch, capsys):
    """Training from two different absolute directories prints the same
    output (apart from the run id) through both entry points."""
    for name, train in _ENTRY_POINTS.items():
        outputs = []
        for cwd in (tmp_path / name / "first-worktree", tmp_path / name / "second" / "nested"):
            cwd.mkdir(parents=True)
            monkeypatch.chdir(cwd)
            outputs.append(_assert_portable_and_persisted(train, cwd, capsys))
        assert outputs[0] == outputs[1], name


def test_display_path_is_the_cwd_relative_persistence_location(tmp_path, monkeypatch):
    """The display path is derived from the persistence root, so it names
    exactly the directory ``NNRun.save()`` writes — relative, with ``/``
    separators. (No training entry point accepts a custom root, so the
    ticket's custom-root criterion does not apply yet.)"""
    from nnx.nn.params.nn_run import _run_display_path, _runs_root

    monkeypatch.chdir(tmp_path)
    run_id = "0123456789abcdef0123456789abcdef"
    display = _run_display_path(run_id)
    assert display == f"runs/{run_id}"
    assert os.path.abspath(display) == os.path.join(_runs_root(), run_id)
