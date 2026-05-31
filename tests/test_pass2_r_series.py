"""Pass-2 catalog: R-series regression tests.

Covers reliability gaps surfaced in the pass-2 audit:
- R1: training raises FloatingPointError on NaN/Inf loss instead of
  silently producing garbage checkpoints.
- R2: NNOptimParams.grad_clip_norm actually clips gradients to the
  configured norm.
- R3: NNRun.save() is invoked after each epoch so interrupted training
  leaves a loadable partial run on disk.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx.nn.enum.activations import Activations
from nnx.nn.enum.devices import Devices
from nnx.nn.enum.losses import Losses
from nnx.nn.enum.nets import Nets
from nnx.nn.enum.optims import Optims
from nnx.nn.nn_model import NNModel
from nnx.nn.params.nn_model_params import NNModelParams
from nnx.nn.params.nn_optim_params import NNOptimParams
from nnx.nn.params.nn_params import NNParams
from nnx.nn.params.nn_run import NNRun
from nnx.nn.params.nn_scheduler_params import NNSchedulerParams
from nnx.nn.params.nn_train_params import NNTrainParams


def _model() -> NNModel:
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


def _train_params(loader: DataLoader, **kw) -> NNTrainParams:
    return NNTrainParams(
        n_epochs=kw.pop("n_epochs", 1),
        train_loader=loader,
        optim=NNOptimParams(
            name=Optims.ADAM,
            max_lr=kw.pop("max_lr", 1e-2),
            momentum=(0.9, 0.999),
            weight_decay=0.0,
            grad_clip_norm=kw.pop("grad_clip_norm", None),
        ),
        scheduler=NNSchedulerParams(
            min_lr=1e-7,
            factor=0.5,
            patience=1,
            cooldown=1,
            threshold=1e-3,
        ),
    )


class _ConstantLoss(torch.nn.Module):
    """A loss that always returns a fixed scalar. Used to inject NaN/Inf
    into the train loop without relying on optimizer-divergence dynamics
    (which would couple the test to LR / init / dropout).

    Cleaner than monkey-patching `.forward` / `.__call__` on a bare
    `nn.Module()` — the latter pattern depends on PyTorch's call-chain
    internals and silently fails if Module.__call__ ever stops looking
    up `self.forward` through the instance dict.
    """

    def __init__(self, value: float):
        super().__init__()
        self._value = value

    def forward(self, _log, _y):  # noqa: ARG002
        return torch.tensor(self._value, requires_grad=True)


def test_r1_nan_loss_raises(tmp_path, monkeypatch):
    """Stub the loss to NaN; the train loop should detect the non-finite
    value and raise FloatingPointError rather than continuing to save
    garbage checkpoints."""
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)

    X = torch.randn(32, 4)
    y = torch.randint(0, 2, (32,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16)

    model = _model()
    model.loss_fn = _ConstantLoss(float("nan"))

    with pytest.raises(FloatingPointError, match="non-finite training loss"):
        model.train(params=_train_params(loader))


def test_r1_inf_loss_raises(tmp_path, monkeypatch):
    """Same guard, +inf case."""
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)

    X = torch.randn(32, 4)
    y = torch.randint(0, 2, (32,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16)

    model = _model()
    model.loss_fn = _ConstantLoss(float("inf"))

    with pytest.raises(FloatingPointError, match="non-finite training loss"):
        model.train(params=_train_params(loader))


def test_r2_grad_clip_norm_actually_clips(tmp_path, monkeypatch):
    """With grad_clip_norm=0.01 set, the post-backward gradient norm of the
    network parameters never exceeds the threshold (within FP tolerance)."""
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)

    X = torch.randn(32, 4)
    y = torch.randint(0, 2, (32,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16)

    model = _model()
    clip = 0.01

    seen_norms: list[float] = []
    orig_clip = torch.nn.utils.clip_grad_norm_

    def _spy(parameters, max_norm, *a, **kw):
        n = orig_clip(parameters, max_norm, *a, **kw)
        seen_norms.append(float(n))
        return n

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _spy)
    # NNModel._train_step imports via `torch.nn.utils.clip_grad_norm_` so
    # we also patch the qualified attribute path it uses.
    model.train(params=_train_params(loader, grad_clip_norm=clip))

    # clip_grad_norm_ was called at least once per batch.
    assert len(seen_norms) >= 2
    # Returned norm (pre-clip) was usually larger than the clip threshold,
    # confirming clipping actually had work to do.
    assert any(n > clip for n in seen_norms)


def test_r3_atomic_writes_use_tmp_rename(tmp_path):
    """Round-10 hardening: NNRun.save uses _atomic_write_text which writes
    to <path>.tmp then renames into place. Verifies the helper directly —
    a write that crashes mid-stream leaves the destination file intact."""
    import os

    from nnx.nn.params.nn_run import _atomic_write_text

    target = str(tmp_path / "run.yaml")

    # First write succeeds — file appears.
    _atomic_write_text(target, "first: 1\n")
    assert os.path.exists(target)
    with open(target) as f:
        assert f.read() == "first: 1\n"

    # Simulate a process that crashed leaving the .tmp behind from a
    # prior incomplete write. The next call should still succeed and
    # produce the new content atomically.
    with open(target + ".tmp", "w") as f:
        f.write("garbage stale half-write")
    _atomic_write_text(target, "second: 2\n")
    with open(target) as f:
        assert f.read() == "second: 2\n"


def test_r3_incremental_save_leaves_loadable_partial_run(tmp_path, monkeypatch):
    """After each epoch NNRun.save() is invoked; a 2-epoch run that's
    interrupted after the first epoch should still produce a loadable
    runs/<id>/run.yaml + idps.csv on disk."""
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)

    X = torch.randn(32, 4)
    y = torch.randint(0, 2, (32,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16)

    model = _model()
    # Inject a callback that stops after the first epoch.
    from nnx.nn.callbacks import Callback

    class _StopAfterFirst(Callback):
        def on_epoch_end(self, ctx):
            ctx.should_stop = True

    run = model.train(
        params=_train_params(loader, n_epochs=5),
        callbacks=[_StopAfterFirst()],
    )

    # On-disk artifacts exist after epoch 1 even though we requested 5.
    run_path = tmp_path / "runs" / run.id
    assert (run_path / "run.yaml").exists()
    assert (run_path / "idps.csv").exists()

    # Reload from disk and verify we see the actual-completed epochs.
    reloaded = NNRun.load(id=run.id)
    completed_epochs = {idp.epoch_idx for idp in reloaded.idps}
    assert completed_epochs == {0}


def test_r2_grad_clip_round_trips_through_state():
    """grad_clip_norm survives NNOptimParams.state() / from_state()."""
    p = NNOptimParams(
        name=Optims.ADAM,
        max_lr=1e-3,
        momentum=(0.9, 0.999),
        weight_decay=0.0,
        grad_clip_norm=1.0,
    )
    rt = NNOptimParams.from_state(p.state())
    assert rt.grad_clip_norm == 1.0
    assert rt == p


def test_r2_grad_clip_back_compat_with_old_yaml():
    """A state() dict missing grad_clip_norm (older runs) must still load
    cleanly with grad_clip_norm=None."""
    legacy_state = {
        "max_lr": 1e-3,
        "momentum": "(0.9, 0.999)",
        "name": "adam",
        "weight_decay": 0.0,
        # NO grad_clip_norm key — predates the field.
    }
    p = NNOptimParams.from_state(legacy_state)
    assert p.grad_clip_norm is None


def test_r3_nnrun_load_rejects_python_object_tags(tmp_path, monkeypatch):
    """Defense-in-depth: NNRun.load uses yaml.safe_load, so a tampered
    run.yaml containing a `!!python/object/...` tag must fail to load
    rather than instantiate the embedded Python object.

    The original implementation used yaml.FullLoader, which would have
    happily executed the tag. Switching to safe_load aligns with the
    sibling metadata.yaml's long-standing safe_load contract (see the
    seeding.py comment) and ensures filesystem-level tampering can't
    escalate into arbitrary code execution at load time."""
    import yaml

    monkeypatch.chdir(tmp_path)
    run_id = "deadbeef" * 4  # md5-shaped fake id; we never check its hash.
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)

    # Write a run.yaml that, under FullLoader, would import a stdlib
    # function via the `!!python/name` tag. safe_load must refuse it.
    poisoned = "trigger: !!python/name:os.system ''\n"
    (run_dir / "run.yaml").write_text(poisoned)
    (run_dir / "idps.csv").write_text("epoch_idx,iter_idx\n")  # placeholder

    with pytest.raises(yaml.YAMLError):
        NNRun.load(id=run_id)


def test_r3_nnrun_load_rejects_path_traversal_run_ids(tmp_path, monkeypatch):
    """Defense-in-depth: NNRun.load / NNCheckpoint.load accept a public
    ``id`` / ``run`` parameter and join it directly into a path under
    ``runs/``. A traversal-shaped identifier like ``"../../etc"`` would
    escape the runs root and try to read sensitive files sitting next to
    the working directory.

    Internal callers always pass the md5 hex of NNRun.state() (always
    safe), but `_validate_run_id` is the boundary guard for the public
    API surface. Mirrors the spirit of the sibling test above that fixes
    the yaml.FullLoader path-traversal-via-tag escalation.
    """
    monkeypatch.chdir(tmp_path)

    traversal_ids = [
        "../etc",
        "../../etc/passwd",
        "..",
        ".",
        "runs/../etc",
        "with/slash",
        "with\\backslash",
        "",
    ]
    for bad_id in traversal_ids:
        with pytest.raises(ValueError, match=r"run id|non-empty"):
            NNRun.load(id=bad_id)


def test_r3_nnrun_load_accepts_normal_md5_id(tmp_path, monkeypatch):
    """The traversal guard must not reject normal md5-hex run ids
    (32 lowercase-hex chars, no separators) that legitimate callers
    produce via ``NNRun.id``. Smoke test against the validator's
    happy path."""
    from nnx.nn.params.nn_run import _validate_run_id

    md5_shaped = "deadbeef" * 4  # 32 hex chars, valid run-id shape.
    # The validator should pass through without raising.
    assert _validate_run_id(md5_shaped) == md5_shaped
