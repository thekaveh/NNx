"""Pass-2 catalog: N-series regression tests.

Covers correctness gaps surfaced in the pass-2 audit:
- N1: NNOptimParams.is_valid() must return a bool (not None) for any input.
- N7: NNModel.evaluate() aggregates predictions across all batches so an
  uneven final batch doesn't over-weight metrics.
- N8: evaluate() raises rather than silently returning NaN on empty loaders.
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


def _model() -> NNModel:
    return NNModel(
        net_params=NNParams(
            input_dim=4, output_dim=2, hidden_dims=[8],
            dropout_prob=0.0, activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
        ),
    )


def test_n1_optim_is_valid_always_returns_bool():
    """is_valid() previously returned None for unknown enum variants,
    which let invalid configs slip through `not params.optim.is_valid()`."""
    p_sgd = NNOptimParams(name=Optims.SGD, max_lr=1e-2, momentum=0.9, weight_decay=0.0)
    assert p_sgd.is_valid() is True

    # SGD with a tuple momentum is invalid (Adam-shaped).
    p_bad = NNOptimParams(name=Optims.SGD, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0)
    assert p_bad.is_valid() is False
    assert isinstance(p_bad.is_valid(), bool)


def test_n7_evaluate_aggregates_across_batches():
    """Last-batch over-weighting bug: with 10 samples split into batches of
    [8, 2], per-batch averaging weighted the 2-sample batch 50% in the mean.
    Aggregating before computing should weight by sample count."""
    torch.manual_seed(0)
    model = _model()

    X = torch.randn(10, 4)
    y = torch.randint(0, 2, (10,))
    # batch_size=8 → batches of [8, 2]
    loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False)

    edp = model.evaluate(loader=loader)
    # accuracy is sample-weighted; should match a single-batch computation.
    big_loader = DataLoader(TensorDataset(X, y), batch_size=10, shuffle=False)
    edp_full = model.evaluate(loader=big_loader)
    assert abs(edp.accuracy - edp_full.accuracy) < 1e-9
    assert abs(edp.error - edp_full.error) < 1e-9


def test_n8_evaluate_raises_on_empty_loader():
    """Empty loaders previously yielded NaN metrics from np.mean over [].
    Should raise instead."""
    model = _model()
    X = torch.empty(0, 4)
    y = torch.empty(0, dtype=torch.long)
    loader = DataLoader(TensorDataset(X, y), batch_size=8)
    with pytest.raises(ValueError, match="zero samples"):
        model.evaluate(loader=loader)


def test_n4_train_works_on_iterable_dataset(tmp_path, monkeypatch):
    """train() must tolerate DataLoaders where len() raises (IterableDataset)."""
    monkeypatch.chdir(tmp_path)

    from torch.utils.data import IterableDataset

    class _IterableSet(IterableDataset):
        def __init__(self, n: int):
            super().__init__()
            self.n = n

        def __iter__(self):
            for _ in range(self.n):
                yield torch.randn(4), torch.randint(0, 2, (1,)).squeeze()

    loader = DataLoader(_IterableSet(n=8), batch_size=4)
    # Sanity: len() on this loader raises.
    with pytest.raises(TypeError):
        len(loader)

    from nnx.nn.params.nn_optim_params import NNOptimParams
    from nnx.nn.params.nn_scheduler_params import NNSchedulerParams
    from nnx.nn.params.nn_train_params import NNTrainParams

    model = _model()
    run = model.train(params=NNTrainParams(
        n_epochs=1,
        train_loader=loader,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
    ))
    # Successfully completed at least the iterable's worth of batches.
    assert len(run.idps) >= 1


def test_review_read_best_pointer_resolves_symlink_or_pointer_file(tmp_path):
    """The helper introduced during the meta-review must extract a run id
    from either layout — a real symlink OR a POINTER.txt file inside the
    `runs/best/` directory."""
    import os

    from nnx.nn.params.nn_run import _point_best, _read_best_pointer

    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    real_run_dir = runs_root / "abc123"
    real_run_dir.mkdir()
    best_path = str(runs_root / "best")

    # Symlink layout (always supported on POSIX).
    _point_best(best_path, str(real_run_dir))
    assert _read_best_pointer(best_path) == "abc123"

    # Re-point to a different run and verify the pointer updates.
    other_run = runs_root / "def456"
    other_run.mkdir()
    _point_best(best_path, str(other_run))
    assert _read_best_pointer(best_path) == "def456"

    # POINTER.txt layout: simulate by removing the symlink and writing the
    # fallback directory by hand.
    os.remove(best_path)
    os.makedirs(best_path)
    with open(os.path.join(best_path, "POINTER.txt"), "w") as f:
        f.write(str(real_run_dir))
    assert _read_best_pointer(best_path) == "abc123"


def test_review_pointer_file_compared_correctly_under_symlink_fallback(tmp_path, monkeypatch):
    """Pre-fix: under the POINTER.txt fallback, `NNCheckpoint.load(run="best", ...)`
    looked for `runs/best/checkpoints/best.pt` which didn't exist (best/ was a
    directory with POINTER.txt inside), so _best_err returned +inf and the new
    run ALWAYS overwrote the pointer. Post-fix: NNRun.save resolves the pointer
    via _read_best_pointer and compares against the right run."""
    monkeypatch.chdir(tmp_path)
    from nnx.nn.params import nn_run as nn_run_mod

    def _raise(*a, **kw):
        raise OSError("symlink not supported (simulated Windows)")
    monkeypatch.setattr(nn_run_mod.os, "symlink", _raise)

    # Drive two distinct runs (different LRs → different run.id) through
    # NNRun.save under the symlink fallback. The key claim is that the
    # second save() does NOT clobber the pointer just because it can't
    # read the prior best — it goes through _read_best_pointer correctly.
    from torch.utils.data import DataLoader, TensorDataset

    from nnx.nn.params.nn_optim_params import NNOptimParams
    from nnx.nn.params.nn_scheduler_params import NNSchedulerParams
    from nnx.nn.params.nn_train_params import NNTrainParams

    def _drive(lr):
        torch.manual_seed(0)
        m = _model()
        return m.train(params=NNTrainParams(
            n_epochs=1,
            train_loader=DataLoader(
                TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,))),
                batch_size=8,
            ),
            optim=NNOptimParams(name=Optims.ADAM, max_lr=lr, momentum=(0.9, 0.999), weight_decay=0.0),
            scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
        ))

    run_a = _drive(1e-2)
    run_b = _drive(1e-3)  # different config → different run.id

    pointer = (tmp_path / "runs" / "best" / "POINTER.txt").read_text().strip()
    # Whichever of the two has the lower train_edp.error should be the
    # pointer target; if they tie, run_a (the incumbent) keeps the slot.
    err_a = run_a.idps[-1].train_edp.error
    err_b = run_b.idps[-1].train_edp.error
    expected_id = run_b.id if err_b < err_a else run_a.id
    assert expected_id in pointer, (
        f"pointer at {pointer!r} should reference {expected_id} "
        f"(err_a={err_a:.4f}, err_b={err_b:.4f})"
    )


def test_review_optim_params_state_omits_default_grad_clip_norm():
    """CRITICAL back-compat regression: NNOptimParams.state() must NOT
    emit grad_clip_norm when it's the default (None) — otherwise every
    existing run.id hash changes when this code loads them.

    The shipped state() shape pre-grad-clip-norm was exactly:
        {max_lr, momentum, name, weight_decay}
    A NNOptimParams with grad_clip_norm=None (default) must produce that
    same dict.
    """
    p = NNOptimParams(name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0)
    state = p.state()
    assert "grad_clip_norm" not in state, (
        "grad_clip_norm=None must be omitted from state() to preserve run.id back-compat; "
        f"got state={state!r}"
    )
    assert "accumulate_grad_batches" not in state
    assert set(state.keys()) == {"max_lr", "momentum", "name", "weight_decay"}


def test_review_optim_params_state_emits_grad_clip_norm_when_set():
    p = NNOptimParams(
        name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999),
        weight_decay=0.0, grad_clip_norm=1.0,
    )
    state = p.state()
    assert state["grad_clip_norm"] == 1.0


def test_review_nnrun_all_handles_missing_runs_dir(tmp_path, monkeypatch):
    """NNRun.all() should return [] when runs/ doesn't exist yet, not
    raise FileNotFoundError."""
    monkeypatch.chdir(tmp_path)
    from nnx.nn.params.nn_run import NNRun
    assert NNRun.all() == []


def test_review_nnrun_all_skips_non_run_entries(tmp_path, monkeypatch):
    """Stray files in runs/ (e.g., .DS_Store) and incomplete directories
    (missing run.yaml) must not crash NNRun.all()."""
    monkeypatch.chdir(tmp_path)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    (runs_root / ".DS_Store").write_text("macOS junk")
    (runs_root / "incomplete_run").mkdir()  # no run.yaml inside

    from nnx.nn.params.nn_run import NNRun
    assert NNRun.all() == []


def test_review_callbacks_module_imports_without_ipython(monkeypatch):
    """Importing nnx.nn.callbacks must NOT trigger an IPython import.
    Otherwise every `import nnx` consumer pulls IPython transitively."""
    import sys

    # Save the ORIGINAL nnx.nn.callbacks module reference so we can
    # restore the exact same Callback / EarlyStopping / etc. class
    # objects after the test. Without this, downstream tests that do
    # `isinstance(cb, Callback)` against the original class will fail —
    # subclassing the re-imported Callback produces a DIFFERENT class
    # tree that's NOT a subclass of the originally-imported one.
    original_callbacks_module = sys.modules.get("nnx.nn.callbacks")

    # Save then sabotage any IPython modules already cached so we'd see
    # the import fail if callbacks pulled it in.
    saved = {k: v for k, v in sys.modules.items() if k.startswith("IPython")}
    for k in list(sys.modules):
        if k.startswith("IPython"):
            sys.modules[k] = None  # make subsequent `import IPython` raise

    try:
        # Force a re-import of callbacks.
        for k in list(sys.modules):
            if k.startswith("nnx.nn.callbacks"):
                del sys.modules[k]
        import importlib
        importlib.import_module("nnx.nn.callbacks")
    finally:
        # Restore IPython modules so other tests aren't affected.
        for k in list(sys.modules):
            if k.startswith("IPython"):
                del sys.modules[k]
        for k, v in saved.items():
            sys.modules[k] = v
        # Restore the original nnx.nn.callbacks module so other modules'
        # cached references to its classes (Callback, EarlyStopping, ...)
        # remain authoritative.
        if original_callbacks_module is not None:
            sys.modules["nnx.nn.callbacks"] = original_callbacks_module


def test_n6_best_symlink_falls_back_to_pointer_file_when_symlink_fails(tmp_path, monkeypatch):
    """On platforms where os.symlink raises (e.g., Windows without dev mode),
    NNRun.save still records the best run via a POINTER.txt file."""
    monkeypatch.chdir(tmp_path)

    from nnx.nn.params import nn_run as nn_run_mod

    def _raise(*a, **kw):
        raise OSError("symlink not supported (simulated Windows)")
    monkeypatch.setattr(nn_run_mod.os, "symlink", _raise)

    # Drive a tiny run end-to-end so NNRun.save() is exercised through
    # the symlink path.
    from torch.utils.data import DataLoader, TensorDataset

    from nnx.nn.params.nn_optim_params import NNOptimParams
    from nnx.nn.params.nn_scheduler_params import NNSchedulerParams
    from nnx.nn.params.nn_train_params import NNTrainParams

    X = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8)

    model = _model()
    run = model.train(params=NNTrainParams(
        n_epochs=1,
        train_loader=loader,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
    ))

    pointer = tmp_path / "runs" / "best" / "POINTER.txt"
    assert pointer.exists()
    assert run.id in pointer.read_text()
