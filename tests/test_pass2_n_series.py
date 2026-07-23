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
from nnx.nn.nn_model import NNModel, _classification_metric_tensors, _loss_normalization_weight
from nnx.nn.params.nn_model_params import NNModelParams
from nnx.nn.params.nn_optim_params import NNOptimParams
from nnx.nn.params.nn_params import NNParams


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


def test_custom_elementwise_loss_subclass_keeps_batch_normalization_contract():
    class CustomMSE(torch.nn.MSELoss):
        pass

    logits = torch.zeros(2, 3)
    target = torch.zeros(2, 3)

    assert _loss_normalization_weight(CustomMSE(), logits, target) == 2.0


def test_cross_entropy_probability_targets_use_class_indices_for_metrics():
    loss_fn = torch.nn.CrossEntropyLoss()
    target = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    prediction = torch.tensor([1, 0])

    metric_target, metric_prediction = _classification_metric_tensors(loss_fn, target, prediction)

    assert torch.equal(metric_target, torch.tensor([1, 0]))
    assert torch.equal(metric_prediction, prediction)


def test_multidimensional_probability_targets_are_flattened_for_metrics():
    loss_fn = torch.nn.CrossEntropyLoss()
    target = torch.softmax(torch.randn(2, 3, 4, 5), dim=1)
    prediction = target.argmax(dim=1)

    metric_target, metric_prediction = _classification_metric_tensors(loss_fn, target, prediction)

    assert metric_target.shape == (40,)
    assert metric_prediction.shape == (40,)


def test_cross_entropy_subclass_keeps_metric_preprocessing():
    class CustomCrossEntropy(torch.nn.CrossEntropyLoss):
        pass

    target = torch.tensor([0, -100, 1])
    prediction = torch.tensor([0, 1, 1])
    metric_target, metric_prediction = _classification_metric_tensors(CustomCrossEntropy(), target, prediction)

    assert torch.equal(metric_target, torch.tensor([0, 1]))
    assert torch.equal(metric_prediction, torch.tensor([0, 1]))


def test_weighted_cross_entropy_subclass_keeps_native_denominator():
    class CustomCrossEntropy(torch.nn.CrossEntropyLoss):
        pass

    loss = CustomCrossEntropy(weight=torch.tensor([1.0, 3.0]))
    target = torch.tensor([0, 1, 1])
    assert _loss_normalization_weight(loss, torch.randn(3, 2), target) == 7.0


def test_binary_logits_use_zero_threshold_for_predictions():
    class BinaryNet(torch.nn.Linear):
        def unpack_batch(self, batch):
            return (batch[0],), batch[1]

    model = _model()
    model.params = NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.BINARY_CROSS_ENTROPY)
    model.loss_fn = model.params.loss()
    model.net = BinaryNet(4, 1)
    with torch.no_grad():
        model.net.weight.zero_()
        model.net.bias.fill_(1.0)

    _x, _y, logits, prediction = model._fwd_pass((torch.zeros(2, 4), torch.ones(2, 1)))

    assert logits.shape == (2, 1)
    assert torch.equal(prediction, torch.ones(2, 1, dtype=torch.long))


def test_soft_binary_targets_are_thresholded_for_metrics():
    target = torch.tensor([[0.8], [0.2]])
    prediction = torch.tensor([[1], [0]])
    metric_target, metric_prediction = _classification_metric_tensors(torch.nn.BCEWithLogitsLoss(), target, prediction)
    assert torch.equal(metric_target, torch.tensor([[1], [0]]))
    assert torch.equal(metric_prediction, prediction)


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


@pytest.mark.parametrize("loss_type", [torch.nn.CrossEntropyLoss, torch.nn.NLLLoss])
def test_n7_evaluate_loss_matches_weighted_ignored_combined_batch(loss_type):
    torch.manual_seed(0)
    model = _model()
    model.loss_fn = loss_type(weight=torch.tensor([1.0, 7.0]), ignore_index=-100)

    X = torch.randn(5, 4)
    y = torch.tensor([-100, -100, 0, 0, 1])
    split = model.evaluate(loader=DataLoader(TensorDataset(X, y), batch_size=2, shuffle=False))
    combined = model.evaluate(loader=DataLoader(TensorDataset(X, y), batch_size=5, shuffle=False))

    assert split.loss == pytest.approx(combined.loss)


@pytest.mark.parametrize("loss_type", [torch.nn.CrossEntropyLoss, torch.nn.NLLLoss])
def test_n7_evaluate_loss_preserves_classification_sum_reduction(loss_type):
    torch.manual_seed(0)
    model = _model()
    model.loss_fn = loss_type(weight=torch.tensor([1.0, 7.0]), ignore_index=-100, reduction="sum")

    X = torch.randn(5, 4)
    y = torch.tensor([-100, -100, 0, 0, 1])
    split = model.evaluate(loader=DataLoader(TensorDataset(X, y), batch_size=2, shuffle=False))
    combined = model.evaluate(loader=DataLoader(TensorDataset(X, y), batch_size=5, shuffle=False))

    assert split.loss == pytest.approx(combined.loss)


@pytest.mark.parametrize(
    "loss_fn",
    [
        pytest.param(torch.nn.MSELoss(reduction="sum"), id="mse"),
        pytest.param(torch.nn.BCEWithLogitsLoss(reduction="sum"), id="bce"),
        pytest.param(torch.nn.MSELoss(), id="mse-mean"),
        pytest.param(torch.nn.BCEWithLogitsLoss(), id="bce-mean"),
    ],
)
def test_n7_evaluate_elementwise_loss_matches_combined_call(loss_fn):
    class BinaryModel:
        evaluate = NNModel.evaluate

        def __init__(self):
            self.net = torch.nn.Linear(1, 1, bias=False)
            self.net.weight.data.fill_(0.25)
            self.loss_fn = loss_fn
            self.device = torch.device("cpu")

        def _fwd_pass(self, batch):
            X, Y = batch
            logits = self.net(X).squeeze(-1)
            return X, Y, logits, (logits >= 0).to(Y.dtype)

    X = torch.tensor([[1.0], [2.0], [3.0]])
    Y = torch.tensor([0.0, 1.0, 1.0])
    model = BinaryModel()

    split = model.evaluate(DataLoader(TensorDataset(X, Y), batch_size=2))
    combined = loss_fn(model.net(X).squeeze(-1), Y)

    assert split.loss == pytest.approx(float(combined.detach()))


def test_n7_evaluate_calls_cross_entropy_subclass_and_hooks():
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
    X = torch.randn(5, 4)
    Y = torch.tensor([0, 1, 0, 1, 0])

    split = model.evaluate(DataLoader(TensorDataset(X, Y), batch_size=2))
    with torch.no_grad():
        combined = loss_fn(model.net(X), Y)

    assert split.loss == pytest.approx(float(combined))
    assert loss_fn.calls == 4
    assert len(hook_calls) == 4


def test_n7_evaluate_uses_batch_weighting_for_cross_entropy_subclasses():
    class RemappingCrossEntropy(torch.nn.CrossEntropyLoss):
        def forward(self, logits, target):
            remapped = torch.where(target == 0, torch.ones_like(target), target)
            return super().forward(logits, remapped)

    model = _model()
    model.loss_fn = RemappingCrossEntropy(weight=torch.tensor([1.0, 7.0]))
    X = torch.randn(3, 4)
    Y = torch.tensor([0, 0, 1])

    split = model.evaluate(DataLoader(TensorDataset(X, Y), batch_size=2))
    with torch.no_grad():
        combined = model.loss_fn(model.net(X), Y)

    assert split.loss == pytest.approx(float(combined))


def test_n7_evaluate_filters_exact_cross_entropy_ignore_index_from_metrics():
    class EvaluationModel:
        evaluate = NNModel.evaluate

        def __init__(self):
            self.net = torch.nn.Identity()
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
            self.device = torch.device("cpu")

        def _fwd_pass(self, batch):
            X, Y = batch
            return X, Y, X, X.argmax(dim=1)

    model = EvaluationModel()
    X = torch.tensor([[0.0, 4.0], [4.0, 0.0], [0.0, 4.0]])
    Y = torch.tensor([-100, 0, 1])

    edp = model.evaluate(
        DataLoader(TensorDataset(X, Y), batch_size=2),
        extra_metrics={"count": lambda y, _y_hat: len(y)},
    )

    assert edp.accuracy == 1.0
    assert edp.error == 0.0
    assert edp.extra == {"count": 2.0}


def test_n7_evaluate_rejects_loader_with_only_ignored_targets():
    model = _model()
    model.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    X = torch.randn(3, 4)
    Y = torch.full((3,), -100)

    with pytest.raises(ValueError, match="zero non-ignored samples"):
        model.evaluate(DataLoader(TensorDataset(X, Y), batch_size=2))


def test_n8_evaluate_raises_on_empty_loader():
    """Empty loaders previously yielded NaN metrics from np.mean over [].
    Should raise instead."""
    model = _model()
    X = torch.empty(0, 4)
    y = torch.empty(0, dtype=torch.long)
    loader = DataLoader(TensorDataset(X, y), batch_size=8)
    with pytest.raises(ValueError, match="zero samples"):
        model.evaluate(loader=loader)


def test_n8_evaluate_handles_batch_of_one():
    """Symmetric edge case to the N7 uneven-last-batch fix: a loader
    whose only batch contains exactly one sample. Several of the
    aggregation paths (np.concatenate over a single 1-row array,
    sample-weighted loss division) would silently produce different
    results with batch_size=1 if a regression added a dimension-squeeze
    or a guard-against-empty branch."""
    torch.manual_seed(0)
    model = _model()

    X = torch.randn(1, 4)
    y = torch.randint(0, 2, (1,))
    loader = DataLoader(TensorDataset(X, y), batch_size=1, shuffle=False)

    edp = model.evaluate(loader=loader)
    # The single sample is either correctly classified (accuracy=1.0,
    # error=0.0) or not (0.0, 1.0). Either way both metrics must be
    # finite and 0/1, not NaN, and accuracy + error must sum to 1.
    assert edp.accuracy in (0.0, 1.0)
    assert edp.error in (0.0, 1.0)
    assert abs((edp.accuracy + edp.error) - 1.0) < 1e-9
    assert edp.loss is not None
    assert torch.isfinite(torch.tensor(edp.loss)).item()


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
    run = model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=loader,
            optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
            scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
        )
    )
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


def test_point_best_symlink_resolves_under_relative_root(tmp_path, monkeypatch):
    """_point_best used the raw run_path as the symlink target, but a
    symlink target resolves relative to the symlink's OWN directory —
    with a relative root= (run.save(root="experiments")) the link
    dangled from birth, so every save took the repoint-unconditionally
    dangling branch and `runs/best` tracked the most RECENT run instead
    of the best. The target is now the sibling run-dir basename, which
    also survives relocating the runs root."""
    import os

    from nnx.nn.params.nn_run import _point_best, _read_best_pointer

    monkeypatch.chdir(tmp_path)
    run_id = "a" * 32
    runs_root = tmp_path / "experiments" / "runs"
    (runs_root / run_id).mkdir(parents=True)
    best_path = str(runs_root / "best")

    # The relative shape NNRun.save builds from root="experiments".
    _point_best(best_path, os.path.join("experiments", "runs", run_id))
    assert os.path.exists(best_path), "best symlink dangles under a relative root"
    assert _read_best_pointer(best_path) == run_id

    # Sibling-basename target keeps resolving after the root moves.
    (tmp_path / "experiments").rename(tmp_path / "elsewhere")
    assert os.path.exists(str(tmp_path / "elsewhere" / "runs" / "best"))


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
        return m.train(
            params=NNTrainParams(
                n_epochs=1,
                train_loader=DataLoader(
                    TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,))),
                    batch_size=8,
                ),
                optim=NNOptimParams(name=Optims.ADAM, max_lr=lr, momentum=(0.9, 0.999), weight_decay=0.0),
                scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
            )
        )

    run_a = _drive(1e-2)
    run_b = _drive(1e-3)  # different config → different run.id

    pointer = (tmp_path / "runs" / "best" / "POINTER.txt").read_text().strip()
    # Whichever of the two has the lower train_edp.error should be the
    # pointer target; if they tie, run_a (the incumbent) keeps the slot.
    err_a = run_a.idps[-1].train_edp.error
    err_b = run_b.idps[-1].train_edp.error
    expected_id = run_b.id if err_b < err_a else run_a.id
    assert expected_id in pointer, (
        f"pointer at {pointer!r} should reference {expected_id} (err_a={err_a:.4f}, err_b={err_b:.4f})"
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
        f"grad_clip_norm=None must be omitted from state() to preserve run.id back-compat; got state={state!r}"
    )
    assert "accumulate_grad_batches" not in state
    assert set(state.keys()) == {"max_lr", "momentum", "name", "weight_decay"}


def test_review_optim_params_state_emits_grad_clip_norm_when_set():
    p = NNOptimParams(
        name=Optims.ADAM,
        max_lr=1e-3,
        momentum=(0.9, 0.999),
        weight_decay=0.0,
        grad_clip_norm=1.0,
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

        reimported = importlib.import_module("nnx.nn.callbacks")
        # Explicit checks: the module reloaded successfully (Callback +
        # standard callbacks reachable) AND no IPython submodule was
        # pulled in as a side effect (the actual invariant this test
        # protects). After this block we set every `IPython*` entry in
        # sys.modules to None as a sentinel that blocks `import IPython`;
        # if the callbacks import had succeeded in loading IPython, that
        # sentinel would have been overwritten by the real module object.
        assert hasattr(reimported, "Callback")
        assert hasattr(reimported, "EarlyStopping")
        for k in sys.modules:
            if k.startswith("IPython"):
                assert sys.modules[k] is None, (
                    f"importing nnx.nn.callbacks loaded {k!r} (lazy "
                    "import in _LegacyCallback leaked out of on_epoch_end)"
                )
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
    run = model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=loader,
            optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
            scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
        )
    )

    pointer = tmp_path / "runs" / "best" / "POINTER.txt"
    assert pointer.exists()
    assert run.id in pointer.read_text()
