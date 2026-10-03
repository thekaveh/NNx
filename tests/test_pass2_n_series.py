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


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_n7_nll_matches_log_softmax_reference(reduction):
    """FIX-001: native NLL with class weights and ignore_index must equal
    `F.nll_loss(F.log_softmax(raw, 1), y, ...)` on the full set — a direct
    oracle, not just split-vs-combined self-consistency (which an
    unnormalized objective also satisfies)."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    model = _model()
    weight = torch.tensor([1.0, 7.0])
    model.loss_fn = torch.nn.NLLLoss(weight=weight, ignore_index=-100, reduction=reduction)

    X = torch.randn(5, 4)
    y = torch.tensor([-100, -100, 0, 0, 1])
    got = model.evaluate(loader=DataLoader(TensorDataset(X, y), batch_size=2, shuffle=False)).loss
    with torch.no_grad():
        raw = model.net(X)
    expected = float(F.nll_loss(F.log_softmax(raw, dim=1), y, weight=weight, ignore_index=-100, reduction=reduction))
    assert got == pytest.approx(expected, rel=1e-6)

    ce_model = _model()
    ce_model.net.load_state_dict(model.net.state_dict())
    ce_model.loss_fn = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=-100, reduction=reduction)
    assert ce_model.evaluate(loader=DataLoader(TensorDataset(X, y), batch_size=5)).loss == pytest.approx(got, rel=1e-6)


def test_n7_nll_transformer_token_logits_use_class_axis():
    """Transformer token logits (B, T, V) are flattened to (tokens, V) by
    `_fwd_pass` before the loss, so log-softmax must run over the class
    axis of that flattened view; predictions are unchanged."""
    import torch.nn.functional as F

    from nnx.nn.params.nn_transformer_params import NNTransformerParams

    torch.manual_seed(0)
    params = NNTransformerParams(
        input_dim=16,
        output_dim=16,
        dropout_prob=0.0,
        vocab_size=16,
        n_layers=1,
        n_heads=2,
        d_model=16,
        ffn_mult=2,
        max_seq_len=8,
    )
    nll = NNModel(
        net_params=params,
        params=NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.NEGATIVE_LOG_LIKELIHOOD),
    )
    ce = NNModel(
        net_params=params,
        params=NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    ce.net.load_state_dict(nll.net.state_dict())
    tokens = torch.randint(0, 16, (2, 4))
    targets = torch.randint(0, 16, (2, 4))
    loader = DataLoader(TensorDataset(tokens, targets), batch_size=2)

    with torch.no_grad():
        _, flat_targets, logits, predictions = nll._fwd_pass((tokens, targets))
    assert logits.shape == (8, 16)
    expected = float(F.nll_loss(F.log_softmax(logits, dim=1), flat_targets))
    nll_edp = nll.evaluate(loader)
    assert nll_edp.loss == pytest.approx(expected, rel=1e-6)
    assert nll_edp.loss == pytest.approx(ce.evaluate(loader).loss, rel=1e-6)
    assert nll_edp.accuracy == ce.evaluate(loader).accuracy
    assert torch.equal(
        nll.predict(tokens).logits
        if isinstance(nll.predict(tokens).logits, torch.Tensor)
        else torch.as_tensor(nll.predict(tokens).logits),
        torch.as_tensor(ce.predict(tokens).logits),
    )


# --- FEAT-020: an opt-in bounded validation step ---------------------------------------------------------


def _masked_task_run(tmp_path, monkeypatch, task, targets, *, eval_step_fn=None, metrics=(), monitor=None):
    """Train the same seeded model on uneven [2, 2, 1] batches whose targets
    are partly masked; return the run and its BEST checkpoint."""
    from nnx import Checkpoints, MonitorSpec, NNCheckpoint, NNTrainParams, set_seed
    from nnx.nn.params.nn_optim_params import NNOptimParams as OptimParams

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    set_seed(0)
    loss = Losses.MEAN_SQUARED_ERROR if task.kind == "regression" else Losses.CROSS_ENTROPY
    output_dim = task.num_outputs
    model = NNModel(
        net_params=NNParams(
            input_dim=4, output_dim=output_dim, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=loss, task=task),
    )
    X = torch.randn(5, 4, generator=torch.Generator().manual_seed(1))
    batches = [(X[0:2], targets[0:2]), (X[2:4], targets[2:4]), (X[4:5], targets[4:5])]
    run = model.train(
        params=NNTrainParams(
            n_epochs=4,
            train_loader=batches,
            val_loader=batches,
            optim=OptimParams.builder().sgd(max_lr=0.2).build(),
            metrics=list(metrics),
            monitor=monitor if monitor is not None else MonitorSpec(metric="loss"),
            seed=0,
        ),
        eval_step_fn=eval_step_fn,
    )
    return run, NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("regression", id="regression-masked"),
        pytest.param("categorical", id="categorical-ignored"),
    ],
)
def test_feat020_streaming_eval_step_matches_the_eager_validation_record(tmp_path, monkeypatch, case):
    from nnx import MetricSpec, MonitorSpec, TaskSpec
    from nnx.streaming import streaming_eval_step

    if case == "regression":
        task = TaskSpec.regression(2)
        nan = float("nan")
        targets = torch.tensor([[1.0, nan], [0.5, -1.0], [nan, nan], [2.0, 0.0], [0.0, nan]])
        metrics, monitor = [MetricSpec("mae")], MonitorSpec(metric="mae")
    else:
        task = TaskSpec.categorical(3, ignore_index=-100)
        targets = torch.tensor([0, -100, 2, 1, -100])
        metrics = [MetricSpec("nll"), MetricSpec("f1"), MetricSpec("accuracy")]
        monitor = MonitorSpec(metric="nll")
    eager, eager_best = _masked_task_run(
        tmp_path / "eager", monkeypatch, task, targets, metrics=metrics, monitor=monitor
    )
    stream, stream_best = _masked_task_run(
        tmp_path / "stream",
        monkeypatch,
        task,
        targets,
        eval_step_fn=streaming_eval_step,
        metrics=metrics,
        monitor=monitor,
    )
    eager_val = [idp.val_edp for idp in eager.idps if idp.val_edp is not None]
    stream_val = [idp.val_edp for idp in stream.idps if idp.val_edp is not None]
    assert len(eager_val) == len(stream_val) == 4
    for expected, actual in zip(eager_val, stream_val, strict=True):
        assert (actual.kind, actual.count, actual.status) == (expected.kind, expected.count, expected.status)
        assert actual.count == (6 if case == "regression" else 3)  # masked targets are not counted
        assert actual.loss == pytest.approx(expected.loss, rel=1e-12)  # summed numerators / summed denominators
        for field in ("accuracy", "f1", "recall", "precision", "error"):
            assert getattr(actual, field) == pytest.approx(getattr(expected, field), rel=1e-12)
        assert dict(actual.metrics) == pytest.approx(dict(expected.metrics), rel=1e-12)
    selections = [(idp.selection.value, idp.selection.improved) for idp in eager.idps if idp.selection is not None]
    streamed = [(idp.selection.value, idp.selection.improved) for idp in stream.idps if idp.selection is not None]
    assert [improved for _, improved in streamed] == [improved for _, improved in selections]
    assert [value for value, _ in streamed] == pytest.approx([value for value, _ in selections], rel=1e-12)
    assert eager_best is not None and stream_best is not None
    assert stream_best.idp.epoch_idx == eager_best.idp.epoch_idx  # the same BEST choice


def test_feat020_streaming_eval_step_matches_legacy_classification_and_keeps_evaluate_eager():
    from nnx import EvalStepContext, MetricSpec
    from nnx.streaming import streaming_eval_step

    model = _model()
    X = torch.randn(5, 4, generator=torch.Generator().manual_seed(2))
    Y = torch.tensor([0, 1, 1, 0, 1])
    loader = DataLoader(TensorDataset(X, Y), batch_size=2)
    declared = (MetricSpec("f1"), MetricSpec("nll"))
    eager = model.evaluate(loader, metrics=declared)
    streamed = streaming_eval_step(
        EvalStepContext(model=model, val_loader=loader, extra_metrics=None, epoch_idx=0, metrics=declared)
    )
    for field in ("accuracy", "f1", "recall", "precision", "loss", "error"):
        assert getattr(streamed, field) == pytest.approx(getattr(eager, field), rel=1e-12)
    assert dict(streamed.metrics) == pytest.approx(dict(eager.metrics), rel=1e-12)
    empty = DataLoader(TensorDataset(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)), batch_size=2)
    with pytest.raises(ValueError, match=r"evaluate\(\) loader produced zero samples"):
        model.evaluate(empty)  # the default path keeps its empty-input exception
    with pytest.raises(ValueError, match="zero samples"):
        streaming_eval_step(EvalStepContext(model=model, val_loader=empty, extra_metrics=None, epoch_idx=0))


def test_feat020_streaming_eval_step_refuses_what_it_cannot_bound_before_training(tmp_path, monkeypatch):
    import os
    from dataclasses import replace

    import numpy as np

    from nnx import MetricSpec, NNTrainParams, register_metric, unregister_metric
    from nnx.streaming import streaming_eval_step

    class _Scores:  # no merge(): needs every stored score
        def update(self, target, prediction):
            pass

        def result(self):
            return 0.5

    monkeypatch.chdir(tmp_path)
    X = torch.randn(4, 4)
    Y = torch.tensor([0, 1, 0, 1])
    batches = [(X[:2], Y[:2]), (X[2:], Y[2:])]
    base = NNTrainParams(
        n_epochs=1, train_loader=batches, val_loader=batches, optim=NNOptimParams.builder().sgd(max_lr=0.1).build()
    )
    with pytest.raises(ValueError, match="extra_metrics"):
        _model().train(
            params=replace(base, extra_metrics={"n": lambda y, y_hat: float(np.size(y))}),
            eval_step_fn=streaming_eval_step,
        )
    register_metric("tests.rank", 1, lambda config: _Scores(), input="probabilities", mode="max")
    try:
        with pytest.raises(ValueError, match="no bounded, mergeable form"):
            _model().train(params=replace(base, metrics=(MetricSpec("tests.rank"),)), eval_step_fn=streaming_eval_step)
    finally:
        unregister_metric("tests.rank", 1)
    assert not os.path.exists("runs")  # refused before any run was reserved


def test_feat020_review_multi_output_bce_without_a_task_scores_like_evaluate():
    from nnx import EvalStepContext
    from nnx.streaming import streaming_eval_step

    class _Indicators:
        """A BCE model without a task whose decisions are fixed indicator rows."""

        evaluate = NNModel.evaluate

        def __init__(self):
            self.net = torch.nn.Identity()
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
            self.device = torch.device("cpu")

        def _fwd_pass(self, batch):
            X, Y = batch
            return X, Y, X, (X >= 0).to(torch.long)

    logits = torch.tensor([[1.0, -1.0, -1.0], [-1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, -1.0, 1.0]])
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    loader = DataLoader(TensorDataset(logits, targets), batch_size=3)  # uneven [3, 1]
    model = _Indicators()
    eager = model.evaluate(loader)
    streamed = streaming_eval_step(
        EvalStepContext(model=model, val_loader=loader, extra_metrics=None, epoch_idx=0)  # type: ignore[arg-type]
    )
    assert eager.accuracy == pytest.approx(0.25)  # exact-row (subset) accuracy, averaged over the 3 labels
    for field in ("accuracy", "f1", "recall", "precision", "loss", "error"):
        assert getattr(streamed, field) == pytest.approx(getattr(eager, field), rel=1e-12)


def test_feat020_review_eager_predict_holds_eval_mode_once_while_a_stream_restores_per_batch(monkeypatch):
    import nnx.nn.nn_model as nn_model_module

    calls = {"n": 0}
    capture = nn_model_module._capture_training_modes

    def counting(module):
        calls["n"] += 1
        return capture(module)

    monkeypatch.setattr(nn_model_module, "_capture_training_modes", counting)
    model = _model()
    loader = DataLoader(TensorDataset(torch.randn(5, 4), torch.zeros(5, dtype=torch.long)), batch_size=2)
    model.predict(loader)
    assert calls["n"] == 1  # one snapshot for the whole eager call
    calls["n"] = 0
    with model.iter_predict(loader) as stream:
        assert len(list(stream)) == 3
    assert calls["n"] == 3  # a stream restores the modes around every batch


def test_feat020_review_the_preflight_survives_functools_wrappers(tmp_path, monkeypatch):
    import functools
    import os

    from nnx import NNTrainParams
    from nnx.streaming import streaming_eval_step

    @functools.wraps(streaming_eval_step)
    def wrapped(ctx):
        return streaming_eval_step(ctx)

    monkeypatch.chdir(tmp_path)
    batches = [(torch.randn(2, 4), torch.tensor([0, 1]))]
    params = NNTrainParams(
        n_epochs=1,
        train_loader=batches,
        val_loader=batches,
        optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
        extra_metrics={"n": lambda y, y_hat: 0.0},
    )
    with pytest.raises(ValueError, match="extra_metrics"):
        _model().train(params=params, eval_step_fn=functools.partial(streaming_eval_step))
    assert not os.path.exists("runs")  # a partial of the step is the step: refused before any run
    with pytest.raises(ValueError, match="extra_metrics"):
        _model().train(params=params, eval_step_fn=wrapped)  # a wrapper is a step of its own ...
    assert os.path.exists("runs")  # ... so the streaming step refuses at its first call instead


def test_feat020_review_round_six_old_task_adapters_and_bounded_extra_metrics():
    from nnx import TaskSpec
    from nnx.tasks import TaskAdapter, task_adapter

    class _Legacy(type(task_adapter(TaskSpec.regression(1)))):  # a subclass written before FEAT-020
        def accumulator(self, *, keep_arrays=False):
            return super().accumulator(keep_arrays=keep_arrays)

    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=1, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.MEAN_SQUARED_ERROR, task=TaskSpec.regression(1)
        ),
    )
    legacy = _Legacy(TaskSpec.regression(1))
    assert isinstance(legacy, TaskAdapter)
    model._task_adapter = legacy  # type: ignore[attr-defined]
    loader = DataLoader(TensorDataset(torch.randn(3, 4), torch.randn(3, 1)), batch_size=2)
    assert model.evaluate(loader).count == 3  # the default path never passes bounded=
    for spec in (TaskSpec.multilabel(2), TaskSpec.regression(1)):
        bounded = task_adapter(spec).accumulator(bounded=True)
        with pytest.raises(ValueError, match="keeps no arrays"):
            bounded.result(loss=0.0, extra_metrics={"n": lambda y, y_hat: 0.0})


def test_feat020_review_round_seven_train_and_plans_refuse_a_broken_factory_alike(tmp_path, monkeypatch):
    import os

    from nnx import MetricSpec, NNTrainParams, register_metric, unregister_metric
    from nnx.streaming import _streaming_problems, streaming_eval_step

    def factory(config):
        raise ImportError("optional dependency missing")

    monkeypatch.chdir(tmp_path)
    batches = [(torch.randn(2, 4), torch.tensor([0, 1]))]
    register_metric("tests.broken", 1, factory, input="labels", mode="max")
    try:
        params = NNTrainParams(
            n_epochs=1,
            train_loader=batches,
            val_loader=batches,
            optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
            metrics=(MetricSpec("tests.broken"),),
        )
        [(_, message)] = _streaming_problems(params)
        with pytest.raises(ValueError) as refused:
            _model().train(params=params, eval_step_fn=streaming_eval_step)
        assert str(refused.value) == message and "optional dependency missing" in message
    finally:
        unregister_metric("tests.broken", 1)
    assert not os.path.exists("runs")


def test_feat020_review_round_eight_the_preflight_checks_the_task_adapter(tmp_path, monkeypatch):
    import os

    from nnx import NNTrainParams, TaskSpec
    from nnx.streaming import streaming_eval_step
    from nnx.tasks import task_adapter

    class _Legacy(type(task_adapter(TaskSpec.regression(1)))):  # accumulator() without bounded=
        def accumulator(self, *, keep_arrays=False):
            return super().accumulator(keep_arrays=keep_arrays)

    monkeypatch.chdir(tmp_path)
    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=1, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.MEAN_SQUARED_ERROR, task=TaskSpec.regression(1)
        ),
    )
    model._task_adapter = _Legacy(TaskSpec.regression(1))  # type: ignore[attr-defined]
    batches = [(torch.randn(2, 4), torch.randn(2, 1))]
    params = NNTrainParams(
        n_epochs=1, train_loader=batches, val_loader=batches, optim=NNOptimParams.builder().sgd(max_lr=0.1).build()
    )
    with pytest.raises(ValueError, match="bounded accumulator"):
        model.train(params=params, eval_step_fn=streaming_eval_step)
    assert not os.path.exists("runs")


def test_feat020_review_round_eleven_bounded_checks_travel_with_the_evaluation(tmp_path, monkeypatch):
    from nnx import EvalStepContext, NNTrainParams, TaskSpec
    from nnx.streaming import streaming_eval_step
    from nnx.tasks import task_adapter

    class _Unbounded(type(task_adapter(TaskSpec.regression(1)))):
        def accumulator(self, *, keep_arrays=False, **ignored):  # never bounded
            return super().accumulator(keep_arrays=keep_arrays)

    monkeypatch.chdir(tmp_path)
    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=1, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.MEAN_SQUARED_ERROR, task=TaskSpec.regression(1)
        ),
    )
    model._task_adapter = _Unbounded(TaskSpec.regression(1))  # type: ignore[attr-defined]
    batches = [(torch.randn(2, 4), torch.randn(2, 1))]
    params = NNTrainParams(
        n_epochs=1, train_loader=batches, val_loader=batches, optim=NNOptimParams.builder().sgd(max_lr=0.1).build()
    )
    with pytest.raises(ValueError, match="not bounded"):  # a wrapped step skips the preflight, not the check
        model.train(params=params, eval_step_fn=lambda ctx: streaming_eval_step(ctx))

    class _ShapeShifter:
        """A legacy BCE stand-in whose label batches change rank."""

        evaluate = NNModel.evaluate

        def __init__(self):
            self.net = torch.nn.Identity()
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
            self.device = torch.device("cpu")
            self._calls = 0

        def _fwd_pass(self, batch):
            self._calls += 1
            shape = (2,) if self._calls == 1 else (2, 3)
            logits, target = torch.zeros(shape), torch.ones(shape)
            return logits, target, logits, (logits >= 0).long()

    stand_in = _ShapeShifter()
    with pytest.raises(ValueError, match="changed shape"):
        streaming_eval_step(
            EvalStepContext(model=stand_in, val_loader=[None, None], extra_metrics=None, epoch_idx=0)  # type: ignore[arg-type]
        )


def test_feat020_review_round_thirteen_an_adapters_own_type_error_is_not_rewrapped():
    from nnx import TaskSpec
    from nnx.nn.nn_model import _bounded_task_accumulator
    from nnx.tasks import task_adapter

    class _Broken(type(task_adapter(TaskSpec.regression(1)))):
        def accumulator(self, *, keep_arrays=False, bounded=False):
            raise TypeError("the adapter's own bug")

    with pytest.raises(TypeError, match="the adapter's own bug"):  # not reported as a missing bounded argument
        _bounded_task_accumulator(_Broken(TaskSpec.regression(1)), "streaming_eval_step()")


def test_feat020_review_round_fourteen_the_adapter_probe_survives_opaque_callables():
    from nnx import TaskSpec
    from nnx.nn.nn_model import _bounded_task_accumulator
    from nnx.tasks import task_adapter

    base = type(task_adapter(TaskSpec.regression(1)))

    class _Positional(base):
        def accumulator(self, *args):  # no bounded keyword
            return super().accumulator()

    with pytest.raises(ValueError, match="takes no bounded keyword"):
        _bounded_task_accumulator(_Positional(TaskSpec.regression(1)), "streaming_eval_step()")
    opaque = base(TaskSpec.regression(1))
    opaque.accumulator = dict  # type: ignore[method-assign]  # inspect.signature(dict) raises
    with pytest.raises(ValueError, match="not bounded"):  # the call decides, with the usual message
        _bounded_task_accumulator(opaque, "streaming_eval_step()")


def test_feat020_review_round_fifteen_opaque_adapters_get_the_bounded_message():
    from nnx import TaskSpec
    from nnx.nn.nn_model import _bounded_task_accumulator
    from nnx.tasks import task_adapter

    class _Opaque:  # no introspectable signature, and no bounded keyword
        @property
        def __signature__(self):
            raise ValueError("no signature")

        def __call__(self):
            return None

    adapter = type(task_adapter(TaskSpec.regression(1)))(TaskSpec.regression(1))
    adapter.accumulator = _Opaque()  # type: ignore[method-assign]
    with pytest.raises(ValueError, match=r"accumulator\(bounded=True\) raised TypeError"):
        _bounded_task_accumulator(adapter, "streaming_eval_step()")


def test_feat020_review_round_eighteen_kwargs_forwarding_adapters_get_the_bounded_message():
    from nnx import TaskSpec
    from nnx.nn.nn_model import _bounded_task_accumulator
    from nnx.tasks import task_adapter

    base = type(task_adapter(TaskSpec.regression(1)))

    class _PreFeat020(base):
        def accumulator(self, *, keep_arrays=False):  # no bounded keyword
            return super().accumulator(keep_arrays=keep_arrays)

    class _Forwarding(_PreFeat020):
        def accumulator(self, **options):  # forwards everything to the older base
            return super().accumulator(**options)

    with pytest.raises(ValueError, match=r"accumulator\(bounded=True\) raised TypeError"):
        _bounded_task_accumulator(_Forwarding(TaskSpec.regression(1)), "streaming_eval_step()")


def test_feat020_review_round_nineteen_a_positional_only_bounded_is_no_keyword():
    from nnx import TaskSpec
    from nnx.nn.nn_model import _bounded_task_accumulator
    from nnx.tasks import task_adapter

    class _PositionalOnly(type(task_adapter(TaskSpec.regression(1)))):
        def accumulator(self, bounded=False, /):
            return super().accumulator(bounded=bounded)

    with pytest.raises(ValueError, match="takes no bounded keyword"):
        _bounded_task_accumulator(_PositionalOnly(TaskSpec.regression(1)), "streaming_eval_step()")
