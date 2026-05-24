"""Pass-2 catalog: V-series tests (reproducibility).

- V1: NNTrainParams.seed pins RNGs at train() entry, producing identical
  weights and metrics across repeated runs with the same seed.
- V1 back-compat: seed=None NNTrainParams.state() omits the `seed` key so
  run.id is unchanged vs. pre-seed runs.
- V2: dataloader_worker_init_fn produces deterministic worker seeds.
- V3: metadata.yaml is written alongside run.yaml and contains env info
  but is NOT part of state() / run.id.
"""
from __future__ import annotations

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
from nnx.nn.params.nn_scheduler_params import NNSchedulerParams
from nnx.nn.params.nn_train_params import NNTrainParams
from nnx.seeding import dataloader_worker_init_fn, env_snapshot, set_seed


def _build_train_params(seed=None, n_epochs=1):
    X = torch.randn(32, 4)
    y = torch.randint(0, 2, (32,))
    return NNTrainParams(
        n_epochs=n_epochs,
        train_loader=DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True),
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
        seed=seed,
    )


def _make_model():
    return NNModel(
        net_params=NNParams(
            input_dim=4, output_dim=2, hidden_dims=[8],
            dropout_prob=0.0, activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
        ),
    )


def test_v1_seed_makes_runs_reproducible(tmp_path, monkeypatch):
    """Two NNModel.train() invocations with the same seed produce identical
    final weights (CPU; deterministic float math)."""
    monkeypatch.chdir(tmp_path)

    set_seed(42)
    m1 = _make_model()
    run1 = m1.train(params=_build_train_params(seed=42))
    w1 = m1.net.state_dict()

    # Reset cwd to avoid the runs/best symlink collision between trials.
    (tmp_path / "runs2").mkdir()
    monkeypatch.chdir(tmp_path / "runs2")

    set_seed(42)
    m2 = _make_model()
    run2 = m2.train(params=_build_train_params(seed=42))
    w2 = m2.net.state_dict()

    # Every parameter should match exactly across the two runs.
    for k in w1.keys():
        assert torch.equal(w1[k], w2[k]), f"mismatch in {k}"
    # And the run.ids match because state() is deterministic.
    assert run1.id == run2.id


def test_v1_seed_none_preserves_back_compat_run_id():
    """A NNTrainParams with seed=None must hash to the same run.id as the
    pre-seed code did — i.e., the `seed` key must NOT appear in state()."""
    p = NNTrainParams(
        n_epochs=10,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=5e-5),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=2, cooldown=1, threshold=1e-3),
        # seed=None (default)
    )
    state = p.state()
    assert 'seed' not in state, "seed=None must not appear in state() to preserve back-compat"


def test_v1_seed_set_appears_in_state():
    """When seed is set, it IS in state() — affecting run.id intentionally
    (different seeds => different runs)."""
    p = NNTrainParams(
        n_epochs=5,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=5e-5),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=2, cooldown=1, threshold=1e-3),
        seed=123,
    )
    state = p.state()
    assert state['seed'] == 123


def test_v1_train_params_round_trip_with_seed():
    p = NNTrainParams(
        n_epochs=5, seed=7,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=2, cooldown=1, threshold=1e-3),
    )
    rt = NNTrainParams.from_state(p.state())
    assert rt.seed == 7


def test_v1_train_params_from_state_legacy_yaml_no_seed_key():
    """A YAML produced before the seed field existed must still load."""
    legacy = {
        'n_epochs': 10,
        'optim': {
            'max_lr': 1e-3, 'momentum': "(0.9, 0.999)",
            'name': 'adam', 'weight_decay': 0.0,
        },
        'scheduler': {
            'min_lr': 1e-7, 'factor': 0.5, 'patience': 2,
            'cooldown': 1, 'threshold': 1e-3, 'kind': None,
            'step_size': None, 'T_max': None, 'max_lr': None,
            'total_steps': None, 'warmup_steps': None,
        },
        # NO 'seed' key
    }
    p = NNTrainParams.from_state(legacy)
    assert p.seed is None


def test_v2_dataloader_worker_init_fn_deterministic():
    """Two calls to worker_init_fn with the same worker_id under the same
    torch seed produce the same numpy RNG state."""
    import numpy as np

    torch.manual_seed(99)
    dataloader_worker_init_fn(worker_id=3)
    state_a = np.random.get_state()[1].tolist()

    torch.manual_seed(99)
    dataloader_worker_init_fn(worker_id=3)
    state_b = np.random.get_state()[1].tolist()

    assert state_a == state_b


def test_v2_dataloader_worker_init_fn_diverges_per_worker():
    """Different worker_ids must produce different numpy seeds; otherwise
    each worker would emit identical samples."""
    import numpy as np

    torch.manual_seed(99)
    dataloader_worker_init_fn(worker_id=0)
    rand_0 = np.random.rand()

    torch.manual_seed(99)
    dataloader_worker_init_fn(worker_id=1)
    rand_1 = np.random.rand()

    assert rand_0 != rand_1


def test_v3_env_snapshot_returns_serializable_dict():
    snap = env_snapshot()
    assert isinstance(snap, dict)
    # Required keys present even when some fail (None is acceptable).
    for k in ("nnx", "python", "torch", "numpy", "platform",
              "cuda_available", "cuda_device_count", "git_commit", "git_dirty"):
        assert k in snap
    # python / torch / numpy / platform always succeed.
    assert snap["python"] is not None
    assert snap["torch"] is not None
    assert snap["numpy"] is not None


def test_v3_metadata_yaml_written_by_run_save(tmp_path, monkeypatch):
    """NNRun.save() writes runs/<id>/metadata.yaml alongside run.yaml.
    The metadata file is NOT used in run.id computation."""
    monkeypatch.chdir(tmp_path)

    m = _make_model()
    run = m.train(params=_build_train_params(seed=11))

    run_dir = tmp_path / "runs" / run.id
    assert (run_dir / "run.yaml").exists()
    assert (run_dir / "metadata.yaml").exists()

    import yaml
    with open(run_dir / "metadata.yaml") as f:
        meta = yaml.safe_load(f)
    assert meta["torch"] is not None
