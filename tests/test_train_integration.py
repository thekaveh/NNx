"""End-to-end integration test for NNModel.train().

Exercises the full path: build model → train for a few epochs on a tiny
in-memory dataset → assert checkpoints + run files land on disk → reload
the run and reconstruct a model from the BEST checkpoint.

Uses a small random dataset so it stays fast (<10s on CPU) and avoids
network downloads. Uses tmp_path + chdir so the runs/ directory lands in
a pytest temp dir and doesn't pollute the repo."""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
import threading

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx.finetune.param_groups import NNParamGroupSpec
from nnx.nn.callbacks import Callback, ModelCheckpoint
from nnx.nn.enum.activations import Activations
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.nn.enum.devices import Devices
from nnx.nn.enum.losses import Losses
from nnx.nn.enum.nets import Nets
from nnx.nn.enum.optims import Optims
from nnx.nn.enum.schedulers import Schedulers
from nnx.nn.nn_model import NNModel, _capture_rng_state, _restore_rng_state
from nnx.nn.params.nn_checkpoint import NNCheckpoint
from nnx.nn.params.nn_model_params import NNModelParams
from nnx.nn.params.nn_optim_params import NNOptimParams
from nnx.nn.params.nn_params import NNParams
from nnx.nn.params.nn_run import NNRun
from nnx.nn.params.nn_scheduler_params import NNSchedulerParams
from nnx.nn.params.nn_train_params import NNTrainParams


def _save_checkpoint_in_process(checkpoint, root: str, marker: int) -> None:
    checkpoint.save(
        run="concurrent-run",
        type=Checkpoints.LAST,
        root=root,
        optimizer_state={"state": {}, "param_groups": [{"marker": marker}]},
        completed_epoch=marker,
    )


def _reserve_run_in_process(root: str) -> str:
    net_params, model_params = _make_params()
    run = NNRun(
        net=net_params,
        model=model_params,
        train=NNTrainParams(n_epochs=1, data_id="concurrent-admission"),
    )
    try:
        run.ensure_writable(root=root)
    except FileExistsError:
        return "blocked"
    return "reserved"


def _make_tiny_loaders(n_train: int = 32, n_val: int = 16, input_dim: int = 8, n_classes: int = 3):
    """Random classification data — just enough to drive a forward/backward."""
    torch.manual_seed(0)
    np.random.seed(0)

    X_train = torch.randn(n_train, input_dim)
    y_train = torch.randint(0, n_classes, (n_train,))
    X_val = torch.randn(n_val, input_dim)
    y_val = torch.randint(0, n_classes, (n_val,))

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=8, shuffle=False)
    return train_loader, val_loader


def _make_params(input_dim: int = 8, output_dim: int = 3):
    net_params = NNParams(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=[16],
        dropout_prob=0.0,
        activation=Activations.RELU,
    )
    model_params = NNModelParams(
        net=Nets.FEED_FWD,
        device=Devices.CPU,
        loss=Losses.CROSS_ENTROPY,
    )
    return net_params, model_params


def _train_params(train_loader, val_loader, n_epochs: int = 2):
    return NNTrainParams(
        n_epochs=n_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=NNOptimParams(
            name=Optims.ADAM,
            max_lr=1e-2,
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
    )


def test_train_end_to_end_produces_run_and_checkpoints(tmp_path, monkeypatch):
    """train() saves a run with one idp per batch and at least BEST+LAST
    checkpoints. Reloading the run reconstructs every idp."""
    monkeypatch.chdir(tmp_path)

    net_params, model_params = _make_params()
    train_loader, val_loader = _make_tiny_loaders()
    train_params = _train_params(train_loader, val_loader, n_epochs=2)

    model = NNModel(net_params=net_params, params=model_params)
    run = model.train(params=train_params)

    # Every batch produces an idp; with batch_size=8 and 32 train samples
    # that's 4 batches × 2 epochs = 8 idps.
    assert len(run.idps) == 8
    # The final idps in each epoch should have val_edp populated.
    last_in_first_epoch = run.idps[3]
    last_in_second_epoch = run.idps[7]
    assert last_in_first_epoch.val_edp is not None
    assert last_in_second_epoch.val_edp is not None

    # On-disk artifacts.
    run_dir = tmp_path / "runs" / run.id
    assert (run_dir / "run.yaml").exists()
    assert (run_dir / "idps.csv").exists()
    assert (run_dir / "checkpoints" / "first.pt").exists()
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert (run_dir / "checkpoints" / "best.pt").exists()
    # runs/best symlink points at this run (no prior runs in tmp_path).
    assert os.path.islink(tmp_path / "runs" / "best")


def test_run_save_load_round_trip(tmp_path, monkeypatch):
    """NNRun.load(run.id) returns an NNRun whose idps/state match the saved one."""
    monkeypatch.chdir(tmp_path)

    net_params, model_params = _make_params()
    train_loader, val_loader = _make_tiny_loaders()
    train_params = _train_params(train_loader, val_loader, n_epochs=1)

    model = NNModel(net_params=net_params, params=model_params)
    original = model.train(params=train_params)

    reloaded = NNRun.load(id=original.id)

    assert reloaded.id == original.id
    assert reloaded.net == original.net
    assert reloaded.model == original.model
    # train_loader / val_loader are runtime-only (repr=False, not serialized);
    # compare the serializable parts.
    assert reloaded.train.n_epochs == original.train.n_epochs
    assert reloaded.train.optim == original.train.optim
    assert reloaded.train.scheduler == original.train.scheduler
    assert len(reloaded.idps) == len(original.idps)
    # Per-iteration metrics survive CSV → DataFrame → dict round-trip.
    for ridp, oidp in zip(reloaded.idps, original.idps, strict=True):
        assert ridp.iter_idx == oidp.iter_idx
        assert ridp.epoch_idx == oidp.epoch_idx
        assert ridp.batch_idx == oidp.batch_idx
        # Floats may differ by float64 → string → float64 noise; tolerate epsilon.
        assert abs(ridp.train_edp.loss - oidp.train_edp.loss) < 1e-9


def test_checkpoint_reconstruct_predicts(tmp_path, monkeypatch):
    """The BEST checkpoint can be loaded and used to build a working NNModel
    that produces predictions of the right shape."""
    monkeypatch.chdir(tmp_path)

    net_params, model_params = _make_params()
    train_loader, val_loader = _make_tiny_loaders()
    train_params = _train_params(train_loader, val_loader, n_epochs=2)

    model = NNModel(net_params=net_params, params=model_params)
    run = model.train(params=train_params)

    ckpt = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert ckpt is not None

    reloaded = NNModel.from_checkpoint(checkpoint=ckpt)
    X = np.random.RandomState(0).randn(4, 8).astype(np.float32)
    log, hat = reloaded.predict(X=X)
    assert log.shape == (4, 3)
    assert hat.shape == (4,)


def test_train_skips_val_loop_when_no_val_loader(tmp_path, monkeypatch):
    """Without a val_loader, idps[*].val_edp is None and the run still saves
    cleanly (regression for the no-val NNRun.save crash)."""
    monkeypatch.chdir(tmp_path)

    net_params, model_params = _make_params()
    train_loader, _ = _make_tiny_loaders()
    train_params = _train_params(train_loader, val_loader=None, n_epochs=1)

    model = NNModel(net_params=net_params, params=model_params)
    run = model.train(params=train_params)

    assert all(idp.val_edp is None for idp in run.idps)
    assert (tmp_path / "runs" / run.id / "run.yaml").exists()


def test_warm_resume_restores_scheduler_and_continues_epoch_progress(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base_loader, _ = _make_tiny_loaders()
    dataset = base_loader.dataset
    full_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        generator=torch.Generator().manual_seed(1234),
    )
    split_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        generator=torch.Generator().manual_seed(1234),
    )
    scheduler = NNSchedulerParams(
        kind=Schedulers.STEP,
        step_size=1,
        factor=0.5,
        min_lr=0.0,
        patience=0,
        cooldown=0,
        threshold=0.0,
    )

    torch.manual_seed(7)
    net_params, model_params = _make_params()
    uninterrupted = NNModel(net_params=net_params, params=model_params)
    full_run = uninterrupted.train(
        params=NNTrainParams(
            n_epochs=4,
            data_id="full",
            train_loader=full_loader,
            optim=NNOptimParams(name=Optims.SGD, max_lr=0.1, momentum=0.0, weight_decay=0.0),
            scheduler=scheduler,
        )
    )

    torch.manual_seed(7)
    first_half = NNModel(net_params=net_params, params=model_params)
    first_run = first_half.train(
        params=NNTrainParams(
            n_epochs=2,
            data_id="split",
            train_loader=split_loader,
            optim=NNOptimParams(name=Optims.SGD, max_lr=0.1, momentum=0.0, weight_decay=0.0),
            scheduler=scheduler,
        )
    )
    resumed_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        generator=torch.Generator().manual_seed(9999),
    )
    resumed = NNModel(net_params=net_params, params=model_params)
    resumed_run = resumed.train(
        params=NNTrainParams(
            n_epochs=2,
            data_id="split",
            train_loader=resumed_loader,
            optim=NNOptimParams(name=Optims.SGD, max_lr=0.1, momentum=0.0, weight_decay=0.0),
            scheduler=scheduler,
            resume_from_run_id=first_run.id,
        )
    )

    full_state = NNCheckpoint.load_training_state(full_run.id, Checkpoints.LAST)
    resumed_state = NNCheckpoint.load_training_state(resumed_run.id, Checkpoints.LAST)
    assert full_state is not None
    assert resumed_state is not None
    assert resumed_state["completed_epoch"] == 3
    assert resumed_run.idps[0].epoch_idx == 2
    assert resumed_state["optimizer"]["param_groups"][0]["lr"] == full_state["optimizer"]["param_groups"][0]["lr"]
    assert resumed_state["scheduler"]["last_epoch"] == full_state["scheduler"]["last_epoch"]
    assert "train_loader_generator" in resumed_state["rng"]
    for name, tensor in uninterrupted.net.state_dict().items():
        assert torch.equal(resumed.net.state_dict()[name], tensor)


def test_training_rng_round_trips_batch_sampler_generator():
    dataset = TensorDataset(torch.arange(8))

    class GeneratedBatchSampler:
        def __init__(self, seed: int):
            self.generator = torch.Generator().manual_seed(seed)

        def __iter__(self):
            order = torch.randperm(8, generator=self.generator).tolist()
            yield from (order[:4], order[4:])

        def __len__(self):
            return 2

    source = DataLoader(dataset, batch_sampler=GeneratedBatchSampler(17))
    state = _capture_rng_state(source)
    expected = list(iter(source.batch_sampler))

    restored = DataLoader(dataset, batch_sampler=GeneratedBatchSampler(99))
    _restore_rng_state(state, restored)
    assert list(iter(restored.batch_sampler)) == expected


def test_training_rng_restores_equivalent_generator_at_different_attachment():
    dataset = TensorDataset(torch.arange(8))
    source_generator = torch.Generator().manual_seed(17)
    source = DataLoader(dataset, batch_size=2, generator=source_generator)
    state = _capture_rng_state(source)
    expected = torch.randperm(8, generator=source_generator)

    target_generator = torch.Generator().manual_seed(99)
    target_sampler = torch.utils.data.RandomSampler(dataset, generator=target_generator)
    target = DataLoader(dataset, batch_size=2, sampler=target_sampler)
    _restore_rng_state(state, target)
    assert torch.equal(torch.randperm(8, generator=target_generator), expected)


def test_training_rng_round_trips_mps_state(monkeypatch):
    saved = torch.tensor([1, 2, 3], dtype=torch.uint8)
    restored = []
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: saved)
    monkeypatch.setattr(torch.mps, "set_rng_state", restored.append)

    state = _capture_rng_state()
    _restore_rng_state(state)

    assert torch.equal(state["mps"], saved)
    assert len(restored) == 1
    assert torch.equal(restored[0], saved)


def test_epoch_end_model_mutation_is_present_in_best_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    model = NNModel(net_params=net_params, params=model_params)

    class ShiftWeights(Callback):
        def on_epoch_end(self, ctx):
            with torch.no_grad():
                next(ctx.model.net.parameters()).add_(1.0)

    run = model.train(params=_train_params(train_loader, None, n_epochs=1), callbacks=[ShiftWeights()])
    best = NNCheckpoint.load(run.id, Checkpoints.BEST)
    assert best is not None
    for name, tensor in model.net.state_dict().items():
        assert torch.equal(best.net_state[name], tensor)


def test_training_state_rejects_checkpoint_sidecar_generation_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    model = NNModel(net_params=net_params, params=model_params)
    run = model.train(params=_train_params(train_loader, None, n_epochs=1))

    checkpoint = NNCheckpoint.load(run.id, Checkpoints.LAST)
    assert checkpoint is not None and checkpoint.training_state_id is not None
    sidecar = tmp_path / "runs" / run.id / "checkpoints" / f"last.pt.opt.{checkpoint.training_state_id}.pt"
    state = torch.load(sidecar, weights_only=True)
    state["checkpoint_id"] = "different-generation"
    torch.save(state, sidecar)

    with pytest.raises(ValueError, match="checkpoint and training-state sidecar do not match"):
        NNCheckpoint.load_training_state(run.id, Checkpoints.LAST)


def test_training_state_rejects_missing_owned_sidecar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    run = NNModel(net_params=net_params, params=model_params).train(
        params=_train_params(train_loader, None, n_epochs=1)
    )
    checkpoint = NNCheckpoint.load(run.id, Checkpoints.LAST)
    assert checkpoint is not None and checkpoint.training_state_id is not None
    checkpoint_dir = tmp_path / "runs" / run.id / "checkpoints"
    (checkpoint_dir / f"last.pt.opt.{checkpoint.training_state_id}.pt").unlink()
    (checkpoint_dir / "last.pt.opt.pt").unlink()

    with pytest.raises(ValueError, match="references a missing training-state sidecar"):
        NNCheckpoint.load_training_state(run.id, Checkpoints.LAST)


def test_versioned_checkpoint_rejects_legacy_fallback_sidecar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    run = NNModel(net_params=net_params, params=model_params).train(
        params=_train_params(train_loader, None, n_epochs=1)
    )
    checkpoint = NNCheckpoint.load(run.id, Checkpoints.LAST)
    assert checkpoint is not None and checkpoint.training_state_id is not None
    checkpoint_dir = tmp_path / "runs" / run.id / "checkpoints"
    (checkpoint_dir / f"last.pt.opt.{checkpoint.training_state_id}.pt").unlink()
    torch.save({"state": {}, "param_groups": []}, checkpoint_dir / "last.pt.opt.pt")

    with pytest.raises(ValueError, match="cannot use a legacy optimizer-only sidecar"):
        NNCheckpoint.load_training_state(run.id, Checkpoints.LAST)


def test_interrupted_optimizerless_cleanup_ignores_stale_sidecars(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    run = NNModel(net_params=net_params, params=model_params).train(
        params=_train_params(train_loader, None, n_epochs=1)
    )
    checkpoint = NNCheckpoint.load(run.id, Checkpoints.LAST)
    assert checkpoint is not None
    generic_sidecar = tmp_path / "runs" / run.id / "checkpoints" / "last.pt.opt.pt"
    original_remove = os.remove

    def interrupt_cleanup(path):
        if os.path.abspath(os.fspath(path)) == os.path.abspath(os.fspath(generic_sidecar)):
            raise KeyboardInterrupt
        original_remove(path)

    monkeypatch.setattr(os, "remove", interrupt_cleanup)
    with pytest.raises(KeyboardInterrupt):
        checkpoint.save(run.id, Checkpoints.LAST)

    reloaded = NNCheckpoint.load(run.id, Checkpoints.LAST)
    assert reloaded is not None and reloaded.training_state_id is None
    assert NNCheckpoint.load_training_state(run.id, Checkpoints.LAST) is None


def test_interrupted_optimizerless_cleanup_ignores_stale_legacy_sidecar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    run = NNModel(net_params=net_params, params=model_params).train(
        params=_train_params(train_loader, None, n_epochs=1)
    )
    checkpoint = NNCheckpoint.load(run.id, Checkpoints.LAST)
    assert checkpoint is not None
    generic_sidecar = tmp_path / "runs" / run.id / "checkpoints" / "last.pt.opt.pt"
    torch.save({"state": {}, "param_groups": []}, generic_sidecar)

    monkeypatch.setattr(os, "remove", lambda _path: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        checkpoint.save(run.id, Checkpoints.LAST)

    assert NNCheckpoint.load_training_state(run.id, Checkpoints.LAST) is None


def test_sidecar_cleanup_treats_root_glob_characters_literally(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    run = NNModel(net_params=net_params, params=model_params).train(
        params=_train_params(train_loader, None, n_epochs=1)
    )
    checkpoint = NNCheckpoint.load(run.id, Checkpoints.LAST)
    assert checkpoint is not None
    optimizer_state = {"state": {}, "param_groups": []}
    literal_root = tmp_path / "root[1]"
    sibling_root = tmp_path / "root1"
    checkpoint.save(run.id, Checkpoints.LAST, root=str(literal_root), optimizer_state=optimizer_state)
    checkpoint.save(run.id, Checkpoints.LAST, root=str(sibling_root), optimizer_state=optimizer_state)
    sibling_generations = list((sibling_root / "runs" / run.id / "checkpoints").glob("last.pt.opt.*.pt"))
    assert sibling_generations

    checkpoint.save(run.id, Checkpoints.LAST, root=str(literal_root))

    assert all(path.exists() for path in sibling_generations)


def test_interrupted_sidecar_publication_preserves_previous_generation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    run = NNModel(net_params=net_params, params=model_params).train(
        params=_train_params(train_loader, None, n_epochs=1)
    )
    checkpoint = NNCheckpoint.load(run.id, Checkpoints.LAST)
    before = NNCheckpoint.load_training_state(run.id, Checkpoints.LAST)
    assert checkpoint is not None and before is not None

    import nnx.nn.params.nn_checkpoint as checkpoint_module

    original_atomic_save = checkpoint_module._atomic_torch_save

    def fail_new_generation(obj, path):
        if ".opt." in path and not path.endswith(".opt.pt"):
            raise KeyboardInterrupt
        return original_atomic_save(obj, path)

    monkeypatch.setattr(checkpoint_module, "_atomic_torch_save", fail_new_generation)
    with pytest.raises(KeyboardInterrupt):
        checkpoint.save(
            run.id,
            Checkpoints.LAST,
            optimizer_state=before["optimizer"],
            scheduler_state=before["scheduler"],
            rng_state=before["rng"],
        )

    reloaded = NNCheckpoint.load(run.id, Checkpoints.LAST)
    after = NNCheckpoint.load_training_state(run.id, Checkpoints.LAST)
    assert reloaded is not None and after is not None
    assert reloaded.training_state_id == checkpoint.training_state_id
    assert after["checkpoint_id"] == before["checkpoint_id"]


def test_training_state_rejects_unsupported_version(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    run = NNModel(net_params=net_params, params=model_params).train(
        params=_train_params(train_loader, None, n_epochs=1)
    )
    checkpoint = NNCheckpoint.load(run.id, Checkpoints.LAST)
    assert checkpoint is not None and checkpoint.training_state_id is not None
    sidecar = tmp_path / "runs" / run.id / "checkpoints" / f"last.pt.opt.{checkpoint.training_state_id}.pt"
    state = torch.load(sidecar, weights_only=True)
    state["nnx_training_state_version"] = 999
    torch.save(state, sidecar)

    with pytest.raises(ValueError, match="unsupported training-state sidecar version"):
        NNCheckpoint.load_training_state(run.id, Checkpoints.LAST)


def test_training_state_rejects_non_mapping_sidecar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    run = NNModel(net_params=net_params, params=model_params).train(
        params=_train_params(train_loader, None, n_epochs=1)
    )
    checkpoint = NNCheckpoint.load(run.id, Checkpoints.LAST)
    assert checkpoint is not None and checkpoint.training_state_id is not None
    sidecar = tmp_path / "runs" / run.id / "checkpoints" / f"last.pt.opt.{checkpoint.training_state_id}.pt"
    torch.save(["not", "a", "mapping"], sidecar)

    with pytest.raises(ValueError, match="expected a mapping"):
        NNCheckpoint.load_training_state(run.id, Checkpoints.LAST)


def test_failed_resume_rolls_back_model_and_loader_rng(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    run = NNModel(net_params=net_params, params=model_params).train(
        params=_train_params(train_loader, None, n_epochs=1)
    )
    checkpoint = NNCheckpoint.load(run.id, Checkpoints.LAST)
    assert checkpoint is not None and checkpoint.training_state_id is not None
    sidecar = tmp_path / "runs" / run.id / "checkpoints" / f"last.pt.opt.{checkpoint.training_state_id}.pt"
    state = torch.load(sidecar, weights_only=True)
    state["optimizer"]["param_groups"] = []
    torch.save(state, sidecar)

    resume_loader = DataLoader(
        train_loader.dataset,
        batch_size=8,
        shuffle=True,
        generator=torch.Generator().manual_seed(123),
    )
    candidate = NNModel(net_params=net_params, params=model_params)
    before_model = {name: tensor.detach().clone() for name, tensor in candidate.net.state_dict().items()}
    before_generator = resume_loader.generator.get_state().clone()

    with pytest.raises(ValueError, match="parameter group"):
        candidate.train(
            params=NNTrainParams(
                n_epochs=1,
                data_id="rollback-resume",
                train_loader=resume_loader,
                resume_from_run_id=run.id,
                optim=_train_params(train_loader, None).optim,
                scheduler=_train_params(train_loader, None).scheduler,
            )
        )
    assert torch.equal(resume_loader.generator.get_state(), before_generator)
    for name, tensor in candidate.net.state_dict().items():
        assert torch.equal(tensor, before_model[name])


def test_legacy_resume_base_exception_rolls_back_model_and_rng(tmp_path, monkeypatch):
    from dataclasses import replace

    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    source = NNModel(net_params=net_params, params=model_params).train(
        params=_train_params(train_loader, None, n_epochs=1)
    )
    source_checkpoint = NNCheckpoint.load(source.id, Checkpoints.LAST)
    assert source_checkpoint is not None
    replace(source_checkpoint, training_state_id=None).save("legacy-resume", Checkpoints.LAST)

    resume_loader = DataLoader(
        train_loader.dataset,
        batch_size=8,
        shuffle=True,
        generator=torch.Generator().manual_seed(123),
    )
    candidate = NNModel(net_params=net_params, params=model_params)
    before_model = {name: value.detach().clone() for name, value in candidate.net.state_dict().items()}
    before_rng = torch.get_rng_state().clone()
    before_loader_rng = resume_loader.generator.get_state().clone()
    original_load = candidate.net.load_state_dict
    calls = 0

    def interrupt_first_load(state, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            with torch.no_grad():
                next(candidate.net.parameters()).add_(9.0)
            torch.rand(1)
            resume_loader.generator.manual_seed(999)
            raise KeyboardInterrupt
        return original_load(state, *args, **kwargs)

    monkeypatch.setattr(candidate.net, "load_state_dict", interrupt_first_load)
    with pytest.raises(KeyboardInterrupt):
        candidate.train(
            params=NNTrainParams(
                n_epochs=1,
                data_id="legacy-base-exception",
                train_loader=resume_loader,
                resume_from_run_id="legacy-resume",
                optim=_train_params(train_loader, None).optim,
                scheduler=_train_params(train_loader, None).scheduler,
            )
        )

    assert torch.equal(torch.get_rng_state(), before_rng)
    assert torch.equal(resume_loader.generator.get_state(), before_loader_rng)
    for name, value in candidate.net.state_dict().items():
        assert torch.equal(value, before_model[name])


def test_run_history_failure_cannot_publish_completed_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    original_save = NNRun.save

    def fail_with_history(self, *args, **kwargs):
        if self.idps:
            raise OSError("injected run history failure")
        return original_save(self, *args, **kwargs)

    monkeypatch.setattr(NNRun, "save", fail_with_history)
    with pytest.raises(OSError, match="injected run history failure"):
        NNModel(net_params=net_params, params=model_params).train(params=_train_params(train_loader, None, n_epochs=1))
    assert not list(tmp_path.glob("runs/*/checkpoints/last.pt"))


def test_checkpoint_failure_rolls_run_history_back(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    params = _train_params(train_loader, None, n_epochs=1)
    run_id = NNRun(net=net_params, model=model_params, train=params).id

    def fail_checkpoint(*args, **kwargs):
        raise OSError("injected checkpoint failure")

    monkeypatch.setattr(NNModel, "_save_checkpoints", fail_checkpoint)
    with pytest.raises(OSError, match="injected checkpoint failure"):
        NNModel(net_params=net_params, params=model_params).train(params=params)

    loaded = NNRun.load(run_id)
    assert loaded.idps == []
    assert not (tmp_path / "runs" / run_id / "checkpoints" / "last.pt").exists()


def test_ancillary_checkpoint_failure_keeps_last_committed_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    params = _train_params(train_loader, None, n_epochs=1)
    run_id = NNRun(net=net_params, model=model_params, train=params).id
    original_save = NNCheckpoint.save

    def fail_after_last(self, run, type, *args, **kwargs):
        if type != Checkpoints.LAST:
            raise OSError("injected ancillary checkpoint failure")
        return original_save(self, run, type, *args, **kwargs)

    monkeypatch.setattr(NNCheckpoint, "save", fail_after_last)
    with pytest.raises(OSError, match="ancillary checkpoint failure"):
        NNModel(net_params=net_params, params=model_params).train(params=params)

    committed = NNCheckpoint.load(run_id, Checkpoints.LAST)
    loaded = NNRun.load(run_id)
    assert committed is not None
    assert loaded.idps is not None
    assert len(loaded.idps) == len(train_loader)


def test_callback_checkpoint_is_not_published_when_later_callback_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()

    class FailingCallback(Callback):
        def on_epoch_end(self, ctx):
            raise RuntimeError("injected callback failure")

    with pytest.raises(RuntimeError, match="injected callback failure"):
        NNModel(net_params=net_params, params=model_params).train(
            params=_train_params(train_loader, None, n_epochs=1),
            callbacks=[ModelCheckpoint(epochs=[0], tag="queued"), FailingCallback()],
        )
    assert not list(tmp_path.glob("runs/*/checkpoints/queued_e0.pt"))


def test_deferred_model_checkpoint_freezes_callback_time_weights(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    captured = {}

    class CaptureThenMutate(Callback):
        def on_epoch_end(self, ctx):
            captured.update({name: tensor.detach().clone() for name, tensor in ctx.model.net.state_dict().items()})
            with torch.no_grad():
                next(ctx.model.net.parameters()).add_(10.0)

    run = NNModel(net_params=net_params, params=model_params).train(
        params=_train_params(train_loader, None, n_epochs=1),
        callbacks=[ModelCheckpoint(epochs=[0], tag="frozen"), CaptureThenMutate()],
    )
    custom = NNCheckpoint.from_file(str(tmp_path / "runs" / run.id / "checkpoints" / "frozen_e0.pt"))
    assert custom is not None
    for name, tensor in captured.items():
        assert torch.equal(custom.net_state[name], tensor)


def test_failed_deferred_callback_does_not_update_global_best(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()

    class DeferredFailure(Callback):
        def on_epoch_end(self, ctx):
            def fail():
                raise OSError("deferred failure")

            ctx.deferred_checkpoint_writes.append(fail)

    with pytest.raises(OSError, match="deferred failure"):
        NNModel(net_params=net_params, params=model_params).train(
            params=_train_params(train_loader, None, n_epochs=1),
            callbacks=[DeferredFailure()],
        )

    assert not os.path.lexists(tmp_path / "runs" / "best")


def test_phase_checkpoints_carry_complete_resume_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    run = NNModel(net_params=net_params, params=model_params).train(
        params=_train_params(train_loader, None, n_epochs=8)
    )

    for checkpoint_type in (Checkpoints.FIRST, Checkpoints.Q1, Checkpoints.Q2, Checkpoints.Q3):
        state = NNCheckpoint.load_training_state(run.id, checkpoint_type)
        assert state is not None
        assert state["optimizer"] is not None
        assert state["optimizer_type"].endswith(".Adam")
        assert state["scheduler"] is not None
        assert state["scheduler_type"].endswith(".ReduceLROnPlateau")
        assert state["rng"] is not None


def test_warm_resume_rejects_optimizer_and_scheduler_type_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    step = NNSchedulerParams(
        kind=Schedulers.STEP,
        step_size=1,
        factor=0.5,
        min_lr=0.0,
        patience=0,
        cooldown=0,
        threshold=0.0,
    )
    first = NNModel(net_params=net_params, params=model_params).train(
        params=NNTrainParams(
            n_epochs=1,
            data_id="typed-resume-source",
            train_loader=train_loader,
            optim=NNOptimParams(name=Optims.ADAM, max_lr=0.01, momentum=(0.9, 0.999), weight_decay=0.0),
            scheduler=step,
        )
    )

    with pytest.raises(ValueError, match="optimizer type mismatch"):
        NNModel(net_params=net_params, params=model_params).train(
            params=NNTrainParams(
                n_epochs=1,
                data_id="typed-resume-optimizer",
                train_loader=train_loader,
                resume_from_run_id=first.id,
                optim=NNOptimParams(name=Optims.SGD, max_lr=0.01, momentum=0.0, weight_decay=0.0),
                scheduler=step,
            )
        )

    cosine = NNSchedulerParams(
        kind=Schedulers.COSINE_ANNEALING,
        T_max=2,
        factor=0.5,
        min_lr=0.0,
        patience=0,
        cooldown=0,
        threshold=0.0,
    )
    with pytest.raises(ValueError, match="scheduler type mismatch"):
        NNModel(net_params=net_params, params=model_params).train(
            params=NNTrainParams(
                n_epochs=1,
                data_id="typed-resume-scheduler",
                train_loader=train_loader,
                resume_from_run_id=first.id,
                optim=NNOptimParams(name=Optims.ADAM, max_lr=0.01, momentum=(0.9, 0.999), weight_decay=0.0),
                scheduler=cosine,
            )
        )


@pytest.mark.parametrize(("saved_scaler", "current_scaler"), [({"scale": 2.0}, None), (None, object())])
def test_warm_resume_rejects_grad_scaler_presence_changes(tmp_path, monkeypatch, saved_scaler, current_scaler):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    source_params = _train_params(train_loader, None, n_epochs=1)
    first = NNModel(net_params=net_params, params=model_params).train(params=source_params)
    checkpoint = NNCheckpoint.load(first.id, Checkpoints.LAST)
    assert checkpoint is not None and checkpoint.training_state_id is not None
    sidecar = tmp_path / "runs" / first.id / "checkpoints" / f"last.pt.opt.{checkpoint.training_state_id}.pt"
    state = torch.load(sidecar, weights_only=True)
    state["scaler"] = saved_scaler
    torch.save(state, sidecar)

    monkeypatch.setattr(NNModel, "_build_grad_scaler", lambda _self: current_scaler)
    with pytest.raises(ValueError, match="GradScaler presence mismatch"):
        NNModel(net_params=net_params, params=model_params).train(
            params=NNTrainParams(
                n_epochs=1,
                train_loader=train_loader,
                resume_from_run_id=first.id,
                optim=source_params.optim,
                scheduler=source_params.scheduler,
            )
        )


def test_warm_resume_rejects_same_optimizer_with_different_parameter_topology(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    source_params = _train_params(train_loader, None, n_epochs=1)
    first = NNModel(net_params=net_params, params=model_params).train(params=source_params)
    grouped_optim = NNOptimParams(
        name=Optims.ADAM,
        max_lr=source_params.optim.max_lr,
        momentum=source_params.optim.momentum,
        weight_decay=source_params.optim.weight_decay,
        param_groups=[NNParamGroupSpec(name_pattern="*.bias", weight_decay=0.0)],
    )

    with pytest.raises(ValueError, match="optimizer parameter topology"):
        NNModel(net_params=net_params, params=model_params).train(
            params=NNTrainParams(
                n_epochs=1,
                train_loader=train_loader,
                resume_from_run_id=first.id,
                optim=grouped_optim,
                scheduler=source_params.scheduler,
            )
        )


def test_multiple_loader_generators_restore_by_seed_identity():
    dataset = TensorDataset(torch.arange(8))
    loader_generator = torch.Generator().manual_seed(11)
    sampler_generator = torch.Generator().manual_seed(22)
    source = DataLoader(dataset, batch_size=2, shuffle=True, generator=loader_generator)
    source.sampler.generator = sampler_generator
    torch.rand(3, generator=loader_generator)
    torch.rand(5, generator=sampler_generator)
    saved = _capture_rng_state(source)

    resumed = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        generator=torch.Generator().manual_seed(22),
    )
    resumed.sampler.generator = torch.Generator().manual_seed(11)
    _restore_rng_state(saved, resumed)

    assert torch.equal(resumed.generator.get_state(), sampler_generator.get_state())
    assert torch.equal(resumed.sampler.generator.get_state(), loader_generator.get_state())


def test_finite_horizon_scheduler_resume_requires_explicit_total_steps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    params = NNTrainParams(
        n_epochs=1,
        train_loader=train_loader,
        resume_from_run_id="prior-run",
        optim=NNOptimParams(name=Optims.SGD, max_lr=0.1, momentum=0.0, weight_decay=0.0),
        scheduler=NNSchedulerParams(
            kind=Schedulers.ONE_CYCLE,
            max_lr=0.1,
            total_steps=None,
            min_lr=0.0,
            factor=0.5,
            patience=0,
            cooldown=0,
            threshold=0.0,
        ),
    )

    with pytest.raises(ValueError, match="requires scheduler.total_steps"):
        NNModel(net_params=net_params, params=model_params).train(params=params)


def test_finite_horizon_resume_rejects_insufficient_explicit_horizon(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    scheduler = NNSchedulerParams(
        kind=Schedulers.ONE_CYCLE,
        max_lr=0.1,
        total_steps=2,
        min_lr=0.0,
        factor=0.5,
        patience=0,
        cooldown=0,
        threshold=0.0,
    )
    first = NNModel(net_params=net_params, params=model_params).train(
        params=NNTrainParams(
            n_epochs=1,
            data_id="finite-first",
            train_loader=train_loader,
            optim=NNOptimParams(name=Optims.SGD, max_lr=0.1, momentum=0.0, weight_decay=0.0),
            scheduler=scheduler,
        )
    )

    candidate = NNModel(net_params=net_params, params=model_params)
    before = {name: tensor.detach().clone() for name, tensor in candidate.net.state_dict().items()}
    with pytest.raises(ValueError, match="beyond scheduler.total_steps=2"):
        candidate.train(
            params=NNTrainParams(
                n_epochs=2,
                data_id="finite-resume",
                train_loader=train_loader,
                resume_from_run_id=first.id,
                optim=NNOptimParams(name=Optims.SGD, max_lr=0.1, momentum=0.0, weight_decay=0.0),
                scheduler=scheduler,
            )
        )
    for name, tensor in candidate.net.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_setup_failure_releases_empty_run_reservation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    params = NNTrainParams(
        n_epochs=2,
        data_id="invalid-scheduler",
        train_loader=train_loader,
        optim=NNOptimParams(name=Optims.SGD, max_lr=0.1, momentum=0.0, weight_decay=0.0),
        scheduler=NNSchedulerParams(
            kind=Schedulers.ONE_CYCLE,
            max_lr=0.1,
            total_steps=1,
            min_lr=0.0,
            factor=0.5,
            patience=0,
            cooldown=0,
            threshold=0.0,
        ),
    )

    for _ in range(2):
        with pytest.raises(ValueError, match="total_steps"):
            NNModel(net_params=net_params, params=model_params).train(params=params)


def test_concurrent_checkpoint_writers_commit_matching_state_pair(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    model = NNModel(net_params=net_params, params=model_params)
    run = model.train(params=_train_params(train_loader, None, n_epochs=1))
    checkpoint = NNCheckpoint.load(run.id, Checkpoints.LAST)
    assert checkpoint is not None

    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        futures = [pool.submit(_save_checkpoint_in_process, checkpoint, str(tmp_path), marker) for marker in (11, 22)]
        for future in futures:
            future.result(timeout=90)

    saved = NNCheckpoint.load("concurrent-run", Checkpoints.LAST, root=str(tmp_path))
    state = NNCheckpoint.load_training_state("concurrent-run", Checkpoints.LAST, root=str(tmp_path))
    assert saved is not None
    assert state is not None
    assert saved.training_state_id == state["checkpoint_id"]
    assert state["completed_epoch"] in {11, 22}
    assert state["optimizer"]["param_groups"][0]["marker"] == state["completed_epoch"]


def test_concurrent_identical_runs_have_single_admission_winner(tmp_path):
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        results = list(pool.map(_reserve_run_in_process, [str(tmp_path), str(tmp_path)]))

    assert sorted(results) == ["blocked", "reserved"]


def test_overwrite_waits_for_active_run_lease(tmp_path):
    net_params, model_params = _make_params()
    params = NNTrainParams(n_epochs=1, data_id="active-overwrite")
    active = NNRun(net=net_params, model=model_params, train=params)
    replacement = NNRun(net=net_params, model=model_params, train=params)
    active_entered = threading.Event()
    allow_active_to_finish = threading.Event()
    replacement_entered = threading.Event()

    def hold_active_run() -> None:
        with active.writable_lease(root=str(tmp_path)):
            active_entered.set()
            assert allow_active_to_finish.wait(timeout=5)

    def overwrite_run() -> None:
        assert active_entered.wait(timeout=5)
        with replacement.writable_lease(root=str(tmp_path), overwrite=True):
            replacement_entered.set()

    active_thread = threading.Thread(target=hold_active_run)
    replacement_thread = threading.Thread(target=overwrite_run)
    active_thread.start()
    replacement_thread.start()

    assert active_entered.wait(timeout=5)
    assert not replacement_entered.wait(timeout=0.1)
    allow_active_to_finish.set()
    active_thread.join(timeout=5)
    replacement_thread.join(timeout=5)

    assert not active_thread.is_alive()
    assert not replacement_thread.is_alive()
    assert replacement_entered.is_set()


def test_train_rejects_none_or_invalid_params():
    """The first guard in NNModel.train() — params=None or an invalid
    optim config — must raise ValueError loudly rather than letting the
    loop start and produce a garbage run. Pre-audit, this branch had
    zero test coverage."""
    import pytest

    from nnx.nn.params.nn_train_params import NNTrainParams

    net_params, model_params = _make_params()
    model = NNModel(net_params=net_params, params=model_params)

    # 1. None params — surfaces a distinct error from the invalid-optim case.
    with pytest.raises(ValueError, match="^train params must be non-None$"):
        model.train(params=None)

    # 2. invalid optim: Adam with a scalar momentum (Adam wants a tuple).
    train_loader, _ = _make_tiny_loaders()
    bad_optim = NNOptimParams(
        name=Optims.ADAM,
        max_lr=1e-3,
        momentum=0.9,
        weight_decay=0.0,
    )
    bad_params = NNTrainParams(
        n_epochs=1,
        train_loader=train_loader,
        optim=bad_optim,
        scheduler=NNSchedulerParams(
            min_lr=1e-7,
            factor=0.5,
            patience=1,
            cooldown=1,
            threshold=1e-3,
        ),
    )
    with pytest.raises(ValueError, match=r"^train params has an invalid optim config:"):
        model.train(params=bad_params)


def test_train_rejects_none_train_loader():
    """params.train_loader=None (the dataclass default — "wire later via
    with_train_loader") must fail fast with an actionable ValueError.
    Pre-fix, train() printed the run-details table and then crashed in
    the epoch loop with a raw `TypeError: 'NoneType' object is not
    iterable`."""
    import pytest

    from nnx.nn.params.nn_train_params import NNTrainParams

    net_params, model_params = _make_params()
    model = NNModel(net_params=net_params, params=model_params)
    params = NNTrainParams(
        n_epochs=1,
        optim=NNOptimParams(
            name=Optims.ADAM,
            max_lr=1e-3,
            momentum=(0.9, 0.999),
            weight_decay=0.0,
        ),
    )
    with pytest.raises(ValueError, match="train_loader is required"):
        model.train(params=params)


def test_run_checkpoints_slots_and_best_exclusion(tmp_path, monkeypatch):
    """NNRun.checkpoints() contract: five cadence slots in order (FIRST,
    Q1, Q2, Q3, LAST); None where the tag was never written (a 1-epoch
    run writes only FIRST and LAST — see phase_tag's small-n_epochs
    caveat); BEST deliberately excluded as a duplicate pointer."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")

    train_loader, val_loader = _make_tiny_loaders()
    net_params, model_params = _make_params()
    model = NNModel(net_params=net_params, params=model_params)
    run = model.train(params=_train_params(train_loader, val_loader, n_epochs=1))

    ckpts = run.checkpoints()
    assert len(ckpts) == 5
    assert ckpts[0] is not None, "FIRST should exist"
    assert ckpts[4] is not None, "LAST should exist"
    assert ckpts[1] is None and ckpts[2] is None and ckpts[3] is None, "Q1-Q3 unwritten for n_epochs=1"


def test_train_rejects_empty_train_loader(tmp_path, monkeypatch):
    """A loader that yields zero batches (dataset smaller than
    batch_size with drop_last=True) must fail fast with an actionable
    error. Pre-fix the first epoch crashed on a bare IndexError at
    idps[-1] — and a later zero-batch epoch would have silently
    attached its val_edp to the previous epoch's last idp."""
    import pytest
    from torch.utils.data import DataLoader, TensorDataset

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")

    X = torch.randn(4, 8)
    y = torch.randint(0, 3, (4,))
    empty_loader = DataLoader(TensorDataset(X, y), batch_size=8, drop_last=True)
    assert len(empty_loader) == 0

    net_params, model_params = _make_params()
    model = NNModel(net_params=net_params, params=model_params)
    with pytest.raises(ValueError, match="yielded no batches"):
        model.train(params=_train_params(empty_loader, None, n_epochs=1))


def test_best_err_falls_back_to_loss_for_paradigm_runs():
    """BEST tracking for paradigm runs (diffusion / SimCLR / DPO / GAN)
    whose custom steps leave `.error` unset: _best_err must fall back to
    `.loss` via the shared resolver. Pre-fix every such checkpoint
    scored +inf, `inf < inf` is False, and runs/best stayed frozen on
    whichever run saved first."""
    from nnx import NNCheckpoint, NNEvaluationDataPoint, NNIterationDataPoint
    from nnx.nn.params.nn_run import _best_err

    net_params, model_params = _make_params()
    model = NNModel(net_params=net_params, params=model_params)
    loss_only_edp = NNEvaluationDataPoint(accuracy=0.0, f1=0.0, recall=0.0, precision=0.0, loss=0.42)
    ckpt = NNCheckpoint(
        idp=NNIterationDataPoint(lr=1e-3, iter_idx=0, epoch_idx=0, batch_idx=0, train_edp=loss_only_edp),
        model_params=model.params,
        net_params=model.net_params,
        net_state=model.net.state_dict(),
    )
    assert _best_err(ckpt) == 0.42
    assert _best_err(None) == float("inf")


def test_nn_run_save_tolerates_default_none_idps(tmp_path):
    """NNRun(...).save() with the dataclass-default idps=None must write
    an empty idps.csv instead of raising TypeError, and load back with
    zero idps."""
    from nnx.nn.params.nn_run import NNRun

    net_params, model_params = _make_params()
    run = NNRun(
        net=net_params,
        train=_train_params(None, None, n_epochs=1),
        model=model_params,
    )
    run.save(root=str(tmp_path))
    loaded = NNRun.load(run.id, root=str(tmp_path))
    assert loaded.idps == []
    assert loaded.id == run.id


def test_train_refuses_accidental_run_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, val_loader = _make_tiny_loaders()
    params = _train_params(train_loader, val_loader, n_epochs=1)
    net_params, model_params = _make_params()
    NNModel(net_params=net_params, params=model_params).train(params=params)

    with pytest.raises(FileExistsError, match="overwrite_existing"):
        NNModel(net_params=net_params, params=model_params).train(params=params)


def test_train_overwrite_removes_stale_run_artifacts(tmp_path, monkeypatch):
    from dataclasses import replace

    monkeypatch.chdir(tmp_path)
    train_loader, val_loader = _make_tiny_loaders()
    params = _train_params(train_loader, val_loader, n_epochs=1)
    net_params, model_params = _make_params()
    first = NNModel(net_params=net_params, params=model_params).train(params=params)
    stale = tmp_path / "runs" / first.id / "checkpoints" / "stale.pt"
    stale.write_bytes(b"old run")

    replacement = NNModel(net_params=net_params, params=model_params).train(
        params=replace(params, overwrite_existing=True)
    )

    assert replacement.id == first.id
    assert not stale.exists()


def test_train_rejects_fully_frozen_model():
    """freeze('*') then train() previously died mid-loop with torch's
    raw 'element 0 of tensors does not require grad' — the boundary now
    names the actual cause."""
    import pytest

    from nnx.finetune import freeze

    net_params, model_params = _make_params()
    model = NNModel(net_params=net_params, params=model_params)
    freeze(model.net, "*")
    train_loader, _ = _make_tiny_loaders()
    with pytest.raises(ValueError, match="no trainable parameters"):
        model.train(params=_train_params(train_loader, None, n_epochs=1))


def test_nn_run_load_names_the_corrupt_file(tmp_path, monkeypatch):
    """Malformed-artifact errors must point at the file that's actually
    broken: a dropped idps.csv column must not be blamed on run.yaml."""
    import pytest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    model = NNModel(net_params=net_params, params=model_params)
    run = model.train(params=_train_params(train_loader, None, n_epochs=1))
    run_dir = tmp_path / "runs" / run.id

    # Drop a CSV column → error names idps.csv.
    import pandas as pd

    from nnx.nn.params.nn_run import NNRun

    csv_path = run_dir / "idps.csv"
    df = pd.read_csv(csv_path)
    df.drop(columns=["lr"]).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="idps.csv"):
        NNRun.load(run.id)
    df.to_csv(csv_path, index=False)  # restore

    # Zero-byte idps.csv (external truncation — our own writes always
    # emit at least the frame header) → error names idps.csv instead of
    # pandas' context-free EmptyDataError.
    csv_text = csv_path.read_text(encoding="utf-8")
    csv_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="idps.csv"):
        NNRun.load(run.id)
    csv_path.write_text(csv_text, encoding="utf-8")  # restore

    # Drop a run.yaml key → error names run.yaml.
    import yaml

    yaml_path = run_dir / "run.yaml"
    rep = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    del rep["net"]
    yaml_path.write_text(yaml.safe_dump(rep, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="run.yaml"):
        NNRun.load(run.id)


def test_nn_run_load_rejects_empty_run_yaml(tmp_path, monkeypatch):
    """An empty / truncated-to-zero run.yaml safe_loads to None — the
    error must name the file instead of a bare AttributeError."""
    import pytest

    from nnx.nn.params.nn_run import NNRun

    monkeypatch.chdir(tmp_path)
    run_id = "a" * 32
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.yaml").write_text("", encoding="utf-8")
    (run_dir / "idps.csv").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a mapping"):
        NNRun.load(run_id)


def test_nn_run_load_truncates_history_beyond_last_commit(tmp_path, monkeypatch):
    from dataclasses import replace

    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    run = NNModel(net_params=net_params, params=model_params).train(
        params=_train_params(train_loader, None, n_epochs=1)
    )
    assert run.idps
    speculative = replace(run.idps[-1], epoch_idx=99, iter_idx=999)
    run.with_idps([*run.idps, speculative]).save(update_best=False)

    loaded = NNRun.load(run.id)
    assert loaded.idps
    assert max(idp.epoch_idx for idp in loaded.idps) == 0
    assert all(idp.iter_idx != 999 for idp in loaded.idps)


def test_nn_run_load_rejects_empty_last_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_loader, _ = _make_tiny_loaders()
    net_params, model_params = _make_params()
    run = NNModel(net_params=net_params, params=model_params).train(
        params=_train_params(train_loader, None, n_epochs=1)
    )
    last = tmp_path / "runs" / run.id / "checkpoints" / "last.pt"
    last.write_bytes(b"")

    with pytest.raises(ValueError, match="malformed LAST checkpoint"):
        NNRun.load(run.id)
