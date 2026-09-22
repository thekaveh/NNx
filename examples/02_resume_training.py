"""Warm-resume training from a prior run.

Round 1 trains for 3 epochs; round 2 picks up from the LAST checkpoint
of round 1 and trains another 4 epochs, with Adam momentum / first /
second-moment buffers, scheduler, scaler, epoch progress, and RNG state
preserved across the boundary. Resume lineage participates in the new run's
content-addressed identity, so the original run remains intact.

Checking for saved state is *observational*: ``NNCheckpoint.load_training_state``
/ ``load_optimizer_state`` / ``load_with_training_state`` return their absent
result for a run that has never been written without creating ``runs/<id>/``,
so a preflight probe never blocks the first fit. ``checkpoint_probe_first_fit``
below is a bounded, self-checking demonstration of that contract.

Overwriting a run (``overwrite_existing=True``) deletes its artifacts and reuses
its content-addressed ID. If that run owned ``runs/best``, NNx re-elects the best
surviving committed run before the old artifacts disappear; the replacement only
becomes a candidate again once it has committed its own BEST checkpoint. While no
committed winner exists, ``runs/best`` is absent. ``overwrite_best_recovery``
below demonstrates this in a fresh temporary root — it is a teaching fixture,
not a recommendation to enable overwrite in ordinary workflows.

Warm resume accepts every batch source ordinary training accepts — a
``DataLoader``, a plain re-iterable list of ``(X, Y)`` batches, or the
one-element full-batch list ``NNGraphDataset(sampler="full")`` produces.
``iterable_graph_resume`` below demonstrates both (the graph branch needs
the optional ``thekaveh-nnx[graph]`` extra and is skipped without it).

Mixed precision (``NNModelParams(mixed_precision=True)``) only activates on
CUDA, where the loop owns a ``torch.amp.GradScaler("cuda")`` whose state
rides along in the training-state sidecar. ``amp_resume_compatibility``
below fits and resumes with AMP requested on CPU (no scaler is built or
saved) and, only on a CUDA host, with AMP actually enabled.

Run:
    python examples/02_resume_training.py
"""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Checkpoints,
    Devices,
    Losses,
    Nets,
    NNCheckpoint,
    NNEvaluationDataPoint,
    NNIterationDataPoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNRun,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    set_seed,
)
from nnx.nn.params.nn_run import _read_best_pointer


def _base_optim() -> NNOptimParams:
    return NNOptimParams(
        name=Optims.ADAM,
        max_lr=1e-2,
        momentum=(0.9, 0.999),
        weight_decay=0.0,
    )


def _base_sched() -> NNSchedulerParams:
    return NNSchedulerParams(
        min_lr=1e-7,
        factor=0.5,
        patience=2,
        cooldown=1,
        threshold=1e-3,
    )


def _make_model_and_loader():
    # No set_seed here — the caller does set_seed(...) in main() before
    # each call, per the [[examples-seed-helper-override]] convention.
    # Centralizing seed management in main() makes the reproducibility
    # contract visible at the entry point and avoids hidden re-seeding
    # inside helpers.
    X = torch.randn(128, 8)
    y = torch.randint(0, 3, (128,))
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    net_params = NNParams(
        input_dim=8,
        output_dim=3,
        hidden_dims=[16],
        dropout_prob=0.0,
        activation=Activations.RELU,
    )
    model_params = NNModelParams(
        net=Nets.FEED_FWD,
        device=Devices.CPU,
        loss=Losses.CROSS_ENTROPY,
    )
    return NNModel(net_params=net_params, params=model_params), loader


def checkpoint_probe_first_fit() -> dict:
    """Bounded demonstration that probing absent training state does not
    reserve a run (writes ``runs/`` under the current working directory;
    the smoke test runs it in a temporary one).

    Derives the prospective run ID from the same params the fit will use,
    probes all three state readers, asserts both the absent return values
    and that no ``runs/<id>/`` directory appeared, then performs the first
    fit with ``overwrite_existing`` left at its default (``False``) and
    reloads LAST together with its training state.
    """
    set_seed(7)
    model, loader = _make_model_and_loader()
    params = NNTrainParams(n_epochs=1, train_loader=loader, optim=_base_optim(), scheduler=_base_sched())
    prospective_id = NNRun(net=model.net_params, train=params, model=model.params).id
    run_dir = os.path.join("runs", prospective_id)

    assert NNCheckpoint.load_training_state(run=prospective_id, type=Checkpoints.LAST) is None
    assert NNCheckpoint.load_optimizer_state(run=prospective_id, type=Checkpoints.LAST) is None
    assert NNCheckpoint.load_with_training_state(run=prospective_id, type=Checkpoints.LAST) == (None, None)
    assert not os.path.exists(run_dir), f"probe must not reserve {run_dir}"

    run = model.train(params=params)  # overwrite_existing stays False
    assert run.id == prospective_id
    last, training_state = NNCheckpoint.load_with_training_state(run=run.id, type=Checkpoints.LAST)
    assert last is not None and training_state is not None
    assert last.idp.epoch_idx == 0 and training_state["completed_epoch"] == 0

    summary = {"run_id": run.id, "probe_reserved_run": False, "last_epoch": last.idp.epoch_idx}
    print(f"probe-then-first-fit workflow: {summary}")
    return summary


def _commit_scored_run(model: NNModel, salt: str, error: float) -> NNRun:
    """Persist a tiny *committed* run (history + LAST + BEST) whose BEST
    scores `error`, using the model's current weights."""
    edp = NNEvaluationDataPoint(f1=0.0, recall=0.0, accuracy=0.0, precision=0.0, loss=1.0, error=error)
    idp = NNIterationDataPoint(lr=0.01, iter_idx=0, epoch_idx=0, batch_idx=0, train_edp=edp)
    run = NNRun(net=model.net_params, model=model.params, train=NNTrainParams(n_epochs=1), salt=salt, idps=[idp])
    checkpoint = NNCheckpoint(
        net_params=model.net_params, model_params=model.params, net_state=model.net.state_dict(), idp=idp
    )
    checkpoint.save(run=run.id, type=Checkpoints.LAST)
    checkpoint.save(run=run.id, type=Checkpoints.BEST)
    return run.save()


def overwrite_best_recovery() -> dict:
    """Bounded demonstration of `runs/best` re-election after the winner is
    overwritten (writes ``runs/`` under the current working directory; the
    smoke test runs it in a temporary one — keep it out of real roots).

    Commits run A (error 0.1, the winner) and run B (error 0.2) from two
    differently initialised models, then overwrites A with a *worse*
    result (0.9). The pointer must move to B — the best surviving
    committed run — and B's BEST must load for inference with the same
    predictions as the model that produced it.
    """
    set_seed(7)
    model_a, _ = _make_model_and_loader()
    model_b, _ = _make_model_and_loader()
    run_a = _commit_scored_run(model_a, "A", 0.1)
    run_b = _commit_scored_run(model_b, "B", 0.2)
    best = os.path.join("runs", "best")
    assert _read_best_pointer(best) == run_a.id

    with run_a.writable_lease(overwrite=True):
        # The winner's artifacts are gone; B was elected before they went.
        assert _read_best_pointer(best) == run_b.id
        _commit_scored_run(model_a, "A", 0.9)
    assert _read_best_pointer(best) == run_b.id, "worse replacement must not reclaim best"

    winner_id = _read_best_pointer(best)
    assert winner_id is not None
    checkpoint = NNCheckpoint.load(run=winner_id, type=Checkpoints.BEST)
    assert checkpoint is not None
    X = torch.randn(4, 8)
    restored = NNModel.from_checkpoint(checkpoint=checkpoint).predict(X).logits
    assert (restored == model_b.predict(X).logits).all()

    summary = {"winner": winner_id, "winner_is_survivor": winner_id == run_b.id}
    print(f"overwrite-best recovery workflow: {summary}")
    return summary


def iterable_graph_resume() -> dict:
    """Bounded demonstration of warm resume over non-DataLoader batch
    sources (writes ``runs/`` under the current working directory; the
    smoke test runs it in a temporary one).

    1. A reusable list of two ``(X, Y)`` batches: one saved epoch, one
       resumed epoch; the resumed run starts at epoch 1.
    2. A tiny in-memory five-node graph through ``NNGraphDataset(sampler="full")``
       and ``Nets.GRAPH_CONV`` (no downloads, no pyg-lib / torch-sparse):
       one saved + one resumed epoch, and evaluation still scores only the
       split's seed node (accuracy is exactly 0.0 or 1.0 for a one-node
       validation split). Skipped when ``torch_geometric`` is missing.
    """
    set_seed(7)
    summary: dict = {}

    batches = [(torch.randn(4, 8), torch.tensor([0, 1, 2, 1])), (torch.randn(4, 8), torch.tensor([2, 0, 1, 0]))]
    model_a, _ = _make_model_and_loader()
    first = model_a.train(params=NNTrainParams(n_epochs=1, data_id="list", train_loader=batches, optim=_base_optim()))
    model_b, _ = _make_model_and_loader()
    resumed = model_b.train(
        params=NNTrainParams(
            n_epochs=1, data_id="list", train_loader=batches, optim=_base_optim(), resume_from_run_id=first.id
        )
    )
    assert resumed.idps[0].epoch_idx == first.idps[-1].epoch_idx + 1 == 1
    summary["list_resumed_epoch"] = resumed.idps[0].epoch_idx

    try:
        from torch_geometric.data import Data
    except ImportError:  # pragma: no cover - exercised only on core-only installs
        summary["graph"] = "skipped (install thekaveh-nnx[graph])"
        print(f"iterable/graph resume workflow: {summary}")
        return summary

    from nnx import NNGraphDataset

    class _TinyGraph:
        """Five-node directed cycle with train nodes {0, 2}, val node {1}."""

        num_features = 3
        num_classes = 2

        def __init__(self, root, transform=None):
            n = 5
            g = torch.Generator().manual_seed(7)
            train_mask = torch.zeros(n, dtype=torch.bool)
            train_mask[[0, 2]] = True
            val_mask = torch.zeros(n, dtype=torch.bool)
            val_mask[[1]] = True
            test_mask = torch.zeros(n, dtype=torch.bool)
            test_mask[[3, 4]] = True
            self._data = Data(
                x=torch.randn(n, self.num_features, generator=g),
                edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long),
                y=torch.tensor([0, 1, 0, 1, 0], dtype=torch.long),
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
            )

        def __getitem__(self, idx):
            return self._data

    ds = NNGraphDataset(ds_class=_TinyGraph, sampler="full")

    def _graph_model() -> NNModel:
        return NNModel(
            net_params=NNParams(
                dropout_prob=0.0,
                activation=Activations.RELU,
                input_dim=ds.input_dim,
                output_dim=ds.output_dim,
                hidden_dims=[8],
            ),
            params=NNModelParams(net=Nets.GRAPH_CONV, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        )

    graph_first = _graph_model().train(
        params=NNTrainParams(n_epochs=1, train_loader=ds.train_loader, val_loader=ds.val_loader, optim=_base_optim())
    )
    graph_model = _graph_model()
    graph_resumed = graph_model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=ds.train_loader,
            val_loader=ds.val_loader,
            optim=_base_optim(),
            resume_from_run_id=graph_first.id,
        )
    )
    assert graph_resumed.idps[0].epoch_idx == 1
    val_edp = graph_model.evaluate(loader=ds.val_loader)
    assert val_edp.accuracy in (0.0, 1.0), "only the single seed node may be scored"
    summary["graph_resumed_epoch"] = graph_resumed.idps[0].epoch_idx
    summary["graph_val_accuracy"] = val_edp.accuracy
    print(f"iterable/graph resume workflow: {summary}")
    return summary


def amp_resume_compatibility() -> dict:
    """Bounded demonstration of mixed-precision warm resume (writes ``runs/``
    under the current working directory; the smoke test runs it in a
    temporary one).

    ``NNModelParams(mixed_precision=True)`` only activates autocast + a
    ``torch.amp.GradScaler("cuda")`` on CUDA — the scaler factory needs
    PyTorch >= 2.3, NNx's declared floor. Everywhere else the same
    configuration trains scaler-free: the training-state sidecar records
    ``scaler=None`` and a resume restores nothing for it. On a CUDA host
    the helper additionally fits one epoch with AMP enabled, resumes for
    one more, and checks the restored scaler state (scale factor and
    growth tracker) — reported as skipped otherwise, never faked.
    """
    set_seed(11)
    summary: dict = {}

    def _fit_and_resume(device: Devices, tag: str) -> tuple[NNRun, NNRun, dict, dict]:
        X = torch.randn(64, 8)
        y = torch.randint(0, 3, (64,))
        loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)
        net_params = NNParams(
            input_dim=8, output_dim=3, hidden_dims=[16], dropout_prob=0.0, activation=Activations.RELU
        )
        model_params = NNModelParams(net=Nets.FEED_FWD, device=device, loss=Losses.CROSS_ENTROPY, mixed_precision=True)
        first = NNModel(net_params=net_params, params=model_params).train(
            params=NNTrainParams(
                n_epochs=1, data_id=tag, train_loader=loader, optim=_base_optim(), scheduler=_base_sched()
            )
        )
        first_state = NNCheckpoint.load_training_state(run=first.id, type=Checkpoints.LAST)
        resumed = NNModel(net_params=net_params, params=model_params).train(
            params=NNTrainParams(
                n_epochs=1,
                data_id=tag,
                train_loader=loader,
                optim=_base_optim(),
                scheduler=_base_sched(),
                resume_from_run_id=first.id,
            )
        )
        resumed_state = NNCheckpoint.load_training_state(run=resumed.id, type=Checkpoints.LAST)
        assert first_state is not None and resumed_state is not None
        assert resumed.idps[0].epoch_idx == 1
        return first, resumed, first_state, resumed_state

    # CPU: AMP is a documented no-op — no scaler is built or persisted.
    _, resumed, first_state, resumed_state = _fit_and_resume(Devices.CPU, "amp-cpu")
    assert first_state["scaler"] is None and resumed_state["scaler"] is None
    summary["cpu"] = {"scaler_state": None, "resumed_epoch": resumed.idps[0].epoch_idx}

    if torch.cuda.is_available():
        _, resumed, first_state, resumed_state = _fit_and_resume(Devices.CUDA, "amp-cuda")
        assert first_state["scaler"] is not None and resumed_state["scaler"] is not None
        # The resumed run continues from the saved scale rather than the
        # factory default (65536.0 → 2**16); growth tracking carries on.
        summary["cuda"] = {
            "saved_scale": first_state["scaler"]["scale"],
            "resumed_scale": resumed_state["scaler"]["scale"],
            "resumed_growth_tracker": resumed_state["scaler"]["_growth_tracker"],
            "resumed_epoch": resumed.idps[0].epoch_idx,
        }
    else:
        summary["cuda"] = "skipped (no CUDA device): enabled-AMP resume not exercised here"
    print(f"amp resume compatibility: {summary}")
    return summary


def main():
    base_optim = _base_optim()
    base_sched = _base_sched()

    # Round 1: train from scratch. Seed pinned so the random model
    # init + DataLoader shuffle order are reproducible.
    set_seed(7)
    model_a, loader = _make_model_and_loader()
    run_a = model_a.train(
        params=NNTrainParams(
            n_epochs=4,
            train_loader=loader,
            optim=base_optim,
            scheduler=base_sched,
        )
    )
    print(f"\nRound 1 done. run.id = {run_a.id}, {len(run_a.idps)} iterations")

    # Round 2: build a NEW model (random weights) and resume from round 1's LAST.
    # Re-seeding is harmless but not required for continuity: loading the
    # training-state bundle restores the RNG after model construction.
    set_seed(7)
    model_b, loader2 = _make_model_and_loader()
    run_b = model_b.train(
        params=NNTrainParams(
            n_epochs=3,
            train_loader=loader2,
            optim=base_optim,
            scheduler=base_sched,
            resume_from_run_id=run_a.id,
            resume_from_checkpoint="last",
        )
    )
    assert run_b.id != run_a.id
    print(f"Round 2 done. run.id = {run_b.id}, {len(run_b.idps)} iterations")
    print("Round 2 continued from round 1's complete LAST training state.")
    print(f"Final round 2 train loss: {run_b.idps[-1].train_edp.loss:.4f}")


if __name__ == "__main__":
    main()
