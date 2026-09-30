"""Bounded training history with an append journal (FEAT-036).

By default ``train()`` keeps every per-batch record in memory and rewrites
``runs/<id>/idps.csv`` each epoch. ``history=HistoryJournal(...)`` keeps a
bounded window instead and appends each record once to
``runs/<id>/history/``:

  1. **Train with a journal.** Three epochs of five batches, ``retention=3``
     and ``chunk_size=2``: ``ctx.idps`` and the returned ``NNRun.idps`` hold
     the last three records, while the journal holds all fifteen.
     ``idps.csv`` is not written.
  2. **Resume lazily.** A continuation reads the source run's LAST
     checkpoint, never its history, and writes its own run directory and
     chunk files. The source run is left byte-for-byte unchanged.
  3. **Render the summary.** The notebook view (``_repr_html_``) charts the
     per-epoch rows the journal wrote; ``NNRun.load`` reads the committed
     tail only.
  4. **Export CSV.** ``export_history_csv`` writes the legacy ``idps.csv``
     layout; ``lineage=True`` puts the parent's committed prefix before the
     child's records, each epoch once.

Fully offline, CPU only.

Run:
    python examples/history_journal.py

The bounded ``history_journal_workflow()`` helper is executed by
``tests/test_examples_smoke.py -k history_journal`` in a temporary working
directory.
"""

from __future__ import annotations

import hashlib
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNRun,
    NNTrainParams,
    Optims,
)
from nnx.history import HistoryJournal, export_history_csv, iter_history


def _model() -> NNModel:
    torch.manual_seed(0)
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def _params(n_epochs: int, **lineage) -> NNTrainParams:
    generator = torch.Generator().manual_seed(1)
    features = torch.randn(20, 4, generator=generator)
    labels = torch.randint(0, 2, (20,), generator=generator)
    loader = DataLoader(TensorDataset(features, labels), batch_size=4)  # five batches
    return NNTrainParams(
        n_epochs=n_epochs,
        seed=0,
        train_loader=loader,
        val_loader=loader,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
        **lineage,
    )


def _digest(path: str) -> str:
    digest = hashlib.sha256()
    for directory, _, files in sorted(os.walk(path)):
        for name in sorted(files):
            if not name.endswith(".lock"):
                with open(os.path.join(directory, name), "rb") as handle:
                    digest.update(name.encode() + handle.read())
    return digest.hexdigest()


def history_journal_workflow() -> dict:
    journal = HistoryJournal(retention=3, chunk_size=2)

    # 1. Train with a journal: a bounded window in memory, every record on disk.
    parent = _model().train(_params(3), history=journal)
    assert parent.idps is not None and len(parent.idps) == 3
    assert [(r.epoch_idx, r.batch_idx) for r in parent.idps] == [(2, 2), (2, 3), (2, 4)]
    assert sum(1 for _ in iter_history(parent.id)) == 15
    assert not os.path.exists(os.path.join("runs", parent.id, "idps.csv"))
    source = os.path.join("runs", parent.id)
    before = _digest(source)

    # 2. Resume lazily: the continuation owns its files; the source is untouched.
    child = _model().train(_params(2, resume_from_run_id=parent.id), history=journal)
    assert _digest(source) == before
    assert child.id != parent.id and {r.epoch_idx for r in iter_history(child.id)} == {3, 4}

    # 3. The summary reads per-epoch rows; load reads the committed tail.
    loaded = NNRun.load(child.id)
    assert loaded.idps is not None and len(loaded.idps) == 3
    assert "<table" in loaded._repr_html_()
    epochs = loaded._epoch_series()["epochs"]
    assert epochs == [3, 4], epochs

    # 4. Export the legacy CSV layout, with and without the parent's prefix.
    own = export_history_csv(child.id, "child.csv")
    whole = export_history_csv(child.id, "lineage.csv", lineage=True)
    frame = pd.read_csv("lineage.csv")
    assert (own, whole) == (10, 25)
    assert list(frame["epoch_idx"]) == sorted(frame["epoch_idx"]) and frame["epoch_idx"].nunique() == 5

    summary = {"window": len(parent.idps), "records": whole, "epochs": epochs, "source_unchanged": True}
    print(f"window kept in memory: {summary['window']} of 15 records; idps.csv not written")
    print(f"continuation {child.id[:8]}… resumed from {parent.id[:8]}… (source unchanged)")
    print(f"summary chart epochs: {epochs}")
    print(f"exported {own} own records, {whole} with the parent's committed prefix")
    return summary


def main() -> None:
    history_journal_workflow()


if __name__ == "__main__":
    main()
