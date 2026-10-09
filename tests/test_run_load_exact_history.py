"""``NNRun.load`` reads a run's ``idps.csv`` history back exactly.

``to_csv`` writes each float's shortest round-trip repr, but pandas' default
float parser misreads roughly one value in five by one ULP; a loaded run's
losses, learning rates and metrics then differed from the trained run's
(``tests/test_trainer.py``'s round-trip assertion failed on a CI leg). The
history is now parsed round-trip, as ``migrate_history`` already did."""

from __future__ import annotations

import pandas as pd
import torch

from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNParams, NNTrainParams
from nnx.nn.params.nn_run import NNRun


def _floats(state, prefix=""):
    found = {}
    for key, value in state.items():
        if isinstance(value, dict):
            found.update(_floats(value, f"{prefix}{key}."))
        elif isinstance(value, float):
            found[prefix + key] = value
    return found


def test_a_loaded_runs_history_equals_the_trained_runs_bit_for_bit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    torch.manual_seed(0)
    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=3, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    data = torch.Generator().manual_seed(1)
    batches = [(torch.randn(8, 4, generator=data), torch.arange(8) % 3) for _ in range(3)]
    run = model.train(params=NNTrainParams(n_epochs=12, train_loader=batches, val_loader=batches[:1]))
    trained = [_floats(idp.state()) for idp in run.idps]
    assert sum(len(values) for values in trained) >= 40  # enough values that the default parser misreads some

    # The default parser does misread some of these values (the bug being guarded).
    csv = tmp_path / "runs" / run.id / "idps.csv"
    default = pd.read_csv(csv)
    exact = pd.read_csv(csv, float_precision="round_trip")
    assert not default.equals(exact)

    loaded = NNRun.load(id=run.id)
    assert [_floats(idp.state()) for idp in loaded.idps] == trained
