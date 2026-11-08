"""Single-node data-parallel training under torchrun.

Launch two processes on one machine (CPU ranks talk over Gloo; with CUDA,
one GPU per process over NCCL)::

    torchrun --standalone --nproc_per_node=2 examples/ddp_supervised.py --device cpu --epochs 2 --output ddp-out

On macOS, a host name that resolves to an unreachable address makes the
rendezvous retry without end; launch over the loopback instead::

    GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 \
        --master_port=29500 examples/ddp_supervised.py --device cpu --epochs 2 --output ddp-out

Six labelled rows with stable ids (0-5) train a small classifier:

  1. ``nnx.distributed.train_loader`` deals each epoch's seeded permutation
     to the ranks (``policy="pad"``: every rank takes the same number of
     steps); ``validation_loader`` shards rows ``rank::world`` unpadded, so
     every validation id is scored exactly once.
  2. ``model.train(params, distributed=DDP())`` — every update equals one
     process training on the ranks' union batch; every rank returns the
     same run id and the same global records.
  3. Only the writer rank (``DDP(writer_rank=0)`` by default) holds the run
     lease and writes ``<output>/runs/<id>/``; the other rank writes nothing
     under ``runs/``.
  4. ``nnx.distributed.shutdown()`` leaves the process group together: a
     bounded teardown barrier, then the group is destroyed.

Each rank prints one JSON report (and writes it to
``<output>/report-rank<r>.json``): its rank, the run id, its validation ids
(together they cover 0-5 exactly once), whether it owns the artifacts (it
is the writer rank NNx recorded in the run's LAST checkpoint, which both
ranks read from the shared output directory) and ``train_rows_accuracy``,
the final epoch's accuracy on the validation pass.
The validation loader shards the same six rows the model trains on, which
suits the sharding demonstration but makes that number a training-row
accuracy, not a held-out one: the labels are row-id parity over random
features, so nothing here can generalize. For a real validation set, give
``validation_loader`` rows the model does not train on.

``--inject-failure`` makes rank 1 raise in epoch 1: torchrun stops the
other rank and the launch exits non-zero within the process group's
timeout — no hang, no orphaned process.

No optional extras. ``tests/test_examples_smoke.py`` runs it under
``torchrun --standalone --nproc_per_node=2`` on CPU, with and without
``--inject-failure``. CUDA claims need a real two-GPU run.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from nnx import (
    Activations,
    Checkpoints,
    Devices,
    Losses,
    Nets,
    NNCheckpoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNTrainParams,
)
from nnx import distributed as nnx_dist
from nnx.nn.callbacks import Callback


class SixRows(torch.utils.data.Dataset):
    """Six rows with stable ids: the id is the index; the label is its parity."""

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(0)
        self.X = torch.randn(6, 4, generator=generator)
        self.y = torch.tensor([row % 2 for row in range(6)])

    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]


class InjectedFailure(Callback):
    """Raise on one rank at the start of an epoch."""

    distributed = "all"

    def __init__(self, rank: int, epoch: int) -> None:
        self.rank, self.epoch = rank, epoch

    def on_epoch_begin(self, ctx) -> None:
        if torch.distributed.get_rank() == self.rank and ctx.epoch == self.epoch:
            raise RuntimeError(f"injected failure on rank {self.rank} in epoch {self.epoch}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NNx single-node DDP under torchrun")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output", default="ddp-out", help="working directory for runs/ (shared by the ranks)")
    parser.add_argument("--inject-failure", action="store_true", help="rank 1 raises in epoch 1")
    args = parser.parse_args()

    rank, world_size = nnx_dist.init_process_group(
        backend="nccl" if args.device == "cuda" else "gloo", timeout_seconds=30
    )
    os.makedirs(args.output, exist_ok=True)
    os.chdir(args.output)

    torch.manual_seed(0)  # the same initial weights on every rank (DDP also broadcasts them)
    device = Devices.CUDA if args.device == "cuda" else Devices.CPU
    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=device, loss=Losses.CROSS_ENTROPY),
    )
    rows = SixRows()
    validation = nnx_dist.validation_loader(rows, batch_size=2)
    callbacks = [InjectedFailure(rank=1, epoch=1)] if args.inject_failure else []
    run = model.train(
        params=NNTrainParams(
            n_epochs=args.epochs,
            train_loader=nnx_dist.train_loader(rows, batch_size=2, seed=0),
            val_loader=validation,
            optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
            overwrite_existing=True,
        ),
        callbacks=callbacks,
        distributed=nnx_dist.DDP(),
    )
    # The writer rank is configurable (DDP(writer_rank=...)): ask the run's
    # LAST checkpoint which rank NNx made the writer, not a hard-coded rank 0.
    writer_rank = NNCheckpoint.load_training_state(run.id, Checkpoints.LAST)["distributed"]["writer_rank"]
    owner = rank == writer_rank
    report = {
        "rank": rank,
        "world_size": world_size,
        "run_id": run.id,
        "validation_ids": validation.ids(),
        "owns_artifacts": owner,
        "artifacts": os.path.join(args.output, "runs", run.id) if owner else None,
        # The validation pass scores the training rows: not a held-out accuracy.
        "train_rows_accuracy": run.idps[-1].val_edp.accuracy if run.idps[-1].val_edp is not None else None,
    }
    # Each rank writes its report to the shared output too: the ranks' stdout
    # streams interleave under torchrun.
    with open(f"report-rank{rank}.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle)
    print(json.dumps(report), flush=True)
    nnx_dist.shutdown()  # leave together: no rank exits while a peer is mid-teardown


if __name__ == "__main__":
    main()
