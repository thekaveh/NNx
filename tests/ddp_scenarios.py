"""Two-process DDP scenarios for tests/test_ddp_adapter.py and
tests/test_ddp_resume.py (FEAT-030).

Launched exactly as users launch NNx under DDP::

    python -m torch.distributed.run --standalone --nproc_per_node=2 tests/ddp_scenarios.py SCENARIO OUT_DIR

Each rank works in OUT_DIR (the shared working directory, so ``runs/`` is
shared like a single node's disk) and saves what it saw to
``OUT_DIR/rank<r>.pt``; the tests compare those with a single-process
reference. Not collected by pytest (no ``test_`` prefix).
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

from nnx import (
    Activations,
    Checkpoints,
    CompileSpec,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNTrainParams,
)
from nnx import distributed as nnx_dist
from nnx.history import HistoryJournal
from nnx.monitors import MonitorSpec
from nnx.nn.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoardCallback
from nnx.nn.params.nn_checkpoint import NNCheckpoint

IGNORE = -100
FEATURES, CLASSES = 4, 3


class Rows(torch.utils.data.Dataset):
    """Deterministic rows: features, a class target, and IGNORE targets where asked."""

    def __init__(self, n: int, *, seed: int, ignore: tuple[int, ...] = (), nan: tuple[int, ...] = ()) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n, FEATURES, generator=generator)
        self.y = torch.randint(0, CLASSES, (n,), generator=generator)
        for row in ignore:
            self.y[row] = IGNORE
        for row in nan:
            self.X[row] = float("nan")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]


def model() -> NNModel:
    torch.manual_seed(0)
    return NNModel(
        net_params=NNParams(
            input_dim=FEATURES, output_dim=CLASSES, hidden_dims=[6], dropout_prob=0.0, activation=Activations.RELU
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def train_set() -> Rows:
    # 11 rows: odd, so the pad policy repeats one; ignored targets fall unevenly on the ranks.
    return Rows(11, seed=1, ignore=(0, 3, 4))


def val_set(n: int = 5) -> Rows:
    return Rows(n, seed=2)


def params(train_loader, val_loader=None, **kwargs) -> NNTrainParams:
    return NNTrainParams(
        n_epochs=kwargs.pop("n_epochs", 2),
        train_loader=train_loader,
        val_loader=val_loader,
        optim=NNOptimParams.builder().sgd(max_lr=0.1).accumulate_grad(kwargs.pop("accumulate", 2)).build(),
        overwrite_existing=True,
        **kwargs,
    )


def summary(run) -> dict:
    return {
        "run_id": run.id,
        "train_loss": [idp.train_edp.loss for idp in run.idps],
        "train_accuracy": [idp.train_edp.accuracy for idp in run.idps],
        "val_loss": [None if idp.val_edp is None else idp.val_edp.loss for idp in run.idps],
        "val_accuracy": [None if idp.val_edp is None else idp.val_edp.accuracy for idp in run.idps],
        "epochs": sorted({idp.epoch_idx for idp in run.idps}),
    }


class Undeclared(Callback):
    pass


class WriterComponent(Callback):
    """A writer-only callback that also carries checkpointed state."""

    distributed = "writer"

    def component_spec(self):
        from nnx.components import ComponentSpec

        return ComponentSpec(name="writer-counter", version=1)

    def component_state(self):
        return {"count": 0}

    def load_component_state(self, state, *, version):
        pass


class EventLog(Callback):
    """Writes a file when constructed, like TensorBoardCallback / WandbCallback."""

    distributed = "unsupported"

    def __init__(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, f"events.{os.getpid()}"), "w", encoding="utf-8") as handle:
            handle.write("constructed\n")


class FailingWriterCallback(Callback):
    distributed = "writer"

    def on_epoch_end(self, ctx) -> None:
        raise OSError("disk full (injected on the writer)")


def batchnorm_model() -> NNModel:
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(FEATURES, 6), torch.nn.BatchNorm1d(6), torch.nn.Linear(6, CLASSES))
    return NNModel(module=net, params=NNModelParams(device=Devices.CPU, loss=Losses.CROSS_ENTROPY))


def scenario(name: str, rank: int, world_size: int) -> dict:
    out: dict = {"rank": rank, "world_size": world_size}
    if name == "parity":
        m = model()
        validation = nnx_dist.validation_loader(val_set(), batch_size=2)
        run = m.train(
            params=params(nnx_dist.train_loader(train_set(), batch_size=2, seed=3), validation),
            distributed=nnx_dist.DDP(),
        )
        out.update(summary(run))
        out["state"] = {k: v.clone() for k, v in m.net.state_dict().items()}
        out["validation_ids"] = validation.ids()
        out["last_keys"] = sorted(NNCheckpoint.load(run.id, Checkpoints.LAST).net_state)
    elif name == "local_equivalent":
        # world 1 under torchrun vs. the local path on the same partition
        m = model()
        run = m.train(
            params=params(
                nnx_dist.train_loader(train_set(), batch_size=2, seed=3),
                nnx_dist.validation_loader(val_set(), batch_size=2),
            ),
            distributed=nnx_dist.DDP(),
        )
        out.update(summary(run))
        out["state"] = {k: v.clone() for k, v in m.net.state_dict().items()}
    elif name == "nonfinite":
        data = Rows(8, seed=1, nan=(5,))  # one rank's batch holds a NaN row
        try:
            model().train(
                params=params(nnx_dist.train_loader(data, batch_size=2, shuffle=False), accumulate=1),
                distributed=nnx_dist.DDP(),
            )
            out["error"] = None
        except BaseException as error:  # noqa: BLE001 - recorded for the test
            out["error"] = type(error).__name__
            out["message"] = str(error)
    elif name == "preflight":
        out["errors"] = {}
        attempts = {
            "undeclared_callback": dict(callbacks=[Undeclared()]),
            "borrowed_tensorboard": dict(tensorboard=True),
            "plain_loader": dict(plain=True),
            "partition_mismatch": dict(mismatch=True),
            "writer_component": dict(callbacks=[WriterComponent()]),
            "writer_only_component": dict(callbacks=[nnx_dist.writer_only(WriterComponent)]),
            "batchnorm": dict(batchnorm=True),
            "compile": dict(train_kwargs=dict(compile=CompileSpec(backend="aot_eager"))),
            "history_journal": dict(train_kwargs=dict(history=HistoryJournal(retention=3, chunk_size=2))),
        }
        for label, options in attempts.items():
            if options.get("tensorboard"):
                try:
                    options = dict(callbacks=[TensorBoardCallback(log_dir=f"tb-borrowed-{rank}")])
                except ImportError:  # the tensorboard extra is not installed here
                    out["errors"][label] = "unavailable"
                    continue
            if options.get("plain"):
                loader = [(torch.randn(2, FEATURES), torch.randint(0, CLASSES, (2,)))]
            elif options.get("mismatch"):
                loader = nnx_dist.train_loader(train_set(), batch_size=2, seed=3 + rank)  # rank-specific seed
            else:
                loader = nnx_dist.train_loader(train_set(), batch_size=2, seed=3)
            try:
                built = batchnorm_model() if options.get("batchnorm") else model()
                built.train(
                    params=params(loader),
                    callbacks=options.get("callbacks"),
                    distributed=nnx_dist.DDP(),
                    **options.get("train_kwargs", {}),
                )
                out["errors"][label] = None
            except BaseException as error:  # noqa: BLE001
                out["errors"][label] = f"{type(error).__name__}: {error}"
        out["runs_after_preflight"] = (
            sorted(n for n in os.listdir("runs") if not n.startswith(".")) if os.path.isdir("runs") else []
        )
    elif name == "single_process_entries":
        # FIX-028: the single-process entry points refuse a multi-rank group
        # before any study, factory call or run; one rank behaves as before.
        from nnx.plans import ExperimentPlan
        from nnx.search import FloatParam, SearchBudget, SearchSpace, search

        built = []

        def train_factory():
            built.append(rank)
            return torch.utils.data.DataLoader(train_set(), batch_size=4)

        plan = (
            ExperimentPlan()
            .with_net(
                NNParams(
                    input_dim=FEATURES,
                    output_dim=CLASSES,
                    hidden_dims=[6],
                    dropout_prob=0.0,
                    activation=Activations.RELU,
                )
            )
            .with_model(NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY))
            .with_train(
                NNTrainParams(
                    n_epochs=1,
                    optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
                    monitor=MonitorSpec("loss"),
                )
            )
            .with_data(train_factory, val=lambda: torch.utils.data.DataLoader(val_set(), batch_size=4), identity="rows")
            .with_seed(3)
        )
        out["errors"] = {}
        try:
            out["fit_run"] = plan.fit().run.id
            out["errors"]["fit"] = None
        except BaseException as error:  # noqa: BLE001
            out["errors"]["fit"] = f"{type(error).__name__}: {error}"
        try:
            result = search(
                plan,
                SearchSpace(FloatParam(name="lr", low=1e-3, high=1e-1, log=True)),
                apply=lambda base, chosen: base.with_optim(NNOptimParams.builder().sgd(max_lr=chosen["lr"]).build()),
                monitor=MonitorSpec("loss"),
                budget=SearchBudget(trials=1),
                study_name="entries",
                storage="sqlite:///study.db",
            )
            out["search_states"] = [outcome.state for outcome in result.outcomes]
            out["errors"]["search"] = None
        except BaseException as error:  # noqa: BLE001
            out["errors"]["search"] = f"{type(error).__name__}: {error}"
        out["factory_calls"] = len(built)
        out["study_written"] = os.path.exists("study.db")
        out["runs"] = sorted(n for n in os.listdir("runs") if not n.startswith(".")) if os.path.isdir("runs") else []
    elif name == "all_ignored_rank":
        # rows 1 and 3 ignored, no shuffle: rank 1's first batch has no valid target
        m = model()
        data = Rows(8, seed=1, ignore=(1, 3))
        run = m.train(
            params=params(nnx_dist.train_loader(data, batch_size=2, shuffle=False), accumulate=1, n_epochs=1),
            distributed=nnx_dist.DDP(),
        )
        out.update(summary(run))
        out["state"] = {k: v.clone() for k, v in m.net.state_dict().items()}
    elif name == "all_ignored_window":
        data = Rows(8, seed=1, ignore=(0, 1, 2, 3))  # the whole first global batch is ignored
        try:
            model().train(
                params=params(nnx_dist.train_loader(data, batch_size=2, shuffle=False), accumulate=1, n_epochs=1),
                distributed=nnx_dist.DDP(),
            )
            out["error"] = None
        except BaseException as error:  # noqa: BLE001
            out["error"] = type(error).__name__
    elif name == "writer_callback_fails":
        import time

        started = time.monotonic()
        try:
            model().train(
                params=params(nnx_dist.train_loader(train_set(), batch_size=2, seed=3)),
                callbacks=[FailingWriterCallback()],
                distributed=nnx_dist.DDP(),
            )
            out["error"] = None
        except BaseException as error:  # noqa: BLE001
            out["error"] = f"{type(error).__name__}: {error}"
        out["seconds"] = time.monotonic() - started
    elif name == "steps_mismatch":
        loader = nnx_dist.train_loader(train_set(), batch_size=2, seed=3)
        if rank == 1:
            loader.batch_size = 1  # more batches on this rank (bypassing the agreed descriptor check)
            loader.descriptor = lambda: {**loader.partition.descriptor(), "batch_size": 2}  # type: ignore[method-assign]
        try:
            model().train(params=params(loader), distributed=nnx_dist.DDP())
            out["error"] = None
        except BaseException as error:  # noqa: BLE001
            out["error"] = type(error).__name__
            out["message"] = str(error)
    elif name == "empty_validation_rank":
        m = model()
        stopper = EarlyStopping(monitor=MonitorSpec("loss"), patience=0)
        run = m.train(
            params=params(
                nnx_dist.train_loader(train_set(), batch_size=2, seed=3),
                nnx_dist.validation_loader(val_set(1), batch_size=2),  # rank 1 has no validation rows
                n_epochs=5,
                monitor=MonitorSpec("loss"),
            ),
            callbacks=[stopper],
            distributed=nnx_dist.DDP(),
        )
        out.update(summary(run))
        out["validation_rows_here"] = len(nnx_dist.validation_loader(val_set(1), batch_size=2).ids())
    elif name == "writer":
        from nnx.nn.params.nn_run import NNRun

        calls = {"lease": 0, "checkpoint_save": 0, "run_save": 0, "factory": 0}
        lease, checkpoint_save, run_save = NNRun.writable_lease, NNCheckpoint.save, NNRun.save

        def counted(key, original):
            def wrapper(*args, **kwargs):
                calls[key] += 1
                return original(*args, **kwargs)

            return wrapper

        NNRun.writable_lease = counted("lease", lease)  # type: ignore[method-assign]
        NNCheckpoint.save = counted("checkpoint_save", checkpoint_save)  # type: ignore[method-assign]
        NNRun.save = counted("run_save", run_save)  # type: ignore[method-assign]

        def tensorboard():
            calls["factory"] += 1
            return EventLog("tb-writer")

        from nnx.provenance import ExperimentManifest

        m = model()
        writer_params = params(
            nnx_dist.train_loader(train_set(), batch_size=2, seed=3),
            nnx_dist.validation_loader(val_set(), batch_size=2),
        )
        run = m.train(
            params=writer_params,
            provenance=ExperimentManifest.for_model(m, train=writer_params),
            callbacks=[
                ModelCheckpoint(epochs=[0], tag="custom"),
                nnx_dist.writer_only(tensorboard),
                EarlyStopping(monitor="val_edp.loss", patience=10),
            ],
            distributed=nnx_dist.DDP(),
        )
        NNRun.writable_lease, NNCheckpoint.save, NNRun.save = lease, checkpoint_save, run_save  # type: ignore[method-assign]
        out.update(summary(run))
        out["calls"] = calls
        out["provenance"] = (
            None if run.provenance is None else (run.provenance.fingerprint, run.provenance.attempt.attempt_id)
        )
        out["state"] = {k: v.clone() for k, v in m.net.state_dict().items()}
        out["last"] = NNCheckpoint.load_training_state(run.id, Checkpoints.LAST)["distributed"]
        out["last_keys"] = sorted(NNCheckpoint.load(run.id, Checkpoints.LAST).net_state)
        out["exported"] = False
        if rank == 0:
            try:  # Hub and ONNX export need the hub / onnx extras
                m.save_pretrained("hub")
                m.to_onnx("net.onnx", torch.randn(2, FEATURES))
                out["exported"] = True
            except ImportError:
                pass
    elif name in ("resume_planned_first", "resume_planned_second"):
        # #394: a planned resume under DDP continues the plan of 2 epochs.
        class StopAfterFirst(Callback):
            distributed = "all"

            def on_epoch_end(self, ctx):
                if ctx.epoch == 0:
                    ctx.should_stop = True

        loader = nnx_dist.train_loader(train_set(), batch_size=2, seed=3)
        m = model()
        if name == "resume_planned_first":
            run = m.train(
                params=params(loader, n_epochs=2, data_id="planned"),
                callbacks=[StopAfterFirst()],
                distributed=nnx_dist.DDP(),
            )
            if rank == 0:
                with open("planned_run_id.txt", "w", encoding="utf-8") as handle:
                    handle.write(run.id)
        else:
            first_id = open("planned_run_id.txt", encoding="utf-8").read().strip()
            run = m.train(
                params=params(
                    loader,
                    n_epochs=2,
                    data_id="planned",
                    resume_from_run_id=first_id,
                    resume_mode="stateful",
                    resume_epochs="planned",
                ),
                distributed=nnx_dist.DDP(),
            )
        out.update(summary(run))
        out["state"] = {k: v.clone() for k, v in m.net.state_dict().items()}
        out["rng_after"] = torch.get_rng_state()
    elif name in ("resume_full", "resume_first", "resume_second", "resume_changed"):
        loader = nnx_dist.train_loader(train_set(), batch_size=2, seed=3)
        m = model()
        if name == "resume_full":
            run = m.train(params=params(loader, n_epochs=2, data_id="full"), distributed=nnx_dist.DDP())
        elif name == "resume_first":
            run = m.train(params=params(loader, n_epochs=1, data_id="split"), distributed=nnx_dist.DDP())
        else:
            first_id = open("first_run_id.txt", encoding="utf-8").read().strip()
            if name == "resume_changed":
                loader = nnx_dist.train_loader(train_set(), batch_size=2, seed=4)  # a different partition
            before = {k: v.clone() for k, v in m.net.state_dict().items()}
            try:
                run = m.train(
                    params=params(loader, n_epochs=1, data_id="split", resume_from_run_id=first_id),
                    distributed=nnx_dist.DDP(),
                )
            except BaseException as error:  # noqa: BLE001
                out["error"] = f"{type(error).__name__}: {error}"
                out["unchanged"] = all(torch.equal(before[k], v) for k, v in m.net.state_dict().items())
                return out
        out.update(summary(run))
        out["state"] = {k: v.clone() for k, v in m.net.state_dict().items()}
        if name == "resume_first" and rank == 0:
            with open("first_run_id.txt", "w", encoding="utf-8") as handle:
                handle.write(run.id)
        out["rng_after"] = torch.get_rng_state()
    else:
        raise SystemExit(f"unknown scenario {name}")
    return out


def main() -> None:
    name, out_dir = sys.argv[1], sys.argv[2]
    os.chdir(out_dir)
    rank, world_size = nnx_dist.init_process_group(timeout_seconds=float(os.environ.get("NNX_DDP_TIMEOUT", "60")))
    try:
        result = scenario(name, rank, world_size)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        pass
    torch.save(result, os.path.join(out_dir, f"rank{rank}.pt"))
    # Tear down together: a rank destroying its group while a peer still
    # holds open Gloo pairs can abort that peer ("terminate called without an
    # active exception").
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
