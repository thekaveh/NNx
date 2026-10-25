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

import atexit
import faulthandler
import gc
import os
import sys
import traceback
from dataclasses import replace

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


class UnreadableRows(Rows):
    """``Rows`` whose every sample fails to read (a corrupt or missing file)."""

    def __getitem__(self, index: int):
        raise OSError(f"cannot read sample {index}")


def _device() -> Devices:
    """CPU (Gloo), or CUDA (NCCL, one device per rank) when the launch sets
    ``NNX_DDP_DEVICE=cuda`` (FEAT-043's two-GPU parity run)."""
    return Devices.CUDA if os.environ.get("NNX_DDP_DEVICE") == "cuda" else Devices.CPU


def model(input_dim: int = FEATURES) -> NNModel:
    torch.manual_seed(0)
    return NNModel(
        net_params=NNParams(
            input_dim=input_dim, output_dim=CLASSES, hidden_dims=[6], dropout_prob=0.0, activation=Activations.RELU
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=_device(), loss=Losses.CROSS_ENTROPY),
    )


def train_set() -> Rows:
    # 11 rows: odd, so the pad policy repeats one; ignored targets fall unevenly on the ranks.
    return Rows(11, seed=1, ignore=(0, 3, 4))


def val_set(n: int = 5) -> Rows:
    return Rows(n, seed=2)


def sgd_optim(accumulate: int = 2) -> NNOptimParams:
    return NNOptimParams.builder().sgd(max_lr=0.1).accumulate_grad(accumulate).build()


def params(train_loader, val_loader=None, **kwargs) -> NNTrainParams:
    return NNTrainParams(
        n_epochs=kwargs.pop("n_epochs", 2),
        train_loader=train_loader,
        val_loader=val_loader,
        optim=kwargs.pop("optim", None) or sgd_optim(kwargs.pop("accumulate", 2)),
        overwrite_existing=kwargs.pop("overwrite_existing", True),
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


class TopologyChanging(Callback):
    """Changes model.net's topology as QAT does: it overrides only checkpoint_transforms."""

    def checkpoint_transforms(self):
        return ()


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


class TrainEndDraw(Callback):
    """#538: a draw on every rank once training ends (a final test score on a
    loader without a generator); records each rank's streams before it. A
    draw in every session's on_train_begin, and a rank-sized one after each
    epoch, give the ranks streams of their own to restore."""

    distributed = "all"

    def __init__(self) -> None:
        self.rng_before = None

    def on_train_begin(self, ctx) -> None:
        torch.rand(1)

    def on_epoch_end(self, ctx) -> None:
        torch.rand(torch.distributed.get_rank() + 1)

    def on_train_end(self, ctx) -> None:
        self.rng_before = torch.get_rng_state()
        torch.rand(1)


def dropout_model() -> NNModel:
    torch.manual_seed(0)
    return NNModel(
        net_params=NNParams(
            input_dim=FEATURES, output_dim=CLASSES, hidden_dims=[6], dropout_prob=0.3, activation=Activations.RELU
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=_device(), loss=Losses.CROSS_ENTROPY),
    )


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
        # #478: the failed fit's DDP wrapper is dropped on every rank, even the
        # one whose error (and its traceback) is still alive here.
        out["live_ddp"] = _live_ddp()
    elif name == "release_on_failure":
        # #539: a fit that fails before the loop (a refused reservation, a
        # peer's failed wrapper build, NNRun or the compile session) or inside
        # DDP's forward drops the wrapper before the error reaches the caller.
        # Each handler counts the live wrappers, then calls shutdown(), whose
        # destroy_process_group is spied on (and skipped, so the group lives on).
        import nnx.nn.nn_model as nn_model_module

        def fit(salt: str, **kwargs):
            input_dim = kwargs.pop("input_dim", FEATURES)
            loader = nnx_dist.train_loader(train_set(), batch_size=2, seed=3)
            return model(input_dim).train(
                params=params(loader, n_epochs=1, **kwargs), salt=salt, distributed=nnx_dist.DDP()
            )

        def failing(what: str):
            def fail(*args, **kwargs):
                raise RuntimeError(f"{what} failed (injected on rank {rank})")

            return fail

        fit("rerun", overwrite_existing=False)  # a finished run, re-run below
        cases = {
            "reservation": ({}, dict(salt="rerun", overwrite_existing=False)),
            "build": ({"Reducer": (torch.distributed, failing("Reducer"))} if rank == 1 else {}, dict(salt="build")),
            "nnrun": ({"NNRun": (nn_model_module, failing("NNRun"))}, dict(salt="nnrun")),
            "compiling": ({"_compiling": (nn_model_module, failing("_compiling"))}, dict(salt="compiling")),
            "forward": ({}, dict(salt="forward", input_dim=FEATURES + 1)),  # the rows have FEATURES columns
        }
        out["cases"] = {}
        for label, (patches, kwargs) in cases.items():
            originals = {attr: getattr(owner, attr) for attr, (owner, _) in patches.items()}
            for attr, (owner, replacement) in patches.items():
                setattr(owner, attr, replacement)
            case: dict = {"error": None}
            try:
                fit(kwargs.pop("salt"), **kwargs)
            except BaseException as error:  # noqa: BLE001 - recorded for the test
                case["error"] = f"{type(error).__name__}: {error}"
                case["live_ddp"] = _live_ddp()
                case["net_forward_locals"] = _net_forward_locals(error)
                case["live_ddp_at_destroy"] = _shutdown_count()
            finally:
                for attr, (owner, _) in patches.items():
                    setattr(owner, attr, originals[attr])
            out["cases"][label] = case
    elif name == "unscorable":
        # A task-free MSE model: one rank's batch holds a fractional target (unscorable as a class).
        from torch.utils.data import TensorDataset

        targets = torch.tensor([[0.0], [1.0], [1.0], [0.5]])
        data = TensorDataset(torch.randn(4, FEATURES, generator=torch.Generator().manual_seed(3)), targets)
        torch.manual_seed(0)
        regression = NNModel(
            net_params=NNParams(
                input_dim=FEATURES, output_dim=1, hidden_dims=[6], dropout_prob=0.0, activation=Activations.RELU
            ),
            params=NNModelParams(net=Nets.FEED_FWD, device=_device(), loss=Losses.MEAN_SQUARED_ERROR),
        )
        try:
            regression.train(
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
            "topology_callback": dict(callbacks=[TopologyChanging()]),
            "writer_only_topology_callback": dict(callbacks=[nnx_dist.writer_only(TopologyChanging)]),
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
    elif name == "shutdown":
        # FEAT-042: leave together; a second call is a no-op.
        nnx_dist.shutdown(timeout_seconds=30)
        out["initialized_after"] = torch.distributed.is_initialized()
        nnx_dist.shutdown(timeout_seconds=30)
        out["second_call"] = "no-op"
    elif name == "shutdown_missing":
        # FEAT-042: rank 1 never reaches the teardown; rank 0's shutdown is bounded.
        import time

        if rank == 0:
            start = time.monotonic()
            try:
                nnx_dist.shutdown(timeout_seconds=3)
                out["error"] = None
            except BaseException as error:  # noqa: BLE001 - recorded for the test
                out["error"] = f"{type(error).__name__}: {error}"
            out["seconds"] = time.monotonic() - start
            out["initialized_after"] = torch.distributed.is_initialized()
            out["teardown_done"] = True  # the group is left for process exit
        else:
            # Arrives after rank 0's wait timed out: rank 0 has raised (its
            # process stays at exit, its barrier still queued), so this
            # barrier completes against the queued one.
            time.sleep(6)
            start = time.monotonic()
            try:
                nnx_dist.shutdown(timeout_seconds=3)
                out["error"] = None
            except BaseException as error:  # noqa: BLE001 - recorded for the test
                out["error"] = f"{type(error).__name__}: {error}"
            out["seconds"] = time.monotonic() - start
            out["initialized_after"] = torch.distributed.is_initialized()

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
        try:
            from nnx import NNParamGroupSpec, Optims
            from nnx.objectives import supervised_objective
            from nnx.trainer import NNTrainerParams, Trainer

            group = NNParamGroupSpec(name_pattern="*", lr=0.1)
            optim = NNOptimParams(name=Optims.SGD, max_lr=0.1, momentum=0.0, weight_decay=0.0, param_groups=[group])
            trainer_params = NNTrainerParams(
                n_epochs=1,
                data_id="trainer-entry",
                train_loader=torch.utils.data.DataLoader(train_set(), batch_size=4),
                optims={"all": optim},
            )
            out["trainer_run"] = Trainer(model()).train(trainer_params, objective=supervised_objective()).id
            out["errors"]["trainer"] = None
        except BaseException as error:  # noqa: BLE001
            out["errors"]["trainer"] = f"{type(error).__name__}: {error}"
        try:
            plain = model().train(
                params(torch.utils.data.DataLoader(train_set(), batch_size=4), None, n_epochs=1, data_id="plain")
            )
            out["plain_run"] = plain.id
            out["errors"]["plain"] = None
        except BaseException as error:  # noqa: BLE001
            out["errors"]["plain"] = f"{type(error).__name__}: {error}"
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
    elif name in (
        "resume_full",
        "resume_first",
        "resume_second",
        "resume_changed",
        "resume_corrupt",
        "resume_bad_id",
        "resume_hyperparameters",
        "resume_republished",
        "resume_republished_between_loads",
    ):
        loader = nnx_dist.train_loader(train_set(), batch_size=2, seed=3)
        m = model()
        import nnx.nn.params.nn_checkpoint as nn_checkpoint

        hashed: list[str] = []  # FEAT-046: the files this rank hashes to verify the resume point
        real_sha256 = nn_checkpoint._sha256

        def counting_sha256(path):
            hashed.append(path)
            return real_sha256(path)

        nn_checkpoint._sha256 = counting_sha256
        verified: list[str] = []  # the ranks' NNCheckpoint.verify calls
        real_verify = nn_checkpoint.NNCheckpoint.verify

        def counting_verify(*args, **kwargs):
            verified.append(str(kwargs.get("run", args[0] if args else None)))
            manifest = real_verify(*args, **kwargs)
            if name == "resume_republished":  # #537: LAST is published again before the ranks load it
                from _checkpoint_files import republish

                republish(verified[-1], Checkpoints.LAST)
            return manifest

        nn_checkpoint.NNCheckpoint.verify = staticmethod(counting_verify)
        if name == "resume_republished_between_loads":
            # #537: the writer loads LAST, then publishes it again before rank 1 loads it.
            real_load = nn_checkpoint.NNCheckpoint._load_resume_point

            def ordered_load(run, *args, **kwargs):
                if rank != 0:
                    torch.distributed.barrier()
                try:
                    return real_load(run, *args, **kwargs)
                finally:
                    if rank == 0:
                        from _checkpoint_files import republish

                        republish(run, Checkpoints.LAST)
                        torch.distributed.barrier()

            nn_checkpoint.NNCheckpoint._load_resume_point = staticmethod(ordered_load)
        if name == "resume_full":
            run = m.train(params=params(loader, n_epochs=2, data_id="full"), distributed=nnx_dist.DDP())
        elif name == "resume_first":
            run = m.train(params=params(loader, n_epochs=1, data_id="split"), distributed=nnx_dist.DDP())
        else:
            if name == "resume_bad_id":
                first_id = "../outside"  # refused by the run-id check, on every rank alike
            else:
                first_id = open("first_run_id.txt", encoding="utf-8").read().strip()
            if name == "resume_changed":
                loader = nnx_dist.train_loader(train_set(), batch_size=2, seed=4)  # a different partition
            if name == "resume_corrupt":  # the writer flips a byte of LAST's training state, then all resume
                if rank == 0:
                    checkpoint_path = nn_checkpoint._checkpoint_path(first_id, Checkpoints.LAST)
                    point = nn_checkpoint.NNCheckpoint.resume_point(first_id, Checkpoints.LAST)
                    (state,) = [name for name in point["files"] if name != os.path.basename(checkpoint_path)]
                    with open(os.path.join(os.path.dirname(checkpoint_path), state), "r+b") as handle:
                        handle.seek(-1, os.SEEK_END)
                        last = handle.read(1)
                        handle.seek(-1, os.SEEK_END)
                        handle.write(bytes([last[0] ^ 0xFF]))
                torch.distributed.barrier()
            # #502: a changed lr / weight_decay / momentum is refused on every rank.
            changed = name == "resume_hyperparameters"
            configured = replace(sgd_optim(), max_lr=1e-4, weight_decay=0.1, momentum=0.0) if changed else None
            republished = name in ("resume_republished", "resume_republished_between_loads")
            mode = {"resume_mode": "weights_only"} if republished else {}
            before = {k: v.clone() for k, v in m.net.state_dict().items()}
            rng_before = torch.get_rng_state()
            try:
                run = m.train(
                    params=params(
                        loader, n_epochs=1, data_id="split", resume_from_run_id=first_id, optim=configured, **mode
                    ),
                    distributed=nnx_dist.DDP(),
                )
            except BaseException as error:  # noqa: BLE001
                out["error"] = f"{type(error).__name__}: {error}"
                out["unchanged"] = all(torch.equal(before[k], v) for k, v in m.net.state_dict().items())
                out["rng_unchanged"] = torch.equal(rng_before, torch.get_rng_state())
                out["hashed"], out["verified"] = len(hashed), len(verified)
                return out
        out.update(summary(run))
        out["state"] = {k: v.clone() for k, v in m.net.state_dict().items()}
        if name == "resume_first" and rank == 0:
            with open("first_run_id.txt", "w", encoding="utf-8") as handle:
                handle.write(run.id)
        out["rng_after"] = torch.get_rng_state()
        out["hashed"], out["verified"] = len(hashed), len(verified)
    elif name in ("train_end_draw_full", "train_end_draw_first", "train_end_draw_second"):
        # #538: with an on_train_end draw on every rank, 1 epoch + a resume of
        # 1 equals 2 straight (dropout makes the weights depend on the RNG).
        loader = nnx_dist.train_loader(train_set(), batch_size=2, seed=3)
        m, draw = dropout_model(), TrainEndDraw()
        resume = {}
        if name == "train_end_draw_second":
            first_id = open("draw_first_run_id.txt", encoding="utf-8").read().strip()
            resume = dict(resume_from_run_id=first_id, resume_mode="stateful")
        run = m.train(
            params=params(
                loader,
                n_epochs=2 if name == "train_end_draw_full" else 1,
                data_id="draw-full" if name == "train_end_draw_full" else "draw-split",
                **resume,
            ),
            callbacks=[draw],
            distributed=nnx_dist.DDP(),
        )
        out.update(summary(run))
        out["state"] = {k: v.clone() for k, v in m.net.state_dict().items()}
        out["rng_before"] = draw.rng_before
        state = NNCheckpoint.load_training_state(run.id, Checkpoints.LAST)
        out["rng_by_rank"] = [entry["torch"] for entry in state["distributed"]["rng_by_rank"]]
        torch.distributed.barrier()  # every rank has read LAST before the writer names the run
        if name == "train_end_draw_first" and rank == 0:
            with open("draw_first_run_id.txt", "w", encoding="utf-8") as handle:
                handle.write(run.id)
    elif name in ("resume_steps_mismatch", "resume_unreadable_rank"):
        # #507: a resume that fails on every rank before its first committed
        # update hands each rank's model and RNG back: the step-count refusal
        # (rank 1 has more batches, past the descriptor check), or rank 1's
        # first fetch failing while rank 0 waits in its first step until the
        # process group's (short) timeout fails it.
        first_id = open("first_run_id.txt", encoding="utf-8").read().strip()
        data = UnreadableRows(11, seed=1) if name == "resume_unreadable_rank" and rank == 1 else train_set()
        loader = nnx_dist.train_loader(data, batch_size=2, seed=3)
        if name == "resume_steps_mismatch" and rank == 1:
            loader.batch_size = 1
            loader.descriptor = lambda: {**loader.partition.descriptor(), "batch_size": 2}  # type: ignore[method-assign]
        import random

        import numpy as np

        m = model()
        before = {k: v.clone() for k, v in m.net.state_dict().items()}
        random.seed(777 + rank)
        np.random.seed(777 + rank)
        torch.manual_seed(777 + rank)
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            m.train(
                params=params(
                    loader, n_epochs=1, data_id="resumed", resume_from_run_id=first_id, resume_mode="stateful"
                ),
                distributed=nnx_dist.DDP(),
            )
            out["error"] = None
        except BaseException as error:  # noqa: BLE001
            out["error"] = f"{type(error).__name__}: {error}"
        out["unchanged"] = all(torch.equal(before[k], v) for k, v in m.net.state_dict().items())
        out["python_rng_unchanged"] = random.getstate() == python_rng
        numpy_now = np.random.get_state()
        out["numpy_rng_unchanged"] = numpy_now[0] == numpy_rng[0] and all(
            np.array_equal(now, then) for now, then in zip(numpy_now[1:], numpy_rng[1:], strict=True)
        )
        out["rng_unchanged"] = torch.equal(torch.get_rng_state(), torch_rng)
        if name == "resume_unreadable_rank":
            out["teardown_done"] = True  # the timed-out group is left for process exit
    elif name == "parked":
        # #478: rank 1 parks while rank 0 waits at a barrier, so the scenario
        # hangs; its faulthandler dump must name where rank 1 is.
        if rank == 1:
            _parked_for_diagnostics()
        torch.distributed.barrier()
    else:
        raise SystemExit(f"unknown scenario {name}")
    return out


def _live_ddp() -> int:
    """The DDP wrappers in ``gc.get_objects()``, garbage included (#478)."""
    from torch.nn.parallel import DistributedDataParallel

    return sum(isinstance(item, DistributedDataParallel) for item in gc.get_objects())


def _net_forward_locals(error: BaseException) -> list[list[str]]:
    """The local names of each net ``forward`` frame in ``error``'s
    traceback (DDP's own frames excluded): #539 clears DDP's frames only."""
    from torch.nn.parallel import DistributedDataParallel

    names = []
    tb = error.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        owner = frame.f_locals.get("self")
        if frame.f_code.co_name == "forward" and isinstance(owner, torch.nn.Module):
            if not isinstance(owner, DistributedDataParallel):
                names.append(sorted(frame.f_locals))
        tb = tb.tb_next
    return names


def _shutdown_count() -> list[int]:
    """``nnx.distributed.shutdown()`` with ``destroy_process_group`` spied on:
    the live DDP wrappers it reaches the destroy with (#539). The destroy is
    skipped, so the group stays up for the next case."""
    seen: list[int] = []
    destroy = torch.distributed.destroy_process_group
    torch.distributed.destroy_process_group = lambda *args, **kwargs: seen.append(_live_ddp())
    try:
        nnx_dist.shutdown(timeout_seconds=30)
    finally:
        torch.distributed.destroy_process_group = destroy
    return seen


def _parked_for_diagnostics() -> None:
    import time

    time.sleep(float(os.environ.get("NNX_DDP_PARK_SECONDS", "20")))


def _watch_for_hangs(out_dir: str) -> None:
    """#478: every thread's stack goes to ``faults-<pid>.txt`` in OUT_DIR when
    the rank aborts (SIGABRT, SIGSEGV, ...) and, once, after
    ``NNX_DDP_DUMP_AFTER`` seconds (default 200, before ``launch()``'s
    240 s timeout), so a failed or hung scenario says where each rank was.
    The watchdog stays armed through teardown and interpreter exit, where a
    rank can also hang."""
    faults = open(os.path.join(out_dir, f"faults-{os.getpid()}.txt"), "w", encoding="utf-8")  # noqa: SIM115
    faulthandler.enable(file=faults, all_threads=True)
    faulthandler.dump_traceback_later(float(os.environ.get("NNX_DDP_DUMP_AFTER", "200")), repeat=False, file=faults)


def _mark(phase: str) -> None:
    """A per-rank progress line on stderr (#478): a failed launch shows the
    last phase each rank reached."""
    print(f"[ddp-scenario pid={os.getpid()} rank={os.environ.get('RANK', '?')}] {phase}", file=sys.stderr, flush=True)


def main() -> None:
    name, out_dir = sys.argv[1], sys.argv[2]
    os.chdir(out_dir)
    _watch_for_hangs(out_dir)
    atexit.register(_mark, "interpreter exiting")
    rank, world_size = nnx_dist.init_process_group(timeout_seconds=float(os.environ.get("NNX_DDP_TIMEOUT", "60")))
    _mark(f"joined the group: {name}")
    try:
        result = scenario(name, rank, world_size)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        pass
    torch.save(result, os.path.join(out_dir, f"rank{rank}.pt"))
    _mark("result saved")
    if not result.get("teardown_done"):
        nnx_dist.shutdown()  # leave together (a no-op when the scenario already did)
    _mark("left the group")


if __name__ == "__main__":
    main()
