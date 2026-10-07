"""Optional single-node DDP, launched by ``torchrun`` (FEAT-030).

NNx runs one process per device under PyTorch's own runtime — ``torchrun``
starts the processes, ``torch.distributed`` connects them and
``DistributedDataParallel`` averages gradients — and adds what a training
library must get right on top:

- **Exact global batches.** Each committed update equals a single process
  training on the union of the ranks' batches: every rank's loss numerator
  is scaled by ``world_size / W`` where ``W`` is the *global* loss
  denominator of the update window (unequal valid counts included), so
  DDP's gradient average is the global-batch gradient. Accumulation
  windows synchronize gradients only at committed updates (``no_sync``
  otherwise); an all-masked global window is skipped on every rank.
- **Agreement, not hangs.** Non-finite losses, step-count mismatches,
  preflight refusals, resume and component-restore failures, epoch-end
  callback failures and writer-side commit failures are decided
  collectively: every rank raises together. Anything else that fails on one
  rank only (a dataset error mid-epoch, say) ends through the process
  group's timeout (``init_process_group(timeout_seconds=...)``) and
  ``torchrun`` stopping the other ranks — bounded, never a silent hang. Records (per batch, per epoch,
  validation) are global — gathered and computed identically on every
  rank — so monitors, early stopping, BEST and the plateau scheduler agree.
- **Partitions declare their policy.** :func:`train_loader` partitions a
  map-style dataset per epoch (seeded by ``seed`` and the epoch) and pads
  (repeats leading samples) or drops the tail so every rank takes the same
  number of steps. :func:`validation_loader` shards rows ``rank::world``
  *unpadded*: every row id is scored exactly once, a rank may hold none,
  and the validation loop itself runs no collective.
- **One writer.** Only the writer rank (``DDP(writer_rank=0)``) holds the
  run lease and writes history, checkpoints (BEST included), deferred
  callback checkpoints and provenance; every rank returns the same run id
  and the same global records, and the writer's provenance record. Checkpoints, Hub and ONNX export keep the
  canonical module's keys (no ``module.`` prefix) — DDP wraps ``model.net``
  per fit and never replaces it.
- **Callbacks declare their rank behaviour** (``Callback.distributed``):
  ``"all"`` runs on every rank (``EarlyStopping``, ``LRMonitor``),
  ``"writer"`` on the writer only (``ModelCheckpoint``, plain function
  callbacks); a writer-only callback may not carry checkpointed component
  state. A
  ``TensorBoardCallback`` / ``WandbCallback`` writes when constructed, so a
  borrowed instance is refused; pass :func:`writer_only` with a factory and
  NNx builds it on the writer rank alone. A callback that declares nothing
  is refused before training, on every rank.
- **Resume** (same world size and partition only) restores each rank's RNG
  streams from the checkpoint; a different world size or partition — or a
  single-process stateful resume of a distributed checkpoint — fails before
  anything is restored (``resume_mode="weights_only"`` starts fresh from the
  weights).

Exactness holds for deterministic forwards: batch normalization (running
statistics per rank) is refused, and stochastic layers such as dropout draw
each rank's own stream, so they match a single process only in
distribution. Global records cost one gather of every batch's outputs and
targets per step, and validation gathers every scored output once — sized
for supervised models, not for huge vocabularies.

Scope: one node, FP32, the default train and validation steps, one
optimizer, a fixed topology without batch normalization, map-style datasets. Out: multi-node or elastic
runs, FSDP / DeepSpeed, mixed precision, ``compile=``, custom steps or
objectives, history journals, graph neighbour sampling.

Recipe::

    torchrun --standalone --nproc_per_node=2 train.py

    # train.py
    import nnx
    from nnx import distributed

    distributed.init_process_group()  # env:// from torchrun; gloo on CPU, nccl on CUDA
    params = nnx.NNTrainParams(
        n_epochs=3,
        train_loader=distributed.train_loader(train_set, batch_size=32, seed=0),
        val_loader=distributed.validation_loader(val_set, batch_size=64),
        optim=...,
    )
    run = model.train(params=params, distributed=distributed.DDP())
    distributed.shutdown()  # leave together: no rank exits while a peer is mid-teardown
"""

from __future__ import annotations

import contextlib
import datetime
import math
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

__all__ = [
    "DDP",
    "DistributedFailure",
    "RankPartition",
    "ShardedLoader",
    "init_process_group",
    "shutdown",
    "train_loader",
    "validation_loader",
    "writer_only",
]

PARTITION_POLICIES: tuple[str, ...] = ("pad", "drop")
"""How a train partition evens out its tail across ranks."""


class DistributedFailure(RuntimeError):
    """Raised on every rank when another rank failed (preflight, a
    non-finite loss, a step-count mismatch, a writer-side commit): the
    failing rank raises its own error, the others this one naming it."""


def _world() -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized: launch with `torchrun --standalone --nproc_per_node=N script.py` "
            "and call nnx.distributed.init_process_group() first"
        )
    return dist.get_rank(), dist.get_world_size()


def shutdown(*, timeout_seconds: float = 60.0) -> None:
    """Leave the process group together, then destroy it: the counterpart of
    :func:`init_process_group`, called once training is over.

    Every rank meets at a teardown barrier before the group is destroyed, so
    no rank tears its connections down while a peer still uses them (with
    Gloo, a peer can otherwise abort: "terminate called without an active
    exception"). With Gloo the wait is bounded by ``timeout_seconds``: when a
    rank does not arrive in time the others raise ``RuntimeError`` instead of
    blocking in the call, and leave the group to process exit (destroying it
    then could block on the missing rank). The timed-out barrier stays queued,
    so the process itself may still wait at exit until the missing rank
    arrives (its barrier then completes against the queued one) or the
    group's own timeout (``init_process_group(timeout_seconds=...)``) expires.
    With NCCL, older torch releases ignore ``timeout_seconds`` unless
    ``TORCH_NCCL_BLOCKING_WAIT=1`` is set, so the wait is bounded by the
    group's timeout instead. Without a process group, or after a successful
    call, it does nothing; after a failed call the group is still
    initialized (``init_process_group()`` returns it as is), so exit."""
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError(f"timeout_seconds must be a finite number > 0, got {timeout_seconds!r}")
    if not dist.is_available() or not dist.is_initialized():
        return
    reason = "it timed out"
    try:
        work = dist.barrier(async_op=True)
        arrived = work is not None and work.wait(timeout=datetime.timedelta(seconds=timeout_seconds))
    except Exception as error:  # a timed-out wait raises (Gloo; NCCL on recent torch)
        arrived, reason = False, f"{type(error).__name__}: {error}"
    if not arrived:
        raise RuntimeError(
            f"nnx.distributed.shutdown(): not every rank reached the teardown barrier within "
            f"{timeout_seconds:g}s ({reason}); the process group is left for process exit"
        )
    dist.destroy_process_group()


def _int(value: Any, what: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{what} must be an int >= {minimum}, got {value!r}")
    return value


def init_process_group(backend: Optional[str] = None, *, timeout_seconds: float = 120.0) -> tuple[int, int]:
    """Join the process group ``torchrun`` described in the environment
    (``env://``) and return ``(rank, world_size)``. ``backend`` defaults to
    ``nccl`` when CUDA is available, else ``gloo``; ``timeout_seconds``
    bounds every collective, so a rank that dies cannot hang the others
    forever. With CUDA, each process takes ``cuda:LOCAL_RANK``."""
    if not dist.is_available():
        raise RuntimeError("this torch build has no torch.distributed")
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    missing = [name for name in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT") if name not in os.environ]
    if missing:
        raise RuntimeError(
            f"no torchrun environment ({', '.join(missing)} unset): launch with "
            "`torchrun --standalone --nproc_per_node=N script.py`"
        )
    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds!r}")
    chosen = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if chosen == "nccl":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group(backend=chosen, timeout=datetime.timedelta(seconds=timeout_seconds))
    return dist.get_rank(), dist.get_world_size()


# --- partitions ---------------------------------------------------------------------------------


class RankPartition(Sampler[int]):
    """This rank's dataset indices, per epoch.

    ``kind="train"``: a seeded permutation of ``range(n)`` (``seed + epoch``;
    identity order with ``shuffle=False``), evened out by ``policy`` —
    ``"pad"`` repeats leading indices, ``"drop"`` drops the tail — and
    dealt ``rank::world_size``, so every rank serves the same count.
    ``kind="validation"``: rows ``rank::world_size`` in order, unpadded
    (counts may differ; a rank may serve none)."""

    def __init__(
        self,
        n: int,
        *,
        rank: int,
        world_size: int,
        kind: str = "train",
        policy: str = "pad",
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.n = _int(n, "n", minimum=0)
        self.world_size = _int(world_size, "world_size", minimum=1)
        self.rank = _int(rank, "rank", minimum=0)
        if self.rank >= self.world_size:
            raise ValueError(f"rank {rank} is outside a world of {world_size}")
        if kind not in ("train", "validation"):
            raise ValueError(f"kind must be 'train' or 'validation', got {kind!r}")
        if policy not in PARTITION_POLICIES:
            raise ValueError(f"policy must be one of {PARTITION_POLICIES}, got {policy!r}")
        if not isinstance(shuffle, bool):
            raise TypeError(f"shuffle must be a bool, got {shuffle!r}")
        self.kind, self.policy, self.shuffle = kind, policy, shuffle
        self.seed = _int(seed, "seed", minimum=0)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = _int(epoch, "epoch", minimum=0)

    def global_order(self) -> list[int]:
        """Every rank's indices for this epoch, before dealing (train: after
        padding or dropping)."""
        if self.kind == "validation" or not self.shuffle:
            order = list(range(self.n))
        else:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(self.n, generator=generator).tolist()
        if self.kind == "train":
            remainder = len(order) % self.world_size
            if remainder and self.policy == "pad":
                padding = self.world_size - remainder
                order += (order * (padding // max(len(order), 1) + 1))[:padding] if order else []
            elif remainder:
                order = order[: len(order) - remainder]
        return order

    def indices(self) -> list[int]:
        return self.global_order()[self.rank :: self.world_size]

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices())

    def __len__(self) -> int:
        return len(self.indices())

    def descriptor(self) -> dict[str, Any]:
        """What a resume must match: the dataset size, world, kind, policy,
        shuffle and seed (never the epoch or rank)."""
        return {
            "n": self.n,
            "world_size": self.world_size,
            "kind": self.kind,
            "policy": self.policy,
            "shuffle": self.shuffle,
            "seed": self.seed,
        }


class ShardedLoader:
    """A re-iterable loader over one rank's :class:`RankPartition` of a
    map-style dataset (``DataLoader`` underneath, in-process, ordered).
    ``set_epoch`` reseeds a train partition; ``ids()`` names the dataset
    rows this rank serves this epoch."""

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        partition: RankPartition,
        *,
        collate_fn: Optional[Callable[[list[Any]], Any]] = None,
    ) -> None:
        if not hasattr(dataset, "__getitem__") or not hasattr(dataset, "__len__"):
            raise TypeError("a sharded loader needs a map-style dataset (with __getitem__ and __len__)")
        if len(dataset) != partition.n:
            raise ValueError(f"the partition covers {partition.n} rows, the dataset has {len(dataset)}")
        self.dataset = dataset
        self.batch_size = _int(batch_size, "batch_size", minimum=1)
        self.partition = partition
        self.collate_fn = collate_fn

    def set_epoch(self, epoch: int) -> None:
        self.partition.set_epoch(epoch)

    def ids(self) -> list[int]:
        return self.partition.indices()

    def __iter__(self) -> Iterator[Any]:
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.partition,
            shuffle=False,
            num_workers=0,
            collate_fn=self.collate_fn,
        )
        return iter(loader)

    def __len__(self) -> int:
        rows = len(self.partition)
        return (rows + self.batch_size - 1) // self.batch_size

    def descriptor(self) -> dict[str, Any]:
        return {**self.partition.descriptor(), "batch_size": self.batch_size}


def train_loader(
    dataset: Any,
    batch_size: int,
    *,
    policy: str = "pad",
    seed: int = 0,
    shuffle: bool = True,
    collate_fn: Optional[Callable[[list[Any]], Any]] = None,
) -> ShardedLoader:
    """This rank's training loader: a per-epoch seeded partition of
    ``dataset`` evened out by ``policy`` (``"pad"`` or ``"drop"``), so
    every rank takes the same number of steps."""
    rank, world_size = _world()
    partition = RankPartition(
        len(dataset), rank=rank, world_size=world_size, kind="train", policy=policy, shuffle=shuffle, seed=seed
    )
    return ShardedLoader(dataset, batch_size, partition, collate_fn=collate_fn)


def validation_loader(
    dataset: Any, batch_size: int, *, collate_fn: Optional[Callable[[list[Any]], Any]] = None
) -> ShardedLoader:
    """This rank's validation shard: rows ``rank::world_size``, unpadded —
    every row is scored exactly once across the ranks."""
    rank, world_size = _world()
    partition = RankPartition(len(dataset), rank=rank, world_size=world_size, kind="validation")
    return ShardedLoader(dataset, batch_size, partition, collate_fn=collate_fn)


# --- the spec and callbacks ---------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True, slots=True)
class DDP:
    """``model.train(params, distributed=DDP())``: data-parallel training
    over the initialized process group. ``writer_rank`` owns every
    artifact; ``find_unused_parameters`` is passed to
    ``DistributedDataParallel``."""

    writer_rank: int = 0
    find_unused_parameters: bool = False

    def __post_init__(self) -> None:
        _int(self.writer_rank, "writer_rank", minimum=0)
        if not isinstance(self.find_unused_parameters, bool):
            raise TypeError(f"find_unused_parameters must be a bool, got {self.find_unused_parameters!r}")


def writer_only(factory: Callable[[], Any]) -> Any:
    """A callback NNx builds on the writer rank only — for callbacks that
    write when constructed (``TensorBoardCallback``, ``WandbCallback``):
    ``callbacks=[writer_only(lambda: TensorBoardCallback("tb"))]``. Without
    ``distributed=``, the factory is simply called."""
    if not callable(factory):
        raise TypeError("writer_only needs a zero-argument factory returning a Callback")
    from .nn.callbacks import _WriterOwned

    return _WriterOwned(factory)


# --- the per-fit session ------------------------------------------------------------------------


def _send(value: Any) -> list[Any]:
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def _summary(error: BaseException) -> tuple[str, str]:
    return type(error).__name__, str(error).splitlines()[0][:500] if str(error) else ""


def agree(error: Optional[BaseException], what: str) -> None:
    """Collective verdict: every rank calls this at the same point; if any
    rank had an error, every rank raises — its own error, or a
    :class:`DistributedFailure` naming the first failing rank."""
    verdicts = _send(None if error is None else _summary(error))
    failed = [(rank, verdict) for rank, verdict in enumerate(verdicts) if verdict is not None]
    if not failed:
        return
    if error is not None:
        raise error
    rank, (kind, message) = failed[0]
    raise DistributedFailure(f"{what} failed on rank {rank}: {kind}: {message}")


@contextlib.contextmanager
def collectively(what: str) -> Iterator[None]:
    """Run a block that may fail on some ranks, then agree on the outcome."""
    error: Optional[BaseException] = None
    try:
        yield
    except Exception as caught:
        error = caught
    agree(error, what)


class _DDPSession:
    """One fit's process-group view: the DDP wrapper (around, never
    assigned to, the canonical module) and the collectives the loop uses."""

    def __init__(self, net: torch.nn.Module, spec: DDP, device: torch.device) -> None:
        self.rank, self.world_size = _world()
        if spec.writer_rank >= self.world_size:
            raise ValueError(f"writer_rank {spec.writer_rank} is outside a world of {self.world_size}")
        self.spec = spec
        self.writer = self.rank == spec.writer_rank
        self.device = device
        from torch.nn.parallel import DistributedDataParallel

        self._net = net
        self.module = DistributedDataParallel(
            net,
            device_ids=[device] if device.type == "cuda" else None,
            find_unused_parameters=spec.find_unused_parameters,
        )
        self.in_step = False
        self.validation_ids: list[int] = []

    # the forward the train step routes through DDP; everything else is canonical
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs) if self.in_step else self._net(*args, **kwargs)

    @contextlib.contextmanager
    def step(self, *, sync: bool) -> Iterator[None]:
        self.in_step = True
        try:
            with contextlib.nullcontext() if sync else self.module.no_sync():
                yield
        finally:
            self.in_step = False

    def sum(self, value: float) -> float:
        tensor = torch.tensor([float(value)], dtype=torch.float64, device=self._collective_device())
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item())

    def gather(self, value: Any) -> list[Any]:
        return _send(value)

    def barrier(self) -> None:
        dist.barrier()

    def _collective_device(self) -> torch.device:
        return self.device if dist.get_backend() == "nccl" else torch.device("cpu")

    def check_steps(self, n_batches: Optional[int], epoch: int) -> None:
        counts = self.gather(n_batches)
        if len(set(counts)) != 1:
            raise DistributedFailure(
                f"epoch {epoch}: ranks would take different numbers of steps {counts}; every rank must serve "
                "the same number of batches (use nnx.distributed.train_loader)"
            )


def _rank_behaviour(callback: Any) -> Optional[str]:
    return getattr(type(callback), "distributed", None)


def classify_callbacks(callbacks: Sequence[Any], *, writer: bool) -> list[Any]:
    """This rank's callbacks under DDP, or a ValueError for one whose rank
    behaviour is unsupported or undeclared."""
    from .nn.callbacks import _WriterOwned

    kept: list[Any] = []
    for callback in callbacks:
        if isinstance(callback, _WriterOwned):
            if writer:
                built = callback.build()
                from .components import StatefulComponent

                if isinstance(built, StatefulComponent):
                    # Raised on the writer inside the agreed preflight, so
                    # every rank refuses: replicas never register it.
                    raise ValueError(
                        f"writer_only built {type(built).__name__}, which carries checkpointed component state that "
                        "every rank must restore alike on resume: pass it without writer_only (distributed = 'all')"
                    )
                kept.append(built)
            continue
        behaviour = _rank_behaviour(callback)
        if behaviour == "all":
            kept.append(callback)
        elif behaviour == "writer":
            from .components import StatefulComponent

            if isinstance(callback, StatefulComponent):
                raise ValueError(
                    f"{type(callback).__name__} runs on the writer rank only but carries checkpointed component "
                    "state, which every rank must restore alike on resume: declare it distributed = 'all' (it must "
                    "then write nothing) or train without it"
                )
            if writer:
                kept.append(callback)
        elif behaviour == "unsupported":
            raise ValueError(
                f"{type(callback).__name__} writes when constructed, so an instance built on every rank cannot be "
                "used under DDP: pass nnx.distributed.writer_only(lambda: "
                f"{type(callback).__name__}(...)) to build it on the writer rank only"
            )
        else:
            raise ValueError(
                f"callback {type(callback).__name__} does not declare its rank behaviour: set its class attribute "
                "distributed = 'all' (side-effect free, run on every rank) or 'writer' (writer rank only)"
            )
    return kept


def rng_restore(states: Optional[Sequence[Any]], rank: int, world_size: int) -> Optional[Mapping[str, Any]]:
    """This rank's saved RNG state from a distributed checkpoint."""
    if states is None:
        return None
    if len(states) != world_size:
        raise ValueError(f"the checkpoint holds RNG state for {len(states)} ranks, this world has {world_size}")
    return states[rank]
