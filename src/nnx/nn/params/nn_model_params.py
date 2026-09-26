from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

from ..enum.devices import Devices
from ..enum.losses import Losses
from ..enum.nets import Nets

if TYPE_CHECKING:
    from ...models import ModelSpec, RuntimeModule
    from ...tasks import TaskSpec


@dataclass(frozen=True, kw_only=True, slots=True)
class NNModelParams:
    # What builds the network (FEAT-006): a built-in ``Nets`` member (with
    # ``NNParams``), a registered ``nnx.models.ModelSpec``, or — filled in by
    # ``NNModel(module=...)`` when left ``None`` — the ``RuntimeModule``
    # descriptor of a caller-owned module.
    net: Optional[Union[Nets, ModelSpec, RuntimeModule]] = None
    device: Devices = Devices.CPU
    loss: Losses = Losses.CROSS_ENTROPY

    # Opt-in fp16 autocast + GradScaler in train(). Only effective on CUDA;
    # silently bypassed on CPU/MPS where torch.cuda.amp is a no-op or unavailable.
    mixed_precision: bool = False

    # Opt-in task declaration (FEAT-002): categorical / multilabel /
    # regression adapters own validation, masking, loss units, decoding and
    # metrics. None keeps the legacy classification path and serialization.
    task: Optional[TaskSpec] = None

    def __post_init__(self) -> None:
        if self.net is not None and not isinstance(self.net, Nets):
            from ...models import ModelSpec, RuntimeModule

            if not isinstance(self.net, (ModelSpec, RuntimeModule)):
                raise TypeError(
                    "NNModelParams.net must be a Nets member, an nnx.models.ModelSpec or a RuntimeModule, "
                    f"got {type(self.net).__name__}"
                )
        if self.task is not None:
            from ...tasks import TaskSpec

            if not isinstance(self.task, TaskSpec):
                raise TypeError(f"NNModelParams.task must be a TaskSpec or None, got {type(self.task).__name__}")

    def __str__(self) -> str:
        task = f", task={self.task}" if self.task is not None else ""
        return f"[net={self.net}, device={self.device}, loss={self.loss}, mixed_precision={self.mixed_precision}{task}]"

    def is_valid(self) -> bool:
        return self.net is not None and self.device is not None and self.loss is not None

    @property
    def builtin(self) -> bool:
        """Whether the net is a built-in ``Nets`` member (FEAT-006)."""
        return isinstance(self.net, Nets)

    def state(self) -> dict:
        if self.net is None:
            raise ValueError("NNModelParams.net is unset; NNModel(module=...) fills it in for a wrapped module")
        # Built-in nets keep their plain string (run ids unchanged); a
        # registered or runtime descriptor is a mapping tagged by `kind`.
        net: Any = str(self.net) if isinstance(self.net, Nets) else self.net.state()
        d: dict[str, object] = dict(
            net=net,
            loss=str(self.loss),
            device=str(self.device),
        )
        # `mixed_precision` is omitted from state() when False so a
        # NNModelParams without AMP enabled hashes to the same run.id
        # as before this field existed. Same omit-when-default invariant
        # as NNTrainParams.seed / NNOptimParams.param_groups.
        if self.mixed_precision:
            d["mixed_precision"] = True
        # `task` (FEAT-002) follows the same rule: absent for legacy models, so
        # their state() and run.id are unchanged; a versioned mapping when set.
        if self.task is not None:
            d["task"] = self.task.state()
        return d

    @staticmethod
    def from_state(state: dict) -> NNModelParams:
        task_state = state.get("task")
        task = None
        if task_state is not None:
            from ...tasks import TaskSpec

            task = TaskSpec.from_state(task_state)
        raw_net = state["net"]
        if isinstance(raw_net, str):
            net: Union[Nets, ModelSpec, RuntimeModule] = Nets(raw_net)
        else:
            from ...models import descriptor_from_state

            net = descriptor_from_state(raw_net)
        return NNModelParams(
            net=net,
            loss=Losses(state["loss"]),
            device=Devices(state["device"]),
            mixed_precision=state.get("mixed_precision", False),
            task=task,
        )
