"""Replayable model transformation recipes (FEAT-016).

A :class:`TransformRecipe` is an ordered, immutable list of topology
operations on explicit ``nn.Linear`` targets of ``model.net``:

- :func:`lora` — wrap each target in a :class:`~nnx.peft.LoRALinear`
  (base frozen, ``lora_A`` / ``lora_B`` trainable);
- :func:`low_rank` — replace each target by its rank-``k`` SVD factors
  ``nn.Sequential(Linear(in, k, bias=False), Linear(k, out))``.

Each operation records its ``id``, ``version``, ``targets`` and ``config``
and their order. :meth:`TransformRecipe.validate` checks a model against
the whole recipe — every target present, an ``nn.Linear``, registered once,
no target inside another operation's subtree, no version this NNx does not
know, no optimizer holding a parameter the recipe replaces — and mutates
nothing; :meth:`TransformRecipe.materialize` then either builds a fresh
registered base (``materialization="fresh"``) or changes the given model
in place (``"in_place"``), and records the operations on the model.

Recorded operations travel with every checkpoint (pickle and safetensors)
and Hub save as :class:`~nnx.nn.params.nn_checkpoint.NNCheckpointTransform`
entries, and ``NNModel.from_checkpoint`` / ``from_pretrained`` replay them
on a freshly built base **before** loading the saved tensors: the low-rank
factors are allocated in their recorded shape and loaded — SVD is never
rerun — and LoRA wrappers are rebuilt around the fresh base layers. A raw
state dict or an adapter-only export carries no recipe and cannot rebuild
the topology alone.

Build an optimizer *after* materializing a recipe: an optimizer built
before holds the replaced parameters, and :func:`check_optimizer` (or
``validate(model, optimizers=...)``) refuses it with a rebuild
instruction. Surgery done outside a recipe stays unrecorded and is still
refused before persistent training.
"""

from __future__ import annotations

import math
import types
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

from torch import nn

if TYPE_CHECKING:
    import torch

    from .nn.nn_model import NNModel
    from .nn.params.nn_checkpoint import NNCheckpointTransform

__all__ = [
    "RecipeError",
    "TransformOp",
    "TransformRecipe",
    "check_optimizer",
    "lora",
    "low_rank",
]

LORA = "lora"
LOW_RANK = "low_rank"
_VERSIONS = {LORA: (1,), LOW_RANK: (1,)}
_PATH_CHARACTERS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.")


class RecipeError(ValueError):
    """A recipe that does not fit a model, or a malformed operation.
    ``problems`` lists ``(operation index, operation id, target, reason)``;
    the message names each."""

    def __init__(self, problems: Sequence[tuple[Optional[int], str, Optional[str], str]]) -> None:
        self.problems = tuple(problems)
        lines = [_describe(index, op_id, target, reason) for index, op_id, target, reason in self.problems]
        super().__init__("; ".join(lines))


def _describe(index: Optional[int], op_id: str, target: Optional[str], reason: str) -> str:
    where = f"recipe operation {index} ({op_id})" if index is not None else f"recipe operation ({op_id})"
    return f"{where}, target {target!r}: {reason}" if target is not None else f"{where}: {reason}"


def _frozen(value: Any) -> Any:
    if isinstance(value, Mapping):
        return types.MappingProxyType({str(k): _frozen(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(v) for v in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


@dataclass(frozen=True)
class TransformOp:
    """One recorded operation: ``id`` (``"lora"`` or ``"low_rank"``),
    ``version``, the explicit ``targets`` (dotted paths under ``model.net``)
    and an immutable ``config``. Build one with :func:`lora` or
    :func:`low_rank`."""

    id: str
    targets: tuple[str, ...]
    config: Mapping[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "config", _frozen(dict(self.config)))

    def state(self) -> dict[str, Any]:
        """The JSON-ready form a checkpoint records (``options`` of its
        :class:`~nnx.nn.params.nn_checkpoint.NNCheckpointTransform`)."""
        return {"targets": list(self.targets), **_plain(self.config)}

    def checkpoint_transform(self) -> NNCheckpointTransform:
        from .nn.params.nn_checkpoint import NNCheckpointTransform

        return NNCheckpointTransform(name=self.id, version=self.version, options=self.state())

    @staticmethod
    def from_checkpoint_transform(transform: NNCheckpointTransform) -> TransformOp:
        options = dict(transform.options)
        targets = options.pop("targets", ())
        return TransformOp(id=transform.name, targets=tuple(targets), config=options, version=transform.version)


def lora(*targets: str, r: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> TransformOp:
    """Wrap each ``nn.Linear`` at ``targets`` in a
    :class:`~nnx.peft.LoRALinear` (``r``, ``alpha``, ``dropout`` as for
    ``apply_lora_to``). Targets are explicit module paths, not globs."""
    return TransformOp(id=LORA, targets=tuple(targets), config={"r": r, "alpha": alpha, "dropout": dropout})


def low_rank(*targets: str, rank: int, method: str = "svd") -> TransformOp:
    """Replace each ``nn.Linear`` at ``targets`` by its rank-``rank``
    factorization (``nnx.surgery.low_rank_factorize``)."""
    return TransformOp(id=LOW_RANK, targets=tuple(targets), config={"rank": rank, "method": method})


def _config_problems(index: Optional[int], op: TransformOp) -> list[tuple[Optional[int], str, Optional[str], str]]:
    """Operation-level problems, independent of any model."""
    problems: list[tuple[Optional[int], str, Optional[str], str]] = []

    def problem(reason: str, target: Optional[str] = None) -> None:
        problems.append((index, op.id, target, reason))

    if op.id not in _VERSIONS:
        problem(f"unknown operation; this NNx replays {sorted(_VERSIONS)}")
        return problems
    if isinstance(op.version, bool) or op.version not in _VERSIONS[op.id]:
        problem(f"unknown version {op.version!r}; this NNx replays versions {list(_VERSIONS[op.id])}")
        return problems
    if not op.targets:
        problem("names no target")
    for target in op.targets:
        if not isinstance(target, str) or not target or not set(target) <= _PATH_CHARACTERS:
            problem(
                "targets must be explicit dotted module paths (no globs)", target if isinstance(target, str) else None
            )
    if len(set(op.targets)) != len(op.targets):
        problem("names a target twice")
    expected = {"r", "alpha", "dropout"} if op.id == LORA else {"rank", "method"}
    if set(op.config) != expected:
        problem(f"config must have exactly {sorted(expected)}, got {sorted(op.config)}")
        return problems
    if op.id == LORA:
        r, alpha, dropout = op.config["r"], op.config["alpha"], op.config["dropout"]
        if isinstance(r, bool) or not isinstance(r, int) or r < 1:
            problem(f"r must be a positive integer, got {r!r}")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
            problem(f"alpha must be a finite positive number, got {alpha!r}")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            problem(f"dropout must be in [0, 1), got {dropout!r}")
    else:
        rank, method = op.config["rank"], op.config["method"]
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            problem(f"rank must be a positive integer, got {rank!r}")
        if method != "svd":
            problem(f"method must be 'svd', got {method!r}")
    return problems


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root + ".")


@dataclass(frozen=True)
class TransformRecipe:
    """An ordered, immutable list of :class:`TransformOp`.

    ``materialization`` says what :meth:`materialize` does: ``"fresh"``
    builds a fresh registered base from the given model's descriptor and
    transforms that (the given model is untouched); ``"in_place"``
    transforms the given model itself. The operations are copied: changing
    the list they came from changes nothing here.
    """

    operations: tuple[TransformOp, ...]
    materialization: Literal["fresh", "in_place"] = "fresh"

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        object.__setattr__(self, "operations", operations)
        problems = []
        for index, op in enumerate(operations):
            if not isinstance(op, TransformOp):
                problems.append((index, type(op).__name__, None, "is not a TransformOp (build one with lora/low_rank)"))
            else:
                problems.extend(_config_problems(index, op))
        if self.materialization not in ("fresh", "in_place"):
            problems.append(
                (None, "recipe", None, f"materialization must be 'fresh' or 'in_place', got {self.materialization!r}")
            )
        if problems:
            raise RecipeError(problems)

    def checkpoint_transforms(self) -> tuple[NNCheckpointTransform, ...]:
        """The operations as checkpoint transforms, in order."""
        return tuple(op.checkpoint_transform() for op in self.operations)

    def validate(self, model: NNModel, *, optimizers: Iterable[torch.optim.Optimizer] = ()) -> None:
        """Check the whole recipe against ``model`` — mutating nothing —
        and raise one :class:`RecipeError` naming every problem."""
        _validate(model.net, self.operations, _recorded_operations(model), list(optimizers))

    def materialize(self, model: NNModel, *, optimizers: Iterable[torch.optim.Optimizer] = ()) -> NNModel:
        """Apply the recipe — to a fresh registered base built from
        ``model``'s descriptor (``"fresh"``, returned) or to ``model``
        itself (``"in_place"``, returned) — after validating it whole; the
        operations are recorded on the returned model, so its checkpoints
        and Hub saves rebuild the same topology. Transactional: a failure
        leaves the model as it was."""
        optimizers = list(optimizers)
        if self.materialization == "fresh":
            if _recorded_operations(model) or getattr(model, "_topology_transforms", ()):
                raise RecipeError(
                    [(None, "recipe", None, "a fresh materialization starts from an untransformed model's descriptor")]
                )
            target = _fresh_base(model)
        else:
            target = model
        _validate(target.net, self.operations, _recorded_operations(target), optimizers)
        replaced: list[tuple[str, nn.Module]] = []
        # Building a LoRA wrapper freezes its base and sets modes, so a
        # rollback also restores every flag and mode as it was.
        flags = [(p, p.requires_grad) for p in target.net.parameters()]
        modes = [(m, m.training) for m in target.net.modules()]
        try:
            for op in self.operations:
                for path in op.targets:
                    from .surgery._utils import get_module, set_module

                    original = get_module(target.net, path)
                    set_module(target.net, path, _build(op, original, allocate_only=False))
                    replaced.append((path, original))
        except BaseException:
            from .surgery._utils import set_module

            for path, original in reversed(replaced):
                set_module(target.net, path, original)
            for parameter, requires_grad in flags:
                parameter.requires_grad_(requires_grad)
            for module, training in modes:
                module.training = training
            raise
        target._topology_transforms = (*target._topology_transforms, *self.checkpoint_transforms())
        return target


def _recorded_operations(model: Any) -> tuple[TransformOp, ...]:
    """The recipe operations already recorded on a model; any other
    recorded transform (a train-end QAT conversion) refuses a recipe."""
    recorded = tuple(getattr(model, "_topology_transforms", ()))
    foreign = [t for t in recorded if t.name not in _VERSIONS]
    if foreign:
        raise RecipeError(
            [
                (
                    None,
                    "recipe",
                    None,
                    f"the model already carries the topology transform {foreign[0].name!r}; apply recipes to a model "
                    "before any train-end transform",
                )
            ]
        )
    return tuple(TransformOp.from_checkpoint_transform(t) for t in recorded)


def _fresh_base(model: NNModel) -> NNModel:
    from .models import ModelSpec, RuntimeModule
    from .nn.enum.nets import Nets

    net = model.params.net
    if isinstance(net, RuntimeModule):
        raise RecipeError(
            [(None, "recipe", None, f"a fresh materialization needs a registered base; {net} is runtime-only")]
        )
    if isinstance(net, ModelSpec):
        return type(model)(params=model.params)
    assert isinstance(net, Nets)
    return type(model)(params=model.params, net_params=model.net_params)


def _registration_paths(net: nn.Module) -> dict[int, list[str]]:
    """Every dotted path each module object is registered under."""
    paths: dict[int, list[str]] = {}

    def walk(module: nn.Module, prefix: str, seen: set[int]) -> None:
        for name, child in module._modules.items():
            if child is None:
                continue
            path = f"{prefix}.{name}" if prefix else name
            paths.setdefault(id(child), []).append(path)
            if id(child) not in seen:
                walk(child, path, seen | {id(child)})

    walk(net, "", {id(net)})
    return paths


def _validate(
    net: nn.Module,
    operations: Sequence[TransformOp],
    recorded: Sequence[TransformOp],
    optimizers: Sequence[Any],
) -> None:
    problems: list[tuple[Optional[int], str, Optional[str], str]] = []
    paths = _registration_paths(net)
    owners = [(op, path) for op in recorded for path in op.targets]
    for index, op in enumerate(operations):
        problems.extend(_config_problems(index, op))
        for position, path in enumerate(op.targets):
            if not isinstance(path, str):
                continue
            for other in op.targets[:position]:
                if isinstance(other, str) and path != other and (_within(path, other) or _within(other, path)):
                    problems.append((index, op.id, path, f"overlaps target {other!r} of the same operation"))
            for owner_op, owner_path in owners:
                if _within(path, owner_path) or _within(owner_path, path):
                    problems.append(
                        (
                            index,
                            op.id,
                            path,
                            f"is inside the subtree {owner_path!r} another operation ({owner_op.id}) transforms",
                        )
                    )
            try:
                module = net.get_submodule(path)
            except AttributeError:
                problems.append((index, op.id, path, "no such module in model.net"))
                continue
            if type(module) is not nn.Linear:
                problems.append((index, op.id, path, f"is a {type(module).__name__}; only nn.Linear is supported"))
                continue
            aliases = paths.get(id(module), [])
            if len(aliases) > 1:
                problems.append((index, op.id, path, f"the same nn.Linear is registered under {aliases}"))
            if op.id == LOW_RANK and isinstance(op.config.get("rank"), int):
                limit = min(module.in_features, module.out_features)
                if op.config["rank"] > limit:
                    problems.append((index, op.id, path, f"rank {op.config['rank']} exceeds min(in, out) = {limit}"))
        owners.extend((op, path) for path in op.targets if isinstance(path, str))
    for number, optimizer in enumerate(optimizers):
        held = {id(p) for group in optimizer.param_groups for p in group["params"]}
        for index, op in enumerate(operations):
            for path in op.targets:
                try:
                    module = net.get_submodule(path)
                except (AttributeError, TypeError):
                    continue
                if any(id(p) in held for p in module.parameters()):
                    problems.append(
                        (
                            index,
                            op.id,
                            path,
                            f"optimizer {number} holds this target's parameters — build the optimizer after "
                            "materializing the recipe (e.g. nnx.optimizers.build_optimizer(model.net, ...))",
                        )
                    )
    if problems:
        raise RecipeError(problems)


def _build(op: TransformOp, linear: nn.Module, *, allocate_only: bool) -> nn.Module:
    """The replacement for one target: a fresh application, or — when
    rebuilding a saved model whose tensors are loaded next — just the
    recorded topology (no SVD, no meaningful values)."""
    assert isinstance(linear, nn.Linear)  # validated: only nn.Linear targets
    if op.id == LORA:
        from .peft.lora import LoRALinear

        return LoRALinear(linear, r=op.config["r"], alpha=op.config["alpha"], dropout=op.config["dropout"])
    rank = op.config["rank"]
    if not allocate_only:
        from .surgery.low_rank import low_rank_factorize

        return low_rank_factorize(linear, rank=rank, method=op.config["method"])
    from typing import cast

    from torch.nn.utils import skip_init

    from .surgery._utils import copy_param_roles

    weight = linear.weight
    down = cast(
        nn.Linear, skip_init(nn.Linear, linear.in_features, rank, bias=False, dtype=weight.dtype, device=weight.device)
    )
    up = cast(
        nn.Linear,
        skip_init(
            nn.Linear, rank, linear.out_features, bias=linear.bias is not None, dtype=weight.dtype, device=weight.device
        ),
    )
    copy_param_roles(linear, down)
    copy_param_roles(linear, up)
    factors = nn.Sequential(down, up)
    factors.train(linear.training)
    return factors


def _replayable(transform: NNCheckpointTransform) -> bool:
    return transform.name in _VERSIONS


def _replay(model: NNModel, transform: NNCheckpointTransform) -> None:
    """Rebuild one recorded operation's topology on a freshly built base
    before its saved tensors are loaded (``from_checkpoint`` /
    ``from_pretrained``)."""
    from .surgery._utils import get_module, set_module

    op = TransformOp.from_checkpoint_transform(transform)
    _validate(model.net, (op,), _recorded_operations(model), ())
    for path in op.targets:
        set_module(model.net, path, _build(op, get_module(model.net, path), allocate_only=True))
    model._topology_transforms = (*model._topology_transforms, transform)


def _expected_keys(base_keys: Iterable[str], transforms: Sequence[NNCheckpointTransform]) -> Optional[set[str]]:
    """The tensor keys a base with ``base_keys`` has after the recorded
    recipe ``transforms`` (``None`` when one is not a recipe operation)."""
    keys = set(base_keys)
    for transform in transforms:
        if not _replayable(transform):
            return None
        op = TransformOp.from_checkpoint_transform(transform)
        for path in op.targets:
            bias = f"{path}.bias" in keys
            keys -= {f"{path}.weight", f"{path}.bias"}
            if op.id == LORA:
                keys |= {f"{path}.lora_A", f"{path}.lora_B", f"{path}.base.weight"}
                keys |= {f"{path}.base.bias"} if bias else set()
            else:
                keys |= {f"{path}.0.weight", f"{path}.1.weight"}
                keys |= {f"{path}.1.bias"} if bias else set()
    return keys


def check_optimizer(model: NNModel, optimizer: torch.optim.Optimizer) -> None:
    """Refuse an optimizer that no longer matches ``model.net`` — one built
    before a recipe replaced layers holds parameters the model no longer
    has, and misses the new ones — with the instruction to rebuild it."""
    live = {id(p) for p in model.net.parameters()}
    trainable = {id(p) for p in model.net.parameters() if p.requires_grad}
    held = {id(p) for group in optimizer.param_groups for p in group["params"]}
    stale = held - live
    missing = trainable - held
    if stale or missing:
        raise ValueError(
            f"this optimizer is stale for the model's current topology ({len(stale)} parameters it holds are no longer "
            f"in model.net, {len(missing)} trainable parameters are missing): build the optimizer after materializing "
            "the recipe, e.g. nnx.optimizers.build_optimizer(model.net, optim_params)"
        )
