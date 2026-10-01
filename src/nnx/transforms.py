"""Replayable model transformation recipes (FEAT-016).

A :class:`TransformRecipe` is an ordered, immutable list of topology
operations on explicit ``nn.Linear`` targets of ``model.net``:

- :func:`lora` — wrap each target in a :class:`~nnx.peft.LoRALinear`
  (base frozen, ``lora_A`` / ``lora_B`` trainable);
- :func:`low_rank` — replace each target by its rank-``k`` SVD factors
  ``nn.Sequential(Linear(in, k, bias=False), Linear(k, out))``.

Each operation records its ``id``, ``version``, ``targets`` and ``config``
and their order. A recipe needs a model NNx can rebuild — built-in
``Nets`` or a registered ``ModelSpec``; a runtime-only ``module=`` is
refused. :meth:`TransformRecipe.validate` checks a model against
the whole recipe — every target present, an ``nn.Linear``, registered once,
no target inside another operation's subtree, no version this NNx does not
know, no optimizer holding a parameter the recipe replaces — and mutates
nothing; :meth:`TransformRecipe.materialize` then either changes the given
model in place (``materialization="in_place"``, the default) or builds a
fresh, randomly initialized registered base from its descriptor
(``"fresh"``), and records the operations on the model it returns.

Recorded operations travel with every checkpoint (pickle and safetensors)
and Hub save as :class:`~nnx.nn.params.nn_checkpoint.NNCheckpointTransform`
entries, and fold into the run id of a run that trains them and into
``ExperimentManifest.for_model``.
``NNModel.from_checkpoint`` / ``from_pretrained`` replay them on a freshly
built base **before** loading the saved tensors: the low-rank factors are
allocated in their recorded shape and loaded — SVD is never rerun — and
LoRA wrappers are rebuilt around the fresh base layers. A raw state dict or
an adapter-only export carries no recipe and cannot rebuild the topology
alone.

Build an optimizer *after* materializing a recipe: an optimizer built
before holds the replaced parameters, and :func:`check_optimizer` (or
``validate(model, optimizers=...)``) refuses it with a rebuild
instruction. A recipe model's live topology must stay exactly its base plus
its recipe: ``NNModel.train`` and ``Trainer.train`` refuse anything else
(and ``NNModel.train`` still refuses unrecorded low-rank surgery on an
untransformed model, as before).
"""

from __future__ import annotations

import contextlib
import math
import weakref
from collections.abc import Iterable, Iterator, Mapping, Sequence
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
_Problem = tuple[Optional[int], str, Optional[str], str]
# Per model, the parameters its recipe made stale for an optimizer: the
# layers a low-rank operation replaced, or the source of a fresh
# materialization. Weak on both sides, so nothing is kept alive.
_SUPERSEDED: weakref.WeakKeyDictionary[Any, dict[int, weakref.ref]] = weakref.WeakKeyDictionary()


@contextlib.contextmanager
def _random_streams_kept() -> Iterator[None]:
    """Run a build whose initial values are discarded without moving the
    global random streams (and without creating a CUDA context)."""
    import torch

    from .seeding import _capture_rng_state, _restore_rng_state

    state = _capture_rng_state(None, cuda=torch.cuda.is_initialized())
    try:
        yield
    finally:
        _restore_rng_state(state, None)


class RecipeError(ValueError):
    """A recipe that does not fit a model, or a malformed operation.
    ``problems`` lists ``(operation index, operation id, target, reason)``;
    the message names each."""

    def __init__(self, problems: Sequence[_Problem]) -> None:
        self.problems = tuple(problems)
        super().__init__("; ".join(_describe(*problem) for problem in self.problems))


def _describe(index: Optional[int], op_id: str, target: Optional[str], reason: str) -> str:
    where = f"recipe operation {index} ({op_id})" if index is not None else f"recipe operation ({op_id})"
    return f"{where}, target {target!r}: {reason}" if target is not None else f"{where}: {reason}"


class _FrozenConfig(Mapping):
    """An operation's read-only config: hashable, picklable, comparable to
    a plain dict."""

    __slots__ = ("_items",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._items = tuple(sorted((str(key), _frozen(value)) for key, value in values.items()))

    def __getitem__(self, key: str) -> Any:
        for name, value in self._items:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (name for name, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(self._items)

    def __repr__(self) -> str:
        return repr(dict(self._items))

    def __reduce__(self) -> tuple[Any, ...]:
        return (_FrozenConfig, (dict(self._items),))


def _frozen(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenConfig(value)
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
        if isinstance(self.targets, (str, bytes)):
            raise TypeError(f"targets must be a sequence of module paths, got the string {self.targets!r}")
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "config", _FrozenConfig(dict(self.config)))

    def state(self) -> dict[str, Any]:
        """The JSON-ready form a checkpoint records (``options`` of its
        :class:`~nnx.nn.params.nn_checkpoint.NNCheckpointTransform`)."""
        return {"targets": list(self.targets), **_plain(self.config)}

    def checkpoint_transform(self) -> NNCheckpointTransform:
        from .nn.params.nn_checkpoint import NNCheckpointTransform

        return NNCheckpointTransform(name=self.id, version=self.version, options=self.state())

    @staticmethod
    def from_checkpoint_transform(transform: NNCheckpointTransform) -> TransformOp:
        """The operation a checkpoint recorded; malformed options raise a
        :class:`RecipeError`."""
        try:
            if not isinstance(transform.options, Mapping):
                raise TypeError(f"options must be a mapping, got {type(transform.options).__name__}")
            options = dict(transform.options)
            targets = options.pop("targets", ())
            if isinstance(targets, (str, bytes)) or not all(isinstance(t, str) for t in targets):
                raise TypeError("targets must be a list of module paths")
            return TransformOp(id=transform.name, targets=tuple(targets), config=options, version=transform.version)
        except (TypeError, ValueError) as error:
            raise RecipeError([(None, str(transform.name), None, f"malformed recorded options: {error}")]) from None


def lora(*targets: str, r: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> TransformOp:
    """Wrap each ``nn.Linear`` at ``targets`` in a
    :class:`~nnx.peft.LoRALinear` (``r``, ``alpha``, ``dropout`` as for
    ``apply_lora_to``). Targets are explicit module paths, not globs."""
    return TransformOp(id=LORA, targets=tuple(targets), config={"r": r, "alpha": alpha, "dropout": dropout})


def low_rank(*targets: str, rank: int, method: str = "svd") -> TransformOp:
    """Replace each ``nn.Linear`` at ``targets`` by its rank-``rank``
    factorization (``nnx.surgery.low_rank_factorize``)."""
    return TransformOp(id=LOW_RANK, targets=tuple(targets), config={"rank": rank, "method": method})


def _config_problems(index: Optional[int], op: TransformOp) -> list[_Problem]:
    """Operation-level problems, independent of any model."""
    problems: list[_Problem] = []

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
        if not isinstance(target, str):
            problem(f"targets must be explicit dotted module paths, got a {type(target).__name__}")
        elif not target or not set(target) <= _PATH_CHARACTERS:
            problem("targets must be explicit dotted module paths (no globs)", target)
    paths = [target for target in op.targets if isinstance(target, str)]
    if len(set(paths)) != len(paths):
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

    ``materialization`` says what :meth:`materialize` does: ``"in_place"``
    (the default) transforms the given model itself — its trained weights
    are what LoRA wraps and SVD factorizes; ``"fresh"`` builds a fresh,
    randomly initialized registered base from the given model's descriptor
    and transforms that, leaving the given model untouched. The operations
    are copied: changing the list they came from changes nothing here.
    """

    operations: tuple[TransformOp, ...]
    materialization: Literal["fresh", "in_place"] = "in_place"

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        object.__setattr__(self, "operations", operations)
        problems: list[_Problem] = []
        for index, op in enumerate(operations):
            if not isinstance(op, TransformOp):
                problems.append((index, type(op).__name__, None, "is not a TransformOp (build one with lora/low_rank)"))
            else:
                problems.extend(_config_problems(index, op))
        if self.materialization not in ("fresh", "in_place"):
            reason = f"materialization must be 'fresh' or 'in_place', got {self.materialization!r}"
            problems.append((None, "recipe", None, reason))
        if problems:
            raise RecipeError(problems)

    def checkpoint_transforms(self) -> tuple[NNCheckpointTransform, ...]:
        """The operations as checkpoint transforms, in order."""
        return tuple(op.checkpoint_transform() for op in self.operations)

    def validate(self, model: NNModel, *, optimizers: Iterable[torch.optim.Optimizer] = ()) -> None:
        """Check the whole recipe as :meth:`materialize` would — against
        ``model`` in place, or against the fresh base ``model`` describes —
        mutating nothing, and raise one :class:`RecipeError` naming every
        problem. ``optimizers`` are checked for an in-place materialization
        (a fresh base's parameters belong to no existing optimizer)."""
        if self.materialization == "fresh":
            _check_source(model, fresh=True)
            # Checked against the base materialize would build — built here
            # and discarded, leaving the global random streams untouched.
            with _random_streams_kept():
                base = _fresh_base(model)
            _validate(base.net, self.operations, (), ())
        else:
            _check_source(model, fresh=False)
            _validate(model.net, self.operations, _recorded_operations(model), list(optimizers))

    def materialize(self, model: NNModel, *, optimizers: Iterable[torch.optim.Optimizer] = ()) -> NNModel:
        """Validate the whole recipe (as :meth:`validate`), then apply it —
        to ``model`` itself (``"in_place"``, returned) or to a fresh
        registered base built from ``model``'s descriptor (``"fresh"``,
        returned). The operations are recorded on the returned model, so
        its checkpoints and Hub saves rebuild the same topology.
        Transactional: a failure leaves the model as it was."""
        if self.materialization == "fresh":
            _check_source(model, fresh=True)
            target = _fresh_base(model)  # built once, then checked as validate would
            _validate(target.net, self.operations, (), ())
        else:
            self.validate(model, optimizers=optimizers)
            target = model
        replaced: list[tuple[str, nn.Module]] = []
        # Building a LoRA wrapper freezes its base and sets modes, so a
        # rollback also restores every flag and mode as it was.
        flags = [(p, p.requires_grad) for p in target.net.parameters()]
        modes = [(m, m.training) for m in target.net.modules()]
        from .surgery._utils import get_module, set_module

        try:
            for op in self.operations:
                for path in op.targets:
                    original = get_module(target.net, path)
                    set_module(target.net, path, _build(op, original, allocate_only=False))
                    replaced.append((path, original))
        except BaseException:
            for path, original in reversed(replaced):
                set_module(target.net, path, original)
            for parameter, requires_grad in flags:
                parameter.requires_grad_(requires_grad)
            for module, training in modes:
                module.training = training
            raise
        target._topology_transforms = (*target._topology_transforms, *self.checkpoint_transforms())
        # What an optimizer built before this materialization holds and
        # check_optimizer refuses: the replaced low-rank layers, or the
        # whole source of a fresh base.
        superseded = list(model.net.parameters()) if target is not model else []
        superseded += [p for (_, original) in replaced if type(original) is nn.Linear for p in original.parameters()]
        live = {id(p) for p in target.net.parameters()}
        record = _SUPERSEDED.setdefault(target, {})
        for parameter in superseded:
            if id(parameter) not in live:
                record[id(parameter)] = weakref.ref(parameter)
        return target


def _recorded_operations(model: Any) -> tuple[TransformOp, ...]:
    """The recipe operations already recorded on a model; any other
    recorded transform (a train-end QAT conversion) refuses a recipe."""
    recorded = tuple(getattr(model, "_topology_transforms", ()))
    foreign = [t for t in recorded if t.name not in _VERSIONS]
    if foreign:
        reason = (
            f"the model already carries the topology transform {foreign[0].name!r}; apply recipes to a model before "
            "any train-end transform"
        )
        raise RecipeError([(None, "recipe", None, reason)])
    return tuple(TransformOp.from_checkpoint_transform(t) for t in recorded)


def _check_source(model: NNModel, *, fresh: bool) -> None:
    """A recipe needs a model NNx can rebuild; a fresh materialization
    builds a plain ``NNModel`` from the descriptor of an untransformed one."""
    from .models import RuntimeModule
    from .nn.nn_model import NNModel

    if isinstance(model.params.net, RuntimeModule):
        # Its descriptor pins the untransformed module and nothing can
        # rebuild it, so a recipe could never be replayed.
        reason = (
            f"{model.params.net} is a runtime-only module (module=), which nothing can rebuild: build the model "
            "from Nets or a registered ModelSpec, or use nnx.peft / nnx.surgery directly"
        )
        raise RecipeError([(None, "recipe", None, reason)])
    if not fresh:
        return
    problems: list[_Problem] = []
    if getattr(model, "_topology_transforms", ()):
        problems.append(
            (None, "recipe", None, "a fresh materialization starts from an untransformed model's descriptor")
        )
    if type(model) is not NNModel:
        problems.append(
            (
                None,
                "recipe",
                None,
                f"a fresh materialization builds an NNModel; use 'in_place' for a {type(model).__name__}",
            )
        )
    if problems:
        raise RecipeError(problems)


def _fresh_base(model: NNModel) -> NNModel:
    from .models import ModelSpec, _UnpackBatch
    from .nn.nn_model import NNModel

    adapter = getattr(model, "_batch_adapter", None)
    # A caller's batch adapter carries over; the default one is bound to the
    # old module, so the fresh model derives its own.
    kwargs = {"batch_adapter": adapter} if adapter is not None and not isinstance(adapter, _UnpackBatch) else {}
    if isinstance(model.params.net, ModelSpec):
        return NNModel(params=model.params, **kwargs)
    return NNModel(params=model.params, net_params=model.net_params, **kwargs)


def _registration_paths(net: nn.Module) -> dict[int, list[str]]:
    """Every dotted path each module object is registered under."""
    from .peft._targets import _registrations

    paths: dict[int, list[str]] = {}
    for path, _, _, child in _registrations(net):
        paths.setdefault(id(child), []).append(path)
    return paths


def _validate(
    net: nn.Module,
    operations: Sequence[TransformOp],
    recorded: Sequence[TransformOp],
    optimizers: Sequence[Any],
    *,
    indexed: bool = True,
) -> None:
    problems: list[_Problem] = []
    paths = _registration_paths(net)
    owners = [(op, path) for op in recorded for path in op.targets]
    for position, op in enumerate(operations):
        index = position if indexed else None
        problems.extend(_config_problems(index, op))
        for slot, path in enumerate(op.targets):
            if not isinstance(path, str):
                continue
            for other in op.targets[:slot]:
                if isinstance(other, str) and path != other and (_within(path, other) or _within(other, path)):
                    problems.append((index, op.id, path, f"overlaps target {other!r} of the same operation"))
            for owner_op, owner_path in owners:
                if _within(path, owner_path) or _within(owner_path, path):
                    reason = f"is inside the subtree {owner_path!r} another operation ({owner_op.id}) transforms"
                    problems.append((index, op.id, path, reason))
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
        for position, op in enumerate(operations):
            for path in op.targets:
                try:
                    module = net.get_submodule(path)
                except (AttributeError, TypeError):
                    continue
                if any(id(p) in held for p in module.parameters()):
                    reason = (
                        f"optimizer {number} holds this target's parameters — build the optimizer after "
                        "materializing the recipe (e.g. nnx.optimizers.build_optimizer(model.net, ...))"
                    )
                    problems.append((position if indexed else None, op.id, path, reason))
    if problems:
        raise RecipeError(problems)


def _build(op: TransformOp, linear: nn.Module, *, allocate_only: bool) -> nn.Module:
    """The replacement for one target: a fresh application, or — when
    rebuilding a saved model whose tensors are loaded next — just the
    recorded topology (no SVD, no meaningful values)."""
    assert isinstance(linear, nn.Linear)  # validated: only nn.Linear targets
    if op.id == LORA:
        from .peft.lora import LoRALinear

        def wrap() -> nn.Module:
            return LoRALinear(linear, r=op.config["r"], alpha=op.config["alpha"], dropout=op.config["dropout"])

        if not allocate_only:
            return wrap()
        # The adapter's initial values are overwritten by the saved ones, so
        # a rebuild leaves the global random streams where they were.
        with _random_streams_kept():
            return wrap()
    if not allocate_only:
        from .surgery.low_rank import low_rank_factorize

        return low_rank_factorize(linear, rank=op.config["rank"], method=op.config["method"])
    import torch

    from .surgery.low_rank import _allocate_factors

    factors = _allocate_factors(linear, op.config["rank"])
    with torch.no_grad():  # defined values (zeros) where a non-strict load leaves a factor out
        for parameter in factors.parameters():
            parameter.zero_()
    return factors


def _replayable(transform: NNCheckpointTransform) -> bool:
    return transform.name in _VERSIONS


def _recipe_transforms(transforms: Iterable[NNCheckpointTransform]) -> tuple[NNCheckpointTransform, ...]:
    """The recorded recipe operations among a model's transforms."""
    return tuple(t for t in transforms if _replayable(t))


def _replay(model: NNModel, transform: NNCheckpointTransform) -> None:
    """Rebuild one recorded operation's topology on a freshly built base
    before its saved tensors are loaded (``from_checkpoint`` /
    ``from_pretrained``)."""
    from .surgery._utils import get_module, set_module

    op = TransformOp.from_checkpoint_transform(transform)
    _validate(model.net, (op,), _recorded_operations(model), (), indexed=False)
    for path in op.targets:
        set_module(model.net, path, _build(op, get_module(model.net, path), allocate_only=True))
    model._topology_transforms = (*model._topology_transforms, transform)


_Shape = tuple[Optional[int], ...]


def _expected_state(
    base: Mapping[str, Any], transforms: Sequence[NNCheckpointTransform]
) -> Optional[dict[str, Optional[_Shape]]]:
    """``{key: shape}`` of a base's tensors after the recorded recipe
    ``transforms`` — a shape is ``None``, or a dimension of it is, where the
    base gives only names (a registered factory's reference keys); ``None``
    when one is not a recipe operation."""
    state: dict[str, Optional[_Shape]] = dict(base)
    for transform in transforms:
        if not _replayable(transform):
            return None
        op = TransformOp.from_checkpoint_transform(transform)
        for path in op.targets:
            weight = state.pop(f"{path}.weight", None)
            has_bias = f"{path}.bias" in state
            bias = state.pop(f"{path}.bias", None)
            out_features, in_features = weight if weight is not None else (None, None)
            if op.id == LORA:
                r = op.config["r"]
                state[f"{path}.lora_A"] = (r, in_features)
                state[f"{path}.lora_B"] = (out_features, r)
                state[f"{path}.base.weight"] = weight
                if has_bias:
                    state[f"{path}.base.bias"] = bias
            else:
                rank = op.config["rank"]
                # A dimension the base does not give stays None (a registered
                # factory's names only), but the recorded rank is checked.
                state[f"{path}.0.weight"] = (rank, in_features)
                state[f"{path}.1.weight"] = (out_features, rank)
                if has_bias:
                    state[f"{path}.1.bias"] = bias
    return state


def _topology_problems(
    net: nn.Module, base_state: Mapping[str, Optional[_Shape]], transforms: Sequence[NNCheckpointTransform]
) -> list[str]:
    """How ``net`` differs from its base plus the recorded recipe: tensor
    names, the shapes the recipe and the base fix, and each target's
    module and configuration."""
    import torch

    from .peft.lora import LoRALinear

    expected = _expected_state(base_state, transforms)
    if expected is None:
        return []
    actual = {key: tuple(value.shape) for key, value in net.state_dict().items() if isinstance(value, torch.Tensor)}
    problems: list[str] = []
    unexpected, missing = sorted(set(actual) - set(expected)), sorted(set(expected) - set(actual))
    if unexpected or missing:
        problems.append(f"unexpected tensors {unexpected[:5]}, missing {missing[:5]}")
    for key, shape in expected.items():
        live = actual.get(key)
        if live is None or shape is None:
            continue
        if len(shape) != len(live) or any(
            want is not None and want != got for want, got in zip(shape, live, strict=False)
        ):
            problems.append(f"{key} has shape {live}, its recipe and base give {shape}")
    for transform in _recipe_transforms(transforms):
        op = TransformOp.from_checkpoint_transform(transform)
        for path in op.targets:
            try:
                module = net.get_submodule(path)
            except AttributeError:
                continue  # reported as missing tensors
            if op.id == LORA:
                dropout = getattr(getattr(module, "lora_dropout", None), "p", 0.0)
                if not isinstance(module, LoRALinear) or (module.r, module.alpha, dropout) != (
                    op.config["r"],
                    op.config["alpha"],
                    op.config["dropout"],
                ):
                    problems.append(f"{path} is not the LoRA wrapper its recipe records ({dict(op.config)})")
            elif not (isinstance(module, nn.Sequential) and len(module) == 2):
                problems.append(f"{path} is not the low-rank factor pair its recipe records")
    return problems


def check_optimizer(model: NNModel, optimizer: torch.optim.Optimizer) -> None:
    """Refuse an optimizer built before the model's recipe: one holding
    parameters the recipe replaced (a low-rank operation's layers, or the
    model a fresh materialization started from), or holding a LoRA
    target's base weights but not the adapter built around them. An
    optimizer built afterwards — over every parameter, a subset, or with
    parameters outside ``model.net`` — passes."""
    from .surgery._utils import get_module

    held = [p for group in optimizer.param_groups for p in group["params"]]
    held_ids = {id(p) for p in held}
    record = _SUPERSEDED.get(model, {})
    stale = sum(1 for p in held if (ref := record.get(id(p))) is not None and ref() is p)
    unadapted: list[str] = []
    for transform in _recipe_transforms(getattr(model, "_topology_transforms", ())):
        op = TransformOp.from_checkpoint_transform(transform)
        if op.id != LORA:
            continue
        for path in op.targets:
            try:
                wrapper = get_module(model.net, path)
            except (AttributeError, KeyError):
                raise ValueError(
                    f"the model's topology differs from its recorded recipe: {op.id} target {path!r} is gone "
                    "(unrecorded surgery?)"
                ) from None
            base, adapter = (
                getattr(wrapper, "base", None),
                (getattr(wrapper, "lora_A", None), getattr(wrapper, "lora_B", None)),
            )
            holds_base = isinstance(base, nn.Module) and any(id(p) in held_ids for p in base.parameters())
            if holds_base and not any(isinstance(p, nn.Parameter) and id(p) in held_ids for p in adapter):
                unadapted.append(path)
    if stale or unadapted:
        raise ValueError(
            f"this optimizer is stale for the model's current topology ({stale} parameters it holds were replaced by its "
            f"recipe; it holds the frozen base but not the adapter of {unadapted}): build the optimizer after "
            "materializing the recipe, e.g. nnx.optimizers.build_optimizer(model.net, optim_params)"
        )
