"""Optimizer construction: the shared build hook and registered factories.

Every NNx training entry point (``NNModel.train`` and ``Trainer.train``)
builds its optimizers through :func:`build_optimizer`, so built-in
:class:`~nnx.nn.params.nn_optim_params.NNOptimParams` variants and
registered factories follow one path with one parameter-ownership rule.

A *registered optimizer factory* lets a run use an optimizer NNx does not
ship without disguising it as a built-in name:

1. Register a callable under a stable ``(id, version)`` pair::

       def lion(param_groups, config):
           return MyLion(param_groups, betas=tuple(config["betas"]))

       register_optimizer_factory("acme.lion", 1, lion)

2. Reference it from a run with :class:`NNOptimFactoryParams`, whose
   :class:`OptimizerFactorySpec` holds only the id, version and a
   JSON-like ``config`` — never the callable itself::

       NNOptimFactoryParams(
           factory=OptimizerFactorySpec(id="acme.lion", version=1, config={"betas": [0.9, 0.99]}),
           max_lr=3e-4,
           weight_decay=0.1,
       )

Lifecycle and guarantees:

* **Resolution** happens before any run directory exists: an unknown id or
  version fails ``train()`` up front. Resolution is a registry lookup —
  NNx never imports a module or evaluates a string to find a factory.
* **Construction.** The factory is called exactly once per optimizer
  with the *resolved* parameter groups — each a dict with ``"params"``,
  ``"lr"`` and ``"weight_decay"``, holding exactly the parameters a
  built-in optimizer would receive (every parameter in one group when
  ``param_groups`` is None; otherwise the ``build_param_groups`` buckets,
  frozen parameters dropped and ``strict_param_groups`` ownership
  applied) — and a read-only view of ``config``. It must return a
  ``torch.optim.Optimizer`` holding exactly those parameters: none
  missing, none duplicated, none foreign. Anything else raises before the
  first epoch. Construction happens before the run is reserved, so a
  factory may be called for a run that is then refused (for example an
  existing run without ``overwrite_existing``): keep factories
  side-effect-free constructors.
* **Persistence.** The run's ``run.yaml`` records the spec (id, version,
  config), so run metadata — ``NNRun.load``, ``str(run)``, the notebook
  view — reloads offline without the factory registered and without
  running factory code. Warm resume (``resume_from_run_id``) additionally
  requires the same id / version / config and the same ordered
  named-parameter topology as the checkpoint; any change raises.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
from torch import nn

from ._config import _SLUG, _freeze_config, _thaw_config
from ._validation import require_count
from .nn.enum.optims import resolve_param_groups
from .nn.params.nn_optim_params import NNOptimParams, _validate_optim_param_groups, _validate_optim_scalars

if TYPE_CHECKING:
    from .finetune.param_groups import NNParamGroupSpec

__all__ = [
    "NNOptimFactoryParams",
    "OptimizerFactory",
    "OptimizerFactorySpec",
    "build_optimizer",
    "optim_params_from_state",
    "register_optimizer_factory",
    "registered_optimizer_factories",
    "resolve_optimizer_factory",
    "unregister_optimizer_factory",
]

OptimizerFactory = Callable[[list[dict[str, Any]], Mapping[str, Any]], torch.optim.Optimizer]
"""``factory(param_groups, config) -> torch.optim.Optimizer``."""

_FACTORY_ID = _SLUG
_REGISTRY: dict[tuple[str, int], OptimizerFactory] = {}


def _require_factory_id(value: object, *, owner: str) -> str:
    if not isinstance(value, str) or _FACTORY_ID.fullmatch(value) is None:
        raise ValueError(
            f"{owner} requires id to be a non-empty string of letters, digits, '_', '.' or '-' "
            f"(starting with a letter or digit), got {value!r}"
        )
    return value


def _require_factory_version(value: object, *, owner: str) -> int:
    return require_count(value, "version", owner=owner, minimum=1)


class OptimizerFactorySpec:
    """Stable, serializable reference to a registered optimizer factory.

    ``id`` and ``version`` name the factory in the registry; ``config`` is
    handed to it read-only and must be JSON-like — ``None``, ``bool``,
    ``int``, finite ``float``, ``str``, lists (exposed as tuples) and
    string-keyed mappings. The spec never holds the callable, so it
    round-trips through ``run.yaml`` and reloads without the factory being
    registered or imported. Immutable; equal specs have equal ``state()``.
    """

    __slots__ = ("id", "version", "config")
    id: str
    version: int
    config: Mapping[str, Any]

    def __init__(self, id: str, version: int, config: Optional[Mapping[str, Any]] = None) -> None:
        owner = "OptimizerFactorySpec"
        object.__setattr__(self, "id", _require_factory_id(id, owner=owner))
        object.__setattr__(self, "version", _require_factory_version(version, owner=owner))
        if config is None:
            config = {}
        if not isinstance(config, Mapping):
            raise TypeError(f"OptimizerFactorySpec config must be a str-keyed mapping, got {type(config).__name__}")
        object.__setattr__(self, "config", _freeze_config(config, ""))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"OptimizerFactorySpec is immutable; cannot set {name!r}")

    def __reduce__(self):
        return (OptimizerFactorySpec, (self.id, self.version, _thaw_config(self.config)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OptimizerFactorySpec):
            return NotImplemented
        return _canonical_factory_state(self.state()) == _canonical_factory_state(other.state())

    def __hash__(self) -> int:
        return hash(_canonical_factory_state(self.state()))

    def __repr__(self) -> str:
        return f"OptimizerFactorySpec(id={self.id!r}, version={self.version}, config={_thaw_config(self.config)!r})"

    def __str__(self) -> str:
        return f"{self.id}@v{self.version}"

    def state(self) -> dict[str, Any]:
        return {"id": self.id, "version": self.version, "config": _thaw_config(self.config)}

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> OptimizerFactorySpec:
        return OptimizerFactorySpec(id=state["id"], version=state["version"], config=state.get("config") or {})


def _canonical_factory_state(state: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Type-strict canonical form of a spec state (``1``, ``1.0`` and
    ``True`` stay distinct, as they do in ``run.yaml`` and the run id).
    Used for spec equality / hashing and for the resume identity check."""
    return None if state is None else json.dumps(state, sort_keys=True)


def register_optimizer_factory(
    id: str,
    version: int,
    factory: OptimizerFactory,
    *,
    replace: bool = False,
) -> None:
    """Register ``factory`` under ``(id, version)`` for this process.

    ``factory(param_groups, config)`` receives the resolved parameter
    groups and the spec's read-only config and must return a
    ``torch.optim.Optimizer`` over exactly those parameters. Registering an
    existing ``(id, version)`` raises unless ``replace=True``; bump
    ``version`` instead when the factory's behaviour changes, so resumed
    runs can tell the difference.
    """
    owner = "register_optimizer_factory"
    key = (_require_factory_id(id, owner=owner), _require_factory_version(version, owner=owner))
    if not callable(factory):
        raise TypeError(f"register_optimizer_factory requires a callable factory, got {type(factory).__name__}")
    if key in _REGISTRY and not replace:
        raise ValueError(
            f"optimizer factory {key[0]}@v{key[1]} is already registered; pass replace=True to "
            "override it, or register a new version"
        )
    _REGISTRY[key] = factory


def unregister_optimizer_factory(id: str, version: int) -> bool:
    """Remove ``(id, version)`` from the registry; returns whether it was
    registered. Run metadata that references it still loads — only
    training with it needs the registration."""
    return _REGISTRY.pop((id, version), None) is not None


def registered_optimizer_factories() -> tuple[tuple[str, int], ...]:
    """Sorted ``(id, version)`` pairs currently registered."""
    return tuple(sorted(_REGISTRY))


def resolve_optimizer_factory(spec: OptimizerFactorySpec) -> OptimizerFactory:
    """Look ``spec`` up in the registry. Never imports anything: an
    unregistered id or version raises ``ValueError`` naming what is
    registered."""
    if not isinstance(spec, OptimizerFactorySpec):
        raise TypeError(f"expected an OptimizerFactorySpec, got {type(spec).__name__}")
    factory = _REGISTRY.get((spec.id, spec.version))
    if factory is not None:
        return factory
    versions = sorted(version for fid, version in _REGISTRY if fid == spec.id)
    if versions:
        raise ValueError(
            f"optimizer factory {spec.id!r} has no registered version {spec.version} "
            f"(registered: {', '.join(f'v{v}' for v in versions)})"
        )
    raise ValueError(
        f"optimizer factory {spec.id!r} is not registered; call "
        f"register_optimizer_factory({spec.id!r}, {spec.version}, factory) before training"
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class NNOptimFactoryParams:
    """Optimizer config backed by a registered factory.

    The registered-variant counterpart of
    :class:`~nnx.nn.params.nn_optim_params.NNOptimParams`: it shares
    ``max_lr`` / ``weight_decay`` (defaults for every resolved group),
    ``grad_clip_norm``, ``accumulate_grad_batches`` and ``param_groups``
    with the same validation, but names its optimizer through ``factory``
    instead of an :class:`~nnx.nn.enum.optims.Optims` value and carries no
    ``momentum`` / ``name`` fields. Its ``state()`` holds a ``factory``
    entry, which is how run decoding tells the two variants apart.
    """

    factory: OptimizerFactorySpec
    max_lr: float
    weight_decay: float = 0.0
    grad_clip_norm: Optional[float] = None
    accumulate_grad_batches: int = 1
    param_groups: Optional[list[NNParamGroupSpec]] = field(default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.factory, OptimizerFactorySpec):
            hint = (
                " — a callable cannot be serialized into run.yaml; register it with "
                "register_optimizer_factory(id, version, factory) and pass "
                "OptimizerFactorySpec(id=..., version=...)"
                if callable(self.factory)
                else ""
            )
            raise TypeError(
                f"NNOptimFactoryParams requires factory to be an OptimizerFactorySpec, "
                f"got {type(self.factory).__name__}{hint}"
            )
        _validate_optim_scalars(self, "NNOptimFactoryParams")
        _validate_optim_param_groups(self)

    def __str__(self) -> str:
        return (
            f"[factory={self.factory}, max_lr={self.max_lr:1.0e}, weight_decay={self.weight_decay:1.0e}, "
            f"grad_clip={self.grad_clip_norm}, accum={self.accumulate_grad_batches}]"
        )

    def is_valid(self) -> bool:
        """Always True: every field was validated at construction (the
        built-in variant's momentum-shape check has no counterpart here).
        Whether the factory is registered is checked when training starts."""
        return True

    def state(self) -> dict[str, Any]:
        d: dict[str, Any] = dict(factory=self.factory.state(), max_lr=self.max_lr, weight_decay=self.weight_decay)
        if self.grad_clip_norm is not None:
            d["grad_clip_norm"] = self.grad_clip_norm
        if self.accumulate_grad_batches != 1:
            d["accumulate_grad_batches"] = self.accumulate_grad_batches
        if self.param_groups is not None:
            d["param_groups"] = [g.state() for g in self.param_groups]
        return d

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> NNOptimFactoryParams:
        from .finetune.param_groups import NNParamGroupSpec

        raw_pg = state.get("param_groups")
        return NNOptimFactoryParams(
            factory=OptimizerFactorySpec.from_state(state["factory"]),
            max_lr=state["max_lr"],
            weight_decay=state.get("weight_decay", 0.0),
            grad_clip_norm=state.get("grad_clip_norm"),
            accumulate_grad_batches=state.get("accumulate_grad_batches", 1),
            param_groups=[NNParamGroupSpec.from_state(g) for g in raw_pg] if raw_pg is not None else None,
        )


AnyOptimParams = Union[NNOptimParams, NNOptimFactoryParams]


def optim_params_from_state(state: Mapping[str, Any]) -> AnyOptimParams:
    """Decode a serialized optimizer config into the right variant.

    A ``factory`` entry is the discriminator for
    :class:`NNOptimFactoryParams`; anything else is a built-in
    :class:`NNOptimParams` (every config written before factories
    existed). Decoding never resolves or runs a factory.
    """
    if "factory" in state:
        return NNOptimFactoryParams.from_state(state)
    return NNOptimParams.from_state(dict(state))


def _resolved_groups(net: nn.Module, params: NNOptimFactoryParams, *, strict: bool) -> list[dict[str, Any]]:
    """The groups a built-in optimizer would receive (the shared
    :func:`~nnx.nn.enum.optims.resolve_param_groups` rule), each with an
    explicit ``lr`` / ``weight_decay`` and its own ``params`` list."""
    groups = resolve_param_groups(
        net,
        None if params.param_groups is None else list(params.param_groups),
        lr_start=params.max_lr,
        weight_decay=params.weight_decay,
        strict_param_groups=strict,
    )
    return [
        {
            **group,
            "params": list(group["params"]),
            "lr": group.get("lr", params.max_lr),
            "weight_decay": group.get("weight_decay", params.weight_decay),
        }
        for group in groups
    ]


def _check_factory_result(
    result: object, expected: set[int], net: nn.Module, spec: OptimizerFactorySpec
) -> torch.optim.Optimizer:
    if not isinstance(result, torch.optim.Optimizer):
        raise TypeError(f"optimizer factory {spec} returned {type(result).__name__}, expected a torch.optim.Optimizer")
    names = {id(param): name for name, param in net.named_parameters()}
    seen: set[int] = set()
    duplicate: list[str] = []
    foreign: list[str] = []
    for group in result.param_groups:
        for param in group["params"]:
            key = id(param)
            label = names.get(key, f"<external tensor {tuple(param.shape)}>")
            if key in seen:
                duplicate.append(label)
            seen.add(key)
            if key not in expected:
                foreign.append(label)
    missing = [names.get(key, "<unknown>") for key in expected - seen]
    problems = []
    if missing:
        problems.append(f"missing {sorted(missing)}")
    if duplicate:
        problems.append(f"duplicated {sorted(set(duplicate))}")
    if foreign:
        problems.append(f"foreign (not in the resolved groups) {sorted(set(foreign))}")
    if problems:
        raise ValueError(
            f"optimizer factory {spec} must return an optimizer over exactly the resolved parameter "
            f"groups; parameters {'; '.join(problems)}"
        )
    return result


def build_optimizer(
    net: nn.Module,
    params: AnyOptimParams,
    *,
    strict_param_groups: bool = False,
) -> torch.optim.Optimizer:
    """Build the optimizer ``params`` describes for ``net``.

    The one construction hook behind ``NNModel.train`` (non-strict: an
    unmatched trainable parameter joins a default group) and
    ``Trainer.train`` (``strict_param_groups=True``: each optimizer owns
    only what its ``param_groups`` specs select, so disjoint optimizers
    never co-own a parameter).

    Built-in :class:`NNOptimParams` dispatch through
    :class:`~nnx.nn.enum.optims.Optims` exactly as before. For
    :class:`NNOptimFactoryParams` the factory is resolved in the registry,
    called once with the groups a built-in would receive (the shared
    :func:`~nnx.nn.enum.optims.resolve_param_groups` rule, each group with
    explicit ``lr`` / ``weight_decay``) and the read-only config, and its
    result is checked: it must be a ``torch.optim.Optimizer`` holding
    exactly those parameters.
    """
    if net is None:
        raise ValueError("net must not be None")
    if isinstance(params, NNOptimParams):
        return params.name(
            net=net,
            lr_start=params.max_lr,
            momentum=params.momentum,
            weight_decay=params.weight_decay,
            param_groups=params.param_groups,
            strict_param_groups=strict_param_groups,
            eps=params.eps,
        )
    if not isinstance(params, NNOptimFactoryParams):
        raise TypeError(f"expected NNOptimParams or NNOptimFactoryParams, got {type(params).__name__}")
    factory = resolve_optimizer_factory(params.factory)
    groups = _resolved_groups(net, params, strict=strict_param_groups)
    if not any(group["params"] for group in groups):
        raise ValueError(f"optimizer factory {params.factory} would own no parameters")
    # Snapshot the ownership before the call: the factory gets fresh lists
    # it may consume or mutate freely.
    expected = {id(param) for group in groups for param in group["params"]}
    result = factory(groups, params.factory.config)
    return _check_factory_result(result, expected, net, params.factory)


def optimizer_factory_state(params: AnyOptimParams) -> Optional[dict[str, Any]]:
    """The factory identity a checkpoint records for resume validation:
    the spec's state for a registered variant, ``None`` for a built-in."""
    return params.factory.state() if isinstance(params, NNOptimFactoryParams) else None
