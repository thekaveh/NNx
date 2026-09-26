"""Arbitrary modules and registered model factories (FEAT-006).

``NNModel`` builds its network from one of three **descriptors**, stored in
``NNModelParams.net``:

- a built-in :class:`~nnx.Nets` member plus ``NNParams`` — the default path,
  unchanged;
- a :class:`ModelSpec` naming a **registered factory** — portable: the
  factory is looked up by ``id`` / ``version`` and called with the spec's
  JSON-like ``config``, so checkpoints, Hub artifacts and runs rebuild the
  module without pickling code::

      from nnx.models import ModelSpec, register_model_factory

      register_model_factory("my.encoder", 1, lambda config: Encoder(**config))
      model = NNModel(params=NNModelParams(net=ModelSpec("my.encoder", 1, {"width": 16}), loss=Losses.MSE))

  Construction runs under ``torch.manual_seed(spec.seed)`` with the ambient
  RNG state captured and restored, so the same spec always initializes the
  same weights and never shifts the caller's random streams;
- a caller-owned ``nn.Module`` instance — ``NNModel(module=encoder,
  params=NNModelParams(loss=...))``. The module is wrapped, never cloned or
  re-initialized (``model.net is encoder``); its descriptor is a
  :class:`RuntimeModule` (class name and a parameter-topology fingerprint)
  marked ``reconstructible=False``. Such a model trains and checkpoints
  normally, but a portable save (safetensors checkpoint, ``save_pretrained``)
  fails with :class:`MissingModelFactoryError`, and reloading its weights
  needs the module again (``NNModel.from_checkpoint(ckpt, module=...)``).

Non-built-in modules see their batches through a **batch adapter**:
:class:`PositionalInputs` (``(x1, ..., xn, y)`` tuples → ``module(x1, ...,
xn)``) or :class:`KeywordInputs` (mapping batches → ``module(**inputs)``).
A module that already defines ``unpack_batch`` keeps using it. The adapter
is runtime-only — it is never serialized and does not change run ids.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional, cast

import torch
from torch import nn

from ._config import _SLUG, _freeze_config, _thaw_config
from ._validation import require_count

__all__ = [
    "BatchAdapter",
    "KeywordInputs",
    "MissingModelFactoryError",
    "ModelFactory",
    "ModelSpec",
    "PositionalInputs",
    "RuntimeModule",
    "build_module",
    "module_topology",
    "register_model_factory",
    "registered_model_factories",
    "resolve_model_factory",
    "unregister_model_factory",
]

ModelFactory = Callable[[Mapping[str, Any]], nn.Module]
"""``factory(config) -> torch.nn.Module``."""

_REGISTRY: dict[tuple[str, int], ModelFactory] = {}


class MissingModelFactoryError(ValueError):
    """A module cannot be rebuilt: its factory is not registered, or it is a
    runtime-only module (``reconstructible=False``) that no factory
    describes."""


def _require_id(value: object, *, owner: str) -> str:
    if not isinstance(value, str) or _SLUG.fullmatch(value) is None:
        raise ValueError(
            f"{owner} requires id to be a non-empty string of letters, digits, '_', '.' or '-' "
            f"(starting with a letter or digit), got {value!r}"
        )
    return value


# --- descriptors ---------------------------------------------------------------


class ModelSpec:
    """Stable, serializable reference to a registered model factory.

    ``id`` and ``version`` name the factory; ``config`` is handed to it
    read-only and must be JSON-like (``None``, ``bool``, ``int``, finite
    ``float``, ``str``, lists and string-keyed mappings); ``seed`` is the
    ``torch.manual_seed`` construction runs under. The spec never holds the
    callable, so it round-trips through ``run.yaml``, checkpoints and Hub
    configs. Immutable; equal specs have equal ``state()``.
    """

    __slots__ = ("id", "version", "config", "seed")
    kind = "registered"
    reconstructible = True
    id: str
    version: int
    config: Mapping[str, Any]
    seed: int

    def __init__(self, id: str, version: int = 1, config: Optional[Mapping[str, Any]] = None, *, seed: int = 0):
        owner = "ModelSpec"
        object.__setattr__(self, "id", _require_id(id, owner=owner))
        object.__setattr__(self, "version", require_count(version, "version", owner=owner, minimum=1))
        if config is None:
            config = {}
        if not isinstance(config, Mapping):
            raise TypeError(f"ModelSpec config must be a str-keyed mapping, got {type(config).__name__}")
        object.__setattr__(self, "config", _freeze_config(config, "", owner=owner))
        object.__setattr__(self, "seed", require_count(seed, "seed", owner=owner, minimum=0))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"ModelSpec is immutable; cannot set {name!r}")

    def __reduce__(self):
        return (_model_spec, (self.id, self.version, _thaw_config(self.config), self.seed))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelSpec):
            return NotImplemented
        return _canonical(self.state()) == _canonical(other.state())

    def __hash__(self) -> int:
        return hash(_canonical(self.state()))

    def __repr__(self) -> str:
        return (
            f"ModelSpec(id={self.id!r}, version={self.version}, config={_thaw_config(self.config)!r}, seed={self.seed})"
        )

    def __str__(self) -> str:
        return f"{self.id}@v{self.version}"

    def state(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "id": self.id,
            "version": self.version,
            "config": _thaw_config(self.config),
            "seed": self.seed,
        }

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> ModelSpec:
        return ModelSpec(state["id"], state["version"], state.get("config") or {}, seed=state.get("seed", 0))


def _model_spec(id: str, version: int, config: Mapping[str, Any], seed: int) -> ModelSpec:
    return ModelSpec(id, version, config, seed=seed)


class RuntimeModule:
    """Descriptor of a caller-owned module that no factory describes.

    ``module`` is the class's qualified name and ``topology`` a fingerprint
    of its parameter and buffer names, shapes and dtypes and of its
    ``repr`` (layer hyperparameters such as dropout rates) — never of
    weight values. Like a built-in net, the run id therefore reflects the
    architecture, not the weights: two differently-initialized instances of
    one architecture share a run id, so give them distinct ``data_id`` /
    ``salt`` values. ``reconstructible`` is always ``False``: the weights
    can only be loaded back into a module the caller supplies again.
    """

    __slots__ = ("module", "topology")
    kind = "runtime"
    reconstructible = False
    module: str
    topology: str

    def __init__(self, module: str, topology: str) -> None:
        if not isinstance(module, str) or not module:
            raise ValueError(f"RuntimeModule.module must be a non-empty string, got {module!r}")
        if not isinstance(topology, str) or not topology:
            raise ValueError(f"RuntimeModule.topology must be a non-empty string, got {topology!r}")
        object.__setattr__(self, "module", module)
        object.__setattr__(self, "topology", topology)

    @classmethod
    def of(cls, module: nn.Module) -> RuntimeModule:
        """The descriptor of ``module`` (its class and topology)."""
        if not isinstance(module, nn.Module):
            raise TypeError(f"expected a torch.nn.Module, got {type(module).__name__}")
        kind = type(module)
        return cls(f"{kind.__module__}.{kind.__qualname__}", module_topology(module))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"RuntimeModule is immutable; cannot set {name!r}")

    def __reduce__(self):
        return (RuntimeModule, (self.module, self.topology))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RuntimeModule):
            return NotImplemented
        return (self.module, self.topology) == (other.module, other.topology)

    def __hash__(self) -> int:
        return hash((self.module, self.topology))

    def __repr__(self) -> str:
        return f"RuntimeModule(module={self.module!r}, topology={self.topology!r})"

    def __str__(self) -> str:
        return f"runtime:{self.module}"

    def state(self) -> dict[str, Any]:
        return {"kind": self.kind, "module": self.module, "topology": self.topology, "reconstructible": False}

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> RuntimeModule:
        return RuntimeModule(state["module"], state["topology"])


def descriptor_from_state(state: Any) -> Any:
    """Decode a non-built-in ``NNModelParams.net`` state by its ``kind``."""
    if not isinstance(state, Mapping):
        raise ValueError(f"model descriptor state must be a mapping, got {type(state).__name__}")
    kind = state.get("kind")
    if kind == ModelSpec.kind:
        return ModelSpec.from_state(state)
    if kind == RuntimeModule.kind:
        return RuntimeModule.from_state(state)
    raise ValueError(f"unknown model descriptor kind {kind!r} (expected 'registered' or 'runtime')")


def _canonical(state: Mapping[str, Any]) -> str:
    return json.dumps(state, sort_keys=True)


def module_topology(module: nn.Module) -> str:
    """Fingerprint of ``module``'s state-dict names, shapes and dtypes plus
    its ``repr`` (layer hyperparameters). Uninitialized lazy parameters
    (``nn.LazyLinear`` before its first forward) count by name only."""
    layout = [
        [name, None if shape is None else list(shape), str(value.dtype)]
        for name, value in module.state_dict().items()
        if isinstance(value, torch.Tensor)
        for shape in (_shape(value),)
    ]
    return hashlib.md5(json.dumps([layout, repr(module)]).encode("utf-8")).hexdigest()


def _shape(value: torch.Tensor) -> Optional[tuple[int, ...]]:
    """A tensor's shape, or ``None`` for an uninitialized lazy parameter."""
    return None if nn.parameter.is_lazy(value) else tuple(value.shape)


def _state_schema(state: Mapping[str, Any]) -> dict[str, Optional[tuple[int, ...]]]:
    return {name: _shape(value) for name, value in state.items() if isinstance(value, torch.Tensor)}


def check_state_schema(module: nn.Module, state: Mapping[str, Any], *, what: str) -> None:
    """Raise before loading when ``state`` does not match ``module``'s
    parameter names and shapes (an uninitialized lazy parameter matches any
    shape — loading materializes it)."""
    expected, actual = _state_schema(module.state_dict()), _state_schema(state)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    reshaped = sorted(
        name
        for name in set(expected) & set(actual)
        if expected[name] is not None and actual[name] is not None and expected[name] != actual[name]
    )
    if not (missing or unexpected or reshaped):
        return
    details = "; ".join(
        f"{label}: {', '.join(names[:5])}{' …' if len(names) > 5 else ''}"
        for label, names in (("missing", missing), ("unexpected", unexpected), ("reshaped", reshaped))
        if names
    )
    raise ValueError(f"{what}: the rebuilt module's topology does not match the saved weights ({details})")


# --- registry ------------------------------------------------------------------


def register_model_factory(id: str, version: int, factory: ModelFactory, *, replace: bool = False) -> None:
    """Register ``factory`` under ``(id, version)`` for this process.

    ``factory(config)`` receives a :class:`ModelSpec`'s read-only config and
    must return a ``torch.nn.Module``. Registering an existing
    ``(id, version)`` raises unless ``replace=True``; bump ``version`` when
    the factory's architecture changes, so saved weights are never loaded
    into a different topology.
    """
    owner = "register_model_factory"
    key = (_require_id(id, owner=owner), require_count(version, "version", owner=owner, minimum=1))
    if not callable(factory):
        raise TypeError(f"register_model_factory requires a callable factory, got {type(factory).__name__}")
    if key in _REGISTRY and not replace:
        raise ValueError(
            f"model factory {key[0]}@v{key[1]} is already registered; pass replace=True to override it, "
            "or register a new version"
        )
    _REGISTRY[key] = factory


def unregister_model_factory(id: str, version: int) -> bool:
    """Remove ``(id, version)``; returns whether it was registered. Runs and
    checkpoints that reference it still load their metadata — only
    rebuilding the module needs the registration."""
    return _REGISTRY.pop((id, version), None) is not None


def registered_model_factories() -> tuple[tuple[str, int], ...]:
    """Sorted ``(id, version)`` pairs currently registered."""
    return tuple(sorted(_REGISTRY))


def resolve_model_factory(spec: ModelSpec) -> ModelFactory:
    """Look ``spec`` up in the registry. Never imports anything: an
    unregistered id or version raises :class:`MissingModelFactoryError`
    naming what is registered."""
    if not isinstance(spec, ModelSpec):
        raise TypeError(f"expected a ModelSpec, got {type(spec).__name__}")
    factory = _REGISTRY.get((spec.id, spec.version))
    if factory is not None:
        return factory
    versions = sorted(version for fid, version in _REGISTRY if fid == spec.id)
    if versions:
        registered = ", ".join(f"v{version}" for version in versions)
        raise MissingModelFactoryError(
            f"model factory {spec} is not registered (registered versions of {spec.id!r}: {registered})"
        )
    raise MissingModelFactoryError(
        f"model factory {spec} is not registered; call nnx.models.register_model_factory({spec.id!r}, "
        f"{spec.version}, factory) in this process before building or loading it"
    )


def build_module(spec: ModelSpec) -> nn.Module:
    """Build ``spec``'s module: resolve the factory (failing before anything
    runs when it is unknown), then call it with Python's ``random``, NumPy
    and torch all seeded from ``spec.seed`` and the ambient RNG states
    restored afterwards."""
    factory = resolve_model_factory(spec)
    import random

    import numpy as np

    from .nn.nn_model import _capture_rng_state, _restore_rng_state

    rng_state = _capture_rng_state(None)
    try:
        random.seed(spec.seed)
        np.random.seed(spec.seed)
        torch.manual_seed(spec.seed)
        module = factory(spec.config)
    finally:
        _restore_rng_state(rng_state, None)
    if not isinstance(module, nn.Module):
        raise TypeError(f"model factory {spec} returned {type(module).__name__}, not a torch.nn.Module")
    return module


# --- batch adapters --------------------------------------------------------------


class BatchAdapter:
    """How a non-built-in module sees a batch.

    :meth:`split` returns ``(args, kwargs, target)``: the module is called as
    ``module(*args, **kwargs)`` and ``target`` (``None`` when the batch has
    none, e.g. for prediction) is what the loss scores. :meth:`output`
    turns the module's raw return value into the output tensor; the default
    requires a tensor. Subclass either to support other layouts.
    """

    def split(self, batch: Any) -> tuple[tuple[Any, ...], dict[str, Any], Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def output(self, raw: Any) -> torch.Tensor:
        if isinstance(raw, torch.Tensor):
            return raw
        raise TypeError(
            f"the module returned {type(raw).__name__}, not a tensor; use a BatchAdapter whose output() "
            "extracts the output tensor"
        )


class PositionalInputs(BatchAdapter):
    """Positional inputs: a batch ``(x1, ..., xn, y)`` calls
    ``module(x1, ..., xn)`` and scores ``y``; ``(x1, ..., xn)`` or a bare
    tensor has no target."""

    def __init__(self, n_inputs: int = 1) -> None:
        self.n_inputs = require_count(n_inputs, "n_inputs", owner="PositionalInputs", minimum=1)

    def __repr__(self) -> str:
        return f"PositionalInputs(n_inputs={self.n_inputs})"

    def split(self, batch: Any) -> tuple[tuple[Any, ...], dict[str, Any], Any]:
        if isinstance(batch, torch.Tensor):
            if self.n_inputs != 1:
                raise ValueError(f"PositionalInputs(n_inputs={self.n_inputs}) got a single tensor batch")
            return (batch,), {}, None
        if isinstance(batch, Sequence) and not isinstance(batch, str):
            items = tuple(batch)
            if len(items) == self.n_inputs + 1:
                return items[: self.n_inputs], {}, items[self.n_inputs]
            if len(items) == self.n_inputs:
                return items, {}, None
            raise ValueError(
                f"PositionalInputs(n_inputs={self.n_inputs}) expects batches of {self.n_inputs} input(s) plus an "
                f"optional target, got {len(items)} item(s)"
            )
        raise TypeError(f"PositionalInputs expects a tensor or a sequence batch, got {type(batch).__name__}")


class KeywordInputs(BatchAdapter):
    """Keyword inputs: a mapping batch calls ``module(**{name: batch[name]
    for name in inputs})`` and scores ``batch[target]`` (``None`` when the
    key is absent or ``target=None``)."""

    def __init__(self, inputs: Sequence[str], target: Optional[str] = "labels") -> None:
        names = tuple(inputs) if not isinstance(inputs, str) else (inputs,)
        if not names or not all(isinstance(name, str) and name for name in names):
            raise ValueError(f"KeywordInputs needs at least one non-empty input name, got {inputs!r}")
        if len(set(names)) != len(names):
            raise ValueError(f"KeywordInputs input names must be unique, got {names!r}")
        if target is not None and (not isinstance(target, str) or not target or target in names):
            raise ValueError(f"KeywordInputs target must be a non-empty name distinct from the inputs, got {target!r}")
        self.inputs = names
        self.target = target

    def __repr__(self) -> str:
        return f"KeywordInputs(inputs={self.inputs!r}, target={self.target!r})"

    def split(self, batch: Any) -> tuple[tuple[Any, ...], dict[str, Any], Any]:
        if not isinstance(batch, Mapping):
            raise TypeError(f"KeywordInputs expects mapping batches, got {type(batch).__name__}")
        missing = [name for name in self.inputs if name not in batch]
        if missing:
            raise KeyError(f"batch is missing input(s) {missing} (has {sorted(batch)})")
        kwargs = {name: batch[name] for name in self.inputs}
        target = batch.get(self.target) if self.target is not None else None
        return (), kwargs, target


class _UnpackBatch(BatchAdapter):
    """A module's own ``unpack_batch(batch) -> ((x1, ...), y)``."""

    def __init__(self, module: nn.Module) -> None:
        self._unpack: Callable[[Any], Any] = cast(Any, module).unpack_batch

    def split(self, batch: Any) -> tuple[tuple[Any, ...], dict[str, Any], Any]:
        inputs, target = self._unpack(batch)
        return _as_inputs(inputs), {}, target


def _as_inputs(inputs: Any) -> tuple[Any, ...]:
    """``unpack_batch`` returns a tuple of inputs; a bare tensor (or other
    single input) is one input, never split into rows."""
    return tuple(inputs) if isinstance(inputs, (tuple, list)) else (inputs,)


def default_batch_adapter(module: nn.Module) -> BatchAdapter:
    """``module.unpack_batch`` when the module defines it, else one
    positional input."""
    if callable(getattr(module, "unpack_batch", None)):
        return _UnpackBatch(module)
    return PositionalInputs(1)
