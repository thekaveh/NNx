"""Checkpointable component state (FEAT-005).

A warm resume already restores the model, optimizer, scheduler, GradScaler
and RNG state from the training-state sidecar of a checkpoint. Anything
else that carries state across epochs — a callback's patience, a JEPA EMA
target encoder, a custom step's running statistics — registers as a
**component**: an object with a unique name, a versioned schema and a pair
of state methods::

    class RunningMean:
        def component_spec(self) -> ComponentSpec:
            return ComponentSpec("my.running_mean", version=1)

        def component_state(self) -> dict:
            return {"mean": self.mean, "count": self.count}

        def load_component_state(self, state, *, version: int) -> None:
            self.mean, self.count = state["mean"], state["count"]

``NNModel.train`` and ``Trainer.train`` collect components from their
callbacks and step function (and any ``components=[...]`` passed
explicitly), write every component's state into the same checkpoint
generation as the model — so a model and component state from different
generations can never be combined — and restore them on resume:

1. every component's saved metadata is validated **before anything is
   mutated**; a missing, unknown or incompatible required component — or a
   saved state that a component's optional
   ``check_component_state(state, *, version)`` hook rejects — yields one
   :class:`ComponentRestoreError` listing every problem;
2. reset hooks (``Callback.on_train_begin``) run once;
3. the saved states are loaded transactionally — if any load fails, every
   component already restored is put back to its pre-call state and the
   error propagates;
4. the first resumed epoch runs.

State must be ``torch.load(weights_only=True)``-safe: tensors, numbers,
strings, booleans, ``None`` and lists / tuples / dicts of those.
``component_state()`` may return live tensors: the state is serialized
right away, and a restore snapshots it with a deep copy.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, cast, runtime_checkable

__all__ = [
    "COMPONENT_STATE_VERSION",
    "ComponentRegistry",
    "ComponentRestoreError",
    "ComponentSpec",
    "ResumeStatus",
    "StatefulComponent",
]

# Version of the {name: {version, required, state}} mapping stored in the
# training-state sidecar under "components".
COMPONENT_STATE_VERSION = 1
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")


class ComponentRestoreError(ValueError):
    """Saved component state that cannot be restored: every problem found
    (missing, unknown or incompatible components) is listed in
    ``problems``, and nothing has been mutated when it is raised by
    validation."""

    def __init__(self, problems: Iterable[str]) -> None:
        self.problems = tuple(problems)
        lines = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"cannot restore component state:\n{lines}")


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """Identity and schema version of a checkpointable component.

    Args:
        name: unique within a training run — a filename-safe slug
            (letters, digits, ``.``, ``_``, ``:``, ``-``).
        version: schema version of ``component_state()`` (a positive
            integer). A checkpoint written by a *newer* schema than the
            component supports is rejected before anything is restored.
        required: when ``True`` (default) a resume from a stateful
            checkpoint that lacks this component's state fails; an optional
            component simply keeps its fresh state.
        numbered: ``name`` is a default the registry may number when several
            components share it — ``name``, ``name.2``, ``name.3`` … in
            registration order — instead of rejecting the duplicate
            (built-in callbacks such as ``EarlyStopping`` use it, so two of
            them in one run still work). A resume then maps state by that
            order. Explicit names leave it ``False`` and must be unique.
    """

    name: str
    version: int = 1
    required: bool = True
    numbered: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise ValueError(
                f"ComponentSpec name must be a filename-safe slug (letters, digits, '.', '_', ':', '-'), "
                f"got {self.name!r}"
            )
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError(f"ComponentSpec version must be a positive integer, got {self.version!r}")
        if not isinstance(self.required, bool):
            raise ValueError(f"ComponentSpec required must be a bool, got {self.required!r}")
        if not isinstance(self.numbered, bool):
            raise ValueError(f"ComponentSpec numbered must be a bool, got {self.numbered!r}")


@runtime_checkable
class StatefulComponent(Protocol):
    """The contract a checkpointable component implements.

    A component may also define ``check_component_state(state, *, version)
    -> Iterable[str]``: the registry calls it while validating a resume,
    before anything is mutated, and reports every string it returns (for
    example a configuration that no longer matches the saved state) in the
    same :class:`ComponentRestoreError` as the other problems.
    """

    def component_spec(self) -> ComponentSpec: ...

    def component_state(self) -> dict[str, Any]: ...

    def load_component_state(self, state: Mapping[str, Any], *, version: int) -> None: ...


@dataclass(frozen=True, slots=True)
class ResumeStatus:
    """How a training session started.

    Attributes:
        mode: ``"fresh"`` (no resume), ``"stateful"`` (model plus the
            complete training-state bundle, components included) or
            ``"weights_only"`` (model weights only — the checkpoint had no
            training state, or ``resume_mode="weights_only"`` was asked).
        source_run_id / source_checkpoint: where the resume came from.
        restored_components: names of the components whose state was
            restored, in registration order.
        fresh_components: registered components that kept their fresh
            state (optional components absent from the checkpoint, or every
            component on a weights-only resume).
    """

    mode: str = "fresh"
    source_run_id: Optional[str] = None
    source_checkpoint: Optional[str] = None
    restored_components: tuple[str, ...] = ()
    fresh_components: tuple[str, ...] = ()

    def state(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "source_run_id": self.source_run_id,
            "source_checkpoint": self.source_checkpoint,
            "restored_components": list(self.restored_components),
            "fresh_components": list(self.fresh_components),
        }

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> ResumeStatus:
        return ResumeStatus(
            mode=state["mode"],
            source_run_id=state.get("source_run_id"),
            source_checkpoint=state.get("source_checkpoint"),
            restored_components=tuple(state.get("restored_components") or ()),
            fresh_components=tuple(state.get("fresh_components") or ()),
        )

    def __post_init__(self) -> None:
        if self.mode not in ("fresh", "stateful", "weights_only"):
            raise ValueError(f"ResumeStatus mode must be 'fresh', 'stateful' or 'weights_only', got {self.mode!r}")


@dataclass
class _Plan:
    # (registered name, component, saved state, saved schema version)
    to_restore: list[tuple[str, Any, Mapping[str, Any], int]] = field(default_factory=list)
    fresh: list[str] = field(default_factory=list)


class ComponentRegistry:
    """Ordered, uniquely named set of the components of one training run."""

    def __init__(self, components: Iterable[Any] = ()) -> None:
        self._components: dict[str, tuple[ComponentSpec, Any]] = {}
        for component in components:
            self.register(component)

    @staticmethod
    def discover(*sources: Any, explicit: Iterable[Any] = ()) -> ComponentRegistry:
        """Register every source that implements :class:`StatefulComponent`
        (callbacks, step functions), in order — ``None`` and lists are
        flattened, anything else is ignored — then every ``explicit``
        component, which must implement the protocol (``TypeError``
        otherwise). An object that appears more than once (say, a callback
        also passed through ``components=[...]``) is registered once."""
        registry = ComponentRegistry()
        for source in sources:
            items = source if isinstance(source, (list, tuple)) else [source]
            for item in items:
                if item is not None and isinstance(item, StatefulComponent):
                    registry.register(item)
        for component in explicit:
            registry.register(component)
        return registry

    def register(self, component: Any) -> str:
        """Register ``component`` and return the name its state is saved
        under. Registering the same object again is a no-op; a different
        object with a taken name raises ``ValueError`` unless its spec is
        ``numbered``."""
        if not isinstance(component, StatefulComponent):
            raise TypeError(
                f"{type(component).__name__} is not a StatefulComponent "
                "(component_spec / component_state / load_component_state)"
            )
        for name, (_, registered) in self._components.items():
            if registered is component:
                return name
        spec = component.component_spec()
        if not isinstance(spec, ComponentSpec):
            raise TypeError(f"{type(component).__name__}.component_spec() must return a ComponentSpec")
        name = spec.name
        if name in self._components:
            if not spec.numbered:
                raise ValueError(
                    f"duplicate component name {spec.name!r}: component names must be unique within a run "
                    "(pass a distinct name to one of them)"
                )
            ordinal = 2
            while f"{spec.name}.{ordinal}" in self._components:
                ordinal += 1
            name = f"{spec.name}.{ordinal}"
        self._components[name] = (spec, component)
        return name

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._components)

    def __len__(self) -> int:
        return len(self._components)

    def collect(self) -> dict[str, dict[str, Any]]:
        """Every component's versioned state, for the training-state sidecar."""
        return {
            name: {"version": spec.version, "required": spec.required, "state": component.component_state()}
            for name, (spec, component) in self._components.items()
        }

    def fresh_plan(self) -> _Plan:
        """A plan that restores nothing: every component keeps its fresh
        state (weights-only resumes, checkpoints written before FEAT-005)."""
        return _Plan(fresh=list(self._components))

    def plan(self, saved: Optional[Mapping[str, Any]]) -> _Plan:
        """Validate saved component metadata against the registered
        components **without mutating anything**; raise one
        :class:`ComponentRestoreError` listing every problem."""
        saved = dict(saved or {})
        problems: list[str] = []
        plan = _Plan()
        for name, (spec, component) in self._components.items():
            entry = saved.get(name)
            if entry is None:
                if spec.required:
                    problems.append(
                        f"missing required component {name!r}: the checkpoint has no state for it "
                        "(it was written without this component, or with a different name)"
                    )
                else:
                    plan.fresh.append(name)
                continue
            if not isinstance(entry, Mapping) or "state" not in entry or "version" not in entry:
                problems.append(f"component {name!r}: malformed saved entry")
                continue
            version = entry["version"]
            if isinstance(version, bool) or not isinstance(version, int) or version < 1:
                problems.append(f"component {name!r}: invalid saved schema version {version!r}")
            elif version > spec.version:
                problems.append(
                    f"incompatible component {name!r}: saved with schema version {version}, but this "
                    f"component supports up to version {spec.version}"
                )
            else:
                check = getattr(component, "check_component_state", None)
                found = list(cast(Iterable[str], check(entry["state"], version=version))) if callable(check) else []
                if found:
                    problems.extend(f"incompatible component {name!r}: {problem}" for problem in found)
                else:
                    plan.to_restore.append((name, component, entry["state"], version))
        for name, entry in saved.items():
            if name in self._components:
                continue
            required = bool(entry.get("required", True)) if isinstance(entry, Mapping) else True
            if required:
                problems.append(
                    f"unknown required component {name!r}: the checkpoint has its state but this run "
                    "registers no component with that name (add it, e.g. the same callback, or resume "
                    "with resume_mode='weights_only')"
                )
        if problems:
            raise ComponentRestoreError(problems)
        return plan

    def restore(self, plan: _Plan) -> tuple[str, ...]:
        """Load a validated plan transactionally: on any failure every
        component attempted — the failing one included — gets a deep copy
        of its pre-call state back, then the original error propagates. A
        component whose own rollback fails does not stop the others from
        being rolled back, nor replace the original error; it is reported
        in a ``RuntimeWarning``."""
        snapshots = [
            (name, component, deepcopy(component.component_state()), self._components[name][0].version)
            for name, component, _, _ in plan.to_restore
        ]
        attempted = 0
        try:
            for _, component, state, version in plan.to_restore:
                attempted += 1
                component.load_component_state(state, version=version)
        except BaseException:
            failed: list[str] = []
            for name, component, snapshot, spec_version in reversed(snapshots[:attempted]):
                try:
                    component.load_component_state(snapshot, version=spec_version)
                except Exception as rollback_error:  # keep rolling the others back
                    failed.append(f"{name!r} ({type(rollback_error).__name__}: {rollback_error})")
            if failed:
                warnings.warn(
                    f"component restore failed and could not roll back {', '.join(failed)}; their state is undefined",
                    RuntimeWarning,
                    stacklevel=2,
                )
            raise
        return tuple(name for name, _, _, _ in plan.to_restore)
