"""FEAT-005: the checkpointable component registry contract.

Unique names, versioned schemas, a single actionable pre-restore report
for missing / unknown / incompatible components (nothing mutated), and
transactional restore that puts every component back on any failure.
"""

from __future__ import annotations

import copy

import pytest
import torch

from nnx import ComponentRegistry, ComponentRestoreError, ComponentSpec, EarlyStopping, StatefulComponent


class Counter:
    """A minimal component: one integer and one tensor."""

    def __init__(self, name: str, *, version: int = 1, required: bool = True):
        self.spec = ComponentSpec(name, version=version, required=required)
        self.count = 0
        self.weights = torch.zeros(3)
        self.fail_next_load = False

    def component_spec(self) -> ComponentSpec:
        return self.spec

    def component_state(self) -> dict:
        return {"count": self.count, "weights": self.weights}

    def load_component_state(self, state, *, version: int) -> None:
        self.count = state["count"]
        self.weights.copy_(state["weights"])  # in place: a live tensor, like a module buffer
        if self.fail_next_load:  # fail after mutating, once
            self.fail_next_load = False
            raise RuntimeError(f"{self.spec.name} cannot load")


def test_component_spec_validation():
    assert ComponentSpec("callback.early_stopping").version == 1
    with pytest.raises(ValueError, match="slug"):
        ComponentSpec("bad name")
    with pytest.raises(ValueError, match="positive integer"):
        ComponentSpec("x", version=0)
    with pytest.raises(ValueError, match="bool"):
        ComponentSpec("x", required="yes")  # type: ignore[arg-type]


def test_names_are_unique_and_discovery_is_ordered():
    registry = ComponentRegistry([Counter("a"), Counter("b")])
    assert registry.names == ("a", "b")
    with pytest.raises(ValueError, match="duplicate component name 'a'"):
        registry.register(Counter("a"))
    with pytest.raises(ValueError, match="duplicate component name 'es'"):
        ComponentRegistry.discover([EarlyStopping(name="es"), EarlyStopping(name="es")])
    discovered = ComponentRegistry.discover([object(), EarlyStopping(name="es")], None, Counter("c"))
    assert discovered.names == ("es", "c")
    assert isinstance(EarlyStopping(), StatefulComponent)


def test_default_named_callbacks_are_numbered_so_existing_runs_keep_working():
    # Two EarlyStopping callbacks with default names worked before component
    # state existed; they are numbered in callback order instead of rejected.
    first, second, third = EarlyStopping(), EarlyStopping(monitor="train_edp.loss"), EarlyStopping()
    registry = ComponentRegistry.discover([first, second, third])
    assert registry.names == ("early_stopping", "early_stopping.2", "early_stopping.3")
    first._wait, second._wait, third._wait = 1, 2, 3
    saved = registry.collect()
    fresh = [EarlyStopping(), EarlyStopping(monitor="train_edp.loss"), EarlyStopping()]
    restored = ComponentRegistry.discover(fresh)
    restored.restore(restored.plan(saved))
    assert [stopper._wait for stopper in fresh] == [1, 2, 3]
    # A numbered default never takes a name an explicit component holds.
    mixed = ComponentRegistry.discover([EarlyStopping(name="early_stopping"), EarlyStopping()])
    assert mixed.names == ("early_stopping", "early_stopping.2")


def test_explicit_components_are_validated_and_one_object_registers_once():
    class Drifted:  # load_state instead of load_component_state
        def component_spec(self):
            return ComponentSpec("drifted")

        def component_state(self):
            return {}

        def load_state(self, state):
            pass

    with pytest.raises(TypeError, match="Drifted is not a StatefulComponent"):
        ComponentRegistry.discover([EarlyStopping()], explicit=[Drifted()])
    # Callbacks and step functions that are not components are still ignored.
    assert ComponentRegistry.discover([Drifted()]).names == ()
    stopper = EarlyStopping()
    both = ComponentRegistry.discover([stopper], explicit=[stopper])
    assert both.names == ("early_stopping",)
    assert both.register(stopper) == "early_stopping"


def test_collect_records_version_requiredness_and_state():
    registry = ComponentRegistry([Counter("a", version=2), Counter("b", required=False)])
    collected = registry.collect()
    assert collected["a"]["version"] == 2 and collected["a"]["required"] is True
    assert collected["b"]["required"] is False and collected["b"]["state"]["count"] == 0


def test_one_pre_restore_report_lists_every_problem_and_mutates_nothing():
    fresh = Counter("present")
    registry = ComponentRegistry([Counter("missing"), Counter("versioned", version=1), fresh])
    saved = {
        "present": {"version": 1, "required": True, "state": {"count": 9, "weights": torch.ones(3)}},
        "versioned": {"version": 2, "required": True, "state": {"count": 1, "weights": torch.ones(3)}},
        "orphan": {"version": 1, "required": True, "state": {}},
        "optional_orphan": {"version": 1, "required": False, "state": {}},
    }
    with pytest.raises(ComponentRestoreError) as info:
        registry.plan(saved)
    problems = "\n".join(info.value.problems)
    assert "missing required component 'missing'" in problems
    assert "incompatible component 'versioned'" in problems and "version 2" in problems
    assert "unknown required component 'orphan'" in problems
    assert "optional_orphan" not in problems
    assert fresh.count == 0 and torch.equal(fresh.weights, torch.zeros(3))  # validation mutated nothing


def test_optional_components_absent_from_the_checkpoint_stay_fresh():
    registry = ComponentRegistry([Counter("kept"), Counter("optional", required=False)])
    plan = registry.plan({"kept": {"version": 1, "required": True, "state": {"count": 4, "weights": torch.ones(3)}}})
    assert registry.restore(plan) == ("kept",)
    assert plan.fresh == ["optional"]


def test_a_failing_restore_rolls_every_component_back_to_its_pre_call_state():
    first, second = Counter("first"), Counter("second")
    first.count, first.weights = 1, torch.full((3,), 7.0)
    second.count = 2
    snapshots = {c.spec.name: copy.deepcopy(c.component_state()) for c in (first, second)}
    registry = ComponentRegistry([first, second])
    saved = {
        name: {"version": 1, "required": True, "state": {"count": 100, "weights": torch.full((3,), -1.0)}}
        for name in ("first", "second")
    }
    plan = registry.plan(saved)
    second.fail_next_load = True  # the second component fails after partially loading
    with pytest.raises(RuntimeError, match="second cannot load"):
        registry.restore(plan)
    for component in (first, second):
        state = component.component_state()
        assert state["count"] == snapshots[component.spec.name]["count"]
        assert torch.equal(state["weights"], snapshots[component.spec.name]["weights"])


class Checked(Counter):
    """A component whose configuration must match the saved state."""

    def __init__(self, name: str, mode: str):
        super().__init__(name)
        self.mode = mode

    def component_state(self) -> dict:
        return {**super().component_state(), "mode": self.mode}

    def check_component_state(self, state, *, version: int):
        return [] if state["mode"] == self.mode else [f"saved mode {state['mode']!r} != configured {self.mode!r}"]


def test_check_hook_problems_join_the_single_pre_restore_report():
    registry = ComponentRegistry([Checked("cfg", mode="max"), Counter("missing")])
    saved = {"cfg": {"version": 1, "required": True, "state": {"count": 3, "weights": torch.ones(3), "mode": "min"}}}
    with pytest.raises(ComponentRestoreError) as info:
        registry.plan(saved)
    problems = "\n".join(info.value.problems)
    assert "incompatible component 'cfg': saved mode 'min' != configured 'max'" in problems
    assert "missing required component 'missing'" in problems


def test_early_stopping_monitor_mismatch_is_reported_before_restoring():
    saved = ComponentRegistry([EarlyStopping(monitor="val_edp.loss")]).collect()
    other = EarlyStopping(monitor="val_edp.error")
    with pytest.raises(ComponentRestoreError, match="monitor='val_edp.loss'"):
        ComponentRegistry([other]).plan(saved)


class FailsRollback(Counter):
    def load_component_state(self, state, *, version: int) -> None:
        if state["count"] == 0:  # its own pre-call snapshot
            raise RuntimeError("rollback broke")
        super().load_component_state(state, version=version)


def test_a_failing_rollback_neither_masks_the_error_nor_stops_the_others():
    first, broken, last = Counter("first"), FailsRollback("broken"), Counter("last")
    registry = ComponentRegistry([first, broken, last])
    saved = {
        name: {"version": 1, "required": True, "state": {"count": 5, "weights": torch.ones(3)}}
        for name in ("first", "broken", "last")
    }
    plan = registry.plan(saved)
    last.fail_next_load = True
    with pytest.warns(RuntimeWarning, match="could not roll back 'broken'"):
        with pytest.raises(RuntimeError, match="last cannot load"):
            registry.restore(plan)
    assert first.count == 0 and torch.equal(first.weights, torch.zeros(3))
    assert last.count == 0 and torch.equal(last.weights, torch.zeros(3))
