"""FEAT-025: the opt-in Result layer — laws, no truthiness, no caught callbacks."""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from nnx.result import Err, Ok, UnwrapError


def double(x):
    return x * 2


def safe_inverse(x):
    return Err("zero") if x == 0 else Ok(1 / x)


def plus_one_ok(x):
    return Ok(x + 1)


@pytest.mark.parametrize("value", [0, None, False, "", 3])
def test_ok_holds_any_value_and_has_no_truth_value(value):
    result = Ok(value)
    assert result.is_ok and not result.is_err and result.unwrap() == value
    with pytest.raises(TypeError, match="no truth value"):
        bool(result)
    with pytest.raises(TypeError, match="no truth value"):
        bool(Err(value))


def test_results_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Ok(1).value = 2  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        Err("e").error = "f"  # type: ignore[misc]


def test_map_changes_only_ok_and_map_error_only_err():
    assert Ok(2).map(double) == Ok(4)
    assert Err("e").map(double) == Err("e")
    assert Err("e").map_error(str.upper) == Err("E")
    assert Ok(2).map_error(str.upper) == Ok(2)


@pytest.mark.parametrize("start", [Ok(4), Ok(0), Err("boom")])
def test_monad_laws(start):
    # left identity: Ok(a).bind(f) == f(a)
    assert Ok(4).bind(safe_inverse) == safe_inverse(4)
    assert Ok(0).bind(safe_inverse) == safe_inverse(0)
    # right identity: m.bind(Ok) == m
    assert start.bind(Ok) == start
    # associativity: m.bind(f).bind(g) == m.bind(lambda x: f(x).bind(g))
    assert start.bind(safe_inverse).bind(plus_one_ok) == start.bind(lambda x: safe_inverse(x).bind(plus_one_ok))
    # functor identity and composition
    assert start.map(lambda x: x) == start
    assert start.map(double).map(double) == start.map(lambda x: double(double(x)))


def test_bind_flattens_one_result_and_recover_composes():
    assert Ok(2).bind(safe_inverse) == Ok(0.5)
    assert Ok(0).bind(safe_inverse) == Err("zero")
    recovered = Err("zero").recover(lambda e: Ok(-1) if e == "zero" else Err(e))
    assert recovered == Ok(-1)
    assert Err("other").recover(lambda e: Ok(-1) if e == "zero" else Err(e)) == Err("other")
    assert Ok(3).recover(lambda e: Ok(-1)) == Ok(3)
    # recover then map compose like any Result
    assert Err("zero").recover(lambda e: Ok(10)).map(double) == Ok(20)


@pytest.mark.parametrize(
    "call",
    [lambda: Ok(1).bind(lambda x: x + 1), lambda: Err("e").recover(lambda e: "fallback")],
    ids=["bind", "recover"],
)
def test_a_callback_returning_a_non_result_raises_type_error(call):
    with pytest.raises(TypeError, match="must return an Ok or an Err"):
        call()


@pytest.mark.parametrize("error", [RuntimeError("boom"), KeyboardInterrupt(), asyncio.CancelledError()])
def test_callbacks_are_never_caught(error):
    def raising(_):
        raise error

    for call in (
        lambda: Ok(1).map(raising),
        lambda: Ok(1).bind(raising),
        lambda: Err("e").map_error(raising),
        lambda: Err("e").recover(raising),
    ):
        with pytest.raises(type(error)):
            call()


def test_unwrap_error_pickles_with_its_error():
    import pickle

    clone = pickle.loads(pickle.dumps(UnwrapError({"code": "x"})))
    assert clone.error == {"code": "x"} and str(clone) == "called unwrap() on an Err: {'code': 'x'}"


def test_unwrap_raises_a_documented_wrapper_with_the_typed_error():
    cause = ValueError("bad")
    with pytest.raises(UnwrapError) as caught:
        Err(cause).unwrap()
    assert caught.value.error is cause and caught.value.__cause__ is cause
    with pytest.raises(UnwrapError) as caught:
        Err({"code": "x"}).unwrap()
    assert caught.value.error == {"code": "x"} and caught.value.__cause__ is None
