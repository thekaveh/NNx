"""Deferred decision jobs: batch independent questions, chain dependent
ones (FEAT-024).

A :class:`DecisionJob` is an **immutable description** of decision work —
building one calls no provider, runs no callback and draws no RNG::

    from nnx.decisions import DecisionJob as Job

    triage = Job.collect({
        "topic": Job.ask(topic_question, id="topic"),
        "urgent": Job.ask(urgent_question, id="urgent"),
    }).map(lambda answers: answers["topic"])
    routed = Job.ask(topic_question, id="topic").then(
        lambda answer: Follow(Job.ask(team_question, id="team"), state=route_state(answer))
    )
    result = triage.run(provider, state=texts, limits=Limits(max_questions=2))

- :meth:`DecisionJob.ask` — one question over the run's (or a
  continuation's) **state**: the inputs the provider reads. Its value is the
  provider's results, one per input row (or, with an abstention ``policy``,
  the selective decisions).
- :meth:`DecisionJob.collect` — independent jobs, keyed; the value is a
  mapping in the given key order.
- :meth:`DecisionJob.map` — a pure function of a job's value.
- :meth:`DecisionJob.then` — a **dependent** job: once its prerequisite
  succeeds, ``fn(value)`` returns a :class:`Follow` — the next job and the
  state it reads, mapped explicitly from the answer. It runs once, never
  feeds a sibling question, and is bounded by ``Limits.max_depth``.

:meth:`DecisionJob.run` and :meth:`DecisionJob.arun` are the only effect
boundaries. They **borrow** the provider (never close it): independent
questions over the same state are batched into the fewest calls —
``ceil(count / cap)`` per state, in a stable order — where ``cap`` is the
provider's ``max_questions`` (a provider answering one question per call,
such as :class:`~nnx.decisions.FixedHeadProvider`, has cap 1) bounded by
``Limits.max_questions``. Duplicate question ids, questions the provider
cannot serve and a token cap the provider cannot enforce are rejected
before any call.

**Answers are checked.** Each call's answers must hold one result list per
question, one result per input row, each a result of that question (its
digest); anything else is an ``InvalidDecisionResponse`` failure.

**Fail-fast.** A provider error stops scheduling: :class:`JobFailed` carries
every outcome known so far (``outcomes``; the answered ones are
``completed``), the failing questions and the skipped ones; nothing
completed is re-run (the job never retries — retries belong to the
provider). Request and depth limits raise :class:`JobLimitExceeded` (in
``arun``, after the calls already in flight finish), the ``timeout``
:class:`JobTimeout`, each with the outcomes so far; questions never sent
are ``skipped``.

**Async.** :meth:`DecisionJob.arun` runs up to ``Limits.max_concurrency``
provider calls at once through a provider's ``adecide_many`` / ``adecide``.
A provider with only synchronous methods is called in a worker thread,
never alongside another call (its methods need not be thread-safe), and the
run waits for a running thread before it returns — a cancellation that
arrives meanwhile is delivered once the thread is done. Setting the
``cancel`` event (an ``asyncio.Event``; set at any point, even if cleared
again) stops scheduling — no continuation runs after it — and returns a
``"cancelled"`` :class:`JobResult`; cancelling the task cleans up the job's own tasks and
re-raises. A request whose call began is reported as ``sent`` — never as
rolled back: whatever the provider did with it stays done.

**What "independent" means.** Computational, not statistical: questions
batched together share a call, nothing more. Their answers are separate
marginals; the job never multiplies them into a joint probability.
"""

from __future__ import annotations

import asyncio
import numbers
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, Union, cast

from .schema import (
    Boolean,
    BooleanResult,
    Choice,
    ChoiceResult,
    DecisionError,
    DecisionResult,
    InvalidDecisionRequest,
    InvalidDecisionResponse,
    ProviderFailure,
    Score,
    ScoreResult,
    UnsupportedCapability,
)

__all__ = [
    "DecisionJob",
    "Follow",
    "InvalidJob",
    "JobError",
    "JobFailed",
    "JobLimitExceeded",
    "JobResult",
    "JobTimeout",
    "Limits",
    "QuestionOutcome",
    "RowOutcome",
]

Question = Union[Choice, Boolean, Score]


# --- errors ---------------------------------------------------------------------------------


class JobError(DecisionError):
    """Base of the decision-job errors. ``outcomes`` maps question ids to
    every :class:`QuestionOutcome` known when the error was raised (answered,
    failed, skipped or cancelled — a request already sent says so), in
    scheduling order; ``completed`` is the answered subset. Both are empty
    when the job was refused before any call."""

    def __init__(self, message: str, *, outcomes: Optional[Mapping[str, QuestionOutcome]] = None) -> None:
        super().__init__(message)
        self.outcomes: dict[str, QuestionOutcome] = dict(outcomes or {})
        self.completed: dict[str, QuestionOutcome] = {
            question_id: outcome for question_id, outcome in self.outcomes.items() if outcome.kind == "answered"
        }

    def __reduce__(self) -> Any:
        # Keyword-only fields would break Exception's default pickling (a
        # process-pool worker's error must reach its parent intact), and so
        # would dropping the provider's error (the __cause__) when it pickles.
        import pickle

        cause = self.__cause__
        try:
            pickle.dumps(cause)
        except Exception:
            cause = None
        return (_restore_error, (type(self), str(self), dict(self.__dict__), cause))


def _restore_error(
    cls: type[JobError], message: str, fields: dict[str, Any], cause: Optional[BaseException] = None
) -> JobError:
    error = cls.__new__(cls)
    Exception.__init__(error, message)
    error.__dict__.update(fields)
    error.__cause__ = cause
    return error


class InvalidJob(JobError, ValueError):
    """A job that cannot run as described: duplicate question ids, an
    unsupported question, a token cap the provider cannot enforce, a
    continuation that returns something other than a :class:`Follow`."""


class JobFailed(JobError):
    """A provider call failed (fail-fast): ``failed`` names the questions of
    that call, ``skipped`` the ready questions that were never sent; the
    provider's error is the ``__cause__``."""

    def __init__(
        self,
        message: str,
        *,
        outcomes: Mapping[str, QuestionOutcome],
        failed: Sequence[str],
        skipped: Sequence[str],
    ) -> None:
        super().__init__(message, outcomes=outcomes)
        self.failed = tuple(failed)
        self.skipped = tuple(skipped)


class JobLimitExceeded(JobError):
    """A request or depth limit stopped the job before the next call."""


class JobTimeout(JobError, TimeoutError):
    """The run's ``timeout`` elapsed; in-flight requests were sent and are
    not rolled back."""


# --- configuration and outcomes ----------------------------------------------------------------


def _optional_positive(value: Any, what: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise InvalidJob(f"{what} must be a positive integer or None, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class Limits:
    """What a run may do.

    Attributes:
        max_questions: the most questions per provider call (``None``: the
            provider's own cap; a provider without one answers one per call).
        max_tokens: the most tokens per provider call — enforceable only by
            a provider with ``count_tokens(questions, state)``; any other is
            refused before any call.
        max_depth: the most nested continuations (``then``).
        max_requests: the most provider calls in the run (``None``:
            unbounded).
        timeout: seconds for the whole run (``None``: none). ``run`` checks
            it between calls; ``arun`` also cancels in-flight calls.
        max_concurrency: ``arun``'s concurrent provider calls.
    """

    max_questions: Optional[int] = None
    max_tokens: Optional[int] = None
    max_depth: int = 8
    max_requests: Optional[int] = None
    timeout: Optional[float] = None
    max_concurrency: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_questions", _optional_positive(self.max_questions, "max_questions"))
        object.__setattr__(self, "max_tokens", _optional_positive(self.max_tokens, "max_tokens"))
        object.__setattr__(self, "max_requests", _optional_positive(self.max_requests, "max_requests"))
        depth = self.max_depth
        if isinstance(depth, bool) or not isinstance(depth, numbers.Integral) or depth < 0:
            raise InvalidJob(f"max_depth must be a non-negative integer, got {depth!r}")
        concurrency = self.max_concurrency
        if isinstance(concurrency, bool) or not isinstance(concurrency, numbers.Integral) or concurrency < 1:
            raise InvalidJob(f"max_concurrency must be a positive integer, got {concurrency!r}")
        object.__setattr__(self, "max_concurrency", int(concurrency))
        timeout = self.timeout
        if timeout is not None and (
            isinstance(timeout, bool) or not isinstance(timeout, numbers.Real) or not timeout > 0
        ):
            raise InvalidJob(f"timeout must be a positive number of seconds or None, got {timeout!r}")


@dataclass(frozen=True)
class RowOutcome:
    """One input row's answer: ``"answered"``, or ``"abstained"`` when the
    question's abstention policy declined it (the result is kept either
    way)."""

    kind: str
    result: DecisionResult
    decision: Any = None  # the SelectiveDecision, when a policy applied


@dataclass(frozen=True)
class QuestionOutcome:
    """What happened to one question.

    ``kind`` is ``"answered"`` (the provider answered; rows may still
    abstain), ``"failed"`` (its provider call raised — ``error``),
    ``"skipped"`` (never sent: a failure, a limit or the timeout stopped
    scheduling) or ``"cancelled"`` (the run was cancelled, or a sibling call
    failed first; ``sent`` says whether the request had already gone out —
    a sent request is not rolled back).
    """

    id: str
    kind: str
    rows: tuple[RowOutcome, ...] = ()
    error: Optional[BaseException] = field(default=None, compare=False)
    sent: bool = False

    @property
    def abstained(self) -> bool:
        return any(row.kind == "abstained" for row in self.rows)


@dataclass(frozen=True)
class JobResult:
    """A finished run: the job's ``value``, every question's outcome in the
    order it was scheduled, the provider ``calls`` made and the ``status``
    (``"completed"`` or ``"cancelled"``; a cancelled run has no value)."""

    value: Any
    outcomes: Mapping[str, QuestionOutcome]
    calls: int
    status: str = "completed"


@dataclass(frozen=True)
class Follow:
    """What a continuation returns: the next ``job`` and the ``state`` it
    reads (the answer mapped explicitly into follow-up inputs)."""

    job: DecisionJob
    state: Any


# --- the job tree -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Ask:
    question: Question
    id: str
    policy: Any = None
    model_id: Optional[str] = None


@dataclass(frozen=True)
class _Collect:
    items: tuple[tuple[str, DecisionJob], ...]


@dataclass(frozen=True)
class _Map:
    job: DecisionJob
    fn: Callable[[Any], Any]


@dataclass(frozen=True)
class _Then:
    job: DecisionJob
    fn: Callable[[Any], Follow]


_Node = Union[_Ask, _Collect, _Map, _Then]


class DecisionJob:
    """An immutable, deferred description of decision work — see the module
    docstring. Build with :meth:`ask` and :meth:`collect`, transform with
    :meth:`map`, chain with :meth:`then`; run with :meth:`run` /
    :meth:`arun`."""

    __slots__ = ("_node",)

    def __init__(self, node: _Node) -> None:
        object.__setattr__(self, "_node", node)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("DecisionJob is immutable")

    # ---------- building (pure) ----------

    @staticmethod
    def ask(question: Question, *, id: str, policy: Any = None, model_id: Optional[str] = None) -> DecisionJob:
        """One question over the state. ``id`` keys its outcome (unique in a
        run); an abstention ``policy`` (with the ``model_id`` it was tuned
        for) turns each row's result into a selective decision."""
        if not isinstance(question, (Choice, Boolean, Score)):
            raise InvalidJob(f"ask() needs a Choice, Boolean or Score, got {type(question).__name__}")
        if not isinstance(id, str) or not id.strip():
            raise InvalidJob(f"a question id must be a non-empty string, got {id!r}")
        if policy is not None:
            from ..abstention import AbstentionPolicy

            if not isinstance(policy, AbstentionPolicy):
                raise InvalidJob(f"policy must be an nnx.abstention.AbstentionPolicy, got {type(policy).__name__}")
            if not isinstance(model_id, str) or not model_id:
                raise InvalidJob("an abstention policy needs the model_id it was tuned for")
            if isinstance(question, Boolean):
                raise InvalidJob("an abstention policy applies to a Choice or Score, not a Boolean")
            from ..abstention import AbstentionSchemaError

            try:  # the checks abstention.decide makes on each answer, made before any call
                policy._check(tuple(question.option_ids), model_id, "probabilities", None)
            except AbstentionSchemaError as error:
                raise InvalidJob(f"question {id!r}: the abstention policy does not fit it: {error}") from error
        return DecisionJob(_Ask(question, id, policy, model_id))

    @staticmethod
    def collect(jobs: Mapping[str, DecisionJob]) -> DecisionJob:
        """Independent jobs; the value maps each key to its job's value, in
        ``jobs``' order."""
        if not isinstance(jobs, Mapping) or not jobs:
            raise InvalidJob("collect() needs a non-empty mapping of key -> DecisionJob")
        items = tuple(jobs.items())
        bad = [key for key, job in items if not isinstance(key, str) or not isinstance(job, DecisionJob)]
        if bad:
            raise InvalidJob(f"collect() needs string keys and DecisionJob values; bad entries {bad}")
        return DecisionJob(_Collect(items))

    def map(self, fn: Callable[[Any], Any]) -> DecisionJob:
        """This job with ``fn`` applied to its value (pure: no provider
        calls)."""
        if not callable(fn):
            raise InvalidJob(f"map() needs a callable, got {type(fn).__name__}")
        return DecisionJob(_Map(self, fn))

    def then(self, fn: Callable[[Any], Follow]) -> DecisionJob:
        """A dependent job: after this one succeeds, ``fn(value)`` returns a
        :class:`Follow` (the next job and its state), run once."""
        if not callable(fn):
            raise InvalidJob(f"then() needs a callable, got {type(fn).__name__}")
        return DecisionJob(_Then(self, fn))

    # ---------- inspection ----------

    def question_ids(self) -> tuple[str, ...]:
        """The ids of the questions known before running (not those a
        continuation will create), in scheduling order."""
        return tuple(ask.id for ask in _static_asks(self))

    def state(self) -> dict[str, Any]:
        """Plain data for a job of ``ask`` / ``collect`` only. A job holding
        a function (``map``, ``then``) is a runtime value: refused."""
        node = self._node
        if isinstance(node, _Ask):
            if node.policy is not None:
                return {
                    "ask": node.id,
                    "question": node.question.state(),
                    "policy": node.policy.state(),
                    "model_id": node.model_id,
                }
            return {"ask": node.id, "question": node.question.state()}
        if isinstance(node, _Collect):
            return {"collect": [[key, job.state()] for key, job in node.items]}
        raise TypeError(
            "a DecisionJob holding a runtime function (map / then) cannot be serialised; rebuild it where it runs"
        )

    def __reduce__(self) -> Any:
        return (_from_state, (self.state(),))

    def __copy__(self) -> DecisionJob:
        return self  # immutable

    def __deepcopy__(self, memo: Any) -> DecisionJob:
        return self  # immutable

    def __repr__(self) -> str:
        return f"DecisionJob({type(self._node).__name__.lstrip('_').lower()}, questions={list(self.question_ids())})"

    # ---------- running (the effect boundaries) ----------

    def run(self, provider: Any, *, state: Any, limits: Optional[Limits] = None) -> JobResult:
        """Run the job synchronously with ``provider`` (borrowed, never
        closed) over ``state``."""
        return _Runner(self, provider, state, limits or Limits()).run()

    async def arun(
        self,
        provider: Any,
        *,
        state: Any,
        limits: Optional[Limits] = None,
        cancel: Optional[asyncio.Event] = None,
    ) -> JobResult:
        """Run the job on the event loop: up to ``limits.max_concurrency``
        calls at once; setting ``cancel`` stops scheduling and returns a
        cancelled result (see the module docstring)."""
        return await _Runner(self, provider, state, limits or Limits()).arun(cancel)


def _from_state(state: Mapping[str, Any]) -> DecisionJob:
    from .schema import question_from_state

    if "ask" in state:
        policy = None
        if state.get("policy") is not None:
            from ..abstention import AbstentionPolicy

            policy = AbstentionPolicy.from_state(state["policy"])
        return DecisionJob.ask(
            question_from_state(state["question"]), id=state["ask"], policy=policy, model_id=state.get("model_id")
        )
    if "collect" in state:
        return DecisionJob.collect({key: _from_state(item) for key, item in state["collect"]})
    raise InvalidJob(f"not a DecisionJob state: {sorted(state)}")


def _static_asks(job: DecisionJob) -> Iterator[_Ask]:
    """The asks reachable without running a continuation, in order."""
    node = job._node
    if isinstance(node, _Ask):
        yield node
    elif isinstance(node, _Collect):
        for _, child in node.items:
            yield from _static_asks(child)
    else:
        yield from _static_asks(node.job)


# --- the runner -------------------------------------------------------------------------------

_PENDING = object()


class _ProviderError(Exception):
    """A provider hook (``check``, ``capabilities``, ``count_tokens``) raised
    something other than a declared refusal; carried to fail-fast."""

    def __init__(self, error: BaseException) -> None:
        super().__init__(str(error))
        self.error = error


def _modality(state: Any) -> tuple[str, int]:
    """The modality and batch size a provider's capabilities are checked
    against, read from the state."""
    import numpy as np
    import torch

    first = state[0] if isinstance(state, tuple) and state and not isinstance(state[0], str) else state
    if isinstance(first, (torch.Tensor, np.ndarray)) and getattr(first, "ndim", 0) >= 1:
        return "tensor", len(first)
    if isinstance(state, Sequence) and not isinstance(state, str) and all(isinstance(x, str) for x in state):
        return "text", len(state)
    raise UnsupportedCapability(
        f"a job's state must be a batch the provider reads (texts or a tensor / array), got {type(state).__name__}"
    )


@dataclass
class _Ready:
    ask: _Ask
    state: Any
    group: int  # the state's group: questions batch only within one state


class _Runner:
    """One run of a job: evaluate the tree in rounds — gather the asks that
    are ready, batch them per state, call the provider, then resolve maps,
    collections and continuations (each continuation once)."""

    def __init__(self, job: DecisionJob, provider: Any, state: Any, limits: Limits) -> None:
        if not isinstance(job, DecisionJob):
            raise InvalidJob(f"not a DecisionJob: {type(job).__name__}")
        if not callable(getattr(provider, "decide", None)) or not callable(getattr(provider, "capabilities", None)):
            raise InvalidJob("a provider needs capabilities() and decide(question, inputs)")
        self.job = job
        self.provider = provider
        self.state = state
        self.limits = limits
        self.answers: dict[str, Any] = {}  # question id -> its value
        self.outcomes: dict[str, QuestionOutcome] = {}
        self.order: dict[str, None] = {}  # question ids in scheduling order
        self.calls = 0  # provider calls made
        self.launched = 0  # calls scheduled (bounded by max_requests)
        self.memo: dict[tuple[int, ...], Any] = {}  # path -> map value / Follow, computed once
        self.states: list[Any] = [state]  # group index -> state
        self.seen_ids: set[str] = set()
        self.registered: dict[str, None] = {}  # question ids known so far, in order
        self.checked: set[tuple[str, int]] = set()
        self.started = time.monotonic()
        if callable(getattr(provider, "decide_many", None)):
            # A batching provider declares its cap (None: any number per call).
            cap = _optional_positive(getattr(provider, "max_questions", None), "provider.max_questions")
        else:
            cap = 1  # one question per call
        caps = [c for c in (cap, limits.max_questions) if c is not None]
        self.cap = min(caps) if caps else sys.maxsize
        if limits.max_tokens is not None and not callable(getattr(provider, "count_tokens", None)):
            raise InvalidJob(
                f"limits.max_tokens={limits.max_tokens} cannot be enforced: the provider has no count_tokens()"
            )
        # Every question known before running is validated before any call.
        try:
            self._register([(ask, 0) for ask in _static_asks(job)])
        except _ProviderError as wrapped:
            failure = JobFailed(
                f"the provider's pre-call check failed ({type(wrapped.error).__name__}: {wrapped.error})",
                outcomes={},
                failed=(),
                skipped=tuple(self.registered),
            )
            raise failure from wrapped.error

    def _ordered(self) -> dict[str, QuestionOutcome]:
        """The outcomes in scheduling order (not completion order)."""
        return {question_id: self.outcomes[question_id] for question_id in self.order if question_id in self.outcomes}

    # ---------- validation ----------

    def _register(self, asks: Sequence[tuple[_Ask, int]]) -> None:
        for ask, group in asks:
            if ask.id in self.seen_ids:
                raise InvalidJob(f"duplicate question id {ask.id!r}", outcomes=self._ordered())
            self.seen_ids.add(ask.id)
            self.registered[ask.id] = None
            self._check(ask, group)

    def _check(self, ask: _Ask, group: int) -> None:
        key = (ask.id, group)
        if key in self.checked:
            return
        state = self.states[group]
        try:
            check = getattr(self.provider, "check", None)
            if callable(check):
                check(ask.question, state)
            else:
                modality, batch_size = _modality(state)
                self.provider.capabilities().check(ask.question, modality=modality, batch_size=batch_size)
        except (UnsupportedCapability, InvalidDecisionRequest) as error:
            raise InvalidJob(f"question {ask.id!r} cannot be served: {error}", outcomes=self._ordered()) from error
        except Exception as error:  # the provider's own check failed: a provider failure
            raise _ProviderError(error) from error
        self.checked.add(key)

    # ---------- evaluation ----------

    def _evaluate(self, job: DecisionJob, group: int, depth: int, path: tuple[int, ...], ready: list[_Ready]) -> Any:
        node = job._node
        if isinstance(node, _Ask):
            if node.id in self.answers:
                return self.answers[node.id]
            ready.append(_Ready(node, self.states[group], group))
            return _PENDING
        if isinstance(node, _Collect):
            values = [self._evaluate(child, group, depth, (*path, i), ready) for i, (_, child) in enumerate(node.items)]
            if any(value is _PENDING for value in values):
                return _PENDING
            return {key: value for (key, _), value in zip(node.items, values, strict=True)}
        if isinstance(node, _Map):
            value = self._evaluate(node.job, group, depth, (*path, 0), ready)
            if value is _PENDING:
                return _PENDING
            if path not in self.memo:
                self.memo[path] = node.fn(value)
            return self.memo[path]
        value = self._evaluate(node.job, group, depth, (*path, 0), ready)
        if value is _PENDING:
            return _PENDING
        follow = self.memo.get(path)
        if follow is None:
            if depth + 1 > self.limits.max_depth:
                raise JobLimitExceeded(
                    f"continuation depth {depth + 1} exceeds max_depth={self.limits.max_depth}",
                    outcomes=self._ordered(),
                )
            follow = node.fn(value)  # runs once, after its prerequisite succeeded
            if not isinstance(follow, Follow) or not isinstance(follow.job, DecisionJob):
                raise InvalidJob(
                    f"a continuation must return Follow(job, state=...), got {type(follow).__name__}",
                    outcomes=self._ordered(),
                )
            # Questions batch per state object: a continuation that returns
            # a state already in use joins that state's group.
            follow_group = next((i for i, known in enumerate(self.states) if known is follow.state), None)
            if follow_group is None:
                self.states.append(follow.state)
                follow_group = len(self.states) - 1
            self.memo[path] = (follow, follow_group)
            self._register([(ask, follow_group) for ask in _static_asks(follow.job)])
        else:
            follow, follow_group = follow
        return self._evaluate(follow.job, follow_group, depth + 1, (*path, 1), ready)

    def _round(self) -> tuple[Any, list[list[_Ready]]]:
        """The job's value (or pending) and this round's provider calls."""
        ready: list[_Ready] = []

        def skip_ready() -> None:  # questions already ready are reported, never sent
            for item in ready:
                self.order.setdefault(item.ask.id, None)
                self.outcomes.setdefault(item.ask.id, QuestionOutcome(item.ask.id, "skipped"))

        try:
            value = self._evaluate(self.job, 0, 0, (), ready)
            if value is not _PENDING:
                return value, []
            groups: dict[int, list[_Ready]] = {}
            for item in ready:
                groups.setdefault(item.group, []).append(item)  # stable: first appearance
            chunks = [chunk for items in groups.values() for chunk in self._chunks(items)]
        except JobError as error:
            skip_ready()
            error.outcomes = self._ordered()
            error.completed = {k: v for k, v in error.outcomes.items() if v.kind == "answered"}
            raise
        except _ProviderError as wrapped:  # fail-fast: answers already received are kept
            skip_ready()
            failure = JobFailed(
                f"a provider hook failed ({type(wrapped.error).__name__}: {wrapped.error}); "
                f"{len(ready)} ready question(s) skipped",
                outcomes=self._ordered(),
                failed=(),
                skipped=[item.ask.id for item in ready],
            )
            raise failure from wrapped.error
        for chunk in chunks:
            self.order.update(dict.fromkeys(item.ask.id for item in chunk))
        return value, chunks

    def _count_tokens(self, questions: list[Question], state: Any) -> Any:
        try:
            return self.provider.count_tokens(questions, state)
        except Exception as error:  # the provider's token count failed: a provider failure
            raise _ProviderError(error) from error

    def _chunks(self, items: list[_Ready]) -> list[list[_Ready]]:
        """``ceil(len / cap)`` calls of up to ``cap`` questions, in order,
        each under the token cap when one is set."""
        if self.limits.max_tokens is None:
            return [items[i : i + self.cap] for i in range(0, len(items), self.cap)]
        chunks: list[list[_Ready]] = []
        current: list[_Ready] = []
        for item in items:
            candidate = [*current, item]
            tokens = self._count_tokens([r.ask.question for r in candidate], item.state)
            if len(candidate) <= self.cap and tokens <= self.limits.max_tokens:
                current = candidate
                continue
            if not current:
                raise InvalidJob(
                    f"question {item.ask.id!r} alone exceeds max_tokens={self.limits.max_tokens} ({tokens} tokens)",
                    outcomes=self._ordered(),
                )
            chunks.append(current)
            current = [item]
            alone = self._count_tokens([item.ask.question], item.state)
            if alone > self.limits.max_tokens:
                raise InvalidJob(
                    f"question {item.ask.id!r} alone exceeds max_tokens={self.limits.max_tokens} ({alone} tokens)",
                    outcomes=self._ordered(),
                )
        if current:
            chunks.append(current)
        return chunks

    # ---------- calling the provider ----------

    def _limit(self) -> Optional[tuple[type[JobError], str]]:
        """The limit the next call would break, if any."""
        if self.limits.max_requests is not None and self.launched >= self.limits.max_requests:
            return JobLimitExceeded, f"the next call would exceed max_requests={self.limits.max_requests}"
        if self._expired():
            return JobTimeout, f"the run exceeded its timeout of {self.limits.timeout}s"
        return None

    def _expired(self) -> bool:
        return self.limits.timeout is not None and time.monotonic() - self.started > self.limits.timeout

    def _remaining(self) -> Optional[float]:
        if self.limits.timeout is None:
            return None
        return max(self.limits.timeout - (time.monotonic() - self.started), 0.0)

    @staticmethod
    def _answers(chunk: list[_Ready], answered: Any) -> list[Sequence[DecisionResult]]:
        """The provider's answers, checked against the questions they claim
        to answer: one result list per question, one result per input row,
        each a result of that question (its digest)."""
        answers = list(answered)
        if len(answers) != len(chunk):
            raise InvalidDecisionResponse(f"the provider answered {len(answers)} of {len(chunk)} questions in one call")
        try:
            rows: Optional[int] = _modality(chunk[0].state)[1]
        except UnsupportedCapability:  # a provider-specific input: its row count is the provider's to check
            rows = None
        checked = []
        for item, results in zip(chunk, answers, strict=True):
            if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
                raise InvalidDecisionResponse(
                    f"question {item.ask.id!r}: expected a sequence of results (one per input row), got "
                    f"{type(results).__name__}"
                )
            results = tuple(results)
            if rows is not None and len(results) != rows:
                raise InvalidDecisionResponse(f"question {item.ask.id!r}: {len(results)} results for {rows} input rows")
            digest = item.ask.question.digest()
            wrong = [
                i
                for i, result in enumerate(results)
                if not isinstance(result, (ChoiceResult, BooleanResult, ScoreResult))
                or result.question_digest != digest
            ]
            if wrong:
                raise InvalidDecisionResponse(
                    f"question {item.ask.id!r}: results {wrong} do not answer it (another question's digest, or "
                    "not a decision result)"
                )
            checked.append(results)
        return checked

    def _call(self, chunk: list[_Ready]) -> list[Sequence[DecisionResult]]:
        questions = [item.ask.question for item in chunk]
        state = chunk[0].state
        if len(chunk) == 1 and not callable(getattr(self.provider, "decide_many", None)):
            return self._answers(chunk, [self.provider.decide(questions[0], state)])
        return self._answers(chunk, self.provider.decide_many(questions, state))

    def _is_async(self, chunk: list[_Ready]) -> bool:
        if callable(getattr(self.provider, "adecide_many", None)):
            return True
        return len(chunk) == 1 and callable(getattr(self.provider, "adecide", None))

    async def _acall(self, chunk: list[_Ready], threads: list[asyncio.Future[Any]]) -> list[Sequence[DecisionResult]]:
        questions = [item.ask.question for item in chunk]
        state = chunk[0].state
        many: Optional[Callable[..., Awaitable[Any]]] = getattr(self.provider, "adecide_many", None)
        if callable(many):
            return self._answers(chunk, await many(questions, state))
        one: Optional[Callable[..., Awaitable[Any]]] = getattr(self.provider, "adecide", None)
        if callable(one) and len(chunk) == 1:
            return self._answers(chunk, [await one(questions[0], state)])
        # A synchronous provider runs in a worker thread, never alongside
        # another call (its methods need not be thread-safe). A thread cannot
        # be interrupted: it is shielded, and the run waits for it before
        # returning, so the borrowed provider is never left in use.
        work = asyncio.ensure_future(asyncio.to_thread(self._call, chunk))
        threads.append(work)
        return await asyncio.shield(work)

    def _record(self, chunk: list[_Ready], answered: list[Sequence[DecisionResult]]) -> None:
        for item, results in zip(chunk, answered, strict=True):
            results = tuple(results)
            ask = item.ask
            if ask.policy is None:
                rows = tuple(RowOutcome("answered", result) for result in results)
                self.answers[ask.id] = results
            else:
                from ..abstention import decide

                decisions = tuple(decide(result, ask.policy, model_id=cast(str, ask.model_id)) for result in results)
                rows = tuple(
                    RowOutcome("answered" if d.outcome.status == "accepted" else "abstained", d.result, d)
                    for d in decisions
                )
                self.answers[ask.id] = decisions
            self.outcomes[ask.id] = QuestionOutcome(ask.id, "answered", rows, sent=True)

    def _mark(self, chunks: Iterable[list[_Ready]], kind: str, *, sent: bool) -> None:
        for chunk in chunks:
            for item in chunk:
                self.outcomes[item.ask.id] = QuestionOutcome(item.ask.id, kind, sent=sent)

    def _fail(
        self, failures: Sequence[tuple[list[_Ready], BaseException]], pending: Sequence[list[_Ready]]
    ) -> JobFailed:
        for chunk, error in failures:
            for item in chunk:
                self.outcomes[item.ask.id] = QuestionOutcome(item.ask.id, "failed", error=error, sent=True)
        self._mark(pending, "skipped", sent=False)
        error = failures[0][1]
        skipped = [item.ask.id for chunk in pending for item in chunk]
        failure = JobFailed(
            f"a provider call failed ({type(error).__name__}: {error}); {len(skipped)} ready question(s) skipped",
            outcomes=self._ordered(),
            failed=[item.ask.id for chunk, _ in failures for item in chunk],
            skipped=skipped,
        )
        failure.__cause__ = error
        return failure

    # ---------- sync ----------

    def run(self) -> JobResult:
        while True:
            value, chunks = self._round()
            if not chunks:
                return JobResult(value, self._ordered(), self.calls)
            for index, chunk in enumerate(chunks):
                limit = self._limit()
                if limit is not None:  # nothing more is sent
                    self._mark(chunks[index:], "skipped", sent=False)
                    raise limit[0](limit[1], outcomes=self._ordered())
                self.launched += 1
                self.calls += 1
                try:
                    answered = self._call(chunk)
                    self._record(chunk, answered)
                except Exception as error:  # fail-fast: nothing more is scheduled
                    raise self._fail([(chunk, error)], chunks[index + 1 :]) from error

    # ---------- async ----------

    async def arun(self, cancel: Optional[asyncio.Event]) -> JobResult:
        threads: list[asyncio.Future[Any]] = []  # worker threads of synchronous calls
        in_flight: dict[asyncio.Future[Any], tuple[int, list[_Ready], bool]] = {}  # task -> (launch, chunk, sync)
        started: set[int] = set()  # launches whose provider call began: those requests were sent
        if cancel is not None and not isinstance(cancel, asyncio.Event):
            raise InvalidJob(f"cancel must be an asyncio.Event, got {type(cancel).__name__}")
        # Set at any point (even if cleared again), the event cancels the run.
        stop: Optional[asyncio.Future[Any]] = None if cancel is None else asyncio.ensure_future(cancel.wait())
        launches = 0

        def cancelled() -> bool:
            if stop is None:
                return False
            if stop.done() and not stop.cancelled() and stop.exception() is not None:
                # The watcher itself failed (an event bound to another loop): not a cancellation.
                raise InvalidJob(
                    f"the cancel event cannot be awaited in this event loop: {stop.exception()}"
                ) from stop.exception()
            return stop.done() or cast(asyncio.Event, cancel).is_set()

        async def call(launch: int, chunk: list[_Ready]) -> list[Sequence[DecisionResult]]:
            started.add(launch)
            self.calls += 1
            return await self._acall(chunk, threads)

        def abandon() -> None:
            """Cancel the job's in-flight calls; a started one was sent."""
            for task, (launch, chunk, _) in in_flight.items():
                task.cancel()
                self._mark([chunk], "cancelled", sent=launch in started)

        try:
            while True:
                if cancelled():  # before any continuation runs; known questions were never sent
                    for question_id in self.registered:
                        if question_id not in self.outcomes:
                            self.order.setdefault(question_id, None)
                            self.outcomes[question_id] = QuestionOutcome(question_id, "cancelled", sent=False)
                    return JobResult(None, self._ordered(), self.calls, status="cancelled")
                value, chunks = self._round()
                if not chunks:
                    return JobResult(value, self._ordered(), self.calls)
                queue = list(chunks)
                stopping: Optional[tuple[type[JobError], str]] = None
                while queue or in_flight:
                    if cancelled():
                        self._mark(queue, "cancelled", sent=False)
                        abandon()
                        return JobResult(None, self._ordered(), self.calls, status="cancelled")
                    while queue and stopping is None:
                        sync = not self._is_async(queue[0])
                        if in_flight and (
                            sync
                            or len(in_flight) >= self.limits.max_concurrency
                            or any(entry[2] for entry in in_flight.values())
                        ):
                            break  # a synchronous call runs alone; others up to max_concurrency
                        stopping = self._limit()
                        if stopping is not None:
                            break  # nothing more is sent; calls in flight finish first
                        chunk = queue.pop(0)
                        self.launched += 1
                        in_flight[asyncio.ensure_future(call(launches, chunk))] = (launches, chunk, sync)
                        launches += 1
                    if stopping is not None and not in_flight:
                        self._mark(queue, "skipped", sent=False)
                        raise stopping[0](stopping[1], outcomes=self._ordered())
                    waiters: list[asyncio.Future[Any]] = [*in_flight, *([stop] if stop is not None else [])]
                    done, _ = await asyncio.wait(
                        waiters, timeout=self._remaining(), return_when=asyncio.FIRST_COMPLETED
                    )
                    if not done:  # the timeout elapsed with calls in flight
                        self._mark(queue, "skipped", sent=False)
                        abandon()
                        raise JobTimeout(
                            f"the run exceeded its timeout of {self.limits.timeout}s; in-flight requests that were "
                            "sent are not rolled back",
                            outcomes=self._ordered(),
                        )
                    failures: list[tuple[list[_Ready], BaseException]] = []
                    for _, task in sorted((in_flight[t][0], t) for t in done if t in in_flight):
                        _, chunk, _ = in_flight.pop(task)
                        if task.cancelled():  # the job cancels nothing before this point: the provider did
                            failures.append((chunk, ProviderFailure("the provider's call was cancelled")))
                            continue
                        try:
                            self._record(chunk, task.result())
                        except Exception as error:  # recorded after every success of this tick
                            failures.append((chunk, error))
                    if failures:  # fail-fast: nothing more is scheduled
                        abandon()
                        raise self._fail(failures, queue) from failures[0][1]
        except asyncio.CancelledError:
            abandon()
            raise
        finally:
            if stop is not None:
                stop.cancel()
            for task, (launch, chunk, _) in in_flight.items():
                task.cancel()
                for item in chunk:  # a sent request is never presented as rolled back
                    self.outcomes.setdefault(
                        item.ask.id, QuestionOutcome(item.ask.id, "cancelled", sent=launch in started)
                    )
            # Wait for the job's own tasks and threads; a cancellation that
            # arrives meanwhile is delivered once they are done, never lost.
            pending = [
                work for work in (*in_flight, *threads, *([stop] if stop is not None else [])) if not work.done()
            ]
            interrupted = False
            while pending:
                try:
                    await asyncio.shield(asyncio.gather(*pending, return_exceptions=True))
                except asyncio.CancelledError:
                    interrupted = True
                pending = [work for work in pending if not work.done()]
            if interrupted:
                raise asyncio.CancelledError
