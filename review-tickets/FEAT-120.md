---
id: FEAT-120
title: "[FEAT-120] Split DecisionJob's _Runner.arun into launch, harvest and drain"
labels: ["enhancement", "area: decisions", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-41"]
depends_on: []
related_issues: [245, 557]
status: READY
---

**Category:** Decisions · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-120: `_Runner.arun` in `src/nnx/decisions/jobs.py` is 121 lines with three nested closures and a cyclomatic complexity of 42: scheduling, completion handling and the exit protocol are interleaved in one body. This ticket turns the closures and the three phases into `_Runner` methods (`_cancelled`, `_started_call`, `_abandon`, `_launch`, `_harvest`, `_drain`) without changing any behaviour.

## Context

**What happens today.** The closures read the locals of `arun`:

```python
        def cancelled() -> bool:
            if stop is None:
                return False
```

```python
        async def call(launch: int, chunk: list[_Ready]) -> list[Sequence[DecisionResult]]:
            started.add(launch)
            self.calls += 1
            return await self._acall(chunk, threads)
```

and the launch loop, the completion loop and the `finally` drain sit inline (L1081-L1095, L1111-L1120, L1127-L1149).

**Why it matters.** The async runner carries the job's cancellation, timeout, fail-fast and "a sent request is never presented as rolled back" guarantees. Small named methods make each guarantee readable and testable on its own.

**History.** #245 (FEAT-024) introduced applicative batching and the async runner; #557 (FIX-089) hardened `Limits` and inputs. Both keep their behaviour: this is a move, not a redesign. Splitting `decisions/benchmark.py` is the other half of SMELL-41 (FEAT-121, blocked by DEC-017).

**Background for the worker.**
- `_Ready`: one ask whose inputs are known and that can be sent (a chunk is a list of them, one provider call).
- `in_flight`: `{task: (launch number, chunk, is_synchronous)}` for the calls started and not yet harvested.
- `started`: the launch numbers whose provider call began, so their requests count as sent.
- `stop`: the future that watches the `cancel` event (`None` when no event was given).
- `threads`: the worker threads of synchronous provider calls run through `asyncio.to_thread`; the run waits for them before returning.
- `self._mark(chunks, kind, sent=...)`, `self._fail(failures, pending)`, `self._limit()`, `self._remaining()`, `self._is_async(chunk)`, `self._acall(chunk, threads)` already exist on `_Runner` and stay unchanged.

**Proposal.**
1. In `src/nnx/decisions/jobs.py`, directly after the `# ---------- async ----------` comment (L1027) and before `async def arun`, add these methods. Each body is the corresponding closure or block of today's `arun`, with the captured locals turned into parameters:

   ```python
       def _cancelled(self, cancel: Optional[asyncio.Event], stop: Optional[asyncio.Future[Any]]) -> bool:
           """Whether ``cancel`` was set since the run began (``stop`` watches
           it). A watcher that failed — an event bound to another loop — is
           no cancellation: it raises :class:`InvalidJob`."""
           if stop is None:
               return False
           if stop.done() and not stop.cancelled() and stop.exception() is not None:
               # The watcher itself failed (an event bound to another loop): not a cancellation.
               raise InvalidJob(
                   f"the cancel event cannot be awaited in this event loop: {stop.exception()}"
               ) from stop.exception()
           return stop.done() or cast(asyncio.Event, cancel).is_set()

       async def _started_call(
           self, launch: int, chunk: list[_Ready], started: set[int], threads: list[asyncio.Future[Any]]
       ) -> list[Sequence[DecisionResult]]:
           """One provider call for ``chunk``: ``launch`` counts as sent once it begins."""
           started.add(launch)
           self.calls += 1
           return await self._acall(chunk, threads)

       def _abandon(
           self, in_flight: dict[asyncio.Future[Any], tuple[int, list[_Ready], bool]], started: set[int]
       ) -> None:
           """Cancel the job's in-flight calls; a started one was sent."""
           for task, (launch, chunk, _) in in_flight.items():
               task.cancel()
               self._mark([chunk], "cancelled", sent=launch in started)

       def _launch(
           self,
           queue: list[list[_Ready]],
           in_flight: dict[asyncio.Future[Any], tuple[int, list[_Ready], bool]],
           started: set[int],
           threads: list[asyncio.Future[Any]],
           launches: int,
       ) -> tuple[Optional[tuple[type[JobError], str]], int]:
           """Start queued chunks while the concurrency rules allow (a
           synchronous call runs alone; others up to ``max_concurrency``).
           Returns the limit that stopped sending (``None`` when none did) and
           the next launch number."""
           while queue:
               sync = not self._is_async(queue[0])
               if in_flight and (
                   sync
                   or len(in_flight) >= self.limits.max_concurrency
                   or any(entry[2] for entry in in_flight.values())
               ):
                   break  # a synchronous call runs alone; others up to max_concurrency
               stopping = self._limit()
               if stopping is not None:
                   return stopping, launches  # nothing more is sent; calls in flight finish first
               chunk = queue.pop(0)
               self.launched += 1
               in_flight[asyncio.ensure_future(self._started_call(launches, chunk, started, threads))] = (
                   launches,
                   chunk,
                   sync,
               )
               launches += 1
           return None, launches

       def _harvest(
           self,
           done: set[asyncio.Future[Any]],
           in_flight: dict[asyncio.Future[Any], tuple[int, list[_Ready], bool]],
       ) -> list[tuple[list[_Ready], BaseException]]:
           """Record each finished call of ``done`` in launch order and remove
           it from ``in_flight``; return the failures, recorded after every
           success of this tick (a call the provider cancelled is one)."""
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
           return failures

       async def _drain(
           self,
           in_flight: dict[asyncio.Future[Any], tuple[int, list[_Ready], bool]],
           threads: list[asyncio.Future[Any]],
           stop: Optional[asyncio.Future[Any]],
           started: set[int],
       ) -> None:
           """The run's exit, however it ends: cancel the watcher and every
           call still in flight (a sent request is never presented as rolled
           back), then wait for the job's own tasks and threads; a
           cancellation that arrives meanwhile is delivered once they are
           done, never lost."""
           <the body of today's `finally:` block, L1128-L1149, unchanged>
   ```

2. Rewrite `arun` (L1029-L1149) to:

   ```python
       async def arun(self, cancel: Optional[asyncio.Event]) -> JobResult:
           threads: list[asyncio.Future[Any]] = []  # worker threads of synchronous calls
           in_flight: dict[asyncio.Future[Any], tuple[int, list[_Ready], bool]] = {}  # task -> (launch, chunk, sync)
           started: set[int] = set()  # launches whose provider call began: those requests were sent
           if cancel is not None and not isinstance(cancel, asyncio.Event):
               raise InvalidJob(f"cancel must be an asyncio.Event, got {type(cancel).__name__}")
           stop: Optional[asyncio.Future[Any]] = None
           launches = 0
           try:
               if cancel is not None:
                   # Set at any point (even if cleared again), the event cancels the run.
                   stop = asyncio.ensure_future(cancel.wait())
                   await asyncio.sleep(0)  # the watcher starts: an unusable event is refused before any call
               while True:
                   if self._cancelled(cancel, stop):  # before any continuation runs; known questions were never sent
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
                       if self._cancelled(cancel, stop):
                           self._mark(queue, "cancelled", sent=False)
                           self._abandon(in_flight, started)
                           return JobResult(None, self._ordered(), self.calls, status="cancelled")
                       if stopping is None:
                           stopping, launches = self._launch(queue, in_flight, started, threads, launches)
                       if stopping is not None and not in_flight:
                           self._mark(queue, "skipped", sent=False)
                           raise stopping[0](stopping[1], outcomes=self._ordered())
                       waiters: list[asyncio.Future[Any]] = [*in_flight, *([stop] if stop is not None else [])]
                       done, _ = await asyncio.wait(
                           waiters, timeout=self._remaining(), return_when=asyncio.FIRST_COMPLETED
                       )
                       if not done:  # the timeout elapsed with calls in flight
                           self._mark(queue, "skipped", sent=False)
                           self._abandon(in_flight, started)
                           raise JobTimeout(
                               f"the run exceeded its timeout of {self.limits.timeout}s; in-flight requests that were "
                               "sent are not rolled back",
                               outcomes=self._ordered(),
                           )
                       failures = self._harvest(done, in_flight)
                       if failures:  # fail-fast: nothing more is scheduled
                           self._abandon(in_flight, started)
                           raise self._fail(failures, queue) from failures[0][1]
           except asyncio.CancelledError:
               self._abandon(in_flight, started)
               raise
           finally:
               await self._drain(in_flight, threads, stop, started)
   ```

   `if stopping is None:` around `_launch` replaces today's `while queue and stopping is None:` condition: once a limit stops sending, nothing more is launched in that round.

3. Keep every comment of the moved code. Run `uv run ruff format src/nnx/decisions/jobs.py`.

No message, outcome, ordering or call count changes, so no test changes and no CHANGELOG entry. Nothing outside `jobs.py` references `_Runner` (`git grep -n "_Runner" src tests` shows only `jobs.py`), and no test monkeypatches its methods.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] `arun` defines no nested function. Check: `uv run python -c "import ast; t=ast.parse(open('src/nnx/decisions/jobs.py').read()); r=next(n for n in ast.walk(t) if isinstance(n, ast.ClassDef) and n.name=='_Runner'); a=next(n for n in r.body if getattr(n, 'name', '')=='arun'); print(sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(a)) - 1)"` prints `0`
- [ ] `_Runner` has the methods `_cancelled`, `_started_call`, `_abandon`, `_launch`, `_harvest` and `_drain`. Check: `grep -cE "    (async )?def _(cancelled|started_call|abandon|launch|harvest|drain)\(" src/nnx/decisions/jobs.py` returns `6`
- [ ] The job suite passes unchanged (cancellation, timeouts, fail-fast, concurrency, sent flags). Check: `uv run pytest tests/test_decision_jobs.py`
- [ ] The other decision suites pass. Check: `uv run pytest tests -k decision`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [jobs.py L1029-L1149](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/jobs.py#L1029-L1149) | `_Runner.arun`: 121 lines, complexity 42 |
| [jobs.py L1038-L1057](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/jobs.py#L1038-L1057) | the closures `cancelled`, `call`, `abandon` |
| [jobs.py L1081-L1095](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/jobs.py#L1081-L1095) | the launch loop |
| [jobs.py L1111-L1123](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/jobs.py#L1111-L1123) | the completion loop and fail-fast |
| [jobs.py L1127-L1149](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/jobs.py#L1127-L1149) | the `finally` drain |
| #245 | applicative batching and the async runner |
| #557 | `Limits` validation and custom inputs |

## Scope

**In:** `_Runner.arun` and the six new methods.

**Out:** any change to scheduling, cancellation or timeout behaviour; the synchronous `run`. Splitting `decisions/benchmark.py` (FEAT-121). The abstention imports in `jobs.py` (FEAT-060) and the other function-local imports (FEAT-136).

## Dependencies

**Blocked by:** none
**Blocks:** none
**Related:** #245, #557
