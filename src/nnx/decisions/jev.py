"""Typed decisions on Jev models through the TypeSafe Python SDK (FEAT-010).

:class:`JevProvider` answers :class:`Choice`, :class:`Boolean` and
:class:`Score` questions about text with one ``system_one`` call per input,
through the ``typesafe-sdk`` client — NNx adds no second transport stack::

    from nnx.decisions import Boolean, Choice, JevProvider

    with JevProvider(model="jev-1.13.0") as provider:  # TYPESAFE_API_KEY from the environment
        topic, spam = provider.decide_many(
            [Choice("Topic?", (("t-sport", "sports"), ("t-econ", "the economy"))), Boolean("Is this spam?")],
            texts,
        )

Install the extra first: ``pip install "thekaveh-nnx[jev]"`` (tested against
``typesafe-sdk>=0.7,<0.8``). Importing :mod:`nnx` or :mod:`nnx.decisions`
never imports the SDK; building a provider without an injected client does,
and raises an :class:`ImportError` naming the extra when it is missing.

**Translation.** A :class:`Choice` becomes a Jev ``choice`` keyed by its
options' descriptions — the model-facing text, never the bookkeeping ids —
a :class:`Score` a ``score`` whose criteria are its levels' descriptions in
order (lowest first), a :class:`Boolean` a ``noul``; the prompt is the
``instructions``. Every question of one :meth:`JevProvider.decide_many` goes
into the same call, one call per input. Answers are realigned to each
question's own option order and validated
(:func:`~nnx.decisions.validate_response`); a Score's expected score stays
in ``vendor_score``.

**Confidence.** A Choice's or Score's ``confidence`` is Jev's own certainty
in its selection, recorded as ``raw["provider_confidence"]`` — it is not a
probability that the answer is correct. A Boolean has none.

**Metadata.** Each result's ``raw`` is plain JSON: ``provider``, the
``model`` the service resolved (even for an alias request), the
``requested_model`` when one was set, the ``request_id`` and the reported
``usage`` token counts. A field the service did not report is absent, and
clients, headers and credentials are never recorded.

**Retries and failures.** The SDK's :class:`~typesafe_sdk.RetryPolicy`
owns retries (pass ``retry=`` when the provider builds its client); the
adapter makes each request once. What still fails becomes a typed
:class:`JevError` — :class:`JevTimeout`, :class:`JevAuthenticationError`,
:class:`JevRateLimited` or :class:`JevMalformedResponse` — keeping the SDK
error as ``__cause__`` and, when the service answered, its ``request_id``
(a timeout or a lost connection has none).

**Cost.** Every text is its own ``system_one`` request, so one provider
call over ``n`` texts sends ``n`` requests: a benchmark ``Budget.max_calls``
or a job's ``Limits.max_requests`` counts provider calls, not requests — set
``max_batch`` to bound requests per call. A failure on one text stops the
call; answers already received for earlier texts are not returned.

**Lifecycle.** An injected ``client`` / ``async_client`` stays the
caller's: the provider never closes it, and never builds a second client
beside it — with only a sync client injected the provider is sync-only
(``adecide`` / ``adecide_many`` are ``None``, so a decision job's ``arun``
runs it in its own worker thread, one call at a time); with only an async
one, the sync methods are refused. With no client injected the provider builds its own (lazily, from
``api_key`` / ``base_url`` / ``timeout`` / ``retry`` or the SDK's
environment variables): :meth:`JevProvider.close` (``with``) closes a built
sync client, :meth:`JevProvider.aclose` (``async with``) both — after an
error or a cancellation too. A built async client belongs to one event
loop; a call (or :meth:`JevProvider.aclose`) from another loop drops it
with a ``ResourceWarning`` — close it with ``async with`` inside each
``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from .providers import Capabilities, Question
from .schema import (
    Boolean,
    Choice,
    DecisionResult,
    InvalidDecisionRequest,
    InvalidDecisionResponse,
    ProviderFailure,
    Score,
    UnsupportedCapability,
    validate_response,
)

__all__ = [
    "SDK_RANGE",
    "JevAuthenticationError",
    "JevError",
    "JevMalformedResponse",
    "JevProvider",
    "JevRateLimited",
    "JevTimeout",
]

SDK_RANGE = ">=0.7,<0.8"
"""The ``typesafe-sdk`` versions the adapter is tested against (the ``jev`` extra's pin)."""

_INSTALL_HINT = 'JevProvider needs the TypeSafe SDK: pip install "thekaveh-nnx[jev]" (or "typesafe-sdk{range}")'


def _sdk() -> Any:
    """The ``typesafe_sdk`` module, imported on demand."""
    try:
        import typesafe_sdk
    except ImportError as error:
        raise ImportError(_INSTALL_HINT.format(range=SDK_RANGE)) from error
    return typesafe_sdk


# --- errors -----------------------------------------------------------------------------------


class JevError(ProviderFailure):
    """A Jev request failed (the SDK error is the ``__cause__``).
    ``request_id`` is the service's request id when it sent one."""

    def __init__(self, message: str, *, request_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.request_id = request_id

    def __str__(self) -> str:
        message = super().__str__()
        if self.request_id is not None and self.request_id not in message:
            message += f" (request_id={self.request_id})"
        return message

    def __reduce__(self) -> Any:
        return (_rebuild_error, (type(self), self.args[0] if self.args else "", self.__dict__.copy()))


def _rebuild_error(cls: type[JevError], message: str, state: dict[str, Any]) -> JevError:
    # An OSError-based class (JevTimeout) must be allocated by OSError.__new__.
    base: Any = OSError if issubclass(cls, OSError) else Exception
    error = base.__new__(cls, message)
    error.args = (message,)
    error.__dict__.update(state)
    return error


class JevTimeout(JevError, TimeoutError):
    """The request exceeded its timeout (after the SDK's retries)."""


class JevAuthenticationError(JevError):
    """The API key was rejected or lacks access (HTTP 401 / 403)."""


class JevRateLimited(JevError):
    """The service refused the request for its rate limit (HTTP 429, after
    the SDK's retries). ``retry_after_ms`` is the wait it asked for, if any."""

    def __init__(self, message: str, *, request_id: Optional[str] = None, retry_after_ms: Optional[float] = None):
        super().__init__(message, request_id=request_id)
        self.retry_after_ms = retry_after_ms


class JevMalformedResponse(JevError, InvalidDecisionResponse):
    """The service answered with data that does not fit the request: a body
    the SDK cannot parse, a missing answer, or a distribution that does not
    fit the question."""


def _translate(error: BaseException) -> Optional[JevError]:
    """The typed adapter error for an SDK failure, or ``None`` when ``error``
    is not one (it then propagates unchanged)."""
    try:
        sdk = _sdk()
    except ImportError:
        return None
    if not isinstance(error, sdk.TypeSafeError):
        return None
    request_id = getattr(error, "request_id", None)
    request_id = request_id if isinstance(request_id, str) else None
    message = f"{type(error).__name__}: {error}"
    if isinstance(error, sdk.TypeSafeAPITimeoutError):
        return JevTimeout(message)
    if isinstance(error, (sdk.TypeSafeAuthenticationError, sdk.TypeSafePermissionDeniedError)):
        return JevAuthenticationError(message, request_id=request_id)
    if isinstance(error, sdk.TypeSafeRateLimitError):
        return JevRateLimited(message, request_id=request_id, retry_after_ms=getattr(error, "retry_after_ms", None))
    if isinstance(error, sdk.TypeSafeAPIResponseValidationError):
        return JevMalformedResponse(message, request_id=request_id)
    return JevError(message, request_id=request_id)


# --- translation ------------------------------------------------------------------------------


def _wire(question: Question) -> dict[str, Any]:
    """The Jev question for an NNx question: only model-facing text."""
    if isinstance(question, Choice):
        return {
            "type": "choice",
            "instructions": question.prompt,
            "criteria": {option.description: None for option in question.options},
        }
    if isinstance(question, Score):
        return {
            "type": "score",
            "instructions": question.prompt,
            "criteria": [level.description for level in question.levels],
        }
    return {"type": "noul", "instructions": question.prompt}


def _texts(inputs: Any) -> list[str]:
    if (
        isinstance(inputs, Sequence)
        and not isinstance(inputs, (str, bytes))
        and all(isinstance(x, str) for x in inputs)
    ):
        return list(inputs)
    raise UnsupportedCapability(f"a Jev provider reads a sequence of texts, got {type(inputs).__name__}")


def _response_metadata(response: Any, provider: str, requested: Optional[str]) -> dict[str, Any]:
    meta: dict[str, Any] = {"provider": provider}
    model = getattr(response, "model", None)
    if isinstance(model, str):
        meta["model"] = model
    if requested is not None:
        meta["requested_model"] = requested
    try:
        request_id = response.request_id
    except Exception:  # noqa: BLE001 - a response built without HTTP metadata has no request id
        request_id = None
    if isinstance(request_id, str):
        meta["request_id"] = request_id
    usage = getattr(response, "usage", None)
    counts = {
        name: int(getattr(usage, name))
        for name in ("input_tokens", "output_tokens")
        if isinstance(getattr(usage, name, None), int) and not isinstance(getattr(usage, name), bool)
    }
    if counts:
        meta["usage"] = counts
    return meta


def _result(question: Question, answer: Any, meta: dict[str, Any], provider: str) -> DecisionResult:
    """One validated result from a Jev answer (a mismatch raises
    :class:`InvalidDecisionResponse`)."""
    kind = getattr(answer, "type", None)
    if isinstance(question, Boolean):
        if kind != "noul":
            raise InvalidDecisionResponse(f"expected a noul answer, got {kind!r}")
        return validate_response(question, getattr(answer, "noul", None), provider=provider, raw=dict(meta))
    expected = "choice" if isinstance(question, Choice) else "score"
    if kind != expected:
        raise InvalidDecisionResponse(f"expected a {expected} answer, got {kind!r}")
    confidence = getattr(answer, "confidence", None)
    raw = dict(meta)
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        raw["provider_confidence"] = float(confidence)
    probabilities = getattr(answer, "probabilities", None)
    if not isinstance(probabilities, Mapping):
        raise InvalidDecisionResponse(f"the {kind!r} answer carries no probabilities")
    if isinstance(question, Choice):
        by_description = {option.description: option.id for option in question.options}
        unknown = [label for label in probabilities if label not in by_description]
        if unknown:
            raise InvalidDecisionResponse(f"the choice answer names labels the question did not ask: {unknown}")
        keyed = [(by_description[label], p) for label, p in probabilities.items()]
        return validate_response(question, keyed, provider=provider, raw=raw)
    ids = question.option_ids
    keyed = []
    for level, p in probabilities.items():
        if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level < len(ids):
            raise InvalidDecisionResponse(f"the score answer names level {level!r}; the question has {len(ids)}")
        keyed.append((ids[level], p))
    return validate_response(question, keyed, provider=provider, raw=raw, vendor_score=getattr(answer, "score", None))


# --- the provider -----------------------------------------------------------------------------


class JevProvider:
    """A :class:`~nnx.decisions.DecisionProvider` for Jev models through the
    TypeSafe SDK. See the module docstring for translation, confidence,
    metadata, retry and lifecycle semantics.

    Args:
        client: a ``typesafe_sdk.TypeSafeClient`` (or anything with its
            ``system_one``) the caller owns; never closed by the provider.
        async_client: an ``AsyncTypeSafeClient`` the caller owns, for
            :meth:`adecide` / :meth:`adecide_many`. Injecting either means
            the provider builds no client at all.
        model: the model to request (pin a version such as
            ``"jev-1.13.0"``; ``None`` uses the client's default).
        api_key / base_url / timeout / retry: settings for the clients the
            provider builds itself (``api_key`` defaults to the
            ``TYPESAFE_API_KEY`` environment variable). Refused with an
            injected client, whose settings are the caller's.
        max_batch: the most inputs per call (``None``: unbounded) — each
            input is one request.
        name: the provider name recorded on results.

    Raises:
        ImportError: no client is injected and the ``jev`` extra is missing.
        JevError: a client the provider builds cannot be configured (for
            example no API key), at the first call.
    """

    max_questions: Optional[int] = None  # every question of one call shares a system_one request

    def __init__(
        self,
        client: Any = None,
        *,
        async_client: Any = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        retry: Any = None,
        max_batch: Optional[int] = None,
        name: str = "jev",
    ) -> None:
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise InvalidDecisionRequest(f"model must be a non-empty string or None, got {model!r}")
        if not isinstance(name, str) or not name.strip():
            raise InvalidDecisionRequest(f"name must be a non-empty string, got {name!r}")
        settings = {"api_key": api_key, "base_url": base_url, "timeout": timeout, "retry": retry}
        given = sorted(key for key, value in settings.items() if value is not None)
        if given and (client is not None or async_client is not None):
            raise InvalidDecisionRequest(
                f"{given} configure the clients the provider builds; an injected client brings its own settings"
            )
        # The provider builds clients only when the caller injects none: an
        # injected client is never paired with one built from the environment.
        self._builds = client is None and async_client is None
        if self._builds:
            _sdk()  # the provider will build its clients: the extra must be installed
        self._client = client
        self._async_client = async_client
        self._async_loop: Any = None  # the event loop a built async client belongs to
        if not self._builds and async_client is None:
            # Sync-only: no async methods, so a decision job runs this provider
            # in its own worker thread, one call at a time, and waits for it.
            self.__dict__["adecide"] = None
            self.__dict__["adecide_many"] = None
        self._settings = {key: value for key, value in settings.items() if value is not None}
        self.model = model
        self.name = name
        self._capabilities = Capabilities(
            primitives=frozenset({"choice", "boolean", "score"}),
            modalities=frozenset({"text"}),
            dynamic_labels=True,
            max_batch=max_batch,
        )

    def __repr__(self) -> str:  # never shows credentials
        return f"JevProvider(model={self.model!r}, name={self.name!r})"

    # -- declared capabilities ---------------------------------------------------------------

    def capabilities(self) -> Capabilities:
        return self._capabilities

    def record(self) -> dict[str, Any]:
        """The provider's identity for benchmarks and replay: no client,
        header or credential."""
        return {"provider": self.name, "backend": "typesafe-sdk", "sdk_range": SDK_RANGE, "model": self.model}

    def check(self, question: Question, inputs: Any) -> None:
        """Every check made before a request: the question, the inputs and
        the provider's capabilities — no network I/O."""
        if not isinstance(question, (Choice, Boolean, Score)):
            raise InvalidDecisionRequest(f"not a decision question: {type(question).__name__}")
        texts = _texts(inputs)
        self._capabilities.check(question, modality="text", batch_size=len(texts))
        if isinstance(question, Choice):
            descriptions = [option.description for option in question.options]
            repeated = sorted({d for d in descriptions if descriptions.count(d) > 1})
            if repeated:
                raise UnsupportedCapability(f"a Jev choice is keyed by option description; repeated: {repeated}")

    # -- clients -----------------------------------------------------------------------------

    def _build(self, factory: str) -> Any:
        try:
            return getattr(_sdk(), factory)(**self._settings)
        except Exception as error:  # a missing / invalid key or setting: typed like a request failure
            mapped = _translate(error)
            if mapped is None:
                raise
            raise mapped from error

    def _sync_client(self) -> Any:
        if self._client is None:
            if not self._builds:
                raise InvalidDecisionRequest(
                    "only an async client was injected: use adecide / adecide_many, or inject a sync client too"
                )
            self._client = self._build("TypeSafeClient")
        return self._client

    def _aio_client(self) -> Any:
        if not self._builds:
            return self._async_client
        loop = asyncio.get_running_loop()
        if self._async_client is not None and self._async_loop is not loop:
            # A built async client belongs to the loop it was built on; that
            # loop is gone, so it cannot be awaited closed from this one.
            self._drop_async_client()
        if self._async_client is None:
            self._async_client = self._build("AsyncTypeSafeClient")
            self._async_loop = loop
        return self._async_client

    def _drop_async_client(self) -> None:
        self._async_client = None
        self._async_loop = None
        warnings.warn(
            "JevProvider dropped an async client built on another event loop without closing it; "
            "close it with 'async with' (or aclose()) inside the asyncio.run that used it",
            ResourceWarning,
            stacklevel=3,
        )

    def close(self) -> None:
        """Close the sync client the provider built (never an injected one).
        A built async client needs :meth:`aclose` (``async with``)."""
        if self._builds and self._client is not None:
            client, self._client = self._client, None
            client.close()
        if self._builds and self._async_client is not None:
            warnings.warn(
                "JevProvider.close() cannot close the async client it built; use aclose() or 'async with'",
                ResourceWarning,
                stacklevel=2,
            )

    async def aclose(self) -> None:
        """Close every client the provider built, async one included."""
        try:
            if self._builds and self._async_client is not None:
                if self._async_loop is not asyncio.get_running_loop():
                    self._drop_async_client()
                else:
                    client, self._async_client = self._async_client, None
                    await client.aclose()
        finally:
            self.close()

    def __enter__(self) -> JevProvider:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    async def __aenter__(self) -> JevProvider:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    # -- answering ---------------------------------------------------------------------------

    def _request(self, questions: Sequence[Question], inputs: Any) -> tuple[list[str], dict[str, Any]]:
        if not questions:
            raise InvalidDecisionRequest("decide_many needs at least one question")
        for question in questions:
            self.check(question, inputs)
        return _texts(inputs), {f"q{i}": _wire(question) for i, question in enumerate(questions)}

    def _call_kwargs(self) -> dict[str, Any]:
        return {} if self.model is None else {"model": self.model}

    def _results(self, questions: Sequence[Question], responses: list[Any]) -> list[list[DecisionResult]]:
        answered: list[list[DecisionResult]] = [[] for _ in questions]
        for response in responses:
            meta = _response_metadata(response, self.name, self.model)
            answers = getattr(response, "answers", None)
            for index, question in enumerate(questions):
                answer = answers.get(f"q{index}") if isinstance(answers, Mapping) else None
                try:
                    if answer is None:
                        raise InvalidDecisionResponse(f"the response holds no answer for question q{index}")
                    answered[index].append(_result(question, answer, meta, self.name))
                except InvalidDecisionResponse as error:
                    raise JevMalformedResponse(
                        f"{type(error).__name__}: {error}", request_id=meta.get("request_id")
                    ) from error
        return answered

    def decide(self, question: Question, inputs: Any) -> list[DecisionResult]:
        """Answer ``question`` for every text in ``inputs``."""
        return self.decide_many([question], inputs)[0]

    def decide_many(self, questions: Sequence[Question], inputs: Any) -> list[list[DecisionResult]]:
        """Answer every question for every text: one ``system_one`` call per
        text holding all the questions. One result list per question, one
        result per text."""
        texts, wire = self._request(questions, inputs)
        client = self._sync_client()
        responses = []  # one request per text, made in order; a failure stops the rest
        for text in texts:
            try:
                responses.append(client.system_one(text, wire, **self._call_kwargs()))
            except Exception as error:
                mapped = _translate(error)
                if mapped is None:
                    raise
                raise mapped from error
        return self._results(questions, responses)

    async def adecide(self, question: Question, inputs: Any) -> list[DecisionResult]:
        """:meth:`decide` through the async client."""
        return (await self.adecide_many([question], inputs))[0]

    async def adecide_many(self, questions: Sequence[Question], inputs: Any) -> list[list[DecisionResult]]:
        """:meth:`decide_many` through the async client (absent — ``None`` —
        when only a sync client was injected)."""
        texts, wire = self._request(questions, inputs)
        client = self._aio_client()
        responses = []
        for text in texts:
            try:
                responses.append(await client.system_one(text, wire, **self._call_kwargs()))
            except Exception as error:
                mapped = _translate(error)
                if mapped is None:
                    raise
                raise mapped from error
        return self._results(questions, responses)
