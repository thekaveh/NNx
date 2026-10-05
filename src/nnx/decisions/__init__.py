"""Provider-neutral typed decisions (FEAT-009).

Ask a typed question, get labelled probabilities back — from any provider
that declares it can answer:

- **Questions** — :class:`Choice` (pick one of 2+ :class:`Option`\\ s),
  :class:`Boolean` (yes / no) and :class:`Score` (2+ ordered levels). An
  option's ``id`` is bookkeeping; its ``description`` is the model-facing
  text (:meth:`Question.model_view` never shows ids). ``digest()`` hashes
  the kind, prompt and every option in order, so a result can never be
  matched to a reordered or reworded question.
- **Results** — :class:`ChoiceResult` / :class:`ScoreResult` (a
  distribution in the question's order, summing to 1 within ``1e-6``) and
  :class:`BooleanResult` (one ``p_true``). A provider's own output stays in
  ``raw``, a Score provider's own number in ``vendor_score``;
  :attr:`ScoreResult.expected_index` (``sum(i * p_i)``) is ordinal, not an
  interval score.
- **Validation** — :func:`validate_response` reorders a provider's keyed
  output into the question's order and rejects missing, duplicate, unknown
  or unlabeled ids and malformed distributions, which it never
  renormalizes.
- **Providers** — the :class:`DecisionProvider` protocol: declared
  :class:`Capabilities` (primitives, modalities, dynamic labels, batch
  limits, inference / training / export) checked before any model call or
  network I/O, and ``decide(question, inputs)``.
  :class:`FixedHeadProvider` adapts a trained NNx classifier;
  :class:`NLIProvider` (FEAT-011) is a local label-conditioned baseline that
  scores candidates supplied at inference with a caller-supplied NLI model;
  :class:`JevProvider` (FEAT-010, the ``jev`` extra) answers on Jev models
  through the TypeSafe SDK, mapping its failures to typed
  :class:`JevError`\\ s.
- **Jobs** (FEAT-024) — :class:`DecisionJob`: an immutable, deferred
  description of decision work (``ask`` / ``collect`` / ``map`` / ``then``)
  whose ``run`` / ``arun`` batch independent questions into the fewest
  provider calls and chain dependent ones, fail-fast under explicit
  :class:`Limits`.
- **Errors** — :class:`UnsupportedCapability`,
  :class:`InvalidDecisionRequest`, :class:`InvalidDecisionResponse` and
  :class:`ProviderFailure`, all :class:`DecisionError`\\ s.

Importing this package starts no provider or model backend and needs no
hosted-SDK extra; inference only — nothing here trains, exports or executes
actions.
"""

from .jev import (
    JevAuthenticationError,
    JevError,
    JevMalformedResponse,
    JevProvider,
    JevRateLimited,
    JevTimeout,
)
from .jobs import (
    DecisionJob,
    Follow,
    InvalidJob,
    JobError,
    JobFailed,
    JobLimitExceeded,
    JobResult,
    JobTimeout,
    Limits,
    QuestionOutcome,
    RowOutcome,
)
from .nli import NLIProvider
from .providers import Capabilities, DecisionProvider, FixedHeadProvider
from .schema import (
    PROBABILITY_TOLERANCE,
    Boolean,
    BooleanResult,
    Choice,
    ChoiceResult,
    DecisionError,
    DecisionResult,
    InvalidDecisionRequest,
    InvalidDecisionResponse,
    Option,
    ProviderFailure,
    Question,
    Score,
    ScoreResult,
    UnsupportedCapability,
    question_from_state,
    validate_response,
)

__all__ = [
    "PROBABILITY_TOLERANCE",
    "Boolean",
    "BooleanResult",
    "Capabilities",
    "Choice",
    "ChoiceResult",
    "DecisionError",
    "DecisionJob",
    "DecisionProvider",
    "DecisionResult",
    "FixedHeadProvider",
    "Follow",
    "InvalidDecisionRequest",
    "InvalidDecisionResponse",
    "InvalidJob",
    "JevAuthenticationError",
    "JevError",
    "JevMalformedResponse",
    "JevProvider",
    "JevRateLimited",
    "JevTimeout",
    "JobError",
    "JobFailed",
    "JobLimitExceeded",
    "JobResult",
    "JobTimeout",
    "Limits",
    "NLIProvider",
    "Option",
    "ProviderFailure",
    "Question",
    "QuestionOutcome",
    "RowOutcome",
    "Score",
    "ScoreResult",
    "UnsupportedCapability",
    "question_from_state",
    "validate_response",
]
