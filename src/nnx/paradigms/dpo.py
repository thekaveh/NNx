"""Direct Preference Optimization (DPO) — preference-tuning an LM.

The factory :func:`dpo_train_step_factory` returns a
:class:`nnx.TrainStepFn` that implements the original DPO objective
(Rafailov et al., 2023). Given a frozen *reference* policy
``π_ref`` and a trainable *policy* ``π_θ`` (the
:class:`GenerativeNNModel` being optimized), each batch supplies a
prompt ``x`` plus a chosen response ``y_w`` and a rejected response
``y_l``. The step computes

::

    Δ_w  = log π_θ(y_w | x) − log π_ref(y_w | x)
    Δ_l  = log π_θ(y_l | x) − log π_ref(y_l | x)
    L    = −log σ(β · (Δ_w − Δ_l))

— i.e., minimize the negative log-sigmoid of the policy-vs-reference
log-ratio margin between chosen and rejected. The reference must be
kept apart from the policy's net; each step checks that, then freezes
the reference's parameters (``requires_grad=False``) and pins its net
to eval mode before every reference forward; the policy is the only
module whose weights move.

Scope: this is the bare DPO objective for the small-LM experimentation
NNx is sized for. It is **not** a production RLHF replacement —
production-scale preference tuning typically needs reference-model
sharing, KL-shaped warmup, IPO/cDPO variants, or PEFT integration
that's out of scope here. See ``docs/dpo.md`` for the honest tradeoffs.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .._step_helpers import finalize_step, freeze_apart, imperative_step, pinned_eval
from .._validation import require_count, require_finite_real, shown
from ..nn.nn_model import NNModel, TrainStepContext, TrainStepFn
from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from ..precision import full_precision_only

# The remedy every refusal of a reference in (or sharing) the policy's net names.
_REFERENCE_REMEDY = "pass a reference kept apart from the policy (copy.deepcopy(policy) before training)"


def dpo_train_step_factory(
    ref_model: NNModel,
    *,
    beta: float = 0.1,
    pad_token_id: Optional[int] = None,
    prompt_packing: bool = False,
) -> TrainStepFn:
    """Build a Direct Preference Optimization :class:`TrainStepFn`.

    Args:
        ref_model: a frozen reference policy — typically a copy of the
            SFT checkpoint that the trainable policy was initialized
            from, kept apart from the policy: not the policy's net, not
            inside it, sharing none of its parameters. Building the step
            leaves it alone; each step checks that placement first, then
            clears ``requires_grad`` on every parameter and puts the net
            in eval mode before its forward (one a callback switches to
            train mode would otherwise run its dropout in train mode).
            Must share ``vocab_size`` and tokenization with the policy.
        beta: strength of the reference-policy constraint (the scale of
            the implicit reward). Larger ``beta`` keeps the policy closer
            to the reference; smaller ``beta`` lets it deviate more. The
            original DPO paper uses 0.1 as the default; values in
            ``[0.01, 0.5]`` are common. Must be > 0.
        pad_token_id: the id the dataset used to right-pad chosen /
            rejected responses (``NNPreferenceDataset.pad_token_id``).
            When set, padded positions are excluded from the response
            log-prob sums. Without it, pad tokens are scored too — the
            pad terms don't cancel between policy/reference or
            chosen/rejected (different contexts), biasing the objective
            and training the policy to emit pads after short responses.
            ``None`` is only appropriate when every response genuinely
            fills ``max_response_len``. Two caveats: masking is by
            token-id equality, so a genuine occurrence of the pad id
            inside a response is dropped too (pick a dedicated pad id);
            and prompt-side padding remains visible to the model (no
            attention mask) — a pre-existing modeling bias this knob
            doesn't address unless ``prompt_packing`` is on.
        prompt_packing: opt-in fix for the prompt-side padding above. The
            default ``False`` keeps the prompt's right-padding in front of
            the response, so the model conditions on pad tokens. ``True``
            packs each row as its valid prompt tokens, then its valid
            response tokens, then padding at the end only; the response
            log-probs are read at the positions the packed row places it
            (a causal model's logits there never see the trailing padding),
            so they match the same row built without padding. Packing drops
            every pad id from the prompt and the response, including one
            inside them, as the pad mask already does for responses. A
            prompt with no valid token is refused at the first step, since
            its response would have no prompt position to be predicted from.

    Returns:
        A ``TrainStepFn`` for ``NNModel.train(..., train_step_fn=...)``.
        The training loader MUST yield batches of three
        ``torch.LongTensor`` of shape ``(B, T_*)``::

            (prompt_ids, chosen_ids, rejected_ids)

        — typically from :class:`nnx.NNPreferenceDataset`. All three
        tensors must be right-padded (content left-aligned) to fixed
        lengths, as ``NNPreferenceDataset`` yields them.

    Raises:
        ValueError: if ``beta`` is not a finite positive number, if
            ``pad_token_id`` is not a non-negative integer, or if
            ``prompt_packing`` is not a ``bool``. At the first
            step, before anything is frozen or run: a reference that is
            the policy's net, lies inside it or shares a parameter with it
            (freezing it would freeze the policy).
        TypeError: if ``ref_model`` has no ``net`` module.
    """
    # A bool, NaN, ±inf or an integer too large for a float (10**400) is refused by name.
    beta = require_finite_real(
        beta,
        "beta",
        owner="dpo_train_step_factory",
        minimum=0,
        exclusive_min=True,
        domain_message=f"beta must be a finite positive number, got {shown(beta, real=True)}",
    )
    if pad_token_id is not None:  # a bool or a negative id would mask nothing
        pad_token_id = require_count(pad_token_id, "pad_token_id", owner="dpo_train_step_factory", minimum=0)
    if not isinstance(prompt_packing, bool):  # 1 or "yes" would silently select a layout
        raise ValueError(
            "dpo_train_step_factory requires prompt_packing to be a bool, "
            f"got {shown(prompt_packing)} of type {type(prompt_packing).__name__}"
        )

    if not isinstance(getattr(ref_model, "net", None), torch.nn.Module):
        raise TypeError(
            f"dpo_train_step_factory needs an NNModel reference (a model with a net), got {type(ref_model).__name__}"
        )
    # The reference is left alone here: until a step sees the policy,
    # freezing it could freeze (part of) the policy.

    def step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        m = ctx.model
        # Before any mode change, freeze or forward: a response with no valid
        # token would enter the batch mean as a pair scored 0 with no gradient.
        prompt_ids, chosen_ids, rejected_ids = _unpack_preference_batch(ctx.batch)
        prompt_ids = prompt_ids.to(m.device)
        chosen_ids = chosen_ids.to(m.device)
        rejected_ids = rejected_ids.to(m.device)
        _require_response_tokens(chosen_ids, pad_token_id, "chosen")
        _require_response_tokens(rejected_ids, pad_token_id, "rejected")
        if prompt_packing:
            _require_prompt_tokens(prompt_ids, pad_token_id)
        # Before any mode change or forward: refuse a reference that is, lies
        # in or shares parameters with the policy's net, then freeze it.
        freeze_apart(m.net, ref_model.net, role="DPO reference model", remedy=_REFERENCE_REMEDY)
        m.net.train()
        m.net.zero_grad()

        # Policy log-probs (with gradient). Each is the sum of per-token
        # log-probs over the response positions only; _response_logprobs
        # picks the padded or the packed layout.
        policy_chosen_logp = _response_logprobs(
            m.net, prompt_ids, chosen_ids, pad_token_id=pad_token_id, prompt_packing=prompt_packing
        )
        policy_rejected_logp = _response_logprobs(
            m.net, prompt_ids, rejected_ids, pad_token_id=pad_token_id, prompt_packing=prompt_packing
        )

        # Reference log-probs — no gradient, but move tensors through
        # ref_model's device frame in case it lives elsewhere.
        # m.net.train() or a callback may have switched the reference to train mode; its modes are given back after.
        with pinned_eval(ref_model.net), torch.no_grad():
            ref_chosen_logp = _response_logprobs(
                ref_model.net,
                prompt_ids.to(ref_model.device),
                chosen_ids.to(ref_model.device),
                pad_token_id=pad_token_id,
                prompt_packing=prompt_packing,
            ).to(m.device)
            ref_rejected_logp = _response_logprobs(
                ref_model.net,
                prompt_ids.to(ref_model.device),
                rejected_ids.to(ref_model.device),
                pad_token_id=pad_token_id,
                prompt_packing=prompt_packing,
            ).to(m.device)

        # DPO loss: −log σ(β · ((logπ_w − logπ_l) − (logπref_w − logπref_l))).
        # Equivalent to −log σ(β · (Δ_w − Δ_l)) with Δ_* the
        # policy-minus-reference margins.
        pi_logratios = policy_chosen_logp - policy_rejected_logp
        ref_logratios = ref_chosen_logp - ref_rejected_logp
        loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()

        loss_val = finalize_step(loss, ctx, paradigm="dpo")

        # No classification target — surface the loss in both slots so
        # BEST tracking + ReduceLROnPlateau have a single signal.
        # Track the chosen-minus-rejected log-prob gap as a side metric
        # via `error`: more negative = bigger preference margin learned.
        gap = float((policy_chosen_logp - policy_rejected_logp).detach().mean())
        # Implicit DPO rewards r(x,y) = β·(logπ_θ(y|x) − logπ_ref(y|x)).
        # reward_accuracy is the fraction of the batch where the chosen
        # response's implicit reward exceeds the rejected one — the
        # headline DPO ranking-quality diagnostic (Rafailov et al., 2023).
        # Detach policy logps so the metrics carry no gradient; ref logps
        # were already computed under torch.no_grad().
        reward_chosen = beta * (policy_chosen_logp.detach() - ref_chosen_logp)
        reward_rejected = beta * (policy_rejected_logp.detach() - ref_rejected_logp)
        reward_accuracy = float((reward_chosen > reward_rejected).float().mean())
        return NNEvaluationDataPoint(
            f1=0.0,
            recall=0.0,
            accuracy=0.0,
            precision=0.0,
            loss=loss_val,
            error=-gap,
            extra={
                "reward_chosen": float(reward_chosen.mean()),
                "reward_rejected": float(reward_rejected.mean()),
                "reward_accuracy": reward_accuracy,
            },
        )

    return full_precision_only(imperative_step(step, paradigm="dpo"))  # finalize_step: full precision


def _unpack_preference_batch(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pull (prompt_ids, chosen_ids, rejected_ids) out of a DPO batch.

    Accepts the canonical 3-tuple shape yielded by
    :class:`nnx.NNPreferenceDataset` and any DataLoader-collated form
    of it (still a 3-tuple of stacked tensors).
    """
    if not (isinstance(batch, (list, tuple)) and len(batch) == 3):
        raise ValueError(
            "DPO step expects a batch of (prompt_ids, chosen_ids, rejected_ids); "
            f"got {type(batch).__name__} "
            f"{'of length ' + str(len(batch)) if isinstance(batch, (list, tuple)) else ''}"
        )
    p, c, r = batch
    if not (torch.is_tensor(p) and torch.is_tensor(c) and torch.is_tensor(r)):
        raise ValueError(
            "DPO step batch entries must be torch.Tensors; got "
            f"({type(p).__name__}, {type(c).__name__}, {type(r).__name__})"
        )
    return p, c, r


def _require_response_tokens(response_ids: torch.Tensor, pad_token_id: Optional[int], name: str) -> None:
    """Refuse a batch holding a response with no valid token: every position
    is the pad id (or the response is empty). Its log-prob would be exactly
    0 with no gradient, yet it would still enter the batch mean as a
    preference pair. Without ``pad_token_id`` every position counts."""
    if pad_token_id is None:
        counts = response_ids.new_full((response_ids.shape[0],), response_ids.shape[1])
    else:
        counts = (response_ids != pad_token_id).sum(dim=-1)
    empty = (counts == 0).nonzero(as_tuple=False)
    if empty.numel():
        row = int(empty[0, 0])
        reason = "it is empty" if pad_token_id is None else f"every position is the pad id {pad_token_id}"
        raise ValueError(
            f"DPO {name} response of batch row {row} has no valid tokens ({reason}): its log-prob would be 0 "
            "with no gradient yet still count as a preference pair. Drop that row from the dataset."
        )


def _require_prompt_tokens(prompt_ids: torch.Tensor, pad_token_id: Optional[int]) -> None:
    """Refuse a prompt row with no valid token under ``prompt_packing``. Packed,
    its response would start at position 0 with no earlier logit to predict
    it from, and the first response token would be silently left unscored."""
    if pad_token_id is None:
        counts = prompt_ids.new_full((prompt_ids.shape[0],), prompt_ids.shape[1])
    else:
        counts = (prompt_ids != pad_token_id).sum(dim=-1)
    empty = (counts == 0).nonzero(as_tuple=False)
    if empty.numel():
        row = int(empty[0, 0])
        reason = "it is empty" if pad_token_id is None else f"every position is the pad id {pad_token_id}"
        raise ValueError(
            f"DPO prompt of batch row {row} has no valid tokens ({reason}): with prompt_packing the response "
            "is placed right after the prompt, so it needs a prompt token to be predicted from. "
            "Drop that row from the dataset."
        )


def _response_logprob(
    net: torch.nn.Module,
    full_ids: torch.Tensor,
    prompt_len: int,
    *,
    pad_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Sum of per-token log-probabilities of the response continuation.

    Standard teacher-forcing log-likelihood scoring: feed the full
    ``(prompt + response)`` token sequence through the LM, line up
    logits at position ``t`` against the target token at position
    ``t+1``, then sum the picked log-probs over the response positions
    only (the prompt positions are conditioning, not part of the
    scored continuation).

    Args:
        net: a TransformerNN-shaped LM that maps ``(B, T)`` token ids
            to ``(B, T, vocab)`` logits.
        full_ids: ``(B, T_p + T_r)`` long tensor of concatenated
            prompt + response ids.
        prompt_len: the length of the prompt prefix; everything after
            this position in ``full_ids`` is the response.

    Returns:
        ``(B,)`` tensor of total response log-likelihood per row.
    """
    # Forward gives (B, T_p+T_r, vocab) logits. Logits at position t
    # predict the token at position t+1, so we predict the response by
    # looking at logits[:, prompt_len-1 : -1, :] and matching against
    # targets at full_ids[:, prompt_len:].
    logits = net(full_ids)
    # log-softmax once, then gather the per-token chosen-id log-probs.
    log_probs = F.log_softmax(logits, dim=-1)
    # Slice to align: positions [prompt_len - 1 .. T-2] in `log_probs`
    # predict tokens at positions [prompt_len .. T-1] in `full_ids`.
    response_logits = log_probs[:, prompt_len - 1 : -1, :]
    response_targets = full_ids[:, prompt_len:]
    # gather along vocab dim using the targets as indices.
    token_logp = response_logits.gather(dim=-1, index=response_targets.unsqueeze(-1)).squeeze(-1)
    if pad_token_id is not None:
        # Zero out padded positions (multiplying keeps the graph intact
        # with zero gradient on pads) so right-padded responses aren't
        # scored on their padding.
        token_logp = token_logp * (response_targets != pad_token_id)
    # Sum across the response positions for a per-row total log-prob.
    return token_logp.sum(dim=-1)


def _response_logprobs(
    net: torch.nn.Module,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    *,
    pad_token_id: Optional[int],
    prompt_packing: bool,
) -> torch.Tensor:
    """``(B,)`` response log-probs under the padded layout (the default) or
    the packed one (``prompt_packing=True``).

    Padded: ``[prompt (incl. its pads) | response (incl. its pads)]``, scored
    by :func:`_response_logprob` at ``prompt_len = T_p``. Packed: each row is
    ``[valid prompt | valid response | pad...]``, so the response sits right
    after the prompt and the trailing pads cannot affect it under causal
    attention; :func:`_packed_response_logprob` scores it at its per-row start.
    """
    if not prompt_packing:
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        return _response_logprob(net, full_ids, prompt_ids.shape[1], pad_token_id=pad_token_id)
    packed_ids, prompt_lens = _pack_prompt_first(prompt_ids, response_ids, pad_token_id)
    return _packed_response_logprob(net, packed_ids, prompt_lens, pad_token_id=pad_token_id)


def _pack_prompt_first(
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    pad_token_id: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack each row as its valid prompt tokens, then its valid response tokens,
    then padding at the end only.

    Returns ``(packed_ids, prompt_lens)``: the ``(B, T_p + T_r)`` packed rows and
    the ``(B,)`` count of valid prompt tokens, which is where each row's
    response starts. Without ``pad_token_id`` every position is valid, so the
    rows are returned as concatenated and each prompt keeps its full width.
    """
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    if pad_token_id is None:
        prompt_lens = torch.full((full_ids.shape[0],), prompt_ids.shape[1], dtype=torch.long, device=full_ids.device)
        return full_ids, prompt_lens
    # A stable sort on "is pad" keeps the valid tokens in their order (prompt
    # before response) and moves every pad to the end of the row.
    order = torch.sort((full_ids == pad_token_id).to(torch.uint8), dim=1, stable=True).indices
    packed_ids = full_ids.gather(dim=1, index=order)
    prompt_lens = (prompt_ids != pad_token_id).sum(dim=1)
    return packed_ids, prompt_lens


def _packed_response_logprob(
    net: torch.nn.Module,
    packed_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    *,
    pad_token_id: Optional[int],
) -> torch.Tensor:
    """Sum of per-token log-probs of each packed row's response, ``(B,)``.

    ``packed_ids`` rows are ``[valid prompt | valid response | pad...]`` with
    ``prompt_lens`` valid prompt tokens. The logit at position ``t-1`` predicts
    the token at ``t``, so every position ``t >= 1`` is scored and kept only if
    it is at or past its row's response start and is not a pad.
    """
    logits = net(packed_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    targets = packed_ids[:, 1:]
    token_logp = log_probs[:, :-1, :].gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(1, packed_ids.shape[1], device=packed_ids.device)
    in_response = positions.unsqueeze(0) >= prompt_lens.to(packed_ids.device).unsqueeze(1)
    if pad_token_id is not None:
        in_response = in_response & (targets != pad_token_id)
    return (token_logp * in_response).sum(dim=-1)
