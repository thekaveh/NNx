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
log-ratio margin between chosen and rejected. The reference's
parameters are frozen on factory call (``requires_grad=False``) and
its net is pinned to eval mode; the policy is the only module whose
weights move.

Scope: this is the bare DPO objective for the small-LM experimentation
NNx is sized for. It is **not** a production RLHF replacement —
production-scale preference tuning typically needs reference-model
sharing, KL-shaped warmup, IPO/cDPO variants, or PEFT integration
that's out of scope here. See ``docs/dpo.md`` for the honest tradeoffs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .._step_helpers import finalize_step
from ..nn.nn_model import NNModel, TrainStepContext, TrainStepFn
from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint


def dpo_train_step_factory(
    ref_model: NNModel,
    *,
    beta: float = 0.1,
) -> TrainStepFn:
    """Build a Direct Preference Optimization :class:`TrainStepFn`.

    Args:
        ref_model: a frozen reference policy — typically a copy of the
            SFT checkpoint that the trainable policy was initialized
            from. Its ``net`` is set to eval mode and every parameter
            has ``requires_grad`` cleared on factory call. Must share
            ``vocab_size`` and tokenization with the policy.
        beta: temperature on the implicit reward. Larger ``beta`` makes
            the loss sharper (closer to a hard preference); smaller
            ``beta`` keeps the policy closer to the reference. The
            original DPO paper uses 0.1 as the default; values in
            ``[0.01, 0.5]`` are common. Must be > 0.

    Returns:
        A ``TrainStepFn`` for ``NNModel.train(..., train_step_fn=...)``.
        The training loader MUST yield batches of three
        ``torch.LongTensor`` of shape ``(B, T_*)``::

            (prompt_ids, chosen_ids, rejected_ids)

        — typically from :class:`nnx.NNPreferenceDataset`. All three
        tensors must already be padded / right-aligned by the dataset.

    Raises:
        ValueError: if ``beta`` ≤ 0.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")

    # Freeze the reference and pin to eval mode. The policy's training
    # never touches the reference; this just guards against accidental
    # gradient flow if the caller wires them into a shared module later.
    ref_model.net.eval()
    for p in ref_model.net.parameters():
        p.requires_grad = False

    def step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        m = ctx.model
        m.net.train()
        m.net.zero_grad()

        prompt_ids, chosen_ids, rejected_ids = _unpack_preference_batch(ctx.batch)
        prompt_ids = prompt_ids.to(m.device)
        chosen_ids = chosen_ids.to(m.device)
        rejected_ids = rejected_ids.to(m.device)

        # Build the (prompt + response) sequences. We compute the
        # response log-prob conditional on the prompt by summing the
        # per-token log-probs over the response positions only.
        chosen_seq = torch.cat([prompt_ids, chosen_ids], dim=1)
        rejected_seq = torch.cat([prompt_ids, rejected_ids], dim=1)
        prompt_len = prompt_ids.shape[1]

        # Policy log-probs (with gradient).
        policy_chosen_logp = _response_logprob(m.net, chosen_seq, prompt_len)
        policy_rejected_logp = _response_logprob(m.net, rejected_seq, prompt_len)

        # Reference log-probs — no gradient, but move tensors through
        # ref_model's device frame in case it lives elsewhere.
        with torch.no_grad():
            ref_chosen_logp = _response_logprob(
                ref_model.net,
                chosen_seq.to(ref_model.device),
                prompt_len,
            ).to(m.device)
            ref_rejected_logp = _response_logprob(
                ref_model.net,
                rejected_seq.to(ref_model.device),
                prompt_len,
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
        return NNEvaluationDataPoint(
            f1=0.0,
            recall=0.0,
            accuracy=0.0,
            precision=0.0,
            loss=loss_val,
            error=-gap,
        )

    return step


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


def _response_logprob(
    net: torch.nn.Module,
    full_ids: torch.Tensor,
    prompt_len: int,
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
    # Sum across the response positions for a per-row total log-prob.
    return token_logp.sum(dim=-1)
