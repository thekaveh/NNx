"""Knowledge distillation — train a small student to mimic a frozen
teacher's softened predictions.

Two factories:

  - :func:`kd_train_step_factory` — classic Hinton/Vinyals/Dean
    formulation: the student's loss is a weighted mix of the standard
    hard-label loss and a temperature-softened KL divergence between
    student and teacher logits::

        L = α · KL(softmax(t/T) || softmax(s/T)) · T² + (1 − α) · L_hard

  - :func:`feature_kd_train_step_factory` — FitNets-style intermediate
    feature distillation. Adds an MSE term between named teacher /
    student intermediate-layer activations, weighted by ``beta``::

        L = α · KL_soft · T² + β · MSE(student_act, teacher_act) + (1 − α) · L_hard

Both factories return a :class:`nnx.TrainStepFn` that plugs straight
into :meth:`NNModel.train` via the ``train_step_fn=`` hook. The
teacher's parameters are frozen on factory call so subsequent training
can never drift the teacher's weights; restored on tear-down would be
tidy but unnecessary in practice — once a teacher is a teacher, it
stays one.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as F

from .._metrics import classification_edp
from .._step_helpers import finalize_step, softened_kl
from ..nn.nn_model import NNModel, TrainStepContext, TrainStepFn
from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint


def kd_train_step_factory(
    teacher: NNModel,
    *,
    alpha: float = 0.5,
    temperature: float = 4.0,
) -> TrainStepFn:
    """Build a knowledge-distillation :class:`TrainStepFn`.

    Args:
        teacher: a fully-trained :class:`NNModel` whose net produces
            logits of the same shape as the student's. The teacher's
            parameters are frozen (``requires_grad=False``) and its
            net is set to eval mode on factory call.
        alpha: weight on the distillation (soft) loss. The hard-label
            loss gets ``1 − α``. ``α=1.0`` is pure distillation;
            ``α=0.0`` collapses to standard supervised training (the
            teacher is loaded but unused). 0.5 is the common default.
        temperature: softmax temperature applied to BOTH student and
            teacher logits before the KL. Higher T flattens the
            distribution and exposes more dark knowledge; the
            ``× T²`` factor in front of the KL keeps gradient
            magnitude comparable to the hard-label term across T.
            4.0 is the classical Hinton choice.

    Returns:
        A ``TrainStepFn`` suitable for ``NNModel.train(..., train_step_fn=...)``.

    Raises:
        ValueError: if ``alpha`` is not in [0, 1], or ``temperature`` ≤ 0.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    # Freeze the teacher and pin to eval mode. The student's training
    # never touches the teacher; this just guards against accidental
    # gradient flow if the caller wires them into a shared module later.
    teacher.net.eval()
    for p in teacher.net.parameters():
        p.requires_grad = False

    def step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        m = ctx.model
        m.net.train()
        m.net.zero_grad()

        # unpack_batch returns ((X,), Y); the singleton destructure asserts
        # a single-input net and binds X to that one tensor.
        (X,), Y = cast(Any, m.net).unpack_batch(ctx.batch)
        X = X.to(m.device)
        Y = Y.to(m.device)

        student_logits = m.net(X)
        with torch.no_grad():
            # Teacher might live on a different device; migrate the
            # input into its frame for the forward pass. Cheap when
            # student.device == teacher.device.
            teacher_logits = teacher.net(X.to(teacher.device)).to(m.device)

        # KL(teacher || student) with both softened by T — the standard
        # Hinton direction (see softened_kl for the F.kl_div contract).
        soft_loss = softened_kl(student_logits, teacher_logits, temperature)
        hard_loss = m.loss_fn(student_logits, Y)
        loss = alpha * soft_loss + (1.0 - alpha) * hard_loss
        loss_val = finalize_step(loss, ctx, paradigm="distillation")

        return classification_edp(
            Y=Y,
            Y_hat=student_logits.argmax(dim=-1),
            loss=loss_val,
            extra_metrics=ctx.extra_metrics,
        )

    return step


def feature_kd_train_step_factory(
    teacher: NNModel,
    *,
    auxiliary_layers: dict[str, str],
    alpha: float = 0.5,
    beta: float = 0.5,
    temperature: float = 4.0,
) -> TrainStepFn:
    """Build a FitNets-style feature-distillation :class:`TrainStepFn`.

    Extends :func:`kd_train_step_factory` with an additional MSE term
    matching named intermediate-layer activations between the (frozen)
    teacher and the trainable student. Forward hooks register on the
    pairs in ``auxiliary_layers``; collected activations feed an
    elementwise MSE that's mixed into the loss via ``beta``::

        L = α · KL_soft · T² + β · MSE(student_act, teacher_act) + (1 − α) · L_hard

    Args:
        teacher: a fully-trained :class:`NNModel` whose net produces
            logits of the same shape as the student's. The teacher's
            parameters are frozen (``requires_grad=False``) and its
            net is set to eval mode on factory call — same guarantee
            as :func:`kd_train_step_factory`.
        auxiliary_layers: dict mapping ``teacher_layer_name ->
            student_layer_name`` for each (teacher, student) pair to
            match. Names are dotted paths resolved via
            :meth:`torch.nn.Module.get_submodule` against the teacher
            / student ``net``. Must be non-empty. The teacher and
            student activations at each pair must share shape — if
            they don't, the factory raises ``ValueError`` on the
            first forward (the projector ``FeatureRegressor`` from
            FitNets is intentionally deferred).
        alpha: weight on the soft (logit-KL) term. The hard-label
            loss gets ``1 − α``. 0.5 is the common default.
        beta: weight on the feature-MSE term. 0.5 is the common
            starting point; tune downward if it dominates the logit
            term, upward to bias the student toward matching internal
            representations.
        temperature: softmax temperature for the logit-KL term —
            identical contract to :func:`kd_train_step_factory`.

    Returns:
        A ``TrainStepFn`` suitable for ``NNModel.train(...,
        train_step_fn=...)``.

    Raises:
        ValueError: if ``alpha`` or ``beta`` is not in [0, 1], if
            ``temperature`` ≤ 0, or if ``auxiliary_layers`` is empty.
            On the first batch, if any paired teacher/student
            activation shapes disagree.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if not auxiliary_layers:
        raise ValueError(
            "auxiliary_layers must be a non-empty mapping of teacher_layer_name -> student_layer_name; got empty dict"
        )

    # Freeze the teacher and pin to eval mode — same guarantee as
    # kd_train_step_factory. Student training never touches the
    # teacher; this just guards against accidental gradient flow.
    teacher.net.eval()
    for p in teacher.net.parameters():
        p.requires_grad = False

    # Resolve named submodules eagerly so a typo in the user's mapping
    # raises a clear error on factory call (not deep inside the first
    # forward where the AttributeError would be cryptic).
    teacher_layers: dict[str, torch.nn.Module] = {}
    for t_name in auxiliary_layers:
        try:
            teacher_layers[t_name] = teacher.net.get_submodule(t_name)
        except AttributeError as e:
            raise ValueError(
                f"auxiliary_layers: teacher has no submodule named {t_name!r} (get_submodule raised: {e})"
            ) from e

    def step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        m = ctx.model
        m.net.train()
        m.net.zero_grad()

        (X,), Y = cast(Any, m.net).unpack_batch(ctx.batch)
        X = X.to(m.device)
        Y = Y.to(m.device)

        # Resolve student layers per-step (cheap, and keeps us robust
        # if the user swapped m.net between factory build and first
        # step). Same typo-surfacing as the teacher resolution above.
        student_acts: dict[str, torch.Tensor] = {}
        teacher_acts: dict[str, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []
        try:
            for t_name, s_name in auxiliary_layers.items():
                try:
                    s_layer = m.net.get_submodule(s_name)
                except AttributeError as e:
                    raise ValueError(
                        f"auxiliary_layers: student has no submodule named {s_name!r} (get_submodule raised: {e})"
                    ) from e

                # Capture by default-arg to bind the loop's current names
                # rather than the closure's late-binding.
                def _t_hook(_module, _inputs, output, _key=t_name):
                    teacher_acts[_key] = output

                def _s_hook(_module, _inputs, output, _key=t_name):
                    student_acts[_key] = output

                handles.append(teacher_layers[t_name].register_forward_hook(_t_hook))
                handles.append(s_layer.register_forward_hook(_s_hook))

            student_logits = m.net(X)
            with torch.no_grad():
                # Teacher might live on a different device; migrate the
                # input into its frame for the forward pass.
                teacher_logits = teacher.net(X.to(teacher.device)).to(m.device)
        finally:
            for h in handles:
                h.remove()

        # Verify every paired activation has matching shape. The
        # FeatureRegressor (a small projection) is deferred — for now
        # we surface a clear error so the user doesn't get cryptic
        # broadcast results from F.mse_loss.
        feature_loss = torch.zeros((), device=m.device)
        for t_name in auxiliary_layers:
            t_act = teacher_acts[t_name].to(m.device)
            s_act = student_acts[t_name]
            if t_act.shape != s_act.shape:
                raise ValueError(
                    f"auxiliary_layers: shape mismatch at pair "
                    f"{t_name!r} -> {auxiliary_layers[t_name]!r} — "
                    f"teacher activation has shape {tuple(t_act.shape)} "
                    f"but student activation has shape {tuple(s_act.shape)}. "
                    "The v1 factory requires shape-matched paired layers; "
                    "the FeatureRegressor projector is deferred."
                )
            feature_loss = feature_loss + F.mse_loss(s_act, t_act)
        # Average across paired layers so beta's scale is invariant
        # to how many pairs the user provided.
        feature_loss = feature_loss / len(auxiliary_layers)

        soft_loss = softened_kl(student_logits, teacher_logits, temperature)
        hard_loss = m.loss_fn(student_logits, Y)
        loss = alpha * soft_loss + beta * feature_loss + (1.0 - alpha) * hard_loss
        loss_val = finalize_step(loss, ctx, paradigm="feature_distillation")

        return classification_edp(
            Y=Y,
            Y_hat=student_logits.argmax(dim=-1),
            loss=loss_val,
            extra_metrics=ctx.extra_metrics,
        )

    return step
