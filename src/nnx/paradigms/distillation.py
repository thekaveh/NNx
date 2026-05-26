"""Knowledge distillation — train a small student to mimic a frozen
teacher's softened predictions.

Classic Hinton/Vinyals/Dean formulation: the student's loss is a
weighted mix of the standard hard-label loss and a temperature-softened
KL divergence between student and teacher logits.

    L = α · KL(softmax(s/T) || softmax(t/T)) · T² + (1 − α) · L_hard

The factory returns a :class:`nnx.TrainStepFn` that plugs straight into
:meth:`NNModel.train` via the ``train_step_fn=`` hook. The teacher's
parameters are frozen on factory call so subsequent training can never
drift the teacher's weights; restored on tear-down would be tidy but
unnecessary in practice — once a teacher is a teacher, it stays one.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

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

        (X,), Y = m.net.unpack_batch(ctx.batch)
        X = tuple(x.to(m.device) for x in X) if isinstance(X, tuple) else (X.to(m.device),)
        Y = Y.to(m.device)

        student_logits = m.net(*X)
        with torch.no_grad():
            # Teacher might live on a different device; migrate the
            # inputs into its frame for the forward pass. Cheap when
            # student.device == teacher.device.
            t_inputs = tuple(x.to(teacher.device) for x in X)
            teacher_logits = teacher.net(*t_inputs).to(m.device)

        # KL(student || teacher) with both softened by T. F.kl_div
        # expects the FIRST argument in log-space and the second in
        # plain probability space; we obey that convention.
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)
        hard_loss = m.loss_fn(student_logits, Y)
        loss = alpha * soft_loss + (1.0 - alpha) * hard_loss

        loss.backward()
        ctx.optimizer.step()

        loss_val = float(loss.detach())
        if not np.isfinite(loss_val):
            raise FloatingPointError(
                f"non-finite distillation loss ({loss_val!r}) — training diverged. "
                "Check learning rate, alpha, temperature, or input normalization."
            )

        Y_hat = student_logits.argmax(dim=-1)
        return (
            NNEvaluationDataPoint.of(
                Y=Y.cpu().numpy(), Y_hat=Y_hat.cpu().numpy(),
                extra_metrics=ctx.extra_metrics,
            )
            .with_loss(value=loss_val)
            .with_error(value=float(1 - (Y_hat == Y).sum().item() / Y.size(0)))
        )

    return step
