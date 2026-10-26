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

import torch
import torch.nn.functional as F

from .._step_helpers import (
    finalize_step,
    finalize_task_step,
    imperative_step,
    predicted_labels,
    require_row_logits,
    softened_kl,
    task_record,
    task_step,
    task_term,
)
from .._validation import require_finite_real
from ..nn.nn_model import (
    NNModel,
    TrainStepContext,
    TrainStepFn,
    _check_scorable_target,
    _classification_edp_for_loss,
    _loss_input,
    _single_input_batch,
)
from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from ..precision import full_precision_only
from ..tasks import _StepTasks

# The task kinds the KD steps score their hard term for (#513); born_again_train
# refuses any other before generation 0 trains.
_KD_STEP_TASKS = _StepTasks("kd_train_step_factory")


def _check_kd_weights(alpha: float, temperature: float, *, owner: str) -> None:
    """Validate Hinton KD's mixing weight and temperature — shared by
    :func:`kd_train_step_factory`, ``nnx.objectives.KDObjective`` and
    ``born_again_train``. A ``bool`` is refused: ``True`` would pass both
    ranges and train as ``1.0``."""
    require_finite_real(alpha, "alpha", owner=owner, minimum=0, maximum=1)
    require_finite_real(temperature, "temperature", owner=owner, minimum=0, exclusive_min=True)


def _freeze_teacher(teacher: NNModel) -> None:
    """Freeze the teacher and pin it to eval mode. The student's training
    never touches the teacher; this just guards against accidental gradient
    flow if the caller wires them into a shared module later. Stored teacher
    probabilities (FEAT-022) are refused: live distillation softens a
    running teacher's logits."""
    from .offline_distillation import _is_teacher_records

    if _is_teacher_records(teacher) or not isinstance(getattr(teacher, "net", None), torch.nn.Module):
        raise TypeError(
            f"live distillation needs a live NNModel teacher whose logits it softens at a temperature, got "
            f"{type(teacher).__name__}; stored, probability-only teacher records train through "
            "nnx.paradigms.offline_distillation (TeacherDataset.objective())"
        )
    teacher.net.eval()
    for p in teacher.net.parameters():
        p.requires_grad = False


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
            loss gets ``1 − α``. ``α=1.0`` is pure distillation (the
            labels are not scored); ``α=0.0`` collapses to standard
            supervised training (the teacher is loaded but unused), as the
            default step trains it. 0.5 is the common default.
        temperature: softmax temperature applied to BOTH student and
            teacher logits before the KL. Higher T flattens the
            distribution and exposes more dark knowledge; the
            ``× T²`` factor in front of the KL keeps gradient
            magnitude comparable to the hard-label term across T.
            4.0 is the classical Hinton choice.

    A model that declares a ``TaskSpec`` scores the hard term through its
    task adapter, as the default step does: categorical targets equal to
    ``ignore_index`` and NaN multilabel labels are left out of it (the soft
    term keeps every row), each batch is validated before any backward
    pass, and the record carries the task's ``kind`` / ``count`` /
    ``status``. The labels are validated against the task even at
    ``alpha=1``, so a placeholder label must then be the task's
    ``ignore_index`` (NaN for a multilabel task). A batch with nothing to
    learn from (every target masked at ``alpha=0``) takes no update and
    records ``status="empty"``. A regression task is refused before any run
    is reserved.

    Returns:
        A ``TrainStepFn`` suitable for ``NNModel.train(..., train_step_fn=...)``.

    Raises:
        ValueError: if ``alpha`` is not a real in [0, 1], or ``temperature``
            not a finite real > 0 (a ``bool`` is refused). At the first
            step: a single-logit head with ``alpha > 0`` (no class
            distribution to distill), or a teacher of another shape.
            ``NNModel.train`` refuses a model that declares a regression
            task (``TaskValidationError``) before any run is reserved.
        TypeError: if ``teacher`` is not an ``NNModel`` (stored teacher
            records train through ``TeacherDataset.objective()``).
    """
    _check_kd_weights(alpha, temperature, owner="kd_train_step_factory")
    _freeze_teacher(teacher)

    def step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        m = ctx.model
        m.net.train()
        m.net.zero_grad()

        # One positional input, split by the model's batch adapter.
        X, Y = _single_input_batch(m, ctx.batch, who="kd_train_step_factory")

        student_logits = m._net_forward((X,), {})
        require_row_logits(student_logits, "distillation")
        with torch.no_grad():
            # Teacher might live on a different device; migrate the
            # input into its frame for the forward pass. Cheap when
            # student.device == teacher.device.
            teacher_logits = teacher._net_forward((X.to(teacher.device),), {}).to(m.device)

        # KL(teacher || student) with both softened by T — the standard
        # Hinton direction (see softened_kl for the F.kl_div contract).
        # alpha=0 is plain supervised training: no soft term (a binary head has none).
        soft_loss = softened_kl(student_logits, teacher_logits, temperature) if alpha > 0 else 0.0
        adapter = m.task_adapter
        if adapter is not None:
            # A declared task scores the labels over its valid targets (#513), validated before any backward.
            hard = task_term(adapter, m.loss_fn, student_logits, Y)
            loss = alpha * soft_loss + (1.0 - alpha) * hard.loss if alpha < 1 else soft_loss
            assert isinstance(loss, torch.Tensor)
            trains = alpha > 0 or hard.trains  # every target masked at alpha=0: nothing to learn from
            loss_val = finalize_task_step(loss, ctx, trains=trains, paradigm="distillation")
            return task_record(adapter, [(1.0, hard)], loss=loss_val, extra_metrics=ctx.extra_metrics)
        # alpha=1 is pure distillation: no hard term (an all-ignored batch would make it NaN).
        hard_loss = m.loss_fn(_loss_input(m.loss_fn, student_logits), Y) if alpha < 1 else 0.0
        loss = alpha * soft_loss + (1.0 - alpha) * hard_loss
        assert isinstance(loss, torch.Tensor)  # 0 <= alpha <= 1: at least one term is a tensor
        _check_scorable_target(
            m.loss_fn, Y, paradigm="distillation"
        )  # before the update: a refused batch changes nothing
        loss_val = finalize_step(loss, ctx, paradigm="distillation")

        return _classification_edp_for_loss(
            loss_fn=m.loss_fn,
            target=Y,
            prediction=predicted_labels(m.loss_fn, student_logits),
            loss=loss_val,
            extra_metrics=ctx.extra_metrics,
        )

    # finalize_step: full precision
    return full_precision_only(imperative_step(task_step(step, _KD_STEP_TASKS), paradigm="kd"))


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

    A declared ``TaskSpec`` is treated as by :func:`kd_train_step_factory`:
    the hard term scores the valid targets only, a batch with nothing to
    learn from (every target masked at ``alpha=0, beta=0``) takes no
    update, and a regression task is refused before any run is reserved.

    Returns:
        A ``TrainStepFn`` suitable for ``NNModel.train(...,
        train_step_fn=...)``.

    Raises:
        ValueError: if ``alpha`` or ``beta`` is not a real in [0, 1], if
            ``temperature`` is not a finite real > 0 (a ``bool`` is refused),
            or if ``auxiliary_layers`` is empty.
            On the first batch, if any paired teacher/student
            activation shapes disagree, or if the head has a single logit
            and ``alpha > 0`` (no class distribution to distill).
            ``NNModel.train`` refuses a model that declares a regression
            task (``TaskValidationError``) before any run is reserved.
        TypeError: if ``teacher`` is not an ``NNModel`` (stored teacher
            records train through ``TeacherDataset.objective()``).
    """
    owner = "feature_kd_train_step_factory"
    require_finite_real(alpha, "alpha", owner=owner, minimum=0, maximum=1)
    require_finite_real(beta, "beta", owner=owner, minimum=0, maximum=1)
    require_finite_real(temperature, "temperature", owner=owner, minimum=0, exclusive_min=True)
    if not auxiliary_layers:
        raise ValueError(
            "auxiliary_layers must be a non-empty mapping of teacher_layer_name -> student_layer_name; got empty dict"
        )

    _freeze_teacher(teacher)  # same guarantee as kd_train_step_factory

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

        X, Y = _single_input_batch(m, ctx.batch, who="feature_kd_train_step_factory")

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

            student_logits = m._net_forward((X,), {})
            require_row_logits(student_logits, "feature distillation")
            with torch.no_grad():
                # Teacher might live on a different device; migrate the
                # input into its frame for the forward pass.
                teacher_logits = teacher._net_forward((X.to(teacher.device),), {}).to(m.device)
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

        # alpha=0 is plain supervised training: no soft term (a binary head has none).
        soft_loss = softened_kl(student_logits, teacher_logits, temperature) if alpha > 0 else 0.0
        adapter = m.task_adapter
        if adapter is not None:
            # As kd_train_step_factory: the labels over the task's valid targets (#513).
            hard = task_term(adapter, m.loss_fn, student_logits, Y)
            loss = alpha * soft_loss + beta * feature_loss + ((1.0 - alpha) * hard.loss if alpha < 1 else 0.0)
            trains = alpha > 0 or beta > 0 or hard.trains  # all masked at alpha=0, beta=0: nothing to learn from
            loss_val = finalize_task_step(loss, ctx, trains=trains, paradigm="feature_distillation")
            return task_record(adapter, [(1.0, hard)], loss=loss_val, extra_metrics=ctx.extra_metrics)
        # alpha=1 is pure distillation: no hard term (an all-ignored batch would make it NaN).
        hard_loss = m.loss_fn(_loss_input(m.loss_fn, student_logits), Y) if alpha < 1 else 0.0
        loss = alpha * soft_loss + beta * feature_loss + (1.0 - alpha) * hard_loss
        _check_scorable_target(m.loss_fn, Y, paradigm="feature distillation")
        loss_val = finalize_step(loss, ctx, paradigm="feature_distillation")

        return _classification_edp_for_loss(
            loss_fn=m.loss_fn,
            target=Y,
            prediction=predicted_labels(m.loss_fn, student_logits),
            loss=loss_val,
            extra_metrics=ctx.extra_metrics,
        )

    # finalize_step: full precision
    return full_precision_only(
        imperative_step(task_step(step, _StepTasks("feature_kd_train_step_factory")), paradigm="feature_kd")
    )
