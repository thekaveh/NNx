"""Born-Again Networks — iterated self-distillation.

A born-again network is a sequence of generations where each generation
is trained to mimic the previous generation's outputs (Furlanello et al.,
"Born Again Neural Networks", ICML 2018). Generation 0 is trained from
scratch with the standard supervised loss; generation k > 0 uses
generation k-1 as the (frozen) teacher in a Hinton-style KD step.

Despite the student and teacher having identical architecture and seeing
the same data, the student often matches or slightly outperforms the
teacher — the soft targets act as an implicit regularizer / label
smoother. The pedagogical value is high: born-again is the cheapest
demonstration that KD's gain isn't purely from compressing a bigger
teacher into a smaller student.

The wrapper is intentionally minimal — it composes
:func:`kd_train_step_factory` with :meth:`NNModel.train` across G
generations. It does not introduce a new params dataclass, does not
touch ``NNModel`` internals, and returns the per-generation
:class:`NNRun` list so callers can plot the convergence trajectory.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

from ..nn.nn_model import NNModel
from ..nn.params.nn_run import NNRun
from ..nn.params.nn_train_params import NNTrainParams
from .distillation import kd_train_step_factory


def born_again_train(
    model: NNModel,
    *,
    generations: int = 3,
    train_params: NNTrainParams,
    **kd_kwargs: Any,
) -> list[NNRun]:
    """Iterate G generations of self-distillation on a single model.

    Generation 0 trains plain (no teacher) — standard supervised loss.
    Each subsequent generation uses a deep-copied, frozen, eval-mode
    snapshot of the model *after* the prior generation completed as the
    teacher for a Hinton-style KD step (via :func:`kd_train_step_factory`).

    The same ``NNModel`` wrapper is reused, but its network weights are reset
    to the caller-provided initialization before every student generation.
    This follows the original Born-Again Networks procedure while keeping
    memory usage to the live student, one frozen teacher, and one initial
    state dictionary.

    Args:
        model: the :class:`NNModel` to train. Its initial weights seed every
            fresh student; its final state corresponds to the LAST generation.
        generations: how many generations to run. ``generations=1`` is
            a plain supervised run (no KD) — kept as a degenerate case
            so callers can sweep generations including the baseline.
            Must be ``≥ 1``.
        train_params: base configuration for every :meth:`NNModel.train`
            call. Generation zero uses it unchanged. Each later generation
            records the preceding run as its parent, producing a distinct
            content-addressed run with independent history and BEST tracking.
        **kd_kwargs: forwarded to :func:`kd_train_step_factory` for
            generations ≥ 1 (``alpha``, ``temperature``). Ignored on
            generation 0 (no teacher).

    Returns:
        A list of :class:`NNRun` objects, one per generation, in order.
        ``runs[0]`` is the plain run; ``runs[k]`` for ``k > 0`` is the
        KD run that used generation ``k-1``'s model as teacher.

    Raises:
        ValueError: if ``generations < 1``.
    """
    if generations < 1:
        raise ValueError(f"generations must be >= 1, got {generations}")

    runs: list[NNRun] = []
    teacher: NNModel | None = None
    initial_state = copy.deepcopy(model.net.state_dict())
    generation_params = train_params
    for g in range(generations):
        if g > 0:
            model.net.load_state_dict(initial_state)
        if teacher is None:
            run = model.train(params=generation_params)
        else:
            step_fn = kd_train_step_factory(teacher=teacher, **kd_kwargs)
            run = model.train(params=generation_params, train_step_fn=step_fn)
        runs.append(run)

        if g == generations - 1:
            # The last generation has no successor — deep-copying a
            # never-used teacher would only double peak memory at exit.
            break

        # Snapshot the just-trained model as the next generation's
        # teacher. deepcopy duplicates the net's parameters into a
        # detached graph; freezing + eval-mode is then enforced here
        # (and again inside kd_train_step_factory as belt-and-braces).
        teacher = copy.deepcopy(model)
        teacher.net.eval()
        for p in teacher.net.parameters():
            p.requires_grad = False
        generation_params = replace(
            train_params,
            resume_from_run_id=None,
            parent_run_id=run.id,
            resume_from_checkpoint="last",
        )

    return runs
