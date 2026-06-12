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

    The same in-place ``model`` is reused across generations; only the
    teacher snapshot is duplicated. This matches the original paper's
    setup and keeps memory usage to two copies of the network at any
    one time (the live student + the frozen teacher snapshot).

    Args:
        model: the :class:`NNModel` to train. Mutated in place across
            generations; the final state corresponds to the LAST
            generation. Restore from a checkpoint if you need an
            intermediate generation's weights.
        generations: how many generations to run. ``generations=1`` is
            a plain supervised run (no KD) — kept as a degenerate case
            so callers can sweep generations including the baseline.
            Must be ``≥ 1``.
        train_params: passed unchanged to every :meth:`NNModel.train`
            call. The same ``run.id`` would be computed every generation
            if nothing else changed, but in practice each generation
            mutates the model's weights between calls, so the underlying
            artifacts (idps / phase checkpoints) diverge generation to
            generation even with the same id. Caveat: the shared id
            also means each generation's BEST tracking seeds from the
            previous generation's on-disk BEST — a generation that
            never beats its predecessor leaves BEST pointing at the
            EARLIER generation's weights. Run each generation from a
            fresh ``runs/`` root (fresh cwd) if you need per-generation
            artifacts or independent BEST tracking.
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
    for g in range(generations):
        if teacher is None:
            run = model.train(params=train_params)
        else:
            step_fn = kd_train_step_factory(teacher=teacher, **kd_kwargs)
            run = model.train(params=train_params, train_step_fn=step_fn)
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

    return runs
