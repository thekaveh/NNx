from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union

from ..._validation import require_count, require_finite_real
from ..enum.optims import Optims
from .nn_params import _ImmutableList

if TYPE_CHECKING:
    from ...finetune.param_groups import NNParamGroupSpec
    from .nn_optim_params_builder import NNOptimParamsBuilder


# torch's Adam / AdamW default. `eps` is omitted from state() at this value,
# so every pre-existing optimizer config keeps its run.id.
_ADAM_EPS_DEFAULT = 1e-8
_ADAM_FAMILY = frozenset({Optims.ADAM, Optims.ADAM_AMSGRAD, Optims.ADAMW})
_SGD_FAMILY = frozenset({Optims.SGD, Optims.SGD_NESTEROV})


def _validate_optim_scalars(obj: Any, owner: str) -> None:
    """Validate the scalar knobs every optimizer config shares (built-in
    ``NNOptimParams`` and registered ``NNOptimFactoryParams``).

    Fails fast on out-of-range scalars — each constructs fine but
    misbehaves silently/obscurely deep in the train loop:
      * accumulate_grad_batches < 1: =0 dies mid-training with
        `ZeroDivisionError` on `batch_idx % accumulate_grad_batches`
        (AFTER printing the whole run-config table); <0 scales the
        loss by 1/N < 0 and silently performs gradient *ascent*.
      * grad_clip_norm <= 0 (when not the None "off" sentinel): 0.0
        passes the `is not None` clip-enable check and zeros every
        gradient, so training runs to completion making no progress.
      * accumulate_grad_batches is an integer count (FIX-021): a
        fractional value such as 2.5 used to reach the modulo and
        silently step every five batches. NumPy integers are
        normalized to a plain `int` so state() stays YAML-portable.
      * max_lr < 0: a negative LR performs gradient *ascent*. max_lr=0
        is allowed — it is an explicit "freeze updates" choice (used as a
        no-update idiom, e.g. probing a loss without changing weights);
        unlike the knobs above, 0 is intended, not a silent footgun.
      * weight_decay < 0: grows weights instead of decaying them (matches
        the per-group NNParamGroupSpec guard).
    Finite-real first (NaN passes every inequality), then the domain
    (FIX-020). Meaningful zeros stay valid; nothing here touches state().
    """
    accumulate = obj.accumulate_grad_batches
    object.__setattr__(
        obj,
        "accumulate_grad_batches",
        require_count(
            accumulate,
            "accumulate_grad_batches",
            owner=owner,
            minimum=1,
            domain_message=(
                f"accumulate_grad_batches must be >= 1, got {accumulate} "
                "(1 = step every batch; N = step every N batches)."
            ),
        ),
    )
    grad_clip_norm = obj.grad_clip_norm
    if grad_clip_norm is not None:
        require_finite_real(
            grad_clip_norm,
            "grad_clip_norm",
            owner=owner,
            minimum=0.0,
            exclusive_min=True,
            domain_message=(
                f"grad_clip_norm must be > 0 when set, got {grad_clip_norm} (use None to disable clipping, not 0)."
            ),
        )
    max_lr = obj.max_lr
    require_finite_real(
        max_lr,
        "max_lr",
        owner=owner,
        minimum=0.0,
        domain_message=(
            f"max_lr must be non-negative, got {max_lr} (a negative LR performs gradient ascent; 0 freezes updates)."
        ),
    )
    weight_decay = obj.weight_decay
    require_finite_real(
        weight_decay,
        "weight_decay",
        owner=owner,
        minimum=0.0,
        domain_message=f"weight_decay must be non-negative, got {weight_decay} (use 0 to disable).",
    )


def _validate_optim_param_groups(obj: Any) -> None:
    """Fail fast on plain dicts / generators in ``param_groups`` and freeze
    the list. Plain dicts construct fine but crash much later inside
    state() during NNRun hashing with an opaque AttributeError; a
    generator would be silently EXHAUSTED by the validation loop (state()
    would then emit an empty param_groups and training would run
    single-group with a shifted run.id)."""
    param_groups = obj.param_groups
    if param_groups is None:
        return
    if not isinstance(param_groups, (list, tuple)):
        raise TypeError(f"param_groups must be a list/tuple of NNParamGroupSpec, got {type(param_groups).__name__}")
    # Lazy import — keeps this low-level dataclass importable without
    # eagerly loading the finetune subpackage (no cycle today).
    from ...finetune.param_groups import NNParamGroupSpec

    for i, g in enumerate(param_groups):
        if not isinstance(g, NNParamGroupSpec):
            raise TypeError(
                f"param_groups[{i}] must be an NNParamGroupSpec, got {type(g).__name__} — "
                "wrap it: NNParamGroupSpec(name_pattern=..., lr=...)."
            )
    object.__setattr__(obj, "param_groups", _ImmutableList(param_groups))


@dataclass(frozen=True, kw_only=True, slots=True)
class NNOptimParams:
    """Optimizer config.

    `momentum` is overloaded by optimizer kind:
      - For SGD / SGD_NESTEROV: a single float, the SGD momentum coefficient.
      - For ADAM / ADAM_AMSGRAD / ADAMW: a (beta1, beta2) tuple, passed as
        the `betas=` argument. The name is retained for backwards
        compatibility — `is_valid()` enforces the per-optim shape.

    `weight_decay` follows the chosen optimizer: ADAM / ADAM_AMSGRAD add it
    to the gradient (coupled L2 penalty, rescaled by the adaptive
    denominator), while ADAMW decays the weights directly
    (``p -= lr * weight_decay * p``) before the Adam update — the
    decoupled decay of Loshchilov & Hutter.

    `eps` is the Adam-family denominator term (``1e-8``, torch's default).
    It is only meaningful for ADAM / ADAM_AMSGRAD / ADAMW; setting it on
    an SGD variant raises. It is omitted from `state()` at its default so
    existing run ids are unchanged.

    `grad_clip_norm` clips gradients by global L2 norm before optimizer.step().
    None = no clipping (back-compat default). Typical values: 1.0 for
    transformers, 5.0 for RNNs.

    `accumulate_grad_batches` enables gradient accumulation — the effective
    batch size becomes batch_size * accumulate_grad_batches. The loss is
    scaled by 1/N so the accumulated gradient is the mean across N batches.
    Default 1 (back-compat: step every batch).

    `param_groups` enables per-layer-group LR / weight_decay overrides — the
    fine-tuning idiom of "small LR on the backbone, large LR on the head."
    None = single-group behavior (every parameter at `max_lr` / `weight_decay`).
    When set, the optimizer factory dispatches via
    :func:`nnx.finetune.param_groups.build_param_groups` to construct
    per-group dicts.
    """

    name: Optims
    max_lr: float
    weight_decay: float
    momentum: Union[float, tuple[float, float]]

    grad_clip_norm: Optional[float] = None
    accumulate_grad_batches: int = 1
    param_groups: Optional[list[NNParamGroupSpec]] = field(default=None)
    eps: float = _ADAM_EPS_DEFAULT

    def __post_init__(self):
        _validate_optim_scalars(self, "NNOptimParams")
        # Adam betas in [0, 1); SGD momentum >= 0 (FIX-020).
        if isinstance(self.momentum, tuple):
            if len(self.momentum) != 2:
                raise ValueError(
                    f"NNOptimParams requires momentum betas as a (beta1, beta2) pair, got {self.momentum!r}"
                )
            for i, beta in enumerate(self.momentum):
                require_finite_real(
                    beta, f"momentum[{i}]", owner="NNOptimParams", minimum=0.0, maximum=1.0, exclusive_max=True
                )
        else:
            require_finite_real(self.momentum, "momentum", owner="NNOptimParams", minimum=0.0)
        # eps is the Adam-family denominator term: finite and > 0, and
        # rejected on SGD variants rather than silently ignored.
        eps = require_finite_real(self.eps, "eps", owner="NNOptimParams", minimum=0.0, exclusive_min=True)
        if self.name not in _ADAM_FAMILY and eps != _ADAM_EPS_DEFAULT:
            raise ValueError(
                f"NNOptimParams eps applies only to Adam-family optimizers (adam, adam_amsgrad, adamw); "
                f"{self.name} got eps={self.eps!r}"
            )
        object.__setattr__(self, "eps", eps)
        _validate_optim_param_groups(self)

    def __str__(self):
        eps = f", eps={self.eps:g}" if self.eps != _ADAM_EPS_DEFAULT else ""
        return f"[name={self.name}, max_lr={self.max_lr:1.0e}, weight_decay={self.weight_decay:1.0e}, momentum={self.momentum}{eps}, grad_clip={self.grad_clip_norm}, accum={self.accumulate_grad_batches}]"

    def state(self):
        d: dict[str, object] = dict(
            max_lr=self.max_lr,
            momentum=str(self.momentum),
            name=str(self.name),
            weight_decay=self.weight_decay,
        )
        # grad_clip_norm / accumulate_grad_batches / param_groups: only emit
        # when set to a non-default value, so a NNOptimParams with none of
        # them set hashes to the same run.id as before these fields existed.
        # Existing on-disk YAML without these keys is loadable via the .get()
        # defaults below. (This invariant was broken once before for
        # grad_clip_norm — every existing run.id shifted. Same omit-when-
        # default pattern is now enforced on every params dataclass; see
        # test_params_round_trip.py for the regression tests.)
        if self.grad_clip_norm is not None:
            d["grad_clip_norm"] = self.grad_clip_norm
        if self.accumulate_grad_batches != 1:
            d["accumulate_grad_batches"] = self.accumulate_grad_batches
        if self.param_groups is not None:
            d["param_groups"] = [g.state() for g in self.param_groups]
        if self.eps != _ADAM_EPS_DEFAULT:
            d["eps"] = self.eps
        return d

    @staticmethod
    def from_state(state: dict) -> NNOptimParams:
        # Lazy import — defers the finetune subpackage so this
        # low-level dataclass stays light at import time (no actual
        # cycle today: param_groups.py imports only stdlib + torch).
        from ...finetune.param_groups import NNParamGroupSpec

        raw_pg = state.get("param_groups")
        param_groups = [NNParamGroupSpec.from_state(g) for g in raw_pg] if raw_pg is not None else None
        return NNOptimParams(
            max_lr=state["max_lr"],
            name=Optims(state["name"]),
            weight_decay=state["weight_decay"],
            momentum=ast.literal_eval(state["momentum"]),
            # .get() preserves back-compat with older YAML that predates
            # grad_clip_norm / accumulate_grad_batches / param_groups.
            grad_clip_norm=state.get("grad_clip_norm"),
            accumulate_grad_batches=state.get("accumulate_grad_batches", 1),
            param_groups=param_groups,
            eps=state.get("eps", _ADAM_EPS_DEFAULT),
        )

    def is_valid(self) -> bool:
        if self.name in _SGD_FAMILY:
            return isinstance(self.momentum, float)
        if self.name in _ADAM_FAMILY:
            return (
                isinstance(self.momentum, tuple)
                and len(self.momentum) == 2
                and all(isinstance(x, float) for x in self.momentum)
            )
        # Unknown enum variant — refuse rather than implicitly returning None
        # (which would short-circuit `not params.optim.is_valid()` in train()).
        return False

    @classmethod
    def builder(cls) -> NNOptimParamsBuilder:
        """Return a variant-aware builder. See `NNOptimParamsBuilder`."""
        from .nn_optim_params_builder import NNOptimParamsBuilder

        return NNOptimParamsBuilder()
