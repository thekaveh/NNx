"""I-JEPA latent prediction as an objective (FEAT-040).

:func:`~nnx.paradigms.jepa_train_step_factory` is an imperative step: it
takes a masked mean squared error, hands it to ``finalize_step`` (which
owns the optimizer step) and runs :func:`~nnx.paradigms.update_ema` at
once — so it cannot accumulate gradients over uneven microbatches or share
the update engine. :func:`jepa_objective` describes the same loss as an
objective (``nnx.objectives``)::

    target = build_target_encoder(model.net)
    predictor = JEPAPredictor(embed_dim=..., n_patches=model.net.n_patches)
    model.net.add_module("_jepa_predictor", predictor)  # optimizer-owned, once
    model.train(params=NNTrainParams(...), objective=jepa_objective(target, predictor, mask_fn))

For each microbatch the objective returns one ``"latent_mse"`` loss term:
the **sum** of squared errors between the predicted and the target
embeddings at the masked positions, normalized by the number of those
elements — so microbatches with different target masks combine as
``(sum_1 + sum_2) / (count_1 + count_2)``. The EMA target encoder runs
under ``no_grad`` and is never optimizer-owned; it advances **once per
committed update**, from the updated online weights (the objective's
``after_update`` hook) — never per microbatch, and not at all for a skipped
window.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional, cast

import torch
from torch import nn

from .._step_helpers import first_input, full_precision
from ..components import ComponentSpec
from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from ..objectives import LossTerm, Objective, ObjectiveContext, ObjectiveResult, UpdateEvent
from .jepa import _broadcast_mask, update_ema

__all__ = ["JEPAObjective", "jepa_objective"]

MaskFn = Callable[[int, torch.device], "tuple[torch.Tensor, torch.Tensor]"]

_TERM = "latent_mse"
_STATE_KEYS = frozenset({"spec", "predictor", "target_encoder", "ema_updates"})


def _submodule_path(root: nn.Module, module: nn.Module) -> Optional[str]:
    """The dotted name ``module`` is registered under in ``root`` (``""``
    for ``root`` itself), or ``None`` when it is not part of ``root``."""
    for name, candidate in root.named_modules():
        if candidate is module:
            return name
    return None


def _check_masks(context: torch.Tensor, target: torch.Tensor, n_patches: int) -> None:
    """The mask contract, checked before any forward pass: two complementary
    ``BoolTensor[n_patches]`` with at least one context and one target patch."""
    if not isinstance(context, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("mask_fn must return two tensors (context_mask, target_mask)")
    if context.shape != (n_patches,) or target.shape != (n_patches,):
        raise ValueError(
            f"mask_fn must return two BoolTensors of shape ({n_patches},); got {tuple(context.shape)} and "
            f"{tuple(target.shape)}"
        )
    if context.dtype != torch.bool or target.dtype != torch.bool:
        raise ValueError(f"mask_fn must return BoolTensors, got {context.dtype} and {target.dtype}")
    # One device sync in the common case; the precise reason only on failure.
    if bool((context == target).any() | ~target.any() | ~context.any()):
        if not torch.equal(context, ~target):
            raise ValueError(
                "context_mask and target_mask must be complementary (every patch is either context or target)."
            )
        if not bool(target.any()):
            raise ValueError("the target mask is empty: there is nothing to predict (every patch is context)")
        raise ValueError("the context mask is empty: there is nothing to predict from (every patch is a target)")


class JEPAObjective(Objective):
    """I-JEPA latent prediction as an objective — see the module docstring.

    Callable as an objective and a checkpointable component (FEAT-005)
    named ``"jepa.objective"``: its state is the objective spec (loss term,
    EMA momentum), the predictor's **reference** (its name inside
    ``model.net`` — its weights are saved once, with the net), the EMA
    target encoder's weights and the EMA update counter, so a stateful
    resume continues exactly where an uninterrupted run would be. A saved
    state carrying a second copy of the predictor's weights, a different
    predictor reference or a different spec is rejected before anything is
    restored.

    Before any run is reserved (:meth:`check_run`) the objective refuses:
    a net without the ViT patch contract (``n_patches``,
    ``patch_positions()``); a predictor that is not a submodule of
    ``model.net`` or whose trainable parameters are not owned exactly once
    by the run's optimizers; a target encoder inside ``model.net``,
    sharing its parameters or owned by an optimizer; a target parameter
    without a same-named, same-shaped online parameter; and callbacks that
    change the net's topology (``checkpoint_transforms``, e.g. QAT), which
    would break the EMA's name correspondence and the predictor reference.

    Args:
        target_encoder: the EMA copy of ``model.net``
            (:func:`~nnx.paradigms.build_target_encoder`); frozen and put in
            eval mode here.
        predictor: the predictor (:class:`~nnx.paradigms.JEPAPredictor` or
            the same ``forward(context_embeds, context_positions,
            target_positions)`` contract), registered under ``model.net``.
        mask_fn: ``(n_patches, device) -> (context_mask, target_mask)``,
            complementary ``BoolTensor[n_patches]``, sampled once per
            microbatch (shared across its rows). Masks drawn from the global
            RNG (:func:`~nnx.paradigms.random_block_mask`'s default) continue
            across a stateful resume, which restores that RNG.
        ema_momentum: fixed EMA decay in ``[0, 1)``:
            ``target ← momentum · target + (1 − momentum) · online``.
        nonfinite: the engine's non-finite policy (``"fail"`` / ``"skip"``).
    """

    def __init__(
        self,
        target_encoder: nn.Module,
        predictor: nn.Module,
        mask_fn: MaskFn,
        *,
        ema_momentum: float = 0.996,
        nonfinite: str = "fail",
    ) -> None:
        super().__init__(nonfinite=nonfinite)
        if (
            isinstance(ema_momentum, bool)
            or not isinstance(ema_momentum, numbers.Real)
            or not 0.0 <= float(ema_momentum) < 1.0
        ):
            raise ValueError(f"ema_momentum must be a number in [0, 1), got {ema_momentum!r}")
        if not isinstance(target_encoder, nn.Module) or not isinstance(predictor, nn.Module):
            raise TypeError("target_encoder and predictor must be torch.nn.Module instances")
        if target_encoder is predictor:
            raise ValueError("target_encoder and predictor must be different modules")
        if not callable(mask_fn):
            raise TypeError(f"mask_fn must be callable, got {type(mask_fn).__name__}")
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.mask_fn = mask_fn
        self.ema_momentum = float(ema_momentum)
        self.ema_updates = 0
        self._predictor_path: Optional[str] = None
        self._online: Optional[nn.Module] = None
        _freeze(target_encoder)

    # ---------- before the run ----------

    def check_run(
        self, model: Any, *, optimizers: Mapping[str, torch.optim.Optimizer], callbacks: Sequence[Any]
    ) -> None:
        from ..nn.callbacks import Callback

        net = model.net
        if not hasattr(net, "n_patches") or not callable(getattr(net, "patch_positions", None)):
            raise ValueError(
                f"jepa_objective needs a ViT-style model.net with n_patches and patch_positions(); got "
                f"{type(net).__name__}"
            )
        path = _submodule_path(net, self.predictor)
        if not path:
            raise ValueError(
                "the JEPA predictor must be a submodule of model.net (e.g. model.net.add_module('_jepa_predictor', "
                "predictor)) so the run's optimizer owns its parameters and checkpoints save them once, with the net"
            )
        if _submodule_path(net, self.target_encoder) is not None:
            raise ValueError("the JEPA target encoder must not be part of model.net: the optimizer would train it")
        online = {id(p) for p in net.parameters()}
        if any(id(p) in online for p in self.target_encoder.parameters()):
            raise ValueError(
                "the JEPA target encoder shares parameters with model.net; build it with build_target_encoder"
            )
        owned: dict[int, int] = {}
        for optimizer in optimizers.values():
            for group in optimizer.param_groups:
                for param in group["params"]:
                    owned[id(param)] = owned.get(id(param), 0) + 1
        if any(id(p) in owned for p in self.target_encoder.parameters()):
            raise ValueError("an optimizer owns JEPA target-encoder parameters; the target is updated only by its EMA")
        unowned = [n for n, p in self.predictor.named_parameters() if p.requires_grad and id(p) not in owned]
        if unowned:
            raise ValueError(
                f"the run's optimizers do not own the JEPA predictor parameters {unowned}; build the optimizer over "
                "model.net after registering the predictor (and leave them trainable)"
            )
        twice = [n for n, p in self.predictor.named_parameters() if owned.get(id(p), 0) > 1]
        if twice:
            raise ValueError(f"the JEPA predictor parameters {twice} are owned by more than one optimizer")
        sources = dict(net.named_parameters())
        unmatched = [
            n for n, p in self.target_encoder.named_parameters() if n not in sources or sources[n].shape != p.shape
        ]
        if unmatched:
            raise ValueError(
                f"JEPA target-encoder parameters {unmatched} have no same-named, same-shaped counterpart in "
                "model.net, so the EMA cannot track them; rebuild the target from the current net"
            )
        changing = [
            type(cb).__name__
            for cb in callbacks
            if isinstance(cb, Callback) and type(cb).checkpoint_transforms is not Callback.checkpoint_transforms
        ]
        if changing:
            raise ValueError(
                f"callbacks {changing} change model.net's topology, which breaks the JEPA target's EMA name "
                "correspondence and the predictor reference; train without them, or with the imperative step"
            )
        self._predictor_path = path

    # ---------- per microbatch ----------

    def __call__(self, ctx: ObjectiveContext) -> ObjectiveResult:
        model = ctx.model
        net = cast(Any, model.net)
        n_patches = int(net.n_patches)
        context_1d, target_1d = self.mask_fn(n_patches, model.device)
        _check_masks(context_1d, target_1d, n_patches)  # before any forward pass
        x = first_input(model, ctx.batch)
        if x.shape[0] == 0:
            raise ValueError("jepa_objective got an empty batch")
        self._online = model.net
        model.net.train()  # the predictor is a submodule: it trains too
        self.target_encoder.eval()

        context_embeds = model.net(x, mask=_broadcast_mask(context_1d, x.shape[0]))
        positions = net.patch_positions()
        context_positions = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=positions.device), positions[context_1d]]
        )
        target_positions = positions[target_1d]
        with torch.no_grad():
            target_embeds = self.target_encoder(x)[:, target_positions, :]
        predicted = self.predictor(context_embeds, context_positions, target_positions)
        error = full_precision(predicted) - full_precision(target_embeds)
        term = LossTerm(_TERM, error.pow(2).sum(), error.numel(), "mean")
        value = term.value
        assert value is not None  # a non-empty target mask always has elements
        # Detached metrics only: no classification fields for a self-supervised loss.
        return ObjectiveResult((term,), NNEvaluationDataPoint(loss=value, metrics={_TERM: value}))

    # ---------- once per committed update ----------

    def after_update(self, events: tuple[UpdateEvent, ...]) -> None:
        """Advance the EMA target from the updated online weights — once per
        committed update, whatever the number of named optimizers."""
        if self._online is None:
            raise RuntimeError("jepa_objective.after_update ran before any microbatch")
        update_ema(self._online, self.target_encoder, self.ema_momentum)
        self.ema_updates += 1

    # ---------- checkpointable component (FEAT-005) ----------

    def _spec(self) -> dict[str, Any]:
        return {"objective": "jepa", "loss": _TERM, "ema_momentum": self.ema_momentum}

    def component_spec(self) -> ComponentSpec:
        return ComponentSpec("jepa.objective", version=1)

    def component_state(self) -> dict[str, Any]:
        # Detached live tensors: checkpoints serialize them at once, and a
        # restore snapshots them with its own deep copy. The predictor's
        # weights are not here: they belong to model.net (saved once).
        return {
            "spec": self._spec(),
            "predictor": self._predictor_path,
            "target_encoder": dict(self.target_encoder.state_dict()),
            "ema_updates": self.ema_updates,
        }

    def check_component_state(self, state: Mapping[str, Any], *, version: int) -> list[str]:
        problems: list[str] = []
        extra = sorted(set(state) - _STATE_KEYS)
        if extra:
            problems.append(
                f"the JEPA objective state carries {extra}: a second predictor (or other) payload — the predictor's "
                "weights are saved once, with model.net"
            )
        if state.get("spec") != self._spec():
            problems.append(
                f"the checkpoint's JEPA objective {state.get('spec')!r} does not match this one {self._spec()!r}"
            )
        if state.get("predictor") != self._predictor_path:
            problems.append(
                f"the checkpoint's JEPA predictor is model.net.{state.get('predictor')}, this run's is "
                f"model.net.{self._predictor_path}"
            )
        saved = state.get("target_encoder")
        current = self.target_encoder.state_dict()
        if not isinstance(saved, Mapping) or set(saved) != set(current):
            problems.append("the checkpoint's JEPA target encoder does not match this target's parameters")
        else:
            reshaped = [k for k, v in saved.items() if not isinstance(v, torch.Tensor) or v.shape != current[k].shape]
            if reshaped:
                problems.append(f"the checkpoint's JEPA target-encoder tensors {reshaped} have different shapes")
        count = state.get("ema_updates")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            problems.append(f"malformed JEPA EMA update counter {count!r}")
        return problems

    def load_component_state(self, state: Mapping[str, Any], *, version: int) -> None:
        self.target_encoder.load_state_dict(state["target_encoder"])
        _freeze(self.target_encoder)  # the EMA copy stays frozen and in eval mode
        self.ema_updates = int(state["ema_updates"])


def _freeze(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def jepa_objective(
    target_encoder: nn.Module,
    predictor: nn.Module,
    mask_fn: MaskFn,
    *,
    ema_momentum: float = 0.996,
    nonfinite: str = "fail",
) -> JEPAObjective:
    """I-JEPA latent prediction as an objective (see :class:`JEPAObjective`)
    — the objective counterpart of :func:`jepa_train_step_factory`, which
    stays available (and unchanged) as the imperative step."""
    return JEPAObjective(target_encoder, predictor, mask_fn, ema_momentum=ema_momentum, nonfinite=nonfinite)
