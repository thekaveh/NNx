"""A causal language-model task: declared alignment, masks and valid-token
normalization (FEAT-034).

A next-token step written by hand flattens ``(B, T, V)`` logits before any
mask exists, divides by the batch count when it accumulates and returns
placeholder classification fields. :class:`CausalLMTask` declares instead:

- **alignment** — ``"shift_inputs"``: a batch is token ids ``(B, T)`` (or
  ``(ids, loss_mask)``) and the task shifts once, predicting token ``t + 1``
  from tokens ``<= t``; ``"pre_shifted"``: a batch is ``(inputs, targets)``
  (or ``(inputs, targets, loss_mask)``) already aligned, never shifted again;
- the **vocabulary** size and axis (the logits' last axis), the **ignore**
  id (``-100``) and an optional **padding** id, an optional per-position
  **loss mask** (a loss mask is never an attention mask), label
  **smoothing** for the objective, and the task **version**.

The ignore id, the padding id and the loss mask select one set of **valid
positions** — the objective's denominator, the NLL and the token accuracy
all use exactly those. Shapes, dtypes, lengths and every non-ignored id are
checked before any forward pass.

- :meth:`CausalLMTask.objective` — a FEAT-004 objective: one ``"token_ce"``
  term per microbatch, the (smoothed) cross-entropy **sum** over valid
  positions with the valid count as its denominator, so accumulation
  windows normalize by the window's valid tokens. An all-masked microbatch
  contributes nothing, and an all-masked window takes no optimizer or
  scaler step (its gradients are cleared; schedulers still advance once per
  epoch, and a plateau scheduler skips an epoch whose validation record is
  unavailable). The task configuration is
  checkpointed component state (``"lm.causal_task"``): a resume with another
  configuration is refused before the first resumed update.
- :meth:`CausalLMTask.eval_step` — an ``eval_step_fn``: the epoch's
  **unsmoothed** NLL is the total valid-token negative log likelihood over
  the valid-token count (independent of batching and padding), perplexity is
  ``exp(NLL)`` (``+inf`` when it overflows, never a clipped finite value) and
  token accuracy is the top-1 rate. They are the record's ``loss`` and its
  ``metrics`` ``nll`` / ``perplexity`` / ``token_accuracy``; an epoch whose
  every position is masked is reported unavailable (no loss, no metrics).
  No classification field is fabricated.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import torch

from .components import ComponentSpec
from .objectives import LossTerm, Objective, ObjectiveContext, ObjectiveResult

__all__ = [
    "ALIGNMENTS",
    "CausalLMEval",
    "CausalLMObjective",
    "CausalLMTask",
    "LMTaskError",
    "perplexity",
]

ALIGNMENTS = ("shift_inputs", "pre_shifted")
TASK_VERSION = 1
KIND = "causal_lm"


class LMTaskError(ValueError):
    """A task configuration or a batch the task rejects."""


def _integer(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise LMTaskError(f"{what} must be an integer, got {value!r}")
    return int(value)


def perplexity(nll: Optional[float]) -> Optional[float]:
    """``exp(nll)``; ``+inf`` when it overflows (never a clipped value)."""
    if nll is None:
        return None
    try:
        return math.exp(nll)
    except OverflowError:
        return math.inf


@dataclass(frozen=True)
class CausalLMTask:
    """A next-token prediction task.

    Args:
        vocab_size: the vocabulary size; the logits' class axis must have
            exactly this width.
        alignment: ``"shift_inputs"`` (the task shifts token ids once) or
            ``"pre_shifted"`` (batches carry aligned inputs and targets).
        ignore_id: the target id that marks a position as not scored.
        pad_id: an optional padding id: target positions holding it are not
            scored either (input positions may hold it).
        smoothing: label smoothing for the **objective** only, in ``[0, 1)``;
            reported NLL and perplexity are always unsmoothed.
        vocab_axis: the logits' vocabulary axis (``-1``, the last).
        tokenizer: an optional declared tokenizer identity (for example the
            tokenizer file's digest), checkpointed with the task.
        version: the task schema version.
    """

    vocab_size: int
    alignment: str = "shift_inputs"
    ignore_id: int = -100
    pad_id: Optional[int] = None
    smoothing: float = 0.0
    vocab_axis: int = -1
    tokenizer: Optional[str] = None
    version: int = TASK_VERSION

    def __post_init__(self) -> None:
        vocab = _integer(self.vocab_size, "vocab_size")
        if vocab < 2:
            raise LMTaskError(f"vocab_size must be at least 2, got {vocab}")
        if self.alignment not in ALIGNMENTS:
            raise LMTaskError(f"alignment must be one of {ALIGNMENTS}, got {self.alignment!r}")
        ignore = _integer(self.ignore_id, "ignore_id")
        if 0 <= ignore < vocab:
            raise LMTaskError(f"ignore_id {ignore} is a real token id (vocabulary 0..{vocab - 1}); use a negative id")
        if self.pad_id is not None:
            pad = _integer(self.pad_id, "pad_id")
            if not 0 <= pad < vocab:
                raise LMTaskError(f"pad_id {pad} is outside the vocabulary 0..{vocab - 1}")
            object.__setattr__(self, "pad_id", pad)  # a builtin int: checkpoint state stays weights_only-safe
        smoothing = self.smoothing
        if isinstance(smoothing, bool) or not isinstance(smoothing, numbers.Real) or not 0.0 <= smoothing < 1.0:
            raise LMTaskError(f"smoothing must be in [0, 1), got {smoothing!r}")
        object.__setattr__(self, "smoothing", float(smoothing))
        if isinstance(self.vocab_axis, bool) or self.vocab_axis not in (-1,):
            raise LMTaskError(f"vocab_axis must be -1 (the logits' last axis), got {self.vocab_axis!r}")
        object.__setattr__(self, "vocab_axis", -1)
        if self.tokenizer is not None and (not isinstance(self.tokenizer, str) or not self.tokenizer):
            raise LMTaskError(f"tokenizer must be a non-empty identity string or None, got {self.tokenizer!r}")
        if isinstance(self.version, bool) or self.version != TASK_VERSION:
            raise LMTaskError(f"this NNx supports causal-LM task version {TASK_VERSION}, got {self.version!r}")
        object.__setattr__(self, "version", TASK_VERSION)
        object.__setattr__(self, "vocab_size", vocab)
        object.__setattr__(self, "ignore_id", ignore)

    # ---------- identity ----------

    def state(self) -> dict[str, Any]:
        return {
            "kind": KIND,
            "version": self.version,
            "vocab_size": self.vocab_size,
            "alignment": self.alignment,
            "ignore_id": self.ignore_id,
            "pad_id": self.pad_id,
            "smoothing": self.smoothing,
            "vocab_axis": self.vocab_axis,
            "tokenizer": self.tokenizer,
        }

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> CausalLMTask:
        if not isinstance(state, Mapping) or state.get("kind") != KIND:
            raise LMTaskError(f"not a causal-LM task state: {state!r}")
        known = {"kind", "version", "vocab_size", "alignment", "ignore_id", "pad_id", "smoothing", "vocab_axis"}
        unknown = sorted(set(state) - known - {"tokenizer"})
        if unknown:
            raise LMTaskError(f"a causal-LM task state has unknown keys {unknown}")
        return CausalLMTask(**{k: state[k] for k in known - {"kind"} if k in state}, tokenizer=state.get("tokenizer"))

    # ---------- batches ----------

    def split(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """``(inputs, targets, loss_mask)`` for one batch, aligned once (see
        the module docstring), with shapes, dtypes and ids checked.

        ``"shift_inputs"`` takes token ids ``(B, T)``, ``(ids,)``, ``(ids,
        loss_mask)`` or a mapping ``{"input_ids", "labels"?, "loss_mask"?}``
        — HuggingFace-style ``labels`` sit at the ids' positions (unshifted,
        ``ignore_id`` where not scored) and are shifted with them.
        ``"pre_shifted"`` takes ``(inputs, targets[, loss_mask])`` or a
        mapping ``{"inputs", "targets", "loss_mask"?}``. A loss mask is
        boolean or 0/1 and covers the ids (``"shift_inputs"``) or the
        targets (``"pre_shifted"``). A mapping's ``attention_mask`` is
        ignored: a loss mask is never an attention mask."""
        if isinstance(batch, Mapping):
            return self._split_mapping(batch)
        if isinstance(batch, torch.Tensor):
            parts: tuple[Any, ...] = (batch,)
        elif isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
            parts = tuple(batch)
        else:
            raise LMTaskError(f"a causal-LM batch is a tensor, a tuple or a mapping, got {type(batch).__name__}")
        if self.alignment == "shift_inputs":
            if len(parts) not in (1, 2):
                raise LMTaskError(
                    "alignment='shift_inputs' takes token ids, (ids,) or (ids, loss_mask); got "
                    f"{len(parts)} parts — pass alignment='pre_shifted' for (inputs, targets) batches"
                )
            mask = parts[1] if len(parts) == 2 else None
            if isinstance(mask, torch.Tensor) and not mask.is_floating_point() and mask.dtype != torch.bool:
                if not bool(((mask == 0) | (mask == 1)).all()):
                    raise LMTaskError(
                        "alignment='shift_inputs' reads (ids, loss_mask), and the second part is not a 0/1 mask; "
                        "for (inputs, targets) batches pass alignment='pre_shifted'"
                    )
            return self._shifted(parts[0], None, mask)
        if len(parts) not in (2, 3):
            raise LMTaskError(
                f"alignment='pre_shifted' takes (inputs, targets) or (inputs, targets, loss_mask); got {len(parts)} parts"
            )
        return self._aligned(parts[0], parts[1], parts[2] if len(parts) == 3 else None)

    def _split_mapping(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        keys = set(batch)
        if self.alignment == "shift_inputs":
            unknown = sorted(keys - {"input_ids", "labels", "loss_mask", "attention_mask"})
            if "input_ids" not in keys or unknown:
                raise LMTaskError(
                    "alignment='shift_inputs' reads a mapping {'input_ids', 'labels'?, 'loss_mask'?}; "
                    f"got keys {sorted(keys)}"
                    + (" — pass alignment='pre_shifted' for inputs/targets" if unknown else "")
                )
            return self._shifted(batch["input_ids"], batch.get("labels"), batch.get("loss_mask"))
        unknown = sorted(keys - {"inputs", "targets", "loss_mask", "attention_mask"})
        if not {"inputs", "targets"} <= keys or unknown:
            hint = (
                " — {'input_ids', 'labels'} is HuggingFace-style (labels unshifted): pass alignment='shift_inputs'"
                if "input_ids" in keys
                else ""
            )
            raise LMTaskError(
                f"alignment='pre_shifted' reads a mapping {{'inputs', 'targets', 'loss_mask'?}}; got keys {sorted(keys)}"
                + hint
            )
        return self._aligned(batch["inputs"], batch["targets"], batch.get("loss_mask"))

    def _shifted(
        self, ids: Any, labels: Any, loss_mask: Any
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        ids = self._ids(ids, "token ids")
        if ids.shape[1] < 2:
            raise LMTaskError(f"alignment='shift_inputs' needs sequences of at least 2 tokens, got {ids.shape[1]}")
        if labels is None:
            source = ids
        else:
            source = self._ids(labels, "labels")
            if source.shape != ids.shape:
                raise LMTaskError(f"labels {tuple(source.shape)} must have the ids' shape {tuple(ids.shape)}")
        inputs, targets = ids[:, :-1], source[:, 1:]
        mask = None if loss_mask is None else self._mask(loss_mask, ids.shape)[:, 1:]  # scores the target positions
        return self._checked(inputs, targets, mask)

    def _aligned(
        self, inputs: Any, targets: Any, loss_mask: Any
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        inputs = self._ids(inputs, "inputs")
        targets = self._ids(targets, "targets")
        if inputs.shape != targets.shape:
            raise LMTaskError(
                f"pre-shifted inputs {tuple(inputs.shape)} and targets {tuple(targets.shape)} differ in shape"
            )
        if inputs.shape[1] < 1:
            raise LMTaskError("pre-shifted windows need at least one position")
        mask = None if loss_mask is None else self._mask(loss_mask, targets.shape)
        return self._checked(inputs, targets, mask)

    def _checked(
        self, inputs: torch.Tensor, targets: torch.Tensor, loss_mask: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        bad_inputs = (inputs < 0) | (inputs >= self.vocab_size)
        if bool(bad_inputs.any()):
            raise LMTaskError(f"input ids must be in 0..{self.vocab_size - 1}; found {inputs[bad_inputs][:5].tolist()}")
        scored = targets != self.ignore_id
        bad_targets = scored & ((targets < 0) | (targets >= self.vocab_size))
        if bool(bad_targets.any()):
            raise LMTaskError(
                f"non-ignored target ids must be in 0..{self.vocab_size - 1} (ignore_id={self.ignore_id}); "
                f"found {targets[bad_targets][:5].tolist()}"
            )
        return inputs, targets, loss_mask

    @staticmethod
    def _ids(value: Any, what: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise LMTaskError(f"{what} must be a tensor, got {type(value).__name__}")
        if value.dtype not in (torch.int64, torch.int32, torch.int16, torch.uint8, torch.int8):
            raise LMTaskError(f"{what} must be integer token ids, got dtype {value.dtype}")
        if value.ndim != 2:
            raise LMTaskError(f"{what} must be (batch, length), got shape {tuple(value.shape)}")
        return value.long()

    @staticmethod
    def _mask(value: Any, shape: torch.Size) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
            got = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            raise LMTaskError(f"the loss mask must be a tensor of shape {tuple(shape)}, got {got}")
        if value.dtype != torch.bool:
            if value.is_floating_point() or not bool(((value == 0) | (value == 1)).all()):
                raise LMTaskError("the loss mask must be boolean or 0/1 integers")
            value = value.bool()
        return value

    def valid(self, targets: torch.Tensor, loss_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """The scored positions: not ``ignore_id``, not ``pad_id`` and, when
        given, inside the loss mask."""
        valid = targets != self.ignore_id
        if self.pad_id is not None:
            valid &= targets != self.pad_id
        if loss_mask is not None:
            valid &= loss_mask.to(valid.device)
        return valid

    def logits(self, model: Any, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """The model's logits ``(B, T, V)`` for ``inputs``, checked."""
        raw = model.net(inputs.to(model.device))
        logits = getattr(raw, "logits", None)
        if logits is None:
            adapter = getattr(model, "_batch_adapter", None)  # a registered module's own output rule (FEAT-006)
            try:
                logits = raw if adapter is None else adapter.output(raw)
            except TypeError as error:
                raise LMTaskError(f"the model must return floating logits: {error}") from error
        if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
            raise LMTaskError(f"the model must return floating logits, got {type(logits).__name__}")
        expected = (*targets.shape, self.vocab_size)
        if tuple(logits.shape) != expected:
            raise LMTaskError(
                f"logits of shape {tuple(logits.shape)} do not fit targets {tuple(targets.shape)} and "
                f"vocab_size={self.vocab_size}: expected {expected}"
            )
        return logits

    def token_sums(
        self, logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, float, int, int]:
        """``(smoothed CE sum (differentiable), unsmoothed NLL sum, correct,
        valid count)`` over the valid positions; flattened once, here."""
        targets = targets.to(logits.device)
        valid = valid.to(logits.device)
        n = int(valid.sum())
        if n == 0:  # an empty sum keeps the graph and stays finite whatever the logits hold
            return logits.reshape(-1)[:0].sum(), 0.0, 0, 0
        flat = logits[valid]  # (n, V)
        chosen = targets[valid]
        ce = torch.nn.functional.cross_entropy(flat, chosen, reduction="sum", label_smoothing=self.smoothing)
        with torch.no_grad():
            detached = flat.detach()
            # Per-token NLL in the logits' dtype (no full-vocabulary float64
            # copy), summed in float64.
            per_token = torch.logsumexp(detached, dim=-1) - detached.gather(1, chosen.unsqueeze(1)).squeeze(1)
            nll = float(per_token.double().sum())
            correct = int((detached.argmax(dim=-1) == chosen).sum())
        return ce, nll, correct, n

    # ---------- training and evaluation ----------

    def objective(self, *, nonfinite: str = "fail") -> CausalLMObjective:
        return CausalLMObjective(self, nonfinite=nonfinite)

    def eval_step(self) -> CausalLMEval:
        return CausalLMEval(self)


def _record(*, loss: Optional[float], nll: float, correct: int, n: int, with_perplexity: bool) -> Any:
    from .nn.params.nn_evaluation_data_point import NNEvaluationDataPoint

    if n == 0:  # every position masked: unavailable, never a fabricated value
        return NNEvaluationDataPoint(kind=KIND, count=0, status="empty")
    mean = nll / n
    metrics = {"nll": mean, "token_accuracy": correct / n}
    if with_perplexity:
        metrics["perplexity"] = perplexity(mean)  # type: ignore[assignment]
    return NNEvaluationDataPoint(loss=loss, kind=KIND, count=n, status="ok", metrics=metrics)


def _no_extra_metrics(ctx: Any) -> None:
    if getattr(ctx, "extra_metrics", None):
        raise LMTaskError(
            "extra_metrics (y_true, y_pred callables) do not apply to the causal-LM task; its records carry "
            "nll, perplexity and token_accuracy — compute anything else in your own eval_step_fn"
        )


class CausalLMObjective(Objective):
    """The task's training objective (see :meth:`CausalLMTask.objective`):
    a ``"token_ce"`` term per microbatch — the cross-entropy sum over valid
    positions (smoothed when the task smooths) over the valid count. Its
    record's ``loss`` is that microbatch's term value and ``metrics`` its
    unsmoothed ``nll`` and ``token_accuracy``. Checkpointed as component
    ``"lm.causal_task"``: the task configuration and the valid tokens the
    run's objective has scored (counted from 0 for each fresh run, restored
    on resume; a window the engine then skips — a non-finite loss, an AMP
    overflow — is still counted)."""

    def __init__(self, task: CausalLMTask, *, nonfinite: str = "fail") -> None:
        super().__init__(nonfinite=nonfinite)
        if not isinstance(task, CausalLMTask):
            raise LMTaskError(f"a CausalLMTask is needed, got {type(task).__name__}")
        self.task = task
        self.tokens = 0  # valid target tokens scored in this run (restored on resume)

    def __call__(self, ctx: ObjectiveContext) -> ObjectiveResult:
        _no_extra_metrics(ctx)
        if ctx.epoch_idx == 0 and ctx.batch_idx == 0:
            self.tokens = 0  # a fresh run (a resume starts at a later epoch, its count restored)
        model = ctx.model
        model.net.train()
        inputs, targets, loss_mask = self.task.split(ctx.batch)
        valid = self.task.valid(targets, loss_mask)
        logits = self.task.logits(model, inputs, targets)
        ce, nll, correct, n = self.task.token_sums(logits, targets, valid)
        term = LossTerm("token_ce", ce, n)
        self.tokens += n
        record = _record(loss=term.value, nll=nll, correct=correct, n=n, with_perplexity=False)
        return ObjectiveResult((term,), record)

    # ---------- component state (FEAT-005) ----------

    def component_spec(self) -> ComponentSpec:
        return ComponentSpec("lm.causal_task", version=1)

    def component_state(self) -> dict[str, Any]:
        return {"task": self.task.state(), "tokens": self.tokens}

    def check_component_state(self, state: Mapping[str, Any], *, version: int) -> list[str]:
        saved = state.get("task") if isinstance(state, Mapping) else None
        if saved != self.task.state():
            changed = sorted(
                key
                for key in set(saved or {}) | set(self.task.state())
                if (saved or {}).get(key) != self.task.state().get(key)
            )
            return [
                f"the causal-LM task changed since the checkpoint ({', '.join(changed)}): {saved} -> {self.task.state()}"
            ]
        tokens = state.get("tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            return [f"the checkpoint's causal-LM token count is not a non-negative integer: {tokens!r}"]
        return []

    def load_component_state(self, state: Mapping[str, Any], *, version: int) -> None:
        self.tokens = int(state["tokens"])


class CausalLMEval:
    """The task's ``eval_step_fn`` (see :meth:`CausalLMTask.eval_step`):
    total valid-token NLL over the valid-token count for the whole
    validation loader, its perplexity and token accuracy. The model runs in
    eval mode and every submodule's training flag is restored."""

    def __init__(self, task: CausalLMTask) -> None:
        if not isinstance(task, CausalLMTask):
            raise LMTaskError(f"a CausalLMTask is needed, got {type(task).__name__}")
        self.task = task

    def __call__(self, ctx: Any) -> Any:
        from .utils import _capture_training_modes, _restore_training_modes

        _no_extra_metrics(ctx)
        model = ctx.model
        modes = _capture_training_modes(model.net)
        total_nll, total_correct, total, batches = 0.0, 0, 0, 0
        try:
            model.net.eval()
            with torch.no_grad():
                for batch in ctx.val_loader:
                    batches += 1
                    inputs, targets, loss_mask = self.task.split(batch)
                    valid = self.task.valid(targets, loss_mask)
                    logits = self.task.logits(model, inputs, targets)
                    _, nll, correct, n = self.task.token_sums(logits, targets, valid)
                    total_nll += nll
                    total_correct += correct
                    total += n
        finally:
            _restore_training_modes(modes)
        if batches == 0:
            raise LMTaskError("the validation loader yielded no batches")
        mean = total_nll / total if total else None
        return _record(loss=mean, nll=total_nll, correct=total_correct, n=total, with_perplexity=True)
