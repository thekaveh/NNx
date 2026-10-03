"""Offline teacher distributions: distill a student from stored, permissioned
teacher probabilities (FEAT-022).

``kd_train_step_factory`` and ``kd_objective`` forward a **live** teacher
every step and compare temperature-softened **logits** (scaled by ``T²``).
Precomputed teacher output is often only a probability distribution — a
decision provider's answers, an export from a model you may no longer run —
with no pre-temperature logits to soften. This module trains on those
records directly, offline:

- :class:`TeacherRecord` — one stored distribution: the sample and question
  ids, the ordered candidate ids and their probabilities (finite,
  nonnegative, summing to 1 within ``1e-6``), the teacher model and
  revision, the schema digest (e.g. a ``nnx.decisions`` question's
  ``digest()``), the probability ``semantics`` and the record's
  ``provenance`` — its ``source`` and the **declared** ``training_rights``
  under which it may train a student. NNx records the declaration; it does
  not verify entitlement. An optional ``label`` (a candidate id) is the
  sample's hard target. :func:`write_teacher_records` /
  :func:`read_teacher_records` keep them as strict JSONL
  (``nnx.teacher-record/1``).
- :class:`TeacherDataset` — records joined to the student's inputs **by
  sample id**, every distribution aligned **by candidate id** to the
  student's output order (``candidates``). A record missing a candidate,
  naming an unknown or duplicate one, another schema digest, a repeated
  sample id or a sample with no input is refused when the dataset is built
  — before any optimizer exists. :meth:`TeacherDataset.loader` yields
  :class:`TeacherBatch`\\ es.
- :class:`OfflineDistillationObjective` (:meth:`TeacherDataset.objective`)
  — an objective for NNx's shared update engine (FEAT-004): ``alpha ·
  KL(teacher ‖ student) + (1 − alpha) · CE(student, label)``, both at
  temperature 1 — no logits are invented and no ``T²`` is applied. The two
  terms share one mask (every row) and one denominator (the microbatch's
  rows), so microbatches accumulate exactly to the full-batch update; a
  partly labelled batch is refused, and ``alpha < 1`` needs every record
  labelled. The teacher's probabilities are detached, and the objective
  never back-propagates, zeroes, steps, scales or schedules. Its identity —
  objective version, schema digest, ``alpha`` and the candidate alignment
  (``identity()``) — is checkpointed as component state
  ``"offline_distillation.objective"``, so a stateful resume with any of
  them changed is refused before the first resumed update; pass the same
  identity to ``ExperimentManifest.for_model(objective=...)`` to record it
  in a run's provenance.
- :func:`evaluate_offline` (and :meth:`TeacherDataset.eval_step` for the
  training loop's validation) reports **teacher agreement** (the student's
  top candidate is the teacher's, over every record), **imitation loss**
  (mean ``KL(teacher ‖ student)`` in nats, over every record) and
  **labelled quality** (accuracy and NLL on the labelled records only),
  each with its own denominator: agreeing with a teacher is not being
  right.

Nothing here reaches the network: records, inputs and the student are
local. Live distillation (``kd_train_step_factory``,
``feature_kd_train_step_factory``, ``born_again_train``, ``kd_objective``)
keeps its logit / temperature contract and refuses these records.
"""

from __future__ import annotations

import math
import numbers
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..components import ComponentSpec
from ..objectives import LossTerm, Objective, ObjectiveContext, ObjectiveResult

if TYPE_CHECKING:
    from ..nn.nn_model import NNModel
    from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint

__all__ = [
    "OBJECTIVE_VERSION",
    "PROBABILITY_TOLERANCE",
    "RECORD_FORMAT",
    "SEMANTICS",
    "DistillationReport",
    "Measure",
    "OfflineDistillationEval",
    "OfflineDistillationObjective",
    "TeacherBatch",
    "TeacherDataset",
    "TeacherRecord",
    "TeacherRecordError",
    "evaluate_offline",
    "read_teacher_records",
    "write_teacher_records",
]

RECORD_FORMAT = "nnx.teacher-record/1"
OBJECTIVE_ID = "offline_distillation"
OBJECTIVE_VERSION = 1
PROBABILITY_TOLERANCE = 1e-6
# What a record's probabilities are: the teacher's own predictive
# distribution, a calibrated one, or empirical frequencies (votes, samples).
SEMANTICS = ("predictive", "calibrated", "empirical")
REQUIRED_PROVENANCE = ("source", "training_rights")
_FIELDS = (
    "sample_id",
    "question_id",
    "candidates",
    "probabilities",
    "teacher",
    "revision",
    "schema_digest",
    "semantics",
    "provenance",
)


class TeacherRecordError(ValueError):
    """A teacher record, dataset or batch that cannot train or be scored."""


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise TeacherRecordError(f"{what} must be a non-empty string, got {value!r}")
    return value


def _ids(values: Any, what: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TeacherRecordError(f"{what} must be a sequence of candidate ids, got {values!r}")
    ids = tuple(values)
    for value in ids:
        _text(value, f"a candidate id in {what}")
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise TeacherRecordError(f"{what} has duplicate candidate ids {duplicates}")
    if len(ids) < 2:
        raise TeacherRecordError(f"{what} needs at least 2 candidates, got {list(ids)}")
    return ids


@dataclass(frozen=True)
class TeacherRecord:
    """One stored teacher distribution over a question's candidates (see
    the module docstring). Every field but ``label`` is required."""

    sample_id: str
    question_id: str
    candidates: tuple[str, ...]
    probabilities: tuple[float, ...]
    teacher: str
    revision: str
    schema_digest: str
    semantics: str
    provenance: Mapping[str, str]
    label: Optional[str] = None
    _nnx_teacher_records = True  # a marker the live-distillation paths refuse

    def __post_init__(self) -> None:
        for name in ("sample_id", "question_id", "teacher", "revision", "schema_digest"):
            _text(getattr(self, name), f"TeacherRecord.{name}")
        candidates = _ids(self.candidates, f"record {self.sample_id!r}'s candidates")
        object.__setattr__(self, "candidates", candidates)
        if isinstance(self.probabilities, (str, bytes)) or not isinstance(self.probabilities, Iterable):
            raise TeacherRecordError(f"record {self.sample_id!r}: probabilities must be a sequence of numbers")
        probabilities = tuple(self.probabilities)
        if len(probabilities) != len(candidates):
            raise TeacherRecordError(
                f"record {self.sample_id!r} needs one probability per candidate: {len(probabilities)} for "
                f"{len(candidates)} candidates"
            )
        for p in probabilities:
            if isinstance(p, bool) or not isinstance(p, numbers.Real) or not math.isfinite(p):
                raise TeacherRecordError(f"record {self.sample_id!r}: probabilities must be finite numbers, got {p!r}")
            if p < 0:
                raise TeacherRecordError(f"record {self.sample_id!r}: probabilities must be nonnegative, got {p!r}")
        probabilities = tuple(float(p) for p in probabilities)
        total = math.fsum(probabilities)
        if abs(total - 1.0) > PROBABILITY_TOLERANCE:
            raise TeacherRecordError(
                f"record {self.sample_id!r}: probabilities must sum to 1 within {PROBABILITY_TOLERANCE}, sum to "
                f"{total!r}"
            )
        object.__setattr__(self, "probabilities", probabilities)
        if self.semantics not in SEMANTICS:
            raise TeacherRecordError(
                f"record {self.sample_id!r}: semantics must be one of {SEMANTICS} (probabilities, never logits), "
                f"got {self.semantics!r}"
            )
        object.__setattr__(self, "semantics", SEMANTICS[SEMANTICS.index(self.semantics)])
        provenance = self.provenance
        if not isinstance(provenance, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str) and value for key, value in provenance.items()
        ):
            raise TeacherRecordError(
                f"record {self.sample_id!r}: provenance must map names to non-empty strings, got {provenance!r}"
            )
        missing = [key for key in REQUIRED_PROVENANCE if key not in provenance]
        if missing:
            raise TeacherRecordError(
                f"record {self.sample_id!r}: provenance lacks {missing} — say where the record came from and "
                "under which declared training rights it may train a student"
            )
        object.__setattr__(self, "provenance", dict(sorted(provenance.items())))
        if self.label is not None and self.label not in candidates:
            raise TeacherRecordError(
                f"record {self.sample_id!r}: label {self.label!r} is not one of its candidates {list(candidates)}"
            )

    def __hash__(self) -> int:
        return hash((self.sample_id, self.question_id, self.candidates, self.probabilities, self.schema_digest))

    def distribution(self) -> dict[str, float]:
        """``{candidate id: probability}``."""
        return dict(zip(self.candidates, self.probabilities, strict=True))

    def state(self) -> dict[str, Any]:
        """The record as strict JSON (``nnx.teacher-record/1``)."""
        return {
            "format": RECORD_FORMAT,
            "sample_id": self.sample_id,
            "question_id": self.question_id,
            "candidates": list(self.candidates),
            "probabilities": list(self.probabilities),
            "teacher": self.teacher,
            "revision": self.revision,
            "schema_digest": self.schema_digest,
            "semantics": self.semantics,
            "provenance": dict(self.provenance),
            "label": self.label,
        }

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> TeacherRecord:
        if not isinstance(state, Mapping) or state.get("format") != RECORD_FORMAT:
            raise TeacherRecordError(f"not a {RECORD_FORMAT} record: {state!r}")
        unknown = sorted(set(state) - {"format", "label", *_FIELDS})
        if unknown:
            raise TeacherRecordError(f"a teacher record has unknown keys {unknown}")
        missing = [name for name in _FIELDS if name not in state]
        if missing:
            raise TeacherRecordError(f"a teacher record lacks {missing}")
        return TeacherRecord(**{name: state[name] for name in _FIELDS}, label=state.get("label"))


def write_teacher_records(path: Union[str, os.PathLike[str]], records: Iterable[TeacherRecord]) -> None:
    """One record per line (strict JSON, sorted keys), written atomically."""
    import json

    from .._artifacts import atomic_write

    lines = [json.dumps(record.state(), sort_keys=True, allow_nan=False) for record in records]
    atomic_write(path, "".join(line + "\n" for line in lines))


def read_teacher_records(path: Union[str, os.PathLike[str]]) -> list[TeacherRecord]:
    """The records of a JSONL file written by :func:`write_teacher_records`;
    a malformed line raises naming ``path:line``."""
    from .._artifacts import parse_json, read_text

    records = []
    for number, line in enumerate(read_text(path, "teacher records", TeacherRecordError).split("\n"), start=1):
        if not line.strip():
            continue
        try:
            records.append(TeacherRecord.from_state(parse_json(line, "teacher records", TeacherRecordError)))
        except TeacherRecordError as error:
            raise TeacherRecordError(f"{os.fspath(path)}:{number}: {error}") from error
    return records


@dataclass(frozen=True)
class TeacherBatch:
    """A microbatch of a :class:`TeacherDataset`: the student's ``inputs``,
    the ``teacher`` probabilities ``(rows, candidates)`` in the dataset's
    candidate order, the hard ``target`` indices (``-1`` where unlabelled),
    the ``labelled`` row mask, the ``sample_ids``, and the dataset's
    ``schema_digest`` and ``candidates`` (the order of the probability
    columns), which an objective or evaluation checks against its own."""

    inputs: torch.Tensor
    teacher: torch.Tensor
    target: torch.Tensor
    labelled: torch.Tensor
    sample_ids: tuple[str, ...]
    schema_digest: str
    candidates: tuple[str, ...]
    _nnx_teacher_records = True


class TeacherDataset(torch.utils.data.Dataset):
    """:class:`TeacherRecord`\\ s joined to the student's ``inputs`` (a
    mapping from sample id to its input tensor) and aligned to the student's
    output order ``candidates`` (see the module docstring).

    Args:
        records: the teacher records — one per sample, all with this
            ``schema_digest`` and exactly these candidates (in any order).
        inputs: ``{sample id: input tensor}``; every record's sample needs
            one (extra inputs are ignored).
        candidates: the student's output order: output ``i`` scores
            ``candidates[i]``.
        schema_digest: the question schema the records answer (e.g. a
            ``nnx.decisions`` question's ``digest()``).
    """

    _nnx_teacher_records = True

    def __init__(
        self,
        records: Iterable[TeacherRecord],
        inputs: Mapping[str, Any],
        *,
        candidates: Sequence[str],
        schema_digest: str,
    ) -> None:
        self.candidates = _ids(candidates, "the dataset's candidates")
        self.schema_digest = _text(schema_digest, "schema_digest")
        if not isinstance(inputs, Mapping):
            raise TeacherRecordError(f"inputs must map sample ids to input tensors, got {type(inputs).__name__}")
        records = tuple(records)
        if not records:
            raise TeacherRecordError("a teacher dataset needs at least one record")
        index = {candidate: i for i, candidate in enumerate(self.candidates)}
        seen: set[str] = set()
        rows: list[list[float]] = []
        targets: list[int] = []
        tensors: list[torch.Tensor] = []
        for record in records:
            if not isinstance(record, TeacherRecord):
                raise TeacherRecordError(f"records must be TeacherRecord, got {type(record).__name__}")
            where = f"record {record.sample_id!r}"
            if record.sample_id in seen:
                raise TeacherRecordError(f"{where}: duplicate sample id — one teacher distribution per sample")
            seen.add(record.sample_id)
            if record.schema_digest != self.schema_digest:
                raise TeacherRecordError(
                    f"{where} answers schema digest {record.schema_digest!r}, not this dataset's {self.schema_digest!r}"
                )
            unknown = sorted(set(record.candidates) - set(self.candidates))
            if unknown:
                raise TeacherRecordError(
                    f"{where} names unknown candidates {unknown} (the student scores {list(self.candidates)})"
                )
            missing = [candidate for candidate in self.candidates if candidate not in record.candidates]
            if missing:
                raise TeacherRecordError(f"{where} lacks candidates {missing}")
            if record.sample_id not in inputs:
                raise TeacherRecordError(f"{where} has no input (no {record.sample_id!r} in inputs)")
            distribution = record.distribution()
            rows.append([distribution[candidate] for candidate in self.candidates])  # aligned by id
            targets.append(-1 if record.label is None else index[record.label])
            tensors.append(torch.as_tensor(inputs[record.sample_id]))
        questions = sorted({record.question_id for record in records})
        if len(questions) > 1:
            raise TeacherRecordError(f"a teacher dataset answers one question; its records answer {questions}")
        shapes = sorted({tuple(tensor.shape) for tensor in tensors})
        if len(shapes) > 1:
            raise TeacherRecordError(f"every input must have one shape to batch, got shapes {shapes}")
        default = torch.get_default_dtype()
        tensors = [tensor.to(default) if tensor.is_floating_point() else tensor for tensor in tensors]
        self.records = records
        self._teacher = torch.tensor(rows, dtype=torch.float32)
        self._target = torch.tensor(targets, dtype=torch.long)
        self._inputs = tensors

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        return self._inputs[i], self._teacher[i], self._target[i], self.records[i].sample_id

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(record.sample_id for record in self.records)

    @property
    def labelled(self) -> str:
        """``"all"``, ``"none"`` or ``"partial"``: which records carry a label."""
        count = int((self._target >= 0).sum())
        return "all" if count == len(self) else "none" if count == 0 else "partial"

    def collate(self, items: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]) -> TeacherBatch:
        inputs, teacher, target, ids = zip(*items, strict=True)
        targets = torch.stack(target)
        return TeacherBatch(
            inputs=torch.stack(inputs),
            teacher=torch.stack(teacher),
            target=targets,
            labelled=targets >= 0,
            sample_ids=tuple(ids),
            schema_digest=self.schema_digest,
            candidates=self.candidates,
        )

    def loader(self, batch_size: int, *, shuffle: bool = False, seed: Optional[int] = None) -> DataLoader:
        """A ``DataLoader`` of :class:`TeacherBatch`\\ es; ``shuffle`` draws
        its order from ``seed`` (a generator a stateful resume restores)."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise TeacherRecordError(f"batch_size must be a positive integer, got {batch_size!r}")
        generator = None
        if shuffle:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TeacherRecordError(f"a shuffling loader needs an integer seed, got {seed!r}")
            generator = torch.Generator().manual_seed(seed)
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, generator=generator, collate_fn=self.collate)

    def objective(self, *, alpha: float = 0.5, nonfinite: str = "fail") -> OfflineDistillationObjective:
        """This dataset's :class:`OfflineDistillationObjective` — refused
        here, before any optimizer exists, for a partly labelled dataset
        (any ``alpha``: its batches would mix labelled and unlabelled rows)
        and, with ``alpha < 1`` (a hard term), for one without labels."""
        objective = OfflineDistillationObjective(
            schema_digest=self.schema_digest, candidates=self.candidates, alpha=alpha, nonfinite=nonfinite
        )
        if self.labelled == "partial":
            raise TeacherRecordError(
                "this dataset is partly labelled: its batches would mix labelled and unlabelled rows, which share "
                "one mask and denominator; label every record or none (drop the labels for alpha=1.0)"
            )
        if objective.alpha < 1.0 and self.labelled != "all":
            raise TeacherRecordError(
                f"alpha={objective.alpha} adds the hard term, so every record needs a label; this dataset's "
                f"labels: {self.labelled} (use alpha=1.0 for teacher distributions alone)"
            )
        return objective

    def eval_step(self) -> OfflineDistillationEval:
        """An ``eval_step_fn`` for ``NNModel.train`` over a validation loader
        of this dataset's kind (see :class:`OfflineDistillationEval`)."""
        return OfflineDistillationEval(schema_digest=self.schema_digest, candidates=self.candidates)


def _check_alpha(alpha: Any) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real):
        raise TypeError(f"alpha must be a number in [0, 1], got {alpha!r}")
    value = float(alpha)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise TeacherRecordError(f"alpha must be finite and in [0, 1], got {alpha!r}")
    return value


def _teacher_batch(batch: Any, schema_digest: str, candidates: tuple[str, ...], who: str) -> TeacherBatch:
    if not isinstance(batch, TeacherBatch):
        raise TeacherRecordError(
            f"{who} trains on TeacherBatch microbatches (TeacherDataset.loader), got {type(batch).__name__}"
        )
    if batch.schema_digest != schema_digest:
        raise TeacherRecordError(
            f"{who}: the batch answers schema digest {batch.schema_digest!r}, not {schema_digest!r}"
        )
    if tuple(batch.candidates) != candidates:
        raise TeacherRecordError(
            f"{who}: the batch's probability columns are candidates {list(batch.candidates)}, not "
            f"{list(candidates)} — build the loader and the objective from datasets with one candidate order"
        )
    if batch.teacher.ndim != 2 or batch.teacher.shape[1] != len(candidates):
        raise TeacherRecordError(
            f"{who}: the batch's teacher probabilities have shape {tuple(batch.teacher.shape)}, not "
            f"(rows, {len(candidates)})"
        )
    return batch


def _log_probabilities(model: Any, batch: TeacherBatch, ctx: Optional[ObjectiveContext], who: str) -> torch.Tensor:
    logits = model._net_forward((batch.inputs.to(model.device),), {})
    if ctx is not None:
        logits = ctx.full_precision(logits)
    rows, classes = batch.teacher.shape
    if logits.ndim != 2 or tuple(logits.shape) != (rows, classes):
        raise TeacherRecordError(
            f"{who}: the student gives outputs of shape {tuple(logits.shape)} for {rows} rows and {classes} "
            "candidates; its head must score each candidate in the dataset's order"
        )
    return F.log_softmax(logits.float(), dim=-1)


def _kl_rows(q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """``KL(q ‖ p)`` per row, in nats; a zero teacher probability adds 0."""
    positive = q > 0
    log_q = torch.log(torch.where(positive, q, torch.ones_like(q)))
    return torch.where(positive, q * (log_q - log_p), torch.zeros_like(q)).sum(dim=-1)


class OfflineDistillationObjective(Objective):
    """``alpha · KL(teacher ‖ student) + (1 − alpha) · CE(student, label)``
    over stored teacher distributions (see the module docstring). Build it
    with :meth:`TeacherDataset.objective`.

    Terms (both normalized by the microbatch's rows — one shared mask and
    denominator): ``"teacher_kl"`` (weight ``alpha``, omitted when
    ``alpha == 0``) and ``"hard_ce"`` (weight ``1 − alpha``, omitted when
    ``alpha == 1``)."""

    def __init__(
        self,
        *,
        schema_digest: str,
        candidates: Sequence[str],
        alpha: float = 0.5,
        nonfinite: str = "fail",
    ) -> None:
        super().__init__(nonfinite=nonfinite)
        self.schema_digest = _text(schema_digest, "schema_digest")
        self.candidates = _ids(candidates, "the objective's candidates")
        self.alpha = _check_alpha(alpha)

    def identity(self) -> dict[str, Any]:
        """The objective's identity: id, version, schema digest, ``alpha``
        and the candidate alignment — checkpointed as component state and
        fit for ``ExperimentManifest.for_model(objective=...)``."""
        return {
            "id": OBJECTIVE_ID,
            "version": OBJECTIVE_VERSION,
            "schema_digest": self.schema_digest,
            "alpha": self.alpha,
            "adapter": {"kind": "candidate_ids", "candidates": list(self.candidates)},
        }

    def __call__(self, ctx: ObjectiveContext) -> ObjectiveResult:
        from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint

        who = "the offline distillation objective"
        batch = _teacher_batch(ctx.batch, self.schema_digest, self.candidates, who)
        rows = int(batch.teacher.shape[0])
        labelled = int(batch.labelled.sum())
        if 0 < labelled < rows:
            raise TeacherRecordError(
                f"{who}: a partly labelled batch ({labelled} of {rows} rows) — the soft and hard terms share one "
                "mask and denominator, so a batch is labelled throughout or not at all"
            )
        if self.alpha < 1.0 and labelled == 0:
            raise TeacherRecordError(f"{who}: alpha={self.alpha} adds the hard term, so every record needs a label")
        model = ctx.model
        model.net.train()
        log_p = _log_probabilities(model, batch, ctx, who)
        q = batch.teacher.detach().to(device=log_p.device, dtype=torch.float32)
        terms = []
        kl = _kl_rows(q, log_p)
        if self.alpha > 0.0:
            terms.append(LossTerm("teacher_kl", kl.sum(), rows, "mean", self.alpha))
        if self.alpha < 1.0:
            ce = F.nll_loss(log_p, batch.target.to(log_p.device), reduction="sum")
            terms.append(LossTerm("hard_ce", ce, rows, "mean", 1.0 - self.alpha))
        result = ObjectiveResult(terms)
        with torch.no_grad():
            agreement = float((log_p.argmax(dim=-1) == q.argmax(dim=-1)).sum()) / rows
            metrics = {"teacher_kl": float(kl.sum()) / rows, "teacher_agreement": agreement}
        record = NNEvaluationDataPoint(loss=result.loss(), metrics=metrics)
        return ObjectiveResult(terms, record)

    def component_spec(self) -> ComponentSpec:
        return ComponentSpec("offline_distillation.objective", version=OBJECTIVE_VERSION)

    def component_state(self) -> dict[str, Any]:
        return self.identity()

    def check_component_state(self, state: Mapping[str, Any], *, version: int) -> list[str]:
        if not isinstance(state, Mapping):
            return [f"the checkpoint's offline distillation state is not a mapping: {state!r}"]
        mine = self.identity()
        names = {
            "id": "objective id",
            "version": "objective version",
            "schema_digest": "schema digest",
            "alpha": "alpha",
            "adapter": "candidate alignment",
        }
        problems = [
            f"the {label} changed since the checkpoint: {state.get(key)!r} -> {mine[key]!r}"
            for key, label in names.items()
            if state.get(key) != mine[key]
        ]
        unknown = sorted(set(state) - set(names))
        if unknown:
            problems.append(f"the checkpoint's offline distillation state has unknown keys {unknown}")
        return problems

    def load_component_state(self, state: Mapping[str, Any], *, version: int) -> None:
        """Nothing to restore: the checked identity is the whole state."""


@dataclass(frozen=True)
class Measure:
    """One reported quantity with its denominator, or ``value=None`` and the
    ``reason`` it is unavailable."""

    value: Optional[float]
    denominator: int
    reason: Optional[str] = None

    def state(self) -> dict[str, Any]:
        return {"value": self.value, "denominator": self.denominator, "reason": self.reason}


@dataclass(frozen=True)
class DistillationReport:
    """A student scored against stored teacher distributions.

    - ``agreement``: records whose student top candidate is the teacher's
      (ties go to the first candidate in the dataset's order), over every
      record;
    - ``imitation_loss``: mean ``KL(teacher ‖ student)`` in nats, over every
      record;
    - ``label_accuracy`` / ``label_nll``: quality on the labelled records
      only — unavailable without any.
    """

    agreement: Measure
    imitation_loss: Measure
    label_accuracy: Measure
    label_nll: Measure
    candidates: tuple[str, ...] = field(default=())
    schema_digest: str = ""

    def state(self) -> dict[str, Any]:
        return {
            "agreement": self.agreement.state(),
            "imitation_loss": self.imitation_loss.state(),
            "label_accuracy": self.label_accuracy.state(),
            "label_nll": self.label_nll.state(),
            "candidates": list(self.candidates),
            "schema_digest": self.schema_digest,
        }

    def text(self) -> str:
        def show(measure: Measure, unit: str) -> str:
            if measure.value is None:
                return f"unavailable ({measure.reason})"
            return f"{measure.value:.4f} ({measure.denominator} {unit})"

        return "\n".join(
            [
                f"teacher agreement {show(self.agreement, 'records')}",
                f"imitation loss (KL, nats) {show(self.imitation_loss, 'records')}",
                f"label accuracy {show(self.label_accuracy, 'labelled records')}",
                f"label NLL (nats) {show(self.label_nll, 'labelled records')}",
            ]
        )


class _Tally:
    def __init__(self) -> None:
        self.rows = 0
        self.agree = 0
        self.kl = 0.0
        self.labelled = 0
        self.correct = 0
        self.nll = 0.0

    def update(self, log_p: torch.Tensor, batch: TeacherBatch) -> None:
        q = batch.teacher.detach().to(device=log_p.device, dtype=torch.float32)
        top = log_p.argmax(dim=-1)
        self.rows += int(q.shape[0])
        self.agree += int((top == q.argmax(dim=-1)).sum())
        self.kl += float(_kl_rows(q, log_p).double().sum())
        mask = batch.labelled.to(log_p.device)
        if bool(mask.any()):
            target = batch.target.to(log_p.device)[mask]
            self.labelled += int(mask.sum())
            self.correct += int((top[mask] == target).sum())
            self.nll += float(F.nll_loss(log_p[mask].double(), target, reduction="sum"))

    def report(self, candidates: tuple[str, ...], schema_digest: str) -> DistillationReport:
        if self.rows == 0:
            raise TeacherRecordError("there is nothing to evaluate: the loader yielded no rows")
        none = "no labelled records"
        return DistillationReport(
            agreement=Measure(self.agree / self.rows, self.rows),
            imitation_loss=Measure(self.kl / self.rows, self.rows),
            label_accuracy=Measure(self.correct / self.labelled, self.labelled)
            if self.labelled
            else Measure(None, 0, none),
            label_nll=Measure(self.nll / self.labelled, self.labelled) if self.labelled else Measure(None, 0, none),
            candidates=candidates,
            schema_digest=schema_digest,
        )


def _score(
    model: NNModel, batches: Iterable[Any], schema_digest: str, candidates: tuple[str, ...], who: str
) -> DistillationReport:
    from ..utils import _capture_training_modes, _restore_training_modes

    tally = _Tally()
    modes = _capture_training_modes(model.net)
    try:
        model.net.eval()
        with torch.no_grad():
            for batch in batches:
                batch = _teacher_batch(batch, schema_digest, candidates, who)
                tally.update(_log_probabilities(model, batch, None, who), batch)
    finally:
        _restore_training_modes(modes)
    return tally.report(candidates, schema_digest)


def evaluate_offline(
    model: NNModel, data: Union[TeacherDataset, Iterable[TeacherBatch]], *, batch_size: int = 256
) -> DistillationReport:
    """Score ``model`` against a :class:`TeacherDataset` (or a loader of its
    batches): teacher agreement, imitation loss and labelled quality, each
    with its own denominator (see :class:`DistillationReport`). Runs in eval
    mode under ``no_grad`` and restores the model's modes."""
    if isinstance(data, TeacherDataset):
        return _score(model, data.loader(batch_size), data.schema_digest, data.candidates, "evaluate_offline")
    dataset = getattr(data, "dataset", None)
    if not isinstance(dataset, TeacherDataset):
        raise TeacherRecordError("evaluate_offline needs a TeacherDataset or a loader over one")
    return _score(model, data, dataset.schema_digest, dataset.candidates, "evaluate_offline")


class OfflineDistillationEval:
    """An ``eval_step_fn`` (``NNModel.train(eval_step_fn=...)``) over a
    validation loader of :class:`TeacherBatch`\\ es: its ``metrics`` are
    ``teacher_kl``, ``teacher_agreement`` and — with labelled records —
    ``label_accuracy`` and ``label_nll``.

    The record's ``loss`` is the **imitation loss** (``teacher_kl``) whatever
    the objective's ``alpha``: it measures agreement with the teacher, not
    quality. A scheduler, early stopping or BEST selection that should
    follow labelled quality must monitor ``label_accuracy`` or ``label_nll``
    by name (``MonitorSpec``). Use :func:`evaluate_offline` for every
    denominator."""

    def __init__(self, *, schema_digest: str, candidates: Sequence[str]) -> None:
        self.schema_digest = _text(schema_digest, "schema_digest")
        self.candidates = _ids(candidates, "the evaluation's candidates")

    def __call__(self, ctx: Any) -> NNEvaluationDataPoint:
        from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint

        report = _score(ctx.model, ctx.val_loader, self.schema_digest, self.candidates, "the offline evaluation")
        measures = {
            "teacher_kl": report.imitation_loss,
            "teacher_agreement": report.agreement,
            "label_accuracy": report.label_accuracy,
            "label_nll": report.label_nll,
        }
        metrics = {name: m.value for name, m in measures.items() if m.value is not None}  # labelled ones may be absent
        return NNEvaluationDataPoint(loss=report.imitation_loss.value, metrics=metrics)


def _is_teacher_records(value: Any) -> bool:
    """Whether ``value`` is offline teacher data — a record, a dataset, a
    batch, a sequence of records or a loader over a dataset — which the
    live-teacher paths refuse."""
    if getattr(value, "_nnx_teacher_records", False) is True:
        return True
    if getattr(getattr(value, "dataset", None), "_nnx_teacher_records", False) is True:
        return True
    if isinstance(value, (list, tuple)) and value:
        return all(getattr(item, "_nnx_teacher_records", False) is True for item in value)
    return False
