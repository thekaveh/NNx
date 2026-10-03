"""Distill a student from stored teacher distributions, offline (FEAT-022).

``examples/10_knowledge_distillation.py`` runs a **live** teacher every step
and softens its **logits** at a temperature. Here the teacher is gone: what
is left is an export of its probabilities, one per sample, with the
schema, teacher revision and the rights under which it may train a
student. This script trains on that export directly:

  1. **Records.** Each ``TeacherRecord`` carries the sample and question
     ids, the ordered candidate ids and probabilities, the teacher and its
     revision, the schema digest, the probability semantics and its
     provenance (``source`` and the **declared** ``training_rights`` — NNx
     records the declaration, it does not verify it). They are written to
     and read back from strict JSONL.
  2. **Dataset.** ``TeacherDataset`` joins the records to the student's
     inputs by sample id and aligns every distribution by candidate id to
     the student's output order — a record listing ``("fox", "cat",
     "dog")`` trains exactly like one listing ``("cat", "dog", "fox")``.
  3. **Train** with ``dataset.objective(alpha=0.7)``: ``0.7 ·
     KL(teacher ‖ student) + 0.3 · CE(student, label)``, both at
     temperature 1 (no logits invented, no ``T²``), validated each epoch by
     ``dataset.eval_step()``.
  4. **Checkpoint and reload** the LAST checkpoint; the objective's
     identity (schema digest, version, alpha, candidate alignment) is
     checkpointed component state, so a resume with another schema is
     refused.
  5. **Report** teacher agreement, imitation loss and labelled quality
     separately, each with its own denominator: agreeing with the teacher
     is not being right — where the teacher is wrong, a faithful student is
     wrong too.

Fully offline, CPU only; nothing is downloaded.

Run:
    python examples/offline_teacher_distillation.py

The bounded ``offline_teacher_distillation_workflow()`` helper is executed
by ``tests/test_examples_smoke.py`` in a temporary working directory, and
the script itself runs end to end there as a subprocess.
"""

from __future__ import annotations

import torch

from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNParams, NNTrainParams
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.nn.enum.optims import Optims
from nnx.nn.params.nn_checkpoint import NNCheckpoint
from nnx.nn.params.nn_optim_params import NNOptimParams
from nnx.paradigms.offline_distillation import (
    TeacherDataset,
    TeacherRecord,
    evaluate_offline,
    read_teacher_records,
    write_teacher_records,
)

CANDIDATES = ("cat", "dog", "fox")
SCHEMA = "sha256:animal-kind-v1"  # e.g. a nnx.decisions question's digest()
PROVENANCE = {
    "source": "animal-teacher export 2026-09 (local file)",
    "training_rights": "declared by the exporter: internal student training permitted",
}


def _records(n: int, seed: int = 0) -> tuple[list[TeacherRecord], dict[str, torch.Tensor]]:
    """``n`` samples: 2-D inputs, a teacher distribution from a hidden
    linear rule, and the true label — which the teacher gets wrong on some
    samples."""
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(n, 2, generator=generator)
    rule = torch.tensor([[2.0, 0.0], [-1.0, 1.5], [-1.0, -1.5]])
    teacher = torch.softmax(inputs @ rule.T, dim=-1).double()
    truth = (inputs @ rule.T + 0.8 * torch.randn(n, 3, generator=generator)).argmax(dim=-1)
    records, by_id = [], {}
    for i in range(n):
        sample_id = f"animal-{i:03d}"
        order = CANDIDATES if i % 2 == 0 else ("fox", "cat", "dog")  # the export's own order varies
        probabilities = [float(teacher[i, CANDIDATES.index(c)]) for c in order]
        probabilities[-1] = 1.0 - sum(probabilities[:-1])  # exact float sum
        records.append(
            TeacherRecord(
                sample_id=sample_id,
                question_id="animal-kind",
                candidates=order,
                probabilities=tuple(probabilities),
                teacher="animal-teacher",
                revision="2026-09-r3",
                schema_digest=SCHEMA,
                semantics="predictive",
                provenance=PROVENANCE,
                label=CANDIDATES[int(truth[i])],
            )
        )
        by_id[sample_id] = inputs[i]
    return records, by_id


def _student(seed: int = 0) -> NNModel:
    torch.manual_seed(seed)
    return NNModel(
        net_params=NNParams(
            input_dim=2, output_dim=len(CANDIDATES), hidden_dims=[16], dropout_prob=0.0, activation=Activations.RELU
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def offline_teacher_distillation_workflow(n: int = 96, epochs: int = 8) -> dict:
    records, inputs = _records(n)
    write_teacher_records("teacher_records.jsonl", records)
    stored = read_teacher_records("teacher_records.jsonl")
    assert stored == records
    split = int(0.75 * n)
    train = TeacherDataset(stored[:split], inputs, candidates=CANDIDATES, schema_digest=SCHEMA)
    val = TeacherDataset(stored[split:], inputs, candidates=CANDIDATES, schema_digest=SCHEMA)

    student = _student()
    objective = train.objective(alpha=0.7)
    run = student.train(
        NNTrainParams(
            n_epochs=epochs,
            train_loader=train.loader(batch_size=16, shuffle=True, seed=0),
            val_loader=val.loader(batch_size=32),
            optim=NNOptimParams(name=Optims.ADAM, max_lr=5e-2, momentum=(0.9, 0.999), weight_decay=0.0),
            seed=0,
            data_id="animal-teacher-export-2026-09",
        ),
        objective=objective,
        eval_step_fn=val.eval_step(),
    )

    last = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert last is not None
    reloaded = NNModel.from_checkpoint(last)
    report = evaluate_offline(reloaded, val)
    assert report == evaluate_offline(student, val)  # the reload scores the same
    return {
        "objective": objective.identity(),
        "epochs": len({idp.epoch_idx for idp in run.idps}),
        "val_imitation_loss": run.idps[-1].val_edp.loss,
        "agreement": report.agreement,
        "imitation_loss": report.imitation_loss,
        "label_accuracy": report.label_accuracy,
        "label_nll": report.label_nll,
        "teacher_label_accuracy": sum(
            max(r.distribution(), key=lambda c: (r.distribution()[c], -CANDIDATES.index(c))) == r.label
            for r in stored[split:]
        )
        / len(stored[split:]),
        "report": report.text(),
    }


def main() -> None:
    result = offline_teacher_distillation_workflow(n=240, epochs=20)
    print(f"objective: {result['objective']}")
    print(result["report"])
    print(f"the teacher itself was right on {result['teacher_label_accuracy']:.4f} of the labelled records")


if __name__ == "__main__":
    main()
