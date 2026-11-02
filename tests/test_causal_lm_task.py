"""FEAT-034: a causal language-model task with declared alignment, masks and
valid-token normalization.

The task declares alignment, vocabulary, ignore and padding ids, a loss mask,
smoothing and its version, and rejects bad batches before any forward pass.
One set of valid positions drives the objective's denominator, the NLL and
the token accuracy; an all-masked window steps nothing; epoch NLL is total
valid-token NLL over the valid-token count, independent of batching and
padding, and perplexity is exp(NLL) or +inf. The configuration is
checkpointed component state: a resume with another configuration is refused
before the first resumed update.
"""

from __future__ import annotations

import math
from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Devices,
    Losses,
    MetricSpec,
    MonitorSpec,
    Nets,
    NNCheckpoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNRun,
    NNTrainParams,
    NNTransformerParams,
)
from nnx.components import ComponentRestoreError
from nnx.lm_tasks import CausalLMTask, LMTaskError, perplexity
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.objectives import ObjectiveContext

V = 12


@pytest.fixture(autouse=True)
def _workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")


def _model(seed: int = 0, precision: Optional[str] = None) -> NNModel:
    from nnx.precision import PrecisionPolicy

    torch.manual_seed(seed)
    net = NNTransformerParams(
        input_dim=V,
        output_dim=V,
        dropout_prob=0.0,
        vocab_size=V,
        n_layers=1,
        n_heads=2,
        d_model=16,
        ffn_mult=2,
        max_seq_len=8,
    )
    return NNModel(
        net_params=net,
        params=NNModelParams(
            net=Nets.TRANSFORMER,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
            precision=None if precision is None else PrecisionPolicy(precision),
        ),
    )


def _ids(n: int = 8, length: int = 6, seed: int = 0) -> torch.Tensor:
    return torch.randint(0, V, (n, length), generator=torch.Generator().manual_seed(seed))


class _Ctx:
    def __init__(self, model, loader):
        self.model, self.val_loader = model, loader


# --- AC1: declaration and validation -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"vocab_size": 1}, "at least 2"),
        ({"vocab_size": V, "alignment": "shifted"}, "alignment"),
        ({"vocab_size": V, "ignore_id": 3}, "real token id"),
        ({"vocab_size": V, "pad_id": V}, "outside the vocabulary"),
        ({"vocab_size": V, "smoothing": 1.0}, "smoothing"),
        ({"vocab_size": V, "vocab_axis": 1}, "vocab_axis"),
        ({"vocab_size": V, "version": 2}, "version"),
    ],
)
def test_the_task_declares_and_validates_its_configuration(fields, match):
    with pytest.raises(LMTaskError, match=match):
        CausalLMTask(**fields)
    task = CausalLMTask(vocab_size=V, pad_id=0, smoothing=0.1, tokenizer="sha256:abc")
    assert CausalLMTask.from_state(task.state()) == task


def test_batches_are_checked_and_aligned_exactly_once():
    shift = CausalLMTask(vocab_size=V)
    ids = _ids(2, 5)
    inputs, targets, mask = shift.split(ids)
    assert torch.equal(inputs, ids[:, :-1]) and torch.equal(targets, ids[:, 1:]) and mask is None
    pre = CausalLMTask(vocab_size=V, alignment="pre_shifted")
    x, y = ids[:, :-1], ids[:, 1:]
    inputs, targets, _ = pre.split((x, y))
    assert torch.equal(inputs, x) and torch.equal(targets, y)  # never shifted a second time
    with pytest.raises(LMTaskError, match="pre_shifted"):
        shift.split((x, y, torch.ones_like(y)))
    with pytest.raises(LMTaskError, match="integer token ids"):
        shift.split(ids.float())
    with pytest.raises(LMTaskError, match=r"\(batch, length\)"):
        shift.split(ids[0])
    with pytest.raises(LMTaskError, match="at least 2 tokens"):
        shift.split(ids[:, :1])
    with pytest.raises(LMTaskError, match="differ in shape"):
        pre.split((x, y[:, :-1]))
    bad = y.clone()
    bad[0, 0] = V
    with pytest.raises(LMTaskError, match="non-ignored target ids"):
        pre.split((x, bad))
    ignored = y.clone()
    ignored[0, 0] = -100
    pre.split((x, ignored))  # the ignore id is fine as a target
    with pytest.raises(LMTaskError, match="input ids"):
        pre.split((ignored, y))
    with pytest.raises(LMTaskError, match="loss mask"):
        pre.split((x, y, torch.ones(2, 3)))


# --- AC2: one set of valid positions -----------------------------------------------------------------------


def test_ignore_padding_and_the_loss_mask_select_one_set_of_positions():
    task = CausalLMTask(vocab_size=V, alignment="pre_shifted", pad_id=0)
    model = _model()
    x = _ids(2, 5, seed=1)
    y = _ids(2, 5, seed=2)
    y[0, 0] = -100
    y[1, 1] = 0  # padding
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[1, 4] = False
    result = task.objective()(ObjectiveContext(model=model, batch=(x, y, mask), epoch_idx=0, batch_idx=0))
    valid = (y != -100) & (y != 0) & mask
    n = int(valid.sum())
    (term,) = result.terms
    assert term.denominator == n == result.record.count
    with torch.no_grad():
        logits = model.net(x)
    reference = F.cross_entropy(logits[valid], y[valid], reduction="sum")
    assert float(term.numerator.detach()) == pytest.approx(float(reference), rel=1e-5)
    assert result.record.metrics["nll"] == pytest.approx(float(reference) / n, rel=1e-5)
    accuracy = float((logits[valid].argmax(-1) == y[valid]).float().mean())
    assert result.record.metrics["token_accuracy"] == pytest.approx(accuracy)
    assert result.record.f1 is None and result.record.accuracy is None  # no classification placeholders


def test_an_all_masked_window_steps_nothing_and_evaluation_is_unavailable(monkeypatch):
    task = CausalLMTask(vocab_size=V, alignment="pre_shifted")
    x = _ids(4, 5)
    y = torch.full_like(x, -100)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    model = _model()
    before = {k: v.clone() for k, v in model.net.state_dict().items()}
    steps = []
    original = torch.optim.SGD.step

    def counted(self, *args, **kwargs):
        steps.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.SGD, "step", counted)
    with pytest.warns(RuntimeWarning, match="skipping ReduceLROnPlateau step"):  # nor is the scheduler stepped
        run = model.train(
            params=NNTrainParams(
                n_epochs=1,
                train_loader=loader,
                val_loader=loader,
                optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
            ),
            objective=task.objective(),
            eval_step_fn=task.eval_step(),
        )
    assert steps == [] and all(torch.equal(before[k], v) for k, v in model.net.state_dict().items())
    assert all(p.grad is None for p in model.net.parameters())  # cleared
    val = run.idps[-1].val_edp
    assert (val.status, val.count, val.loss, dict(val.metrics)) == ("empty", 0, None, {})
    reloaded = NNRun.load(run.id).idps[-1].val_edp
    assert (reloaded.status, reloaded.count, reloaded.loss) == ("empty", 0, None)  # unavailability survives decoding


# --- AC3: epoch NLL and perplexity --------------------------------------------------------------------------


def test_epoch_nll_is_total_over_valid_tokens_whatever_the_batching_or_padding():
    task = CausalLMTask(vocab_size=V, alignment="pre_shifted", pad_id=0)
    model = _model()
    x = _ids(6, 5, seed=3).clamp(min=1)
    y = _ids(6, 5, seed=4).clamp(min=1)
    y[0, 3:] = 0  # an uneven row: two padded positions
    # The invariance under test is the task's token-weighted aggregation, not
    # the batch-invariance of float32 CPU kernels (which round differently per
    # batch shape on some platforms): compare the batchings in float64, where
    # kernel rounding (~1e-15) sits far below the bound.
    exact = _model()
    exact.net.load_state_dict(model.net.state_dict())
    exact.net.double()
    values = []
    for batch_size in (1, 2, 4, 6):
        record = task.eval_step()(_Ctx(exact, DataLoader(TensorDataset(x, y), batch_size=batch_size)))
        values.append((record.loss, record.count))
    assert len({count for _, count in values}) == 1
    assert max(v for v, _ in values) - min(v for v, _ in values) < 1e-9
    # A wrong aggregation, the mean of per-batch means, misses the bound: the
    # uneven row puts 18 valid tokens in one batch of 4 and 10 in the other.
    with torch.no_grad():
        means = [
            F.cross_entropy(exact.net(xb)[yb != 0], yb[yb != 0], reduction="mean").item()
            for xb, yb in DataLoader(TensorDataset(x, y), batch_size=4)
        ]
    assert abs(sum(means) / len(means) - values[0][0]) > 1e-6
    # Extra padding columns change nothing.
    padded_x = torch.cat([x, torch.ones(6, 2, dtype=torch.long)], dim=1)
    padded_y = torch.cat([y, torch.zeros(6, 2, dtype=torch.long)], dim=1)
    padded_model = _model()
    padded_model.net.load_state_dict(model.net.state_dict())
    with torch.no_grad():
        valid = y != 0
        reference = F.cross_entropy(model.net(x)[valid].double(), y[valid], reduction="mean")
    padded = task.eval_step()(_Ctx(padded_model, DataLoader(TensorDataset(padded_x, padded_y), batch_size=4)))
    assert padded.loss == pytest.approx(float(reference), rel=1e-6) and padded.count == values[0][1]
    assert padded.metrics["perplexity"] == pytest.approx(math.exp(padded.loss))
    assert perplexity(1000.0) == math.inf and perplexity(None) is None


# --- AC4: unsmoothed reports, the named monitor ---------------------------------------------------------------


def test_reported_nll_stays_unsmoothed_while_the_objective_smooths():
    task = CausalLMTask(vocab_size=V, alignment="pre_shifted", smoothing=0.2)
    model = _model()
    x, y = _ids(2, 5, seed=5), _ids(2, 5, seed=6)
    result = task.objective()(ObjectiveContext(model=model, batch=(x, y), epoch_idx=0, batch_idx=0))
    with torch.no_grad():
        logits = model.net(x)
    smoothed = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), label_smoothing=0.2)
    plain = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    assert result.terms[0].value == pytest.approx(float(smoothed), rel=1e-5)
    assert result.record.metrics["nll"] == pytest.approx(float(plain), rel=1e-5)
    evaluated = task.eval_step()(_Ctx(model, DataLoader(TensorDataset(x, y), batch_size=1)))
    assert evaluated.loss == pytest.approx(float(plain), rel=1e-5)


# --- AC5: train / evaluate / resume ----------------------------------------------------------------------------


def _params(epochs: int = 2, **kwargs) -> NNTrainParams:
    x = _ids(8, 6, seed=7).clamp(min=1)  # no padding id: every target position is valid
    return NNTrainParams(
        n_epochs=epochs,
        train_loader=DataLoader(TensorDataset(x), batch_size=4),
        val_loader=DataLoader(TensorDataset(x[:4]), batch_size=2),
        optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
        metrics=[MetricSpec("nll")],
        monitor=MonitorSpec("nll"),
        **kwargs,
    )


def test_train_evaluate_and_resume_carry_the_task():
    task = CausalLMTask(vocab_size=V, pad_id=0, tokenizer="sha256:demo-tokenizer")
    run = _model().train(params=_params(), objective=task.objective(), eval_step_fn=task.eval_step())
    reloaded = NNRun.load(run.id)
    val = [idp.val_edp for idp in reloaded.idps if idp.val_edp is not None]
    assert len(val) == 2 and all(set(edp.metrics) == {"nll", "perplexity", "token_accuracy"} for edp in val)
    assert all(edp.kind == "causal_lm" and edp.f1 is None and edp.precision is None for edp in val)
    assert [edp.loss for edp in val] == pytest.approx([edp.metrics["nll"] for edp in val])
    assert str(reloaded) and reloaded._repr_html_()  # rendering needs no classification field
    selections = [idp.selection for idp in reloaded.idps if idp.selection is not None]
    assert selections and selections[0].monitor.metric == "nll" and selections[0].improved  # BEST by val NLL
    best = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert best is not None and best.idp.val_edp.metrics["nll"] == pytest.approx(min(edp.metrics["nll"] for edp in val))
    state = NNCheckpoint.load_training_state(run=run.id, type=Checkpoints.LAST)["components"]["lm.causal_task"]["state"]
    assert state["task"] == task.state() and state["task"]["tokenizer"] == "sha256:demo-tokenizer"
    assert state["tokens"] == 2 * 8 * 5  # every target position is valid: 8 rows x 5 targets, 2 epochs

    resumed = task.objective()
    child = _model().train(
        params=_params(1, resume_from_run_id=run.id), objective=resumed, eval_step_fn=task.eval_step()
    )
    assert resumed.tokens == 3 * 8 * 5 and child.resume_status.mode == "stateful"
    changed = CausalLMTask(vocab_size=V, pad_id=0, smoothing=0.1, tokenizer="sha256:demo-tokenizer")
    model = _model()
    before = {k: v.clone() for k, v in model.net.state_dict().items()}
    with pytest.raises(ComponentRestoreError, match="causal-LM task changed"):
        model.train(
            params=_params(1, resume_from_run_id=run.id, overwrite_existing=True),
            objective=changed.objective(),
            eval_step_fn=changed.eval_step(),
            salt="changed",
        )
    assert all(torch.equal(before[k], v) for k, v in model.net.state_dict().items())  # refused before any update


def test_accumulation_normalizes_by_the_windows_valid_tokens():
    task = CausalLMTask(vocab_size=V, alignment="pre_shifted")
    x, y = _ids(3, 5, seed=8), _ids(3, 5, seed=9)
    y[0, :4] = -100  # microbatch 1 has 1 + 5 valid tokens, microbatch 2 has 5
    reference = _model()
    accumulated = _model()
    accumulated.net.load_state_dict(reference.net.state_dict())
    optimizer = torch.optim.SGD(reference.net.parameters(), lr=0.1)
    valid = y != -100
    loss = F.cross_entropy(reference.net(x)[valid], y[valid])  # one full-batch mean over all valid tokens
    loss.backward()
    optimizer.step()
    accumulated.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=DataLoader(TensorDataset(x, y), batch_size=2),
            optim=NNOptimParams.builder().sgd(max_lr=0.1, momentum=0.0).accumulate_grad(2).build(),
        ),
        objective=task.objective(),
    )
    for (name, want), got in zip(
        reference.net.state_dict().items(), accumulated.net.state_dict().values(), strict=True
    ):
        assert torch.allclose(want, got, atol=1e-6), name


def test_an_all_masked_microbatch_in_a_trained_window_records_no_loss():
    """#504: an empty task record keeps its None loss inside an accumulated
    window after a microbatch that trained (never the window's running loss)."""
    task = CausalLMTask(vocab_size=V, alignment="pre_shifted")
    x, y = _ids(4, 5, seed=10), _ids(4, 5, seed=11)
    y[2:] = -100  # microbatch 2: every position masked
    run = _model().train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=DataLoader(TensorDataset(x, y), batch_size=2),
            optim=NNOptimParams.builder().sgd(max_lr=0.1).accumulate_grad(2).build(),
            save_phase_checkpoints=False,
        ),
        objective=task.objective(),
    )
    (trained, empty) = (idp.train_edp for idp in run.idps)
    assert (trained.status, trained.count) == ("ok", 10) and trained.loss is not None
    assert (empty.status, empty.count, empty.loss, dict(empty.metrics)) == ("empty", 0, None, {})


# --- review hardening -------------------------------------------------------------------------------------


class _MaskedSecondTime:
    """A validation loader whose second pass is entirely padding."""

    def __init__(self, ids):
        self.ids, self.passes = ids, 0

    def __iter__(self):
        self.passes += 1
        ids = self.ids if self.passes == 1 else torch.zeros_like(self.ids)
        return iter([ids[:2], ids[2:]])


def test_without_a_monitor_an_unavailable_validation_epoch_never_becomes_best():
    from nnx import NNSchedulerParams
    from nnx._metrics import _resolve_metric
    from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint

    empty = NNEvaluationDataPoint(kind="causal_lm", count=0, status="empty")
    assert _resolve_metric(empty, NNEvaluationDataPoint(loss=0.5)) is None  # never the training loss
    task = CausalLMTask(vocab_size=V, pad_id=0)
    x = _ids(8, 6, seed=7).clamp(min=1)
    params = NNTrainParams(
        n_epochs=2,
        train_loader=DataLoader(TensorDataset(x), batch_size=4),
        val_loader=_MaskedSecondTime(x[:4]),
        optim=NNOptimParams.builder().sgd(max_lr=0.5).build(),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=0, cooldown=0, threshold=1e-3),
    )
    with pytest.warns(RuntimeWarning, match="validation record is unavailable"):
        run = _model().train(params=params, objective=task.objective(), eval_step_fn=task.eval_step())
    best = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert best is not None and best.idp.epoch_idx == 0 and best.idp.val_edp.status == "ok"


def test_numpy_integers_give_a_weights_only_checkpoint_state():
    import io

    import numpy as np

    task = CausalLMTask(vocab_size=np.int64(V), pad_id=np.int64(0), ignore_id=np.int32(-100))
    assert all(type(task.state()[k]) is int for k in ("vocab_size", "pad_id", "ignore_id", "version", "vocab_axis"))
    buffer = io.BytesIO()
    torch.save(task.state(), buffer)
    buffer.seek(0)
    assert CausalLMTask.from_state(torch.load(buffer, weights_only=True)) == task
    run = _model().train(params=_params(1), objective=task.objective(), eval_step_fn=task.eval_step())
    state = NNCheckpoint.load_training_state(run=run.id, type=Checkpoints.LAST)["components"]["lm.causal_task"]
    assert state["state"]["task"] == task.state()


def test_batch_forms_are_read_one_way_and_huggingface_labels_shift_with_the_ids():
    shifted = CausalLMTask(vocab_size=V)
    ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, 2, -100, 4]])
    inputs, targets, mask = shifted.split({"input_ids": ids, "labels": labels, "attention_mask": torch.ones(1, 4)})
    assert torch.equal(inputs, ids[:, :-1]) and torch.equal(targets, labels[:, 1:]) and mask is None
    assert shifted.valid(targets, mask).tolist() == [[True, False, True]]
    with pytest.raises(
        LMTaskError, match="with a boolean mask; for \\(inputs, targets\\) batches pass alignment='pre_shifted'"
    ):
        shifted.split((ids[:, :-1], ids[:, 1:]))
    with pytest.raises(LMTaskError, match="labels .* must have the ids' shape"):
        shifted.split({"input_ids": ids, "labels": labels[:, 1:]})
    aligned = CausalLMTask(vocab_size=V, alignment="pre_shifted")
    with pytest.raises(LMTaskError, match="HuggingFace-style"):
        aligned.split({"input_ids": ids, "labels": ids})
    inputs, targets, _ = aligned.split({"inputs": ids[:, :-1], "targets": ids[:, 1:]})
    assert torch.equal(targets, ids[:, 1:])
    with pytest.raises(LMTaskError, match="at least one position"):
        aligned.split((ids[:, :0], ids[:, :0]))


def test_the_token_count_restarts_for_each_fresh_run_of_one_objective():
    task = CausalLMTask(vocab_size=V)
    objective = task.objective()
    for index in range(2):
        run = _model().train(params=_params(1), objective=objective, eval_step_fn=task.eval_step(), salt=f"fit-{index}")
        state = NNCheckpoint.load_training_state(run=run.id, type=Checkpoints.LAST)["components"]["lm.causal_task"]
        assert state["state"]["tokens"] == 8 * 5  # one epoch, not the sum over runs


def test_a_module_output_goes_through_its_batch_adapter():
    from torch import nn

    from nnx import NNModelParams as Params
    from nnx.models import BatchAdapter

    class Pair(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(V, V)

        def forward(self, ids):
            return self.embed(ids), None

    class First(BatchAdapter):
        def split(self, batch):
            return (batch,), {}, None

        def output(self, raw):
            return raw[0]

    model = NNModel(module=Pair(), params=Params(loss=Losses.CROSS_ENTROPY), batch_adapter=First())
    task = CausalLMTask(vocab_size=V)
    logits = task.logits(model, *task.split(_ids(2, 5))[:2])
    assert logits.shape == (2, 4, V)
    plain = NNModel(module=Pair(), params=Params(loss=Losses.CROSS_ENTROPY))
    with pytest.raises(LMTaskError, match="floating logits"):
        task.logits(plain, *task.split(_ids(2, 5))[:2])


def test_an_all_masked_microbatch_is_finite_even_with_infinite_logits():
    task = CausalLMTask(vocab_size=V, pad_id=0)
    logits = torch.full((1, 3, V), math.inf, requires_grad=True)
    ce, nll, correct, n = task.token_sums(
        logits, torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 3, dtype=torch.bool)
    )
    assert float(ce.detach()) == 0.0 and (nll, correct, n) == (0.0, 0, 0)
    ce.backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) == 0.0


def test_an_empty_validation_loader_and_extra_metrics_are_refused():
    task = CausalLMTask(vocab_size=V)
    with pytest.raises(LMTaskError, match="yielded no batches"):
        task.eval_step()(_Ctx(_model(), []))
    params = _params(1, extra_metrics={"acc": lambda y, p: 0.0})
    with pytest.raises(LMTaskError, match="extra_metrics"):
        _model().train(params=params, objective=task.objective(), eval_step_fn=task.eval_step())


def test_the_nll_matches_a_float64_reference_without_a_float64_vocabulary_copy(monkeypatch):
    task = CausalLMTask(vocab_size=V)
    logits = torch.randn(3, 5, V, generator=torch.Generator().manual_seed(3)) * 4
    targets = torch.randint(0, V, (3, 5), generator=torch.Generator().manual_seed(4))
    valid = torch.ones(3, 5, dtype=torch.bool)
    expected = float(-F.log_softmax(logits.double(), dim=-1).gather(-1, targets.unsqueeze(-1)).sum())
    original = torch.Tensor.double
    widths = []

    def spy(self, *args, **kwargs):
        widths.append(self.shape[-1] if self.ndim else 0)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "double", spy)
    _, nll, _, _ = task.token_sums(logits, targets, valid)
    assert nll == pytest.approx(expected, rel=1e-6) and V not in widths  # only per-token values are widened


def test_round_two_edges():
    import io

    import numpy as np

    task = CausalLMTask(vocab_size=V)
    ids = torch.tensor([[1, 2, 3, 0, 0]])
    with pytest.raises(LMTaskError, match="padded position"):
        task.split({"input_ids": ids, "attention_mask": torch.tensor([[1, 1, 1, 0, 0]])})
    padded = CausalLMTask(vocab_size=V, pad_id=0)
    padded.split({"input_ids": ids, "attention_mask": torch.tensor([[1, 1, 1, 0, 0]])})  # pad_id: unscored
    for second in (torch.tensor([[1, 0, 1, 1, 0]]), torch.tensor([[0, 1, 1, 0, 1]])):  # 0/1 ints: could be targets
        with pytest.raises(LMTaskError, match="boolean mask"):
            CausalLMTask(vocab_size=2).split((torch.tensor([[1, 0, 1, 1, 0]]), second))
    _, _, mask = task.split({"input_ids": ids, "loss_mask": torch.tensor([[1, 1, 1, 0, 0]])})
    assert mask is not None and mask.dtype == torch.bool
    named = CausalLMTask(vocab_size=V, alignment=np.str_("pre_shifted"), tokenizer=np.str_("sha256:abc"))
    assert type(named.state()["alignment"]) is str and type(named.state()["tokenizer"]) is str
    buffer = io.BytesIO()
    torch.save(named.state(), buffer)
    buffer.seek(0)
    assert CausalLMTask.from_state(torch.load(buffer, weights_only=True)) == named
    # Half precision: the reported NLL is computed in float32.
    logits = torch.randn(64, V, generator=torch.Generator().manual_seed(0)) * 20
    targets = logits.argmax(dim=-1)  # confident tokens: tiny NLLs half precision rounds to 0
    valid = torch.ones(64, dtype=torch.bool)
    _, exact, _, _ = task.token_sums(logits.double(), targets, valid)
    _, half, _, _ = task.token_sums(logits.to(torch.bfloat16), targets, valid)
    reference = float(-F.log_softmax(logits.to(torch.bfloat16).double(), dim=-1).gather(1, targets[:, None]).sum())
    assert half == pytest.approx(reference, rel=1e-4, abs=1e-6) and exact > 0


def test_a_task_state_without_its_vocab_size_is_a_task_error():
    from nnx.lm_tasks import KIND, CausalLMTask, LMTaskError

    with pytest.raises(LMTaskError, match="'vocab_size'"):
        CausalLMTask.from_state({"kind": KIND})


def test_a_one_dimensional_attention_mask_is_a_task_error():
    task = CausalLMTask(vocab_size=V, pad_id=0)
    with pytest.raises(LMTaskError, match="attention_mask"):
        task.split({"input_ids": torch.tensor([[1, 2, 3, 0]]), "attention_mask": torch.tensor([1, 1, 1, 0])})


@pytest.mark.parametrize("mask", [[[0, 0, 1, 1, 1]], [[1, 0, 1, 1, 1]]], ids=["left", "interior"])
def test_padding_before_a_real_token_is_refused(mask):
    # The model never receives the mask: real tokens would attend to the padding.
    task = CausalLMTask(vocab_size=V, pad_id=0)
    with pytest.raises(LMTaskError, match="right-pad the batch"):
        task.split({"input_ids": torch.tensor([[0, 0, 1, 2, 3]]), "attention_mask": torch.tensor(mask)})


# --- #577: one vocabulary pass, and no logits copy when every position is scored ---------------------------

COPIES = {"index", "masked_select", "clone", "_to_copy", "copy", "copy_"}  # ops that would copy (n, V) logits
CROSS_ENTROPY = {"_log_softmax", "log_softmax", "nll_loss_forward", "nll_loss2d_forward", "cross_entropy_loss"}


class _Census(TorchDispatchMode):
    """The aten ops a call runs: name, output size and dtype, and the
    storages of the first tensor argument and of the output."""

    def __init__(self):
        super().__init__()
        self.ops: list[tuple[str, int, Optional[torch.dtype], int, int]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        first = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
        tensor = out if isinstance(out, torch.Tensor) else None
        self.ops.append(
            (
                func.overloadpacket.__name__,
                0 if tensor is None else tensor.numel(),
                None if tensor is None else tensor.dtype,
                0 if first is None else first.untyped_storage().data_ptr(),
                0 if tensor is None else tensor.untyped_storage().data_ptr(),
            )
        )
        return out


def _token_sums_census(monkeypatch) -> list[dict]:
    """Every ``token_sums`` call: its logits and the ops it ran."""
    calls: list[dict] = []
    original = CausalLMTask.token_sums

    def spy(self, logits, targets, valid, **kwargs):
        census = _Census()
        with census:
            out = original(self, logits, targets, valid, **kwargs)
        calls.append(
            {
                "with_ce": kwargs.get("with_ce", True),
                "size": logits.numel(),
                "dtype": logits.dtype,
                "storage": logits.untyped_storage().data_ptr(),
                "n": out[3],
                "ops": census.ops,
            }
        )
        return out

    monkeypatch.setattr(CausalLMTask, "token_sums", spy)
    return calls


def _trainer_fit(model: NNModel, objective) -> None:
    from nnx.trainer import NNTrainerParams, Trainer

    params = (
        NNTrainerParams.builder()
        .n_epochs(1)
        .train_loader(DataLoader(TensorDataset(_ids(8, 6, seed=7).clamp(min=1)), batch_size=4))
        .optimizer("default", NNOptimParams.builder().sgd(max_lr=0.1).build())
        .save_phase_checkpoints(False)
        .build()
    )
    Trainer(model).train(params=params, objective=objective)


def test_an_all_scored_batch_costs_one_vocabulary_pass_and_no_logits_copy(monkeypatch):
    task = CausalLMTask(vocab_size=V)  # smoothing 0; _params' batches score every position
    calls = _token_sums_census(monkeypatch)
    _model().train(params=_params(1), objective=task.objective(), eval_step_fn=task.eval_step())
    _trainer_fit(_model(), task.objective())
    trained = [call for call in calls if call["with_ce"]]
    evaluated = [call for call in calls if not call["with_ce"]]
    assert len(trained) == 4 and len(evaluated) == 2  # two training batches per fit; two validation batches
    for call in calls:
        assert call["dtype"] == torch.float32 and call["n"] * V == call["size"]  # every position scored
        assert not [op for op, size, *_ in call["ops"] if op in COPIES and size == call["size"]]  # no (n, V) copy
        assert not [op for op, *_ in call["ops"] if op in CROSS_ENTROPY]  # no cross-entropy pass
        # One logsumexp, over the logits themselves: the loss and the NLL in training, the NLL in evaluation.
        assert [read == call["storage"] for op, _, _, read, _ in call["ops"] if op == "logsumexp"] == [True]


@pytest.mark.parametrize("dtype", ["float32", "bf16", "bf16-autocast"])
def test_the_smoothing_0_backward_fills_one_vocabulary_buffer_the_gradient(dtype):
    task = CausalLMTask(vocab_size=V)
    targets = _ids(4, 7, seed=14)
    logits = torch.randn(4, 7, V, generator=torch.Generator().manual_seed(15))
    logits = (logits if dtype == "float32" else logits.to(torch.bfloat16)).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == "bf16-autocast"):
        ce, _, _, n = task.token_sums(logits, targets, task.valid(targets, None))
    census = _Census()
    with census:
        ce.backward()
    seen: set[int] = {logits.untyped_storage().data_ptr()}
    fresh = []
    for _, size, out_dtype, read, wrote in census.ops:
        if wrote not in seen and size >= n * V:
            fresh.append(out_dtype)
        seen |= {read, wrote}
    # One (n, V) buffer in float32, the gradient itself; for bf16 logits it is cast once more, to bf16.
    assert n * V == logits.numel() and fresh == [torch.float32] + [logits.dtype] * (dtype != "float32")
    assert logits.grad is not None and logits.grad.dtype == logits.dtype
    expected = torch.softmax(logits.detach().float(), dim=-1) - F.one_hot(targets, V)
    assert torch.allclose(logits.grad.float(), expected, rtol=1e-5 if dtype == "float32" else 2**-8, atol=1e-7)


@pytest.mark.parametrize("transform", ["func.grad", "func.jvp", "forward_ad", "func.hessian"])
@pytest.mark.parametrize("masked", [False, True], ids=["all-scored", "masked"])
def test_torch_func_and_forward_mode_ad_run_through_the_smoothing_0_loss_as_through_cross_entropy(masked, transform):
    import torch.autograd.forward_ad as fwAD

    task = CausalLMTask(vocab_size=V)
    g = torch.Generator().manual_seed(20)
    base = torch.randn(2, 3, V, generator=g, dtype=torch.float64) * 2
    tangent = torch.randn(2, 3, V, generator=g, dtype=torch.float64)
    targets = torch.randint(0, V, (2, 3), generator=g)
    if masked:
        targets[0, :2] = task.ignore_id
    valid = task.valid(targets, None)

    def apply(fn):
        if transform == "func.grad":
            return torch.func.grad(fn)(base)
        if transform == "func.jvp":
            return torch.func.jvp(fn, (base,), (tangent,))[1]
        if transform == "func.hessian":  # forward over reverse mode, vmapped
            return torch.func.hessian(fn)(base)
        with fwAD.dual_level():
            return fwAD.unpack_dual(fn(fwAD.make_dual(base, tangent))).tangent

    ours = apply(lambda x: task.token_sums(x, targets, valid)[0])
    reference = apply(lambda x: F.cross_entropy(x[valid], targets[valid], reduction="sum"))
    assert torch.allclose(ours, reference, rtol=1e-10, atol=1e-12)


def test_the_smoothing_0_loss_keeps_a_second_derivative():
    task = CausalLMTask(vocab_size=V)
    targets = _ids(3, 4, seed=18)
    targets[:, 0] = task.ignore_id
    valid = task.valid(targets, None)
    base = torch.randn(3, 4, V, dtype=torch.float64, generator=torch.Generator().manual_seed(19))
    assert torch.autograd.gradgradcheck(lambda x: task.token_sums(x, targets, valid)[0], (base.requires_grad_(),))
    penalties = []
    for fn in (task.token_sums, lambda *args: _e69fd74_token_sums(task, *args)):  # a gradient penalty
        logits = base.detach().clone().requires_grad_()
        (grad,) = torch.autograd.grad(fn(logits, targets, valid)[0], logits, create_graph=True)
        grad.pow(2).sum().backward()
        penalties.append(logits.grad)
    assert torch.allclose(penalties[0], penalties[1], rtol=1e-10, atol=1e-12)


def test_bf16_logits_under_autocast_cost_one_vocabulary_pass_in_float32(monkeypatch):
    task = CausalLMTask(vocab_size=V)
    calls = _token_sums_census(monkeypatch)
    _model(precision="bf16").train(params=_params(1), objective=task.objective(), eval_step_fn=task.eval_step())
    _trainer_fit(_model(precision="bf16"), task.objective())
    trained = [call for call in calls if call["with_ce"]]
    assert len(trained) == 4 and all(call["dtype"] == torch.bfloat16 for call in trained)
    for call in trained:
        assert not [op for op, *_ in call["ops"] if op in CROSS_ENTROPY]
        assert [dtype for op, _, dtype, _, _ in call["ops"] if op == "logsumexp"] == [torch.float32]
    # Its one float32 widening of the (n, V) logits is one e69fd74 made too, for the NLL.
    logits = torch.randn(4, 7, V, generator=torch.Generator().manual_seed(16)).to(torch.bfloat16)
    targets = _ids(4, 7, seed=17)
    valid = task.valid(targets, None)
    widened = []
    for fn in (task.token_sums, lambda *args: _e69fd74_token_sums(task, *args)):
        census = _Census()
        with torch.autocast("cpu", dtype=torch.bfloat16), census:
            fn(logits.clone().requires_grad_(), targets, valid)
        widened.append(
            sum(
                op == "_to_copy" and size == logits.numel() and dtype == torch.float32
                for op, size, dtype, *_ in census.ops
            )
        )
    assert widened[0] == 1 and widened[1] >= 1


def _e69fd74_token_sums(task, logits, targets, valid):
    """``token_sums`` before #577: a copy of the scored rows, then a second pass for the NLL."""
    n = int(valid.sum())
    flat, chosen = logits[valid], targets[valid]
    ce = F.cross_entropy(flat, chosen, reduction="sum", label_smoothing=task.smoothing)
    with torch.no_grad():
        detached = flat.detach()
        if detached.dtype in (torch.float16, torch.bfloat16):
            detached = detached.float()
        per_token = torch.logsumexp(detached, dim=-1) - detached.gather(1, chosen.unsqueeze(1)).squeeze(1)
        nll = float(per_token.cpu().double().sum())
        correct = int((detached.argmax(dim=-1) == chosen).sum())
    return ce, nll, correct, n


@pytest.mark.parametrize("smoothing", [0.0, 0.1])
@pytest.mark.parametrize("masked", [False, True], ids=["all-scored", "masked"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "bf16-autocast", "bf16"])
def test_token_sums_match_the_two_pass_reference(smoothing, masked, dtype):
    task = CausalLMTask(vocab_size=V, smoothing=smoothing)
    g = torch.Generator().manual_seed(11)
    base = torch.randn(4, 7, V, generator=g) * 3
    targets = torch.randint(0, V, (4, 7), generator=g)
    if masked:
        targets[:, :2] = task.ignore_id
    valid = task.valid(targets, None)
    width = torch.float64 if dtype == "float64" else torch.bfloat16 if dtype.startswith("bf16") else torch.float32
    # Without smoothing, bf16 logits outside autocast now get their loss in float32, like their NLL.
    widen = dtype == "bf16" and not smoothing
    results = []
    for fn in (task.token_sums, lambda x, *args: _e69fd74_token_sums(task, x.float() if widen else x, *args)):
        logits = base.to(width).requires_grad_()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == "bf16-autocast"):
            ce, nll, correct, n = fn(logits, targets, valid)
        ce.backward()
        results.append((ce.detach(), nll, correct, n, logits.grad))
    (ce, nll, correct, n, grad), (ce_ref, nll_ref, correct_ref, n_ref, grad_ref) = results
    assert ce.dtype == ce_ref.dtype and torch.allclose(ce, ce_ref, rtol=1e-2 if ce.dtype == torch.bfloat16 else 1e-6)
    assert nll == pytest.approx(nll_ref, rel=1e-5) and (correct, n) == (correct_ref, n_ref)
    assert torch.allclose(grad, grad_ref, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("dtype", ["float32", "bf16-autocast"])
@pytest.mark.parametrize(("margin", "noise"), [(16.0, 2.0), (16.0, 1.0)], ids=["nll-0.14", "nll-0.011"])
def test_confident_tokens_keep_e69fd74s_nll_at_a_realistic_vocabulary(margin, noise, dtype):
    vocab = 32000
    task = CausalLMTask(vocab_size=vocab)
    g = torch.Generator().manual_seed(3)
    base = torch.randn(2, 128, vocab, generator=g) * noise
    targets = torch.randint(0, vocab, (2, 128), generator=g)
    base.scatter_add_(2, targets.unsqueeze(-1), torch.full((2, 128, 1), margin))  # each target leads by `margin`
    if dtype == "bf16-autocast":
        base = base.to(torch.bfloat16)
    valid = task.valid(targets, None)
    results = []
    for fn in (task.token_sums, lambda *args: _e69fd74_token_sums(task, *args)):
        logits = base.clone().requires_grad_()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == "bf16-autocast"):
            ce, nll, correct, n = fn(logits, targets, valid)
        ce.backward()
        assert logits.grad is not None
        results.append((float(ce.detach()), nll, correct, n, logits.grad.double()))
    (ce, nll, correct, n, grad), (_, nll_ref, correct_ref, n_ref, grad_ref) = results
    exact = base.double().reshape(-1, vocab).requires_grad_()
    reference = F.cross_entropy(exact, targets.reshape(-1), reduction="sum")
    reference.backward()
    assert exact.grad is not None
    assert nll == pytest.approx(nll_ref, rel=1e-5) and (correct, n) == (correct_ref, n_ref)  # e69fd74's formula
    assert nll == pytest.approx(float(reference.detach()), rel=1e-5)
    assert ce == pytest.approx(nll, rel=1e-6)  # the loss is that NLL
    rtol = 1e-5 if dtype == "float32" else 2**-7  # a bf16 gradient is rounded to bf16
    assert torch.allclose(grad, exact.grad.reshape(base.shape), rtol=rtol, atol=5e-6)
    # e69fd74's gradient came from torch's cross-entropy, which on these tokens is itself up to 1.1e-4 off.
    assert torch.allclose(grad, grad_ref, rtol=rtol, atol=2e-4)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an MPS device")
@pytest.mark.parametrize("half", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_mps_autocast_keeps_the_loss_and_the_nll_in_float32(half):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with torch.autocast("mps", dtype=half):
                supported = torch.is_autocast_enabled("mps")
        except RuntimeError:  # no MPS autocast before torch 2.5
            supported = False
    if not supported:
        pytest.skip("this torch has no MPS autocast")
    task = CausalLMTask(vocab_size=V)
    g = torch.Generator().manual_seed(5)
    base = (torch.randn(3, 9, V, generator=g) * 4).to(half)
    targets = torch.randint(0, V, (3, 9), generator=g)
    valid = task.valid(targets, None)
    logits = base.to("mps").requires_grad_()
    census = _Census()
    with torch.autocast("mps", dtype=half), census:
        ce, nll, correct, n = task.token_sums(logits, targets, valid)
    ce.backward()
    _, nll_ref, correct_ref, n_ref = _e69fd74_token_sums(task, base, targets, valid)  # its NLL, in float32 on the CPU
    assert nll == pytest.approx(nll_ref, rel=1e-5) and float(ce.detach()) == pytest.approx(nll, rel=1e-6)
    assert (correct, n) == (correct_ref, n_ref)
    assert not [op for op, *_ in census.ops if op in CROSS_ENTROPY]  # no half-precision log_softmax
    assert [dtype for op, _, dtype, _, _ in census.ops if op == "logsumexp"] == [torch.float32]
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())


def test_a_partially_masked_batch_with_nan_logits_at_unscored_positions_stays_finite():
    task = CausalLMTask(vocab_size=V)
    targets = _ids(3, 6, seed=12)
    targets[:, :2] = task.ignore_id  # a prompt-masked batch
    logits = torch.randn(3, 6, V, generator=torch.Generator().manual_seed(13))
    logits[:, :2] = math.nan  # unscored positions never enter the softmax
    logits.requires_grad_()
    ce, nll, correct, n = task.token_sums(logits, targets, task.valid(targets, None))
    ce.backward()
    assert n == 12 and math.isfinite(float(ce.detach())) and math.isfinite(nll) and 0 <= correct <= n
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())
    assert float(logits.grad[:, :2].abs().sum()) == 0.0
    expected = float(
        F.cross_entropy(logits.detach()[:, 2:].reshape(-1, V), targets[:, 2:].reshape(-1), reduction="sum")
    )
    assert float(ce.detach()) == pytest.approx(expected, rel=1e-6) and nll == pytest.approx(expected, rel=1e-5)
