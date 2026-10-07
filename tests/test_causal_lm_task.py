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

import pytest
import torch
import torch.nn.functional as F
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


def _model(seed: int = 0) -> NNModel:
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
        net_params=net, params=NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
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
