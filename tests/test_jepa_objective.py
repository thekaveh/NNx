"""FEAT-040: I-JEPA latent prediction as an objective.

The adapter returns a differentiable squared-error sum over the masked
target elements and detached metrics, and never zeroes, back-propagates,
scales or steps; uneven microbatches with fixed masks take the full-batch
update and unequal target counts combine as ``(s1 + s2) / (c1 + c2)``;
empty masks are rejected first; the target stays frozen under no-grad, the
predictor is optimizer-owned once and unsupported combinations fail before
any run; the EMA advances once per committed update and never after a skip.
"""

from __future__ import annotations

import copy
import os

import pytest
import torch

from nnx import (
    Activations,
    Callback,
    Devices,
    JEPAObjective,
    JEPAPredictor,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParamGroupSpec,
    NNParams,
    NNSchedulerParams,
    NNTrainerParams,
    NNTrainParams,
    ObjectiveContext,
    Optims,
    Trainer,
    ViTNN,
    build_target_encoder,
    jepa_objective,
    jepa_train_step_factory,
    set_seed,
)

GRID = 4  # 16x16 images, 4x4 patches
_PLATEAU = NNSchedulerParams(min_lr=0.0, factor=0.5, patience=10, cooldown=0, threshold=0.0)


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.chdir(tmp_path)


def _parts(seed: int = 0, *, attach: bool = True):
    """The small ViT / predictor fixture of tests/test_component_resume.py."""
    set_seed(seed)
    model = NNModel(
        net_params=NNParams(input_dim=12, output_dim=4, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    model.net = ViTNN(image_size=16, patch_size=4, in_channels=3, d_model=16, n_layers=1, n_heads=2)
    target = build_target_encoder(model.net)
    predictor = JEPAPredictor(embed_dim=16, n_patches=model.net.n_patches, predictor_dim=8, n_layers=1, n_heads=2)
    if attach:
        model.net.add_module("_jepa_predictor", predictor)
    return model, target, predictor


def _block(rows: slice, cols: slice):
    grid = torch.zeros(GRID, GRID, dtype=torch.bool)
    grid[rows, cols] = True
    target = grid.flatten()
    return ~target, target


def _masks(*blocks):
    """A mask_fn handing out the given (context, target) masks in turn
    (cycling)."""
    state = {"i": 0}

    def mask_fn(n_patches, device):
        context, target = blocks[state["i"] % len(blocks)]
        state["i"] += 1
        return context.to(device), target.to(device)

    return mask_fn


FOUR = _block(slice(0, 2), slice(0, 2))  # 4 target patches
TWO = _block(slice(3, 4), slice(2, 4))  # 2 target patches


def _images(n: int, seed: int = 1) -> torch.Tensor:
    return torch.randn(n, 3, 16, 16, generator=torch.Generator().manual_seed(seed))


def _params(loader, *, n_epochs: int = 1, accumulate: int = 1, **fields) -> NNTrainParams:
    fields.setdefault("overwrite_existing", True)
    return NNTrainParams(
        n_epochs=n_epochs,
        train_loader=loader,
        optim=NNOptimParams(
            name=Optims.SGD,
            max_lr=0.05,
            momentum=0.0,
            weight_decay=0.0,
            accumulate_grad_batches=accumulate,
        ),
        scheduler=_PLATEAU,
        save_phase_checkpoints=False,
        **fields,
    )


def _batch(x: torch.Tensor):
    return (x, torch.zeros(x.shape[0], dtype=torch.long))


# --- AC1: what the adapter returns, and what it never does ----------------------------------------------------


def test_the_adapter_returns_a_squared_error_sum_and_never_owns_the_update(monkeypatch):
    model, target, predictor = _parts()
    objective = jepa_objective(target, predictor, _masks(FOUR), ema_momentum=0.5)

    def forbidden(*args, **kwargs):
        raise AssertionError("an objective never zeroes, back-propagates, scales or steps")

    monkeypatch.setattr(torch.optim.SGD, "step", forbidden)
    monkeypatch.setattr(torch.nn.Module, "zero_grad", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    result = objective(ObjectiveContext(model=model, batch=_batch(_images(3)), epoch_idx=0, batch_idx=0))

    (term,) = result.terms
    # 3 images × 4 target patches × 16 embedding dims.
    assert (term.name, term.reduction, term.denominator) == ("latent_mse", "mean", 3.0 * 4 * 16)
    assert term.numerator.requires_grad
    record = result.record
    assert record is not None and dict(record.metrics) == {"latent_mse": record.loss}
    assert (record.f1, record.recall, record.accuracy, record.precision, record.error) == (None,) * 5
    assert all(p.grad is None for p in model.net.parameters())
    assert objective.ema_updates == 0 and all(not p.requires_grad for p in target.parameters())


def test_empty_masks_are_rejected_before_any_forward():
    model, target, predictor = _parts()
    calls: list[int] = []
    model.net.register_forward_hook(lambda *args: calls.append(1))
    target.register_forward_hook(lambda *args: calls.append(1))
    everything = torch.ones(GRID * GRID, dtype=torch.bool)
    ctx = ObjectiveContext(model=model, batch=_batch(_images(2)), epoch_idx=0, batch_idx=0)
    with pytest.raises(ValueError, match="target mask is empty"):
        jepa_objective(target, predictor, _masks((everything, ~everything)))(ctx)
    with pytest.raises(ValueError, match="context mask is empty"):
        jepa_objective(target, predictor, _masks((~everything, everything)))(ctx)
    with pytest.raises(ValueError, match="complementary"):
        jepa_objective(target, predictor, _masks((FOUR[0], FOUR[0])))(ctx)
    assert calls == []


def test_arguments_are_validated():
    _, target, predictor = _parts()
    for momentum in (1.0, -0.1, True):
        with pytest.raises(ValueError, match="ema_momentum"):
            JEPAObjective(target, predictor, _masks(FOUR), ema_momentum=momentum)
    with pytest.raises(ValueError, match="different modules"):
        jepa_objective(target, target, _masks(FOUR))
    with pytest.raises(TypeError, match="mask_fn"):
        jepa_objective(target, predictor, None)  # type: ignore[arg-type]


# --- AC2: uneven microbatches and unequal counts ----------------------------------------------------------------


def test_uneven_microbatches_match_the_full_batch_reference():
    x = _images(3)
    full, full_target, full_predictor = _parts()
    split, split_target, split_predictor = _parts()
    full.train(
        _params([_batch(x)]), objective=jepa_objective(full_target, full_predictor, _masks(FOUR), ema_momentum=0.5)
    )
    split.train(
        _params([_batch(x[:2]), _batch(x[2:])], accumulate=4),  # a short [2, 1] window
        objective=jepa_objective(split_target, split_predictor, _masks(FOUR), ema_momentum=0.5),
    )
    for key, value in full.net.state_dict().items():
        torch.testing.assert_close(split.net.state_dict()[key], value, rtol=1e-6, atol=1e-7, msg=key)
    for key, value in full_target.state_dict().items():
        torch.testing.assert_close(split_target.state_dict()[key], value, rtol=1e-6, atol=1e-7, msg=f"target {key}")


def test_unequal_target_counts_combine_as_summed_errors_over_summed_counts():
    x = _images(4)
    model, target, predictor = _parts()
    reference, reference_target = copy.deepcopy(model.net), copy.deepcopy(target)  # before the EMA moves it
    objective = jepa_objective(target, predictor, _masks(FOUR, TWO), ema_momentum=0.5)
    events: list = []
    model.train(
        _params([_batch(x[:2]), _batch(x[2:])], accumulate=2),
        objective=objective,
        callbacks=[_Events(events)],
    )
    # The reference: one SGD step on (s1 + s2) / (c1 + c2), computed by hand.
    ref_objective = jepa_objective(reference_target, reference._jepa_predictor, _masks(FOUR, TWO), ema_momentum=0.5)
    ref_model = copy.copy(model)
    ref_model.net = reference
    terms = [
        ref_objective(ObjectiveContext(model=ref_model, batch=_batch(part), epoch_idx=0, batch_idx=i)).terms[0]
        for i, part in enumerate((x[:2], x[2:]))
    ]
    s1, c1, s2, c2 = terms[0].numerator, terms[0].denominator, terms[1].numerator, terms[1].denominator
    assert (c1, c2) == (2 * 4 * 16, 2 * 2 * 16)
    loss = (s1 + s2) / (c1 + c2)
    assert events[0].losses["latent_mse"] == pytest.approx(float(loss.detach()), rel=1e-6)
    loss.backward()
    with torch.no_grad():
        for p in reference.parameters():
            if p.grad is not None:
                p -= 0.05 * p.grad
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(model.net.state_dict()[key], value, rtol=1e-6, atol=1e-7, msg=key)


class _Events(Callback):
    def __init__(self, events: list) -> None:
        self.events = events

    def on_optimizer_update(self, ctx, event) -> None:
        self.events.append(event)


# --- AC3: ownership and refusals before any run ----------------------------------------------------------------


def test_the_target_stays_frozen_and_never_optimizer_owned():
    model, target, predictor = _parts()
    target_before = copy.deepcopy(target.state_dict())
    outputs: list[bool] = []
    target.register_forward_hook(lambda module, inputs, output: outputs.append(output.requires_grad))
    model.train(
        _params([_batch(_images(2))]), objective=jepa_objective(target, predictor, _masks(FOUR), ema_momentum=0.5)
    )
    assert outputs == [False]  # under no_grad
    assert all(not p.requires_grad and p.grad is None for p in target.parameters()) and not target.training
    assert any(not torch.equal(target.state_dict()[k], v) for k, v in target_before.items())  # moved by the EMA only


@pytest.mark.parametrize(
    ("break_it", "match"),
    [
        ("detached", "must be a submodule of model.net"),
        ("target_inside", "must not be part of model.net"),
        ("not_a_vit", "ViT-style model.net"),
        ("topology", "change model.net's topology"),
        ("renamed_target", "no same-named, same-shaped counterpart"),
    ],
)
def test_unsupported_combinations_fail_before_any_run(break_it, match):
    model, target, predictor = _parts(attach=break_it != "detached")
    callbacks: list = []
    if break_it == "target_inside":
        model.net.add_module("_target", target)
    elif break_it == "not_a_vit":
        model.net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(768, 4))
        model.net.add_module("_jepa_predictor", predictor)
    elif break_it == "topology":
        callbacks.append(_TopologyChanging())
    elif break_it == "renamed_target":
        target = torch.nn.Sequential(torch.nn.Linear(2, 2))
    objective = jepa_objective(target, predictor, _masks(FOUR))
    before = copy.deepcopy(model.net.state_dict())
    with pytest.raises(ValueError, match=match):
        model.train(_params([_batch(_images(2))]), objective=objective, callbacks=callbacks)
    assert not os.path.exists("runs")
    assert all(torch.equal(model.net.state_dict()[k], v) for k, v in before.items())


class _TopologyChanging(Callback):
    def checkpoint_transforms(self):  # a QAT-style callback declares its topology transforms
        return ()


def test_the_predictor_must_be_optimizer_owned_exactly_once():
    model, target, predictor = _parts()
    encoder_only = torch.optim.SGD([p for n, p in model.net.named_parameters() if not n.startswith("_jepa")], lr=0.1)
    with pytest.raises(ValueError, match="do not own the JEPA predictor parameters"):
        jepa_objective(target, predictor, _masks(FOUR)).check_run(
            model, optimizers={"default": encoder_only}, callbacks=[]
        )
    shared = list(predictor.parameters())
    optimizers = {"a": torch.optim.SGD(shared, lr=0.1), "b": torch.optim.SGD(shared[:1], lr=0.1)}
    with pytest.raises(ValueError, match="owned by more than one optimizer"):
        jepa_objective(target, predictor, _masks(FOUR)).check_run(model, optimizers=optimizers, callbacks=[])
    with pytest.raises(ValueError, match="optimizer owns JEPA target-encoder parameters"):
        jepa_objective(target, predictor, _masks(FOUR)).check_run(
            model,
            optimizers={"a": torch.optim.SGD(list(model.net.parameters()) + list(target.parameters()), lr=0.1)},
            callbacks=[],
        )


def test_the_imperative_jepa_step_is_refused_as_an_objective():
    model, target, predictor = _parts()
    step = jepa_train_step_factory(target_encoder=target, predictor=predictor, mask_fn=_masks(FOUR))
    with pytest.raises(ValueError, match="imperative jepa step"):
        model.train(_params([_batch(_images(2))]), objective=step)
    assert not os.path.exists("runs")


# --- AC4: the EMA advances once per committed update ------------------------------------------------------------


def test_the_ema_advances_once_per_committed_update_from_the_updated_weights():
    model, target, predictor = _parts()
    old = copy.deepcopy(target.state_dict())
    objective = jepa_objective(target, predictor, _masks(FOUR), ema_momentum=0.5)
    x = _images(4)
    seen: list[int] = []

    class Count(Callback):
        def on_optimizer_update(self, ctx, event) -> None:
            seen.append(objective.ema_updates)  # the EMA ran before callbacks see the update

    model.train(_params([_batch(x[:2]), _batch(x[2:])], accumulate=2), objective=objective, callbacks=[Count()])
    assert objective.ema_updates == 1 and seen == [1]  # two microbatches, one committed update
    online = dict(model.net.named_parameters())
    for name, value in target.named_parameters():
        torch.testing.assert_close(value, 0.5 * old[name] + 0.5 * online[name].detach(), msg=name)


def test_a_skipped_window_leaves_the_target_unchanged():
    model, target, predictor = _parts()
    old = copy.deepcopy(target.state_dict())
    objective = jepa_objective(target, predictor, _masks(FOUR), ema_momentum=0.5, nonfinite="skip")
    poisoned = torch.full((2, 3, 16, 16), float("nan"))
    run = model.train(_params([_batch(poisoned)]), objective=objective)
    assert run.idps[-1].update_count == 0 and objective.ema_updates == 0
    assert all(torch.equal(target.state_dict()[k], v) for k, v in old.items())


def test_a_trainer_with_two_optimizers_advances_the_ema_once_per_commit():
    model, target, predictor = _parts()
    objective = jepa_objective(target, predictor, _masks(FOUR), ema_momentum=0.5)
    sgd = dict(name=Optims.SGD, max_lr=0.05, momentum=0.0, weight_decay=0.0)
    encoder = sorted({n.split(".")[0] for n, _ in model.net.named_parameters() if not n.startswith("_jepa")})
    params = NNTrainerParams(
        n_epochs=1,
        train_loader=[_batch(_images(2, seed=s)) for s in range(3)],
        optims={
            "encoder": NNOptimParams(**sgd, param_groups=[NNParamGroupSpec(name_pattern=f"{n}*") for n in encoder]),
            "predictor": NNOptimParams(**sgd, param_groups=[NNParamGroupSpec(name_pattern="_jepa_predictor.*")]),
        },
        save_phase_checkpoints=False,
        overwrite_existing=True,
    )
    events: list = []
    Trainer(model).train(params, objective=objective, callbacks=[_Events(events)])
    assert len(events) == 6 and objective.ema_updates == 3  # one event per optimizer, one EMA per commit


# --- review round 2 ---------------------------------------------------------------------------------------


def test_a_misplaced_target_never_freezes_the_online_network():
    model, _, predictor = _parts()
    objective = jepa_objective(model.net, predictor, _masks(FOUR))  # the online net passed as its own target
    assert all(p.requires_grad for p in model.net.parameters()) and model.net.training
    with pytest.raises(ValueError, match="must not be part of model.net"):
        model.train(_params([_batch(_images(2))]), objective=objective)
    assert all(p.requires_grad for p in model.net.parameters())
    assert not os.path.exists("runs")


def test_a_target_on_another_dtype_is_refused_before_any_run():
    model, target, predictor = _parts()
    model.net.double()  # the target stayed float32
    with pytest.raises(ValueError, match="another device or dtype"):
        model.train(_params([_batch(_images(2).double())]), objective=jepa_objective(target, predictor, _masks(FOUR)))
    assert not os.path.exists("runs")


def test_the_ema_counter_counts_each_runs_steps():
    model, target, predictor = _parts()
    objective = jepa_objective(target, predictor, _masks(FOUR), ema_momentum=0.5)
    loader = [_batch(_images(2, seed=s)) for s in range(2)]
    model.train(_params(loader), objective=objective)
    assert objective.ema_updates == 2
    model.train(_params(loader, seed=1), objective=objective)  # a fresh run, not a resume
    assert objective.ema_updates == 2
