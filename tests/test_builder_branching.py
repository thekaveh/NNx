"""FEAT-041: branch mutable parameter builders with ``copy()`` and rebuild
them from an existing params value with ``from_params()``.

Covers all four builder families (optimizer, scheduler, transformer,
trainer): branch isolation for complete and partial builders, exact
``from_params`` round trips (every supported init field, ``state()`` key
order and default omission), precise rejection of unsupported inputs,
shared identity for runtime objects, freedom from side effects, and the
``NNTransformerParams.from_state`` per-layer list fix.
"""

from __future__ import annotations

import builtins
import dataclasses
import random
import socket

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNEvaluationDataPoint,
    NNModel,
    NNModelParams,
    NNOptimFactoryParams,
    NNOptimParams,
    NNOptimParamsBuilder,
    NNParamGroupSpec,
    NNParams,
    NNRun,
    NNSchedulerParams,
    NNSchedulerParamsBuilder,
    NNTrainerParams,
    NNTrainerParamsBuilder,
    NNTrainParams,
    NNTransformerParams,
    NNTransformerParamsBuilder,
    OptimizerFactorySpec,
    Optims,
    Schedulers,
    Trainer,
    TrainerStepContext,
    set_seed,
)
from nnx.nn.params.nn_checkpoint import NNCheckpoint
from nnx.nn.params.nn_iteration_data_point import NNIterationDataPoint

# ---------------------------------------------------------------- helpers


def _adam(max_lr: float = 1e-3) -> NNOptimParams:
    return NNOptimParams.builder().adam(max_lr=max_lr).build()


def _plateau() -> NNSchedulerParams:
    return (
        NNSchedulerParams.builder()
        .reduce_on_plateau(min_lr=1e-6, factor=0.5, patience=2, cooldown=0, threshold=1e-4)
        .build()
    )


def _step_sched() -> NNSchedulerParams:
    return (
        NNSchedulerParams.builder()
        .step(step_size=3, min_lr=1e-6, factor=0.5, patience=2, cooldown=0, threshold=1e-4)
        .build()
    )


def _items(state: dict) -> list:
    """State as an ordered item list: equality then also checks key order."""
    return list(state.items())


class _RaisingLoader(DataLoader):
    """A loader any iteration of which fails the test."""

    def __iter__(self):
        raise AssertionError("builder copy/from_params must never iterate data")


def _raising_loader() -> DataLoader:
    return _RaisingLoader(TensorDataset(torch.zeros(2, 1)), batch_size=1)


def _metric(y_true, y_pred) -> float:
    return 0.0


def _full_transformer(**overrides) -> NNTransformerParams:
    kwargs = dict(
        input_dim=64,
        output_dim=64,
        dropout_prob=0.0,
        activation=Activations.LEAKY_RELU,
        hidden_dims=[7, 5],
        activations=[Activations.RELU, Activations.TANH],
        dropout_probs=[0.1, 0.3],
        n_heads=2,
        vocab_size=64,
        n_layers=2,
        d_model=16,
        max_seq_len=12,
        ffn_mult=2,
        rope_base=500000.0,
        tie_embeddings=False,
        attn_dropout=0.1,
        resid_dropout=0.2,
    )
    kwargs.update(overrides)
    return NNTransformerParams(**kwargs)  # type: ignore[arg-type]


# ------------------------------------------------------------- copy(): optim


def test_optim_copy_branches_complete_builder_without_cross_talk():
    groups = [NNParamGroupSpec(name_pattern="head.*", lr=1e-2)]
    source = NNOptimParams.builder().adamw(max_lr=1e-3, weight_decay=0.05, eps=1e-6).param_groups(groups)
    branch = source.copy()

    assert isinstance(branch, NNOptimParamsBuilder)
    assert branch is not source
    before = source.build()

    # Caller mutates the list it handed to the source builder.
    groups.append(NNParamGroupSpec(name_pattern="body.*", lr=1e-4))
    branch.sgd(max_lr=0.1).accumulate_grad(4)
    source.grad_clip(1.0)

    branched = branch.build()
    assert branched.name == Optims.SGD
    assert branched.grad_clip_norm is None
    assert branched.accumulate_grad_batches == 4
    assert [g.name_pattern for g in branched.param_groups or []] == ["head.*"]

    rebuilt = source.build()
    assert rebuilt.name == Optims.ADAMW and rebuilt.eps == 1e-6
    assert rebuilt.grad_clip_norm == 1.0 and rebuilt.accumulate_grad_batches == 1
    # An already-built value is never touched by later builder activity.
    assert before.grad_clip_norm is None
    assert [g.name_pattern for g in before.param_groups or []] == ["head.*"]


def test_optim_copy_branches_partial_builder():
    source = NNOptimParams.builder().grad_clip(0.5)
    branch = source.copy().adam(max_lr=1e-3)
    assert branch.build().grad_clip_norm == 0.5
    with pytest.raises(ValueError, match="call one of"):
        source.build()
    # The branch's variant did not leak back into the partial source.
    assert "name" not in source._fields


# --------------------------------------------------------- copy(): scheduler


def test_scheduler_copy_branches_and_variant_replacement_stays_local():
    source = NNSchedulerParams.builder().one_cycle(
        max_lr=1e-2, total_steps=10, min_lr=1e-6, factor=0.5, patience=1, cooldown=0, threshold=1e-4
    )
    branch = source.copy().cosine_annealing(T_max=5, min_lr=1e-6, factor=0.5, patience=1, cooldown=0, threshold=1e-4)
    assert source.build().kind == Schedulers.ONE_CYCLE
    built = branch.build()
    assert built.kind == Schedulers.COSINE_ANNEALING and built.max_lr is None and built.T_max == 5


def test_scheduler_copy_of_empty_builder_still_requires_a_variant():
    branch = NNSchedulerParams.builder().copy()
    with pytest.raises(ValueError, match="reduce_on_plateau"):
        branch.build()


# ------------------------------------------------------- copy(): transformer


def test_transformer_copy_branches_partial_builder():
    source = NNTransformerParams.builder().vocab(64).layers(n=2, heads=2, d_model=16)
    branch = source.copy().context(max_seq_len=32).ffn(mult=2)
    with pytest.raises(ValueError, match=r"\.context\(max_seq_len=\.\.\.\)"):
        source.build()
    built = branch.build()
    assert built.max_seq_len == 32 and built.ffn_mult == 2
    source.context(max_seq_len=8).vocab(128)
    assert branch.build().vocab_size == 64
    assert source.build().ffn_mult == 4


# ----------------------------------------------------------- copy(): trainer


def test_trainer_copy_isolates_configuration_collections_and_shares_runtime_objects():
    loader = _raising_loader()
    metrics = {"zero": _metric}
    source = (
        NNTrainerParams.builder()
        .n_epochs(2)
        .optimizer("g", _adam())
        .scheduler("g", _plateau())
        .train_loader(loader)
        .extra_metrics(metrics)
    )
    branch = source.copy()
    metrics["late"] = _metric  # caller-owned map mutated after the branch
    branch.optimizer("d", _adam(2e-4)).scheduler("d", _step_sched()).n_epochs(3)
    source.seed(7)

    built_branch = branch.build()
    assert sorted(built_branch.optims) == ["d", "g"]
    assert sorted(built_branch.schedulers) == ["d", "g"]
    assert built_branch.n_epochs == 3 and built_branch.seed is None
    assert set(built_branch.extra_metrics or {}) == {"zero"}
    # Runtime objects keep identity: never copied, never iterated.
    assert built_branch.train_loader is loader
    assert (built_branch.extra_metrics or {})["zero"] is _metric

    built_source = source.build()
    assert sorted(built_source.optims) == ["g"] and built_source.seed == 7 and built_source.n_epochs == 2


def test_trainer_branch_with_scheduler_key_absent_from_optims_still_fails():
    source = NNTrainerParams.builder().n_epochs(1).optimizer("g", _adam())
    branch = source.copy().scheduler("ghost", _plateau())
    with pytest.raises(ValueError, match="ghost"):
        branch.build()
    source.build()  # the source is unaffected


def test_trainer_builder_data_id_and_overwrite_existing_setters():
    builder = NNTrainerParams.builder().n_epochs(1).optimizer("g", _adam())
    assert builder.data_id("split-a") is builder
    assert builder.overwrite_existing(True) is builder
    built = builder.build()
    assert built.data_id == "split-a" and built.overwrite_existing is True
    # Defaults stay omitted from state() when the setters are not called.
    plain = NNTrainerParams.builder().n_epochs(1).optimizer("g", _adam()).build()
    assert "data_id" not in plain.state() and plain.overwrite_existing is False


# ---------------------------------------------------------- from_params()


@pytest.mark.parametrize(
    "params",
    [
        NNOptimParams(
            name=Optims.ADAMW,
            max_lr=3e-4,
            weight_decay=0.05,
            momentum=(0.8, 0.95),
            grad_clip_norm=1.0,
            accumulate_grad_batches=3,
            param_groups=[NNParamGroupSpec(name_pattern="a.*", lr_multiplier=0.1, weight_decay=0.0)],
            eps=1e-6,
        ),
        NNOptimParams(name=Optims.SGD_NESTEROV, max_lr=0.1, weight_decay=0.0, momentum=0.9),
        NNOptimParams.builder().adam_amsgrad(max_lr=1e-3).build(),
    ],
    ids=["adamw-all-fields", "sgd-nesterov-defaults", "adam-amsgrad"],
)
def test_optim_from_params_round_trips_exactly(params):
    builder = NNOptimParamsBuilder.from_params(params)
    rebuilt = builder.build()
    assert rebuilt == params
    assert _items(rebuilt.state()) == _items(params.state())


def test_optim_from_params_keeps_modifiers_across_a_new_variant():
    params = NNOptimParams.builder().adamw(max_lr=1e-3, eps=1e-6).grad_clip(2.0).accumulate_grad(2).build()
    switched = NNOptimParamsBuilder.from_params(params).sgd(max_lr=0.05).build()
    assert switched.name == Optims.SGD and switched.eps == 1e-8
    assert switched.grad_clip_norm == 2.0 and switched.accumulate_grad_batches == 2


@pytest.mark.parametrize(
    "params",
    [
        NNSchedulerParams(min_lr=1e-6, factor=0.5, patience=2, cooldown=1, threshold=1e-3),
        NNSchedulerParams.builder()
        .step(step_size=4, min_lr=0.0, factor=0.1, patience=0, cooldown=0, threshold=0.0)
        .build(),
        NNSchedulerParams.builder()
        .cosine_annealing(T_max=9, min_lr=1e-7, factor=0.5, patience=1, cooldown=0, threshold=1e-4)
        .build(),
        NNSchedulerParams.builder()
        .one_cycle(max_lr=1e-2, total_steps=20, min_lr=1e-7, factor=0.5, patience=1, cooldown=0, threshold=1e-4)
        .build(),
        NNSchedulerParams.builder()
        .linear_warmup_decay(
            warmup_steps=2, total_steps=20, min_lr=1e-7, factor=0.5, patience=1, cooldown=0, threshold=1e-4
        )
        .build(),
    ],
    ids=["plateau", "step", "cosine", "one-cycle", "warmup-decay"],
)
def test_scheduler_from_params_round_trips_exactly(params):
    rebuilt = NNSchedulerParamsBuilder.from_params(params).build()
    assert rebuilt == params
    assert _items(rebuilt.state()) == _items(params.state())


@pytest.mark.parametrize(
    "params",
    [
        _full_transformer(),
        _full_transformer(activation=None),
        NNTransformerParams.builder().vocab(32).layers(n=1, heads=2, d_model=8).context(max_seq_len=4).build(),
    ],
    ids=["every-field", "activation-none", "builder-defaults"],
)
def test_transformer_from_params_round_trips_exactly(params):
    builder = NNTransformerParamsBuilder.from_params(params)
    assert "_dims" not in builder._fields
    rebuilt = builder.build()
    assert rebuilt == params
    assert _items(rebuilt.state()) == _items(params.state())


def test_transformer_from_params_then_vocab_resets_dims_but_keeps_inherited_fields():
    rebuilt = NNTransformerParamsBuilder.from_params(_full_transformer()).vocab(128).build()
    assert (rebuilt.vocab_size, rebuilt.input_dim, rebuilt.output_dim) == (128, 128, 128)
    assert list(rebuilt.hidden_dims or []) == [7, 5]
    assert list(rebuilt.activations or []) == [Activations.RELU, Activations.TANH]
    assert list(rebuilt.dropout_probs or []) == [0.1, 0.3]


def test_trainer_from_params_round_trips_every_field_and_shares_runtime_objects():
    loader, val_loader = _raising_loader(), _raising_loader()
    factory = NNOptimFactoryParams(
        factory=OptimizerFactorySpec(id="tests.branching", version=1, config={"momentum": 0.9}),
        max_lr=0.05,
    )
    params = NNTrainerParams(
        n_epochs=4,
        optims={"g": _adam(), "d": factory},
        schedulers={"g": _plateau()},
        seed=11,
        data_id="split-b",
        save_phase_checkpoints=False,
        auto_step_schedulers=False,
        overwrite_existing=True,
        train_loader=loader,
        val_loader=val_loader,
        extra_metrics={"zero": _metric},
    )
    builder = NNTrainerParamsBuilder.from_params(params)
    rebuilt = builder.build()
    assert _items(rebuilt.state()) == _items(params.state())
    for f in dataclasses.fields(NNTrainerParams):
        assert getattr(rebuilt, f.name) == getattr(params, f.name), f.name
    assert list(rebuilt.optims) == ["g", "d"]  # runtime insertion order kept
    assert rebuilt.optims["d"] is factory  # factory params keep identity
    assert rebuilt.train_loader is loader and rebuilt.val_loader is val_loader
    assert (rebuilt.extra_metrics or {})["zero"] is _metric

    # The rebuilt builder owns its collections: extending it leaves the source value alone.
    builder.optimizer("extra", _adam()).scheduler("d", _plateau())
    assert sorted(params.optims) == ["d", "g"] and sorted(params.schedulers) == ["g"]


def test_trainer_from_params_default_omission_matches_source():
    params = NNTrainerParams(n_epochs=1, optims={"main": _adam()})
    rebuilt = NNTrainerParamsBuilder.from_params(params).build()
    assert _items(rebuilt.state()) == _items(params.state())
    assert rebuilt.schedulers == {} and rebuilt.extra_metrics is None


def test_iterable_loaders_keep_identity_and_are_never_compared_or_copied():
    """Any iterable is a loader (NNModel accepts lists; arrays are iterable
    too): copy/from_params share it by identity and never compare it
    element-wise against the `None` default."""
    batches = [(torch.zeros(2, 4), torch.zeros(2, dtype=torch.long))]
    array_batches = np.zeros((2, 3))
    params = NNTrainerParams(
        n_epochs=1,
        optims={"m": _adam()},
        train_loader=batches,  # type: ignore[arg-type]
        val_loader=array_batches,  # type: ignore[arg-type]
    )
    rebuilt = NNTrainerParamsBuilder.from_params(params)
    for builder in (rebuilt, rebuilt.copy()):
        built = builder.build()
        assert built.train_loader is batches and built.val_loader is array_batches
    chained = NNTrainerParams.builder().n_epochs(1).optimizer("m", _adam())
    chained.train_loader(batches)  # type: ignore[arg-type]
    assert chained.copy().build().train_loader is batches


# ------------------------------------------------ from_params(): rejections


def test_optim_from_params_rejects_factory_params_precisely():
    factory = NNOptimFactoryParams(factory=OptimizerFactorySpec(id="tests.branching", version=1), max_lr=0.1)
    with pytest.raises(TypeError, match=r"NNOptimParamsBuilder\.from_params.*NNOptimFactoryParams"):
        NNOptimParamsBuilder.from_params(factory)  # type: ignore[arg-type]


def test_from_params_rejects_unsupported_subclasses_and_foreign_values():
    @dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
    class TaggedOptim(NNOptimParams):
        tag: str = "x"

    @dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
    class TaggedTransformer(NNTransformerParams):
        tag: str = "x"

    with pytest.raises(TypeError, match="TaggedOptim"):
        NNOptimParamsBuilder.from_params(TaggedOptim(name=Optims.SGD, max_lr=0.1, weight_decay=0.0, momentum=0.9))
    with pytest.raises(TypeError, match="TaggedTransformer"):
        NNTransformerParamsBuilder.from_params(TaggedTransformer(**_full_transformer_kwargs()))
    with pytest.raises(TypeError, match=r"NNTransformerParamsBuilder\.from_params.*NNParams"):
        NNTransformerParamsBuilder.from_params(
            NNParams(input_dim=2, output_dim=2, dropout_prob=0.0)  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match=r"NNSchedulerParamsBuilder\.from_params.*dict"):
        NNSchedulerParamsBuilder.from_params(_plateau().state())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"NNTrainerParamsBuilder\.from_params.*NNTrainParams"):
        NNTrainerParamsBuilder.from_params(NNTrainParams(n_epochs=1))  # type: ignore[arg-type]


def _full_transformer_kwargs() -> dict:
    p = _full_transformer()
    return {f.name: getattr(p, f.name) for f in dataclasses.fields(p) if f.init}


@pytest.mark.parametrize(
    ("builder_cls", "params_cls", "params"),
    [
        (NNOptimParamsBuilder, NNOptimParams, _adam()),
        (NNSchedulerParamsBuilder, NNSchedulerParams, _plateau()),
        (NNTransformerParamsBuilder, NNTransformerParams, _full_transformer()),
        (NNTrainerParamsBuilder, NNTrainerParams, NNTrainerParams(n_epochs=1, optims={"m": _adam()})),
    ],
    ids=["optim", "scheduler", "transformer", "trainer"],
)
def test_from_params_field_coverage_matches_the_dataclass(builder_cls, params_cls, params, monkeypatch):
    init_fields = {f.name for f in dataclasses.fields(params_cls) if f.init}
    # Drift guard: every init field is supported and `_dims` never is.
    assert set(builder_cls._PARAMS_FIELDS) == init_fields
    assert "_dims" not in builder_cls._PARAMS_FIELDS
    # A field the builder does not know about is reported by name.
    dropped = sorted(init_fields)[0]
    monkeypatch.setattr(builder_cls, "_PARAMS_FIELDS", tuple(n for n in builder_cls._PARAMS_FIELDS if n != dropped))
    with pytest.raises(ValueError, match=rf"unsupported field.*{dropped}"):
        builder_cls.from_params(params)


# ---------------------------------------------- setters and last-call rules


def test_setters_still_mutate_and_return_the_same_builder_after_branching():
    optim = NNOptimParams.builder().copy()
    assert optim.adam(max_lr=1e-3) is optim and optim.sgd(max_lr=0.1) is optim
    assert optim.build().name == Optims.SGD  # last variant wins
    sched = NNSchedulerParamsBuilder.from_params(_plateau())
    assert sched.step(step_size=2, min_lr=0.0, factor=0.5, patience=0, cooldown=0, threshold=0.0) is sched
    assert sched.build().kind == Schedulers.STEP
    transformer = NNTransformerParamsBuilder.from_params(_full_transformer()).copy()
    assert transformer.context(max_seq_len=8) is transformer
    assert transformer.build().rope_base == 10000.0  # `.context` last-call-wins reset
    trainer = NNTrainerParamsBuilder.from_params(NNTrainerParams(n_epochs=1, optims={"m": _adam()})).copy()
    assert trainer.n_epochs(5) is trainer and trainer.build().n_epochs == 5


# ------------------------------------------------------------- side effects


def test_copy_and_from_params_have_no_side_effects(monkeypatch):
    loader = _raising_loader()
    trainer_params = NNTrainerParams(
        n_epochs=1, optims={"m": _adam()}, train_loader=loader, extra_metrics={"zero": _metric}
    )
    transformer = _full_transformer()
    torch_state, np_state, py_state = torch.get_rng_state(), np.random.get_state(), random.getstate()

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("builder copy/from_params must not do I/O or build models")

    monkeypatch.setattr(builtins, "open", _forbidden)
    monkeypatch.setattr(socket, "socket", _forbidden)
    monkeypatch.setattr(torch.nn.Module, "__init__", _forbidden)

    for builder in (
        NNTrainerParamsBuilder.from_params(trainer_params),
        NNOptimParamsBuilder.from_params(_adam()),
        NNSchedulerParamsBuilder.from_params(_plateau()),
        NNTransformerParamsBuilder.from_params(transformer),
    ):
        builder.copy().build()

    monkeypatch.undo()
    assert torch.equal(torch_state, torch.get_rng_state())
    assert random.getstate() == py_state
    after = np.random.get_state()
    assert np_state[0] == after[0] and np.array_equal(np_state[1], after[1]) and np_state[2:] == after[2:]


# ------------------------------------- NNTransformerParams.from_state fix


def test_transformer_resolve_from_state_keeps_per_layer_lists():
    params = _full_transformer()
    state = params.state()
    assert "activations" in state and "dropout_probs" in state
    dispatched = NNParams.resolve_from_state(state)
    assert isinstance(dispatched, NNTransformerParams) and dispatched == params
    assert list(dispatched.activations or []) == [Activations.RELU, Activations.TANH]
    assert list(dispatched.dropout_probs or []) == [0.1, 0.3]


def test_transformer_from_state_legacy_states_without_list_keys_are_unchanged():
    legacy = NNTransformerParams.builder().vocab(32).layers(n=1, heads=2, d_model=8).context(max_seq_len=4).build()
    state = legacy.state()
    assert "activations" not in state and "dropout_probs" not in state
    restored = NNTransformerParams.from_state(state)
    assert restored == legacy and restored.activations is None and restored.dropout_probs is None


def test_transformer_per_layer_lists_survive_run_yaml_and_safetensors(tmp_path):
    params = NNTransformerParamsBuilder.from_params(_full_transformer()).build()
    edp = NNEvaluationDataPoint(loss=1.0, error=0.5, accuracy=0.5, f1=0.5, recall=0.5, precision=0.5)
    idp = NNIterationDataPoint(lr=1e-3, iter_idx=0, epoch_idx=0, batch_idx=0, train_edp=edp)
    model_params = NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)

    run = NNRun(net=params, train=NNTrainParams(n_epochs=1), model=model_params, idps=[idp])
    run.save(root=str(tmp_path))
    loaded = NNRun.load(run.id, root=str(tmp_path))
    assert loaded.net == params and loaded.id == run.id

    path = tmp_path / "ckpt.safetensors"
    NNCheckpoint(idp=idp, model_params=model_params, net_params=params, net_state={"w": torch.zeros(1)}).to_file(
        str(path), format="safetensors"
    )
    restored = NNCheckpoint.from_file(str(path))
    assert restored is not None and restored.net_params == params


# ------------------------------------------- sibling Trainer runs (CPU)


def _one_update(ctx: TrainerStepContext) -> NNEvaluationDataPoint:
    model = ctx.model
    model.net.train()
    for optimizer in ctx.optimizers.values():
        optimizer.zero_grad()
    (x,), y = model.net.unpack_batch(ctx.batch)
    loss = model.loss_fn(model.net(x), y)
    loss.backward()
    for optimizer in ctx.optimizers.values():
        optimizer.step()
    return NNEvaluationDataPoint(f1=0.0, recall=0.0, accuracy=0.0, precision=0.0, loss=float(loss.detach()), error=0.0)


def _tiny_model() -> NNModel:
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def test_sibling_trainer_branches_train_and_save_independently(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    set_seed(0)
    loader = DataLoader(TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,))), batch_size=8)
    base = NNTrainerParams.builder().n_epochs(1).train_loader(loader).optimizer("body", _adam())
    adam_branch = base.copy().data_id("branch-adam")
    sgd_branch = (
        base.copy()
        .data_id("branch-sgd")
        .optimizer("body", NNOptimParams.builder().sgd(max_lr=0.05).build())
        .scheduler("body", _step_sched())
    )
    configs = [adam_branch.build(), sgd_branch.build()]
    assert configs[0].optims["body"].name == Optims.ADAM and not configs[0].schedulers
    assert configs[1].optims["body"].name == Optims.SGD and sorted(configs[1].schedulers) == ["body"]

    runs = [Trainer(_tiny_model()).train(params=config, trainer_step_fn=_one_update) for config in configs]
    assert runs[0].id != runs[1].id
    for run, config in zip(runs, configs, strict=True):
        loaded = NNRun.load(run.id)
        assert loaded.trainer is not None
        assert loaded.trainer.state() == config.state()
        assert (tmp_path / "runs" / run.id).is_dir()
    # The shared base is untouched by either branch.
    base_built = base.build()
    assert base_built.data_id is None and base_built.optims["body"].name == Optims.ADAM
