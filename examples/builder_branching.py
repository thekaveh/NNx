"""Branching mutable params builders with copy() and from_params() (FEAT-041).

The four params builders are mutable and reusable. Two methods let a run
share its setup instead of retyping it per variant:

  1. **copy()** branches a builder — complete or still partial. After
     the branch, setters on either builder never affect the other (nor
     values already built); loaders and metric callables stay shared by
     identity, and nothing is iterated, built or written.
  2. **from_params(params)** rebuilds a builder from an existing params
     value, so ``from_params(p).build() == p`` with the same ``state()``,
     and the builder can then be varied like any hand-written chain.

This script branches an optimizer builder into an AdamW and an SGD
variant, branches a named ``Trainer`` configuration into two sibling
runs with distinct ``data_id`` s (one extra scheduler on one branch),
trains each for one CPU update, reloads both runs' metadata, and checks
that a transformer config — per-layer ``activations`` / ``dropout_probs``
included — round-trips through ``NNTransformerParamsBuilder.from_params``
and ``NNTransformerParams.from_state``.

Fully offline, CPU only.

Run:
    python examples/builder_branching.py

The bounded ``builder_branching_workflow()`` helper is executed by
``tests/test_examples_smoke.py`` in a temporary working directory.
"""

from __future__ import annotations

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
    NNOptimParams,
    NNOptimParamsBuilder,
    NNParamGroupSpec,
    NNParams,
    NNRun,
    NNSchedulerParams,
    NNTrainerParams,
    NNTransformerParams,
    NNTransformerParamsBuilder,
    Optims,
    Trainer,
    TrainerStepContext,
    set_seed,
)


def _step(ctx: TrainerStepContext) -> NNEvaluationDataPoint:
    """One supervised update that steps every optimizer over its own group."""
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


def _model() -> NNModel:
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def builder_branching_workflow() -> None:
    # 1. Optimizer branches: a shared, partial base (clipping + a head group)
    #    that each branch completes with its own variant.
    base_optim = NNOptimParams.builder().grad_clip(1.0).param_groups([NNParamGroupSpec(name_pattern="layers.1.*")])
    adamw = base_optim.copy().adamw(max_lr=1e-3, weight_decay=0.01).build()
    sgd = base_optim.copy().sgd(max_lr=0.05).build()
    assert (adamw.name, sgd.name) == (Optims.ADAMW, Optims.SGD)
    assert adamw.grad_clip_norm == sgd.grad_clip_norm == 1.0
    assert "name" not in base_optim._fields  # the partial base is untouched
    # from_params rebuilds an equal builder that can then be varied.
    assert NNOptimParamsBuilder.from_params(adamw).build() == adamw
    assert NNOptimParamsBuilder.from_params(adamw).accumulate_grad(2).build().accumulate_grad_batches == 2

    # 2. Named Trainer branches: one shared base, two sibling runs.
    set_seed(0)
    loader = DataLoader(TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,))), batch_size=8)
    base = NNTrainerParams.builder().n_epochs(1).train_loader(loader).optimizer("head", adamw)
    step_decay = (
        NNSchedulerParams.builder()
        .step(step_size=1, min_lr=0.0, factor=0.5, patience=0, cooldown=0, threshold=0.0)
        .build()
    )
    plain = base.copy().data_id("branch-plain").build()
    scheduled = base.copy().data_id("branch-scheduled").optimizer("head", sgd).scheduler("head", step_decay).build()
    assert plain.train_loader is loader and scheduled.train_loader is loader  # shared, never copied

    runs = [Trainer(_model()).train(params=config, trainer_step_fn=_step) for config in (plain, scheduled)]
    assert runs[0].id != runs[1].id
    for run, config in zip(runs, (plain, scheduled), strict=True):
        loaded = NNRun.load(run.id)
        assert loaded.trainer is not None and loaded.trainer.state() == config.state()
    assert base.build().data_id is None  # neither branch leaked back

    # 3. Transformer round trip, inherited per-layer lists included.
    lm = NNTransformerParams(
        input_dim=64,
        output_dim=64,
        dropout_prob=0.0,
        hidden_dims=[7, 5],
        activations=[Activations.RELU, Activations.TANH],
        dropout_probs=[0.1, 0.3],
        n_heads=2,
        vocab_size=64,
        n_layers=2,
        d_model=16,
        max_seq_len=32,
    )
    assert NNTransformerParamsBuilder.from_params(lm).build() == lm
    assert NNTransformerParams.from_state(lm.state()) == lm
    # `.context()` sets max_seq_len AND rope_base (an omitted rope_base resets
    # to 10000.0), so pass the carried value again to keep it.
    longer = NNTransformerParamsBuilder.from_params(lm).context(max_seq_len=128, rope_base=lm.rope_base).build()
    assert longer.max_seq_len == 128 and longer.rope_base == lm.rope_base and longer.activations == lm.activations

    print(f"optimizer branches: {adamw.name} / {sgd.name}; sibling runs: {runs[0].id}, {runs[1].id}")


def main() -> None:
    builder_branching_workflow()


if __name__ == "__main__":
    main()
