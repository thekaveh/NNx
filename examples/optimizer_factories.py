"""Registered optimizer factories and AdamW (FEAT-013).

Demonstrates how a run uses an optimizer NNx does not ship, without
disguising it as a built-in name:

  1. **Register** a factory under a stable ``(id, version)``:
     ``register_optimizer_factory("examples.counting_sgd", 1, factory)``.
     The factory receives the *resolved* parameter groups — the ones a
     built-in optimizer would get, each with explicit ``lr`` /
     ``weight_decay`` and ownership already applied — plus a read-only
     JSON-like ``config``, and returns a
     ``torch.optim.Optimizer`` over exactly those parameters.
  2. **Reference** it from a run with ``NNOptimFactoryParams`` +
     ``OptimizerFactorySpec(id, version, config)``. Here a ``Trainer`` run
     gives the factory the ``layers.0.*`` group and a built-in
     ``Optims.ADAMW`` (decoupled weight decay) the disjoint ``layers.1.*``
     group; one CPU batch makes one update of each, and the factory is
     called exactly once.
  3. **Reload offline.** After unregistering the factory, ``NNRun.load``
     still restores the full run metadata from ``run.yaml`` — the spec is
     plain data, so no factory code runs. (Warm *resume* is different: it
     rebuilds the optimizer, so it needs the same id / version / config
     registered again.)

Fully offline, CPU only.

Run:
    python examples/optimizer_factories.py

The bounded ``optimizer_factories_workflow()`` helper is executed by
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
    NNOptimFactoryParams,
    NNOptimParams,
    NNParamGroupSpec,
    NNParams,
    NNRun,
    NNTrainerParams,
    OptimizerFactorySpec,
    Optims,
    Trainer,
    TrainerStepContext,
    register_optimizer_factory,
    set_seed,
    unregister_optimizer_factory,
)

FACTORY_ID = "examples.counting_sgd"


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


def optimizer_factories_workflow() -> None:
    calls: list[list[tuple[float, int]]] = []

    def counting_sgd(param_groups, config):
        # Resolved groups: every group carries explicit lr / weight_decay.
        calls.append([(g["lr"], len(g["params"])) for g in param_groups])
        return torch.optim.SGD(param_groups, momentum=config["momentum"], nesterov=config["nesterov"])

    register_optimizer_factory(FACTORY_ID, 1, counting_sgd, replace=True)
    try:
        set_seed(0)
        model = NNModel(
            net_params=NNParams(
                input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU
            ),
            params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        )
        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=8)  # one batch -> one update per optimizer

        body = NNOptimFactoryParams(
            factory=OptimizerFactorySpec(id=FACTORY_ID, version=1, config={"momentum": 0.9, "nesterov": True}),
            max_lr=0.05,
            param_groups=[NNParamGroupSpec(name_pattern="layers.0.*")],
        )
        head = (
            NNOptimParams.builder()
            .adamw(max_lr=1e-3, weight_decay=0.01)
            .param_groups([NNParamGroupSpec(name_pattern="layers.1.*")])
            .build()
        )
        before = {name: param.detach().clone() for name, param in model.net.named_parameters()}
        run = Trainer(model).train(
            params=NNTrainerParams(n_epochs=1, train_loader=loader, optims={"body": body, "head": head}),
            trainer_step_fn=_step,
        )

        # One factory call, one resolved group: layers.0.weight + layers.0.bias at max_lr.
        assert calls == [[(0.05, 2)]], calls
        # Both disjoint groups were stepped — the factory's and AdamW's.
        for name, param in model.net.named_parameters():
            assert not torch.equal(param.detach(), before[name]), f"{name} was not updated"
    finally:
        unregister_optimizer_factory(FACTORY_ID, 1)

    # Offline metadata reload: the factory is no longer registered, yet the
    # run (both the full trainer block and the representative train block)
    # decodes from run.yaml without calling any factory code.
    loaded = NNRun.load(run.id)
    assert loaded.id == run.id and loaded.state() == run.state()
    assert loaded.trainer is not None
    assert loaded.trainer.optims["body"] == body and loaded.trainer.optims["head"].name == Optims.ADAMW
    assert isinstance(loaded.train.optim, NNOptimFactoryParams)  # "body" sorts first
    assert len(calls) == 1
    print(f"trained 1 update with {body.factory} + adamw; reloaded offline: {loaded.train.optim}")


def main() -> None:
    optimizer_factories_workflow()


if __name__ == "__main__":
    main()
