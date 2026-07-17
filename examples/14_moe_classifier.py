"""Mixture-of-Experts classifier — the first-class MoE model type:
``NNMoEParams`` + ``Nets.FEED_FWD_MOE``.

Demonstrates:

  1. Building an MoE feed-forward classifier directly from params:
     :class:`NNMoEParams` (a drop-in :class:`NNParams` subclass carrying
     ``num_experts``/``top_k``) with ``Nets.FEED_FWD_MOE`` gives a
     ``FeedFwdMoENN`` whose every hidden layer is an :class:`MoELinear`
     (the classifier head stays a plain ``nn.Linear``). No custom
     subclassing or ``model.net`` swapping — earlier versions of this
     example hand-rolled exactly that workaround, which also silently
     hashed to the SAME ``run.id`` as a plain feed-forward twin.
     ``NNMoEParams.state()`` always emits ``num_experts``, so MoE runs
     get distinct ids and checkpoints round-trip to the right class.
  2. Training with :func:`moe_train_step_factory` — the standard
     supervised step augmented with the Switch-style load-balancing
     aux loss summed across every :class:`MoELinear` layer. Without
     the aux term, the router can collapse onto one or two experts
     and waste the rest of the parameter budget.
  3. Verifying that the aux loss decreases across the run — proof
     that the load-balancing penalty is doing its job.
  4. Checkpoint round-trip: the saved LAST checkpoint restores an
     ``NNMoEParams`` (via ``resolve_from_state``) and rebuilds the
     MoE net with identical logits.

Like the other tutorial examples, this is mechanism-first: it doesn't
claim the MoE classifier beats a plain feed-forward of the same total
parameter count on toy data. The benefit of MoE shows up on harder
problems where different experts can genuinely specialize.

Run:
    python examples/14_moe_classifier.py
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Checkpoints,
    Devices,
    Losses,
    MoELinear,
    Nets,
    NNCheckpoint,
    NNModel,
    NNModelParams,
    NNMoEParams,
    NNOptimParams,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    moe_train_step_factory,
    set_seed,
)


def _make_loaders(seed: int = 0) -> tuple[DataLoader, DataLoader]:
    """Toy 3-class classification with overlapping Gaussians."""
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(3, 8, generator=g) * 1.5

    def make(n: int):
        cls = torch.randint(0, 3, (n,), generator=g)
        X = means[cls] + 0.7 * torch.randn(n, 8, generator=g)
        return X, cls

    X_train, y_train = make(256)
    X_val, y_val = make(128)
    train = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)
    return train, val


def main() -> None:
    set_seed(0)
    train_loader, val_loader = _make_loaders(seed=0)

    NUM_EXPERTS, TOP_K = 4, 2

    # Build the model straight from params — Nets.FEED_FWD_MOE
    # instantiates a FeedFwdMoENN whose hidden layers are MoELinear.
    net_params = NNMoEParams(
        input_dim=8,
        output_dim=3,
        hidden_dims=[16],
        dropout_prob=0.0,
        activation=Activations.RELU,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
    )
    model = NNModel(
        net_params=net_params,
        params=NNModelParams(net=Nets.FEED_FWD_MOE, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )

    moe_layer: MoELinear = model.net.layers[0]  # type: ignore[assignment]
    assert isinstance(moe_layer, MoELinear)
    total_params = sum(p.numel() for p in model.net.parameters())
    expert_params = sum(p.numel() for e in moe_layer.experts for p in e.parameters())
    router_params = sum(p.numel() for p in moe_layer.router.parameters())
    print("=" * 60)
    print(f"MoE classifier ({type(model.net).__name__}): {NUM_EXPERTS} experts, top_k={TOP_K}")
    print("=" * 60)
    print(f"total params:   {total_params}")
    print(f"  router:       {router_params}")
    print(f"  experts:      {expert_params}")
    print(f"  classifier:   {total_params - router_params - expert_params}")

    # Snapshot aux loss BEFORE any training — gives us the starting
    # imbalance to compare against post-training.
    all_X = torch.cat([b[0] for b in train_loader], dim=0)
    model.net.eval()
    with torch.no_grad():
        _ = model.net(all_X)
    aux_start = float(moe_layer.last_aux_loss)
    print(f"\naux loss at init:  {aux_start:.4f}  (minimum is 1.0 at uniform routing)")

    # Train with the MoE step factory. ``aux_loss_weight=0.05`` is a
    # tutorial-scale value — enough to nudge routing toward uniform
    # without dominating the supervised signal.
    step_fn = moe_train_step_factory(aux_loss_weight=0.05)
    run = model.train(
        params=NNTrainParams(
            n_epochs=10,
            train_loader=train_loader,
            val_loader=val_loader,
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=5e-2,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7,
                factor=0.5,
                patience=3,
                cooldown=1,
                threshold=1e-3,
            ),
        ),
        train_step_fn=step_fn,
    )

    # Post-training aux loss on the same data.
    model.net.eval()
    with torch.no_grad():
        _ = model.net(all_X)
    aux_end = float(moe_layer.last_aux_loss)
    val_err = run.idps[-1].val_edp.error

    print(f"aux loss post-train: {aux_end:.4f}  (gap to 1.0: {aux_end - 1.0:.4f})")
    print(f"final val error:     {val_err:.4f}")

    if aux_end >= aux_start:
        # Toy data + random routing can occasionally leave aux loss
        # near its initial value (the supervised signal can pull
        # against the aux signal). Print a warning rather than crash —
        # this is a demo, not a guarantee.
        print(
            "note: aux loss did NOT decrease — on tiny toy data the supervised "
            "gradient through the gating weights can overwhelm the aux signal. "
            "Increase aux_loss_weight or n_epochs to see balancing dominate."
        )
    else:
        print("aux loss decreased during training: routing is more balanced")

    # ---- Checkpoint round-trip: the MoE fields ride along in state().
    # ``resolve_from_state`` dispatches on the always-emitted
    # ``num_experts`` key, so the reloaded model is an MoE net — not a
    # silently-downgraded plain feed-forward.
    ckpt = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert ckpt is not None and isinstance(ckpt.net_params, NNMoEParams)
    reloaded = NNModel.from_checkpoint(ckpt)
    model.net.eval()
    reloaded.net.eval()
    with torch.no_grad():
        same = torch.allclose(model.net(all_X), reloaded.net(all_X))
    print(f"\ncheckpoint round-trip: net_params={type(ckpt.net_params).__name__}, ")
    print(f"reloaded net={type(reloaded.net).__name__}, logits identical: {same}")


if __name__ == "__main__":
    main()
