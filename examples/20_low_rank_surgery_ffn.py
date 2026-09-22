"""Model surgery demo — low-rank factorize a Linear, retrain at lower rank.

Pipeline:

  1. Build + train a small classifier with a wide Linear layer.
  2. Snapshot FP32 val accuracy + Linear parameter count.
  3. Apply `nnx.surgery.low_rank_factorize(layer, rank=8)`.
     Replaces the named Linear with two stacked Linears (out×r, r×in).
     At max rank the factorization is exact; below max rank it is the
     SVD truncation — the bottleneck.
  4. Compare:
     - Total param count before vs after.
     - Validation accuracy before vs after.
  5. Briefly fine-tune the surgically modified model and verify accuracy
     recovers.

Note: `low_rank_factorize` takes a `nn.Linear` directly and returns a
`nn.Sequential` of two Linears. The caller is responsible for swapping
the layer back into the network's ModuleList.

``surgery_freeze_roles`` below is a bounded companion: surgery preserves
each replaced tensor's ``requires_grad`` and each replaced module's
train/eval mode, so a weight-frozen / bias-trainable layer factorizes into
frozen factors plus a trainable bias, and an optimizer built afterwards
from explicit parameter groups updates only what was trainable before.
Want the factors themselves to learn? ``unfreeze`` them explicitly before
creating the optimizer.

``widen_supported_workflow`` below is a second bounded companion for the
``widen`` primitive: it shows eval-mode parity through a direct Linear and
through an elementwise activation, the rejection of a width-dependent
Softmax boundary (source left untouched), and the honest reconstruction
path — widen ``model.net``, rebuild the *immutable params* with
``dataclasses.replace`` so the new width and every per-layer override stay
aligned, load the widened state into a fresh correctly-described
``NNModel``, train with a new optimizer, and reload BEST.

``deepen_override_workflow`` below is the ``deepen`` companion: starting
from explicit per-layer activation/dropout overrides it inserts one
identity layer, shows that the returned ``FeedFwdNN.params`` now describes
the deeper topology (one more hidden dim, aligned overrides, zero dropout
at the new site), rebuilds a fresh ``NNModel`` from ``deeper.params``,
fits one tiny epoch with a new optimizer and reloads BEST.

Run:
    pip install thekaveh-nnx
    python examples/20_low_rank_surgery_ffn.py
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    set_seed,
)
from nnx.finetune.param_groups import build_param_groups
from nnx.surgery import deepen, low_rank_factorize, widen


def _make_data():
    # No torch.manual_seed here — the caller does set_seed(42) before
    # calling us. Re-seeding torch inside this helper would silently
    # override the caller's seed (the same bug that PR #31's review
    # caught in examples 19 / 21 / 23).
    X = torch.randn(1024, 16)
    proj = torch.randn(16, 3)
    y = (X @ proj).argmax(dim=1)
    train = TensorDataset(X[:800], y[:800])
    val = TensorDataset(X[800:], y[800:])
    return DataLoader(train, batch_size=64, shuffle=True), DataLoader(val, batch_size=64)


def _param_count(net: torch.nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def _val_acc(net: torch.nn.Module, loader: DataLoader) -> float:
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            preds = net(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / total


def widen_supported_workflow() -> dict:
    """Bounded ``widen`` contract demonstration (writes ``runs/`` under the
    current working directory; the smoke test runs it in a temporary one).

    1. ``Linear -> ReLU -> Linear``: widening the first layer keeps the
       eval-mode forward identical (Net2WiderNet through an elementwise op).
    2. ``Linear -> Softmax -> Linear``: rejected with a ``ValueError``
       naming the width-dependent op; the source model is untouched.
    3. A ``FeedFwdNN`` model with per-layer activation overrides: widen
       ``layers.0``, rebuild the params with ``dataclasses.replace`` (new
       width, overrides preserved), load the widened state into a fresh
       ``NNModel``, verify parity, train one epoch with a new optimizer
       and reload the BEST checkpoint for a finite evaluation.
    """
    from dataclasses import replace

    from nnx import Checkpoints, NNCheckpoint

    set_seed(0)
    x = torch.randn(6, 8)

    plain = torch.nn.Sequential(torch.nn.Linear(8, 6), torch.nn.ReLU(), torch.nn.Linear(6, 3)).eval()
    wider = widen(plain, layer_name="0", new_width=10)
    assert wider[0].out_features == 10 and wider[2].in_features == 10
    assert torch.allclose(plain(x), wider(x), atol=1e-5), "widen through ReLU must preserve the forward"

    softmax_net = torch.nn.Sequential(torch.nn.Linear(8, 6), torch.nn.Softmax(dim=-1), torch.nn.Linear(6, 3))
    before = {k: v.clone() for k, v in softmax_net.state_dict().items()}
    try:
        widen(softmax_net, layer_name="0", new_width=10)
    except ValueError as exc:
        assert "Softmax" in str(exc), exc
    else:
        raise AssertionError("a width-dependent Softmax boundary must be rejected")
    assert all(torch.equal(before[k], v) for k, v in softmax_net.state_dict().items()), (
        "rejection must not mutate the source"
    )

    net_params = NNParams(
        input_dim=8,
        output_dim=3,
        hidden_dims=[8, 6],
        dropout_prob=0.0,
        activation=Activations.RELU,
        activations=[Activations.TANH, Activations.RELU],  # per-layer overrides must survive the rebuild
    )
    model_params = NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    model = NNModel(net_params=net_params, params=model_params)
    model.net.eval()
    wider_net = widen(model.net, layer_name="layers.0", new_width=12)
    rebuilt_params = replace(net_params, hidden_dims=[12, 6])  # width changes; activations stay aligned
    assert rebuilt_params.activation_for(0) is Activations.TANH
    refined = NNModel(net_params=rebuilt_params, params=model_params)
    refined.net.load_state_dict(wider_net.state_dict())
    refined.net.eval()
    with torch.no_grad():
        assert torch.allclose(model.net(x), refined.net(x), atol=1e-5), "fresh correctly-described model must match"

    y = torch.randint(0, 3, (6,))
    loader = DataLoader(TensorDataset(x, y), batch_size=3)
    run = refined.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=loader,
            val_loader=loader,
            optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0),
            scheduler=NNSchedulerParams(min_lr=1e-6, factor=0.5, patience=2, cooldown=1, threshold=1e-3),
        )
    )
    best = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert best is not None
    reloaded = NNModel.from_checkpoint(checkpoint=best)
    assert reloaded.net_params.hidden_dims == [12, 6]
    loss = reloaded.evaluate(loader).loss
    assert loss is not None and torch.isfinite(torch.tensor(loss))

    summary = {"widened_hidden_dims": rebuilt_params.hidden_dims, "best_eval_loss": float(loss), "run_id": run.id}
    print(f"widen supported-workflow: {summary}")
    return summary


def deepen_override_workflow() -> dict:
    """Bounded ``deepen`` contract demonstration (writes ``runs/`` under the
    current working directory; the smoke test runs it in a temporary one).

    A ``FeedFwdNN`` with hidden dims ``[8, 6]``, explicit ReLU overrides
    and mixed dropout ``[0.2, 0.0]`` is deepened after ``layers.0``. The
    returned network's ``params`` must describe the new topology — hidden
    ``[8, 8, 6]``, three ReLU entries, dropout ``[0.2, 0.0, 0.0]`` with
    zero at the new identity site — and the eval forward must match the
    original. A fresh ``NNModel`` is then built from ``deeper.params``
    (never from the old descriptor), the deeper state is loaded, one
    tiny epoch trains with a new optimizer, and BEST reloads with the
    deeper descriptor for a finite evaluation. The original params and
    network are left unchanged.
    """
    from nnx import Checkpoints, NNCheckpoint

    set_seed(0)
    x = torch.randn(6, 8)
    y = torch.randint(0, 3, (6,))

    net_params = NNParams(
        input_dim=8,
        output_dim=3,
        hidden_dims=[8, 6],
        dropout_prob=0.0,
        activation=Activations.RELU,
        activations=[Activations.RELU, Activations.RELU],
        dropout_probs=[0.2, 0.0],
    )
    model_params = NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    model = NNModel(net_params=net_params, params=model_params)
    model.net.eval()

    deeper_net = deepen(model.net, after_layer_name="layers.0")
    deeper_net.eval()
    with torch.no_grad():
        assert torch.allclose(model.net(x), deeper_net(x), atol=1e-5), "deepen must preserve the eval forward"
    new_params = deeper_net.params  # the transformed descriptor travels with the network
    assert list(new_params.hidden_dims) == [8, 8, 6], list(new_params.hidden_dims)
    assert list(new_params.activations) == [Activations.RELU] * 3
    assert list(new_params.dropout_probs) == [0.2, 0.0, 0.0], list(new_params.dropout_probs)
    assert list(net_params.hidden_dims) == [8, 6] and len(model.net.layers) == 3, "source untouched"

    refined = NNModel(net_params=new_params, params=model_params)  # fresh, correctly described
    refined.net.load_state_dict(deeper_net.state_dict())
    refined.net.eval()
    with torch.no_grad():
        assert torch.allclose(model.net(x), refined.net(x), atol=1e-5)

    loader = DataLoader(TensorDataset(x, y), batch_size=3)
    run = refined.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=loader,
            val_loader=loader,
            optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0),
            scheduler=NNSchedulerParams(min_lr=1e-6, factor=0.5, patience=2, cooldown=1, threshold=1e-3),
        )
    )
    best = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert best is not None
    reloaded = NNModel.from_checkpoint(checkpoint=best)
    assert list(reloaded.net_params.hidden_dims) == [8, 8, 6]
    assert [reloaded.net_params.dropout_for(i) for i in range(3)] == [0.2, 0.0, 0.0]
    loss = reloaded.evaluate(loader).loss
    assert loss is not None and torch.isfinite(torch.tensor(loss))

    summary = {
        "hidden_dims": list(new_params.hidden_dims),
        "dropout_probs": list(new_params.dropout_probs),
        "run_id": run.id,
    }
    print(f"deepen override workflow: {summary}")
    return summary


def surgery_freeze_roles() -> dict:
    """Bounded demonstration that surgery preserves trainability roles.

    Builds ``Linear(8, 16) -> ReLU -> Linear(16, 3)``, freezes the first
    layer's weight but leaves its bias trainable, factorizes that layer at
    rank 8 and reattaches the factors. Then it builds *fresh* explicit
    parameter groups (``strict=True``, so frozen tensors cannot enter),
    takes one SGD step and checks: both factor weights are frozen and
    unchanged, the up-projection bias (the only intended trainable role
    in that layer) moved, the untouched second layer trained normally,
    and the source Linear's parameters are unchanged. No temporary files.
    """
    from nnx import NNParamGroupSpec

    set_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
    source = net[0]
    source.weight.requires_grad_(False)
    source.bias.requires_grad_(True)
    source_snapshot = {n: p.detach().clone() for n, p in source.named_parameters()}

    factors = low_rank_factorize(source, rank=8)
    net[0] = factors  # reattach — low_rank_factorize returns a NEW nn.Sequential
    down, up = factors[0], factors[1]
    assert not down.weight.requires_grad and not up.weight.requires_grad, "factor weights inherit the frozen role"
    assert up.bias is not None and up.bias.requires_grad, "the bias keeps its own trainable role"
    assert factors.training == source.training

    groups = build_param_groups(
        net,
        [NNParamGroupSpec(name_pattern="0.*", lr=1e-2), NNParamGroupSpec(name_pattern="2.*", lr=1e-3)],
        default_lr=1e-3,
        default_weight_decay=0.0,
        strict=True,
    )
    grouped = {id(p) for g in groups for p in g["params"]}
    assert grouped == {id(up.bias), id(net[2].weight), id(net[2].bias)}, "frozen factors must not enter the groups"

    before = {n: p.detach().clone() for n, p in net.named_parameters()}
    optimizer = torch.optim.SGD(groups)
    x = torch.randn(5, 8)
    loss = net(x).square().mean()
    loss.backward()
    optimizer.step()

    assert torch.equal(down.weight.detach(), before["0.0.weight"]) and torch.equal(
        up.weight.detach(), before["0.1.weight"]
    )
    assert not torch.equal(up.bias.detach(), before["0.1.bias"]), "the intended trainable bias must move"
    assert not torch.equal(net[2].weight.detach(), before["2.weight"])
    assert all(torch.equal(p.detach(), source_snapshot[n]) for n, p in source.named_parameters()), "source untouched"

    summary = {
        "trainable_after_surgery": sorted(n for n, p in net.named_parameters() if p.requires_grad),
        "loss": float(loss.detach()),
    }
    print(f"surgery freeze-roles workflow: {summary}")
    return summary


def main() -> None:
    set_seed(42)
    train_loader, val_loader = _make_data()

    # Wide hidden_dims so the factorization has compressible structure.
    # FeedFwdNN with hidden_dims=[64, 128, 32] yields:
    #   layers.0: Linear(16 → 64)
    #   layers.1: Linear(64 → 128)   ← widest; factorize this one
    #   layers.2: Linear(128 → 32)
    #   layers.3: Linear(32 → 3)
    net_params = NNParams(
        input_dim=16,
        output_dim=3,
        hidden_dims=[64, 128, 32],
        dropout_prob=0.0,
        activation=Activations.RELU,
    )
    model_params = NNModelParams(
        net=Nets.FEED_FWD,
        device=Devices.CPU,
        loss=Losses.CROSS_ENTROPY,
    )
    train_params = NNTrainParams(
        n_epochs=5,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=NNOptimParams(
            name=Optims.ADAM,
            max_lr=1e-2,
            momentum=(0.9, 0.999),
            weight_decay=0.0,
        ),
        scheduler=NNSchedulerParams(
            min_lr=1e-6,
            factor=0.5,
            patience=2,
            cooldown=1,
            threshold=1e-3,
        ),
    )

    print("─── Phase 1: train the wide net ───")
    model = NNModel(net_params=net_params, params=model_params)
    model.train(params=train_params)
    print(f"FP32 val accuracy: {_val_acc(model.net, val_loader):.3f}")
    print(f"FP32 params:       {_param_count(model.net):,}")

    print("─── Phase 2: low-rank factorize the widest Linear at rank=8 ───")
    # FeedFwdNN stores Linears in a ModuleList; layers.1 is the 64→128 Linear.
    # low_rank_factorize takes the nn.Linear directly and returns nn.Sequential.
    target_linear = model.net.layers[1]
    factored = low_rank_factorize(target_linear, rank=8)
    model.net.layers[1] = factored
    print(f"Surgically reduced rank: 8 (max was {min(target_linear.in_features, target_linear.out_features)})")
    print(f"Post-surgery params:    {_param_count(model.net):,}")
    print(f"Post-surgery val acc:    {_val_acc(model.net, val_loader):.3f}  # expect drop before refinement")

    print("─── Phase 3: refine to recover accuracy ───")
    # Topology-changing surgery no longer matches net_params, so use a
    # manual refinement loop and export_state_dict() for persistence.
    optimizer = torch.optim.Adam(model.net.parameters(), lr=5e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.net.train()
    for _ in range(3):
        for features, labels in train_loader:
            optimizer.zero_grad()
            loss_fn(model.net(features), labels).backward()
            optimizer.step()
    print(f"Refined val accuracy:   {_val_acc(model.net, val_loader):.3f}")


if __name__ == "__main__":
    main()
