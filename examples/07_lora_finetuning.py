"""LoRA fine-tuning — parameter-efficient adaptation of a pretrained
classifier.

Flow:

  1. Pretrain a small classifier on distribution A.
  2. Wrap every :class:`nn.Linear` in the trained net with
     :class:`LoRALinear`. The wrap freezes each base layer's
     full-rank weight; only the new ``lora_A`` and ``lora_B``
     matrices (~r/dim of the original size) train.
  3. Fine-tune on distribution B.
  4. Verify every base ``weight`` / ``bias`` is BIT-EXACTLY unchanged
     and LoRA params have moved.
  5. Save a LoRA-only checkpoint and compare its size to a full
     state-dict snapshot.

The point isn't accuracy comparison — for a toy task, full fine-tuning
and LoRA fine-tuning land near the same val error. The point is
demonstrating PEFT's storage and update efficiency: a few hundred new
parameters per layer instead of tens of thousands.

``dora_zero_row_composition`` below is a bounded DoRA companion: a wrapped
layer with mixed zero / nonzero base rows feeding a following projection,
converted to half after wrapping. DoRA normalizes rows in FP32 for
FP16/BF16, so zero rows stay finite (equal to the base) and learned
updates remain differentiable; the helper takes one optimizer step on a
float32 loss reduction, checks the frozen base and round-trips the full
``state_dict()`` (the LoRA-only helpers omit ``magnitude``).

Run:
    python examples/07_lora_finetuning.py
"""

from __future__ import annotations

import os
import tempfile

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
    apply_dora_to,
    apply_lora_to,
    save_lora_weights,
    set_seed,
)


def _classifier() -> NNModel:
    return NNModel(
        net_params=NNParams(
            input_dim=8,
            output_dim=4,
            hidden_dims=[32, 32],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )


def _loader(seed: int, n: int = 256) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(4, 8, generator=g) * 1.5
    cls = torch.randint(0, 4, (n,), generator=g)
    X = means[cls] + 0.4 * torch.randn(n, 8, generator=g)
    return DataLoader(TensorDataset(X, cls), batch_size=32, shuffle=True)


def _train_params(n_epochs: int, train_loader, lr: float = 1e-2, data_id: str | None = None):
    # `data_id` keeps the pretraining run and the fine-tuning run distinct:
    # loaders are not part of run.id, so two phases with identical params
    # would otherwise collide on the same run directory.
    return NNTrainParams(
        n_epochs=n_epochs,
        data_id=data_id,
        train_loader=train_loader,
        optim=NNOptimParams(
            name=Optims.ADAM,
            max_lr=lr,
            momentum=(0.9, 0.999),
            weight_decay=0.0,
        ),
        scheduler=NNSchedulerParams(
            min_lr=1e-7,
            factor=0.5,
            patience=2,
            cooldown=1,
            threshold=1e-3,
        ),
    )


def dora_zero_row_composition(dtype: torch.dtype = torch.float16) -> dict:
    """Bounded DoRA composition with zero rows in reduced precision.

    Builds ``Linear(8, 6) -> ReLU -> Linear(6, 3)``, zeroes two rows of
    the first weight (as pruning or explicit zero init would), wraps both
    layers with ``apply_dora_to`` and converts the *whole* net to ``dtype``
    afterwards (so this exercises the normalization path, not parameter
    placement). Checks a finite forward through the following projection
    with the zero rows still producing the base's zeros, one SGD step on
    a float32 loss reduction (half MSE backward is not supported on CPU)
    that moves only adapter/magnitude parameters, the bit-exact frozen
    base, and a complete ``state_dict()`` reload including ``magnitude``.
    Runs on CPU in FP32 and FP16; BF16/CUDA lanes are the same call with
    a different ``dtype``.
    """
    set_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(8, 6), torch.nn.ReLU(), torch.nn.Linear(6, 3))
    with torch.no_grad():
        net[0].weight[0].zero_()
        net[0].weight[1].zero_()
    n_wrapped = apply_dora_to(net, "*", r=2, alpha=4.0)
    assert n_wrapped == 2, n_wrapped
    net = net.to(dtype)
    base_snapshot = {n: p.detach().clone() for n, p in net.named_parameters() if not p.requires_grad}

    x = torch.randn(4, 8).to(dtype)
    hidden = net[0](x)
    assert hidden.dtype == dtype and torch.isfinite(hidden).all(), "zero rows must not produce NaN"
    assert torch.equal(hidden[:, :2], net[0].base(x)[:, :2]), "zero rows keep the base's output at init"
    out = net(x)
    assert out.dtype == dtype and torch.isfinite(out).all()

    trainable = [p for p in net.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in trainable]
    optimizer = torch.optim.SGD(trainable, lr=0.05)
    loss = (net(x).float() ** 2).mean()  # float32 reduction: supported half backward on CPU
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
    optimizer.step()
    assert any(not torch.equal(a, b.detach()) for a, b in zip(before, trainable, strict=True)), "adapter must move"
    assert all(torch.equal(p.detach(), base_snapshot[n]) for n, p in net.named_parameters() if n in base_snapshot)

    reloaded = torch.nn.Sequential(torch.nn.Linear(8, 6), torch.nn.ReLU(), torch.nn.Linear(6, 3))
    apply_dora_to(reloaded, "*", r=2, alpha=4.0)
    reloaded = reloaded.to(dtype)
    state = net.state_dict()
    assert "0.magnitude" in state and "2.magnitude" in state
    reloaded.load_state_dict(state)
    assert torch.equal(reloaded(x), net(x)) and torch.isfinite(reloaded(x)).all()

    summary = {"dtype": str(dtype), "wrapped": n_wrapped, "loss": float(loss.detach()), "state_keys": len(state)}
    print(f"DoRA zero-row composition workflow: {summary}")
    return summary


def main():
    set_seed(0)

    # ---- Phase 1: pretrain on distribution A.
    print("=" * 60)
    print("Phase 1: pretraining on distribution A")
    print("=" * 60)
    model = _classifier()
    pre_total = sum(p.numel() for p in model.net.parameters())
    print(f"net: {pre_total} total parameters\n")
    model.train(params=_train_params(5, _loader(seed=0), data_id="pretrain-A"))

    # Snapshot every parameter for the strict equality check after
    # LoRA fine-tuning. We snapshot by name BEFORE wrapping; after
    # apply_lora_to the names will have `.base.` inserted.
    pretrain_snapshot = {n: p.clone() for n, p in model.net.named_parameters()}

    # ---- Phase 2: wrap every Linear with LoRA.
    print("\n" + "=" * 60)
    print("Phase 2: wrapping with LoRA (r=4, alpha=8)")
    print("=" * 60)
    n_wrapped = apply_lora_to(model.net, "layers.*", r=4, alpha=8.0)
    print(f"wrapped {n_wrapped} Linear layers")

    # Count trainable parameters now. The base layers are frozen, so
    # only the LoRA A/B matrices remain trainable.
    trainable = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    print(f"trainable params: {trainable} / {pre_total} ({trainable * 100 / pre_total:.1f}%)")

    # ---- Phase 3: fine-tune on distribution B.
    print("\n" + "=" * 60)
    print("Phase 3: LoRA fine-tuning on distribution B")
    print("=" * 60)
    set_seed(1)
    model.train(params=_train_params(5, _loader(seed=42), data_id="finetune-B"))

    # ---- Phase 4: verify the PEFT contract.
    print("\n" + "=" * 60)
    print("Phase 4: verifying base-frozen invariant")
    print("=" * 60)
    drifted = []
    for n, post in model.net.named_parameters():
        if "lora_" in n:
            continue
        # apply_lora_to inserted a single `.base.` segment into every
        # wrapped layer's parameter name (e.g. `layers.0.weight` →
        # `layers.0.base.weight`), so strip it to recover the pre-wrap key.
        pre_key = n.replace(".base.", ".")
        if not torch.equal(post.detach(), pretrain_snapshot[pre_key]):
            drifted.append(n)
    if drifted:
        raise RuntimeError(f"base parameters drifted during LoRA fine-tuning: {drifted}")
    print("every base parameter is bit-exactly unchanged after fine-tuning")

    # ---- Phase 5: save LoRA-only checkpoint, compare sizes.
    print("\n" + "=" * 60)
    print("Phase 5: saving LoRA-only checkpoint")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as tmp:
        lora_path = os.path.join(tmp, "lora.pt")
        full_path = os.path.join(tmp, "full.pt")
        save_lora_weights(model.net, lora_path)
        torch.save(model.net.state_dict(), full_path)
        lora_size = os.path.getsize(lora_path)
        full_size = os.path.getsize(full_path)
        print(f"LoRA-only:  {lora_size:>8} bytes")
        print(f"full state: {full_size:>8} bytes")
        print(f"LoRA is {lora_size * 100 / full_size:.1f}% the size of the full state")


if __name__ == "__main__":
    main()
