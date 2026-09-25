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

``peft_preconverted_base`` below shows placement: adapters are allocated
with the wrapped weight's dtype and device, so a base converted to
float64 (or moved to an accelerator) *before* ``apply_lora_to`` composes
immediately — one adapter-only SGD step, no second ``.to()``, base
bit-exactly unchanged.

``peft_eval_injection`` below covers deployment order: the classifier is
put in ``eval()`` *before* adapters with nonzero dropout are injected and
trained weights are loaded. Each wrapper inherits the mode of the layer
it wraps, so repeated ``predict`` calls stay bit-identical and the
per-module mode map is unchanged; a later ``train()`` still re-enables
the adapter dropout.

``peft_alias_preflight`` below shows the shared-module boundary: a Linear
registered under two names cannot be wrapped without splitting it, so the
apply helpers reject it (naming both aliases) before building any wrapper,
and an independent target can still be wrapped and trained.

``lora_artifact_roundtrip`` below closes the loop the main flow leaves
open: it inspects the exact keys ``save_lora_weights`` writes (adapter
ownership, never a name substring — a layer named ``lora_A_projection``
does not leak its base), reloads the artifact into a freshly wrapped
matching base and checks the load count, output parity and base
identity. An adapter file is *not* a resumable model checkpoint: it
carries only the trainable delta; the base weights come from wherever
the base model is loaded from.

Run:
    python examples/07_lora_finetuning.py
"""

from __future__ import annotations

import os
import tempfile
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    LoRALinear,
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
    load_lora_weights,
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


def _lora_owned_params(net: torch.nn.Module) -> set[int]:
    """Identities of the parameters the LoRA wrappers in ``net`` own —
    the ownership-aware alternative to matching ``"lora_" in name``,
    which a colliding base name could satisfy."""
    return {id(p) for m in net.modules() if isinstance(m, LoRALinear) for p in (m.lora_A, m.lora_B)}


def lora_artifact_roundtrip() -> dict:
    """Bounded adapter-artifact demonstration (writes one file under a
    temporary directory).

    Wraps a tiny classifier whose first layer is deliberately *named*
    ``lora_A_projection`` (a name that contains the LoRA marker), moves
    the adapters off their zero init, saves the LoRA-only artifact and
    asserts its exact key set: the two adapter tensors per wrapped layer
    and nothing else — no ``*.base.*`` tensors. Then it rebuilds a
    matching wrapped destination from the same base weights, loads the
    artifact, and checks the load count, bit-exact output parity, and
    that the destination's base tensors are the same objects with the
    same values as before the load.
    """
    set_seed(0)

    def _build() -> torch.nn.Module:
        return torch.nn.Sequential(
            OrderedDict(
                [
                    ("lora_A_projection", torch.nn.Linear(8, 6)),
                    ("act", torch.nn.ReLU()),
                    ("head", torch.nn.Linear(6, 3)),
                ]
            )
        )

    source = _build()
    base_state = {k: v.detach().clone() for k, v in source.state_dict().items()}
    n_wrapped = apply_lora_to(source, "*", r=2, alpha=4.0)
    assert n_wrapped == 2, n_wrapped
    with torch.no_grad():
        for p in source.parameters():
            if id(p) in _lora_owned_params(source):
                p.normal_(std=0.1)

    x = torch.randn(4, 8)
    expected = source(x)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_lora_weights(source, os.path.join(tmp, "lora.pt"))
        saved_keys = set(torch.load(path, weights_only=True))
        assert saved_keys == {
            "lora_A_projection.lora_A",
            "lora_A_projection.lora_B",
            "head.lora_A",
            "head.lora_B",
        }, saved_keys

        destination = _build()
        destination.load_state_dict(base_state)  # same pretrained base, fresh adapters
        apply_lora_to(destination, "*", r=2, alpha=4.0)
        base_tensors = {n: p for n, p in destination.named_parameters() if id(p) not in _lora_owned_params(destination)}
        base_values = {n: p.detach().clone() for n, p in base_tensors.items()}
        n_loaded = load_lora_weights(destination, path)

    assert n_loaded == len(saved_keys) == 4, n_loaded
    assert torch.equal(destination(x), expected), "reloaded adapter must reproduce the source output"
    for n, p in destination.named_parameters():
        if n in base_tensors:
            assert p is base_tensors[n] and torch.equal(p.detach(), base_values[n]), n

    summary = {"wrapped": n_wrapped, "saved_keys": sorted(saved_keys), "loaded": n_loaded}
    print(f"LoRA artifact round-trip workflow: {summary}")
    return summary


def peft_preconverted_base() -> dict:
    """Bounded placement demonstration: convert the base BEFORE injecting
    LoRA, then train the adapter without any corrective ``.to()``.

    A tiny ``Linear -> ReLU -> Linear`` net is converted to float64 first.
    ``apply_lora_to`` then allocates every ``lora_A`` / ``lora_B`` in
    float64 on the base's device, the forward and backward run at once,
    an optimizer built afterwards owns exactly the new adapter
    parameters, and the frozen base tensors are the very same objects
    with bit-exact values after the step.
    """
    set_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(8, 6), torch.nn.ReLU(), torch.nn.Linear(6, 3)).double()
    base_tensors = {name: p for name, p in net.named_parameters()}
    base_values = {name: p.detach().clone() for name, p in net.named_parameters()}

    n_wrapped = apply_lora_to(net, "*", r=2, alpha=4.0)
    assert n_wrapped == 2, n_wrapped
    assert all(p.dtype == torch.float64 for p in net.parameters()), "adapters must inherit float64"
    assert net[0].base.weight is base_tensors["0.weight"], "the base weight must be the same tensor"

    trainable = [p for p in net.parameters() if p.requires_grad]
    assert len(trainable) == 4  # two LoRA pairs
    optimizer = torch.optim.SGD(trainable, lr=0.1)
    x = torch.randn(5, 8, dtype=torch.float64)
    loss = (net(x) ** 2).mean()
    loss.backward()
    optimizer.step()
    out = net(x)
    assert out.dtype == torch.float64 and torch.isfinite(out).all()

    for name, value in base_values.items():
        wrapped_name = name.replace(".weight", ".base.weight").replace(".bias", ".base.bias")
        current = dict(net.named_parameters())[wrapped_name]
        assert current is base_tensors[name] and torch.equal(current.detach(), value), name

    summary = {"wrapped": n_wrapped, "adapter_dtype": str(net[0].lora_A.dtype), "loss": float(loss.detach())}
    print(f"PEFT pre-converted base workflow: {summary}")
    return summary


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


def peft_eval_injection() -> dict:
    """Bounded eval-before-injection demonstration (writes one file under a
    temporary directory).

    A deployed classifier is switched to ``eval()`` *before* LoRA adapters
    with nonzero dropout are injected and trained weights are loaded into
    them. Every wrapper inherits the eval mode of the layer it wraps, so
    the dropout on the adapter path stays inactive: repeated ``predict``
    calls are bit-identical and the per-module mode map is all-eval before
    and after. A later ``net.train()`` still activates the adapter dropout
    (preservation never pins a wrapper in eval), and modes are runtime
    state — the adapter file holds only ``lora_A`` / ``lora_B`` tensors.
    """
    set_seed(0)
    source = _classifier()
    base_state = {k: v.detach().clone() for k, v in source.net.state_dict().items()}
    apply_lora_to(source.net, "*", r=2, alpha=4.0, dropout=0.3)
    with torch.no_grad():
        for m in source.net.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(std=0.2)  # a trained, nonzero adapter

    deployed = _classifier()
    deployed.net.load_state_dict(base_state)  # the same pretrained base weights
    deployed.net.eval()  # eval BEFORE injection
    n_wrapped = apply_lora_to(deployed.net, "*", r=2, alpha=4.0, dropout=0.3)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_lora_weights(source.net, os.path.join(tmp, "adapter.pt"))
        saved_keys = sorted(torch.load(path, weights_only=True))
        n_loaded = load_lora_weights(deployed.net, path)
    assert n_loaded == len(saved_keys) == 2 * n_wrapped, (n_loaded, saved_keys)
    assert all(key.endswith(("lora_A", "lora_B")) for key in saved_keys), "no mode field is serialized"

    modes = {name: m.training for name, m in deployed.net.named_modules()}
    assert not any(modes.values()), f"injection flipped modules into train mode: {modes}"
    x = torch.randn(16, 8)
    first, second = deployed.predict(x), deployed.predict(x)
    assert (first.logits == second.logits).all(), "eval-mode inference must be deterministic"
    assert {name: m.training for name, m in deployed.net.named_modules()} == modes

    deployed.net.train()
    dropouts = [m.lora_dropout for m in deployed.net.modules() if isinstance(m, LoRALinear)]
    assert dropouts and all(d.training for d in dropouts), "a later train() must activate adapter dropout"
    deployed.net.eval()

    summary = {"wrapped": n_wrapped, "loaded": n_loaded, "eval_modules": len(modes)}
    print(f"PEFT eval-injection workflow: {summary}")
    return summary


def peft_alias_preflight() -> dict:
    """Bounded shared-module preflight demonstration (FIX-014).

    One ``Linear`` is registered under two names (``encoder`` and
    ``decoder``) next to an independent ``head``. Wrapping any alias would
    split the shared layer into two independent layers, so
    ``apply_lora_to`` rejects the wildcard *and* the second name alone with
    a ``ValueError`` naming both aliases — before building any wrapper, so
    every module identity, weight and ``requires_grad`` flag is unchanged.
    Selecting only the independent ``head`` then succeeds, and one
    optimizer step trains exactly its adapter while the shared layer stays
    untouched.
    """
    set_seed(0)
    shared = torch.nn.Linear(6, 6)
    net = torch.nn.ModuleDict(OrderedDict([("encoder", shared), ("decoder", shared), ("head", torch.nn.Linear(6, 3))]))
    snapshot = {k: v.detach().clone() for k, v in net.state_dict().items()}

    rejected = []
    for pattern in ("*", "decoder"):
        try:
            apply_lora_to(net, pattern, r=2, alpha=4.0)
        except ValueError as exc:
            assert "(encoder, decoder)" in str(exc), exc
            rejected.append(pattern)
    assert rejected == ["*", "decoder"], rejected
    assert net["encoder"] is net["decoder"] is shared and type(shared) is torch.nn.Linear
    assert all(p.requires_grad for p in net.parameters()), "a rejected call must not freeze anything"
    assert all(torch.equal(v, snapshot[k]) for k, v in net.state_dict().items())

    n_wrapped = apply_lora_to(net, "head", r=2, alpha=4.0)
    assert n_wrapped == 1 and isinstance(net["head"], LoRALinear)
    adapter = [net["head"].lora_A, net["head"].lora_B]
    optimizer = torch.optim.SGD(adapter, lr=0.1)
    x = torch.randn(4, 6)
    with torch.no_grad():
        features = net["decoder"](net["encoder"](x))
    loss = (net["head"](features) ** 2).mean() + net["head"].lora_B.sum()
    loss.backward()
    before = [p.detach().clone() for p in adapter]
    optimizer.step()
    assert any(not torch.equal(a, p.detach()) for a, p in zip(before, adapter, strict=True)), "adapter must move"
    assert torch.equal(shared.weight.detach(), snapshot["encoder.weight"]), "the shared layer is untouched"

    summary = {"rejected": rejected, "wrapped": n_wrapped, "loss": float(loss.detach())}
    print(f"PEFT alias preflight workflow: {summary}")
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
    lora_owned = _lora_owned_params(model.net)  # wrapper-owned identities, not a name heuristic
    for n, post in model.net.named_parameters():
        if id(post) in lora_owned:
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
