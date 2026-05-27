"""Transfer learning via layer freezing.

Trains a small classifier from scratch on synthetic distribution A,
exports its weights, then loads them into a fresh model with the
backbone frozen and fine-tunes only the head on distribution B. Shows
the standard "freeze backbone, train head" recipe.

The same pattern works against any external weights — torchvision
checkpoints, HuggingFace state-dicts, weights from a colleague. The
key is `load_pretrained` (with optional `key_map=` for naming
mismatches) plus `freeze(...)` with glob patterns.

Run:
    python examples/06_finetune_with_layer_freezing.py
"""

from __future__ import annotations

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
    frozen,
    load_pretrained,
    set_seed,
)


def _make_loaders(seed: int, n: int = 256, d: int = 8, n_classes: int = 3):
    """Random-feature / random-label dataset. The seed differs between the
    pretrain and fine-tune calls so the two loaders draw distinct samples
    — sufficient for showing that loss decreases on the new distribution
    while the backbone stays frozen and only the head trains. (A
    class-conditional Gaussian setup would make the demo more
    informative; kept simple to minimize dependencies.)
    """
    torch.manual_seed(seed)
    X = torch.randn(n, d)
    y = torch.randint(0, n_classes, (n,))
    train = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
    return train


def _make_model(seed: int):
    set_seed(seed)
    return NNModel(
        net_params=NNParams(
            input_dim=8,
            output_dim=3,
            hidden_dims=[16, 8],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )


def main():
    train_params_template = lambda lr: NNTrainParams(  # noqa: E731
        n_epochs=3,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=lr, momentum=(0.9, 0.999), weight_decay=0.0),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=2, cooldown=1, threshold=1e-3),
    )

    # ── Phase 1: train a model from scratch on distribution A ──────────
    print("Phase 1: pretrain on distribution A")
    pretrained = _make_model(seed=0)
    loader_a = _make_loaders(seed=0)
    run_a = pretrained.train(
        params=train_params_template(lr=1e-2).with_train_loader(loader_a),
    )
    print(f"  pretrained {len(run_a.idps)} iterations\n")

    # ── Save the pretrained backbone as a plain state-dict ─────────────
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        weights_path = f.name
    pretrained.export_state_dict(weights_path)
    print(f"  exported weights to {weights_path}")

    # ── Phase 2: load into a fresh model, freeze backbone, fine-tune ──
    print("\nPhase 2: fine-tune on distribution B (backbone frozen)")
    fine = _make_model(seed=1)  # different random init
    result = load_pretrained(fine.net, weights_path)
    print(
        f"  loaded {len(result.loaded_keys)} keys, "
        f"{len(result.missing_keys)} missing, "
        f"{len(result.unexpected_keys)} unexpected"
    )

    # Freeze every parameter except the final classifier head
    # (FeedFwdNN names its layers `layers.0`, `layers.1`, `layers.2`).
    n_frozen = fine.freeze("layers.0.*", "layers.1.*")
    total = sum(1 for _ in fine.net.parameters())
    n_frozen_total = len(frozen(fine.net))
    print(
        f"  froze {n_frozen} params this call; "
        f"{n_frozen_total}/{total} now frozen, "
        f"{total - n_frozen_total}/{total} trainable"
    )

    loader_b = _make_loaders(seed=42)  # distribution B
    run_b = fine.train(
        params=train_params_template(lr=1e-3).with_train_loader(loader_b),
    )
    last_loss = run_b.idps[-1].train_edp.loss
    first_loss = run_b.idps[0].train_edp.loss
    print(f"\nFine-tune loss: {first_loss:.4f} → {last_loss:.4f}")
    print(f"  trained {len(run_b.idps)} iterations on the head only")


if __name__ == "__main__":
    main()
