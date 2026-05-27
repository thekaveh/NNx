"""Tests for nnx.paradigms.contrastive — SimCLR / NT-Xent."""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

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
    nt_xent_loss,
    set_seed,
    simclr_train_step_factory,
)

# -------------------------------------------------------------------------
# nt_xent_loss
# -------------------------------------------------------------------------

def test_nt_xent_validates_shape_match():
    with pytest.raises(ValueError, match="shape mismatch"):
        nt_xent_loss(torch.randn(8, 16), torch.randn(8, 32))


def test_nt_xent_validates_temperature():
    with pytest.raises(ValueError, match="temperature"):
        nt_xent_loss(torch.randn(8, 16), torch.randn(8, 16), temperature=0.0)


def test_nt_xent_finite_and_scalar():
    z1 = torch.randn(8, 16)
    z2 = torch.randn(8, 16)
    loss = nt_xent_loss(z1, z2, temperature=0.5)
    assert loss.dim() == 0
    assert torch.isfinite(loss).item()


def test_nt_xent_minimized_when_pairs_aligned():
    """When (z1, z2) are perfectly aligned (z2 == z1) but distinct
    SAMPLES are orthogonal, the loss should be very small — every
    positive pair is the most-similar in its row by construction."""
    torch.manual_seed(0)
    # 8 mutually orthogonal directions in 16D.
    base = torch.eye(8, 16)
    z1 = base
    z2 = base.clone()
    loss_aligned = nt_xent_loss(z1, z2, temperature=0.1).item()
    # Random embeddings: no special alignment between views.
    z1r = torch.randn(8, 16)
    z2r = torch.randn(8, 16)
    loss_random = nt_xent_loss(z1r, z2r, temperature=0.1).item()
    assert loss_aligned < loss_random


# -------------------------------------------------------------------------
# simclr_train_step_factory
# -------------------------------------------------------------------------

class _PairedViewDataset(Dataset):
    """Yields (view1, view2) where view2 is view1 plus a small jitter —
    enough to verify the step's plumbing without modeling a real
    augmentation pipeline."""

    def __init__(self, n: int = 32, d: int = 8, noise: float = 0.05):
        torch.manual_seed(0)
        self.x = torch.randn(n, d)
        self.noise = noise

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        v1 = self.x[idx]
        v2 = v1 + self.noise * torch.randn_like(v1)
        return v1, v2


def _embedding_model(input_dim: int = 8, output_dim: int = 16) -> NNModel:
    """An NNModel whose net produces D-dim embeddings. Losses.CROSS_ENTROPY
    is unused by the contrastive step — only the forward path matters."""
    return NNModel(
        net_params=NNParams(
            input_dim=input_dim, output_dim=output_dim, hidden_dims=[32],
            dropout_prob=0.0, activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
        ),
    )


def test_simclr_factory_validates_temperature():
    with pytest.raises(ValueError, match="temperature"):
        simclr_train_step_factory(temperature=0.0)


def test_simclr_factory_rejects_bad_batch_shape():
    """Step should fail loudly when the loader yields the wrong shape."""
    model = _embedding_model()
    step_fn = simclr_train_step_factory(temperature=0.5)

    # Bad: a single tensor (no second view).
    from nnx.nn.nn_model import TrainStepContext
    ctx = TrainStepContext(
        model=model, batch=torch.randn(4, 8), optimizer=None, scaler=None,
        grad_clip_norm=None, extra_metrics=None,
        accumulate_grad_batches=1, batch_idx=0, epoch_idx=0,
    )
    with pytest.raises(ValueError, match="view1, view2"):
        step_fn(ctx)


def test_simclr_train_loop_runs(tmp_path, monkeypatch):
    """End-to-end smoke: build a paired-view loader, run a few epochs,
    verify the loss is finite throughout AND that embeddings actually
    moved (a constant-output net would fool a finite-only check)."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)

    dataset = _PairedViewDataset(n=32, d=8)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = _embedding_model()
    # Pre-train weights so we can detect that the optimizer actually ran.
    pre = {n: p.clone() for n, p in model.net.named_parameters()}

    run = model.train(
        params=NNTrainParams(
            n_epochs=2,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3,
            ),
        ),
        train_step_fn=simclr_train_step_factory(temperature=0.5),
    )
    losses = [idp.train_edp.loss for idp in run.idps]
    assert len(losses) > 0
    assert all(lo is not None and torch.isfinite(torch.tensor(lo)).item() for lo in losses)
    moved = any(
        not torch.equal(pre[n], p.detach()) for n, p in model.net.named_parameters()
    )
    assert moved, "SimCLR step ran but embedding net weights did not change"
