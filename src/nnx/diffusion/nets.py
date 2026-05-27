"""Small denoising networks for diffusion.

``DiffusionMLP`` is a tabular / low-dimensional denoiser: it takes the
noisy sample ``x_t`` plus the timestep ``t`` and returns the predicted
noise ``ε_θ(x_t, t)``. The timestep is sinusoid-embedded then concatenated
with x for input to the MLP — the standard DDPM time-conditioning idiom
in its smallest form.

This is *intentionally small*. The point is to make diffusion training
visible end-to-end without a full U-Net implementation; the same
``train_step_fn`` / sampler machinery works against any nn.Module with
the ``forward(x, t)`` signature, so larger architectures can be slotted
in by the user as needed.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_time_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer-style sinusoidal positional embedding,
    applied to scalar timesteps so the denoising network can condition
    on ``t``.

    Args:
        t: integer or float tensor of shape ``(B,)`` — per-sample timesteps.
        dim: embedding dimension. Half of it carries sin frequencies,
            half carries cos; ``dim`` must be even.

    Returns:
        Tensor of shape ``(B, dim)``.
    """
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal_time_embed dim must be even, got {dim}")
    half = dim // 2
    # Inverse-frequency scaling, matching the original Transformer paper
    # and ho:DDPM.
    decay = math.log(10000.0) / (half - 1)
    freqs = torch.exp(-decay * torch.arange(half, dtype=torch.float32, device=t.device))
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


class DiffusionMLP(nn.Module):
    """Conditional MLP for low-dim diffusion: ``forward(x_t, t) -> ε_pred``.

    Architecture: sinusoidal time embed → small projection → concat with
    flat x_t → MLP → linear head producing a noise prediction of the same
    shape as x_t. Bare ReLU activations, no skip connections — a single
    file's worth of code, enough to learn a 2D Gaussian mixture or a
    small tabular distribution.

    Inputs of any rank are supported by flattening dimensions ≥ 1 before
    the MLP and un-flattening at the output. The network is *NOT* a
    U-Net — it has no spatial structure. For image-space diffusion, the
    same train/sample/schedule machinery works against a user-supplied
    U-Net.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        time_embed_dim: int = 32,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dims is None:
            hidden_dims = [128, 128]
        self.input_dim = input_dim
        self.time_embed_dim = time_embed_dim

        # Project the sinusoidal embedding through one Linear+GELU before
        # concatenation — DDPM uses an MLP, but a single layer is enough
        # for low-dim cases and keeps the param count minimal.
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.GELU(),
        )

        layer_dims = [input_dim + time_embed_dim, *hidden_dims, input_dim]
        layers = []
        for in_d, out_d in zip(layer_dims[:-1], layer_dims[1:], strict=True):
            layers.append(nn.Linear(in_d, out_d))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise added to ``x`` at timestep ``t``.

        Args:
            x: ``(B, *)`` clean shape; flattened internally to ``(B, D)``.
            t: ``(B,)`` integer timesteps.

        Returns:
            Tensor of the same shape as ``x``.
        """
        orig_shape = x.shape
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        if x_flat.shape[1] != self.input_dim:
            raise ValueError(f"DiffusionMLP expects flattened input dim {self.input_dim}, got {x_flat.shape[1]}")

        # Time conditioning: sinusoidal embed → MLP → concat with x.
        t_emb = sinusoidal_time_embed(t, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)

        h = torch.cat([x_flat, t_emb], dim=-1)
        for layer in self.layers[:-1]:
            h = torch.relu(layer(h))
        out = self.layers[-1](h)

        return out.reshape(orig_shape)

    def unpack_batch(self, batch):
        """Standard ``(X-tuple, Y)`` adapter so this net plays nicely with
        the NNx dataloader contract. ``Y`` is unused by diffusion — every
        consumer that calls ``unpack_batch`` discards it."""
        if isinstance(batch, (list, tuple)):
            x, y = batch[0], batch[1] if len(batch) > 1 else None
        else:
            x, y = batch, None
        return (x,), y
