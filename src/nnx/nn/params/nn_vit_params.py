"""Vision Transformer encoder params (#395).

``NNViTParams`` subclasses ``NNParams`` — the same lift-via-subclassing
pattern ``NNTransformerParams`` / ``NNConvParams`` use — and carries every
setting :class:`~nnx.nn.net.vit_nn.ViTNN` takes, so ``Nets.VIT(params)``
builds the encoder and a stored run rebuilds it with no caller-side swap:

- ``image_size`` / ``patch_size`` / ``d_model`` / ``n_layers`` (required)
  and ``n_heads`` (lifted from ``NNParams``, required here) — the
  architecture. Always emitted by ``state()``; ``patch_size`` is the key
  ``NNParams.resolve_from_state`` dispatches on (mirroring ``vocab_size`` →
  transformer, ``conv_channels`` → conv).
- ``in_channels`` (3), ``ffn_mult`` (4), ``attn_dropout`` / ``resid_dropout``
  (0.0) — omitted from ``state()`` at their defaults, the omit-when-default
  invariant that keeps a vanilla config hashing to a stable run id as knobs
  accrue (a default takes part in the identity by its absence).

The base fields follow the encoder's shape: ``input_dim`` is the pixel count
``in_channels * image_size ** 2``, ``output_dim`` is ``d_model`` (one
``d_model`` vector per token), there are no hidden layers, and
``dropout_prob`` stays 0.0 — the encoder drops out through ``attn_dropout``
and ``resid_dropout``. Any other value raises, so no setting is silently
ignored. ``activation`` is unused (the blocks use SwiGLU), as for
``NNTransformerParams``.

Every count is validated as a non-boolean integer at construction (Python and
NumPy integers accepted and normalized to plain ``int``); sizes must be
positive, ``image_size`` divisible by ``patch_size`` and ``d_model`` by
``n_heads``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..._validation import require_count, require_finite_real
from .nn_params import NNParams

_DEFAULTS = {"in_channels": 3, "ffn_mult": 4, "attn_dropout": 0.0, "resid_dropout": 0.0}


@dataclass(frozen=True, kw_only=True, slots=True)
class NNViTParams(NNParams):
    """Parameters for :class:`~nnx.nn.net.vit_nn.ViTNN`, built by ``Nets.VIT``."""

    image_size: int
    patch_size: int
    d_model: int
    n_layers: int
    in_channels: int = 3
    ffn_mult: int = 4
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0

    def __post_init__(self):
        # Explicit unbound call — same slotted-dataclass reasoning as
        # NNTransformerParams.__post_init__.
        NNParams.__post_init__(self)
        if self.n_heads is None:
            raise ValueError(f"NNViTParams requires n_heads > 0, got {self.n_heads!r}")
        # Positive dimensions are validated BEFORE the divisibility checks, so
        # a zero or negative value cannot mask itself (`0 % n == 0`).
        for name in ("image_size", "patch_size", "d_model", "n_layers", "in_channels", "ffn_mult"):
            object.__setattr__(
                self,
                name,
                require_count(getattr(self, name), name, owner="NNViTParams", minimum=0, exclusive_min=True),
            )
        # Probabilities: finite reals in [0, 1], normalized to a plain float so
        # 1 and 1.0 (or a NumPy scalar) hash and serialize alike.
        for name in ("attn_dropout", "resid_dropout"):
            value = require_finite_real(getattr(self, name), name, owner="NNViTParams", minimum=0.0, maximum=1.0)
            object.__setattr__(self, name, float(value))
        if self.image_size % self.patch_size != 0:
            raise ValueError(f"image_size={self.image_size} must be divisible by patch_size={self.patch_size}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")
        pixels = self.in_channels * self.image_size**2
        if self.input_dim != pixels:
            raise ValueError(
                f"NNViTParams requires input_dim = in_channels * image_size ** 2 = {pixels}, got {self.input_dim}"
            )
        if self.output_dim != self.d_model:
            raise ValueError(
                f"NNViTParams requires output_dim = d_model = {self.d_model} (one vector per token), "
                f"got {self.output_dim}"
            )
        if self.hidden_dims:
            raise ValueError(f"NNViTParams has no hidden layers: hidden_dims must be empty, got {self.hidden_dims}")
        if self.dropout_prob != 0.0:
            raise ValueError(
                f"NNViTParams drops out through attn_dropout / resid_dropout: dropout_prob must be 0.0, "
                f"got {self.dropout_prob}"
            )

    @property
    def n_patches(self) -> int:
        """Patch tokens per image (the encoder adds one CLS token)."""
        return (self.image_size // self.patch_size) ** 2

    def state(self) -> dict:
        d = NNParams.state(self)  # n_heads is always set here, so always emitted
        d.update(image_size=self.image_size, patch_size=self.patch_size, d_model=self.d_model, n_layers=self.n_layers)
        for key, default in _DEFAULTS.items():
            value = getattr(self, key)
            if value != default:
                d[key] = value
        return d

    @staticmethod
    def from_state(state: dict) -> NNViTParams:
        base = NNParams.from_state(state)
        return NNViTParams(
            input_dim=base.input_dim,
            output_dim=base.output_dim,
            hidden_dims=base.hidden_dims,
            dropout_prob=base.dropout_prob,
            activation=base.activation,
            n_heads=base.n_heads,
            image_size=state["image_size"],
            patch_size=state["patch_size"],
            d_model=state["d_model"],
            n_layers=state["n_layers"],
            **{key: state.get(key, default) for key, default in _DEFAULTS.items()},
        )

    def __str__(self) -> str:
        return (
            f"[vit, image={self.image_size}, patch={self.patch_size}, channels={self.in_channels}, "
            f"d_model={self.d_model}, layers={self.n_layers}, heads={self.n_heads}]"
        )
