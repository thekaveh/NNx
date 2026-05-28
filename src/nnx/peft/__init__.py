"""Parameter-efficient fine-tuning (PEFT).

Two complementary patterns for adapting a pretrained network to a new
task without re-training the full weight matrix:

  - **LoRA** (Low-Rank Adaptation) — wraps an existing :class:`nn.Linear`
    with a frozen base and trainable low-rank residual. See :mod:`lora`.
  - **Adapter layers** — bottleneck residual blocks the user inserts
    into the forward pass. See :mod:`adapters`.

LoRA is the dominant choice in modern PEFT pipelines; adapters
predate LoRA and are still useful when the user controls the network
architecture and wants explicit residual insertion points.
"""

from __future__ import annotations

from .adapters import AdapterLayer
from .dora import DoRALinear, apply_dora_to
from .ia3 import IA3Linear, apply_ia3_to, load_ia3_weights, save_ia3_weights
from .lora import LoRALinear, apply_lora_to, load_lora_weights, save_lora_weights

__all__ = [
    "LoRALinear",
    "apply_lora_to",
    "save_lora_weights",
    "load_lora_weights",
    "AdapterLayer",
    "DoRALinear",
    "apply_dora_to",
    "IA3Linear",
    "apply_ia3_to",
    "save_ia3_weights",
    "load_ia3_weights",
]
