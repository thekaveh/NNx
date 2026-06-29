"""Parameter-efficient fine-tuning (PEFT).

Complementary patterns for adapting a pretrained network to a new task
without re-training the full weight matrix:

  - **LoRA** (Low-Rank Adaptation) — wraps an existing :class:`nn.Linear`
    with a frozen base and trainable low-rank residual. See :mod:`lora`.
  - **DoRA** (Weight-Decomposed Low-Rank Adaptation) — extends LoRA with a
    trainable per-output-row magnitude vector. See :mod:`dora`.
  - **IA³** (Infused Adapter by Inhibiting and Amplifying Inner Activations)
    — a single learned per-output-dim scaling vector on a frozen Linear; the
    smallest adapter in the family. See :mod:`ia3`.
  - **Adapter layers** — bottleneck residual blocks the user inserts
    into the forward pass. See :mod:`adapters`.
  - **Prefix tuning** — learnable per-layer K/V prefixes attached to a
    frozen TransformerNN. See :mod:`prefix`.
  - **Prompt tuning** — learnable soft-prompt embeddings prepended to
    the input of a frozen TransformerNN. See :mod:`prompt`.

LoRA is the dominant choice in modern PEFT pipelines; adapters predate
LoRA and are still useful when the user controls the network architecture
and wants explicit residual insertion points. Prefix and prompt tuning
target the transformer LM path specifically — both leave the base
weights bit-exactly unchanged at all times.
"""

from __future__ import annotations

from .adapters import AdapterLayer
from .dora import DoRALinear, apply_dora_to
from .ia3 import IA3Linear, apply_ia3_to, load_ia3_weights, save_ia3_weights
from .lora import LoRALinear, apply_lora_to, load_lora_weights, save_lora_weights
from .prefix import PrefixTuner, load_prefix_weights, save_prefix_weights
from .prompt import PromptTuner, load_prompt_weights, save_prompt_weights

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
    "PrefixTuner",
    "save_prefix_weights",
    "load_prefix_weights",
    "PromptTuner",
    "save_prompt_weights",
    "load_prompt_weights",
]
