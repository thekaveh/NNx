"""Fine-tuning infrastructure for nnx.

Three concerns, three modules:

- :mod:`nnx.finetune.freezing` — toggle ``requires_grad`` on submodule
  parameters via fnmatch glob patterns.
- :mod:`nnx.finetune.loading` — load pretrained weights from external
  state-dicts / paths / modules, with optional key remapping.
- :mod:`nnx.finetune.param_groups` — declarative per-layer LR /
  weight-decay overrides via :class:`NNParamGroupSpec`, plumbed into
  the existing :class:`nnx.NNOptimParams`.

The typical transfer-learning recipe:

    from nnx import NNModel
    from nnx.finetune import load_pretrained, freeze

    model = NNModel(net_params=..., params=...)
    load_pretrained(model.net, "resnet18.pt", key_map={"fc.": "head."})
    freeze(model.net, "encoder.*")               # train only the head
    model.train(params=NNTrainParams(...))

For more granular control (per-group LRs), set ``NNOptimParams.param_groups``
to a list of :class:`NNParamGroupSpec` and let the optimizer factory
build the right per-group dicts.
"""

from __future__ import annotations

from .freezing import freeze, frozen, unfreeze
from .loading import LoadPretrainedResult, load_pretrained
from .param_groups import NNParamGroupSpec, build_param_groups

__all__ = [
    "freeze",
    "unfreeze",
    "frozen",
    "load_pretrained",
    "LoadPretrainedResult",
    "NNParamGroupSpec",
    "build_param_groups",
]
