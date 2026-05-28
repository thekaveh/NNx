"""nnx — lightweight PyTorch training / eval / visualization toolkit.

The package is organized under `nnx.nn` (model, params, datasets, enums, nets,
callbacks) and two top-level helpers (`nnx.utils.Utils`, `nnx.vis_utils.VisUtils`).
The curated re-exports below give a flat surface for the most common imports
without forbidding the deep paths existing notebook code relies on.
"""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    try:
        __version__ = _version("nnx")
    except PackageNotFoundError:
        # Editable install before metadata exists, or run from the source
        # tree without installation.
        __version__ = "0.1.0+local"
except ImportError:  # pragma: no cover — Python <3.8.
    __version__ = "0.1.0+local"

from .diffusion import (
    DiffusionMLP,
    NoiseSchedule,
    NoiseSchedulers,
    diffusion_train_step_factory,
    sample,
)
from .finetune import (
    LoadPretrainedResult,
    NNParamGroupSpec,
    freeze,
    frozen,
    load_pretrained,
    unfreeze,
)
from .nn.callbacks import (
    Callback,
    EarlyStopping,
    LRMonitor,
    ModelCheckpoint,
    TensorBoardCallback,
    WandbCallback,
)
from .nn.dataset.nn_dataset import NNDataset
from .nn.dataset.nn_dataset_base import NNDatasetBase
from .nn.dataset.nn_graph_dataset import NNGraphDataset
from .nn.dataset.nn_tabular_dataset import NNTabularDataset
from .nn.enum.activations import Activations
from .nn.enum.checkpoints import Checkpoints
from .nn.enum.devices import Devices
from .nn.enum.losses import Losses
from .nn.enum.nets import Nets
from .nn.enum.optims import Optims
from .nn.enum.schedulers import Schedulers
from .nn.generative_nn_model import GenerativeNNModel
from .nn.net.feed_fwd_nn import FeedFwdNN
from .nn.net.graph_att_nn import GraphAttNN
from .nn.net.graph_conv_nn import GraphConvNN
from .nn.net.graph_nn_base import GraphNNBase
from .nn.net.graph_sage_nn import GraphSageNN
from .nn.net.transformer_nn import TransformerNN
from .nn.nn_model import (
    NNModel,
    PredictResult,
    TrainStepContext,
    TrainStepFn,
    default_train_step,
)
from .nn.params.nn_checkpoint import NNCheckpoint
from .nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from .nn.params.nn_iteration_data_point import NNIterationDataPoint
from .nn.params.nn_model_params import NNModelParams
from .nn.params.nn_optim_params import NNOptimParams
from .nn.params.nn_params import NNParams
from .nn.params.nn_run import NNRun
from .nn.params.nn_scheduler_params import NNSchedulerParams
from .nn.params.nn_train_params import NNTrainParams
from .nn.params.nn_transformer_params import NNTransformerParams

# NNTokenizerParams + train_bpe depend on the optional `tokenizers`
# extra (the `lm` extra in pyproject.toml). Re-exported only when the
# dep is available so non-LM users importing `nnx` don't hit an
# ImportError at top-level.
try:
    from .nn.params.nn_tokenizer_params import NNTokenizerParams, train_bpe

    _HAS_LM_EXTRA = True
except ImportError:  # pragma: no cover — exercised in CI without the lm extra
    NNTokenizerParams = None  # type: ignore[assignment,misc]
    train_bpe = None  # type: ignore[assignment]
    _HAS_LM_EXTRA = False

# LogitsProcessor chain — pure-torch, no optional deps; always available.
from .generation import (
    LogitsProcessor,
    RepetitionPenalty,
    TemperatureScaling,
    TopKFilter,
    TopPFilter,
    apply_chain,
    sample_next_token,
)
from .paradigms import (
    cutmix_train_step_factory,
    kd_train_step_factory,
    mixup_train_step_factory,
    nt_xent_loss,
    simclr_train_step_factory,
)
from .peft import (
    AdapterLayer,
    LoRALinear,
    apply_lora_to,
    load_lora_weights,
    save_lora_weights,
)
from .seeding import dataloader_worker_init_fn, env_snapshot, set_seed
from .trainer import NNTrainerParams, Trainer, TrainerStepContext, TrainerStepFn
from .utils import Utils
from .vis_utils import VisUtils

__all__ = [
    # Orchestration
    "NNModel",
    "PredictResult",
    "TrainStepContext",
    "TrainStepFn",
    "default_train_step",
    # Callbacks
    "Callback",
    "EarlyStopping",
    "LRMonitor",
    "ModelCheckpoint",
    "TensorBoardCallback",
    "WandbCallback",
    # Params
    "NNParams",
    "NNRun",
    "NNCheckpoint",
    "NNModelParams",
    "NNTrainParams",
    "NNOptimParams",
    "NNSchedulerParams",
    "NNIterationDataPoint",
    "NNEvaluationDataPoint",
    # Enums
    "Activations",
    "Checkpoints",
    "Devices",
    "Losses",
    "Nets",
    "Optims",
    "Schedulers",
    # Networks
    "FeedFwdNN",
    "GraphNNBase",
    "GraphConvNN",
    "GraphSageNN",
    "GraphAttNN",
    # Decoder-only transformer / LM path (SP-4)
    "TransformerNN",
    "NNTransformerParams",
    "NNTokenizerParams",
    "GenerativeNNModel",
    "train_bpe",
    "LogitsProcessor",
    "TemperatureScaling",
    "TopKFilter",
    "TopPFilter",
    "RepetitionPenalty",
    "apply_chain",
    "sample_next_token",
    # Datasets
    "NNDataset",
    "NNGraphDataset",
    "NNTabularDataset",
    "NNDatasetBase",
    # Helpers
    "Utils",
    "VisUtils",
    # Fine-tuning
    "freeze",
    "unfreeze",
    "frozen",
    "load_pretrained",
    "LoadPretrainedResult",
    "NNParamGroupSpec",
    # Multi-optimizer Trainer
    "Trainer",
    "TrainerStepContext",
    "TrainerStepFn",
    "NNTrainerParams",
    # Diffusion
    "DiffusionMLP",
    "NoiseSchedule",
    "NoiseSchedulers",
    "diffusion_train_step_factory",
    "sample",
    # Training paradigms
    "kd_train_step_factory",
    "simclr_train_step_factory",
    "nt_xent_loss",
    "mixup_train_step_factory",
    "cutmix_train_step_factory",
    # PEFT (LoRA + adapters)
    "LoRALinear",
    "apply_lora_to",
    "save_lora_weights",
    "load_lora_weights",
    "AdapterLayer",
    # Reproducibility
    "set_seed",
    "dataloader_worker_init_fn",
    "env_snapshot",
    # Metadata
    "__version__",
]
