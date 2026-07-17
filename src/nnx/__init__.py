"""nnx — lightweight PyTorch training / eval / visualization toolkit.

The package is organized under `nnx.nn` (model, params, datasets, enums, nets,
callbacks) and two top-level helpers (`nnx.utils.Utils`, `nnx.vis_utils.VisUtils`).
The curated re-exports below give a flat surface for the most common imports
without forbidding the deep paths existing notebook code relies on.
"""

from __future__ import annotations

try:
    # Underscore-aliased so neither symbol leaks into `nnx.*` — these are
    # implementation details of `__version__` lookup, not public nnx API.
    from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
    from importlib.metadata import version as _version

    try:
        __version__ = _version("thekaveh-nnx")
    except _PackageNotFoundError:
        # Editable install before metadata exists, or run from the source
        # tree without installation. Keep in sync with pyproject.toml
        # [project] version on every bump.
        __version__ = "0.2.0+local"
except ImportError:  # pragma: no cover — Python <3.8.
    __version__ = "0.2.0+local"

from . import embeddings, interop, prune, viz
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
from .nn.dataset.nn_preference_dataset import NNPreferenceDataset
from .nn.dataset.nn_tabular_dataset import NNTabularDataset
from .nn.enum.activations import Activations
from .nn.enum.checkpoints import Checkpoints
from .nn.enum.devices import Devices
from .nn.enum.losses import Losses
from .nn.enum.nets import Nets
from .nn.enum.optims import Optims
from .nn.enum.schedulers import Schedulers
from .nn.generative_nn_model import GenerativeNNModel
from .nn.moe import MoELinear
from .nn.net.feed_fwd_nn import FeedFwdNN
from .nn.net.graph_att_nn import GraphAttNN
from .nn.net.graph_conv_nn import GraphConvNN
from .nn.net.graph_nn_base import GraphNNBase
from .nn.net.graph_sage_nn import GraphSageNN
from .nn.net.transformer_nn import TransformerNN
from .nn.net.vit_nn import ViTBlock, ViTNN
from .nn.nn_model import (
    EvalStepContext,
    EvalStepFn,
    NNModel,
    PredictResult,
    TrainStepContext,
    TrainStepFn,
    default_train_step,
)
from .nn.params.nn_checkpoint import NNCheckpoint
from .nn.params.nn_conv_params import NNConvParams
from .nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from .nn.params.nn_iteration_data_point import NNIterationDataPoint
from .nn.params.nn_model_params import NNModelParams
from .nn.params.nn_moe_params import NNMoEParams
from .nn.params.nn_optim_params import NNOptimParams
from .nn.params.nn_optim_params_builder import NNOptimParamsBuilder
from .nn.params.nn_params import NNParams
from .nn.params.nn_run import NNRun
from .nn.params.nn_scheduler_params import NNSchedulerParams
from .nn.params.nn_scheduler_params_builder import NNSchedulerParamsBuilder
from .nn.params.nn_train_params import NNTrainParams
from .nn.params.nn_transformer_params import NNTransformerParams
from .nn.params.nn_transformer_params_builder import NNTransformerParamsBuilder

# NNTokenizerParams + train_bpe depend on the optional `tokenizers`
# extra (the `lm` extra in pyproject.toml). Re-exported only when the
# dep is available so non-LM users importing `nnx` don't hit an
# ImportError at top-level.
try:
    from .nn.params.nn_tokenizer_params import NNTokenizerParams, train_bpe
except ImportError:  # pragma: no cover — exercised in CI without the lm extra
    NNTokenizerParams = None  # type: ignore[assignment,misc]
    train_bpe = None  # type: ignore[assignment]

# Lower-level NNModel.train compatible factory for the embeddings path,
# promoted alongside the other 10 *_train_step_factory functions for
# uniform `nnx.<TAB>` discoverability. High-level `train_contrastive`
# and FAISS export stay under `nnx.embeddings.*` to keep the top-level
# surface focused on the train-step entry points users will most often
# reach for.
from .embeddings import text_contrastive_train_step_factory

# LogitsProcessor chain — pure-torch, no optional deps; always available.
from .generation import (
    LogitsChain,
    LogitsChainBuilder,
    LogitsProcessor,
    RepetitionPenalty,
    TemperatureScaling,
    TopKFilter,
    TopPFilter,
    apply_chain,
    sample_next_token,
)
from .lr_finder import LRFinderResult, lr_finder
from .paradigms import (
    JEPAPredictor,
    born_again_train,
    build_target_encoder,
    cutmix_train_step_factory,
    dpo_train_step_factory,
    feature_kd_train_step_factory,
    jepa_train_step_factory,
    kd_train_step_factory,
    mixup_train_step_factory,
    moe_train_step_factory,
    nt_xent_loss,
    random_block_mask,
    simclr_train_step_factory,
    update_ema,
)
from .peft import (
    AdapterLayer,
    DoRALinear,
    IA3Linear,
    LoRALinear,
    PrefixTuner,
    PromptTuner,
    apply_dora_to,
    apply_ia3_to,
    apply_lora_to,
    load_ia3_weights,
    load_lora_weights,
    load_prefix_weights,
    load_prompt_weights,
    save_ia3_weights,
    save_lora_weights,
    save_prefix_weights,
    save_prompt_weights,
)
from .quantize import QATLifecycleCallback, qat_train_step_factory, quantize_int8
from .seeding import dataloader_worker_init_fn, env_snapshot, set_seed
from .surgery import (
    deepen,
    drop_layer,
    expand_embedding,
    low_rank_factorize,
    widen,
)
from .trainer import NNTrainerParams, Trainer, TrainerStepContext, TrainerStepFn
from .trainer.params_builder import NNTrainerParamsBuilder
from .utils import Utils
from .vis_utils import VisUtils

__all__ = [
    # Orchestration
    "NNModel",
    "PredictResult",
    "TrainStepContext",
    "TrainStepFn",
    "EvalStepContext",
    "EvalStepFn",
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
    "NNConvParams",
    "NNMoEParams",
    "NNRun",
    "NNCheckpoint",
    "NNModelParams",
    "NNTrainParams",
    "NNOptimParams",
    "NNOptimParamsBuilder",
    "NNSchedulerParams",
    "NNSchedulerParamsBuilder",
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
    # Decoder-only transformer / LM path
    "generation",
    "TransformerNN",
    "NNTransformerParams",
    "NNTransformerParamsBuilder",
    "NNTokenizerParams",
    "GenerativeNNModel",
    "train_bpe",
    "LogitsChain",
    "LogitsChainBuilder",
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
    "NNPreferenceDataset",
    # Helpers
    "Utils",
    "VisUtils",
    "viz",
    # Fine-tuning
    "finetune",
    "freeze",
    "unfreeze",
    "frozen",
    "load_pretrained",
    "LoadPretrainedResult",
    "NNParamGroupSpec",
    # Multi-optimizer Trainer
    "trainer",
    "Trainer",
    "TrainerStepContext",
    "TrainerStepFn",
    "NNTrainerParams",
    "NNTrainerParamsBuilder",
    # Diffusion
    "diffusion",
    "DiffusionMLP",
    "NoiseSchedule",
    "NoiseSchedulers",
    "diffusion_train_step_factory",
    "sample",
    # Training paradigms
    "paradigms",
    "kd_train_step_factory",
    "feature_kd_train_step_factory",
    "born_again_train",
    "simclr_train_step_factory",
    "nt_xent_loss",
    "mixup_train_step_factory",
    "cutmix_train_step_factory",
    # I-JEPA (joint embedding predictive architecture)
    "jepa_train_step_factory",
    "build_target_encoder",
    "update_ema",
    "random_block_mask",
    "JEPAPredictor",
    "ViTNN",
    "ViTBlock",
    "dpo_train_step_factory",
    # Mixture-of-Experts
    "MoELinear",
    "moe_train_step_factory",
    # PEFT (LoRA + DoRA + IA3 + Prefix + Prompt + adapters)
    "peft",
    "LoRALinear",
    "apply_lora_to",
    "save_lora_weights",
    "load_lora_weights",
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
    "AdapterLayer",
    # Quantization (PTQ INT8 weight-only + QAT 8da4w via torchao)
    "quantize",
    "quantize_int8",
    "qat_train_step_factory",
    "QATLifecycleCallback",
    # Pruning (magnitude unstructured + 2:4 semi-structured)
    "prune",
    # Surgery (Net2Net + drop + low-rank + embedding)
    "surgery",
    "widen",
    "deepen",
    "drop_layer",
    "low_rank_factorize",
    "expand_embedding",
    # Embeddings (contrastive trainer + FAISS export)
    "embeddings",
    "text_contrastive_train_step_factory",
    # Interop (GGUF / Ollama exporters; bound so README's
    # `nnx.interop.write_gguf(...)` works after a plain `import nnx`)
    "interop",
    # Reproducibility
    "set_seed",
    "dataloader_worker_init_fn",
    "env_snapshot",
    "lr_finder",
    "LRFinderResult",
    # Metadata
    "__version__",
]
