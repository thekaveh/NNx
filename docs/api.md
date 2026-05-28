# API Reference

Auto-generated from docstrings via [mkdocstrings](https://mkdocstrings.github.io/). Sections are ordered from most foundational to most specialized; within each section, classes precede free functions and type aliases.

## 1. Top-level package

::: nnx
    options:
      members:
        - __version__
        - set_seed
        - dataloader_worker_init_fn
        - env_snapshot

## 2. Orchestrators

### 2.1. NNModel — supervised orchestrator

::: nnx.nn.nn_model.NNModel

::: nnx.nn.nn_model.PredictResult

::: nnx.nn.nn_model.TrainStepContext

::: nnx.nn.nn_model.default_train_step

### 2.2. Trainer — multi-optimizer orchestrator

::: nnx.trainer.trainer.Trainer

::: nnx.trainer.trainer.TrainerStepContext

::: nnx.trainer.trainer.TrainerStepFn

::: nnx.trainer.params.NNTrainerParams

## 3. Params

::: nnx.nn.params.nn_params.NNParams

::: nnx.nn.params.nn_model_params.NNModelParams

::: nnx.nn.params.nn_train_params.NNTrainParams

::: nnx.nn.params.nn_optim_params.NNOptimParams

::: nnx.nn.params.nn_scheduler_params.NNSchedulerParams

::: nnx.nn.params.nn_run.NNRun

::: nnx.nn.params.nn_checkpoint.NNCheckpoint

::: nnx.nn.params.nn_iteration_data_point.NNIterationDataPoint

::: nnx.nn.params.nn_evaluation_data_point.NNEvaluationDataPoint

## 4. Networks

::: nnx.nn.net.feed_fwd_nn.FeedFwdNN

::: nnx.nn.net.graph_nn_base.GraphNNBase

::: nnx.nn.net.graph_conv_nn.GraphConvNN

::: nnx.nn.net.graph_sage_nn.GraphSageNN

::: nnx.nn.net.graph_att_nn.GraphAttNN

## 5. Datasets

::: nnx.nn.dataset.nn_dataset_base.NNDatasetBase

::: nnx.nn.dataset.nn_dataset.NNDataset

::: nnx.nn.dataset.nn_graph_dataset.NNGraphDataset

::: nnx.nn.dataset.nn_tabular_dataset.NNTabularDataset

## 6. Enums

::: nnx.nn.enum.activations.Activations

::: nnx.nn.enum.checkpoints.Checkpoints

::: nnx.nn.enum.devices.Devices

::: nnx.nn.enum.losses.Losses

::: nnx.nn.enum.nets.Nets

::: nnx.nn.enum.optims.Optims

::: nnx.nn.enum.schedulers.Schedulers

## 7. Callbacks

::: nnx.nn.callbacks.Callback

::: nnx.nn.callbacks.EarlyStopping

::: nnx.nn.callbacks.LRMonitor

::: nnx.nn.callbacks.ModelCheckpoint

::: nnx.nn.callbacks.TensorBoardCallback

::: nnx.nn.callbacks.WandbCallback

## 8. Fine-tuning (`nnx.finetune`)

::: nnx.finetune.freezing.freeze

::: nnx.finetune.freezing.unfreeze

::: nnx.finetune.freezing.frozen

::: nnx.finetune.loading.load_pretrained

::: nnx.finetune.loading.LoadPretrainedResult

::: nnx.finetune.param_groups.NNParamGroupSpec

::: nnx.finetune.param_groups.build_param_groups

## 9. Parameter-efficient fine-tuning (`nnx.peft`)

::: nnx.peft.lora.LoRALinear

::: nnx.peft.lora.apply_lora_to

::: nnx.peft.lora.save_lora_weights

::: nnx.peft.lora.load_lora_weights

::: nnx.peft.adapters.AdapterLayer

## 10. Pruning (`nnx.prune`)

::: nnx.prune.magnitude.magnitude_prune

::: nnx.prune.semi_structured.semi_structured_24

## 11. Diffusion (`nnx.diffusion`)

::: nnx.diffusion.schedules.NoiseSchedulers

::: nnx.diffusion.schedules.NoiseSchedule

::: nnx.diffusion.nets.DiffusionMLP

::: nnx.diffusion.nets.sinusoidal_time_embed

::: nnx.diffusion.training.diffusion_train_step_factory

::: nnx.diffusion.sampling.sample

## 12. Training paradigms (`nnx.paradigms`)

::: nnx.paradigms.distillation.kd_train_step_factory

::: nnx.paradigms.distillation.feature_kd_train_step_factory

::: nnx.paradigms.contrastive.simclr_train_step_factory

::: nnx.paradigms.contrastive.nt_xent_loss

::: nnx.paradigms.augmentation.mixup_train_step_factory

::: nnx.paradigms.augmentation.cutmix_train_step_factory

## 13. Visualization

### 12.1. Run-output viz (`nnx.vis_utils`)

::: nnx.vis_utils
    options:
      members:
        - confusion_matrix
        - classification_report
        - multi_line_plot
        - scatter_plot
        - two_dim_tsne_checkpoint_logits

### 12.2. Model-internals viz (`nnx.viz`)

::: nnx.viz.summary.summary

::: nnx.viz.weight_histogram.weight_histogram

## 13. Utilities
## 14. Utilities

::: nnx.utils
    options:
      members:
        - print_tree
        - print_table
        - flatten_dict

### 14.1. `Utils` back-compat facade

`nnx.Utils` is a thin staticmethod facade over the module-level functions above, kept so existing notebook code that calls `Utils.print_tree(...)` / `Utils.print_table(...)` / `Utils.flatten_dict(...)` continues to work. New code should prefer the module-level functions directly.
