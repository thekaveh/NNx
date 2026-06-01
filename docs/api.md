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
        - LRFinderResult
        - lr_finder

## 2. Orchestrators

### 2.1. NNModel — supervised orchestrator

::: nnx.nn.nn_model.NNModel

::: nnx.nn.nn_model.PredictResult

::: nnx.nn.nn_model.TrainStepContext

::: nnx.nn.nn_model.default_train_step

### 2.2. GenerativeNNModel — decoder-only LM orchestrator

::: nnx.nn.generative_nn_model.GenerativeNNModel

### 2.3. Trainer — multi-optimizer orchestrator

::: nnx.trainer.trainer.Trainer

::: nnx.trainer.trainer.TrainerStepContext

::: nnx.trainer.trainer.TrainerStepFn

::: nnx.trainer.params.NNTrainerParams

## 3. Params

::: nnx.nn.params.nn_params.NNParams

::: nnx.nn.params.nn_model_params.NNModelParams

::: nnx.nn.params.nn_train_params.NNTrainParams

::: nnx.nn.params.nn_optim_params.NNOptimParams

::: nnx.nn.params.nn_optim_params_builder.NNOptimParamsBuilder

::: nnx.nn.params.nn_scheduler_params.NNSchedulerParams

::: nnx.nn.params.nn_scheduler_params_builder.NNSchedulerParamsBuilder

::: nnx.nn.params.nn_transformer_params.NNTransformerParams

::: nnx.nn.params.nn_tokenizer_params.NNTokenizerParams

::: nnx.nn.params.nn_tokenizer_params.train_bpe

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

::: nnx.nn.net.transformer_nn.TransformerNN

::: nnx.nn.net.vit_nn.ViTNN

::: nnx.nn.net.vit_nn.ViTBlock

::: nnx.nn.moe.MoELinear

## 5. Datasets

::: nnx.nn.dataset.nn_dataset_base.NNDatasetBase

::: nnx.nn.dataset.nn_dataset.NNDataset

::: nnx.nn.dataset.nn_graph_dataset.NNGraphDataset

::: nnx.nn.dataset.nn_tabular_dataset.NNTabularDataset

::: nnx.nn.dataset.nn_preference_dataset.NNPreferenceDataset

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

LoRA + DoRA + IA3 + Prefix-Tuning + Prompt-Tuning + Adapters. All methods share the same in-place wrap + save/load idiom (per-method `save_*_weights` / `load_*_weights` persist only the trainable delta).

### 9.1. LoRA

::: nnx.peft.lora.LoRALinear

::: nnx.peft.lora.apply_lora_to

::: nnx.peft.lora.save_lora_weights

::: nnx.peft.lora.load_lora_weights

### 9.2. DoRA

::: nnx.peft.dora.DoRALinear

::: nnx.peft.dora.apply_dora_to

### 9.3. IA3

::: nnx.peft.ia3.IA3Linear

::: nnx.peft.ia3.apply_ia3_to

::: nnx.peft.ia3.save_ia3_weights

::: nnx.peft.ia3.load_ia3_weights

### 9.4. Prefix Tuning

::: nnx.peft.prefix.PrefixTuner

::: nnx.peft.prefix.save_prefix_weights

::: nnx.peft.prefix.load_prefix_weights

### 9.5. Prompt Tuning

::: nnx.peft.prompt.PromptTuner

::: nnx.peft.prompt.save_prompt_weights

::: nnx.peft.prompt.load_prompt_weights

### 9.6. Adapters

::: nnx.peft.adapters.AdapterLayer

## 10. Pruning (`nnx.prune`)

::: nnx.prune.magnitude.magnitude_prune

::: nnx.prune.semi_structured.semi_structured_24

## 11. Model surgery (`nnx.surgery`)

Walkthrough at [Model surgery](surgery.md). Every primitive returns a fresh `nn.Module` and composes with `NNModel.train()` for the "load checkpoint → surgery → refine" loop.

::: nnx.surgery.widen.widen

::: nnx.surgery.deepen.deepen

::: nnx.surgery.drop_layer.drop_layer

::: nnx.surgery.low_rank.low_rank_factorize

::: nnx.surgery.embedding.expand_embedding

## 12. Quantization (`nnx.quantize`)

PTQ INT8 weight-only + QAT 8da4w via [`torchao`](https://github.com/pytorch/ao) (the replacement for the removed `torch.ao.quantization`). Opt-in via `pip install "nnx[quantize]"`.

::: nnx.quantize.ptq.quantize_int8

::: nnx.quantize.qat.qat_train_step_factory

::: nnx.quantize.qat.QATLifecycleCallback

## 13. Diffusion (`nnx.diffusion`)

::: nnx.diffusion.schedules.NoiseSchedulers

::: nnx.diffusion.schedules.NoiseSchedule

::: nnx.diffusion.nets.DiffusionMLP

::: nnx.diffusion.nets.sinusoidal_time_embed

::: nnx.diffusion.training.diffusion_train_step_factory

::: nnx.diffusion.sampling.sample

## 14. Training paradigms (`nnx.paradigms`)

Each factory returns a `TrainStepFn` for the `train_step_fn=` hook on `NNModel.train`. The training loop, checkpoint cadence, callbacks, and persistence are unchanged — only the per-batch update is swapped.

### 14.1. Knowledge distillation

::: nnx.paradigms.distillation.kd_train_step_factory

::: nnx.paradigms.distillation.feature_kd_train_step_factory

::: nnx.paradigms.born_again.born_again_train

### 14.2. Contrastive

::: nnx.paradigms.contrastive.simclr_train_step_factory

::: nnx.paradigms.contrastive.nt_xent_loss

### 14.3. Augmentation

::: nnx.paradigms.augmentation.mixup_train_step_factory

::: nnx.paradigms.augmentation.cutmix_train_step_factory

### 14.4. Mixture-of-Experts

`MoELinear` is the drop-in layer (documented in §4); `moe_train_step_factory` adds the Switch-style load-balancing aux loss to the supervised step.

::: nnx.paradigms.moe.moe_train_step_factory

### 14.5. I-JEPA

Walkthrough at [I-JEPA](jepa.md). The `ViTNN` encoder is documented in §4.

::: nnx.paradigms.jepa.jepa_train_step_factory

::: nnx.paradigms.jepa.JEPAPredictor

::: nnx.paradigms.jepa.build_target_encoder

::: nnx.paradigms.jepa.update_ema

::: nnx.paradigms.jepa.random_block_mask

### 14.6. DPO

Walkthrough at [DPO](dpo.md).

::: nnx.paradigms.dpo.dpo_train_step_factory

## 15. Embeddings (`nnx.embeddings`)

End-to-end walkthrough at [Embeddings](embeddings.md). Opt-in via `pip install "nnx[embeddings]"`.

::: nnx.embeddings.contrastive_trainer.ContrastiveTextDataset

::: nnx.embeddings.contrastive_trainer.train_contrastive

::: nnx.embeddings.contrastive_trainer.embed_texts

::: nnx.embeddings.contrastive_trainer.text_contrastive_train_step_factory

::: nnx.embeddings.faiss_export.export_to_faiss

::: nnx.embeddings.faiss_export.export_to_safetensors

## 16. Interop (`nnx.interop`)

### 16.1. GGUF + Ollama

End-to-end walkthrough at [GGUF & Ollama](gguf.md). Opt-in via `pip install "nnx[gguf-write]"`.

::: nnx.interop.gguf.writer.write_gguf

::: nnx.interop.gguf.tensor_name_map.map_tensors

::: nnx.interop.ollama.export_ollama_modelfile

## 17. HuggingFace Hub + safetensors

Opt-in via `pip install "nnx[hub]"`. Two integration surfaces:

- **safetensors checkpoints** — `NNCheckpoint.to_file(..., format="safetensors")` and `NNCheckpoint.from_file(..., format="safetensors")` (see §3 `NNCheckpoint`) read and write checkpoints in the safetensors format alongside the default pickle path. Loadable by outside-Python tools (ComfyUI, vLLM, AutoGPTQ).
- **Hub publish / load** — `NNModel` mixes in `huggingface_hub.PyTorchModelHubMixin`, so `save_pretrained(local_dir)`, `push_to_hub(repo_id)`, and `NNModel.from_pretrained(repo_id)` work directly on a trained model. The mixin methods are inherited and live on `NNModel` itself — see §2.1.

Walkthrough at [HuggingFace Hub](hub.md).

## 18. Generation (`nnx.generation`)

`LogitsProcessor` chain for autoregressive sampling. Used by `GenerativeNNModel.generate()` (§2.2). Pure-torch — no optional deps.

::: nnx.generation.LogitsProcessor

::: nnx.generation.TemperatureScaling

::: nnx.generation.TopKFilter

::: nnx.generation.TopPFilter

::: nnx.generation.RepetitionPenalty

::: nnx.generation.apply_chain

::: nnx.generation.sample_next_token

## 19. Visualization

### 19.1. Run-output viz (`nnx.vis_utils`)

::: nnx.vis_utils
    options:
      members:
        - confusion_matrix
        - classification_report
        - multi_line_plot
        - scatter_plot
        - two_dim_tsne_checkpoint_logits

### 19.2. Model-internals viz (`nnx.viz`)

Opt-in via `pip install "nnx[viz]"` (pulls `torchinfo` + `captum`) and `pip install "nnx[viz-interactive]"` (adds the `netron` browser viewer for `nnx.viz.netron_export(..., launch=True)`).

::: nnx.viz.activation.activation_map

::: nnx.viz.attribute.attribute

::: nnx.viz.gradient_flow.gradient_flow

::: nnx.viz.netron.netron_export

::: nnx.viz.summary.summary

::: nnx.viz.weight_histogram.weight_histogram

## 20. Utilities

::: nnx.utils
    options:
      members:
        - print_tree
        - print_table
        - flatten_dict

### 20.1. `Utils` back-compat facade

`nnx.Utils` is a thin staticmethod facade over the module-level functions above, kept so existing notebook code that calls `Utils.print_tree(...)` / `Utils.print_table(...)` / `Utils.flatten_dict(...)` continues to work. New code should prefer the module-level functions directly.
