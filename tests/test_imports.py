"""Smoke tests: every module in nnx imports without error.

This catches broken imports early — most common rot when restructuring.
The full list below is intentionally exhaustive: when a new module lands
under src/nnx/, add it here so the next refactor surfaces broken imports
loudly rather than letting them linger until someone tries to use the
affected feature.

Each test asserts one named symbol per module rather than `module is
not None` (Python guarantees the latter on successful import — the
assertion was tautological). The named-symbol check additionally
catches a regression that empties `__all__` or drops a public export.
"""


def test_top_level_imports():
    import nnx
    from nnx import diffusion, finetune, paradigms, peft, seeding, trainer, utils, vis_utils, viz

    # One probe per subpackage to verify the public surface stayed intact.
    assert hasattr(nnx, "NNModel")
    assert hasattr(utils, "print_tree")
    assert hasattr(vis_utils, "VisUtils")
    assert hasattr(seeding, "set_seed")
    assert hasattr(finetune, "freeze")
    assert hasattr(trainer, "Trainer")
    assert hasattr(diffusion, "DiffusionMLP")
    assert hasattr(paradigms, "kd_train_step_factory")
    assert hasattr(paradigms, "feature_kd_train_step_factory")
    assert hasattr(peft, "LoRALinear")
    assert hasattr(viz, "summary")
    assert hasattr(viz, "weight_histogram")


def test_subpackages_attribute_accessible_after_plain_import():
    """README §1.2 advertises dotted-submodule access (e.g.
    ``nnx.interop.write_gguf(...)``) after a plain ``import nnx``.
    For that to work, every advertised subpackage must be bound as
    an attribute of the top-level package — i.e. ``__init__.py``
    must perform either ``from .X import ...`` (which side-effects
    the attribute) or an explicit ``from . import X``.

    Regression: ``nnx.interop`` was previously omitted from the
    explicit ``from . import ...`` line, so ``hasattr(nnx, 'interop')``
    returned False after a plain ``import nnx`` despite the README
    advertising ``nnx.interop.write_gguf``."""
    import nnx

    for name in (
        "diffusion",
        "embeddings",
        "finetune",
        "generation",
        "interop",
        "paradigms",
        "peft",
        "prune",
        "quantize",
        "surgery",
        "trainer",
        "viz",
    ):
        assert hasattr(nnx, name), f"nnx.{name} must be attribute-accessible after plain `import nnx`"


def test_finetune_submodules_import():
    from nnx.finetune import freezing, loading, param_groups

    assert hasattr(freezing, "freeze")
    assert hasattr(loading, "load_pretrained")
    assert hasattr(param_groups, "NNParamGroupSpec")


def test_trainer_submodules_import():
    from nnx.trainer import params, trainer

    assert hasattr(params, "NNTrainerParams")
    assert hasattr(trainer, "Trainer")


def test_diffusion_submodules_import():
    from nnx.diffusion import nets, sampling, schedules, training

    assert hasattr(nets, "DiffusionMLP")
    assert hasattr(sampling, "sample")
    assert hasattr(schedules, "NoiseSchedulers")
    assert hasattr(training, "diffusion_train_step_factory")


def test_paradigms_submodules_import():
    from nnx.paradigms import augmentation, contrastive, distillation

    assert hasattr(augmentation, "mixup_train_step_factory")
    assert hasattr(contrastive, "simclr_train_step_factory")
    assert hasattr(distillation, "kd_train_step_factory")
    assert hasattr(distillation, "feature_kd_train_step_factory")


def test_peft_submodules_import():
    from nnx.peft import adapters, lora

    assert hasattr(adapters, "AdapterLayer")
    assert hasattr(lora, "LoRALinear")


def test_viz_submodules_import():
    # The `summary` / `weight_histogram` re-exports in `nnx.viz.__init__`
    # shadow the same-named submodules under `from nnx.viz import ...`,
    # so probe the dotted-path import directly.
    import importlib

    summary_mod = importlib.import_module("nnx.viz.summary")
    weight_histogram_mod = importlib.import_module("nnx.viz.weight_histogram")
    assert callable(summary_mod.summary)
    assert callable(weight_histogram_mod.weight_histogram)


def test_nn_subpackage_imports():
    from nnx.nn import callbacks, nn_model

    assert hasattr(nn_model, "NNModel")
    assert hasattr(callbacks, "Callback")


def test_net_modules_import():
    from nnx.nn.net import (
        feed_fwd_nn,
        graph_att_nn,
        graph_conv_nn,
        graph_nn_base,
        graph_sage_nn,
        transformer_layers,
        transformer_nn,
    )

    assert hasattr(feed_fwd_nn, "FeedFwdNN")
    assert hasattr(graph_conv_nn, "GraphConvNN")
    assert hasattr(graph_sage_nn, "GraphSageNN")
    assert hasattr(graph_att_nn, "GraphAttNN")
    assert hasattr(graph_nn_base, "GraphNNBase")
    assert hasattr(transformer_layers, "TransformerBlock")
    assert hasattr(transformer_nn, "TransformerNN")


def test_interop_subpackage_imports():
    """``nnx.interop`` is the GGUF / Ollama export surface. The top-level
    package import must succeed even when the optional ``gguf`` dep is
    missing (the writer imports it lazily inside the function body)."""
    from nnx import interop
    from nnx.interop import ollama
    from nnx.interop.gguf import tensor_name_map, writer

    assert hasattr(interop, "write_gguf")
    assert hasattr(interop, "export_ollama_modelfile")
    assert hasattr(ollama, "export_ollama_modelfile")
    assert hasattr(writer, "write_gguf")
    assert hasattr(tensor_name_map, "map_tensors")


def test_generation_subpackage_imports():
    """LogitsProcessor chain — pure-torch, no optional deps. Should
    import without `tokenizers` available."""
    from nnx import generation
    from nnx.generation import logits_processors, sampling

    assert hasattr(generation, "TemperatureScaling")
    assert hasattr(logits_processors, "apply_chain")
    assert hasattr(sampling, "sample_next_token")


def test_transformer_public_surface():
    """Top-level re-exports for the LM-path surface."""
    import nnx

    assert hasattr(nnx, "TransformerNN")
    assert hasattr(nnx, "NNTransformerParams")
    assert hasattr(nnx, "GenerativeNNModel")
    assert hasattr(nnx, "TemperatureScaling")
    assert hasattr(nnx, "TopKFilter")
    assert hasattr(nnx, "TopPFilter")
    assert hasattr(nnx, "RepetitionPenalty")
    assert hasattr(nnx, "apply_chain")
    assert hasattr(nnx, "sample_next_token")
    # The Nets enum gained the new variant.
    assert nnx.Nets.TRANSFORMER.value == "transformer"


def test_dataset_modules_import():
    from nnx.nn.dataset import (
        nn_dataset,
        nn_dataset_base,
        nn_graph_dataset,
        nn_tabular_dataset,
    )

    assert hasattr(nn_dataset_base, "NNDatasetBase")
    assert hasattr(nn_dataset, "NNDataset")
    assert hasattr(nn_graph_dataset, "NNGraphDataset")
    assert hasattr(nn_tabular_dataset, "NNTabularDataset")


def test_enum_modules_import():
    from nnx.nn.enum import (
        activations,
        checkpoints,
        devices,
        losses,
        nets,
        optims,
        schedulers,
    )

    assert hasattr(activations, "Activations")
    assert hasattr(checkpoints, "Checkpoints")
    assert hasattr(devices, "Devices")
    assert hasattr(losses, "Losses")
    assert hasattr(nets, "Nets")
    assert hasattr(optims, "Optims")
    assert hasattr(schedulers, "Schedulers")


def test_params_modules_import():
    from nnx.nn.params import (
        nn_checkpoint,
        nn_evaluation_data_point,
        nn_iteration_data_point,
        nn_model_params,
        nn_optim_params,
        nn_params,
        nn_run,
        nn_scheduler_params,
        nn_train_params,
        nn_transformer_params,
    )

    assert hasattr(nn_model_params, "NNModelParams")
    assert hasattr(nn_train_params, "NNTrainParams")
    assert hasattr(nn_optim_params, "NNOptimParams")
    assert hasattr(nn_checkpoint, "NNCheckpoint")
    assert hasattr(nn_run, "NNRun")
    assert hasattr(nn_evaluation_data_point, "NNEvaluationDataPoint")
    assert hasattr(nn_iteration_data_point, "NNIterationDataPoint")
    assert hasattr(nn_scheduler_params, "NNSchedulerParams")
    assert hasattr(nn_params, "NNParams")
    assert hasattr(nn_transformer_params, "NNTransformerParams")


def test_top_level_scheduler_builder_importable():
    """`nnx.NNSchedulerParamsBuilder` is the canonical top-level handle
    for the new Builder. Reachable via `nnx.NNSchedulerParams.builder()`
    too, but the explicit name lets users `from nnx import
    NNSchedulerParamsBuilder` if they want to type-annotate against it.
    """
    import nnx

    assert hasattr(nnx, "NNSchedulerParamsBuilder")
    builder = nnx.NNSchedulerParams.builder()
    assert isinstance(builder, nnx.NNSchedulerParamsBuilder)


def test_top_level_optim_builder_importable():
    """`nnx.NNOptimParamsBuilder` is the canonical top-level handle
    for the new Builder."""
    import nnx

    assert hasattr(nnx, "NNOptimParamsBuilder")
    builder = nnx.NNOptimParams.builder()
    assert isinstance(builder, nnx.NNOptimParamsBuilder)


def test_top_level_transformer_params_builder_importable():
    import nnx

    assert hasattr(nnx, "NNTransformerParamsBuilder")
    builder = nnx.NNTransformerParams.builder()
    assert isinstance(builder, nnx.NNTransformerParamsBuilder)


def test_top_level_trainer_params_builder_importable():
    import nnx

    assert hasattr(nnx, "NNTrainerParamsBuilder")
    builder = nnx.NNTrainerParams.builder()
    assert isinstance(builder, nnx.NNTrainerParamsBuilder)
