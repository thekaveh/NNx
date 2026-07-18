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


def test_packagenotfounderror_is_not_a_public_nnx_symbol():
    """Regression for PR #51: `importlib.metadata.PackageNotFoundError`
    is used inside the `__version__` try/except in `src/nnx/__init__.py`
    and must be underscore-aliased so it doesn't leak as a public
    `nnx.PackageNotFoundError` attribute. Pre-fix the unaliased import
    accidentally exposed the importlib exception as part of the nnx
    public surface — confusing for downstream readers / autocomplete
    consumers and not part of NNx's API. The same pattern applies to any
    future implementation-detail import that lives in the top-level
    `nnx/__init__.py` — alias with a leading underscore."""
    import nnx

    assert not hasattr(nnx, "PackageNotFoundError"), (
        "nnx.PackageNotFoundError leaked into the public namespace — "
        "ensure the importlib.metadata import in src/nnx/__init__.py is "
        "underscore-aliased (e.g. `as _PackageNotFoundError`)."
    )


def test_finetune_submodules_import():
    from nnx.finetune import freezing, loading, param_groups

    assert hasattr(freezing, "freeze")
    assert hasattr(loading, "load_pretrained")
    assert hasattr(param_groups, "NNParamGroupSpec")


def test_trainer_submodules_import():
    from nnx.trainer import params, trainer

    assert hasattr(params, "NNTrainerParams")
    assert hasattr(trainer, "Trainer")


def test_nn_trainer_params_builder_importable_from_subpackage():
    """``from nnx.trainer import NNTrainerParamsBuilder`` must work
    just like ``from nnx.peft import LoRALinear`` does. Pre-fix the
    subpackage ``__init__.py`` re-exported ``NNTrainerParams`` /
    ``Trainer`` / ``TrainerStepContext`` / ``TrainerStepFn`` but
    silently omitted the Builder — even though the top-level
    ``nnx.NNTrainerParamsBuilder`` worked because the top-level
    ``__init__.py`` imports it from ``trainer.params_builder``
    directly, bypassing the subpackage's ``__all__``. Two access
    paths, one missing — a small asymmetry that confuses users who
    follow the ``from <subpackage> import <class>`` convention every
    other builder supports."""
    from nnx.trainer import NNTrainerParams, NNTrainerParamsBuilder

    builder = NNTrainerParams.builder()
    assert isinstance(builder, NNTrainerParamsBuilder)


def test_every_name_in_top_level_all_is_attribute_accessible():
    """The inverse of ``test_subpackages_appear_in_top_level_all``: every
    name listed in ``nnx.__all__`` must actually be bound on the package.
    Catches the regression where a symbol lands in ``__all__`` (typed by
    the author, or pasted from a sibling list) but the corresponding
    ``from .X import Y`` line is missing — ``import nnx`` succeeds, but
    ``from nnx import Y`` and ``nnx.Y`` both fail.

    The existing per-module probes only check 13 named symbols + the
    train-step-factory AST sweep, so a typo'd new entry in ``__all__``
    (or a renamed symbol whose ``__all__`` entry didn't track) would
    silently pass CI."""
    import nnx

    missing = sorted(name for name in nnx.__all__ if not hasattr(nnx, name))
    assert not missing, (
        f"{len(missing)} name(s) in nnx.__all__ are not bound on nnx: {missing}. "
        "Either add the missing `from .X import Y` to src/nnx/__init__.py, "
        "or remove the entry from `__all__`."
    )


def test_subpackages_appear_in_top_level_all():
    """Every subpackage that's attribute-accessible after a plain
    ``import nnx`` (per ``test_subpackages_attribute_accessible_after_plain_import``)
    must also appear in ``nnx.__all__``. ``__all__`` is the
    documented public surface — IDEs, doc generators, Sphinx
    autosummary, and ``from nnx import *`` all read it. Pre-fix only
    four (viz / embeddings / interop / prune) were listed; the other
    eight specialization subpackages (peft / diffusion / finetune /
    generation / paradigms / quantize / surgery / trainer) were
    attribute-accessible but absent from ``__all__``."""
    import nnx

    public_all = set(nnx.__all__)
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
        assert name in public_all, (
            f"nnx.{name} is attribute-accessible but missing from nnx.__all__; "
            "add it to the appropriate section in src/nnx/__init__.py"
        )


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
    """``nnx.interop`` is the experimental GGUF export surface. The top-level
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


def test_top_level_transformer_builder_importable():
    import nnx

    assert hasattr(nnx, "NNTransformerParamsBuilder")
    builder = nnx.NNTransformerParams.builder()
    assert isinstance(builder, nnx.NNTransformerParamsBuilder)


def test_top_level_trainer_builder_importable():
    import nnx

    assert hasattr(nnx, "NNTrainerParamsBuilder")
    builder = nnx.NNTrainerParams.builder()
    assert isinstance(builder, nnx.NNTrainerParamsBuilder)


def test_top_level_logits_chain_builder_importable():
    import nnx

    assert hasattr(nnx, "LogitsChain")
    assert hasattr(nnx, "LogitsChainBuilder")
    chain = nnx.LogitsChain.builder().build()
    assert isinstance(chain, nnx.LogitsChain)
    builder = nnx.LogitsChain.builder()
    assert isinstance(builder, nnx.LogitsChainBuilder)


def test_every_train_step_factory_is_top_level():
    """Convention: every `*_train_step_factory` defined anywhere in the
    package must be re-exported at top-level `nnx.*` AND listed in
    `nnx.__all__` — 11 factories visible in one `nnx.<TAB>`. The
    convention was broken once (PR #54 caught
    `text_contrastive_train_step_factory` reachable only via
    `nnx.embeddings.*`); this test makes a 12th factory — or a refactor
    dropping an existing one from `__all__` — fail loudly. Discovery is
    AST-based over src/nnx so a factory in a brand-new subpackage can't
    hide from the scan."""
    import ast
    from pathlib import Path

    import nnx

    src_root = Path(nnx.__file__).parent
    defined: set[str] = set()
    for py in src_root.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.endswith("_train_step_factory"):
                defined.add(node.name)

    assert len(defined) >= 11, f"factory scan looks broken — found only {sorted(defined)}"
    missing_attr = sorted(n for n in defined if not hasattr(nnx, n))
    missing_all = sorted(n for n in defined if n not in nnx.__all__)
    assert not missing_attr, f"not reachable at top-level nnx.*: {missing_attr}"
    assert not missing_all, f"missing from nnx.__all__: {missing_all}"
