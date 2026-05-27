"""Smoke tests: every module in nnx imports without error.

This catches broken imports early — most common rot when restructuring.
The full list below is intentionally exhaustive: when a new module lands
under src/nnx/, add it here so the next refactor surfaces broken imports
loudly rather than letting them linger until someone tries to use the
affected feature.
"""


def test_top_level_imports():
    import nnx
    from nnx import diffusion, finetune, paradigms, peft, seeding, trainer, utils, vis_utils
    assert nnx is not None
    assert utils is not None
    assert vis_utils is not None
    assert seeding is not None
    assert finetune is not None
    assert trainer is not None
    assert diffusion is not None
    assert paradigms is not None
    assert peft is not None


def test_finetune_submodules_import():
    from nnx.finetune import freezing, loading, param_groups
    assert freezing is not None
    assert loading is not None
    assert param_groups is not None


def test_trainer_submodules_import():
    from nnx.trainer import params, trainer
    assert params is not None
    assert trainer is not None


def test_diffusion_submodules_import():
    from nnx.diffusion import nets, sampling, schedules, training
    assert nets is not None
    assert sampling is not None
    assert schedules is not None
    assert training is not None


def test_paradigms_submodules_import():
    from nnx.paradigms import augmentation, contrastive, distillation
    assert augmentation is not None
    assert contrastive is not None
    assert distillation is not None


def test_peft_submodules_import():
    from nnx.peft import adapters, lora
    assert adapters is not None
    assert lora is not None


def test_nn_subpackage_imports():
    from nnx.nn import callbacks, nn_model
    assert nn_model is not None
    assert callbacks is not None


def test_net_modules_import():
    from nnx.nn.net import (
        feed_fwd_nn,
        graph_att_nn,
        graph_conv_nn,
        graph_nn_base,
        graph_sage_nn,
    )
    assert feed_fwd_nn is not None
    assert graph_conv_nn is not None
    assert graph_sage_nn is not None
    assert graph_att_nn is not None
    assert graph_nn_base is not None


def test_dataset_modules_import():
    from nnx.nn.dataset import (
        nn_dataset,
        nn_dataset_base,
        nn_graph_dataset,
        nn_tabular_dataset,
    )
    assert nn_dataset_base is not None
    assert nn_dataset is not None
    assert nn_graph_dataset is not None
    assert nn_tabular_dataset is not None


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
    assert activations is not None
    assert checkpoints is not None
    assert devices is not None
    assert losses is not None
    assert nets is not None
    assert optims is not None
    assert schedulers is not None


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
    )
    assert nn_model_params is not None
    assert nn_train_params is not None
    assert nn_optim_params is not None
    assert nn_checkpoint is not None
    assert nn_run is not None
    assert nn_evaluation_data_point is not None
    assert nn_iteration_data_point is not None
    assert nn_scheduler_params is not None
    assert nn_params is not None
