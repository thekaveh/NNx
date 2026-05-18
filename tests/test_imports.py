"""Smoke tests: every module in nnx imports without error.

This catches broken imports early — most common rot when restructuring.
"""


def test_top_level_imports():
    import nnx
    from nnx import utils, vis_utils
    assert utils is not None
    assert vis_utils is not None


def test_nn_subpackage_imports():
    from nnx.nn import nn_model
    assert nn_model is not None


def test_net_modules_import():
    from nnx.nn.net import feed_fwd_nn, graph_conv_nn, graph_sage_nn, graph_att_nn
    assert feed_fwd_nn is not None
    assert graph_conv_nn is not None
    assert graph_sage_nn is not None
    assert graph_att_nn is not None


def test_dataset_modules_import():
    from nnx.nn.dataset import nn_dataset_base, nn_dataset, nn_graph_dataset
    assert nn_dataset_base is not None
    assert nn_dataset is not None
    assert nn_graph_dataset is not None


def test_enum_modules_import():
    from nnx.nn.enum import activations, checkpoints, devices, losses, nets, optims
    assert activations is not None
    assert checkpoints is not None
    assert devices is not None
    assert losses is not None
    assert nets is not None
    assert optims is not None


def test_params_modules_import():
    from nnx.nn.params import (
        nn_model_params,
        nn_train_params,
        nn_optim_params,
        nn_checkpoint,
        nn_run,
        nn_evaluation_data_point,
        nn_iteration_data_point,
        nn_scheduler_params,
        nn_params,
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
