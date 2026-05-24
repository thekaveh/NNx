# API Reference

Auto-generated from docstrings via [mkdocstrings](https://mkdocstrings.github.io/).

## Top-level package

::: nnx
    options:
      members:
        - __version__
        - set_seed
        - dataloader_worker_init_fn
        - env_snapshot

## NNModel — the orchestrator

::: nnx.nn.nn_model.NNModel

::: nnx.nn.nn_model.PredictResult

## Params

::: nnx.nn.params.nn_params.NNParams

::: nnx.nn.params.nn_model_params.NNModelParams

::: nnx.nn.params.nn_train_params.NNTrainParams

::: nnx.nn.params.nn_optim_params.NNOptimParams

::: nnx.nn.params.nn_scheduler_params.NNSchedulerParams

::: nnx.nn.params.nn_run.NNRun

::: nnx.nn.params.nn_checkpoint.NNCheckpoint

::: nnx.nn.params.nn_iteration_data_point.NNIterationDataPoint

::: nnx.nn.params.nn_evaluation_data_point.NNEvaluationDataPoint

## Callbacks

::: nnx.nn.callbacks.Callback

::: nnx.nn.callbacks.EarlyStopping

::: nnx.nn.callbacks.LRMonitor

::: nnx.nn.callbacks.ModelCheckpoint

::: nnx.nn.callbacks.TensorBoardCallback

::: nnx.nn.callbacks.WandbCallback

## Networks

::: nnx.nn.net.feed_fwd_nn.FeedFwdNN

::: nnx.nn.net.graph_nn_base.GraphNNBase

::: nnx.nn.net.graph_conv_nn.GraphConvNN

::: nnx.nn.net.graph_sage_nn.GraphSageNN

::: nnx.nn.net.graph_att_nn.GraphAttNN

## Datasets

::: nnx.nn.dataset.nn_dataset.NNDataset

::: nnx.nn.dataset.nn_graph_dataset.NNGraphDataset

::: nnx.nn.dataset.nn_tabular_dataset.NNTabularDataset

## Enums

::: nnx.nn.enum.activations.Activations

::: nnx.nn.enum.checkpoints.Checkpoints

::: nnx.nn.enum.devices.Devices

::: nnx.nn.enum.losses.Losses

::: nnx.nn.enum.nets.Nets

::: nnx.nn.enum.optims.Optims

::: nnx.nn.enum.schedulers.Schedulers

## Visualization

::: nnx.vis_utils
    options:
      members:
        - confusion_matrix
        - classification_report
        - multi_line_plot
        - scatter_plot
        - two_dim_tsne_checkpoint_logits

## Utilities

::: nnx.utils
    options:
      members:
        - print_tree
        - print_table
        - flatten_dict
