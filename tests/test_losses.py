"""Tests for the Losses enum factory.

Each variant must return the correct torch.nn loss class. Catches the regression
where MEAN_SQUARED_ERROR and BINARY_CROSS_ENTROPY were swapped in the match
expression — a silent bug shipped between commits 3962995 and the next release.
"""

from __future__ import annotations

from torch import nn

from nnx.nn.enum.losses import Losses


def test_cross_entropy_returns_ce_loss():
    loss = Losses.CROSS_ENTROPY()
    assert isinstance(loss, nn.CrossEntropyLoss)


def test_mean_squared_error_returns_mse_loss():
    """Regression: was returning nn.BCELoss() in a swap with BCE."""
    loss = Losses.MEAN_SQUARED_ERROR()
    assert isinstance(loss, nn.MSELoss)


def test_binary_cross_entropy_returns_logits_loss():
    """Regression: was returning nn.MSELoss() in a swap with MSE."""
    loss = Losses.BINARY_CROSS_ENTROPY()
    assert isinstance(loss, nn.BCEWithLogitsLoss)


def test_negative_log_likelihood_returns_nll_loss():
    loss = Losses.NEGATIVE_LOG_LIKELIHOOD()
    assert isinstance(loss, nn.NLLLoss)


def test_all_enum_variants_have_a_factory_branch():
    """If a new variant is added to Losses, this test fails until the factory
    is updated. Belt-and-braces against another silent miss in __call__."""
    for variant in Losses:
        loss = variant()
        assert loss is not None
        assert isinstance(loss, nn.Module)


# --- FIX-001: native NLLLoss must see log-probabilities, not raw logits ----


def _nll_two_class_model(loss_enum=Losses.NEGATIVE_LOG_LIKELIHOOD):
    import torch

    from nnx import Devices, Nets, NNModel, NNModelParams, NNParams

    model = NNModel(
        net_params=NNParams(input_dim=2, output_dim=2, hidden_dims=[], dropout_prob=0),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=loss_enum),
    )
    with torch.no_grad():
        model.net.layers[-1].weight.zero_()
        model.net.layers[-1].bias.copy_(torch.tensor([2.0, 1.0]))
    return model


def test_nll_normalizes_builtin_logits():
    """Built-in nets emit raw logits; `torch.nn.NLLLoss` requires log
    probabilities. Evaluation and the default training step must report
    the normalized NLL (0.31326166 for logits [2, 1] and target 0) while
    `predict().logits` stays raw."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from nnx import Optims, TrainStepContext, default_train_step

    model = _nll_two_class_model()
    x, y = torch.zeros(2, 2), torch.zeros(2, dtype=torch.long)
    loss = model.evaluate(DataLoader(TensorDataset(x, y), batch_size=2)).loss
    assert abs(loss - 0.31326166) < 1e-6
    assert model.predict(x).logits.tolist() == [[2.0, 1.0], [2.0, 1.0]]

    optimizer = Optims.ADAM(net=model.net, lr_start=0.0, momentum=(0.9, 0.999), weight_decay=0.0)
    ctx = TrainStepContext(
        model=model,
        batch=(x, y),
        optimizer=optimizer,
        scaler=None,
        grad_clip_norm=None,
        extra_metrics=None,
        accumulate_grad_batches=1,
        batch_idx=0,
        epoch_idx=0,
        is_last_batch=True,
    )
    edp = default_train_step(ctx)
    assert edp.loss is not None and abs(edp.loss - 0.31326166) < 1e-6
    # The backpropagated objective is the normalized one: its gradient on
    # the target logit is (softmax - onehot), never the raw -1 of an
    # unnormalized NLL.
    bias_grad = model.net.layers[-1].bias.grad
    assert bias_grad is not None
    expected = torch.softmax(torch.tensor([2.0, 1.0]), dim=0) - torch.tensor([1.0, 0.0])
    assert torch.allclose(bias_grad, expected, atol=1e-6)


def test_nll_preserves_custom_subclass_input():
    """A user-defined NLLLoss *subclass* with its own forward keeps its
    documented supplied-input contract: it receives the network's raw
    output unchanged (no inferred log-softmax)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    seen: list[torch.Tensor] = []

    class RecordingNLL(nn.NLLLoss):
        def forward(self, input, target):  # type: ignore[override]
            seen.append(input.detach().clone())
            return super().forward(input, target)

    model = _nll_two_class_model()
    model.loss_fn = RecordingNLL()
    x, y = torch.zeros(2, 2), torch.zeros(2, dtype=torch.long)
    loss = model.evaluate(DataLoader(TensorDataset(x, y), batch_size=2)).loss
    assert seen and torch.equal(seen[0], torch.tensor([[2.0, 1.0], [2.0, 1.0]]))
    assert loss == -2.0
