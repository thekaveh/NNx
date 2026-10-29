"""Tests for ``nnx.quantize.qat`` — torchao QAT 8da4w.

torchao is an opt-in extra (``pip install thekaveh-nnx[quantize]``); the entire
module is skipped when it isn't importable so contributors who don't
install the extra still get a clean ``pytest -q``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Callback,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNTrainParams,
    Optims,
    QATLifecycleCallback,
    default_train_step,
    qat_train_step_factory,
)

# Skip the whole module when torchao is unavailable. The factory + callback
# raise clear ImportErrors on construction; running the rest of the suite
# without torchao still has value (just not these tests).
pytest.importorskip("torchao")


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


def _make_model(hidden_dims: list[int] | None = None) -> NNModel:
    """Small classifier whose hidden widths divide the default int4
    groupsize (32). Width 64 = 32 * 2, so the 8da4w quantizer applies
    cleanly without requiring ``padding_allowed=True``."""
    return NNModel(
        net_params=NNParams(
            input_dim=32,
            output_dim=3,
            hidden_dims=hidden_dims or [64, 64],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )


def _make_loader(n: int = 64, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(3, 32, generator=g) * 2.0
    cls = torch.randint(0, 3, (n,), generator=g)
    X = means[cls] + 0.5 * torch.randn(n, 32, generator=g)
    return DataLoader(TensorDataset(X, cls), batch_size=16, shuffle=True)


@pytest.fixture
def tiny_model() -> NNModel:
    torch.manual_seed(0)
    return _make_model()


# -------------------------------------------------------------------------
# Factory validation
# -------------------------------------------------------------------------


def test_qat_train_step_factory_rejects_unknown_config():
    """Bad ``qat_config`` must raise eagerly at factory construction —
    before ``model.train(...)`` spins up — so typos surface fast."""
    with pytest.raises(ValueError, match="unknown qat_config"):
        qat_train_step_factory(qat_config="bogus-recipe")


@pytest.mark.parametrize("bad_groupsize", [0, -32])
def test_qat_lifecycle_callback_rejects_non_positive_groupsize(bad_groupsize):
    """``groupsize`` is the int4 weight grouping width. A negative value
    silently builds a mis-quantized model and 0 surfaces only as a cryptic
    ZeroDivisionError deep inside prepare(); both fail fast at construction
    via the shared ``_build_quantizer`` chokepoint — the
    [[params-boundary-validation]] class."""
    with pytest.raises(ValueError, match="groupsize must be a positive int"):
        QATLifecycleCallback(qat_config="8da4w", groupsize=bad_groupsize)


@pytest.mark.parametrize("bad_groupsize", [True, 16.0, 0, -32, np.float64(32.0)])
def test_a_refused_groupsize_is_named_with_its_repr(bad_groupsize):
    """#544: the refusal names groupsize and shows the value's repr (a
    NumPy float printed as an accepted-looking ``32.0`` before)."""
    with pytest.raises(ValueError, match="groupsize") as refused:
        QATLifecycleCallback(groupsize=bad_groupsize)
    assert repr(bad_groupsize) in str(refused.value)


def _qat_trained(groupsize) -> tuple[NNModel, QATLifecycleCallback]:
    torch.manual_seed(0)
    model = _make_model(hidden_dims=[64, 64])
    callback = QATLifecycleCallback(groupsize=groupsize)
    loader = DataLoader(_make_loader(n=32, seed=0).dataset, batch_size=16)
    model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=loader,
            optim=NNOptimParams(name=Optims.SGD, max_lr=1e-2, momentum=0.0, weight_decay=0.0),
            overwrite_existing=True,
        ),
        callbacks=[callback],
        train_step_fn=qat_train_step_factory(),
        salt=f"groupsize-{type(groupsize).__name__}",
    )
    return model, callback


def test_a_numpy_groupsize_trains_and_converts_like_the_plain_int(tmp_path, monkeypatch):
    """#544: torchao takes a NumPy groupsize, and so does the callback; it
    stores the plain int, which the checkpoint transform options (JSON in
    safetensors metadata) hold."""
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.chdir(tmp_path)
    reference, _ = _qat_trained(32)
    x = torch.randn(4, 32, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = reference.net(x)
    for groupsize in (np.int64(32), np.int32(32)):
        model, callback = _qat_trained(groupsize)
        assert callback.is_converted and callback.groupsize == 32 and type(callback.groupsize) is int
        (transform,) = callback.checkpoint_transforms()
        assert transform.options["groupsize"] == 32 and type(transform.options["groupsize"]) is int
        with torch.no_grad():
            torch.testing.assert_close(model.net(x), expected)


def test_qat_train_step_factory_returns_base_step_unchanged():
    """The factory is a thin wrapper: no base => default_train_step;
    custom base => same callable returned. The fake-quant insertion
    happens in the callback, not in the per-batch step."""
    fn_default = qat_train_step_factory()
    assert fn_default is default_train_step

    def my_step(ctx):  # type: ignore[no-untyped-def] — type signature carried by TrainStepFn
        return None

    fn_custom = qat_train_step_factory(base_step=my_step)
    assert fn_custom is my_step


# -------------------------------------------------------------------------
# Callback lifecycle: prepares + converts at the right times
# -------------------------------------------------------------------------


def _module_class_names(net: torch.nn.Module) -> set[str]:
    """All concrete class names in ``net``'s module tree.

    Used instead of an ``isinstance(m, nn.Linear)`` filter because
    torchao's *converted* linear (``Int8DynActInt4WeightLinear``) does
    NOT subclass ``nn.Linear``: it's a standalone ``nn.Module`` that
    holds packed int4 weights. Filtering by ``nn.Linear`` would hide
    exactly the modules we want to assert on post-convert."""
    return {type(m).__name__ for m in net.modules()}


def test_qat_lifecycle_callback_prepares_and_converts(tiny_model):
    """The callback's on_train_begin must swap nn.Linear for the
    fake-quantized counterpart; on_train_end must convert to the
    truly-quantized variant. Driven manually (bypassing model.train)
    so we exercise the callback contract in isolation."""
    cb = QATLifecycleCallback(qat_config="8da4w")
    assert not cb.is_prepared and not cb.is_converted

    # Snapshot the layer types pre-prepare. Plain nn.Linear from FeedFwdNN.
    pre = _module_class_names(tiny_model.net)
    assert "Linear" in pre, pre

    # Build a minimal stand-in for the train()'s _CallbackContext that
    # exposes just .model — that's all our callback consumes.
    class _Ctx:
        def __init__(self, model):
            self.model = model

    ctx = _Ctx(tiny_model)

    cb.on_train_begin(ctx)
    assert cb.is_prepared and not cb.is_converted
    # Post-prepare: torchao swaps in Int8DynActInt4WeightQATLinear (a
    # subclass of nn.Linear) — identify by class name to stay decoupled
    # from torchao's internal symbol path.
    prepared_types = _module_class_names(tiny_model.net)
    assert any("QAT" in t and "Linear" in t for t in prepared_types), f"no QAT-wrapped Linear found: {prepared_types}"

    # Forward pass still works on the prepared model — fake-quant is
    # transparent to the standard forward.
    out = tiny_model.net(torch.randn(2, 32))
    assert out.shape == (2, 3)

    cb.on_train_end(ctx)
    assert cb.is_prepared and cb.is_converted
    converted_types = _module_class_names(tiny_model.net)
    # After convert: Int8DynActInt4WeightLinear (no "QAT" in the name —
    # this is the inference-time module, not the training-time one).
    # Note: the converted linear is NOT an nn.Linear subclass.
    assert any("Int8" in t and "Int4" in t and "QAT" not in t for t in converted_types), (
        f"no truly-quantized Linear found post-convert: {converted_types}"
    )


def test_qat_lifecycle_callback_idempotent(tiny_model):
    """Calling on_train_begin / on_train_end twice must not double-wrap
    or re-convert. Guards against a regression if a caller registers
    the same callback in two back-to-back train() invocations."""
    cb = QATLifecycleCallback(qat_config="8da4w")

    class _Ctx:
        def __init__(self, model):
            self.model = model

    ctx = _Ctx(tiny_model)
    cb.on_train_begin(ctx)
    types_after_first = _module_class_names(tiny_model.net)
    # Second call must be a no-op (early return on is_prepared).
    cb.on_train_begin(ctx)
    types_after_second = _module_class_names(tiny_model.net)
    assert types_after_first == types_after_second

    cb.on_train_end(ctx)
    converted = _module_class_names(tiny_model.net)
    cb.on_train_end(ctx)
    still_converted = _module_class_names(tiny_model.net)
    assert converted == still_converted


def test_qat_lifecycle_callback_convert_without_prepare_noop(tiny_model):
    """Calling on_train_end without ever calling on_train_begin must
    NOT mutate the model (would otherwise yield a partially-quantized
    state that fails inference). It's a silent no-op since
    ``is_prepared`` is False."""
    cb = QATLifecycleCallback(qat_config="8da4w")

    class _Ctx:
        def __init__(self, model):
            self.model = model

    pre = _module_class_names(tiny_model.net)
    cb.on_train_end(_Ctx(tiny_model))
    post = _module_class_names(tiny_model.net)
    assert pre == post
    assert not cb.is_converted


# -------------------------------------------------------------------------
# End-to-end: QAT integrates with model.train()
# -------------------------------------------------------------------------


def test_qat_end_to_end_training(tmp_path, monkeypatch):
    """Full integration: build a tiny classifier, attach the
    QATLifecycleCallback + factory step, run a couple of epochs via
    NNModel.train(), confirm the model emerges converted and still
    produces sane outputs. This is the contract the user actually
    cares about."""
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    model = _make_model(hidden_dims=[64, 64])
    train_loader = _make_loader(n=64, seed=0)

    cb = QATLifecycleCallback(qat_config="8da4w")
    step_fn = qat_train_step_factory(qat_config="8da4w")

    model.train(
        params=NNTrainParams(
            n_epochs=2,
            train_loader=train_loader,
            val_loader=None,
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=1e-2,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            ),
        ),
        callbacks=[cb],
        train_step_fn=step_fn,
    )

    # After train(): callback advanced through the full lifecycle.
    assert cb.is_prepared and cb.is_converted

    # The converted model's linears are the truly-quantized variant —
    # not the QAT (fake-quant) variant, which would mean on_train_end
    # silently skipped. Converted module isn't an nn.Linear subclass,
    # so scan all modules by class name.
    classes = _module_class_names(model.net)
    assert any("Int8" in t and "Int4" in t and "QAT" not in t for t in classes), (
        f"converted model still has QAT linears: {classes}"
    )

    # Forward through the converted model returns logits with the right shape.
    with torch.no_grad():
        out = model.net(torch.randn(4, 32))
    assert out.shape == (4, 3)
    assert torch.isfinite(out).all(), "converted model produced non-finite outputs"


# -------------------------------------------------------------------------
# Post-convert ONNX export
# -------------------------------------------------------------------------


def test_qat_converted_model_onnx_exports(tmp_path, monkeypatch, skip_on_dynamo_dispatch_error):
    """The converted model must round-trip through ``NNModel.to_onnx`` —
    same contract as PTQ. torchao's converted int4-weight linear is
    not a TorchScript-traceable module (legacy ``torch.onnx.export``
    path raises an internal assertion), so we use the ``dynamo=True``
    path, which goes through ``torch.export`` and handles the packed
    quantized weights correctly.

    Requires ``onnxscript`` — install via ``thekaveh-nnx[onnx-dynamo]``. Skipped
    when the installed torch / onnxscript combo can't dispatch one of the
    ops the quantized model emits (see ``_skip_if_dynamo_dispatch_error``
    in ``conftest.py``)."""
    pytest.importorskip("onnxscript")

    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    model = _make_model(hidden_dims=[64, 64])

    cb = QATLifecycleCallback(qat_config="8da4w")
    step_fn = qat_train_step_factory(qat_config="8da4w")
    model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=_make_loader(n=32, seed=0),
            val_loader=None,
            optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
        ),
        callbacks=[cb],
        train_step_fn=step_fn,
    )
    assert cb.is_converted

    onnx_path = tmp_path / "qat.onnx"
    try:
        model.to_onnx(str(onnx_path), example_input=torch.randn(1, 32), dynamo=True)
    except Exception as e:
        skip_on_dynamo_dispatch_error(e)
        raise  # unreachable — helper either skips or re-raises
    assert onnx_path.exists() and onnx_path.stat().st_size > 0


def test_a_converted_recipe_model_trained_again_writes_loadable_tags(tmp_path, monkeypatch):
    """FEAT-016: a recipe model converted by QAT records both on LAST; when
    that converted model trains again, its in-loop tags record the full
    transform list too (not just the recipe), so they replay and load."""
    from nnx.nn.enum.checkpoints import Checkpoints
    from nnx.nn.params.nn_checkpoint import NNCheckpoint
    from nnx.transforms import TransformRecipe, lora

    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    optim = NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0)
    model = TransformRecipe([lora("layers.0", r=4, alpha=8.0)]).materialize(_make_model(hidden_dims=[64, 64]))
    first = model.train(
        params=NNTrainParams(n_epochs=1, train_loader=_make_loader(), optim=optim),
        callbacks=[QATLifecycleCallback(qat_config="8da4w")],
        train_step_fn=qat_train_step_factory(qat_config="8da4w"),
    )
    converted = NNModel.from_checkpoint(NNCheckpoint.load(run=first.id, type=Checkpoints.LAST))
    assert [t.name for t in converted._topology_transforms] == ["lora", "torchao_qat"]
    again = converted.train(params=NNTrainParams(n_epochs=1, train_loader=_make_loader(), optim=optim, data_id="again"))
    for tag in (Checkpoints.FIRST, Checkpoints.BEST):
        checkpoint = NNCheckpoint.load(run=again.id, type=tag)
        assert checkpoint is not None and [t.name for t in checkpoint.transforms] == ["lora", "torchao_qat"]
        NNModel.from_checkpoint(checkpoint)


def test_a_converted_recipe_model_resumes_its_own_in_loop_tags(tmp_path, monkeypatch):
    """FEAT-016: in-loop tags of a converted recipe-plus-QAT model trained
    again record (recipe, conversion); the same converted model resumes them."""
    from nnx.nn.enum.checkpoints import Checkpoints
    from nnx.nn.params.nn_checkpoint import NNCheckpoint
    from nnx.transforms import TransformRecipe, lora

    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    optim = NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0)
    model = TransformRecipe([lora("layers.0", r=4, alpha=8.0)]).materialize(_make_model(hidden_dims=[64, 64]))
    first = model.train(
        params=NNTrainParams(n_epochs=1, train_loader=_make_loader(), optim=optim),
        callbacks=[QATLifecycleCallback(qat_config="8da4w")],
        train_step_fn=qat_train_step_factory(qat_config="8da4w"),
    )

    def converted() -> NNModel:
        return NNModel.from_checkpoint(NNCheckpoint.load(run=first.id, type=Checkpoints.LAST))

    again = converted().train(
        params=NNTrainParams(n_epochs=2, train_loader=_make_loader(), optim=optim, data_id="again")
    )
    resumed = converted().train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=_make_loader(),
            optim=optim,
            data_id="again",
            resume_from_run_id=again.id,
            resume_from_checkpoint="first",
        )
    )
    assert resumed.resume_status is not None and resumed.resume_status.mode == "stateful"


def test_a_split_qat_run_with_dropout_equals_the_uninterrupted_run(tmp_path, monkeypatch):
    """#538: QAT preparation draws from the global RNG in on_train_begin, after
    a stateful resume has restored it; the resumed session's first epoch must
    still start from the restored streams, so 2 + 1 equals 3 straight."""
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.chdir(tmp_path)
    g = torch.Generator().manual_seed(0)
    X, y = torch.randn(48, 32, generator=g), torch.randint(0, 3, (48,), generator=g)
    batches = [(X[i : i + 16], y[i : i + 16]) for i in range(0, 48, 16)]

    def model() -> NNModel:
        torch.manual_seed(7)
        return NNModel(
            net_params=NNParams(
                input_dim=32, output_dim=3, hidden_dims=[64], dropout_prob=0.1, activation=Activations.RELU
            ),
            params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        )

    def fit(target: NNModel, n_epochs: int, data_id: str, **resume):
        return target.train(
            params=NNTrainParams(
                n_epochs=n_epochs,
                data_id=data_id,
                train_loader=batches,
                seed=0,
                optim=NNOptimParams.builder().adam(max_lr=0.01).build(),
                **resume,
            ),
            callbacks=[QATLifecycleCallback()],
            train_step_fn=qat_train_step_factory(),
        )

    whole = model()
    fit(whole, 3, "whole")
    first = fit(model(), 2, "split")
    resumed = model()
    run = fit(resumed, 1, "split", resume_from_run_id=first.id, resume_mode="stateful")
    assert run.resume_status is not None and run.resume_status.mode == "stateful"
    for name, weight in whole.net.state_dict().items():
        assert torch.equal(weight, resumed.net.state_dict()[name]), name


# -------------------------------------------------------------------------
# Layers QAT cannot fake-quantize, and fits that fail before training (#526)
# -------------------------------------------------------------------------


def _fit(model: NNModel, loop: str, callbacks: list, loader, *, n_epochs: int = 1) -> None:
    """A QAT fit through ``NNModel.train`` or ``Trainer.train``."""
    from nnx import NNEvaluationDataPoint, NNTrainerParams, Trainer

    optim = NNOptimParams.builder().sgd(max_lr=0.1).build()
    if loop == "nnmodel":
        params = NNTrainParams(n_epochs=n_epochs, train_loader=loader, optim=optim)
        model.train(params=params, callbacks=callbacks, train_step_fn=qat_train_step_factory())
        return

    def trainer_step(ctx):
        ctx.model.net.train()
        optimizer = ctx.optimizers["main"]
        optimizer.zero_grad()
        x, target = ctx.batch
        loss = ctx.model.loss_fn(ctx.model.net(x), target)
        loss.backward()
        optimizer.step()
        return NNEvaluationDataPoint(loss=float(loss.detach()), error=float(loss.detach()))

    params = NNTrainerParams(n_epochs=n_epochs, train_loader=loader, optims={"main": optim})
    Trainer(model).train(params=params, trainer_step_fn=trainer_step, callbacks=callbacks)


def _types(net: torch.nn.Module) -> dict[str, str]:
    return {name: type(module).__name__ for name, module in net.named_modules() if name}


class _AttentionNet(torch.nn.Module):
    """The ticket's net: MultiheadAttention reads ``out_proj.weight`` and
    never calls ``out_proj``."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(64, 4, batch_first=True)
        self.head = torch.nn.Linear(64, 2)

    def forward(self, x):
        h, _ = self.attn(x, x, x, need_weights=False)
        return self.head(h.mean(1))


@pytest.mark.parametrize("loop", ["nnmodel", "trainer"])
def test_qat_refuses_a_linear_its_parent_reads_directly(loop, tmp_path, monkeypatch):
    """#526: QAT swapped MultiheadAttention's out_proj, which the parent
    never calls: training skipped its fake quantization and the converted
    forward failed (Float and Char). It is refused before prepare."""
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    model = NNModel(module=_AttentionNet(), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    x, y = torch.randn(4, 5, 64), torch.tensor([0, 1, 0, 1])
    before = _types(model.net)
    state = {key: value.clone() for key, value in model.net.state_dict().items()}
    callback = QATLifecycleCallback(groupsize=32)

    with pytest.raises(
        ValueError, match=r"QAT cannot prepare attn\.out_proj \(read by its parent MultiheadAttention\)"
    ):
        _fit(model, loop, [callback], [(x, y)])

    assert _types(model.net) == before
    assert all(torch.equal(value, state[key]) for key, value in model.net.state_dict().items())
    assert not callback.is_prepared and not callback.is_converted
    model.net.eval()
    with torch.no_grad():
        assert model.net(x).shape == (4, 2)


def test_qat_refuses_a_transformer_layers_linears_naming_each():
    """The torch Transformer layers' eval fast path reads linear1 / linear2
    directly as well (``WEIGHT_READING_PARENTS``)."""
    net = torch.nn.Sequential(torch.nn.TransformerEncoderLayer(64, 4, dim_feedforward=64, batch_first=True))

    class _Ctx:
        model = type("M", (), {"net": net})()

    before = _types(net)
    callback = QATLifecycleCallback(groupsize=32)
    with pytest.raises(ValueError) as refusal:
        callback.on_train_begin(_Ctx())
    message = str(refusal.value)
    assert "0.linear1 (read by its parent TransformerEncoderLayer)" in message
    assert "0.linear2 (read by its parent TransformerEncoderLayer)" in message
    assert "0.self_attn.out_proj (read by its parent MultiheadAttention)" in message
    assert _types(net) == before and not callback.is_prepared


@pytest.mark.parametrize("adapter", ["lora", "ia3"])
def test_qat_still_trains_and_converts_lora_and_ia3_adapted_nets(adapter, tmp_path, monkeypatch):
    """LoRA and IA3 call their base layer, so QAT fake-quantizes it as before."""
    from nnx.peft import apply_ia3_to, apply_lora_to

    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    model = _make_model()
    (apply_lora_to if adapter == "lora" else apply_ia3_to)(model.net, "layers.0")
    callback = QATLifecycleCallback()

    _fit(model, "nnmodel", [callback], _make_loader())

    assert callback.is_prepared and callback.is_converted
    assert _types(model.net)["layers.0.base"] == "Int8DynActInt4WeightLinear"
    with torch.no_grad():
        assert torch.isfinite(model.net(torch.randn(4, 32))).all()


class _RaisesOnTrainBegin(Callback):
    def on_train_begin(self, ctx) -> None:
        raise RuntimeError("a later callback failed")


class _RaisesAfterAnEpoch(Callback):
    def on_epoch_end(self, ctx) -> None:
        raise RuntimeError("failed after an update")


@pytest.mark.parametrize("loop", ["nnmodel", "trainer"])
def test_a_fit_failing_before_its_first_update_hands_back_the_unprepared_net(loop, tmp_path, monkeypatch):
    """#526: QAT prepared the net, a later callback's on_train_begin raised,
    and on_train_end still converted it: an int8 net from a fit that ran no
    step. The fit now gets back the very layers it had, unconverted, and
    the callback's flags read as if it never ran."""
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    model = _make_model()
    before, layers = _types(model.net), list(model.net.modules())
    state = {key: value.clone() for key, value in model.net.state_dict().items()}
    x = torch.randn(4, 32)
    with torch.no_grad():
        expected = model.net(x)
    callback = QATLifecycleCallback()

    with pytest.raises(RuntimeError, match="a later callback failed"):
        _fit(model, loop, [callback, _RaisesOnTrainBegin()], _make_loader())

    assert _types(model.net) == before
    assert all(a is b for a, b in zip(model.net.modules(), layers, strict=True))
    assert all(torch.equal(value, state[key]) for key, value in model.net.state_dict().items())
    assert not callback.is_prepared and not callback.is_converted
    assert callback.checkpoint_transforms() == ()
    with torch.no_grad():
        assert torch.equal(model.net(x), expected)

    # The same callback prepares and converts on the retry.
    _fit(model, loop, [callback], _make_loader())
    assert callback.is_prepared and callback.is_converted
    assert _types(model.net)["layers.0"] == "Int8DynActInt4WeightLinear"


@pytest.mark.parametrize("loop", ["nnmodel", "trainer"])
def test_a_fit_failing_after_an_update_still_converts(loop, tmp_path, monkeypatch):
    """Pins the other side of #526: once an update has trained the
    fake-quantized net, a failure keeps that training, converted as before."""
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    model = _make_model()
    callback = QATLifecycleCallback()

    with pytest.raises(RuntimeError, match="failed after an update"):
        _fit(model, loop, [callback, _RaisesAfterAnEpoch()], _make_loader())

    assert callback.is_prepared and callback.is_converted
    assert _types(model.net)["layers.0"] == "Int8DynActInt4WeightLinear"
