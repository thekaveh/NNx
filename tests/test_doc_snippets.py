"""Executable smoke-tests for the canonical NNx documentation snippets.

Each test exercises one documented workflow end-to-end so CI catches
regressions between docs and the shipped library.  Tests are intentionally
tiny (small tensors, 1–2 epochs) to keep the total suite runtime modest.

Intentionally excluded workflows (not unit-testable):
  - TransformerNN / GenerativeNNModel.generate — requires `lm` extra and
    tokenizer downloads (HuggingFace hub, real vocab).
  - Experimental GGUF export — covered by the dedicated writer tests.
  - HuggingFace Hub push / `NNModel.push_to_hub` — requires valid HF token
    and network access.
  - Embeddings / FAISS index — requires `faiss-cpu` extra and real text data.
  - GraphConvNN / GraphSageNN / GraphAttNN — require `torch_geometric` extra
    and a PyG `Data` object.
  - `semi_structured_24` — requires Ampere GPU (CUDA sm_80+); CPU raises
    RuntimeError inside torchao.
  - `to_onnx` — covered by the dedicated test_onnx_dynamo.py suite.
  - Warm-resume training — covered by test_train_integration.py.

Optional-extra gates (tests skip rather than fail when dep absent):
  - Quantization  (`quantize_int8`, QAT): ``pytest.importorskip("torchao")``
  - Viz           (``summary``, ``weight_histogram``, etc.): ``pytest.importorskip("torchinfo")``
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Shared helpers — kept private to this module
# ---------------------------------------------------------------------------


def _make_tiny_loader(
    n: int = 32,
    input_dim: int = 8,
    n_classes: int = 3,
    batch_size: int = 16,
    seed: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(n, input_dim, generator=gen)
    y = torch.randint(0, n_classes, (n,), generator=gen)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


def _make_model(input_dim: int = 8, output_dim: int = 3, hidden: int = 16):
    """Construct a tiny CPU FeedFwdNN NNModel."""
    from nnx import (
        Activations,
        Devices,
        Losses,
        Nets,
        NNModel,
        NNModelParams,
        NNParams,
    )

    net_params = NNParams(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=[hidden],
        dropout_prob=0.0,
        activation=Activations.RELU,
    )
    model_params = NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    return NNModel(net_params=net_params, params=model_params)


def _make_train_params(train_loader, val_loader=None, n_epochs: int = 1):
    from nnx import NNOptimParams, NNSchedulerParams, NNTrainParams, Optims

    return NNTrainParams(
        n_epochs=n_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=2, cooldown=1, threshold=1e-3),
    )


# ---------------------------------------------------------------------------
# 1.  Quickstart
# ---------------------------------------------------------------------------


def test_quickstart_end_to_end(tmp_path, monkeypatch):
    """Docs/quickstart.md §1 — full train+predict pipeline.

    Asserts:
    - run.id is a 32-char hex string (md5 content-address).
    - model.predict returns logits (B, output_dim) and classes (B,).
    - run.idps has the expected count (n_batches × n_epochs).
    """
    monkeypatch.chdir(tmp_path)

    from nnx import (
        Activations,
        Devices,
        EarlyStopping,
        Losses,
        Nets,
        NNModel,
        NNModelParams,
        NNOptimParams,
        NNParams,
        NNSchedulerParams,
        NNTrainParams,
        Optims,
    )

    torch.manual_seed(42)
    train_loader = _make_tiny_loader(n=32, batch_size=16)
    val_loader = _make_tiny_loader(n=16, batch_size=8, seed=1, shuffle=False)

    net_params = NNParams(
        input_dim=8,
        output_dim=3,
        hidden_dims=[32, 16],
        dropout_prob=0.0,
        activation=Activations.RELU,
    )
    model_params = NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    model = NNModel(net_params=net_params, params=model_params)

    train_params = NNTrainParams(
        n_epochs=2,
        seed=42,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=5e-5),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=3, cooldown=1, threshold=1e-3),
    )
    run = model.train(params=train_params, callbacks=[EarlyStopping(patience=5)])

    assert isinstance(run.id, str) and len(run.id) == 32
    assert (tmp_path / "runs" / run.id / "run.yaml").exists()
    assert len(run.idps) == 4  # 2 batches/epoch (32/16) × 2 epochs — one IDP per batch

    import numpy as np

    X_pred = np.random.default_rng(0).standard_normal((4, 8)).astype("float32")
    result = model.predict(X=X_pred)
    assert result.logits.shape == (4, 3)
    assert result.classes.shape == (4,)


# ---------------------------------------------------------------------------
# 2.  Fluent Builders
# ---------------------------------------------------------------------------


def test_builder_optim_params_adam():
    """Fluent-Builders wiki — NNOptimParams.builder().adam(...).build()."""
    from nnx import NNOptimParams, Optims

    opt = NNOptimParams.builder().adam(max_lr=3e-4, betas=(0.9, 0.999), weight_decay=0.0).build()
    assert opt.name == Optims.ADAM
    assert abs(opt.max_lr - 3e-4) < 1e-9

    opt_sgd = (
        NNOptimParams.builder()
        .sgd(max_lr=1e-2, momentum=0.9, weight_decay=5e-5)
        .grad_clip(1.0)
        .accumulate_grad(4)
        .build()
    )
    assert opt_sgd.grad_clip_norm == 1.0
    assert opt_sgd.accumulate_grad_batches == 4


def test_builder_scheduler_params_variants():
    """Fluent-Builders wiki — NNSchedulerParams.builder() variants.

    NNSchedulerParams stores the scheduler kind in the ``kind`` field (None
    means the default ReduceLROnPlateau).  Other variants surface their
    specific keyword arguments as direct fields instead.
    """
    from nnx import NNSchedulerParams

    # ReduceLROnPlateau — the default; kind is None, patience is set.
    sched_plateau = (
        NNSchedulerParams.builder()
        .reduce_on_plateau(min_lr=1e-7, factor=0.5, patience=10, cooldown=2, threshold=1e-3)
        .build()
    )
    assert sched_plateau.kind is None  # default variant has no kind tag
    assert sched_plateau.patience == 10

    # CosineAnnealingLR — kind tag set, T_max surfaced.
    sched_cosine = (
        NNSchedulerParams.builder()
        .cosine_annealing(T_max=100, min_lr=1e-7, factor=0.5, patience=10, cooldown=2, threshold=1e-3)
        .build()
    )
    from nnx import Schedulers

    assert sched_cosine.kind == Schedulers.COSINE_ANNEALING
    assert sched_cosine.T_max == 100

    # LinearWarmupDecay — warmup_steps field must match.
    sched_warmup = (
        NNSchedulerParams.builder()
        .linear_warmup_decay(
            warmup_steps=50,
            total_steps=500,
            min_lr=1e-7,
            factor=0.5,
            patience=10,
            cooldown=2,
            threshold=1e-3,
        )
        .build()
    )
    assert sched_warmup.warmup_steps == 50


def test_builder_transformer_params():
    """Fluent-Builders wiki — NNTransformerParams.builder() full chain."""
    from nnx import NNTransformerParams

    params = (
        NNTransformerParams.builder()
        .vocab(size=1_000)
        .layers(n=2, heads=2, d_model=16)  # 16 % 2 == 0 → ok
        .context(max_seq_len=32, rope_base=500_000.0)
        .ffn(mult=4)
        .dropout(attn=0.1, resid=0.1)
        .tied_embeddings(True)
        .build()
    )
    assert params.vocab_size == 1_000
    assert params.n_layers == 2
    assert params.n_heads == 2
    assert params.d_model == 16
    assert params.max_seq_len == 32
    assert abs(params.rope_base - 500_000.0) < 1.0
    assert params.tie_embeddings is True


def test_builder_trainer_params():
    """Fluent-Builders wiki — NNTrainerParams.builder() for multi-optim."""
    from nnx import NNOptimParams, NNSchedulerParams, NNTrainerParams

    g_opt = NNOptimParams.builder().adam(max_lr=2e-4, betas=(0.5, 0.999)).build()
    d_opt = NNOptimParams.builder().adam(max_lr=2e-4, betas=(0.5, 0.999)).build()
    plateau = (
        NNSchedulerParams.builder()
        .reduce_on_plateau(min_lr=1e-7, factor=0.5, patience=2, cooldown=1, threshold=1e-3)
        .build()
    )

    trainer_params = (
        NNTrainerParams.builder()
        .n_epochs(10)
        .optimizer("G", g_opt)
        .optimizer("D", d_opt)
        .scheduler("G", plateau)
        .build()
    )
    assert trainer_params.n_epochs == 10
    assert "G" in trainer_params.optims
    assert "D" in trainer_params.optims
    assert "G" in trainer_params.schedulers


def test_builder_logits_chain():
    """Fluent-Builders wiki — LogitsChain.builder() canonical order."""
    from nnx import LogitsChain
    from nnx.generation.logits_processors import (
        RepetitionPenalty,
        TemperatureScaling,
        TopKFilter,
        TopPFilter,
    )

    chain = LogitsChain.builder().top_k(50).top_p(0.9).temperature(0.8).repetition_penalty(1.2).build()
    types = [type(p) for p in chain.processors]
    # Canonical order: RepetitionPenalty → TopKFilter → TopPFilter → TemperatureScaling
    assert types == [RepetitionPenalty, TopKFilter, TopPFilter, TemperatureScaling]

    # Forward pass sanity-check
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.5, -0.5]])
    out = chain.apply(logits, token_history=[0])
    assert out.shape == logits.shape


# ---------------------------------------------------------------------------
# 3.  Networks
# ---------------------------------------------------------------------------


def test_networks_feed_fwd_nn_forward():
    """Networks wiki — FeedFwdNN construction and forward pass."""
    from nnx import (
        Activations,
        Devices,
        Losses,
        Nets,
        NNModel,
        NNModelParams,
        NNParams,
    )

    net_params = NNParams(
        input_dim=16,
        output_dim=3,
        hidden_dims=[64, 32],
        dropout_prob=0.1,
        activation=Activations.RELU,
    )
    model = NNModel(
        net_params=net_params,
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    x = torch.randn(4, 16)
    with torch.no_grad():
        out = model.net(x)
    assert out.shape == (4, 3)


def test_networks_moe_linear_forward():
    """Networks wiki — MoELinear drop-in forward; last_aux_loss populated."""
    from nnx import MoELinear

    layer = MoELinear(in_features=16, out_features=16, num_experts=4, top_k=2)
    x = torch.randn(8, 16)
    out = layer(x)
    assert out.shape == (8, 16)
    assert layer.last_aux_loss is not None


def test_networks_vit_nn_forward():
    """Networks wiki — ViTNN (tiny 8×8 image) forward returns CLS + patches."""
    from nnx import ViTNN

    vit = ViTNN(
        image_size=8,
        patch_size=4,
        in_channels=1,
        d_model=16,
        n_layers=2,
        n_heads=2,
        ffn_mult=2,
        attn_dropout=0.0,
        resid_dropout=0.0,
    )
    x = torch.randn(2, 1, 8, 8)  # (B, C, H, W)
    with torch.no_grad():
        out = vit(x)
    # n_patches = (8/4)² = 4, output shape = (B, 4+1, d_model)
    assert out.shape == (2, 5, 16)


# ---------------------------------------------------------------------------
# 4.  Training Loop and Callbacks
# ---------------------------------------------------------------------------


def test_freeze_unfreeze_return_int():
    """Training-Loop wiki — model.freeze / unfreeze return int counts."""
    model = _make_model(input_dim=4, output_dim=2, hidden=8)

    n_frozen = model.freeze("layers.0.*")
    assert isinstance(n_frozen, int) and n_frozen == 2  # weight + bias

    n_unfrozen = model.unfreeze("layers.0.*")
    assert isinstance(n_unfrozen, int) and n_unfrozen == 2


def test_callbacks_early_stopping_lr_monitor_model_checkpoint(tmp_path, monkeypatch):
    """Training-Loop wiki — EarlyStopping, LRMonitor, ModelCheckpoint wired.

    Tiny 2-epoch run; asserts:
    - LRMonitor.history has one entry per epoch.
    - ModelCheckpoint written the file for epoch index 0 (first of `epochs=[0]`).
    """
    monkeypatch.chdir(tmp_path)

    from nnx import EarlyStopping, LRMonitor, ModelCheckpoint

    model = _make_model()
    train_loader = _make_tiny_loader()
    val_loader = _make_tiny_loader(n=16, seed=1, shuffle=False)
    train_params = _make_train_params(train_loader, val_loader, n_epochs=2)

    lr_cb = LRMonitor()
    ckpt_cb = ModelCheckpoint(epochs=[0], tag="milestone")
    model.train(
        params=train_params,
        callbacks=[EarlyStopping(patience=3), lr_cb, ckpt_cb],
    )

    assert len(lr_cb.history) == 2  # one per epoch

    run_dirs = list((tmp_path / "runs").iterdir())
    assert run_dirs, "Expected at least one run directory"
    ckpt_dir = run_dirs[0] / "checkpoints"
    milestone_files = list(ckpt_dir.glob("milestone_e*.pt"))
    assert milestone_files, "ModelCheckpoint did not write milestone file"


# ---------------------------------------------------------------------------
# 5.  Persistence (Runs and Checkpoints)
# ---------------------------------------------------------------------------


def test_persistence_run_load_checkpoint_from_checkpoint(tmp_path, monkeypatch):
    """Persistence wiki — NNRun.load, NNCheckpoint.load, NNModel.from_checkpoint."""
    monkeypatch.chdir(tmp_path)

    import numpy as np

    from nnx import Checkpoints, NNCheckpoint, NNModel, NNRun

    model = _make_model()
    train_loader = _make_tiny_loader()
    run = model.train(params=_make_train_params(train_loader))

    # --- NNRun.load ---
    reloaded = NNRun.load(id=run.id)
    assert reloaded.id == run.id
    assert len(reloaded.idps) == len(run.idps)

    # --- NNCheckpoint.load ---
    ckpt = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert ckpt is not None

    # --- NNModel.from_checkpoint ---
    reconstructed = NNModel.from_checkpoint(checkpoint=ckpt)
    X = np.random.default_rng(7).standard_normal((4, 8)).astype("float32")
    logits, classes = reconstructed.predict(X=X)
    assert logits.shape == (4, 3)
    assert classes.shape == (4,)


# ---------------------------------------------------------------------------
# 6.  Reproducibility
# ---------------------------------------------------------------------------


def test_reproducibility_set_seed():
    """Reproducibility — set_seed produces deterministic FeedFwdNN init."""
    from nnx import set_seed

    set_seed(42)
    model_a = _make_model()

    set_seed(42)
    model_b = _make_model()

    for pa, pb in zip(model_a.net.parameters(), model_b.net.parameters(), strict=True):
        assert torch.equal(pa, pb), "set_seed did not produce identical weights"


def test_reproducibility_dataloader_worker_init_fn():
    """Reproducibility — dataloader_worker_init_fn wires without error."""
    from nnx import dataloader_worker_init_fn

    X = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=4,
        num_workers=0,  # single-process; init_fn is a no-op but must accept
        worker_init_fn=dataloader_worker_init_fn,
    )
    batch = next(iter(loader))
    assert batch[0].shape == (4, 4)


def test_reproducibility_lr_finder():
    """Quickstart §2.8 / Reproducibility — lr_finder runs and returns LRFinderResult."""
    from nnx import lr_finder

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    loader = _make_tiny_loader(n=32, input_dim=4, n_classes=3, batch_size=8, shuffle=False)

    result = lr_finder(
        net,
        loader,
        loss_fn=F.cross_entropy,
        start_lr=1e-6,
        end_lr=1.0,
        num_iter=20,
    )
    assert isinstance(result.suggested_lr, float) and result.suggested_lr > 0
    assert len(result.lrs) == len(result.losses) == 20


# ---------------------------------------------------------------------------
# 7.  Model Surgery
# ---------------------------------------------------------------------------


def test_surgery_widen_function_preserving():
    """Surgery docs §2 — widen preserves the forward output on a FeedFwdNN."""
    from nnx import widen
    from nnx.nn.enum.activations import Activations
    from nnx.nn.net.feed_fwd_nn import FeedFwdNN
    from nnx.nn.params.nn_params import NNParams

    torch.manual_seed(7)
    params = NNParams(
        input_dim=6,
        output_dim=2,
        hidden_dims=[8],
        dropout_prob=0.0,
        activation=Activations.RELU,
    )
    net = FeedFwdNN(params)
    net.eval()
    x = torch.randn(4, 6)
    orig_out = net(x)

    wider = widen(net, layer_name="layers.0", new_width=16)
    wider.eval()
    new_out = wider(x)

    assert torch.allclose(orig_out, new_out, atol=1e-5), (
        f"widen broke function-preservation: max diff {(orig_out - new_out).abs().max().item():.2e}"
    )


def test_surgery_deepen_function_preserving():
    """Surgery docs §2 — deepen preserves forward output on a ReLU net."""
    from nnx import deepen

    torch.manual_seed(8)
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    net.eval()
    x = torch.randn(4, 4)
    orig_out = net(x)

    deeper = deepen(net, after_layer_name="1")  # insert after the ReLU at index 1
    deeper.eval()
    new_out = deeper(x)

    assert torch.allclose(orig_out, new_out, atol=1e-5), (
        f"deepen broke function-preservation: max diff {(orig_out - new_out).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# 8.  Pruning
# ---------------------------------------------------------------------------


def test_pruning_magnitude_prune(tmp_path, monkeypatch):
    """Pruning wiki — magnitude_prune zeros 50% of weights; count > 0."""
    monkeypatch.chdir(tmp_path)

    from nnx.prune import magnitude_prune

    model = _make_model()
    n = magnitude_prune(model.net, sparsity=0.5)
    assert isinstance(n, int) and n > 0

    # After bake=True (default), weight tensors should have zeros
    total_zeros = 0
    for module in model.net.modules():
        if isinstance(module, nn.Linear):
            total_zeros += (module.weight == 0.0).sum().item()
    assert total_zeros > 0


# ---------------------------------------------------------------------------
# 9.  Training Paradigms
# ---------------------------------------------------------------------------


def test_paradigms_mixup(tmp_path, monkeypatch):
    """Training-Paradigms wiki — mixup_train_step_factory runs a tiny epoch."""
    monkeypatch.chdir(tmp_path)

    from nnx import mixup_train_step_factory

    model = _make_model()
    train_loader = _make_tiny_loader()
    step_fn = mixup_train_step_factory(alpha=0.4)
    run = model.train(params=_make_train_params(train_loader), train_step_fn=step_fn)
    assert len(run.idps) > 0


def test_paradigms_born_again(tmp_path, monkeypatch):
    """Training-Paradigms wiki — born_again_train returns list[NNRun] per generation."""
    monkeypatch.chdir(tmp_path)

    from nnx import born_again_train

    model = _make_model(input_dim=4, output_dim=2, hidden=8)
    train_loader = _make_tiny_loader(n=32, input_dim=4, n_classes=2, batch_size=16)
    train_params = _make_train_params(train_loader)

    runs = born_again_train(model, generations=2, train_params=train_params, alpha=0.5, temperature=4.0)
    assert len(runs) == 2


# ---------------------------------------------------------------------------
# 10.  Text-Generation (pure logits — no network / tokenizer)
# ---------------------------------------------------------------------------


def test_text_generation_processors_individually():
    """Text-Generation wiki — each processor transforms logits correctly."""
    from nnx.generation.logits_processors import (
        RepetitionPenalty,
        TemperatureScaling,
        TopKFilter,
        TopPFilter,
    )

    logits = torch.tensor([[1.0, 2.0, 3.0, 0.5, -0.5]])

    # TemperatureScaling — divides by temperature
    scaled = TemperatureScaling(temperature=2.0)(logits, token_history=[])
    assert torch.allclose(scaled, logits / 2.0)

    # TopKFilter — only top-2 finite; rest -inf
    topk = TopKFilter(top_k=2)(logits, token_history=[])
    assert (topk == float("-inf")).sum().item() == 3

    # TopPFilter — small p keeps only top tokens
    topp = TopPFilter(top_p=0.01)(logits, token_history=[])
    assert (topp == float("-inf")).sum().item() >= 1

    # RepetitionPenalty — seen token logit changes
    history = [2]  # token at index 2 has logit 3.0 (positive)
    penalized = RepetitionPenalty(penalty=2.0)(logits.clone(), token_history=history)
    # positive logit for seen token divided by penalty → smaller
    assert penalized[0, 2] < logits[0, 2]


def test_text_generation_apply_chain():
    """Text-Generation wiki — apply_chain runs processors in order."""
    from nnx.generation.logits_processors import (
        RepetitionPenalty,
        TemperatureScaling,
        TopKFilter,
        apply_chain,
    )

    logits = torch.tensor([[1.0, 2.0, 3.0, 0.5, -0.5]])
    out = apply_chain(
        logits,
        token_history=[0],
        processors=[RepetitionPenalty(1.1), TopKFilter(3), TemperatureScaling(0.8)],
    )
    assert out.shape == logits.shape


def test_text_generation_logits_chain_apply():
    """Text-Generation wiki — LogitsChain.apply is semantically equivalent to apply_chain."""
    from nnx.generation.logits_chain import LogitsChain
    from nnx.generation.logits_processors import TemperatureScaling, TopKFilter, apply_chain

    chain = LogitsChain(processors=[TopKFilter(top_k=3), TemperatureScaling(temperature=0.8)])
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.5, -0.5]])
    history = [1, 2]
    via_chain = chain.apply(logits, token_history=history)
    via_direct = apply_chain(logits, token_history=history, processors=chain.processors)
    assert torch.equal(via_chain, via_direct)


# ---------------------------------------------------------------------------
# 11.  Visualization (optional — gate on torchinfo)
# ---------------------------------------------------------------------------


def test_viz_summary_gated():
    """Viz — nnx.viz.summary wraps torchinfo; gated on [viz] extra."""
    torchinfo = pytest.importorskip("torchinfo")  # noqa: F841

    from nnx.viz import summary

    model = _make_model()
    stats = summary(model, input_size=(1, 8))
    assert stats.total_params > 0


def test_viz_weight_histogram_gated():
    """Viz — nnx.viz.weight_histogram runs without error; gated on [viz] extra."""
    pytest.importorskip("torchinfo")

    from nnx.viz import weight_histogram

    model = _make_model()
    fig = weight_histogram(model)
    assert fig is not None


# ---------------------------------------------------------------------------
# 12.  Quantization (optional — gate on torchao)
# ---------------------------------------------------------------------------


def test_quantize_int8_gated(tmp_path, monkeypatch):
    """Quantization wiki — quantize_int8 returns a new NNModel; original untouched."""
    pytest.importorskip("torchao")
    monkeypatch.chdir(tmp_path)

    from nnx import quantize_int8

    model = _make_model()
    model_q = quantize_int8(model)

    assert model_q is not model
    # Original net still FP32
    for p in model.net.parameters():
        assert p.dtype == torch.float32

    # Quantized model can still predict
    import numpy as np

    X = np.random.default_rng(0).standard_normal((4, 8)).astype("float32")
    logits, classes = model_q.predict(X=X)
    assert logits.shape == (4, 3)


def test_quantize_qat_gated(tmp_path, monkeypatch):
    """Quantization wiki — QAT lifecycle: prepare on train_begin, convert on train_end."""
    pytest.importorskip("torchao")
    monkeypatch.chdir(tmp_path)

    from nnx import (
        Activations,
        Devices,
        Losses,
        Nets,
        NNModel,
        NNModelParams,
        NNOptimParams,
        NNParams,
        NNSchedulerParams,
        NNTrainParams,
        Optims,
    )
    from nnx.quantize import QATLifecycleCallback, qat_train_step_factory

    # hidden_dim must be ≥ groupsize (32) for 8da4w to work
    net_params = NNParams(
        input_dim=32,
        output_dim=2,
        hidden_dims=[64],
        dropout_prob=0.0,
        activation=Activations.RELU,
    )
    model = NNModel(
        net_params=net_params,
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )

    torch.manual_seed(0)
    X = torch.randn(32, 32)
    y = torch.randint(0, 2, (32,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16)

    callback = QATLifecycleCallback(qat_config="8da4w", groupsize=32)
    step_fn = qat_train_step_factory(qat_config="8da4w")

    model.train(
        params=NNTrainParams(
            n_epochs=2,
            train_loader=loader,
            optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, weight_decay=0.0, momentum=(0.9, 0.999)),
            scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=2, cooldown=1, threshold=1e-3),
        ),
        callbacks=[callback],
        train_step_fn=step_fn,
    )

    assert callback.is_prepared
    assert callback.is_converted
