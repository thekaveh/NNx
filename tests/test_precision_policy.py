"""FEAT-028: the precision policy, its resolution matrix and run inspection.

``NNModelParams(precision=PrecisionPolicy(...))`` resolves against the
device before any run is reserved: FP32 everywhere, FP16 on CUDA (with a
``GradScaler``), BF16 on CPU and on CUDA devices with bf16 support. An
unsupported request fails unless the policy names ``fallback="fp32"``.
"""

from __future__ import annotations

import os
import pickle

import pytest
import torch

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
    PrecisionPolicy,
    PrecisionUnsupportedError,
    ResolvedPrecision,
    precision_support,
)
from nnx.nn.params.nn_run import NNRun
from nnx.precision import REFERENCE_TOLERANCES, resolve_precision

_NET = NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU)


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.chdir(tmp_path)


def _params(**fields) -> NNModelParams:
    return NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY, **fields)


def _batches(n: int = 4):
    generator = torch.Generator().manual_seed(1)
    return [(torch.randn(4, 4, generator=generator), torch.randint(0, 2, (4,), generator=generator)) for _ in range(n)]


def _train_params(**fields) -> NNTrainParams:
    fields.setdefault("overwrite_existing", True)
    return NNTrainParams(
        n_epochs=fields.pop("n_epochs", 1),
        train_loader=fields.pop("train_loader", _batches()),
        optim=NNOptimParams(name=Optims.SGD, max_lr=0.05, momentum=0.0, weight_decay=0.0),
        scheduler=NNSchedulerParams(min_lr=0.0, factor=0.5, patience=0, cooldown=0, threshold=0.0),
        save_phase_checkpoints=False,
        **fields,
    )


def _cuda(monkeypatch, *, bf16: bool) -> None:
    """Resolution-only stand-in for a CUDA host (no CUDA tensor is made)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: bf16)


# --- the resolution matrix ---------------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_fp32_resolves_everywhere_without_autocast_or_scaler(device):
    resolved = PrecisionPolicy("fp32").resolve(device)
    assert (resolved.requested, resolved.effective, resolved.device_type) == ("fp32", "fp32", device)
    assert resolved.autocast_dtype is None and not resolved.uses_scaler and not resolved.reduced
    assert resolved.build_scaler() is None


def test_fp16_runs_on_cuda_with_a_scaler(monkeypatch):
    _cuda(monkeypatch, bf16=False)
    resolved = PrecisionPolicy("fp16").resolve("cuda")
    assert resolved.effective == "fp16" and resolved.autocast_dtype is torch.float16 and resolved.uses_scaler


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_fp16_off_cuda_fails_unless_a_fallback_is_chosen(device):
    with pytest.raises(PrecisionUnsupportedError, match="fp16 runs .* on CUDA only.*fallback='fp32'"):
        PrecisionPolicy("fp16").resolve(device)
    fallen = PrecisionPolicy("fp16", fallback="fp32").resolve(device)
    assert (fallen.requested, fallen.effective) == ("fp16", "fp32")
    assert fallen.fallback_reason is not None and "CUDA only" in fallen.fallback_reason


def test_bf16_runs_on_cpu_without_a_scaler():
    resolved = PrecisionPolicy("bf16").resolve("cpu")
    assert resolved.effective == "bf16" and resolved.autocast_dtype is torch.bfloat16
    assert not resolved.uses_scaler and resolved.build_scaler() is None


def test_bf16_on_cuda_needs_bf16_support(monkeypatch):
    _cuda(monkeypatch, bf16=True)
    assert PrecisionPolicy("bf16").resolve("cuda").effective == "bf16"
    _cuda(monkeypatch, bf16=False)
    with pytest.raises(PrecisionUnsupportedError, match="is_bf16_supported"):
        PrecisionPolicy("bf16").resolve("cuda")
    assert PrecisionPolicy("bf16", fallback="fp32").resolve("cuda").effective == "fp32"


def test_bf16_on_mps_is_unsupported():
    with pytest.raises(PrecisionUnsupportedError):
        PrecisionPolicy("bf16").resolve("mps")


def test_an_unsupported_request_fails_before_any_run_is_reserved(tmp_path):
    with pytest.raises(PrecisionUnsupportedError):
        NNModel(net_params=_NET, params=_params(precision=PrecisionPolicy("fp16")))
    assert not os.path.exists(tmp_path / "runs")  # the model is never built, so nothing is reserved


def test_policy_fields_are_validated():
    with pytest.raises(ValueError, match="mode must be one of"):
        PrecisionPolicy("fp8")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fallback must be one of"):
        PrecisionPolicy("bf16", fallback="ignore")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a PrecisionPolicy"):
        _params(precision="bf16")  # type: ignore[arg-type]


# --- serialization, the legacy flag and run ids ------------------------------------------------------------


def test_precision_is_omitted_from_state_when_unset_and_round_trips_when_set():
    assert "precision" not in _params().state()
    for policy in (PrecisionPolicy("bf16"), PrecisionPolicy("fp16", fallback="fp32")):
        params = _params(precision=policy)
        assert NNModelParams.from_state(params.state()) == params
    assert _params(precision=PrecisionPolicy("bf16")).state()["precision"] == {
        "mode": "bf16"
    }  # default fallback omitted


def test_the_legacy_flag_keeps_its_meaning_and_its_run_id(monkeypatch):
    legacy = _params(mixed_precision=True)
    assert list(legacy.state().items()) == [
        ("net", "feed_fwd"),
        ("loss", "cross_entropy"),
        ("device", "cpu"),
        ("mixed_precision", True),
    ]
    run = NNRun(net=_NET, train=_train_params(), model=legacy)
    assert run.id == NNRun(net=_NET, train=_train_params(), model=NNModelParams.from_state(legacy.state())).id
    on_cpu = resolve_precision(legacy, "cpu")
    assert (on_cpu.requested, on_cpu.effective, on_cpu.source) == ("fp16", "fp32", "legacy")
    assert on_cpu.fallback_reason is not None and "CUDA only" in on_cpu.fallback_reason
    _cuda(monkeypatch, bf16=False)
    on_cuda = resolve_precision(legacy, "cuda")
    assert on_cuda.effective == "fp16" and on_cuda.uses_scaler


def test_a_contradicting_mode_is_rejected():
    with pytest.raises(ValueError, match="contradicts precision='bf16'"):
        _params(mixed_precision=True, precision=PrecisionPolicy("bf16"))
    _params(mixed_precision=True, precision=PrecisionPolicy("fp16"))  # consistent: allowed


def test_a_pickle_from_before_the_precision_field_still_loads():
    params = _params(mixed_precision=True)
    for kept in (4, 5):  # written before task (FEAT-002) / before precision (FEAT-028)
        old = NNModelParams.__new__(NNModelParams)
        old.__setstate__(params.__getstate__()[:kept])
        assert old == params and old.precision is None and old.task is None
    restored = pickle.loads(pickle.dumps(_params(precision=PrecisionPolicy("bf16"))))
    assert restored.precision == PrecisionPolicy("bf16")


# --- support report and run inspection ---------------------------------------------------------------------


def test_the_support_matrix_claims_nothing_for_absent_hardware():
    matrix = precision_support()
    assert matrix["cpu"] == {"fp32": "verified", "fp16": "unsupported", "bf16": "verified"}
    assert matrix["mps"]["fp16"] == matrix["mps"]["bf16"] == "unsupported"
    if not torch.cuda.is_available():
        assert set(matrix["cuda"].values()) == {"unverified"}  # absent cells report unverified, not support
    else:
        assert matrix["cuda"]["fp32"] == "supported"
    assert set(REFERENCE_TOLERANCES) == {"fp16", "bf16"}
    assert REFERENCE_TOLERANCES["fp16"].loss == 5e-3 and REFERENCE_TOLERANCES["fp16"].weights == 5e-4
    assert REFERENCE_TOLERANCES["bf16"].loss == 5e-2 and REFERENCE_TOLERANCES["bf16"].weights == 5e-3


def test_run_inspection_reports_requested_effective_fallback_and_tf32_separately():
    model = NNModel(net_params=_NET, params=_params(precision=PrecisionPolicy("bf16")))
    run = model.train(_train_params())
    record = run.precision.record() if run.precision is not None else None
    assert record is not None
    assert (record["requested"], record["effective"], record["fallback_reason"]) == ("bf16", "bf16", None)
    assert record["autocast_dtype"] == "bfloat16" and record["grad_scaler"] is False
    assert set(record["tf32"]) == {"cuda_matmul", "cudnn"}  # reported, never set
    assert record["tf32"]["cuda_matmul"] == torch.backends.cuda.matmul.allow_tf32
    assert not any("lr_finder" in item or "diffusion" in item for item in record["covers"])
    assert {"nnx.lr_finder", "nnx.diffusion.sampling"} <= set(record["not_covered"])
    loaded = NNRun.load(run.id)
    assert loaded.precision is not None and loaded.precision.record() == record


def test_a_fallback_is_recorded_on_the_run():
    model = NNModel(net_params=_NET, params=_params(precision=PrecisionPolicy("fp16", fallback="fp32")))
    run = model.train(_train_params())
    assert isinstance(run.precision, ResolvedPrecision)
    assert (run.precision.requested, run.precision.effective) == ("fp16", "fp32")
    assert run.precision.fallback_reason is not None and "CUDA only" in run.precision.fallback_reason
    assert model.resolved_precision.effective == "fp32"
