from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "nunchaku" / "models" / "linear.py"


def load_linear_module(monkeypatch) -> types.ModuleType:
    nunchaku_pkg = types.ModuleType("nunchaku")
    nunchaku_pkg.__path__ = [str(MODULE_PATH.parents[1])]
    models_pkg = types.ModuleType("nunchaku.models")
    models_pkg.__path__ = [str(MODULE_PATH.parent)]
    ops_pkg = types.ModuleType("nunchaku.ops")
    ops_pkg.__path__ = [str(MODULE_PATH.parents[1] / "ops")]

    gemm_module = types.ModuleType("nunchaku.ops.gemm")
    gemm_module.svdq_gemm_w4a4_cuda = lambda *args, **kwargs: None
    gemv_module = types.ModuleType("nunchaku.ops.gemv")
    gemv_module.awq_gemv_w4a16_cuda = lambda *args, **kwargs: None
    quantize_module = types.ModuleType("nunchaku.ops.quantize")
    quantize_module.svdq_quantize_w4a4_act_fuse_lora_cuda = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "nunchaku", nunchaku_pkg)
    monkeypatch.setitem(sys.modules, "nunchaku.models", models_pkg)
    monkeypatch.setitem(sys.modules, "nunchaku.ops", ops_pkg)
    monkeypatch.setitem(sys.modules, "nunchaku.ops.gemm", gemm_module)
    monkeypatch.setitem(sys.modules, "nunchaku.ops.gemv", gemv_module)
    monkeypatch.setitem(sys.modules, "nunchaku.ops.quantize", quantize_module)

    spec = importlib.util.spec_from_file_location("nunchaku.models.linear", MODULE_PATH)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "nunchaku.models.linear", module)
    spec.loader.exec_module(module)
    return module


def test_from_linear_supports_lazy_initialized_linear(monkeypatch):
    module = load_linear_module(monkeypatch)
    linear = types.SimpleNamespace(
        in_features=16,
        out_features=32,
        weight=None,
        bias=None,
        weight_comfy_model_dtype=torch.float16,
    )

    quant_linear = module.SVDQW4A4Linear.from_linear(linear, rank=8, precision="int4")

    assert quant_linear.in_features == 16
    assert quant_linear.out_features == 32
    assert quant_linear.torch_dtype == torch.float16
    assert quant_linear.qweight.device.type == "cpu"
    assert quant_linear.bias is None


def test_from_linear_prefers_materialized_weight_dtype(monkeypatch):
    module = load_linear_module(monkeypatch)
    linear = torch.nn.Linear(8, 12, bias=False, dtype=torch.float16)

    quant_linear = module.SVDQW4A4Linear.from_linear(linear, rank=4, precision="int4")

    assert quant_linear.in_features == 8
    assert quant_linear.out_features == 12
    assert quant_linear.torch_dtype == torch.float16
    assert quant_linear.qweight.device == linear.weight.device
