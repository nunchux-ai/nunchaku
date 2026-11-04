import pytest
import torch
import torch.nn.functional as F

from nunchaku.ops.misc import layer_norm, layer_norm_, mul_add_, silu_and_mul


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_silu_and_mul(dtype: torch.dtype):
    x = torch.randn(2, 300, 2 * 1536, dtype=dtype, device="cuda")
    ref = F.silu(x[..., :1536].float()) * x[..., 1536:].float()
    assert rel(silu_and_mul(x), ref) < 1e-2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_layer_norm(dtype: torch.dtype):
    x = (torch.randn(2, 300, 3072, device="cuda") * 3 + 1).to(dtype)
    ref = F.layer_norm(x.float(), (3072,), eps=1e-6)
    assert rel(layer_norm(x, 1e-6), ref) < 1e-2
    assert rel(layer_norm_(x.clone(), 1e-6), ref) < 1e-2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mul_add_strided(dtype: torch.dtype):
    """Per-batch scale/bias vectors applied to a slice along the sequence (batch stride larger than the slice)."""
    buf = torch.randn(2, 400, 3072, dtype=dtype, device="cuda")
    mod = torch.randn(2, 1, 3 * 3072, dtype=dtype, device="cuda")
    shift, scale, _ = mod.chunk(3, dim=-1)
    residual = torch.randn(2, 200, 3072, dtype=dtype, device="cuda")
    before = buf.clone()

    x = buf[:, 100:300]
    ref = x.float() * (scale.float() + 1) + shift.float()
    mul_add_(x, scale, shift, 1.0)
    assert rel(x, ref) < 1e-2
    assert torch.equal(buf[:, :100], before[:, :100]) and torch.equal(buf[:, 300:], before[:, 300:])

    ref = x.float() * scale.float() + residual.float()
    mul_add_(x, scale, residual)
    assert rel(x, ref) < 1e-2
