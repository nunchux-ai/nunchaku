"""
Python wrappers for Nunchaku's elementwise and normalization CUDA kernels.
"""

from typing import Optional

import torch

from .._C import ops


def silu_and_mul(x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    SwiGLU activation: ``silu(x[..., :d]) * x[..., d:]``.

    Parameters
    ----------
    x : torch.Tensor, shape (..., 2 * d), dtype float16, bfloat16 or float32
        Input tensor (contiguous).
    out : torch.Tensor, shape (..., d), optional
        Output tensor. If None, a new tensor is allocated.

    Returns
    -------
    torch.Tensor, shape (..., d)
        Output tensor.
    """
    assert x.is_contiguous()
    if out is None:
        out = x.new_empty(*x.shape[:-1], x.shape[-1] // 2)
    ops.silu_and_mul(out, x)
    return out


def mul_add_(
    x: torch.Tensor, scale: Optional[torch.Tensor], bias: torch.Tensor, scale_shift: float = 0.0
) -> torch.Tensor:
    """
    In-place ``x = x * (scale + scale_shift) + bias``, with per-batch broadcasting.

    Parameters
    ----------
    x : torch.Tensor, shape (B, S, D)
        Input tensor, updated in place. Each batch must be contiguous; the batch stride may be larger
        (e.g., a slice along the sequence dimension).
    scale : torch.Tensor, shape (B, 1, D) or (B, S, D), optional
        Scale. If None, only ``bias`` is added.
    bias : torch.Tensor, shape (B, 1, D) or (B, S, D)
        Bias.
    scale_shift : float, optional
        Value added to ``scale``. Default is 0.

    Returns
    -------
    torch.Tensor
        ``x``.
    """
    assert x.ndim == 3 and x[0].is_contiguous()
    if scale is not None:
        assert scale[0].is_contiguous()
    assert bias[0].is_contiguous()
    ops.mul_add_batch(x, scale, True, scale_shift, bias, True)
    return x


def layer_norm(x: torch.Tensor, eps: float, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Layer normalization over the last dimension, without affine parameters.

    Parameters
    ----------
    x : torch.Tensor, shape (..., D)
        Input tensor (contiguous).
    eps : float
        Epsilon for numerical stability.
    out : torch.Tensor, shape (..., D), optional
        Output tensor (contiguous). If None, a new tensor is allocated.

    Returns
    -------
    torch.Tensor, shape (..., D)
        Normalized tensor.
    """
    assert x.is_contiguous()
    if out is None:
        out = torch.empty_like(x)
    assert out.is_contiguous()
    ops.layernorm_general(out, x, None, None, eps)
    return out
