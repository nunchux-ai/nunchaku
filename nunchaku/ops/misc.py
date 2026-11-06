"""
Python wrappers for Nunchaku's elementwise and normalization CUDA kernels.
"""

import torch

from .._C import ops


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """
    SwiGLU activation: ``silu(x[..., :d]) * x[..., d:]``.

    Parameters
    ----------
    x : torch.Tensor, shape (..., 2 * d), dtype float16, bfloat16 or float32
        Input tensor (contiguous).

    Returns
    -------
    torch.Tensor, shape (..., d)
        Output tensor.
    """
    assert x.is_contiguous() and x.shape[-1] % 2 == 0
    out = x.new_empty(*x.shape[:-1], x.shape[-1] // 2)
    ops.silu_and_mul(out, x)
    return out


def mul_add_(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, scale_shift: float = 0.0) -> torch.Tensor:
    """
    In-place ``x = x * (scale + scale_shift) + bias``, with per-batch broadcasting.

    Parameters
    ----------
    x : torch.Tensor, shape (B, S, D)
        Input tensor, updated in place. Each batch must be contiguous; the batch stride may be larger
        (e.g., a slice along the sequence dimension).
    scale : torch.Tensor, shape (B, 1, D) or (B, S, D)
        Scale, with each batch contiguous.
    bias : torch.Tensor, shape (B, 1, D) or (B, S, D)
        Bias, with each batch contiguous.
    scale_shift : float, optional
        Value added to ``scale``. Default is 0.

    Returns
    -------
    torch.Tensor
        ``x``.
    """
    assert x.ndim == scale.ndim == bias.ndim == 3
    assert scale.dtype == bias.dtype == x.dtype
    assert scale.shape[0] == bias.shape[0] == x.shape[0] and scale.shape[2] == bias.shape[2] == x.shape[2]
    assert scale.shape[1] in (1, x.shape[1]) and bias.shape[1] in (1, x.shape[1])
    assert x[0].is_contiguous() and scale[0].is_contiguous() and bias[0].is_contiguous()
    ops.mul_add_batch(x, scale, True, scale_shift, bias, True)
    return x


def layer_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Layer normalization over the last dimension, without affine parameters.

    Parameters
    ----------
    x : torch.Tensor, shape (..., D)
        Input tensor (contiguous).
    eps : float
        Epsilon for numerical stability.

    Returns
    -------
    torch.Tensor, shape (..., D)
        Normalized tensor.
    """
    assert x.is_contiguous()
    out = torch.empty_like(x)
    ops.layernorm_general(out, x, None, None, eps)
    return out


def layer_norm_(x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    In-place variant of :func:`layer_norm`. The kernel caches each row in shared memory before writing it, so the
    output may overwrite the input.

    Parameters
    ----------
    x : torch.Tensor, shape (..., D)
        Input tensor (contiguous), normalized in place.
    eps : float
        Epsilon for numerical stability.

    Returns
    -------
    torch.Tensor
        ``x``.
    """
    assert x.is_contiguous()
    ops.layernorm_general(x, x, None, None, eps)
    return x
