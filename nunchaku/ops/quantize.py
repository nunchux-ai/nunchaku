"""
This module provides Python wrappers for Nunchaku's high-performance SVDQuant quantization CUDA kernels.
"""

import torch

from .._C import ops
from ..utils import ceil_divide


def svdq_quantize_w4a4_act_fuse_lora_cuda(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    lora_act_out: torch.Tensor | None = None,
    smooth: torch.Tensor | None = None,
    act_unsigned: bool = False,
    fuse_glu: bool = False,
    fp4: bool = False,
    pad_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantizes activations and computes LoRA down-projection using SVDQuant W4A4 CUDA kernel.

    Parameters
    ----------
    input : torch.Tensor, shape (M, K), dtype bfloat16/float16
        Input activations.
    output : torch.Tensor or None, shape (M_pad, K // 2), dtype uint8, optional
        Packed output tensor for quantized activations. Allocated if None.
    oscales : torch.Tensor or None, shape (K // G, M_pad), dtype float8_e4m3fn for NVFP4 or input dtype for INT4, optional
        Output scales tensor. Allocated if None.
    lora_down : torch.Tensor or None, shape (K, R), dtype bfloat16/float16, optional
        Packed LoRA down-projection weights.
    lora_act_out : torch.Tensor or None, shape (M_pad, R), dtype float32, optional
        Packed output tensor for LoRA activations. Allocated if None.
    smooth : torch.Tensor or None, optional, dtype bfloat16/float16
        Smoothing factor for quantization.
    act_unsigned : bool, default=False
        If True, quantize INT4 activations as unsigned. This should match how the
        packed activations will be consumed by the downstream GEMM.
    fuse_glu : bool, default=False
        If True, fuse GLU activation.
    fp4 : bool, default=False
        If True, use NVFP4 quantization; else INT4.
    pad_size : int, default=256
        Pad batch size to a multiple of this value for efficient CUDA execution.

    Returns
    -------
    output : torch.Tensor, shape (M_pad, K // 2), dtype uint8
        Packed quantized activations.
    oscales : torch.Tensor, shape (K // G, M_pad), dtype float8_e4m3fn for NVFP4 or input dtype for INT4
        Output scales.
    lora_act_out : torch.Tensor, shape (M_pad, R), dtype float32
        Packed LoRA activation output.

    Notes
    -----
    Notations:

    - M: batch size
    - K: input channels
    - R: LoRA rank
    - G: group size (64 for INT4, 16 for NVFP4)
    - M_pad: padded batch size = ceil(M / pad_size) * pad_size
    """
    batch_size, channels = input.shape
    rank = lora_down.shape[1]
    batch_size_pad = ceil_divide(batch_size, pad_size) * pad_size
    if output is None:
        output = torch.empty(batch_size_pad, channels // 2, dtype=torch.uint8, device=input.device)
    if oscales is None:
        if fp4:
            assert channels % 16 == 0
            oscales = torch.empty(channels // 16, batch_size_pad, dtype=torch.float8_e4m3fn, device=input.device)
        else:
            assert channels % 64 == 0
            oscales = torch.empty(channels // 64, batch_size_pad, dtype=input.dtype, device=input.device)
    if lora_act_out is None:
        lora_act_out = torch.empty(batch_size_pad, rank, dtype=torch.float32, device=input.device)

    ops.quantize_w4a4_act_fuse_lora(
        input, output, oscales, lora_down, lora_act_out, smooth, act_unsigned, fuse_glu, fp4
    )
    return output, oscales, lora_act_out
