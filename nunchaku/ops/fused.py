"""
High-performance fused operators for quantized neural network inference.
"""

import torch
from diffusers.models.normalization import RMSNorm as DiffUsersRMSNorm
from torch.nn import RMSNorm

from nunchaku.models.linear import SVDQW4A4Linear

from ..utils import ceil_divide
from .gemm import svdq_gemm_w4a4_cuda


def fused_gelu_mlp(x: torch.Tensor, fc1: SVDQW4A4Linear, fc2: SVDQW4A4Linear, pad_size: int = 256) -> torch.Tensor:
    """
    Fused quantized MLP with GELU activation.

    Combines the first quantized linear layer, GELU activation, and the second quantized linear layer into a single CUDA kernel. Supports INT4 and NVFP4 quantization.

    Parameters
    ----------
    x : torch.Tensor, shape (B, S, C_in), dtype float16 or bfloat16
        Input tensor.
    fc1 : SVDQW4A4Linear
        First quantized linear layer (input → hidden).
    fc2 : SVDQW4A4Linear
        Second quantized linear layer (hidden → output).
    pad_size : int, optional
        Batch padding size for CUDA kernel efficiency. Default is 256.

    Returns
    -------
    torch.Tensor, shape (B, S, C_out), dtype as input
        Output tensor.

    Notes
    -----
    - Notations:

      - B: batch size
      - S: sequence length
      - C_in: input features
      - C_out: output features
    - For INT4 quantization, GELU activations are shifted by 0.171875 to ensure non-negativity, enabling unsigned quantization for improved quality. See: https://github.com/nunchux-ai/nunchaku/blob/433f0b228a61a53fb700ac676fd2e290368ac94d/src/kernels/zgemm/gemm_w4a4_launch_impl.cuh#L286
    """
    batch_size, seq_len, channels = x.shape
    x = x.view(batch_size * seq_len, channels)
    quantized_x, ascales, lora_act = fc1.quantize(x)

    batch_size_pad = ceil_divide(batch_size * seq_len, pad_size) * pad_size

    qout_act = torch.empty(batch_size_pad, fc1.out_features // 2, dtype=torch.uint8, device=x.device)
    if fc2.precision == "nvfp4":
        qout_ascales = torch.empty(fc1.out_features // 16, batch_size_pad, dtype=torch.float8_e4m3fn, device=x.device)
    else:
        qout_ascales = torch.empty(fc1.out_features // 64, batch_size_pad, dtype=x.dtype, device=x.device)
    qout_lora_act = torch.empty(batch_size_pad, fc2.proj_down.shape[1], dtype=torch.float32, device=x.device)

    svdq_gemm_w4a4_cuda(
        act=quantized_x,
        wgt=fc1.qweight,
        qout=qout_act,
        ascales=ascales,
        wscales=fc1.wscales,
        oscales=qout_ascales,
        lora_act_in=lora_act,
        lora_up=fc1.proj_up,
        lora_down=fc2.proj_down,
        lora_act_out=qout_lora_act,
        bias=fc1.bias,
        smooth_factor=fc2.smooth_factor,
        fp4=fc1.precision == "nvfp4",
        alpha=fc1.wtscale,
        wcscales=fc1.wcscales,
    )
    output = torch.empty(batch_size * seq_len, fc2.out_features, dtype=x.dtype, device=x.device)
    output = fc2.forward_quant(qout_act, qout_ascales, qout_lora_act, output=output)
    output = output.view(batch_size, seq_len, -1)
    return output


def fused_qkv_norm_rottary(
    x: torch.Tensor,
    proj: SVDQW4A4Linear,
    norm_q: RMSNorm | None = None,
    norm_k: RMSNorm | None = None,
    rotary_emb: torch.Tensor | None = None,
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    attn_tokens: int = 0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused quantized QKV projection with RMSNorm and rotary embeddings.

    Performs quantized QKV projection, applies RMS normalization to Q and K, and fuses rotary embeddings in a single CUDA kernel call.

    Parameters
    ----------
    x : torch.Tensor, shape (B, S, C_in), dtype float16 or bfloat16
        Input tensor.
    proj : SVDQW4A4Linear
        Quantized QKV projection layer.
    norm_q : RMSNorm
        RMSNorm for query.
    norm_k : RMSNorm
        RMSNorm for key.
    rotary_emb : torch.Tensor
        Packed rotary embedding tensor (see :func:`~nunchaku.models.embeddings.pack_rotemb`).
    output : torch.Tensor or tuple of torch.Tensor, optional
        Output tensor(s). If None, a new tensor is allocated.
        If tuple, should be (output_q, output_k, output_v) for fused attention packing.
    attn_tokens : int, optional
        Number of attention tokens. Default is 0.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        Output tensor of shape (B, S, C_out), or tuple (output_q, output_k, output_v).

    Notes
    -----
    Notations:
    - B: batch size
    - S: sequence length
    - C_in: input features
    - C_out: output features
    """
    assert (
        norm_q is None or isinstance(norm_q, RMSNorm) or (isinstance(norm_q, DiffUsersRMSNorm) and norm_q.bias is None)
    )
    assert (
        norm_k is None or isinstance(norm_k, RMSNorm) or (isinstance(norm_k, DiffUsersRMSNorm) and norm_k.bias is None)
    )

    batch_size, seq_len, channels = x.shape
    x = x.view(batch_size * seq_len, channels)
    quantized_x, ascales, lora_act = proj.quantize(x)

    if output is None:
        output = torch.empty(batch_size * seq_len, proj.out_features, dtype=x.dtype, device=x.device)

    if isinstance(output, tuple):
        assert len(output) == 3
        output_q, output_k, output_v = output
        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=proj.qweight,
            ascales=ascales,
            wscales=proj.wscales,
            lora_act_in=lora_act,
            lora_up=proj.proj_up,
            bias=proj.bias,
            fp4=proj.precision == "nvfp4",
            alpha=proj.wtscale,
            wcscales=proj.wcscales,
            norm_q=norm_q.weight if norm_q is not None else None,
            norm_k=norm_k.weight if norm_k is not None else None,
            rotary_emb=rotary_emb,
            out_q=output_q,
            out_k=output_k,
            out_v=output_v,
            attn_tokens=attn_tokens,
        )
        return output_q, output_k, output_v
    else:
        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=proj.qweight,
            out=output,
            ascales=ascales,
            wscales=proj.wscales,
            lora_act_in=lora_act,
            lora_up=proj.proj_up,
            bias=proj.bias,
            fp4=proj.precision == "nvfp4",
            alpha=proj.wtscale,
            wcscales=proj.wcscales,
            norm_q=norm_q.weight if norm_q is not None else None,
            norm_k=norm_k.weight if norm_k is not None else None,
            rotary_emb=rotary_emb,
        )
        output = output.view(batch_size, seq_len, -1)
        return output
