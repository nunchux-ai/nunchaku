"""
Triton/PyTorch-based NVFP4 quantization backend.
"""

import torch
from ...utils import ceil_divide

FP4_E2M1_MAGNITUDES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)
QVALUE_MAX = 6.0
MSCALE_MAX = 448.0

import triton
import triton.language as tl

@triton.jit
def fp4_encode_value_triton(x):
    sign = (x < 0.0).to(tl.int32)
    a = tl.abs(x)
    
    idx = tl.where(a <= 0.25, 0,
          tl.where(a < 0.75, 1,
          tl.where(a <= 1.25, 2,
          tl.where(a < 1.75, 3,
          tl.where(a <= 2.5, 4,
          tl.where(a < 3.5, 5,
          tl.where(a <= 5.0, 6, 7)))))))
          
    nibble = (sign << 3) | idx
    return nibble.to(tl.uint8)

@triton.jit
def quantize_fp4_pack_kernel(
    x_ptr, output_ptr, scales_ptr,
    stride_x_m, stride_x_k,
    stride_out_m, stride_out_k,
    stride_scale_m, stride_scale_k,
    M, K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        x_ptrs = x_ptr + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
        
        # Group-size = 16
        x_reshaped = tl.reshape(x, (BLOCK_M, BLOCK_K // 16, 16))
        x_abs = tl.abs(x_reshaped)
        amax = tl.max(x_abs, axis=2)
        
        sA = amax / 6.0
        sA = tl.minimum(sA, 448.0)
        sA_f8 = sA.to(tl.float8e4nv)
        sA_scaled = sA_f8.to(tl.float32)
        
        sA_safe = tl.where(sA_scaled == 0.0, 1.0, sA_scaled)
        x_scaled = x_reshaped / sA_safe[:, :, None]
        
        nibbles = fp4_encode_value_triton(x_scaled)
        sA_is_zero = sA_scaled == 0.0
        nibbles = tl.where(sA_is_zero[:, :, None], 7, nibbles)
        
        nibbles_2d = tl.reshape(nibbles, (BLOCK_M, BLOCK_K))
        nibbles_split = tl.reshape(nibbles_2d, (BLOCK_M, BLOCK_K // 2, 2)).to(tl.int32)
        
        mask_lo = (tl.arange(0, 2) == 0)[None, None, :]
        mask_hi = (tl.arange(0, 2) == 1)[None, None, :]
        
        lo = tl.sum(nibbles_split * mask_lo, axis=2)
        hi = tl.sum(nibbles_split * mask_hi, axis=2)
        packed = (lo | (hi << 4)).to(tl.uint8)
        
        out_ptrs = output_ptr + offs_m[:, None] * stride_out_m + (k_start // 2 + tl.arange(0, BLOCK_K // 2))[None, :] * stride_out_k
        tl.store(out_ptrs, packed, mask=mask_m[:, None])
        
        scale_ptrs = scales_ptr + offs_m[:, None] * stride_scale_m + (k_start // 16 + tl.arange(0, BLOCK_K // 16))[None, :] * stride_scale_k
        tl.store(scale_ptrs, sA_f8, mask=mask_m[:, None])


def fp8_e4m3_round(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    x_sat = x.clamp(min=-MSCALE_MAX, max=MSCALE_MAX)
    return x_sat.to(torch.float8_e4m3fn).to(torch.float32)

def fp4_e2m1_encode_value(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    sign = (x < 0).to(torch.int32)
    a = x.abs().clamp(max=QVALUE_MAX)
    mags = FP4_E2M1_MAGNITUDES.to(x.device)
    diffs = (a[..., None] - mags).abs()
    even_bonus = torch.tensor(
        [0.0, 1e-9, 0.0, 1e-9, 0.0, 1e-9, 0.0, 1e-9],
        dtype=torch.float32, device=x.device,
    )
    scored = diffs + even_bonus
    idx = scored.argmin(dim=-1).to(torch.int32)
    nibble = (sign << 3) | idx
    return nibble.to(torch.uint8)

def fp4_encode_ascales_warp_interleaved(sA: torch.Tensor, M_pad: int, K: int) -> torch.Tensor:
    device = sA.device
    G_len = K // 16
    
    rs = torch.arange(M_pad, device=device)
    gs = torch.arange(G_len, device=device)
    R, G = torch.meshgrid(rs, gs, indexing="ij")  # (M_pad, G_len)
    
    bm = R // 256
    r_in_block = R % 256
    warp_id = r_in_block // 32
    r_in_warp = r_in_block % 32
    lane_id = (r_in_warp % 8) * 4 + (r_in_warp // 8)
    bk = G // 4
    pack_lane = G % 4
    
    per_bm_chunk = G_len * 256
    flat = (
        bm * per_bm_chunk
        + (bk * 8 + warp_id) * 128
        + lane_id * 4
        + pack_lane
    )
    
    oscales_flat = torch.empty(G_len * M_pad, dtype=torch.float8_e4m3fn, device=device)
    oscales_flat[flat.long()] = sA.to(torch.float8_e4m3fn)
    return oscales_flat.view(G_len, M_pad)

def quantize_w4a4_act_fuse_lora_triton(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    lora_act_out: torch.Tensor | None = None,
    smooth: torch.Tensor | None = None,
    fuse_glu: bool = False,
    fp4: bool = False,
    pad_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Triton/PyTorch fallback emulator/compiled path for W4A4 quantization.
    """
    assert fp4, "Triton/PyTorch backend currently only supports fp4=True"
    
    batch_size, channels = input.shape
    batch_size_pad = ceil_divide(batch_size, pad_size) * pad_size
    rank = lora_down.shape[1] if lora_down is not None else 0
    
    if output is None:
        output = torch.empty(batch_size_pad, channels // 2, dtype=torch.uint8, device=input.device)
    if oscales is None:
        oscales = torch.empty(channels // 16, batch_size_pad, dtype=torch.float8_e4m3fn, device=input.device)
    if lora_act_out is None and rank > 0:
        lora_act_out = torch.empty(batch_size_pad, rank, dtype=torch.float32, device=input.device)

    # 1. Apply GLU if fuse_glu is True
    if fuse_glu:
        assert channels % 2 == 0
        x_glu = input[..., 0::2] * torch.nn.functional.silu(input[..., 1::2])
    else:
        x_glu = input
        
    K_pad_active = x_glu.shape[1]
    
    # 2. Pad x_glu to batch_size_pad
    x_pad = torch.zeros(batch_size_pad, K_pad_active, dtype=input.dtype, device=input.device)
    x_pad[:batch_size, :] = x_glu
    
    # 3. Compute LoRA down-projection on the unsmoothed/GLU'ed input.
    # NOTE: `lora_down` is stored on disk (shape (K, R)) in deepcompressor's
    # permuted `pack_lowrank_weight(..., down=True)` layout. Unpacking yields
    # the logical PyTorch-standard tensor of shape (R, K) - same convention as
    # ``nn.Linear(K, R).weight``. We then compute ``x @ proj_down_logical.T``.
    if lora_down is not None and lora_act_out is not None:
        from .unpack import unpack_proj_down
        lora_down_logical = unpack_proj_down(lora_down)  # (R, K)
        lora_act_out.zero_()
        # x_pad (M, K) @ proj_down_logical.T (K, R) -> (M, R)
        if lora_down_logical.shape == lora_down.shape:
            # Fallback path (no unpack done); use the old direct matmul to
            # preserve previous behavior on non-128-aligned shapes.
            lora_act_out.copy_(torch.matmul(x_pad.float(), lora_down_logical.float()))
        else:
            lora_act_out.copy_(torch.matmul(x_pad.float(), lora_down_logical.float().t()))
        
    # 4. Apply smooth factor if provided.
    # NOTE: smooth_factor on disk is in deepcompressor's fragment-interleaved
    # pack_scale layout (see comment in pack_scale at
    # packages/deepcompressor/deepcompressor/backend/nunchaku/utils.py:62).
    # To use as a flat channel-ordered divisor, we unpack via the inverse
    # permutation. Falls back to flat read for sub-WARP_N tensors.
    if smooth is not None:
        from .unpack import fragment_to_channel_order, _smooth_cache_lookup
        smooth_ordered = _smooth_cache_lookup(smooth)
        s_factor = torch.ones(K_pad_active, dtype=input.dtype, device=input.device)
        s_len = min(K_pad_active, smooth_ordered.shape[0])
        s_factor[:s_len] = smooth_ordered[:s_len]
        x_pad = x_pad / s_factor
        
    # 5. High-performance Triton quantization kernel
    sA_row_major = torch.empty(batch_size_pad, K_pad_active // 16, dtype=torch.float8_e4m3fn, device=input.device)
    
    BLOCK_M = 64
    BLOCK_K = min(256, K_pad_active)
    grid = (ceil_divide(batch_size_pad, BLOCK_M),)
    
    quantize_fp4_pack_kernel[grid](
        x_pad, output, sA_row_major,
        x_pad.stride(0), x_pad.stride(1),
        output.stride(0), output.stride(1),
        sA_row_major.stride(0), sA_row_major.stride(1),
        batch_size_pad, K_pad_active,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )
    
    # 6. Warp-interleave activation scales
    interleaved_oscales = fp4_encode_ascales_warp_interleaved(sA_row_major, batch_size_pad, K_pad_active)
    oscales[:, :batch_size_pad] = interleaved_oscales
    
    return output, oscales, lora_act_out
