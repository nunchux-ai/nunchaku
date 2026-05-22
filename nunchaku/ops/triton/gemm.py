"""
Triton/PyTorch-based NVFP4 GEMM backend.
"""

import torch
import math
import triton
import triton.language as tl
from .unpack import unpack_w4a4_weights, unpack_w4a4_scales
from .quantize import quantize_w4a4_act_fuse_lora_triton

_BLOCK_M = 256
_NUM_WARPS = 8
_WARP_M = 32
_GROUP_SIZE = 16

def fp4_decode_ascales(oscales: torch.Tensor, M_pad: int, K: int) -> torch.Tensor:
    """
    Decode the raw warp-interleaved `oscales` tensor (shape (K/16, M_pad), dtype
    float8_e4m3fn) into a logical (M_pad, K/16) FP32 tensor of scales.
    """
    assert oscales.dtype == torch.float8_e4m3fn, oscales.dtype
    assert oscales.numel() == (K // _GROUP_SIZE) * M_pad, (
        f"oscales has {oscales.numel()} elements, expected {(K//_GROUP_SIZE)*M_pad}"
    )
    assert M_pad % _BLOCK_M == 0
    assert K % (_GROUP_SIZE * 4) == 0  # need full WARP_K worth of groups

    rs = torch.arange(M_pad, device=oscales.device)
    gs = torch.arange(K // _GROUP_SIZE, device=oscales.device)
    R, G = torch.meshgrid(rs, gs, indexing="ij")  # (M_pad, K/16)

    bm = R // _BLOCK_M
    r_in_block = R % _BLOCK_M
    warp_id = r_in_block // _WARP_M
    r_in_warp = r_in_block % _WARP_M
    lane_id = (r_in_warp % 8) * 4 + (r_in_warp // 8)
    bk = G // 4
    pack_lane = G % 4

    per_bm_chunk = (K // _GROUP_SIZE) * _BLOCK_M
    flat = (
        bm * per_bm_chunk
        + (bk * _NUM_WARPS + warp_id) * 128
        + lane_id * 4
        + pack_lane
    )
    flat_view = oscales.contiguous().view(-1)
    return flat_view[flat.long()].to(torch.float32)

def unpack_rotemb(packed: torch.Tensor) -> torch.Tensor:
    B, M, D = packed.shape
    x = packed.view(B, M // 16, D // 8, 8, 4, 2, 2)
    x = x.permute(0, 1, 2, 5, 3, 4, 6)  # -> (B, M // 16, D // 8, 2, 8, 4, 2)
    x = x.reshape(B, M // 16, D // 8, 16, 8)
    x = x.permute(0, 1, 3, 2, 4)  # -> (B, M // 16, 16, D // 8, 8)
    x = x.reshape(B, M, D // 2, 1, 2)
    return x

def apply_rmsnorm_rope_epilogue(
    temp_out: torch.Tensor,
    norm_q: torch.Tensor | None,
    norm_k: torch.Tensor | None,
    rotary_emb: torch.Tensor | None,
) -> torch.Tensor:
    if norm_q is None and norm_k is None and rotary_emb is None:
        return temp_out

    M_pad, N = temp_out.shape
    assert N % 3 == 0, f"Expected N to be divisible by 3 for QKV, got N={N}"
    N_chunk = N // 3
    
    res = temp_out.clone()
    q = res[:, :N_chunk]
    k = res[:, N_chunk : 2 * N_chunk]
    
    if rotary_emb is not None:
        unpacked = unpack_rotemb(rotary_emb)
        B_rot, M_rot, D_half, _, _ = unpacked.shape
        unpacked_flat = unpacked.view(B_rot * M_rot, D_half, 2)
        
        if B_rot * M_rot == M_pad:
            sincos = unpacked_flat
        else:
            repeat_factor = math.ceil(M_pad / (B_rot * M_rot))
            sincos = unpacked_flat.repeat(repeat_factor, 1, 1)[:M_pad]
    else:
        sincos = None

    HEAD_DIM = 128
    num_heads = N_chunk // HEAD_DIM

    # Process Query (Q)
    if norm_q is not None:
        q_reshaped = q.view(M_pad, num_heads, HEAD_DIM)
        rms_q = torch.sqrt(torch.mean(q_reshaped.float() ** 2, dim=-1, keepdim=True) + 1e-6)
        if norm_q.numel() == HEAD_DIM:
            norm_q_reshaped = norm_q.view(1, 1, HEAD_DIM)
        else:
            norm_q_reshaped = norm_q.view(1, num_heads, HEAD_DIM)
        q_norm = (q_reshaped.float() / rms_q) * norm_q_reshaped.float()
    else:
        q_norm = q.view(M_pad, num_heads, HEAD_DIM).float()

    if sincos is not None:
        q_norm = q_norm.view(M_pad, num_heads, HEAD_DIM // 2, 2)
        sin = sincos[:, None, :, 0]
        cos = sincos[:, None, :, 1]
        ix = q_norm[..., 0]
        iy = q_norm[..., 1]
        q_rotated = torch.stack([
            ix * cos - iy * sin,
            ix * sin + iy * cos
        ], dim=-1)
        q_final = q_rotated.view(M_pad, N_chunk).to(temp_out.dtype)
    else:
        q_final = q_norm.view(M_pad, N_chunk).to(temp_out.dtype)
    res[:, :N_chunk] = q_final

    # Process Key (K)
    if norm_k is not None:
        k_reshaped = k.view(M_pad, num_heads, HEAD_DIM)
        rms_k = torch.sqrt(torch.mean(k_reshaped.float() ** 2, dim=-1, keepdim=True) + 1e-6)
        if norm_k.numel() == HEAD_DIM:
            norm_k_reshaped = norm_k.view(1, 1, HEAD_DIM)
        else:
            norm_k_reshaped = norm_k.view(1, num_heads, HEAD_DIM)
        k_norm = (k_reshaped.float() / rms_k) * norm_k_reshaped.float()
    else:
        k_norm = k.view(M_pad, num_heads, HEAD_DIM).float()

    if sincos is not None:
        k_norm = k_norm.view(M_pad, num_heads, HEAD_DIM // 2, 2)
        sin = sincos[:, None, :, 0]
        cos = sincos[:, None, :, 1]
        ix = k_norm[..., 0]
        iy = k_norm[..., 1]
        k_rotated = torch.stack([
            ix * cos - iy * sin,
            ix * sin + iy * cos
        ], dim=-1)
        k_final = k_rotated.view(M_pad, N_chunk).to(temp_out.dtype)
    else:
        k_final = k_norm.view(M_pad, N_chunk).to(temp_out.dtype)
    res[:, N_chunk : 2 * N_chunk] = k_final

    return res


@triton.jit
def gemm_fp4_kernel(
    a_ptr, b_ptr, c_ptr,
    scale_a_ptr, scale_b_ptr,
    bias_ptr, wcscales_ptr,
    M, N, K,
    alpha,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_sam, stride_sak,
    stride_sbn, stride_sbk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_WCSCALES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 2
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    offs_k = tl.arange(0, BLOCK_K_PACKED)
    offs_scale_k = tl.arange(0, BLOCK_K // 16)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
    
    scale_a_ptrs = scale_a_ptr + offs_m[:, None] * stride_sam + offs_scale_k[None, :] * stride_sak
    scale_b_ptrs = scale_b_ptr + offs_scale_k[None, :] * stride_sbn + offs_n[:, None] * stride_sbk
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0)
        b = tl.load(b_ptrs, mask=mask_n[:, None], other=0)
        
        scale_a = tl.load(scale_a_ptrs, mask=mask_m[:, None], other=0.0)
        scale_b = tl.load(scale_b_ptrs, mask=mask_n[:, None], other=0.0)
        
        accumulator = tl.dot_scaled(
            a, scale_a, "e2m1",
            b.T, scale_b, "e2m1",
            accumulator,
            lhs_k_pack=True,
            rhs_k_pack=True
        )
        
        a_ptrs += BLOCK_K_PACKED * stride_ak
        b_ptrs += BLOCK_K_PACKED * stride_bk
        scale_a_ptrs += (BLOCK_K // 16) * stride_sak
        scale_b_ptrs += (BLOCK_K // 16) * stride_sbn
        
    if alpha != 1.0:
        accumulator = accumulator * alpha
        
    if HAS_WCSCALES:
        wc_ptrs = wcscales_ptr + offs_n
        wc = tl.load(wc_ptrs, mask=mask_n, other=1.0).to(tl.float32)
        accumulator = accumulator * wc[None, :]
        
    if HAS_BIAS:
        bias_ptrs = bias_ptr + offs_n
        bias = tl.load(bias_ptrs, mask=mask_n, other=0.0).to(tl.float32)
        accumulator = accumulator + bias[None, :]
        
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, accumulator.to(c_ptr.type.element_ty), mask=mask_c)

def gemm_w4a4_fp4_triton(
    act: torch.Tensor,
    wgt: torch.Tensor,
    out: torch.Tensor | None = None,
    qout: torch.Tensor | None = None,
    ascales: torch.Tensor | None = None,
    wscales: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    poolout: torch.Tensor | None = None,
    lora_act_in: torch.Tensor | None = None,
    lora_up: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    lora_act_out: torch.Tensor | None = None,
    norm_q: torch.Tensor | None = None,
    norm_k: torch.Tensor | None = None,
    rotary_emb: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    smooth_factor: torch.Tensor | None = None,
    out_vk: torch.Tensor | None = None,
    out_linearattn: torch.Tensor | None = None,
    act_unsigned: bool = False,
    lora_scales: list[float] | None = None,
    fuse_silu: bool = False,
    fp4: bool = False,
    alpha: float | None = 1.0,
    wcscales: torch.Tensor | None = None,
    out_q: torch.Tensor | None = None,
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
    attn_tokens: int = 0,
    wgt_logical: torch.Tensor | None = None,
    wscales_logical: torch.Tensor | None = None,
) -> None:
    """
    Triton/PyTorch compiled/fallback path for W4A4 NVFP4 GEMM.
    """
    assert fp4, "Triton/PyTorch backend currently only supports fp4=True"
    assert act.dim() == 2, f"Expected 2D act, got shape {act.shape}"
    
    if isinstance(alpha, torch.Tensor):
        alpha = float(alpha.item())
        
    M_pad, K_half = act.shape
    K = K_half * 2
    N = wgt.shape[0]
    
    # Check if we should use the fast Triton JIT GEMM kernel
    use_triton = (
        act.device.type == "cuda"
        and (K % 64 == 0)
        and (M_pad % 64 == 0)
        and (N % 64 == 0)
    )
    
    if use_triton:
        # 1. Unpack/cache packed weights and scales if needed
        if not hasattr(wgt, "_packed_cache") or wgt._packed_cache is None:
            from .unpack import unpack_w4a4_weights_packed
            wgt._packed_cache = unpack_w4a4_weights_packed(wgt, K, N)
            
        if not hasattr(wscales, "_fp8_cache") or wscales._fp8_cache is None:
            from .unpack import unpack_w4a4_scales_fp8
            wscales._fp8_cache = unpack_w4a4_scales_fp8(wscales, K, N)
            
        wgt_packed = wgt._packed_cache
        scale_b = wscales._fp8_cache
        
        # 2. Decode and convert activation scales to float8_e4m3fn
        assert ascales is not None
        sA = fp4_decode_ascales(ascales, M_pad, K)  # shape (M_pad, K // 16)
        scale_a = sA.to(torch.float8_e4m3fn)
        
        # 3. Create a temporary padded output tensor for Triton GEMM
        if out is not None:
            M_actual = out.shape[0]
            out_dtype = out.dtype
            out_device = out.device
        else:
            assert out_q is not None
            B_q, H_q, M_q, D_q = out_q.shape
            M_actual = B_q * M_q
            out_dtype = out_q.dtype
            out_device = out_q.device
            
        temp_out = torch.empty(M_pad, N, dtype=out_dtype, device=out_device)
        
        # 4. Run Triton compiled GEMM JIT kernel
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (triton.cdiv(M_pad, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        
        has_bias = bias is not None
        has_wcscales = wcscales is not None
        
        gemm_fp4_kernel[grid](
            act, wgt_packed, temp_out,
            scale_a, scale_b,
            bias if has_bias else act,
            wcscales if has_wcscales else act,
            M_pad, N, K,
            alpha if alpha is not None else 1.0,
            act.stride(0), act.stride(1),
            wgt_packed.stride(0), wgt_packed.stride(1),
            temp_out.stride(0), temp_out.stride(1),
            scale_a.stride(0), scale_a.stride(1),
            scale_b.stride(0), scale_b.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            HAS_BIAS=has_bias,
            HAS_WCSCALES=has_wcscales,
        )
        
        # 5. Add LoRA up-projection in-place on temp_out.
        # NOTE: `lora_up` is stored on disk in deepcompressor's permuted
        # `pack_lowrank_weight(..., down=False)` layout. We must unpack it
        # before doing the matmul.
        if lora_act_in is not None and lora_up is not None:
            from .unpack import unpack_proj_up
            lora_up_logical = unpack_proj_up(lora_up)
            R = lora_act_in.shape[1]
            if lora_scales is not None:
                scales_tensor = torch.tensor(
                    [lora_scales[r // 16] for r in range(R)],
                    dtype=lora_act_in.dtype,
                    device=lora_act_in.device
                )
                lora_act_scaled = lora_act_in * scales_tensor
            else:
                lora_act_scaled = lora_act_in
            temp_out = temp_out + lora_act_scaled.to(lora_up_logical.dtype) @ lora_up_logical.t()
            
        # 6. Apply SiLU if requested
        if fuse_silu:
            temp_out = torch.nn.functional.silu(temp_out)
            
        # 6.5 Apply RMSNorm & RoPE epilogue if requested
        temp_out = apply_rmsnorm_rope_epilogue(temp_out, norm_q, norm_k, rotary_emb)
        
        # 7. Write back to output
        if out is not None:
            out.copy_(temp_out[:M_actual, :])
        if out_q is not None and out_k is not None and out_v is not None:
            assert N % 3 == 0
            N_chunk = N // 3
            q_final = temp_out[:, :N_chunk]
            k_final = temp_out[:, N_chunk : 2 * N_chunk]
            v_final = temp_out[:, 2 * N_chunk :]
            B_q, H_q, M_q, D_q = out_q.shape
            out_q.copy_(q_final.view(B_q, M_q, H_q, D_q).permute(0, 2, 1, 3))
            B_k, H_k, M_k, D_k = out_k.shape
            out_k.copy_(k_final.view(B_k, M_k, H_k, D_k).permute(0, 2, 1, 3))
            B_v, H_v, M_v, D_v = out_v.shape
            out_v.copy_(v_final.view(B_v, M_v, H_v, D_v).permute(0, 2, 1, 3))
        
        # 8. Next-layer re-quantization
        if qout is not None and oscales is not None:
            quantize_w4a4_act_fuse_lora_triton(
                input=temp_out,
                output=qout,
                oscales=oscales,
                lora_down=lora_down,
                lora_act_out=lora_act_out,
                smooth=smooth_factor,
                fp4=True
            )
            
    else:
        # Fallback PyTorch eager emulator path
        # 1. Unpack activation values
        u = act.to(torch.int32) & 0xFF
        lo = u & 0x0F
        hi = (u >> 4) & 0x0F
        
        codebook = torch.tensor(
            [+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float32, device=act.device
        )
        
        qact_lo = codebook[lo.long()]
        qact_hi = codebook[hi.long()]
        
        qA = torch.stack([qact_lo, qact_hi], dim=-1).reshape(M_pad, K)
        
        # 2. Decode activation scales
        assert ascales is not None
        sA = fp4_decode_ascales(ascales, M_pad, K) # shape (M_pad, K // 16)
        
        qA_reshaped = qA.reshape(M_pad, K // 16, 16)
        A_dq = (qA_reshaped * sA[..., None]).reshape(M_pad, K)
        
        # 3. Unpack weights and scales
        if wgt_logical is None:
            qW = unpack_w4a4_weights(wgt, K, N)
        else:
            qW = wgt_logical
            
        if wscales_logical is None:
            assert wscales is not None
            sW = unpack_w4a4_scales(wscales, K, N)
        else:
            sW = wscales_logical
            
        qW_reshaped = qW.reshape(N, K // 16, 16)
        W_dq = (qW_reshaped * sW.t()[..., None]).reshape(N, K)
        
        # 4. Perform base matmul
        if out is not None:
            M_actual = out.shape[0]
        else:
            assert out_q is not None
            B_q, H_q, M_q, D_q = out_q.shape
            M_actual = B_q * M_q
        
        gemm_out = A_dq @ W_dq.t()
        
        if alpha is not None:
            gemm_out = gemm_out * alpha
            
        if wcscales is not None:
            gemm_out = gemm_out * wcscales.to(gemm_out.dtype)
            
        # 5. Add LoRA (unpack `lora_up` to logical layout first; see comment above)
        if lora_act_in is not None and lora_up is not None:
            from .unpack import unpack_proj_up
            lora_up_logical = unpack_proj_up(lora_up)
            R = lora_act_in.shape[1]
            if lora_scales is not None:
                scales_tensor = torch.tensor(
                    [lora_scales[r // 16] for r in range(R)],
                    dtype=lora_act_in.dtype,
                    device=lora_act_in.device
                )
                lora_act_scaled = lora_act_in * scales_tensor
            else:
                lora_act_scaled = lora_act_in
            lora_out = lora_act_scaled.to(lora_up_logical.dtype) @ lora_up_logical.t()
            gemm_out = gemm_out + lora_out
            
        if bias is not None:
            gemm_out = gemm_out + bias.to(gemm_out.dtype)
            
        if fuse_silu:
            gemm_out = torch.nn.functional.silu(gemm_out)
            
        # 6.5 Apply RMSNorm & RoPE epilogue if requested
        gemm_out = apply_rmsnorm_rope_epilogue(gemm_out, norm_q, norm_k, rotary_emb)
        
        # 7. Write back to output
        if out is not None:
            out.copy_(gemm_out[:M_actual, :])
        if out_q is not None and out_k is not None and out_v is not None:
            assert N % 3 == 0
            N_chunk = N // 3
            q_final = gemm_out[:, :N_chunk]
            k_final = gemm_out[:, N_chunk : 2 * N_chunk]
            v_final = gemm_out[:, 2 * N_chunk :]
            B_q, H_q, M_q, D_q = out_q.shape
            out_q.copy_(q_final.view(B_q, M_q, H_q, D_q).permute(0, 2, 1, 3))
            B_k, H_k, M_k, D_k = out_k.shape
            out_k.copy_(k_final.view(B_k, M_k, H_k, D_k).permute(0, 2, 1, 3))
            B_v, H_v, M_v, D_v = out_v.shape
            out_v.copy_(v_final.view(B_v, M_v, H_v, D_v).permute(0, 2, 1, 3))
        
        if qout is not None and oscales is not None:
            quantize_w4a4_act_fuse_lora_triton(
                input=gemm_out,
                output=qout,
                oscales=oscales,
                lora_down=lora_down,
                lora_act_out=lora_act_out,
                smooth=smooth_factor,
                fp4=True
            )
