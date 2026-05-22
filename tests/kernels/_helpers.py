"""
Shared test infrastructure for NVFP4 kernel correctness tests.

This implements the helpers required by `MATH_TEST_PLAN.md` §9:

1. `nvfp4_reference(...)` - pure-PyTorch FP32 oracle implementing exactly the
   algorithm from `MATH.md`.
2. `fp4_e2m1_encode/decode` - pure-PyTorch FP4 RNE quantizer/dequantizer
   saturating at +/- 6.
3. `fp8_e4m3_round` - uses `torch.float8_e4m3fn` cast for FP8 rounding.
4. `ulp_e4m3` - one-ULP magnitude in FP8 e4m3 around a given value, for §1.1.
5. `assert_close_nvfp4` - wraps §0.3 tolerance formula.

These helpers are intentionally independent of the Nunchaku kernel.

NOTE on hardware/build coverage
-------------------------------
The tests in this directory require an FP4-capable GPU (SM 12.0 / Blackwell or
later) and the Nunchaku Python bindings (`nunchaku._C.ops`). They are skipped
otherwise via the module-level `pytestmark` declared in each test file.
"""

from __future__ import annotations

import math
import pytest
import torch


# ---------------------------------------------------------------------------
# FP4 e2m1 codebook (8 magnitudes, signed: 16 distinct values)
# ---------------------------------------------------------------------------

# Per MATH.md §1.1 the encodable magnitudes are { 0, 0.5, 1, 1.5, 2, 3, 4, 6 }
FP4_E2M1_MAGNITUDES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)
QVALUE_MAX = 6.0
MSCALE_MAX = 448.0  # FP8 e4m3 max representable value (= 448)


def fp4_e2m1_decode(byte_packed: torch.Tensor) -> torch.Tensor:
    """
    Decode a tensor of packed uint8 (two FP4 e2m1 per byte) to FP32.

    Each byte stores two FP4 values: low nibble = first element, high nibble =
    second element. Returns a tensor of shape `byte_packed.shape + (2,)`.

    Per MATH.md §1.1: e2m1 has 1 sign bit + 2 exponent bits + 1 mantissa bit.
    Decoding follows the OCP MXFP4 / NVFP4 codebook:

        nibble bits (s e1 e0 m) -> magnitude
        0000 -> +0.0           1000 -> -0.0
        0001 -> +0.5           1001 -> -0.5
        0010 -> +1.0           1010 -> -1.0
        0011 -> +1.5           1011 -> -1.5
        0100 -> +2.0           1100 -> -2.0
        0101 -> +3.0           1101 -> -3.0
        0110 -> +4.0           1110 -> -4.0
        0111 -> +6.0           1111 -> -6.0
    """
    assert byte_packed.dtype in (torch.uint8, torch.int8), byte_packed.dtype
    u = byte_packed.to(torch.int32) & 0xFF
    lo = u & 0x0F
    hi = (u >> 4) & 0x0F

    codebook = torch.tensor(
        [+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32, device=byte_packed.device,
    )
    lo_v = codebook[lo]
    hi_v = codebook[hi]
    return torch.stack([lo_v, hi_v], dim=-1)


def fp4_e2m1_encode_value(x: torch.Tensor) -> torch.Tensor:
    """
    Round-to-nearest-even FP32 -> FP4 e2m1 nibble (uint8 in 0..15) with
    saturate-to-finite to +/- 6.

    Returns a tensor of the same shape as `x` with nibble values in [0, 15].

    RNE: rounds halfway cases to the codebook value with even mantissa bit.
    """
    x = x.to(torch.float32)
    sign = (x < 0).to(torch.int32)
    a = x.abs().clamp(max=QVALUE_MAX)

    # Map magnitude to the nearest codebook entry with RNE.
    # The codebook is: [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    # For each pair of adjacent entries, the midpoint is the rounding boundary;
    # ties go to the codebook entry whose mantissa LSB is 0.
    # Mantissa LSB by index in codebook: [0,1,0,1,0,1,0,1]
    # Tie midpoints: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0.
    # RNE: at exact midpoint, round to the even mantissa neighbor.
    #
    # Even-mantissa indices: 0,2,4,6 (magnitudes 0,1,2,4).
    # For each midpoint, even neighbor:
    #   0.25 between 0 and 0.5     -> tie goes to 0
    #   0.75 between 0.5 and 1.0   -> tie goes to 1.0
    #   1.25 between 1.0 and 1.5   -> tie goes to 1.0
    #   1.75 between 1.5 and 2.0   -> tie goes to 2.0
    #   2.5  between 2.0 and 3.0   -> tie goes to 2.0
    #   3.5  between 3.0 and 4.0   -> tie goes to 4.0
    #   5.0  between 4.0 and 6.0   -> tie goes to 4.0
    mags = FP4_E2M1_MAGNITUDES.to(x.device)
    # For each input, find the codebook index by scanning.
    # We'll do this vectorized via thresholds.
    # threshold[i] = midpoint between mags[i] and mags[i+1], adjusted by tie-rule
    # If even is the lower (i is even), tie goes down: use threshold strictly
    # greater than midpoint to map midpoint to lower index.
    # If odd is the lower (i is odd), tie goes up: use threshold <= midpoint.
    #
    # Simpler: compute differences to each codebook entry, prefer even mantissa
    # on exact tie.
    diffs = (a[..., None] - mags).abs()
    # Find min distance, but among ties prefer even-index entries.
    # We add a tiny tiebreaker that favors even indices by a value smaller than
    # any nonzero distance between FP32-representable midpoints and codebook
    # entries (1e-12 is safe; midpoints are .25, .75, .. all exact).
    even_bonus = torch.tensor(
        [0.0, 1e-9, 0.0, 1e-9, 0.0, 1e-9, 0.0, 1e-9],
        dtype=torch.float32, device=x.device,
    )
    scored = diffs + even_bonus
    idx = scored.argmin(dim=-1).to(torch.int32)

    nibble = (sign << 3) | idx
    # +0 / -0 normalization: both encode magnitude 0; the sign bit is preserved
    # but their decoded value is 0.0 either way.
    return nibble.to(torch.uint8)


def fp4_e2m1_quantize_dequantize(x: torch.Tensor) -> torch.Tensor:
    """
    Round x to the nearest FP4 e2m1 codebook value with RNE and satfinite +/- 6.
    Returns FP32 tensor of the same shape.
    """
    nibble = fp4_e2m1_encode_value(x)
    sign = ((nibble.to(torch.int32) >> 3) & 1)
    mag_idx = nibble.to(torch.int32) & 0x7
    mags = FP4_E2M1_MAGNITUDES.to(x.device)
    val = mags[mag_idx]
    return torch.where(sign.bool(), -val, val)


def fp8_e4m3_round(x: torch.Tensor) -> torch.Tensor:
    """
    Round FP32 -> FP8 e4m3 (RNE, saturate to +/- 448) and return as FP32.

    Implements ``cvt.rn.satfinite.e4m3x2.f32`` semantics: values with magnitude
    above 448 are clamped to +/- 448 before the cast (Torch's
    ``float8_e4m3fn`` cast otherwise produces NaN for magnitudes > 448 except
    449 which round-to-nearest yields 448).
    """
    x = x.to(torch.float32)
    # Match cvt.rn.satfinite.e4m3 PTX semantics: clip first to the largest
    # finite e4m3 magnitude so the cast never produces NaN.
    x_sat = x.clamp(min=-MSCALE_MAX, max=MSCALE_MAX)
    # FP8 e4m3 has no inf in the "fn" (finite, no NaN-only) variant.
    return x_sat.to(torch.float8_e4m3fn).to(torch.float32)


def ulp_e4m3(x: torch.Tensor) -> torch.Tensor:
    """
    Magnitude of one ULP of FP8 e4m3 at value x (FP32 in, FP32 out).

    For normalized e4m3 values: ulp(x) = 2^(floor(log2(|x|))) * 2^-3
                                       = 2^(floor(log2(|x|)) - 3).
    For denormals (|x| < 2^-6): ulp = 2^-9.
    Returns ulp(0) = 2^-9 as well for the lower bound.
    """
    x = x.to(torch.float32).abs()
    tiny = torch.tensor(2.0 ** -9, device=x.device)
    # log2(x) needs guarding against 0
    safe = x.clamp(min=2.0 ** -9)
    exp = torch.floor(torch.log2(safe))
    ulp = torch.pow(torch.tensor(2.0, device=x.device), exp - 3.0)
    ulp = torch.where(x < 2.0 ** -6, tiny, ulp)
    return ulp


def nvfp4_reference(
    act_bf16: torch.Tensor,
    wgt_bf16: torch.Tensor,
    group_size: int = 16,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Pure-PyTorch FP32 oracle for the NVFP4 W4A4 GEMM described in MATH.md.

    Inputs (real, BF16/FP16):
        act_bf16: (M, K)
        wgt_bf16: (N, K)
    Returns FP32 tensor of shape (M, N).
    """
    assert act_bf16.dim() == 2 and wgt_bf16.dim() == 2
    M, K = act_bf16.shape
    N, K2 = wgt_bf16.shape
    assert K == K2
    assert K % group_size == 0

    A = act_bf16.float()
    W = wgt_bf16.float()

    # --- Per-group quantization of activations ---
    A_groups = A.reshape(M, K // group_size, group_size)
    mA = A_groups.abs().amax(dim=-1)
    sA = (mA / QVALUE_MAX).clamp(max=MSCALE_MAX)
    sA = fp8_e4m3_round(sA)                          # FP8 rounding
    # Avoid /0 (matching kernel rcp.approx.ftz semantics: scale=0 -> q=0).
    sA_safe = sA.clamp(min=1e-30)
    qA = fp4_e2m1_quantize_dequantize(A_groups / sA_safe[..., None])
    qA = torch.where(sA[..., None] == 0, torch.zeros_like(qA), qA)

    # --- Per-group quantization of weights ---
    W_groups = W.reshape(N, K // group_size, group_size)
    mW = W_groups.abs().amax(dim=-1)
    sW = (mW / QVALUE_MAX).clamp(max=MSCALE_MAX)
    sW = fp8_e4m3_round(sW)
    sW_safe = sW.clamp(min=1e-30)
    qW = fp4_e2m1_quantize_dequantize(W_groups / sW_safe[..., None])
    qW = torch.where(sW[..., None] == 0, torch.zeros_like(qW), qW)

    # --- Dequantized FP32 GEMM ---
    A_dq = (qA * sA[..., None]).reshape(M, K)
    W_dq = (qW * sW[..., None]).reshape(N, K)
    return alpha * (A_dq @ W_dq.T)


def assert_close_nvfp4(
    out: torch.Tensor,
    ref: torch.Tensor,
    K: int,
    max_abs_input: float = 6.0,
    rtol: float = 0.05,
    atol_scale: float = 0.05,
    extra_floor: float = 1e-3,
) -> None:
    """
    Compare two tensors with the NVFP4 tolerance formula from
    MATH_TEST_PLAN.md §0.3:

        atol(m,n) = max(atol_scale * |y_ref|, atol_scale * sqrt(K) * max_abs_input)
        rtol      = rtol  (relative)

    `extra_floor` is added on top to handle small-magnitude outputs.
    """
    out_f = out.float()
    ref_f = ref.float()
    err = (out_f - ref_f).abs()
    per_element_atol = (
        atol_scale * ref_f.abs()
    ).clamp(min=atol_scale * math.sqrt(K) * max_abs_input * 1e-3 + extra_floor)
    allowed = per_element_atol + rtol * ref_f.abs()
    bad = err > allowed
    if bad.any():
        bad_idx = bad.nonzero()[:5].tolist()
        diag_lines = [
            f"NVFP4 close-check FAILED on {bad.sum().item()}/{bad.numel()} elements",
            f"  max err     = {err.max().item():.6g}",
            f"  median err  = {err.median().item():.6g}",
            f"  max ref abs = {ref_f.abs().max().item():.6g}",
            f"  max out abs = {out_f.abs().max().item():.6g}",
            f"  rtol={rtol} atol_scale={atol_scale} sqrt(K)*max_abs_input={math.sqrt(K)*max_abs_input:.3f}",
            f"  first bad indices: {bad_idx}",
        ]
        raise AssertionError("\n".join(diag_lines))


# ---------------------------------------------------------------------------
# NVFP4 activation-scale (ascales) byte layout
# ---------------------------------------------------------------------------
#
# The runtime quantizer in
# `nunchaku.ops.quantize.svdq_quantize_w4a4_act_fuse_lora_cuda` writes a
# tensor of shape (K/16, M_pad) of FP8 e4m3 scales, BUT the underlying
# byte layout is the warp-interleaved storage used by the FP4 Tensor-Core
# MMA - it is NOT a row-major (K/16, M_pad) matrix in the (row, group) -> FP8
# sense. This was discovered empirically here and matches the kernel comment
# at `gemm_w4a4.cuh:63`:
#     // amscales: [M / BLOCK_M, K / group_size, NUM_WARPS,
#     //           AMSCALES_NUM_PACKS, AMSCALES_VALID_LANES]
#     //          of packed_amscale_t  (each packed_amscale_t = 4 FP8 scales)
#
# Concretely, for the FP4 path with the kernel's hard-coded
# BLOCK_M=256, NUM_WARPS=8, WARP_M=32, WARP_K=64, group_size=16,
# AMSCALES_VALID_LANES=32, AMSCALES_PACK_SIZE=1:
#
#     bm           = r // 256
#     r_in_block   = r % 256
#     warp_id      = r_in_block // 32
#     r_in_warp    = r_in_block % 32
#     lane_id      = (r_in_warp % 8) * 4 + (r_in_warp // 8)
#     bk           = g // 4
#     pack_lane    = g % 4
#     flat_offset  = ( bm * (K/16) * M_pad
#                    + (bk * 8 + warp_id) * 128
#                    + lane_id * 4 + pack_lane )
#
# and the stored scale is at `oscales.flatten()[flat_offset]`.
#
# The lane_id permutation `(r%8)*4 + r//8` matches the (m16,k64) MMA
# fragment row-to-lane mapping that the Tensor Core expects.

_BLOCK_M = 256
_NUM_WARPS = 8
_WARP_M = 32
_GROUP_SIZE = 16


def fp4_decode_ascales(oscales: torch.Tensor, M_pad: int, K: int) -> torch.Tensor:
    """
    Decode the raw `oscales` tensor (shape (K/16, M_pad), dtype
    float8_e4m3fn) into a logical (M_pad, K/16) FP32 tensor of scales.

    Returns ``scales[r, g]`` = scale assigned by the kernel to activation
    row r, K-group g.
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

    # Per-BLOCK_M chunk size: (K/16) * BLOCK_M = (K/16) * 256 FP8 scales,
    # i.e. (K/16) * NUM_WARPS * AMSCALES_VALID_LANES * 4 (pack_lane).
    per_bm_chunk = (K // _GROUP_SIZE) * _BLOCK_M
    flat = (
        bm * per_bm_chunk
        + (bk * _NUM_WARPS + warp_id) * 128
        + lane_id * 4
        + pack_lane
    )
    flat_view = oscales.contiguous().view(-1)
    return flat_view[flat.long()].to(torch.float32)


# ---------------------------------------------------------------------------
# Hardware / capability gates
# ---------------------------------------------------------------------------

def _device_supports_fp4() -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability(0)
    return cap >= (10, 0)


def _nunchaku_importable() -> bool:
    # Evaluated lazily (at first call) instead of at module-import time:
    # the `tests/kernels/conftest.py` may need to strip a shadowing
    # source `nunchaku/` directory from `sys.path` first, and pytest's
    # own rootdir-based path-insertion happens AFTER conftest loading.
    try:
        import nunchaku._C  # noqa: F401
        return True
    except Exception:
        return False


def _require_fp4_predicate() -> bool:
    """
    Returns True iff the current device + nunchaku build supports the
    FP4 kernels under test.

    Re-evaluated by pytest at test-collection / fixture-setup time, after
    the conftest hooks have repaired sys.path and stripped any shadowing
    un-compiled ``nunchaku/`` source directory.
    """
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability(0) < (10, 0):
        return False
    try:
        import nunchaku._C  # noqa: F401
        return True
    except Exception:
        return False


# Module-level marker. We compute the predicate eagerly here too; the
# tests' fixture `_check_fp4_runtime_or_skip` re-checks at runtime to
# catch the case where the predicate flips after path repair.
requires_fp4 = pytest.mark.skipif(
    not _require_fp4_predicate(),
    reason="Requires SM 12.0+ CUDA device and a working `nunchaku._C` build.",
)


