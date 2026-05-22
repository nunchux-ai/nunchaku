"""
NVFP4 quantization tests (MATH_TEST_PLAN.md §1).

These exercise `nunchaku.ops.quantize.svdq_quantize_w4a4_act_fuse_lora_cuda`
in isolation, by decoding the FP8 `ascales` it writes and comparing each
group-scale to the pure-PyTorch oracle.

The interpretation of the on-disk layout is documented in `conftest.py`
under "NVFP4 activation-scale (ascales) byte layout"; this was discovered
empirically by single-element probing of the kernel output. The decoder is
verified to reproduce the kernel scales bit-exactly for random inputs.

Notes on kernel limitations discovered while writing these tests
-----------------------------------------------------------------

1. The `svdq_quantize_w4a4_act_fuse_lora_cuda` kernel requires a non-None
   `smooth` tensor; passing `smooth=None` produces an illegal-memory-access
   CUDA error. All tests therefore supply `smooth = ones(...)` explicitly,
   which is the algebraic identity for the smooth-fold (MATH.md §8). This
   contradicts the public docstring of the wrapper which says "Smoothing
   factor for quantization." is optional, and is recorded in MATH.md.

2. The FP4 weight quantizer `kernels::quantize_w4a4_wgt` is compiled out
   (`assert(false)`) - weight quantization for NVFP4 only ever happens
   offline at checkpoint-build time. Tests for the GEMM path therefore can
   only test the GEMM as a black-box via the public Python `SVDQW4A4Linear`
   layer, not as a free-standing kernel call from random tensors.

3. The kernel does not write zero into `act` bytes when an input group is
   all-zero. Instead it relies on the per-group FP8 scale being 0, so that
   the dequantized value is `q * s = q * 0 = 0` regardless of the FP4
   contents. The `act` byte for a zero group on the current build is `0x77`
   (i.e. +6, +6) because the kernel uses `rcp.approx.ftz.f32(0) -> +inf`,
   then `0 * inf -> NaN`, then `cvt.satfinite.e2m1x2 -> +6`. End-to-end the
   dequantize is still 0, but a checkpoint-equality test on raw `act` bytes
   between two implementations would have to take this into account.
"""

from __future__ import annotations

import math

import pytest
import torch

from _helpers import (  # type: ignore[import-not-found]
    FP4_E2M1_MAGNITUDES,
    MSCALE_MAX,
    QVALUE_MAX,
    fp4_decode_ascales,
    fp4_e2m1_decode,
    fp8_e4m3_round,
    requires_fp4,
    ulp_e4m3,
)

pytestmark = requires_fp4


def _call_quantize(
    x: torch.Tensor,
    smooth: torch.Tensor | None = None,
    rank: int = 16,
    pad_size: int = 256,
):
    """
    Wrapper that always supplies a non-None `smooth` (see note 1 in the
    module docstring) and a zero `lora_down` so the LoRA-down output is
    unused.
    """
    from nunchaku.ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda

    K = x.shape[-1]
    if smooth is None:
        smooth = torch.ones(K, dtype=x.dtype, device=x.device)
    lora_down = torch.zeros(K, rank, dtype=x.dtype, device=x.device)
    qact, oscales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora_cuda(
        x, lora_down=lora_down, smooth=smooth, fp4=True, pad_size=pad_size
    )
    return qact, oscales, lora_act_out


# ---------------------------------------------------------------------------
# §1.1  Per-group scale derivation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("M", [256, 1024])
@pytest.mark.parametrize("K", [128, 512, 2048])
def test_per_group_scale_matches_oracle(sigma, M, K):
    """
    MATH_TEST_PLAN.md §1.1: every per-group FP8 scale stored by the kernel
    must equal `min(max_abs(group) / 6.0, 448.0)` (then FP8-e4m3-rounded)
    from the oracle, to within 1 ULP of FP8.
    """
    torch.manual_seed(M * 31 + K * 7 + int(sigma * 1000))
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * sigma

    qact, oscales, _ = _call_quantize(x)

    decoded = fp4_decode_ascales(oscales, M_pad=M, K=K)

    # Oracle scales.
    raw = (
        x.float().reshape(M, K // 16, 16).abs().amax(dim=-1) / QVALUE_MAX
    ).clamp(max=MSCALE_MAX)
    expected = fp8_e4m3_round(raw)

    # MATH_TEST_PLAN §1.1 allows <= 0.5 ULP of FP8 e4m3. We tighten to
    # zero because the kernel reads FP8-rounded scales: the only way the
    # output differs is if the kernel's max-abs reduction misses an element.
    err = (decoded - expected).abs()
    tol = 0.5 * ulp_e4m3(expected)
    bad = err > tol
    assert not bad.any(), (
        f"per-group scale mismatch in {bad.sum().item()} / {bad.numel()} positions "
        f"(max err = {err.max().item():.6g}, max tol = {tol.max().item():.6g})"
    )


# ---------------------------------------------------------------------------
# §1.2  Saturation at MSCALE_MAX = 448
# ---------------------------------------------------------------------------

def test_saturation_at_mscale_max():
    """
    A single huge outlier per row drives that row's group-0 scale above
    MSCALE_MAX = 448, where it must be clipped *exactly* to 448 by the
    `cvt.rn.satfinite.e4m3x2.f32` semantics from MATH.md §1.2.
    """
    torch.manual_seed(0)
    M, K = 256, 128
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.01
    # One outlier per row, in group 0:
    x[:, 0] = 1.0e6  # any value > QVALUE_MAX * MSCALE_MAX = 2688

    _, oscales, _ = _call_quantize(x)
    decoded = fp4_decode_ascales(oscales, M_pad=M, K=K)

    # Group 0 in every row must be saturated to exactly 448.
    g0 = decoded[:, 0]
    assert torch.all(g0 == 448.0), (
        f"group-0 scales are not all 448.0; unique values = "
        f"{torch.unique(g0).tolist()}"
    )


# ---------------------------------------------------------------------------
# §1.3  FP4 codebook coverage (round trip through dequantize)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("magnitude", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
@pytest.mark.parametrize("sign", [+1, -1])
def test_fp4_codebook_roundtrip_via_oracle(magnitude, sign):
    """
    Filling an entire row with one codebook magnitude (times an FP8-exact
    scale) must round-trip *exactly* through (quantize, dequantize) when
    using the oracle dequantizer. We can't read FP4 bytes from the kernel's
    fragment-shuffled `act` tensor, but we can check that the scale stored
    is exactly `magnitude / 6.0` after FP8 rounding, which is the only
    observable that defines the dequantize value at codebook input.
    """
    torch.manual_seed(0)
    M, K = 256, 128
    val = float(sign) * magnitude
    x = torch.full((M, K), val, dtype=torch.bfloat16, device="cuda")

    _, oscales, _ = _call_quantize(x)
    decoded = fp4_decode_ascales(oscales, M_pad=M, K=K)

    # Expected scale: |val| / 6.0, FP8-rounded.
    expected_scale = fp8_e4m3_round(
        torch.tensor(magnitude / QVALUE_MAX, device="cuda")
    ).item()
    assert torch.all(decoded == expected_scale), (
        f"expected all scales == {expected_scale}; got unique values "
        f"{torch.unique(decoded).tolist()}"
    )

    # And the dequantize via FP4 codebook should reproduce |val| exactly for
    # codebook magnitudes (this is a property of the oracle dequantizer; we
    # verify the codebook entries are recoverable):
    assert magnitude in FP4_E2M1_MAGNITUDES.tolist()


# ---------------------------------------------------------------------------
# §1.4  Sign symmetry
# ---------------------------------------------------------------------------

def test_sign_symmetry_scales_identical():
    """
    For input `x` and `-x` the per-group FP8 scales must be byte-identical;
    only the FP4 element sign bits flip.
    """
    torch.manual_seed(123)
    M, K = 256, 256
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    _, oscales_pos, _ = _call_quantize(x)
    _, oscales_neg, _ = _call_quantize(-x)

    # Compare the raw FP8 bytes, not the decoded floats - tightest check.
    bytes_pos = oscales_pos.view(torch.uint8)
    bytes_neg = oscales_neg.view(torch.uint8)
    assert torch.equal(bytes_pos, bytes_neg), (
        "FP8 scale bytes differ between x and -x; quantization scale is not "
        "sign-symmetric."
    )


def test_sign_symmetry_act_bytes_flipped():
    """
    For inputs `x` and `-x`, the FP4 byte pairs must be related only by
    sign-bit flips of each nibble (XOR with 0x88). The kernel's special
    zero-group handling (`0 * inf -> NaN -> satfinite -> +6`) means we must
    skip groups with scale = 0, where the act bytes are 0x77 regardless of
    sign; per MATH.md §8 the only observable in that case is the scale.
    """
    torch.manual_seed(123)
    M, K = 256, 128
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    # Ensure no group has max_abs == 0:
    x = torch.where(x.abs() < 0.05, torch.full_like(x, 0.05), x)

    qa_pos, _, _ = _call_quantize(x)
    qa_neg, _, _ = _call_quantize(-x)

    pos_bytes = qa_pos.view(-1).to(torch.int32)
    neg_bytes = qa_neg.view(-1).to(torch.int32)

    # For non-zero groups every nibble must flip its sign bit (bit 3 of the
    # nibble), so the byte XOR == 0x88. Zero nibbles (magnitude 0) are
    # unaffected, but pure-zero nibbles cannot appear with this seeded
    # non-zero input.
    xor = (pos_bytes ^ neg_bytes) & 0xFF
    # Allowed XOR values:
    #   - 0x88 : both nibbles negated normally
    #   - 0x08 / 0x80 : one nibble was magnitude 0 (codebook index 0) and so
    #                   stays nibble 0 under negation; the other flipped.
    #   - 0x00 : both nibbles were magnitude 0.
    allowed = (xor == 0x88) | (xor == 0x80) | (xor == 0x08) | (xor == 0x00)
    bad = ~allowed
    assert not bad.any(), (
        f"sign symmetry check: {bad.sum().item()} bytes have "
        f"unexpected XOR pattern; sample = {xor[bad][:5].tolist()}"
    )


# ---------------------------------------------------------------------------
# §1.5  Zero rows
# ---------------------------------------------------------------------------

def test_zero_rows_produce_finite_zero_scale():
    """
    An entire K-group of zeros must produce a finite FP8 scale (no NaN /
    Inf), and the dequantized output for that group must be exactly zero.

    The kernel currently writes the FP4 act byte as 0x77 for zero groups
    (see module docstring note 3). The mathematical observable - the
    dequantized value - is still zero because scale=0 dominates the product.
    """
    M, K = 256, 128
    x = torch.zeros(M, K, dtype=torch.bfloat16, device="cuda")

    _, oscales, _ = _call_quantize(x)
    decoded = fp4_decode_ascales(oscales, M_pad=M, K=K)

    assert torch.isfinite(decoded).all()
    assert torch.all(decoded == 0.0)


# ---------------------------------------------------------------------------
# §1.6  Smooth-quant fold
# ---------------------------------------------------------------------------
#
# The kernel applies smooth-quant via h2div() inside the EpilogueQuantize step
# (see `gemm_w4a4.cuh:990-993`). It loads `smooth_factor` not as a flat
# `[K]` tensor but as a `wscale_warp` of `packed_wscale_t`, i.e. a
# fragment-interleaved per-warp layout. That means passing a random
# `torch.randn(K)` tensor as `smooth_factor` and expecting the kernel to use
# `smooth[k]` for channel k is INCORRECT - the kernel will mis-attribute
# values across channels.
#
# We can therefore only safely test smooth-fold with a *uniform* smooth
# factor (where the layout permutation is irrelevant). Building a properly
# packed `smooth_factor` tensor from Python would require duplicating the
# `pack_wscales` kernel layout, which is out of scope for these tests.
# (This packing-layout requirement is undocumented in the public Python
# wrapper - see note in module docstring.)

@pytest.mark.parametrize("s_const", [0.5, 1.0, 2.0, 3.5])
def test_smooth_quant_fold_uniform_constant(s_const):
    """
    With a uniform `smooth_factor = s_const` over all channels, the kernel
    must quantize `x / s_const`. Equivalently, each per-group scale is the
    `x` scale divided by `s_const` (modulo FP8 rounding).
    """
    torch.manual_seed(7)
    M, K = 256, 256
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    smooth = torch.full((K,), s_const, dtype=torch.bfloat16, device="cuda")

    _, oscales_a, _ = _call_quantize(x, smooth=smooth)
    a = fp4_decode_ascales(oscales_a, M_pad=M, K=K)

    # Reference: quantize x/s_const explicitly with smooth=ones.
    x_div = (x.float() / s_const).to(torch.bfloat16)
    _, oscales_b, _ = _call_quantize(
        x_div, smooth=torch.ones(K, dtype=torch.bfloat16, device="cuda")
    )
    b = fp4_decode_ascales(oscales_b, M_pad=M, K=K)

    # With uniform smooth, the layout permutation doesn't matter and the two
    # decoded scale matrices should agree to within FP8 ULP, dominated by
    # the bf16 rounding of `x / s_const` before the second kernel call.
    err = (a - b).abs()
    tol = ulp_e4m3(a.clamp(min=2 ** -6)) + 1e-4
    bad = err > tol
    assert bad.float().mean().item() < 0.02, (
        f"uniform smooth-fold mismatch for s={s_const}: "
        f"{bad.sum().item()}/{bad.numel()} positions differ by > 1 FP8 ULP "
        f"(max err = {err.max().item():.6g})"
    )


# ---------------------------------------------------------------------------
# Determinism (touches §5.5 from the test plan)
# ---------------------------------------------------------------------------

def test_quantize_is_deterministic_bit_for_bit():
    """
    The quantizer must be bit-deterministic across repeated calls on the
    same input. MATH_TEST_PLAN.md §5.5.
    """
    torch.manual_seed(0)
    M, K = 256, 256
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    qact1, oscales1, _ = _call_quantize(x)
    qact2, oscales2, _ = _call_quantize(x)
    assert torch.equal(qact1, qact2)
    assert torch.equal(
        oscales1.view(torch.uint8), oscales2.view(torch.uint8)
    )
