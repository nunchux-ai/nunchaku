"""
NVFP4 numerical-property tests (MATH_TEST_PLAN.md §5).

These tests treat a fully-built `SVDQW4A4Linear` as the black-box GEMM kernel
under test. The layer holds randomly-initialized but valid FP4 weight buffers
(``qweight``, ``wscales``, ``wcscales``, ``proj_down``, ``proj_up``,
``smooth_factor``, ``wtscale``); we never need to decode them, only to keep
them constant across the property check.

Why a black-box test on a random layer is a meaningful test of the FP4 GEMM
math:

* The quantize+GEMM kernel is a *deterministic* function of ``x`` (the BF16
  input) for fixed weight buffers, so the equations from MATH.md §1-2 imply
  algebraic invariants that the layer output must satisfy regardless of what
  the (un-decoded) weight buffers happen to mean.
* For example, scale-equivariance ``y(c*x) ≈ c * y(x)`` and sign symmetry
  ``y(-x) = -y(x)`` follow directly from the math because (a) all per-group
  scales are linear in ``|x|``, and (b) FP4 quantization is sign-symmetric
  by construction (MATH.md §1.1 codebook). A bug that violates these
  invariants is by definition a bug in the quantization or GEMM math.
"""

from __future__ import annotations

import math

import pytest
import torch

from _helpers import requires_fp4  # type: ignore[import-not-found]

pytestmark = requires_fp4


def _make_layer(in_features: int, out_features: int, rank: int = 32,
                 seed: int = 1, device: str = "cuda"):
    """
    Build an SVDQW4A4Linear with random-but-valid FP4 weight buffers.

    No buffer needs to be interpretable to us; we only need it to stay
    constant across the property check.
    """
    from nunchaku.models.linear import SVDQW4A4Linear

    g = torch.Generator(device=device).manual_seed(seed)
    layer = SVDQW4A4Linear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        rank=rank,
        precision="nvfp4",
        torch_dtype=torch.bfloat16,
        device=device,
    )
    layer.qweight.data = torch.randint(
        -128, 128, (out_features, in_features // 2),
        dtype=torch.int8, device=device, generator=g,
    )
    # FP8 e4m3 weight scales (in [0, 0.5]).
    layer.wscales.data = (
        torch.rand(in_features // 16, out_features, device=device, generator=g)
        * 0.5
    ).to(torch.float8_e4m3fn)
    layer.wcscales.data = torch.ones(
        out_features, dtype=torch.bfloat16, device=device
    )
    layer.smooth_factor.data = torch.ones(
        in_features, dtype=torch.bfloat16, device=device
    )
    layer.proj_down.data = torch.randn(
        in_features, rank, dtype=torch.bfloat16, device=device, generator=g
    ) * 0.05
    layer.proj_up.data = torch.randn(
        out_features, rank, dtype=torch.bfloat16, device=device, generator=g
    ) * 0.05
    layer.wtscale = 1.0
    return layer


# ---------------------------------------------------------------------------
# §5.3 / §5.5: Determinism + Bilinearity / scale equivariance with c = 2^k
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("c", [-2.0, -1.0, 0.5, 1.0, 2.0, 4.0])
def test_scale_equivariance_power_of_two(c):
    """
    For any constant ``c`` whose magnitude is a power of 2 (or 0), the layer
    output must satisfy ``y(c*x) == c * y(x)`` **bit-exactly**.

    Mathematical justification (MATH.md §1.2):
      ``c * x`` has per-group max_abs ``|c| * max_abs(x)`` and therefore
      per-group scale ``|c| * s_g``. After FP8 rounding, when ``|c|`` is a
      power of 2 below the saturation limit, this rounding commutes
      exactly with the multiply (FP8 e4m3 stores exponent + mantissa, and a
      power-of-two factor adjusts only the exponent). Then ``q[i] = x[i] /
      s_g`` is unchanged, so the dequantized inputs scale exactly by ``c``,
      the FP32 GEMM scales linearly, and the FP32 -> BF16 down-cast in the
      epilogue is also exact for power-of-two factors below saturation.

    A failure indicates either (i) a non-deterministic kernel, or (ii) a
    bug that violates the linearity of the per-group scale path.
    """
    layer = _make_layer(in_features=512, out_features=256, rank=32, seed=11)
    x = torch.randn(1, 64, 512, dtype=torch.bfloat16, device="cuda")
    y_ref = layer(x)
    y = layer(x * c)
    diff = (y - c * y_ref).abs()
    assert torch.equal(y, c * y_ref), (
        f"y(c*x) != c * y(x) for c={c}: max diff = {diff.max().item():.6g}, "
        f"reference max = {y_ref.abs().max().item():.6g}"
    )


def test_sign_symmetry():
    """
    MATH_TEST_PLAN.md §5.6: ``y(-x) == -y(x)`` (bit-exact, since negation
    only flips sign bits in BF16/FP4/FP8).
    """
    layer = _make_layer(in_features=512, out_features=256, rank=32, seed=12)
    x = torch.randn(1, 64, 512, dtype=torch.bfloat16, device="cuda")
    y_pos = layer(x)
    y_neg = layer(-x)
    assert torch.equal(y_neg, -y_pos), (
        f"y(-x) != -y(x); max |sum| = {(y_pos + y_neg).abs().max().item():.6g}"
    )


def test_zero_input_zero_output():
    """
    ``y(0) == 0`` follows from MATH.md §1.2 (s_g(0) = 0 -> dequant = 0)
    and from the linearity of the GEMM in the dequantized activations. We
    additionally use a layer whose ``proj_down`` and ``proj_up`` are
    non-zero, so the test also catches LoRA-down/up paths missing the zero
    case.
    """
    layer = _make_layer(in_features=512, out_features=256, rank=32, seed=13)
    x = torch.zeros(1, 64, 512, dtype=torch.bfloat16, device="cuda")
    y = layer(x)
    assert torch.equal(y, torch.zeros_like(y)), (
        f"y(0) != 0; max |y| = {y.abs().max().item():.6g}"
    )


def test_determinism_bit_exact():
    """
    MATH_TEST_PLAN.md §5.5: the kernel must be bit-deterministic across
    repeated calls.
    """
    layer = _make_layer(in_features=512, out_features=256, rank=32, seed=14)
    x = torch.randn(1, 64, 512, dtype=torch.bfloat16, device="cuda")
    y1 = layer(x)
    y2 = layer(x)
    y3 = layer(x)
    assert torch.equal(y1, y2)
    assert torch.equal(y2, y3)


# ---------------------------------------------------------------------------
# §5.1 / §5.2  Linearity in A  (within nonlinear-quantization tolerance)
# ---------------------------------------------------------------------------

def test_linearity_in_A_approximate():
    """
    MATH_TEST_PLAN.md §5.1: ``y(A1 + A2) ≈ y(A1) + y(A2)`` within
    additive tolerance. Quantization is non-linear, so this is only
    approximate; we require the residual to be small relative to the
    typical output magnitude (per the test plan's relaxed 2x tolerance).
    """
    layer = _make_layer(in_features=512, out_features=256, rank=32, seed=15)
    A1 = torch.randn(1, 64, 512, dtype=torch.bfloat16, device="cuda")
    A2 = torch.randn(1, 64, 512, dtype=torch.bfloat16, device="cuda")
    y1 = layer(A1)
    y2 = layer(A2)
    ysum = layer(A1 + A2)

    diff = (ysum - (y1 + y2)).abs()
    # The output magnitudes here can run high because the random weight
    # bytes were arbitrary. We compare against the typical scale of the
    # summed output, not against the per-element ref.
    typical = (y1 + y2).abs().mean().clamp(min=1e-3)
    rel_err = diff.mean() / typical
    assert rel_err.item() < 0.5, (
        f"approximate linearity-in-A violated: rel_err = {rel_err.item():.3f}, "
        f"max abs diff = {diff.max().item():.3f}, "
        f"typical = {typical.item():.3f}"
    )


# ---------------------------------------------------------------------------
# §5.4  Output-element max bound  (probabilistic check)
# ---------------------------------------------------------------------------

def test_output_max_bound_holds():
    """
    MATH_TEST_PLAN.md §5.4: ``|y[m,n]| <= QVALUE_MAX**2 * MSCALE_MAX**2 * K
    * |alpha|`` should always hold by construction (each dequantized
    operand is bounded by ``QVALUE_MAX * MSCALE_MAX``).

    We use the layer's default ``wtscale = 1.0`` and verify the bound holds
    on random input.
    """
    in_f, out_f, M = 512, 256, 64
    layer = _make_layer(in_features=in_f, out_features=out_f, rank=32, seed=16)
    x = torch.randn(1, M, in_f, dtype=torch.bfloat16, device="cuda")
    y = layer(x)

    # MATH.md §1: bound per element = QVALUE_MAX (=6) * MSCALE_MAX (=448)
    # for each operand. Per output element y[m,n] = sum over k=0..K-1 of
    # dequant(A[m,k]) * dequant(W[n,k]). With both operands bounded:
    bound = 6.0 * 448.0 * 6.0 * 448.0 * in_f * 1.0  # = QVALUE_MAX^2 * MSCALE_MAX^2 * K * |alpha|
    max_abs = y.abs().max().item()
    assert max_abs <= bound, (
        f"output bound violated: max|y|={max_abs:.6g} > bound={bound:.6g}"
    )
    # And it should never be NaN/Inf:
    assert torch.isfinite(y).all(), "output contains non-finite values"
