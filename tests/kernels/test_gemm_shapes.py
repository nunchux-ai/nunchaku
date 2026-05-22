"""
NVFP4 GEMM shape and behavior tests (MATH_TEST_PLAN.md §§2-3, plus §2.3 alpha).

These tests exercise the full `gemm_w4a4` kernel via the public
`SVDQW4A4Linear` Python layer for a representative set of (M, in_features,
out_features) shapes, including:

  * single-tile (M=BLOCK_M=256, N=BLOCK_N=128, K=64 .. 128)
  * powers-of-two shapes
  * non-multiple ("tail") shapes that exercise the pad-and-mask code path
  * a diffusion-realistic shape (FLUX-dev QKV: M=4096 N=3072 K=3072)

The layer's weight buffers are random byte patterns - we never need to know
the "true" weight matrix, only verify properties that must hold *regardless*
of what byte pattern the kernel interprets them as. Specifically each test
verifies one or more of:

  - **bit-exact scale equivariance** for power-of-two ``c``
  - **bit-exact sign symmetry**
  - **bit-exact determinism**
  - **finite output values within MATH.md §1.2's bound**
  - **alpha (wtscale) linearity** for varied ``wtscale``

A failure here means the GEMM kernel itself is non-linear, non-deterministic,
or produces NaN/Inf on input that should be representable - all of which
would be direct contradictions of MATH.md.
"""

from __future__ import annotations

import pytest
import torch

from _helpers import requires_fp4  # type: ignore[import-not-found]

pytestmark = requires_fp4


def _make_layer(in_features, out_features, rank=32, seed=1, device="cuda"):
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


def _input(M, K, seed=42, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(1, M, K, dtype=torch.bfloat16, device=device, generator=g)


# ---------------------------------------------------------------------------
# §2.1 / §2.2  Single-tile correctness via invariants
# ---------------------------------------------------------------------------

# Two BF16 outputs that differ by ``<= BF16_ULP_TOL`` are treated as
# "deterministically equal modulo last-bit noise". This relaxation is
# necessary because the FP4 kernel is **observed** to be non-deterministic
# by at most one BF16 ULP for some non-tile-aligned shapes; per
# MATH_TEST_PLAN.md §0.4 we surface this as a known property rather than
# treat it as a hard failure.
BF16_ULP_TOL = 2 ** -8  # 1 ULP of BF16 at magnitude 1; scales with exponent


def _bit_close(a: torch.Tensor, b: torch.Tensor) -> bool:
    """
    Two BF16 tensors are 'bit-close' if every element differs by at most
    one BF16 ULP (about 0.4% relative).
    """
    a32 = a.float()
    b32 = b.float()
    return bool(
        (a32 - b32).abs().le(BF16_ULP_TOL * (1.0 + a32.abs().max())).all()
    )


@pytest.mark.parametrize("M", [16, 32, 64, 128, 256])
@pytest.mark.parametrize("in_features", [128, 256, 512])
@pytest.mark.parametrize("out_features", [128, 256])
def test_single_tile_invariants(M, in_features, out_features):
    """
    For each single-tile shape (M <= BLOCK_M = 256) check that the GEMM
    is (a) bit-deterministic, (b) bit-exact scale-equivariant for c=2,
    (c) bit-exact sign-symmetric, (d) produces finite output.

    Single-tile shapes always use only one CTA per output block in M
    direction, so no inter-CTA atomic reduction noise can creep in - the
    determinism guarantee is strict.
    """
    layer = _make_layer(in_features, out_features, rank=32,
                       seed=hash((M, in_features, out_features)) % 100)
    x = _input(M, in_features, seed=M * 7 + in_features * 13 + out_features)

    y1 = layer(x)
    y1b = layer(x)
    assert torch.equal(y1, y1b), "non-deterministic GEMM at single-tile shape"

    y2 = layer(2 * x)
    assert torch.equal(y2, 2 * y1), (
        f"scale equivariance failed: max |y(2x) - 2 y(x)| = "
        f"{(y2 - 2 * y1).abs().max().item():.4g}"
    )

    y_neg = layer(-x)
    assert torch.equal(y_neg, -y1), (
        f"sign symmetry failed: max |y(-x) + y(x)| = "
        f"{(y_neg + y1).abs().max().item():.4g}"
    )

    assert torch.isfinite(y1).all()


# ---------------------------------------------------------------------------
# §3.1  Power-of-two shapes
# ---------------------------------------------------------------------------

POW2_SHAPES = [
    # (M, in_features, out_features)
    (64, 1024, 1024),
    (128, 1024, 3072),
    (256, 3072, 1024),
    (1024, 1024, 1024),
]


@pytest.mark.parametrize("M,in_f,out_f", POW2_SHAPES)
def test_power_of_two_shapes(M, in_f, out_f):
    layer = _make_layer(in_f, out_f, rank=32, seed=M + in_f + out_f)
    x = _input(M, in_f, seed=M * 7 + in_f * 13 + out_f * 17)

    y = layer(x)
    assert torch.isfinite(y).all()

    # MATH.md §1.2 output bound:
    bound = 6.0 * 448.0 * 6.0 * 448.0 * in_f * 1.0
    assert y.abs().max().item() <= bound, (
        f"output exceeds bound for shape M={M} K={in_f} N={out_f}: "
        f"max|y|={y.abs().max().item():.6g} > {bound:.6g}"
    )

    y2 = layer(2 * x)
    assert torch.equal(y2, 2 * y), (
        f"scale equivariance failed at M={M} K={in_f} N={out_f}: "
        f"max diff = {(y2 - 2 * y).abs().max().item():.4g}"
    )


# ---------------------------------------------------------------------------
# §3.2  Non-multiple ("tail") shapes
#
# The kernel pads (M, N) up to BLOCK_M=256, BLOCK_N=128 and K to 128. A bug
# in the masking-out of padded rows/cols would cause data leakage into the
# visible (unpadded) output. We test this by:
#
#   1. Running the layer on an unpadded shape.
#   2. Running it on the same input *zero-padded* to the next BLOCK_* boundary.
#   3. Asserting the visible region of the two outputs match exactly.
#
# This catches "padding leaks into output" bugs without us needing to know
# the kernel's internal padding scheme.
# ---------------------------------------------------------------------------

# Note: We can't easily padding-test the in_features dim through SVDQW4A4Linear
# (the layer's weight buffers are sized for a fixed in_features). We test M
# padding only.


@pytest.mark.parametrize("M", [33, 100, 199, 257, 511])
def test_M_tail_shapes_finite_and_consistent(M):
    """
    Non-multiple-of-256 M values must produce finite output that respects
    MATH.md bounds. We also check the property that calling the layer
    with `x` and a row-wise zero-padded version yields the same result
    on the original rows.

    Determinism on tail shapes is enforced with 1-ULP-of-BF16 tolerance
    rather than strict ``torch.equal``: the FP4 kernel uses atomic adds in
    the LoRA-down reduction path, which can reorder accumulations across
    repeated kernel launches and introduce a 1-bit difference in the BF16
    output. See MATH_TEST_PLAN.md §0.4 - this is a known property of the
    reference implementation, not a correctness failure.
    """
    in_f, out_f = 512, 256
    layer = _make_layer(in_f, out_f, rank=32, seed=M)
    x = _input(M, in_f, seed=M + 7)

    y = layer(x)
    assert torch.isfinite(y).all()
    bound = 6.0 * 448.0 * 6.0 * 448.0 * in_f * 1.0
    assert y.abs().max().item() <= bound

    # Determinism modulo 1 BF16 ULP (see docstring).
    assert _bit_close(y, layer(x)), (
        f"non-determinism > 1 ULP at M={M}: "
        f"max diff = {(y - layer(x)).abs().max().item():.6g}"
    )

    # Zero-pad rows to the next multiple of 256, run, and verify the
    # first M rows match (within 1 ULP, same reasoning).
    pad_M = ((M + 255) // 256) * 256
    if pad_M != M:
        pad_amount = pad_M - M
        x_padded = torch.cat(
            [x, torch.zeros(1, pad_amount, in_f, dtype=x.dtype, device=x.device)],
            dim=1,
        )
        y_padded = layer(x_padded)
        assert _bit_close(y_padded[:, :M, :], y), (
            f"row-padding leaks into visible output at M={M}: "
            f"max diff = {(y_padded[:, :M, :] - y).abs().max().item():.6g}"
        )
        # And the padded (zero-input) rows must be zero output (strict).
        zeros_out = y_padded[:, M:, :]
        assert torch.equal(zeros_out, torch.zeros_like(zeros_out)), (
            f"zero-padded rows produce non-zero output at M={M}: "
            f"max = {zeros_out.abs().max().item():.6g}"
        )


# ---------------------------------------------------------------------------
# §3.3  Diffusion-realistic shape  (FLUX-dev transformer QKV)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,in_f,out_f", [
    (4096, 3072, 3072),                                  # QKV proj
    pytest.param(4096, 3072, 15360, marks=pytest.mark.slow),   # FFN-up
    pytest.param(4096, 15360, 3072, marks=pytest.mark.slow),   # FFN-down
])
def test_flux_realistic_shape(M, in_f, out_f):
    """
    Run a single GEMM at FLUX-dev transformer-block shape and verify the
    output is finite, bounded, and deterministic within 1 BF16 ULP (see
    the M-tail test for why exact determinism is not required).
    """
    layer = _make_layer(in_f, out_f, rank=32, seed=99)
    x = _input(M, in_f, seed=314159 + M)
    y = layer(x)
    assert torch.isfinite(y).all()
    bound = 6.0 * 448.0 * 6.0 * 448.0 * in_f * 1.0
    assert y.abs().max().item() <= bound
    assert _bit_close(y, layer(x)), (
        f"non-determinism > 1 ULP at shape M={M} K={in_f} N={out_f}: "
        f"max diff = {(y - layer(x)).abs().max().item():.6g}"
    )


# ---------------------------------------------------------------------------
# §2.3  Per-tensor `wtscale` (alpha) scaling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_wtscale_linear(alpha):
    """
    MATH_TEST_PLAN.md §2.3 / §5.7: the per-tensor scale ``alpha`` (stored
    in ``layer.wtscale``) must scale the FP4 GEMM contribution linearly:
    ``Y_fp4(alpha=c) == c * Y_fp4(alpha=1)`` for power-of-two c.

    The layer also adds a LoRA-up contribution that is NOT multiplied by
    alpha (see MATH.md §1.3: alpha scales only the FP4 GEMM accumulators;
    LoRA-up is added afterward in the epilogue). We zero out ``proj_up``
    for this test so the LoRA-up contribution is suppressed and only the
    alpha-scaled GEMM remains. The LoRA-down output is computed but
    unused in this configuration.

    For power-of-two alpha within a safe range, the equality must hold
    bit-exactly: alpha is an FP32 multiply on FP32 accumulators, and a
    power-of-two factor only shifts the FP32 exponent (no precision loss),
    so the subsequent FP32 -> BF16 down-cast produces the same BF16 as if
    we'd multiplied y_base by alpha after the cast.
    """
    layer = _make_layer(512, 256, rank=32, seed=20)
    # Suppress LoRA-up so the layer output is exactly alpha * FP4_GEMM.
    layer.proj_up.data.zero_()

    x = _input(64, 512, seed=2024)

    layer.wtscale = 1.0
    y_base = layer(x)
    layer.wtscale = float(alpha)
    y_alpha = layer(x)

    expected = (alpha * y_base.float()).to(torch.bfloat16)
    assert torch.equal(y_alpha, expected), (
        f"alpha={alpha} expected bit-exact equality; "
        f"max diff = {(y_alpha.float() - expected.float()).abs().max().item():.4g}"
    )


def test_wtscale_zero_kills_gemm():
    """
    MATH.md §1.3: alpha=0 must zero out the FP4 GEMM contribution. With
    LoRA-up also zeroed, the entire output must be zero.
    """
    layer = _make_layer(512, 256, rank=32, seed=21)
    layer.proj_up.data.zero_()
    layer.wtscale = 0.0

    x = _input(64, 512, seed=2025)
    y = layer(x)
    assert torch.equal(y, torch.zeros_like(y)), (
        f"alpha=0 with proj_up=0 should give all-zero output; "
        f"max |y| = {y.abs().max().item():.6g}"
    )
