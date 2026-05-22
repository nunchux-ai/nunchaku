# NVFP4 Kernel Tests

These tests validate that [`packages/nunchaku/MATH.md`](../../MATH.md)
correctly describes the SM 12.0 NVFP4 W4A4 GEMM kernel, following the test
plan in [`packages/nunchaku/MATH_TEST_PLAN.md`](../../MATH_TEST_PLAN.md).

## Layout

```
packages/nunchaku/tests/kernels/
├── conftest.py              # path repair + custom-mark registration
├── _helpers.py              # FP4/FP8 oracle, ascales decoder, requires_fp4
├── test_quantize.py         # §1: quantize_w4a4_act_fuse_lora correctness
├── test_gemm_shapes.py      # §§2-3: GEMM correctness at varied shapes
└── test_math_properties.py  # §5: linearity / determinism / scale invariance
```

## Requirements

- A CUDA device of capability ≥ **(12, 0)** (Blackwell / consumer SM 12.0).
- An installed `nunchaku` wheel with the compiled `nunchaku._C` extension
  (`pip install nunchaku-*+cu13.0torch2.12-*.whl`).
- `pytest`, `torch` ≥ 2.12 (with FP4/FP8 dtype support).

If any of the above is missing, every test is skipped via the
[`requires_fp4`](./_helpers.py) marker; nothing fails.

## Running

```bash
# Run all the fast tests.
~/venv/diffusers/bin/python -m pytest packages/nunchaku/tests/kernels/ -m "not slow"

# Include the diffusion-realistic shapes (FFN-up / FFN-down at K=15360).
~/venv/diffusers/bin/python -m pytest packages/nunchaku/tests/kernels/
```

## Coverage map (vs. MATH_TEST_PLAN.md)

| Test plan section | File / function | Notes |
|---|---|---|
| §1.1 per-group scales | [`test_quantize.py::test_per_group_scale_matches_oracle`](test_quantize.py) | parametrized over (sigma, M, K) |
| §1.2 MSCALE_MAX saturation | [`test_quantize.py::test_saturation_at_mscale_max`](test_quantize.py) | exact 448.0 expected |
| §1.3 FP4 codebook | [`test_quantize.py::test_fp4_codebook_roundtrip_via_oracle`](test_quantize.py) | one test per (sign, magnitude) |
| §1.4 sign symmetry | [`test_quantize.py::test_sign_symmetry_scales_identical`](test_quantize.py), `..._act_bytes_flipped` | byte-exact |
| §1.5 zero rows | [`test_quantize.py::test_zero_rows_produce_finite_zero_scale`](test_quantize.py) | scale=0, **act bytes != 0** - see MATH.md §8.4 |
| §1.6 smooth-fold | [`test_quantize.py::test_smooth_quant_fold_uniform_constant`](test_quantize.py) | uniform smooth only - see MATH.md §8.2 |
| §2.1-2.3, §3.1-3.3 GEMM | [`test_gemm_shapes.py`](test_gemm_shapes.py) | black-box invariants via `SVDQW4A4Linear` |
| §2.3 alpha | [`test_gemm_shapes.py::test_wtscale_linear`](test_gemm_shapes.py), `test_wtscale_zero_kills_gemm` | LoRA-up zeroed |
| §3.2 tail / padding | [`test_gemm_shapes.py::test_M_tail_shapes_finite_and_consistent`](test_gemm_shapes.py) | row-padding doesn't leak |
| §3.3 diffusion shapes | [`test_gemm_shapes.py::test_flux_realistic_shape`](test_gemm_shapes.py) | marked `slow` for FFN sizes |
| §5.1 linearity in A | [`test_math_properties.py::test_linearity_in_A_approximate`](test_math_properties.py) | 2x tolerance |
| §5.3 bilinearity (c · A) | [`test_math_properties.py::test_scale_equivariance_power_of_two`](test_math_properties.py) | bit-exact for c=2^k |
| §5.4 output bound | [`test_math_properties.py::test_output_max_bound_holds`](test_math_properties.py) | QVALUE_MAX² · MSCALE_MAX² · K · α |
| §5.5 determinism | [`test_math_properties.py::test_determinism_bit_exact`](test_math_properties.py) | strict at single-tile; 1-ULP at scale - see MATH.md §8.5 |
| §5.6 sign symmetry (GEMM) | [`test_math_properties.py::test_sign_symmetry`](test_math_properties.py) | bit-exact |

Sections **§4 (fused epilogues)**, **§6 (cross-impl)**, and **§7 (LPIPS)**
are not covered here: §4 requires building the kernel directly via the
private `gemm_w4a4_launch` C++ interface (not exposed to Python), §6
requires a second implementation to diff against, and §7 already exists
under [`tests/v1/flux/`](../v1/flux/) and friends.

## Findings recorded in MATH.md

While developing these tests, the following kernel behaviours were
discovered and added to [`MATH.md`](../../MATH.md) §8:

1. FP4 weight quantization is offline only (`assert(false)` at runtime).
2. `ascales` storage is warp-interleaved, not row-major.
3. `smooth_factor` requires a `packed_wscale_t` layout, not a flat `[N]` tensor.
4. `quantize_w4a4_act_fuse_lora` crashes if `smooth=None` is passed.
5. Zero-input groups encode as FP4 byte `0x77`, not `0x00`.
6. The kernel is 1-BF16-ULP non-deterministic on some non-tile-aligned shapes.

Each finding is testable via one of the cases above.
