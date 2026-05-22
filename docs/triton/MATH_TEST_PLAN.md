# NVFP4 Math Test Plan

This document defines the test coverage needed to validate that an
implementation of the NVFP4 W4A4 GEMM described in [`MATH.md`](MATH.md)
behaves correctly on a real CUDA device.

The test plan is **deliberately implementation-agnostic**. It does not assume
the kernel uses `mma.sync.mxf4nvf4` (current SM 12.0 consumer path), nor
`tcgen05.mma` (TMEM-based SM 10.0a / 11.0 datacenter path), nor a software
table-lookup emulation. All tests interact only with the **public host
operations** defined by Nunchaku:

| Op (host-side) | Defined at | What it does |
|---|---|---|
| `quantize_w4a4_act_fuse_lora` | [`gemm_w4a4_launch.cuh:51`](src/kernels/zgemm/gemm_w4a4_launch.cuh:51) | BF16/FP16 → (FP4 act, FP8 ascales) + optional fused LoRA-down |
| `gemm_w4a4(..., fp4=true)` | [`gemm_w4a4_launch.cuh:22`](src/kernels/zgemm/gemm_w4a4_launch.cuh:22) | FP4 × FP4 → BF16/FP16 (+ optional fused requant, LoRA-up, GELU, bias, smooth-fold) |
| `gemm_w4a4(..., fp4=true)` with `qout` set | same | FP4 × FP4 → next-layer FP4 inputs (fused requantize) |

Every test below specifies its **input tensors**, **what numerical property it
checks**, and **what tolerance is acceptable**. Implementations that produce
identical observable behaviour pass; internal microarchitecture is irrelevant.

All tests target **real CUDA devices** (no CPU-only path) and must be runnable
under `pytest packages/nunchaku/tests/kernels/`.

---

## 0. Conventions and shared infrastructure

### 0.1 Tensor formats (must be observable)

- `act`     : `INT8[M, K/2]`, two packed FP4 `e2m1` per byte, fragment-shuffled per [`MATH.md`](MATH.md#33-weight-wgt-layout).
- `wgt`     : `INT8[N_pad, K_pad/2]`, packed FP4 `e2m1`, repacked column-major-by-tile.
- `ascales` : `FP8_E4M3[K_pad/16, M]`.
- `wscales` : `FP8_E4M3[K_pad/16, N_pad]`.
- `wtscale` : `FP32` scalar (per-tensor `alpha`).
- `out`     : `BF16` or `FP16[M, N]`.

`M`, `N`, `K` may be padded to 128 internally; tests must report results at
the unpadded shape.

### 0.2 Reference (oracle) math

Every kernel-level test references a **pure-PyTorch FP32 oracle** implementing
exactly the algorithm from [`MATH.md`](MATH.md):

```python
def nvfp4_reference(act_bf16, wgt_bf16, group_size=16,
                    qvalue_max=6.0, mscale_max=448.0, alpha=1.0):
    # Per-group quantization of activations
    A_groups = act_bf16.reshape(M, K // group_size, group_size).float()
    sA  = (A_groups.abs().amax(dim=-1) / qvalue_max).clamp(max=mscale_max)
    sA  = sA.to(torch.float8_e4m3fn).float()        # round to FP8 e4m3
    qA  = (A_groups / sA[..., None]).clamp(-qvalue_max, qvalue_max)
    qA  = round_to_e2m1(qA)                          # exact FP4 e2m1 RNE

    # Same for weights
    W_groups = wgt_bf16.reshape(N, K // group_size, group_size).float()
    sW  = (W_groups.abs().amax(dim=-1) / qvalue_max).clamp(max=mscale_max)
    sW  = sW.to(torch.float8_e4m3fn).float()
    qW  = (W_groups / sW[..., None]).clamp(-qvalue_max, qvalue_max)
    qW  = round_to_e2m1(qW)

    # Dequantized GEMM in FP32
    A_dq = (qA * sA[..., None]).reshape(M, K)
    W_dq = (qW * sW[..., None]).reshape(N, K)
    return alpha * (A_dq @ W_dq.T)                   # FP32
```

`round_to_e2m1` and FP8-`e4m3` rounding are provided by
`torch.float8_e4m3fn` casts and a small helper for FP4 RNE (8 representable
magnitudes, RNE tie-breaking). The oracle is FP32 to avoid double-quantization
noise.

### 0.3 Tolerances

FP4 has ~7% relative quantization error per element, FP8 scales another
~5%. Absolute tolerances are therefore set relative to per-output-element max
expected magnitude, not to fixed machine-epsilon values.

For an output element `y[m,n]` and oracle `y_ref[m,n]`:

```
atol(m,n) = max(0.05 * |y_ref[m,n]|, 0.05 * K**0.5 * max_abs_input)
rtol      = 0.05
```

The `K**0.5` term reflects RMS accumulation noise. Tests that require
**bit-exact** behaviour (e.g. scale-layout tests) state so explicitly and use
`==` not `allclose`.

### 0.4 Determinism

All tests fix `torch.manual_seed(...)` and require deterministic kernel
behaviour. Kernels that internally reorder K-reductions in a data-dependent
way must report this and be marked `@pytest.mark.nondeterministic`.

---

## 1. Quantization tests (`quantize_w4a4_act_fuse_lora`)

These tests validate the BF16 → (FP4, FP8) path **in isolation**, before
involving any GEMM. They are the cheapest signal that a port preserves the
on-disk encoding.

### 1.1 Per-group scale derivation

**Inputs**: `x = torch.randn(M, K, dtype=bfloat16) * sigma`, for
`sigma ∈ {0.1, 1.0, 10.0}`, `M ∈ {32, 256, 1024}`, `K ∈ {64, 128, 512, 2048}`.

**Procedure**: call quantize op, read back `ascales` (FP8 `e4m3`), decode to
FP32, compare against `min(max_abs(group) / 6.0, 448.0)` from the oracle.

**Assertion**: `|s_kernel - s_oracle| <= 0.5 * ulp_e4m3(s_oracle)` (one ULP of
FP8 `e4m3` rounding).

### 1.2 Saturation at `MSCALE_MAX = 448`

**Inputs**: `x` with one outlier set to `1e6` in each of `M` rows; other
elements `randn(...) * 0.01`.

**Assertion**: the corresponding scale is **exactly** `448.0` (after FP8
decode), and the matching quantized FP4 element decodes to a value in
`{-6, -4, -3, -2, ..., 6}` (the FP4 codebook).

### 1.3 FP4 codebook coverage

**Inputs**: `x[m, k] = c` for every `c` in the FP4 reference codebook
multiplied by a known FP8-representable scale.

**Assertion**: round-trip through quantize → dequantize matches `c` exactly
(no error). This catches mantissa/exponent-bit miswirings.

### 1.4 Sign symmetry

**Inputs**: `x` and `-x`.

**Assertion**: scales identical, FP4 values are bit-flipped sign bit only.

### 1.5 Zero rows

**Inputs**: a 16-K group of all zeros.

**Assertion**: the corresponding scale is **finite** (no NaN / Inf) and the
dequantized output is exactly zero. (Implementations that compute `1/scale`
with `scale = 0` must guard this — most current code does, but it's a common
porting bug.)

### 1.6 Smooth-quant fold

**Inputs**: random `x`, random `smooth_factor`. Run quantize with smooth-fold
ON.

**Assertion**: the dequantized output equals `x / smooth_factor` (oracle), to
within tolerance from §0.3. This is the *only* test that exercises the
`h2div` path on the FP4 quant side.

### 1.7 Bit-exact layout test

**Inputs**: fixed seed, M=256, N=128, K=64, BF16 input.

**Assertion**: the `act` byte buffer and `ascales` byte buffer match a
**checked-in golden binary** (`tests/kernels/data/golden_nvfp4_quant.bin`)
byte-for-byte. This is what catches checkpoint-format regressions across
implementations.

---

## 2. Single-tile GEMM correctness

These tests fix a single CTA tile (`M=BLOCK_M=256`, `N=BLOCK_N=128`,
`K=WARP_K=64`) so that the kernel issues exactly one block of MMAs. They are
the simplest microbenchmark of correctness.

### 2.1 Identity-weight test

**Inputs**: `W = I` (logically), encoded as NVFP4 with all unit scales.
`A = random BF16`.

**Assertion**: `out ≈ A` within tolerance from §0.3. Catches transpose /
fragment-mapping bugs.

### 2.2 Random small GEMM

**Inputs**: random BF16 `A, W`, quantized through the *same* quantize op as
the activations (so we test the GEMM, not the quantizer).

**Procedure**: kernel-quantize `A`, kernel-quantize `W`, run
`gemm_w4a4(..., fp4=true)`, compare BF16 output against the FP32 oracle.

**Assertion**: `torch.allclose(out, oracle, atol=…, rtol=0.05)` per §0.3.

### 2.3 `alpha` (per-tensor) scaling

**Inputs**: same as §2.2 but `alpha ∈ {0.5, 1.0, 2.0, 100.0}`.

**Assertion**: output scales linearly with `alpha` to FP32 precision (no
double-saturation). Catches whether `alpha` is applied pre- or post-FP16
downcast.

### 2.4 Outlier robustness

**Inputs**: insert outliers `|x| ∈ {1e3, 1e4, 1e5}` into one K-group of `A` or
`W`.

**Assertion**: oracle is computed with the *same* satfinite saturation; result
must match oracle within tolerance. Catches missing `satfinite` on the scale
or element path.

### 2.5 K-sweep at fixed M, N

**Inputs**: `M=256, N=128`, `K ∈ {64, 128, 256, 512, 1024, 4096}`.

**Assertion**: error stays bounded; specifically, RMS error grows no faster
than `K**0.5` (concentration-of-measure expectation for unbiased quantization
noise).

### 2.6 Mixed-precision accumulator

**Inputs**: a deliberately adversarial pattern where many small FP4 products
must accumulate to a value larger than FP16 max
(`~1e3 * 1e3 * K = 6.5e6` for `K=64`).

**Assertion**: result is finite and matches oracle. This validates the
**FP32 accumulator** requirement; a kernel that down-cast to FP16 mid-K would
silently saturate.

---

## 3. Multi-tile / shape-tail correctness

These tests exercise `M`/`N`/`K` values that **don't** divide cleanly into the
CTA tile, including the values actually used by FLUX/Qwen-Image transformer
blocks.

### 3.1 Power-of-two shapes

**Inputs**: cartesian product
`M ∈ {16, 32, 64, 128, 256, 1024, 4096}` × `N ∈ {16, 64, 128, 256, 1024, 3072}` ×
`K ∈ {64, 128, 1024, 3072, 12288}`.

### 3.2 Non-multiple shapes (padding)

**Inputs**: `M=4097` (one past tile), `N=129`, `K=65`.

**Assertion**: kernel returns shape `[M, N]` matching unpadded oracle.
Padding zeros must not leak into the visible output.

### 3.3 Diffusion-realistic shapes

**Inputs**: shapes observed in FLUX-dev transformer blocks
(`M=4096, N=3072, K=3072` for QKV projection;
`M=4096, N=15360, K=3072` for FFN-up;
`M=4096, N=3072, K=15360` for FFN-down).

**Assertion**: oracle match within tolerance.

### 3.4 Batch / grouped dimension

**Inputs**: stacked GEMMs `B ∈ {1, 2, 8}` along the M dimension.

**Assertion**: each slice matches its individual-call oracle (catches stride
bugs in `act` global addressing).

---

## 4. Fused-epilogue tests

The same `gemm_w4a4` host call can fuse several post-matmul operations. Each
must be tested both in isolation and combined.

### 4.1 Bias add

**Inputs**: random `bias[N]` in BF16.

**Assertion**: `out = matmul + bias` within §0.3 tolerance.

### 4.2 SiLU activation

**Inputs**: `fuse_silu = True`.

**Assertion**: `out = silu(matmul + bias)` within tolerance.

### 4.3 Fused re-quantize for next layer

**Inputs**: enable `qout`, `oscales`, `smooth_factor`.

**Procedure**:

1. Run kernel with re-quantize fused; capture `qout`, `oscales`.
2. Run *unfused* equivalent: same GEMM with no re-quantize, then call
   `quantize_w4a4_act_fuse_lora` on the BF16 result; capture its outputs.
3. **Bit-compare** `qout` and `oscales` against the unfused result.

**Assertion**: byte-equal. This is the test that gates checkpoint
compatibility across implementations.

### 4.4 LoRA-up fusion

**Inputs**: random `lora_up`, `lora_act_in`.

**Assertion**: `out = matmul + lora_up @ lora_act_in.T` within tolerance.
Catches rank-axis bugs.

### 4.5 LoRA-down fusion (inside `quantize_w4a4_act_fuse_lora`)

**Inputs**: random `lora_down`, expect fused LoRA-down output in addition to
the quantized activations.

**Assertion**: `lora_act_out ≈ x @ lora_down.T` (FP32), and quantized act
matches the unfused quantizer.

### 4.6 RMSNorm + RoPE epilogue

**Inputs**: random `norm_q`, `norm_k`, `rotary_emb`.

**Assertion**: matches a pure-PyTorch reference of
`rope(rmsnorm(matmul, norm_weight))` for the Q and K projection paths.

### 4.7 QKV packing epilogue

**Inputs**: fused QKV projection emitting `out_q`, `out_k`, `out_v`.

**Assertion**: per-head splits match the equivalent
`out.split([H_q, H_k, H_v], dim=-1)`.

---

## 5. Numerical-property tests (math-level)

These tests probe properties that must hold regardless of implementation
strategy.

### 5.1 Linearity in `A`

`gemm(A1 + A2, W) ≈ gemm(A1, W) + gemm(A2, W)` within additive tolerance.
*Caveat*: only approximate because quantization is non-linear; assertion uses
2x normal tolerance.

### 5.2 Linearity in `W`

Same as §5.1 with `W1 + W2`.

### 5.3 Bilinearity / scale equivariance

`gemm(c*A, W) ≈ c * gemm(A, W)` for `c ∈ {0.5, 2, 10}` (within saturation
limits). Verifies that no `MSCALE_MAX` clipping is hit mid-test.

### 5.4 Output-element max bound

`|out[m,n]| <= 6 * 448 * K * alpha` should hold by construction
(`QVALUE_MAX * MSCALE_MAX * K * alpha`). Probabilistic check: assert no
element exceeds this hard bound.

### 5.5 Deterministic reproducibility

Run the same inputs twice; require **bit-identical** outputs (or document
non-determinism explicitly with `@pytest.mark.nondeterministic`).

### 5.6 Symmetry: `(-A) @ W == -(A @ W)`

Within tolerance. Catches asymmetric quantization rounding.

### 5.7 `alpha` invariance under rescaling

`gemm(A, W, alpha=c)` should equal `gemm(A, W, alpha=1) * c` for any `c` that
doesn't trigger output saturation. Within tolerance.

---

## 6. Cross-implementation differential tests

If two implementations are available (e.g. the existing SM 12.0 kernel and a
new SM 11.0 TMEM kernel), they must agree on the *same observable surface*.

### 6.1 Activation-encoding equivalence

For the same BF16 input, `quantize_w4a4_act_fuse_lora` must produce
**byte-identical** `act` and `ascales` across implementations.

### 6.2 GEMM-output equivalence

For the same pre-quantized inputs, `gemm_w4a4(..., fp4=true)` must produce
outputs within a much tighter tolerance than §0.3:

```
rtol = 1e-3
atol = K * 1e-5 * max_abs_input
```

Any larger discrepancy signals a non-equivalent implementation (e.g. different
accumulator order, different scale broadcast).

### 6.3 Requantize-roundtrip equivalence

Chain `gemm → requantize → gemm → requantize` across two layers using both
implementations. The intermediate `qout`/`oscales` must be byte-identical.

### 6.4 Performance bounds

Not a correctness test, but record:

- Time per `gemm_w4a4` call at the shapes in §3.3.
- Memory traffic (`nsys nvprof` GPU counters).

A new implementation should be within 0.5x – 2x of the reference; gross
deviations indicate a different algorithm (good or bad), worth investigating.

---

## 7. End-to-end backstop (existing tests)

The existing LPIPS-based pipeline tests in
[`tests/v1/flux/`](tests/v1/flux/) and [`tests/v1/qwenimage/`](tests/v1/qwenimage/)
remain valuable as the final acceptance gate. They are insufficient on their
own (see [`MATH.md`](MATH.md) §7 / the review of coverage in the project
README) but they catch *systemic* drift that microtests might miss
(model-specific quantization sensitivities, KV-cache interactions, etc.).

Test plan recommendation: **microtests (§§ 1–6) must all pass green before
running the LPIPS suite at all.** LPIPS is for assurance, not debugging.

---

## 8. File layout

Recommended directory structure under [`packages/nunchaku/tests`](tests/):

```
tests/
  kernels/
    __init__.py
    conftest.py              # device fixtures, oracle helpers, ULP/FP8 utilities
    data/
      golden_nvfp4_quant.bin # see §1.7
      golden_gemm_small.npz  # see §6 cross-impl
    test_quantize.py         # §1
    test_gemm_single_tile.py # §2
    test_gemm_shapes.py      # §3
    test_epilogues.py        # §4
    test_math_properties.py  # §5
    test_diff_impl.py        # §6 (skipif only-one-impl-available)
```

Each test file should:

- Skip the entire file if `get_precision() != "fp4"` or
  `not torch.cuda.is_available()` or
  `device_capability < (10, 0)` (TMEM) or
  `device_capability < (12, 0)` (consumer Blackwell), whichever is the
  declared target of the implementation.
- Use `pytest.mark.parametrize` to enumerate shapes.
- Print actual vs. expected error magnitudes on failure for diagnostics.

---

## 9. Required test utilities (`tests/kernels/conftest.py`)

The following helpers must be implemented once, in the test suite (not in the
kernel under test):

1. `def nvfp4_reference(...)` — pure-PyTorch FP32 oracle (see §0.2).
2. `def fp4_e2m1_encode(x: Tensor) -> Tensor` and `def fp4_e2m1_decode(...)`
   — pure-PyTorch FP4 RNE quantizer with saturation to ±6.
3. `def fp8_e4m3_round(x: Tensor) -> Tensor` — uses `torch.float8_e4m3fn` cast.
4. `def ulp_e4m3(x) -> Tensor` — for tolerance computation in §1.1.
5. `def assert_close_nvfp4(out, ref, K)` — wraps §0.3 tolerance formula.
6. `def assert_bytes_equal(actual: Tensor, golden_path: Path)` — for §1.7,
   §4.3, §6.1, §6.3.
7. `def synthesize_outlier_input(shape, magnitude, density)` — for §1.2, §2.4.

These helpers are intentionally **independent of the Nunchaku kernel**, so
they validate any future implementation including pure software emulation,
TMEM, or `mma.sync`.

---

## 10. Coverage summary

| Concern                              | Tests              |
|---|---|
| Bit-exact on-disk encoding           | §1.7, §4.3, §6.1, §6.3 |
| FP4 codebook & rounding              | §1.3, §1.4, §1.5     |
| Per-group scale derivation           | §1.1                 |
| Scale saturation (`MSCALE_MAX`)      | §1.2, §2.4           |
| Element saturation (`QVALUE_MAX`)    | §1.2, §5.4           |
| GEMM correctness                     | §2.1, §2.2, §3.1–3.3 |
| FP32 accumulator                     | §2.6                 |
| Per-tensor `alpha`                   | §2.3, §5.7           |
| Smooth-quant fold                    | §1.6, §4.3           |
| LoRA up / down                       | §4.4, §4.5           |
| GELU / SiLU fusion                   | §4.2                 |
| Bias                                 | §4.1                 |
| RMSNorm + RoPE                       | §4.6                 |
| QKV pack                             | §4.7                 |
| Determinism                          | §5.5                 |
| Cross-impl byte-level equality       | §6.1, §6.3           |
| Cross-impl numerical equivalence     | §6.2                 |