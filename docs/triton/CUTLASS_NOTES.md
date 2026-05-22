# CUTLASS Notes for a Nunchaku NVFP4 → SM110 (Thor) TMEM Port

Companion to [`MATH.md`](MATH.md). Survey of NVIDIA CUTLASS 4.5 (the tree at
[`packages/cutlass`](../cutlass)) to decide whether its kernels can be reused or
adapted to port the W4A4-NVFP4 GEMM in
[`src/kernels/zgemm/gemm_w4a4.cuh`](src/kernels/zgemm/gemm_w4a4.cuh) from its
current **SM 12.0 register-resident `mma.sync` path** to the new
**SM 11.0 (NVIDIA Thor) `tcgen05.mma` / TMEM path**.

Bottom line up front: **yes, CUTLASS is useful for this port**, with caveats
documented at the end.

---

## 0. Target arch reminder

| Arch | Family | NVFP4 path | Scale dtype | Notes |
|---|---|---|---|---|
| SM 10.0 / 10.0a | Datacenter Blackwell (B100/B200/GB200) | `tcgen05.mma.kind::mxf4nvf4` (TMEM) | `ue4m3` or `ue8m0` | Full TMEM, has `mma.sync` too |
| SM 10.3 / 10.3a | Datacenter Blackwell "Ultra" (B300) | `tcgen05.mma.kind::mxf4nvf4` (TMEM, ultra) | `ue4m3` or `ue8m0` | TMEM + 2-SM cluster MMA |
| **SM 11.0 / 11.0a (Thor)** | **Automotive / robotics Blackwell** | **`tcgen05.mma.kind::mxf4nvf4` (TMEM only)** | **`ue4m3` or `ue8m0`** | **No `mma.sync` block-scale; CUDA 13.0+** |
| SM 12.0 / 12.0a | Consumer Blackwell (RTX 50, GeForce) | `mma.sync.kind::mxf4nvf4` (register) | `ue4m3` or `ue8m0` | **No TMEM**, this is what nunchaku currently uses |

The two paths are *not* interchangeable. SM110 inherits the SM100 (TMEM)
codegen, **not** the SM120 (register) codegen. Confirmed from
[`packages/cutlass/include/cute/arch/config.hpp`](../cutlass/include/cute/arch/config.hpp:100-118):

```cpp
// SM110 specific configs
#if (defined(CUTLASS_ARCH_MMA_SM110A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM110F_ENABLED))
#  define CUTE_ARCH_TMA_SM90_ENABLED
#  define CUTE_ARCH_TCGEN05_MXF4_MMA_ENABLED
#  define CUTE_ARCH_TCGEN05_MXF4NVF4_MMA_ENABLED   // <-- NVFP4 TMEM MMA
#  define CUTE_ARCH_TCGEN05_TMEM_ENABLED
#  define CUTE_ARCH_TMA_SM100_ENABLED
...
```

Notably the SM120 macros (`CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED`, etc.,
[`config.hpp`](../cutlass/include/cute/arch/config.hpp:159-169)) are **not**
defined on SM110. This means nunchaku's current FP4 kernel
([`gemm_w4a4.cuh:200`](src/kernels/zgemm/gemm_w4a4.cuh:200), which emits
`mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X ... .ue4m3`) will
**not** compile or run on Thor — there is no register-resident block-scaled FP4
`mma.sync` PTX in that arch.

---

## 1. What CUTLASS actually has for NVFP4

### 1.1 PTX-level atoms

| Atom file | What it wraps | Useful for SM110? |
|---|---|---|
| [`include/cute/arch/mma_sm120.hpp:3216`](../cutlass/include/cute/arch/mma_sm120.hpp:3216) | `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64...ue4m3` — *exact PTX nunchaku emits today* | **No**: same family as nunchaku's current code, register-resident, not present on SM110 |
| [`include/cute/arch/mma_sm120.hpp:3137`](../cutlass/include/cute/arch/mma_sm120.hpp:3137), [`:3159`](../cutlass/include/cute/arch/mma_sm120.hpp:3159) | `ue8m0` (MXFP4) variants of the same instruction | No |
| [`include/cute/arch/mma_sm100_umma.hpp:1614`](../cutlass/include/cute/arch/mma_sm100_umma.hpp:1614) (`SM100_MMA_MXF4_SS`) | `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [tmem_c], desc_a, desc_b, idesc, [tsfa], [tsfb], p` | **Yes**: this is the TMEM atom enabled for SM110 |
| [`mma_sm100_umma.hpp:1685`](../cutlass/include/cute/arch/mma_sm100_umma.hpp:1685) (`SM100_MMA_MXF4NVF4_SS_SPARSE`) | Sparse variant of the same TMEM MMA | Not needed (nunchaku has no sparsity) |
| [`mma_sm100_umma.hpp:1754`](../cutlass/include/cute/arch/mma_sm100_umma.hpp:1754) (`SM100_MMA_MXF4_2x1SM_SS`) | 2-SM cluster MMA | Probably not on Thor (Thor has limited cluster sizes, must verify) |

`SM100_MMA_MXF4_SS` is the **core building block** for an SM110 port. Its
descriptor (`idescE`) carries the scale-format field (`UE4M3` vs `UE8M0`),
selected by template type, so the NVFP4 case (`float_ue4m3_t`) maps to the
same atom: see
[`include/cute/arch/mma_sm100_desc.hpp:217-231`](../cutlass/include/cute/arch/mma_sm100_desc.hpp:217).

Constraints from `SM100_MMA_MXF4_SS`:

- `M == 128` (atom M is fixed at 128, *not* 16 like SM120; this is a big shape change).
- `N % 8 == 0`, `8 ≤ N ≤ 256`.
- `VS ∈ {16, 32}`: 16 = NVFP4 micro-block (our case), 32 = MXFP4.
- A/B operands live in **shared memory** (referenced by `desc_a`, `desc_b`),
  not registers; scale-factor operands live in **TMEM** (`tsfa_addr`,
  `tsfb_addr`).
- Output (C/D) lives in **TMEM** (`tmem_c` address), not registers.

Compare with nunchaku's current atom which is `m16n8k64`, register-resident
A/B/scales/accumulator — *none* of the operand storage matches.

### 1.2 Collective mainloops (one level up from atoms)

These are the "compose a CTA-level GEMM around the atom" pieces.

| File | Lines | Purpose |
|---|---|---|
| [`include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp`](../cutlass/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp) | 1108 | TMA + warp-specialized + TMEM mainloop for `tcgen05.mma` block-scaled FP4/FP6/FP8 |
| [`sm100_blockscaled_mma_array_warpspecialized.hpp`](../cutlass/include/cutlass/gemm/collective/sm100_blockscaled_mma_array_warpspecialized.hpp) | — | Grouped/MoE variant (relevant for MiniMax NVFP4) |
| [`sm103_blockscaled_mma_warpspecialized.hpp`](../cutlass/include/cutlass/gemm/collective/sm103_blockscaled_mma_warpspecialized.hpp) | — | SM103 "Ultra" variant; SM110 reuses SM100, not SM103 |
| [`builders/sm100_blockscaled_umma_builder.inl`](../cutlass/include/cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl) | 317 | Type-level builder that picks the right atoms, pipeline depth, etc. |

The builder is what `CollectiveBuilder<arch::Sm100, OpClassBlockScaledTensorOp, ...>` resolves to. It picks `SM100_MMA_MXF4_SS` automatically when ElementA = `cutlass::nv_float4_t<float_e2m1_t>` and ElementB = same.

### 1.3 Layout helper for the scale tensors

[`include/cutlass/detail/sm100_blockscaled_layout.hpp:62`](../cutlass/include/cutlass/detail/sm100_blockscaled_layout.hpp:62)
defines `Sm1xxBlockScaledConfig<SFVecSize>` (yes, "1xx" — used for SM100, SM103
and SM110 alike). For SFVecSize = 16, the K-major scale atom is:

```cpp
using SfKMajorAtom = Layout< Shape< Shape<_32,_4>, Shape<Int<SFVecSize>, _4>>,
                             Stride<Stride<_16,_4>, Stride<           _0, _1>>>;
using Blk_MN = _128;   // atom holds 128 rows
using Blk_SF =   _4;   // atom holds 4 SF per row tile
```

Comment in the file: *"A single indivisible block will hold 4 scale factors of
128 rows/columns (A/B matrix). 4 is chosen to make consecutive 32 bits of data
to have scale factors for only a single row (col). 32 bits corresponds to the
TMEM word size."*

This is the **canonical on-disk SF layout** that TMA + UTCCP will consume on
SM100/110/103. Compare with nunchaku's
[`MATH.md` §8.1](MATH.md) layout:

```
[M / BLOCK_M, K / 16, NUM_WARPS, AMSCALES_NUM_PACKS, AMSCALES_VALID_LANES]
   BLOCK_M=256,         NUM_WARPS=8
```

i.e. nunchaku tiles SF in 256-row CTA blocks split across 8 warps of 32 lanes
each, with a lane permutation `(r%8)*4 + r//8` matching the `m16n8k64` MMA
fragment. CUTLASS instead uses a 128-row atom with a `(_32,_4)` block shape
matched to TMEM word width.

**These two layouts are structurally similar (both pack 4 SFs per 32-bit word
for one row/col of an MMA atom) but not byte-identical.** Reusing a nunchaku
checkpoint file with CUTLASS would require an offline transpose-permute pass,
or the offline quantizer (`Linear.cpp`) would have to emit the
CUTLASS-canonical layout directly. See "Unresolved Q3" below.

### 1.4 Epilogues — the re-quantize-to-FP4 path

Nunchaku's
[`EpilogueQuantize<…, USE_FP4=true>`](src/kernels/zgemm/gemm_w4a4.cuh:930)
fuses FP32 accumulator → BF16 → smooth-fold → quantize back to FP4 + new SF,
in registers, before writing the next layer's activation.

CUTLASS has an exact analog: the **`LinCombBlockScaleFactor`** fusion
operation, instantiated in
[`examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu:130`](../cutlass/examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu:130):

```cpp
using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
    OutputSFVectorSize,   // 16 = NVFP4 micro-block
    ElementD,             // float_e2m1_t
    ElementCompute,       // float
    ElementSFD, LayoutSFDTag, // float_ue4m3_t for NVFP4, ue8m0 for MXFP4
    ElementC>;
```

with callback machinery in
[`include/cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp`](../cutlass/include/cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp).
This is the equivalent of nunchaku's
`quantize_w4a4_fp4_from_fpsum_warp`+output epilogue, on the TMEM/TMA path,
and writes both `D` (FP4) and `SFD` (FP8 e4m3) tensors. Re-using it would
make a fused two-layer-FP4 pipeline (the common diffusion-block case
described in [`MATH.md` §7](MATH.md)) cheap to express.

There is **no LoRA epilogue** in CUTLASS (`grep -r lora include/cutlass/`
returns nothing). Nunchaku's
[`EpilogueLoraDown`](src/kernels/zgemm/lora.cuh:1) and
`reduce_lora_act` paths would have to be ported as a hand-written extension
of `LinCombBlockScaleFactor` (or stacked as a separate kernel).

### 1.5 Working SM100/SM103 reference examples

These are the closest off-the-shelf demonstrations of what a Thor port would
look like. They compile against the SM110-shared atoms and collectives:

| Example | What | Why useful |
|---|---|---|
| [`examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu`](../cutlass/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu) | NVFP4 × NVFP4 → BF16 GEMM, dense, SM100 TMEM | Skeleton for any single nunchaku linear layer |
| [`examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu`](../cutlass/examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu) | NVFP4 × NVFP4 → NVFP4 + SFD, SM100 TMEM, with `LinCombBlockScaleFactor` | Skeleton for the fused "next layer's input" path |
| [`examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_fp4_grouped.cu`](../cutlass/examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_fp4_grouped.cu) | NVFP4 grouped GEMM (MoE), SM100 TMEM | Directly relevant for the **MiniMax-M2 NVFP4** MoE checkpoint in this repo (`models/MiniMax-M2.7-REAP-172B-A10B-NVFP4`) |
| [`examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_fp4_regular.cu`](../cutlass/examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_fp4_regular.cu) | NVFP4 regular (non-grouped) MoE GEMM | Alt MoE strategy |
| [`examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm_block_scaled.cu`](../cutlass/examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm_block_scaled.cu) | Grouped block-scaled GEMM | General grouped-GEMM scaffolding |

Tests under
[`test/unit/gemm/device/sm100_*f4*`](../cutlass/test/unit/gemm/device/) exercise
the same paths and are good correctness oracles.

There is also a real working integration to crib from in
[`packages/vllm/csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_kernels.cu`](../vllm/csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_kernels.cu)
(vLLM's SM100 NVFP4 GEMM wrapper), which calls into exactly this CUTLASS path
and shows the host-side SF-tensor layout convention
(`Sm100BlkScaledConfig::tile_atom_to_shape_SFA`).

---

## 2. How CUTLASS would slot into the nunchaku port

A reasonable plan, mapped onto the items in [`MATH.md` §11](MATH.md):

1. **The MMA atom** (`mma_fp4_tmem`): just be a thin wrapper around
   `cute::SM100_MMA_MXF4_SS<float_e2m1_t, float_e2m1_t, float,
   float_ue4m3_t, 128, N, 16, …>` — or, more practically, let the CUTLASS
   `CollectiveBuilder` pick it.
2. **TMEM allocation + descriptors**: use
   [`include/cute/arch/tmem_allocator_sm100.hpp`](../cutlass/include/cute/arch/tmem_allocator_sm100.hpp)
   and the TMEM copy/load atoms in
   [`include/cute/arch/copy_sm100.hpp`](../cutlass/include/cute/arch/copy_sm100.hpp);
   they're already SM110-compatible because they're gated on
   `CUTE_ARCH_TCGEN05_TMEM_ENABLED`.
3. **A/B + SF load pipeline**: use the `sm100_blockscaled_umma_builder.inl`
   builder, which sets up TMA loads of A/B, UTCCP transfers of SF tiles
   into TMEM, and a multi-stage pipeline (replaces nunchaku's
   `NUM_STAGES = 2` register-resident pipeline).
4. **Output**: write the FP32 accumulator into TMEM via `tcgen05.mma`,
   then download with `tcgen05.ld` in the epilogue warp.
5. **Re-quantize** with `LinCombBlockScaleFactor<16, float_e2m1_t, float,
   float_ue4m3_t, ...>` — output is FP4 data + ue4m3 SFD in the canonical
   CUTLASS SF layout.
6. **LoRA**: not provided by CUTLASS. Either
   - launch a separate small CUTLASS GEMM for LoRA-down (FP16 → FP32 acc)
     and add it into the main GEMM accumulator before the FP4 epilogue, or
   - write a custom epilogue callback that consumes a low-rank delta — this
     is the harder but more performant path.

Tile-shape choice on SM110 is constrained by `SM100_MMA_MXF4_SS`:
**M-atom is 128**, N is 8-256 in steps of 8, K is 64 per MMA. A reasonable
CTA tile starting point — taken straight from the 72b example —
is `MmaTileShape = Shape<_128, _128, _256>`, `ClusterShape = Shape<_1,_1,_1>`.
This is *completely different* from nunchaku's current
`BLOCK_M=256, BLOCK_N=128, WARP_M=32, WARP_N=128` register-tile geometry —
the entire warp layout is replaced by warp-specialization (producer/consumer
warps) under the hood by `KernelTmaWarpSpecializedBlockScaledSm100`.

---

## 3. Why CUTLASS is the right starting point (and why partially)

**Yes, use it.** Concrete reasons:

1. **The exact PTX for SM110 NVFP4 already exists** in CUTLASS, fully wrapped
   in C++ atoms (`SM100_MMA_MXF4_SS` with `ue4m3` scale type). Hand-writing
   the inline-PTX wrapper that mirrors nunchaku's current `asm volatile` block
   would just be re-implementing
   [`mma_sm100_umma.hpp:1639-1652`](../cutlass/include/cute/arch/mma_sm100_umma.hpp:1639).
2. **TMEM + UTCCP scaffolding is non-trivial** (allocator, descriptors,
   pipeline barriers, warp specialization, TMA multicast); rolling our own
   to replace nunchaku's register pipeline is several weeks of work that
   CUTLASS already did.
3. **An exact analog for the fused re-quantize epilogue** exists
   (`LinCombBlockScaleFactor`), so the cross-layer FP4 fusion described in
   [`MATH.md` §7](MATH.md) is expressible without inventing new PTX.
4. **MoE support** (`92_blackwell_moe_gemm_fp4_grouped.cu`) directly matches
   the actual deployment target in this repo
   ([`models/MiniMax-M2.7-REAP-172B-A10B-NVFP4`](../../models/MiniMax-M2.7-REAP-172B-A10B-NVFP4)).
5. **Existing real-world integrations** (vLLM, TensorRT-LLM) prove this path
   works for production NVFP4 inference.

**But** — partial reuse only. Things CUTLASS does *not* give:

- A pre-built FP4-from-FP16 *activation* quantize kernel that matches
  nunchaku's `quantize_w4a4_act_fuse_lora` (different semantics, no LoRA fold).
- A fused **LoRA-down** path. Must be added on top.
- Byte-compatible scale-tensor layout with existing nunchaku checkpoints
  (different atomic blocking — see Q3 below).
- A drop-in `__CUDA_ARCH__ >= 1200` → `__CUDA_ARCH__ >= 1100` patch. The whole
  kernel architecture has to change.

---

## 4. Unresolved questions / hazards

These need answers before committing to a CUTLASS-based port. Treat each as a
hard prerequisite, not as polish.

### Q1. Does CUTLASS actually compile *and run* against `-arch=sm_110a` today?

CUTLASS 4.5 only defines macros; it has **no `cutlass::arch::Sm110` tag**
(`grep arch::Sm110 packages/cutlass/include` is empty). The standard pattern
must therefore be: instantiate kernels with `ArchTag = cutlass::arch::Sm100`
(or `Sm103`) and let `CUTLASS_ARCH_MMA_SM110F_ENABLED` route the codegen.
Things that need to be empirically verified once Thor silicon / nvcc 13.x is
in hand:

- (a) Does `nvcc -arch=sm_110a` accept `cute::SM100_MMA_MXF4_SS<…>` and emit
  the right SASS? The atom is gated on `CUTE_ARCH_TCGEN05_MXF4NVF4_MMA_ENABLED`
  which **is** defined on SM110F per
  [`config.hpp:109`](../cutlass/include/cute/arch/config.hpp:109), so on paper
  yes.
- (b) Do the `Sm100`-flavored collectives compile under `arch::Sm100` while
  targeting sm_110? The kernels and builders contain numerous `if constexpr
  (cute::is_same_v<ArchTag, arch::Sm100>)` style checks, not `__CUDA_ARCH__`
  checks, so this *should* work, but it has only been smoke-tested on real
  SM100 hardware historically. We do not have an SM110-targeted example
  anywhere in the tree.
- (c) Are there any PTX features the SM100 collective uses
  (e.g. cluster-launch-control, multicast TMA, 2-SM MMA, certain `tcgen05.cp`
  variants) that are **not** present on Thor's SM110 silicon (Thor's
  per-die SM count and TMEM size differ from B200)? Specifically:
  - 2-SM `tcgen05.mma.cta_group::2` instructions exist on SM100/103 — must
    verify Thor's max `cluster_dim`.
  - TMA multicast requires `ClusterShape != _1,_1,_1` — Thor may force
    cluster shape to 1×1×1 like SM120 does (see comment in
    [`79a_blackwell_geforce_nvfp4_bf16_gemm.cu:48`](../cutlass/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu:48)).

### Q2. TMEM word-level scale layout vs `scale_vec::4X` lane semantics

Nunchaku's `scale_vec::4X` semantics
([`MATH.md` §4.1](MATH.md)) pack 4 FP8 scales × 32 lanes into one register per
warp, with very specific lane permutation. CUTLASS's `Sm1xxBlockScaledConfig`
packs 4 FP8 scales × 32-bit TMEM words *per row* of A/B with a different
permutation, and the SF tile is **transferred through TMEM via UTCCP**, not
held in registers. The math is the same (still "4 scales per MMA per
operand"), but:

- It is *not* obvious that the lane mapping comment
  [`gemm_w4a4.cuh:144`](src/kernels/zgemm/gemm_w4a4.cuh:144) ("8-row × 4-scale
  layout") matches `SfKMajorAtom`'s `Shape<<_32,_4>, <SFVec,_4>>`.
- A unit test that takes a known nunchaku checkpoint, runs both kernels, and
  bit-compares the dequantized A·B tensors at FP32 is the only credible way
  to settle this.

### Q3. Checkpoint compatibility with existing nunchaku weights

[`MATH.md` §8.0](MATH.md) notes that FP4 weight quantization is **offline**:
the `qweight`, `wscales`, `wtscale`, and `wcscales` buffers are produced by
[`Linear.cpp`](src/Linear.cpp:90) / the checkpoint-builder scripts, not at
runtime. The on-disk layout is the warp-interleaved one in
[`MATH.md` §8.1](MATH.md).

If we adopt CUTLASS for the TMEM path, the offline quantizer **must change**
to emit the CUTLASS-canonical layout via
[`Sm1xxBlockScaledConfig::tile_atom_to_shape_SFA`](../cutlass/include/cutlass/detail/sm100_blockscaled_layout.hpp:90).
This means:

- Existing FP4 checkpoints (e.g.
  `models/MiniMax-M2.7-REAP-172B-A10B-NVFP4`) cannot be loaded
  unchanged on Thor; they need a re-pack pass.
- Or we keep two layouts: nunchaku-legacy for SM120 GeForce path, CUTLASS for
  SM110/SM100 path. This is uglier but preserves existing artifacts.

Decision needed: re-pack at load time, re-pack offline once, or maintain
two layouts.

### Q4. NaN/Inf handling for the zero-input group

[`MATH.md` §8.4](MATH.md) documents a specific byte-level quirk: an all-zero
K-group becomes `q = 0x77 (= +6, +6)` with `s = 0`, because
`rcp.approx.ftz(0) = +inf`, `0 * inf = NaN`, and
`cvt.rn.satfinite.e2m1x2.f32(NaN) = +6`. CUTLASS's quantize fusion paths
(`LinCombBlockScaleFactor`) almost certainly do not reproduce this exact
bit-pattern. The dequantized values are still 0 either way, but any test
that does byte-equality with the legacy reference will fail. Need to
explicitly decide:

- Match CUTLASS bytes and re-bake reference outputs, or
- Patch CUTLASS to use the same `0 / 0 = +6` fast path (probably wrong; not
  recommended), or
- Mark the test class as bytewise-not-comparable across implementations.

### Q5. `alpha` (per-tensor weight scale, `wtscale`)

Nunchaku's NVFP4 path supports an FP32 `alpha` that scales the whole GEMM
output ([`MATH.md` §1.3](MATH.md), [`gemm_w4a4.cuh:341`](src/kernels/zgemm/gemm_w4a4.cuh:341)).
CUTLASS's stock `LinCombBlockScaleFactor` exposes `alpha` and `beta` as
standard linear-combination scalars, so this is straightforward — provided
we keep `alpha` strictly in FP32 and apply it on the FP32 accumulator (not
after FP4 quantize). Worth double-checking against
`fusion::LinCombBlockScaleFactor`'s exact application order.

### Q6. Smooth-quant fold (`smooth_factor`)

[`MATH.md` §8.2 and §8.3](MATH.md) document a per-channel `smooth_factor` that
is folded into the FP16 result before re-quantization. CUTLASS does not have
a stock "divide-by-per-N-vector before SF generation" fusion. Options:

- Treat `smooth_factor` as a constant in the offline checkpoint and absorb it
  into the weight (`W' = W · diag(smooth)`), then never apply at runtime.
  Mathematically equivalent and removes a runtime dependency.
- Add a custom epilogue callback chained before `LinCombBlockScaleFactor`.

The first is preferable if no algorithm assumes runtime adjustable smooth.

### Q7. LoRA fusion

Nunchaku's `quantize_w4a4_act_fuse_lora` and `EpilogueLoraDown` paths
([`MATH.md` §3.5](MATH.md), [`MATH.md` §7](MATH.md)) are not expressible in any
CUTLASS fusion we found. Either:

- Stage LoRA as a separate small GEMM (one BF16 GEMM into the FP32
  accumulator before the FP4 quantize), with a synchronization point.
  Simple, costs an extra kernel launch + memory traffic.
- Write a custom CUTLASS fusion callback that injects the LoRA branch into
  the epilogue. Performance-correct but probably the single hardest engineering
  item in the whole port.

### Q8. 1-ULP non-determinism

[`MATH.md` §8.5](MATH.md) notes ~1-BF16-ULP non-determinism on non-aligned M
shapes due to atomic LoRA reduction. CUTLASS warp-specialized kernels also
have non-deterministic ordering in stream-K and split-K paths. We need to
pick a determinism contract (probably "1-ULP-close" with deterministic split
within a single problem size) and test it.

### Q9. Performance: does CUTLASS beat a hand-rolled kernel on Thor?

The current nunchaku FP4 kernel on SM120 keeps everything register-resident
and skips shared memory for operands ([`MATH.md` §4.3](MATH.md)). On SM110 we
cannot do that — operands have to come through SMEM and SF through TMEM.
CUTLASS's warp-specialized + persistent + cluster-launch design is the
state-of-the-art template for this kind of pipeline, but there is no public
benchmark of NVFP4 on Thor yet. It is plausible (not certain) that a custom
kernel tuned for the specific tile shapes nunchaku cares about could beat
CUTLASS by 10-20%, but at much higher engineering cost. The pragmatic stance:
use CUTLASS, then revisit if benchmarks demand otherwise.

### Q10. CUDA-toolkit / driver availability

CUTLASS 4.5 gates SM110 on CUDA 13.0+
([`arch/config.h:132`](../cutlass/include/cutlass/arch/config.h:132)). The host
container in this repo currently ships CUDA 12.8 / 12.9 for the SM120 path
(see [`packages/vllm/CMakeLists.txt`](../vllm/CMakeLists.txt)). The Thor port
requires a separate toolchain. Confirm the CI / build matrix can handle two
CUDA major versions side-by-side, or pin the Thor build to its own image.

---

## 5. Recommendation

1. **Use CUTLASS 4.5+** as the GEMM core for the SM110 (Thor) port. Do **not**
   try to write `tcgen05.mma` PTX wrappers from scratch — the
   [`SM100_MMA_MXF4_SS`](../cutlass/include/cute/arch/mma_sm100_umma.hpp:1614)
   atom plus the `Sm100` collective+builder give us the right primitive at
   roughly the same level of abstraction as nunchaku's current
   [`mma_fp4`](src/kernels/zgemm/gemm_w4a4.cuh:191) wrapper.
2. **Start from the 72b example** and the
   [`vllm nvfp4_scaled_mm_kernels.cu`](../vllm/csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_kernels.cu)
   wrapper for the API shape; extend with the `LinCombBlockScaleFactor`
   fusion for the cross-layer FP4 chain.
3. **Treat layout/byte-compat (Q3) and LoRA (Q7) as the two big design
   decisions**; the rest of the unresolved items are mostly verify-and-test.
4. **Keep the SM120 register-resident path in tree** so consumer Blackwell
   (RTX 50) is not regressed. Dispatch on `__CUDA_ARCH__` (or runtime
   `cudaGetDeviceProperties` for the launch wrapper):
   - `__CUDA_ARCH__ == 1200/1210` → existing nunchaku kernel.
   - `__CUDA_ARCH__ == 1000/1003/1100` → new CUTLASS-based kernel.
5. **Write a portable reference implementation in PyTorch / Triton** before
   the CUDA work, validated against the existing SM120 kernel at FP32 acc,
   to use as an oracle while iterating on the SM110 kernel (avoiding the
   "kernel produces plausible garbage and we discover it three weeks in"
   failure mode).

In short: CUTLASS sources are highly useful — but they replace ~60% of
nunchaku's FP4 kernel work, not 100%. The remaining 40% is the
nunchaku-specific fusion (LoRA, smooth-quant, the exact on-disk SF layout)
and the validation pipeline.