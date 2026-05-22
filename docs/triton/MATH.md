# NVFP4 Math and CUDA Instructions in Nunchaku

This document describes the mathematical formulation and CUDA / PTX instructions
used by the **NVFP4** (NVIDIA FP4 with micro-block scaling) code path inside
Nunchaku's W4A4 GEMM kernels. The intent is to capture the current state of
the SM 12.0 (Blackwell) Tensor-Core based implementation so we can later port it
to the new **TMEM NVFP4** architecture (sm_110 that only has TMEM NVFP4 instructions)

All file references below are relative to [`packages/nunchaku`](packages/nunchaku).

---

## 1. NVFP4 numeric formats

Three packed numeric formats are involved in a single matmul.

### 1.1 Data elements (4-bit FP4, `e2m1`)

Each activation `A[i,k]` and weight `W[n,k]` element is stored as
**FP4 `e2m1`** (1 sign bit, 2 exponent bits, 1 mantissa bit). Encodable
magnitudes:

```
{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 }
```

so the **maximum representable magnitude is `QVALUE_MAX = 6.0`**, used in
[`gemm_w4a4.cuh`](src/kernels/zgemm/gemm_w4a4.cuh:91):

```cpp
constexpr float QVALUE_MAX       = 6.0f;
constexpr float RECPI_QVALUE_MAX = 1 / QVALUE_MAX;
```

Two FP4 values are packed per byte; an entire 16x64 MMA-A tile fits in
`uint4` (`packed_act_t = uint4`, 16 B/lane in [`gemm_base.cuh`](src/kernels/zgemm/gemm_base.cuh:156)).

### 1.2 Micro-block (per-group) scale (FP8 `e4m3`, "ue4m3")

Every **16 contiguous K-elements** share one **FP8 `e4m3`** scale (sometimes
written `ue4m3` because it carries no sign on the MMA-side scaling, although the
PTX hardware encoding is the standard signed `e4m3`).

Maximum representable magnitude (`MSCALE_MAX = 448.0f`,
[`gemm_w4a4.cuh`](src/kernels/zgemm/gemm_w4a4.cuh:93)):

```cpp
constexpr float MSCALE_MAX = 448.0f;
```

For each 1x16 micro-block `g` of a row, the scale is

```
s_g = min( max_abs(x_g) / QVALUE_MAX , MSCALE_MAX )
    = min( max_abs(x_g) / 6.0      , 448.0 )
```

and each element is stored as

```
q_i = round_to_e2m1_satfinite( x_i / s_g )      // FP4
```

so that the de-quantized value is approximately `q_i * s_g`.

### 1.3 Per-tile (per-tensor) outer scale (FP32, "alpha")

Because micro-block scales themselves saturate at 448, an additional global FP32
scalar `alpha` (`wtscale`) can re-scale the GEMM output. The kernel calls this
the *per-tensor scale of weight* in [`gemm_w4a4.cuh`](src/kernels/zgemm/gemm_w4a4.cuh:279):

```cpp
template<typename Epilogue, bool USE_ALPHA>
__device__ __forceinline__ static void gemm_w4a4_fp4_block(
    const BlockInfo binfo, const packed_act_t *act, const packed_wgt_t *wgt,
    const packed_amscale_t *ascales, const packed_wmscale_t *wscales,
    float alpha, /* per-tensor scale of weight */
    int M, int N, int K, ...);
```

After accumulation:

```
Y = alpha * sum_k ( dequant_fp4(A_k) * dequant_fp4(W_k) )
```

The factor `alpha` is just a float multiplied into the FP32 accumulators
(loop in [`gemm_w4a4.cuh`](src/kernels/zgemm/gemm_w4a4.cuh:341)):

```cpp
if constexpr (USE_ALPHA) {
    for (auto &pack : fpsum)
        for (int i = 0; i < 8; i++)
            pack.data[i] *= alpha;
}
```

---

## 2. Two-level quantization formula

Given an unquantized FP16/BF16 row `x[k]` of length `K`:

1. Split into groups of 16: `G = K / 16` micro-blocks.
2. For each group `g`:
   - `m_g     = max_{k in g} |x[k]|`
   - `s_g     = min(m_g / 6.0, 448.0)`            (FP8 `e4m3`)
   - `rs_g    = 1 / s_g`                          (FP32, computed via `rcp.approx.ftz.f32`)
   - `q[k]    = quantize_e2m1_satfinite(x[k] * rs_g)`   (FP4)
3. Store `q[k]` (FP4) and `s_g` (FP8 `e4m3`) into the activation / weight
   tensors respectively.

The matmul itself is then performed natively by the Tensor Core, which on
SM 12.0 reads `(q, s_g)` pairs directly via the new `block_scale` flavor of
`mma.sync`.

The corresponding code is split into three places:

| Where                                                                                      | What it produces                                                                  |
|---|---|
| [`Linear.cpp`](src/Linear.cpp:90)                                                          | Allocates `qweight` (INT8 = packed FP4) and `wscales` (FP8 `e4m3`) tensors. **Weight quantization itself is offline** for FP4: the runtime kernel `quantize_w4a4_wgt` is compiled out for FP4 (`assert(false)` in [`gemm_w4a4_launch_impl.cuh:549`](src/kernels/zgemm/gemm_w4a4_launch_impl.cuh:549)), so the `qweight`, `wscales`, `wtscale`, and `wcscales` buffers are produced at checkpoint-build time, not at runtime. |
| [`quantize_w4a4_act_fuse_lora`](src/kernels/zgemm/gemm_w4a4_launch.cuh:51)                  | Quantizes activations + scales on-line (used between two W4A4 layers).            |
| [`quantize_w4a4_fp4_from_fpsum_warp`](src/kernels/zgemm/gemm_w4a4.cuh:85)                  | Inline quantization of an FP16 GEMM result back to FP4 + FP8 for the next layer. |

---

## 3. Tensor layouts

### 3.1 Block tiling (CTA tile)

From [`gemm_base.cuh`](src/kernels/zgemm/gemm_base.cuh:34):

```
BLOCK_M = 256, BLOCK_N = 128
NUM_WARPS = 8, WARP_SIZE = 32

INSN_M = 16, INSN_N = 16, INSN_K = 64
WARP_M = BLOCK_M / NUM_WARPS = 32
WARP_N = BLOCK_N             = 128
WARP_K = INSN_K              = 64
```

So one CTA computes a `256x128xK` C-tile across 8 warps, each warp computing
`32x128` using `WARP_M_TILES=2`, `WARP_N_TILES=8` MMAs of shape
`m16n16k64`.

### 3.2 Activation `act` layout

`act` shape: `[M, K/2]` in `INT8` (two FP4s per byte).

Packed per-thread type is `packed_act_t = uint4` (16 B / lane), holding one
`m16k64` A-fragment for the FP4 MMA (one warp = 32 lanes = 16x64 elements x 4 bits
= 2048 bits = 32 lanes x 64 bits, but with the 4-fragment layout = `uint4`).

### 3.3 Weight `wgt` layout

`wgt` shape: `[N, K/2]`, repacked column-major-by-tile so each warp loads its
`n16k64` B-fragment as `packed_wgt_t = uint4`. Repacking is done at load time:

```cpp
// Linear.cpp quantize_w4a4_wgt swaps y/z to match MMA fragment order
std::swap(tmpout.y, tmpout.z);
```

### 3.4 Scales: `amscales` (act micro-scales) and `wmscales` (wgt micro-scales)

Both are stored as FP8 `e4m3` (`Tensor::FP8_E4M3` in [`Linear.cpp`](src/Linear.cpp:96)):

| Scale tensor | Shape                                | Dtype     | Grouping in K |
|---|---|---|---|
| `wscales`    | `[in_features / 16, out_features]`   | `FP8_E4M3`| 1 scale / 16 K |
| `ascales` (FP4 path) | `[K / 16, M]` (transposed)  | `FP8_E4M3`| 1 scale / 16 K |

Note the key contrast with the INT4 path:

```cpp
if (use_fp4) {
    this->wscales = Tensor::allocate({in_features_pad / 16, out_features_pad}, FP8_E4M3, ...);
} else {
    this->wscales = Tensor::allocate({in_features_pad / 64, out_features_pad}, dtype, ...);
}
```

i.e. FP4 uses **4x more scales** (one per 16-K group vs. one per 64-K group)
because micro-block size = 16.

The per-warp register layout is `packed_amscale_t` / `packed_wmscale_t`:

```cpp
struct packed_wmscale_t { uint32_t data[WMSCALES_PACK_SIZE]; }; // each uint32 = 4 FP8 scales
struct packed_amscale_t { uint32_t data[AMSCALES_PACK_SIZE]; };
```

Each `uint32_t` packs **4 FP8 e4m3 scales** = the 4 micro-blocks that the
`m16n8k64` Tensor-Core consumes per MMA, exactly matching the
`scale_vec::4X` modifier (see Section 4).

The kernel commentary at [`gemm_w4a4.cuh`](src/kernels/zgemm/gemm_w4a4.cuh:25):

```cpp
// micro-scales for FP4 MMA
// each uint32_t is a 4*32 matrix of scales (for MMA of 64*32)
```

### 3.5 Block-Info, swizzles, LoRA

Block coordinates are stored in `BlockInfo` (`bm`, `bn`, `numBlocksM`,
`numBlocksN`, with optional XY swap for L2-friendly tiling). LoRA up/down ranks
are folded into the same kernel via `EpilogueLoraDown` and `lora_up` arguments;
the LoRA branch reuses *the same* FP4 quantization step for fused LoRA-down
output (`quantize_w4a4_fuse_lora_kernel<use_fp4=true>` at
[`gemm_w4a4.cuh`](src/kernels/zgemm/gemm_w4a4.cuh:1097)).

---

## 4. The core MMA instruction: `mma.sync ... mxf4nvf4.block_scale`

### 4.1 PTX (current implementation)

The FP4 MMA is a single Blackwell-class instruction emitted with `asm volatile`
in [`mma_fp4`](src/kernels/zgemm/gemm_w4a4.cuh:191):

```ptx
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
       .f32.e2m1.e2m1.f32.ue4m3
       {%0, %1, %2, %3},        // D :  4x f32 (accumulator out)
       {%4, %5, %6, %7},        // A :  uint4 packed FP4 (m16xk64 = 32 lanes x 16B)
       {%8, %9},                // B :  2x uint32 packed FP4 (n8xk64)
       {%10,%11,%12,%13},       // C :  4x f32 (accumulator in)
       {%14}, {%15,%16},        // amscale (1 reg) + selector (2 imm)
       {%17}, {%18,%19};        // wmscale (1 reg) + selector (2 imm)
```

Decoding the modifier string:

| Token                   | Meaning                                                                          |
|---|---|
| `m16n8k64`              | Tile shape: 16 (M) x 8 (N) x 64 (K) per warp issue                              |
| `row.col`               | A row-major, B column-major (standard CUTLASS layout)                            |
| `kind::mxf4nvf4`        | "MX-FP4 / NV-FP4"; both A and B are 4-bit FP `e2m1` with per-block FP8 scales    |
| `block_scale`           | Use per-block scale operands (next groups of operands)                           |
| `scale_vec::4X`         | Each scale register provides **4** scales (one per 16-K micro-block, x4 = 64 K) |
| `.f32`                  | C / D accumulator dtype = FP32                                                   |
| `.e2m1.e2m1`            | A and B element dtype = FP4 `e2m1`                                               |
| `.f32`                  | Accumulator dtype repeated (PTX syntax)                                          |
| `.ue4m3`                | Scale element dtype = FP8 `e4m3` (unsigned interpretation for MX/NVFP4)          |

The two extra `{}, {imm, imm}` groups specify which lanes within the scale
register feed which K-sub-block. In the kernel these are populated by `ida` /
`idb`:

```cpp
// ida, idb in {0, 1}, encoded as (idb*2) and (idb*2+1) for the two halves of N
"{%17}, {%18, %19};"
...
: "r"(wmscale), "n"(0), "h"((short)(idb * 2 + 1))
```

Because one MMA covers 64 K but a micro-block is 16 K, four scales are needed
per matrix per MMA: those four are packed into one `uint32_t` (`scale_vec::4X`).

Two `m16n8k64` issues are stacked to cover `m16n16k64` (one MMA per N half).

### 4.2 Equivalent math computed by the instruction

For `ida = idb = 0` (one quarter), letting `i = 0..15`, `j = 0..7`,
`k = 0..63`, the hardware effectively computes

```
g = k >> 4                                 // micro-block index, 0..3
D[i,j] = C[i,j]
       + sum_{k=0..63}  dequant(A[i,k])  *  dequant(B[k,j])
       =       C[i,j]
       + sum_{g=0..3}
            s_A[i, g] * s_B[g, j] *
            sum_{k=16g..16g+15} fp4_to_f32(A[i,k]) * fp4_to_f32(B[k,j])
```

where `s_A`, `s_B` are FP8-`e4m3`-decoded scales delivered through the scale
registers. The accumulators are pure FP32 and are read/written as `{%0..%3}`
and `{%10..%13}`.

### 4.3 Warp scheduling around the MMA

The `compute_fp4` driver at [`gemm_w4a4.cuh`](src/kernels/zgemm/gemm_w4a4.cuh:253)
issues `WARP_M_TILES x WARP_N_TILES = 2 x 8 = 16` MMAs per K-stage, indexing
into the packed scale registers as:

```cpp
amscale[i / 2 / AMSCALES_PACK_SIZE].data[i / 2 % AMSCALES_PACK_SIZE]
wmscale[j / 2 / WMSCALES_PACK_SIZE].data[j / 2 % WMSCALES_PACK_SIZE]
```

The `/2` is because each `uint32_t` scale register feeds **two M-tiles** (the
8-row x 4-scale layout described in the comment block in
[`gemm_w4a4.cuh`](src/kernels/zgemm/gemm_w4a4.cuh:144)).

A double-buffered K loop (`NUM_STAGES = 2`) prefetches the next K-stage while
the current one issues MMAs:

```cpp
for (int k1 = 0; k1 < K / WARP_K; k1 += NUM_STAGES) {
  for (int k2 = 0; k2 < NUM_STAGES; k2++) {
      int nextk = k1 + k2 + NUM_STAGES - 1;
      int idx   = (k2 + NUM_STAGES - 1) % NUM_STAGES;
      load_act(act, nextk, K, A[idx], pred);
      load_wgt(wgt, nextk, K, W[idx], pred);
      load_amscale(ascales, nextk, amscale[idx], pred);
      load_wmscale(wscales, nextk, wmscale[idx], pred);
      compute_fp4(A[k2], W[k2], amscale[k2], wmscale[k2], fpsum);
  }
}
```

This is a classic register-resident, software-pipelined GEMM. There is no
shared memory used for the operands (other than for ldmatrix-style fragment
loads done inside `load_act` / `load_wgt`).

---

## 5. Supporting PTX conversion / quantization instructions

The kernel uses several special PTX conversions to produce the FP4 / FP8 values
exactly the way the Tensor Core expects them.

### 5.1 FP32 -> FP4 `e2m1` (2 values packed into 1 byte)

In [`gemm_utils.cuh`](src/kernels/zgemm/gemm_utils.cuh:239):

```cpp
__device__ __forceinline__ uint32_t quantize_float2_fp4(float2 value) {
    uint32_t result;
    asm volatile(
        "{ .reg .b8 tmp; "
        "  cvt.rn.satfinite.e2m1x2.f32 tmp, %1, %2; "
        "  cvt.u32.u8 %0, tmp; }"
        : "=r"(result) : "f"(value.y), "f"(value.x));
    return result;
}
```

`cvt.rn.satfinite.e2m1x2.f32` converts two FP32 values to two packed FP4 `e2m1`
in one byte, with round-to-nearest-even and saturation to the finite range.

### 5.2 FP32 -> FP8 `e4m3` (2 or 4 values packed)

In [`gemm_utils.cuh`](src/kernels/zgemm/gemm_utils.cuh:247):

```cpp
__device__ __forceinline__ uint32_t quantize_float4_fp8(float4 value) {
    uint16_t lo, hi;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                 : "=h"(lo) : "f"(value.y), "f"(value.x));
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                 : "=h"(hi) : "f"(value.w), "f"(value.z));
    return uint32_t(lo) | (uint32_t(hi) << 16);
}
```

This produces a 4-FP8 packed scale register exactly in the layout demanded by
`scale_vec::4X`.

### 5.3 FP32 reciprocal (for `1/s_g`)

In [`gemm_utils.cuh`](src/kernels/zgemm/gemm_utils.cuh:260):

```cpp
__device__ __forceinline__ static float cuda_frcp(float x) {
    float result;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}
```

Used both at quantization time (`rscale = 1 / scale`) and in inline fusion
paths.

### 5.4 Warp shuffles for cross-lane max/share

The intra-warp 4-lane reduction for `max_abs` and the broadcast of the packed
qresult use `__shfl_xor_sync` (`shfl.sync.bfly.b32`), e.g.
[`gemm_w4a4.cuh`](src/kernels/zgemm/gemm_w4a4.cuh:121):

```cpp
for (int mask = 2; mask > 0; mask /= 2) {
    for (int i = 0; i < NUM_GROUPS; i++) {
        maxvalue[0][i] = __hmax(maxvalue[0][i], __shfl_xor_sync(~0, maxvalue[0][i], mask));
        maxvalue[1][i] = __hmax(maxvalue[1][i], __shfl_xor_sync(~0, maxvalue[1][i], mask));
    }
}
```

This reduces across the 4 lanes of a `quad` (matching the m16-n8 fragment row
mapping).

### 5.5 `ldmatrix` for fragment loading

Used by the non-FP4 quantize path; the FP4 path skips shared memory and uses
direct global loads via `load_pred` (see `load_amscale` /
[`gemm_w4a4.cuh`](src/kernels/zgemm/gemm_w4a4.cuh:64-83)).

---

## 6. Capability gating and traps

Compile-time and runtime gating of the FP4 path is done with:

```cpp
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    static constexpr bool FP4_AVAILABLE = true;
#else
    static constexpr bool FP4_AVAILABLE = false;
#endif

__device__ __forceinline__ static void trap_no_fp4() {
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
        printf("FP4 is not available on this device\n");
    __syncthreads();
    __nanosleep(1000000);
    __trap();
}
```

(from [`gemm_w4a4.cuh`](src/kernels/zgemm/gemm_w4a4.cuh:28))

`MIN_ARCH = 1200` (SM 12.0, Blackwell) is enforced on the FP4 kernel struct:

```cpp
template<typename Epilogue, bool USE_ALPHA>
struct gemm_w4a4_fp4_kernel {
    static constexpr int MIN_ARCH = 1200;
    ...
};
```

---

## 7. Putting it together: end-to-end image-generation step

For a typical FLUX / SD-class diffusion forward pass:

1. **Linear-layer weights** are pre-quantized offline to NVFP4 layout:
   - `qweight ∈ INT8[N_pad, K_pad / 2]` (two FP4 per byte).
   - `wscales ∈ FP8_E4M3[K_pad / 16, N_pad]`.
   - `wtscale ∈ FP32` (per-tensor `alpha`).
2. **Activations** start as BF16/FP16. The pre-layer
   [`quantize_w4a4_act_fuse_lora`](src/kernels/zgemm/gemm_w4a4_launch.cuh:51)
   kernel:
   - reads the BF16 activation,
   - computes per-16-element max-abs,
   - emits `q_act ∈ INT8[M, K_pad/2]` and `ascales ∈ FP8_E4M3[K_pad/16, M]`,
   - simultaneously runs the LoRA-down GEMM into FP32 (fused).
3. **W4A4 GEMM** is launched via
   [`GEMM_W4A4_Launch<Config, true>::gemm_w4a4`](src/kernels/zgemm/gemm_w4a4_launch.cuh:22),
   which dispatches `gemm_w4a4_fp4_kernel<Epilogue, USE_ALPHA>` over a
   `(M/BLOCK_M, N/BLOCK_N)` grid with 8 warps per CTA. Inside each CTA:
   - prefetch `(act, wgt, amscale, wmscale)` into registers (2 K-stages),
   - issue `m16n8k64` `mxf4nvf4.block_scale.scale_vec::4X` MMAs into FP32
     accumulators,
   - optionally scale by `alpha`,
   - down-cast FP32 -> FP16/BF16 via `packed_fp32_to_fp16`.
4. **Epilogue** runs one of several `Epilogues<Config>::*` flavors -
   plain GEMM, GEMM + bias + GELU + re-quantize for next layer (via
   `EpilogueQuantize<FUSE_GELU, USE_UNSIGNED=false, USE_FP4=true>`), GEMM +
   LoRA-up, or attention QKV packing.
5. **Re-quantize on the way out**: `EpilogueQuantize<…, USE_FP4=true>` at
   [`gemm_w4a4.cuh`](src/kernels/zgemm/gemm_w4a4.cuh:930) calls
   `quantize_w4a4_fp4_from_fpsum_warp` which:
   - keeps the result in registers (no shared-mem round-trip),
   - computes per-16-K max-abs across the warp,
   - emits new `q_act ∈ INT8` + `ascales ∈ FP8_E4M3`,
   - writes them at the location consumed by the *next* layer's FP4 MMA -
     end-to-end fusion across consecutive W4A4-FP4 layers.

Steps 3-5 are inside the same kernel launch when the next layer is also FP4,
which is the common case in the diffusion transformer blocks.

---

## 8. Implementation gotchas validated by `tests/kernels/`

These notes were uncovered while building the test suite under
[`tests/kernels/`](tests/kernels/) and validated against the SM 12.0
reference implementation. A future port must respect them or its behavior
will diverge from the reference.

### 8.0 FP4 weight quantization is offline-only

The runtime kernel `quantize_w4a4_wgt` is **compiled out for the FP4 path**
(`assert(false)` in
[`gemm_w4a4_launch_impl.cuh:549`](src/kernels/zgemm/gemm_w4a4_launch_impl.cuh:549)).
For NVFP4 models the `qweight + wscales + wtscale + wcscales` buffers are
written by the offline checkpoint builder (`src/Linear.cpp` / the export
tools in `scripts/`), not by any runtime call. The runtime only quantizes
activations.

### 8.1 `ascales` storage is warp-interleaved, not row-major

The PyTorch shape of `ascales` is `[K/16, M_pad]` but the byte layout is
the warp-interleaved storage that the FP4 Tensor Core MMA expects:

    layout = [M / BLOCK_M, K / 16, NUM_WARPS, AMSCALES_NUM_PACKS, AMSCALES_VALID_LANES]
             of packed_amscale_t  (each packed_amscale_t = 4 FP8 e4m3 scales)

so the FP8 scale assigned to activation row `r` and K-group `g` lives at
flat offset

    bm           = r // 256
    r_in_block   = r %  256
    warp_id      = r_in_block // 32
    r_in_warp    = r_in_block %  32
    lane_id      = (r_in_warp % 8) * 4 + (r_in_warp // 8)
    bk           = g // 4
    pack_lane    = g %  4
    flat_offset  = bm * (K/16) * BLOCK_M
                 + (bk * NUM_WARPS + warp_id) * 128
                 + lane_id * 4 + pack_lane

The lane permutation `(r%8)*4 + r//8` matches the `m16k64` MMA fragment
row-to-lane mapping. The full Python decoder is in
[`tests/kernels/_helpers.py`](tests/kernels/_helpers.py:1) (`fp4_decode_ascales`).
A port that emits a row-major `[K/16, M]` tensor will produce checkpoints
that the reference GEMM rejects silently.

### 8.2 `smooth_factor` is fragment-interleaved (packed_wscale_t)

The `EpilogueQuantize` smooth-fold (lines 990-993 of
[`gemm_w4a4.cuh`](src/kernels/zgemm/gemm_w4a4.cuh:990)) loads
`smooth_factor` via `load_wscale(..., packed_wscale_t)` which expects the
fragment-interleaved layout described at
[`gemm_base.cuh:123-129`](src/kernels/zgemm/gemm_base.cuh:123); it is
**not** a plain `[N]` 1D tensor in channel order. Passing a flat
`[N]` BF16 tensor (as the public Python wrapper currently does) only works
when the smooth values are uniform across channels.

### 8.3 `quantize_w4a4_act_fuse_lora` requires non-null `smooth`

Passing `smooth_factor = None` (a `Tensor::valid() == false` tensor in
C++) into the FP4 path triggers an illegal memory access in the
load_wscale call inside `EpilogueQuantize`. Until that branch is fixed,
callers must always pass a `smooth_factor` of `1.0`-filled BF16, which is
the algebraic identity. (The Python wrapper at
[`quantize.py:80`](nunchaku/ops/quantize.py:80) accepts `smooth=None`,
which is the source of the bug.)

### 8.4 Zero-input group encodes as `q=+6, s=0`, not `q=0, s=0`

For an all-zero K-group of activations, the reference kernel writes:

  * `oscales[g, m] = 0.0`  (FP8 e4m3 zero)
  * `act[m, k]   = 0x77 = (+6, +6)`  (two saturated FP4 values)

because the kernel computes the dequantize factor as
`rcp.approx.ftz.f32(scale)`, which returns `+inf` for `scale = 0`; then
`0.0 * inf -> NaN`, then `cvt.rn.satfinite.e2m1x2.f32(NaN) -> +6`.

End-to-end the dequantized value is still exactly `0`, because the per-group
scale dominates the product, but a port that writes `q = 0` for the zero
group will produce **byte-different** but algebraically-equivalent
checkpoints. Any cross-implementation byte-equality test (§6 of the test
plan) must take this into account.

### 8.5 1-ULP non-determinism on some shapes

For M values that are not multiples of `BLOCK_M = 256`, and for the larger
diffusion-realistic shapes, the kernel can produce BF16 outputs that
differ by **at most one BF16 ULP** across repeated calls with identical
inputs. This is caused by the atomic-add path inside the LoRA-down
reduction (`reduce_lora_act` in
[`lora.cuh:82`](src/kernels/zgemm/lora.cuh:82)) reordering accumulations
across CTAs. Tests in [`tests/kernels/`](tests/kernels/) treat this as a
known property and use a "1-ULP-close" comparison rather than
`torch.equal` for those shapes. Per
[`MATH_TEST_PLAN.md`](MATH_TEST_PLAN.md) §0.4 this is recorded as a
deliberate, non-correctness behavior of the reference; a TMEM port should
either preserve it or document the change.

---

## 9. Numerical-fidelity notes

- **FP32 accumulation** is mandatory: the kernel always uses
  `packed_f32psum_t` for FP4 (`f32psum_warp fpsum` in
  [`gemm_w4a4.cuh`](src/kernels/zgemm/gemm_w4a4.cuh:294)). The INT4 path can
  optionally use FP16 accumulation via `USE_FP32_ACCUM = false`.
- **Saturation** is `satfinite` everywhere - both during
  `cvt.satfinite.e2m1x2.f32` and `cvt.satfinite.e4m3x2.f32`. This avoids
  Inf/NaN propagation when an activation has outliers; the side effect is that
  any `|x| > 6 * 448 = 2688` is clipped at quantization time. The optional
  `wtscale` (`alpha`) shifts this clip range globally.
- **Rounding** is round-to-nearest-even (`cvt.rn.*`) for both FP4 and FP8
  conversion, matching the OCP MXFP4 / NVFP4 reference math.
- **Reciprocal scale** uses `rcp.approx.ftz.f32` instead of an IEEE division.
  This is ~1 ULP off in the mantissa, which is below FP4 quantization noise.
- **Optional NaN-check** path (`#define ENABLE_NAN_CHECK 1`) instruments every
  accumulator with `checkNan` (`isfinite` + `__trap()`); disabled in release
  builds.
- **Smooth-quant fold**: each layer carries a `smooth_factor` (per-N FP16/BF16
  vector). The `EpilogueQuantize` divides the FP16 result by the
  `smooth_factor` *before* FP4 quantization (lines 990-993). This compensates
  for the inverse multiply that the next layer's weight has absorbed at
  calibration time. It is mathematically `(x / s_smooth) · (W · diag(s_smooth))`
  but executed for free in registers.

---

## 10. Inventory of CUDA / PTX instructions touched by the FP4 path

| Instruction                                                          | Purpose                                          | Cited line |
|---|---|---|
| `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3` | The actual FP4 Tensor-Core MMA | [`gemm_w4a4.cuh:200`](src/kernels/zgemm/gemm_w4a4.cuh:200) |
| `cvt.rn.satfinite.e2m1x2.f32`                                        | FP32 -> 2x FP4 `e2m1` pack                       | [`gemm_utils.cuh:241`](src/kernels/zgemm/gemm_utils.cuh:241) |
| `cvt.rn.satfinite.e4m3x2.f32`                                        | FP32 -> 2x FP8 `e4m3` pack (block-scale)         | [`gemm_utils.cuh:249`](src/kernels/zgemm/gemm_utils.cuh:249) |
| `cvt.u32.u8`                                                         | Zero-extend 8-bit FP4 pair to a 32-bit register  | [`gemm_utils.cuh:241`](src/kernels/zgemm/gemm_utils.cuh:241) |
| `rcp.approx.ftz.f32`                                                 | `1 / scale` reciprocal for quantize step         | [`gemm_utils.cuh:262`](src/kernels/zgemm/gemm_utils.cuh:262) |
| `shfl.sync.bfly.b32` (via `__shfl_xor_sync`)                         | Cross-lane max-abs / pack broadcast             | [`gemm_w4a4.cuh:121`](src/kernels/zgemm/gemm_w4a4.cuh:121) |
| `ldmatrix.sync` (via `ldmatrix(...)`)                                | Fragment loads (INT4 path; FP4 uses direct LDG)  | [`gemm_w4a4.cuh:619`](src/kernels/zgemm/gemm_w4a4.cuh:619) |
| `__nanosleep`, `__trap`                                              | Capability-trap when not SM 12.0+                | [`gemm_w4a4.cuh:34`](src/kernels/zgemm/gemm_w4a4.cuh:34) |
| `__habs2`, `__hmax`, `__hmul2`, `h2div`, `gelu_half2`                | Per-lane FP16/BF16 epilogue math                | various in `gemm_w4a4.cuh` |
| `int2half2_fast_512`, `int2float_fast`                               | Fast int->float dequant (not used in FP4 path)   | [`gemm_w4a4.cuh:691-719`](src/kernels/zgemm/gemm_w4a4.cuh:691) |

---

## 11. What changes for TMEM NVFP4 (future work)

The current kernel keeps **all** accumulators, activations, weights and scales
in **register memory** (`act_warp`, `wgt_warp`, `amscale_warp`, `wmscale_warp`,
`f32psum_warp`). Each warp owns a `256x128` slice of the C tile in 64 FP32
registers per lane.

The new 5th-gen Tensor-Core path on SM 12.0a moves to a different shape and
storage:

1. `mma.sync` -> **`tcgen05.mma`** (or `tcgen05.mma.async`) which writes
   accumulators directly into **tensor memory (TMEM)** rather than registers.
2. Operands stay in shared memory but are addressed via TMEM descriptors;
   scales must be loaded into TMEM with `tcgen05.cp` or similar.
3. The 4-lane "quad" packing of scales done in
   `quantize_w4a4_fp4_from_fpsum_warp` will need to be replaced by a
   TMEM-friendly layout (likely a contiguous FP8 `e4m3` line per micro-block,
   transposed for B).
4. The two-stage software pipeline (`NUM_STAGES = 2`) can be replaced by
   `cp.async.bulk.tensor` + `mbarrier` for asynchronous A/B/scale fetches.
5. `EpilogueQuantize` will need a TMEM->register download (`tcgen05.ld`) before
   it can do the FP16-domain smooth-fold + re-quantize step. Alternatively the
   epilogue can be re-expressed entirely on TMEM-resident accumulators.

The math from sections 1-2 is unchanged - **only the data movement and the
exact PTX form of the MMA differ**. In particular:

- The element formats stay `e2m1` (FP4) and `e4m3` (FP8 scale).
- The micro-block size stays 16.
- The per-tensor `alpha` stays an FP32 multiply.
- The `scale_vec::4X` semantics carry over; on TMEM-class hardware it becomes
  `tcgen05.mma.kind::mxf4nvf4` with the equivalent `scale_vec` modifier.

A port should therefore concentrate on:

1. A new tile shape (`m64n128k64`, or `m128n128k64`) consistent with
   `tcgen05.mma`.
2. A TMEM allocator and descriptor layer in [`Tensor.h`](src/Tensor.h) /
   [`gemm_base.cuh`](src/kernels/zgemm/gemm_base.cuh).
3. A new `mma_fp4_tmem(...)` wrapper paralleling the current `mma_fp4(...)`.
4. Rewriting `compute_fp4` / `gemm_w4a4_fp4_block` to drive `tcgen05.mma.async`
   loops with `mbarrier` synchronization.
5. Adapting `EpilogueQuantize<…, USE_FP4=true>` to read from TMEM and write
   the same on-disk layout (so model files stay binary-compatible).

Items 1-3 are mechanical; items 4-5 are where most of the
TMEM-architecture-specific reasoning will live.