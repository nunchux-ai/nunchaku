# Triton as a Replacement for Hand-Coded TMEM NVFP4 Kernels

This note evaluates whether [Triton](../triton) (specifically the in-tree
checkout at [`packages/triton`](../triton)) can be used to implement the
sm_110 / sm_120 NVFP4 W4A4 math described in
[`MATH.md`](MATH.md), so we can skip hand-writing `tcgen05.mma.kind::mxf4nvf4`
PTX kernels.

The short answer is **yes for the sm_100 / sm_103 / sm_110 (TMEM) target, with
caveats for sm_120 consumer Blackwell, and the work primarily reduces to weight
re-packing at checkpoint-load time plus a handful of Triton kernels**.

The **primary target workload is diffusion transformers** like
[`models/nunchaku-z-image-turbo`](../../models/nunchaku-z-image-turbo)
(Z-Image-Turbo, ~6B params), FLUX.1-dev (~12B), Sana, QwenImage. See
[`src/imagegen/ZImageTurboBackend.py`](../../src/imagegen/ZImageTurboBackend.py)
and [`OmniImageEditServer.py`](../../src/imagegen/OmniImageEditServer.py)
for how they get loaded; see
[`nunchaku/models/linear.py`](nunchaku/models/linear.py) and the per-arch
files under [`nunchaku/models/transformers/`](nunchaku/models/transformers/)
for the actual call sites (`SVDQW4A4Linear`,
`NunchakuZImageTransformer2DModel`, etc.). This use case differs in
important ways from a 172B MoE LLM and **simplifies the Triton port
considerably**; see §9 below.

All references below to Triton files are inside
[`packages/triton`](../triton); references to Nunchaku files are inside
[`packages/nunchaku`](.).

---

## 1. What MATH.md actually requires from a kernel author

From [`MATH.md`](MATH.md) §§1-4 and §11, an sm_110 port must produce the
following pieces of math, in this order, every forward pass:

1. **BF16/FP16 -> NVFP4 activation quantization** (with fused LoRA-down
   matmul):
   - per-row group of 16 K, `s_g = min(max_abs(x_g)/6, 448)` in FP8 e4m3;
   - `q_i = cvt.rn.satfinite.e2m1x2.f32(x_i / s_g)` packed 2-per-byte;
   - simultaneous BF16 GEMM into the LoRA-down rank-R buffer;
   - output `ascales` in the fragment-permuted layout described in
     [`MATH.md`](MATH.md) §8.1.
2. **FP4 x FP4 block-scaled MMA into FP32**, with one FP8 `e4m3` micro-scale
   per 16 K-elements on each operand (`scale_vec::4X`), then multiplied by an
   optional FP32 per-tensor `alpha`.
3. **Epilogue**: bias add, optional GELU/SiLU, optional LoRA-up GEMM (FP32 in,
   BF16 out), optional smooth-fold divide, optional **re-quantize back to
   NVFP4** for the next layer.
4. **Numerical contract** preserved bit-for-bit where the reference is bit-
   exact, with the documented loosenings (§§8.4, 8.5).

The PTX form of (2) is the only piece that the existing CUDA implementation
expresses directly; (1) and (3) are warp-cooperative kernels over FP32/FP16
tensors plus a few `cvt.satfinite.*` instructions.

---

## 2. What Triton already supports natively

### 2.1 The block-scaled MMA itself

Triton emits exactly the same PTX family that
[`MATH.md`](MATH.md) §4.1 documents:

| Triton path | Generated PTX | File |
|---|---|---|
| `tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", ...)` on **sm_120 / sm_121** | `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3` | [`MMAv2.cpp`](../triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp:507) |
| `tcgen05_mma_scaled(a_smem, b_smem.permute((1,0)), acc_tmem, a_scale_tmem, b_scale_tmem, "e2m1", "e2m1", ...)` on **sm_100 / sm_103 / sm_110** | `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X` | [`MMAv5.cpp`](../triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp:300) |

The dispatch is driven by `getMMAVersionSafe(computeCapability, op)` in
[`AccelerateMatmul.cpp:42`](../triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:42),
which is exactly the table we need:

```cpp
if (computeCapability < 100) versionsSupported = {3, 2};   // Hopper
else if (computeCapability < 120) versionsSupported = {5, 2}; // sm_100 / sm_103 / sm_110 -> tcgen05
else if (computeCapability < 130) versionsSupported = {2};    // sm_120 / sm_121 -> MMAv2 only
```

So:

* **sm_110 (datacenter Blackwell, TMEM)**: Triton automatically picks `tcgen05.mma`
  v5 - the very instruction `MATH.md` §11 says we'd otherwise have to hand-emit.
  No special opt-in required; just pass `tl.dot_scaled(...)` or
  `tcgen05_mma_scaled(...)` from Gluon. There's no "sm_110-only" gating
  anywhere in Triton's NVIDIA backend - it falls into the `< 120` bucket the
  same way sm_100/103 do.
* **sm_120 (consumer Blackwell, no TMEM)**: Triton still emits `mxf4nvf4`
  block-scaled MMA but uses the **register-resident** MMAv2 form. This is
  functionally what the current `gemm_w4a4_fp4_kernel` in
  [`gemm_w4a4.cuh`](src/kernels/zgemm/gemm_w4a4.cuh:191) does today.

### 2.2 Per-block FP4 quantization

Both halves of the activation-quantization step in §1.1 are available as
ready-to-use Triton kernels:

* [`_quantize_nvfp4_fn`](../triton/python/triton_kernels/triton_kernels/numerics_details/mxfp_details/_downcast_to_mxfp.py:228)
  - takes BF16/FP32 input, computes per-16 max-abs, divides, and packs
  two `e2m1` per byte with `cvt.rn.satfinite.e2m1x2.f32`. Emits FP8 `e4m3`
  scales.
* [`_quantize_mxfp4_fn`](../triton/python/triton_kernels/triton_kernels/numerics_details/mxfp_details/_downcast_to_mxfp.py:223)
  - same but FP8 `e8m0` scales (MXFP4; **not** what we want).
* [`_upcast_from_mxfp`](../triton/python/triton_kernels/triton_kernels/numerics_details/mxfp_details/_upcast_from_mxfp.py:1)
  - reverse, uses `cvt.rn.f16x2.e2m1x2` for the FP4 -> FP16 step.

These are the exact PTX conversion instructions listed in
[`MATH.md`](MATH.md) §5.1-5.2, so the **numerical** behaviour of activation
quantization will already match `quantize_w4a4_act_fuse_lora` up to rounding
and the warp-shuffle reduction tree.

### 2.3 End-to-end reference: the block-scaled matmul tutorials

We have two production-quality, tested reference implementations to copy from:

* [`10-block-scaled-matmul.py`](../triton/python/tutorials/10-block-scaled-matmul.py)
  - generic `tl.dot_scaled` matmul that handles `nvfp4`, `mxfp4`, `mxfp8`, and
  mixed precision; emits both `tcgen05.mma.kind::mxf4nvf4.block_scale` and
  `mma.sync.m16n8k32.kind::mxf8f6f4.block_scale` depending on capability.
* [`11-tcgen05-mma-scaled.py`](../triton/python/tutorials/gluon/11-tcgen05-mma-scaled.py)
  - low-level Gluon tutorial showing software-pipelined and warp-specialized
  NVFP4 matmul on TMEM, including the `TensorMemoryScalesLayout`, `tcgen05_copy`
  pipelining, and `swizzle_scales_packed_block` for HBM-layout scales.

Performance numbers in the tutorial header for sm_100 (warp-specialized,
pipelined, `nvfp4 x nvfp4`, 8192^3): **~4847 TFLOPs**, which is in the same
ballpark as cuBLAS on the same hardware. That is far above what the current
sm_120 register-resident kernel can sustain.

### 2.4 Layout helpers we get for free

The `triton_kernels` package already includes:

* [`make_default_matmul_mxfp4_w_layout`](../triton/python/triton_kernels/triton_kernels/tensor_details/layout.py:27)
  / `make_default_matmul_mxfp4_w_scale_layout` for weight and scale HBM
  swizzling, with capability-aware variants (`BlackwellMXScaleLayout`,
  `BlackwellMXValueLayout`, `BlackwellMX4ValueShuffledLayout`).
* `TensorMemoryScalesLayout` and `tcgen05_copy` for staging scales into TMEM
  on Blackwell/sm_110.
* `swizzle_scales_packed_block` / `unswizzle_scales_shared_memory` (in the
  block-scaled tutorial) for the 5-D `(M/128, K/16/4, 32, 4, 4)` layout that
  matches the PTX
  [tcgen05-mma-scale-factor-a-layout-1x](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x)
  spec.

None of these need to be re-derived; we just feed our data into them.

---

## 3. The mismatch: Nunchaku's scale layout is custom

Triton's NVFP4 scale layouts are **not byte-compatible** with Nunchaku's
on-disk layout. Specifically:

* Nunchaku stores `ascales` as
  `[M/256, K/16, NUM_WARPS=8, AMSCALES_NUM_PACKS, AMSCALES_VALID_LANES]` with
  the lane permutation `(r%8)*4 + r//8` baked into the byte order
  ([`MATH.md`](MATH.md) §8.1). This is what `mma.sync.m16n8k64` consumes
  directly through its scale operand registers.
* Triton's NVFP4 scales (in HBM) are
  `[M/128, K/(16*4), 32, 4, 4]` of `float8_e4m3fn` for the
  `tcgen05.mma.scale_vec::4X` path, transformed by `swizzle_scales_packed_block`.

Similarly the weight layouts differ:

* Nunchaku: `INT8[N_pad, K_pad/2]` with the `std::swap(tmpout.y, tmpout.z)`
  fragment-order swap ([`MATH.md`](MATH.md) §3.3).
* Triton (Blackwell-MX): `BlackwellMX4ValueShuffledLayout` for shuffled FP4
  weights, or the simpler `BlackwellMXValueLayout`.

This is the **single largest cost** of a Triton port: not the math, but
re-expressing the on-disk checkpoint layout.

---

## 4. Repacking strategy (the high-leverage idea)

The model checkpoints
([`models/MiniMax-M2.7-REAP-172B-A10B-NVFP4`](../../models/MiniMax-M2.7-REAP-172B-A10B-NVFP4)
and similar) store the FP4 weights and FP8 scales in Nunchaku's custom
fragment-permuted format. Two implementation options:

### 4.1 Option A: load-time repack (recommended)

Add a Python load hook (in
[`nunchaku/models/`](nunchaku/models/) or in the safetensors loader at
[`merge_safetensors.py`](nunchaku/merge_safetensors.py)) that, when the
target capability is sm_110 or sm_120, decodes Nunchaku's fragment-permuted
`(qweight, wscales, wtscale)` into the layout the Triton kernel expects.

Concretely:

1. Use the existing Python decoder
   [`tests/kernels/_helpers.py`](tests/kernels/_helpers.py:1) (`fp4_decode_ascales`,
   `fp4_decode_wscales`) to recover the logical `[N, K/16]` and `[N, K/2]`
   tensors. These already exist and are tested against the reference
   CUDA kernel.
2. Apply Triton's layout conversion via
   `convert_layout(wrap_torch_tensor(w, dtype=FP4), value_layout)` and
   `convert_layout(wrap_torch_tensor(wscale, dtype=UINT8), scale_layout)`
   from
   [`triton_kernels.tensor`](../triton/python/triton_kernels/triton_kernels/tensor.py).
3. Cache the repacked tensors on disk on first use (mtime-keyed sidecar in
   `~/.cache/nunchaku/nvfp4-tmem/`) so the cost is amortized.

**Pros:**
- Zero changes to the on-disk checkpoint format -> no breakage of existing
  models or downstream tooling.
- Repacking is pure PyTorch and runs once on `model.load()`, so we don't
  pay it on every forward pass.
- The reference CUDA kernel ([`gemm_w4a4_fp4_kernel`](src/kernels/zgemm/gemm_w4a4.cuh:279))
  can stay as the byte-equality oracle in tests.

**Cons:**
- We hold two copies of the weights in host RAM during the conversion. For
  a 172 B model in NVFP4 that's ~86 GB twice; we'd need to convert one
  safetensors shard at a time.
- Activation `ascales` also has a fragment-permuted layout. Either we
  repack them every forward pass (small, but a kernel) or we restructure
  `quantize_w4a4_act_fuse_lora` to write the Triton-style layout
  directly.

### 4.2 Option B: rewrite the activation quantizer + GEMM both in Triton

Replace both
[`quantize_w4a4_act_fuse_lora`](src/kernels/zgemm/gemm_w4a4_launch.cuh:51)
and
[`gemm_w4a4_fp4_kernel`](src/kernels/zgemm/gemm_w4a4.cuh:279)
with two Triton kernels:

1. `nunchaku_quantize_nvfp4_fuse_lora.py` - thin wrapper around
   `_quantize_nvfp4_fn` plus a tiled BF16 LoRA-down GEMM
   (`tl.dot`). Writes activations + scales in **the layout that the
   matching Triton GEMM expects**, not the legacy fragment-permuted layout.
2. `nunchaku_w4a4_fp4_gemm.py` - a copy of
   [`tutorials/10-block-scaled-matmul.py`](../triton/python/tutorials/10-block-scaled-matmul.py)
   plus a fused bias/GELU/LoRA-up epilogue, optional `qout` re-quantize
   epilogue, and `wtscale` (`alpha`) multiply.

The legacy CUDA kernels stay compiled in and remain the sm_100/sm_120
fallback (and the test oracle).

**Pros:**
- We never have to massage Nunchaku's old fragment-permuted layout on the hot
  path - it's an offline weight repack only.
- A single source of truth for the algorithm (in Python), much easier to
  maintain than 1500 lines of `gemm_w4a4.cuh`.
- Free port to sm_120 (consumer) via the same code path; just falls back to
  MMAv2.

**Cons:**
- We have to re-implement the LoRA fusion epilogue inside the Triton GEMM.
  That's non-trivial - the current CUDA epilogue handles up to 6 distinct
  output paths (plain, GELU, LoRA-up, attention QKV pack, re-quantize, etc.)
  See `Epilogues<Config>` in [`gemm_w4a4.cuh`](src/kernels/zgemm/gemm_w4a4.cuh).
- The re-quantize epilogue (§4 step 5) has fragment-level cooperation across
  the warp; rewriting it as a Triton epilogue while keeping the same
  numerical contract (especially §8.4 zero-input encoding) is finicky.

### 4.3 Recommended path: hybrid

1. **Offline-repack the weights** (one-time, Option A) so the GEMM consumes
   Triton-style scale and value layouts.
2. **Rewrite activation quant + W4A4 GEMM in Triton** (Option B) but in
   isolation - each linear layer in the network calls the same two
   Triton kernels. Make `ascales` use Triton's layout natively; the
   legacy fragment-permuted layout never appears in the new code path.
3. **Keep the legacy CUDA kernels** as the validation oracle for the
   `tests/kernels/` suite ([`MATH_TEST_PLAN.md`](MATH_TEST_PLAN.md))
   and as a fallback for older GPUs (sm_89, sm_90 Hopper).
4. **Skip §8.4 byte-equality** for the new path - the test plan already
   acknowledges this is a non-portable property. The Triton path produces
   the same dequantized value but not the same packed bits.

The §8.5 1-ULP non-determinism property in turn becomes irrelevant - Triton's
split-k path uses deterministic accumulation by default.

---

## 5. Concrete coverage table

For each piece of NVFP4 math in `MATH.md`, what does Triton give us?

| MATH.md section | Math piece | Triton support | Effort to use |
|---|---|---|---|
| §1.1, §5.1 | FP32 -> FP4 e2m1 pack | `cvt.rn.satfinite.e2m1x2.f32` already emitted by `_quantize_nvfp4_fn` | trivial - call the helper |
| §1.2, §5.2 | FP32 -> FP8 e4m3 scale | emitted by same helper | trivial |
| §1.3 | per-tensor `alpha` (`wtscale`) | scalar fp32 multiply in epilogue | trivial - `acc * alpha` in Triton |
| §2 | two-level quant | `_quantize_nvfp4_fn` does it end-to-end | trivial |
| §3 | tensor layouts | Triton has its own (better) layouts; ours need repack (§4) | **medium - one-shot offline repack** |
| §4 | `mma.sync.kind::mxf4nvf4.block_scale.scale_vec::4X` | `tl.dot_scaled` (sm_120) and `tcgen05_mma_scaled` (sm_110) | trivial - one Triton call |
| §5.3 | `rcp.approx.ftz.f32` | implicit in `_quantize_nvfp4_fn` divide; not bit-equal but within FP4 noise | trivial |
| §5.4 | warp-shuffle max-abs | Triton emits the equivalent reduction tree under `tl.max` | trivial |
| §5.5 | `ldmatrix` | Triton emits via `tl.load` + dot operand layout | automatic |
| §6 | capability gating | `cuda_capability_geq(10,0)` + `cuda_capability_geq(12,0)` | trivial |
| §7 step 2 | fused LoRA-down + quant | two `tl.dot` calls in the same kernel | **medium** - need to write the kernel |
| §7 step 4-5 | epilogues (bias/GELU/LoRA-up/re-quant) | composable in Triton; re-quant fuses `_quantize_nvfp4_fn` at the end | **medium-hard** - the fused re-quant epilogue is the trickiest part |
| §8.0 | weight quant is offline | unchanged; we just repack on load | trivial |
| §8.1-8.3 | warp-interleaved storage | irrelevant once we repack | n/a |
| §8.4 | `q=+6, s=0` zero-encoding | byte-different in new path; algebraically identical | document |
| §8.5 | 1-ULP non-determinism | absent in deterministic Triton path | document |
| §9 | numerical fidelity | same FP32 accumulation, same `satfinite`, same rounding | preserved by construction |
| §11 | TMEM port (sm_110) | **this is exactly what `tcgen05_mma_scaled` does**; we'd have written ~1000 lines of PTX otherwise | **biggest win** |

---

## 6. Why this is the right hammer for this nail

The reason Triton fits is that the only **architecturally** new thing on
sm_110 is the migration of accumulators from registers to TMEM and the
replacement of `mma.sync` with `tcgen05.mma.async`. Everything else in
`MATH.md` (the FP4 e2m1 format, the FP8 e4m3 block scales, the
`scale_vec::4X` semantics, the per-tensor alpha, the smooth-fold) carries
over unchanged. Triton encapsulates exactly that architectural delta:

* `tl.dot_scaled` is the abstract instruction.
* The compiler picks `mma.sync.m16n8k64.mxf4nvf4.block_scale.scale_vec::4X`
  on sm_120 and `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X`
  on sm_110 - no source change in the kernel.
* TMEM allocation, `tcgen05_copy` pipelining, mbarrier choreography, and
  warp specialization are all handled by Triton's
  [`tritongpu-pipeline`](../triton/lib/Dialect/TritonGPU/Transforms/) and
  TMEM passes.

In contrast, a hand-written `tcgen05.mma.async` kernel must also re-derive
the TMEM allocator, the scale-to-TMEM `tcgen05.cp` pattern, the mbarrier
phase tracking, and the 2-CTA broadcast semantics (`cta_group::2`). Triton
already has working test coverage for every one of those.

---

## 7. What this does **not** solve

Triton is not a free lunch:

1. **Byte-equality with the legacy kernel is gone.** Tests in
   [`MATH_TEST_PLAN.md`](MATH_TEST_PLAN.md) §6 that compare against
   pre-recorded golden bytes need to be relaxed to "numerically equal at
   FP4 precision". The MATH_TEST_PLAN explicitly anticipates this case
   (§0.4 marks 1-ULP differences as known-acceptable).
2. **LoRA-down fusion is non-trivial in Triton.** The current CUDA kernel
   computes the LoRA-down GEMM in the same warp that does the activation
   quantization, sharing FP32 reduction registers across both. In Triton we
   either keep them as two separate kernels (with an extra global-memory
   round-trip for the FP32 reduction), or hand-write a single Triton kernel
   that interleaves the rank-R matmul into the per-16-K reduction loop.
   The latter is doable (Triton supports two `tl.dot`s in one kernel and
   software-pipelines them) but is one of the more delicate kernels we'd
   write.
3. **The re-quantize epilogue with smooth-fold** (`USE_FP4=true` branch of
   `EpilogueQuantize` in
   [`gemm_w4a4.cuh`](src/kernels/zgemm/gemm_w4a4.cuh:930)) folds three
   things together: bias add, smooth divide, and FP4 re-quantization that
   writes the next layer's inputs. Reproducing this as a Triton epilogue
   without a TMEM->register->TMEM round-trip is possible but requires
   careful use of `tcgen05_copy` and `gl.convert_layout` (the Gluon
   tutorials in [`11-tcgen05-mma-scaled.py`](../triton/python/tutorials/gluon/11-tcgen05-mma-scaled.py)
   show roughly the right pattern). Expect a measurable performance gap
   versus the legacy kernel on this specific epilogue until it's tuned.
4. **Compile time.** Triton kernels are AOT-compiled per launch shape on
   first call. The current Nunchaku binary ships a single fat AOT-compiled
   CUDA fatbin; Triton will add a one-time per-shape JIT step (cached to
   `~/.triton/cache/`). This is invisible after warm-up but adds a few
   seconds to the first forward pass.
5. **Triton version pin.** The block-scaled MMA tutorial uses APIs that
   landed in Triton 3.4-3.6 (`tcgen05_mma_scaled`, `BlackwellMXScaleLayout`,
   `BlackwellMX4ValueShuffledLayout`). The wheel we already ship
   ([`triton-3.6.0+git565c0852-cp312-cp312-linux_aarch64.whl`](../triton/triton-3.6.0+git565c0852-cp312-cp312-linux_aarch64.whl))
   includes them, so no upstream pinning is needed.

---

## 8. Recommendation

**Use Triton.** The pieces of math in
[`MATH.md`](MATH.md) that would otherwise require hand-coded
`tcgen05.mma.kind::mxf4nvf4.block_scale.scale_vec::4X` PTX kernels for
sm_110 are exactly the pieces that Triton's `tl.dot_scaled` /
`tcgen05_mma_scaled` already produce, with working tests. The remaining
work is:

1. A one-time **offline weight-and-scale repacker** (Python, ~100 lines)
   that turns Nunchaku's fragment-permuted on-disk layout into the
   `BlackwellMXValueLayout` + `BlackwellMXScaleLayout` pair that the
   Triton kernels expect. The decoder logic is already in
   [`tests/kernels/_helpers.py`](tests/kernels/_helpers.py:1).
2. Two Triton kernels:
   - `nunchaku_quantize_nvfp4.py` - activation quantization + optional
     fused LoRA-down BF16 GEMM. Built on top of `_quantize_nvfp4_fn`.
   - `nunchaku_gemm_w4a4_fp4.py` - block-scaled GEMM + epilogue (bias,
     GELU, LoRA-up, optional next-layer re-quant). Built on top of
     `tl.dot_scaled` (Gluon variant on sm_100/103/110 for TMEM, generic
     variant elsewhere).
3. A thin Python dispatcher in
   [`nunchaku/ops/gemm.py`](nunchaku/ops/gemm.py) that switches between the
   legacy CUDA kernel (sm_89/sm_90/sm_120 fallback) and the new Triton
   path (sm_100/103/110), keyed on
   `torch.cuda.get_device_capability()`.
4. A relaxed numerical-equivalence harness alongside
   [`MATH_TEST_PLAN.md`](MATH_TEST_PLAN.md) §6 that uses dequantized-value
   equality at FP4 precision rather than byte equality.

**Do not look elsewhere.** The realistic alternatives are CUTLASS 3.6+
(which has `Sm100BlockScaled` but ties us into another large C++ template
metaprogramming codebase and a separate build system), or hand-writing PTX
(which is the very thing this exercise is trying to avoid). Triton gives
us the right level of abstraction - a Python kernel that the compiler
lowers into either `mma.sync` or `tcgen05.mma` based on capability - and
brings two tutorial-quality reference implementations
([`10-block-scaled-matmul.py`](../triton/python/tutorials/10-block-scaled-matmul.py),
[`11-tcgen05-mma-scaled.py`](../triton/python/tutorials/gluon/11-tcgen05-mma-scaled.py))
that we can use as a starting point.

The only place where Triton might not be a win is the very fused
`EpilogueQuantize` next-layer re-quant path, where the legacy CUDA kernel
has been micro-tuned for register reuse across two consecutive W4A4 layers.
That path is worth profiling on real workloads before we commit to a full
port; if Triton gives us 80-90% of the legacy performance on it, the
maintenance win of having a single Python kernel covers the gap many times
over. If it gives us 50%, we either keep the legacy kernel as the fast
path for that single epilogue or invest in a custom Gluon implementation
along the lines of
[`14-multicta.py`](../triton/python/tutorials/gluon/14-multicta.py)
warp-specialized examples.

---

## 9. Adjustments for diffusion-transformer use case

The picture changes substantially once we accept that the target workload
is diffusion transformers (Z-Image-Turbo, FLUX, Sana, QwenImage) rather
than a 172B sparse-MoE LLM. The relevant differences:

### 9.1 Workload shape

| Property | Diffusion DiT (Z-Image-Turbo) | LLM (MiniMax-M2.7 NVFP4) |
|---|---|---|
| Parameter count | ~6-13 B (FP4) | 172 B (FP4 MoE) |
| FP4 weight bytes on disk | 3-6 GB | ~85 GB |
| Linear-layer call pattern | Same `~30` blocks called for `~25` denoise steps | One token at a time per layer, KV-cache-sized inputs |
| Typical M at each linear | 4096-65536 (image tokens x batch) | 1-32 (decode) or 512-2048 (prefill) |
| Typical K, N | 1536-3072 | 4096-6144 |
| Latency target | One image in ~1-4 s | Tokens/s, sub-100ms TTFT |
| **Same weights re-used per step** | **Yes - same Linear called 25-50x** | No - new tokens each step |
| LoRA hot-swapping at runtime | Yes (`update_lora_params`, `set_lora_strength`) | Rare |

The dominant fact is the second to last row: each `SVDQW4A4Linear`
processes activations 25-50 times per image with the **exact same
weights**. Any one-time cost paid against the weights (repacking,
layout conversion, Triton JIT, autotune) is amortized over 25-50 launches
per inference, then over an entire user session. This is the opposite
of LLM inference where weights are touched once per token and amortization
windows are short.

### 9.2 Repacking is essentially free

For a Z-Image-Turbo r128 NVFP4 checkpoint
([`svdq-fp4_r128-z-image-turbo.safetensors`](../../models/nunchaku-z-image-turbo/svdq-fp4_r128-z-image-turbo.safetensors)):

* Total FP4 weight bytes on disk: a few GB.
* Repacking decodes Nunchaku's fragment-permuted layout
  ([`MATH.md`](MATH.md) §8.1) into Triton's
  `BlackwellMXValueLayout` + `BlackwellMXScaleLayout` (§§3.2-3.4 in
  this doc). This is bandwidth-bound, GPU-resident, and takes well under
  a second per checkpoint on any Blackwell GPU.
* The repacked tensors fit comfortably in HBM; no need to stream shards
  for these model sizes.

Concretely: replace `SVDQW4A4Linear.__init__`'s allocation of
`self.qweight`, `self.wscales`, `self.wcscales` with **two** parameter
buffers - a legacy-format one (kept for the CUDA fallback) and a
Triton-format one (used on sm_100/103/110). After
[`from_pretrained()`](nunchaku/models/transformers/transformer_zimage.py:1)
finishes, walk the module tree and convert. **No safetensors changes
needed; no checkpoint-build-tool changes needed; no cross-binary
compatibility concerns.**

For very large MoE models we discussed in §4 (cache-to-disk, one shard at
a time), that's also still available, but for the actual diffusion DiTs
this is not required.

### 9.3 LoRA semantics need to stay intact

Diffusion DiTs are the heaviest user of dynamic LoRA. Nunchaku exposes
[`set_lora_strength()`](nunchaku/models/transformers/transformer_flux.py:836)
and
[`update_lora_params()`](nunchaku/models/transformers/transformer_flux.py:785)
which mutate `proj_down` / `proj_up` between generations. Key
implications:

1. `proj_down` and `proj_up` are **already** stored as plain BF16/FP16
   matrices (see
   [`linear.py:121-122`](nunchaku/models/linear.py:121)). They are not
   touched by the FP4 layout transformations; the LoRA-down GEMM is a
   BF16 GEMM on top of FP32 activations, and the LoRA-up GEMM is a
   BF16 GEMM into the BF16 output. **Triton has no problem with these
   shapes** - they go through plain `tl.dot` (or `tl.dot` with
   appropriate `kWidth`).
2. We must keep the LoRA-down fusion semantics: BF16 input is read
   exactly once, scattered to (a) FP4 quantized activations and (b)
   LoRA-down rank-R BF16 GEMM, both in the same kernel. The
   straightforward Triton port writes two kernels and pays one extra
   global-memory round-trip on the activations; for the M=4096-65536
   shapes typical in diffusion that's actually fine - we're not bandwidth-
   bound. If profiling later shows otherwise, the LoRA-down `tl.dot`
   can be fused into the activation quantizer with a `K`-pipelined inner
   loop.
3. `update_lora_params(...)` must continue to be a pure-Python op that
   only writes `proj_down` / `proj_up`. It must **not** trigger a Triton
   recompile or a weight repack. Since both LoRA matrices live outside
   the FP4 layout, this is naturally true.

### 9.4 Autotune & JIT cost

Triton kernels are compiled per unique launch shape. For a DiT:

* The number of **distinct M values** in a forward pass is at most one
  or two (image-token count + a small text-context count).
* Across an inference session the M values are fixed (the user picks an
  image resolution, then runs N denoising steps at that resolution).
* K and N vary per layer but the model has on the order of 30-40
  blocks * 6-8 linears per block = ~200 unique (M, N, K) triples.

So we autotune ~200 kernels once during a warmup step, cache to
`~/.triton/cache/`, and never recompile again for the same checkpoint.
The first inference is slower (a few extra seconds); subsequent ones are
unaffected. This is **strictly better** than the current static
nvcc-compiled CUDA fatbin, which is tuned for one shape and underperforms
on others.

Compared to LLM serving, where M ranges across `[1, max_batch_size *
max_seq]`, this is a vastly easier autotune surface.

### 9.5 sm_120 is at least as important as sm_110 here

Most diffusion-DiT users sit on **consumer Blackwell (sm_120)** - RTX
5090, 5080, 5070 Ti, and friends. sm_110 is datacenter (B100/B200) and is
where TMEM lives. From [`getMMAVersionSafe`](../triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:51):

* sm_100/103/**110** (`< 120`) -> Triton emits `tcgen05.mma` v5 (TMEM)
* sm_120/121 (`< 130`) -> Triton emits `mma.sync` v2 (registers)

Both paths use the same `kind::mxf4nvf4.block_scale.scale_vec::4X`
encoding and the same FP8 e4m3 scale format
([`MMAv2.cpp:507`](../triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp:507),
[`MMAv5.cpp:300`](../triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp:300)).
**One Triton kernel covers both architectures** with no source change
because the dispatch happens at compile time inside Triton itself.

This is a major shift in calculus: the existing hand-coded sm_120 kernel
in [`gemm_w4a4.cuh`](src/kernels/zgemm/gemm_w4a4.cuh:191) is also
something we'd be glad to retire, because Triton gives us a single
source of truth for both the TMEM (sm_110) and non-TMEM (sm_120) paths.

### 9.6 The "fused next-layer re-quant" epilogue matters less here

`MATH.md` §7 step 5 (`EpilogueQuantize<USE_FP4=true>`) fuses the
output-quantization of one Linear directly into the FP32 accumulators
of the previous Linear, saving one BF16 write + one BF16 read. In a
diffusion DiT:

1. Most Linears are followed by a non-linear activation, an
   `nn.Linear`-shaped attention op, or a residual+LayerNorm - not directly
   by another `SVDQW4A4Linear`. The chained-W4A4 case applies to
   roughly the `mlp_fc1 -> GELU -> mlp_fc2` path and a few QKV/projection
   pairs.
2. M is large (4096-65536), so the kernel is firmly compute-bound. Saving
   one BF16 round-trip per FFN layer is well under 5% of forward-pass time
   in our regime.

So the place where the legacy CUDA kernel is genuinely hard to beat -
fragment-level fusion of FP32 accumulators into the next quantizer - is
not on the critical path for our use case. The Triton port can do
"un-fused" re-quantize (write BF16, read BF16 in next layer, quantize
there) and still hit close to legacy performance. This removes the
hardest open question from §7.

### 9.7 Updated repacking strategy (concrete, DiT-tuned)

Drop §4's complicated hybrid; for diffusion DiTs the right plan is:

1. **`nunchaku.models.linear.SVDQW4A4LinearTriton`**: new subclass of
   `nn.Module`. Holds `qweight_triton`, `wscales_triton`, `wcscales`,
   `wtscale`, `proj_down`, `proj_up`, `smooth_factor`, `bias`. No
   fragment-permuted tensors at all.
2. **`SVDQW4A4Linear.upgrade_to_triton(self)`**: in-place method that
   decodes the existing legacy fragment-permuted tensors into the new
   Triton layout (using
   [`tests/kernels/_helpers.py`](tests/kernels/_helpers.py:1)) and
   returns a `SVDQW4A4LinearTriton`. Called by a one-line walk of the
   module tree after `from_pretrained()`.
3. **Two Triton kernels** under
   [`nunchaku/ops/triton/`](nunchaku/ops/) (new directory):
   * `quantize_w4a4_act_fuse_lora_triton.py` - BF16 activation -> FP4
     act + FP8 ascales + FP32 LoRA-down output. Built on
     [`_quantize_nvfp4_fn`](../triton/python/triton_kernels/triton_kernels/numerics_details/mxfp_details/_downcast_to_mxfp.py:228)
     for the quant half and `tl.dot` for the LoRA-down half.
   * `gemm_w4a4_fp4_triton.py` - FP4 act + FP4 wgt + scales -> BF16
     out, with bias, optional fused LoRA-up (rank R BF16 GEMM into the
     same accumulator), and (optionally) GELU/SiLU. Built on
     `tl.dot_scaled(..., "e2m1", ..., "e2m1", ...)`.
4. **Dispatch in `SVDQW4A4Linear.forward`**: if
   `torch.cuda.get_device_capability() >= (10, 0)` and a precompiled
   Triton path is registered for this `(precision, group_size)`, call
   the Triton path; else fall back to the existing CUDA kernel. Single
   Python branch.
5. **Autotune warmup**: optional, called from
   [`ZImageTurboBackend.load()`](../../src/imagegen/ZImageTurboBackend.py)
   after `pipeline = ZImagePipeline.from_pretrained(...)`. Run one dummy
   denoising step at the target resolution to fill the autotune cache.
6. **Test against the legacy kernel** end-to-end (one denoising step,
   compare BF16 image-domain output) rather than at the byte level. The
   §8.4 zero-encoding quirk doesn't survive any forward pass beyond the
   first GEMM anyway.

### 9.8 Memory savings (small but worth noting)

The legacy `SVDQW4A4Linear` has `smooth_factor_orig` as a duplicate of
`smooth_factor` (see [`linear.py:117`](nunchaku/models/linear.py:117) -
the docstring at line 55 even says "Unused"). The Triton path can drop it.
Per linear that saves `in_features * 2` bytes (BF16); for a ~13 B FLUX
checkpoint this is on the order of tens of MB. Not a reason to do
anything, but a free cleanup while we're touching this code.

### 9.9 Revised top-level recommendation

For the diffusion-transformer use case the recommendation is **stronger
than the original §8**:

* Switch to Triton on **both sm_110 (TMEM) and sm_120 (consumer Blackwell)**.
  Retire the hand-coded `gemm_w4a4_fp4_kernel`
  ([`gemm_w4a4.cuh`](src/kernels/zgemm/gemm_w4a4.cuh:191)) from those
  capability paths.
* Keep the legacy CUDA kernel as the fallback for sm_89/sm_90 (Ada/Hopper
  - older RTX 4090 / H100), where Triton would emit a worse path anyway
  and we have a working binary today.
* Repack at load time (§9.7 step 2). Don't change the on-disk format.
* Don't bother fusing the next-layer re-quantize (§9.6).
* Cache Triton autotune results in
  `~/.cache/nunchaku/triton/<sha>/`; ship them with the wheel for the
  handful of (M, N, K) triples that matter for the popular diffusion
  models. Hand-pick configs for Z-Image-Turbo and FLUX so first-run
  latency is acceptable.

This drops a large fraction of the unmaintainable C++/PTX from
`packages/nunchaku/src/kernels/zgemm/`, gives us **a single Python
implementation that covers sm_100, sm_103, sm_110, sm_120, sm_121** out
of the box, and keeps the door open for a CUDA fallback on Hopper/Ada
where Triton's MMAv3/MMAv2 paths are less competitive.

The risk profile is low: the **only** place we don't have direct Triton
support is the fragment-level fused next-layer re-quant, which §9.6 shows
isn't worth fusing for DiT shapes anyway.

---

## 10. 2026-05 Triton-FP4 debug postmortem

This section documents the two bugs that prevented the initial Triton FP4
path from producing valid images, and the fixes that landed in
[`packages/nunchaku/nunchaku/ops/triton/unpack.py`](nunchaku/ops/triton/unpack.py),
[`packages/nunchaku/nunchaku/ops/triton/gemm.py`](nunchaku/ops/triton/gemm.py),
and [`packages/nunchaku/nunchaku/ops/triton/quantize.py`](nunchaku/ops/triton/quantize.py).

### 10.1 Symptom

Z-Image-Turbo via the Triton sm_110 fallback produced pure VAE-pattern
noise output, while the same checkpoint produced valid images via the
INT4 path (also a CUDA fallback) and the un-quantized BF16 reference.
Per-step latent statistics showed the FP4 trajectory denoising correctly
for the first 4 steps, then **monotonically diverging** (latent std blew
up from 0.6 to 1.4-3.8 by the end of the schedule).

All five MATH_TEST_PLAN.md invariant tests passed pre-fix because they
only checked scale equivariance / sign symmetry / determinism, which a
deterministic-but-algebraically-wrong kernel still satisfies. The
[`tests/kernels/test_quantize.py`](tests/kernels/test_quantize.py) suite
did not have a smooth-quant-fold byte-equality test.

### 10.2 Root causes

1. **SVDQuant LoRA factors `proj_down` and `proj_up` were used as flat
   tensors** in the Triton path, but they are stored on disk in
   [`NunchakuWeightPacker.pack_lowrank_weight()`](../deepcompressor/deepcompressor/backend/nunchaku/utils.py:153)'s
   permuted MMA-fragment layout (7-level reshape + 8-axis permute over
   `pack_n=16, pack_k=16` tiles). Reading them as `(K, R)` / `(N, R)`
   PyTorch matrices yielded a LoRA contribution with
   ``cosine_similarity ≈ 0.015`` versus the original BF16 weight - i.e.
   essentially random direction. This dominated the GEMM output and
   produced the all-noise images.

2. **`smooth_factor` was used as a flat `[K]` tensor** in the Triton
   activation quantizer, but it is also stored in deepcompressor's
   [`pack_scale()`](../deepcompressor/deepcompressor/backend/nunchaku/utils.py:62)
   fragment-interleaved layout (`WARP_N=128`, `WSCALES_PACK_SIZE=4`,
   lane `L` holds positions
   `(L/4)*16 + (L%4)*2 + {0,1,8,9}` of the per-warp slice). This was
   already documented at MATH.md §8.2 as a "known issue that goes
   unnoticed because the official Python wrapper passes a flat tensor".
   On real diffusion DiT checkpoints, smooth_factor has ~5% of channels
   with distinct non-uniform values (Z-Image-Turbo: 192 unique values
   in 3840 channels), so the flat-vs-permuted mismatch is a real
   per-channel scale error.

### 10.3 Fixes

* Added [`_unpack_lowrank_weight()`](nunchaku/ops/triton/unpack.py:282)
  which is byte-identical to the deepcompressor reference (verified by
  100% roundtrip-equal in
  [`packages/nunchaku/scratch/verify_lora_unpack.py`](scratch/verify_lora_unpack.py)),
  plus thin `unpack_proj_down` / `unpack_proj_up` cache wrappers that
  store the unpacked tensor as an attribute on the input
  `nn.Parameter` so subsequent denoise steps re-use the allocation.
* Added [`fragment_to_channel_order()`](nunchaku/ops/triton/unpack.py:207)
  for `smooth_factor` (re-uses the same warp-slice inverse permutation
  that already lived in the file for `ascales`) plus a
  [`_smooth_cache_lookup()`](nunchaku/ops/triton/unpack.py:228) helper
  with the same per-tensor attribute caching pattern.
* The Triton GEMM / quantize call sites now invoke the appropriate
  unpacker before each matmul or divide.

### 10.4 Effect

End-to-end Z-Image-Turbo at 8 denoise steps (prompt "A photograph of a
sleeping orange tabby cat on a sofa", seed=42, 512x512):

| run                      | latent std at t=0 | image-quality score | block-grid score | image content |
|---|---:|---:|---:|---|
| BF16 unquantized          | 0.61 | 0.287 | 0.00 | clear cat |
| INT4 (CUDA fallback)      | 0.63 | 0.412 | 0.00 | clear cat |
| FP4-Triton **before fix** | **1.41** | **0.000** | **2.05** | **pure noise grid** |
| FP4-Triton **after fix**  | **0.63** | **0.394** | **0.00** | **clear cat** |

Per-layer reconstruction of the BF16 weight from the FP4 dequant + LoRA
also went from cos sim 0.256 (random direction) to 0.993 (within FP4
quantization noise). Image-quality regression harness:
[`packages/nunchaku/scratch/evaluate.py`](scratch/evaluate.py) +
[`packages/nunchaku/scratch/img_quality.py`](scratch/img_quality.py).

All 100 tests in [`tests/kernels/`](tests/kernels/) pass after the fix.