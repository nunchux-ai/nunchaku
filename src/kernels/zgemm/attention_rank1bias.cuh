#pragma once

#include "attention.cuh"
#include <limits>

namespace nunchaku::kernels {

// A variant of `Attention` that adds a rank-1 additive bias term:
//   logits_ij = (q_i · k_j) * scale + (m_i * m_j)
//
// Notes:
// - `scale` passed into the kernel should already be multiplied by log2(e),
//   as the attention kernels use exp2().
// - `m` is FP16 and indexed by token position (same for all heads).
template<typename AttentionConfig>
class AttentionRank1Bias : public Attention<AttentionConfig> {
public:
    using Base = Attention<AttentionConfig>;
    using typename Base::BlockInfo;
    using Base::BLOCK_M;
    using Base::HEAD_DIM;
    using Base::INSN_N;
    using Base::NUM_WARPS;
    using Base::WARP_K;
    using Base::WARP_K_TILES_PV;
    using Base::WARP_K_TILES_QK;
    using Base::WARP_M;
    using Base::WARP_M_TILES;
    using Base::WARP_N_TILES;
    using Base::WARP_SIZE;
    using typename Base::packed_fpsum_t;
    using typename Base::packed_f32psum_t;
    using typename Base::packed_k_t;
    using typename Base::packed_p_t;
    using typename Base::packed_q_t;
    using typename Base::packed_v_t;
    using typename Base::q_warp;
    using typename Base::k_warp;
    using typename Base::p_warp;
    using typename Base::qk_warp;
    using typename Base::rowval_warp;
    using typename Base::v_warp;
    using typename Base::o_warp;

    static constexpr float LOG2E = 1.4426950408889634074f;

    __device__ __forceinline__ static float half_to_float(const half &x) { return __half2float(x); }

    // lane-local mapping for the MMA accumulator fragment:
    // - rows are split as {row0=row_base, row1=row_base+8}, where row_base = laneId / 4 in [0,7]
    // - within each 16-col half-tile, cols are 4 consecutive columns:
    //     col_base = (laneId & 3) * 4
    // - accumulator ordering matches `compute_rowmax` grouping:
    //     data[0,1,4,5] -> row0 cols 0,1,2,3
    //     data[2,3,6,7] -> row1 cols 0,1,2,3
    __device__ __forceinline__ static void get_rowcol_base(int laneId, int &row_base, int &col_base) {
        row_base = laneId >> 2;
        col_base = (laneId & 3) << 2;
    }

    __device__ __forceinline__ static void load_mq(const half *__restrict__ m,
                                                   int base_q,
                                                   int laneId,
                                                   float &mq0,
                                                   float &mq1) {
        int row_base, col_base;
        get_rowcol_base(laneId, row_base, col_base);
        (void)col_base;
        mq0 = half_to_float(m[base_q + row_base]);
        mq1 = half_to_float(m[base_q + row_base + 8]);
    }

    __device__ __forceinline__ static void load_mk4(const half *__restrict__ m, int base_k, int laneId, float mk[4]) {
        int row_base, col_base;
        get_rowcol_base(laneId, row_base, col_base);
        (void)row_base;
        mk[0] = half_to_float(m[base_k + col_base + 0]);
        mk[1] = half_to_float(m[base_k + col_base + 1]);
        mk[2] = half_to_float(m[base_k + col_base + 2]);
        mk[3] = half_to_float(m[base_k + col_base + 3]);
    }

    __device__ __forceinline__ static rowval_warp compute_rowmax_bias(qk_warp QK,
                                                                      rowval_warp rowmax,
                                                                      float scale_log2e,
                                                                      const half *__restrict__ m,
                                                                      int base_q,
                                                                      int base_k,
                                                                      int laneId) {
        float mq0, mq1;
        load_mq(m, base_q, laneId, mq0, mq1);

#pragma unroll
        for (int mt = 0; mt < (int)WARP_M_TILES; mt++) {
            float2 maxv = make_float2(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());

#pragma unroll
            for (int kt = 0; kt < (int)WARP_K_TILES_QK; kt++) {
                float mk[4];
                load_mk4(m, base_k + kt * 16, laneId, mk);

                packed_f32psum_t &val = QK[mt * WARP_K_TILES_QK + kt];
                // scaled logits (exp2 domain): qk*scale_log2e + (mq*mk)*LOG2E
                float x0 = fmaf(val.data[0], scale_log2e, mq0 * mk[0] * LOG2E);
                float x1 = fmaf(val.data[1], scale_log2e, mq0 * mk[1] * LOG2E);
                float x2 = fmaf(val.data[4], scale_log2e, mq0 * mk[2] * LOG2E);
                float x3 = fmaf(val.data[5], scale_log2e, mq0 * mk[3] * LOG2E);
                float y0 = fmaf(val.data[2], scale_log2e, mq1 * mk[0] * LOG2E);
                float y1 = fmaf(val.data[3], scale_log2e, mq1 * mk[1] * LOG2E);
                float y2 = fmaf(val.data[6], scale_log2e, mq1 * mk[2] * LOG2E);
                float y3 = fmaf(val.data[7], scale_log2e, mq1 * mk[3] * LOG2E);

                float x = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
                float y = fmaxf(fmaxf(y0, y1), fmaxf(y2, y3));
                maxv.x   = fmaxf(maxv.x, x);
                maxv.y   = fmaxf(maxv.y, y);
            }

            // Reduce among the 4 lanes covering the same two rows.
            for (int mask = 1; mask <= 2; mask *= 2) {
                maxv.x = fmaxf(maxv.x, __shfl_xor_sync(~0, maxv.x, mask));
                maxv.y = fmaxf(maxv.y, __shfl_xor_sync(~0, maxv.y, mask));
            }
            rowmax[mt].x = fmaxf(rowmax[mt].x, maxv.x);
            rowmax[mt].y = fmaxf(rowmax[mt].y, maxv.y);
        }
        return rowmax;
    }

    __device__ __forceinline__ static qk_warp softmax_bias(qk_warp QK,
                                                           rowval_warp rowmax_scaled,
                                                           float scale_log2e,
                                                           const half *__restrict__ m,
                                                           int base_q,
                                                           int base_k,
                                                           int laneId) {
        float mq0, mq1;
        load_mq(m, base_q, laneId, mq0, mq1);

#pragma unroll
        for (int mt = 0; mt < (int)WARP_M_TILES; mt++) {
            float2 shift = rowmax_scaled[mt];
#pragma unroll
            for (int kt = 0; kt < (int)WARP_K_TILES_QK; kt++) {
                float mk[4];
                load_mk4(m, base_k + kt * 16, laneId, mk);

                packed_f32psum_t &val = QK[mt * WARP_K_TILES_QK + kt];
                // exp2( (qk*scale_log2e + bias*LOG2E) - rowmax )
                val.data[0] = cuda_exp2(fmaf(val.data[0], scale_log2e, mq0 * mk[0] * LOG2E - shift.x));
                val.data[1] = cuda_exp2(fmaf(val.data[1], scale_log2e, mq0 * mk[1] * LOG2E - shift.x));
                val.data[4] = cuda_exp2(fmaf(val.data[4], scale_log2e, mq0 * mk[2] * LOG2E - shift.x));
                val.data[5] = cuda_exp2(fmaf(val.data[5], scale_log2e, mq0 * mk[3] * LOG2E - shift.x));

                val.data[2] = cuda_exp2(fmaf(val.data[2], scale_log2e, mq1 * mk[0] * LOG2E - shift.y));
                val.data[3] = cuda_exp2(fmaf(val.data[3], scale_log2e, mq1 * mk[1] * LOG2E - shift.y));
                val.data[6] = cuda_exp2(fmaf(val.data[6], scale_log2e, mq1 * mk[2] * LOG2E - shift.y));
                val.data[7] = cuda_exp2(fmaf(val.data[7], scale_log2e, mq1 * mk[3] * LOG2E - shift.y));
            }
        }
        return QK;
    }

    __device__ __forceinline__ static std::tuple<p_warp, rowval_warp>
    compute(q_warp Q,
            k_warp K,
            rowval_warp &M,
            rowval_warp &L,
            float scale_log2e,
            const half *__restrict__ m,
            int base_q,
            int base_k,
            int laneId) {
        qk_warp qk          = Base::compute_qk(Q, K);
        rowval_warp M1      = compute_rowmax_bias(qk, M, scale_log2e, m, base_q, base_k, laneId);
        qk                  = softmax_bias(qk, M1, scale_log2e, m, base_q, base_k, laneId);
        rowval_warp rowsum  = Base::compute_rowsum(qk);
        p_warp P            = Base::qk_to_p(qk);
        rowval_warp rescale = Base::compute_rescale(M, M1);
        M                   = M1;
        L                   = Base::compute_l(L, rescale, rowsum);
        return {P, rescale};
    }

    template<typename Epilogue>
    __device__ __forceinline__ static void attention_fp16_rank1bias_block(BlockInfo binfo,
                                                                          const packed_q_t *ptr_q,
                                                                          const packed_k_t *ptr_k,
                                                                          const packed_v_t *ptr_v,
                                                                          const half *ptr_m_batch,
                                                                          float scale_log2e,
                                                                          int ntokens_q,
                                                                          int ntokens_kv,
                                                                          typename Epilogue::Arguments epilogueArgs,
                                                                          bool alwaysfalse) {
        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

        q_warp Q;
        k_warp K;
        v_warp V;
        o_warp O;
        rowval_warp L;
        rowval_warp M;

        Base::load_q(ptr_q, Q, true);
        Base::load_k(ptr_k, 0, K, true);

#pragma unroll
        for (auto &pack : O) {
#pragma unroll
            for (int i = 0; i < 8; i++) {
                pack.data[i] = 0;
            }
        }

        static constexpr float neginf = -std::numeric_limits<float>::max();
        L.fill(make_float2(0.0f, 0.0f));
        M.fill(make_float2(neginf, neginf));

        static constexpr int SHMEM_TILES = Base::IS_SM80 ? 4 : 7;
        static_assert(SHMEM_TILES <= (int)Q.size());
        using q_shmem_t = packed_q_t[NUM_WARPS][SHMEM_TILES][WARP_SIZE];
        __shared__ q_shmem_t Q_shmem;

#pragma unroll
        for (int i = 0; i < SHMEM_TILES; i++) {
            store<true>(&Q_shmem[warpId][i][laneId], Q[Q.size() - 1 - i]);
        }
        __syncwarp();

        int dummy = 0;

        // Each block processes BLOCK_M query tokens, split across warps by WARP_M.
        const int base_q = binfo.bm * BLOCK_M + warpId * WARP_M;

        for (int k1 = 0; k1 < ntokens_kv / WARP_K; k1++) {
            if (alwaysfalse) {
                ptr_v += K[0].x;
            }

#pragma unroll
            for (int i = 0; i < SHMEM_TILES; i++) {
                Q[Q.size() - 1 - i] = load<true>(&Q_shmem[warpId][i][laneId]);
            }

            if constexpr (!Base::IS_SM80) {
                if (k1 % 2 == 1) {
                    __syncthreads();
                }
            }

            if (alwaysfalse) {
                dummy = clock();
            }

            Base::load_v(ptr_v, k1, V, true);

            if (alwaysfalse) {
                dummy = clock();
            }

            // Key token base for this tile (WARP_K tokens).
            const int base_k = k1 * WARP_K;
            auto [P, rescale] = compute(Q, K, M, L, scale_log2e, ptr_m_batch, base_q, base_k, laneId);

            if (alwaysfalse) {
                dummy = clock();
            }

            if (alwaysfalse) {
                ptr_k += V[0].x;
            }

            Base::load_k(ptr_k, k1 + 1, K, k1 + 1 < ntokens_kv / WARP_K);

            O = Base::compute_pv(P, V, O, rescale);

            if (alwaysfalse) {
                dummy = clock();
            }
        }

        unused_var(dummy, alwaysfalse);

        O = Base::compute_o(O, L);

        auto f16psum = Base::GEMM::packed_fp32_to_fp16(O);

        Epilogue()(
            typename Base::GEMM::BlockInfo{
                .bm         = binfo.batch * binfo.numBlocksM + binfo.bm,
                .bn         = binfo.head,
                .numBlocksM = binfo.numBatch * binfo.numBlocksM,
                .numBlocksN = binfo.numHeads,
            },
            f16psum,
            binfo.numBatch * binfo.numBlocksM * BLOCK_M,
            binfo.numHeads * HEAD_DIM,
            0,
            epilogueArgs);
    }

    template<typename Epilogue>
    struct attention_fp16_rank1bias_kernel {
        static constexpr int MIN_ARCH   = std::is_same_v<typename Base::half_t, __nv_bfloat16> ? 800 : 750;
        static constexpr int SHMEM_SIZE = 0;

        __device__ void operator()(const packed_q_t *ptr_q,
                                   const packed_k_t *ptr_k,
                                   const packed_v_t *ptr_v,
                                   const half *ptr_m,
                                   float scale_log2e,
                                   int ntokens_q,
                                   int ntokens_kv,
                                   typename Epilogue::Arguments epilogueArgs,
                                   bool alwaysfalse) {
            BlockInfo binfo = {
                .bm         = (int)blockIdx.x,
                .head       = (int)blockIdx.y,
                .batch      = (int)blockIdx.z,
                .numBlocksM = (int)gridDim.x,
                .numHeads   = (int)gridDim.y,
                .numBatch   = (int)gridDim.z,
            };

            const int ktiles = ceilDiv(ntokens_kv, WARP_K);

            attention_fp16_rank1bias_block<Epilogue>(
                binfo,
                ptr_q + ((binfo.batch * binfo.numHeads + binfo.head) * binfo.numBlocksM + binfo.bm) * NUM_WARPS *
                            WARP_M_TILES * Base::WARP_D_TILES * WARP_SIZE,
                ptr_k + (binfo.batch * binfo.numHeads + binfo.head) * ktiles * WARP_K_TILES_QK * Base::WARP_D_TILES *
                            WARP_SIZE,
                ptr_v + (binfo.batch * binfo.numHeads + binfo.head) * ktiles * WARP_K_TILES_PV * WARP_N_TILES * WARP_SIZE,
                ptr_m + binfo.batch * ntokens_kv,
                scale_log2e,
                ntokens_q,
                ntokens_kv,
                epilogueArgs,
                alwaysfalse);
        }
    };
};

} // namespace nunchaku::kernels
