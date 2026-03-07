#include "zgemm.h"
#include "attention_rank1bias.h"
#include "attention_rank1bias.cuh"

#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634074
#endif

namespace nunchaku::kernels {

void attention_fp16_rank1bias(Tensor q, // [B, H, TokensQ, HEAD_DIM] FP16
                              Tensor k, // [B, H, TokensKV, HEAD_DIM] FP16
                              Tensor v, // [B, H, TokensKV, HEAD_DIM] FP16
                              Tensor m, // [B, TokensKV] FP16
                              Tensor o, // [B, TokensQ, H*HEAD_DIM] FP16/BF16
                              float scale) {
    int sizeBatch   = q.shape[0];
    int numHeads    = q.shape[1];
    int numTokensQ  = q.shape[2];
    int headDim     = q.shape[3];
    int numTokensKV = k.shape[2];

    assert(o.ndims() == 3);
    assert(o.shape[0] == sizeBatch);
    assert(o.shape[1] == numTokensQ);
    assert(o.shape[2] == numHeads * headDim);

    assert(m.ndims() == 2);
    assert(m.shape[0] == sizeBatch);
    assert(m.shape[1] == numTokensKV);
    assert(m.scalar_type() == Tensor::FP16);

    // we use exp2 instead of exp in the kernel
    scale *= M_LOG2E;

    dispatchBool(o.scalar_type() == Tensor::BF16, [&]<bool bf16out>() {
#ifndef __INTELLISENSE__
        using Attention = typename nunchaku::kernels::AttentionRank1Bias<AttentionFP16Config<bf16out>>;
#else
        using Attention = typename nunchaku::kernels::AttentionRank1Bias<AttentionFP16Config<true>>;
#endif

        using packed_q_t = typename Attention::packed_q_t;
        using packed_k_t = typename Attention::packed_k_t;
        using packed_v_t = typename Attention::packed_v_t;

        assert(isTypeMatch<typename Attention::half_t>(q.scalar_type()));
        assert(isTypeMatch<typename Attention::half_t>(k.scalar_type()));
        assert(isTypeMatch<typename Attention::half_t>(v.scalar_type()));
        assert(isTypeMatch<typename Attention::epilogue_half_t>(o.scalar_type()));

        assert(numTokensQ % Attention::BLOCK_M == 0);
        assert(numTokensKV % Attention::WARP_K == 0);
        assert(headDim == Attention::HEAD_DIM);

        // kernel launch
        dim3 grid(numTokensQ / Attention::BLOCK_M, numHeads, sizeBatch);
        auto func = invoke_kernel<typename Attention::template attention_fp16_rank1bias_kernel<typename Attention::GEMM::EpilogueDefault>,
                                  const packed_q_t *,
                                  const packed_k_t *,
                                  const packed_v_t *,
                                  const half *,
                                  float,
                                  int,
                                  int,
                                  typename Attention::GEMM::EpilogueDefault::Arguments,
                                  bool>;

        func<<<grid, Attention::GEMM::WARP_SIZE * Attention::GEMM::NUM_WARPS, 0, getCurrentCUDAStream()>>>(
            q.data_ptr<packed_q_t>(),
            k.data_ptr<packed_k_t>(),
            v.data_ptr<packed_v_t>(),
            m.data_ptr<half>(),
            scale,
            numTokensQ,
            numTokensKV,
            typename Attention::GEMM::EpilogueDefault::Arguments{
                .out     = o.data_ptr<typename Attention::GEMM::half_t>(),
                .actualM = sizeBatch * numTokensQ,
                .actualN = numHeads * headDim,
            },
            false);
        checkCUDA(cudaGetLastError());
    });
}

} // namespace nunchaku::kernels
