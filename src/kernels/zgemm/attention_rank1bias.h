#pragma once

#include "Tensor.h"

namespace nunchaku::kernels {

// Attention kernel with Chroma rank-1 additive bias:
//   logits_ij = (q_i · k_j) / sqrt(D) + (m_i * m_j)
//
// Inputs:
// - q/k/v: [B, H, Tokens, HEAD_DIM] FP16 packed
// - m:     [B, Tokens] FP16
// - o:     [B, Tokens, H*HEAD_DIM] FP16 or BF16
//
// Constraints match attention_fp16: TokensQ multiple of 128, TokensKV multiple of 32, HEAD_DIM=128.
void attention_fp16_rank1bias(Tensor q, Tensor k, Tensor v, Tensor m, Tensor o, float scale);

} // namespace nunchaku::kernels
