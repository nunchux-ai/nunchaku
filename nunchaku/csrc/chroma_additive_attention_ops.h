#pragma once

#include <torch/extension.h>

namespace nunchaku::ops {

// Packed variant (matches `fused_qkv_norm_rottary(..., output=(q,k,v), attn_tokens=...)` layout):
// Inputs:
// - q/k/v:   [B, H, T_pad, D] contiguous, dtype FP16, CUDA (packed layout for nunchaku attention)
// - m:       [B, T_pad] contiguous, dtype FP16, CUDA (padded mask values; pad tokens should be 0)
// - out:     [B, T_pad, H*D] contiguous, dtype FP16 or BF16, CUDA
//
// This runs the rank-1 bias attention kernel:
//   logits_ij = (q_i·k_j)/sqrt(D) + (m_i*m_j)
void chroma_additive_attention_packed_fp16(torch::Tensor q,
                                           torch::Tensor k,
                                           torch::Tensor v,
                                           torch::Tensor m,
                                           torch::Tensor out,
                                           double scale);

} // namespace nunchaku::ops
