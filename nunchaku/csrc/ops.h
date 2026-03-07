#pragma once

#include "interop/torch.h"
#include "kernels/zgemm/zgemm.h"
#include "kernels/awq/gemv_awq.h"
#include "kernels/awq/gemm_awq.h"

namespace nunchaku::ops {

void gemm_w4a4(std::optional<torch::Tensor> act,            // packed act [M, K / 2]
               std::optional<torch::Tensor> wgt,            // packed act [N, K / 2]
               std::optional<torch::Tensor> out,            // linear     [M, N]
               std::optional<torch::Tensor> qout,           // packed act [M, N / 2]
               std::optional<torch::Tensor> ascales,        // packed as  [K / 64, M]
               std::optional<torch::Tensor> wscales,        // packed ws  [K / 64, N]
               std::optional<torch::Tensor> oscales,        // packed as  [N / 64, M]
               std::optional<torch::Tensor> poolout,        // linear     [M / PoolSize, N]
               std::optional<torch::Tensor> lora_act_in,    // packed lora_act [M, R]
               std::optional<torch::Tensor> lora_up,        // packed lora_wgt [N, R]
               std::optional<torch::Tensor> lora_down,      // packed lora_wgt [N, R]
               std::optional<torch::Tensor> lora_act_out,   // packed lora_act [M, R]
               std::optional<torch::Tensor> norm_q,         // linear     [HEAD_DIM]
               std::optional<torch::Tensor> norm_k,         // linear     [HEAD_DIM]
               std::optional<torch::Tensor> rotary_emb,     // linear     [M, HEAD_DIM / 2, 2, 2]
               std::optional<torch::Tensor> bias,           // packed ws  [N]
               std::optional<torch::Tensor> smooth_factor,  // packed ws  [N], for quantization of the next layer
               std::optional<torch::Tensor> out_vk,         // linear     [B, num_heads, head_dim + 1, head_dim]
               std::optional<torch::Tensor> out_linearattn, // linear     [B, (M), N / 3]
               bool act_unsigned,
               bool qout_act_unsigned,
               std::vector<float> lora_scales,
               bool fuse_silu,
               bool fp4,
               float alpha,
               std::optional<torch::Tensor> wcscales,
               std::optional<torch::Tensor> out_q, // packed attention [B, H, M, D]
               std::optional<torch::Tensor> out_k, // packed attention [B, H, M, D]
               std::optional<torch::Tensor> out_v, // packed attention [B, H, M, D]
               int attn_tokens) {
    TorchOpContext ctx;
    spdlog::trace("running gemm_w4a4: ");

    auto getTensor = [](std::optional<torch::Tensor> &t) {
        Tensor ret = t.has_value() ? from_torch(t.value()) : Tensor{};
        if (ret.valid()) {
            spdlog::trace("  {}", ret.shape.str());
        } else {
            spdlog::trace("  <invalid>");
        }
        return ret;
    };
    nunchaku::kernels::gemm_w4a4(getTensor(act),
                                 getTensor(wgt),
                                 getTensor(out),
                                 getTensor(qout),
                                 getTensor(ascales),
                                 getTensor(wscales),
                                 getTensor(oscales),
                                 getTensor(poolout),
                                 getTensor(lora_act_in),
                                 getTensor(lora_up),
                                 getTensor(lora_down),
                                 getTensor(lora_act_out),
                                 getTensor(norm_q),
                                 getTensor(norm_k),
                                 getTensor(rotary_emb),
                                 getTensor(bias),
                                 getTensor(smooth_factor),
                                 getTensor(out_vk),
                                 getTensor(out_linearattn),
                                 act_unsigned,
                                 qout_act_unsigned,
                                 lora_scales,
                                 fuse_silu,
                                 fp4,
                                 alpha,
                                 getTensor(wcscales),
                                 getTensor(out_q),
                                 getTensor(out_k),
                                 getTensor(out_v),
                                 attn_tokens);
    // Tensor::synchronizeDevice();
}

void quantize_w4a4_act_fuse_lora(std::optional<torch::Tensor> input,
                                 std::optional<torch::Tensor> output,
                                 std::optional<torch::Tensor> oscales,
                                 std::optional<torch::Tensor> lora_down,
                                 std::optional<torch::Tensor> lora_act_out,
                                 std::optional<torch::Tensor> smooth,
                                 bool act_unsigned,
                                 bool fuse_glu,
                                 bool fp4) {
    TorchOpContext ctx;

    spdlog::trace("running quantize_w4a4_act_fuse_lora: ");

    auto getTensor = [](std::optional<torch::Tensor> &t) {
        Tensor ret = t.has_value() ? from_torch(t.value()) : Tensor{};
        if (ret.valid()) {
            spdlog::trace("  {}", ret.shape.str());
        } else {
            spdlog::trace("  <invalid>");
        }
        return ret;
    };
    nunchaku::kernels::quantize_w4a4_act_fuse_lora(getTensor(input),
                                                   getTensor(output),
                                                   getTensor(oscales),
                                                   getTensor(lora_down),
                                                   getTensor(lora_act_out),
                                                   getTensor(smooth),
                                                   act_unsigned,
                                                   fuse_glu,
                                                   fp4);
}

void attention_fp16(torch::Tensor q, // packed [Batch, Head, TokensQ, HEAD_DIM]
                    torch::Tensor k, // packed [Batch, Head, TokensKV, HEAD_DIM]
                    torch::Tensor v, // packed [Batch, Head, TokensKV, HEAD_DIM]
                    torch::Tensor o, // linear [Batch, TokensQ, Head * HEAD_DIM]
                    float scale) {
    TorchOpContext ctx;
    nunchaku::kernels::attention_fp16(from_torch(q), from_torch(k), from_torch(v), from_torch(o), scale);
}

torch::Tensor gemv_awq(torch::Tensor _in_feats,
                       torch::Tensor _kernel,
                       torch::Tensor _scaling_factors,
                       torch::Tensor _zeros,
                       int64_t m,
                       int64_t n,
                       int64_t k,
                       int64_t group_size) {
    TorchOpContext ctx;
    Tensor result = ::gemv_awq(from_torch(_in_feats.contiguous()),
                               from_torch(_kernel.contiguous()),
                               from_torch(_scaling_factors.contiguous()),
                               from_torch(_zeros.contiguous()),
                               (int)m,
                               (int)n,
                               (int)k,
                               (int)group_size);

    torch::Tensor output = to_torch(result);
    // Tensor::synchronizeDevice();

    return output;
}

torch::Tensor
gemm_awq(torch::Tensor _in_feats, torch::Tensor _kernel, torch::Tensor _scaling_factors, torch::Tensor _zeros) {
    Tensor result = ::awq_gemm_forward_cuda(from_torch(_in_feats.contiguous()),
                                            from_torch(_kernel.contiguous()),
                                            from_torch(_scaling_factors.contiguous()),
                                            from_torch(_zeros.contiguous()));

    TorchOpContext ctx;
    // TODO: allocate output in torch and use from_torch instead (to_torch needs an extra copy)
    torch::Tensor output = to_torch(result);
    // Tensor::synchronizeDevice();

    return output;
}

void test_rmsnorm_rope(
    torch::Tensor input, torch::Tensor output, torch::Tensor norm_q, torch::Tensor norm_k, torch::Tensor rotary_emb) {
    nunchaku::kernels::test_rmsnorm_rope(
        from_torch(input), from_torch(output), from_torch(norm_q), from_torch(norm_k), from_torch(rotary_emb));
}

void test_pack_qkv(torch::Tensor input, torch::Tensor out_q, torch::Tensor out_k, torch::Tensor out_v, int numTokens) {
    nunchaku::kernels::test_pack_qkv(
        from_torch(input), from_torch(out_q), from_torch(out_k), from_torch(out_v), numTokens);
}

}; // namespace nunchaku::ops
