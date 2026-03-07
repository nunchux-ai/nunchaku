#include <iostream>

#include "SanaModel.h"
#include "kernels/zgemm/zgemm.h"
#include "flash_api.h"
#include "kernels/misc_kernels.h"

#include <nvtx3/nvToolsExt.h>

using spdlog::fmt_lib::format;
using namespace nunchaku;

SanaLinearAttention::SanaLinearAttention(
    int dim, bool bias, bool pag, bool use_fp4, Tensor::ScalarType dtype, Device device)
    : dim(dim), dim_pad(ceilDiv(dim, 128) * 128), qkv_proj(dim, dim_pad * 3, bias, use_fp4, dtype, device),
      out_proj(dim_pad, dim, bias, use_fp4, dtype, device), pag_to_v(std::nullopt) {
    registerChildren(qkv_proj, "qkv_proj")(out_proj, "out_proj");

    if (pag) {
        pag_to_v.emplace(dim, dim_pad, bias, use_fp4, dtype, device);
        registerChildren(pag_to_v.value(), "pag_to_v");
    }
}

Tensor SanaLinearAttention::forward(Tensor x, Tensor out) {
    constexpr int HEAD_DIM = 32;

    assert(x.ndims() == 3);
    const int batch_size     = x.shape[0];
    const int num_tokens     = x.shape[1];
    const int num_tokens_pad = ceilDiv(num_tokens, 256) * 256;
    assert(x.shape[2] == dim);

    const int num_heads = dim_pad / HEAD_DIM;

    if (num_tokens_pad != num_tokens) {
        spdlog::debug("SanaLinearAttention: pad num_tokens from {} to {}", num_tokens, num_tokens_pad);

        Tensor x_pad = Tensor::allocate({batch_size, num_tokens_pad, dim}, x.dtype(), x.device());
        x_pad.zero_();
        for (int i = 0; i < batch_size; i++) {
            x_pad.slice(0, i, i + 1).slice(1, 0, num_tokens).copy_(x.slice(0, i, i + 1));
        }

        x = x_pad;
    }

    auto qact = qkv_proj.quantize(x, false);

    Tensor q  = Tensor::allocate({batch_size, num_tokens_pad, dim_pad}, x.dtype(), x.device());
    Tensor vk = Tensor::allocate({batch_size, num_heads, HEAD_DIM + 1, HEAD_DIM}, Tensor::FP32, x.device());

    kernels::gemm_w4a4(qact.act,
                       qkv_proj.qweight,
                       {},
                       {},
                       qact.ascales,
                       qkv_proj.wscales,
                       {},
                       {},
                       qact.lora_act,
                       qkv_proj.lora_up,
                       {},
                       {},
                       {},
                       {},
                       {},
                       qkv_proj.bias,
                       {},
                       vk,
                       q,
                       qact.is_unsigned,
                       false,
                       qkv_proj.lora_scales,
                       false,
                       qkv_proj.use_fp4,
                       *qkv_proj.wtscale.data_ptr<float>(),
                       qkv_proj.wcscales.numel() > 0 ? qkv_proj.wcscales : Tensor{},
                       {},
                       {},
                       {},
                       0);

    debug("vk", vk);
    debug("q", q);

    kernels::linearattn_vk_mul_q(q, vk);

    debug("raw_attn_output", q);

    if (num_tokens_pad != num_tokens) {
        Tensor q_unpad = Tensor::allocate({batch_size, num_tokens, dim_pad}, q.dtype(), q.device());
        for (int i = 0; i < batch_size; i++) {
            q_unpad.slice(0, i, i + 1).copy_(q.slice(0, i, i + 1).slice(1, 0, num_tokens));
        }
        q = q_unpad;
    }

    // kernels::gemm_w8a8_fuse_litela(qact.act, qkv.qweight, q, vk, qact.ascales, qkv.wscales);

    // return out_proj.forward(q);
    if (!out.valid()) {
        out = Tensor::allocate({batch_size, num_tokens, dim}, q.dtype(), q.device());
    }
    out_proj.forward(q, out);
    return out;
}

Tensor SanaLinearAttention::forward_pag(Tensor x, bool cfg) {
    const int batch_size = x.shape[0];
    const int num_tokens = x.shape[1];

    Tensor out = Tensor::allocate({batch_size, num_tokens, dim}, x.dtype(), x.device());
    Tensor x_org, x_ptb;
    Tensor out_org, out_ptb;

    if (cfg) {
        assert(batch_size % 3 == 0);
        x_org   = x.slice(0, 0, batch_size * 2 / 3);
        x_ptb   = x.slice(0, batch_size * 2 / 3, batch_size);
        out_org = out.slice(0, 0, batch_size * 2 / 3);
        out_ptb = out.slice(0, batch_size * 2 / 3, batch_size);
    } else {
        assert(batch_size % 2 == 0);
        x_org   = x.slice(0, 0, batch_size / 2);
        x_ptb   = x.slice(0, batch_size / 2, batch_size);
        out_org = out.slice(0, 0, batch_size / 2);
        out_ptb = out.slice(0, batch_size / 2, batch_size);
    }

    this->forward(x_org, out_org);

    Tensor v_ptb = this->pag_to_v.value().forward(x_ptb);
    this->out_proj.forward(v_ptb, out_ptb);

    return out;
}

MultiHeadCrossAttention::MultiHeadCrossAttention(
    int num_heads, int head_dim, bool use_fp4, Tensor::ScalarType dtype, Device device)
    : num_heads(num_heads), head_dim(head_dim),
      q_linear(num_heads * head_dim, num_heads * head_dim, true, use_fp4, dtype, device),
      kv_linear(num_heads * head_dim, num_heads * head_dim * 2, true, dtype, device),
      out_proj(num_heads * head_dim, num_heads * head_dim, true, use_fp4, dtype, device) {
    registerChildren(q_linear, "q_linear")(kv_linear, "kv_linear")(out_proj, "out_proj");
}

Tensor MultiHeadCrossAttention::forward(Tensor x, Tensor cond, Tensor cu_seqlens_img, Tensor cu_seqlens_txt) {
    assert(x.ndims() == 3);
    assert(cond.ndims() == 2);
    assert(cu_seqlens_img.ndims() == 1);
    assert(cu_seqlens_txt.ndims() == 1);

    const int batch_size     = x.shape[0];
    const int num_tokens_img = x.shape[1];
    const int num_tokens_txt = cond.shape[0];

    assert(cu_seqlens_img.shape[0] == batch_size + 1);
    assert(cu_seqlens_txt.shape[0] == batch_size + 1);

    Tensor q  = q_linear.forward(x).view({batch_size * num_tokens_img, num_heads, head_dim});
    Tensor kv = kv_linear.forward(cond).view({num_tokens_txt, num_heads * 2, head_dim});

    Tensor k = kv.slice(1, 0, num_heads);
    Tensor v = kv.slice(1, num_heads, num_heads * 2);

    Tensor attn_output = mha_varlen_fwd(q,
                                        k,
                                        v,
                                        cu_seqlens_img,
                                        cu_seqlens_txt,
                                        num_tokens_img,
                                        num_tokens_txt,
                                        0.0f,
                                        pow(q.shape[-1], (-0.5)),
                                        false,
                                        false,
                                        -1,
                                        -1,
                                        false)
                             .front()
                             .view({batch_size, num_tokens_img, num_heads * head_dim});

    // Tensor attn_output = mha_fwd(q, k, v,
    //     0.0f,
    //     pow(q.shape[-1], (-0.5)),
    //     false, -1, -1, false
    // ).front().view({B, N, num_heads * head_dim});

    return out_proj.forward(attn_output);
}

SanaGLUMBConv::SanaGLUMBConv(
    int in_features, int hidden_features, bool use_fp4, Tensor::ScalarType dtype, Device device)
    : in_features(in_features), hidden_features(hidden_features),
      inverted_conv(in_features, hidden_features * 2, true, use_fp4, dtype, device),
      depth_conv(hidden_features * 2, true, dtype, device),
      point_conv(hidden_features, in_features, false, use_fp4, dtype, device) {
    registerChildren(inverted_conv, "inverted_conv")(depth_conv, "depth_conv")(point_conv, "point_conv");
}

Tensor SanaGLUMBConv::forward(Tensor x, int H, int W) {
    if (H <= 0 || W <= 0) {
        H = W = sqrt(x.shape[1]);
    }
    x = inverted_conv.forward_silu(x);
    x = x.view({x.shape[0], H, W, x.shape[-1]});
    debug("inverted_conv_output", x);
    x = depth_conv.forward(x);
    debug("depth_conv_output", x);
    x         = x.view({x.shape[0], H * W, x.shape[-1]});
    auto qact = point_conv.quantize(x, true);
    return point_conv.forward_quant(qact);
}

SanaLinearTransformerBlock::SanaLinearTransformerBlock(int hidden_size,
                                                       int intermediate_size,
                                                       int num_cross_attention_heads,
                                                       bool pag,
                                                       bool use_fp4,
                                                       Tensor::ScalarType dtype,
                                                       Device device)
    : hidden_size(hidden_size), num_cross_attention_heads(num_cross_attention_heads),
      attn(hidden_size, false, pag, use_fp4, dtype, device),
      cross_attn(num_cross_attention_heads, hidden_size / num_cross_attention_heads, use_fp4, dtype, device),
      ff(hidden_size, intermediate_size, use_fp4, dtype, device), norm1(hidden_size, 1e-6, false, dtype, device),
      norm2(hidden_size, 1e-6, false, dtype, device) {
    this->scale_shift_table = Tensor::allocate({6, hidden_size}, dtype, device);

    registerChildren(attn, "attn")(cross_attn, "cross_attn")(ff, "ff");

    registerParams(this->scale_shift_table, "scale_shift_table");
}

Tensor SanaLinearTransformerBlock::forward(Tensor hidden_states,
                                           Tensor encoder_hidden_states,
                                           Tensor timestep,
                                           Tensor cu_seqlens_img,
                                           Tensor cu_seqlens_txt,
                                           int H,
                                           int W,
                                           bool pag,
                                           bool cfg) {

    nvtxRangePushA("SanaLinearTransformerBlock");

    nvtxRangePushA("chunk");

    // Tensor ones = Tensor::ones({hidden_size}, Tensor::FP16, x.device());

    const int batch_size = timestep.shape[0];

    timestep = timestep.copy(timestep.device());
    timestep = timestep.view({batch_size, 6, hidden_size});

    kernels::mul_add_batch(timestep, {}, false, 0, this->scale_shift_table, false);
    debug("shifted_timestep", timestep);

    std::array<Tensor, 6> chunked;
    for (int i = 0; i < 6; i++) {
        chunked[i] = timestep.slice(1, i, i + 1);
    }
    auto &&[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] = chunked;
    // auto &&[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] = kernels::split_mod<6>(timestep);

    nvtxRangePop();

    {
        nvtxRangePushA("LinearAttention");

        Tensor residual           = hidden_states;
        Tensor norm_hidden_states = norm1.forward(hidden_states);
        kernels::mul_add_batch(norm_hidden_states, scale_msa, true, 1, shift_msa, true);
        debug("norm_hidden_states_la", norm_hidden_states);

        Tensor attn_output = pag ? attn.forward_pag(norm_hidden_states, cfg) : attn.forward(norm_hidden_states);
        debug("attn_output_la", attn_output);

        kernels::mul_add_batch(attn_output, gate_msa, true, 0, residual, true);

        hidden_states = attn_output;

        nvtxRangePop();
    }

    {
        nvtxRangePushA("CrossAttention");

        debug("norm_hidden_states_cross", hidden_states);
        Tensor attn_output = cross_attn.forward(hidden_states, encoder_hidden_states, cu_seqlens_img, cu_seqlens_txt);
        debug("attn_output_cross", attn_output);

        kernels::mul_add_batch(attn_output, {}, false, 0, hidden_states, true);

        hidden_states = attn_output;

        nvtxRangePop();
    }

    {
        nvtxRangePushA("Feed-forward");

        debug("hidden_states_ff", hidden_states);
        Tensor norm_hidden_states = norm2.forward(hidden_states);
        kernels::mul_add_batch(norm_hidden_states, scale_mlp, true, 1, shift_mlp, true);
        debug("norm_hidden_states_ff", norm_hidden_states);

        Tensor ff_output = ff.forward(norm_hidden_states, H, W);
        debug("ff_output", ff_output);

        kernels::mul_add_batch(ff_output, gate_mlp, true, 0, hidden_states, true);

        hidden_states = ff_output;

        nvtxRangePop();
    }

    nvtxRangePop();

    debug("hidden_states_out", hidden_states);

    return hidden_states;
}

SanaModel::SanaModel(SanaConfig config, Tensor::ScalarType dtype, Device device) : config(config) {
    const int inner_dim = config.num_attention_heads * config.attention_head_dim;
    for (int i = 0; i < config.num_layers; i++) {
        transformer_blocks.push_back(std::make_unique<SanaLinearTransformerBlock>(
            inner_dim,
            ceilDiv(int(round(config.expand_ratio * inner_dim)), 64) * 64,
            config.num_cross_attention_heads,
            std::find(config.pag_layers.begin(), config.pag_layers.end(), i) != config.pag_layers.end(),
            config.use_fp4,
            dtype,
            device));
        registerChildren(*transformer_blocks.back(), format("transformer_blocks.{}", i));
    }
}

Tensor SanaModel::forward(Tensor hidden_states,
                          Tensor encoder_hidden_states,
                          Tensor timestep,
                          Tensor cu_seqlens_img,
                          Tensor cu_seqlens_txt,
                          int H,
                          int W,
                          bool pag,
                          bool cfg,
                          bool skip_first_layer) {
    for (int i = (skip_first_layer ? 1 : 0); i < config.num_layers; i++) {
        auto &&block  = transformer_blocks[i];
        hidden_states = block->forward(hidden_states,
                                       encoder_hidden_states,
                                       timestep,
                                       cu_seqlens_img,
                                       cu_seqlens_txt,
                                       H,
                                       W,
                                       pag && std::find(config.pag_layers.begin(), config.pag_layers.end(), i) !=
                                                  config.pag_layers.end(),
                                       cfg);
    }
    return hidden_states;
}
