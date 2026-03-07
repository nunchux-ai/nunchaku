#include "chroma_additive_attention_ops.h"

#include "interop/torch.h"

#include "kernels/zgemm/attention_rank1bias.h"

#include <cmath>

namespace nunchaku::ops {

static void require_cuda(const torch::Tensor &t, const char *name) {
    if (!t.is_cuda()) {
        throw std::runtime_error(std::string(name) + " must be a CUDA tensor");
    }
}

static void require_contiguous(const torch::Tensor &t, const char *name) {
    if (!t.is_contiguous()) {
        throw std::runtime_error(std::string(name) + " must be contiguous");
    }
}

void chroma_additive_attention_packed_fp16(torch::Tensor q,
                                           torch::Tensor k,
                                           torch::Tensor v,
                                           torch::Tensor m,
                                           torch::Tensor out,
                                           double scale) {
    TorchOpContext ctx;

    require_cuda(q, "q");
    require_cuda(k, "k");
    require_cuda(v, "v");
    require_cuda(m, "m");
    require_cuda(out, "out");

    require_contiguous(q, "q");
    require_contiguous(k, "k");
    require_contiguous(v, "v");
    require_contiguous(m, "m");
    require_contiguous(out, "out");

    if (q.ndimension() != 4 || k.ndimension() != 4 || v.ndimension() != 4) {
        throw std::runtime_error("q/k/v must have shape (B, H, T, D)");
    }
    if (!q.sizes().equals(k.sizes()) || !q.sizes().equals(v.sizes())) {
        throw std::runtime_error("q/k/v shapes must match");
    }

    const int64_t B = q.size(0);
    const int64_t H = q.size(1);
    const int64_t T = q.size(2);
    const int64_t D = q.size(3);

    if (D != 128) {
        throw std::runtime_error("chroma_additive_attention_packed_fp16 requires head_dim=128");
    }
    if (m.ndimension() != 2 || m.size(0) != B || m.size(1) != T) {
        throw std::runtime_error("m must have shape (B, T_pad)");
    }
    if (out.ndimension() != 3 || out.size(0) != B || out.size(1) != T || out.size(2) != H * D) {
        throw std::runtime_error("out must have shape (B, T_pad, H*D)");
    }

    if (q.scalar_type() != torch::kFloat16 || k.scalar_type() != torch::kFloat16 || v.scalar_type() != torch::kFloat16) {
        throw std::runtime_error("q/k/v dtype must be fp16 (packed attention layout)");
    }
    if (m.scalar_type() != torch::kFloat16) {
        throw std::runtime_error("m dtype must be fp16");
    }
    if (!(out.scalar_type() == torch::kFloat16 || out.scalar_type() == torch::kBFloat16)) {
        throw std::runtime_error("out dtype must be fp16 or bf16");
    }

    if (T <= 0 || (T % 128) != 0 || (T % 32) != 0) {
        throw std::runtime_error("T_pad must be a positive multiple of 128 (and 32)");
    }
    if (scale == 0.0) {
        scale = std::pow(double(D), -0.5);
    }

    nunchaku::kernels::attention_fp16_rank1bias(
        from_torch(q), from_torch(k), from_torch(v), from_torch(m), from_torch(out), float(scale)
    );
}

} // namespace nunchaku::ops
