#include "Linear.h"
#include "kernels/zgemm/zgemm.h"
#include "kernels/gemm_f16.h"
#include "kernels/misc_kernels.h"
#include "kernels/awq/gemv_awq.h"
#include "kernels/dwconv.h"

#include <nvtx3/nvToolsExt.h>

using namespace nunchaku;

GEMM_F16::GEMM_F16(int in_features, int out_features, bool use_bias, Tensor::ScalarType dtype, Device device)
    : in_features(in_features), out_features(out_features) {
    this->weight = Tensor::allocate({out_features, in_features}, dtype, device);
    this->bias   = use_bias ? Tensor::allocate({out_features}, dtype, device) : Tensor{};

    registerParams(weight, "weight", ParamFlags::LazyLoad)(bias, "bias");
}

Tensor GEMM_F16::forward(Tensor x) {
    Tensor out = gemm_f16(x, this->weight, {}, this->bias, 1.0f);
    return out;
}

GEMV_AWQ::GEMV_AWQ(int in_features, int out_features, bool use_bias, Tensor::ScalarType dtype, Device device)
    : in_features(in_features), out_features(out_features), group_size(64), lora_rank(0), lora_scale(1.0f),
      device(device) {
    this->qweight = Tensor::allocate({out_features / 4, ceilDiv(in_features, 8) * 4}, Tensor::INT32, device);
    this->wscales = Tensor::allocate({ceilDiv(in_features, group_size), out_features}, dtype, device);
    this->wzeros  = Tensor::allocate({ceilDiv(in_features, group_size), out_features}, dtype, device);
    this->bias    = use_bias ? Tensor::allocate({out_features}, dtype, device) : Tensor{};

    // !!! lora layout is different from w4a4 !!!
    this->lora_down = Tensor::allocate({lora_rank, in_features}, dtype, device, true);
    this->lora_up   = Tensor::allocate({out_features, lora_rank}, dtype, device, true);

    registerParams(qweight, "qweight", ParamFlags::LazyLoad)(wscales, "wscales")(wzeros, "wzeros")(bias, "bias")(
        lora_down, "lora_down", ParamFlags::Optional)(lora_up, "lora_up", ParamFlags::Optional);
}

void GEMV_AWQ::loadParam(std::string key, Tensor &dst, Tensor src) {
    if (key == "lora_down" || key == "lora_up") {
        assert(src.ndims() == 2);
        if (dst.shape.dataExtent != src.shape.dataExtent) {
            dst = Tensor::allocate(src.shape.dataExtent, dst.scalar_type(), this->device);
            Module::loadParam(key, dst, src);
            if (key == "lora_down") {
                const int new_rank = dst.shape[0];
                this->lora_rank    = new_rank;
            }
        } else {
            Module::loadParam(key, dst, src);
        }
    } else {
        Module::loadParam(key, dst, src);
    }
}

Tensor GEMV_AWQ::forward(Tensor x) {
    debug("x", x);

    const int M = (int)x.numel() / x.shape[-1];
    Tensor out  = gemv_awq(x, this->qweight, this->wscales, this->wzeros, M, out_features, in_features, group_size);
    if (bias.valid()) {
        // TODO: batch
        // assert(out.numel() == bias.numel());
        // out = kernels::add(out, bias.view(out.shape.dataExtent));
        kernels::mul_add_batch(out, {}, false, 0.0, bias, false);
    }

    debug("out_before_lora", out);

    if (this->lora_rank > 0) {
        Tensor lora_act = gemm_f16(x, this->lora_down, {}, {}, 1.0f);
        debug("lora_act", lora_act);

        Tensor lora_out = gemm_f16(lora_act, this->lora_up, {}, {}, this->lora_scale);
        debug("lora_out", lora_out);

        out = kernels::add(out, lora_out);
    }

    debug("out", out);

    return out;
}

#define NO_LORA_FUSION 0

GEMM_W4A4::GEMM_W4A4(
    int in_features, int out_features, bool bias, bool use_fp4, Tensor::ScalarType dtype, Device device)
    : in_features(in_features), out_features(out_features), in_features_pad(ceilDiv(in_features, 128) * 128),
      out_features_pad(ceilDiv(out_features, 128) * 128), use_fp4(use_fp4), lora_rank(0), dtype(dtype), device(device) {
    this->qweight = Tensor::allocate({out_features_pad, in_features_pad / 2}, Tensor::INT8, device, true);
    if (use_fp4) {
        this->wscales = Tensor::allocate({in_features_pad / 16, out_features_pad}, Tensor::FP8_E4M3, device, true);
    } else {
        this->wscales = Tensor::allocate({in_features_pad / 64, out_features_pad}, dtype, device, true);
    }

    this->bias = bias ? Tensor::allocate({out_features_pad}, dtype, device, true) : Tensor{};

    this->lora_down = Tensor::allocate({in_features_pad, lora_rank}, dtype, device, true);
    this->lora_up   = Tensor::allocate({out_features_pad, lora_rank}, dtype, device, true);

    // TODO: smooth factor in non-Lora fusion
    this->smooth = Tensor::allocate({in_features_pad}, dtype, device, true);

    // FIXME: reset wtscale and wcscales to default values when reloading the weights
    this->wtscale                    = Tensor::allocate({1}, Tensor::FP32, Device::cpu(), true);
    *this->wtscale.data_ptr<float>() = 1.0f;

    this->wcscales = Tensor::allocate({0}, dtype, device, true);

    registerParams(qweight, "qweight", ParamFlags::LazyLoad)(wscales, "wscales")(this->bias, "bias")(
        lora_down, "lora_down", ParamFlags::Optional)(lora_up, "lora_up", ParamFlags::Optional)(smooth, "smooth")(
        wtscale, "wtscale", ParamFlags::Optional)(wcscales, "wcscales", ParamFlags::Optional);

#if NO_LORA_FUSION
    checkCUBLAS(cublasCreate(&handle));
#endif
}

void GEMM_W4A4::loadParam(std::string key, Tensor &dst, Tensor src) {
    if (key == "lora_down" || key == "lora_up") {
        assert(src.ndims() == 2);
        if (dst.shape.dataExtent != src.shape.dataExtent) {
            dst = Tensor::allocate(src.shape.dataExtent, dst.scalar_type(), this->device);
            Module::loadParam(key, dst, src);
            this->lora_rank = dst.shape[1];
            this->lora_scales.resize(ceilDiv(this->lora_rank, 16), 1.0f);
        } else {
            Module::loadParam(key, dst, src);
        }
    } else if (key == "wcscales") {
        assert(src.ndims() == 1);
        assert(src.shape[0] == out_features_pad);
        dst = Tensor::allocate(src.shape.dataExtent, dst.scalar_type(), this->device);
        Module::loadParam(key, dst, src);
    } else if (key == "wtscale") {
        assert(src.numel() == 1);
        if (src.dtype() == Tensor::BF16) {
            *dst.data_ptr<float>() = float(*src.data_ptr<__nv_bfloat16>());
        } else if (src.dtype() == Tensor::FP16) {
            *dst.data_ptr<float>() = float(*src.data_ptr<half>());
        } else if (src.dtype() == Tensor::FP32) {
            Module::loadParam(key, dst, src);
        } else {
            assert(false);
        }
    } else {
        Module::loadParam(key, dst, src);
    }
}

Tensor GEMM_W4A4::forward(Tensor x) {
    return std::get<Tensor>(this->forward(x, FuseOptions::EMPTY, nullptr));
}

Tensor GEMM_W4A4::forward_silu(Tensor x) {
    return std::get<Tensor>(this->forward(x, FuseOptions::SILU, nullptr));
}

std::variant<Tensor, GEMM_W4A4::QuantizedActivation>
GEMM_W4A4::forward(Tensor x, FuseOptions fuse, GEMM_W4A4 *nextGEMM) {
    return forward_quant(quantize(x, false), fuse, nextGEMM);
}

void GEMM_W4A4::forward(Tensor x,
                        Tensor out,
                        Tensor pool,
                        Tensor norm_q,
                        Tensor norm_k,
                        Tensor rotary_emb,
                        Tensor out_q,
                        Tensor out_k,
                        Tensor out_v,
                        int numTokens) {
    QuantizedActivation qact = quantize(x, false);

#if !NO_LORA_FUSION

#if 0
    Tensor dummy = Tensor::empty_like(qact.lora_act);
    dummy.zero_();

    gemm_w4a4(qact.act, qweight, out, {}, qact.ascales, wscales, {}, pool, dummy, this->lora_up, {}, {}, norm_q, norm_k, rotary_emb, this->bias, {}, qact.is_unsigned);
    debug("gemm.nolora.out", out);
#endif

    kernels::gemm_w4a4(qact.act,
                       qweight,
                       out,
                       {},
                       qact.ascales,
                       wscales,
                       {},
                       pool,
                       qact.lora_act,
                       this->lora_up,
                       {},
                       {},
                       norm_q,
                       norm_k,
                       rotary_emb,
                       this->bias,
                       {},
                       {},
                       {},
                       qact.is_unsigned,
                       false,
                       this->lora_scales,
                       false,
                       use_fp4,
                       *this->wtscale.data_ptr<float>(),
                       wcscales.numel() > 0 ? wcscales : Tensor{},
                       out_q,
                       out_k,
                       out_v,
                       numTokens);

    debug("gemm.out", out);
#else
    const int M = (int)qact.act.numel() / qact.act.shape[-1];

    kernels::gemm_w4a4(qact.act,
                       qweight,
                       out,
                       {},
                       qact.ascales,
                       wscales,
                       {},
                       pool,
                       {},
                       {},
                       {},
                       {},
                       norm_q,
                       norm_k,
                       rotary_emb,
                       this->bias,
                       {},
                       qact.is_unsigned,
                       false,
                       this->lora_scales);

    nvtxRangePushA("LoraUp");

    static const half one  = 1.0;
    static const half zero = 0.0;
    // lora_up: [M, R] * [OC, R] => [M, OC]
    // cublas view: [OC, R] * [M, R]^T
    checkCUBLAS(cublasHgemm(handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            this->out_features,
                            M,
                            this->lora_rank,
                            &one,
                            this->lora_up.data_ptr<half>(),
                            this->lora_rank,
                            qact.lora_act.data_ptr<half>(),
                            this->lora_rank,
                            &one,
                            out.data_ptr<half>(),
                            this->out_features));

    nvtxRangePop();
#endif
}

std::variant<Tensor, GEMM_W4A4::QuantizedActivation>
GEMM_W4A4::forward_quant(QuantizedActivation qact, FuseOptions fuse, GEMM_W4A4 *nextGEMM) {
    Tensor out;
    QuantizedActivation qout;

    Tensor next_lora;
    Tensor next_smooth;

    const int M = (int)qact.act.numel() / qact.act.shape[-1];

    if (fuse == FuseOptions::EMPTY || fuse == FuseOptions::SILU) {
        // auto shape = TensorShape(qact.act.shape.dataExtent);
        // shape[-1] = out_features;
        auto shape = TensorShape(qact.actShape.dataExtent);
        shape[-1]  = out_features;
        out        = Tensor::allocate(shape, dtype, device);
    } else {
        qout.act = Tensor::allocate({M, out_features_pad / 2}, Tensor::INT8, device);
        if (use_fp4) {
            qout.ascales = Tensor::allocate({out_features_pad / 16, M}, Tensor::FP8_E4M3, device);
        } else {
            qout.ascales = Tensor::allocate({out_features_pad / 64, M}, dtype, device);
        }
        qout.lora_act    = Tensor::allocate({M, lora_rank}, Tensor::FP32, device);
        qout.is_unsigned = !use_fp4;
        qout.actShape    = qact.actShape;

        next_lora   = nextGEMM->lora_down;
        next_smooth = nextGEMM->smooth;
    }

#if !NO_LORA_FUSION

#if 0
    Tensor dummy = Tensor::empty_like(qact.lora_act);
    dummy.zero_();

    gemm_w4a4(qact.act, qweight, out, qout.act, qact.ascales, wscales, qout.ascales, {}, dummy, this->lora_up, next_lora, qout.lora_act, {}, {}, {}, this->bias, next_smooth, qact.is_unsigned);

    if (fuse == FuseOptions::EMPTY) {
        debug("gemm.nolora.out", out);
    } else {
        debug("gemm.nolora.qout", qout.act);
        debug("gemm.nolora.oscales", qout.ascales);
        debug("gemm.nolora.lora_act_out", qout.lora_act);
    }
#endif

    kernels::gemm_w4a4(qact.act,
                       qweight,
                       out,
                       qout.act,
                       qact.ascales,
                       wscales,
                       qout.ascales,
                       {},
                       qact.lora_act,
                       this->lora_up,
                       next_lora,
                       qout.lora_act,
                       {},
                       {},
                       {},
                       this->bias,
                       next_smooth,
                       {},
                       {},
                       qact.is_unsigned,
                       qout.is_unsigned,
                       this->lora_scales,
                       fuse == FuseOptions::SILU,
                       use_fp4,
                       *this->wtscale.data_ptr<float>(),
                       wcscales.numel() > 0 ? wcscales : Tensor{},
                       {},
                       {},
                       {},
                       0);

    if (fuse == FuseOptions::EMPTY || fuse == FuseOptions::SILU) {
        debug("gemm.out", out);
    } else {
        debug("gemm.qout", qout.act);
        debug("gemm.oscales", qout.ascales);
        debug("gemm.lora_act_out", qout.lora_act);
    }

#else
    if (!out.valid()) {
        auto shape = TensorShape(qact.act.shape.dataExtent);
        shape[-1]  = out_features;
        out        = Tensor::allocate(shape, Tensor::FP16, qweight.device());
    }

    kernels::gemm_w4a4(qact.act,
                       qweight,
                       out,
                       qout.act,
                       qact.ascales,
                       wscales,
                       qout.ascales,
                       {},
                       {},
                       {},
                       {},
                       {},
                       {},
                       {},
                       {},
                       this->bias,
                       next_smooth,
                       qact.is_unsigned,
                       qout.is_unsigned,
                       this->lora_scales);

    nvtxRangePushA("LoraUp");

    static const half one  = 1.0;
    static const half zero = 0.0;

    // lora_up: [M, R] * [OC, R]^T => [M, OC]
    // cublas view: [R, OC]^T * [R, M] => [OC, M]
    // lora_up layout wrong?
    checkCUBLAS(cublasHgemm(handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            this->out_features,
                            M,
                            this->lora_rank,
                            &one,
                            this->lora_up.data_ptr<half>(),
                            this->lora_rank,
                            qact.lora_act.data_ptr<half>(),
                            this->lora_rank,
                            &one,
                            out.data_ptr<half>(),
                            this->out_features));

    nvtxRangePop();

    if (fuse == FuseOptions::GELU_QUANT) {
        nvtxRangePushA("LoraDown");
        // IC is for next lora (OC of this layer)
        // lora_down: [M, IC] * [IC, R] => [M, R]
        // cublas view: [R, IC] * [IC, M] => [R, M]
        checkCUBLAS(cublasHgemm(handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                this->lora_rank,
                                M,
                                this->out_features,
                                &one,
                                next_lora.data_ptr<half>(),
                                this->lora_rank,
                                out.data_ptr<half>(),
                                this->out_features,
                                &zero,
                                qout.lora_act.data_ptr<half>(),
                                this->lora_rank));

        out = {};

        nvtxRangePop();
    }

#endif
    if (out.valid()) {
        return out;
    }
    return qout;
}

Tensor GEMM_W4A4::forward_quant(QuantizedActivation qact) {
    return std::get<Tensor>(this->forward_quant(qact, FuseOptions::EMPTY, nullptr));
}

GEMM_W4A4::QuantizedActivation GEMM_W4A4::quantize(Tensor x, bool fuse_glu) {
    const int actualM = x.numel() / x.shape[-1];
    const int M       = ceilDiv(actualM, 256) * 256;

    // auto shape = TensorShape(x.shape.dataExtent);
    // shape[-1] = in_features / 2;

    QuantizedActivation qact;
    qact.act = Tensor::allocate({M, in_features_pad / 2}, Tensor::INT8, device);
    if (use_fp4) {
        qact.ascales = Tensor::allocate({in_features_pad / 16, M}, Tensor::FP8_E4M3, device);
    } else {
        qact.ascales = Tensor::allocate({in_features_pad / 64, M}, dtype, device);
    }
    qact.lora_act    = Tensor::allocate({M, lora_rank}, Tensor::FP32, device);
    qact.is_unsigned = false;
    qact.actShape    = x.shape.dataExtent;

#if !NO_LORA_FUSION
    debug("quantize.x", x);
    debug("quantize.smooth", this->smooth);

    kernels::quantize_w4a4_act_fuse_lora(
        x, qact.act, qact.ascales, this->lora_down, qact.lora_act, this->smooth, false, fuse_glu, use_fp4);

    debug("quantize.qact", qact.act);
    debug("quantize.ascales", qact.ascales);
    debug("quantize.lora_act", qact.lora_act);
#else
    static const half one  = 1.0;
    static const half zero = 0.0;

    nvtxRangePushA("LoraDown");

    // lora_down: [M, IC] * [IC, R] => [M, R]
    // cublas view: [R, IC] * [IC, M]
    checkCUBLAS(cublasHgemm(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            this->lora_rank,
                            M,
                            this->in_features,
                            &one,
                            lora_down.data_ptr<half>(),
                            this->lora_rank,
                            x.data_ptr<half>(),
                            this->in_features,
                            &zero,
                            qact.lora_act.data_ptr<half>(),
                            this->lora_rank));

    nvtxRangePop();

    kernels::quantize_w4a4_act(x, qact.act, qact.ascales);

#endif

    return qact;
}

GEMM_W8A8::GEMM_W8A8(int in_features, int out_features, bool bias, Tensor::ScalarType dtype, Device device)
    : in_features(in_features), out_features(out_features), dtype(dtype) {
    this->qweight = Tensor::allocate({out_features, in_features}, Tensor::INT8, device);
    this->wscales = Tensor::allocate({out_features}, dtype, device);
    this->bias    = bias ? Tensor::allocate({out_features}, dtype, device, true) : Tensor{};

    registerParams(qweight, "qweight", ParamFlags::LazyLoad)(wscales, "wscales")(this->bias, "bias");
}

GEMM_W8A8::QuantizedActivation GEMM_W8A8::quantize(Tensor x, bool fuse_glu) {
    QuantizedActivation qact;
    auto qshape = x.shape;
    if (fuse_glu) {
        qshape[-1] /= 2;
    }
    qact.act     = Tensor::allocate(qshape, Tensor::INT8, x.device());
    qact.ascales = Tensor::allocate({(int)x.numel() / x.shape[-1]}, this->dtype, x.device());

    debug("quantize.x", x);

    kernels::quantize_w8a8_act(x, qact.act, qact.ascales, fuse_glu);

    debug("quantize.qact", qact.act);
    debug("quantize.ascales", qact.ascales);

    return qact;
}

Tensor GEMM_W8A8::forward_quant(QuantizedActivation qact) {
    auto shape = TensorShape(qact.act.shape.dataExtent);
    shape[-1]  = out_features;
    Tensor out = Tensor::allocate(shape, this->dtype, qact.act.device());
    kernels::gemm_w8a8(qact.act, this->qweight, out, qact.ascales, this->wscales, this->bias);

    debug("gemm.out", out);
    return out;
}

DWCONV::DWCONV(int in_features, bool use_bias, Tensor::ScalarType dtype, Device device) : in_features(in_features) {
    this->weight = Tensor::allocate({in_features, 3, 3, 1}, dtype, device);
    this->bias   = use_bias ? Tensor::allocate({in_features}, dtype, device) : Tensor{};

    registerParams(this->weight, "weight")(this->bias, "bias");
}

Tensor DWCONV::forward(Tensor x) {
    return dwconv_f16(x, this->weight, {}, this->bias);
}
