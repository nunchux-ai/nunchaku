#include "misc_kernels_impl.cuh"
#include "misc_kernels.h"
#include "dispatch_utils.h"

namespace nunchaku::kernels {

Tensor add(Tensor a, Tensor b) {
    assert(a.shape.dataExtent == b.shape.dataExtent);
    assert(a.dtype() == b.dtype());
    assert(a.is_contiguous());
    assert(b.is_contiguous());

    int threadsPerBlock = 1024;

    auto stream = getCurrentCUDAStream();

    Tensor out = Tensor::empty_like(a);

    constexpr int unroll = 8;
    const bool can_vectorize = (a.numel() % unroll == 0) &&
                               ((uintptr_t)a.data_ptr() % (a.scalar_size() * unroll) == 0) &&
                               ((uintptr_t)b.data_ptr() % (b.scalar_size() * unroll) == 0);

    dispatch(out.scalar_type(), [&]<typename scalar_t>() {
        if (can_vectorize) {
            int blocksPerGrid = (a.numel() + threadsPerBlock * unroll - 1) / (threadsPerBlock * unroll);
            add_kernel<scalar_t, unroll><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), out.numel());
        } else {
            int blocksPerGrid = (a.numel() + threadsPerBlock - 1) / threadsPerBlock;
            add_kernel<scalar_t, 1><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), out.numel());
        }
    });

    return out;
}

void mul_add(Tensor x, Tensor scale, Tensor bias) {
    // assert(scale.shape.data == bias.shape.data);
    // FIXME FIXME
    assert(x.numel() % scale.numel() == 0);
    assert(x.numel() % bias.numel() == 0);
    assert(x.dtype() == scale.dtype());
    assert(x.dtype() == bias.dtype());

    constexpr int unroll = 8;

    assert((uintptr_t)x.data_ptr() % (x.scalar_size() * unroll) == 0);
    assert(!scale.valid() || (uintptr_t)scale.data_ptr() % (x.scalar_size() * unroll) == 0);
    assert((uintptr_t)bias.data_ptr() % (x.scalar_size() * unroll) == 0);

    assert(x.numel() % unroll == 0);
    assert(!scale.valid() || scale.numel() % unroll == 0);
    assert(bias.numel() % unroll == 0);

    int threadsPerBlock = 1024;
    int blocksPerGrid   = (x.numel() + threadsPerBlock * unroll - 1) / (threadsPerBlock * unroll);

    auto stream = getCurrentCUDAStream();

    dispatch(x.scalar_type(), [&]<typename scalar_t>() {
        if (scale.valid()) {
            mul_add_kernel<scalar_t, unroll, false>
                <<<blocksPerGrid, threadsPerBlock, 0, stream>>>(x.data_ptr<scalar_t>(),
                                                                scale.data_ptr<scalar_t>(),
                                                                bias.data_ptr<scalar_t>(),
                                                                0,
                                                                x.numel(),
                                                                scale.numel(),
                                                                bias.numel(),
                                                                0,
                                                                0,
                                                                0);
        } else {
            mul_add_kernel<scalar_t, unroll, true><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                x.data_ptr<scalar_t>(), nullptr, bias.data_ptr<scalar_t>(), 0, x.numel(), 1, bias.numel(), 0, 0, 0);
        }
    });
}

void mul_add_batch(Tensor x, Tensor scale, bool batch_scale, double scale_shift, Tensor bias, bool batch_bias) {

    const int batch_size = x.shape[0];
    assert(!batch_scale || scale.shape[0] == batch_size);
    assert(!batch_bias || bias.shape[0] == batch_size);

    const int numel       = x.numel() / batch_size;
    const int numel_scale = scale.valid() ? (scale.numel() / (batch_scale ? batch_size : 1)) : 1;
    const int numel_bias  = bias.numel() / (batch_bias ? batch_size : 1);

    assert(numel % numel_scale == 0);
    assert(numel % numel_bias == 0);
    assert(!scale.valid() || x.dtype() == scale.dtype());
    assert(x.dtype() == bias.dtype());

    constexpr int unroll = 8;

    assert((uintptr_t)x.data_ptr() % (x.scalar_size() * unroll) == 0);
    assert(!scale.valid() || (uintptr_t)scale.data_ptr() % (x.scalar_size() * unroll) == 0);
    assert((uintptr_t)bias.data_ptr() % (x.scalar_size() * unroll) == 0);

    assert(numel % unroll == 0);
    assert(!scale.valid() || numel_scale % unroll == 0);
    assert(numel_bias % unroll == 0);

    int threadsPerBlock = 1024;
    dim3 grid(ceilDiv(numel, threadsPerBlock * unroll), batch_size);

    auto stream = getCurrentCUDAStream();

    dispatch(x.scalar_type(), [&]<typename scalar_t>() {
        if (scale.valid()) {
            mul_add_kernel<scalar_t, unroll, false>
                <<<grid, threadsPerBlock, 0, stream>>>(x.data_ptr<scalar_t>(),
                                                       scale.data_ptr<scalar_t>(),
                                                       bias.data_ptr<scalar_t>(),
                                                       (scalar_t)scale_shift,
                                                       numel,
                                                       numel_scale,
                                                       numel_bias,
                                                       x.stride(0),
                                                       batch_scale ? scale.stride(0) : 0,
                                                       batch_bias ? bias.stride(0) : 0);
        } else {
            mul_add_kernel<scalar_t, unroll, true>
                <<<grid, threadsPerBlock, 0, stream>>>(x.data_ptr<scalar_t>(),
                                                       nullptr,
                                                       bias.data_ptr<scalar_t>(),
                                                       (scalar_t)scale_shift,
                                                       numel,
                                                       1,
                                                       numel_bias,
                                                       x.stride(0),
                                                       0,
                                                       batch_bias ? bias.stride(0) : 0);
        }
    });
}

Tensor embedding(Tensor input_id, Tensor lookup) {
    assert(input_id.dtype() == Tensor::INT32);
    assert(lookup.ndims() == 2);

    auto shapeOut = input_id.shape;
    shapeOut.dataExtent.push_back(lookup.shape[-1]);

    auto stream = getCurrentCUDAStream();

    Tensor out = Tensor::empty(shapeOut, lookup.scalar_type(), input_id.device());

    dispatch(out.scalar_type(), [&]<typename scalar_t>() {
        EmbeddingKernel<<<input_id.numel(), std::min(lookup.shape[-1], 1024), 0, stream>>>(
            input_id.data_ptr<int32_t>(), out.data_ptr<scalar_t>(), lookup.data_ptr<scalar_t>(), lookup.shape[-1]);
    });

    return out;
}

Tensor argmax_sample(Tensor logits) {
    assert(logits.ndims() == 2);

    auto stream = getCurrentCUDAStream();

    Tensor out = Tensor::empty({logits.shape[0]}, Tensor::INT32, logits.device());

    dispatch(logits.scalar_type(), [&]<typename scalar_t>() {
        argmax_sample_kernel<<<logits.shape[0], std::min(logits.shape[1], 1024), 0, stream>>>(
            logits.data_ptr<scalar_t>(), out.data_ptr<int32_t>(), logits.shape[1]);
    });

    return out;
}

void splitqkv(Tensor qkv, Tensor q, Tensor k, Tensor v) {
    // FIXME FIXME
    // assert(qkv.shape[0] == q.shape[0]);
    // assert(qkv.shape[0] == k.shape[0]);
    // assert(qkv.shape[0] == v.shape[0]);

    auto stream = getCurrentCUDAStream();

    int dim_q = q.shape[-1] * q.shape[-2];
    int dim_k = k.shape[-1] * k.shape[-2];
    int dim_v = v.shape[-1] * v.shape[-2];

    assert(dim_k == dim_v);
    assert(dim_q + dim_k + dim_v == qkv.shape[-1]);

    int num_tokens = qkv.numel() / qkv.shape[-1];

    dispatch(qkv.scalar_type(), [&]<typename scalar_t>() {
        splitqkv_kernel<<<num_tokens, std::min(qkv.shape[-1], 1024), 0, stream>>>(qkv.data_ptr<scalar_t>(),
                                                                                  q.data_ptr<scalar_t>(),
                                                                                  k.data_ptr<scalar_t>(),
                                                                                  v.data_ptr<scalar_t>(),
                                                                                  dim_q,
                                                                                  dim_k);
    });
}

template<size_t N>
std::array<Tensor, N> split_mod(Tensor input) {
    assert(input.shape[-1] % N == 0);

    int threadsPerBlock = 1024;
    int blocksPerGrid   = (input.numel() + threadsPerBlock - 1) / threadsPerBlock;

    auto stream = getCurrentCUDAStream();

    auto shapeOut = TensorShape(input.shape.dataExtent);
    shapeOut[-1] /= N;

    std::array<Tensor, N> out;
    for (int k = 0; k < N; k++) {
        out[k] = Tensor::empty(shapeOut, input.scalar_type(), input.device());
    }

    dispatch(input.scalar_type(), [&]<typename scalar_t>() {
        std::array<scalar_t *, N> outPtr;
        for (int k = 0; k < N; k++) {
            outPtr[k] = out[k].template data_ptr<scalar_t>();
        }
        split_mod_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            input.data_ptr<scalar_t>(), outPtr, input.numel());
    });

    return out;
}

Tensor quant_static(Tensor x, float scale) {
    Tensor out = Tensor::empty(x.shape, Tensor::INT8, x.device());

    constexpr int unroll = 8;

    assert((uintptr_t)x.data_ptr() % (x.scalar_size() * unroll) == 0);

    int threadsPerBlock = 1024;
    int blocksPerGrid   = (x.numel() + threadsPerBlock * unroll - 1) / (threadsPerBlock * unroll);

    auto stream = getCurrentCUDAStream();

    dispatch(x.scalar_type(), [&]<typename scalar_t>() {
        quant_kernel_static<scalar_t, unroll><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            x.data_ptr<scalar_t>(), out.data_ptr<int8_t>(), (scalar_t)scale, x.numel());
    });

    return out;
}

Tensor quant_static_fuse_gelu(Tensor x, float scale) {
    Tensor out = Tensor::empty(x.shape, Tensor::INT8, x.device());

    constexpr int unroll = 8;

    assert((uintptr_t)x.data_ptr() % (x.scalar_size() * unroll) == 0);

    int threadsPerBlock = 1024;
    int blocksPerGrid   = (x.numel() + threadsPerBlock * unroll - 1) / (threadsPerBlock * unroll);

    auto stream = getCurrentCUDAStream();

    dispatch(x.scalar_type(), [&]<typename scalar_t>() {
        quant_kernel_static_fuse_gelu<scalar_t, unroll><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            x.data_ptr<scalar_t>(), out.data_ptr<int8_t>(), (scalar_t)scale, x.numel());
    });

    return out;
}

void cast(Tensor input, Tensor output) {
    assert(input.is_contiguous());
    assert(output.is_contiguous());
    assert(input.shape.dataExtent == output.shape.dataExtent);

    if (input.data_ptr() == output.data_ptr()) {
        assert(input.scalar_size() == output.scalar_size());
    }

    auto stream = getCurrentCUDAStream();

    dispatch(input.scalar_type(), [&]<typename input_t>() {
        dispatch(output.scalar_type(), [&]<typename output_t>() {
            constexpr int unroll = 16 / std::max(sizeof(input_t), sizeof(output_t));

            int threadsPerBlock = 1024;
            int blocksPerGrid   = (int)ceilDiv<int64_t>(input.numel(), threadsPerBlock * unroll);

            cast_kernel<input_t, output_t, unroll><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                input.data_ptr<input_t>(), output.data_ptr<output_t>(), input.numel());

            checkCUDA(cudaGetLastError());
        });
    });
}

Tensor topk(Tensor x, int k) {
    constexpr int MAXK = 64 + 4;

    const int N     = x.shape[-1];
    const int batch = x.numel() / N;

    assert(k <= N);
    assert(k <= MAXK);

    auto outShape = TensorShape(x.shape.dataExtent);
    outShape[-1]  = k;
    outShape.dataStride.clear();

    Tensor out = Tensor::empty(outShape, Tensor::INT32, x.device());

    auto stream = getCurrentCUDAStream();

    dispatchVal(k, std::make_integer_sequence<int, MAXK + 1>(), [&]<int K>() {
        if constexpr (K == 0) {
            assert(false);
            return;
        }
        if constexpr (K > 0) {
            dispatch(x.scalar_type(), [&]<typename scalar_t>() {
                topk_kernel<scalar_t, K><<<ceilDiv(batch, 32), 32, 0, stream>>>(
                    x.data_ptr<scalar_t>(), out.data_ptr<int>(), N, x.stride(-2), batch);
                checkCUDA(cudaGetLastError());
            });
        }
    });

    return out;
}

template std::array<Tensor, 2> split_mod<2>(Tensor input);
template std::array<Tensor, 3> split_mod<3>(Tensor input);
template std::array<Tensor, 4> split_mod<4>(Tensor input);
template std::array<Tensor, 5> split_mod<5>(Tensor input);
template std::array<Tensor, 6> split_mod<6>(Tensor input);

}; // namespace nunchaku::kernels
