#include "reduction_utils.cuh"
#include <array>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "utils.cuh"
#include "activation_kernels_impl.cuh"

namespace nunchaku::kernels {

template<typename T, int unroll>
struct alignas(sizeof(T) * unroll) Tvec {
    T data[unroll];
};

template<typename T, int unroll>
__global__ void add_kernel(T *a, T *b, T *c, size_t length) {
    int i = (threadIdx.x + blockIdx.x * blockDim.x) * unroll;
    if (i >= length) return;

    using Tvec = nunchaku::kernels::Tvec<T, unroll>;

    Tvec ra = *reinterpret_cast<Tvec *>(&a[i]);
    Tvec rb = *reinterpret_cast<Tvec *>(&b[i]);
    Tvec rc;

#pragma unroll
    for (int k = 0; k < unroll; k++) {
        rc.data[k] = ra.data[k] + rb.data[k];
    }

    *reinterpret_cast<Tvec *>(&c[i]) = rc;
}

template<typename T, int unroll, bool no_scale>
__global__ void mul_add_kernel(T *x,
                               T *scale,
                               T *bias,
                               T scale_shift,
                               size_t length,
                               int mod_scale,
                               int mod_bias,
                               int64_t batch_stride_x,
                               int64_t batch_stride_scale,
                               int64_t batch_stride_bias) {
    const int batch_id = blockIdx.y;
    int thread         = threadIdx.x + blockIdx.x * blockDim.x;
    int i              = thread * unroll;
    int i_scale        = i % mod_scale;
    int i_bias         = i % mod_bias;

    if (i >= length) {
        return;
    }

    using Tvec = nunchaku::kernels::Tvec<T, unroll>;

    Tvec rx     = *reinterpret_cast<Tvec *>(&x[i + batch_stride_x * batch_id]);
    Tvec rscale = *reinterpret_cast<Tvec *>(&scale[i_scale + batch_stride_scale * batch_id]);
    Tvec rbias  = *reinterpret_cast<Tvec *>(&bias[i_bias + batch_stride_bias * batch_id]);

#pragma unroll
    for (int k = 0; k < unroll; k++) {
        T tmp;
        if constexpr (no_scale) {
            tmp = rx.data[k] + rbias.data[k];
        } else {
            tmp = rx.data[k] * (rscale.data[k] + scale_shift) + rbias.data[k];
        }
        if constexpr (std::is_same_v<T, half>) {
            tmp = __hmin(tmp, (half)65504);
            tmp = __hmax(tmp, (half)-65504);
        }
        rx.data[k] = tmp;
    }

    *reinterpret_cast<Tvec *>(&x[i + batch_stride_x * batch_id]) = rx;

    // #pragma unroll
    //     for (int k = 0; k < unroll; k++) {
    //         // assert(i < length);
    //         x[i] = x[i] * scale[i_scale] + bias[i_bias];
    //         i++;
    //         i_scale++;
    //         i_bias++;
    //         // assert(i_scale < mod_scale);
    //         // assert(i_bias < mod_bias);
    //     }
}

template<typename T, size_t N>
__global__ void split_mod_kernel(T *input, std::array<T *, N> output, size_t length) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i * N < length) {
#pragma unroll
        for (int k = 0; k < N; k++) {
            output[k][i] = input[i * N + k];
        }
    }
}

template<typename T>
__global__ void
EmbeddingKernel(int32_t *__restrict__ input_id, T *__restrict__ output, T *__restrict__ lookup, int embed_dim) {
    int i = blockIdx.x;

    int32_t token_id     = input_id[i];
    T *output_sample_ptr = output + i * embed_dim;
    T *target_embed      = lookup + token_id * embed_dim;

    for (int j = threadIdx.x; j < embed_dim; j += blockDim.x) {
        output_sample_ptr[j] = target_embed[j];
    }
}

template<typename T>
__global__ void argmax_sample_kernel(T *input, int32_t *output, int hidden_dim) {
    float maxValue = -1e20;
    int argmax     = 0;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float data = (float)input[blockIdx.x * hidden_dim + i];
        if (data > maxValue) {
            maxValue = data;
            argmax   = i;
        }
    }
    // blockAllReduceMax seems to be broken when T=half
    float maxValueBlock = vllm::blockAllReduceMax(maxValue);
    if (maxValue == maxValueBlock) {
        output[blockIdx.x] = argmax;
    }
}

template<typename T>
__global__ void splitqkv_kernel(T *qkv, T *q, T *k, T *v, int q_size, int kv_size) {
    int qkv_size = q_size + 2 * kv_size;
    for (int i = threadIdx.x; i < qkv_size; i += blockDim.x) {
        T data = qkv[blockIdx.x * qkv_size + i];
        if (i < q_size) {
            q[blockIdx.x * q_size + i] = data;
        } else if (i < q_size + kv_size) {
            k[blockIdx.x * kv_size + i - q_size] = data;
        } else {
            v[blockIdx.x * kv_size + i - q_size - kv_size] = data;
        }
    }
}

template<typename T, int unroll>
__global__ void quant_kernel_static(const T *input, int8_t *output, T scale, size_t length) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * unroll;
    if (i >= length) {
        return;
    }

    using Tvec  = nunchaku::kernels::Tvec<T, unroll>;
    using I8vec = nunchaku::kernels::Tvec<int8_t, unroll>;

    Tvec rinput = *reinterpret_cast<const Tvec *>(&input[i]);
    I8vec routput;
    float fscale = 1.0f / (float)scale;

#pragma unroll
    for (int k = 0; k < unroll; k++) {
        routput.data[k] = float_to_int8_rn(((float)rinput.data[k]) * fscale);
    }

    *reinterpret_cast<I8vec *>(&output[i]) = routput;
}

template<typename T, int unroll>
__global__ void quant_kernel_static_fuse_gelu(const T *input, int8_t *output, T scale, size_t length) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * unroll;
    if (i >= length) {
        return;
    }

    using Tvec  = nunchaku::kernels::Tvec<T, unroll>;
    using I8vec = nunchaku::kernels::Tvec<int8_t, unroll>;

    Tvec rinput = *reinterpret_cast<const Tvec *>(&input[i]);
    I8vec routput;
    float fscale = 1.0f / (float)scale;

#pragma unroll
    for (int k = 0; k < unroll; k++) {
        routput.data[k] = float_to_int8_rn(((float)vllm::gelu_new_kernel(rinput.data[k])) * fscale);
    }

    *reinterpret_cast<I8vec *>(&output[i]) = routput;
}

template<typename Tin, typename Tout, int unroll>
__global__ void cast_kernel(const Tin *input, Tout *output, size_t length) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * unroll;

    using Tvec_in  = nunchaku::kernels::Tvec<Tin, unroll>;
    using Tvec_out = nunchaku::kernels::Tvec<Tout, unroll>;

    Tvec_in rinput = *reinterpret_cast<const Tvec_in *>(&input[i]);
    Tvec_out routput;

#pragma unroll
    for (int k = 0; k < unroll; k++) {
        routput.data[k] = cuda_cast<Tout, Tin>(rinput.data[k]);
        if constexpr (std::is_same_v<Tout, half>) {
            routput.data[k] = __hmin(routput.data[k], (half)65504);
            routput.data[k] = __hmax(routput.data[k], (half)-65504);
        }
    }

    *reinterpret_cast<Tvec_out *>(&output[i]) = routput;
}

// input:  [..., N]
// output: [..., K] of index in reverse order
template<typename T, int K>
__global__ void topk_kernel(const T *input, int *output, int N, int strideInput, int numRows) {
    const int row    = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = row * strideInput;

    if (row >= numRows) {
        return;
    }

    T val[K];
    int16_t idx[K];

#pragma unroll
    for (int i = 0; i < K; i++) {
        val[i] = input[offset + i];
        idx[i] = i;
    }

    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     for (int i = 0; i < K; i++) {
    //         printf("%d ", idx[i]);
    //     }
    //     printf("\n");
    // }

    for (int i = K; i < N; i++) {
        T newval = input[offset + i];

        T minval   = val[0];
        int minpos = 0;
#pragma unroll
        for (int j = 1; j < K; j++) {
            if (val[j] < minval) {
                minval = val[j];
                minpos = j;
            }
        }

        if (newval >= minval) {
#pragma unroll
            for (int j = 0; j < K; j++) {
                if (j >= minpos) {
                    val[j] = val[j + 1];
                    idx[j] = idx[j + 1];
                }
            }
            val[K - 1] = newval;
            idx[K - 1] = i;
        }

        // if (blockIdx.x == 0 && threadIdx.x == 0) {
        //     for (int i = 0; i < K; i++) {
        //         printf("%d ", idx[i]);
        //     }
        //     printf("\n");
        // }
    }

    for (int i = 0; i < K; i++) {
        output[row * K + i] = idx[K - i - 1];
    }
}

}; // namespace nunchaku::kernels
