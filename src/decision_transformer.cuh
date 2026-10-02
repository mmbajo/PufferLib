#ifndef PUFFER_DECISION_TRANSFORMER_CUH
#define PUFFER_DECISION_TRANSFORMER_CUH

// A native FP32 implementation of decisions.policy.SnakePolicy. All trainable
// operations, including attention and embedding lookup, have explicit backward
// passes. Parameter names and row-major layouts match the PyTorch state_dict.
// Legal-action masking belongs to the caller, after the four action logits.
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace decision {

struct Config {
    int width = 32;
    int layers = 2;
    bool head_pooling = true;
    bool relative_coordinates = true;
    std::string body_features = "ordered";
};

struct Parameter {
    std::string name;
    std::vector<int> shape;
    size_t count;
    float* data;
    float* grad;
};

struct Output {
    float* logits;  // [batch, 4], unmasked
    float* values;  // [batch]
};

namespace transformer_detail {

constexpr int tokens = 101;
constexpr int cells = 100;
constexpr int heads = 4;
constexpr int threads = 256;

inline void cuda_check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("decision Transformer CUDA: ") +
                                 cudaGetErrorString(status));
}

inline void blas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("decision Transformer cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
}

inline unsigned int grid(size_t count) {
    size_t blocks = (count + threads - 1) / threads;
    if (blocks > std::numeric_limits<unsigned int>::max())
        throw std::invalid_argument("decision Transformer tensor is too large");
    return static_cast<unsigned int>(blocks);
}

__device__ inline float block_sum(float value) {
    __shared__ float work[threads];
    int t = threadIdx.x;
    work[t] = value;
    __syncthreads();
    for (int stride = threads / 2; stride; stride /= 2) {
        if (t < stride) work[t] += work[t + stride];
        __syncthreads();
    }
    float result = work[0];
    __syncthreads();
    return result;
}

__device__ inline float block_max(float value) {
    __shared__ float work[threads];
    int t = threadIdx.x;
    work[t] = value;
    __syncthreads();
    for (int stride = threads / 2; stride; stride /= 2) {
        if (t < stride) work[t] = fmaxf(work[t], work[t + stride]);
        __syncthreads();
    }
    float result = work[0];
    __syncthreads();
    return result;
}

static __global__ void find_heads(const int* boards, int* head, int batch) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    // Match (boards == 1).long().argmax(): first head, or zero if absent.
    int found = 0;
    for (int t = 0; t < cells; ++t) {
        if (boards[b * cells + t] == 1) {
            found = t;
            break;
        }
    }
    head[b] = found;
}

static __global__ void embed_forward(const int* boards, const int* head,
        const float* cell, const float* cls, const float* position,
        const float* rel_row, const float* rel_col, float* output,
        int batch, int width) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t count = static_cast<size_t>(batch) * tokens * width;
    if (i >= count) return;
    int d = i % width;
    int token = (i / width) % tokens;
    int b = i / (static_cast<size_t>(tokens) * width);
    float value;
    if (token == 0) {
        value = cls[d];
    } else {
        int t = token - 1;
        int id = boards[b * cells + t] + 1;
        // Out-of-contract input produces NaN, never an out-of-bounds lookup.
        if (id < 0 || id >= 102) {
            output[i] = nanf("");
            return;
        }
        value = cell[id * width + d];
        if (rel_row) {
            int h = head[b];
            int row = t / 10 - h / 10 + 9;
            int col = t % 10 - h % 10 + 9;
            value = value + rel_row[row * width + d];
            value = value + rel_col[col * width + d];
        }
    }
    output[i] = value + position[token * width + d];
}

static __global__ void embed_backward(const float* dy, const int* boards,
        const int* head, float* cell_grad, float* cls_grad,
        float* position_grad, float* row_grad, float* col_grad,
        int batch, int width) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(batch) * tokens * width) return;
    int d = i % width;
    int token = (i / width) % tokens;
    int b = i / (static_cast<size_t>(tokens) * width);
    float g = dy[i];
    atomicAdd(position_grad + token * width + d, g);
    if (token == 0) {
        atomicAdd(cls_grad + d, g);
    } else {
        int t = token - 1;
        int id = boards[b * cells + t] + 1;
        if (id < 0 || id >= 102) return;
        atomicAdd(cell_grad + id * width + d, g);
        if (row_grad) {
            int h = head[b];
            atomicAdd(row_grad + (t / 10 - h / 10 + 9) * width + d, g);
            atomicAdd(col_grad + (t % 10 - h % 10 + 9) * width + d, g);
        }
    }
}

static __global__ void bias_forward(float* output, const float* bias,
        size_t count, int width) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] += bias[i % width];
}

static __global__ void bias_backward(const float* dy, float* bias_grad,
        int rows, int width) {
    int d = blockIdx.x;
    float sum = 0.0f;
    for (int row = threadIdx.x; row < rows; row += blockDim.x)
        sum += dy[static_cast<size_t>(row) * width + d];
    sum = block_sum(sum);
    if (threadIdx.x == 0) bias_grad[d] += sum;
}

static __global__ void residual_add(float* output, const float* residual,
        size_t count) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] += residual[i];
}

static __global__ void norm_forward(const float* input, const float* gamma,
        const float* beta, float* output, float* means, float* invstd,
        int width) {
    int row = blockIdx.x;
    size_t base = static_cast<size_t>(row) * width;
    float sum = 0.0f;
    for (int d = threadIdx.x; d < width; d += blockDim.x)
        sum += input[base + d];
    float mean = block_sum(sum) / width;
    float variance = 0.0f;
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
        float centered = input[base + d] - mean;
        variance += centered * centered;
    }
    float rstd = rsqrtf(block_sum(variance) / width + 1.0e-5f);
    if (threadIdx.x == 0) {
        means[row] = mean;
        invstd[row] = rstd;
    }
    for (int d = threadIdx.x; d < width; d += blockDim.x)
        output[base + d] = (input[base + d] - mean) * rstd * gamma[d] + beta[d];
}

static __global__ void norm_backward_input(const float* dy, const float* input,
        const float* gamma, const float* means, const float* invstd,
        const float* residual_gradient, float* dx, int width) {
    int row = blockIdx.x;
    size_t base = static_cast<size_t>(row) * width;
    float mean = means[row], rstd = invstd[row];
    float sum = 0.0f, sum_z = 0.0f;
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
        float du = dy[base + d] * gamma[d];
        float z = (input[base + d] - mean) * rstd;
        sum += du;
        sum_z += du * z;
    }
    float mean_du = block_sum(sum) / width;
    float mean_du_z = block_sum(sum_z) / width;
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
        float z = (input[base + d] - mean) * rstd;
        float value = rstd * (dy[base + d] * gamma[d] - mean_du - z * mean_du_z);
        dx[base + d] = value + (residual_gradient ? residual_gradient[base + d] : 0.0f);
    }
}

static __global__ void norm_backward_params(const float* dy, const float* input,
        const float* means, const float* invstd, float* gamma_grad,
        float* beta_grad, int rows, int width) {
    int d = blockIdx.x;
    float dg = 0.0f, db = 0.0f;
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        size_t i = static_cast<size_t>(row) * width + d;
        float g = dy[i];
        dg += g * ((input[i] - means[row]) * invstd[row]);
        db += g;
    }
    dg = block_sum(dg);
    db = block_sum(db);
    if (threadIdx.x == 0) {
        gamma_grad[d] += dg;
        beta_grad[d] += db;
    }
}

static __global__ void gelu_forward(const float* input, float* output, size_t count) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + erff(x * 0.7071067811865475244f));
    }
}

static __global__ void gelu_backward(const float* input, float* gradient, size_t count) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        float x = input[i];
        float derivative = 0.5f * (1.0f + erff(x * 0.7071067811865475244f)) +
                           x * 0.39894228040143267794f * expf(-0.5f * x * x);
        gradient[i] *= derivative;
    }
}

// Separate [B,T,3*D] into three contiguous [B,H,T,D/H] arrays.
static __global__ void split_qkv(const float* qkv, float* q, float* k, float* v,
        int batch, int width) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(batch) * tokens * width) return;
    int dim = width / heads;
    int c = i % dim;
    int t = (i / dim) % tokens;
    int h = (i / (static_cast<size_t>(dim) * tokens)) % heads;
    int b = i / (static_cast<size_t>(tokens) * width);
    size_t source = static_cast<size_t>(b * tokens + t) * 3 * width + h * dim + c;
    q[i] = qkv[source];
    k[i] = qkv[source + width];
    v[i] = qkv[source + 2 * width];
}

static __global__ void merge_qkv_grad(const float* q, const float* k,
        const float* v, float* qkv, int batch, int width) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(batch) * tokens * width) return;
    int dim = width / heads;
    int c = i % dim;
    int t = (i / dim) % tokens;
    int h = (i / (static_cast<size_t>(dim) * tokens)) % heads;
    int b = i / (static_cast<size_t>(tokens) * width);
    size_t target = static_cast<size_t>(b * tokens + t) * 3 * width + h * dim + c;
    qkv[target] = q[i];
    qkv[target + width] = k[i];
    qkv[target + 2 * width] = v[i];
}

static __global__ void merge_heads(const float* input, float* output,
        int batch, int width) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(batch) * tokens * width) return;
    int d = i % width;
    int t = (i / width) % tokens;
    int b = i / (static_cast<size_t>(tokens) * width);
    int dim = width / heads;
    size_t source = ((static_cast<size_t>(b) * heads + d / dim) * tokens + t) * dim + d % dim;
    output[i] = input[source];
}

static __global__ void split_heads(const float* input, float* output,
        int batch, int width) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(batch) * tokens * width) return;
    int d = i % width;
    int t = (i / width) % tokens;
    int b = i / (static_cast<size_t>(tokens) * width);
    int dim = width / heads;
    size_t target = ((static_cast<size_t>(b) * heads + d / dim) * tokens + t) * dim + d % dim;
    output[target] = input[i];
}

static __global__ void softmax_forward(float* scores) {
    size_t base = static_cast<size_t>(blockIdx.x) * tokens;
    int t = threadIdx.x;
    float value = t < tokens ? scores[base + t] : -FLT_MAX;
    float maximum = block_max(value);
    float e = t < tokens ? expf(value - maximum) : 0.0f;
    float sum = block_sum(e);
    if (t < tokens) scores[base + t] = e / sum;
}

static __global__ void softmax_backward(const float* probabilities,
        float* gradient, float scale) {
    size_t base = static_cast<size_t>(blockIdx.x) * tokens;
    int t = threadIdx.x;
    float p = t < tokens ? probabilities[base + t] : 0.0f;
    float g = t < tokens ? gradient[base + t] : 0.0f;
    float mean = block_sum(p * g);
    if (t < tokens) gradient[base + t] = p * (g - mean) * scale;
}

static __global__ void pool_forward(const float* input, const int* head,
        float* pooled, int batch, int width, bool head_pooling) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(batch) * width) return;
    int b = i / width, d = i % width;
    int token = head_pooling ? head[b] + 1 : 0;
    pooled[i] = input[(static_cast<size_t>(b) * tokens + token) * width + d];
}

static __global__ void pool_backward(const float* pooled, const int* head,
        float* dx, int batch, int width, bool head_pooling) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(batch) * tokens * width) return;
    int b = i / (static_cast<size_t>(tokens) * width);
    int token = (i / width) % tokens;
    int d = i % width;
    int selected = head_pooling ? head[b] + 1 : 0;
    dx[i] = token == selected ? pooled[static_cast<size_t>(b) * width + d] : 0.0f;
}

}  // namespace transformer_detail

class Model {
public:
    explicit Model(const Config& config, int max_batch, uint64_t seed = 1,
                   cudaStream_t stream = nullptr)
        : cfg_(config), max_batch_(max_batch), stream_(stream) {
        if (cfg_.width < 4 || cfg_.width % 4 != 0 || cfg_.width > 16384)
            throw std::invalid_argument("Transformer width must be a multiple of four in [4, 16384]");
        if (cfg_.layers < 1)
            throw std::invalid_argument("Transformer layers must be positive");
        if (cfg_.body_features != "ordered")
            throw std::invalid_argument("native Transformer supports ordered body features only; tail-distance is not implemented");
        if (max_batch_ < 1 || max_batch_ > std::numeric_limits<int>::max() / 404)
            throw std::invalid_argument("Transformer max_batch is outside supported bounds");
        try {
            transformer_detail::blas_check(cublasCreate(&handle_));
            transformer_detail::blas_check(cublasSetStream(handle_, stream_));
            // Do not silently replace FP32 products with TF32 during parity work.
            transformer_detail::blas_check(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH));
            initialize_parameters(seed);
            allocate_activations();
            zero_grad();
        } catch (...) {
            release();
            throw;
        }
    }

    ~Model() { release(); }
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = delete;
    Model& operator=(Model&&) = delete;

    const Config& config() const { return cfg_; }
    const std::vector<Parameter>& parameters() const { return params_; }
    cudaStream_t stream() const { return stream_; }
    int max_batch() const { return max_batch_; }

    // The caller orders uses of each workspace; changing streams introduces no
    // implicit synchronization or cross-stream dependency.
    void set_stream(cudaStream_t stream) {
        transformer_detail::blas_check(cublasSetStream(handle_, stream));
        stream_ = stream;
    }

    // Bind Puffer's shared parameter and gradient allocations to this model's
    // private activation workspace. External buffers remain owned by the caller
    // and must outlive this Model. Their shapes follow parameters(), in order.
    // Binding synchronizes this stream and discards the latest forward cache.
    void bind_parameters(const std::vector<float*>& data,
                         const std::vector<float*>& gradients) {
        if (data.size() != params_.size() || gradients.size() != params_.size())
            throw std::invalid_argument("Transformer parameter binding count mismatch");
        for (size_t i = 0; i < params_.size(); ++i) {
            if (!data[i] || !gradients[i])
                throw std::invalid_argument("Transformer parameter binding cannot contain null pointers");
        }
        transformer_detail::cuda_check(cudaStreamSynchronize(stream_));
        for (Parameter& param : params_) {
            release_replaced_parameter(param.data, data, gradients);
            release_replaced_parameter(param.grad, data, gradients);
        }
        for (size_t i = 0; i < params_.size(); ++i) {
            params_[i].data = data[i];
            params_[i].grad = gradients[i];
        }
        last_batch_ = 0;
    }

    // Device input is int32 [batch,100], with raw board values -1..100.
    // The model owns a copy, so the caller may reuse its input after this stream
    // completes. A new forward invalidates the previous backward activations.
    Output forward(const int* device_boards, int batch) {
        using namespace transformer_detail;
        if (!device_boards || batch < 1 || batch > max_batch_)
            throw std::invalid_argument("invalid Transformer forward input or batch");
        last_batch_ = batch;
        int width = cfg_.width, rows = batch * tokens;
        size_t hidden = static_cast<size_t>(rows) * width;
        cuda_check(cudaMemcpyAsync(boards_, device_boards,
            static_cast<size_t>(batch) * cells * sizeof(int), cudaMemcpyDeviceToDevice, stream_));
        find_heads<<<grid(batch), threads, 0, stream_>>>(boards_, head_, batch);
        embed_forward<<<grid(hidden), threads, 0, stream_>>>(boards_, head_,
            p(cell_).data, p(cls_).data, p(position_).data,
            cfg_.relative_coordinates ? p(relative_row_).data : nullptr,
            cfg_.relative_coordinates ? p(relative_col_).data : nullptr,
            embedded_, batch, width);

        const float* x = embedded_;
        for (int l = 0; l < cfg_.layers; ++l) {
            const BlockParameters& w = block_params_[l];
            BlockActivations& a = blocks_[l];
            norm(x, w.norm1_weight, w.norm1_bias, a.norm1, a.mean1, a.rstd1, rows);
            linear(a.norm1, w.qkv_weight, w.qkv_bias, a.qkv, rows, width, 3 * width);
            split_qkv<<<grid(hidden), threads, 0, stream_>>>(a.qkv, a.q, a.k, a.v, batch, width);
            attention_forward(a, batch);
            merge_heads<<<grid(hidden), threads, 0, stream_>>>(a.context_heads, a.context, batch, width);
            linear(a.context, w.output_weight, w.output_bias, a.residual1, rows, width, width);
            residual_add<<<grid(hidden), threads, 0, stream_>>>(a.residual1, x, hidden);
            norm(a.residual1, w.norm2_weight, w.norm2_bias, a.norm2, a.mean2, a.rstd2, rows);
            linear(a.norm2, w.ff1_weight, w.ff1_bias, a.ff1, rows, width, 4 * width);
            gelu_forward<<<grid(hidden * 4), threads, 0, stream_>>>(a.ff1, a.gelu, hidden * 4);
            linear(a.gelu, w.ff2_weight, w.ff2_bias, a.output, rows, 4 * width, width);
            residual_add<<<grid(hidden), threads, 0, stream_>>>(a.output, a.residual1, hidden);
            x = a.output;
        }
        norm(x, final_weight_, final_bias_, normalized_, final_mean_, final_rstd_, rows);
        pool_forward<<<grid(static_cast<size_t>(batch) * width), threads, 0, stream_>>>(
            normalized_, head_, pooled_, batch, width, cfg_.head_pooling);
        linear(pooled_, action_weight_, action_bias_, logits_, batch, width, 4);
        linear(pooled_, value_weight_, value_bias_, values_, batch, width, 1);
        cuda_check(cudaGetLastError());
        return {logits_, values_};
    }

    // Both upstream gradients are device arrays. Parameter gradients accumulate
    // until zero_grad(); this supports combining losses or microbatches.
    void backward(const float* logits_gradient, const float* values_gradient) {
        using namespace transformer_detail;
        if (last_batch_ < 1 || !logits_gradient || !values_gradient)
            throw std::invalid_argument("Transformer backward requires a forward and both output gradients");
        int batch = last_batch_, width = cfg_.width, rows = batch * tokens;
        size_t hidden = static_cast<size_t>(rows) * width;
        linear_backward(pooled_, logits_gradient, action_weight_, action_bias_,
                        d_pooled_, batch, width, 4);
        linear_backward(pooled_, values_gradient, value_weight_, value_bias_,
                        d_pooled_, batch, width, 1, 1.0f);
        pool_backward<<<grid(hidden), threads, 0, stream_>>>(
            d_pooled_, head_, d_norm_, batch, width, cfg_.head_pooling);
        norm_backward(d_norm_, blocks_.back().output, final_weight_, final_bias_,
            final_mean_, final_rstd_, nullptr, d_hidden_, rows);

        for (int l = cfg_.layers - 1; l >= 0; --l) {
            const BlockParameters& w = block_params_[l];
            const BlockActivations& a = blocks_[l];
            const float* x = l ? blocks_[l - 1].output : embedded_;
            linear_backward(a.gelu, d_hidden_, w.ff2_weight, w.ff2_bias,
                            d_ff_, rows, 4 * width, width);
            gelu_backward<<<grid(hidden * 4), threads, 0, stream_>>>(a.ff1, d_ff_, hidden * 4);
            linear_backward(a.norm2, d_ff_, w.ff1_weight, w.ff1_bias,
                            d_norm_, rows, width, 4 * width);
            norm_backward(d_norm_, a.residual1, w.norm2_weight, w.norm2_bias,
                a.mean2, a.rstd2, d_hidden_, d_residual_, rows);
            linear_backward(a.context, d_residual_, w.output_weight, w.output_bias,
                            d_context_, rows, width, width);
            split_heads<<<grid(hidden), threads, 0, stream_>>>(d_context_, d_context_heads_, batch, width);
            attention_backward(a, batch);
            merge_qkv_grad<<<grid(hidden), threads, 0, stream_>>>(d_q_, d_k_, d_v_, d_qkv_, batch, width);
            linear_backward(a.norm1, d_qkv_, w.qkv_weight, w.qkv_bias,
                            d_norm_, rows, width, 3 * width);
            norm_backward(d_norm_, x, w.norm1_weight, w.norm1_bias,
                a.mean1, a.rstd1, d_residual_, d_hidden_, rows);
        }
        embed_backward<<<grid(hidden), threads, 0, stream_>>>(d_hidden_, boards_, head_,
            p(cell_).grad, p(cls_).grad, p(position_).grad,
            cfg_.relative_coordinates ? p(relative_row_).grad : nullptr,
            cfg_.relative_coordinates ? p(relative_col_).grad : nullptr,
            batch, width);
        cuda_check(cudaGetLastError());
    }

    void zero_grad() {
        for (const Parameter& param : params_)
            transformer_detail::cuda_check(cudaMemsetAsync(param.grad, 0,
                param.count * sizeof(float), stream_));
    }

private:
    struct BlockParameters {
        size_t qkv_weight, qkv_bias, output_weight, output_bias;
        size_t ff1_weight, ff1_bias, ff2_weight, ff2_bias;
        size_t norm1_weight, norm1_bias, norm2_weight, norm2_bias;
    };
    struct BlockActivations {
        float *norm1, *mean1, *rstd1, *qkv, *q, *k, *v, *probabilities;
        float *context_heads, *context, *residual1, *norm2, *mean2, *rstd2;
        float *ff1, *gelu, *output;
    };
    Config cfg_;
    int max_batch_, last_batch_ = 0;
    cudaStream_t stream_;
    cublasHandle_t handle_ = nullptr;
    std::vector<void*> allocations_;
    std::vector<Parameter> params_;
    std::vector<BlockParameters> block_params_;
    std::vector<BlockActivations> blocks_;
    size_t cell_, cls_, position_, relative_row_ = 0, relative_col_ = 0;
    size_t final_weight_, final_bias_, action_weight_, action_bias_, value_weight_, value_bias_;
    int *boards_ = nullptr, *head_ = nullptr;
    float *embedded_ = nullptr, *normalized_ = nullptr, *final_mean_ = nullptr;
    float *final_rstd_ = nullptr, *pooled_ = nullptr, *logits_ = nullptr, *values_ = nullptr;
    float *d_pooled_ = nullptr, *d_hidden_ = nullptr, *d_residual_ = nullptr;
    float *d_norm_ = nullptr, *d_ff_ = nullptr, *d_context_ = nullptr;
    float *d_context_heads_ = nullptr, *d_qkv_ = nullptr, *d_q_ = nullptr;
    float *d_k_ = nullptr, *d_v_ = nullptr, *d_attention_ = nullptr;

    const Parameter& p(size_t i) const { return params_[i]; }

    template <typename T> T* allocate(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::invalid_argument("Transformer allocation size overflow");
        T* pointer = nullptr;
        transformer_detail::cuda_check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)));
        try {
            allocations_.push_back(pointer);
        } catch (...) {
            cudaFree(pointer);
            throw;
        }
        return pointer;
    }

    void release() noexcept {
        if (handle_) cublasDestroy(handle_);
        handle_ = nullptr;
        for (void* pointer : allocations_) cudaFree(pointer);
        allocations_.clear();
    }

    void release_replaced_parameter(float* pointer,
            const std::vector<float*>& data,
            const std::vector<float*>& gradients) {
        // Binding the model's existing buffers back to itself is a no-op. Other
        // pointers may already be external from an earlier bind; never free them.
        if (std::find(data.begin(), data.end(), pointer) != data.end() ||
                std::find(gradients.begin(), gradients.end(), pointer) != gradients.end())
            return;
        auto owner = std::find(allocations_.begin(), allocations_.end(), pointer);
        if (owner != allocations_.end()) {
            transformer_detail::cuda_check(cudaFree(*owner));
            *owner = nullptr;
        }
    }

    enum class Init { zero, one, normal, uniform };

    size_t add_parameter(const std::string& name, const std::vector<int>& shape,
            Init init, float scale, std::mt19937_64& rng) {
        size_t count = 1;
        for (int d : shape) {
            if (d < 1 || count > std::numeric_limits<size_t>::max() / static_cast<size_t>(d))
                throw std::invalid_argument("invalid Transformer parameter shape");
            count *= d;
        }
        float* data = allocate<float>(count);
        float* grad = allocate<float>(count);
        std::vector<float> host(count);
        std::normal_distribution<float> normal(0.0f, init == Init::normal ? scale : 1.0f);
        std::uniform_real_distribution<float> uniform(-scale, scale);
        for (float& value : host) {
            switch (init) {
                case Init::zero: value = 0.0f; break;
                case Init::one: value = 1.0f; break;
                case Init::normal: value = normal(rng); break;
                case Init::uniform: value = uniform(rng); break;
            }
        }
        transformer_detail::cuda_check(cudaMemcpy(data, host.data(), count * sizeof(float), cudaMemcpyHostToDevice));
        params_.push_back({name, shape, count, data, grad});
        return params_.size() - 1;
    }

    void initialize_parameters(uint64_t seed) {
        int d = cfg_.width, f = 4 * d;
        std::mt19937_64 rng(seed);
        cls_ = add_parameter("cls", {1, 1, d}, Init::normal, 0.02f, rng);
        position_ = add_parameter("positions", {1, 101, d}, Init::normal, 0.02f, rng);
        cell_ = add_parameter("cells.weight", {102, d}, Init::normal, 0.02f, rng);
        for (int l = 0; l < cfg_.layers; ++l) {
            std::string name = "encoder.layers." + std::to_string(l) + ".";
            size_t start = params_.size();
            BlockParameters w;
            w.qkv_weight = add_parameter(name + "self_attn.in_proj_weight", {3 * d, d}, Init::uniform, std::sqrt(1.5f / d), rng);
            w.qkv_bias = add_parameter(name + "self_attn.in_proj_bias", {3 * d}, Init::zero, 0.0f, rng);
            w.output_weight = add_parameter(name + "self_attn.out_proj.weight", {d, d}, Init::uniform, 1.0f / std::sqrt(static_cast<float>(d)), rng);
            w.output_bias = add_parameter(name + "self_attn.out_proj.bias", {d}, Init::zero, 0.0f, rng);
            w.ff1_weight = add_parameter(name + "linear1.weight", {f, d}, Init::uniform, 1.0f / std::sqrt(static_cast<float>(d)), rng);
            w.ff1_bias = add_parameter(name + "linear1.bias", {f}, Init::uniform, 1.0f / std::sqrt(static_cast<float>(d)), rng);
            w.ff2_weight = add_parameter(name + "linear2.weight", {d, f}, Init::uniform, 1.0f / std::sqrt(static_cast<float>(f)), rng);
            w.ff2_bias = add_parameter(name + "linear2.bias", {d}, Init::uniform, 1.0f / std::sqrt(static_cast<float>(f)), rng);
            w.norm1_weight = add_parameter(name + "norm1.weight", {d}, Init::one, 1.0f, rng);
            w.norm1_bias = add_parameter(name + "norm1.bias", {d}, Init::zero, 0.0f, rng);
            w.norm2_weight = add_parameter(name + "norm2.weight", {d}, Init::one, 1.0f, rng);
            w.norm2_bias = add_parameter(name + "norm2.bias", {d}, Init::zero, 0.0f, rng);
            // TransformerEncoder clones one initialized layer. Keep that useful
            // symmetry-breaking baseline, without promising PyTorch RNG parity.
            if (l > 0) {
                for (size_t i = 0; i < 12; ++i)
                    transformer_detail::cuda_check(cudaMemcpy(p(start + i).data,
                        p(3 + i).data, p(start + i).count * sizeof(float), cudaMemcpyDeviceToDevice));
            }
            block_params_.push_back(w);
        }
        final_weight_ = add_parameter("encoder.norm.weight", {d}, Init::one, 1.0f, rng);
        final_bias_ = add_parameter("encoder.norm.bias", {d}, Init::zero, 0.0f, rng);
        float bound = 1.0f / std::sqrt(static_cast<float>(d));
        action_weight_ = add_parameter("action_head.weight", {4, d}, Init::uniform, bound, rng);
        action_bias_ = add_parameter("action_head.bias", {4}, Init::uniform, bound, rng);
        value_weight_ = add_parameter("value_head.weight", {1, d}, Init::uniform, bound, rng);
        value_bias_ = add_parameter("value_head.bias", {1}, Init::uniform, bound, rng);
        if (cfg_.relative_coordinates) {
            relative_row_ = add_parameter("relative_rows.weight", {19, d}, Init::normal, 0.02f, rng);
            relative_col_ = add_parameter("relative_columns.weight", {19, d}, Init::normal, 0.02f, rng);
        }
    }

    void allocate_activations() {
        using namespace transformer_detail;
        size_t rows = static_cast<size_t>(max_batch_) * tokens;
        size_t n = rows * cfg_.width;
        size_t attention = static_cast<size_t>(max_batch_) * heads * tokens * tokens;
        boards_ = allocate<int>(static_cast<size_t>(max_batch_) * cells);
        head_ = allocate<int>(max_batch_);
        embedded_ = allocate<float>(n);
        for (int l = 0; l < cfg_.layers; ++l) {
            BlockActivations a;
            a.norm1 = allocate<float>(n); a.mean1 = allocate<float>(rows); a.rstd1 = allocate<float>(rows);
            a.qkv = allocate<float>(3 * n);
            a.q = allocate<float>(n); a.k = allocate<float>(n); a.v = allocate<float>(n);
            a.probabilities = allocate<float>(attention);
            a.context_heads = allocate<float>(n); a.context = allocate<float>(n);
            a.residual1 = allocate<float>(n);
            a.norm2 = allocate<float>(n); a.mean2 = allocate<float>(rows); a.rstd2 = allocate<float>(rows);
            a.ff1 = allocate<float>(4 * n); a.gelu = allocate<float>(4 * n); a.output = allocate<float>(n);
            blocks_.push_back(a);
        }
        normalized_ = allocate<float>(n);
        final_mean_ = allocate<float>(rows); final_rstd_ = allocate<float>(rows);
        pooled_ = allocate<float>(static_cast<size_t>(max_batch_) * cfg_.width);
        logits_ = allocate<float>(static_cast<size_t>(max_batch_) * 4);
        values_ = allocate<float>(max_batch_);
        d_pooled_ = allocate<float>(static_cast<size_t>(max_batch_) * cfg_.width);
        d_hidden_ = allocate<float>(n); d_residual_ = allocate<float>(n); d_norm_ = allocate<float>(n);
        d_ff_ = allocate<float>(4 * n); d_context_ = allocate<float>(n);
        d_context_heads_ = allocate<float>(n); d_qkv_ = allocate<float>(3 * n);
        d_q_ = allocate<float>(n); d_k_ = allocate<float>(n); d_v_ = allocate<float>(n);
        d_attention_ = allocate<float>(attention);
    }

    void linear(const float* input, size_t weight, size_t bias, float* output,
            int rows, int in, int out) {
        using namespace transformer_detail;
        const float one = 1.0f, zero = 0.0f;
        // Y = X W^T. cuBLAS sees the transpose of every row-major array.
        blas_check(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, out, rows, in,
            &one, p(weight).data, in, input, in, &zero, output, out));
        bias_forward<<<grid(static_cast<size_t>(rows) * out), threads, 0, stream_>>>(
            output, p(bias).data, static_cast<size_t>(rows) * out, out);
    }

    void linear_backward(const float* input, const float* dy,
            size_t weight, size_t bias, float* dx,
            int rows, int in, int out, float dx_beta = 0.0f) {
        using namespace transformer_detail;
        const float one = 1.0f;
        blas_check(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, in, rows, out,
            &one, p(weight).data, in, dy, out, &dx_beta, dx, in));
        blas_check(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, in, out, rows,
            &one, input, in, dy, out, &one, p(weight).grad, in));
        bias_backward<<<out, threads, 0, stream_>>>(dy, p(bias).grad, rows, out);
    }

    void norm(const float* input, size_t gamma, size_t beta, float* output,
            float* means, float* rstd, int rows) {
        transformer_detail::norm_forward<<<rows, transformer_detail::threads, 0, stream_>>>(
            input, p(gamma).data, p(beta).data, output, means, rstd, cfg_.width);
    }

    void norm_backward(const float* dy, const float* input, size_t gamma,
            size_t beta, const float* means, const float* rstd,
            const float* residual, float* dx, int rows) {
        using namespace transformer_detail;
        norm_backward_input<<<rows, threads, 0, stream_>>>(
            dy, input, p(gamma).data, means, rstd, residual, dx, cfg_.width);
        norm_backward_params<<<cfg_.width, threads, 0, stream_>>>(
            dy, input, means, rstd, p(gamma).grad, p(beta).grad, rows, cfg_.width);
    }

    void attention_forward(BlockActivations& a, int batch) {
        using namespace transformer_detail;
        int dim = cfg_.width / heads;
        int batches = batch * heads;
        long long vectors = static_cast<long long>(tokens) * dim;
        long long matrices = static_cast<long long>(tokens) * tokens;
        const float one = 1.0f, zero = 0.0f, scale = 1.0f / std::sqrt(static_cast<float>(dim));
        blas_check(cublasSgemmStridedBatched(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            tokens, tokens, dim, &scale, a.k, dim, vectors, a.q, dim, vectors,
            &zero, a.probabilities, tokens, matrices, batches));
        softmax_forward<<<batches * tokens, threads, 0, stream_>>>(a.probabilities);
        blas_check(cublasSgemmStridedBatched(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim, tokens, tokens, &one, a.v, dim, vectors,
            a.probabilities, tokens, matrices, &zero, a.context_heads, dim, vectors, batches));
    }

    void attention_backward(const BlockActivations& a, int batch) {
        using namespace transformer_detail;
        int dim = cfg_.width / heads;
        int batches = batch * heads;
        long long vectors = static_cast<long long>(tokens) * dim;
        long long matrices = static_cast<long long>(tokens) * tokens;
        const float one = 1.0f, zero = 0.0f;
        // dV = P^T dContext; dP = dContext V^T.
        blas_check(cublasSgemmStridedBatched(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim, tokens, tokens, &one, d_context_heads_, dim, vectors,
            a.probabilities, tokens, matrices, &zero, d_v_, dim, vectors, batches));
        blas_check(cublasSgemmStridedBatched(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            tokens, tokens, dim, &one, a.v, dim, vectors,
            d_context_heads_, dim, vectors, &zero, d_attention_, tokens, matrices, batches));
        softmax_backward<<<batches * tokens, threads, 0, stream_>>>(
            a.probabilities, d_attention_, 1.0f / std::sqrt(static_cast<float>(dim)));
        // dQ = dScore K; dK = dScore^T Q.
        blas_check(cublasSgemmStridedBatched(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim, tokens, tokens, &one, a.k, dim, vectors,
            d_attention_, tokens, matrices, &zero, d_q_, dim, vectors, batches));
        blas_check(cublasSgemmStridedBatched(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim, tokens, tokens, &one, a.q, dim, vectors,
            d_attention_, tokens, matrices, &zero, d_k_, dim, vectors, batches));
    }
};

}  // namespace decision
#endif  // PUFFER_DECISION_TRANSFORMER_CUH
