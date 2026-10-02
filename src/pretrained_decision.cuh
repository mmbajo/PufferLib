#ifndef PUFFER_PRETRAINED_DECISION_CUH
#define PUFFER_PRETRAINED_DECISION_CUH

#include "pretrained_encoder.cuh"
#include <memory>

namespace pretrained {

struct DecisionConfig {
    Config encoder;
    int head_layers = 2;
    int n_act = 2;
    bool value_head = false; // Additional RL critic; not part of imported Laya.
};

inline Config head_config(const DecisionConfig& config) {
    Config head = config.encoder;
    head.family = "laya_head";
    head.layers = config.head_layers;
    head.heads = std::max(1, head.width / 64);
    head.intermediate = 4 * head.width;
    head.epsilon = 1e-5f;
    head.attention_bias = head.mlp_bias = head.norm_bias = true;
    head.layer_types.clear();
    return head;
}

inline std::vector<Parameter> decision_parameter_specs(const DecisionConfig& config) {
    auto result = parameter_specs(config.encoder);
    for (auto& p : result) p.name = "encoder." + p.name;
    if (config.head_layers) {
        auto head = parameter_specs(head_config(config));
        for (auto& p : head) { p.name = "head." + p.name; result.push_back(p); }
    }
    auto add = [&](const std::string& name, std::vector<int> shape) {
        size_t count = 1;
        for (int dim : shape) count *= dim;
        result.push_back({name, shape, count, nullptr, nullptr});
    };
    int d = config.encoder.width;
    add("type_emb.weight", {3, d});
    add("scorer.0.weight", {d}); add("scorer.0.bias", {d});
    add("scorer.1.weight", {d, d}); add("scorer.1.bias", {d});
    add("scorer.3.weight", {1, d}); add("scorer.3.bias", {1});
    add("act_head.0.weight", {256, d + 4}); add("act_head.0.bias", {256});
    add("act_head.2.weight", {config.n_act, 256}); add("act_head.2.bias", {config.n_act});
    if (config.value_head) {
        add("value_head.weight", {1, d}); add("value_head.bias", {1});
    }
    return result;
}

namespace decision_detail {
using namespace decision::transformer_detail;

static __global__ void add_type(float* output, const float* hidden, const float* type,
        const int* qtype, int B, int T, int D) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)B * T * D) return;
    int q = qtype[i / ((size_t)T * D)];
    output[i] = q >= 0 && q < 3 ? hidden[i] + type[q * D + i % D] : nanf("");
}

static __global__ void type_backward(const float* gradient, float* type,
        const int* qtype, int B, int T, int D) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)B * T * D) return;
    int q = qtype[i / ((size_t)T * D)];
    if (q >= 0 && q < 3) atomicAdd(type + q * D + i % D, gradient[i]);
}

static __global__ void gather(float* output, const float* hidden,
        const int* markers, int B, int T, int K, int D) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)B * K * D) return;
    int row = i / D, token = max(0, markers[row]);
    output[i] = token < T ? hidden[((size_t)(row / K) * T + token) * D + i % D] : nanf("");
}

static __global__ void scatter(float* output, const float* gradient,
        const int* markers, int B, int T, int K, int D) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)B * K * D) return;
    int row = i / D, token = max(0, markers[row]);
    if (token < T) atomicAdd(output + ((size_t)(row / K) * T + token) * D + i % D, gradient[i]);
}

static __global__ void mask_logits(float* logits, const int* mask, int count, float fill) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && !mask[i]) logits[i] = fill;
}

// These four confidence features are detached in Laya's original graph.
static __global__ void act_input(float* output, float* pooled, const float* hidden,
        const float* logits, const int* marker_mask, int B, int T, int K, int D) {
    int b = blockIdx.x;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float value = hidden[(size_t)b * T * D + d];
        output[(size_t)b * (D + 4) + d] = value;
        pooled[(size_t)b * D + d] = value;
    }
    if (threadIdx.x) return;
    float maximum = -FLT_MAX, total = 0, top1 = 0, top2 = 0, entropy = 0;
    int valid = 0;
    for (int k = 0; k < K; ++k) { maximum = fmaxf(maximum, logits[b * K + k]); valid += !!marker_mask[b * K + k]; }
    for (int k = 0; k < K; ++k) total += expf(logits[b * K + k] - maximum);
    for (int k = 0; k < K; ++k) {
        float p = expf(logits[b * K + k] - maximum) / total;
        entropy -= p * logf(fmaxf(p, 1e-9f));
        if (p >= top1) { top2 = top1; top1 = p; } else top2 = fmaxf(top2, p);
    }
    int count = max(valid, 2);
    size_t offset = (size_t)b * (D + 4) + D;
    output[offset] = top1;
    output[offset + 1] = top1 - top2;
    output[offset + 2] = entropy / logf((float)count);
    output[offset + 3] = count / 255.0f;
}

static __global__ void scatter_cls(float* hidden, const float* gradient,
        int B, int T, int D, int input_stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * D) hidden[(size_t)(i / D) * T * D + i % D] += gradient[(size_t)(i / D) * input_stride + i % D];
}
} // namespace decision_detail

struct DecisionOutput {
    float* logits; // Raw option logits [B,K], masked with Laya's -1e4 sentinel.
    float* act_logits; // [B,n_act], preserves the imported act/escalate head.
    float* values; // Optional additional Puffer critic [B].
};

class DecisionModel {
public:
    DecisionModel(const DecisionConfig& config, int max_batch, int max_tokens,
            int max_options, cudaStream_t stream = nullptr, bool trainable = true,
            bool allocate_parameters = true)
        : cfg_(config), max_batch_(max_batch), max_tokens_(max_tokens), max_options_(max_options),
          stream_(stream), trainable_(trainable) {
        using namespace decision_detail;
        if (max_batch < 1 || max_tokens < 1 || max_options < 1 || max_options > 4096 ||
                config.head_layers < 0 || config.head_layers > 32 || config.n_act < 1 || config.n_act > 256)
            throw std::invalid_argument("invalid native decision model dimensions");
        try {
            blas_check(cublasCreate(&blas_));
            blas_check(cublasSetStream(blas_, stream));
            blas_check(cublasSetMathMode(blas_, CUBLAS_PEDANTIC_MATH));
            encoder_.reset(new Encoder(config.encoder, max_batch, max_tokens, stream, trainable, false));
            encoder_count_ = encoder_->parameters().size();
            if (config.head_layers) {
                head_.reset(new Encoder(head_config(config), max_batch, max_tokens, stream, trainable, false));
                head_count_ = head_->parameters().size();
            }
            params_ = decision_parameter_specs(config);
            own_start_ = encoder_count_ + head_count_;
            for (size_t i = 0; i < params_.size(); ++i) {
                auto& p = params_[i];
                if (allocate_parameters) {
                    p.data = allocate<float>(p.count);
                    cuda_check(cudaMemsetAsync(p.data, 0, p.count * sizeof(float), stream));
                    if (trainable) {
                        p.grad = allocate<float>(p.count);
                        cuda_check(cudaMemsetAsync(p.grad, 0, p.count * sizeof(float), stream));
                    }
                }
            }
            if (allocate_parameters) bind_children();
            allocate_workspaces();
        } catch (...) { release(); throw; }
    }
    ~DecisionModel() { release(); }
    DecisionModel(const DecisionModel&) = delete;
    DecisionModel& operator=(const DecisionModel&) = delete;

    const DecisionConfig& config() const { return cfg_; }
    const std::vector<Parameter>& parameters() const { return params_; }
    cudaStream_t stream() const { return stream_; }
    void set_stream(cudaStream_t stream) {
        decision_detail::blas_check(cublasSetStream(blas_, stream));
        encoder_->set_stream(stream); if (head_) head_->set_stream(stream); stream_ = stream;
    }
    void bind_parameters(const std::vector<float*>& data, const std::vector<float*>& gradients = {}) {
        using namespace decision_detail;
        if (data.size() != params_.size() || (trainable_ && gradients.size() != params_.size()))
            throw std::invalid_argument("decision parameter binding count mismatch");
        for (size_t i = 0; i < params_.size(); ++i)
            if (!data[i] || (trainable_ && !gradients[i])) throw std::invalid_argument("null decision parameter storage");
        cuda_check(cudaStreamSynchronize(stream_));
        for (auto& p : params_) {
            release_parameter(p.data, data, gradients);
            release_parameter(p.grad, data, gradients);
        }
        for (size_t i = 0; i < params_.size(); ++i) {
            params_[i].data = data[i]; params_[i].grad = trainable_ ? gradients[i] : nullptr;
        }
        bind_children(); last_batch_ = 0;
    }
    void zero_grad() {
        if (!trainable_) throw std::logic_error("inference-only decision model has no gradients");
        for (auto& p : params_) decision_detail::cuda_check(cudaMemsetAsync(p.grad, 0, p.count * sizeof(float), stream_));
    }

    // Initialize only newly attached heads/critic. Imported encoder and Laya
    // parameters are loaded by name afterwards; no imported tensor is reshaped.
    void initialize_new_heads(uint64_t seed = 1, bool critic_only = false) {
        using namespace decision_detail;
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> random(0.0f, 0.02f);
        size_t start = critic_only ? params_.size() - (cfg_.value_head ? 2 : 0) : encoder_count_;
        for (size_t i = start; i < params_.size(); ++i) {
            auto& p = params_[i]; std::vector<float> values(p.count);
            bool norm = p.name.find("norm") != std::string::npos || p.name == "scorer.0.weight";
            bool bias = p.name.size() >= 4 && p.name.compare(p.name.size() - 4, 4, "bias") == 0;
            for (float& value : values) value = bias ? 0.0f : norm ? 1.0f : random(rng);
            cuda_check(cudaMemcpyAsync(p.data, values.data(), p.count * sizeof(float), cudaMemcpyHostToDevice, stream_));
            cuda_check(cudaStreamSynchronize(stream_));
        }
    }

    DecisionOutput forward(const int* ids, const int* attention_mask, const int* token_types,
            const int* markers, const int* marker_mask, const int* qtype, int B, int T, int K) {
        using namespace decision_detail;
        if (B < 1 || B > max_batch_ || T < 1 || T > max_tokens_ || K < 1 || K > max_options_ ||
                !ids || !markers || !marker_mask || !qtype)
            throw std::invalid_argument("invalid native decision input dimensions");
        if (!params_[0].data) throw std::logic_error("decision weights are not bound");
        int D = cfg_.encoder.width, N = B * K;
        last_batch_ = B; last_tokens_ = T; last_options_ = K;
        cuda_check(cudaMemcpyAsync(markers_, markers, N * sizeof(int), cudaMemcpyDeviceToDevice, stream_));
        cuda_check(cudaMemcpyAsync(marker_mask_, marker_mask, N * sizeof(int), cudaMemcpyDeviceToDevice, stream_));
        cuda_check(cudaMemcpyAsync(qtype_, qtype, B * sizeof(int), cudaMemcpyDeviceToDevice, stream_));
        auto hidden = encoder_->forward(ids, attention_mask, token_types, B, T);
        add_type<<<grid((size_t)B * T * D), threads, 0, stream_>>>(typed_, hidden, p(0).data, qtype_, B, T, D);
        final_hidden_ = head_ ? head_->forward_hidden(typed_, attention_mask, B, T) : typed_;
        gather<<<grid((size_t)N * D), threads, 0, stream_>>>(gathered_, final_hidden_, markers_, B, T, K, D);
        norm_forward<<<N, threads, 0, stream_>>>(gathered_, p(1).data, p(2).data, normalized_, means_, rstd_, D);
        linear(normalized_, 3, 4, score_pre_, N, D, D);
        gelu_forward<<<grid((size_t)N * D), threads, 0, stream_>>>(score_pre_, score_act_, (size_t)N * D);
        linear(score_act_, 5, 6, logits_, N, D, 1);
        mask_logits<<<grid(N), threads, 0, stream_>>>(logits_, marker_mask_, N, -1e4f);
        act_input<<<B, threads, 0, stream_>>>(act_input_, pooled_, final_hidden_, logits_, marker_mask_, B, T, K, D);
        linear(act_input_, 7, 8, act_pre_, B, D + 4, 256);
        gelu_forward<<<grid((size_t)B * 256), threads, 0, stream_>>>(act_pre_, act_act_, (size_t)B * 256);
        linear(act_act_, 9, 10, act_logits_, B, 256, cfg_.n_act);
        if (cfg_.value_head) linear(pooled_, 11, 12, values_, B, D, 1);
        cuda_check(cudaGetLastError());
        return {logits_, act_logits_, values_};
    }

    void backward(const float* grad_logits, const float* grad_act = nullptr, const float* grad_value = nullptr) {
        using namespace decision_detail;
        if (!trainable_ || !last_batch_) throw std::logic_error("decision backward requires a trainable forward");
        int B = last_batch_, T = last_tokens_, K = last_options_, D = cfg_.encoder.width, N = B * K;
        cuda_check(cudaMemsetAsync(d_hidden_, 0, (size_t)B * T * D * sizeof(float), stream_));
        if (grad_logits) {
            cuda_check(cudaMemcpyAsync(d_logits_, grad_logits, N * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
            mask_logits<<<grid(N), threads, 0, stream_>>>(d_logits_, marker_mask_, N, 0);
            linear_backward(score_act_, d_logits_, 5, 6, d_score_, N, D, 1);
            gelu_backward<<<grid((size_t)N * D), threads, 0, stream_>>>(score_pre_, d_score_, (size_t)N * D);
            linear_backward(normalized_, d_score_, 3, 4, d_normalized_, N, D, D);
            norm_backward_input<<<N, threads, 0, stream_>>>(d_normalized_, gathered_, p(1).data,
                means_, rstd_, nullptr, d_gathered_, D);
            norm_backward_params<<<D, threads, 0, stream_>>>(d_normalized_, gathered_, means_, rstd_,
                p(1).grad, p(2).grad, N, D);
            scatter<<<grid((size_t)N * D), threads, 0, stream_>>>(d_hidden_, d_gathered_, markers_, B, T, K, D);
        }
        if (grad_act) {
            linear_backward(act_act_, grad_act, 9, 10, d_act_, B, 256, cfg_.n_act);
            gelu_backward<<<grid((size_t)B * 256), threads, 0, stream_>>>(act_pre_, d_act_, (size_t)B * 256);
            linear_backward(act_input_, d_act_, 7, 8, d_act_input_, B, D + 4, 256);
            scatter_cls<<<grid(B * D), threads, 0, stream_>>>(d_hidden_, d_act_input_, B, T, D, D + 4);
        }
        if (grad_value) {
            if (!cfg_.value_head) throw std::invalid_argument("no decision value head");
            linear_backward(pooled_, grad_value, 11, 12, d_pooled_, B, D, 1);
            scatter_cls<<<grid(B * D), threads, 0, stream_>>>(d_hidden_, d_pooled_, B, T, D, D);
        }
        float* d_typed = head_ ? head_->backward(d_hidden_) : d_hidden_;
        type_backward<<<grid((size_t)B * T * D), threads, 0, stream_>>>(d_typed, p(0).grad, qtype_, B, T, D);
        encoder_->backward(d_typed);
        cuda_check(cudaGetLastError());
    }

private:
    DecisionConfig cfg_;
    int max_batch_, max_tokens_, max_options_, last_batch_ = 0, last_tokens_ = 0, last_options_ = 0;
    cudaStream_t stream_;
    bool trainable_;
    cublasHandle_t blas_ = nullptr;
    std::unique_ptr<Encoder> encoder_, head_;
    size_t encoder_count_ = 0, head_count_ = 0, own_start_ = 0;
    std::vector<Parameter> params_;
    std::vector<void*> allocations_;
    int *markers_ = nullptr, *marker_mask_ = nullptr, *qtype_ = nullptr;
    float *typed_ = nullptr, *final_hidden_ = nullptr, *gathered_ = nullptr, *normalized_ = nullptr;
    float *means_ = nullptr, *rstd_ = nullptr, *score_pre_ = nullptr, *score_act_ = nullptr, *logits_ = nullptr;
    float *act_input_ = nullptr, *pooled_ = nullptr, *act_pre_ = nullptr, *act_act_ = nullptr;
    float *act_logits_ = nullptr, *values_ = nullptr;
    float *d_hidden_ = nullptr, *d_logits_ = nullptr, *d_score_ = nullptr, *d_normalized_ = nullptr;
    float *d_gathered_ = nullptr, *d_act_ = nullptr, *d_act_input_ = nullptr, *d_pooled_ = nullptr;

    Parameter& p(size_t index) { return params_[own_start_ + index]; }
    template<class T> T* allocate(size_t count) {
        T* pointer = nullptr;
        decision_detail::cuda_check(cudaMalloc(&pointer, count * sizeof(T)));
        allocations_.push_back(pointer); return pointer;
    }
    void release() {
        if (stream_) cudaStreamSynchronize(stream_);
        head_.reset(); encoder_.reset();
        for (void* p : allocations_) cudaFree(p);
        allocations_.clear();
        if (blas_) cublasDestroy(blas_);
        blas_ = nullptr;
    }
    void release_parameter(float* pointer, const std::vector<float*>& data, const std::vector<float*>& gradients) {
        auto it = std::find(allocations_.begin(), allocations_.end(), pointer);
        if (it == allocations_.end() || std::find(data.begin(), data.end(), pointer) != data.end() ||
                std::find(gradients.begin(), gradients.end(), pointer) != gradients.end()) return;
        decision_detail::cuda_check(cudaFree(pointer)); allocations_.erase(it);
    }
    void bind_children() {
        auto bind = [&](Encoder& model, size_t start, size_t count) {
            std::vector<float*> data, gradient;
            for (size_t i = start; i < start + count; ++i) {
                data.push_back(params_[i].data);
                if (trainable_) gradient.push_back(params_[i].grad);
            }
            model.bind_parameters(data, gradient);
        };
        bind(*encoder_, 0, encoder_count_);
        if (head_) bind(*head_, encoder_count_, head_count_);
    }
    void allocate_workspaces() {
        size_t B = max_batch_, N = B * max_options_, D = cfg_.encoder.width;
        markers_ = allocate<int>(N); marker_mask_ = allocate<int>(N); qtype_ = allocate<int>(B);
        typed_ = allocate<float>(B * max_tokens_ * D);
        gathered_ = allocate<float>(N * D); normalized_ = allocate<float>(N * D);
        means_ = allocate<float>(N); rstd_ = allocate<float>(N);
        score_pre_ = allocate<float>(N * D); score_act_ = allocate<float>(N * D); logits_ = allocate<float>(N);
        act_input_ = allocate<float>(B * (D + 4)); pooled_ = allocate<float>(B * D);
        act_pre_ = allocate<float>(B * 256); act_act_ = allocate<float>(B * 256); act_logits_ = allocate<float>(B * cfg_.n_act);
        if (cfg_.value_head) values_ = allocate<float>(B);
        if (trainable_) {
            d_hidden_ = allocate<float>(B * max_tokens_ * D); d_logits_ = allocate<float>(N);
            d_score_ = allocate<float>(N * D); d_normalized_ = allocate<float>(N * D); d_gathered_ = allocate<float>(N * D);
            d_act_ = allocate<float>(B * 256); d_act_input_ = allocate<float>(B * (D + 4)); d_pooled_ = allocate<float>(B * D);
        }
    }
    void linear(const float* input, size_t weight, size_t bias, float* output, int rows, int in, int out) {
        using namespace decision_detail;
        const float one = 1, zero = 0;
        blas_check(cublasSgemm(blas_, CUBLAS_OP_T, CUBLAS_OP_N, out, rows, in,
            &one, p(weight).data, in, input, in, &zero, output, out));
        bias_forward<<<grid((size_t)rows * out), threads, 0, stream_>>>(output, p(bias).data, (size_t)rows * out, out);
    }
    void linear_backward(const float* input, const float* gradient, size_t weight, size_t bias,
            float* dx, int rows, int in, int out) {
        using namespace decision_detail;
        const float one = 1, zero = 0;
        blas_check(cublasSgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, in, rows, out,
            &one, p(weight).data, in, gradient, out, &zero, dx, in));
        blas_check(cublasSgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_T, in, out, rows,
            &one, input, in, gradient, out, &one, p(weight).grad, in));
        bias_backward<<<out, threads, 0, stream_>>>(gradient, p(bias).grad, rows, out);
    }
};
} // namespace pretrained
#endif
