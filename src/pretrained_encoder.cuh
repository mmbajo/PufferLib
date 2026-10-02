#ifndef PUFFER_PRETRAINED_ENCODER_CUH
#define PUFFER_PRETRAINED_ENCODER_CUH

#include "decision_transformer.cuh"

// Dropout-free FP32 execution of imported encoder weights. This matches the
// supported Hugging Face models in eval mode; PPO deliberately disables dropout.
// BertModel's optional pooler is excluded: forward returns all token features.
namespace pretrained {

struct Config {
    std::string family = "modernbert"; // modernbert, bert, laya_head
    int width = 768, layers = 22, heads = 12, intermediate = 1152;
    int vocab = 50368, max_positions = 8192, type_vocab = 2, pad_token_id = 50283;
    float epsilon = 1e-5f;
    bool attention_bias = false, mlp_bias = false, norm_bias = false;
    int local_window = 128, global_every = 3;
    float global_rope_theta = 160000.0f, local_rope_theta = 10000.0f;
    // Empty uses global_every; otherwise 0=full and 1=sliding for each layer.
    std::vector<int> layer_types;
};

using Parameter = decision::Parameter;

// CPU-only registry; data/grad are null. Names match the bare HF model, or the
// torch.nn.TransformerEncoder used for Laya heads, without a parent prefix.
inline std::vector<Parameter> parameter_specs(const Config& config);

class Encoder {
public:
    // With allocate_parameters=false, bind external buffers before execution.
    // Owned parameters start at zero and require imported weights before use.
    Encoder(const Config&, int max_batch, int max_tokens, cudaStream_t stream = nullptr,
            bool trainable = true, bool allocate_parameters = true);
    ~Encoder();
    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;
    Encoder(Encoder&&) = delete;
    Encoder& operator=(Encoder&&) = delete;
    const Config& config() const;
    const std::vector<Parameter>& parameters() const;
    cudaStream_t stream() const;
    void set_stream(cudaStream_t);
    void bind_parameters(const std::vector<float*>& data, const std::vector<float*>& gradients = {});
    void zero_grad();
    // Device int32 IDs/masks/types, shape [B,T]. A null mask means all keys are
    // visible; a null token_type_ids means zero. Position IDs are 0..T-1.
    float* forward(const int* ids, const int* mask, const int* token_type_ids, int B, int T);
    // Embedding-free Laya head stack: device float32 [B,T,D].
    float* forward_hidden(const float* hidden, const int* mask, int B, int T);
    // Accumulates parameter gradients; returns gradient w.r.t. input features
    // (before the embedding normalization for models with token embeddings).
    float* backward(const float* gradient);
    // Diagnostics for trainable workspaces; raw block output before final norm.
    float* layer_output(int layer) const;
private:
    struct Impl;
    Impl* impl_;
};

namespace detail {
using decision::transformer_detail::cuda_check;
using decision::transformer_detail::blas_check;
using decision::transformer_detail::block_sum;
using decision::transformer_detail::block_max;
using decision::transformer_detail::grid;
constexpr int threads = 256;

inline void validate(const Config& c) {
    if (c.family != "modernbert" && c.family != "bert" && c.family != "laya_head")
        throw std::invalid_argument("unsupported pretrained encoder family");
    if (c.width < 1 || c.width > 16384 || c.layers < 1 || c.layers > 256 ||
            c.heads < 1 || c.width % c.heads || c.intermediate < 1 || c.intermediate > 65536 ||
            !std::isfinite(c.epsilon) || c.epsilon <= 0)
        throw std::invalid_argument("invalid pretrained encoder dimensions or epsilon");
    if (c.family != "laya_head" && (c.vocab < 1 || c.max_positions < 1 ||
            c.pad_token_id < -1 || c.pad_token_id >= c.vocab))
        throw std::invalid_argument("invalid pretrained embedding configuration");
    if (c.family == "bert" && c.type_vocab < 1)
        throw std::invalid_argument("BERT token type vocabulary must be positive");
    if (c.family == "modernbert") {
        if ((c.width / c.heads) % 2 || c.local_window < 0 || c.global_every < 1 ||
                !std::isfinite(c.global_rope_theta) || c.global_rope_theta <= 0 ||
                !std::isfinite(c.local_rope_theta) || c.local_rope_theta <= 0)
            throw std::invalid_argument("invalid ModernBERT rotary or window configuration");
        if (!c.layer_types.empty() && c.layer_types.size() != static_cast<size_t>(c.layers))
            throw std::invalid_argument("ModernBERT layer_types must describe every layer");
        for (int type : c.layer_types)
            if (type != 0 && type != 1) throw std::invalid_argument("unknown attention layer type");
    }
}

static __global__ void norm_forward(const float* x, const float* gamma,
        const float* beta, float* y, float* means, float* invstd, int D, float eps) {
    size_t base = static_cast<size_t>(blockIdx.x) * D;
    float sum = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) sum += x[base + d];
    float mean = block_sum(sum) / D;
    float variance = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float z = x[base + d] - mean; variance += z * z;
    }
    float rstd = rsqrtf(block_sum(variance) / D + eps);
    if (threadIdx.x == 0) { means[blockIdx.x] = mean; invstd[blockIdx.x] = rstd; }
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        y[base + d] = (x[base + d] - mean) * rstd * gamma[d] + (beta ? beta[d] : 0.0f);
}

static __global__ void norm_parameters_backward(const float* dy, const float* x,
        const float* means, const float* invstd, float* dg, float* db, int M, int D) {
    int d = blockIdx.x;
    float g = 0, b = 0;
    for (int row = threadIdx.x; row < M; row += blockDim.x) {
        size_t i = static_cast<size_t>(row) * D + d;
        g += dy[i] * ((x[i] - means[row]) * invstd[row]); b += dy[i];
    }
    g = block_sum(g); b = block_sum(b);
    if (threadIdx.x == 0) { dg[d] += g; if (db) db[d] += b; }
}

static __global__ void embedding_forward(const int* ids, const int* types,
        const float* word, const float* positions, const float* type_embeddings,
        float* x, int B, int T, int D, int vocab, int type_vocab) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(B) * T * D) return;
    int row = i / D, d = i % D, id = ids[row];
    int type = types ? types[row] : 0;
    if (id < 0 || id >= vocab || (type_embeddings && (type < 0 || type >= type_vocab))) {
        x[i] = nanf(""); return;
    }
    float value = word[static_cast<size_t>(id) * D + d];
    if (type_embeddings) value += type_embeddings[static_cast<size_t>(type) * D + d];
    if (positions) value += positions[static_cast<size_t>(row % T) * D + d];
    x[i] = value;
}

static __global__ void embedding_backward(const float* dy, const int* ids,
        const int* types, float* word, float* positions, float* type_embeddings,
        int B, int T, int D, int vocab, int type_vocab, int pad) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(B) * T * D) return;
    int row = i / D, d = i % D, id = ids[row];
    int type = types ? types[row] : 0;
    if (id < 0 || id >= vocab || (type_embeddings && (type < 0 || type >= type_vocab))) return;
    if (id != pad) atomicAdd(word + static_cast<size_t>(id) * D + d, dy[i]);
    if (positions) atomicAdd(positions + static_cast<size_t>(row % T) * D + d, dy[i]);
    if (type_embeddings) atomicAdd(type_embeddings + static_cast<size_t>(type) * D + d, dy[i]);
}

// planar=true means three [B,T,D] planes (BERT's separate Q/K/V projections).
static __global__ void split_qkv(const float* qkv, float* q, float* k, float* v,
        int B, int T, int D, int H, bool planar) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t N = static_cast<size_t>(B) * T * D;
    if (i >= N) return;
    int K = D / H, c = i % K, t = (i / K) % T;
    int h = (i / (static_cast<size_t>(K) * T)) % H;
    int b = i / (static_cast<size_t>(T) * D);
    size_t row = static_cast<size_t>(b) * T + t;
    size_t source = row * (planar ? D : 3 * D) + h * K + c;
    size_t stride = planar ? N : static_cast<size_t>(D);
    q[i] = qkv[source]; k[i] = qkv[source + stride]; v[i] = qkv[source + 2 * stride];
}

static __global__ void merge_qkv(const float* q, const float* k, const float* v,
        float* qkv, int B, int T, int D, int H, bool planar) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t N = static_cast<size_t>(B) * T * D;
    if (i >= N) return;
    int K = D / H, c = i % K, t = (i / K) % T;
    int h = (i / (static_cast<size_t>(K) * T)) % H;
    int b = i / (static_cast<size_t>(T) * D);
    size_t target = (static_cast<size_t>(b) * T + t) * (planar ? D : 3 * D) + h * K + c;
    size_t stride = planar ? N : static_cast<size_t>(D);
    qkv[target] = q[i]; qkv[target + stride] = k[i]; qkv[target + 2 * stride] = v[i];
}

static __global__ void transpose_heads(const float* input, float* output,
        int B, int T, int D, int H, bool split) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(B) * T * D) return;
    int K = D / H, d = i % D, t = (i / D) % T;
    int b = i / (static_cast<size_t>(T) * D);
    size_t other = ((static_cast<size_t>(b) * H + d / K) * T + t) * K + d % K;
    if (split) output[other] = input[i]; else output[i] = input[other];
}

static __global__ void rotary_tables(float* cosine, float* sine, int T, int K, float theta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half = K / 2;
    if (i >= T * half) return;
    int d = i % half, t = i / half;
    float inv = 1.0f / powf(theta, static_cast<float>(2 * d) / K);
    sincosf(t * inv, sine + i, cosine + i);
}

static __global__ void rotary(float* q, float* k, const float* cosine,
        const float* sine, int B, int T, int D, int H, bool backward) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int K = D / H, half = K / 2;
    if (i >= static_cast<size_t>(B) * T * D / 2) return;
    int d = i % half, t = (i / half) % T;
    size_t base = (i / half) * K + d;
    float c = cosine[t * half + d], s = sine[t * half + d] * (backward ? -1.0f : 1.0f);
    float q1 = q[base], q2 = q[base + half], k1 = k[base], k2 = k[base + half];
    q[base] = q1 * c - q2 * s; q[base + half] = q2 * c + q1 * s;
    k[base] = k1 * c - k2 * s; k[base + half] = k2 * c + k1 * s;
}

static __global__ void attention_softmax(float* scores, const int* mask,
        int T, int H, int half_window) {
    size_t base = static_cast<size_t>(blockIdx.x) * T;
    int b = blockIdx.x / (H * T), q = blockIdx.x % T;
    float maximum = -FLT_MAX;
    for (int k = threadIdx.x; k < T; k += blockDim.x) {
        bool visible = (!mask || mask[b * T + k]) && (half_window < 0 || abs(q - k) <= half_window);
        float value = visible ? scores[base + k] : -FLT_MAX;
        scores[base + k] = value; maximum = fmaxf(maximum, value);
    }
    maximum = block_max(maximum);
    float sum = 0;
    for (int k = threadIdx.x; k < T; k += blockDim.x) {
        float value = expf(scores[base + k] - maximum);
        scores[base + k] = value; sum += value;
    }
    sum = block_sum(sum);
    for (int k = threadIdx.x; k < T; k += blockDim.x) scores[base + k] /= sum;
}

static __global__ void attention_softmax_backward(const float* p, float* g, int T, float scale) {
    size_t base = static_cast<size_t>(blockIdx.x) * T;
    float sum = 0;
    for (int k = threadIdx.x; k < T; k += blockDim.x) sum += p[base + k] * g[base + k];
    sum = block_sum(sum);
    for (int k = threadIdx.x; k < T; k += blockDim.x)
        g[base + k] = scale * p[base + k] * (g[base + k] - sum);
}

static __global__ void activation_forward(const float* input, float* output,
        size_t count, int F, int mode) { // 0 GELU, 1 GEGLU, 2 ReLU
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    size_t source = mode == 1 ? (i / F) * 2 * F + i % F : i;
    float x = input[source];
    float value = mode == 2 ? fmaxf(0, x) : 0.5f * x * (1 + erff(x * 0.7071067811865475244f));
    output[i] = mode == 1 ? value * input[source + F] : value;
}

static __global__ void activation_backward(const float* input, const float* dy,
        float* dx, size_t count, int F, int mode) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    size_t source = mode == 1 ? (i / F) * 2 * F + i % F : i;
    float x = input[source];
    float cdf = 0.5f * (1 + erff(x * 0.7071067811865475244f));
    float derivative = mode == 2 ? (x > 0 ? 1.0f : 0.0f) :
        cdf + x * 0.39894228040143267794f * expf(-0.5f * x * x);
    dx[source] = dy[i] * derivative * (mode == 1 ? input[source + F] : 1.0f);
    if (mode == 1) dx[source + F] = dy[i] * x * cdf;
}

} // namespace detail

inline std::vector<Parameter> parameter_specs(const Config& c) {
    detail::validate(c);
    std::vector<Parameter> result;
    auto add = [&](const std::string& name, std::vector<int> shape) {
        size_t count = 1;
        for (int d : shape) {
            if (count > std::numeric_limits<size_t>::max() / static_cast<size_t>(d))
                throw std::invalid_argument("pretrained parameter shape overflow");
            count *= d;
        }
        result.push_back({name, shape, count, nullptr, nullptr});
    };
    auto norm = [&](const std::string& name, bool bias) {
        add(name + ".weight", {c.width}); if (bias) add(name + ".bias", {c.width});
    };
    auto linear = [&](const std::string& name, int out, int in, bool bias) {
        add(name + ".weight", {out, in}); if (bias) add(name + ".bias", {out});
    };
    int D = c.width, F = c.intermediate;
    if (c.family == "modernbert") {
        add("embeddings.tok_embeddings.weight", {c.vocab, D});
        norm("embeddings.norm", c.norm_bias);
        for (int l = 0; l < c.layers; ++l) {
            std::string p = "layers." + std::to_string(l) + ".";
            if (l) norm(p + "attn_norm", c.norm_bias);
            linear(p + "attn.Wqkv", 3 * D, D, c.attention_bias);
            linear(p + "attn.Wo", D, D, c.attention_bias);
            norm(p + "mlp_norm", c.norm_bias);
            linear(p + "mlp.Wi", 2 * F, D, c.mlp_bias);
            linear(p + "mlp.Wo", D, F, c.mlp_bias);
        }
        norm("final_norm", c.norm_bias);
    } else if (c.family == "bert") {
        add("embeddings.word_embeddings.weight", {c.vocab, D});
        add("embeddings.position_embeddings.weight", {c.max_positions, D});
        add("embeddings.token_type_embeddings.weight", {c.type_vocab, D});
        norm("embeddings.LayerNorm", true);
        for (int l = 0; l < c.layers; ++l) {
            std::string p = "encoder.layer." + std::to_string(l) + ".";
            for (const char* name : {"query", "key", "value"}) linear(p + "attention.self." + name, D, D, true);
            linear(p + "attention.output.dense", D, D, true);
            norm(p + "attention.output.LayerNorm", true);
            linear(p + "intermediate.dense", F, D, true);
            linear(p + "output.dense", D, F, true);
            norm(p + "output.LayerNorm", true);
        }
    } else {
        for (int l = 0; l < c.layers; ++l) {
            std::string p = "layers." + std::to_string(l) + ".";
            add(p + "self_attn.in_proj_weight", {3 * D, D});
            add(p + "self_attn.in_proj_bias", {3 * D});
            linear(p + "self_attn.out_proj", D, D, true);
            linear(p + "linear1", F, D, true); linear(p + "linear2", D, F, true);
            norm(p + "norm1", true); norm(p + "norm2", true);
        }
    }
    return result;
}

struct Encoder::Impl {
    struct Linear { int w = -1, b = -1, in = 0, out = 0; };
    struct Norm { int w = -1, b = -1; };
    struct Block { Linear qkv, q, k, v, output, ff1, ff2; Norm n1, n2; };
    struct Work {
        float *n1, *mean1, *rstd1, *qkv, *q, *k, *v, *probabilities;
        float *context_heads, *context, *res1, *n2, *mean2, *rstd2;
        float *ff1, *activated, *res2, *output;
    };
    Config cfg;
    int max_batch, max_tokens, B = 0, T = 0;
    bool trainable, has_mask = false;
    cudaStream_t stream;
    cublasHandle_t handle = nullptr;
    std::vector<Parameter> params;
    std::vector<void*> allocations;
    std::vector<Block> blocks;
    std::vector<Work> work;
    int word = -1, position = -1, type_embedding = -1;
    Norm embedding_norm, final_norm;
    int *ids = nullptr, *mask = nullptr, *types = nullptr;
    float *embedded = nullptr, *embedding_output = nullptr, *embedding_mean = nullptr, *embedding_rstd = nullptr;
    float *output = nullptr, *final_mean = nullptr, *final_rstd = nullptr;
    float *cos_global = nullptr, *sin_global = nullptr, *cos_local = nullptr, *sin_local = nullptr;
    float *dh = nullptr, *dn = nullptr, *dr = nullptr, *dc = nullptr, *dch = nullptr;
    float *dq = nullptr, *dk = nullptr, *dv = nullptr, *dqkv = nullptr, *da = nullptr;
    float *df = nullptr, *dfraw = nullptr, *dinput = nullptr;

    Impl(const Config& c, int mb, int mt, cudaStream_t s, bool train, bool allocate_params)
            : cfg(c), max_batch(mb), max_tokens(mt), trainable(train), stream(s), params(parameter_specs(c)) {
        if (mb < 1 || mt < 1 || static_cast<int64_t>(mb) * mt * c.heads > INT_MAX ||
                (c.family != "laya_head" && mt > c.max_positions))
            throw std::invalid_argument("invalid pretrained encoder workspace dimensions");
        try {
            detail::blas_check(cublasCreate(&handle));
            detail::blas_check(cublasSetStream(handle, stream));
            detail::blas_check(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
            if (allocate_params) {
                for (auto& p : params) {
                    p.data = allocate<float>(p.count);
                    detail::cuda_check(cudaMemsetAsync(p.data, 0, p.count * sizeof(float), stream));
                    if (trainable) {
                        p.grad = allocate<float>(p.count);
                        detail::cuda_check(cudaMemsetAsync(p.grad, 0, p.count * sizeof(float), stream));
                    }
                }
            }
            describe_blocks();
            allocate_workspace();
        } catch (...) { release(); throw; }
    }
    ~Impl() { release(); }
    void release() noexcept {
        if (handle) cublasDestroy(handle);
        handle = nullptr;
        for (void* p : allocations) cudaFree(p);
        allocations.clear();
    }
    template <typename V> V* allocate(size_t n) {
        if (!n || n > std::numeric_limits<size_t>::max() / sizeof(V))
            throw std::invalid_argument("pretrained allocation size overflow");
        V* p = nullptr;
        detail::cuda_check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(V)));
        try { allocations.push_back(p); } catch (...) { cudaFree(p); throw; }
        return p;
    }
    int index(const std::string& name) const {
        for (size_t i = 0; i < params.size(); ++i) if (params[i].name == name) return static_cast<int>(i);
        return -1;
    }
    float* data(int i) const { return i < 0 ? nullptr : params[i].data; }
    float* grad(int i) const { return i < 0 ? nullptr : params[i].grad; }
    Linear linear_spec(const std::string& name, int in, int out) const {
        return {index(name + ".weight"), index(name + ".bias"), in, out};
    }
    Norm norm_spec(const std::string& name) const { return {index(name + ".weight"), index(name + ".bias")}; }
    bool modern() const { return cfg.family == "modernbert"; }
    bool bert() const { return cfg.family == "bert"; }
    bool head() const { return cfg.family == "laya_head"; }
    bool local(int l) const { return modern() && (cfg.layer_types.empty() ? l % cfg.global_every != 0 : cfg.layer_types[l] == 1); }
    int activation() const { return modern() ? 1 : head() ? 2 : 0; }

    void describe_blocks() {
        int D = cfg.width, F = cfg.intermediate;
        if (modern()) {
            word = index("embeddings.tok_embeddings.weight");
            embedding_norm = norm_spec("embeddings.norm"); final_norm = norm_spec("final_norm");
        } else if (bert()) {
            word = index("embeddings.word_embeddings.weight");
            position = index("embeddings.position_embeddings.weight");
            type_embedding = index("embeddings.token_type_embeddings.weight");
            embedding_norm = norm_spec("embeddings.LayerNorm");
        }
        for (int l = 0; l < cfg.layers; ++l) {
            Block w;
            if (modern()) {
                std::string p = "layers." + std::to_string(l) + ".";
                if (l) w.n1 = norm_spec(p + "attn_norm");
                w.qkv = linear_spec(p + "attn.Wqkv", D, 3 * D);
                w.output = linear_spec(p + "attn.Wo", D, D);
                w.n2 = norm_spec(p + "mlp_norm");
                w.ff1 = linear_spec(p + "mlp.Wi", D, 2 * F);
                w.ff2 = linear_spec(p + "mlp.Wo", F, D);
            } else if (bert()) {
                std::string p = "encoder.layer." + std::to_string(l) + ".";
                w.q = linear_spec(p + "attention.self.query", D, D);
                w.k = linear_spec(p + "attention.self.key", D, D);
                w.v = linear_spec(p + "attention.self.value", D, D);
                w.output = linear_spec(p + "attention.output.dense", D, D);
                w.n1 = norm_spec(p + "attention.output.LayerNorm");
                w.ff1 = linear_spec(p + "intermediate.dense", D, F);
                w.ff2 = linear_spec(p + "output.dense", F, D);
                w.n2 = norm_spec(p + "output.LayerNorm");
            } else {
                std::string p = "layers." + std::to_string(l) + ".";
                w.qkv = {index(p + "self_attn.in_proj_weight"), index(p + "self_attn.in_proj_bias"), D, 3 * D};
                w.output = linear_spec(p + "self_attn.out_proj", D, D);
                w.ff1 = linear_spec(p + "linear1", D, F); w.ff2 = linear_spec(p + "linear2", F, D);
                w.n1 = norm_spec(p + "norm1"); w.n2 = norm_spec(p + "norm2");
            }
            blocks.push_back(w);
        }
    }

    void allocate_workspace() {
        size_t M = static_cast<size_t>(max_batch) * max_tokens, N = M * cfg.width;
        size_t A = M * cfg.heads * max_tokens, F = M * cfg.intermediate;
        ids = allocate<int>(M); mask = allocate<int>(M); types = allocate<int>(M);
        embedded = allocate<float>(N);
        if (!head()) {
            embedding_output = allocate<float>(N);
            embedding_mean = allocate<float>(M); embedding_rstd = allocate<float>(M);
        } else embedding_output = embedded;
        for (int l = 0; l < (trainable ? cfg.layers : 1); ++l) {
            Work a;
            a.n1 = allocate<float>(N); a.mean1 = allocate<float>(M); a.rstd1 = allocate<float>(M);
            a.qkv = allocate<float>(3 * N);
            a.q = allocate<float>(N); a.k = allocate<float>(N); a.v = allocate<float>(N);
            a.probabilities = allocate<float>(A);
            a.context_heads = allocate<float>(N); a.context = allocate<float>(N);
            a.res1 = allocate<float>(N); a.n2 = allocate<float>(N);
            a.mean2 = allocate<float>(M); a.rstd2 = allocate<float>(M);
            a.ff1 = allocate<float>(F * (modern() ? 2 : 1)); a.activated = allocate<float>(F);
            a.res2 = allocate<float>(N); a.output = allocate<float>(N);
            work.push_back(a);
        }
        output = allocate<float>(N);
        if (modern()) {
            final_mean = allocate<float>(M); final_rstd = allocate<float>(M);
            size_t frequencies = static_cast<size_t>(max_tokens) * (cfg.width / cfg.heads) / 2;
            cos_global = allocate<float>(frequencies); sin_global = allocate<float>(frequencies);
            cos_local = allocate<float>(frequencies); sin_local = allocate<float>(frequencies);
            detail::rotary_tables<<<detail::grid(frequencies), detail::threads, 0, stream>>>(
                cos_global, sin_global, max_tokens, cfg.width / cfg.heads, cfg.global_rope_theta);
            detail::rotary_tables<<<detail::grid(frequencies), detail::threads, 0, stream>>>(
                cos_local, sin_local, max_tokens, cfg.width / cfg.heads, cfg.local_rope_theta);
        }
        if (trainable) {
            dh = allocate<float>(N); dn = allocate<float>(N); dr = allocate<float>(N);
            dc = allocate<float>(N); dch = allocate<float>(N);
            dq = allocate<float>(N); dk = allocate<float>(N); dv = allocate<float>(N);
            dqkv = allocate<float>(3 * N); da = allocate<float>(A);
            df = allocate<float>(F); dfraw = allocate<float>(F * (modern() ? 2 : 1));
            dinput = allocate<float>(N);
        }
        detail::cuda_check(cudaGetLastError());
    }

    void linear(const float* x, Linear w, float* y, int M) {
        const float one = 1, zero = 0;
        detail::blas_check(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            w.out, M, w.in, &one, data(w.w), w.in, x, w.in, &zero, y, w.out));
        if (w.b >= 0) decision::transformer_detail::bias_forward<<<detail::grid(static_cast<size_t>(M) * w.out), detail::threads, 0, stream>>>(
            y, data(w.b), static_cast<size_t>(M) * w.out, w.out);
    }
    void linear_backward(const float* x, const float* dy, Linear w, float* dx, int M, float beta = 0) {
        const float one = 1;
        detail::blas_check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            w.in, M, w.out, &one, data(w.w), w.in, dy, w.out, &beta, dx, w.in));
        detail::blas_check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
            w.in, w.out, M, &one, x, w.in, dy, w.out, &one, grad(w.w), w.in));
        if (w.b >= 0) decision::transformer_detail::bias_backward<<<w.out, detail::threads, 0, stream>>>(
            dy, grad(w.b), M, w.out);
    }
    void norm(const float* x, Norm w, float* y, float* mean, float* rstd, int M) {
        detail::norm_forward<<<M, detail::threads, 0, stream>>>(
            x, data(w.w), data(w.b), y, mean, rstd, cfg.width, cfg.epsilon);
    }
    void norm_backward(const float* dy, const float* x, Norm w,
            const float* mean, const float* rstd, const float* skip, float* dx, int M) {
        decision::transformer_detail::norm_backward_input<<<M, detail::threads, 0, stream>>>(
            dy, x, data(w.w), mean, rstd, skip, dx, cfg.width);
        detail::norm_parameters_backward<<<cfg.width, detail::threads, 0, stream>>>(
            dy, x, mean, rstd, grad(w.w), grad(w.b), M, cfg.width);
    }
    void add(float* x, const float* skip, size_t n) {
        decision::transformer_detail::residual_add<<<detail::grid(n), detail::threads, 0, stream>>>(x, skip, n);
    }
    void copy(float* dst, const float* src, size_t n) {
        detail::cuda_check(cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }
    void attention_forward(Work& a, int l) {
        int D = cfg.width, H = cfg.heads, K = D / H;
        size_t N = static_cast<size_t>(B) * T * D;
        detail::split_qkv<<<detail::grid(N), detail::threads, 0, stream>>>(a.qkv, a.q, a.k, a.v, B, T, D, H, bert());
        if (modern()) detail::rotary<<<detail::grid(N / 2), detail::threads, 0, stream>>>(
            a.q, a.k, local(l) ? cos_local : cos_global, local(l) ? sin_local : sin_global, B, T, D, H, false);
        const float one = 1, zero = 0, scale = 1.0f / std::sqrt(static_cast<float>(K));
        long long vectors = static_cast<long long>(T) * K, matrices = static_cast<long long>(T) * T;
        detail::blas_check(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            T, T, K, &scale, a.k, K, vectors, a.q, K, vectors, &zero, a.probabilities, T, matrices, B * H));
        detail::attention_softmax<<<B * H * T, detail::threads, 0, stream>>>(
            a.probabilities, has_mask ? mask : nullptr, T, H, local(l) ? cfg.local_window / 2 : -1);
        detail::blas_check(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            K, T, T, &one, a.v, K, vectors, a.probabilities, T, matrices, &zero, a.context_heads, K, vectors, B * H));
        detail::transpose_heads<<<detail::grid(N), detail::threads, 0, stream>>>(a.context_heads, a.context, B, T, D, H, false);
    }
    void attention_backward(const Work& a, int l) {
        int D = cfg.width, H = cfg.heads, K = D / H;
        size_t N = static_cast<size_t>(B) * T * D;
        detail::transpose_heads<<<detail::grid(N), detail::threads, 0, stream>>>(dc, dch, B, T, D, H, true);
        const float one = 1, zero = 0;
        long long vectors = static_cast<long long>(T) * K, matrices = static_cast<long long>(T) * T;
        detail::blas_check(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
            K, T, T, &one, dch, K, vectors, a.probabilities, T, matrices, &zero, dv, K, vectors, B * H));
        detail::blas_check(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            T, T, K, &one, a.v, K, vectors, dch, K, vectors, &zero, da, T, matrices, B * H));
        detail::attention_softmax_backward<<<B * H * T, detail::threads, 0, stream>>>(a.probabilities, da, T, 1.0f / std::sqrt(static_cast<float>(K)));
        detail::blas_check(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            K, T, T, &one, a.k, K, vectors, da, T, matrices, &zero, dq, K, vectors, B * H));
        detail::blas_check(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
            K, T, T, &one, a.q, K, vectors, da, T, matrices, &zero, dk, K, vectors, B * H));
        if (modern()) detail::rotary<<<detail::grid(N / 2), detail::threads, 0, stream>>>(
            dq, dk, local(l) ? cos_local : cos_global, local(l) ? sin_local : sin_global, B, T, D, H, true);
        detail::merge_qkv<<<detail::grid(N), detail::threads, 0, stream>>>(dq, dk, dv, dqkv, B, T, D, H, bert());
    }

    void start(const int* incoming_mask, int batch, int tokens) {
        if (batch < 1 || batch > max_batch || tokens < 1 || tokens > max_tokens)
            throw std::invalid_argument("pretrained encoder batch or token count exceeds workspace");
        for (const auto& p : params) if (!p.data) throw std::invalid_argument("pretrained encoder parameters are not bound");
        B = batch; T = tokens; has_mask = incoming_mask != nullptr;
        if (has_mask) detail::cuda_check(cudaMemcpyAsync(mask, incoming_mask,
            static_cast<size_t>(B) * T * sizeof(int), cudaMemcpyDeviceToDevice, stream));
    }
    float* forward_layers() {
        int M = B * T, D = cfg.width, F = cfg.intermediate;
        size_t N = static_cast<size_t>(M) * D;
        const float* x = embedding_output;
        for (int l = 0; l < cfg.layers; ++l) {
            auto& a = work[trainable ? l : 0]; const auto& w = blocks[l];
            const float* attn_input = x;
            if (!bert() && w.n1.w >= 0) {
                norm(x, w.n1, a.n1, a.mean1, a.rstd1, M); attn_input = a.n1;
            }
            if (bert()) {
                linear(attn_input, w.q, a.qkv, M); linear(attn_input, w.k, a.qkv + N, M);
                linear(attn_input, w.v, a.qkv + 2 * N, M);
            } else linear(attn_input, w.qkv, a.qkv, M);
            attention_forward(a, l);
            linear(a.context, w.output, a.res1, M); add(a.res1, x, N);
            const float* ff_input;
            if (bert()) {
                norm(a.res1, w.n1, a.n1, a.mean1, a.rstd1, M); ff_input = a.n1;
            } else {
                norm(a.res1, w.n2, a.n2, a.mean2, a.rstd2, M); ff_input = a.n2;
            }
            linear(ff_input, w.ff1, a.ff1, M);
            detail::activation_forward<<<detail::grid(static_cast<size_t>(M) * F), detail::threads, 0, stream>>>(a.ff1, a.activated, static_cast<size_t>(M) * F, F, activation());
            if (bert()) {
                linear(a.activated, w.ff2, a.res2, M); add(a.res2, a.n1, N);
                norm(a.res2, w.n2, a.output, a.mean2, a.rstd2, M);
            } else {
                linear(a.activated, w.ff2, a.output, M); add(a.output, a.res1, N);
            }
            x = a.output;
        }
        if (modern()) norm(x, final_norm, output, final_mean, final_rstd, M);
        else copy(output, x, N);
        detail::cuda_check(cudaGetLastError()); return output;
    }

    float* backward(const float* upstream) {
        if (!trainable || B < 1 || !upstream) throw std::invalid_argument("pretrained backward requires a trainable workspace and prior forward");
        for (const auto& p : params) if (!p.grad) throw std::invalid_argument("pretrained encoder gradients are not bound");
        int M = B * T, D = cfg.width, F = cfg.intermediate;
        size_t N = static_cast<size_t>(M) * D;
        if (modern()) norm_backward(upstream, work.back().output, final_norm, final_mean, final_rstd, nullptr, dh, M);
        else copy(dh, upstream, N);
        for (int l = cfg.layers - 1; l >= 0; --l) {
            const auto& a = work[l]; const auto& w = blocks[l];
            const float* x = l ? work[l - 1].output : embedding_output;
            if (bert()) {
                norm_backward(dh, a.res2, w.n2, a.mean2, a.rstd2, nullptr, dr, M);
                linear_backward(a.activated, dr, w.ff2, df, M);
                detail::activation_backward<<<detail::grid(static_cast<size_t>(M) * F), detail::threads, 0, stream>>>(a.ff1, df, dfraw, static_cast<size_t>(M) * F, F, activation());
                linear_backward(a.n1, dfraw, w.ff1, dn, M); add(dn, dr, N);
                norm_backward(dn, a.res1, w.n1, a.mean1, a.rstd1, nullptr, dr, M);
            } else {
                linear_backward(a.activated, dh, w.ff2, df, M);
                detail::activation_backward<<<detail::grid(static_cast<size_t>(M) * F), detail::threads, 0, stream>>>(a.ff1, df, dfraw, static_cast<size_t>(M) * F, F, activation());
                linear_backward(a.n2, dfraw, w.ff1, dn, M);
                norm_backward(dn, a.res1, w.n2, a.mean2, a.rstd2, dh, dr, M);
            }
            linear_backward(a.context, dr, w.output, dc, M);
            attention_backward(a, l);
            if (bert()) {
                linear_backward(x, dqkv, w.q, dn, M);
                linear_backward(x, dqkv + N, w.k, dn, M, 1);
                linear_backward(x, dqkv + 2 * N, w.v, dn, M, 1);
                copy(dh, dn, N); add(dh, dr, N);
            } else {
                linear_backward(w.n1.w >= 0 ? a.n1 : x, dqkv, w.qkv, dn, M);
                if (w.n1.w >= 0) norm_backward(dn, x, w.n1, a.mean1, a.rstd1, dr, dh, M);
                else { copy(dh, dn, N); add(dh, dr, N); }
            }
        }
        if (head()) copy(dinput, dh, N);
        else {
            norm_backward(dh, embedded, embedding_norm, embedding_mean, embedding_rstd, nullptr, dinput, M);
            detail::embedding_backward<<<detail::grid(N), detail::threads, 0, stream>>>(dinput, ids,
                bert() ? types : nullptr, grad(word), grad(position), grad(type_embedding),
                B, T, D, cfg.vocab, cfg.type_vocab, cfg.pad_token_id);
        }
        detail::cuda_check(cudaGetLastError()); return dinput;
    }
};

inline Encoder::Encoder(const Config& c, int mb, int mt, cudaStream_t stream,
        bool trainable, bool allocate_parameters)
    : impl_(new Impl(c, mb, mt, stream, trainable, allocate_parameters)) {}
inline Encoder::~Encoder() { delete impl_; }
inline const Config& Encoder::config() const { return impl_->cfg; }
inline const std::vector<Parameter>& Encoder::parameters() const { return impl_->params; }
inline cudaStream_t Encoder::stream() const { return impl_->stream; }
inline void Encoder::set_stream(cudaStream_t stream) {
    detail::blas_check(cublasSetStream(impl_->handle, stream)); impl_->stream = stream;
}
inline void Encoder::bind_parameters(const std::vector<float*>& data, const std::vector<float*>& gradients) {
    auto& m = *impl_;
    if (data.size() != m.params.size() || (!gradients.empty() && gradients.size() != m.params.size()) ||
            (m.trainable && gradients.size() != m.params.size()))
        throw std::invalid_argument("pretrained parameter binding count mismatch");
    for (size_t i = 0; i < data.size(); ++i)
        if (!data[i] || (m.trainable && !gradients[i])) throw std::invalid_argument("null pretrained parameter binding");
    detail::cuda_check(cudaStreamSynchronize(m.stream));
    for (auto& p : m.params) for (float* old : {p.data, p.grad}) {
        if (!old || std::find(data.begin(), data.end(), old) != data.end() ||
                std::find(gradients.begin(), gradients.end(), old) != gradients.end()) continue;
        auto owner = std::find(m.allocations.begin(), m.allocations.end(), old);
        if (owner != m.allocations.end()) { detail::cuda_check(cudaFree(*owner)); *owner = nullptr; }
    }
    for (size_t i = 0; i < data.size(); ++i) {
        m.params[i].data = data[i]; m.params[i].grad = gradients.empty() ? nullptr : gradients[i];
    }
    m.B = 0;
}
inline void Encoder::zero_grad() {
    if (!impl_->trainable) return;
    for (const auto& p : impl_->params) {
        if (!p.grad) throw std::invalid_argument("pretrained encoder gradients are not bound");
        detail::cuda_check(cudaMemsetAsync(p.grad, 0, p.count * sizeof(float), impl_->stream));
    }
}
inline float* Encoder::forward(const int* ids, const int* mask, const int* types, int B, int T) {
    auto& m = *impl_;
    if (m.head() || !ids) throw std::invalid_argument("token forward requires BERT/ModernBERT and input IDs");
    m.start(mask, B, T);
    size_t rows = static_cast<size_t>(B) * T;
    detail::cuda_check(cudaMemcpyAsync(m.ids, ids, rows * sizeof(int), cudaMemcpyDeviceToDevice, m.stream));
    if (m.bert()) {
        if (types) detail::cuda_check(cudaMemcpyAsync(m.types, types, rows * sizeof(int), cudaMemcpyDeviceToDevice, m.stream));
        else detail::cuda_check(cudaMemsetAsync(m.types, 0, rows * sizeof(int), m.stream));
    }
    detail::embedding_forward<<<detail::grid(rows * m.cfg.width), detail::threads, 0, m.stream>>>(
        m.ids, m.bert() ? m.types : nullptr, m.data(m.word), m.data(m.position), m.data(m.type_embedding),
        m.embedded, B, T, m.cfg.width, m.cfg.vocab, m.cfg.type_vocab);
    m.norm(m.embedded, m.embedding_norm, m.embedding_output, m.embedding_mean, m.embedding_rstd, B * T);
    return m.forward_layers();
}
inline float* Encoder::forward_hidden(const float* hidden, const int* mask, int B, int T) {
    auto& m = *impl_;
    if (!m.head() || !hidden) throw std::invalid_argument("hidden forward requires a Laya head and input features");
    m.start(mask, B, T); m.copy(m.embedded, hidden, static_cast<size_t>(B) * T * m.cfg.width);
    return m.forward_layers();
}
inline float* Encoder::backward(const float* gradient) { return impl_->backward(gradient); }
inline float* Encoder::layer_output(int layer) const {
    if (!impl_->trainable || impl_->B < 1 || layer < 0 || layer >= impl_->cfg.layers)
        throw std::invalid_argument("layer outputs require a completed trainable forward and valid layer");
    return impl_->work[layer].output;
}

} // namespace pretrained

#endif
