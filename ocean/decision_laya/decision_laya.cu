#ifndef PRECISION_FLOAT
#error "decision_laya currently requires FP32"
#endif

// Reuse the stateless pass-through network/decoder and packed action/value ABI.
#include "../decision_snake/decision_snake.cu"

struct DecisionLayaWeights {
    pretrained::DecisionConfig config;
    std::vector<pretrained::Parameter> registry;
    std::vector<Prec> parameters;
};
struct DecisionLayaWorkspace {
    pretrained::DecisionModel* model;
    std::vector<Prec> gradients;
    bool bound = false;
};
struct DecisionLayaActivations {
    DecisionLayaWorkspace* workspace;
    Int ids, mask, markers, marker_mask, qtype;
    Prec output;
    Float grad_logits, grad_values;
};

static __device__ uint32_t decision_laya_byte(const float* input, int offset) {
    return static_cast<uint32_t>(input[offset]);
}
static __global__ void decision_laya_decode(const float* input, int* ids,
        int* mask, int* markers, int* marker_mask, int* qtype, int B, int T, int pad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * T) return;
    int b = i / T, t = i % T;
    const float* row = input + (size_t)b * OBS_SIZE;
    int length = decision_laya_byte(row, 0) | (decision_laya_byte(row, 1) << 8);
    int offset = DECISION_LAYA_HEADER_BYTES + 4 * t;
    uint32_t token = 0;
    for (int byte = 0; byte < 4; ++byte) token |= decision_laya_byte(row, offset + byte) << (8 * byte);
    ids[i] = t < length ? static_cast<int>(token) : pad;
    mask[i] = t < length;
    if (t == 0) {
        qtype[b] = decision_laya_byte(row, 10);
        for (int k = 0; k < 4; ++k) {
            markers[4 * b + k] = decision_laya_byte(row, 2 + 2 * k) |
                (decision_laya_byte(row, 3 + 2 * k) << 8);
            marker_mask[4 * b + k] = 1;
        }
    }
}
static __global__ void decision_laya_pack(float* out, const float* logits,
        const float* values, int B, float temperature) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * 5) out[i] = i % 5 == 4 ? values[i / 5] : logits[i / 5 * 4 + i % 5] / temperature;
}
static __global__ void decision_laya_unpack(const float* input, float* logits,
        float* values, int B, float temperature) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * 5) return;
    if (i % 5 == 4) values[i / 5] = input[i];
    else logits[i / 5 * 4 + i % 5] = input[i] / temperature;
}

static void* decision_laya_weights(void*) {
    auto* weights = new DecisionLayaWeights;
    weights->config = decision_laya_context->config;
    weights->registry = pretrained::decision_parameter_specs(weights->config);
    for (const auto& p : weights->registry) {
        Prec tensor{};
        for (size_t d = 0; d < p.shape.size(); ++d) tensor.shape[d] = p.shape[d];
        // Every flat slot is a multiple of the native allocator's 16-byte ABI.
        if (p.count % 4) {
            if (p.shape.size() != 1) throw std::logic_error("unaligned Laya matrix");
            tensor.shape[0] = (p.count + 3) & ~size_t(3);
        }
        weights->parameters.push_back(tensor);
    }
    return weights;
}
static void decision_laya_register_parameters(void* opaque, Allocator* alloc) {
    auto* w = static_cast<DecisionLayaWeights*>(opaque);
    for (auto& p : w->parameters) alloc_register(alloc, &p);
}
static void decision_laya_initialize(void* opaque, ulong* seed, cudaStream_t stream) {
    using decision::transformer_detail::cuda_check;
    auto* w = static_cast<DecisionLayaWeights*>(opaque);
    auto registry = w->registry;
    std::mt19937_64 rng((*seed)++);
    std::normal_distribution<float> normal(0, 0.02f);
    for (size_t i = 0; i < registry.size(); ++i) {
        auto& p = registry[i]; p.data = w->parameters[i].data;
        cuda_check(cudaMemsetAsync(p.data, 0, numel(w->parameters[i].shape) * sizeof(float), stream));
        bool fresh = p.name.compare(0, 11, "value_head.") == 0 ||
            (decision_laya_context->bundle.kind == "encoder" && p.name.compare(0, 8, "encoder.") != 0);
        if (!fresh) continue;
        std::vector<float> values(p.count);
        bool bias = p.name.size() >= 4 && p.name.compare(p.name.size() - 4, 4, "bias") == 0;
        bool norm = p.name.find("norm") != std::string::npos || p.name == "scorer.0.weight";
        for (float& value : values) value = bias ? 0 : norm ? 1 : normal(rng);
        cuda_check(cudaMemcpyAsync(p.data, values.data(), p.count * sizeof(float), cudaMemcpyHostToDevice, stream));
        cuda_check(cudaStreamSynchronize(stream));
    }
    decision_laya_context->bundle.load_parameters(registry, stream);
}
static void decision_laya_register_common(void* opaque, void* activations,
        Allocator* acts, Allocator* grads, int B) {
    auto* w = static_cast<DecisionLayaWeights*>(opaque);
    auto* a = static_cast<DecisionLayaActivations*>(activations);
    int T = decision_laya_context->bundle.max_len;
    a->workspace = new DecisionLayaWorkspace;
    a->workspace->model = new pretrained::DecisionModel(w->config, B, T, 4, nullptr, grads != nullptr, false);
    a->ids = {.shape = {B, T}}; alloc_register(acts, &a->ids);
    a->mask = {.shape = {B, T}}; alloc_register(acts, &a->mask);
    a->markers = {.shape = {B, 4}}; alloc_register(acts, &a->markers);
    a->marker_mask = {.shape = {B, 4}}; alloc_register(acts, &a->marker_mask);
    a->qtype = {.shape = {B}}; alloc_register(acts, &a->qtype);
    a->output = {.shape = {B, 5}}; alloc_register(acts, &a->output);
    if (grads) {
        a->workspace->gradients = w->parameters;
        for (auto& p : a->workspace->gradients) { p.data = nullptr; alloc_register(grads, &p); }
        a->grad_logits = {.shape = {B, 4}}; alloc_register(acts, &a->grad_logits);
        a->grad_values = {.shape = {B}}; alloc_register(acts, &a->grad_values);
    }
}
static void decision_laya_register_rollout(void* w, void* a, Allocator* acts, int B) {
    decision_laya_register_common(w, a, acts, nullptr, B);
}
static void decision_laya_register_train(void* w, void* a, Allocator* acts, Allocator* grads, int B) {
    decision_laya_register_common(w, a, acts, grads, B);
}
static Prec decision_laya_forward(void* opaque, void* activations, Prec input, cudaStream_t stream) {
    auto* w = static_cast<DecisionLayaWeights*>(opaque);
    auto* a = static_cast<DecisionLayaActivations*>(activations);
    auto* model = a->workspace->model;
    if (!a->workspace->bound) {
        std::vector<float*> data, gradients;
        for (auto& p : w->parameters) data.push_back(p.data);
        for (auto& p : a->workspace->gradients) gradients.push_back(p.data);
        model->bind_parameters(data, gradients); a->workspace->bound = true;
    }
    model->set_stream(stream);
    int B = numel(input.shape) / OBS_SIZE, T = decision_laya_context->bundle.max_len;
    decision_laya_decode<<<grid_size(B * T), BLOCK_SIZE, 0, stream>>>(input.data,
        a->ids.data, a->mask.data, a->markers.data, a->marker_mask.data, a->qtype.data,
        B, T, decision_laya_context->bundle.pad_id);
    auto output = model->forward(a->ids.data, a->mask.data, nullptr, a->markers.data,
        a->marker_mask.data, a->qtype.data, B, T, 4);
    decision_laya_pack<<<grid_size(B * 5), BLOCK_SIZE, 0, stream>>>(a->output.data,
        output.logits, output.values, B, decision_laya_context->temperature);
    return Prec{.data = a->output.data, .shape = {B, 5}};
}
static void decision_laya_backward(void*, void* activations, Prec gradient, cudaStream_t stream) {
    auto* a = static_cast<DecisionLayaActivations*>(activations);
    int B = numel(gradient.shape) / 5;
    decision_laya_unpack<<<grid_size(B * 5), BLOCK_SIZE, 0, stream>>>(gradient.data,
        a->grad_logits.data, a->grad_values.data, B, decision_laya_context->temperature);
    a->workspace->model->set_stream(stream);
    a->workspace->model->zero_grad();
    // The imported act/escalate branch remains intact. Snake PPO supervises
    // option selection and the additional critic, not escalation decisions.
    a->workspace->model->backward(a->grad_logits.data, nullptr, a->grad_values.data);
}
static void create_decision_laya_encoder(Encoder* encoder) {
    encoder->create_weights = decision_laya_weights;
    encoder->reg_params = decision_laya_register_parameters;
    encoder->reg_train = decision_laya_register_train;
    encoder->reg_rollout = decision_laya_register_rollout;
    encoder->init_weights = decision_laya_initialize;
    encoder->forward = decision_laya_forward;
    encoder->backward = decision_laya_backward;
    encoder->activation_size = sizeof(DecisionLayaActivations);
}
