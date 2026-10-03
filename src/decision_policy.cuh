#ifndef PUFFER_DECISION_POLICY_CUH
#define PUFFER_DECISION_POLICY_CUH
#ifndef PRECISION_FLOAT
#error "native decision policies currently require FP32"
#endif
#include "decision_policy.h"
#include "decision_passthrough.cuh"

static_assert(NUM_ATNS == 1, "decision policies require one discrete action head");
static constexpr int decision_policy_action_sizes[] = ACT_SIZES;
static_assert(sizeof(decision_policy_action_sizes) / sizeof(int) == 1 &&
    decision_policy_action_sizes[0] == DECISION_ACTIONS,
    "ACT_SIZES must contain exactly DECISION_ACTIONS");

struct DecisionPolicyWeights {
    pretrained::DecisionConfig config;
    bool zero_init_critic;
    std::vector<pretrained::Parameter> registry;
    std::vector<Prec> parameters;
};
struct DecisionPolicyWorkspace {
    pretrained::DecisionModel* model;
    std::vector<Prec> gradients;
    bool bound = false;
};
struct DecisionPolicyActivations {
    DecisionPolicyWorkspace* workspace;
    Int ids, mask, markers, marker_mask, qtype;
    Prec output;
    Float grad_logits, grad_values;
};

static __device__ uint32_t decision_policy_byte(const float* input, int offset) {
    return static_cast<uint32_t>(input[offset]);
}
static __global__ void decision_policy_decode(const float* input, int* ids,
        int* mask, int* markers, int* marker_mask, int* qtype, int B, int T, int pad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * T) return;
    int b = i / T, t = i % T;
    const float* row = input + (size_t)b * OBS_SIZE;
    int length = decision_policy_byte(row, 0) | (decision_policy_byte(row, 1) << 8);
    int offset = DECISION_POLICY_HEADER_BYTES + 4 * t;
    uint32_t token = 0;
    for (int byte = 0; byte < 4; ++byte) token |= decision_policy_byte(row, offset + byte) << (8 * byte);
    ids[i] = t < length ? static_cast<int>(token) : pad;
    mask[i] = t < length;
    if (t == 0) {
        qtype[b] = decision_policy_byte(row, DECISION_POLICY_QTYPE_OFFSET);
        for (int k = 0; k < DECISION_ACTIONS; ++k) {
            markers[DECISION_ACTIONS * b + k] = decision_policy_byte(row, 2 + 2 * k) |
                (decision_policy_byte(row, 3 + 2 * k) << 8);
            marker_mask[DECISION_ACTIONS * b + k] = 1;
        }
    }
}
static __global__ void decision_policy_pack(float* out, const float* logits,
        const float* values, int B, float temperature) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = DECISION_ACTIONS + 1;
    if (i < B * stride) out[i] = i % stride == DECISION_ACTIONS ? values[i / stride] :
        logits[i / stride * DECISION_ACTIONS + i % stride] / temperature;
}
static __global__ void decision_policy_unpack(const float* input, float* logits,
        float* values, int B, float temperature) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = DECISION_ACTIONS + 1;
    if (i >= B * stride) return;
    if (i % stride == DECISION_ACTIONS) values[i / stride] = input[i];
    else logits[i / stride * DECISION_ACTIONS + i % stride] = input[i] / temperature;
}

static void* decision_policy_weights(void*) {
    auto* weights = new DecisionPolicyWeights;
    weights->config = decision_policy_context->config;
    weights->zero_init_critic = decision_policy_context->zero_init_critic;
    weights->registry = pretrained::decision_parameter_specs(weights->config);
    for (const auto& p : weights->registry) {
        Prec tensor{};
        for (size_t d = 0; d < p.shape.size(); ++d) tensor.shape[d] = p.shape[d];
        // Every flat slot is a multiple of the native allocator's 16-byte ABI.
        if (p.count % 4) {
            if (p.shape.size() != 1) throw std::logic_error("unaligned decision policy matrix");
            tensor.shape[0] = (p.count + 3) & ~size_t(3);
        }
        weights->parameters.push_back(tensor);
    }
    return weights;
}
static void decision_policy_register_parameters(void* opaque, Allocator* alloc) {
    auto* w = static_cast<DecisionPolicyWeights*>(opaque);
    for (auto& p : w->parameters) alloc_register(alloc, &p);
}
static void decision_policy_initialize(void* opaque, ulong* seed, cudaStream_t stream) {
    using decision::transformer_detail::cuda_check;
    auto* w = static_cast<DecisionPolicyWeights*>(opaque);
    auto registry = w->registry;
    std::mt19937_64 rng((*seed)++);
    std::normal_distribution<float> normal(0, 0.02f);
    for (size_t i = 0; i < registry.size(); ++i) {
        auto& p = registry[i]; p.data = w->parameters[i].data;
        cuda_check(cudaMemsetAsync(p.data, 0, numel(w->parameters[i].shape) * sizeof(float), stream));
        bool critic = p.name.compare(0, 11, "value_head.") == 0;
        bool fresh = critic ||
            (decision_policy_context->bundle.kind == "encoder" && p.name.compare(0, 8, "encoder.") != 0);
        if (!fresh) continue;
        std::vector<float> values(p.count);
        bool bias = p.name.size() >= 4 && p.name.compare(p.name.size() - 4, 4, "bias") == 0;
        bool norm = p.name.find("norm") != std::string::npos || p.name == "scorer.0.weight";
        for (float& value : values) value = bias ? 0 : norm ? 1 : normal(rng);
        // Consume exactly the historical RNG sequence before this opt-in
        // override, so every imported/fresh policy parameter remains identical.
        if (critic && w->zero_init_critic)
            for (float& value : values) value = 0;
        cuda_check(cudaMemcpyAsync(p.data, values.data(), p.count * sizeof(float), cudaMemcpyHostToDevice, stream));
        cuda_check(cudaStreamSynchronize(stream));
    }
    decision_policy_context->bundle.load_parameters(registry, stream);
}
static void decision_policy_register_common(void* opaque, void* activations,
        Allocator* acts, Allocator* grads, int B) {
    auto* w = static_cast<DecisionPolicyWeights*>(opaque);
    auto* a = static_cast<DecisionPolicyActivations*>(activations);
    int T = decision_policy_context->execution_tokens;
    a->workspace = new DecisionPolicyWorkspace;
    a->workspace->model = new pretrained::DecisionModel(w->config, B, T, DECISION_ACTIONS, nullptr, grads != nullptr, false);
    a->ids = {.shape = {B, T}}; alloc_register(acts, &a->ids);
    a->mask = {.shape = {B, T}}; alloc_register(acts, &a->mask);
    a->markers = {.shape = {B, DECISION_ACTIONS}}; alloc_register(acts, &a->markers);
    a->marker_mask = {.shape = {B, DECISION_ACTIONS}}; alloc_register(acts, &a->marker_mask);
    a->qtype = {.shape = {B}}; alloc_register(acts, &a->qtype);
    a->output = {.shape = {B, DECISION_ACTIONS + 1}}; alloc_register(acts, &a->output);
    if (grads) {
        a->workspace->gradients = w->parameters;
        for (auto& p : a->workspace->gradients) { p.data = nullptr; alloc_register(grads, &p); }
        a->grad_logits = {.shape = {B, DECISION_ACTIONS}}; alloc_register(acts, &a->grad_logits);
        a->grad_values = {.shape = {B}}; alloc_register(acts, &a->grad_values);
    }
}
static void decision_policy_register_rollout(void* w, void* a, Allocator* acts, int B) {
    decision_policy_register_common(w, a, acts, nullptr, B);
}
static void decision_policy_register_train(void* w, void* a, Allocator* acts, Allocator* grads, int B) {
    decision_policy_register_common(w, a, acts, grads, B);
}
static Prec decision_policy_forward(void* opaque, void* activations, Prec input, cudaStream_t stream) {
    auto* w = static_cast<DecisionPolicyWeights*>(opaque);
    auto* a = static_cast<DecisionPolicyActivations*>(activations);
    auto* model = a->workspace->model;
    if (!a->workspace->bound) {
        std::vector<float*> data, gradients;
        for (auto& p : w->parameters) data.push_back(p.data);
        for (auto& p : a->workspace->gradients) gradients.push_back(p.data);
        model->bind_parameters(data, gradients); a->workspace->bound = true;
    }
    model->set_stream(stream);
    int B = numel(input.shape) / OBS_SIZE, T = decision_policy_context->execution_tokens;
    decision_policy_decode<<<grid_size(B * T), BLOCK_SIZE, 0, stream>>>(input.data,
        a->ids.data, a->mask.data, a->markers.data, a->marker_mask.data, a->qtype.data,
        B, T, decision_policy_context->bundle.pad_id);
    auto output = model->forward(a->ids.data, a->mask.data, nullptr, a->markers.data,
        a->marker_mask.data, a->qtype.data, B, T, DECISION_ACTIONS);
    decision_policy_pack<<<grid_size(B * (DECISION_ACTIONS + 1)), BLOCK_SIZE, 0, stream>>>(a->output.data,
        output.logits, output.values, B, decision_policy_context->temperature);
    return Prec{.data = a->output.data, .shape = {B, DECISION_ACTIONS + 1}};
}
static void decision_policy_backward(void*, void* activations, Prec gradient, cudaStream_t stream) {
    auto* a = static_cast<DecisionPolicyActivations*>(activations);
    int B = numel(gradient.shape) / (DECISION_ACTIONS + 1);
    decision_policy_unpack<<<grid_size(B * (DECISION_ACTIONS + 1)), BLOCK_SIZE, 0, stream>>>(gradient.data,
        a->grad_logits.data, a->grad_values.data, B, decision_policy_context->temperature);
    a->workspace->model->set_stream(stream);
    a->workspace->model->zero_grad();
    // The imported act/escalate branch remains intact. Environment PPO supervises
    // option selection and the additional critic, not escalation decisions.
    a->workspace->model->backward(a->grad_logits.data, nullptr, a->grad_values.data);
}
static void create_decision_policy_encoder(Encoder* encoder) {
    encoder->create_weights = decision_policy_weights;
    encoder->reg_params = decision_policy_register_parameters;
    encoder->reg_train = decision_policy_register_train;
    encoder->reg_rollout = decision_policy_register_rollout;
    encoder->init_weights = decision_policy_initialize;
    encoder->forward = decision_policy_forward;
    encoder->backward = decision_policy_backward;
    encoder->activation_size = sizeof(DecisionPolicyActivations);
}

#endif
