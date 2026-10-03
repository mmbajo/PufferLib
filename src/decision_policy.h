#ifndef PUFFER_DECISION_POLICY_H
#define PUFFER_DECISION_POLICY_H

// Environment contract: define the size of one discrete action head before
// including this header, then serialize each state with decision_policy_encode.
// Options keep the same index as the environment's action and action-mask slots.
#ifndef DECISION_ACTIONS
#error "define DECISION_ACTIONS before including decision_policy.h"
#endif
// Puffer reserves an action size of one for continuous control.
static_assert(DECISION_ACTIONS >= 2 && DECISION_ACTIONS <= 255,
    "native decision policies support one discrete head with 2..255 actions");
#define PUFFER_DECISION_POLICY
typedef unsigned char obs_t;
#define DECISION_POLICY_MAX_TOKENS 2048
// Preserve the original four-action observation ABI: 16 header bytes followed
// by little-endian uint32 token IDs. Larger action sets extend only the header.
#define DECISION_POLICY_HEADER_BYTES ((4 + 2 * DECISION_ACTIONS) > 16 ? (4 + 2 * DECISION_ACTIONS) : 16)
#define DECISION_POLICY_QTYPE_OFFSET (2 + 2 * DECISION_ACTIONS)
#define OBS_SIZE (DECISION_POLICY_HEADER_BYTES + 4 * DECISION_POLICY_MAX_TOKENS)

#include "ini.h"
#include "pretrained_bundle.cuh"
#include "pretrained_input.h"
#include "pretrained_decision.cuh"
#include <memory>

struct DecisionPolicyContext {
    std::string path;
    pretrained::Bundle bundle;
    pretrained::Tokenizer tokenizer;
    pretrained::DecisionConfig config;
    float temperature;
    bool zero_init_critic;
    int execution_tokens;
    explicit DecisionPolicyContext(const std::string& source, bool zero_critic = false,
            int sequence_length = 0)
        : path(source), bundle(source), tokenizer(bundle.tokenizer_path),
          zero_init_critic(zero_critic),
          execution_tokens(sequence_length ? sequence_length : bundle.max_len) {
        if (bundle.max_len > DECISION_POLICY_MAX_TOKENS || bundle.encoder.width % 4)
            throw std::invalid_argument("decision policy requires max_len <= 2048 and width divisible by four");
        if (execution_tokens < 1 || execution_tokens > bundle.max_len)
            throw std::invalid_argument("policy.sequence_length must be 0 or in [1, bundle.max_len]");
        config.encoder = bundle.encoder;
        config.head_layers = bundle.head_layers;
        config.n_act = bundle.n_act;
        config.value_head = true;
        temperature = pretrained::decision_temperature(bundle, 0, DECISION_ACTIONS);
        bundle.validate_parameters(pretrained::decision_parameter_specs(config));
    }
};
static std::unique_ptr<DecisionPolicyContext> decision_policy_context;

// Unlike the general INI reader, these controls require numeric scalar values;
// malformed strings/lists must not silently become zero/default experiments.
static int decision_policy_integer_option(Ini* ini, const char* key, int maximum) {
    DictItem* item = dict_find(puf_ini_section(ini, "policy", 0), key);
    if (!item) {
        puf_ini_set(puf_ini_section(ini, "policy", 0), key, "0");
        return 0;
    }
    double value = item->value;
    bool valid = item->len == 0;
    if (item->str) {
        char* end = nullptr;
        errno = 0;
        value = strtod(item->str, &end);
        valid = valid && end != item->str && errno != ERANGE;
        while (end && isspace((unsigned char)*end)) ++end;
        valid = valid && end && !*end;
    }
    if (!valid || !std::isfinite(value) || value < 0 || value > maximum || value != std::floor(value))
        throw std::invalid_argument(std::string("policy.") + key +
            " must be a numeric integer in [0, " + std::to_string(maximum) + "]");
    return (int)value;
}

// Called before Puffer constructs its policy shapes or starts environment
// workers. The immutable tokenizer/context is then shared by those workers.
static void decision_policy_configure(Ini* ini) {
    const bool zero_critic = decision_policy_integer_option(ini, "zero_init_critic", 1) != 0;
    const int sequence_length = decision_policy_integer_option(ini, "sequence_length", DECISION_POLICY_MAX_TOKENS);
    const char* path = puf_ini_get_str(ini, "policy", "bundle");
    if (!path || !*path || strcmp(path, "None") == 0)
        throw std::invalid_argument("decision policy requires --policy.bundle=/path/to/imported-bundle");
    if (!decision_policy_context)
        decision_policy_context.reset(new DecisionPolicyContext(path, zero_critic, sequence_length));
    if (decision_policy_context->path != path)
        throw std::invalid_argument("one process cannot switch decision policy bundles");
    const int tokens = sequence_length ? sequence_length : decision_policy_context->bundle.max_len;
    if (decision_policy_context->zero_init_critic != zero_critic ||
            decision_policy_context->execution_tokens != tokens)
        throw std::invalid_argument("one process cannot switch decision policy initialization or sequence length");
    const auto& cfg = decision_policy_context->config.encoder;
    puf_ini_put(ini, "policy.hidden_size", std::to_string(cfg.width).c_str());
    puf_ini_put(ini, "policy.num_layers", std::to_string(cfg.layers).c_str());
}

static void decision_policy_encode(const std::string& state,
        const std::string& instructions, const std::vector<std::string>& actions,
        unsigned char* output) {
    if (!decision_policy_context) throw std::logic_error("decision policy context is not initialized");
    if (!output) throw std::invalid_argument("decision policy observation storage is null");
    if (actions.size() != DECISION_ACTIONS)
        throw std::invalid_argument("decision policy option count must equal DECISION_ACTIONS");
    const auto& ctx = *decision_policy_context;
    pretrained::SequenceOptions options;
    options.reject_truncated_state = true;
    const auto sequence = pretrained::build_sequence(ctx.bundle, ctx.tokenizer,
        state, "choice", instructions, actions, options);
    if (sequence.ids.size() > (size_t)ctx.execution_tokens)
        throw std::runtime_error("decision policy input exceeds policy.sequence_length; increase it without truncating the state");
    if (sequence.ids.size() > DECISION_POLICY_MAX_TOKENS ||
            sequence.marker_positions.size() != DECISION_ACTIONS)
        throw std::runtime_error("decision policy sequence does not fit observation storage");
    memset(output, 0, OBS_SIZE);
    auto u16 = [&](int offset, uint32_t value) {
        output[offset] = value & 255; output[offset + 1] = (value >> 8) & 255;
    };
    u16(0, sequence.ids.size());
    for (int k = 0; k < DECISION_ACTIONS; ++k)
        u16(2 + 2 * k, sequence.marker_positions[k]);
    output[DECISION_POLICY_QTYPE_OFFSET] = sequence.qtype;
    for (size_t t = 0; t < sequence.ids.size(); ++t) {
        uint32_t id = sequence.ids[t];
        for (int byte = 0; byte < 4; ++byte)
            output[DECISION_POLICY_HEADER_BYTES + 4 * t + byte] = (id >> (8 * byte)) & 255;
    }
}

#endif
