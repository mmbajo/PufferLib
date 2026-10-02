#ifndef PUFFER_DECISION_LAYA_H
#define PUFFER_DECISION_LAYA_H

// Token IDs travel through Puffer's byte observation storage without losing
// integer precision: 16 header bytes followed by little-endian uint32 IDs.
typedef unsigned char obs_t;
#define DECISION_LAYA_MAX_TOKENS 2048
#define DECISION_LAYA_HEADER_BYTES 16
#define OBS_SIZE (DECISION_LAYA_HEADER_BYTES + 4 * DECISION_LAYA_MAX_TOKENS)
#ifndef PUFFER_DECISION_SNAKE
#define PUFFER_DECISION_SNAKE
#endif

#include "../../src/pretrained_bundle.cuh"
#include "../../src/pretrained_input.h"
#include "../../src/pretrained_decision.cuh"
#include <sstream>

struct DecisionLayaContext {
    std::string path;
    pretrained::Bundle bundle;
    pretrained::Tokenizer tokenizer;
    pretrained::DecisionConfig config;
    float temperature;
    explicit DecisionLayaContext(const std::string& source)
        : path(source), bundle(source), tokenizer(bundle.tokenizer_path) {
        if (bundle.max_len > DECISION_LAYA_MAX_TOKENS || bundle.encoder.width % 4)
            throw std::invalid_argument("decision_laya requires max_len <= 2048 and width divisible by four");
        config.encoder = bundle.encoder;
        config.head_layers = bundle.head_layers;
        config.n_act = bundle.n_act;
        config.value_head = true;
        temperature = pretrained::decision_temperature(bundle, 0, 4);
        bundle.validate_parameters(pretrained::decision_parameter_specs(config));
    }
};
static std::unique_ptr<DecisionLayaContext> decision_laya_context;

static void decision_laya_configure(Ini* ini) {
    const char* path = puf_ini_get_str(ini, "policy", "bundle");
    if (!path || !*path || strcmp(path, "None") == 0)
        throw std::invalid_argument("decision_laya requires --policy.bundle=/path/to/imported-bundle");
    if (!decision_laya_context) decision_laya_context.reset(new DecisionLayaContext(path));
    if (decision_laya_context->path != path)
        throw std::invalid_argument("one process cannot switch decision_laya bundles");
    const auto& cfg = decision_laya_context->config.encoder;
    puf_ini_put(ini, "policy.hidden_size", std::to_string(cfg.width).c_str());
    puf_ini_put(ini, "policy.num_layers", std::to_string(cfg.layers).c_str());
}

static void decision_laya_encode(const int32_t* board, int steps, int max_steps,
        unsigned char* output) {
    if (!decision_laya_context) throw std::logic_error("decision_laya context is not initialized");
    const auto& ctx = *decision_laya_context;
    std::ostringstream state;
    state << "Snake on a 10 by 10 grid. Rows run top to bottom; columns left to right. "
          << "0=empty, -1=food, 1=head, 2=neck, larger numbers follow the body toward the tail. "
          << "Step " << steps << " of " << max_steps << ". Board:\n";
    for (int row = 0; row < 10; ++row) {
        for (int col = 0; col < 10; ++col) {
            if (col) state << ',';
            state << board[row * 10 + col];
        }
        state << '\n';
    }
    pretrained::SequenceOptions options;
    options.reject_truncated_state = true;
    const auto sequence = pretrained::build_sequence(ctx.bundle, ctx.tokenizer,
        state.str(), "choice", "Choose the next move. Eat food and avoid the walls and snake body.",
        {"Move up", "Move down", "Move left", "Move right"}, options);
    if (sequence.ids.size() > DECISION_LAYA_MAX_TOKENS || sequence.marker_positions.size() != 4)
        throw std::runtime_error("decision_laya sequence does not fit observation storage");
    memset(output, 0, OBS_SIZE);
    auto u16 = [&](int offset, uint32_t value) {
        output[offset] = value & 255; output[offset + 1] = (value >> 8) & 255;
    };
    u16(0, sequence.ids.size());
    for (int k = 0; k < 4; ++k) u16(2 + 2 * k, sequence.marker_positions[k]);
    output[10] = sequence.qtype;
    for (size_t t = 0; t < sequence.ids.size(); ++t) {
        uint32_t id = sequence.ids[t];
        for (int byte = 0; byte < 4; ++byte)
            output[DECISION_LAYA_HEADER_BYTES + 4 * t + byte] = (id >> (8 * byte)) & 255;
    }
}

#define DECISION_SNAKE_ENCODE_OBSERVATION(env, board, observations) \
    decision_laya_encode(board, ib_snake_steps((env)->snake), (env)->max_steps, observations)
#include "../decision_snake/decision_snake.h"
#endif
