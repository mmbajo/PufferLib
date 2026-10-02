// CPU-only environment/packing validation. Compile with NVCC for the included
// model declarations; execution does not create a CUDA context or use a GPU.
#include "../src/ini.h"
#define PUF_HEADLESS
#include "../ocean/decision_laya/decision_laya.h"
#include <cassert>
#include <iostream>

static int validate(const unsigned char* observations) {
    const auto& bundle = decision_policy_context->bundle;
    int length = observations[0] | (observations[1] << 8);
    assert(length > 0 && length <= bundle.max_len);
    assert(observations[DECISION_POLICY_QTYPE_OFFSET] == 0);
    for (int k = 0; k < 4; ++k) {
        int marker = observations[2 + 2 * k] | (observations[3 + 2 * k] << 8);
        assert(marker < length);
        uint32_t token = 0;
        for (int byte = 0; byte < 4; ++byte)
            token |= uint32_t(observations[DECISION_POLICY_HEADER_BYTES + 4 * marker + byte]) << (8 * byte);
        assert(token == bundle.mask_id);
    }
    return length;
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: test_decision_laya_observation BUNDLE\n"; return 2; }
    decision_policy_context.reset(new DecisionPolicyContext(argv[1]));
    unsigned char observations[OBS_SIZE], expected[OBS_SIZE], masks[4];
    int32_t board[100]{};
    for (int fixture = 0; fixture < 3; ++fixture) {
        for (int i = 0; i < 100; ++i) board[i] = fixture == 0 ? 0 : fixture == 1 ? i + 1 : -(i % 2);
        if (fixture == 0) { board[45] = 1; board[46] = 2; board[47] = 3; board[72] = -1; }
        decision_laya_encode(board, 499, 500, observations);
        std::cout << "fixture=" << fixture << " tokens=" << validate(observations) << '\n';
    }

    Dict kwargs{};
    dict_set(&kwargs, "max_steps", 1);
    dict_set(&kwargs, "num_agents", 1);
    Env env{}; env.rng = 73;
    puf_init(&env, &kwargs);
    float action = 0, reward = 0, terminal = 0;
    env.agents[0].observations = observations;
    env.agents[0].actions = &action;
    env.agents[0].rewards = &reward;
    env.agents[0].terminals = &terminal;
    env.agents[0].action_mask = masks;
    puf_reset(&env); validate(observations);
    ib_snake_observe(env.snake, board, 100);
    int head = 0;
    while (board[head] != 1) ++head;
    const int dr[4] = {-1, 1, 0, 0}, dc[4] = {0, 0, -1, 1};
    int selected = -1;
    for (int k = 0; k < 4; ++k) {
        int r = head / 10 + dr[k], c = head % 10 + dc[k];
        if (masks[k] && r >= 0 && r < 10 && c >= 0 && c < 10 && board[r * 10 + c] <= 0) { selected = k; break; }
    }
    assert(selected >= 0);
    IBSnake* reference = ib_snake_create(73, 1);
    assert(reference && ib_snake_step(reference, selected) == 0);
    ib_snake_observe(reference, board, 100);
    decision_laya_encode(board, 1, 1, expected);
    action = selected;
    puf_step(&env);
    assert(terminal == 1 && env.transition.truncated && !env.transition.terminated);
    assert(memcmp(expected, env.transition.observations, OBS_SIZE) == 0);
    assert(ib_snake_reset(reference, 74, 1) == 0);
    ib_snake_observe(reference, board, 100);
    decision_laya_encode(board, 0, 1, expected);
    assert(memcmp(expected, observations, OBS_SIZE) == 0);
    assert(memcmp(observations, env.transition.observations, OBS_SIZE) != 0);
    validate(env.transition.observations); validate(observations);
    ib_snake_destroy(reference); puf_close(&env);
    std::cout << "timeout final tokens and autoreset tokens preserved independently\n";
}
