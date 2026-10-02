// CPU execution; compile with NVCC because the shared policy declares CUDA code.
#define PUF_HEADLESS
#include "../ocean/decision_lightsout/decision_lightsout.h"
#include <cassert>
#include <iostream>
#include <limits>
#include <set>

struct Fixture {
    Env env{};
    obs_t observations[OBS_SIZE]{};
    unsigned char mask[DECISION_ACTIONS]{};
    float action = 0, reward = 0, terminal = 0;
    explicit Fixture(uint32_t seed, int limit) {
        Dict kwargs{};
        dict_set(&kwargs, "max_steps", limit);
        env.rng = seed;
        puf_init(&env, &kwargs);
        dict_clear(&kwargs);
        env.agents[0] = {observations, &action, &reward, &terminal, mask, 0};
        puf_reset(&env);
    }
    ~Fixture() { puf_close(&env); }
};

static int u16(const obs_t* data, int offset) {
    return data[offset] | (data[offset + 1] << 8);
}

static int longest_sequence = 0;
static void check_tokens(const obs_t* data) {
    int length = u16(data, 0);
    assert(length > 0 && length <= decision_policy_context->bundle.max_len);
    longest_sequence = std::max(longest_sequence, length);
    assert(data[DECISION_POLICY_QTYPE_OFFSET] == 0);
    std::vector<uint32_t> tokens(length);
    for (int t = 0; t < length; ++t)
        for (int byte = 0; byte < 4; ++byte)
            tokens[t] |= uint32_t(data[DECISION_POLICY_HEADER_BYTES + 4 * t + byte]) << (8 * byte);
    std::set<std::vector<uint32_t>> spans;
    for (int action = 0; action < DECISION_ACTIONS; ++action) {
        int begin = u16(data, 2 + 2 * action);
        int end = action + 1 < DECISION_ACTIONS ? u16(data, 4 + 2 * action) : begin + 1;
        if (action + 1 == DECISION_ACTIONS)
            while (end < length && tokens[end] != decision_policy_context->bundle.sep_id) ++end;
        assert(begin > 0 && begin < end && end <= length);
        assert(tokens[begin] == decision_policy_context->bundle.mask_id);
        std::vector<uint32_t> span(tokens.begin() + begin + 1, tokens.begin() + end);
        spans.insert(span);
        auto expected = decision_policy_context->tokenizer.Encode(" " + std::string(1, 'A' + action), false);
        assert(span == expected); // No letter was dropped or remapped by head capping.
    }
    assert(spans.size() == DECISION_ACTIONS);
}

static void compare_engine(const Fixture& f, const IBLightsOut* oracle) {
    int32_t actual[25], expected[25];
    assert(ib_lightsout_observe(f.env.lightsout, actual, 25) == 0);
    assert(ib_lightsout_observe(oracle, expected, 25) == 0);
    assert(memcmp(actual, expected, sizeof(actual)) == 0);
    assert(ib_lightsout_steps(f.env.lightsout) == ib_lightsout_steps(oracle));
    assert(ib_lightsout_terminated(f.env.lightsout) == ib_lightsout_terminated(oracle));
    assert(ib_lightsout_truncated(f.env.lightsout) == ib_lightsout_truncated(oracle));
    assert(ib_lightsout_reward(f.env.lightsout) == ib_lightsout_reward(oracle));
    assert(ib_lightsout_episode_return(f.env.lightsout) == ib_lightsout_episode_return(oracle));
    uint32_t mask = ib_lightsout_legal_actions(oracle);
    for (int action = 0; action < 25; ++action) assert(f.mask[action] == ((mask >> action) & 1));
    check_tokens(f.observations);
}

// Independent GF(2) elimination solves the visible board, without engine state.
static uint32_t solve(const IBLightsOut* engine) {
    int32_t board[25];
    assert(ib_lightsout_observe(engine, board, 25) == 0);
    uint32_t rows[25];
    int pivots[25], rank = 0;
    for (int cell = 0; cell < 25; ++cell) {
        rows[cell] = uint32_t(board[cell]) << 25;
        for (int action = 0; action < 25; ++action)
            if (abs(cell / 5 - action / 5) + abs(cell % 5 - action % 5) <= 1)
                rows[cell] |= UINT32_C(1) << action;
    }
    for (int col = 0; col < 25; ++col) {
        int pivot = rank;
        while (pivot < 25 && !(rows[pivot] & (UINT32_C(1) << col))) ++pivot;
        if (pivot == 25) continue;
        std::swap(rows[rank], rows[pivot]);
        for (int row = 0; row < 25; ++row)
            if (row != rank && (rows[row] & (UINT32_C(1) << col))) rows[row] ^= rows[rank];
        pivots[rank++] = col;
    }
    for (int row = rank; row < 25; ++row) assert(rows[row] == 0);
    uint32_t actions = 0;
    for (int row = 0; row < rank; ++row)
        if (rows[row] & (UINT32_C(1) << 25)) actions |= UINT32_C(1) << pivots[row];
    assert(actions);
    return actions;
}

static int seeded_traces() {
    int steps = 0;
    for (uint32_t seed = 0; seed < 128; ++seed) {
        Fixture f(seed, 25);
        IBLightsOut* oracle = ib_lightsout_create(seed, 25);
        assert(oracle && f.env.episode_seed == seed);
        compare_engine(f, oracle);
        uint32_t solution = solve(oracle);
        int previous = -1, last = -1;
        for (int action = 0; action < 25 && !ib_lightsout_terminated(oracle); ++action) {
            if (!(solution & (UINT32_C(1) << action))) continue;
            assert(ib_lightsout_step(oracle, action) == 0);
            assert(decision_lightsout_step(&f.env, action) == 0);
            previous = last; last = action;
            assert(f.env.prev_action == previous && f.env.last_action == last);
            assert(f.reward == ib_lightsout_reward(oracle));
            assert(f.terminal == (ib_lightsout_terminated(oracle) || ib_lightsout_truncated(oracle)));
            assert(f.env.transition.seed == seed && f.env.transition.action == action);
            compare_engine(f, oracle);
            ++steps;
        }
        assert(f.env.transition.terminated && !f.env.transition.truncated);
        assert(f.reward == 2 && f.terminal == 1 && f.env.log.score == 1 && f.env.log.n == 1);
        assert(!puf_truncation_observation(&f.env, &f.env.agents[0]));
        auto transition = f.env.transition;
        assert(decision_lightsout_step(&f.env, 0) == -1);
        assert(memcmp(&transition, &f.env.transition, sizeof(transition)) == 0);
        ib_lightsout_destroy(oracle);
    }
    return steps;
}

static void history_and_boundaries() {
    Fixture f(5, 100);
    IBLightsOut* oracle = ib_lightsout_create(5, 100);
    for (int action : {0, 0, 24, 0}) {
        assert(ib_lightsout_step(oracle, action) == 0);
        assert(decision_lightsout_step(&f.env, action) == 0);
        compare_engine(f, oracle);
    }
    assert(f.env.last_action == 0 && f.env.prev_action == 24);
    auto state = decision_lightsout_state(&f.env);
    assert(state.find("Last action=A; preceding action=Y") != std::string::npos);
    assert(state.find("Step 4 of 100") != std::string::npos);
    auto saved = f.env.transition;
    for (float bad : {-1.f, 25.f, .5f, std::numeric_limits<float>::quiet_NaN()}) {
        f.action = bad;
        bool rejected = false;
        try { puf_step(&f.env); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected && memcmp(&saved, &f.env.transition, sizeof(saved)) == 0);
        compare_engine(f, oracle);
    }
    ib_lightsout_destroy(oracle);

    Fixture timeout(19, 1);
    oracle = ib_lightsout_create(19, 1);
    int action = 0;
    while (solve(oracle) == (UINT32_C(1) << action)) ++action;
    assert(ib_lightsout_step(oracle, action) == 0 && ib_lightsout_truncated(oracle));
    timeout.action = action;
    puf_step(&timeout.env);
    assert(timeout.terminal == 1 && timeout.reward == ib_lightsout_reward(oracle));
    assert(timeout.env.transition.truncated && !timeout.env.transition.terminated);
    assert(timeout.env.transition.seed == 19 && timeout.env.episode_seed == 20);
    assert(timeout.env.last_action == -1 && timeout.env.prev_action == -1);
    assert(ib_lightsout_steps(timeout.env.lightsout) == 0);
    for (int a = 0; a < 25; ++a) {
        assert(timeout.mask[a] == 1 && timeout.env.transition.action_mask[a] == 0);
    }
    assert(puf_truncation_observation(&timeout.env, &timeout.env.agents[0]) ==
        timeout.env.transition.observations);
    assert(memcmp(timeout.observations, timeout.env.transition.observations, OBS_SIZE) != 0);
    check_tokens(timeout.env.transition.observations);
    Agent other{};
    assert(!puf_truncation_observation(&timeout.env, &other));
    puf_reset(&timeout.env);
    assert(!timeout.env.transition.valid && timeout.reward == 0 && timeout.terminal == 0);
    assert(!puf_truncation_observation(&timeout.env, &timeout.env.agents[0]));
    assert(decision_lightsout_reset(&timeout.env, UINT32_MAX) == 0);
    assert(timeout.env.next_seed == 0);
    puf_reset(&timeout.env);
    assert(timeout.env.episode_seed == 0);
    ib_lightsout_destroy(oracle);

    // A solve on the configured final step takes precedence over timeout.
    Fixture win(44, 25);
    uint32_t solution = solve(win.env.lightsout);
    int presses = __builtin_popcount(solution);
    win.env.max_steps = presses;
    assert(decision_lightsout_reset(&win.env, 44) == 0);
    for (int a = 0; a < 25 && !win.terminal; ++a) {
        if (!(solution & (UINT32_C(1) << a))) continue;
        win.action = a;
        puf_step(&win.env);
    }
    assert(win.env.transition.terminated && !win.env.transition.truncated);
    assert(win.env.transition.steps == presses && win.reward == 2);
    assert(!puf_truncation_observation(&win.env, &win.env.agents[0]));
}

static void config_rejections() {
    for (double cap : {0., -1., 1.5, double(INT_MAX) + 1,
            std::numeric_limits<double>::quiet_NaN()}) {
        Dict kwargs{}; dict_set(&kwargs, "max_steps", cap);
        Env env{};
        bool rejected = false;
        try { puf_init(&env, &kwargs); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected); dict_clear(&kwargs);
    }
    Dict kwargs{}; dict_set(&kwargs, "max_steps", 100); dict_set(&kwargs, "num_agents", 2);
    Env env{};
    bool rejected = false;
    try { puf_init(&env, &kwargs); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected); dict_clear(&kwargs);
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: test_decision_lightsout_env BUNDLE\n"; return 2; }
    decision_policy_context.reset(new DecisionPolicyContext(argv[1]));
    config_rejections();
    history_and_boundaries();
    int steps = seeded_traces();
    std::cout << "PASS Lights Out: " << steps << " exact engine transitions over 128 solved seeds; "
              << "25 distinct packed options, repeat/ABA history, masks, autoreset, timeout and solve precedence; "
              << "maximum observed sequence=" << longest_sequence << " tokens\n";
}
