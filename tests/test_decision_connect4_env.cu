#define PUF_HEADLESS
#include "../ocean/decision_connect4/decision_connect4.h"
#include <cassert>
#include <iostream>
#include <limits>
#include <vector>

struct Fixture {
    Env env{};
    obs_t observations[OBS_SIZE]{};
    unsigned char masks[DECISION_ACTIONS]{};
    float action = 0, reward = 0, terminal = 0;
    Fixture(uint32_t seed, int max_steps) {
        Dict kwargs{};
        dict_set(&kwargs, "max_steps", max_steps);
        env.rng = seed;
        puf_init(&env, &kwargs);
        dict_clear(&kwargs);
        env.agents[0].observations = observations;
        env.agents[0].actions = &action;
        env.agents[0].rewards = &reward;
        env.agents[0].terminals = &terminal;
        env.agents[0].action_mask = masks;
        puf_reset(&env);
    }
    ~Fixture() { puf_close(&env); }
};

static std::vector<uint32_t> unpack(const obs_t* observation) {
    int count = observation[0] | (observation[1] << 8);
    assert(count > 0 && count <= decision_policy_context->bundle.max_len);
    std::vector<uint32_t> ids(count);
    for (int t = 0; t < count; ++t)
        for (int byte = 0; byte < 4; ++byte)
            ids[t] |= uint32_t(observation[DECISION_POLICY_HEADER_BYTES + 4 * t + byte]) << (8 * byte);
    int last = 0;
    for (int action = 0; action < DECISION_ACTIONS; ++action) {
        int marker = observation[2 + 2 * action] | (observation[3 + 2 * action] << 8);
        assert(marker > last && marker < count);
        assert(ids[marker] == decision_policy_context->bundle.mask_id);
        last = marker;
    }
    assert(observation[DECISION_POLICY_QTYPE_OFFSET] == 0);
    return ids;
}

static void check_observation(const obs_t* observation, const unsigned char* mask,
        const IBConnect4* oracle, int max_steps) {
    auto& tokenizer = decision_policy_context->tokenizer;
    std::string text = tokenizer.Decode(unpack(observation));
    int32_t board[IB_CONNECT4_CELLS];
    assert(ib_connect4_observe(oracle, board, IB_CONNECT4_CELLS) == 0);
    std::string expected_board;
    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 7; ++col) {
            if (col) expected_board += ',';
            expected_board += std::to_string(board[row * 7 + col]);
        }
        expected_board += '\n';
    }
    assert(text.find(expected_board) != std::string::npos);
    assert(text.find("Step " + std::to_string(ib_connect4_steps(oracle)) + " of " +
        std::to_string(max_steps)) != std::string::npos);
    for (int col = 0; col < 7; ++col) {
        assert(text.find("Column " + std::to_string(col)) != std::string::npos);
        assert(mask[col] == ((ib_connect4_legal_actions(oracle) >> col) & 1));
    }
}

static int choose(uint32_t mask, uint32_t selector) {
    std::vector<int> actions;
    for (int col = 0; col < 7; ++col) if ((mask >> col) & 1) actions.push_back(col);
    assert(!actions.empty());
    return actions[selector % actions.size()];
}

static int parity_games() {
    int total_steps = 0;
    int rejected_full = 0;
    for (uint32_t seed = 0; seed < 64; ++seed) {
        Fixture f(seed, 21);
        IBConnect4* oracle = ib_connect4_create(seed, 21);
        assert(oracle);
        check_observation(f.observations, f.masks, oracle, 21);
        uint32_t selector = seed + 1;
        std::vector<int> actions;
        while (uint32_t mask = ib_connect4_legal_actions(oracle)) {
            if (mask != 127) {
                int invalid = 0;
                while ((mask >> invalid) & 1) ++invalid;
                auto before = f.env.transition;
                std::vector<obs_t> bytes(f.observations, f.observations + OBS_SIZE);
                assert(decision_connect4_step(&f.env, invalid) == -1);
                assert(memcmp(&before, &f.env.transition, sizeof(before)) == 0);
                assert(memcmp(bytes.data(), f.observations, OBS_SIZE) == 0);
                ++rejected_full;
            }
            selector = selector * 1664525u + 1013904223u;
            int action = choose(mask, selector);
            actions.push_back(action);
            assert(ib_connect4_step(oracle, action) == 0);
            f.action = (float)action;
            puf_step(&f.env);
            ++total_steps;
            const auto& t = f.env.transition;
            assert(t.valid && t.seed == seed && t.action == action);
            assert(t.steps == ib_connect4_steps(oracle));
            assert(t.outcome == ib_connect4_outcome(oracle));
            assert(t.terminated == ib_connect4_terminated(oracle));
            assert(t.truncated == ib_connect4_truncated(oracle));
            assert(f.reward == (float)ib_connect4_reward(oracle));
            assert(f.terminal == (t.terminated || t.truncated));
            check_observation(t.observations, t.action_mask, oracle, 21);
            assert(!puf_truncation_observation(&f.env, &f.env.agents[0]));
            if (t.terminated) {
                assert(f.env.log.n == 1 && f.env.log.episode_length == t.steps);
                assert(f.env.log.score == t.outcome && f.env.log.episode_return == t.outcome);
                assert(f.env.log.perf == (t.outcome == 1 ? 1.0f : 0.0f));
                assert(f.env.episode_seed == seed + 1);
                assert(ib_connect4_steps(f.env.game) == 0);
                assert(memcmp(f.observations, t.observations, OBS_SIZE) != 0);
            } else check_observation(f.observations, f.masks, oracle, 21);
        }
        assert(ib_connect4_terminated(oracle));
        // Replay the same game with a cap exactly at its natural end. A terminal
        // result on the last allowed move must not become a bootstrapped timeout.
        Fixture capped(seed, (int)actions.size());
        for (int action : actions) {
            capped.action = (float)action;
            puf_step(&capped.env);
        }
        assert(capped.env.transition.terminated && !capped.env.transition.truncated);
        assert(capped.reward == (float)ib_connect4_reward(oracle));
        assert(!puf_truncation_observation(&capped.env, &capped.env.agents[0]));
        ib_connect4_destroy(oracle);
    }
    assert(rejected_full > 0);
    return total_steps;
}

static void timeouts_and_rejections() {
    Fixture f(UINT32_MAX, 1);
    IBConnect4* oracle = ib_connect4_create(UINT32_MAX, 1);
    f.action = 3;
    puf_step(&f.env);
    assert(ib_connect4_step(oracle, 3) == 0);
    const auto& t = f.env.transition;
    assert(t.valid && t.truncated && !t.terminated && t.steps == 1);
    assert(f.reward == 0 && f.terminal == 1);
    assert(f.env.episode_seed == 0 && f.env.next_seed == 1);
    assert(puf_truncation_observation(&f.env, &f.env.agents[0]) == t.observations);
    Agent wrong{};
    assert(puf_truncation_observation(&f.env, &wrong) == nullptr);
    check_observation(t.observations, t.action_mask, oracle, 1);
    assert(memcmp(f.observations, t.observations, OBS_SIZE) != 0);
    puf_reset(&f.env);
    assert(!t.valid && f.reward == 0 && f.terminal == 0);
    assert(!puf_truncation_observation(&f.env, &f.env.agents[0]));
    for (float action : {-1.0f, 7.0f, .5f, std::numeric_limits<float>::quiet_NaN()}) {
        f.action = action;
        bool rejected = false;
        try { puf_step(&f.env); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected && ib_connect4_steps(f.env.game) == 0);
    }
    ib_connect4_destroy(oracle);
    for (double cap : {0.0, -1.0, 1.5, double(INT_MAX) + 1,
            std::numeric_limits<double>::quiet_NaN()}) {
        Dict kwargs{};
        dict_set(&kwargs, "max_steps", cap);
        Env invalid{};
        bool rejected = false;
        try { puf_init(&invalid, &kwargs); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
        dict_clear(&kwargs);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: test_decision_connect4_env BUNDLE\n"; return 2; }
    decision_policy_context.reset(new DecisionPolicyContext(argv[1]));
    timeouts_and_rejections();
    int steps = parity_games();
    std::cout << "decision_connect4: " << steps << " engine/adapter transition checks, "
        << "64 seeded games and cap replays; timeout/reset/mask/token checks passed\n";
}
