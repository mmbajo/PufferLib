#define PUF_HEADLESS
#include "../ocean/decision_2048/decision_2048.h"
#include <cassert>
#include <iostream>
#include <limits>

extern "C" void test_decision_2048_fixture(IBG2048*, const unsigned char*, int);

struct Fixture {
    Env env{};
    obs_t observations[OBS_SIZE]{};
    unsigned char mask[4]{};
    float action = 0, reward = 0, terminal = 0;
    explicit Fixture(uint32_t seed, int max_steps) {
        Dict kwargs{};
        dict_set(&kwargs, "max_steps", max_steps);
        env.rng = seed;
        puf_init(&env, &kwargs);
        dict_clear(&kwargs);
        env.agents[0].observations = observations;
        env.agents[0].actions = &action;
        env.agents[0].action_mask = mask;
        env.agents[0].rewards = &reward;
        env.agents[0].terminals = &terminal;
        puf_reset(&env);
    }
    ~Fixture() { puf_close(&env); }
};

static void same_board(const IBG2048* left, const IBG2048* right) {
    int32_t a[16], b[16];
    assert(ib_g2048_observe(left, a, 16) == 0);
    assert(ib_g2048_observe(right, b, 16) == 0);
    assert(memcmp(a, b, sizeof(a)) == 0);
}

static void tokens(const obs_t* observation) {
    int count = observation[0] | observation[1] << 8;
    assert(count > 0 && count <= decision_policy_context->bundle.max_len);
    assert(observation[DECISION_POLICY_QTYPE_OFFSET] == 0);
    int previous = -1;
    for (int action = 0; action < 4; ++action) {
        int marker = observation[2 + 2 * action] | observation[3 + 2 * action] << 8;
        assert(marker > previous && marker < count);
        uint32_t id = 0;
        for (int byte = 0; byte < 4; ++byte)
            id |= uint32_t(observation[DECISION_POLICY_HEADER_BYTES + 4 * marker + byte]) << (8 * byte);
        assert(id == decision_policy_context->bundle.mask_id);
        previous = marker;
    }
}

static int replays() {
    int transitions = 0;
    for (uint32_t seed = 0; seed < 24; ++seed) {
        Fixture f(seed, 127), noise(seed + 100, 7);
        IBG2048* reference = ib_g2048_create(seed, 127);
        uint32_t current_seed = seed;
        unsigned int rng = seed + 1;
        same_board(f.env.game, reference);
        for (int step = 0; step < 256; ++step) {
            uint32_t legal = ib_g2048_legal_actions(reference);
            for (int a = 0; a < 4; ++a) assert(f.mask[a] == ((legal >> a) & 1));
            rng = rng * 1664525u + 1013904223u;
            int action = (rng >> 24) & 3;
            f.action = float(action);
            assert(ib_g2048_step(reference, action) == 0);
            puf_step(&f.env);
            assert(f.reward == float(ib_g2048_reward(reference)));
            const auto& t = f.env.transition;
            assert(t.valid && t.action == action && t.seed == current_seed);
            assert(t.steps == ib_g2048_steps(reference));
            assert(t.terminated == ib_g2048_terminated(reference));
            assert(t.truncated == ib_g2048_truncated(reference));
            assert(t.score == ib_g2048_score(reference));
            assert(t.episode_return == ib_g2048_episode_return(reference));
            assert(f.terminal == float(t.terminated || t.truncated));
            Env expected = f.env;
            expected.game = reference;
            obs_t expected_observation[OBS_SIZE];
            unsigned char expected_mask[4];
            decision_2048_observe(&expected, expected_observation, expected_mask);
            assert(memcmp(t.observations, expected_observation, OBS_SIZE) == 0);
            assert(memcmp(t.action_mask, expected_mask, 4) == 0);
            assert(bool(puf_truncation_observation(&f.env, &f.env.agents[0])) == bool(t.truncated));
            if (f.terminal) {
                ++current_seed;
                assert(ib_g2048_reset(reference, current_seed, 127) == 0);
                assert(f.env.episode_seed == current_seed);
                assert(memcmp(t.observations, f.observations, OBS_SIZE) != 0);
            }
            same_board(f.env.game, reference);
            tokens(f.observations);
            assert(decision_2048_reset(&noise.env, rng) == 0);
            noise.action = (action + 1) % 4;
            puf_step(&noise.env);
            ++transitions;
        }
        ib_g2048_destroy(reference);
    }
    return transitions;
}

static void boundaries() {
    Fixture f(42, 1);
    const unsigned char merge[16] = {1, 1};
    test_decision_2048_fixture(f.env.game, merge, 1);
    f.action = 2;
    puf_step(&f.env);
    assert(f.terminal == 1 && f.env.transition.truncated && !f.env.transition.terminated);
    assert(f.env.transition.score == 4 && f.reward == float(.05));
    assert(f.env.transition.episode_return == .05 && f.env.transition.max_tile == 4);
    assert(f.env.log.score == 4 && f.env.log.episode_return == float(.05));
    assert(f.env.log.perf == 0 && f.env.log.n == 1 && f.env.log.truncated == 1);
    assert(puf_truncation_observation(&f.env, &f.env.agents[0]) == f.env.transition.observations);
    Agent wrong{};
    assert(!puf_truncation_observation(&f.env, &wrong));
    for (unsigned char legal : f.env.transition.action_mask) assert(!legal);
    assert(ib_g2048_steps(f.env.game) == 0);
    tokens(f.env.transition.observations);
    puf_reset(&f.env);
    assert(!f.env.transition.valid && f.terminal == 0 && f.reward == 0);
    assert(!puf_truncation_observation(&f.env, &f.env.agents[0]));

    // A full-board game over on the cap step must not receive value bootstrap.
    const unsigned char blocked[16] = {1,1,3,4,4,5,6,7,5,6,7,8,6,7,8,9};
    test_decision_2048_fixture(f.env.game, blocked, 1);
    puf_step(&f.env);
    assert(f.terminal && f.env.transition.terminated && !f.env.transition.truncated);
    assert(f.reward == float(-.95) && f.env.transition.score == 4);
    assert(!puf_truncation_observation(&f.env, &f.env.agents[0]));

    f.env.max_steps = 8;
    const unsigned char one[16] = {1};
    test_decision_2048_fixture(f.env.game, one, 8);
    decision_2048_observe(&f.env, f.observations, f.mask);
    assert(!f.mask[0] && f.mask[1] && !f.mask[2] && f.mask[3]);
    assert(decision_2048_step(&f.env, 2) == 0); // Accepted no-op, but masked during training.
    assert(f.reward == float(-.05) && !f.terminal && f.env.transition.steps == 1);
    int32_t board[16];
    ib_g2048_observe(f.env.game, board, 16);
    assert(board[0] == 2);
    for (int i = 1; i < 16; ++i) assert(board[i] == 0);
    const std::string state = decision_2048_state(&f.env);
    assert(state.find("2,0,0,0\n0,0,0,0\n") != std::string::npos);
    assert(state.find("Step 1 of 8") != std::string::npos);
    for (float invalid : {-1.f, 4.f, .5f, std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::infinity()}) {
        auto before = f.env.transition;
        f.action = invalid;
        bool rejected = false;
        try { puf_step(&f.env); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected && memcmp(&before, &f.env.transition, sizeof(before)) == 0);
        assert(decision_2048_state(&f.env) == state);
    }
    unsigned char large[16];
    memset(large, 30, sizeof(large));
    test_decision_2048_fixture(f.env.game, large, 8);
    decision_2048_observe(&f.env, f.observations, f.mask);
    tokens(f.observations);
    assert(decision_2048_state(&f.env).find("1073741824,1073741824") != std::string::npos);

    assert(decision_2048_reset(&f.env, UINT32_MAX) == 0);
    assert(f.env.next_seed == 0 && f.env.episode_seed == UINT32_MAX);
    puf_reset(&f.env);
    assert(f.env.episode_seed == 0 && f.env.next_seed == 1);
}

static void invalid_config() {
    for (double bad : {0., -1., 1.5, double(INT_MAX) + 1,
            std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        Dict kwargs{};
        dict_set(&kwargs, "max_steps", bad);
        Env env{};
        bool rejected = false;
        try { puf_init(&env, &kwargs); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected && !env.game);
        dict_clear(&kwargs);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: test_decision_2048_env BUNDLE\n"; return 2; }
    decision_policy_context.reset(new DecisionPolicyContext(argv[1]));
    boundaries();
    invalid_config();
    int count = replays();
    std::cout << "PASS: " << count << " campaign/adapter transitions; merge score/reward, "
              << "masks, no-ops, seeded autoresets, tokenizer and timeout boundaries\n";
}
