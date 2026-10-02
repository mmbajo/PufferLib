// Build twice: normal native-tokenizer adapter and -DTEST_STOCK_CARTPOLE.
// Both executables print identical physics/reward/reset traces; the adapter also
// checks timeout snapshots, action ordering and input rejection without a GPU.
#define PUF_HEADLESS
#ifdef TEST_STOCK_CARTPOLE
#include "../ocean/cartpole/cartpole.h"
#else
#include "../ocean/decision_cartpole/decision_cartpole.h"
#endif
#include <cassert>
#include <iostream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>

static Dict defaults() {
    Dict kwargs{};
    dict_set(&kwargs, "cart_mass", 1);
    dict_set(&kwargs, "pole_mass", .1);
    dict_set(&kwargs, "pole_length", .5);
    dict_set(&kwargs, "gravity", 9.8);
    dict_set(&kwargs, "force_mag", 10);
    dict_set(&kwargs, "dt", .02);
    dict_set(&kwargs, "continuous", 0);
    dict_set(&kwargs, "max_steps", 200);
    return kwargs;
}

struct Fixture {
    Env env{};
    obs_t observations[OBS_SIZE]{};
    float action = 0, reward = 0, terminal = 0;
    explicit Fixture(Dict& kwargs, unsigned int seed = 173) {
        env.rng = seed;
        puf_init(&env, &kwargs);
        env.agents[0].observations = observations;
        env.agents[0].actions = &action;
        env.agents[0].rewards = &reward;
        env.agents[0].terminals = &terminal;
        puf_reset(&env);
    }
    ~Fixture() { puf_close(&env); }
};

static void print_trace(const Fixture& f, int step) {
#ifdef TEST_STOCK_CARTPOLE
    const float physical_terminal = f.terminal;
    assert(f.observations[0] == f.env.x && f.observations[1] == f.env.x_dot);
    assert(f.observations[2] == f.env.theta && f.observations[3] == f.env.theta_dot);
#else
    const float physical_terminal = f.env.transition.terminated;
    assert(f.terminal == (f.env.transition.terminated || f.env.transition.truncated));
#endif
    std::cout << step << ' ' << f.env.tick << ' ' << f.env.rng << ' '
              << std::hexfloat << f.env.x << ' ' << f.env.x_dot << ' '
              << f.env.theta << ' ' << f.env.theta_dot << ' '
              << f.reward << ' ' << physical_terminal << ' ' << f.env.episode_return << ' '
              << f.env.log.perf << ' ' << f.env.log.episode_length << ' '
              << f.env.log.x_threshold_termination << ' ' << f.env.log.pole_angle_termination
              << ' ' << f.env.log.max_steps_termination << ' ' << f.env.log.n << ' '
              << f.env.log.score << '\n';
}

static void traces() {
    Dict kwargs = defaults();
    Fixture normal(kwargs);
    print_trace(normal, -1);
    unsigned int action_rng = 17;
    for (int i = 0; i < 2048; ++i) {
        action_rng = action_rng * 1664525u + 1013904223u;
        normal.action = (action_rng >> 24) & 1;
        puf_step(&normal.env);
        print_trace(normal, i);
    }
    dict_set(&kwargs, "gravity", 0);
    dict_set(&kwargs, "force_mag", 0);
    Fixture stable(kwargs);
    stable.env.x = stable.env.x_dot = stable.env.theta = stable.env.theta_dot = 0;
    compute_observations(&stable.env);
    for (int i = 0; i < 200; ++i) {
        puf_step(&stable.env);
        print_trace(stable, i);
    }
    assert(stable.env.tick == 0 && stable.env.log.max_steps_termination == 1);
    assert(stable.env.log.score == 200 && stable.env.log.episode_length == 200);
    assert(stable.env.log.perf == 199.0f / 200.0f && stable.reward == 0);
#ifdef TEST_STOCK_CARTPOLE
    assert(stable.terminal == 0); // Preserve the stock API's historical timeout flag.
#else
    assert(stable.terminal == 1);
    assert(puf_truncation_observation(&stable.env, &stable.env.agents[0]));
#endif
    dict_clear(&kwargs);
}

#ifndef TEST_STOCK_CARTPOLE
static void validate_tokens(const obs_t* observation) {
    int count = observation[0] | (observation[1] << 8);
    assert(count > 0 && count <= decision_policy_context->bundle.max_len);
    assert(observation[DECISION_POLICY_QTYPE_OFFSET] == 0);
    for (int action = 0; action < 2; ++action) {
        int marker = observation[2 + 2 * action] | (observation[3 + 2 * action] << 8);
        assert(marker > 0 && marker < count);
        unsigned int token = 0;
        for (int byte = 0; byte < 4; ++byte)
            token |= unsigned(observation[DECISION_POLICY_HEADER_BYTES + 4 * marker + byte]) << (8 * byte);
        assert(token == decision_policy_context->bundle.mask_id);
    }
}

static void adapter_boundaries() {
    Dict kwargs = defaults();
    dict_set(&kwargs, "gravity", 0);
    dict_set(&kwargs, "force_mag", 0);
    dict_set(&kwargs, "max_steps", 1);
    Fixture f(kwargs);
    f.env.x = .25f; f.env.x_dot = .125f; f.env.theta = 0; f.env.theta_dot = 0;
    Env expected = f.env;
    expected.x += expected.tau * expected.x_dot;
    expected.tick = 1;
    obs_t expected_final[OBS_SIZE];
    decision_cartpole_observe(&expected, expected_final);
    puf_step(&f.env);
    assert(f.terminal == 1 && f.reward == 0 && f.env.tick == 0);
    assert(f.env.transition.valid && f.env.transition.truncated && !f.env.transition.terminated);
    assert(f.env.transition.steps == 1);
    assert(memcmp(expected_final, f.env.transition.observations, OBS_SIZE) == 0);
    assert(memcmp(f.observations, f.env.transition.observations, OBS_SIZE) != 0);
    assert(puf_truncation_observation(&f.env, &f.env.agents[0]) == f.env.transition.observations);
    Agent wrong_agent{};
    assert(puf_truncation_observation(&f.env, &wrong_agent) == nullptr);
    validate_tokens(f.observations); validate_tokens(f.env.transition.observations);
    puf_reset(&f.env);
    assert(!f.env.transition.valid && !puf_truncation_observation(&f.env, &f.env.agents[0]));
    assert(f.reward == 0 && f.terminal == 0);

    f.env.x = X_THRESHOLD + .1f;
    f.env.x_dot = f.env.theta = f.env.theta_dot = 0;
    puf_step(&f.env);
    assert(f.env.transition.terminated && f.env.transition.truncated && f.terminal == 1);
    assert(puf_truncation_observation(&f.env, &f.env.agents[0]) == nullptr);
    puf_reset(&f.env);
    f.env.max_steps = 200;
    f.env.theta = THETA_THRESHOLD_RADIANS + .1f;
    f.env.x = f.env.x_dot = f.env.theta_dot = 0;
    puf_step(&f.env);
    assert(f.env.transition.terminated && !f.env.transition.truncated && f.terminal == 1);
    assert(puf_truncation_observation(&f.env, &f.env.agents[0]) == nullptr);
    puf_step(&f.env);
    assert(!f.env.transition.terminated && !f.env.transition.truncated && f.terminal == 0);

    // Action indices retain the stock left/right meaning.
    dict_set(&kwargs, "force_mag", 10);
    dict_set(&kwargs, "max_steps", 200);
    Fixture left(kwargs), right(kwargs);
    left.env.x = left.env.x_dot = left.env.theta = left.env.theta_dot = 0;
    right.env.x = right.env.x_dot = right.env.theta = right.env.theta_dot = 0;
    left.action = 0; right.action = 1;
    puf_step(&left.env); puf_step(&right.env);
    assert(left.env.x_dot < 0 && right.env.x_dot > 0);
    assert(left.env.x_dot == -right.env.x_dot && left.env.theta_dot == -right.env.theta_dot);

    // A non-classic process locale cannot change the observation representation.
    struct Comma : std::numpunct<char> { char do_decimal_point() const override { return ','; } };
    auto old_locale = std::locale();
    std::string before = decision_cartpole_state(&f.env);
    std::locale::global(std::locale(std::locale::classic(), new Comma));
    assert(decision_cartpole_state(&f.env) == before);
    std::locale::global(old_locale);
    assert(before.find("x=") != std::string::npos && before.find("x_dot=") != std::string::npos);
    assert(before.find("theta=") != std::string::npos && before.find("theta_dot=") != std::string::npos);
    assert(before.find("Step 1 of 200") != std::string::npos);
    for (float invalid : {-1.0f, 2.0f, .5f, std::numeric_limits<float>::quiet_NaN()}) {
        auto rng = f.env.rng;
        auto tick = f.env.tick;
        f.action = invalid;
        bool rejected = false;
        try { puf_step(&f.env); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected && rng == f.env.rng && tick == f.env.tick);
    }
    dict_clear(&kwargs);
}

static void reject_invalid_config() {
    struct Invalid { const char* key; double value; };
    for (const auto& invalid : {
            Invalid{"continuous", 1}, Invalid{"continuous", .5},
            Invalid{"continuous", std::numeric_limits<double>::quiet_NaN()},
            Invalid{"cart_mass", 0}, Invalid{"pole_mass", -1}, Invalid{"pole_length", 0},
            Invalid{"dt", 1e300}, Invalid{"dt", 1e-300}, Invalid{"gravity", -1},
            Invalid{"force_mag", std::numeric_limits<double>::infinity()},
            Invalid{"max_steps", 0}, Invalid{"max_steps", 1.5},
            Invalid{"max_steps", double(INT_MAX) + 1},
            Invalid{"max_steps", std::numeric_limits<double>::quiet_NaN()}}) {
        Dict kwargs = defaults();
        dict_set(&kwargs, invalid.key, invalid.value);
        Env env{};
        bool rejected = false;
        try { puf_init(&env, &kwargs); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
        dict_clear(&kwargs);
    }
}
#endif

int main(int argc, char** argv) {
#ifndef TEST_STOCK_CARTPOLE
    if (argc != 2) { std::cerr << "usage: test_decision_cartpole_env BUNDLE\n"; return 2; }
    decision_policy_context.reset(new DecisionPolicyContext(argv[1]));
    adapter_boundaries();
    reject_invalid_config();
#else
    (void)argc; (void)argv;
#endif
    traces();
}
