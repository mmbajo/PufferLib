#ifndef PUFFER_DECISION_CARTPOLE_H
#define PUFFER_DECISION_CARTPOLE_H

#define DECISION_ACTIONS 2
#include "../../src/decision_policy.h"
#include <climits>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>

#define PUF_HAS_TRUNCATION 1

struct DecisionCartpoleTransition {
    obs_t observations[OBS_SIZE];
    int valid;
    int terminated;
    int truncated;
    int steps;
};

struct Env;
static void decision_cartpole_observe(const Env* env, obs_t* output);
static void decision_cartpole_after_step(Env* env, bool terminated, bool truncated);

// Reuse stock CartPole, changing only its observation and episode-boundary
// interface. Its integration equations and reward convention remain unchanged.
#define CARTPOLE_CUSTOM_OBSERVATIONS
#define CARTPOLE_EXTRA_FIELDS int max_steps; DecisionCartpoleTransition transition;
#define CARTPOLE_STEP_LIMIT(env) ((env)->max_steps)
#define CARTPOLE_LEFT_ACTION 0.0f
#define CARTPOLE_ENCODE_OBSERVATION(env, output) decision_cartpole_observe(env, output)
#define CARTPOLE_AFTER_STEP(env, terminated, truncated) \
    decision_cartpole_after_step(env, terminated, truncated)
#define puf_init cartpole_init
#define puf_reset cartpole_reset
#define puf_step cartpole_step
#include "../cartpole/cartpole.h"
#undef puf_init
#undef puf_reset
#undef puf_step

static std::string decision_cartpole_state(const Env* env) {
    if (!std::isfinite(env->x) || !std::isfinite(env->x_dot) ||
            !std::isfinite(env->theta) || !std::isfinite(env->theta_dot))
        throw std::runtime_error("decision_cartpole state contains a non-finite value");
    std::ostringstream state;
    state.imbue(std::locale::classic());
    state << std::setprecision(std::numeric_limits<float>::max_digits10)
          << "CartPole. Positive positions, velocities and angles point right. "
          << "x=" << env->x << " m; x_dot=" << env->x_dot << " m/s; "
          << "theta=" << env->theta << " radians; theta_dot=" << env->theta_dot
          << " radians/s. Step " << env->tick << " of " << env->max_steps
          << ". The cart must stay within -2.4 to 2.4 m; the pole within -12 to 12 degrees.";
    return state.str();
}

static void decision_cartpole_observe(const Env* env, obs_t* output) {
    decision_policy_encode(decision_cartpole_state(env),
        "Keep the pole upright and the cart within the track for as many steps as possible.",
        {"Push left", "Push right"}, output);
}

static void decision_cartpole_after_step(Env* env, bool terminated, bool truncated) {
    auto& transition = env->transition;
    transition.valid = 1;
    transition.terminated = terminated;
    transition.truncated = truncated;
    transition.steps = env->tick;
    if (terminated || truncated)
        decision_cartpole_observe(env, transition.observations);
    // Puffer's done mask must stop GAE at both boundaries. The final observation
    // below supplies bootstrap value only when no physical termination occurred.
    env->agents[0].terminals[0] = (terminated || truncated) ? 1.0f : 0.0f;
}

static const obs_t* puf_truncation_observation(Env* env, const Agent* agent) {
    const auto& transition = env->transition;
    return agent == &env->agents[0] && transition.valid && transition.truncated &&
        !transition.terminated ? transition.observations : nullptr;
}

static void decision_cartpole_validate_config(Dict* kwargs) {
    if (dict_get(kwargs, "continuous") != 0)
        throw std::invalid_argument("decision_cartpole requires continuous=0 (two discrete actions)");
    const char* positive[] = {"cart_mass", "pole_mass", "pole_length", "dt"};
    for (const char* key : positive) {
        double value = dict_get(kwargs, key);
        if (!std::isfinite(value) || !std::isfinite((float)value) || !((float)value > 0))
            throw std::invalid_argument(std::string("decision_cartpole requires finite positive ") + key);
    }
    const char* nonnegative[] = {"gravity", "force_mag"};
    for (const char* key : nonnegative) {
        double value = dict_get(kwargs, key);
        if (!std::isfinite(value) || !std::isfinite((float)value) || value < 0)
            throw std::invalid_argument(std::string("decision_cartpole requires finite nonnegative ") + key);
    }
}

void puf_init(Env* env, Dict* kwargs) {
    decision_cartpole_validate_config(kwargs);
    DictItem* limit = dict_find(kwargs, "max_steps");
    double max_steps = limit ? limit->value : MAX_STEPS;
    if (!(max_steps >= 1 && max_steps <= INT_MAX) || max_steps != std::floor(max_steps))
        throw std::invalid_argument("decision_cartpole max_steps must be a positive int32");
    env->max_steps = (int)max_steps;
    memset(&env->transition, 0, sizeof(env->transition));
    cartpole_init(env, kwargs);
}

void puf_reset(Env* env) {
    memset(&env->transition, 0, sizeof(env->transition));
    cartpole_reset(env);
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
}

void puf_step(Env* env) {
    float action = env->agents[0].actions[0];
    if (action != 0.0f && action != 1.0f)
        throw std::invalid_argument("decision_cartpole action must be 0 (left) or 1 (right)");
    cartpole_step(env);
}

#endif
