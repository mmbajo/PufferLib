#ifndef PUFFER_DECISION_LIGHTSOUT_H
#define PUFFER_DECISION_LIGHTSOUT_H

#define DECISION_ACTIONS 25
#include "../../src/decision_policy.h"
#include "pufferenv.h"
#include "engine/lightsout.h"
#include <climits>
#include <cmath>
#include <locale>
#include <sstream>

#define ACT_SIZES {DECISION_ACTIONS}
#define NUM_ATNS 1
#define PUF_STEPS_PER_SEC 10
#define PUF_HAS_TRUNCATION 1

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float terminated;
    float truncated;
    float n;
};

struct DecisionLightsOutTransition {
    obs_t observations[OBS_SIZE];
    unsigned char action_mask[DECISION_ACTIONS];
    int valid;
    int action;
    int terminated;
    int truncated;
    int steps;
    float reward;
    float score;
    float episode_return;
    uint32_t seed;
};

struct Env {
    Log log;
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;
    IBLightsOut* lightsout;
    int max_steps;
    int last_action;
    int prev_action;
    uint32_t episode_seed;
    uint32_t next_seed;
    DecisionLightsOutTransition transition;
    int render_initialized;
};

static std::string decision_lightsout_state(const Env* env) {
    int32_t board[IB_LIGHTSOUT_CELLS];
    if (ib_lightsout_observe(env->lightsout, board, IB_LIGHTSOUT_CELLS) != 0)
        throw std::logic_error("decision_lightsout has no initialized engine");
    std::ostringstream state;
    state.imbue(std::locale::classic());
    state << "Lights Out on a 5 by 5 board. 1=on, 0=off. Cells A through Y run "
          << "left to right, top to bottom. Pressing a cell toggles it and its "
          << "orthogonal neighbors. Step " << ib_lightsout_steps(env->lightsout)
          << " of " << env->max_steps << ". Last action="
          << (env->last_action < 0 ? '-' : char('A' + env->last_action))
          << "; preceding action="
          << (env->prev_action < 0 ? '-' : char('A' + env->prev_action))
          << "; -=none. Board:\n";
    for (int row = 0; row < IB_LIGHTSOUT_GRID_SIZE; ++row) {
        for (int col = 0; col < IB_LIGHTSOUT_GRID_SIZE; ++col) {
            int index = row * IB_LIGHTSOUT_GRID_SIZE + col;
            if (col) state << ' ';
            state << char('A' + index) << '=' << board[index];
        }
        state << '\n';
    }
    return state.str();
}

static void decision_lightsout_observe(const Env* env, obs_t* observations,
        unsigned char* action_mask) {
    static const std::vector<std::string> options = [] {
        std::vector<std::string> result;
        for (int action = 0; action < DECISION_ACTIONS; ++action)
            result.emplace_back(1, char('A' + action));
        return result;
    }();
    decision_policy_encode(decision_lightsout_state(env),
        "Turn all lights off. Avoid immediate repeats and ABA sequences. Select a cell.",
        options, observations);
    uint32_t mask = ib_lightsout_legal_actions(env->lightsout);
    for (int action = 0; action < DECISION_ACTIONS; ++action)
        action_mask[action] = (mask >> action) & 1;
}

static const obs_t* puf_truncation_observation(Env* env, const Agent* agent) {
    const auto& transition = env->transition;
    return agent == &env->agents[0] && transition.valid && transition.truncated &&
        !transition.terminated ? transition.observations : nullptr;
}

static bool decision_lightsout_buffers_ready(const Env* env) {
    if (!env) return false;
    const auto& agent = env->agents[0];
    return agent.observations && agent.rewards && agent.terminals && agent.action_mask;
}

static int decision_lightsout_start_episode(Env* env, uint32_t seed) {
    if (ib_lightsout_reset(env->lightsout, seed, env->max_steps) != 0) return -1;
    env->episode_seed = seed;
    env->next_seed = seed + UINT32_C(1);
    env->last_action = env->prev_action = -1;
    decision_lightsout_observe(env, env->agents[0].observations, env->agents[0].action_mask);
    return 0;
}

// Explicit seed and no-autoreset operations also support exact evaluation traces.
static int decision_lightsout_reset(Env* env, uint32_t seed) {
    if (!decision_lightsout_buffers_ready(env) ||
            decision_lightsout_start_episode(env, seed) != 0) return -1;
    memset(&env->transition, 0, sizeof(env->transition));
    env->agents[0].rewards[0] = 0;
    env->agents[0].terminals[0] = 0;
    return 0;
}

static int decision_lightsout_step(Env* env, int action) {
    if (!decision_lightsout_buffers_ready(env) ||
            ib_lightsout_step(env->lightsout, action) != 0) return -1;
    env->prev_action = env->last_action;
    env->last_action = action;
    auto& transition = env->transition;
    decision_lightsout_observe(env, transition.observations, transition.action_mask);
    transition.valid = 1;
    transition.action = action;
    transition.terminated = ib_lightsout_terminated(env->lightsout);
    transition.truncated = ib_lightsout_truncated(env->lightsout);
    transition.steps = ib_lightsout_steps(env->lightsout);
    transition.reward = (float)ib_lightsout_reward(env->lightsout);
    transition.score = (float)ib_lightsout_score(env->lightsout);
    transition.episode_return = (float)ib_lightsout_episode_return(env->lightsout);
    transition.seed = env->episode_seed;
    auto& agent = env->agents[0];
    memcpy(agent.observations, transition.observations, OBS_SIZE);
    memcpy(agent.action_mask, transition.action_mask, DECISION_ACTIONS);
    agent.rewards[0] = transition.reward;
    agent.terminals[0] = transition.terminated || transition.truncated;
    if (agent.terminals[0]) {
        env->log.perf += transition.score;
        env->log.score += transition.score;
        env->log.episode_return += transition.episode_return;
        env->log.episode_length += transition.steps;
        env->log.terminated += transition.terminated;
        env->log.truncated += transition.truncated;
        env->log.n += 1;
    }
    return 0;
}

void puf_init(Env* env, Dict* kwargs) {
    double max_steps = dict_get(kwargs, "max_steps");
    DictItem* agents = dict_find(kwargs, "num_agents");
    if (!(max_steps >= 1 && max_steps <= INT_MAX) ||
            max_steps != std::floor(max_steps) || (agents && agents->value != 1))
        throw std::invalid_argument("decision_lightsout requires positive int32 max_steps and num_agents=1");
    env->num_agents = 1;
    env->max_steps = (int)max_steps;
    env->episode_seed = env->next_seed = (uint32_t)env->rng;
    env->last_action = env->prev_action = -1;
    env->agents[0].policy = 0;
    env->lightsout = ib_lightsout_create(env->episode_seed, env->max_steps);
    if (!env->lightsout) throw std::bad_alloc();
    memset(&env->transition, 0, sizeof(env->transition));
}

void puf_reset(Env* env) {
    if (decision_lightsout_reset(env, env->next_seed) != 0)
        throw std::logic_error("decision_lightsout reset requires initialized engine and bound buffers");
}

void puf_step(Env* env) {
    float action = env->agents[0].actions[0];
    if (!(action >= 0 && action < DECISION_ACTIONS) || action != std::floor(action) ||
            decision_lightsout_step(env, (int)action) != 0)
        throw std::invalid_argument("decision_lightsout action must be a live cell index 0..24");
    if (env->transition.terminated || env->transition.truncated) {
        // Keep final tokens/reward/done for PPO while publishing the next state.
        if (decision_lightsout_start_episode(env, env->next_seed) != 0)
            throw std::logic_error("decision_lightsout autoreset failed");
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "terminated", log->terminated);
    dict_set(out, "truncated", log->truncated);
    dict_set(out, "n", log->n);
}

void puf_render(Env* env) {
#ifndef PUF_HEADLESS
    const int cell = 80, size = IB_LIGHTSOUT_GRID_SIZE * cell;
    if (!env->render_initialized) {
        InitWindow(size, size, "PufferLib Decision Lights Out");
        SetTargetFPS(PUF_STEPS_PER_SEC);
        env->render_initialized = 1;
    }
    int32_t board[IB_LIGHTSOUT_CELLS];
    ib_lightsout_observe(env->lightsout, board, IB_LIGHTSOUT_CELLS);
    BeginDrawing();
    ClearBackground((Color){16, 24, 24, 255});
    for (int index = 0; index < IB_LIGHTSOUT_CELLS; ++index) {
        int x = (index % IB_LIGHTSOUT_GRID_SIZE) * cell;
        int y = (index / IB_LIGHTSOUT_GRID_SIZE) * cell;
        DrawRectangle(x + 2, y + 2, cell - 4, cell - 4,
            board[index] ? (Color){255, 222, 64, 255} : (Color){45, 65, 65, 255});
        char label[2] = {char('A' + index), 0};
        DrawText(label, x + 30, y + 26, 24, board[index] ? BLACK : WHITE);
    }
    EndDrawing();
    puf_web_vsync();
#else
    (void)env;
#endif
}

void puf_close(Env* env) {
#ifndef PUF_HEADLESS
    if (env->render_initialized) CloseWindow();
#endif
    env->render_initialized = 0;
    ib_lightsout_destroy(env->lightsout);
    env->lightsout = nullptr;
}

#endif
