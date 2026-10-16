#ifndef PUFFER_DECISION_CONNECT4_H
#define PUFFER_DECISION_CONNECT4_H

#define DECISION_ACTIONS 7
#include "../../src/decision_policy.h"
#include "pufferenv.h"
#include "engine/connect4.h"
#include <climits>
#include <cmath>
#include <locale>
#include <sstream>

#define ACT_SIZES {DECISION_ACTIONS}
#define NUM_ATNS 1
#define PUF_STEPS_PER_SEC 4
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

struct DecisionConnect4Transition {
    obs_t observations[OBS_SIZE];
    unsigned char action_mask[DECISION_ACTIONS];
    int valid;
    int action;
    int terminated;
    int truncated;
    int outcome;
    int steps;
    float reward;
    uint32_t seed;
};

struct Env {
    Log log;
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;
    IBConnect4* game;
    int max_steps;
    uint32_t episode_seed;
    uint32_t next_seed;
    DecisionConnect4Transition transition;
    int render_initialized;
};

static std::string decision_connect4_state(const IBConnect4* game, int max_steps) {
    int32_t board[IB_CONNECT4_CELLS];
    if (ib_connect4_observe(game, board, IB_CONNECT4_CELLS) != 0)
        throw std::runtime_error("decision_connect4 could not read board");
    std::ostringstream state;
    state.imbue(std::locale::classic());
    state << "Connect Four. Rows run top to bottom; columns 0 to 6 run left to right. "
          << "0=empty, 1=your piece, -1=opponent. Pieces fall to the lowest empty cell. "
          << "Step " << ib_connect4_steps(game) << " of " << max_steps << ". Board:\n";
    for (int row = 0; row < IB_CONNECT4_ROWS; ++row) {
        for (int col = 0; col < IB_CONNECT4_COLUMNS; ++col) {
            if (col) state << ',';
            state << board[row * IB_CONNECT4_COLUMNS + col];
        }
        state << '\n';
    }
    return state.str();
}

static void decision_connect4_observe(const Env* env, obs_t* output,
        unsigned char* action_mask) {
    decision_policy_encode(decision_connect4_state(env->game, env->max_steps),
        "Connect four of your pieces horizontally, vertically or diagonally. Choose a non-full column.",
        {"Column 0", "Column 1", "Column 2", "Column 3", "Column 4", "Column 5", "Column 6"}, output);
    uint32_t mask = ib_connect4_legal_actions(env->game);
    for (int col = 0; col < DECISION_ACTIONS; ++col)
        action_mask[col] = (mask >> col) & 1;
}

static const obs_t* puf_truncation_observation(Env* env, const Agent* agent) {
    const auto& transition = env->transition;
    return agent == &env->agents[0] && transition.valid && transition.truncated &&
        !transition.terminated ? transition.observations : nullptr;
}

static bool decision_connect4_buffers_ready(const Env* env) {
    const Agent& agent = env->agents[0];
    return agent.observations && agent.rewards && agent.terminals && agent.action_mask;
}

static int decision_connect4_start_episode(Env* env, uint32_t seed) {
    if (ib_connect4_reset(env->game, seed, env->max_steps) != 0) return -1;
    env->episode_seed = seed;
    env->next_seed = seed + UINT32_C(1);
    decision_connect4_observe(env, env->agents[0].observations, env->agents[0].action_mask);
    return 0;
}

// Explicit reset and step support deterministic collectors without autoreset.
// Engine rejection leaves the episode, transition and bound buffers unchanged.
int decision_connect4_reset(Env* env, uint32_t seed) {
    if (!env || !decision_connect4_buffers_ready(env) ||
            decision_connect4_start_episode(env, seed) != 0) return -1;
    memset(&env->transition, 0, sizeof(env->transition));
    env->agents[0].rewards[0] = env->agents[0].terminals[0] = 0;
    return 0;
}

int decision_connect4_step(Env* env, int action) {
    if (!env || !decision_connect4_buffers_ready(env) ||
            ib_connect4_step(env->game, action) != 0) return -1;
    auto& transition = env->transition;
    decision_connect4_observe(env, transition.observations, transition.action_mask);
    transition.valid = 1;
    transition.action = action;
    transition.terminated = ib_connect4_terminated(env->game);
    transition.truncated = ib_connect4_truncated(env->game);
    transition.outcome = ib_connect4_outcome(env->game);
    transition.steps = ib_connect4_steps(env->game);
    transition.reward = (float)ib_connect4_reward(env->game);
    transition.seed = env->episode_seed;
    Agent& agent = env->agents[0];
    memcpy(agent.observations, transition.observations, OBS_SIZE);
    memcpy(agent.action_mask, transition.action_mask, DECISION_ACTIONS);
    agent.rewards[0] = transition.reward;
    agent.terminals[0] = transition.terminated || transition.truncated;
    if (agent.terminals[0]) {
        // perf is win rate, as in stock Connect Four. score and episode_return
        // retain the signed result; timeouts are separately reported.
        env->log.perf += transition.outcome == 1 ? 1.0f : 0.0f;
        env->log.score += (float)ib_connect4_score(env->game);
        env->log.episode_return += (float)ib_connect4_episode_return(env->game);
        env->log.episode_length += (float)transition.steps;
        env->log.terminated += (float)transition.terminated;
        env->log.truncated += (float)transition.truncated;
        env->log.n += 1;
    }
    return 0;
}

void puf_init(Env* env, Dict* kwargs) {
    double max_steps = dict_get(kwargs, "max_steps");
    DictItem* agents = dict_find(kwargs, "num_agents");
    if (!(max_steps >= 1 && max_steps <= INT_MAX) || max_steps != std::floor(max_steps) ||
            (agents && agents->value != 1))
        throw std::invalid_argument("decision_connect4 requires a positive int32 max_steps and num_agents=1");
    env->num_agents = 1;
    env->max_steps = (int)max_steps;
    env->episode_seed = env->next_seed = (uint32_t)env->rng;
    env->agents[0].policy = 0;
    env->game = ib_connect4_create(env->episode_seed, env->max_steps);
    if (!env->game) throw std::runtime_error("decision_connect4 could not allocate engine");
}

void puf_reset(Env* env) {
    if (decision_connect4_reset(env, env->next_seed) != 0)
        throw std::runtime_error("decision_connect4 reset requires bound Agent buffers");
}

void puf_step(Env* env) {
    float action = env->agents[0].actions[0];
    if (!(action >= 0 && action < DECISION_ACTIONS) || action != (float)(int)action ||
            decision_connect4_step(env, (int)action) != 0)
        throw std::invalid_argument("decision_connect4 requires a legal integer column 0..6");
    if (env->transition.terminated || env->transition.truncated) {
        // Preserve final transition/reward/done while publishing the next board.
        if (decision_connect4_start_episode(env, env->next_seed) != 0)
            throw std::runtime_error("decision_connect4 autoreset failed");
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
    constexpr int cell = 64;
    if (!env->render_initialized) {
        InitWindow(7 * cell, 6 * cell, "PufferLib Decision Connect Four");
        SetTargetFPS(PUF_STEPS_PER_SEC);
        env->render_initialized = 1;
    }
    int32_t board[IB_CONNECT4_CELLS];
    ib_connect4_observe(env->game, board, IB_CONNECT4_CELLS);
    BeginDrawing();
    ClearBackground((Color){20, 40, 120, 255});
    for (int i = 0; i < IB_CONNECT4_CELLS; ++i) {
        Color color = board[i] == 1 ? (Color){245, 210, 45, 255} :
            board[i] == -1 ? (Color){225, 60, 60, 255} : (Color){20, 20, 30, 255};
        DrawCircle((i % 7) * cell + cell / 2, (i / 7) * cell + cell / 2, cell * .4f, color);
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
    ib_connect4_destroy(env->game);
    env->game = nullptr;
}

#endif
