#ifndef PUFFER_DECISION_2048_H
#define PUFFER_DECISION_2048_H

#define DECISION_ACTIONS 4
#include "../../src/decision_policy.h"
#include "pufferenv.h"
#include "engine/g2048.h"
#include <climits>
#include <locale>
#include <sstream>

#define ACT_SIZES {DECISION_ACTIONS}
#define NUM_ATNS 1
#define PUF_STEPS_PER_SEC 5
#define PUF_HAS_TRUNCATION 1

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float max_tile;
    float terminated;
    float truncated;
    float n;
};

struct Decision2048Transition {
    obs_t observations[OBS_SIZE];
    unsigned char action_mask[DECISION_ACTIONS];
    int valid;
    int action;
    int terminated;
    int truncated;
    int steps;
    float reward;
    double score;
    double episode_return;
    int32_t max_tile;
    uint32_t seed;
};

struct Env {
    Log log;
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;
    IBG2048* game;
    int max_steps;
    uint32_t episode_seed;
    uint32_t next_seed;
    Decision2048Transition transition;
    int render_initialized;
};

static std::string decision_2048_state(const Env* env) {
    int32_t board[16];
    if (ib_g2048_observe(env->game, board, 16) != 0)
        throw std::runtime_error("decision_2048 observation requires a valid engine");
    std::ostringstream state;
    state.imbue(std::locale::classic());
    state << "2048 on a 4 by 4 grid. Rows run top to bottom; columns left to right. "
          << "0=empty; other numbers are tile values. Step " << ib_g2048_steps(env->game)
          << " of " << env->max_steps << ". Board:\n";
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            if (col) state << ',';
            state << board[4 * row + col];
        }
        state << '\n';
    }
    return state.str();
}

static void decision_2048_observe(const Env* env, obs_t* observations,
        unsigned char* action_mask) {
    decision_policy_encode(decision_2048_state(env),
        "Slide tiles to merge equal neighbors once per move. A changed board spawns a 2 or 4. "
        "Build larger tiles and avoid a board with no available moves.",
        {"Move up", "Move down", "Move left", "Move right"}, observations);
    uint32_t mask = ib_g2048_legal_actions(env->game);
    for (int action = 0; action < DECISION_ACTIONS; ++action)
        action_mask[action] = (mask >> action) & 1;
}

static bool decision_2048_buffers_ready(const Env* env) {
    if (!env || !env->game) return false;
    const auto& agent = env->agents[0];
    return agent.observations && agent.action_mask && agent.rewards && agent.terminals;
}

static int decision_2048_start_episode(Env* env, uint32_t seed) {
    if (ib_g2048_reset(env->game, seed, env->max_steps) != 0) return -1;
    env->episode_seed = seed;
    env->next_seed = seed + UINT32_C(1);
    decision_2048_observe(env, env->agents[0].observations, env->agents[0].action_mask);
    return 0;
}

// Explicit seed control and non-autoresetting step API mirror the campaign engine.
static int decision_2048_reset(Env* env, uint32_t seed) {
    if (!decision_2048_buffers_ready(env) || decision_2048_start_episode(env, seed) != 0)
        return -1;
    memset(&env->transition, 0, sizeof(env->transition));
    env->agents[0].rewards[0] = env->agents[0].terminals[0] = 0;
    return 0;
}

static int decision_2048_step(Env* env, int action) {
    if (!decision_2048_buffers_ready(env) || ib_g2048_step(env->game, action) != 0)
        return -1;
    auto& transition = env->transition;
    decision_2048_observe(env, transition.observations, transition.action_mask);
    transition.valid = 1;
    transition.action = action;
    transition.terminated = ib_g2048_terminated(env->game);
    transition.truncated = ib_g2048_truncated(env->game);
    transition.steps = ib_g2048_steps(env->game);
    transition.reward = float(ib_g2048_reward(env->game));
    transition.score = ib_g2048_score(env->game);
    transition.episode_return = ib_g2048_episode_return(env->game);
    transition.seed = env->episode_seed;
    int32_t board[16];
    ib_g2048_observe(env->game, board, 16);
    transition.max_tile = 0;
    for (int32_t tile : board) transition.max_tile = std::max(transition.max_tile, tile);

    auto& agent = env->agents[0];
    memcpy(agent.observations, transition.observations, OBS_SIZE);
    memcpy(agent.action_mask, transition.action_mask, DECISION_ACTIONS);
    agent.rewards[0] = transition.reward;
    agent.terminals[0] = transition.terminated || transition.truncated;
    if (agent.terminals[0]) {
        env->log.perf += transition.max_tile >= 2048 ? 1.0f : 0.0f;
        env->log.score += float(transition.score);
        env->log.episode_return += float(transition.episode_return);
        env->log.episode_length += transition.steps;
        env->log.max_tile += transition.max_tile;
        env->log.terminated += transition.terminated;
        env->log.truncated += transition.truncated;
        env->log.n += 1;
    }
    return 0;
}

static const obs_t* puf_truncation_observation(Env* env, const Agent* agent) {
    const auto& t = env->transition;
    return agent == &env->agents[0] && t.valid && t.truncated && !t.terminated
        ? t.observations : nullptr;
}

void puf_init(Env* env, Dict* kwargs) {
    double max_steps = dict_get(kwargs, "max_steps");
    auto* agents = dict_find(kwargs, "num_agents");
    if (!(max_steps >= 1 && max_steps <= INT_MAX) ||
            max_steps != double(int(max_steps)) || (agents && agents->value != 1))
        throw std::invalid_argument("decision_2048 requires positive int32 max_steps and num_agents=1");
    env->num_agents = 1;
    env->max_steps = int(max_steps);
    env->episode_seed = env->rng;
    env->next_seed = env->episode_seed;
    env->agents[0].policy = 0;
    env->game = ib_g2048_create(env->episode_seed, env->max_steps);
    if (!env->game) throw std::bad_alloc();
}

void puf_reset(Env* env) {
    if (decision_2048_reset(env, env->next_seed) != 0)
        throw std::runtime_error("decision_2048 reset requires an engine and bound agent buffers");
}

void puf_step(Env* env) {
    float action = env->agents[0].actions[0];
    if (!(action >= 0 && action < DECISION_ACTIONS) || action != float(int(action)) ||
            decision_2048_step(env, int(action)) != 0)
        throw std::invalid_argument("decision_2048 requires an integer action in [0,3]");
    if ((env->transition.terminated || env->transition.truncated) &&
            decision_2048_start_episode(env, env->next_seed) != 0)
        throw std::runtime_error("decision_2048 autoreset failed");
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "max_tile", log->max_tile);
    dict_set(out, "terminated", log->terminated);
    dict_set(out, "truncated", log->truncated);
    dict_set(out, "n", log->n);
}

void puf_render(Env* env) {
#ifndef PUF_HEADLESS
    const int cell = 110;
    if (!env->render_initialized) {
        InitWindow(4 * cell, 4 * cell, "PufferLib Decision 2048");
        SetTargetFPS(PUF_STEPS_PER_SEC);
        env->render_initialized = 1;
    }
    int32_t board[16];
    ib_g2048_observe(env->game, board, 16);
    BeginDrawing();
    ClearBackground((Color){30, 34, 38, 255});
    for (int i = 0; i < 16; ++i) {
        int x = i % 4 * cell, y = i / 4 * cell;
        DrawRectangle(x + 3, y + 3, cell - 6, cell - 6,
            board[i] ? (Color){211, 173, 107, 255} : (Color){68, 73, 77, 255});
        if (board[i]) {
            char value[16];
            snprintf(value, sizeof(value), "%d", board[i]);
            int size = strlen(value) > 6 ? 15 : 24;
            DrawText(value, x + (cell - MeasureText(value, size)) / 2, y + 42, size,
                (Color){24, 28, 32, 255});
        }
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
    ib_g2048_destroy(env->game);
    env->game = nullptr;
}

#endif
