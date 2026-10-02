#ifndef PUFFER_DECISION_SNAKE_H
#define PUFFER_DECISION_SNAKE_H

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned char obs_t;
#include "pufferenv.h"
#include "engine/snake.h"

#define ACT_SIZES {IB_SNAKE_ACTIONS}
#ifndef OBS_SIZE
#define OBS_SIZE IB_SNAKE_CELLS
#endif
#define NUM_ATNS 1
#define PUF_STEPS_PER_SEC 10
// The core's single terminal buffer cannot represent timeout bootstrapping.
#define PUF_HAS_TRUNCATION 1

// Float-only layout, with n last, is required by PufferLib's log reduction.
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float terminated;
    float truncated;
    float n;
};

// Owns the result of the most recent accepted action, before any autoreset.
// The final mask is zero at either episode boundary, as in the engine API.
typedef struct DecisionSnakeTransition {
    obs_t observations[OBS_SIZE];
    unsigned char action_mask[IB_SNAKE_ACTIONS];
    int valid;
    int action;
    int terminated;
    int truncated;
    int outcome;
    int steps;
    float reward;
    float score;
    float episode_return;
    uint32_t seed;
} DecisionSnakeTransition;

struct Env {
    Log log;
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng; // Initial instance seed supplied by PufferLib.
    IBSnake* snake;
    int max_steps;
    uint32_t episode_seed;
    uint32_t next_seed;
    DecisionSnakeTransition transition;
    int render_initialized;
};

static void decision_snake_observe(Env* env, obs_t* observations,
        unsigned char* action_mask) {
    int32_t board[IB_SNAKE_CELLS];
    ib_snake_observe(env->snake, board, IB_SNAKE_CELLS);
#ifdef DECISION_SNAKE_ENCODE_OBSERVATION
    DECISION_SNAKE_ENCODE_OBSERVATION(env, board, observations);
#else
    for (int i = 0; i < IB_SNAKE_CELLS; i++) {
        observations[i] = (obs_t)(board[i] + 1);
    }
#endif
    uint32_t mask = ib_snake_legal_actions(env->snake);
    for (int i = 0; i < IB_SNAKE_ACTIONS; i++) {
        action_mask[i] = (unsigned char)((mask >> i) & 1);
    }
}

static int decision_snake_buffers_ready(const Env* env) {
    const Agent* agent = &env->agents[0];
    return agent->observations != NULL && agent->rewards != NULL &&
        agent->terminals != NULL && agent->action_mask != NULL;
}

// Reset only the live episode; the caller decides whether to clear a transition.
static int decision_snake_start_episode(Env* env, uint32_t seed) {
    if (ib_snake_reset(env->snake, seed, env->max_steps) != 0) {
        return -1;
    }
    env->episode_seed = seed;
    env->next_seed = seed + UINT32_C(1);
    decision_snake_observe(env, env->agents[0].observations,
        env->agents[0].action_mask);
    return 0;
}

// Explicit seeded reset for collectors that own episode seed scheduling.
// Call puf_init and bind Agent buffers first. Rejection changes no state.
int decision_snake_reset(Env* env, uint32_t seed) {
    if (env == NULL || !decision_snake_buffers_ready(env) ||
            decision_snake_start_episode(env, seed) != 0) {
        return -1;
    }
    memset(&env->transition, 0, sizeof(env->transition));
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
    return 0;
}

// One exact engine step, without autoreset. Invalid actions (including a
// reversal or an action after completion) return -1 without changing buffers.
int decision_snake_step(Env* env, int action) {
    if (env == NULL || !decision_snake_buffers_ready(env) ||
            ib_snake_step(env->snake, action) != 0) {
        return -1;
    }
    DecisionSnakeTransition* transition = &env->transition;
    decision_snake_observe(env, transition->observations, transition->action_mask);
    transition->valid = 1;
    transition->action = action;
    transition->terminated = ib_snake_terminated(env->snake);
    transition->truncated = ib_snake_truncated(env->snake);
    transition->outcome = ib_snake_outcome(env->snake);
    transition->steps = ib_snake_steps(env->snake);
    transition->reward = (float)ib_snake_reward(env->snake);
    transition->score = (float)ib_snake_score(env->snake);
    transition->episode_return = (float)ib_snake_episode_return(env->snake);
    transition->seed = env->episode_seed;

    Agent* agent = &env->agents[0];
    memcpy(agent->observations, transition->observations, sizeof(transition->observations));
    memcpy(agent->action_mask, transition->action_mask, sizeof(transition->action_mask));
    agent->rewards[0] = transition->reward;
    agent->terminals[0] = (float)(transition->terminated || transition->truncated);
    if (agent->terminals[0]) {
        env->log.perf += transition->score / 97.0f;
        env->log.score += transition->score;
        env->log.episode_return += transition->episode_return;
        env->log.episode_length += (float)transition->steps;
        env->log.terminated += (float)transition->terminated;
        env->log.truncated += (float)transition->truncated;
        env->log.n += 1.0f;
    }
    return 0;
}

void puf_init(Env* env, Dict* kwargs) {
    double max_steps = dict_get(kwargs, "max_steps");
    DictItem* agents = dict_find(kwargs, "num_agents");
    if (!(max_steps >= 1 && max_steps <= INT_MAX) ||
            max_steps != (double)(int)max_steps || (agents && agents->value != 1)) {
        fprintf(stderr, "decision_snake: max_steps must be a positive int32; num_agents must be 1\n");
        exit(1);
    }
    env->num_agents = 1;
    env->max_steps = (int)max_steps;
    env->episode_seed = (uint32_t)env->rng;
    env->next_seed = env->episode_seed;
    env->agents[0].policy = 0;
    env->snake = ib_snake_create(env->episode_seed, env->max_steps);
    if (env->snake == NULL) {
        fprintf(stderr, "decision_snake: could not allocate engine\n");
        exit(1);
    }
}

void puf_reset(Env* env) {
    if (decision_snake_reset(env, env->next_seed) != 0) {
        fprintf(stderr, "decision_snake: reset requires a valid engine and bound Agent buffers\n");
        exit(1);
    }
}

void puf_step(Env* env) {
    float action = env->agents[0].actions[0];
    if (!(action >= 0 && action < IB_SNAKE_ACTIONS) ||
            action != (float)(int)action || decision_snake_step(env, (int)action) != 0) {
        fprintf(stderr, "decision_snake: rejected action; the policy must respect the reverse-only mask\n");
        exit(1);
    }
    if (env->transition.terminated || env->transition.truncated) {
        // Preserve transition, reward and done; publish the new episode's board
        // and mask for the next policy decision. No transition is discarded.
        if (decision_snake_start_episode(env, env->next_seed) != 0) {
            fprintf(stderr, "decision_snake: autoreset failed\n");
            exit(1);
        }
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

// Define PUF_HEADLESS for standalone tests without a window system.
void puf_render(Env* env) {
#ifndef PUF_HEADLESS
    const int cell = 48;
    const int size = IB_SNAKE_GRID_SIZE * cell;
    if (!env->render_initialized) {
        InitWindow(size, size, "PufferLib Decision Snake");
        SetTargetFPS(PUF_STEPS_PER_SEC);
        env->render_initialized = 1;
    }
    int32_t board[IB_SNAKE_CELLS];
    ib_snake_observe(env->snake, board, IB_SNAKE_CELLS);
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    for (int i = 0; i < IB_SNAKE_CELLS; i++) {
        if (board[i] == 0) {
            continue;
        }
        Color color = board[i] == -1 ? (Color){255, 85, 85, 255} :
            board[i] == 1 ? (Color){241, 241, 241, 255} : (Color){0, 187, 187, 255};
        DrawRectangle((i % IB_SNAKE_GRID_SIZE) * cell + 1,
            (i / IB_SNAKE_GRID_SIZE) * cell + 1, cell - 2, cell - 2, color);
    }
    EndDrawing();
    puf_web_vsync();
#else
    (void)env;
#endif
}

void puf_close(Env* env) {
#ifndef PUF_HEADLESS
    if (env->render_initialized) {
        CloseWindow();
    }
#endif
    env->render_initialized = 0;
    ib_snake_destroy(env->snake);
    env->snake = NULL;
}

#endif
