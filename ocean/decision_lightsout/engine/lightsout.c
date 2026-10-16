/* Adapted from PufferLib Lights Out; see SOURCE.md and LICENSE. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "lightsout.h"

#include <stdlib.h>
#include <string.h>

struct IBLightsOut {
    unsigned char grid[IB_LIGHTSOUT_CELLS];
    unsigned int rng;
    int max_steps;
    int steps;
    int lights_on;
    int prev_action;
    int last_action;
    int terminated;
    int truncated;
    float reward;
    float episode_return;
};

static void toggle(IBLightsOut *env, int action) {
    static const int dirs[5][2] = {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    int row = action / IB_LIGHTSOUT_GRID_SIZE;
    int col = action % IB_LIGHTSOUT_GRID_SIZE;
    for (int i = 0; i < 5; ++i) {
        int r = row + dirs[i][0];
        int c = col + dirs[i][1];
        if (r >= 0 && r < IB_LIGHTSOUT_GRID_SIZE &&
            c >= 0 && c < IB_LIGHTSOUT_GRID_SIZE) {
            int offset = r * IB_LIGHTSOUT_GRID_SIZE + c;
            unsigned char old = env->grid[offset];
            env->grid[offset] = (unsigned char)!old;
            env->lights_on += old ? -1 : 1;
        }
    }
}

int ib_lightsout_reset(IBLightsOut *env, uint32_t seed, int max_steps) {
    if (env == NULL || max_steps <= 0) return -1;

    IBLightsOut fresh = {0};
    fresh.rng = (unsigned int)seed;
    fresh.max_steps = max_steps;
    fresh.prev_action = -1;
    fresh.last_action = -1;
    /* Presses commute and undo themselves, so every scramble is solvable. */
    do {
        memset(fresh.grid, 0, sizeof(fresh.grid));
        fresh.lights_on = 0;
        for (int i = 0; i < IB_LIGHTSOUT_CELLS; ++i) {
            float u = (float)rand_r(&fresh.rng) / (float)RAND_MAX;
            if (u < 0.15f) toggle(&fresh, i);
        }
        /* Exclude trivial starts by drawing the next full scramble. */
    } while (fresh.lights_on == 0);
    *env = fresh;
    return 0;
}

IBLightsOut *ib_lightsout_create(uint32_t seed, int max_steps) {
    if (max_steps <= 0) return NULL;
    IBLightsOut *env = malloc(sizeof(*env));
    if (env == NULL) return NULL;
    if (ib_lightsout_reset(env, seed, max_steps) != 0) {
        free(env);
        return NULL;
    }
    return env;
}

void ib_lightsout_destroy(IBLightsOut *env) {
    free(env);
}

int ib_lightsout_step(IBLightsOut *env, int action) {
    if (env == NULL || action < 0 || action >= IB_LIGHTSOUT_CELLS ||
        env->terminated || env->truncated) return -1;

    /* Keep the upstream float reward and accumulation arithmetic. */
    float reward = -0.02 * (36.0 / IB_LIGHTSOUT_CELLS);
    int prev_on = env->lights_on;
    if (action == env->last_action) {
        reward -= 0.03f;
    } else if (action == env->prev_action) {
        reward -= 0.02f;
    }
    toggle(env, action);
    env->prev_action = env->last_action;
    env->last_action = action;
    reward += 0.005f * (float)(prev_on - env->lights_on);
    env->steps += 1;

    if (env->lights_on == 0) {
        reward = 2.0f;
        env->terminated = 1;
    } else if (env->steps >= env->max_steps) {
        reward -= 0.5f;
        env->truncated = 1;
    }
    env->reward = reward;
    env->episode_return += reward;
    return 0;
}

int ib_lightsout_observe(const IBLightsOut *env, int32_t *board, int capacity) {
    if (env == NULL || board == NULL || capacity < IB_LIGHTSOUT_CELLS) return -1;
    for (int i = 0; i < IB_LIGHTSOUT_CELLS; ++i) board[i] = env->grid[i];
    return 0;
}

uint32_t ib_lightsout_legal_actions(const IBLightsOut *env) {
    if (env == NULL || env->terminated || env->truncated) return 0;
    return (UINT32_C(1) << IB_LIGHTSOUT_CELLS) - UINT32_C(1);
}

int ib_lightsout_steps(const IBLightsOut *env) { return env ? env->steps : 0; }
int ib_lightsout_terminated(const IBLightsOut *env) { return env ? env->terminated : 0; }
int ib_lightsout_truncated(const IBLightsOut *env) { return env ? env->truncated : 0; }
int ib_lightsout_outcome(const IBLightsOut *env) { return env ? env->terminated : 0; }
double ib_lightsout_reward(const IBLightsOut *env) { return env ? env->reward : 0; }
double ib_lightsout_episode_return(const IBLightsOut *env) {
    return env ? env->episode_return : 0;
}
double ib_lightsout_score(const IBLightsOut *env) { return env ? env->terminated : 0; }
