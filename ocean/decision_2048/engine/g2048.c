/* Adapted from PufferAI/PufferLib, MIT licensed. See LICENSE and SOURCE.md. */
#define _POSIX_C_SOURCE 200809L
#include "g2048.h"

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define SIDE 4
#define CELLS 16
#define MAX_EXPONENT 30

struct IBG2048 {
    unsigned char grid[SIDE][SIDE];
    unsigned int rng;
    int steps;
    int max_steps;
    int terminated;
    int truncated;
    double reward;
    double episode_return;
    double score;
};

/* Preserve upstream's rounded reward table for resulting tiles 128..131072. */
static const double pow15_table[12] = {
    0.0, 1.0, 2.83, 5.20, 8.0, 11.18, 14.70, 18.52, 22.63, 27.0, 31.62, 36.48
};

_Static_assert(UINT_MAX == UINT32_MAX, "2048 requires a 32-bit unsigned int PRNG state");

/* Preserve upstream's rand_r generator, with state private to each episode. */
static uint32_t random_u32(IBG2048 *env) {
    return (uint32_t)rand_r(&env->rng);
}

static void spawn_tile(IBG2048 *env) {
    int empty_count = 0;
    for (int row = 0; row < SIDE; row++) {
        for (int col = 0; col < SIDE; col++) {
            empty_count += env->grid[row][col] == 0;
        }
    }
    if (empty_count == 0) return;

    unsigned char tile = random_u32(env) % 10 == 0 ? 2 : 1;
    int target = (int)(random_u32(env) % (uint32_t)empty_count);
    for (int row = 0; row < SIDE; row++) {
        for (int col = 0; col < SIDE; col++) {
            if (env->grid[row][col] == 0 && target-- == 0) {
                env->grid[row][col] = tile;
                return;
            }
        }
    }
}

/* Upstream slide/merge order: compact, merge adjacent equals once, compact.
 * A newly merged tile cannot merge a second time within the same action. */
static bool slide_and_merge(unsigned char row[SIDE], double *reward, double *score) {
    bool moved = false;
    int write_pos = 0;
    for (int read_pos = 0; read_pos < SIDE; read_pos++) {
        if (row[read_pos] != 0) {
            if (write_pos != read_pos) {
                row[write_pos] = row[read_pos];
                row[read_pos] = 0;
                moved = true;
            }
            write_pos++;
        }
    }
    for (int i = 0; i < SIDE - 1; i++) {
        if (row[i] != 0 && row[i] == row[i + 1] && row[i] < MAX_EXPONENT) {
            row[i]++;
            *reward += 0.05;
            if (row[i] > 6) {
                int index = row[i] - 6;
                double bonus = index < 12 ? pow15_table[index] : pow((double)index, 1.5);
                *reward += bonus * 0.03;
            }
            *score += (double)(UINT32_C(1) << row[i]);
            for (int j = i + 1; j < SIDE - 1; j++) row[j] = row[j + 1];
            row[SIDE - 1] = 0;
            moved = true;
        }
    }
    return moved;
}

static bool move(IBG2048 *env, int action, double *reward, double *score) {
    bool moved = false;
    unsigned char line[SIDE];
    for (int outer = 0; outer < SIDE; outer++) {
        for (int i = 0; i < SIDE; i++) {
            int index = action == 1 || action == 3 ? SIDE - 1 - i : i;
            line[i] = action < 2 ? env->grid[index][outer] : env->grid[outer][index];
        }
        if (slide_and_merge(line, reward, score)) {
            moved = true;
            for (int i = 0; i < SIDE; i++) {
                int index = action == 1 || action == 3 ? SIDE - 1 - i : i;
                if (action < 2) env->grid[index][outer] = line[i];
                else env->grid[outer][index] = line[i];
            }
        }
    }
    return moved;
}

static uint32_t legal_mask(const IBG2048 *env) {
    uint32_t mask = 0;
    for (int action = 0; action < 4; action++) {
        IBG2048 trial = *env;
        double reward = 0;
        double score = 0;
        if (move(&trial, action, &reward, &score)) mask |= UINT32_C(1) << action;
    }
    return mask;
}

IBG2048 *ib_g2048_create(uint32_t seed, int max_steps) {
    if (max_steps <= 0) return NULL;
    IBG2048 *env = calloc(1, sizeof(*env));
    if (env != NULL) ib_g2048_reset(env, seed, max_steps);
    return env;
}

void ib_g2048_destroy(IBG2048 *env) {
    free(env);
}

int ib_g2048_reset(IBG2048 *env, uint32_t seed, int max_steps) {
    if (env == NULL || max_steps <= 0) return -1;
    memset(env, 0, sizeof(*env));
    env->rng = seed;
    env->max_steps = max_steps;
    spawn_tile(env);
    spawn_tile(env);
    return 0;
}

int ib_g2048_step(IBG2048 *env, int action) {
    if (env == NULL || action < 0 || action > 3 || env->terminated || env->truncated) return -1;
    double reward = 0;
    double score = 0;
    if (move(env, action, &reward, &score)) {
        spawn_tile(env);
        env->score += score;
    } else {
        reward = -0.05;
    }
    env->steps++;
    env->terminated = legal_mask(env) == 0;
    /* Natural game over wins when it coincides with the explicit step cap. */
    env->truncated = !env->terminated && env->steps >= env->max_steps;
    if (env->terminated) reward -= 1.0;
    env->reward = reward;
    env->episode_return += reward;
    return 0;
}

int ib_g2048_observe(const IBG2048 *env, int32_t *board, int capacity) {
    if (env == NULL || board == NULL || capacity < CELLS) return -1;
    for (int row = 0; row < SIDE; row++) {
        for (int col = 0; col < SIDE; col++) {
            unsigned char exponent = env->grid[row][col];
            board[row * SIDE + col] = exponent == 0 ? 0 : (int32_t)(UINT32_C(1) << exponent);
        }
    }
    return 0;
}

uint32_t ib_g2048_legal_actions(const IBG2048 *env) {
    if (env == NULL || env->terminated || env->truncated) return 0;
    return legal_mask(env);
}

int ib_g2048_steps(const IBG2048 *env) { return env == NULL ? 0 : env->steps; }
int ib_g2048_terminated(const IBG2048 *env) { return env == NULL ? 0 : env->terminated; }
int ib_g2048_truncated(const IBG2048 *env) { return env == NULL ? 0 : env->truncated; }
int ib_g2048_outcome(const IBG2048 *env) { return env != NULL && env->terminated ? -1 : 0; }
double ib_g2048_reward(const IBG2048 *env) { return env == NULL ? 0 : env->reward; }
double ib_g2048_episode_return(const IBG2048 *env) { return env == NULL ? 0 : env->episode_return; }
double ib_g2048_score(const IBG2048 *env) { return env == NULL ? 0 : env->score; }
