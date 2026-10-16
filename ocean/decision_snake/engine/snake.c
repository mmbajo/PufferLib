/* Adapted from PufferLib Snake; see SOURCE.md and LICENSE. */
#include "snake.h"

#include <stdlib.h>
#include <string.h>

struct IBSnake {
    int snake[IB_SNAKE_CELLS]; /* Head first; unused entries have no meaning. */
    int length;
    int food;
    uint32_t rng;
    int max_steps;
    int steps;
    int terminated;
    int truncated;
    int outcome;
    int score;
    double reward;
    double episode_return;
};

static uint32_t next_random(IBSnake *env) {
    env->rng = env->rng * UINT32_C(1664525) + UINT32_C(1013904223);
    return env->rng;
}

/* Select among empty cells without retrying occupied cells or a full board. */
static void spawn_food(IBSnake *env) {
    unsigned char occupied[IB_SNAKE_CELLS] = {0};
    for (int i = 0; i < env->length; ++i) occupied[env->snake[i]] = 1;
    int empty_count = IB_SNAKE_CELLS - env->length;
    env->food = -1;
    if (empty_count == 0) return;
    int choice = (int)(next_random(env) % (uint32_t)empty_count);
    for (int i = 0; i < IB_SNAKE_CELLS; ++i) {
        if (!occupied[i] && choice-- == 0) {
            env->food = i;
            return;
        }
    }
}

static int next_position(const IBSnake *env, int action) {
    static const int dr[IB_SNAKE_ACTIONS] = {-1, 1, 0, 0};
    static const int dc[IB_SNAKE_ACTIONS] = {0, 0, -1, 1};
    int row = env->snake[0] / IB_SNAKE_GRID_SIZE + dr[action];
    int col = env->snake[0] % IB_SNAKE_GRID_SIZE + dc[action];
    if (row < 0 || row >= IB_SNAKE_GRID_SIZE ||
        col < 0 || col >= IB_SNAKE_GRID_SIZE) return -1;
    return row * IB_SNAKE_GRID_SIZE + col;
}

int ib_snake_reset(IBSnake *env, uint32_t seed, int max_steps) {
    if (env == NULL || max_steps <= 0) return -1;
    IBSnake fresh = {0};
    fresh.rng = seed;
    fresh.max_steps = max_steps;
    fresh.length = 3;
    fresh.snake[0] = 55;
    fresh.snake[1] = 54;
    fresh.snake[2] = 53;
    spawn_food(&fresh);
    *env = fresh;
    return 0;
}

IBSnake *ib_snake_create(uint32_t seed, int max_steps) {
    if (max_steps <= 0) return NULL;
    IBSnake *env = malloc(sizeof(*env));
    if (env == NULL) return NULL;
    if (ib_snake_reset(env, seed, max_steps) != 0) {
        free(env);
        return NULL;
    }
    return env;
}

void ib_snake_destroy(IBSnake *env) {
    free(env);
}

int ib_snake_step(IBSnake *env, int action) {
    if (env == NULL || action < 0 || action >= IB_SNAKE_ACTIONS ||
        env->terminated || env->truncated) return -1;
    int next = next_position(env, action);
    if (env->length > 1 && next == env->snake[1]) return -1;

    int grow = next >= 0 && next == env->food;
    int collision = next < 0;
    /* The old tail departs on this tick unless eating food grows the snake. */
    for (int i = 0; i < env->length - !grow; ++i) {
        if (env->snake[i] == next) collision = 1;
    }
    env->steps += 1;
    env->reward = 0;
    if (collision) {
        env->reward = -1;
        env->terminated = 1;
        env->outcome = -1;
    } else {
        int new_length = env->length + grow;
        memmove(env->snake + 1, env->snake,
                (size_t)(new_length - 1) * sizeof(env->snake[0]));
        env->snake[0] = next;
        env->length = new_length;
        if (grow) {
            env->score += 1;
            env->reward = 1;
            spawn_food(env);
            if (env->length == IB_SNAKE_CELLS) {
                env->terminated = 1;
                env->outcome = 1;
            }
        }
    }
    if (!env->terminated && env->steps >= env->max_steps) env->truncated = 1;
    env->episode_return += env->reward;
    return 0;
}

int ib_snake_observe(const IBSnake *env, int32_t *board, int capacity) {
    if (env == NULL || board == NULL || capacity < IB_SNAKE_CELLS) return -1;
    for (int i = 0; i < IB_SNAKE_CELLS; ++i) board[i] = 0;
    if (env->food >= 0) board[env->food] = -1;
    for (int i = 0; i < env->length; ++i) board[env->snake[i]] = i + 1;
    return 0;
}

uint32_t ib_snake_legal_actions(const IBSnake *env) {
    if (env == NULL || env->terminated || env->truncated) return 0;
    uint32_t mask = 0;
    for (int action = 0; action < IB_SNAKE_ACTIONS; ++action) {
        if (env->length < 2 || next_position(env, action) != env->snake[1])
            mask |= UINT32_C(1) << action;
    }
    return mask;
}

int ib_snake_steps(const IBSnake *env) { return env ? env->steps : 0; }
int ib_snake_terminated(const IBSnake *env) { return env ? env->terminated : 0; }
int ib_snake_truncated(const IBSnake *env) { return env ? env->truncated : 0; }
int ib_snake_outcome(const IBSnake *env) { return env ? env->outcome : 0; }
double ib_snake_reward(const IBSnake *env) { return env ? env->reward : 0; }
double ib_snake_episode_return(const IBSnake *env) {
    return env ? env->episode_return : 0;
}
double ib_snake_score(const IBSnake *env) { return env ? env->score : 0; }
