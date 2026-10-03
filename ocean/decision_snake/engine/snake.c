/* Adapted from PufferLib Snake; see SOURCE.md and LICENSE. */
#include "snake.h"

#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <math.h>

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

/* Snapshot support is independent of the transition/RNG implementation above. */
static void state_put32(unsigned char *p, uint32_t value) {
    for (int i = 0; i < 4; ++i) p[i] = (unsigned char)(value >> (8 * i));
}
static uint32_t state_get32(const unsigned char *p) {
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) value |= (uint32_t)p[i] << (8 * i);
    return value;
}
static void state_put64(unsigned char *p, double value) {
    uint64_t bits; memcpy(&bits, &value, sizeof(bits));
    for (int i = 0; i < 8; ++i) p[i] = (unsigned char)(bits >> (8 * i));
}
static double state_get64(const unsigned char *p) {
    uint64_t bits = 0; double value;
    for (int i = 0; i < 8; ++i) bits |= (uint64_t)p[i] << (8 * i);
    memcpy(&value, &bits, sizeof(value)); return value;
}
static uint32_t state_crc32(const unsigned char *p, size_t bytes) {
    uint32_t crc = UINT32_MAX;
    for (size_t i = 0; i < bytes; ++i) {
        crc ^= p[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ ((crc & 1) ? UINT32_C(0xedb88320) : 0);
    }
    return ~crc;
}
static int state_valid(const IBSnake *env) {
    if (!env || sizeof(double) != 8 || DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024 ||
            env->max_steps < 1 || env->steps < 0 || env->steps > env->max_steps ||
            env->length < 3 || env->length > IB_SNAKE_CELLS ||
            env->score != env->length - 3 || env->score > env->steps ||
            (env->terminated != 0 && env->terminated != 1) ||
            (env->truncated != 0 && env->truncated != 1) ||
            (env->terminated && env->truncated) ||
            !isfinite(env->reward) || !isfinite(env->episode_return)) return 0;
    int collision = env->outcome == -1, full = env->length == IB_SNAKE_CELLS;
    if (env->outcome < -1 || env->outcome > 1 ||
            env->terminated != (collision || full) || (env->outcome == 1) != full ||
            env->truncated != (!env->terminated && env->steps == env->max_steps) ||
            env->episode_return != env->score - collision ||
            (collision && (env->steps == 0 || env->score >= env->steps || env->reward != -1)) ||
            (!collision && env->reward != 0 && env->reward != 1) ||
            (env->reward == 1 && env->score == 0) || (full && env->reward != 1)) return 0;
    unsigned char occupied[IB_SNAKE_CELLS] = {0};
    for (int i = 0; i < env->length; ++i) {
        int cell = env->snake[i];
        if (cell < 0 || cell >= IB_SNAKE_CELLS || occupied[cell]) return 0;
        if (i && abs(cell / 10 - env->snake[i-1] / 10) +
                abs(cell % 10 - env->snake[i-1] % 10) != 1) return 0;
        occupied[cell] = 1;
    }
    if (full ? env->food != -1 : (env->food < 0 || env->food >= IB_SNAKE_CELLS || occupied[env->food])) return 0;
    if (env->steps == 0 && (env->length != 3 || env->reward != 0 ||
            env->snake[0] != 55 || env->snake[1] != 54 || env->snake[2] != 53)) return 0;
    return 1;
}

size_t ib_snake_state_size(void) { return IB_SNAKE_STATE_BYTES; }

int ib_snake_state_save(const IBSnake *env, void *output, size_t bytes) {
    if (!output || bytes != IB_SNAKE_STATE_BYTES || !state_valid(env)) return -1;
    unsigned char data[IB_SNAKE_STATE_BYTES] = {0};
    memcpy(data, "IBSNAK01", 8);
    state_put32(data + 8, 1); state_put32(data + 12, IB_SNAKE_STATE_BYTES);
    state_put32(data + 16, IB_SNAKE_GRID_SIZE);
    state_put32(data + 20, (uint32_t)env->max_steps);
    state_put32(data + 24, (uint32_t)env->steps); state_put32(data + 28, env->rng);
    state_put32(data + 32, (uint32_t)env->length);
    state_put32(data + 36, env->food < 0 ? UINT32_MAX : (uint32_t)env->food);
    state_put32(data + 40, (uint32_t)env->terminated);
    state_put32(data + 44, (uint32_t)env->truncated);
    state_put32(data + 48, env->outcome < 0 ? UINT32_MAX : (uint32_t)env->outcome);
    state_put32(data + 52, (uint32_t)env->score);
    state_put64(data + 56, env->reward); state_put64(data + 64, env->episode_return);
    for (int i = 0; i < IB_SNAKE_CELLS; ++i)
        state_put32(data + 72 + 4*i, i < env->length ? (uint32_t)env->snake[i] : UINT32_MAX);
    state_put32(data + 472, state_crc32(data, 472));
    memcpy(output, data, sizeof(data)); return 0;
}

int ib_snake_state_load(IBSnake *env, const void *input, size_t bytes) {
    if (!env || !input || bytes != IB_SNAKE_STATE_BYTES || sizeof(double) != 8 ||
            DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024) return -1;
    const unsigned char *data = (const unsigned char *)input;
    if (memcmp(data, "IBSNAK01", 8) || state_get32(data+8) != 1 ||
            state_get32(data+12) != bytes || state_get32(data+16) != IB_SNAKE_GRID_SIZE ||
            state_get32(data+472) != state_crc32(data, 472)) return -1;
    IBSnake next = {0};
    uint32_t max_steps = state_get32(data+20), steps = state_get32(data+24);
    uint32_t length = state_get32(data+32), food = state_get32(data+36);
    uint32_t terminated = state_get32(data+40), truncated = state_get32(data+44);
    uint32_t outcome = state_get32(data+48), score = state_get32(data+52);
    if (max_steps < 1 || max_steps > INT_MAX || steps > max_steps || length < 3 ||
            length > IB_SNAKE_CELLS || (food != UINT32_MAX && food >= IB_SNAKE_CELLS) ||
            terminated > 1 || truncated > 1 || (outcome != UINT32_MAX && outcome > 1) || score > 97) return -1;
    next.max_steps = (int)max_steps; next.steps = (int)steps; next.rng = state_get32(data+28);
    next.length = (int)length; next.food = food == UINT32_MAX ? -1 : (int)food;
    next.terminated = (int)terminated; next.truncated = (int)truncated;
    next.outcome = outcome == UINT32_MAX ? -1 : (int)outcome; next.score = (int)score;
    next.reward = state_get64(data+56); next.episode_return = state_get64(data+64);
    for (int i = 0; i < IB_SNAKE_CELLS; ++i) {
        uint32_t cell = state_get32(data+72+4*i);
        if (i >= next.length) { if (cell != UINT32_MAX) return -1; }
        else { if (cell >= IB_SNAKE_CELLS) return -1; next.snake[i] = (int)cell; }
    }
    if (!state_valid(&next)) return -1;
    *env = next; return 0;
}
