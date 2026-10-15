/* Adapted from PufferLib; see SOURCE.md and LICENSE. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "connect4.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

struct IBConnect4 {
    uint64_t player_pieces;
    uint64_t env_pieces;
    unsigned int rng;
    int steps;
    int max_steps;
    int terminated;
    int truncated;
    int outcome;
    double reward;
};

/* Each column has six playable bits followed by an unused sentinel bit. */
static uint64_t top_mask(int column) {
    return UINT64_C(1) << (column * 7 + 5);
}

static uint64_t bottom_mask(int column) {
    return UINT64_C(1) << (column * 7);
}

static bool invalid_move(int column, uint64_t mask) {
    return column < 0 || column >= IB_CONNECT4_COLUMNS ||
        (mask & top_mask(column)) != 0;
}

static uint64_t play(int column, uint64_t mask, uint64_t other_pieces) {
    mask |= mask + bottom_mask(column);
    return other_pieces ^ mask;
}

static bool draw(uint64_t mask) {
    uint64_t full_board = 0;
    for (int column = 0; column < IB_CONNECT4_COLUMNS; column++) {
        full_board |= UINT64_C(63) << (column * 7);
    }
    /* Upstream compares against 4432406249472, which is not the 42-cell
     * playable mask. Check the actual playable bits (regression tested). */
    return (mask & full_board) == full_board;
}

static bool won(uint64_t pieces) {
    uint64_t m = pieces & (pieces >> 7);
    if (m & (m >> 14)) return true;
    m = pieces & (pieces >> 6);
    if (m & (m >> 12)) return true;
    m = pieces & (pieces >> 8);
    if (m & (m >> 16)) return true;
    m = pieces & (pieces >> 1);
    return (m & (m >> 2)) != 0;
}

/* Preserve the native opponent's evaluation and move order, including its
 * summed recursive evaluation (this is not a replacement optimal solver). */
static float negamax(uint64_t pieces, uint64_t other_pieces, int depth) {
    uint64_t piece_mask = pieces | other_pieces;
    if (won(other_pieces)) return (float)pow(10, depth);
    if (won(pieces)) return 0;
    if (depth == 0 || draw(piece_mask)) return 0;

    float value = 0;
    for (int column = 0; column < IB_CONNECT4_COLUMNS; column++) {
        if (invalid_move(column, piece_mask)) continue;
        uint64_t child_pieces = play(column, piece_mask, other_pieces);
        value -= negamax(other_pieces, child_pieces, depth - 1);
    }
    return value;
}

static int compute_env_move(IBConnect4 *env) {
    uint64_t piece_mask = env->player_pieces | env->env_pieces;
    uint64_t hash = env->player_pieces + piece_mask + (UINT64_C(1) << 42);

    switch (hash) {
        case UINT64_C(4398050705408): return 2;
        case UINT64_C(4398583382016): return 3;
    }

    float best_value = 9999;
    float values[IB_CONNECT4_COLUMNS];
    for (int column = 0; column < IB_CONNECT4_COLUMNS; column++) {
        values[column] = 9999;
    }
    for (int column = 0; column < IB_CONNECT4_COLUMNS; column++) {
        if (invalid_move(column, piece_mask)) continue;
        uint64_t child = play(column, piece_mask, env->player_pieces);
        if (won(child)) return column;
        float value = -negamax(env->player_pieces, child, 3);
        values[column] = value;
        if (value < best_value) best_value = value;
    }

    int num_ties = 0;
    for (int column = 0; column < IB_CONNECT4_COLUMNS; column++) {
        if (values[column] == best_value) num_ties++;
    }
    if (num_ties <= 0) return 0;
    int best_tie = (int)(rand_r(&env->rng) % (unsigned int)num_ties);
    for (int column = 0; column < IB_CONNECT4_COLUMNS; column++) {
        if (values[column] == best_value) {
            if (best_tie == 0) return column;
            best_tie--;
        }
    }
    return 0;
}

static void finish_game(IBConnect4 *env, int outcome) {
    env->terminated = 1;
    env->outcome = outcome;
    env->reward = outcome;
}

IBConnect4 *ib_connect4_create(uint32_t seed, int max_steps) {
    if (max_steps <= 0) return NULL;
    IBConnect4 *env = calloc(1, sizeof(*env));
    if (env != NULL) ib_connect4_reset(env, seed, max_steps);
    return env;
}

void ib_connect4_destroy(IBConnect4 *env) {
    free(env);
}

int ib_connect4_reset(IBConnect4 *env, uint32_t seed, int max_steps) {
    if (env == NULL || max_steps <= 0) return -1;
    *env = (IBConnect4){.rng = seed, .max_steps = max_steps};
    return 0;
}

int ib_connect4_step(IBConnect4 *env, int action) {
    if (env == NULL || env->terminated || env->truncated) return -1;
    uint64_t piece_mask = env->player_pieces | env->env_pieces;
    if (invalid_move(action, piece_mask)) return -1;

    env->steps++;
    env->reward = 0;
    env->player_pieces = play(action, piece_mask, env->env_pieces);
    if (won(env->player_pieces)) {
        finish_game(env, 1);
        return 0;
    }
    piece_mask = env->player_pieces | env->env_pieces;
    if (draw(piece_mask)) {
        finish_game(env, 0);
        return 0;
    }

    int opponent_action = compute_env_move(env);
    if (invalid_move(opponent_action, piece_mask)) {
        /* Preserve upstream's result if its native opponent cannot move. */
        finish_game(env, 1);
        return 0;
    }
    env->env_pieces = play(opponent_action, piece_mask, env->player_pieces);
    if (won(env->env_pieces)) {
        finish_game(env, -1);
        return 0;
    }
    if (draw(env->player_pieces | env->env_pieces)) {
        finish_game(env, 0);
        return 0;
    }
    if (env->steps >= env->max_steps) env->truncated = 1;
    return 0;
}

int ib_connect4_observe(const IBConnect4 *env, int32_t *board, int capacity) {
    if (env == NULL || board == NULL || capacity < IB_CONNECT4_CELLS) return -1;
    for (int row = 0; row < IB_CONNECT4_ROWS; row++) {
        for (int column = 0; column < IB_CONNECT4_COLUMNS; column++) {
            uint64_t bit = UINT64_C(1) << (column * 7 + 5 - row);
            board[row * IB_CONNECT4_COLUMNS + column] =
                (env->player_pieces & bit) ? 1 : ((env->env_pieces & bit) ? -1 : 0);
        }
    }
    return 0;
}

uint32_t ib_connect4_legal_actions(const IBConnect4 *env) {
    if (env == NULL || env->terminated || env->truncated) return 0;
    uint32_t actions = 0;
    uint64_t mask = env->player_pieces | env->env_pieces;
    for (int column = 0; column < IB_CONNECT4_COLUMNS; column++) {
        if (!invalid_move(column, mask)) actions |= UINT32_C(1) << column;
    }
    return actions;
}

int ib_connect4_steps(const IBConnect4 *env) { return env ? env->steps : 0; }
int ib_connect4_terminated(const IBConnect4 *env) { return env ? env->terminated : 0; }
int ib_connect4_truncated(const IBConnect4 *env) { return env ? env->truncated : 0; }
int ib_connect4_outcome(const IBConnect4 *env) { return env ? env->outcome : 0; }
double ib_connect4_reward(const IBConnect4 *env) { return env ? env->reward : 0; }
double ib_connect4_episode_return(const IBConnect4 *env) { return env ? env->outcome : 0; }
double ib_connect4_score(const IBConnect4 *env) { return env ? env->outcome : 0; }
