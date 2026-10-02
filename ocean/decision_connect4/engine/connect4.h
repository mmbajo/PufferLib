#ifndef IB_CONNECT4_H
#define IB_CONNECT4_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IB_CONNECT4_ROWS 6
#define IB_CONNECT4_COLUMNS 7
#define IB_CONNECT4_CELLS 42

typedef struct IBConnect4 IBConnect4;

/* max_steps must be positive. A step is one player move and, unless the game
 * ends first, one native-bot reply. create returns NULL on invalid arguments
 * or allocation failure. reset returns -1 on invalid arguments, unchanged. */
IBConnect4 *ib_connect4_create(uint32_t seed, int max_steps);
void ib_connect4_destroy(IBConnect4 *env);
int ib_connect4_reset(IBConnect4 *env, uint32_t seed, int max_steps);

/* Columns are 0..6 from left to right. Return 0 for an accepted action, -1
 * for a NULL/finished environment, out-of-range action, or full column.
 * Rejection leaves all state, including RNG and last reward, unchanged.
 * Natural termination takes precedence over a step cap. No automatic reset. */
int ib_connect4_step(IBConnect4 *env, int action);

/* Write 42 top-row-first, row-major cells: 0 empty, 1 player, -1 opponent.
 * Capacity >= 42 is required; on error return -1 without writing. */
int ib_connect4_observe(const IBConnect4 *env, int32_t *board, int capacity);
/* Bit c indicates playable column c; zero after termination/truncation. */
uint32_t ib_connect4_legal_actions(const IBConnect4 *env);

/* NULL getters return zero. outcome: 1 player win, -1 opponent win,
 * 0 draw/ongoing/truncated. Use flags to distinguish these cases.
 * score and episode_return equal the terminal result; reward is the result
 * of the last accepted step. Truncation has zero reward and score. */
int ib_connect4_steps(const IBConnect4 *env);
int ib_connect4_terminated(const IBConnect4 *env);
int ib_connect4_truncated(const IBConnect4 *env);
int ib_connect4_outcome(const IBConnect4 *env);
double ib_connect4_reward(const IBConnect4 *env);
double ib_connect4_episode_return(const IBConnect4 *env);
double ib_connect4_score(const IBConnect4 *env);

#ifdef __cplusplus
}
#endif
#endif
