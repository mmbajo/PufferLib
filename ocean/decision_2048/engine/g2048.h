#ifndef IB_G2048_H
#define IB_G2048_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct IBG2048 IBG2048;

/* Actions: 0 up, 1 down, 2 left, 3 right. max_steps must be positive. */
IBG2048 *ib_g2048_create(uint32_t seed, int max_steps);
void ib_g2048_destroy(IBG2048 *env);
/* Returns 0 on success; -1 for invalid arguments, without changing state. */
int ib_g2048_reset(IBG2048 *env, uint32_t seed, int max_steps);
/* In-range no-op moves are accepted and consume a step (reward -0.05).
 * Out-of-range actions and steps after episode completion return -1 unchanged. */
int ib_g2048_step(IBG2048 *env, int action);
/* Writes 16 actual tile values (0, 2, 4, ...) in top-row-first row-major order.
 * Returns 0 on success; -1 for NULL arguments or capacity < 16. */
int ib_g2048_observe(const IBG2048 *env, int32_t *board, int capacity);
/* Bit i is set iff action i changes the board; zero after completion or for NULL. */
uint32_t ib_g2048_legal_actions(const IBG2048 *env);
/* Getters return zero for NULL. outcome: 0 ongoing/truncated, -1 game over. */
int ib_g2048_steps(const IBG2048 *env);
int ib_g2048_terminated(const IBG2048 *env);
int ib_g2048_truncated(const IBG2048 *env);
int ib_g2048_outcome(const IBG2048 *env);
/* score is the sum of merged tile values. */
double ib_g2048_reward(const IBG2048 *env);
double ib_g2048_episode_return(const IBG2048 *env);
double ib_g2048_score(const IBG2048 *env);

#ifdef __cplusplus
}
#endif

#endif
