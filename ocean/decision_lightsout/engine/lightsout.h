#ifndef IB_LIGHTSOUT_H
#define IB_LIGHTSOUT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { IB_LIGHTSOUT_GRID_SIZE = 5, IB_LIGHTSOUT_CELLS = 25 };

typedef struct IBLightsOut IBLightsOut;

/* max_steps must be positive. The seed is applied before scrambling. */
IBLightsOut *ib_lightsout_create(uint32_t seed, int max_steps);
void ib_lightsout_destroy(IBLightsOut *env);
/* Return 0 on success, -1 on invalid input. Rejection leaves state unchanged. */
int ib_lightsout_reset(IBLightsOut *env, uint32_t seed, int max_steps);
int ib_lightsout_step(IBLightsOut *env, int action);
/* Top-row-first row-major 0/1 cells; capacity must be at least 25. */
int ib_lightsout_observe(const IBLightsOut *env, int32_t *board, int capacity);
/* Bits 0..24 are set while live; zero after termination/truncation or for NULL. */
uint32_t ib_lightsout_legal_actions(const IBLightsOut *env);

/* Getters return zero for NULL. outcome/score are 1 iff solved, otherwise 0. */
int ib_lightsout_steps(const IBLightsOut *env);
int ib_lightsout_terminated(const IBLightsOut *env);
int ib_lightsout_truncated(const IBLightsOut *env);
int ib_lightsout_outcome(const IBLightsOut *env);
double ib_lightsout_reward(const IBLightsOut *env);
double ib_lightsout_episode_return(const IBLightsOut *env);
double ib_lightsout_score(const IBLightsOut *env);

#ifdef __cplusplus
}
#endif

#endif
