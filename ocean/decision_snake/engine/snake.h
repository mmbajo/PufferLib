#ifndef IB_SNAKE_H
#define IB_SNAKE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { IB_SNAKE_GRID_SIZE = 10, IB_SNAKE_CELLS = 100, IB_SNAKE_ACTIONS = 4 };

typedef struct IBSnake IBSnake;

/* max_steps must be positive. Each instance owns its seeded RNG. */
IBSnake *ib_snake_create(uint32_t seed, int max_steps);
void ib_snake_destroy(IBSnake *env);
/* Return 0 on success, -1 on invalid input. Rejection leaves state unchanged. */
int ib_snake_reset(IBSnake *env, uint32_t seed, int max_steps);
/* Actions: 0 up, 1 down, 2 left, 3 right. Reversing into the neck is invalid. */
int ib_snake_step(IBSnake *env, int action);
/* Top-row-first row-major cells: -1 food, 0 empty, 1 head, 2 neck, ... tail. */
int ib_snake_observe(const IBSnake *env, int32_t *board, int capacity);
/* Direction bits, excluding reverse; fatal directions remain legal. */
uint32_t ib_snake_legal_actions(const IBSnake *env);

/* Getters return zero for NULL. outcome: 1 full board, -1 collision, 0 otherwise. */
int ib_snake_steps(const IBSnake *env);
int ib_snake_terminated(const IBSnake *env);
int ib_snake_truncated(const IBSnake *env);
int ib_snake_outcome(const IBSnake *env);
double ib_snake_reward(const IBSnake *env);
double ib_snake_episode_return(const IBSnake *env);
/* Number of food items eaten, independent of the death penalty in return. */
double ib_snake_score(const IBSnake *env);

/* Version 1 snapshot: little-endian integers/IEEE binary64, explicit lengths
 * and CRC32. No pointers/padding are serialized. Exact-sized buffers required.
 * Save/load return 0 on success, -1 on invalid data or unsupported float ABI.
 * Rejected loads leave the destination unchanged. Unused body slots are zeroed. */
enum { IB_SNAKE_STATE_BYTES = 476 };
size_t ib_snake_state_size(void);
int ib_snake_state_save(const IBSnake *env, void *output, size_t bytes);
int ib_snake_state_load(IBSnake *env, const void *input, size_t bytes);

#ifdef __cplusplus
}
#endif

#endif
