/* Include the implementation for private fixtures; no state editing is public. */
#include "../g2048.c"

#include <assert.h>
#include <stdio.h>

static void near(double actual, double expected) {
    assert(fabs(actual - expected) < 1e-10);
}

static IBG2048 fixture(const unsigned char values[16], int max_steps) {
    IBG2048 env = {0};
    memcpy(env.grid, values, 16);
    env.rng = 42;
    env.max_steps = max_steps;
    return env;
}

static void test_directions_and_merge_order(void) {
    /* Each action reads toward its movement edge. [2,2,4,0] -> [4,4,0,0],
     * never [8,0,0,0]; each original tile participates in one merge. */
    for (int action = 0; action < 4; action++) {
        IBG2048 env = {0};
        const unsigned char input[4] = {1, 1, 2, 0};
        const unsigned char output[4] = {2, 2, 0, 0};
        for (int i = 0; i < 4; i++) {
            int index = action == 1 || action == 3 ? 3 - i : i;
            if (action < 2) env.grid[index][0] = input[i];
            else env.grid[0][index] = input[i];
        }
        double reward = 0;
        double score = 0;
        assert(move(&env, action, &reward, &score));
        near(reward, 0.05);
        near(score, 4);
        for (int i = 0; i < 4; i++) {
            int index = action == 1 || action == 3 ? 3 - i : i;
            assert((action < 2 ? env.grid[index][0] : env.grid[0][index]) == output[i]);
        }
    }
    unsigned char four_equal[4] = {1, 1, 1, 1};
    double reward = 0;
    double score = 0;
    assert(slide_and_merge(four_equal, &reward, &score));
    assert(memcmp(four_equal, (unsigned char[4]){2, 2, 0, 0}, 4) == 0);
    near(reward, 0.10);
    near(score, 8);

    unsigned char spaced[4] = {0, 2, 0, 2};
    reward = score = 0;
    assert(slide_and_merge(spaced, &reward, &score));
    assert(memcmp(spaced, (unsigned char[4]){3, 0, 0, 0}, 4) == 0);
    near(score, 8);
}

static void test_spawn_score_and_observation(void) {
    const unsigned char input[16] = {1, 1};
    IBG2048 env = fixture(input, 20);
    assert(ib_g2048_step(&env, 2) == 0);
    assert(env.grid[0][0] == 2);
    int occupied = 0;
    int sum = 0;
    int32_t board[16];
    assert(ib_g2048_observe(&env, board, 16) == 0);
    for (int i = 0; i < 16; i++) {
        assert(board[i] == 0 || board[i] == 2 || board[i] == 4);
        occupied += board[i] != 0;
        sum += board[i];
    }
    assert(occupied == 2);
    assert(sum == 6 || sum == 8);
    near(ib_g2048_score(&env), 4);
    near(ib_g2048_reward(&env), 0.05);
    near(ib_g2048_episode_return(&env), 0.05);
    assert(ib_g2048_steps(&env) == 1);
    assert(ib_g2048_outcome(&env) == 0);

    const unsigned char distinct[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    env = fixture(distinct, 20);
    assert(ib_g2048_observe(&env, board, 16) == 0);
    for (int i = 0; i < 16; i++) assert(board[i] == (int32_t)(UINT32_C(1) << (i + 1)));
}

static void test_noop_and_invalid_calls(void) {
    const unsigned char input[16] = {1};
    IBG2048 env = fixture(input, 20);
    IBG2048 snapshot = env;
    assert(ib_g2048_legal_actions(&env) == ((1u << 1) | (1u << 3)));
    assert(memcmp(&snapshot, &env, sizeof(env)) == 0);
    assert(ib_g2048_step(&env, -1) == -1);
    assert(ib_g2048_step(&env, 4) == -1);
    assert(ib_g2048_reset(&env, 10, 0) == -1);
    assert(ib_g2048_reset(&env, 10, -1) == -1);
    assert(memcmp(&snapshot, &env, sizeof(env)) == 0);
    int32_t output[16];
    for (int i = 0; i < 16; i++) output[i] = -9;
    assert(ib_g2048_observe(&env, output, 15) == -1);
    for (int i = 0; i < 16; i++) assert(output[i] == -9);
    assert(ib_g2048_observe(&env, NULL, 16) == -1);
    assert(ib_g2048_step(&env, 2) == 0);
    assert(memcmp(env.grid, snapshot.grid, 16) == 0);
    assert(env.rng == snapshot.rng);
    near(ib_g2048_reward(&env), -0.05);
    near(ib_g2048_episode_return(&env), -0.05);
    near(ib_g2048_score(&env), 0);
    assert(ib_g2048_steps(&env) == 1);

    assert(ib_g2048_create(1, 0) == NULL);
    assert(ib_g2048_reset(NULL, 1, 20) == -1);
    assert(ib_g2048_step(NULL, 0) == -1);
    assert(ib_g2048_observe(NULL, output, 16) == -1);
    assert(ib_g2048_legal_actions(NULL) == 0);
    assert(ib_g2048_steps(NULL) == 0);
    assert(ib_g2048_terminated(NULL) == 0);
    assert(ib_g2048_truncated(NULL) == 0);
    assert(ib_g2048_outcome(NULL) == 0);
    near(ib_g2048_reward(NULL), 0);
    near(ib_g2048_episode_return(NULL), 0);
    near(ib_g2048_score(NULL), 0);
    ib_g2048_destroy(NULL);
}

static void test_terminal_state_and_cap(void) {
    const unsigned char input[16] = {
        1, 1, 3, 4,
        4, 5, 6, 7,
        5, 6, 7, 8,
        6, 7, 8, 9
    };
    IBG2048 env = fixture(input, 1);
    assert(ib_g2048_step(&env, 2) == 0);
    assert(ib_g2048_terminated(&env) == 1);
    assert(ib_g2048_truncated(&env) == 0);
    assert(ib_g2048_outcome(&env) == -1);
    near(ib_g2048_reward(&env), -0.95);
    near(ib_g2048_episode_return(&env), -0.95);
    near(ib_g2048_score(&env), 4);
    assert(env.grid[0][0] == 2 && env.grid[0][1] == 3 && env.grid[0][2] == 4);
    assert(env.grid[0][3] == 1 || env.grid[0][3] == 2);
    assert(ib_g2048_legal_actions(&env) == 0);
    IBG2048 snapshot = env;
    assert(ib_g2048_step(&env, 0) == -1);
    assert(memcmp(&snapshot, &env, sizeof(env)) == 0);

    env = fixture((unsigned char[16]){1}, 1);
    assert(ib_g2048_step(&env, 2) == 0);
    assert(ib_g2048_terminated(&env) == 0);
    assert(ib_g2048_truncated(&env) == 1);
    assert(ib_g2048_outcome(&env) == 0);
    assert(ib_g2048_legal_actions(&env) == 0);
    assert(env.grid[0][0] == 1);
    snapshot = env;
    assert(ib_g2048_step(&env, 1) == -1);
    assert(memcmp(&snapshot, &env, sizeof(env)) == 0);
    assert(ib_g2048_reset(&env, 1234, 7) == 0);
    assert(!env.terminated && !env.truncated && env.steps == 0);
    near(env.reward, 0);
    near(env.score, 0);
    near(env.episode_return, 0);
}

static void test_large_tiles(void) {
    unsigned char line[4] = {6, 6, 0, 0};
    double reward = 0;
    double score = 0;
    assert(slide_and_merge(line, &reward, &score));
    near(reward, 0.08);
    near(score, 128);
    line[0] = line[1] = 16;
    reward = score = 0;
    assert(slide_and_merge(line, &reward, &score));
    near(reward, 0.05 + 36.48 * 0.03);
    near(score, 131072);
    line[0] = line[1] = 17;
    reward = score = 0;
    assert(slide_and_merge(line, &reward, &score));
    near(reward, 0.05 + pow(12, 1.5) * 0.03);
    near(score, 262144);

    line[0] = line[1] = 29;
    line[2] = line[3] = 30;
    reward = score = 0;
    assert(slide_and_merge(line, &reward, &score));
    assert(memcmp(line, (unsigned char[4]){30, 30, 30, 0}, 4) == 0);
    near(score, 1073741824.0);
    assert(isfinite(reward));
    reward = score = 0;
    assert(!slide_and_merge(line, &reward, &score));
    near(score, 0);

    unsigned char capped[16];
    memset(capped, 30, sizeof(capped));
    IBG2048 env = fixture(capped, 100);
    int32_t board[16];
    assert(ib_g2048_observe(&env, board, 16) == 0);
    for (int i = 0; i < 16; i++) assert(board[i] == INT32_C(1073741824));
    assert(ib_g2048_legal_actions(&env) == 0);
    assert(ib_g2048_step(&env, 0) == 0);
    assert(env.terminated);
}

static void test_determinism_and_interleaving(void) {
    /* Separate instances must not share PRNG state, even when resets interleave. */
    for (uint32_t seed = 0; seed < 40; seed++) {
        IBG2048 *first = ib_g2048_create(seed, 500);
        IBG2048 *second = ib_g2048_create(seed, 500);
        IBG2048 *noise = ib_g2048_create(seed + 1000, 10);
        assert(first && second && noise);
        assert(memcmp(first, second, sizeof(*first)) == 0);
        int occupied = 0;
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                assert(first->grid[row][col] <= 2);
                occupied += first->grid[row][col] != 0;
            }
        }
        assert(occupied == 2);
        for (int step = 0; step < 500; step++) {
            int action = (step * 7 + step / 9 + (int)seed) % 4;
            int status = ib_g2048_step(first, action);
            ib_g2048_reset(noise, (uint32_t)step, 10);
            ib_g2048_step(noise, (action + 1) % 4);
            assert(ib_g2048_step(second, action) == status);
            assert(memcmp(first, second, sizeof(*first)) == 0);
            assert(isfinite(first->episode_return));
            if (first->terminated || first->truncated) break;
        }
        assert(first->terminated || first->truncated);
        assert(ib_g2048_reset(first, seed, 500) == 0);
        assert(ib_g2048_reset(second, seed, 500) == 0);
        assert(memcmp(first, second, sizeof(*first)) == 0);
        ib_g2048_destroy(first);
        ib_g2048_destroy(second);
        ib_g2048_destroy(noise);
    }
}

int main(void) {
    test_directions_and_merge_order();
    test_spawn_score_and_observation();
    test_noop_and_invalid_calls();
    test_terminal_state_and_cap();
    test_large_tiles();
    test_determinism_and_interleaving();
    puts("g2048: all tests passed");
    return 0;
}
