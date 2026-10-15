/* Internal fixtures stay in this translation unit, outside the public API. */
#include "../lightsout.c"

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>

static void fixture(IBLightsOut *env, const int32_t board[IB_LIGHTSOUT_CELLS], int cap) {
    memset(env, 0, sizeof(*env));
    env->max_steps = cap;
    env->last_action = -1;
    env->prev_action = -1;
    for (int i = 0; i < IB_LIGHTSOUT_CELLS; ++i) {
        assert(board[i] == 0 || board[i] == 1);
        env->grid[i] = (unsigned char)board[i];
        env->lights_on += board[i];
    }
}

static void assert_close(double actual, double expected) {
    assert(fabs(actual - expected) < 1e-6);
}

static void assert_same(const IBLightsOut *a, const IBLightsOut *b) {
    int32_t ba[IB_LIGHTSOUT_CELLS], bb[IB_LIGHTSOUT_CELLS];
    assert(ib_lightsout_observe(a, ba, IB_LIGHTSOUT_CELLS) == 0);
    assert(ib_lightsout_observe(b, bb, IB_LIGHTSOUT_CELLS) == 0);
    assert(memcmp(ba, bb, sizeof(ba)) == 0);
    assert(a->rng == b->rng);
    assert(a->lights_on == b->lights_on);
    assert(a->last_action == b->last_action && a->prev_action == b->prev_action);
    assert(ib_lightsout_steps(a) == ib_lightsout_steps(b));
    assert(ib_lightsout_terminated(a) == ib_lightsout_terminated(b));
    assert(ib_lightsout_truncated(a) == ib_lightsout_truncated(b));
    assert(ib_lightsout_reward(a) == ib_lightsout_reward(b));
    assert(ib_lightsout_episode_return(a) == ib_lightsout_episode_return(b));
}

static void test_toggle_and_shaping(void) {
    IBLightsOut env;
    int32_t full[25], board[25];
    for (int i = 0; i < 25; ++i) full[i] = 1;

    fixture(&env, full, 20);
    assert(ib_lightsout_legal_actions(&env) == UINT32_C(0x1ffffff));
    assert(ib_lightsout_step(&env, 0) == 0);
    assert(ib_lightsout_observe(&env, board, 25) == 0);
    for (int i = 0; i < 25; ++i) assert(board[i] == (i == 0 || i == 1 || i == 5 ? 0 : 1));
    assert_close(ib_lightsout_reward(&env), -0.0288 + 0.015);
    assert(ib_lightsout_step(&env, 0) == 0);
    assert(ib_lightsout_observe(&env, board, 25) == 0);
    assert(memcmp(board, full, sizeof(full)) == 0);
    assert_close(ib_lightsout_reward(&env), -0.0288 - 0.03 - 0.015);
    assert_close(ib_lightsout_episode_return(&env), -0.0876);

    fixture(&env, full, 20);
    assert(ib_lightsout_step(&env, 12) == 0);
    assert(ib_lightsout_observe(&env, board, 25) == 0);
    for (int i = 0; i < 25; ++i) {
        int changed = i == 7 || i == 11 || i == 12 || i == 13 || i == 17;
        assert(board[i] == (changed ? 0 : 1));
    }
    assert_close(ib_lightsout_reward(&env), -0.0288 + 0.025);

    fixture(&env, full, 20);
    assert(ib_lightsout_step(&env, 0) == 0);
    assert(ib_lightsout_step(&env, 24) == 0);
    assert(ib_lightsout_step(&env, 0) == 0);
    assert_close(ib_lightsout_reward(&env), -0.0288 - 0.02 - 0.015);
}

static void test_solve_caps_and_rejections(void) {
    IBLightsOut env, before;
    int32_t corner[25] = {1, 1, 0, 0, 0, 1};
    int32_t board[26];
    fixture(&env, corner, 1);
    assert(ib_lightsout_step(&env, 0) == 0);
    assert(ib_lightsout_steps(&env) == 1);
    assert(ib_lightsout_terminated(&env) == 1);
    assert(ib_lightsout_truncated(&env) == 0);
    assert(ib_lightsout_outcome(&env) == 1);
    assert(ib_lightsout_score(&env) == 1);
    assert(ib_lightsout_reward(&env) == 2);
    assert(ib_lightsout_episode_return(&env) == 2);
    assert(ib_lightsout_legal_actions(&env) == 0);
    for (int i = 0; i < 26; ++i) board[i] = -99;
    assert(ib_lightsout_observe(&env, board, 26) == 0);
    for (int i = 0; i < 25; ++i) assert(board[i] == 0);
    assert(board[25] == -99);
    memcpy(&before, &env, sizeof(env));
    assert(ib_lightsout_step(&env, 1) == -1);
    assert(memcmp(&before, &env, sizeof(env)) == 0);

    fixture(&env, corner, 1);
    assert(ib_lightsout_step(&env, 24) == 0);
    assert(ib_lightsout_terminated(&env) == 0);
    assert(ib_lightsout_truncated(&env) == 1);
    assert(ib_lightsout_outcome(&env) == 0);
    assert(ib_lightsout_score(&env) == 0);
    assert(ib_lightsout_legal_actions(&env) == 0);
    assert_close(ib_lightsout_reward(&env), -0.0288 - 0.015 - 0.5);
    memcpy(&before, &env, sizeof(env));
    assert(ib_lightsout_step(&env, 0) == -1);
    assert(memcmp(&before, &env, sizeof(env)) == 0);

    fixture(&env, corner, 10);
    memcpy(&before, &env, sizeof(env));
    int bad[] = {-1, 25, INT_MIN, INT_MAX};
    for (unsigned i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        assert(ib_lightsout_step(&env, bad[i]) == -1);
        assert(memcmp(&before, &env, sizeof(env)) == 0);
    }
    assert(ib_lightsout_reset(&env, 42, 0) == -1);
    assert(ib_lightsout_reset(&env, 42, -1) == -1);
    assert(memcmp(&before, &env, sizeof(env)) == 0);
    for (int i = 0; i < 26; ++i) board[i] = -99;
    assert(ib_lightsout_observe(&env, board, 24) == -1);
    for (int i = 0; i < 26; ++i) assert(board[i] == -99);
    assert(ib_lightsout_observe(&env, NULL, 25) == -1);
    assert(ib_lightsout_create(0, 0) == NULL);
    assert(ib_lightsout_step(NULL, 0) == -1);
    assert(ib_lightsout_reset(NULL, 0, 10) == -1);
    assert(ib_lightsout_observe(NULL, board, 25) == -1);
    assert(ib_lightsout_legal_actions(NULL) == 0);
    assert(ib_lightsout_steps(NULL) == 0);
    assert(ib_lightsout_terminated(NULL) == 0);
    assert(ib_lightsout_truncated(NULL) == 0);
    assert(ib_lightsout_outcome(NULL) == 0);
    assert(ib_lightsout_reward(NULL) == 0);
    assert(ib_lightsout_episode_return(NULL) == 0);
    assert(ib_lightsout_score(NULL) == 0);
    ib_lightsout_destroy(NULL);
}

/* Independent GF(2) elimination verifies that generated boards are solvable. */
static uint32_t solve_board(const int32_t board[25]) {
    uint32_t rows[25];
    int pivot[25], rank = 0;
    for (int i = 0; i < 25; ++i) {
        rows[i] = (uint32_t)board[i] << 25;
        for (int j = 0; j < 25; ++j) {
            int distance = abs(i / 5 - j / 5) + abs(i % 5 - j % 5);
            if (distance <= 1) rows[i] |= UINT32_C(1) << j;
        }
    }
    for (int col = 0; col < 25; ++col) {
        int found = rank;
        while (found < 25 && (rows[found] & (UINT32_C(1) << col)) == 0) ++found;
        if (found == 25) continue;
        uint32_t temp = rows[rank];
        rows[rank] = rows[found];
        rows[found] = temp;
        for (int r = 0; r < 25; ++r) {
            if (r != rank && (rows[r] & (UINT32_C(1) << col))) rows[r] ^= rows[rank];
        }
        pivot[rank++] = col;
    }
    for (int r = rank; r < 25; ++r) assert(rows[r] == 0);
    uint32_t solution = 0;
    for (int r = 0; r < rank; ++r) {
        if (rows[r] & (UINT32_C(1) << 25)) solution |= UINT32_C(1) << pivot[r];
    }
    return solution;
}

static void test_seeds_and_solvability(void) {
    IBLightsOut *a = ib_lightsout_create(0, 25);
    IBLightsOut *b = ib_lightsout_create(0, 25);
    IBLightsOut *noise = ib_lightsout_create(UINT32_MAX, 25);
    assert(a && b && noise);
    int saw_resample = 0;
    for (uint32_t seed = 0; seed < 512; ++seed) {
        assert(ib_lightsout_reset(a, seed, 25) == 0);
        assert(ib_lightsout_reset(noise, seed + 99, 25) == 0);
        assert(ib_lightsout_step(noise, 12) == 0);
        assert(ib_lightsout_reset(b, seed, 25) == 0);
        assert_same(a, b);
        assert(ib_lightsout_steps(a) == 0);
        assert(ib_lightsout_reward(a) == 0);
        assert(ib_lightsout_episode_return(a) == 0);
        assert(!ib_lightsout_terminated(a) && !ib_lightsout_truncated(a));

        unsigned int single_scramble_rng = seed;
        for (int i = 0; i < 25; ++i) (void)rand_r(&single_scramble_rng);
        if (a->rng != single_scramble_rng) saw_resample = 1;

        int32_t board[25];
        assert(ib_lightsout_observe(a, board, 25) == 0);
        uint32_t solution = solve_board(board);
        assert(solution != 0); /* No initially solved episodes. */
        for (int action = 0; action < 25 && !ib_lightsout_terminated(a); ++action) {
            if ((solution & (UINT32_C(1) << action)) == 0) continue;
            assert(ib_lightsout_step(a, action) == 0);
            if (ib_lightsout_legal_actions(noise)) assert(ib_lightsout_step(noise, action) == 0);
            assert(ib_lightsout_step(b, action) == 0);
            assert_same(a, b);
        }
        assert(ib_lightsout_terminated(a) == 1);
        assert(ib_lightsout_truncated(a) == 0);
        assert(ib_lightsout_score(a) == 1);
        assert(ib_lightsout_observe(a, board, 25) == 0);
        for (int i = 0; i < 25; ++i) assert(board[i] == 0);
    }
    assert(saw_resample);
    assert(ib_lightsout_reset(a, UINT32_MAX, 25) == 0);
    assert(ib_lightsout_reset(b, UINT32_MAX, 25) == 0);
    assert_same(a, b);
    ib_lightsout_destroy(a);
    ib_lightsout_destroy(b);
    ib_lightsout_destroy(noise);
}

int main(void) {
    test_toggle_and_shaping();
    test_solve_caps_and_rejections();
    test_seeds_and_solvability();
    puts("Lights Out: all tests passed (including 512 solved seeded puzzles)");
    return 0;
}
