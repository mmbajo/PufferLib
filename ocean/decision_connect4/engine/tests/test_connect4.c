/* Including the implementation allows private fixtures without an unsafe
 * public state-injection API. The shared library exports only the public API. */
#include "../connect4.c"

#include <assert.h>
#include <limits.h>
#include <stdio.h>

static uint64_t cell(int row, int column) {
    return UINT64_C(1) << (column * 7 + 5 - row);
}

static bool grid_won(uint64_t pieces) {
    const int directions[][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    for (int row = 0; row < 6; row++) {
        for (int column = 0; column < 7; column++) {
            for (int d = 0; d < 4; d++) {
                int end_row = row + 3 * directions[d][0];
                int end_column = column + 3 * directions[d][1];
                if (end_row < 0 || end_row >= 6 || end_column < 0 || end_column >= 7) continue;
                bool line = true;
                for (int k = 0; k < 4; k++) {
                    line = line && (pieces & cell(row + k * directions[d][0],
                                                column + k * directions[d][1]));
                }
                if (line) return true;
            }
        }
    }
    return false;
}

static void assert_equal(const IBConnect4 *a, const IBConnect4 *b) {
    assert(a->player_pieces == b->player_pieces);
    assert(a->env_pieces == b->env_pieces);
    assert(a->rng == b->rng);
    assert(a->steps == b->steps);
    assert(a->max_steps == b->max_steps);
    assert(a->terminated == b->terminated);
    assert(a->truncated == b->truncated);
    assert(a->outcome == b->outcome);
    assert(a->reward == b->reward);
}

static void test_wins(void) {
    const int directions[][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    int lines = 0;
    for (int row = 0; row < 6; row++) {
        for (int column = 0; column < 7; column++) {
            for (int d = 0; d < 4; d++) {
                int end_row = row + 3 * directions[d][0];
                int end_column = column + 3 * directions[d][1];
                if (end_row < 0 || end_row >= 6 || end_column < 0 || end_column >= 7) continue;
                uint64_t pieces = 0;
                for (int k = 0; k < 4; k++) {
                    pieces |= cell(row + k * directions[d][0], column + k * directions[d][1]);
                }
                assert(won(pieces));
                assert(grid_won(pieces));
                for (int k = 0; k < 4; k++) {
                    assert(!won(pieces & ~cell(row + k * directions[d][0], column + k * directions[d][1])));
                }
                lines++;
            }
        }
    }
    assert(lines == 69);
    assert(!won(0));
    /* Adjacent packed bits across a sentinel are not an actual board line. */
    assert(!won(cell(0, 0) | cell(1, 0) | cell(5, 1) | cell(4, 1)));
}

static void test_observation_and_opening(void) {
    IBConnect4 *env = ib_connect4_create(42, 21);
    assert(env != NULL);
    int32_t board[43];
    for (int i = 0; i < 43; i++) board[i] = 123;
    assert(ib_connect4_observe(env, board, 41) == -1);
    for (int i = 0; i < 43; i++) assert(board[i] == 123);
    assert(ib_connect4_observe(env, board, 43) == 0);
    for (int i = 0; i < 42; i++) assert(board[i] == 0);
    assert(board[42] == 123);
    assert(ib_connect4_legal_actions(env) == 127);
    assert(ib_connect4_step(env, 3) == 0);
    assert(ib_connect4_observe(env, board, 42) == 0);
    for (int i = 0; i < 42; i++) {
        assert(board[i] == (i == 38 ? 1 : (i == 37 ? -1 : 0)));
    }
    assert(ib_connect4_reset(env, 42, 21) == 0);
    assert(ib_connect4_step(env, 4) == 0);
    assert(env->player_pieces == cell(5, 4));
    assert(env->env_pieces == cell(5, 3));
    ib_connect4_destroy(env);
}

static void test_bot_tactics(void) {
    IBConnect4 env = {.rng = 42, .max_steps = 21};
    env.player_pieces = cell(5, 4) | cell(5, 5) | cell(4, 5);
    env.env_pieces = cell(5, 0) | cell(5, 1) | cell(5, 2);
    assert(compute_env_move(&env) == 3); /* Take an immediate win. */

    env.player_pieces = cell(5, 0) | cell(5, 1) | cell(5, 2);
    env.env_pieces = cell(5, 4) | cell(5, 5) | cell(4, 5);
    assert(compute_env_move(&env) == 3); /* Block the only immediate threat. */
}

static void test_terminal_and_caps(void) {
    IBConnect4 env = {.rng = 42, .max_steps = 1};
    assert(ib_connect4_step(&env, 3) == 0);
    assert(ib_connect4_steps(&env) == 1);
    assert(ib_connect4_truncated(&env) == 1);
    assert(ib_connect4_terminated(&env) == 0);
    assert(ib_connect4_outcome(&env) == 0);
    assert(ib_connect4_reward(&env) == 0);
    assert(ib_connect4_score(&env) == 0);
    assert(ib_connect4_legal_actions(&env) == 0);
    IBConnect4 before = env;
    assert(ib_connect4_step(&env, 0) == -1);
    assert_equal(&env, &before);

    assert(ib_connect4_reset(&env, 1, 4) == 0);
    env.player_pieces = cell(5, 0) | cell(5, 1) | cell(5, 2);
    env.env_pieces = cell(5, 4) | cell(5, 5) | cell(4, 5);
    env.steps = 3;
    assert(ib_connect4_step(&env, 3) == 0);
    assert(ib_connect4_terminated(&env) == 1);
    assert(ib_connect4_truncated(&env) == 0); /* Win at cap wins precedence. */
    assert(ib_connect4_outcome(&env) == 1);
    assert(ib_connect4_reward(&env) == 1);
    assert(ib_connect4_episode_return(&env) == 1);
    assert(ib_connect4_score(&env) == 1);
    assert(won(env.player_pieces)); /* Final board survives. */
    assert(env.env_pieces == (cell(5, 4) | cell(5, 5) | cell(4, 5)));
    before = env;
    assert(ib_connect4_step(&env, 0) == -1);
    assert_equal(&env, &before);

    assert(ib_connect4_reset(&env, 1, 4) == 0);
    env.env_pieces = cell(5, 0) | cell(5, 1) | cell(5, 2);
    env.player_pieces = cell(5, 4) | cell(5, 5) | cell(4, 5);
    env.steps = 3;
    assert(ib_connect4_step(&env, 6) == 0);
    assert(ib_connect4_terminated(&env) == 1);
    assert(ib_connect4_truncated(&env) == 0);
    assert(ib_connect4_outcome(&env) == -1);
    assert(ib_connect4_reward(&env) == -1);
    assert(ib_connect4_episode_return(&env) == -1);
    assert(ib_connect4_score(&env) == -1);
    assert(won(env.env_pieces));
}

static void test_draw_regression(void) {
    /* Alternating pairs make a full board with no horizontal, vertical,
     * or diagonal four. Remove one top cell from each player for a final
     * two-move draw, where column 2 is the bot's only remaining action. */
    IBConnect4 env = {.rng = 42, .steps = 20, .max_steps = 21};
    for (int row = 0; row < 6; row++) {
        for (int column = 0; column < 7; column++) {
            if (((column / 2) + row) % 2 == 0) env.player_pieces |= cell(row, column);
            else env.env_pieces |= cell(row, column);
        }
    }
    uint64_t full = env.player_pieces | env.env_pieces;
    assert(full == UINT64_C(279258638311359));
    assert(full != UINT64_C(4432406249472)); /* Original comparison fails. */
    assert(draw(full));
    assert(!draw(UINT64_C(4432406249472)));
    assert(!grid_won(env.player_pieces));
    assert(!grid_won(env.env_pieces));
    env.player_pieces &= ~cell(0, 0);
    env.env_pieces &= ~cell(0, 2);
    assert(ib_connect4_legal_actions(&env) == 5);
    assert(ib_connect4_step(&env, 0) == 0);
    assert((env.player_pieces | env.env_pieces) == full);
    assert(ib_connect4_terminated(&env) == 1);
    assert(ib_connect4_truncated(&env) == 0);
    assert(ib_connect4_outcome(&env) == 0);
    assert(ib_connect4_reward(&env) == 0);
    assert(ib_connect4_score(&env) == 0);
    assert(ib_connect4_legal_actions(&env) == 0);
}

static void test_rejected_actions(void) {
    IBConnect4 env = {.rng = 42, .max_steps = 21};
    const int invalids[] = {-1, 7, INT_MIN, INT_MAX};
    for (int i = 0; i < 4; i++) {
        IBConnect4 before = env;
        assert(ib_connect4_step(&env, invalids[i]) == -1);
        assert_equal(&env, &before);
    }
    env.player_pieces = cell(5, 0) | cell(3, 0) | cell(1, 0);
    env.env_pieces = cell(4, 0) | cell(2, 0) | cell(0, 0);
    assert(ib_connect4_legal_actions(&env) == 126);
    IBConnect4 before = env;
    assert(ib_connect4_step(&env, 0) == -1);
    assert_equal(&env, &before);
    assert(ib_connect4_reset(&env, 9, 0) == -1);
    assert_equal(&env, &before);
    assert(ib_connect4_reset(&env, 9, -1) == -1);
    assert_equal(&env, &before);
}

static int choose_action(uint32_t mask, uint32_t value) {
    int choices[7], count = 0;
    for (int column = 0; column < 7; column++) {
        if (mask & (UINT32_C(1) << column)) choices[count++] = column;
    }
    assert(count > 0);
    return choices[value % (uint32_t)count];
}

static void test_seeded_games(void) {
    for (uint32_t seed = 0; seed < 128; seed++) {
        IBConnect4 *a = ib_connect4_create(seed, 21);
        IBConnect4 *b = ib_connect4_create(seed, 21);
        IBConnect4 *noise = ib_connect4_create(seed + 1000, 21);
        assert(a && b && noise);
        int actions[21], count = 0;
        uint32_t selector = seed + 1;
        while (ib_connect4_legal_actions(a)) {
            selector = selector * UINT32_C(1664525) + UINT32_C(1013904223);
            int action = choose_action(ib_connect4_legal_actions(a), selector);
            actions[count++] = action;
            assert(ib_connect4_step(a, action) == 0);
            if (!ib_connect4_legal_actions(noise)) assert(ib_connect4_reset(noise, selector, 21) == 0);
            assert(ib_connect4_step(noise, choose_action(ib_connect4_legal_actions(noise), selector)) == 0);
            assert(ib_connect4_step(b, action) == 0);
            assert_equal(a, b); /* An interleaved environment cannot consume RNG. */
            assert(won(a->player_pieces) == grid_won(a->player_pieces));
            assert(won(a->env_pieces) == grid_won(a->env_pieces));
            assert((a->player_pieces & a->env_pieces) == 0);
            assert(a->steps == count);
        }
        assert(a->terminated);
        assert(!a->truncated);
        assert(count <= 21);
        assert(ib_connect4_reset(b, seed, 21) == 0);
        for (int i = 0; i < count; i++) assert(ib_connect4_step(b, actions[i]) == 0);
        assert_equal(a, b); /* Reset reproduces the entire episode. */
        ib_connect4_destroy(a);
        ib_connect4_destroy(b);
        ib_connect4_destroy(noise);
    }
}

static void test_nulls(void) {
    int32_t board[42] = {0};
    IBConnect4 env = {.max_steps = 1};
    assert(ib_connect4_create(1, 0) == NULL);
    assert(ib_connect4_create(1, -1) == NULL);
    assert(ib_connect4_reset(NULL, 1, 1) == -1);
    assert(ib_connect4_step(NULL, 1) == -1);
    assert(ib_connect4_observe(NULL, board, 42) == -1);
    assert(ib_connect4_observe(&env, NULL, 42) == -1);
    assert(ib_connect4_legal_actions(NULL) == 0);
    assert(ib_connect4_steps(NULL) == 0);
    assert(ib_connect4_terminated(NULL) == 0);
    assert(ib_connect4_truncated(NULL) == 0);
    assert(ib_connect4_outcome(NULL) == 0);
    assert(ib_connect4_reward(NULL) == 0);
    assert(ib_connect4_episode_return(NULL) == 0);
    assert(ib_connect4_score(NULL) == 0);
    ib_connect4_destroy(NULL);
}

int main(void) {
    test_wins();
    test_observation_and_opening();
    test_bot_tactics();
    test_terminal_and_caps();
    test_draw_regression();
    test_rejected_actions();
    test_seeded_games();
    test_nulls();
    puts("connect4: all tests passed (69 win lines; 128 seeded replay/interleaving games)");
    return 0;
}
