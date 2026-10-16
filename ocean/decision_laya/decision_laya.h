#ifndef PUFFER_DECISION_LAYA_H
#define PUFFER_DECISION_LAYA_H

#define DECISION_ACTIONS 4
#include "../../src/decision_policy.h"
#include <sstream>
#ifndef PUFFER_DECISION_SNAKE
#define PUFFER_DECISION_SNAKE
#endif

static void decision_laya_encode(const int32_t* board, int steps, int max_steps,
        unsigned char* output) {
    std::ostringstream state;
    state << "Snake on a 10 by 10 grid. Rows run top to bottom; columns left to right. "
          << "0=empty, -1=food, 1=head, 2=neck, larger numbers follow the body toward the tail. "
          << "Step " << steps << " of " << max_steps << ". Board:\n";
    for (int row = 0; row < 10; ++row) {
        for (int col = 0; col < 10; ++col) {
            if (col) state << ',';
            state << board[row * 10 + col];
        }
        state << '\n';
    }
    decision_policy_encode(state.str(),
        "Choose the next move. Eat food and avoid the walls and snake body.",
        {"Move up", "Move down", "Move left", "Move right"}, output);
}

#define DECISION_SNAKE_ENCODE_OBSERVATION(env, board, observations) \
    decision_laya_encode(board, ib_snake_steps((env)->snake), (env)->max_steps, observations)
#include "../decision_snake/decision_snake.h"
#endif
