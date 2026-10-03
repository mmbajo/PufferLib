#ifndef PUFFER_DECISION_LAYA_H
#define PUFFER_DECISION_LAYA_H

#define DECISION_ACTIONS 4
#include "../../src/decision_policy.h"
#include <sstream>
#ifndef PUFFER_DECISION_SNAKE
#define PUFFER_DECISION_SNAKE
#endif

static void decision_laya_encode(const int32_t* board, int steps, int max_steps,
        unsigned char* output, int observation_format = 0) {
    if (observation_format != 0 && observation_format != 1)
        throw std::invalid_argument("decision_laya observation_format must be 0 or 1");
    std::ostringstream state;
    if (observation_format == 1) {
        int head = -1, food = -1;
        for (int cell = 0; cell < 100; ++cell) {
            if (board[cell] == 1) head = cell;
            if (board[cell] == -1) food = cell;
        }
        if (head < 0) throw std::invalid_argument("decision_laya board has no head");
        state << "Coordinates are zero-based (row, column). Head: ("
              << head / 10 << ", " << head % 10 << "). ";
        if (food >= 0) {
            state << "Food: (" << food / 10 << ", " << food % 10 << "). "
                  << "Food minus head: row " << std::showpos << food / 10 - head / 10
                  << ", column " << food % 10 - head % 10 << std::noshowpos << ". ";
        } else {
            state << "Food: none. ";
        }
        state << "Negative row=up; positive row=down; negative column=left; positive column=right.\n";
    }
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
    decision_laya_encode(board, ib_snake_steps((env)->snake), (env)->max_steps, \
        observations, (env)->observation_format)
#include "../decision_snake/decision_snake.h"
#endif
