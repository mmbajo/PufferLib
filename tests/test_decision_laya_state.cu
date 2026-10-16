// CPU-only snapshot validation with real packed Laya observations/tokenizer.
#include "../src/ini.h"
#define PUF_HEADLESS
#include "../ocean/decision_laya/decision_laya.h"
#define TEST_SNAKE_STATE_LAYA
#include "test_decision_snake_state.c"

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr,"usage: test_decision_laya_state BUNDLE\n"); return 2; }
    decision_policy_context.reset(new DecisionPolicyContext(argv[1]));
    return snake_state_tests();
}
