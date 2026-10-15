/* Private fixtures are linked only into the adapter test, never the trainer. */
#include "../ocean/decision_2048/engine/g2048.c"

void test_decision_2048_fixture(IBG2048* env, const unsigned char* exponents,
        int max_steps) {
    memset(env, 0, sizeof(*env));
    memcpy(env->grid, exponents, 16);
    env->rng = 42;
    env->max_steps = max_steps;
}
