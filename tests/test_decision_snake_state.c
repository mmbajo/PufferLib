/* CPU snapshot tests; compiled both as C11 and C++17. The Laya variant includes
 * this file after configuring the real tokenizer, so both adapters run the same
 * continuation and malformed-input checks. No CUDA context is created. */
#ifndef TEST_SNAKE_STATE_LAYA
#define PUF_HEADLESS
#include "../ocean/decision_snake/decision_snake.h"
#endif
#include <assert.h>

typedef struct StateHarness {
    Env env;
    obs_t observation[OBS_SIZE];
    unsigned char mask[4];
    float action, reward, terminal;
} StateHarness;

static void state_initialize(StateHarness* h, uint32_t seed, int cap, int format) {
    memset(h, 0, sizeof(*h));
    Dict kwargs = {0};
    dict_set(&kwargs, "max_steps", cap);
    dict_set(&kwargs, "observation_format", format);
    h->env.rng = seed;
    puf_init(&h->env, &kwargs);
    h->env.agents[0].observations = h->observation;
    h->env.agents[0].actions = &h->action;
    h->env.agents[0].rewards = &h->reward;
    h->env.agents[0].terminals = &h->terminal;
    h->env.agents[0].action_mask = h->mask;
    puf_reset(&h->env);
    dict_clear(&kwargs);
}

static int cycle_action(const IBSnake* snake) {
    int32_t board[100];
    assert(ib_snake_observe(snake, board, 100) == 0);
    int head = 0;
    while (head < 100 && board[head] != 1) ++head;
    assert(head < 100);
    int r = head / 10, c = head % 10;
    if (c == 0) return r == 9 ? 3 : 1;
    if (r % 2) return c == 9 ? 0 : 3;
    return c > 1 ? 2 : r == 0 ? 2 : 0;
}

static void state_write32(unsigned char* p, uint32_t value) {
    for (int i = 0; i < 4; ++i) p[i] = (unsigned char)(value >> (8*i));
}
static void state_checksum(unsigned char* data, size_t n) {
    state_write32(data+n-4, puf_snake_state_crc32(data, n-4));
}
static void engine_equal(const IBSnake* a, const IBSnake* b) {
    unsigned char left[IB_SNAKE_STATE_BYTES], right[IB_SNAKE_STATE_BYTES];
    assert(ib_snake_state_save(a, left, sizeof(left)) == 0);
    assert(ib_snake_state_save(b, right, sizeof(right)) == 0);
    assert(memcmp(left, right, sizeof(left)) == 0);
}
static void wrapper_equal(const StateHarness* a, const StateHarness* b) {
    unsigned char left[PUF_SNAKE_STATE_BYTES], right[PUF_SNAKE_STATE_BYTES];
    assert(puf_state_save(&a->env, left, sizeof(left)) == 0);
    assert(puf_state_save(&b->env, right, sizeof(right)) == 0);
    assert(memcmp(left, right, sizeof(left)) == 0);
}
static void restore(const StateHarness* from, StateHarness* to) {
    unsigned char data[PUF_SNAKE_STATE_BYTES];
    Agent bindings = to->env.agents[0];
    to->env.render_initialized = 123; // Process-local resources stay bound.
    assert(puf_state_size(&from->env) == sizeof(data));
    assert(puf_state_save(&from->env, data, sizeof(data)) == 0);
    IBSnake* previous = to->env.snake;
    assert(puf_state_validate(&to->env, data, sizeof(data)) == 0);
    assert(to->env.snake == previous);
    assert(puf_state_load(&to->env, data, sizeof(data)) == 0);
    assert(to->env.agents[0].observations == bindings.observations);
    assert(to->env.agents[0].actions == bindings.actions);
    assert(to->env.agents[0].rewards == bindings.rewards);
    assert(to->env.agents[0].terminals == bindings.terminals);
    assert(to->env.agents[0].action_mask == bindings.action_mask);
    assert(to->env.render_initialized == 123);
    wrapper_equal(from, to);
}

static void engine_continuation(void) {
    IBSnake* a = ib_snake_create(73, 20000);
    IBSnake* b = ib_snake_create(9001, 1);
    unsigned char data[IB_SNAKE_STATE_BYTES];
    assert(a && b && ib_snake_state_size() == sizeof(data));
    assert(ib_snake_state_save(a, data, sizeof(data)) == 0);
    assert(ib_snake_state_load(b, data, sizeof(data)) == 0);
    int steps = 0, foods = 0;
    while (!ib_snake_terminated(a) && !ib_snake_truncated(a)) {
        int action = cycle_action(a);
        assert(ib_snake_step(a, action) == 0 && ib_snake_step(b, action) == 0);
        foods += ib_snake_reward(a) == 1;
        engine_equal(a, b);
        if (++steps % 13 == 0) {
            assert(ib_snake_reset(b, 999, 1) == 0);
            assert(ib_snake_state_save(a, data, sizeof(data)) == 0);
            assert(ib_snake_state_load(b, data, sizeof(data)) == 0);
        }
    }
    assert(foods == 97 && ib_snake_score(a) == 97 && ib_snake_outcome(a) == 1);
    assert(ib_snake_state_save(a, data, sizeof(data)) == 0);
    assert(ib_snake_reset(b, 0, 1) == 0);
    assert(ib_snake_state_load(b, data, sizeof(data)) == 0);
    engine_equal(a, b);
    ib_snake_destroy(a); ib_snake_destroy(b);
    printf("engine full-board continuation: %d steps, %d food spawns\n", steps, foods);
}

static void wrapper_continuation(int format) {
    StateHarness a, b;
    state_initialize(&a, UINT32_MAX, 137, format);
    state_initialize(&b, 999, 137, format);
    restore(&a, &b);
    int foods = 0, timeouts = 0;
    for (int step = 0; step < 700; ++step) {
        a.action = b.action = (float)cycle_action(a.env.snake);
        puf_step(&a.env); puf_step(&b.env);
        foods += a.reward == 1; timeouts += a.env.transition.truncated;
        wrapper_equal(&a, &b);
        if (a.terminal || step % 31 == 0) {
            assert(decision_snake_reset(&b.env, 55) == 0);
            restore(&a, &b);
        }
    }
    assert(foods > 5 && timeouts == 5 && a.env.episode_seed == 4);
    // Log accumulation is checkpointed, but clearing logs at a reporting
    // boundary is also a valid state while the last transition remains live.
    memset(&a.env.log, 0, sizeof(a.env.log)); restore(&a, &b);
    puf_close(&a.env); puf_close(&b.env);

    state_initialize(&a, 73, 500, format);
    state_initialize(&b, 999, 500, format);
    for (int step = 0; step < 6; ++step) {
        a.action = 0; // Straight up reaches the wall on accepted action six.
        assert(decision_snake_step(&a.env, 0) == 0);
        restore(&a, &b); // Includes final explicit, non-autoreset state.
    }
    assert(a.env.transition.terminated && a.reward == -1);
    assert(decision_snake_step(&b.env, 0) == -1);
    wrapper_equal(&a, &b);
    puf_reset(&a.env); puf_reset(&b.env); wrapper_equal(&a, &b);
    for (int step = 0; step < 6; ++step) {
        a.action = b.action = 0; puf_step(&a.env); puf_step(&b.env);
    }
    assert(a.terminal && a.env.transition.terminated && ib_snake_steps(a.env.snake) == 0);
    restore(&a, &b);
    puf_close(&a.env); puf_close(&b.env);

    // A full board has no food, a success outcome, and a zero terminal mask.
    // After autoreset its complete final observation still belongs to the
    // transition while the live observation belongs to the next seed.
    state_initialize(&a, 73, 20000, format);
    state_initialize(&b, 999, 20000, format);
    do {
        a.action = (float)cycle_action(a.env.snake);
        puf_step(&a.env);
    } while (!a.terminal);
    assert(a.env.transition.outcome == 1 && a.env.transition.score == 97);
    assert(a.env.transition.terminated && !a.env.transition.truncated);
    restore(&a, &b);
    for (int step = 0; step < 20; ++step) {
        a.action = b.action = (float)cycle_action(a.env.snake);
        puf_step(&a.env); puf_step(&b.env); wrapper_equal(&a, &b);
    }
    puf_close(&a.env); puf_close(&b.env);
    printf("wrapper continuation format=%d: food=%d, timeouts=%d, collisions, seed wrap\n", format, foods, timeouts);
}

static void reject_engine(IBSnake* target, const void* data, size_t n) {
    unsigned char before[IB_SNAKE_STATE_BYTES], after[IB_SNAKE_STATE_BYTES];
    assert(ib_snake_state_save(target, before, sizeof(before)) == 0);
    assert(ib_snake_state_load(target, data, n) == -1);
    assert(ib_snake_state_save(target, after, sizeof(after)) == 0);
    assert(memcmp(before, after, sizeof(before)) == 0);
}
static void reject_wrapper(StateHarness* target, const void* data, size_t n) {
    unsigned char before[PUF_SNAKE_STATE_BYTES], after[PUF_SNAKE_STATE_BYTES];
    Env object = target->env;
    assert(puf_state_save(&target->env, before, sizeof(before)) == 0);
    assert(puf_state_validate(&target->env, data, n) == -1);
    assert(puf_state_load(&target->env, data, n) == -1);
    assert(memcmp(&object, &target->env, sizeof(object)) == 0);
    assert(puf_state_save(&target->env, after, sizeof(after)) == 0);
    assert(memcmp(before, after, sizeof(before)) == 0);
}

static void malformed_states(int format) {
    StateHarness source, target;
    state_initialize(&source, 73, 137, format);
    state_initialize(&target, 789, 137, format);
    source.action = 3; puf_step(&source.env);
    unsigned char engine[IB_SNAKE_STATE_BYTES], changed_engine[IB_SNAKE_STATE_BYTES];
    assert(ib_snake_state_save(source.env.snake, engine, sizeof(engine)) == 0);
    for (size_t n = 0; n < sizeof(engine); ++n) reject_engine(target.env.snake, engine, n);
    reject_engine(target.env.snake, engine, sizeof(engine)+1);
    reject_engine(target.env.snake, NULL, sizeof(engine));
    for (size_t i = 0; i < sizeof(engine); ++i) {
        memcpy(changed_engine, engine, sizeof(engine)); changed_engine[i] ^= 1;
        reject_engine(target.env.snake, changed_engine, sizeof(engine));
    }
    const uint32_t engine_bad[][2] = {
        {8,2}, {12,0}, {16,11}, {20,0}, {24,138}, {32,2}, {32,101},
        {36,56}, {40,2}, {44,1}, {48,2}, {52,98}, {60,0x7ff80000},
        {68,0x7ff00000}, {72,100}, {76,56}, {80,0}, {84,0}
    };
    for (size_t k = 0; k < sizeof(engine_bad)/sizeof(engine_bad[0]); ++k) {
        memcpy(changed_engine, engine, sizeof(engine));
        state_write32(changed_engine+engine_bad[k][0], engine_bad[k][1]);
        state_checksum(changed_engine, sizeof(engine));
        reject_engine(target.env.snake, changed_engine, sizeof(engine));
    }

    unsigned char data[PUF_SNAKE_STATE_BYTES], changed[PUF_SNAKE_STATE_BYTES];
    assert(puf_state_save(&source.env, data, sizeof(data)) == 0);
    const size_t lengths[] = {0,1,8,12,87,88,100,564,sizeof(data)-1,sizeof(data)+1};
    for (size_t k=0;k<sizeof(lengths)/sizeof(lengths[0]);++k) reject_wrapper(&target,data,lengths[k]);
    reject_wrapper(&target,NULL,sizeof(data));
    // Every byte is protected; keep the large text-record test CPU-bounded by
    // exercising all header bytes and evenly spread observation/padding bytes.
    for (size_t i = 0; i < sizeof(data); i += i < 620 ? 1 : 37) {
        memcpy(changed,data,sizeof(data)); changed[i] ^= 1;
        reject_wrapper(&target,changed,sizeof(data));
    }
    const uint32_t wrapper_bad[][2] = {
        {8,2},{12,0},{16,OBS_SIZE+1},{20,0},{24,138},{28,(uint32_t)(1-format)},
        {32,2},{36,UINT32_MAX},{40,2},{52,0},{56,1},{60,0x7fc00000},
        {64,0x7f800000},{68,0xbf800000},{72,0x3f800000},{76,0x3f800000},
        {80,0x3f800000},{84,0xbf800000},
        {564,2},{568,4},{572,1},{576,1},{580,2},{584,0},
        {588,0x7fc00000},{592,0x7f800000},{596,0x3f800000},{600,74},
        {612+2*OBS_SIZE,0x7fc00000},{616+2*OBS_SIZE,0x3f800000},
        {620+2*OBS_SIZE,0x3f800000}
    };
    for (size_t k=0;k<sizeof(wrapper_bad)/sizeof(wrapper_bad[0]);++k) {
        memcpy(changed,data,sizeof(data));
        state_write32(changed+wrapper_bad[k][0],wrapper_bad[k][1]); state_checksum(changed,sizeof(data));
        reject_wrapper(&target,changed,sizeof(data));
    }
    const size_t invalid_byte[] = {604,608,608+OBS_SIZE,612+OBS_SIZE};
    for (size_t k=0;k<sizeof(invalid_byte)/sizeof(invalid_byte[0]);++k) {
        memcpy(changed,data,sizeof(data)); changed[invalid_byte[k]] = 255;
        state_checksum(changed,sizeof(data)); reject_wrapper(&target,changed,sizeof(data));
    }
    // A bad nested engine remains invalid even if the enclosing checksum is
    // recomputed, as does an internally checksummed domain-invalid engine.
    memcpy(changed,data,sizeof(data)); changed[88+28] ^= 1; state_checksum(changed,sizeof(data));
    reject_wrapper(&target,changed,sizeof(data));
    memcpy(changed,data,sizeof(data)); state_write32(changed+88+32,101);
    state_checksum(changed+88,IB_SNAKE_STATE_BYTES); state_checksum(changed,sizeof(data));
    reject_wrapper(&target,changed,sizeof(data));
    memset(changed,0xa5,sizeof(changed)); source.env.transition.outcome = -2;
    assert(puf_state_save(&source.env,changed,sizeof(changed)) == -1);
    for (size_t i=0;i<sizeof(changed);++i) assert(changed[i] == 0xa5);
    puf_close(&source.env); puf_close(&target.env);
    printf("transactional malformed-state checks format=%d passed\n", format);
}

static int snake_state_tests(void) {
    engine_continuation();
    for (int format=0;format<=1;++format) {
        wrapper_continuation(format); malformed_states(format);
    }
    puts("Snake state snapshot tests passed");
    return 0;
}
#ifndef TEST_SNAKE_STATE_LAYA
int main(void) { return snake_state_tests(); }
#endif
