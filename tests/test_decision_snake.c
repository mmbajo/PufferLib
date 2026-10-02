/* Native adapter harness: exercise the real PufferLib headers without a display.
 * Expected gameplay traces are captured independently, in the JSON fixture. */
#include "../ocean/decision_snake/decision_snake.h"

#include <assert.h>
#include <limits.h>

typedef struct {
    Env env;
    obs_t observations[OBS_SIZE];
    unsigned char action_mask[4];
    float action, reward, terminal;
} Harness;

static void initialize(Harness* harness, uint32_t seed, int max_steps) {
    memset(harness, 0, sizeof(*harness));
    Dict kwargs = {0};
    dict_set(&kwargs, "max_steps", max_steps);
    dict_set(&kwargs, "num_agents", 1);
    dict_set(&kwargs, "cell_size", 32);
    harness->env.rng = seed;
    puf_init(&harness->env, &kwargs);
    assert(harness->env.num_agents == 1);
    harness->env.agents[0].observations = harness->observations;
    harness->env.agents[0].actions = &harness->action;
    harness->env.agents[0].rewards = &harness->reward;
    harness->env.agents[0].terminals = &harness->terminal;
    harness->env.agents[0].action_mask = harness->action_mask;
    puf_reset(&harness->env);
    assert(harness->env.episode_seed == seed);
    dict_clear(&kwargs);
}

static unsigned int mask_bits(const unsigned char* mask) {
    unsigned int bits = 0;
    assert(mask != NULL);
    for (int i = 0; i < 4; ++i) {
        assert(mask[i] == 0 || mask[i] == 1);
        bits |= (unsigned int)mask[i] << i;
    }
    return bits;
}

static void board_row(const obs_t* board) {
    for (int i = 0; i < OBS_SIZE; ++i) printf(",%d", (int)board[i] - 1);
    putchar('\n');
}

static void initial_row(const Harness* harness) {
    assert(harness->env.transition.valid == 0);
    assert(harness->reward == 0 && harness->terminal == 0);
    printf("0,0,0,0,0,0,0,%u", mask_bits(harness->env.agents[0].action_mask));
    board_row(harness->observations);
}

static void transition_row(const Harness* harness, int action, uint32_t seed) {
    const DecisionSnakeTransition* transition = &harness->env.transition;
    assert(transition->valid);
    assert(transition->action == action);
    assert(transition->seed == seed);
    assert(transition->reward == (int)transition->reward);
    assert(transition->score == (int)transition->score);
    assert(transition->episode_return == (int)transition->episode_return);
    assert(harness->reward == transition->reward);
    assert(harness->terminal == (transition->terminated || transition->truncated));
    if (!harness->terminal) {
        assert(memcmp(transition->observations, harness->observations,
                      sizeof(harness->observations)) == 0);
        assert(memcmp(transition->action_mask, harness->env.agents[0].action_mask, 4) == 0);
    }
    printf("%d,%d,%d,%d,%d,%d,%d,%u", transition->steps,
           transition->terminated, transition->truncated, transition->outcome,
           (int)transition->score, (int)transition->episode_return,
           (int)transition->reward, mask_bits(transition->action_mask));
    board_row(transition->observations);
}

static void live_row(const Harness* harness) {
    printf("LIVE,%u,%u,%d,%d,%u", harness->env.episode_seed,
           harness->env.next_seed, (int)harness->terminal, (int)harness->reward,
           mask_bits(harness->env.agents[0].action_mask));
    board_row(harness->observations);
}

static void assert_rejected(Harness* harness, int action) {
    DecisionSnakeTransition before = harness->env.transition;
    obs_t observation[OBS_SIZE];
    unsigned char mask[4];
    memcpy(observation, harness->observations, sizeof(observation));
    memcpy(mask, harness->env.agents[0].action_mask, sizeof(mask));
    float reward = harness->reward, terminal = harness->terminal;
    uint32_t episode_seed = harness->env.episode_seed, next_seed = harness->env.next_seed;
    assert(decision_snake_step(&harness->env, action) == -1);
    assert(memcmp(&before, &harness->env.transition, sizeof(before)) == 0);
    assert(memcmp(observation, harness->observations, sizeof(observation)) == 0);
    assert(memcmp(mask, harness->env.agents[0].action_mask, sizeof(mask)) == 0);
    assert(harness->reward == reward && harness->terminal == terminal);
    assert(harness->env.episode_seed == episode_seed && harness->env.next_seed == next_seed);
}

static void noise_step(Harness* noise) {
    unsigned int mask = mask_bits(noise->env.agents[0].action_mask);
    assert(mask);
    int action = 0;
    while (!(mask & (1u << action))) ++action;
    noise->action = (float)action;
    puf_step(&noise->env);
}

int main(int argc, char** argv) {
    assert(argc == 5);
    const char* mode = argv[1];
    uint32_t seed = (uint32_t)strtoull(argv[2], NULL, 10);
    int cap = atoi(argv[3]);
    const char* actions = argv[4];
    Harness harness, noise;
    initialize(&harness, seed, cap);
    initialize(&noise, seed ^ UINT32_C(0x73ac921f), 7);
    int automatic = strcmp(mode, "automatic") == 0 || strcmp(mode, "interleaved") == 0;
    int interleaved = strcmp(mode, "interleaved") == 0;
    int reject = strcmp(mode, "reject") == 0;
    int repeat = strcmp(mode, "repeat") == 0;

    for (int pass = 0; pass < (repeat ? 2 : 1); ++pass) {
        if (pass) {
            assert(decision_snake_reset(&harness.env, seed) == 0);
            puts("REPEAT");
        }
        initial_row(&harness);
        if (reject) {
            const int invalid[] = {-1, 4, INT_MIN, INT_MAX, 2};
            for (unsigned int i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i)
                assert_rejected(&harness, invalid[i]);
        }
        for (const char* item = actions; *item; ++item) {
            int action = *item - '0';
            assert(action >= 0 && action < 4);
            assert(mask_bits(harness.env.agents[0].action_mask) & (1u << action));
            harness.action = (float)action;
            if (interleaved) noise_step(&noise);
            if (automatic) puf_step(&harness.env);
            else assert(decision_snake_step(&harness.env, action) == 0);
            if (interleaved) noise_step(&noise);
            transition_row(&harness, action, seed);
            if (harness.terminal) {
                assert(!item[1]);
                if (!automatic) {
                    assert_rejected(&harness, 0);
                    assert(memcmp(harness.env.transition.observations, harness.observations,
                                  sizeof(harness.observations)) == 0);
                }
            }
        }
        assert(harness.terminal == 1);
        live_row(&harness);
        if (automatic) {
            Log* log = &harness.env.log;
            Dict out = {0};
            puf_log(log, &out);
            assert(dict_get(&out, "n") == 1);
            assert(dict_get(&out, "score") == harness.env.transition.score);
            assert(dict_get(&out, "episode_return") == harness.env.transition.episode_return);
            assert(dict_get(&out, "episode_length") == harness.env.transition.steps);
            dict_clear(&out);
            // The next transition must clear the previous terminal/reward and
            // refer to the new episode, even after an eating timeout or death.
            harness.action = 0;
            puf_step(&harness.env);
            assert(harness.env.transition.seed == seed + UINT32_C(1));
            assert(harness.env.transition.steps == 1);
            assert(harness.reward == harness.env.transition.reward);
            assert(harness.terminal == (cap == 1));
            if (cap > 1) assert(!harness.env.transition.truncated);
        }
    }
    puf_close(&harness.env);
    puf_close(&noise.env);
    return 0;
}
