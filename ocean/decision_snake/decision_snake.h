#ifndef PUFFER_DECISION_SNAKE_H
#define PUFFER_DECISION_SNAKE_H

#include <limits.h>
#include <errno.h>
#include <ctype.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned char obs_t;
#include "pufferenv.h"
#include "engine/snake.h"

#define ACT_SIZES {IB_SNAKE_ACTIONS}
#ifndef OBS_SIZE
#define OBS_SIZE IB_SNAKE_CELLS
#endif
#define NUM_ATNS 1
#define PUF_STEPS_PER_SEC 10
// The core's single terminal buffer cannot represent timeout bootstrapping.
#define PUF_HAS_TRUNCATION 1
#define PUF_ENV_STATE 1

// Float-only layout, with n last, is required by PufferLib's log reduction.
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float terminated;
    float truncated;
    float n;
};

// Owns the result of the most recent accepted action, before any autoreset.
// The final mask is zero at either episode boundary, as in the engine API.
typedef struct DecisionSnakeTransition {
    obs_t observations[OBS_SIZE];
    unsigned char action_mask[IB_SNAKE_ACTIONS];
    int valid;
    int action;
    int terminated;
    int truncated;
    int outcome;
    int steps;
    float reward;
    float score;
    float episode_return;
    uint32_t seed;
} DecisionSnakeTransition;

struct Env {
    Log log;
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng; // Initial instance seed supplied by PufferLib.
    IBSnake* snake;
    int max_steps;
    int observation_format; // Text adapter setting; numeric board observations ignore it.
    uint32_t episode_seed;
    uint32_t next_seed;
    DecisionSnakeTransition transition;
    int render_initialized;
};

// Native collector hook: retain the pre-reset input only for a pure time limit.
// A true terminal, including one coinciding with the limit, has no bootstrap.
static const obs_t* puf_truncation_observation(Env* env, const Agent* agent) {
    const DecisionSnakeTransition* transition = &env->transition;
    return agent == &env->agents[0] && transition->valid &&
        transition->truncated && !transition->terminated ? transition->observations : NULL;
}

static void decision_snake_observe(Env* env, obs_t* observations,
        unsigned char* action_mask) {
    int32_t board[IB_SNAKE_CELLS];
    ib_snake_observe(env->snake, board, IB_SNAKE_CELLS);
#ifdef DECISION_SNAKE_ENCODE_OBSERVATION
    DECISION_SNAKE_ENCODE_OBSERVATION(env, board, observations);
#else
    for (int i = 0; i < IB_SNAKE_CELLS; i++) {
        observations[i] = (obs_t)(board[i] + 1);
    }
#endif
    uint32_t mask = ib_snake_legal_actions(env->snake);
    for (int i = 0; i < IB_SNAKE_ACTIONS; i++) {
        action_mask[i] = (unsigned char)((mask >> i) & 1);
    }
}

static int decision_snake_buffers_ready(const Env* env) {
    const Agent* agent = &env->agents[0];
    return agent->observations != NULL && agent->rewards != NULL &&
        agent->terminals != NULL && agent->action_mask != NULL;
}

// Reset only the live episode; the caller decides whether to clear a transition.
static int decision_snake_start_episode(Env* env, uint32_t seed) {
    if (ib_snake_reset(env->snake, seed, env->max_steps) != 0) {
        return -1;
    }
    env->episode_seed = seed;
    env->next_seed = seed + UINT32_C(1);
    decision_snake_observe(env, env->agents[0].observations,
        env->agents[0].action_mask);
    return 0;
}

// Explicit seeded reset for collectors that own episode seed scheduling.
// Call puf_init and bind Agent buffers first. Rejection changes no state.
int decision_snake_reset(Env* env, uint32_t seed) {
    if (env == NULL || !decision_snake_buffers_ready(env) ||
            decision_snake_start_episode(env, seed) != 0) {
        return -1;
    }
    memset(&env->transition, 0, sizeof(env->transition));
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
    return 0;
}

// One exact engine step, without autoreset. Invalid actions (including a
// reversal or an action after completion) return -1 without changing buffers.
int decision_snake_step(Env* env, int action) {
    if (env == NULL || !decision_snake_buffers_ready(env) ||
            ib_snake_step(env->snake, action) != 0) {
        return -1;
    }
    DecisionSnakeTransition* transition = &env->transition;
    decision_snake_observe(env, transition->observations, transition->action_mask);
    transition->valid = 1;
    transition->action = action;
    transition->terminated = ib_snake_terminated(env->snake);
    transition->truncated = ib_snake_truncated(env->snake);
    transition->outcome = ib_snake_outcome(env->snake);
    transition->steps = ib_snake_steps(env->snake);
    transition->reward = (float)ib_snake_reward(env->snake);
    transition->score = (float)ib_snake_score(env->snake);
    transition->episode_return = (float)ib_snake_episode_return(env->snake);
    transition->seed = env->episode_seed;

    Agent* agent = &env->agents[0];
    memcpy(agent->observations, transition->observations, sizeof(transition->observations));
    memcpy(agent->action_mask, transition->action_mask, sizeof(transition->action_mask));
    agent->rewards[0] = transition->reward;
    agent->terminals[0] = (float)(transition->terminated || transition->truncated);
    if (agent->terminals[0]) {
        env->log.perf += transition->score / 97.0f;
        env->log.score += transition->score;
        env->log.episode_return += transition->episode_return;
        env->log.episode_length += (float)transition->steps;
        env->log.terminated += (float)transition->terminated;
        env->log.truncated += (float)transition->truncated;
        env->log.n += 1.0f;
    }
    return 0;
}

void puf_init(Env* env, Dict* kwargs) {
    double max_steps = dict_get(kwargs, "max_steps");
    DictItem* agents = dict_find(kwargs, "num_agents");
    DictItem* format = dict_find(kwargs, "observation_format");
    double observation_format = format ? format->value : 0;
    int format_valid = !format || format->len == 0;
    if (format && format->str) {
        char* end = NULL;
        errno = 0;
        observation_format = strtod(format->str, &end);
        format_valid = format_valid && end != format->str && errno != ERANGE;
        while (end && isspace((unsigned char)*end)) ++end;
        format_valid = format_valid && end && !*end;
    }
    if (!(max_steps >= 1 && max_steps <= INT_MAX) ||
            max_steps != (double)(int)max_steps || (agents && agents->value != 1)) {
        fprintf(stderr, "decision_snake: max_steps must be a positive int32; num_agents must be 1\n");
        exit(1);
    }
    if (!format_valid || (observation_format != 0 && observation_format != 1)) {
        fprintf(stderr, "decision_snake: observation_format must be 0 or 1\n");
        exit(1);
    }
    env->num_agents = 1;
    env->max_steps = (int)max_steps;
    env->observation_format = (int)observation_format;
    env->episode_seed = (uint32_t)env->rng;
    env->next_seed = env->episode_seed;
    env->agents[0].policy = 0;
    env->snake = ib_snake_create(env->episode_seed, env->max_steps);
    if (env->snake == NULL) {
        fprintf(stderr, "decision_snake: could not allocate engine\n");
        exit(1);
    }
}

void puf_reset(Env* env) {
    if (decision_snake_reset(env, env->next_seed) != 0) {
        fprintf(stderr, "decision_snake: reset requires a valid engine and bound Agent buffers\n");
        exit(1);
    }
}

void puf_step(Env* env) {
    float action = env->agents[0].actions[0];
    if (!(action >= 0 && action < IB_SNAKE_ACTIONS) ||
            action != (float)(int)action || decision_snake_step(env, (int)action) != 0) {
        fprintf(stderr, "decision_snake: rejected action; the policy must respect the reverse-only mask\n");
        exit(1);
    }
    if (env->transition.terminated || env->transition.truncated) {
        // Preserve transition, reward and done; publish the new episode's board
        // and mask for the next policy decision. No transition is discarded.
        if (decision_snake_start_episode(env, env->next_seed) != 0) {
            fprintf(stderr, "decision_snake: autoreset failed\n");
            exit(1);
        }
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "terminated", log->terminated);
    dict_set(out, "truncated", log->truncated);
    dict_set(out, "n", log->n);
}

// Versioned, pointer-free state for synchronous rollout-boundary checkpoints.
// Engine and wrapper records have independent CRC32 checks. Agent bindings and
// render resources belong to the destination process and are never serialized.
enum { PUF_SNAKE_STATE_BYTES = 152 + IB_SNAKE_STATE_BYTES + 2 * OBS_SIZE };
static void puf_snake_state_u32(unsigned char** cursor, uint32_t value) {
    for (int i = 0; i < 4; ++i) (*cursor)[i] = (unsigned char)(value >> (8*i));
    *cursor += 4;
}
static uint32_t puf_snake_state_read_u32(const unsigned char** cursor) {
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) value |= (uint32_t)(*cursor)[i] << (8*i);
    *cursor += 4; return value;
}
static void puf_snake_state_float(unsigned char** cursor, float value) {
    uint32_t bits; memcpy(&bits, &value, 4); puf_snake_state_u32(cursor, bits);
}
static float puf_snake_state_read_float(const unsigned char** cursor) {
    uint32_t bits = puf_snake_state_read_u32(cursor); float value;
    memcpy(&value, &bits, 4); return value;
}
static int puf_snake_state_read_int(const unsigned char** cursor, int* valid) {
    uint32_t value = puf_snake_state_read_u32(cursor);
    if (value > INT_MAX) { *valid = 0; return 0; }
    return (int)value;
}
static uint32_t puf_snake_state_crc32(const unsigned char* data, size_t bytes) {
    uint32_t crc = UINT32_MAX;
    for (size_t i = 0; i < bytes; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ ((crc & 1) ? UINT32_C(0xedb88320) : 0);
    }
    return ~crc;
}
static size_t puf_state_size(const Env* env) {
    (void)env; return PUF_SNAKE_STATE_BYTES;
}

typedef struct PufSnakeSavedState {
    Env env;
    obs_t observations[OBS_SIZE];
    unsigned char mask[IB_SNAKE_ACTIONS];
    float action, reward, terminal;
} PufSnakeSavedState;

static int puf_snake_state_observation_valid(const obs_t* observation, int score) {
#ifdef PUFFER_DECISION_POLICY
    if (!decision_policy_context) return 0;
    const DecisionPolicyContext& ctx = *decision_policy_context;
    int length = observation[0] | ((int)observation[1] << 8);
    if (length < 1 || length > ctx.execution_tokens ||
            observation[DECISION_POLICY_QTYPE_OFFSET] != 0) return 0;
    for (int i = DECISION_POLICY_QTYPE_OFFSET + 1; i < DECISION_POLICY_HEADER_BYTES; ++i)
        if (observation[i]) return 0;
    int previous = 0;
    for (int k = 0; k < IB_SNAKE_ACTIONS; ++k) {
        int marker = observation[2+2*k] | ((int)observation[3+2*k] << 8);
        if (marker <= previous || marker >= length) return 0;
        previous = marker;
        const unsigned char* token = observation + DECISION_POLICY_HEADER_BYTES + 4*marker;
        if (puf_snake_state_read_u32(&token) != ctx.bundle.mask_id) return 0;
    }
    for (int i = 0; i < DECISION_POLICY_MAX_TOKENS; ++i) {
        const unsigned char* token = observation + DECISION_POLICY_HEADER_BYTES + 4*i;
        uint32_t id = puf_snake_state_read_u32(&token);
        if (i >= length ? id != 0 : id >= (uint32_t)ctx.config.encoder.vocab) return 0;
        if ((i == 0 && id != ctx.bundle.cls_id) || (i == length-1 && id != ctx.bundle.sep_id)) return 0;
    }
    (void)score;
#else
    int positions[101], food = 0, length = 0;
    for (int i = 0; i <= 100; ++i) positions[i] = -1;
    for (int i = 0; i < OBS_SIZE; ++i) {
        int value = (int)observation[i] - 1;
        if (value < -1 || value > 100) return 0;
        if (value == -1) ++food;
        if (value > 0) {
            if (positions[value] >= 0) return 0;
            positions[value] = i; ++length;
        }
    }
    if (length != score + 3 || food != (length < 100)) return 0;
    for (int i = 1; i <= length; ++i) {
        if (positions[i] < 0) return 0;
        if (i > 1 && abs(positions[i]/10 - positions[i-1]/10) +
                abs(positions[i]%10 - positions[i-1]%10) != 1) return 0;
    }
#endif
    return 1;
}

static int puf_snake_state_domains(const Env* original, PufSnakeSavedState* saved) {
    Env* next = &saved->env;
    const DecisionSnakeTransition* t = &next->transition;
    const Log* log = &next->log;
    const float logs[] = {log->perf, log->score, log->episode_return, log->episode_length,
        log->terminated, log->truncated, log->n};
    for (int i = 0; i < 7; ++i) if (!isfinite(logs[i])) return 0;
    double tolerance = 1e-4 * (1.0 + log->n);
    if (next->num_agents != 1 || next->tag < 0 ||
            (next->boundary_reached != 0 && next->boundary_reached != 1) ||
            next->max_steps != original->max_steps ||
            next->observation_format != original->observation_format ||
            next->agents[0].policy != original->agents[0].policy ||
            next->next_seed != next->episode_seed + UINT32_C(1) ||
            log->n < 0 || log->n != floorf(log->n) ||
            log->perf < 0 || log->perf > log->n + tolerance ||
            log->score < 0 || log->score > 97.0 * log->n || log->score != floorf(log->score) ||
            log->episode_return < -log->n || log->episode_return > 97.0 * log->n ||
            log->episode_return != floorf(log->episode_return) ||
            log->episode_length < log->n || log->episode_length > (double)next->max_steps * log->n ||
            log->episode_length != floorf(log->episode_length) ||
            log->terminated < 0 || log->truncated < 0 ||
            log->terminated != floorf(log->terminated) || log->truncated != floorf(log->truncated) ||
            fabs((double)log->terminated + log->truncated - log->n) > tolerance ||
            !isfinite(saved->action) || saved->action < 0 || saved->action >= IB_SNAKE_ACTIONS ||
            saved->action != floorf(saved->action) || !isfinite(saved->reward) ||
            !isfinite(saved->terminal) || (t->valid != 0 && t->valid != 1) ||
            !isfinite(t->reward) || !isfinite(t->score) || !isfinite(t->episode_return)) return 0;
    if (!t->valid) {
        if (t->action || t->terminated || t->truncated || t->outcome || t->steps ||
                t->reward || t->score || t->episode_return || t->seed ||
                saved->reward || saved->terminal || ib_snake_steps(next->snake)) return 0;
        for (int i = 0; i < OBS_SIZE; ++i) if (t->observations[i]) return 0;
        for (int i = 0; i < IB_SNAKE_ACTIONS; ++i) if (t->action_mask[i]) return 0;
    } else {
        int collision = t->outcome == -1, full = t->score == 97;
        if (t->action < 0 || t->action >= IB_SNAKE_ACTIONS || t->steps < 1 || t->steps > next->max_steps ||
                t->score < 0 || t->score > 97 || t->score != floorf(t->score) ||
                t->score > t->steps - collision || t->outcome < -1 || t->outcome > 1 ||
                t->terminated != (collision || full) || (t->outcome == 1) != full ||
                t->truncated != (!t->terminated && t->steps == next->max_steps) ||
                t->episode_return != t->score - collision ||
                (collision ? t->reward != -1 : (t->reward != 0 && t->reward != 1)) ||
                (t->reward == 1 && t->score == 0) || (full && t->reward != 1) ||
                saved->reward != t->reward || saved->terminal != (t->terminated || t->truncated) ||
                !puf_snake_state_observation_valid(t->observations, (int)t->score)) return 0;
        for (int k = 0; k < IB_SNAKE_ACTIONS; ++k)
            if (t->action_mask[k] > 1 || ((t->terminated || t->truncated) && t->action_mask[k])) return 0;
        if (t->seed == next->episode_seed) {
            if (t->steps != ib_snake_steps(next->snake) || t->reward != ib_snake_reward(next->snake) ||
                    t->score != ib_snake_score(next->snake) || t->episode_return != ib_snake_episode_return(next->snake) ||
                    t->terminated != ib_snake_terminated(next->snake) || t->truncated != ib_snake_truncated(next->snake) ||
                    t->outcome != ib_snake_outcome(next->snake) ||
                    memcmp(t->observations, saved->observations, OBS_SIZE) ||
                    memcmp(t->action_mask, saved->mask, IB_SNAKE_ACTIONS)) return 0;
        } else if (t->seed + UINT32_C(1) != next->episode_seed ||
                !(t->terminated || t->truncated) || ib_snake_steps(next->snake)) return 0;
    }
    if (ib_snake_steps(next->snake) == 0) {
        IBSnake* reset = ib_snake_create(next->episode_seed, next->max_steps);
        unsigned char actual[IB_SNAKE_STATE_BYTES], expected[IB_SNAKE_STATE_BYTES];
        if (!reset) return 0;
        int ok = ib_snake_state_save(next->snake, actual, sizeof(actual)) == 0 &&
            ib_snake_state_save(reset, expected, sizeof(expected)) == 0 && !memcmp(actual, expected, sizeof(actual));
        ib_snake_destroy(reset); if (!ok) return 0;
    }
    obs_t expected_observation[OBS_SIZE]; unsigned char expected_mask[IB_SNAKE_ACTIONS];
#ifdef __cplusplus
    try {
#endif
        decision_snake_observe(next, expected_observation, expected_mask);
#ifdef __cplusplus
    } catch (...) { return 0; }
#endif
    return !memcmp(saved->observations, expected_observation, OBS_SIZE) &&
        !memcmp(saved->mask, expected_mask, IB_SNAKE_ACTIONS);
}

static int puf_snake_state_decode(const Env* env, const void* input, size_t bytes,
        PufSnakeSavedState* saved) {
    if (!env || !input || !env->snake || !decision_snake_buffers_ready(env) || !env->agents[0].actions ||
            bytes != PUF_SNAKE_STATE_BYTES || sizeof(float) != 4 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128) return -1;
    const unsigned char* data = (const unsigned char*)input;
    const unsigned char* trailer = data + bytes - 4;
    if (memcmp(data, "PUFSNK01", 8) || puf_snake_state_read_u32(&trailer) != puf_snake_state_crc32(data, bytes-4)) return -1;
    const unsigned char* cursor = data + 8;
    if (puf_snake_state_read_u32(&cursor) != 1 || puf_snake_state_read_u32(&cursor) != bytes ||
            puf_snake_state_read_u32(&cursor) != OBS_SIZE || puf_snake_state_read_u32(&cursor) != IB_SNAKE_STATE_BYTES) return -1;
    memset(saved, 0, sizeof(*saved)); saved->env = *env;
    Env* next = &saved->env; int valid = 1;
    next->max_steps = puf_snake_state_read_int(&cursor, &valid);
    next->observation_format = puf_snake_state_read_int(&cursor, &valid);
    next->num_agents = puf_snake_state_read_int(&cursor, &valid);
    next->tag = puf_snake_state_read_int(&cursor, &valid);
    next->boundary_reached = puf_snake_state_read_int(&cursor, &valid);
    next->rng = puf_snake_state_read_u32(&cursor);
    next->episode_seed = puf_snake_state_read_u32(&cursor);
    next->next_seed = puf_snake_state_read_u32(&cursor);
    next->agents[0].policy = puf_snake_state_read_int(&cursor, &valid);
    float logs[7]; for (int i = 0; i < 7; ++i) logs[i] = puf_snake_state_read_float(&cursor);
    next->log.perf=logs[0]; next->log.score=logs[1]; next->log.episode_return=logs[2];
    next->log.episode_length=logs[3]; next->log.terminated=logs[4]; next->log.truncated=logs[5]; next->log.n=logs[6];
    const unsigned char* engine = cursor; cursor += IB_SNAKE_STATE_BYTES;
    DecisionSnakeTransition* t = &next->transition;
    t->valid=puf_snake_state_read_int(&cursor,&valid); t->action=puf_snake_state_read_int(&cursor,&valid);
    t->terminated=puf_snake_state_read_int(&cursor,&valid); t->truncated=puf_snake_state_read_int(&cursor,&valid);
    uint32_t outcome=puf_snake_state_read_u32(&cursor);
    if (outcome != UINT32_MAX && outcome > 1) valid=0;
    t->outcome=outcome == UINT32_MAX ? -1 : (int)(outcome & 1);
    t->steps=puf_snake_state_read_int(&cursor,&valid);
    t->reward=puf_snake_state_read_float(&cursor); t->score=puf_snake_state_read_float(&cursor);
    t->episode_return=puf_snake_state_read_float(&cursor); t->seed=puf_snake_state_read_u32(&cursor);
    memcpy(t->action_mask,cursor,IB_SNAKE_ACTIONS); cursor+=IB_SNAKE_ACTIONS;
    memcpy(t->observations,cursor,OBS_SIZE); cursor+=OBS_SIZE;
    memcpy(saved->mask,cursor,IB_SNAKE_ACTIONS); cursor+=IB_SNAKE_ACTIONS;
    memcpy(saved->observations,cursor,OBS_SIZE); cursor+=OBS_SIZE;
    saved->action=puf_snake_state_read_float(&cursor); saved->reward=puf_snake_state_read_float(&cursor);
    saved->terminal=puf_snake_state_read_float(&cursor);
    const unsigned char* engine_config=engine+20;
    if (!valid || cursor != data+bytes-4 || next->max_steps < 1 ||
            puf_snake_state_read_u32(&engine_config) != (uint32_t)next->max_steps) return -1;
    next->snake=ib_snake_create(0,next->max_steps);
    if (!next->snake) return -1;
    if (ib_snake_state_load(next->snake,engine,IB_SNAKE_STATE_BYTES) || !puf_snake_state_domains(env,saved)) {
        ib_snake_destroy(next->snake); next->snake=NULL; return -1;
    }
    return 0;
}

static int puf_state_validate(const Env* env, const void* input, size_t bytes) {
    PufSnakeSavedState saved;
    if (puf_snake_state_decode(env,input,bytes,&saved)) return -1;
    ib_snake_destroy(saved.env.snake); return 0;
}
static int puf_state_load(Env* env, const void* input, size_t bytes) {
    PufSnakeSavedState saved;
    if (puf_snake_state_decode(env,input,bytes,&saved)) return -1;
    IBSnake* previous=env->snake;
    *env=saved.env;
    memcpy(env->agents[0].observations,saved.observations,OBS_SIZE);
    memcpy(env->agents[0].action_mask,saved.mask,IB_SNAKE_ACTIONS);
    env->agents[0].actions[0]=saved.action; env->agents[0].rewards[0]=saved.reward;
    env->agents[0].terminals[0]=saved.terminal;
    ib_snake_destroy(previous); return 0;
}
static int puf_state_save(const Env* env, void* output, size_t bytes) {
    if (!env || !output || bytes != PUF_SNAKE_STATE_BYTES || !env->snake ||
            !decision_snake_buffers_ready(env) || !env->agents[0].actions ||
            sizeof(float) != 4 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128) return -1;
    unsigned char data[PUF_SNAKE_STATE_BYTES]={0};
    memcpy(data,"PUFSNK01",8); unsigned char* cursor=data+8;
    puf_snake_state_u32(&cursor,1); puf_snake_state_u32(&cursor,PUF_SNAKE_STATE_BYTES);
    puf_snake_state_u32(&cursor,OBS_SIZE); puf_snake_state_u32(&cursor,IB_SNAKE_STATE_BYTES);
    puf_snake_state_u32(&cursor,(uint32_t)env->max_steps);
    puf_snake_state_u32(&cursor,(uint32_t)env->observation_format);
    puf_snake_state_u32(&cursor,(uint32_t)env->num_agents);
    puf_snake_state_u32(&cursor,(uint32_t)env->tag);
    puf_snake_state_u32(&cursor,(uint32_t)env->boundary_reached);
    puf_snake_state_u32(&cursor,env->rng); puf_snake_state_u32(&cursor,env->episode_seed);
    puf_snake_state_u32(&cursor,env->next_seed); puf_snake_state_u32(&cursor,(uint32_t)env->agents[0].policy);
    const float logs[]={env->log.perf,env->log.score,env->log.episode_return,env->log.episode_length,
        env->log.terminated,env->log.truncated,env->log.n};
    for (int i=0;i<7;++i) puf_snake_state_float(&cursor,logs[i]);
    if (ib_snake_state_save(env->snake,cursor,IB_SNAKE_STATE_BYTES)) return -1;
    cursor+=IB_SNAKE_STATE_BYTES;
    const DecisionSnakeTransition* t=&env->transition;
    if (t->outcome < -1 || t->outcome > 1) return -1;
    puf_snake_state_u32(&cursor,(uint32_t)t->valid); puf_snake_state_u32(&cursor,(uint32_t)t->action);
    puf_snake_state_u32(&cursor,(uint32_t)t->terminated); puf_snake_state_u32(&cursor,(uint32_t)t->truncated);
    puf_snake_state_u32(&cursor,t->outcome < 0 ? UINT32_MAX : (uint32_t)t->outcome);
    puf_snake_state_u32(&cursor,(uint32_t)t->steps);
    puf_snake_state_float(&cursor,t->reward); puf_snake_state_float(&cursor,t->score);
    puf_snake_state_float(&cursor,t->episode_return); puf_snake_state_u32(&cursor,t->seed);
    memcpy(cursor,t->action_mask,IB_SNAKE_ACTIONS); cursor+=IB_SNAKE_ACTIONS;
    memcpy(cursor,t->observations,OBS_SIZE); cursor+=OBS_SIZE;
    memcpy(cursor,env->agents[0].action_mask,IB_SNAKE_ACTIONS); cursor+=IB_SNAKE_ACTIONS;
    memcpy(cursor,env->agents[0].observations,OBS_SIZE); cursor+=OBS_SIZE;
    puf_snake_state_float(&cursor,env->agents[0].actions[0]);
    puf_snake_state_float(&cursor,env->agents[0].rewards[0]);
    puf_snake_state_float(&cursor,env->agents[0].terminals[0]);
    if (cursor != data+bytes-4) return -1;
    puf_snake_state_u32(&cursor,puf_snake_state_crc32(data,bytes-4));
    if (puf_state_validate(env,data,bytes)) return -1;
    memcpy(output,data,bytes); return 0;
}

// Define PUF_HEADLESS for standalone tests without a window system.
void puf_render(Env* env) {
#ifndef PUF_HEADLESS
    const int cell = 48;
    const int size = IB_SNAKE_GRID_SIZE * cell;
    if (!env->render_initialized) {
        InitWindow(size, size, "PufferLib Decision Snake");
        SetTargetFPS(PUF_STEPS_PER_SEC);
        env->render_initialized = 1;
    }
    int32_t board[IB_SNAKE_CELLS];
    ib_snake_observe(env->snake, board, IB_SNAKE_CELLS);
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    for (int i = 0; i < IB_SNAKE_CELLS; i++) {
        if (board[i] == 0) {
            continue;
        }
        Color color = board[i] == -1 ? (Color){255, 85, 85, 255} :
            board[i] == 1 ? (Color){241, 241, 241, 255} : (Color){0, 187, 187, 255};
        DrawRectangle((i % IB_SNAKE_GRID_SIZE) * cell + 1,
            (i / IB_SNAKE_GRID_SIZE) * cell + 1, cell - 2, cell - 2, color);
    }
    EndDrawing();
    puf_web_vsync();
#else
    (void)env;
#endif
}

void puf_close(Env* env) {
#ifndef PUF_HEADLESS
    if (env->render_initialized) {
        CloseWindow();
    }
#endif
    env->render_initialized = 0;
    ib_snake_destroy(env->snake);
    env->snake = NULL;
}

#endif
