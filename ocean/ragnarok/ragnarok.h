// ragnarok: novices leveling on a Ragnarok Online field, played on real rAthena servers.
//
// Each env drives one throwaway world from slurm/world.sh with num_agents bots, which
// puf_init logs in through client/rathena_client.h. Env i uses world i (127.0.0.1, ports
// $RO_WORLD_PORT_BASE+10i, default 7000), the accounts in $RO_WORLD_ROOT/w<i>/accounts and
// rAthena's db/ in $RO_DB_DIR. One env per world: two envs on one world would wait for each
// other's bots. Each puf_init claims the next block of num_agents accounts (w<i>/next_block), since
// rAthena keeps a just-logged-out account "online" for up to 30 s and a sweep starts a new trainer
// process per trial.
//
// A step is 100 ms of game time. With lockstep = 1 (training worlds) it is one step of the world:
// the env turns every action into requests, sends READY for every bot, and waits for the server's
// DONE. With lockstep = 0 (a real-time server, such as ro-server) it waits 100 ms of wall time.
//
// Observation: own state, the walkable cells around the bot, the nearest monsters and the nearest
// items on the ground. Action: a verb (wait, move, attack, pick up, potion, sit/stand), a direction
// for move and a slot for attack and pick up (into the monster and item lists the agent last saw).
// Attacking and picking up go on over the next steps: the env first walks the bot into range,
// since rAthena never walks a player there. Reward: reward_scale times (base exp gained divided by
// the exp the level needs, plus reward_job times the same for job exp, plus reward_death on
// death). An episode lasts
// episode_steps steps or until death; the training world's @rlreset starts the next one. The
// logged perf is exp per step relative to the scripted policy (1.0 = as good as that bot).

#include <poll.h>
#include <sys/file.h>

#include "client/rathena_client.h"

typedef float obs_t;
#include "pufferenv.h"

#define RG_VIEW 15
#define RG_MOBS 8
#define RG_MOB_F 13
#define RG_ITEMS 4
#define RG_ITEM_F 4
#define RG_SELF 14
#define OBS_SIZE (RG_SELF + RG_VIEW * RG_VIEW + RG_MOBS * RG_MOB_F + RG_ITEMS * RG_ITEM_F)
#define ACT_SIZES {6, 8, 8}
#define NUM_ATNS 3
#define RG_RED_POTION 501
#define RG_SIGHT 14.0f
// Base exp per step of the scripted policy in this env (ragnarok_check, 2026-11-05): 32.4 to 34.7
// per 1,125 to 1,200 steps. perf = an episode's exp per step over this, so 1.0 matches that bot.
#define RG_SCRIPTED_EXP_PER_STEP 0.0288f

enum { RG_WAIT, RG_MOVE, RG_ATTACK, RG_PICKUP, RG_POTION, RG_SIT };
enum { RG_NONE, RG_ATTACKING, RG_PICKING };

static const int RG_DX[8] = {1, 1, 0, -1, -1, -1, 0, 1};
static const int RG_DY[8] = {0, 1, 1, 1, 0, -1, -1, -1};
// Monster classes the observation names; any other is "other".
static const int RG_CLASSES[4] = {1002, 1063, 1113, 1008};  // Poring, Lunatic, Drops, Pupa

struct Log {
    float perf;
    float score;
    float episode_length;
    float base_level;
    float job_level;
    float kills;
    float deaths;
    float potions;
    float picked;
    float base_exp;
    float n;
};

typedef struct RgBot {
    RoClient c;
    int intent, attacking;
    uint32_t target, item;
    double intent_since, last_attack, last_pickup;
    uint32_t slot_mob[RG_MOBS], slot_item[RG_ITEMS];
    int64_t base_seen, job_seen;  // cumulative exp at the end of the last step
    int deaths_seen;
    // This episode.
    int steps, kills0, potions0, picked0;
    int64_t base0;
    float ret;
} RgBot;

struct Env {
    Log log;
    Agent* agents;
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;
    RgBot* bots;
    RoClient** clients;  // &bots[i].c, for the client's lockstep helpers
    int lockstep, episode_steps;
    float reward_scale, reward_job, reward_death;
    double next_step;  // real time: when the current step ends
};
typedef Env Ragnarok;

static int rg_dist(int x0, int y0, int x1, int y1) {
    int dx = abs(x1 - x0), dy = abs(y1 - y0);
    return dx > dy ? dx : dy;
}

static void rg_die(const char* fmt, const char* a, const char* b) {
    fprintf(stderr, "ragnarok env: ");
    fprintf(stderr, fmt, a, b);
    fprintf(stderr, "\n");
    exit(1);
}

static void rg_world_step(Env* env) {
    if (env->lockstep) {
        if (ro_lockstep_step_all(env->clients, env->num_agents, 60000) < 0)
            rg_die("world %s: no DONE within 60 s of wall time%s", env->bots[0].c.user, "");
        return;
    }
    double now = ro_ms();
    if (env->next_step < now - 1000) env->next_step = now;
    env->next_step += 100;
    while ((now = ro_ms()) < env->next_step) {
        for (int i = 0; i < env->num_agents; i++) ro_poll(&env->bots[i].c);
        usleep(2000);
    }
    for (int i = 0; i < env->num_agents; i++) ro_poll(&env->bots[i].c);
}

// The verb sets what the bot is doing; attacking and picking up carry on in rg_continue.
static void rg_act(RgBot* b, const float* a) {
    RoClient* c = &b->c;
    if (c->state != RO_IN_GAME || c->dead) return;
    int verb = (int)a[0], dir = (int)a[1], slot = (int)a[2];
    int x, y;
    ro_self_pos(c, &x, &y);
    switch (verb) {
    case RG_MOVE:
        if (dir < 0 || dir > 7) break;
        for (int d = 3; d >= 1; d--) {
            int tx = x + RG_DX[dir] * d, ty = y + RG_DY[dir] * d;
            if (ro_walkable(c->cells, tx, ty)) {
                if (c->sitting) ro_sit(c, 0);
                ro_walk(c, tx, ty);
                b->intent = RG_NONE;
                break;
            }
        }
        break;
    case RG_ATTACK:
        if (slot >= 0 && slot < RG_MOBS && b->slot_mob[slot] && ro_unit_find(c, b->slot_mob[slot])) {
            if (b->intent != RG_ATTACKING || b->target != b->slot_mob[slot]) {
                b->intent = RG_ATTACKING;
                b->target = b->slot_mob[slot];
                b->attacking = 0;
                b->intent_since = c->now;
            }
            if (c->sitting) ro_sit(c, 0);
        }
        break;
    case RG_PICKUP:
        if (slot >= 0 && slot < RG_ITEMS && b->slot_item[slot]) {
            b->intent = RG_PICKING;
            b->item = b->slot_item[slot];
            b->intent_since = c->now;
            if (c->sitting) ro_sit(c, 0);
        }
        break;
    case RG_POTION: {  // only below 70% HP, so exploring does not drink them all
        const RoInv* potion = ro_inv_find(c, RG_RED_POTION);
        if (potion && c->maxhp > 0 && c->hp < 0.7 * c->maxhp) ro_use_item(c, potion->index);
        break;
    }
    case RG_SIT:
        ro_sit(c, !c->sitting);
        b->intent = RG_NONE;
        break;
    default:
        break;
    }
}

static void rg_continue(RgBot* b) {
    RoClient* c = &b->c;
    if (c->state != RO_IN_GAME || c->dead || b->intent == RG_NONE) return;
    int x, y;
    ro_self_pos(c, &x, &y);
    if (c->now - b->intent_since > 30000) {
        b->intent = RG_NONE;
        return;
    }
    if (b->intent == RG_ATTACKING) {
        RoUnit* t = ro_unit_find(c, b->target);
        if (!t) {
            b->intent = RG_NONE;
            return;
        }
        int tx, ty, range = c->attack_range > 0 ? c->attack_range : 1;
        ro_unit_pos(c, t, &tx, &ty);
        int too_far = c->too_far_id == t->id && c->now - c->too_far_at < 1000;
        if (rg_dist(x, y, tx, ty) > range || too_far) {
            int ax = tx + (x > tx) - (x < tx), ay = ty + (y > ty) - (y < ty);
            if (!ro_walkable(c->cells, ax, ay)) ax = tx, ay = ty;
            if (!ro_self_moving(c) || rg_dist(c->x1, c->y1, tx, ty) > range || too_far) ro_walk(c, ax, ay);
            b->attacking = 0;
        } else if (!b->attacking || c->now - b->last_attack > 2000) {
            ro_attack(c, t->id);
            b->attacking = 1;
            b->last_attack = c->now;
        }
        return;
    }
    const RoGround* g = NULL;
    for (int i = 0; i < c->nground; i++)
        if (c->ground[i].id == b->item) g = &c->ground[i];
    if (!g) {
        b->intent = RG_NONE;
        return;
    }
    if (rg_dist(x, y, g->x, g->y) <= 1) {
        if (c->now - b->last_pickup > 500) {
            ro_pickup(c, g->id);
            b->last_pickup = c->now;
        }
    } else if (!ro_self_moving(c) || c->x1 != g->x || c->y1 != g->y) {
        ro_walk(c, g->x, g->y);
    }
}

static int rg_class_slot(int cls) {
    for (int i = 0; i < 4; i++)
        if (RG_CLASSES[i] == cls) return i;
    return 4;
}

static void rg_observe(Env* env, int i) {
    RgBot* b = &env->bots[i];
    RoClient* c = &b->c;
    float* o = env->agents[i].observations;
    memset(o, 0, OBS_SIZE * sizeof(float));
    memset(b->slot_mob, 0, sizeof b->slot_mob);
    memset(b->slot_item, 0, sizeof b->slot_item);
    if (c->state != RO_IN_GAME) return;
    int x, y;
    ro_self_pos(c, &x, &y);
    const RoInv* potion = ro_inv_find(c, RG_RED_POTION);
    o[0] = c->maxhp > 0 ? (float)c->hp / c->maxhp : 0;
    o[1] = c->maxsp > 0 ? (float)c->sp / c->maxsp : 0;
    o[2] = c->base_level / 10.0f;
    o[3] = c->job_level / 10.0f;
    o[4] = c->next_base_exp > 0 ? (float)c->base_exp / c->next_base_exp : 0;
    o[5] = c->next_job_exp > 0 ? (float)c->job_exp / c->next_job_exp : 0;
    o[6] = c->max_weight > 0 ? (float)c->weight / c->max_weight : 0;
    o[7] = c->sitting;
    o[8] = c->dead;
    o[9] = potion ? potion->amount / 30.0f : 0;
    o[10] = ro_self_moving(c);
    o[11] = b->intent == RG_ATTACKING;
    o[12] = b->intent == RG_PICKING;
    o[13] = env->episode_steps > 0 ? (float)b->steps / env->episode_steps : 0;
    float* view = o + RG_SELF;
    for (int dy = 0; dy < RG_VIEW; dy++)
        for (int dx = 0; dx < RG_VIEW; dx++)
            view[dx + dy * RG_VIEW] = ro_walkable(c->cells, x + dx - RG_VIEW / 2, y + dy - RG_VIEW / 2);

    // The nearest monsters, nearest first, by insertion into a short sorted list.
    int order[RG_MOBS], dists[RG_MOBS], n = 0;
    for (int u = 0; u < c->nunits; u++) {
        if (c->units[u].type != RO_OBJ_MOB) continue;
        int ux, uy;
        ro_unit_pos(c, &c->units[u], &ux, &uy);
        int d = rg_dist(x, y, ux, uy), k = n < RG_MOBS ? n++ : RG_MOBS;
        if (k == RG_MOBS && d >= dists[RG_MOBS - 1]) continue;
        if (k == RG_MOBS) k = RG_MOBS - 1;
        while (k > 0 && dists[k - 1] > d) {
            order[k] = order[k - 1];
            dists[k] = dists[k - 1];
            k--;
        }
        order[k] = u;
        dists[k] = d;
    }
    float* mobs = view + RG_VIEW * RG_VIEW;
    for (int k = 0; k < n; k++) {
        const RoUnit* u = &c->units[order[k]];
        float* m = mobs + k * RG_MOB_F;
        int ux, uy;
        ro_unit_pos(c, u, &ux, &uy);
        m[0] = 1;
        m[1] = (ux - x) / RG_SIGHT;
        m[2] = (uy - y) / RG_SIGHT;
        m[3] = dists[k] / RG_SIGHT;
        m[4] = u->maxhp > 0 ? (float)u->hp / u->maxhp : 1;
        m[5] = u->last_hit_me && c->now - u->last_hit_me < 3000;
        m[6] = b->intent == RG_ATTACKING && b->target == u->id;
        m[7] = u->boss;
        m[8 + rg_class_slot(u->cls)] = 1;
        b->slot_mob[k] = u->id;
    }
    float* items = mobs + RG_MOBS * RG_MOB_F;
    int used[RO_MAX_GROUND] = {0};
    for (int k = 0; k < RG_ITEMS; k++) {
        int best = -1, best_d = 1 << 30;
        for (int g = 0; g < c->nground; g++) {
            int d = rg_dist(x, y, c->ground[g].x, c->ground[g].y);
            if (!used[g] && d < best_d) best = g, best_d = d;
        }
        if (best < 0 || best_d > RG_SIGHT) break;
        used[best] = 1;
        float* it = items + k * RG_ITEM_F;
        it[0] = 1;
        it[1] = (c->ground[best].x - x) / RG_SIGHT;
        it[2] = (c->ground[best].y - y) / RG_SIGHT;
        it[3] = best_d / RG_SIGHT;
        b->slot_item[k] = c->ground[best].id;
    }
}

static void rg_begin_episode(RgBot* b) {
    RoClient* c = &b->c;
    b->steps = 0;
    b->ret = 0;
    b->kills0 = c->kills;
    b->potions0 = c->potions;
    b->picked0 = c->picked;
    b->base0 = ro_total_base_exp(c);
    b->intent = RG_NONE;
    b->attacking = 0;
}

static void rg_end_episode(Env* env, RgBot* b) {
    RoClient* c = &b->c;
    if (b->steps > 0)
        env->log.perf += (float)(ro_total_base_exp(c) - b->base0) / b->steps / RG_SCRIPTED_EXP_PER_STEP;
    env->log.score += b->ret;
    env->log.episode_length += b->steps;
    env->log.base_level += c->base_level;
    env->log.job_level += c->job_level;
    env->log.kills += c->kills - b->kills0;
    env->log.potions += c->potions - b->potions0;
    env->log.picked += c->picked - b->picked0;
    env->log.base_exp += (float)(ro_total_base_exp(c) - b->base0);
    env->log.deaths += c->dead;
    env->log.n += 1;
    if (env->lockstep) ro_chat(c, "@rlreset");  // training world: takes effect in the next step
    else if (c->dead) ro_respawn(c);              // a real-time server has no @rlreset
    rg_begin_episode(b);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = dict_get(kwargs, "num_agents");
    env->lockstep = dict_get(kwargs, "lockstep");
    env->episode_steps = dict_get(kwargs, "episode_steps");
    env->reward_scale = dict_get(kwargs, "reward_scale");
    env->reward_job = dict_get(kwargs, "reward_job");
    env->reward_death = dict_get(kwargs, "reward_death");
    env->agents = (Agent*)calloc(env->num_agents, sizeof(Agent));
    env->bots = (RgBot*)calloc(env->num_agents, sizeof(RgBot));
    env->clients = (RoClient**)calloc(env->num_agents, sizeof(RoClient*));
    for (int i = 0; i < env->num_agents; i++) env->clients[i] = &env->bots[i].c;
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].action_mask = NULL;
        env->agents[i].policy = 0;
    }
    const char* root = getenv("RO_WORLD_ROOT");
    const char* db = getenv("RO_DB_DIR");
    const char* host = getenv("RO_WORLD_HOST");
    if (!root || !db) rg_die("set RO_WORLD_ROOT and RO_DB_DIR%s%s", "", "");
    int world = (int)env->rng;
    char path[512];
    snprintf(path, sizeof path, "%s/w%d/accounts", root, world);
    FILE* f = fopen(path, "r");
    if (!f) rg_die("cannot open %s%s (one env per world: is total_agents / num_agents <= the worlds started?)", path, "");
    // The next unused block of accounts (see the top of this file), plus RG_ACCOUNT_OFFSET if set.
    int block = 0;
    char block_path[512];
    snprintf(block_path, sizeof block_path, "%s/w%d/next_block", root, world);
    FILE* bf = fopen(block_path, "r+");
    if (bf) {
        flock(fileno(bf), LOCK_EX);
        if (fscanf(bf, "%d", &block) != 1) block = 0;
        rewind(bf);
        fprintf(bf, "%d\n", block + 1);
        fflush(bf);
        flock(fileno(bf), LOCK_UN);
        fclose(bf);
    }
    const char* skip = getenv("RG_ACCOUNT_OFFSET");
    for (int i = 0, n = block * env->num_agents + (skip ? atoi(skip) : 0); i < n; i++) {
        char user[64], pass[64];
        if (fscanf(f, "%63s %63s", user, pass) != 2)
            rg_die("%s has too few accounts%s (raise WORLD_ACCOUNT_BLOCKS)", path, "");
    }
    for (int i = 0; i < env->num_agents; i++) {
        RoClient* c = &env->bots[i].c;
        char user[64], pass[64];
        if (fscanf(f, "%63s %63s", user, pass) != 2) rg_die("%s has too few accounts%s", path, "");
        snprintf(c->user, sizeof c->user, "%.23s", user);
        snprintf(c->pass, sizeof c->pass, "%.23s", pass);
        snprintf(c->host, sizeof c->host, "%s", host ? host : "127.0.0.1");
        const char* port_base = getenv("RO_WORLD_PORT_BASE");
        c->login_port = (port_base ? atoi(port_base) : 7000) + 10 * world;
        c->db_dir = db;
        // A bot of an earlier run may still count as online: retry for up to 40 s.
        int ok = 0;
        for (int tries = 0; tries < 40 && !ok; tries++) {
            ok = ro_login(c, 0) == 0;
            if (!ok) usleep(1000000);
        }
        if (!ok) rg_die("login failed: %s%s", c->err, "");
    }
    fclose(f);
    // Every bot on the map with its status before the first step.
    for (double t0 = ro_ms(); ro_ms() - t0 < 15000;) {
        int ready = 1;
        for (int i = 0; i < env->num_agents; i++) {
            ro_poll(&env->bots[i].c);
            if (env->bots[i].c.state != RO_IN_GAME) ready = 0;
        }
        if (ready) break;
        usleep(10000);
    }
    for (int i = 0; i < env->num_agents; i++) env->bots[i].c.lockstep = env->lockstep;
}

void puf_reset(Env* env) {
    if (env->lockstep) {
        double t0 = ro_ms();
        if (ro_lockstep_join(env->clients, env->num_agents, 60000) < 0)
            rg_die("world of %s: could not join the step barrier within 60 s%s", env->bots[0].c.user, "");
        if (getenv("RG_DEBUG"))
            fprintf(stderr, "ragnarok env %u: %d bots joined in %.0f ms at step %u\n", env->rng, env->num_agents,
                    ro_ms() - t0, env->bots[0].c.steps_done);
        for (int i = 0; i < env->num_agents; i++) ro_chat(&env->bots[i].c, "@rlreset");
    }
    rg_world_step(env);
    for (int i = 0; i < env->num_agents; i++) {
        RgBot* b = &env->bots[i];
        rg_begin_episode(b);
        b->base_seen = ro_total_base_exp(&b->c);
        b->job_seen = ro_total_job_exp(&b->c);
        b->deaths_seen = b->c.deaths;
        rg_observe(env, i);
        env->agents[i].rewards[0] = 0;
        env->agents[i].terminals[0] = 0;
    }
}

void puf_step(Env* env) {
    for (int i = 0; i < env->num_agents; i++) {
        rg_act(&env->bots[i], env->agents[i].actions);
        rg_continue(&env->bots[i]);
    }
    rg_world_step(env);
    for (int i = 0; i < env->num_agents; i++) {
        RgBot* b = &env->bots[i];
        RoClient* c = &b->c;
        int64_t base = ro_total_base_exp(c), job = ro_total_job_exp(c);
        float r = 0;
        if (c->next_base_exp > 0) r += (float)(base - b->base_seen) / c->next_base_exp;
        if (c->next_job_exp > 0) r += env->reward_job * (float)(job - b->job_seen) / c->next_job_exp;
        int died = c->deaths > b->deaths_seen;
        if (died) r += env->reward_death;
        r *= env->reward_scale;
        b->base_seen = base;
        b->job_seen = job;
        b->deaths_seen = c->deaths;
        b->ret += r;
        b->steps++;
        env->agents[i].rewards[0] = r;
        int done = died || (env->episode_steps > 0 && b->steps >= env->episode_steps);
        env->agents[i].terminals[0] = done;
        if (done) rg_end_episode(env, b);
        rg_observe(env, i);
    }
}

// RG_HEADLESS: no window, for tools that do not link raylib.
#ifdef RG_HEADLESS
void puf_render(Env* env) {
    (void)env;
}
#else
void puf_render(Env* env) {
    const int cell = 24, size = RG_VIEW * cell;
    if (!IsWindowReady()) {
        InitWindow(size, size + 40, "Ragnarok: agent 0");
        SetTargetFPS(10);
    }
    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    RoClient* c = &env->bots[0].c;
    int x, y;
    ro_self_pos(c, &x, &y);
    BeginDrawing();
    ClearBackground(CLITERAL(Color){20, 20, 28, 255});
    for (int dy = 0; dy < RG_VIEW; dy++)
        for (int dx = 0; dx < RG_VIEW; dx++)
            if (ro_walkable(c->cells, x + dx - RG_VIEW / 2, y + dy - RG_VIEW / 2))
                DrawRectangle(dx * cell, (RG_VIEW - 1 - dy) * cell, cell - 1, cell - 1, CLITERAL(Color){60, 70, 60, 255});
    for (int g = 0; g < c->nground; g++) {
        int dx = c->ground[g].x - x + RG_VIEW / 2, dy = c->ground[g].y - y + RG_VIEW / 2;
        if (dx >= 0 && dy >= 0 && dx < RG_VIEW && dy < RG_VIEW)
            DrawRectangle(dx * cell + 8, (RG_VIEW - 1 - dy) * cell + 8, 8, 8, GOLD);
    }
    for (int u = 0; u < c->nunits; u++) {
        int ux, uy;
        ro_unit_pos(c, &c->units[u], &ux, &uy);
        int dx = ux - x + RG_VIEW / 2, dy = uy - y + RG_VIEW / 2;
        if (dx >= 0 && dy >= 0 && dx < RG_VIEW && dy < RG_VIEW)
            DrawRectangle(dx * cell + 3, (RG_VIEW - 1 - dy) * cell + 3, cell - 7, cell - 7,
                          c->units[u].type == RO_OBJ_MOB ? RED : SKYBLUE);
    }
    DrawRectangle((RG_VIEW / 2) * cell + 2, (RG_VIEW / 2) * cell + 2, cell - 5, cell - 5, BLUE);
    DrawText(TextFormat("lv %d/%d  hp %d/%d  kills %d  %s %d,%d", c->base_level, c->job_level, c->hp, c->maxhp,
                        c->kills, c->map, x, y), 8, size + 10, 18, RAYWHITE);
    EndDrawing();
}
#endif

void puf_close(Env* env) {
    for (int i = 0; i < env->num_agents; i++)
        if (env->bots[i].c.fd >= 0) close(env->bots[i].c.fd);
    free(env->bots);
    free(env->clients);
    free(env->agents);
#ifndef RG_HEADLESS
    if (IsWindowReady()) CloseWindow();
#endif
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "base_level", log->base_level);
    dict_set(out, "job_level", log->job_level);
    dict_set(out, "kills", log->kills);
    dict_set(out, "deaths", log->deaths);
    dict_set(out, "potions", log->potions);
    dict_set(out, "picked", log->picked);
    dict_set(out, "base_exp", log->base_exp);
    dict_set(out, "n", log->n);
}
