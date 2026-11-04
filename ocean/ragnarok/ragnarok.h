// ragnarok: one goal-conditioned model for Ragnarok Online tasks, played on real rAthena servers.
//
// Worlds. Each env drives one throwaway world from slurm/world.sh with num_agents bots, which
// puf_init logs in through client/rathena_client.h. Env i uses world i (127.0.0.1, ports
// $RO_WORLD_PORT_BASE+10i, default 7000), the accounts in $RO_WORLD_ROOT/w<i>/accounts and
// rAthena's db/ in $RO_DB_DIR. One env per world: two envs on one world would wait for each other's
// bots. Each puf_init claims the next block of num_agents accounts (w<i>/next_block), since rAthena
// keeps a just-logged-out account "online" for up to 30 s and a sweep starts a new process per trial.
//
// Steps. A step is 100 ms of game time. With lockstep = 1 (training worlds) it is one step of the
// world: the env turns every action into requests, sends READY for every bot, and waits for the
// server's DONE. With lockstep = 0 (a real-time server) it waits 100 ms of wall time.
//
// Tasks. Every episode, each bot draws a task by the taskN_weight settings, and the training world's
// `@rlreset <task>` (server/npc/rl_train.txt) sets its character up for it. The observation starts
// with a goal vector (the task, its difficulty, the job), so one model learns every task and can be
// told which to do. Reward: reward_scale x (base exp gained / exp the level needs + reward_job x the
// same for job exp + reward_death on death), scaled by the task's baseline reward rate so that every
// task pays the same at its scripted bot's level. perf is exp per step over the scripted bot's
// (1.0 = as good as it), logged per task.
//
// Layout v2 is fixed (changing it invalidates checkpoints), so it has spare room; see the RG_OBS_*
// offsets. Actions: verb, direction (for move), slot (a monster or item slot, for attack, pick up and
// targeted skills) and skill (an index into RG_SKILLS). Attacking and picking up go on over the next
// steps: the env walks the bot into range first, since rAthena never walks a player there. Action
// masks rule out reserved verbs, empty slots and skills the bot cannot use right now.

#include <poll.h>
#include <sys/file.h>

#include "client/rathena_client.h"
#include "client/rathena_mobs.h"

typedef float obs_t;
#include "pufferenv.h"

// ---- tasks -----------------------------------------------------------------------------------

#define RG_TASKS_MAX 16  // goal one-hot width, with room for later tasks
enum { RG_TASK_FIELD, RG_TASK_SWORD_SKILLS, RG_TASK_SURVIVAL, RG_NTASKS };
static const char* RG_TASK_NAMES[RG_NTASKS] = {"field", "sword_skills", "survival"};
// Each task's scripted policy (ragnarok_check, 64 bots, 6,000 steps, job 5005277): its base exp
// per step, against which perf is measured, and its reward per step (level-normalized exp, deaths
// included, reward_scale 1), by which rewards are normalized so every task pays the same there.
static const float RG_TASK_BASELINE[RG_NTASKS] = {0.0312f, 0.419f, 1.19f};
static const float RG_TASK_REWARD[RG_NTASKS] = {0.00246f, 0.00017f, 0.00097f};

// ---- layout v2 -------------------------------------------------------------------------------

#define RG_VIEW 15
#define RG_MOBS 8
#define RG_ITEMS 4
#define RG_NSKILLS 24
#define RG_PARTY 6
#define RG_SIGHT 14.0f

#define RG_GOAL_F (RG_TASKS_MAX + 4 + 8)  // task one-hot, 4 difficulty values, job one-hot
#define RG_SELF_F 20
#define RG_CAST_F 4
#define RG_SKILL_F 5
#define RG_MOB_F 21
#define RG_ITEM_F 4
#define RG_PARTY_F 4
#define RG_DIALOG_F 8

#define RG_OBS_GOAL 0
#define RG_OBS_SELF (RG_OBS_GOAL + RG_GOAL_F)
#define RG_OBS_CAST (RG_OBS_SELF + RG_SELF_F)
#define RG_OBS_VIEW (RG_OBS_CAST + RG_CAST_F)
#define RG_OBS_SKILLS (RG_OBS_VIEW + RG_VIEW * RG_VIEW)
#define RG_OBS_MOBS (RG_OBS_SKILLS + RG_NSKILLS * RG_SKILL_F)
#define RG_OBS_ITEMS (RG_OBS_MOBS + RG_MOBS * RG_MOB_F)
#define RG_OBS_PARTY (RG_OBS_ITEMS + RG_ITEMS * RG_ITEM_F)
#define RG_OBS_DIALOG (RG_OBS_PARTY + RG_PARTY * RG_PARTY_F)
#define OBS_SIZE (RG_OBS_DIALOG + RG_DIALOG_F)

#define RG_VERBS 16
#define ACT_SIZES {RG_VERBS, 8, 8, RG_NSKILLS}
#define NUM_ATNS 4
#define RG_MASK_SIZE (RG_VERBS + 8 + 8 + RG_NSKILLS)
enum { RG_WAIT, RG_MOVE, RG_ATTACK, RG_PICKUP, RG_POTION, RG_SIT, RG_SKILL };
enum { RG_NONE, RG_ATTACKING, RG_PICKING };

// First-class skills the model can see and use (rAthena skill ids); 0 = a reserved slot.
static const int RG_SKILLS[RG_NSKILLS] = {
    5, 7, 6, 8,                  // Swordman: Bash, Magnum Break, Provoke, Endure
    19, 14, 20, 13, 11, 17, 15,  // Mage: Fire, Cold and Lightning Bolt, Soul Strike, Napalm Beat,
                                 //       Fire Ball, Frost Diver
    28, 34, 29,                  // Acolyte: Heal, Blessing, Increase AGI
    46, 47,                      // Archer: Double Strafe, Arrow Shower
    52, 50, 51,                  // Thief: Envenom, Steal, Hiding
    42,                          // Merchant: Mammonite
    142,                         // Novice: First Aid
    0, 0, 0};

#define RG_RED_POTION 501
static const int RG_DX[8] = {1, 1, 0, -1, -1, -1, 0, 1};
static const int RG_DY[8] = {0, 1, 1, 1, 0, -1, -1, -1};
// Status points go by a fixed rule: STR, AGI, VIT, INT, DEX, LUK weights per job (0 Novice,
// 1 Swordman, 2 Mage, 3 Archer, 4 Acolyte, 5 Merchant, 6 Thief).
static const int RG_STAT_WEIGHTS[7][6] = {
    {3, 2, 2, 0, 1, 0}, {3, 2, 2, 0, 1, 0}, {0, 1, 1, 3, 2, 0}, {1, 2, 1, 0, 3, 0},
    {1, 1, 2, 3, 1, 0}, {3, 1, 2, 0, 1, 0}, {2, 3, 1, 0, 1, 0}};

#define RG_LOG_TASKS 8
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
    float skills;
    float mobs_seen;  // monsters in sight per step
    float edge;       // distance to the nearest map edge, as a fraction of half the map
    float task_perf[RG_LOG_TASKS];
    float task_n[RG_LOG_TASKS];
    float n;
};

typedef struct RgBot {
    RoClient c;
    int task, reset_wait;  // reset_wait: steps left before a fresh @rlreset has landed
    int intent, attacking;
    uint32_t target, item;
    double intent_since, last_attack, last_pickup;
    uint32_t slot_mob[RG_MOBS], slot_item[RG_ITEMS];
    int64_t base_seen, job_seen;  // cumulative exp at the end of the last step
    int deaths_seen;
    // This episode.
    int steps, kills0, potions0, picked0, skills0;
    int64_t base0;
    float ret, mobs_seen, edge;
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
    float task_weight[RG_NTASKS];
    unsigned int seed;
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

static int rg_draw_task(Env* env) {
    float total = 0;
    for (int t = 0; t < RG_NTASKS; t++) total += env->task_weight[t];
    if (total <= 0) return RG_TASK_FIELD;
    float u = (float)rand_r(&env->seed) / RAND_MAX * total;
    for (int t = 0; t < RG_NTASKS; t++) {
        u -= env->task_weight[t];
        if (u <= 0 && env->task_weight[t] > 0) return t;
    }
    return RG_NTASKS - 1;
}

static void rg_world_step(Env* env) {
    if (env->lockstep) {
        if (ro_lockstep_step_all(env->clients, env->num_agents, 60000) < 0)
            rg_die("world of %s: no DONE within 60 s of wall time%s", env->bots[0].c.user, "");
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

static int rg_job_slot(int job) { return job >= 0 && job <= 6 ? job : 7; }

// Can the bot cast RG_SKILLS[k] right now? Returns its level, or 0.
static int rg_skill_ready(const RoClient* c, int k) {
    if (!RG_SKILLS[k] || c->cast_until > c->now) return 0;
    const RoSkill* s = ro_skill_find(c, RG_SKILLS[k]);
    if (!s || s->sp > c->sp || s->ready_at > c->now) return 0;
    return s->level;
}

// The verb sets what the bot is doing; attacking and picking up carry on in rg_continue.
static void rg_act(RgBot* b, const float* a) {
    RoClient* c = &b->c;
    if (c->state != RO_IN_GAME || c->dead || b->reset_wait) return;
    int verb = (int)a[0], dir = (int)a[1], slot = (int)a[2], skill = (int)a[3];
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
    case RG_SKILL: {
        if (skill < 0 || skill >= RG_NSKILLS) break;
        int level = rg_skill_ready(c, skill);
        const RoSkill* s = level ? ro_skill_find(c, RG_SKILLS[skill]) : NULL;
        if (!s) break;
        if (c->sitting) ro_sit(c, 0);
        RoUnit* u = slot >= 0 && slot < RG_MOBS && b->slot_mob[slot] ? ro_unit_find(c, b->slot_mob[slot]) : NULL;
        if (s->inf & 1) {  // on an enemy: the monster in the slot
            if (!u) break;
            ro_use_skill(c, s->id, level, u->id);
            b->intent = RG_ATTACKING;
            b->target = u->id;
            b->intent_since = c->now;
        } else if (s->inf & 2) {  // on the ground: the monster's cell, else under the bot
            int tx = x, ty = y;
            if (u) ro_unit_pos(c, u, &tx, &ty);
            ro_use_skill_pos(c, s->id, level, tx, ty);
        } else {  // on itself (self and support skills, for now)
            ro_use_skill(c, s->id, level, c->aid);
        }
        b->attacking = 0;  // a skill interrupts the attack loop; rg_continue resumes it
        break;
    }
    default:
        break;
    }
}

static void rg_continue(RgBot* b) {
    RoClient* c = &b->c;
    if (c->state != RO_IN_GAME || c->dead || b->reset_wait || b->intent == RG_NONE) return;
    if (c->cast_until > c->now) return;  // let the cast finish
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

static void rg_observe(Env* env, int i) {
    RgBot* b = &env->bots[i];
    RoClient* c = &b->c;
    float* o = env->agents[i].observations;
    unsigned char* mask = env->agents[i].action_mask;
    memset(o, 0, OBS_SIZE * sizeof(float));
    memset(b->slot_mob, 0, sizeof b->slot_mob);
    memset(b->slot_item, 0, sizeof b->slot_item);

    // Goal: the task one-hot, its difficulty (the first tasks have none yet), the job.
    o[RG_OBS_GOAL + b->task] = 1;
    o[RG_OBS_GOAL + RG_TASKS_MAX + 4 + rg_job_slot(c->job)] = 1;

    const RoInv* potion = NULL;
    if (c->state == RO_IN_GAME) {
        int x, y;
        ro_self_pos(c, &x, &y);
        potion = ro_inv_find(c, RG_RED_POTION);
        float* s = o + RG_OBS_SELF;
        s[0] = c->maxhp > 0 ? (float)c->hp / c->maxhp : 0;
        s[1] = c->maxsp > 0 ? (float)c->sp / c->maxsp : 0;
        s[2] = c->base_level / 50.0f;
        s[3] = c->job_level / 50.0f;
        s[4] = c->next_base_exp > 0 ? (float)c->base_exp / c->next_base_exp : 0;
        s[5] = c->next_job_exp > 0 ? (float)c->job_exp / c->next_job_exp : 0;
        s[6] = c->max_weight > 0 ? (float)c->weight / c->max_weight : 0;
        s[7] = c->sitting;
        s[8] = c->dead;
        s[9] = potion ? potion->amount / 30.0f : 0;
        s[10] = ro_self_moving(c);
        s[11] = b->intent == RG_ATTACKING;
        s[12] = b->intent == RG_PICKING;
        s[13] = env->episode_steps > 0 ? (float)b->steps / env->episode_steps : 0;
        s[14] = c->maxhp / 1000.0f;
        s[15] = c->maxsp / 500.0f;
        // Where on the map, and how many monsters are in sight: without these, bots sharing one
        // policy can all drift the same way into a corner.
        if (c->cells) {
            s[16] = (float)x / c->cells->xs;
            s[17] = (float)y / c->cells->ys;
        }
        int seen = 0;
        for (int u = 0; u < c->nunits; u++) seen += c->units[u].type == RO_OBJ_MOB;
        s[18] = seen / 8.0f;
        float* k = o + RG_OBS_CAST;
        k[0] = c->cast_until > c->now;
        k[1] = c->cast_until > c->now ? (float)(c->cast_until - c->now) / 2000.0f : 0;
        k[2] = c->skill_failed_at > 0 && c->now - c->skill_failed_at < 1000;
        k[3] = c->status_points / 50.0f;
        float* view = o + RG_OBS_VIEW;
        for (int dy = 0; dy < RG_VIEW; dy++)
            for (int dx = 0; dx < RG_VIEW; dx++)
                view[dx + dy * RG_VIEW] = ro_walkable(c->cells, x + dx - RG_VIEW / 2, y + dy - RG_VIEW / 2);
        for (int sk = 0; sk < RG_NSKILLS; sk++) {
            const RoSkill* s2 = RG_SKILLS[sk] ? ro_skill_find(c, RG_SKILLS[sk]) : NULL;
            if (!s2) continue;
            float* f = o + RG_OBS_SKILLS + sk * RG_SKILL_F;
            f[0] = 1;
            f[1] = s2->level / 10.0f;
            f[2] = s2->sp / 50.0f;
            f[3] = s2->sp <= c->sp;
            f[4] = rg_skill_ready(c, sk) > 0;
        }

        // The nearest monsters, nearest first, by insertion into a short sorted list.
        int order[RG_MOBS], dists[RG_MOBS], n = 0;
        for (int u = 0; u < c->nunits; u++) {
            if (c->units[u].type != RO_OBJ_MOB) continue;
            int ux, uy;
            ro_unit_pos(c, &c->units[u], &ux, &uy);
            int d = rg_dist(x, y, ux, uy), at = n < RG_MOBS ? n++ : RG_MOBS;
            if (at == RG_MOBS && d >= dists[RG_MOBS - 1]) continue;
            if (at == RG_MOBS) at = RG_MOBS - 1;
            while (at > 0 && dists[at - 1] > d) {
                order[at] = order[at - 1];
                dists[at] = dists[at - 1];
                at--;
            }
            order[at] = u;
            dists[at] = d;
        }
        for (int m = 0; m < n; m++) {
            const RoUnit* u = &c->units[order[m]];
            const RoMobInfo* info = ro_mob_info(u->cls);
            float* f = o + RG_OBS_MOBS + m * RG_MOB_F;
            int ux, uy;
            ro_unit_pos(c, u, &ux, &uy);
            f[0] = 1;
            f[1] = (ux - x) / RG_SIGHT;
            f[2] = (uy - y) / RG_SIGHT;
            f[3] = dists[m] / RG_SIGHT;
            f[4] = u->maxhp > 0 ? (float)u->hp / u->maxhp : 1;
            f[5] = u->last_hit_me && c->now - u->last_hit_me < 3000;
            f[6] = b->intent == RG_ATTACKING && b->target == u->id;
            f[7] = u->boss || (info && info->boss);
            if (info) {
                f[8] = info->level / 100.0f;
                f[9] = log10f((float)info->hp + 1) / 6.0f;
                f[10] = info->aggressive;
                f[11 + (info->element < 10 ? info->element : 0)] = 1;
            }
            b->slot_mob[m] = u->id;
        }
        int used[RO_MAX_GROUND] = {0};
        for (int k2 = 0; k2 < RG_ITEMS; k2++) {
            int best = -1, best_d = 1 << 30;
            for (int g = 0; g < c->nground; g++) {
                int d = rg_dist(x, y, c->ground[g].x, c->ground[g].y);
                if (!used[g] && d < best_d) best = g, best_d = d;
            }
            if (best < 0 || best_d > RG_SIGHT) break;
            used[best] = 1;
            float* it = o + RG_OBS_ITEMS + k2 * RG_ITEM_F;
            it[0] = 1;
            it[1] = (c->ground[best].x - x) / RG_SIGHT;
            it[2] = (c->ground[best].y - y) / RG_SIGHT;
            it[3] = best_d / RG_SIGHT;
            b->slot_item[k2] = c->ground[best].id;
        }
    }

    // Masks: verbs in use; slots holding a monster or an item; skills the bot can cast now. Each
    // head keeps at least one allowed option.
    if (mask) {
        memset(mask, 0, RG_MASK_SIZE);
        unsigned char* verbs = mask;
        unsigned char* dirs = mask + RG_VERBS;
        unsigned char* slots = dirs + 8;
        unsigned char* skills = slots + 8;
        int any_slot = 0, any_skill = 0;
        for (int k2 = 0; k2 < 8; k2++) {
            slots[k2] = b->slot_mob[k2] || (k2 < RG_ITEMS && b->slot_item[k2]);
            any_slot |= slots[k2];
        }
        for (int sk = 0; sk < RG_NSKILLS; sk++) {
            skills[sk] = c->state == RO_IN_GAME && rg_skill_ready(c, sk) > 0;
            any_skill |= skills[sk];
        }
        if (!any_slot) slots[0] = 1;
        if (!any_skill) skills[0] = 1;
        for (int d = 0; d < 8; d++) dirs[d] = 1;
        verbs[RG_WAIT] = verbs[RG_MOVE] = verbs[RG_SIT] = 1;
        verbs[RG_ATTACK] = b->slot_mob[0] != 0;
        verbs[RG_PICKUP] = b->slot_item[0] != 0;
        verbs[RG_POTION] = potion != NULL;
        verbs[RG_SKILL] = any_skill;
    }
}

// Start counting an episode once its @rlreset has landed.
static void rg_begin_episode(RgBot* b) {
    RoClient* c = &b->c;
    b->steps = 0;
    b->ret = 0;
    b->mobs_seen = b->edge = 0;
    b->kills0 = c->kills;
    b->potions0 = c->potions;
    b->picked0 = c->picked;
    b->skills0 = c->skills_used;
    b->base0 = b->base_seen = ro_total_base_exp(c);
    b->job_seen = ro_total_job_exp(c);
    b->deaths_seen = c->deaths;
    b->intent = RG_NONE;
    b->attacking = 0;
}

// Draw the bot's next task and send its reset; the reset lands during the next step.
static void rg_start_reset(Env* env, RgBot* b) {
    b->task = rg_draw_task(env);
    if (env->lockstep) {
        char cmd[32];
        snprintf(cmd, sizeof cmd, "@rlreset %d", b->task);
        ro_chat(&b->c, cmd);
        b->reset_wait = 2;
    } else {
        if (b->c.dead) ro_respawn(&b->c);  // a real-time server has no @rlreset
        b->reset_wait = 0;
        rg_begin_episode(b);
    }
}

static void rg_end_episode(Env* env, RgBot* b) {
    RoClient* c = &b->c;
    float exp_per_step = b->steps > 0 ? (float)(ro_total_base_exp(c) - b->base0) / b->steps : 0;
    float perf = exp_per_step / RG_TASK_BASELINE[b->task];
    env->log.perf += perf;
    env->log.score += b->ret;
    env->log.episode_length += b->steps;
    env->log.base_level += c->base_level;
    env->log.job_level += c->job_level;
    env->log.kills += c->kills - b->kills0;
    env->log.potions += c->potions - b->potions0;
    env->log.picked += c->picked - b->picked0;
    env->log.skills += c->skills_used - b->skills0;
    env->log.base_exp += (float)(ro_total_base_exp(c) - b->base0);
    env->log.mobs_seen += b->steps > 0 ? b->mobs_seen / b->steps : 0;
    env->log.edge += b->steps > 0 ? b->edge / b->steps : 0;
    env->log.deaths += c->dead;
    if (b->task < RG_LOG_TASKS) {
        env->log.task_perf[b->task] += perf;
        env->log.task_n[b->task] += 1;
    }
    env->log.n += 1;
    rg_start_reset(env, b);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = dict_get(kwargs, "num_agents");
    env->lockstep = dict_get(kwargs, "lockstep");
    env->episode_steps = dict_get(kwargs, "episode_steps");
    env->reward_scale = dict_get(kwargs, "reward_scale");
    env->reward_job = dict_get(kwargs, "reward_job");
    env->reward_death = dict_get(kwargs, "reward_death");
    for (int t = 0; t < RG_NTASKS; t++) {
        char key[32];
        snprintf(key, sizeof key, "task%d_weight", t);
        env->task_weight[t] = dict_get(kwargs, key);
    }
    env->seed = 12345u + 7919u * env->rng;
    env->agents = (Agent*)calloc(env->num_agents, sizeof(Agent));
    env->bots = (RgBot*)calloc(env->num_agents, sizeof(RgBot));
    env->clients = (RoClient**)calloc(env->num_agents, sizeof(RoClient*));
    for (int i = 0; i < env->num_agents; i++) {
        env->clients[i] = &env->bots[i].c;
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
    const char* port_base = getenv("RO_WORLD_PORT_BASE");
    for (int i = 0; i < env->num_agents; i++) {
        RoClient* c = &env->bots[i].c;
        char user[64], pass[64];
        if (fscanf(f, "%63s %63s", user, pass) != 2) rg_die("%s has too few accounts%s", path, "");
        snprintf(c->user, sizeof c->user, "%.23s", user);
        snprintf(c->pass, sizeof c->pass, "%.23s", pass);
        snprintf(c->host, sizeof c->host, "%s", host ? host : "127.0.0.1");
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
    }
    for (int i = 0; i < env->num_agents; i++) rg_start_reset(env, &env->bots[i]);
    for (int k = 0; k < 2; k++) rg_world_step(env);
    for (int i = 0; i < env->num_agents; i++) {
        RgBot* b = &env->bots[i];
        b->reset_wait = 0;
        rg_begin_episode(b);
        rg_observe(env, i);
        env->agents[i].rewards[0] = 0;
        env->agents[i].terminals[0] = 0;
    }
}

void puf_step(Env* env) {
    for (int i = 0; i < env->num_agents; i++) {
        RgBot* b = &env->bots[i];
        rg_act(b, env->agents[i].actions);
        rg_continue(b);
        if (!b->reset_wait && b->c.status_points > 0 && b->c.state == RO_IN_GAME)
            ro_spend_status(&b->c, RG_STAT_WEIGHTS[b->c.job >= 0 && b->c.job <= 6 ? b->c.job : 0]);
    }
    rg_world_step(env);
    for (int i = 0; i < env->num_agents; i++) {
        RgBot* b = &env->bots[i];
        RoClient* c = &b->c;
        float r = 0;
        int done = 0;
        if (b->reset_wait) {
            // The reset is landing: no reward, and the episode starts counting once it has.
            if (--b->reset_wait == 0) rg_begin_episode(b);
        } else {
            int64_t base = ro_total_base_exp(c), job = ro_total_job_exp(c);
            if (c->next_base_exp > 0) r += (float)(base - b->base_seen) / c->next_base_exp;
            if (c->next_job_exp > 0) r += env->reward_job * (float)(job - b->job_seen) / c->next_job_exp;
            // Exp is scaled per task so every task pays the same at its baseline; a death costs the
            // same everywhere.
            r *= RG_TASK_REWARD[RG_TASK_FIELD] / RG_TASK_REWARD[b->task];
            int died = c->deaths > b->deaths_seen;
            if (died) r += env->reward_death;
            r *= env->reward_scale;
            b->base_seen = base;
            b->job_seen = job;
            b->deaths_seen = c->deaths;
            b->ret += r;
            b->steps++;
            int x, y, seen = 0;
            ro_self_pos(c, &x, &y);
            for (int u = 0; u < c->nunits; u++) seen += c->units[u].type == RO_OBJ_MOB;
            b->mobs_seen += seen;
            if (c->cells) {
                int ex = x < c->cells->xs - 1 - x ? x : c->cells->xs - 1 - x;
                int ey = y < c->cells->ys - 1 - y ? y : c->cells->ys - 1 - y;
                b->edge += 2.0f * (ex < ey ? ex : ey) / (c->cells->xs < c->cells->ys ? c->cells->xs : c->cells->ys);
            }
            done = died || (env->episode_steps > 0 && b->steps >= env->episode_steps);
        }
        env->agents[i].rewards[0] = r;
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
    DrawText(TextFormat("%s  lv %d/%d  hp %d/%d  sp %d  kills %d  %s %d,%d", RG_TASK_NAMES[env->bots[0].task],
                        c->base_level, c->job_level, c->hp, c->maxhp, c->sp, c->kills, c->map, x, y),
             8, size + 10, 16, RAYWHITE);
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
    dict_set(out, "skills", log->skills);
    dict_set(out, "mobs_seen", log->mobs_seen);
    dict_set(out, "edge", log->edge);
    // Per task: the trainer divides every field by n, which leaves these ratios as they are.
    for (int t = 0; t < RG_NTASKS && t < RG_LOG_TASKS; t++) {
        if (log->task_n[t] <= 0) continue;
        char key[48];
        snprintf(key, sizeof key, "%s_perf", RG_TASK_NAMES[t]);
        dict_set(out, key, log->task_perf[t] / log->task_n[t]);
        snprintf(key, sizeof key, "%s_share", RG_TASK_NAMES[t]);
        dict_set(out, key, log->task_n[t]);
    }
    dict_set(out, "n", log->n);
}
