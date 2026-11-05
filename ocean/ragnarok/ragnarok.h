// ragnarok: one goal-conditioned model for Ragnarok Online tasks, played on real rAthena servers.
//
// Worlds. Each env drives one throwaway world from slurm/world.sh with num_agents bots, which
// puf_init logs in through client/rathena_client.h. Env i uses world i (127.0.0.1, ports
// $RO_WORLD_PORT_BASE+10i, default 7000), the accounts in $RO_WORLD_ROOT/w<i>/accounts and
// rAthena's db/ in $RO_DB_DIR. One env per world: two envs on one world would wait for each other's
// bots. Each puf_init claims the next block of num_agents accounts (w<i>/next_block, wrapping around),
// since rAthena keeps a just-logged-out account "online" for up to 30 s and a sweep starts a new
// process per trial.
//
// Steps. A world step is 100 ms of game time. With lockstep = 1 (training worlds) the env turns
// every action into requests, sends READY for every bot, and waits for the server's DONE; with
// lockstep = 0 (a real-time server) it waits 100 ms of wall time. One env step (one decision) spans
// action_repeat world steps: the verb is sent once, attacks and pick-ups go on, rewards add up.
// Episode lengths, perf and the time-in-episode input count world steps.
//
// Tasks. Every episode, each bot draws a task by the taskN_weight settings, and the training world's
// `@rlreset <task>` (server/npc/rl_train.txt) sets its character up for it. The observation starts
// with a goal vector (the task, its difficulty, the job), so one model learns every task and can be
// told which to do. Reward: reward_scale x (base exp gained / exp the level needs + reward_job x the
// same for job exp, scaled by the task's baseline reward rate so that every task pays the same at
// its scripted bot's level, + reward_death on death). perf is exp per step over the scripted bot's
// (1.0 = as good as it), logged per task. Navigation pays and scores the way to its target instead
// (see RG_NAV_MIN).
//
// Layout v3 is fixed (changing it invalidates checkpoints), so it has spare room; see the RG_OBS_*
// offsets. v3 added the effects on the bot (status icons and ailments), monsters' ailments, more
// difficulty values and a spare block. Actions: verb, direction (for move), slot (a monster or item slot, for attack, pick up and
// targeted skills) and skill (an index into RG_SKILLS). Attacking and picking up go on over the next
// steps: the env walks the bot into range first, since rAthena never walks a player there. A skill
// on a monster does not start an attack. The monster last attacked or cast at stays in slot 0, so
// acting on slot 0 again keeps at it. Action masks rule out reserved verbs, empty slots, skills the
// bot cannot use right now, potions above 70% HP and sitting while busy.

#include <poll.h>
#include <sys/file.h>

#include "client/rathena_client.h"
#ifdef RO_MOBS_H
#include RO_MOBS_H
#else
#include "client/rathena_mobs.h"
#endif

typedef float obs_t;
#include "pufferenv.h"

// ---- tasks -----------------------------------------------------------------------------------

#define RG_TASKS_MAX 16  // goal one-hot width, with room for later tasks
// Field leveling (a Novice), the first classes' skills (each at levels 25/25, on a field for its
// level), survival (a weak Swordman among Peco Pecos) and navigation (a Novice walking to a target cell);
// server/npc/rl_train.txt sets each one up.
enum {
    RG_TASK_FIELD, RG_TASK_SWORD_SKILLS, RG_TASK_SURVIVAL, RG_TASK_MAGE_SKILLS, RG_TASK_ARCHER_SKILLS,
    RG_TASK_ACOLYTE_SKILLS, RG_TASK_THIEF_SKILLS, RG_TASK_MERCHANT_SKILLS, RG_TASK_NAVIGATION, RG_NTASKS
};
static const char* RG_TASK_NAMES[RG_NTASKS] = {"field",       "sword_skills",  "survival",
                                               "mage_skills", "archer_skills", "acolyte_skills",
                                               "thief_skills", "merchant_skills", "navigation"};
// Each task's scripted policy (ragnarok_check, 64 bots): its base exp per step, against which perf
// is measured, and the exp part of its reward per step (level-normalized exp, reward_scale 1), by
// which the exp reward is normalized so that every task pays the same there; a death costs the same
// everywhere. Job 5007175 measured all of them on renewal (client 20221005, 12,000 steps per task),
// so perf does not compare with the pre-renewal G0 and G1 runs.
static const float RG_TASK_BASELINE[RG_NTASKS] = {2.314f, 2.293f, 2.379f, 2.246f, 3.541f,
                                                  2.646f, 2.944f, 2.958f, 0.394f};
static const float RG_TASK_REWARD[RG_NTASKS] = {0.005121f, 0.000218f, 0.000428f, 0.000230f, 0.000361f,
                                                0.000251f, 0.000303f, 0.000300f, 0.005184f};
// Navigation: a target the bot can walk to, RG_NAV_MIN to RG_NAV_MAX cells away (the larger of the
// two axes), reached within 1 cell. Its perf is cells per step on the way there (0 if it never arrives) over the
// scripted bot's; its reward is the progress toward the target (adding up to 1 on the way) plus 1
// on arrival, instead of exp. The goal's difficulty values hold the way there: target minus
// position in x and y, and the distance, in units of RG_NAV_UNIT cells.
#define RG_NAV_MIN 10
#define RG_NAV_MAX 120
#define RG_NAV_UNIT 100.0f
// The curriculum's prerequisites: the tasks (bits) each task builds on. The class skills and
// navigation build on field leveling, survival on the Swordman skills.
static const unsigned RG_TASK_NEEDS[RG_NTASKS] = {0, 1u << RG_TASK_FIELD, 1u << RG_TASK_SWORD_SKILLS,
                                                  1u << RG_TASK_FIELD, 1u << RG_TASK_FIELD,
                                                  1u << RG_TASK_FIELD, 1u << RG_TASK_FIELD,
                                                  1u << RG_TASK_FIELD, 1u << RG_TASK_FIELD};

// ---- layout v3 -------------------------------------------------------------------------------

#define RG_VIEW 15
#define RG_MOBS 8
#define RG_ITEMS 4
#define RG_NSKILLS 24
#define RG_PARTY 6
#define RG_EFFECTS 32  // effects on the bot it can see: RG_EFFECT_EFST, then the ailments, then room
#define RG_SIGHT 14.0f

#define RG_GOAL_DIFF 8                          // the task's difficulty values
#define RG_GOAL_JOB (RG_TASKS_MAX + RG_GOAL_DIFF)  // where the job one-hot starts
#define RG_GOAL_F (RG_GOAL_JOB + 8)             // task one-hot, difficulty values, job one-hot
#define RG_SELF_F 24
#define RG_CAST_F 4
#define RG_EFFECT_F 2  // on, and the share of its time left
#define RG_SKILL_F 5
#define RG_MOB_F 27
#define RG_ITEM_F 4
#define RG_PARTY_F 4
#define RG_DIALOG_F 8
#define RG_SPARE_F 32  // room for inputs to come

#define RG_OBS_GOAL 0
#define RG_OBS_SELF (RG_OBS_GOAL + RG_GOAL_F)
#define RG_OBS_CAST (RG_OBS_SELF + RG_SELF_F)
#define RG_OBS_EFFECTS (RG_OBS_CAST + RG_CAST_F)
#define RG_OBS_VIEW (RG_OBS_EFFECTS + RG_EFFECTS * RG_EFFECT_F)
#define RG_OBS_SKILLS (RG_OBS_VIEW + RG_VIEW * RG_VIEW)
#define RG_OBS_MOBS (RG_OBS_SKILLS + RG_NSKILLS * RG_SKILL_F)
#define RG_OBS_ITEMS (RG_OBS_MOBS + RG_MOBS * RG_MOB_F)
#define RG_OBS_PARTY (RG_OBS_ITEMS + RG_ITEMS * RG_ITEM_F)
#define RG_OBS_DIALOG (RG_OBS_PARTY + RG_PARTY * RG_PARTY_F)
#define RG_OBS_SPARE (RG_OBS_DIALOG + RG_DIALOG_F)
#define OBS_SIZE (RG_OBS_SPARE + RG_SPARE_F)

// The status icons the bot sees on itself (rAthena's EFST_* ids), each as on and the share of its
// time left; then the ailments the server shows as body and health states (on only), the same for
// itself and for each monster.
#define RG_EFFECT_ICONS 16
static const int RG_EFFECT_EFST[RG_EFFECT_ICONS] = {
    10, 12, 3, 1, 4,     // Blessing, Increase AGI, Improve Concentration, Endure, Hiding
    9, 0, 13, 8, 11,     // Angelus, Provoke, Decrease AGI, Quagmire, Signum Crucis
    15, 19, 25, 30, 37,  // Impositio Manus, Kyrie Eleison, Over Thrust, Loud Exclamation, a speed potion
    46};                 // the after-cast delay of the skills that show it
enum { RG_AIL_STONE, RG_AIL_FREEZE, RG_AIL_STUN, RG_AIL_SLEEP, RG_AIL_POISON, RG_AIL_OTHER, RG_NAILMENTS };

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

#define RG_LOG_TASKS RG_TASKS_MAX
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
    float mobs_seen;     // monsters in sight per step
    float players_seen;  // other players in sight per step
    float edge;          // distance to the nearest map edge, as a fraction of half the map
    float moved;         // cells walked per step
    // Share of the steps a bot could act on which it chose each verb.
    float act_wait, act_move, act_attack, act_pickup, act_potion, act_sit, act_skill;
    float task_perf[RG_LOG_TASKS];
    float task_n[RG_LOG_TASKS];
    float task_w[RG_LOG_TASKS];  // nonzero for the tasks in the mix, so their keys are always logged
    float task_p[RG_LOG_TASKS];  // the curriculum's chance of drawing each task, when an episode ended
    float curriculum;            // nonzero when the curriculum draws the tasks
    float n;
};

typedef struct RgBot {
    RoClient c;
    int task, reset_wait;  // reset_wait: steps left before a fresh @rlreset has landed
    int intent, attacking;
    uint32_t target, item;
    uint32_t focus;  // the monster last attacked or targeted by a skill: it stays in slot 0
    int nav_x, nav_y, nav_d0, nav_d, arrived;  // navigation: target, starting and last distance
    double intent_since, last_attack, last_pickup;
    uint32_t slot_mob[RG_MOBS], slot_item[RG_ITEMS];
    int64_t base_seen, job_seen;  // cumulative exp at the end of the last step
    int deaths_seen;
    // This episode.
    int steps, kills0, potions0, picked0, skills0;
    int64_t base0;
    float ret, mobs_seen, players_seen, edge, moved;
    int acts, verbs[RG_SKILL + 1], last_x, last_y;
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
    int lockstep, episode_steps, action_repeat;
    float reward_scale, reward_job, reward_death;
    float task_weight[RG_NTASKS];
    int curriculum;
    float curriculum_floor, curriculum_unlock, curriculum_fast, curriculum_slow, curriculum_patience;
    unsigned int seed;
    double next_step;  // real time: when the current step ends
    unsigned char* nav_reached;  // navigation's search: cells within RG_NAV_MAX of a bot
    int* nav_queue;
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

// The curriculum (curriculum = 1). The task weights then say only which tasks are in the mix and how
// much each counts; one sampler, shared by every env of the process, picks each episode's task:
// - a task unlocks, for good, once each task it builds on (RG_TASK_NEEDS) that is in the mix averages
//   perf >= curriculum_unlock, or has had curriculum_patience episodes (so a task that levels off
//   below the bar does not lock out the ones after it);
// - unlocked tasks are drawn in proportion to their learning progress, |fast - slow| for two moving
//   averages of their perf over episodes (rates curriculum_fast and curriculum_slow), times their
//   weight, and each gets at least curriculum_floor of the episodes.
// Envs step in parallel threads, so the sampler updates under a lock.
typedef struct {
    float fast[RG_NTASKS], slow[RG_NTASKS];
    long episodes[RG_NTASKS];
    int unlocked[RG_NTASKS];
} RgCurriculum;
static RgCurriculum rg_curriculum;
static pthread_mutex_t rg_curriculum_lock = PTHREAD_MUTEX_INITIALIZER;

// Each task's chance of being drawn next.
static void rg_task_chances(Env* env, float* p) {
    float total = 0;
    for (int t = 0; t < RG_NTASKS; t++) total += p[t] = env->task_weight[t] > 0 ? env->task_weight[t] : 0;
    if (env->curriculum) {
        RgCurriculum* k = &rg_curriculum;
        pthread_mutex_lock(&rg_curriculum_lock);
        int open = 0;
        float progress = 0;
        for (int t = 0; t < RG_NTASKS; t++) {
            if (p[t] <= 0) continue;
            int ready = 1;
            for (int need = 0; need < RG_NTASKS; need++)
                if ((RG_TASK_NEEDS[t] >> need & 1) && env->task_weight[need] > 0
                    && !(k->unlocked[need] && k->episodes[need] > 0
                         && (k->slow[need] >= env->curriculum_unlock || k->episodes[need] >= env->curriculum_patience)))
                    ready = 0;
            k->unlocked[t] |= ready;
            if (!k->unlocked[t]) {
                p[t] = 0;
                continue;
            }
            open++;
            p[t] *= fabsf(k->fast[t] - k->slow[t]) + 1e-3f;
            progress += p[t];
        }
        pthread_mutex_unlock(&rg_curriculum_lock);
        float floor = open * env->curriculum_floor < 1 ? env->curriculum_floor : 1.0f / (open > 0 ? open : 1);
        total = 0;
        for (int t = 0; t < RG_NTASKS; t++) {
            if (p[t] > 0) p[t] = floor + (1 - floor * open) * p[t] / progress;
            total += p[t];
        }
    }
    for (int t = 0; t < RG_NTASKS; t++) p[t] = total > 0 ? p[t] / total : t == RG_TASK_FIELD;
}

static int rg_draw_task(Env* env) {
    float p[RG_NTASKS];
    rg_task_chances(env, p);
    float u = (float)rand_r(&env->seed) / RAND_MAX;
    int last = RG_TASK_FIELD;
    for (int t = 0; t < RG_NTASKS; t++) {
        if (p[t] <= 0) continue;
        last = t;
        u -= p[t];
        if (u <= 0) return t;
    }
    return last;
}

// An episode of the task ended with this perf: update its moving averages (running means at first).
static void rg_curriculum_add(Env* env, int task, float perf) {
    if (!env->curriculum) return;
    RgCurriculum* k = &rg_curriculum;
    pthread_mutex_lock(&rg_curriculum_lock);
    long n = ++k->episodes[task];
    float fast = env->curriculum_fast > 1.0f / n ? env->curriculum_fast : 1.0f / n;
    float slow = env->curriculum_slow > 1.0f / n ? env->curriculum_slow : 1.0f / n;
    k->fast[task] += fast * (perf - k->fast[task]);
    k->slow[task] += slow * (perf - k->slow[task]);
    pthread_mutex_unlock(&rg_curriculum_lock);
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

// Ailments from a body state (rAthena's OPT1_*: 1 stone, 6 turning to stone, 2 frozen, 3 stunned,
// 4 asleep) and a health state (OPT2_* bits): RG_NAILMENTS flags.
static void rg_ailments(int opt1, int opt2, float* out) {
    out[RG_AIL_STONE] = opt1 == 1 || opt1 == 6;
    out[RG_AIL_FREEZE] = opt1 == 2;
    out[RG_AIL_STUN] = opt1 == 3;
    out[RG_AIL_SLEEP] = opt1 == 4;
    out[RG_AIL_POISON] = (opt2 & (RO_OPT2_POISON | RO_OPT2_DPOISON)) != 0;
    out[RG_AIL_OTHER] = (opt2 & (RO_OPT2_CURSE | RO_OPT2_SILENCE | RO_OPT2_BLIND | RO_OPT2_BLEEDING)) != 0;
}

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
    b->acts++;
    if (verb >= 0 && verb <= RG_SKILL) b->verbs[verb]++;
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
            b->focus = b->target;
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
            // It becomes the focus. A skill does not start an attack (a Mage would walk up to melee
            // after every bolt), but an attack under way moves to the skill's target.
            b->focus = u->id;
            if (b->intent == RG_ATTACKING && b->target != u->id) {
                b->target = u->id;
                b->intent_since = c->now;
            }
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

    // Goal: the task one-hot, its difficulty (navigation's way to the target; the other tasks have
    // none yet), the job.
    o[RG_OBS_GOAL + b->task] = 1;
    o[RG_OBS_GOAL + RG_GOAL_JOB + rg_job_slot(c->job)] = 1;
    if (b->task == RG_TASK_NAVIGATION && c->state == RO_IN_GAME) {
        int x, y;
        ro_self_pos(c, &x, &y);
        float* g = o + RG_OBS_GOAL + RG_TASKS_MAX;
        g[0] = (b->nav_x - x) / RG_NAV_UNIT;
        g[1] = (b->nav_y - y) / RG_NAV_UNIT;
        g[2] = rg_dist(x, y, b->nav_x, b->nav_y) / RG_NAV_UNIT;
    }

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
        // Effects on the bot: status icons with the share of their time left, then ailments.
        float* e = o + RG_OBS_EFFECTS;
        for (int i = 0; i < RG_EFFECT_ICONS; i++) {
            e[i * RG_EFFECT_F] = ro_effect_left(c, RG_EFFECT_EFST[i]) > 0;
            e[i * RG_EFFECT_F + 1] = ro_effect_share(c, RG_EFFECT_EFST[i]);
        }
        float ail[RG_NAILMENTS];
        rg_ailments(c->opt1, c->opt2, ail);
        for (int i = 0; i < RG_NAILMENTS; i++) e[(RG_EFFECT_ICONS + i) * RG_EFFECT_F] = ail[i];
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

        // The focus first (the monster last attacked or cast at), so that slot 0 keeps at it; then
        // the nearest, by insertion into a short sorted list.
        int order[RG_MOBS], keys[RG_MOBS], dists[RG_MOBS], n = 0;
        for (int u = 0; u < c->nunits; u++) {
            if (c->units[u].type != RO_OBJ_MOB) continue;
            int ux, uy;
            ro_unit_pos(c, &c->units[u], &ux, &uy);
            int d = rg_dist(x, y, ux, uy), at = n < RG_MOBS ? n++ : RG_MOBS;
            int key = c->units[u].id == b->focus ? -1 : d;
            if (at == RG_MOBS && key >= keys[RG_MOBS - 1]) continue;
            if (at == RG_MOBS) at = RG_MOBS - 1;
            while (at > 0 && keys[at - 1] > key) {
                order[at] = order[at - 1];
                keys[at] = keys[at - 1];
                dists[at] = dists[at - 1];
                at--;
            }
            order[at] = u;
            keys[at] = key;
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
            rg_ailments(u->opt1, u->opt2, f + 21);
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
        verbs[RG_WAIT] = verbs[RG_MOVE] = 1;
        verbs[RG_SIT] = b->intent == RG_NONE;  // sitting would drop an attack or a pick-up
        verbs[RG_ATTACK] = b->slot_mob[0] != 0;
        verbs[RG_PICKUP] = b->slot_item[0] != 0;
        verbs[RG_POTION] = potion != NULL && c->hp < 0.7 * c->maxhp;  // rg_act drinks only then
        verbs[RG_SKILL] = any_skill;
    }
}

// Navigation's target: a breadth-first search over walkable cells finds what the bot can walk to
// within RG_NAV_MAX cells (a diagonal step needs both cells beside it free, as in rAthena's paths);
// then a random distance from RG_NAV_MIN to RG_NAV_MAX that has such cells, and a random one of
// them. Without one (a bot walled in), the target is where it stands and it cannot arrive.
#define RG_NAV_W (2 * RG_NAV_MAX + 1)
static void rg_nav_target(Env* env, RgBot* b) {
    RoClient* c = &b->c;
    int x, y;
    ro_self_pos(c, &x, &y);
    b->nav_x = x;
    b->nav_y = y;
    b->nav_d0 = b->nav_d = 0;
    if (!c->cells) return;
    unsigned char* reached = env->nav_reached;
    int* queue = env->nav_queue;
    memset(reached, 0, RG_NAV_W * RG_NAV_W);
    int head = 0, tail = 0, center = RG_NAV_MAX + RG_NAV_MAX * RG_NAV_W;
    reached[center] = 1;
    queue[tail++] = center;
    while (head < tail) {
        int k = queue[head++], kx = k % RG_NAV_W - RG_NAV_MAX, ky = k / RG_NAV_W - RG_NAV_MAX;
        for (int dir = 0; dir < 8; dir++) {
            int nx = kx + RG_DX[dir], ny = ky + RG_DY[dir];
            int n = nx + RG_NAV_MAX + (ny + RG_NAV_MAX) * RG_NAV_W;
            if (nx < -RG_NAV_MAX || ny < -RG_NAV_MAX || nx > RG_NAV_MAX || ny > RG_NAV_MAX || reached[n]) continue;
            if (!ro_walkable(c->cells, x + nx, y + ny)) continue;
            if (RG_DX[dir] && RG_DY[dir]
                && !(ro_walkable(c->cells, x + nx, y + ky) && ro_walkable(c->cells, x + kx, y + ny)))
                continue;
            reached[n] = 1;
            queue[tail++] = n;
        }
    }
    for (int tries = 0; tries < 50; tries++) {
        int d = RG_NAV_MIN + rand_r(&env->seed) % (RG_NAV_MAX - RG_NAV_MIN + 1);
        // The ring of cells d away: the rows y = -d and y = d, and the columns x = -d and x = d
        // without their corners. Count the reached ones, then take a random one.
        int count = 0, pick = -1;
        for (int pass = 0; pass < 2; pass++) {
            for (int t = -d; t <= d; t++)
                for (int side = 0; side < 4; side++) {
                    if (side >= 2 && (t == -d || t == d)) continue;
                    int ox = side < 2 ? t : side == 2 ? -d : d, oy = side < 2 ? (side ? d : -d) : t;
                    if (!reached[ox + RG_NAV_MAX + (oy + RG_NAV_MAX) * RG_NAV_W]) continue;
                    if (!pass) {
                        count++;
                    } else if (pick-- == 0) {
                        b->nav_x = x + ox;
                        b->nav_y = y + oy;
                        b->nav_d0 = b->nav_d = d;
                        return;
                    }
                }
            if (!count) break;
            pick = rand_r(&env->seed) % count;
        }
    }
}

// Start counting an episode once its @rlreset has landed.
static void rg_begin_episode(Env* env, RgBot* b) {
    RoClient* c = &b->c;
    b->steps = 0;
    b->ret = 0;
    b->mobs_seen = b->players_seen = b->edge = b->moved = 0;
    b->acts = 0;
    memset(b->verbs, 0, sizeof b->verbs);
    ro_self_pos(c, &b->last_x, &b->last_y);
    b->kills0 = c->kills;
    b->potions0 = c->potions;
    b->picked0 = c->picked;
    b->skills0 = c->skills_used;
    b->base0 = b->base_seen = ro_total_base_exp(c);
    b->job_seen = ro_total_job_exp(c);
    b->deaths_seen = c->deaths;
    b->intent = RG_NONE;
    b->attacking = 0;
    b->focus = 0;
    b->arrived = 0;
    if (b->task == RG_TASK_NAVIGATION) rg_nav_target(env, b);
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
        rg_begin_episode(env, b);
    }
}

static void rg_end_episode(Env* env, RgBot* b) {
    RoClient* c = &b->c;
    float exp_per_step = b->steps > 0 ? (float)(ro_total_base_exp(c) - b->base0) / b->steps : 0;
    float perf = exp_per_step / RG_TASK_BASELINE[b->task];
    if (b->task == RG_TASK_NAVIGATION)
        perf = b->arrived && b->steps > 0 ? (float)b->nav_d0 / b->steps / RG_TASK_BASELINE[b->task] : 0;
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
    if (b->steps > 0) {
        env->log.mobs_seen += b->mobs_seen / b->steps;
        env->log.players_seen += b->players_seen / b->steps;
        env->log.edge += b->edge / b->steps;
        env->log.moved += b->moved / b->steps;
    }
    if (b->acts > 0) {
        float* share[RG_SKILL + 1] = {&env->log.act_wait, &env->log.act_move, &env->log.act_attack,
                                      &env->log.act_pickup, &env->log.act_potion, &env->log.act_sit,
                                      &env->log.act_skill};
        for (int v = 0; v <= RG_SKILL; v++) *share[v] += (float)b->verbs[v] / b->acts;
    }
    env->log.deaths += c->dead;
    if (b->task < RG_LOG_TASKS) {
        env->log.task_perf[b->task] += perf;
        env->log.task_n[b->task] += 1;
    }
    for (int t = 0; t < RG_NTASKS && t < RG_LOG_TASKS; t++) env->log.task_w[t] = env->task_weight[t] > 0;
    rg_curriculum_add(env, b->task, perf);
    if (env->curriculum) {
        float p[RG_NTASKS];
        rg_task_chances(env, p);
        for (int t = 0; t < RG_NTASKS && t < RG_LOG_TASKS; t++) env->log.task_p[t] += p[t];
        env->log.curriculum += 1;
    }
    env->log.n += 1;
    rg_start_reset(env, b);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = dict_get(kwargs, "num_agents");
    env->lockstep = dict_get(kwargs, "lockstep");
    env->episode_steps = dict_get(kwargs, "episode_steps");
    env->action_repeat = dict_get(kwargs, "action_repeat");
    if (env->action_repeat < 1) env->action_repeat = 1;
    env->reward_scale = dict_get(kwargs, "reward_scale");
    env->reward_job = dict_get(kwargs, "reward_job");
    env->reward_death = dict_get(kwargs, "reward_death");
    for (int t = 0; t < RG_NTASKS; t++) {
        char key[32];
        snprintf(key, sizeof key, "task%d_weight", t);
        env->task_weight[t] = dict_get(kwargs, key);
    }
    env->curriculum = dict_get(kwargs, "curriculum");
    env->curriculum_floor = dict_get(kwargs, "curriculum_floor");
    env->curriculum_unlock = dict_get(kwargs, "curriculum_unlock");
    env->curriculum_patience = dict_get(kwargs, "curriculum_patience");
    env->curriculum_fast = dict_get(kwargs, "curriculum_fast");
    env->curriculum_slow = dict_get(kwargs, "curriculum_slow");
    env->seed = 12345u + 7919u * env->rng;
    env->agents = (Agent*)calloc(env->num_agents, sizeof(Agent));
    env->bots = (RgBot*)calloc(env->num_agents, sizeof(RgBot));
    env->clients = (RoClient**)calloc(env->num_agents, sizeof(RoClient*));
    env->nav_reached = (unsigned char*)malloc(RG_NAV_W * RG_NAV_W);
    env->nav_queue = (int*)malloc(RG_NAV_W * RG_NAV_W * sizeof(int));
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
    // The next block of accounts (see the top of this file), after RG_ACCOUNT_OFFSET accounts if set.
    // Blocks are taken in turn and wrap around, so a block comes back only after every other one.
    const char* skip = getenv("RG_ACCOUNT_OFFSET");
    int offset = skip ? atoi(skip) : 0, accounts = 0;
    char user[64], pass[64];
    while (fscanf(f, "%63s %63s", user, pass) == 2) accounts++;
    int blocks = (accounts - offset) / env->num_agents;
    if (blocks < 1) rg_die("%s has too few accounts%s (raise WORLD_ACCOUNT_BLOCKS)", path, "");
    int block = 0;
    char block_path[512];
    snprintf(block_path, sizeof block_path, "%s/w%d/next_block", root, world);
    FILE* bf = fopen(block_path, "r+");
    if (bf) {
        flock(fileno(bf), LOCK_EX);
        if (fscanf(bf, "%d", &block) != 1 || block < 0) block = 0;
        block %= blocks;
        rewind(bf);
        fprintf(bf, "%d\n", (block + 1) % blocks);
        fflush(bf);
        flock(fileno(bf), LOCK_UN);
        fclose(bf);
    }
    rewind(f);
    for (int i = 0; i < offset + block * env->num_agents; i++)
        if (fscanf(f, "%63s %63s", user, pass) != 2) rg_die("%s has too few accounts%s", path, "");
    const char* port_base = getenv("RO_WORLD_PORT_BASE");
    for (int i = 0; i < env->num_agents; i++) {
        RoClient* c = &env->bots[i].c;
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
        rg_begin_episode(env, b);
        rg_observe(env, i);
        env->agents[i].rewards[0] = 0;
        env->agents[i].terminals[0] = 0;
    }
}

void puf_step(Env* env) {
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].rewards[0] = 0;
        env->agents[i].terminals[0] = 0;
    }
    for (int rep = 0; rep < env->action_repeat; rep++) {
        for (int i = 0; i < env->num_agents; i++) {
            RgBot* b = &env->bots[i];
            if (env->agents[i].terminals[0]) continue;
            if (rep == 0) rg_act(b, env->agents[i].actions);
            rg_continue(b);
            if (!b->reset_wait && b->c.status_points > 0 && b->c.state == RO_IN_GAME)
                ro_spend_status(&b->c, RG_STAT_WEIGHTS[b->c.job >= 0 && b->c.job <= 6 ? b->c.job : 0]);
        }
        rg_world_step(env);
        for (int i = 0; i < env->num_agents; i++) {
            RgBot* b = &env->bots[i];
            RoClient* c = &b->c;
            if (env->agents[i].terminals[0]) {
                // The episode ended earlier in this decision: the reset lands, and the next episode
                // starts with the next decision.
                if (b->reset_wait > 1) b->reset_wait--;
                continue;
            }
            float r = 0;
            int done = 0;
            if (b->reset_wait) {
                // The reset is landing: no reward, and the episode starts counting once it has.
                if (--b->reset_wait == 0) rg_begin_episode(env, b);
            } else {
                int64_t base = ro_total_base_exp(c), job = ro_total_job_exp(c);
                int x, y;
                ro_self_pos(c, &x, &y);
                if (b->task == RG_TASK_NAVIGATION) {
                    // Progress toward the target and 1 on arrival, instead of exp.
                    int d = rg_dist(x, y, b->nav_x, b->nav_y);
                    if (b->nav_d0 > 0) r += (float)(b->nav_d - d) / b->nav_d0;
                    b->nav_d = d;
                    if (b->nav_d0 > 0 && d <= 1) {
                        r += 1;
                        b->arrived = 1;
                    }
                } else {
                    if (c->next_base_exp > 0) r += (float)(base - b->base_seen) / c->next_base_exp;
                    if (c->next_job_exp > 0) r += env->reward_job * (float)(job - b->job_seen) / c->next_job_exp;
                }
                // Scaled per task so every task pays the same at its baseline; a death costs the same
                // everywhere.
                r *= RG_TASK_REWARD[RG_TASK_FIELD] / RG_TASK_REWARD[b->task];
                int died = c->deaths > b->deaths_seen;
                if (died) r += env->reward_death;
                r *= env->reward_scale;
                b->base_seen = base;
                b->job_seen = job;
                b->deaths_seen = c->deaths;
                b->ret += r;
                b->steps++;
                int seen = 0, players = 0;
                for (int u = 0; u < c->nunits; u++) {
                    seen += c->units[u].type == RO_OBJ_MOB;
                    players += c->units[u].type == RO_OBJ_PC && c->units[u].id != c->aid;
                }
                b->mobs_seen += seen;
                b->players_seen += players;
                int step = rg_dist(b->last_x, b->last_y, x, y);
                if (step <= 3) b->moved += step;  // more is a warp
                b->last_x = x;
                b->last_y = y;
                if (c->cells) {
                    int ex = x < c->cells->xs - 1 - x ? x : c->cells->xs - 1 - x;
                    int ey = y < c->cells->ys - 1 - y ? y : c->cells->ys - 1 - y;
                    b->edge += 2.0f * (ex < ey ? ex : ey) / (c->cells->xs < c->cells->ys ? c->cells->xs : c->cells->ys);
                }
                done = died || b->arrived || (env->episode_steps > 0 && b->steps >= env->episode_steps);
            }
            env->agents[i].rewards[0] += r;
            if (done) {
                env->agents[i].terminals[0] = 1;
                rg_end_episode(env, b);
            }
        }
    }
    for (int i = 0; i < env->num_agents; i++) rg_observe(env, i);
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
    free(env->nav_reached);
    free(env->nav_queue);
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
    dict_set(out, "players_seen", log->players_seen);
    dict_set(out, "edge", log->edge);
    dict_set(out, "moved", log->moved);
    dict_set(out, "act_wait", log->act_wait);
    dict_set(out, "act_move", log->act_move);
    dict_set(out, "act_attack", log->act_attack);
    dict_set(out, "act_pickup", log->act_pickup);
    dict_set(out, "act_potion", log->act_potion);
    dict_set(out, "act_sit", log->act_sit);
    dict_set(out, "act_skill", log->act_skill);
    // Per task: the trainer divides every field by n, which leaves these ratios as they are. Every task
    // in the mix logs its keys in every interval (0 if none of its episodes ended there), since the
    // trainer's final metrics expect the same keys in every interval.
    for (int t = 0; t < RG_NTASKS && t < RG_LOG_TASKS; t++) {
        if (log->task_w[t] <= 0) continue;
        char key[48];
        snprintf(key, sizeof key, "%s_perf", RG_TASK_NAMES[t]);
        dict_set(out, key, log->task_n[t] > 0 ? log->task_perf[t] / log->task_n[t] : 0);
        snprintf(key, sizeof key, "%s_share", RG_TASK_NAMES[t]);
        dict_set(out, key, log->task_n[t]);
        if (log->curriculum > 0) {
            snprintf(key, sizeof key, "%s_chance", RG_TASK_NAMES[t]);
            dict_set(out, key, log->task_p[t]);
        }
    }
    dict_set(out, "n", log->n);
}
