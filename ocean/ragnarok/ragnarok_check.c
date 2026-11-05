// ragnarok_check: drive one ragnarok env (world 0) without the trainer, to check the env and to
// measure each task's scripted baseline.
// Usage: ragnarok_check NUM_AGENTS STEPS POLICY TASK [ACTION_REPEAT]
//   POLICY: random or scripted. TASK: a task number, or "mix" for all tasks equally. STEPS counts
//   world steps (100 ms each); the policy decides every ACTION_REPEAT of them (default 1).
// Needs a lockstep world (see slurm/env-check.sbatch) and RO_WORLD_ROOT, RO_DB_DIR.
//
// The scripted policy uses only the env's actions. Every job: drink a potion below 40% HP; keep
// attacking, else attack the nearest monster, else pick up the nearest item, else walk on. On top:
//   Swordman  Bash the target when it is next to the bot, Magnum Break when 3 monsters are close;
//             in the survival task, also back away when hurt and ganged up on, and sit to rest
//             when safe
//   Mage      never melee: walk to within 8 cells of the nearest monster and cast the bolt its
//             element is weakest to (Fire Bolt on earth and undead, Cold Bolt on fire, Lightning
//             Bolt on water, Soul Strike on ghosts, else Fire Bolt); when hit or the monster is
//             close, the quick Soul Strike or Napalm Beat; sit when SP runs low
//   Archer    Double Strafe the target within 9 cells, Arrow Shower when 3 monsters are close
//   Acolyte   Heal itself below 50% HP; Blessing and Increase AGI every 2 minutes
//   Merchant  Mammonite the target next to the bot
//   Thief     Envenom the target next to the bot
// In the navigation task it only walks: every 3 steps, or when it has stopped, toward the cell of its
// 15x15 view that it can reach and that is nearest the target, else in a random direction.
#define RG_HEADLESS
#include "ragnarok.h"

static int skill_index(int id) {
    for (int k = 0; k < RG_NSKILLS; k++)
        if (RG_SKILLS[k] == id) return k;
    return -1;
}

// A scripted bot's memory: its heading while walking on, and when it last cast each buff, in world
// steps.
typedef struct {
    int heading;
    long blessed_at, agi_at;
} Script;

static int job_of(const float* goal) {
    for (int j = 0; j < 8; j++)
        if (goal[RG_TASKS_MAX + 4 + j] > 0) return j;
    return 0;
}

static int ready(const float* skills, int k) { return k >= 0 && skills[k * RG_SKILL_F + 4] > 0; }

// The direction closest to (dx, dy).
static int toward(float dx, float dy) {
    int best = 0;
    float best_dot = -1e9f;
    for (int d = 0; d < 8; d++) {
        float dot = dx * RG_DX[d] + dy * RG_DY[d];
        if (dot > best_dot) best_dot = dot, best = d;
    }
    return best;
}

// The bolt a monster's element is weakest to (rAthena's elements: 1 water, 2 earth, 3 fire, 8
// ghost, 9 undead).
static int mage_bolt(const float* mob) {
    int element = 0;
    for (int e = 0; e < 10; e++)
        if (mob[11 + e] > 0) element = e;
    if (element == 3) return skill_index(14);
    if (element == 1) return skill_index(20);
    if (element == 8) return skill_index(13);
    return skill_index(19);
}

// Navigation's next direction: of the cells in the view that the bot can reach through walkable
// cells, the one nearest the target (fewest steps among equals), by the first step of its path; -1
// if no cell in view is nearer than where the bot stands.
static int nav_dir(const float* o) {
    const float* g = o + RG_OBS_GOAL + RG_TASKS_MAX;
    const float* view = o + RG_OBS_VIEW;
    enum { N = RG_VIEW * RG_VIEW, C = RG_VIEW / 2, CENTER = C + C * RG_VIEW };
    int tx = (int)lroundf(g[0] * RG_NAV_UNIT), ty = (int)lroundf(g[1] * RG_NAV_UNIT);
    int prev[N], queue[N], head = 0, tail = 0, best = CENTER, best_d = rg_dist(0, 0, tx, ty);
    for (int k = 0; k < N; k++) prev[k] = -1;
    prev[CENTER] = CENTER;
    queue[tail++] = CENTER;
    while (head < tail) {
        int k = queue[head++], kx = k % RG_VIEW, ky = k / RG_VIEW;
        int d = rg_dist(kx - C, ky - C, tx, ty);
        if (d < best_d) best_d = d, best = k;
        for (int dir = 0; dir < 8; dir++) {
            int nx = kx + RG_DX[dir], ny = ky + RG_DY[dir], n = nx + ny * RG_VIEW;
            if (nx < 0 || ny < 0 || nx >= RG_VIEW || ny >= RG_VIEW || prev[n] >= 0 || view[n] <= 0) continue;
            prev[n] = k;
            queue[tail++] = n;
        }
    }
    if (best == CENTER) return -1;
    while (prev[best] != CENTER) best = prev[best];
    for (int dir = 0; dir < 8; dir++)
        if (best == CENTER + RG_DX[dir] + RG_DY[dir] * RG_VIEW) return dir;
    return -1;
}

static void scripted(const float* o, float* a, Script* st, long now) {
    const float* goal = o + RG_OBS_GOAL;
    const float* self = o + RG_OBS_SELF;
    const float* cast = o + RG_OBS_CAST;
    const float* mobs = o + RG_OBS_MOBS;
    const float* items = o + RG_OBS_ITEMS;
    const float* skills = o + RG_OBS_SKILLS;
    int survival = goal[RG_TASK_SURVIVAL] > 0, job = job_of(goal);
    float hp = self[0], sp = self[1], dist = mobs[3] * RG_SIGHT;
    int near = 0, hitting = 0, idle = !self[11] && !self[12];
    for (int m = 0; m < RG_MOBS; m++) {
        const float* f = mobs + m * RG_MOB_F;
        if (f[0] > 0 && f[3] * RG_SIGHT <= 2) near++;
        if (f[0] > 0 && f[5] > 0) hitting++;
    }
    a[0] = RG_WAIT; a[1] = 0; a[2] = 0; a[3] = 0;
    if (goal[RG_TASK_NAVIGATION] > 0) {
        if (hp < 0.4f && self[9] > 0) {
            a[0] = RG_POTION;
        } else if (!self[10] || now % 3 == 0) {
            int dir = nav_dir(o);
            if (dir < 0) {
                if (rand() % 5 == 0) st->heading = rand() % 8;
                dir = st->heading;
            }
            a[0] = RG_MOVE; a[1] = dir;
        }
        return;
    }
    // The job's skill for this step, on slot 0 (the focus, else the nearest monster), or -1.
    int skill = -1;
    if (job == 1) {
        int bash = skill_index(5), magnum = skill_index(7);
        if (near >= 3 && ready(skills, magnum)) skill = magnum;
        else if (mobs[0] > 0 && dist <= 1.5f && mobs[4] > 0.3f && ready(skills, bash)) skill = bash;
    } else if (job == 2) {
        // A hit breaks a cast, so up close the quick ones: Soul Strike, else Napalm Beat.
        int bolt = mage_bolt(mobs), soul = skill_index(13), napalm = skill_index(11);
        if (mobs[0] > 0 && (hitting || dist <= 2)) skill = ready(skills, soul) ? soul : ready(skills, napalm) ? napalm : -1;
        else if (mobs[0] > 0 && dist <= 8 && ready(skills, bolt)) skill = bolt;
    } else if (job == 3) {
        int strafe = skill_index(46), shower = skill_index(47);
        if (near >= 3 && ready(skills, shower)) skill = shower;
        else if (mobs[0] > 0 && dist <= 9 && ready(skills, strafe)) skill = strafe;
    } else if (job == 4) {
        int heal = skill_index(28), bless = skill_index(34), agi = skill_index(29);
        if (hp < 0.5f && ready(skills, heal)) skill = heal;
        else if (now - st->blessed_at > 1200 && ready(skills, bless)) skill = bless;
        else if (now - st->agi_at > 1200 && ready(skills, agi)) skill = agi;
    } else if (job == 5 || job == 6) {
        int k = skill_index(job == 5 ? 42 : 52);
        if (mobs[0] > 0 && dist <= 1.5f && ready(skills, k)) skill = k;
    }

    if (hp < 0.4f && self[9] > 0) {
        a[0] = RG_POTION;
    } else if (survival && hp < 0.4f && hitting >= 2) {
        a[0] = RG_MOVE; a[1] = toward(-mobs[1], -mobs[2]);  // back away from the nearest attacker
    } else if (survival && !hitting && idle && (self[7] ? hp < 0.9f : hp < 0.6f)) {
        if (!self[7]) a[0] = RG_SIT;  // sit down, or stay seated until healed
    } else if (job == 2 && !hitting && idle && (self[7] ? sp < 0.9f : sp < 0.25f)) {
        if (!self[7]) a[0] = RG_SIT;  // a Mage out of SP rests
    } else if (cast[0] > 0) {
        a[0] = RG_WAIT;  // moving would break the cast
    } else if (skill >= 0) {
        a[0] = RG_SKILL; a[2] = 0; a[3] = skill;
        if (skill == skill_index(34)) st->blessed_at = now;
        if (skill == skill_index(29)) st->agi_at = now;
    } else if (self[11]) {
        a[0] = RG_WAIT;  // already attacking
    } else if (mobs[0] > 0 && job == 2) {
        if (dist > 8) { a[0] = RG_MOVE; a[1] = toward(mobs[1], mobs[2]); }  // else wait for SP or the delay
    } else if (mobs[0] > 0) {
        a[0] = RG_ATTACK; a[2] = 0;
    } else if (items[0] > 0 && !self[12]) {
        a[0] = RG_PICKUP; a[2] = 0;
    } else if (!self[10] && !self[12]) {
        if (rand() % 5 == 0) st->heading = rand() % 8;
        a[0] = RG_MOVE; a[1] = st->heading;
    }
}

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 64, steps = argc > 2 ? atoi(argv[2]) : 600;
    int use_script = argc > 3 && !strcmp(argv[3], "scripted");
    const char* task = argc > 4 ? argv[4] : "0";
    int repeat = argc > 5 ? atoi(argv[5]) : 1;
    if (repeat < 1) repeat = 1;
    steps = steps / repeat * repeat;
    Dict kw = {0};
    dict_set(&kw, "num_agents", n);
    dict_set(&kw, "lockstep", 1);
    dict_set(&kw, "episode_steps", 1200);
    dict_set(&kw, "action_repeat", repeat);
    dict_set(&kw, "reward_scale", 1.0);
    dict_set(&kw, "reward_job", 0.5);
    dict_set(&kw, "reward_death", -1.0);
    for (int t = 0; t < RG_NTASKS; t++) {
        char key[32];
        snprintf(key, sizeof key, "task%d_weight", t);
        dict_set(&kw, key, !strcmp(task, "mix") || atoi(task) == t ? 1.0 : 0.0);
    }
    Env env;
    memset(&env, 0, sizeof env);
    double t = ro_ms();
    puf_init(&env, &kw);
    float* obs = calloc((size_t)n * OBS_SIZE, sizeof(float));
    float* act = calloc((size_t)n * NUM_ATNS, sizeof(float));
    float* rew = calloc(n, sizeof(float));
    float* term = calloc(n, sizeof(float));
    unsigned char* masks = calloc((size_t)n * RG_MASK_SIZE, 1);
    for (int i = 0; i < n; i++) {
        env.agents[i].observations = obs + (size_t)i * OBS_SIZE;
        env.agents[i].actions = act + (size_t)i * NUM_ATNS;
        env.agents[i].rewards = rew + i;
        env.agents[i].terminals = term + i;
        env.agents[i].action_mask = masks + (size_t)i * RG_MASK_SIZE;
    }
    printf("init %d bots in %.1f s; observation %d floats, actions %d heads\n", n, (ro_ms() - t) / 1000, OBS_SIZE,
           NUM_ATNS);
    t = ro_ms();
    puf_reset(&env);
    printf("reset in %.1f s\n", (ro_ms() - t) / 1000);
    double ret = 0;
    Script* script = (Script*)calloc(n, sizeof(Script));
    for (int i = 0; i < n; i++) script[i].blessed_at = script[i].agi_at = -100000;
    long masked_violations = 0;
    t = ro_ms();
    for (int s = 0; s < steps / repeat; s++) {
        for (int i = 0; i < n; i++) {
            float* a = act + i * NUM_ATNS;
            if (use_script) scripted(obs + (size_t)i * OBS_SIZE, a, &script[i], (long)s * repeat);
            else for (int h = 0, sizes[NUM_ATNS] = ACT_SIZES; h < NUM_ATNS; h++) a[h] = rand() % sizes[h];
            // How often the scripted choice breaks the env's own mask (should be never).
            const unsigned char* m = masks + (size_t)i * RG_MASK_SIZE;
            int verb = (int)a[0], bad = !m[verb] || !m[RG_VERBS + (int)a[1]];
            if (verb == RG_ATTACK || verb == RG_PICKUP || verb == RG_SKILL) bad |= !m[RG_VERBS + 8 + (int)a[2]];
            if (verb == RG_SKILL) bad |= !m[RG_VERBS + 16 + (int)a[3]];
            if (use_script && bad) masked_violations++;
        }
        puf_step(&env);
        for (int i = 0; i < n; i++) ret += rew[i];
    }
    double wall = (ro_ms() - t) / 1000;
    printf("%s, task %s, action repeat %d: %d world steps x %d bots in %.2f s wall (%.0f bot steps/s), mean reward"
           " per bot-step %.5f\n", use_script ? "scripted" : "random", task, repeat, steps, n, wall,
           steps * (double)n / wall, ret / steps / n);
    Log* l = &env.log;
    if (l->n > 0 && strcmp(task, "mix")) {
        // The exp part of the reward, before the task's normalization: RG_TASK_REWARD for this task.
        int k = atoi(task);
        double exp_reward = (ret - l->deaths * -1.0) / steps / n * RG_TASK_REWARD[k] / RG_TASK_REWARD[RG_TASK_FIELD];
        printf("exp reward per bot-step %.6f, base exp per step %.5f (RG_TASK_REWARD and RG_TASK_BASELINE)\n",
               exp_reward, l->episode_length > 0 ? l->base_exp / l->episode_length : 0);
    }
    if (l->n > 0) {
        printf("episodes %.0f: perf %.3f, return %.3f, length %.0f, base lv %.2f, job lv %.2f, kills %.1f, deaths %.2f,"
               " potions %.1f, picked %.1f, skills %.1f, base exp %.1f (%.5f per step), monsters in sight %.2f,"
               " players in sight %.2f, edge distance %.2f, cells walked per step %.3f\n",
               l->n, l->perf / l->n, l->score / l->n, l->episode_length / l->n, l->base_level / l->n,
               l->job_level / l->n, l->kills / l->n, l->deaths / l->n, l->potions / l->n, l->picked / l->n,
               l->skills / l->n, l->base_exp / l->n, l->episode_length > 0 ? l->base_exp / l->episode_length : 0,
               l->mobs_seen / l->n, l->players_seen / l->n, l->edge / l->n, l->moved / l->n);
        printf("  verbs: wait %.2f move %.2f attack %.2f pickup %.2f potion %.2f sit %.2f skill %.2f\n",
               l->act_wait / l->n, l->act_move / l->n, l->act_attack / l->n, l->act_pickup / l->n,
               l->act_potion / l->n, l->act_sit / l->n, l->act_skill / l->n);
        for (int k = 0; k < RG_NTASKS; k++)
            if (l->task_n[k] > 0)
                printf("  task %d %-13s episodes %.0f, perf %.3f\n", k, RG_TASK_NAMES[k], l->task_n[k],
                       l->task_perf[k] / l->task_n[k]);
    }
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (env.bots[i].c.state == RO_ERROR) {
            if (!errors) printf("bot %s: %s\n", env.bots[i].c.user, env.bots[i].c.err);
            errors++;
        }
    }
    const RoClient* c0 = &env.bots[0].c;
    printf("bot 0 now: task %d job %d lv %d/%d hp %d/%d sp %d/%d skills %d on %s; scripted choices against the"
           " mask: %ld\n", env.bots[0].task, c0->job, c0->base_level, c0->job_level, c0->hp, c0->maxhp, c0->sp,
           c0->maxsp, c0->nskills, c0->map, masked_violations);
    printf("bots with errors: %d\n", errors);
    puf_close(&env);
    return errors ? 1 : 0;
}
