// ragnarok_check: drive one ragnarok env (world 0) without the trainer, to check the env and to
// measure each task's scripted baseline.
// Usage: ragnarok_check NUM_AGENTS STEPS POLICY TASK
//   POLICY: random or scripted. TASK: a task number, or "mix" for all tasks equally.
// Needs a lockstep world (see slurm/env-check.sbatch) and RO_WORLD_ROOT, RO_DB_DIR.
//
// The scripted policy uses only the env's actions: drink a potion below 40% HP; for Swordman
// tasks, Bash the target when it is next to the bot and Magnum Break when 3 monsters are close;
// in the survival task, also back away when hurt and ganged up on, and sit to rest when safe;
// otherwise keep attacking, else attack the nearest monster, else pick up the nearest item, else
// walk on.
#define RG_HEADLESS
#include "ragnarok.h"

static int skill_index(int id) {
    for (int k = 0; k < RG_NSKILLS; k++)
        if (RG_SKILLS[k] == id) return k;
    return -1;
}

static void scripted(const float* o, float* a, int* heading) {
    const float* goal = o + RG_OBS_GOAL;
    const float* self = o + RG_OBS_SELF;
    const float* mobs = o + RG_OBS_MOBS;
    const float* items = o + RG_OBS_ITEMS;
    const float* skills = o + RG_OBS_SKILLS;
    int survival = goal[RG_TASK_SURVIVAL] > 0, swordman = goal[RG_TASKS_MAX + 4 + 1] > 0;
    float hp = self[0];
    int near = 0, hitting = 0;
    for (int m = 0; m < RG_MOBS; m++) {
        const float* f = mobs + m * RG_MOB_F;
        if (f[0] > 0 && f[3] * RG_SIGHT <= 2) near++;
        if (f[0] > 0 && f[5] > 0) hitting++;
    }
    int bash = skill_index(5), magnum = skill_index(7);
    a[0] = RG_WAIT; a[1] = 0; a[2] = 0; a[3] = 0;
    if (hp < 0.4f && self[9] > 0) {
        a[0] = RG_POTION;
    } else if (survival && hp < 0.4f && hitting >= 2) {
        // Back away from the nearest attacker.
        float dx = -mobs[1], dy = -mobs[2];
        int best = 0;
        float best_dot = -1e9f;
        for (int d = 0; d < 8; d++) {
            float dot = dx * RG_DX[d] + dy * RG_DY[d];
            if (dot > best_dot) best_dot = dot, best = d;
        }
        a[0] = RG_MOVE; a[1] = best;
    } else if (survival && !hitting && (self[7] ? hp < 0.9f : hp < 0.6f)) {
        if (!self[7]) a[0] = RG_SIT;  // sit down, or stay seated until healed
    } else if (swordman && magnum >= 0 && near >= 3 && skills[magnum * RG_SKILL_F + 4] > 0) {
        a[0] = RG_SKILL; a[3] = magnum;
    } else if (swordman && bash >= 0 && mobs[0] > 0 && mobs[3] * RG_SIGHT <= 1.5f && mobs[4] > 0.3f
               && skills[bash * RG_SKILL_F + 4] > 0) {
        a[0] = RG_SKILL; a[2] = 0; a[3] = bash;
    } else if (self[11]) {
        a[0] = RG_WAIT;  // already attacking
    } else if (mobs[0] > 0) {
        a[0] = RG_ATTACK; a[2] = 0;
    } else if (items[0] > 0 && !self[12]) {
        a[0] = RG_PICKUP; a[2] = 0;
    } else if (!self[10] && !self[12]) {
        if (rand() % 5 == 0) *heading = rand() % 8;
        a[0] = RG_MOVE; a[1] = *heading;
    }
}

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 64, steps = argc > 2 ? atoi(argv[2]) : 600;
    int use_script = argc > 3 && !strcmp(argv[3], "scripted");
    const char* task = argc > 4 ? argv[4] : "0";
    Dict kw = {0};
    dict_set(&kw, "num_agents", n);
    dict_set(&kw, "lockstep", 1);
    dict_set(&kw, "episode_steps", 1200);
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
    int heading[1024] = {0};
    long masked_violations = 0;
    t = ro_ms();
    for (int s = 0; s < steps; s++) {
        for (int i = 0; i < n; i++) {
            float* a = act + i * NUM_ATNS;
            if (use_script) scripted(obs + (size_t)i * OBS_SIZE, a, &heading[i]);
            else for (int h = 0, sizes[NUM_ATNS] = ACT_SIZES; h < NUM_ATNS; h++) a[h] = rand() % sizes[h];
            // How often the scripted choice breaks the env's own mask (should be never).
            const unsigned char* m = masks + (size_t)i * RG_MASK_SIZE;
            if (use_script && (!m[(int)a[0]] || !m[RG_VERBS + (int)a[1]])) masked_violations++;
        }
        puf_step(&env);
        for (int i = 0; i < n; i++) ret += rew[i];
    }
    double wall = (ro_ms() - t) / 1000;
    printf("%s, task %s: %d steps x %d bots in %.2f s wall (%.0f bot steps/s), mean reward per bot-step %.5f\n",
           use_script ? "scripted" : "random", task, steps, n, wall, steps * (double)n / wall, ret / steps / n);
    Log* l = &env.log;
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
