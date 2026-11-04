// ragnarok_check: drive one ragnarok env (world 0) without the trainer, to check the env.
// Usage: ragnarok_check NUM_AGENTS STEPS POLICY   (POLICY: random or scripted)
// Needs a lockstep world (see slurm/env-check.sbatch) and RO_WORLD_ROOT, RO_DB_DIR.
// The scripted policy uses only the env's actions: drink a potion below 40% HP, keep attacking,
// else attack the nearest monster, else pick up the nearest item, else walk on.
#define RG_HEADLESS
#include "ragnarok.h"

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 64, steps = argc > 2 ? atoi(argv[2]) : 600;
    int scripted = argc > 3 && !strcmp(argv[3], "scripted");
    Dict kw = {0};
    dict_set(&kw, "num_agents", n);
    dict_set(&kw, "lockstep", 1);
    dict_set(&kw, "episode_steps", 1200);
    dict_set(&kw, "reward_scale", 1.0);
    dict_set(&kw, "reward_job", 0.5);
    dict_set(&kw, "reward_death", -1.0);
    Env env;
    memset(&env, 0, sizeof env);
    double t = ro_ms();
    puf_init(&env, &kw);
    float* obs = calloc((size_t)n * OBS_SIZE, sizeof(float));
    float* act = calloc((size_t)n * NUM_ATNS, sizeof(float));
    float* rew = calloc(n, sizeof(float));
    float* term = calloc(n, sizeof(float));
    for (int i = 0; i < n; i++) {
        env.agents[i].observations = obs + (size_t)i * OBS_SIZE;
        env.agents[i].actions = act + (size_t)i * NUM_ATNS;
        env.agents[i].rewards = rew + i;
        env.agents[i].terminals = term + i;
    }
    printf("init %d bots in %.1f s\n", n, (ro_ms() - t) / 1000);
    t = ro_ms();
    puf_reset(&env);
    printf("reset in %.1f s\n", (ro_ms() - t) / 1000);
    double ret = 0;
    int heading[512] = {0};
    t = ro_ms();
    for (int s = 0; s < steps; s++) {
        for (int i = 0; i < n; i++) {
            float* a = act + i * NUM_ATNS;
            const float* o = obs + (size_t)i * OBS_SIZE;
            const float* mobs = o + RG_SELF + RG_VIEW * RG_VIEW;
            const float* items = mobs + RG_MOBS * RG_MOB_F;
            if (!scripted) {
                a[0] = rand() % 6; a[1] = rand() % 8; a[2] = rand() % 8;
            } else if (o[0] < 0.4f && o[9] > 0) {
                a[0] = RG_POTION;
            } else if (o[11]) {
                a[0] = RG_WAIT;  // already attacking
            } else if (mobs[0] > 0) {
                a[0] = RG_ATTACK; a[2] = 0;
            } else if (items[0] > 0 && !o[12]) {
                a[0] = RG_PICKUP; a[2] = 0;
            } else if (!o[10] && !o[12]) {
                if (rand() % 5 == 0) heading[i] = rand() % 8;
                a[0] = RG_MOVE; a[1] = heading[i];
            } else {
                a[0] = RG_WAIT;
            }
        }
        puf_step(&env);
        for (int i = 0; i < n; i++) ret += rew[i];
    }
    double wall = (ro_ms() - t) / 1000;
    printf("%s: %d steps x %d bots in %.2f s wall (%.0f bot steps/s), mean reward per bot-step %.5f\n",
           scripted ? "scripted" : "random", steps, n, wall, steps * (double)n / wall, ret / steps / n);
    Log* l = &env.log;
    if (l->n > 0)
        printf("episodes %.0f: mean return %.3f, length %.0f, base lv %.2f, job lv %.2f, kills %.1f, deaths %.2f,"
               " potions %.1f, picked %.1f, base exp %.1f\n", l->n, l->score / l->n, l->episode_length / l->n,
               l->base_level / l->n, l->job_level / l->n, l->kills / l->n, l->deaths / l->n, l->potions / l->n,
               l->picked / l->n, l->base_exp / l->n);
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (env.bots[i].c.state == RO_ERROR) {
            if (!errors) printf("bot %s: %s\n", env.bots[i].c.user, env.bots[i].c.err);
            errors++;
        }
    }
    printf("bots with errors: %d\n", errors);
    puf_close(&env);
    return errors ? 1 : 0;
}
