// ragnarok_check: drive one ragnarok env (world 0) without the trainer, to check the env and to
// measure each task's scripted baseline.
// Usage: ragnarok_check NUM_AGENTS STEPS POLICY TASK [ACTION_REPEAT]
//   POLICY: random or scripted. TASK: a task number, "mix" for all tasks equally, or "curriculum"
//   for all tasks drawn by the curriculum (see ragnarok.h). STEPS counts
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
//   Acolyte   Heal itself below 50% HP; Blessing and Increase AGI whenever they have run out
//   Merchant  Mammonite the target next to the bot
//   Thief     Envenom the target next to the bot
// In the navigation task it only walks: every 3 steps, or when it has stopped, toward the cell of its
// 15x15 view that it can reach and that is nearest the target, else in a random direction. In the
// travel task it reads the region from client/rathena_atlas.h and the maps from RO_DB_DIR (the policy
// has only the compass and the portals in its view): it walks the shortest path, by a breadth-first
// search over the whole map, to the next warp toward the target's map, then to the target.
#define RG_HEADLESS
#include "ragnarok.h"

static int skill_index(int id) {
    for (int k = 0; k < RG_NSKILLS; k++)
        if (RG_SKILLS[k] == id) return k;
    return -1;
}

// A scripted bot's memory: its heading while walking on, and for travel, the distance of every cell of
// a map to its current goal (-1 where unreachable).
typedef struct {
    int heading;
    int map, goal_x, goal_y;
    int16_t* dist;
} Script;

// The steps from every cell of the map to (gx, gy), by a breadth-first search over walkable cells (a
// diagonal step needs both cells beside it free); kept until the goal changes.
static void plan(Script* st, const RoMap* cells, int map, int gx, int gy) {
    if (st->dist && st->map == map && st->goal_x == gx && st->goal_y == gy) return;
    int n = cells->xs * cells->ys, head = 0, tail = 0;
    int* queue = (int*)malloc(n * sizeof(int));
    st->dist = (int16_t*)realloc(st->dist, n * sizeof(int16_t));
    for (int k = 0; k < n; k++) st->dist[k] = -1;
    st->map = map, st->goal_x = gx, st->goal_y = gy;
    if (gx < 0 || gy < 0 || gx >= cells->xs || gy >= cells->ys) {
        free(queue);
        return;
    }
    st->dist[gx + gy * cells->xs] = 0;
    queue[tail++] = gx + gy * cells->xs;
    while (head < tail) {
        int k = queue[head++], kx = k % cells->xs, ky = k / cells->xs;
        for (int dir = 0; dir < 8; dir++) {
            int nx = kx + RG_DX[dir], ny = ky + RG_DY[dir];
            if (!ro_walkable(cells, nx, ny) || st->dist[nx + ny * cells->xs] >= 0) continue;
            if (RG_DX[dir] && RG_DY[dir] && !(ro_walkable(cells, nx, ky) && ro_walkable(cells, kx, ny))) continue;
            st->dist[nx + ny * cells->xs] = st->dist[k] + 1;
            queue[tail++] = nx + ny * cells->xs;
        }
    }
    free(queue);
}

// The direction to the neighbor cell nearest the goal, or -1.
static int plan_dir(const Script* st, const RoMap* cells, int x, int y) {
    int best = -1, best_d = ro_walkable(cells, x, y) && st->dist[x + y * cells->xs] >= 0 ? st->dist[x + y * cells->xs] : 1 << 30;
    for (int dir = 0; dir < 8; dir++) {
        int nx = x + RG_DX[dir], ny = y + RG_DY[dir];
        if (!ro_walkable(cells, nx, ny)) continue;
        int d = st->dist[nx + ny * cells->xs];
        if (d >= 0 && d < best_d) best_d = d, best = dir;
    }
    return best;
}

static int job_of(const float* goal) {
    for (int j = 0; j < 8; j++)
        if (goal[RG_GOAL_JOB + j] > 0) return j;
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

// The next direction toward a target (tx, ty) cells away: of the cells in the view that the bot can
// reach through walkable cells, the one nearest the target (fewest steps among equals), by the first
// step of its path; -1 if no cell in view is nearer than where the bot stands.
static int nav_dir(const float* o, int tx, int ty) {
    const float* view = o + RG_OBS_VIEW;
    enum { N = RG_VIEW * RG_VIEW, C = RG_VIEW / 2, CENTER = C + C * RG_VIEW };
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

static int floor_div(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

// Travel's next direction: along the shortest path to the target on its map, else to the nearest
// warp that leads a map closer to it; -1 off the travel region.
static int travel_dir(const float* o, Script* st) {
    const float* self = o + RG_OBS_SELF;
    const float* g = o + RG_OBS_GOAL + RG_TASKS_MAX;
    int m = -1, t = -1;
    for (int k = 0; self[19] > 0 && k < RO_ATLAS_MAPS; k++)
        if (RO_ATLAS_MAP[k].gx == lroundf(self[20] * 4) && RO_ATLAS_MAP[k].gy == lroundf(self[21] * 4)) m = k;
    if (m < 0) return -1;
    int x = lroundf(self[16] * RO_ATLAS_MAP[m].xs), y = lroundf(self[17] * RO_ATLAS_MAP[m].ys);
    int dx = lroundf(g[0] * RO_ATLAS_SQUARE), dy = lroundf(g[1] * RO_ATLAS_SQUARE);
    const RoMap* cells = ro_map_get(getenv("RO_DB_DIR"), RO_ATLAS_MAP[m].name);
    if (!cells) return -1;
    if (g[3] > 0) {
        plan(st, cells, m, x + dx, y + dy);
        return plan_dir(st, cells, x, y);
    }
    int wx = RO_ATLAS_MAP[m].gx * RO_ATLAS_SQUARE + x + dx, wy = RO_ATLAS_MAP[m].gy * RO_ATLAS_SQUARE + y + dy;
    for (int k = 0; k < RO_ATLAS_MAPS; k++)
        if (RO_ATLAS_MAP[k].gx == floor_div(wx, RO_ATLAS_SQUARE) && RO_ATLAS_MAP[k].gy == floor_div(wy, RO_ATLAS_SQUARE))
            t = k;
    if (t < 0) return -1;
    int hops[RO_ATLAS_MAPS], best = -1, best_d = 1 << 30;
    rg_atlas_hops(t, hops);  // the warps go both ways
    for (int w = 0; w < RO_ATLAS_WARPS; w++) {
        const RoAtlasWarp* p = &RO_ATLAS_WARP[w];
        if (p->from != m || hops[p->to] < 0 || hops[p->to] >= hops[m]) continue;
        int d = rg_dist(x, y, p->x, p->y);
        if (d < best_d) best_d = d, best = w;
    }
    if (best < 0) return -1;
    plan(st, cells, m, RO_ATLAS_WARP[best].x, RO_ATLAS_WARP[best].y);
    return plan_dir(st, cells, x, y);
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
    if (goal[RG_TASK_NAVIGATION] > 0 || goal[RG_TASK_TRAVEL] > 0) {
        const float* g = goal + RG_TASKS_MAX;
        if (hp < 0.4f && self[9] > 0) {
            a[0] = RG_POTION;
        } else if (!self[10] || now % 3 == 0) {
            int dir = goal[RG_TASK_TRAVEL] > 0 ? travel_dir(o, st)
                                               : nav_dir(o, lroundf(g[0] * RG_NAV_UNIT), lroundf(g[1] * RG_NAV_UNIT));
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
        // Blessing and Increase AGI are the first two effects the bot sees on itself.
        const float* effects = o + RG_OBS_EFFECTS;
        int heal = skill_index(28), bless = skill_index(34), agi = skill_index(29);
        if (hp < 0.5f && ready(skills, heal)) skill = heal;
        else if (effects[0] <= 0 && ready(skills, bless)) skill = bless;
        else if (effects[RG_EFFECT_F] <= 0 && ready(skills, agi)) skill = agi;
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
    // Episodes of 1,200 world steps (EPISODE_STEPS to change it), shorter than training's 3,000 so
    // that a short run ends many of them.
    dict_set(&kw, "episode_steps", getenv("EPISODE_STEPS") ? atoi(getenv("EPISODE_STEPS")) : 1200);
    dict_set(&kw, "action_repeat", repeat);
    dict_set(&kw, "reward_scale", 1.0);
    dict_set(&kw, "reward_job", 0.5);
    dict_set(&kw, "reward_death", -1.0);
    int mix = !strcmp(task, "mix") || !strcmp(task, "curriculum");
    for (int t = 0; t < RG_NTASKS; t++) {
        char key[32];
        snprintf(key, sizeof key, "task%d_weight", t);
        dict_set(&kw, key, mix || atoi(task) == t ? 1.0 : 0.0);
    }
    dict_set(&kw, "curriculum", !strcmp(task, "curriculum"));
    dict_set(&kw, "curriculum_floor", 0.05);
    dict_set(&kw, "curriculum_unlock", 0.8);
    dict_set(&kw, "curriculum_patience", 20000);
    dict_set(&kw, "curriculum_fast", 0.002);
    dict_set(&kw, "curriculum_slow", 0.0002);
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
    if (l->n > 0 && !mix) {
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
            if (l->task_n[k] > 0 || l->curriculum > 0)
                printf("  task %d %-15s episodes %.0f, perf %.3f, chance %.3f\n", k, RG_TASK_NAMES[k],
                       l->task_n[k], l->task_n[k] > 0 ? l->task_perf[k] / l->task_n[k] : 0,
                       l->curriculum > 0 ? l->task_p[k] / l->curriculum : 0);
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
