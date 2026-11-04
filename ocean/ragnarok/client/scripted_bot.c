// scripted_bot: N hand-written bots that level up novices on a field, as the baseline and as the
// end-to-end test of rathena_client.h.
//
// Each bot: kill the nearest easy monster (or whatever is hitting it), pick up what drops, drink a
// Red Potion below 40% HP, sit to rest when out of potions, respawn when dead, spend status
// points by a fixed rule. A step is 100 ms of game time: wall-clock time by default, or, with
// --lockstep, one step of a lockstep map server (as fast as the server and the bots allow).
//
// Usage: scripted_bot --host H --port P --accounts FILE --db DIR [--bots N] [--minutes M]
//                     [--log-every S] [--die-after S] [--reset-every S] [--lockstep] [--trace]
//                     [--verbose]
// --die-after S: the first bot types @die after S seconds, to test death and respawn; it needs a
// GM account (train/world.sh with WORLD_GM_BOT=1). --reset-every S: every bot types @rlreset
// every S seconds (a lean training world's episode reset). --minutes, --log-every, --die-after and
// --reset-every count game time.
// FILE has one "userid password" per line. DIR is rAthena's db/ directory (for map_cache.dat).
// Prints one summary line per bot every --log-every seconds and a final table; exits 1 if any bot
// hit a protocol error.
#include <poll.h>

#include "rathena_client.h"

#define STEP_MS 100.0
#define RED_POTION 501
static const int EASY_MOBS[] = {1002, 1063, 1113, 1007, 1049, 1052, 1012};  // Poring Lunatic Drops Fabre
                                                                         // Picky Rocker Roda Frog
// Status points go to the stat with the lowest value / weight: STR, AGI, VIT, INT, DEX, LUK.
static const int STAT_WEIGHT[6] = {3, 3, 2, 0, 2, 0};

typedef struct Brain {
    uint32_t target, item, ignore_items[32];
    int nignore;
    double target_since, last_attack, item_since, last_pickup, last_potion, last_wander, last_stat;
    double died_at, last_respawn, last_reset;
    int resets;
    int resting, heading, attacking;
    int64_t start_base, start_job;
    int start_level, start_job_level;
    double start_time;
    int reported_error;
} Brain;

static int dist(int x0, int y0, int x1, int y1) {
    int dx = abs(x1 - x0), dy = abs(y1 - y0);
    return dx > dy ? dx : dy;
}

static int easy(int cls) {
    for (size_t i = 0; i < sizeof EASY_MOBS / sizeof EASY_MOBS[0]; i++)
        if (EASY_MOBS[i] == cls) return 1;
    return 0;
}

static int ignored(const Brain* b, uint32_t id) {
    for (int i = 0; i < b->nignore; i++)
        if (b->ignore_items[i] == id) return 1;
    return 0;
}

static void ignore_item(Brain* b, uint32_t id) {
    b->ignore_items[b->nignore++ % 32] = id;
    if (b->nignore > 32) b->nignore = 32;
}

// --trace: the first bot prints what it decides and what it sees, once a second at most.
static int trace = 0;
static const RoClient* trace_client;
static void trace_view(RoClient* c, const char* what) {
    static double last;
    if (!trace || c != trace_client || c->now - last < 1000) return;
    last = c->now;
    int x, y, mobs = 0, easy_mobs = 0, types[16] = {0};
    ro_self_pos(c, &x, &y);
    for (int i = 0; i < c->nunits; i++) {
        types[c->units[i].type & 15]++;
        if (c->units[i].type == RO_OBJ_MOB) {
            mobs++;
            for (size_t k = 0; k < sizeof EASY_MOBS / sizeof EASY_MOBS[0]; k++)
                if (EASY_MOBS[k] == c->units[i].cls) easy_mobs++;
        }
    }
    printf("trace %s at %d,%d -> %d,%d hp %d/%d: %s; %d units (pc %d npc %d mob %d easy %d), %d on ground\n",
           c->user, x, y, c->x1, c->y1, c->hp, c->maxhp, what, c->nunits, types[0], types[1] + types[6], mobs,
           easy_mobs, c->nground);
    for (int i = 0; i < c->nunits && i < 4; i++) {
        const RoUnit* u = &c->units[i];
        int ux, uy;
        ro_unit_pos(c, u, &ux, &uy);
        printf("trace   unit %u type %d class %d at %d,%d hp %d/%d\n", u->id, u->type, u->cls, ux, uy, u->hp, u->maxhp);
    }
    fflush(stdout);
}

static void spend_status(RoClient* c, Brain* b) {
    if (c->status_points <= 0 || c->now - b->last_stat < 300) return;
    int value[6] = {c->str, c->agi, c->vit, c->int_, c->dex, c->luk};
    int best = -1;
    double best_score = 1e9;
    for (int s = 0; s < 6; s++) {
        int cost = (value[s] - 1) / 10 + 2;  // pre-renewal raise cost
        if (!STAT_WEIGHT[s] || value[s] >= 99 || cost > c->status_points) continue;
        double score = (double)value[s] / STAT_WEIGHT[s];
        if (score < best_score) best_score = score, best = s;
    }
    if (best >= 0) {
        ro_status_up(c, RO_SP_STR + best, 1);
        b->last_stat = c->now;
    }
}

static void step(RoClient* c, Brain* b) {
    if (c->state != RO_IN_GAME) return;
    double now = c->now;
    if (c->dead) {
        if (!b->died_at) b->died_at = now;
        if (now - b->died_at > 1500 && now - b->last_respawn > 3000) {
            ro_respawn(c);
            b->last_respawn = now;
            b->target = b->item = 0;
            b->resting = 0;
        }
        return;
    }
    b->died_at = 0;
    spend_status(c, b);

    int x, y;
    ro_self_pos(c, &x, &y);
    double hp = c->maxhp > 0 ? (double)c->hp / c->maxhp : 1.0;
    int attacked = 0;
    for (int i = 0; i < c->nunits; i++)
        if (c->units[i].last_hit_me && now - c->units[i].last_hit_me < 3000) attacked = 1;

    const RoInv* potion = ro_inv_find(c, RED_POTION);
    if (hp < 0.4 && potion && now - b->last_potion > 1000) {
        ro_use_item(c, potion->index);
        b->last_potion = now;
        return;
    }
    if (b->resting) {
        if (hp > 0.9 || attacked) {
            ro_sit(c, 0);
            b->resting = 0;
        }
        return;
    }
    if (hp < 0.3 && !potion && !attacked) {
        ro_sit(c, 1);
        b->resting = 1;
        b->target = 0;
        return;
    }

    // The current target: walk next to it (the server does not walk a player into range), then
    // attack; the attack repeats on its own while the target stays in reach.
    RoUnit* t = b->target ? ro_unit_find(c, b->target) : NULL;
    if (t && now - b->target_since < 30000) {
        int tx, ty, range = c->attack_range > 0 ? c->attack_range : 1;
        ro_unit_pos(c, t, &tx, &ty);
        int too_far = c->too_far_id == t->id && now - c->too_far_at < 1000;
        if (dist(x, y, tx, ty) > range || too_far) {
            // The cell next to the target on our side, or the target's own cell.
            int ax = tx + (x > tx) - (x < tx), ay = ty + (y > ty) - (y < ty);
            if (!ro_walkable(c->cells, ax, ay)) ax = tx, ay = ty;
            if (!ro_self_moving(c) || dist(c->x1, c->y1, tx, ty) > range || too_far) ro_walk(c, ax, ay);
            b->attacking = 0;
            trace_view(c, "approaching");
        } else {
            if (!b->attacking || now - b->last_attack > 2000) {
                ro_attack(c, t->id);
                b->attacking = 1;
                b->last_attack = now;
            }
            trace_view(c, "attacking");
        }
        return;
    }
    b->target = 0;
    b->attacking = 0;

    // Loot: the nearest item on the ground within 8 cells.
    RoGround* g = NULL;
    for (int i = 0; i < c->nground; i++) {
        RoGround* it = &c->ground[i];
        if (ignored(b, it->id) || dist(x, y, it->x, it->y) > 8) continue;
        if (!g || dist(x, y, it->x, it->y) < dist(x, y, g->x, g->y)) g = it;
    }
    if (g && !attacked) {
        if (b->item != g->id) {
            b->item = g->id;
            b->item_since = now;
        }
        if (now - b->item_since > 8000) {
            ignore_item(b, g->id);
            b->item = 0;
        } else if (dist(x, y, g->x, g->y) <= 1) {
            if (now - b->last_pickup > 500) {
                ro_pickup(c, g->id);
                b->last_pickup = now;
            }
        } else if (!ro_self_moving(c) || (c->x1 != g->x || c->y1 != g->y)) {
            ro_walk(c, g->x, g->y);
        }
        trace_view(c, "looting");
        return;
    }

    // A new target: whatever is hitting us, else the nearest easy monster in view.
    RoUnit* best = NULL;
    int best_d = 1 << 30;
    for (int i = 0; i < c->nunits; i++) {
        RoUnit* u = &c->units[i];
        if (u->type != RO_OBJ_MOB || u->boss) continue;
        int ux, uy;
        ro_unit_pos(c, u, &ux, &uy);
        int d = dist(x, y, ux, uy);
        int hitting = u->last_hit_me && now - u->last_hit_me < 3000;
        if (!hitting && !easy(u->cls)) continue;
        if (hitting) d -= 1000;
        if (d < best_d) best_d = d, best = u;
    }
    if (best) {
        b->target = best->id;
        b->target_since = now;
        b->attacking = 0;
        trace_view(c, "new target");
        return;
    }
    trace_view(c, "exploring");

    // Nothing in view: explore in long straight legs, turning when blocked or now and then.
    // Monsters are sparse (140 on a 400x400 field) and a bot sees 14 cells around it.
    if (!ro_self_moving(c) || now - b->last_wander > 6000) {
        static const int DX[8] = {1, 1, 0, -1, -1, -1, 0, 1}, DY[8] = {0, 1, 1, 1, 0, -1, -1, -1};
        if (rand() % 5 == 0) b->heading = (b->heading + 7 + rand() % 3) % 8;
        for (int tries = 0; tries < 16; tries++) {
            int wx = x + DX[b->heading] * 15 + rand() % 5 - 2, wy = y + DY[b->heading] * 15 + rand() % 5 - 2;
            if (ro_walkable(c->cells, wx, wy)) {
                ro_walk(c, wx, wy);
                break;
            }
            b->heading = rand() % 8;
        }
        b->last_wander = now;
    }
}

// Lockstep: the newest game time any bot has seen.
static double game_clock(const RoClient* bots, int n) {
    double t = 0;
    for (int i = 0; i < n; i++)
        if (bots[i].lockstep_joined && bots[i].now > t) t = bots[i].now;
    return t;
}

static void report(const RoClient* c, const Brain* b, const char* tag) {
    double hours = (c->now - b->start_time) / 3.6e6;
    int64_t base = ro_total_base_exp(c) - b->start_base, job = ro_total_job_exp(c) - b->start_job;
    int x, y;
    ro_self_pos(c, &x, &y);
    printf("%s %-8s lv %2d/%2d base_exp %6lld (%7.0f/h) job_exp %6lld kills %4d deaths %2d potions %3d "
           "picked %3d hp %4d/%4d at %s %d,%d%s%s\n",
           tag, c->user, c->base_level, c->job_level, (long long)base, hours > 0 ? base / hours : 0.0,
           (long long)job, c->kills, c->deaths, c->potions, c->picked, c->hp, c->maxhp, c->map, x, y,
           c->state == RO_ERROR ? " ERROR: " : "", c->state == RO_ERROR ? c->err : "");
}

int main(int argc, char** argv) {
    const char *host = "127.0.0.1", *accounts = NULL, *db = NULL;
    int port = 6900, nbots = 1, verbose = 0, lockstep = 0;
    double minutes = 10, log_every = 60, die_after = -1, reset_every = -1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--accounts") && i + 1 < argc) accounts = argv[++i];
        else if (!strcmp(argv[i], "--db") && i + 1 < argc) db = argv[++i];
        else if (!strcmp(argv[i], "--bots") && i + 1 < argc) nbots = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--minutes") && i + 1 < argc) minutes = atof(argv[++i]);
        else if (!strcmp(argv[i], "--log-every") && i + 1 < argc) log_every = atof(argv[++i]);
        else if (!strcmp(argv[i], "--die-after") && i + 1 < argc) die_after = atof(argv[++i]);
        else if (!strcmp(argv[i], "--reset-every") && i + 1 < argc) reset_every = atof(argv[++i]);
        else if (!strcmp(argv[i], "--verbose")) verbose = 1;
        else if (!strcmp(argv[i], "--lockstep")) lockstep = 1;
        else if (!strcmp(argv[i], "--trace")) trace = 1;
        else {
            fprintf(stderr, "usage: %s --host H --port P --accounts FILE --db DIR [--bots N] [--minutes M]"
                            " [--log-every S] [--die-after S] [--reset-every S] [--lockstep] [--trace] [--verbose]\n", argv[0]);
            return 2;
        }
    }
    if (!accounts || !db) {
        fprintf(stderr, "--accounts and --db are required\n");
        return 2;
    }
    FILE* f = fopen(accounts, "r");
    if (!f) {
        perror(accounts);
        return 2;
    }
    RoClient* bots = calloc(nbots, sizeof *bots);
    Brain* brains = calloc(nbots, sizeof *brains);
    int n = 0;
    char user[64], pass[64];
    while (n < nbots && fscanf(f, "%63s %63s", user, pass) == 2) {
        if (strlen(user) > 23 || strlen(pass) > 23) {  // CA_LOGIN holds 23 characters of each
            fprintf(stderr, "account %d in %s: userid or password longer than 23 characters\n", n + 1, accounts);
            return 2;
        }
        snprintf(bots[n].user, sizeof bots[n].user, "%s", user);
        snprintf(bots[n].pass, sizeof bots[n].pass, "%s", pass);
        n++;
    }
    fclose(f);
    if (n < nbots) {
        fprintf(stderr, "only %d accounts in %s\n", n, accounts);
        nbots = n;
    }
    srand(1234);
    trace_client = &bots[0];
    for (int i = 0; i < nbots; i++) {
        RoClient* c = &bots[i];
        snprintf(c->host, sizeof c->host, "%s", host);
        c->login_port = port;
        c->db_dir = db;
        c->verbose = verbose;
        double t = ro_ms();
        if (ro_login(c, 0) < 0) printf("login %s failed: %s\n", c->user, c->err);
        else printf("login %s ok in %.0f ms: char %s on %s\n", c->user, ro_ms() - t, c->name, c->map);
        fflush(stdout);
    }

    // Wait for every bot to be on the map and to have its status, then start counting.
    double t0 = ro_ms();
    for (int ready = 0; !ready && ro_ms() - t0 < 15000;) {
        ready = 1;
        for (int i = 0; i < nbots; i++) {
            ro_poll(&bots[i]);
            if (bots[i].state == RO_ENTERING || (bots[i].state == RO_IN_GAME && !bots[i].base_level)) ready = 0;
        }
        usleep(20000);
    }
    // In lockstep mode every bot sends READY once; the first DONE starts the game clock.
    RoClient** ptrs = calloc((size_t)(nbots > 0 ? nbots : 1), sizeof *ptrs);
    for (int i = 0; i < nbots; i++) ptrs[i] = &bots[i];
    if (lockstep) {
        if (ro_lockstep_join(ptrs, nbots, 30000) < 0) {
            printf("lockstep: no DONE within 30 s of wall time; is the map server built with the patch"
                   " and configured with lockstep: yes?\n");
            return 1;
        }
    }
    double clock0 = lockstep ? game_clock(bots, nbots) : ro_ms(), wall0 = ro_ms();
    double next = clock0, next_log = clock0 + log_every * 1000;
    long steps = 0;
    for (int i = 0; i < nbots; i++) {
        Brain* b = &brains[i];
        b->start_base = ro_total_base_exp(&bots[i]);
        b->start_job = ro_total_job_exp(&bots[i]);
        b->start_level = bots[i].base_level;
        b->start_job_level = bots[i].job_level;
        b->start_time = clock0;
        b->last_reset = clock0;
        if (bots[i].state == RO_IN_GAME)
            printf("ready %s: lv %d hp %d/%d at %s, %d items in inventory, potions %d\n", bots[i].user,
                   bots[i].base_level, bots[i].hp, bots[i].maxhp, bots[i].map, bots[i].ninv,
                   ro_inv_find(&bots[i], RED_POTION) ? ro_inv_find(&bots[i], RED_POTION)->amount : 0);
    }
    fflush(stdout);

    for (;;) {
        double clock = lockstep ? game_clock(bots, nbots) : ro_ms();
        if (clock - clock0 >= minutes * 60000) break;
        for (int i = 0; i < nbots; i++) {
            RoClient* c = &bots[i];
            if (ro_poll(c) < 0) {
                if (!brains[i].reported_error) {
                    printf("bot %s stopped: %s\n", c->user, c->err);
                    fflush(stdout);
                    brains[i].reported_error = 1;
                }
                continue;
            }
            Brain* b = &brains[i];
            if (reset_every > 0 && clock - b->last_reset > reset_every * 1000) {
                if (b->resets++ == 0 || b->resets % 10 == 0)
                    printf("reset %s: episode %d ended at lv %d/%d, %d kills so far\n", c->user, b->resets,
                           c->base_level, c->job_level, c->kills);
                ro_chat(c, "@rlreset");
                b->last_reset = clock;
                b->target = b->item = 0;
                b->resting = b->attacking = 0;
                continue;
            }
            step(c, b);
        }
        if (die_after >= 0 && clock - clock0 > die_after * 1000 && bots[0].state == RO_IN_GAME) {
            printf("test: %s types @die (deaths so far %d)\n", bots[0].user, bots[0].deaths);
            fflush(stdout);
            ro_chat(&bots[0], "@die");
            die_after = -1;
        }
        if (clock >= next_log) {
            for (int i = 0; i < nbots; i++) report(&bots[i], &brains[i], "status");
            if (lockstep)
                printf("lockstep: %ld steps, game %.1f s in wall %.1f s (%.1fx), %.0f bot steps/s\n", steps,
                       (clock - clock0) / 1000, (ro_ms() - wall0) / 1000,
                       (clock - clock0) / (ro_ms() - wall0), steps * (double)nbots * 1000 / (ro_ms() - wall0));
            fflush(stdout);
            next_log += log_every * 1000;
        }
        steps++;
        if (lockstep) {
            if (ro_lockstep_step_all(ptrs, nbots, 30000) < 0) {
                printf("lockstep: step %ld got no DONE within 30 s of wall time\n", steps);
                break;
            }
        } else {
            next += STEP_MS;
            double wait = next - ro_ms();
            if (wait > 0) usleep((useconds_t)(wait * 1000));
        }
    }
    double wall = ro_ms() - wall0, game = (lockstep ? game_clock(bots, nbots) : ro_ms()) - clock0;
    printf("clock: %ld steps, game %.1f s, wall %.1f s (%.2fx real time), %.0f bot steps/s\n", steps, game / 1000,
           wall / 1000, game / wall, steps * (double)nbots * 1000 / wall);

    int errors = 0, kills = 0;
    double hours = game / 3.6e6;
    int64_t exp = 0;
    for (int i = 0; i < nbots; i++) {
        report(&bots[i], &brains[i], "final");
        errors += bots[i].state == RO_ERROR;
        kills += bots[i].kills;
        exp += ro_total_base_exp(&bots[i]) - brains[i].start_base;
    }
    printf("summary: %d bots, %.1f min, %d kills, base exp %lld (%.0f per bot-hour), %d with errors\n", nbots,
           hours * 60, kills, (long long)exp, nbots && hours > 0 ? exp / hours / nbots : 0.0, errors);
    return errors ? 1 : 0;
}
