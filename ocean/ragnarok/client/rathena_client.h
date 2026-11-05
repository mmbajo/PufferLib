// rathena_client.h: a small Ragnarok Online client for rAthena, written for bots.
//
// Speaks the client version and rules rathena_packets.h was generated for (gen/gen_packets.py, from
// rAthena's own source), with packet obfuscation off: 20130618 pre-renewal and 20221005 renewal are
// tested. Packet ids, lengths and layouts all come from that header, so another version needs only a
// regenerated header (RO_PACKETS_H picks one other than the default). Header-only: every function is
// static. Link with -lz (zlib, for the map cache).
//
// Use: fill host/login_port/user/pass, call ro_login() (blocking, a few round trips), then call
// ro_poll() often; it reads what the map server sent and keeps the world model in RoClient up to
// date. The ro_* action functions send one request each. Positions are map cells.
//
// Two clocks. Real time (the default): c->now is the wall clock and a step is however long the
// caller waits. Lockstep (c->lockstep = 1, for a map server built with the lockstep patch): the
// server's clock stands still until every bot has called ro_lockstep_ready(); it then runs one
// step and sends each bot ZC_LOCKSTEP_DONE, and c->now becomes the server's game time.
// For many bots on one world use ro_lockstep_join() once, then ro_lockstep_step_all() per step.
// A bot's first READY may miss the step that is running (another bot's READY can arrive first), so
// joining keeps steps going until every bot has had a DONE, then runs one step with all of them.
#pragma once
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

// zlib's uncompress, declared here rather than through <zlib.h>, whose typedefs (Byte) clash with
// PufferLib's trainer, which compiles this header too. Same prototype, so <zlib.h> may come first.
#ifdef __cplusplus
extern "C" {
#endif
int uncompress(unsigned char* dest, unsigned long* dest_len, const unsigned char* source, unsigned long source_len);
#ifdef __cplusplus
}
#endif
#define RO_Z_OK 0

#ifdef RO_PACKETS_H
#include RO_PACKETS_H
#else
#include "rathena_packets.h"
#endif
// The lockstep patch's step-done packet (server/patches), the same in every version.
#define RO_ZC_LOCKSTEP_DONE 0x0cf1

#define RO_API static __attribute__((unused))

#define RO_MAX_UNITS 512
#define RO_MAX_GROUND 256
#define RO_MAX_INV 128
#define RO_MAX_SKILLS 64
#define RO_RBUF (1 << 17)

// rAthena's status ids (enum _sp in src/map/map.hpp).
enum {
    RO_SP_SPEED = 0, RO_SP_BASEEXP = 1, RO_SP_JOBEXP = 2, RO_SP_HP = 5, RO_SP_MAXHP = 6, RO_SP_SP = 7,
    RO_SP_MAXSP = 8, RO_SP_STATUSPOINT = 9, RO_SP_BASELEVEL = 11, RO_SP_SKILLPOINT = 12, RO_SP_STR = 13,
    RO_SP_AGI = 14, RO_SP_VIT = 15, RO_SP_INT = 16, RO_SP_DEX = 17, RO_SP_LUK = 18, RO_SP_ZENY = 20,
    RO_SP_NEXTBASEEXP = 22, RO_SP_NEXTJOBEXP = 23, RO_SP_WEIGHT = 24, RO_SP_MAXWEIGHT = 25,
    RO_SP_JOBLEVEL = 55,
};
// Unit object types in the spawn/idle/walking packets (clif_bl_type in src/map/clif.cpp).
enum { RO_OBJ_PC = 0, RO_OBJ_NPC = 1, RO_OBJ_ITEM = 2, RO_OBJ_SKILL = 3, RO_OBJ_MOB = 5, RO_OBJ_EVT_NPC = 6 };
// ZC_NOTIFY_ACT types (e_damage_type in src/map/clif.hpp).
enum { RO_DMG_NORMAL = 0, RO_DMG_SIT_DOWN = 2, RO_DMG_STAND_UP = 3, RO_DMG_ENDURE = 4, RO_DMG_MULTI_HIT = 8,
       RO_DMG_CRITICAL = 10, RO_DMG_LUCKY_DODGE = 11 };
// CZ_REQUEST_ACT actions.
enum { RO_ACT_ATTACK = 0, RO_ACT_SIT = 2, RO_ACT_STAND = 3, RO_ACT_ATTACK_REPEAT = 7 };
enum { RO_OFFLINE, RO_ENTERING, RO_IN_GAME, RO_ERROR };

typedef struct RoMap {
    char name[16];
    int xs, ys;
    uint8_t* walkable;  // xs * ys, 1 where a unit can stand
} RoMap;

typedef struct RoUnit {
    uint32_t id;
    uint8_t type;  // RO_OBJ_*
    int16_t cls;   // job or monster id
    int16_t speed; // ms per cell
    int16_t x0, y0, x1, y1;
    double t0;     // when the move from (x0, y0) to (x1, y1) started, client ms
    int32_t hp, maxhp;  // monsters only; maxhp 0 when unknown
    uint8_t boss;
    uint8_t hit_by_me;
    double last_hit_me;  // client ms, 0 if never
    int16_t opt1, opt2;  // body state (1 stone, 2 frozen, 3 stunned, 4 asleep, 6 turning to stone) and
    int32_t option;      // health state bits (RO_OPT2_*); option bits (hidden, cloaked, cart, ...)
} RoUnit;

// Health state bits (rAthena's OPT2_*): poisoned (and deadly poison), cursed, silenced, blind,
// bleeding.
#define RO_OPT2_POISON 0x01
#define RO_OPT2_CURSE 0x02
#define RO_OPT2_SILENCE 0x04
#define RO_OPT2_BLIND 0x10
#define RO_OPT2_BLEEDING 0x40
#define RO_OPT2_DPOISON 0x80

// A status effect on the bot (a status icon: rAthena's EFST_* id), with when it ends.
#define RO_MAX_EFFECTS 64
typedef struct RoEffect {
    int16_t id;
    double until, total;  // client ms; until 0: no end was given
} RoEffect;

typedef struct RoGround {
    uint32_t id;
    uint16_t item;
    int16_t x, y, amount;
    double seen;
} RoGround;

typedef struct RoSkill {
    uint16_t id, level, sp, range;
    int32_t inf;       // target type (rAthena's skill_db "TargetType": 1 enemy, 2 ground, 4 self, 16 friend)
    double ready_at;   // client ms; set from ZC_SKILL_POSTDELAY (cooldowns)
} RoSkill;

typedef struct RoInv {
    int16_t index;   // as the client sends it back (server index + 2)
    uint16_t item;
    int16_t amount;
    uint32_t location, worn;
    uint8_t type;
} RoInv;

typedef struct RoClient {
    // Settings, filled by the caller.
    char host[64];
    int login_port;
    char user[24], pass[24];
    const char* db_dir;  // rAthena's db/ directory, for map_cache.dat
    int verbose;
    int lockstep;        // 1: the server steps only when told (see the top of this file)

    // Session.
    int state, fd;
    char err[200];
    uint32_t aid, login_id1, login_id2, cid;
    uint8_t sex;
    char name[24];
    uint8_t rbuf[RO_RBUF];
    int rlen;
    double now, last_ticksend;
    int step_pending;       // lockstep: READY sent, DONE not yet received
    int lockstep_joined;    // lockstep: has had a DONE, so the server's barrier waits for it
    uint32_t steps_done;    // lockstep: the server's step count in the last DONE
    uint32_t too_far_id;    // the last target the server said is out of reach (it never walks
    double too_far_at;      // a player into range: the client has to)

    // Self.
    char map[16];
    const RoMap* cells;
    int16_t x0, y0, x1, y1;
    double t0;
    int speed;
    int hp, maxhp, sp, maxsp, base_level, job_level, status_points, skill_points, zeny, weight, max_weight;
    int str, agi, vit, int_, dex, luk, attack_range;
    int64_t base_exp, job_exp, next_base_exp, next_job_exp, base_offset, job_offset;
    int sitting, dead;
    int job;                   // rAthena job id: 0 Novice, 1 Swordman, 2 Mage, 3 Archer, 4 Acolyte, ...
    RoSkill skills[RO_MAX_SKILLS];
    int nskills;
    double cast_until;         // casting a skill until then (client ms), 0 when not
    uint16_t cast_skill;
    double skill_failed_at;    // the last rejected skill request (ZC_ACK_TOUSESKILL)
    uint16_t skill_failed_id, skill_failed_cause;
    RoEffect effects[RO_MAX_EFFECTS];  // status icons on the bot
    int neffects;
    int16_t opt1, opt2;        // the bot's body and health state, as for units
    int32_t option;

    // World around it.
    RoUnit units[RO_MAX_UNITS];
    int nunits;
    RoGround ground[RO_MAX_GROUND];
    int nground;
    RoInv inv[RO_MAX_INV];
    int ninv;

    // Counters.
    int kills, deaths, picked, potions, map_loads, unknown_packets, skills_used, skills_failed;
    int64_t damage_dealt, damage_taken;
    uint32_t packets_seen;
} RoClient;

// ---- small helpers ---------------------------------------------------------------------------

RO_API double ro_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

RO_API int ro_fail(RoClient* c, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(c->err, sizeof c->err, fmt, ap);
    va_end(ap);
    c->state = RO_ERROR;
    if (c->fd >= 0) {
        close(c->fd);
        c->fd = -1;
    }
    return -1;
}

RO_API uint16_t ro_u16(const uint8_t* p) { return (uint16_t)(p[0] | p[1] << 8); }

RO_API uint32_t ro_u32(const uint8_t* p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

RO_API void ro_put16(uint8_t* p, uint16_t v) { p[0] = v & 0xff; p[1] = v >> 8; }

RO_API void ro_put32(uint8_t* p, uint32_t v) {
    p[0] = v & 0xff; p[1] = (v >> 8) & 0xff; p[2] = (v >> 16) & 0xff; p[3] = v >> 24;
}

// Positions are packed into 10-bit fields (WBUFPOS/WBUFPOS2 in rAthena).
RO_API void ro_get_pos(const uint8_t* p, int16_t* x, int16_t* y) {
    *x = (int16_t)((p[0] << 2) | (p[1] >> 6));
    *y = (int16_t)(((p[1] & 0x3f) << 4) | (p[2] >> 4));
}

RO_API void ro_get_pos2(const uint8_t* p, int16_t* x0, int16_t* y0, int16_t* x1, int16_t* y1) {
    *x0 = (int16_t)((p[0] << 2) | (p[1] >> 6));
    *y0 = (int16_t)(((p[1] & 0x3f) << 4) | (p[2] >> 4));
    *x1 = (int16_t)(((p[2] & 0x0f) << 6) | (p[3] >> 2));
    *y1 = (int16_t)(((p[3] & 0x03) << 8) | p[4]);
}

RO_API void ro_put_pos(uint8_t* p, int x, int y, int dir) {
    p[0] = (uint8_t)(x >> 2);
    p[1] = (uint8_t)((x << 6) | ((y >> 4) & 0x3f));
    p[2] = (uint8_t)((y << 4) | (dir & 0x0f));
}

// Where a unit moving from (x0, y0) to (x1, y1) since t0 at `speed` ms per cell is at `now`.
RO_API void ro_interp(int x0, int y0, int x1, int y1, double t0, int speed, double now, int* x, int* y) {
    int dx = x1 - x0, dy = y1 - y0;
    int steps = abs(dx) > abs(dy) ? abs(dx) : abs(dy);
    if (steps == 0 || speed <= 0) {
        *x = x1; *y = y1;
        return;
    }
    // Diagonal steps cost about 1.4 straight ones in rAthena (MOVE_DIAGONAL_COST 14 vs 10).
    double diag = abs(dx) < abs(dy) ? abs(dx) : abs(dy);
    double cost = speed * (steps - diag + diag * 1.4) / steps;
    double done = (now - t0) / cost;
    if (done >= steps) {
        *x = x1; *y = y1;
        return;
    }
    if (done < 0) done = 0;
    *x = x0 + (int)lround(dx * done / steps);
    *y = y0 + (int)lround(dy * done / steps);
}

RO_API void ro_self_pos(const RoClient* c, int* x, int* y) {
    ro_interp(c->x0, c->y0, c->x1, c->y1, c->t0, c->speed, c->now, x, y);
}

RO_API void ro_unit_pos(const RoClient* c, const RoUnit* u, int* x, int* y) {
    ro_interp(u->x0, u->y0, u->x1, u->y1, u->t0, u->speed, c->now, x, y);
}

RO_API int ro_self_moving(const RoClient* c) {
    int x, y;
    ro_self_pos(c, &x, &y);
    return x != c->x1 || y != c->y1;
}

RO_API int64_t ro_total_base_exp(const RoClient* c) { return c->base_offset + c->base_exp; }
RO_API int64_t ro_total_job_exp(const RoClient* c) { return c->job_offset + c->job_exp; }

// ---- map cache -------------------------------------------------------------------------------

// rAthena's map_cache.dat: u32 size, u16 count, u16 pad, then per map name[12] i16 xs i16 ys i32 len
// and len bytes of zlib-compressed gat cell types. The map server reads db/import, then db/pre-re or
// db/re by its rules (renewal has its own prt_fild08, for one), then db.
static pthread_mutex_t ro_maps_lock = PTHREAD_MUTEX_INITIALIZER;
static RoMap ro_maps[64];
static int ro_nmaps;

RO_API int ro_cache_find(const char* path, const char* name, RoMap* out) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    uint8_t head[8], entry[20];
    int found = 0;
    if (fread(head, 1, 8, f) == 8) {
        int count = ro_u16(head + 4);
        for (int i = 0; i < count && fread(entry, 1, 20, f) == 20; i++) {
            int xs = (int16_t)ro_u16(entry + 12), ys = (int16_t)ro_u16(entry + 14);
            uint32_t len = entry[16] | entry[17] << 8 | entry[18] << 16 | (uint32_t)entry[19] << 24;
            if (strncmp((const char*)entry, name, 12) != 0) {
                fseek(f, len, SEEK_CUR);
                continue;
            }
            uint8_t* z = (uint8_t*)malloc(len);
            unsigned long n = (unsigned long)xs * ys;
            uint8_t* gat = (uint8_t*)malloc(n);
            if (fread(z, 1, len, f) == len && uncompress(gat, &n, z, len) == RO_Z_OK && n == (unsigned long)xs * ys) {
                for (unsigned long k = 0; k < n; k++) gat[k] = gat[k] != 1 && gat[k] != 5;  // map_gat2cell
                snprintf(out->name, sizeof out->name, "%s", name);
                out->xs = xs; out->ys = ys; out->walkable = gat;
                found = 1;
            } else {
                free(gat);
            }
            free(z);
            break;
        }
    }
    fclose(f);
    return found;
}

RO_API const RoMap* ro_map_get(const char* db_dir, const char* name) {
    const RoMap* m = NULL;
    pthread_mutex_lock(&ro_maps_lock);
    for (int i = 0; i < ro_nmaps && !m; i++)
        if (strcmp(ro_maps[i].name, name) == 0) m = &ro_maps[i];
    if (!m && db_dir && ro_nmaps < 64) {
        const char* subdirs[] = {"import/", RO_RULES_DIR "/", ""};
        char path[512];
        for (int i = 0; i < 3 && !m; i++) {
            snprintf(path, sizeof path, "%s/%smap_cache.dat", db_dir, subdirs[i]);
            if (ro_cache_find(path, name, &ro_maps[ro_nmaps])) m = &ro_maps[ro_nmaps++];
        }
    }
    pthread_mutex_unlock(&ro_maps_lock);
    return m;
}

RO_API int ro_walkable(const RoMap* m, int x, int y) {
    return m && x >= 0 && y >= 0 && x < m->xs && y < m->ys && m->walkable[x + y * m->xs];
}

// ---- sockets ---------------------------------------------------------------------------------

RO_API int ro_dial(RoClient* c, int port) {
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof hints);
    char ports[16];
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    snprintf(ports, sizeof ports, "%d", port);
    if (getaddrinfo(c->host, ports, &hints, &res) != 0 || !res) return ro_fail(c, "cannot resolve %s", c->host);
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct timeval tv = {10, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    if (connect(fd, res->ai_addr, res->ai_addrlen) != 0) {
        freeaddrinfo(res);
        close(fd);
        return ro_fail(c, "connect %s:%d: %s", c->host, port, strerror(errno));
    }
    freeaddrinfo(res);
    if (c->fd >= 0) close(c->fd);
    c->fd = fd;
    c->rlen = 0;
    return 0;
}

RO_API int ro_send(RoClient* c, const void* buf, int len) {
    if (c->fd < 0) return -1;
    const uint8_t* p = (const uint8_t*)buf;
    while (len > 0) {
        ssize_t n = send(c->fd, p, len, MSG_NOSIGNAL);
        if (n <= 0) return ro_fail(c, "send: %s", n == 0 ? "closed" : strerror(errno));
        p += n;
        len -= n;
    }
    return 0;
}

// Blocking read of exactly n bytes (login and char phases).
RO_API int ro_read_exact(RoClient* c, uint8_t* buf, int n) {
    for (int got = 0; got < n;) {
        ssize_t r = recv(c->fd, buf + got, n - got, 0);
        if (r <= 0) return ro_fail(c, "recv: %s", r == 0 ? "closed by server" : strerror(errno));
        got += r;
    }
    return 0;
}

// Blocking read of one whole packet into buf; returns its length or -1.
RO_API int ro_read_packet(RoClient* c, uint8_t* buf, int cap) {
    if (ro_read_exact(c, buf, 2) < 0) return -1;
    int cmd = ro_u16(buf), len = ro_packet_len(cmd);
    if (len == 0) return ro_fail(c, "unknown packet 0x%04x during login", cmd);
    if (len < 0) {
        if (ro_read_exact(c, buf + 2, 2) < 0) return -1;
        len = ro_u16(buf + 2);
    }
    if (len < 4 && ro_packet_len(cmd) < 0) return ro_fail(c, "bad length %d for 0x%04x", len, cmd);
    if (len > cap) return ro_fail(c, "packet 0x%04x of %d bytes is too large", cmd, len);
    int have = ro_packet_len(cmd) < 0 ? 4 : 2;
    if (len > have && ro_read_exact(c, buf + have, len - have) < 0) return -1;
    return len;
}

// ---- login: login server, char server, map server --------------------------------------------

RO_API void ro_reset_world(RoClient* c) {
    c->nunits = 0;
    c->nground = 0;
}

// The map server does not resend what the char list already told the client (levels, exp, zeny).
RO_API void ro_from_charinfo(RoClient* c, const RO_CHARACTER_INFO* ci) {
    c->cid = ci->GID;
    snprintf(c->name, sizeof c->name, "%.23s", ci->name);
    c->base_level = ci->level;
    c->job_level = ci->joblevel;
    c->base_exp = ci->exp;
    c->job_exp = ci->jobexp;
    c->zeny = ci->money;
    c->hp = ci->hp; c->maxhp = ci->maxhp; c->sp = ci->sp; c->maxsp = ci->maxsp;
    c->status_points = ci->jobpoint;
    c->skill_points = ci->sppoint;
    c->str = ci->Str; c->agi = ci->Agi; c->vit = ci->Vit; c->int_ = ci->Int; c->dex = ci->Dex; c->luk = ci->Luk;
    c->speed = ci->speed > 0 ? ci->speed : 150;
    c->job = ci->job;
}

RO_API int ro_login(RoClient* c, int slot) {
    static uint8_t buf[1 << 16];
    c->fd = -1;
    c->state = RO_ENTERING;
    c->speed = 150;
    c->now = ro_ms();

    // 1. Login server: CA_LOGIN, then AC_ACCEPT_LOGIN with the char server list.
    if (ro_dial(c, c->login_port) < 0) return -1;
    RO_PACKET_CA_LOGIN login;
    memset(&login, 0, sizeof login);
    login.packetType = RO_HEADER_CA_LOGIN;
    login.version = 55;
    snprintf(login.username, sizeof login.username, "%s", c->user);
    snprintf(login.password, sizeof login.password, "%s", c->pass);
    if (ro_send(c, &login, sizeof login) < 0) return -1;
    int len = ro_read_packet(c, buf, sizeof buf);
    if (len < 0) return -1;
    if (ro_u16(buf) != RO_HEADER_AC_ACCEPT_LOGIN) {
        // AC_REFUSE_LOGIN (0x83e from 2010 on, 0x6a before): 0 unknown id, 1 wrong password,
        // 3 rejected, 5 client version, 6 banned.
        uint32_t error = 0;
        if (ro_u16(buf) == RO_HEADER_SC_NOTIFY_BAN) error = buf[2];  // 2 = already logged in
        else if (len >= 6) memcpy(&error, buf + 2, 4);
        return ro_fail(c, "login refused: packet 0x%04x, error %u", ro_u16(buf), error);
    }
    RO_PACKET_AC_ACCEPT_LOGIN* acc = (RO_PACKET_AC_ACCEPT_LOGIN*)(buf);
    const RO_PACKET_AC_ACCEPT_LOGIN_sub* servers = (const RO_PACKET_AC_ACCEPT_LOGIN_sub*)(buf + sizeof *acc);
    if (len < (int)(sizeof *acc + sizeof *servers)) return ro_fail(c, "no char server listed");
    c->aid = acc->AID;
    c->login_id1 = acc->login_id1;
    c->login_id2 = acc->login_id2;
    c->sex = acc->sex;
    int char_port = servers[0].port;

    // 2. Char server: CH_ENTER; it echoes the account id, then sends the character list.
    if (ro_dial(c, char_port) < 0) return -1;
    uint8_t enter[17] = {0};
    ro_put16(enter, 0x65);  // CH_ENTER, the same in every version (rAthena has no name for it)
    ro_put32(enter + 2, c->aid);
    ro_put32(enter + 6, c->login_id1);
    ro_put32(enter + 10, c->login_id2);
    enter[16] = c->sex;
    if (ro_send(c, enter, sizeof enter) < 0) return -1;
    if (ro_read_exact(c, buf, 4) < 0) return -1;
    int have_char = 0;
    for (int got_list = 0; !got_list;) {
        if ((len = ro_read_packet(c, buf, sizeof buf)) < 0) return -1;
        int cmd = ro_u16(buf);
        if (cmd == RO_HEADER_HC_REFUSE_ENTER) return ro_fail(c, "char server refused: error %d", buf[2]);
        if (cmd != RO_HEADER_HC_ACCEPT_ENTER) continue;  // HC_ACCEPT_ENTER2 (0x82d) comes first
        const RO_CHARACTER_INFO* chars = (const RO_CHARACTER_INFO*)(buf + sizeof(RO_PACKET_HC_ACCEPT_ENTER));
        int n = (len - (int)sizeof(RO_PACKET_HC_ACCEPT_ENTER)) / (int)sizeof(RO_CHARACTER_INFO);
        for (int i = 0; i < n; i++) {
            if (chars[i].CharNum == slot) {
                have_char = 1;
                ro_from_charinfo(c, &chars[i]);
            }
        }
        got_list = 1;
    }
    // HC_CHARLIST_NOTIFY (0x9a0) and HC_BLOCK_CHARACTER (0x20d) follow; the next read skips them.
    if (!have_char) {
        RO_PACKET_CH_MAKE_CHAR make;
        memset(&make, 0, sizeof make);
        make.packetType = RO_HEADER_CH_MAKE_CHAR;
        snprintf(make.name, sizeof make.name, "%s", c->user);
        make.slot = slot;
        make.hair_color = 0;
        make.hair_style = 1;
#ifdef RO_HAS_PACKET_CH_MAKE_CHAR_job
        make.job = 0;  // Novice (0x0a39, from 2015-10 on, names the starting job and the sex)
#endif
#ifdef RO_HAS_PACKET_CH_MAKE_CHAR_sex
        make.sex = c->sex;
#endif
        if (ro_send(c, &make, sizeof make) < 0) return -1;
    }
    for (int made = have_char; !made;) {
        if ((len = ro_read_packet(c, buf, sizeof buf)) < 0) return -1;
        int cmd = ro_u16(buf);
        if (cmd == RO_HEADER_HC_REFUSE_MAKECHAR) return ro_fail(c, "character creation refused: error %d", buf[2]);
        if (cmd != RO_HEADER_HC_ACCEPT_MAKECHAR) continue;
        ro_from_charinfo(c, &((RO_PACKET_HC_ACCEPT_MAKECHAR*)buf)->character);
        made = 1;
    }
    RO_PACKET_CH_SELECT_CHAR sel = {RO_HEADER_CH_SELECT_CHAR, (uint8_t)slot};
    if (ro_send(c, &sel, sizeof sel) < 0) return -1;
    for (;;) {
        if ((len = ro_read_packet(c, buf, sizeof buf)) < 0) return -1;
        int cmd = ro_u16(buf);
        if (cmd == RO_HEADER_HC_REFUSE_ENTER) return ro_fail(c, "character select refused: error %d", buf[2]);
        if (cmd == RO_HEADER_HC_NOTIFY_ZONESVR) break;
    }
    RO_PACKET_HC_NOTIFY_ZONESVR* zone = (RO_PACKET_HC_NOTIFY_ZONESVR*)(buf);
    c->cid = zone->CID;
    snprintf(c->map, sizeof c->map, "%.15s", zone->mapname);
    char* dot = strchr(c->map, '.');
    if (dot) *dot = 0;
    int map_port = zone->port;  // the advertised ip is ignored: the client always dials c->host

    // 3. Map server: CZ_ENTER, then ZC_ACCEPT_ENTER arrives in ro_poll.
    if (ro_dial(c, map_port) < 0) return -1;
    uint8_t want[RO_CZ_WANTTOCONNECTION_LEN] = {0};
    ro_put16(want, RO_CZ_WANTTOCONNECTION);
    ro_put32(want + RO_CZ_WANTTOCONNECTION_P0, c->aid);
    ro_put32(want + RO_CZ_WANTTOCONNECTION_P1, c->cid);
    ro_put32(want + RO_CZ_WANTTOCONNECTION_P2, c->login_id1);
    ro_put32(want + RO_CZ_WANTTOCONNECTION_P3, (uint32_t)c->now);
    want[RO_CZ_WANTTOCONNECTION_P4] = c->sex;
    if (ro_send(c, want, sizeof want) < 0) return -1;
    fcntl(c->fd, F_SETFL, fcntl(c->fd, F_GETFL) | O_NONBLOCK);
    c->cells = ro_map_get(c->db_dir, c->map);
    ro_reset_world(c);
    return 0;
}

// ---- actions (map server) --------------------------------------------------------------------

RO_API int ro_load_end(RoClient* c) {
    uint8_t p[2];
    ro_put16(p, RO_CZ_LOADENDACK);
    return ro_send(c, p, 2);
}

RO_API int ro_ticksend(RoClient* c) {
    uint8_t p[RO_CZ_TICKSEND_LEN] = {0};
    ro_put16(p, RO_CZ_TICKSEND);
    ro_put32(p + RO_CZ_TICKSEND_P0, (uint32_t)c->now);
    c->last_ticksend = c->now;
    return ro_send(c, p, sizeof p);
}

RO_API int ro_walk(RoClient* c, int x, int y) {
    uint8_t p[RO_CZ_WALKTOXY_LEN] = {0};
    ro_put16(p, RO_CZ_WALKTOXY);
    ro_put_pos(p + RO_CZ_WALKTOXY_P0, x, y, 0);
    return ro_send(c, p, sizeof p);
}

RO_API int ro_act(RoClient* c, uint32_t target, int action) {
    uint8_t p[RO_CZ_ACTIONREQUEST_LEN] = {0};
    ro_put16(p, RO_CZ_ACTIONREQUEST);
    ro_put32(p + RO_CZ_ACTIONREQUEST_P0, target);
    p[RO_CZ_ACTIONREQUEST_P1] = (uint8_t)action;
    return ro_send(c, p, sizeof p);
}

RO_API int ro_attack(RoClient* c, uint32_t target) { return ro_act(c, target, RO_ACT_ATTACK_REPEAT); }
RO_API int ro_sit(RoClient* c, int sit) { return ro_act(c, 0, sit ? RO_ACT_SIT : RO_ACT_STAND); }

RO_API int ro_pickup(RoClient* c, uint32_t ground_id) {
    uint8_t p[RO_CZ_TAKEITEM_LEN] = {0};
    ro_put16(p, RO_CZ_TAKEITEM);
    ro_put32(p + RO_CZ_TAKEITEM_P0, ground_id);
    return ro_send(c, p, sizeof p);
}

RO_API int ro_use_item(RoClient* c, int index) {
    uint8_t p[RO_CZ_USEITEM_LEN] = {0};
    ro_put16(p, RO_CZ_USEITEM);
    ro_put16(p + RO_CZ_USEITEM_P0, (uint16_t)index);
    ro_put32(p + RO_CZ_USEITEM_P1, c->aid);
    return ro_send(c, p, sizeof p);
}

RO_API int ro_equip(RoClient* c, int index, uint32_t location) {
    RO_PACKET_CZ_REQ_WEAR_EQUIP p = {RO_CZ_EQUIPITEM, (uint16_t)index, location};
    return ro_send(c, &p, sizeof p);
}

RO_API int ro_respawn(RoClient* c) {
    uint8_t p[RO_CZ_RESTART_LEN] = {0};
    ro_put16(p, RO_CZ_RESTART);
    p[RO_CZ_RESTART_P0] = 0;  // back to the save point
    return ro_send(c, p, sizeof p);
}

RO_API int ro_status_up(RoClient* c, int sp_type, int amount) {
    uint8_t p[RO_CZ_STATUSUP_LEN] = {0};
    ro_put16(p, RO_CZ_STATUSUP);
    ro_put16(p + RO_CZ_STATUSUP_P0, (uint16_t)sp_type);
    p[RO_CZ_STATUSUP_P1] = (uint8_t)amount;
    return ro_send(c, p, sizeof p);
}

RO_API const RoSkill* ro_skill_find(const RoClient* c, int id) {
    for (int i = 0; i < c->nskills; i++)
        if (c->skills[i].id == id && c->skills[i].level > 0) return &c->skills[i];
    return NULL;
}

// A status effect on the bot: the ms it has left (1e9 when no end was given), or 0 if it is off.
RO_API double ro_effect_left(const RoClient* c, int id) {
    for (int i = 0; i < c->neffects; i++)
        if (c->effects[i].id == id) {
            if (c->effects[i].until <= 0) return 1e9;
            return c->effects[i].until > c->now ? c->effects[i].until - c->now : 0;
        }
    return 0;
}

// The share of a status effect's time that is left: 1 when no end was given, 0 if it is off.
RO_API float ro_effect_share(const RoClient* c, int id) {
    for (int i = 0; i < c->neffects; i++)
        if (c->effects[i].id == id) {
            if (c->effects[i].until <= 0 || c->effects[i].total <= 0) return 1;
            return c->effects[i].until > c->now ? (float)((c->effects[i].until - c->now) / c->effects[i].total) : 0;
        }
    return 0;
}

static void ro_effect_set(RoClient* c, int id, int on, double total, double left) {
    int i = 0;
    while (i < c->neffects && c->effects[i].id != id) i++;
    if (!on) {
        if (i < c->neffects) c->effects[i] = c->effects[--c->neffects];
        return;
    }
    if (i == c->neffects) {
        if (c->neffects == RO_MAX_EFFECTS) return;
        c->neffects++;
    }
    c->effects[i].id = (int16_t)id;
    c->effects[i].until = left > 0 ? c->now + left : 0;
    c->effects[i].total = total > 0 ? total : left;
}

// Use a skill on a unit (a monster, a player, or ourselves: target = c->aid).
RO_API int ro_use_skill(RoClient* c, int id, int level, uint32_t target) {
    uint8_t p[RO_CZ_USESKILLTOID_LEN] = {0};
    ro_put16(p, RO_CZ_USESKILLTOID);
    ro_put16(p + RO_CZ_USESKILLTOID_P0, (uint16_t)level);
    ro_put16(p + RO_CZ_USESKILLTOID_P1, (uint16_t)id);
    ro_put32(p + RO_CZ_USESKILLTOID_P2, target);
    c->skills_used++;
    return ro_send(c, p, sizeof p);
}

// Use a skill on a cell (ground skills such as Arrow Shower).
RO_API int ro_use_skill_pos(RoClient* c, int id, int level, int x, int y) {
    uint8_t p[RO_CZ_USESKILLTOPOS_LEN] = {0};
    ro_put16(p, RO_CZ_USESKILLTOPOS);
    ro_put16(p + RO_CZ_USESKILLTOPOS_P0, (uint16_t)level);
    ro_put16(p + RO_CZ_USESKILLTOPOS_P1, (uint16_t)id);
    ro_put16(p + RO_CZ_USESKILLTOPOS_P2, (uint16_t)x);
    ro_put16(p + RO_CZ_USESKILLTOPOS_P3, (uint16_t)y);
    c->skills_used++;
    return ro_send(c, p, sizeof p);
}

// Spend one status point by a fixed rule: the stat with the lowest value / weight among those with
// weight > 0 (weights for STR, AGI, VIT, INT, DEX, LUK). Returns 1 if a request went out.
RO_API int ro_spend_status(RoClient* c, const int weight[6]) {
    if (c->status_points <= 0) return 0;
    int value[6] = {c->str, c->agi, c->vit, c->int_, c->dex, c->luk};
    int best = -1;
    double best_score = 1e9;
    for (int s = 0; s < 6; s++) {
        int cost = (value[s] - 1) / 10 + 2;  // pre-renewal raise cost
        if (weight[s] <= 0 || value[s] >= 99 || cost > c->status_points) continue;
        double score = (double)value[s] / weight[s];
        if (score < best_score) best_score = score, best = s;
    }
    if (best < 0) return 0;
    return ro_status_up(c, RO_SP_STR + best, 1) == 0;
}

RO_API int ro_chat(RoClient* c, const char* text) {
    uint8_t p[256];
    int n = snprintf((char*)p + 4, sizeof p - 4, "%s : %s", c->name, text);
    if (n < 0 || n >= (int)sizeof p - 5) return -1;
    ro_put16(p, RO_CZ_GLOBALMESSAGE);
    ro_put16(p + 2, (uint16_t)(n + 5));
    return ro_send(c, p, n + 5);
}

// Lockstep: this bot's actions for the coming step are sent.
RO_API int ro_lockstep_ready(RoClient* c) {
    uint8_t p[RO_CZ_LOCKSTEPREADY_LEN];
    ro_put16(p, RO_CZ_LOCKSTEPREADY);
    c->step_pending = 1;
    return ro_send(c, p, sizeof p);
}

RO_API const RoInv* ro_inv_find(const RoClient* c, int item) {
    for (int i = 0; i < c->ninv; i++)
        if (c->inv[i].item == item && c->inv[i].amount > 0) return &c->inv[i];
    return NULL;
}

// ---- world model -----------------------------------------------------------------------------

RO_API RoUnit* ro_unit_find(RoClient* c, uint32_t id) {
    for (int i = 0; i < c->nunits; i++)
        if (c->units[i].id == id) return &c->units[i];
    return NULL;
}

RO_API RoUnit* ro_unit_upsert(RoClient* c, uint32_t id) {
    RoUnit* u = ro_unit_find(c, id);
    if (u) return u;
    if (c->nunits == RO_MAX_UNITS) return NULL;
    u = &c->units[c->nunits++];
    memset(u, 0, sizeof *u);
    u->id = id;
    return u;
}

RO_API void ro_unit_remove(RoClient* c, uint32_t id) {
    for (int i = 0; i < c->nunits; i++)
        if (c->units[i].id == id) {
            c->units[i] = c->units[--c->nunits];
            return;
        }
}

RO_API void ro_ground_remove(RoClient* c, uint32_t id) {
    for (int i = 0; i < c->nground; i++)
        if (c->ground[i].id == id) {
            c->ground[i] = c->ground[--c->nground];
            return;
        }
}

RO_API void ro_ground_add(RoClient* c, uint32_t id, int item, int x, int y, int amount) {
    ro_ground_remove(c, id);
    if (c->nground == RO_MAX_GROUND) return;
    RoGround* g = &c->ground[c->nground++];
    g->id = id; g->item = (uint16_t)item; g->x = (int16_t)x; g->y = (int16_t)y; g->amount = (int16_t)amount;
    g->seen = c->now;
}

RO_API RoInv* ro_inv_slot(RoClient* c, int index) {
    for (int i = 0; i < c->ninv; i++)
        if (c->inv[i].index == index) return &c->inv[i];
    if (c->ninv == RO_MAX_INV) return NULL;
    RoInv* v = &c->inv[c->ninv++];
    memset(v, 0, sizeof *v);
    v->index = (int16_t)index;
    return v;
}

RO_API void ro_inv_take(RoClient* c, int index, int count) {
    for (int i = 0; i < c->ninv; i++)
        if (c->inv[i].index == index) {
            c->inv[i].amount -= count;
            if (c->inv[i].amount <= 0) c->inv[i] = c->inv[--c->ninv];
            return;
        }
}

RO_API void ro_set_status(RoClient* c, int var, int64_t v) {
    switch (var) {
    case RO_SP_SPEED: c->speed = (int)v; break;
    case RO_SP_HP:
        c->hp = (int)v;
        if (c->hp <= 0 && !c->dead) { c->dead = 1; c->deaths++; }
        if (c->hp > 0) c->dead = 0;
        break;
    case RO_SP_MAXHP: c->maxhp = (int)v; break;
    case RO_SP_SP: c->sp = (int)v; break;
    case RO_SP_MAXSP: c->maxsp = (int)v; break;
    case RO_SP_STATUSPOINT: c->status_points = (int)v; break;
    case RO_SP_SKILLPOINT: c->skill_points = (int)v; break;
    case RO_SP_ZENY: c->zeny = (int)v; break;
    case RO_SP_WEIGHT: c->weight = (int)v; break;
    case RO_SP_MAXWEIGHT: c->max_weight = (int)v; break;
    case RO_SP_STR: c->str = (int)v; break;
    case RO_SP_AGI: c->agi = (int)v; break;
    case RO_SP_VIT: c->vit = (int)v; break;
    case RO_SP_INT: c->int_ = (int)v; break;
    case RO_SP_DEX: c->dex = (int)v; break;
    case RO_SP_LUK: c->luk = (int)v; break;
    // A level-up sends the new level before the leftover exp, so the old "next" is still current.
    // A lower level is a reset (@rlreset): the exp so far is banked, so totals never go down.
    case RO_SP_BASELEVEL:
        if (c->base_level && v > c->base_level) c->base_offset += c->next_base_exp;
        if (c->base_level && v < c->base_level) { c->base_offset += c->base_exp; c->base_exp = 0; }
        c->base_level = (int)v;
        break;
    case RO_SP_JOBLEVEL:
        if (c->job_level && v > c->job_level) c->job_offset += c->next_job_exp;
        if (c->job_level && v < c->job_level) { c->job_offset += c->job_exp; c->job_exp = 0; }
        c->job_level = (int)v;
        break;
    case RO_SP_BASEEXP: c->base_exp = v; break;
    case RO_SP_JOBEXP: c->job_exp = v; break;
    case RO_SP_NEXTBASEEXP: c->next_base_exp = v; break;
    case RO_SP_NEXTJOBEXP: c->next_job_exp = v; break;
    }
}

RO_API void ro_unit_from_packet(RoClient* c, const uint8_t* buf, int cmd) {
    // The three unit packets share their head; position fields differ (see rathena_packets.h).
    const RO_packet_idle_unit* idle = (const RO_packet_idle_unit*)(buf);
    RO_STATIC_ASSERT(offsetof(RO_packet_idle_unit, speed) == offsetof(RO_packet_unit_walking, speed)
                     && offsetof(RO_packet_idle_unit, job) == offsetof(RO_packet_spawn_unit, job),
                     "unit packets share their head");
#ifdef RO_HAS_packet_idle_unit_AID
    uint32_t id = idle->AID;  // from 2013-12 on, GID is a player's character id and 0 for the rest
#else
    uint32_t id = idle->GID;
#endif
    if (id == c->aid) return;
    RoUnit* u = ro_unit_upsert(c, id);
    if (!u) return;
    u->type = idle->objecttype;
    u->cls = idle->job;
    u->speed = idle->speed;
#ifdef RO_HAS_packet_idle_unit_bodyState
    u->opt1 = idle->bodyState;
    u->opt2 = idle->healthState;
    u->option = idle->effectState;
#endif
    u->t0 = c->now;
    if (cmd == RO_TYPE_unit_walkingType) {
        const RO_packet_unit_walking* w = (const RO_packet_unit_walking*)(buf);
        ro_get_pos2(w->MoveData, &u->x0, &u->y0, &u->x1, &u->y1);
        u->maxhp = w->maxHP; u->hp = w->HP; u->boss = w->isBoss;
        if (c->lockstep) u->t0 = (double)w->moveStartTime;
    } else {
        const uint8_t* pos = cmd == RO_TYPE_spawn_unitType ? ((const RO_packet_spawn_unit*)buf)->PosDir : idle->PosDir;
        ro_get_pos(pos, &u->x0, &u->y0);
        u->x1 = u->x0; u->y1 = u->y0;
        if (cmd == RO_TYPE_spawn_unitType) {
            const RO_packet_spawn_unit* s = (const RO_packet_spawn_unit*)(buf);
            u->maxhp = s->maxHP; u->hp = s->HP; u->boss = s->isBoss;
        } else {
            u->maxhp = idle->maxHP; u->hp = idle->HP; u->boss = idle->isBoss;
        }
    }
}

RO_API void ro_handle(RoClient* c, const uint8_t* buf, int len) {
    int cmd = ro_u16(buf);
    c->packets_seen++;
    switch (cmd) {
    case RO_HEADER_ZC_ACCEPT_ENTER: {  // we are on the map
        const RO_PACKET_ZC_ACCEPT_ENTER* p = (const RO_PACKET_ZC_ACCEPT_ENTER*)(buf);
        ro_get_pos(p->posDir, &c->x0, &c->y0);
        c->x1 = c->x0; c->y1 = c->y0; c->t0 = c->now;
        c->state = RO_IN_GAME;
        c->map_loads++;
        ro_load_end(c);
        ro_ticksend(c);
        break;
    }
    case RO_HEADER_ZC_NPCACK_MAPMOVE: {  // warped, possibly on the same map
        const RO_PACKET_ZC_NPCACK_MAPMOVE* p = (const RO_PACKET_ZC_NPCACK_MAPMOVE*)(buf);
        snprintf(c->map, sizeof c->map, "%.15s", p->mapName);
        char* dot = strchr(c->map, '.');
        if (dot) *dot = 0;
        c->cells = ro_map_get(c->db_dir, c->map);
        c->x0 = c->x1 = (int16_t)p->xPos; c->y0 = c->y1 = (int16_t)p->yPos; c->t0 = c->now;
        c->sitting = 0;
        ro_reset_world(c);
        c->map_loads++;
        ro_load_end(c);
        break;
    }
    case RO_HEADER_ZC_NPCACK_SERVERMOVE: ro_fail(c, "moved to another map server; not supported"); break;
    case RO_HEADER_SC_NOTIFY_BAN: ro_fail(c, "map server refused the connection: error %d", buf[2]); break;
    case RO_HEADER_ZC_NOTIFY_VANISH: {  // type 1 = died
        const RO_PACKET_ZC_NOTIFY_VANISH* p = (const RO_PACKET_ZC_NOTIFY_VANISH*)(buf);
        if (p->gid == c->aid) {
            if (p->type == 1 && !c->dead) { c->dead = 1; c->deaths++; }
            break;
        }
        RoUnit* u = ro_unit_find(c, p->gid);
        if (u && p->type == 1 && u->type == RO_OBJ_MOB && u->hit_by_me) c->kills++;
        ro_unit_remove(c, p->gid);
        break;
    }
    case RO_HEADER_ZC_NOTIFY_PLAYERMOVE: {  // our own walk
        const RO_PACKET_ZC_NOTIFY_PLAYERMOVE* p = (const RO_PACKET_ZC_NOTIFY_PLAYERMOVE*)(buf);
        ro_get_pos2(p->moveData, &c->x0, &c->y0, &c->x1, &c->y1);
        c->t0 = c->lockstep ? (double)p->moveStartTime : c->now;
        break;
    }
    case RO_HEADER_ZC_STOPMOVE: {
        const RO_PACKET_ZC_STOPMOVE* p = (const RO_PACKET_ZC_STOPMOVE*)(buf);
        if (p->AID == c->aid) {
            c->x0 = c->x1 = (int16_t)p->xPos; c->y0 = c->y1 = (int16_t)p->yPos; c->t0 = c->now;
        } else {
            RoUnit* u = ro_unit_find(c, p->AID);
            if (u) { u->x0 = u->x1 = (int16_t)p->xPos; u->y0 = u->y1 = (int16_t)p->yPos; u->t0 = c->now; }
        }
        break;
    }
    case RO_TYPE_spawn_unitType: case RO_TYPE_unit_walkingType: case RO_TYPE_idle_unitType:
        ro_unit_from_packet(c, buf, cmd);
        break;
    case RO_HEADER_ZC_NOTIFY_ACT: {
        const RO_PACKET_ZC_NOTIFY_ACT* p = (const RO_PACKET_ZC_NOTIFY_ACT*)(buf);
        int hit = p->type == RO_DMG_NORMAL || p->type == RO_DMG_MULTI_HIT || p->type == RO_DMG_CRITICAL
                  || p->type == RO_DMG_ENDURE || p->type == RO_DMG_LUCKY_DODGE;
        if ((uint32_t)p->srcID == c->aid) {
            if (p->type == RO_DMG_SIT_DOWN) c->sitting = 1;
            else if (p->type == RO_DMG_STAND_UP) c->sitting = 0;
            else if (hit) {
                RoUnit* u = ro_unit_find(c, (uint32_t)p->targetID);
                if (u) u->hit_by_me = 1;
                if (p->damage > 0) c->damage_dealt += p->damage + (p->damage2 > 0 ? p->damage2 : 0);
            }
        } else if ((uint32_t)p->targetID == c->aid && hit) {
            RoUnit* u = ro_unit_find(c, (uint32_t)p->srcID);
            if (u) u->last_hit_me = c->now;
            if (p->damage > 0) c->damage_taken += p->damage + (p->damage2 > 0 ? p->damage2 : 0);
        }
        break;
    }
    case RO_HEADER_ZC_SKILLINFO_LIST: {  // every skill the character has
        c->nskills = 0;
        for (int off = (int)sizeof(RO_PACKET_ZC_SKILLINFO_LIST); off + (int)sizeof(RO_SKILLDATA) <= len
             && c->nskills < RO_MAX_SKILLS; off += (int)sizeof(RO_SKILLDATA)) {
            const RO_SKILLDATA* d = (const RO_SKILLDATA*)(buf + off);
            RoSkill* k = &c->skills[c->nskills++];
            k->id = d->id; k->level = d->level; k->sp = d->sp; k->range = d->range2; k->inf = d->inf;
            k->ready_at = 0;
        }
        break;
    }
    case RO_HEADER_ZC_SKILLINFO_UPDATE: {  // one skill changed
        const RO_PACKET_ZC_SKILLINFO_UPDATE* p = (const RO_PACKET_ZC_SKILLINFO_UPDATE*)(buf);
        for (int i = 0; i < c->nskills; i++)
            if (c->skills[i].id == p->skillId) {
                c->skills[i].level = p->level; c->skills[i].sp = p->sp; c->skills[i].range = p->range2;
            }
        break;
    }
    case RO_HEADER_ZC_USESKILL_ACK: {  // someone started casting
        const RO_PACKET_ZC_USESKILL_ACK* p = (const RO_PACKET_ZC_USESKILL_ACK*)(buf);
        if (p->srcId == c->aid) {
            c->cast_until = c->now + p->delayTime;
            c->cast_skill = p->skillId;
        }
        break;
    }
    case RO_HEADER_ZC_ACK_TOUSESKILL: {  // a skill request was refused
        const RO_PACKET_ZC_ACK_TOUSESKILL* p = (const RO_PACKET_ZC_ACK_TOUSESKILL*)(buf);
        if (!p->flag) {
            c->skill_failed_at = c->now;
            c->skill_failed_id = p->skillId;
            c->skill_failed_cause = p->cause;
            c->skills_failed++;
            c->cast_until = 0;
        }
        break;
    }
    case RO_HEADER_ZC_SKILL_POSTDELAY: {  // a skill's cooldown
        const RO_PACKET_ZC_SKILL_POSTDELAY* p = (const RO_PACKET_ZC_SKILL_POSTDELAY*)(buf);
        for (int i = 0; i < c->nskills; i++)
            if (c->skills[i].id == p->SKID) c->skills[i].ready_at = c->now + p->DelayTM;
        break;
    }
    case RO_HEADER_ZC_NOTIFY_SKILL: {  // skill damage
        const RO_PACKET_ZC_NOTIFY_SKILL* p = (const RO_PACKET_ZC_NOTIFY_SKILL*)(buf);
        if (p->AID == c->aid) {
            RoUnit* u = ro_unit_find(c, p->targetID);
            if (u) u->hit_by_me = 1;
            if (p->damage > 0) c->damage_dealt += p->damage;
            if (c->cast_skill == p->SKID) c->cast_until = 0;
        } else if (p->targetID == c->aid) {
            RoUnit* u = ro_unit_find(c, p->AID);
            if (u) u->last_hit_me = c->now;
            if (p->damage > 0) c->damage_taken += p->damage;
        }
        break;
    }
    case RO_HEADER_ZC_USE_SKILL: {  // a support skill took effect
        const RO_PACKET_ZC_USE_SKILL* p = (const RO_PACKET_ZC_USE_SKILL*)(buf);
        if (p->srcAID == c->aid && c->cast_skill == p->SKID) c->cast_until = 0;
        break;
    }
    case RO_TYPE_sendLookType: {  // ZC_SPRITE_CHANGE2: look type 0 is the job (a job change)
        const RO_PACKET_ZC_SPRITE_CHANGE* p = (const RO_PACKET_ZC_SPRITE_CHANGE*)(buf);
        if (p->AID == c->aid && p->type == 0) c->job = p->val;
        break;
    }
    case RO_HEADER_ZC_ATTACK_FAILURE_FOR_DISTANCE: {  // <target>.L <tx>.W <ty>.W <x>.W <y>.W <range>.W
        memcpy(&c->too_far_id, buf + 2, 4);
        c->too_far_at = c->now;
        break;
    }
    case RO_TYPE_monsterhpType: {  // ZC_HP_INFO: <id>.L <hp>.L <maxhp>.L
        uint32_t id = buf[2] | buf[3] << 8 | buf[4] << 16 | (uint32_t)buf[5] << 24;
        RoUnit* u = ro_unit_find(c, id);
        if (u) {
            memcpy(&u->hp, buf + 6, 4);
            memcpy(&u->maxhp, buf + 10, 4);
        }
        break;
    }
    case RO_TYPE_dropflooritemType: {  // ZC_ITEM_FALL_ENTRY: an item dropped
        const RO_packet_dropflooritem* p = (const RO_packet_dropflooritem*)(buf);
        ro_ground_add(c, p->ITAID, p->ITID, p->xPos, p->yPos, p->count);
        break;
    }
    case RO_HEADER_ZC_ITEM_ENTRY: {  // an item already on the ground came into view
        const RO_PACKET_ZC_ITEM_ENTRY* p = (const RO_PACKET_ZC_ITEM_ENTRY*)(buf);
        ro_ground_add(c, p->AID, p->itemId, p->x, p->y, p->amount);
        break;
    }
    case RO_HEADER_ZC_ITEM_DISAPPEAR: ro_ground_remove(c, ((const RO_PACKET_ZC_ITEM_DISAPPEAR*)buf)->itemAid); break;
    case RO_HEADER_ZC_ITEM_PICKUP_ACK: {
        const RO_PACKET_ZC_ITEM_PICKUP_ACK* p = (const RO_PACKET_ZC_ITEM_PICKUP_ACK*)(buf);
        if (p->result == 0) {
            RoInv* v = ro_inv_slot(c, p->Index);
            if (v) {
                v->item = p->nameid; v->amount += p->count; v->type = p->type; v->location = p->location;
            }
            c->picked++;
        }
        break;
    }
    case RO_TYPE_inventorylistnormalType: case RO_TYPE_inventorylistequipType: {  // inventory lists
        int normal = cmd == RO_TYPE_inventorylistnormalType;
        int head = normal ? RO_ITEMLIST_NORMAL_HEAD : RO_ITEMLIST_EQUIP_HEAD;
        int size = normal ? (int)sizeof(RO_NORMALITEM_INFO) : (int)sizeof(RO_EQUIPITEM_INFO);
        // From 2018 on the cart and storage lists share these ids; byte 4 says whose list it is.
        if (head > 4 && buf[4] != 0) break;
        if ((len - head) % size != 0) {
            ro_fail(c, "inventory list 0x%04x of %d bytes does not divide into %d-byte items", cmd, len, size);
            break;
        }
        for (int off = head; off + size <= len; off += size) {
            if (normal) {
                const RO_NORMALITEM_INFO* it = (const RO_NORMALITEM_INFO*)(buf + off);
                RoInv* v = ro_inv_slot(c, it->index);
                if (v) { v->item = it->ITID; v->amount = it->count; v->type = it->type; v->worn = 0; }
            } else {
                const RO_EQUIPITEM_INFO* it = (const RO_EQUIPITEM_INFO*)(buf + off);
                RoInv* v = ro_inv_slot(c, it->index);
                if (v) {
                    v->item = it->ITID; v->amount = 1; v->type = it->type;
                    v->location = it->location; v->worn = it->WearState;
                }
            }
        }
        break;
    }
    case RO_TYPE_useItemAckType: {  // ZC_USE_ITEM_ACK: amount is what is left
        const RO_PACKET_ZC_USE_ITEM_ACK* p = (const RO_PACKET_ZC_USE_ITEM_ACK*)(buf);
        if (p->AID == c->aid && p->result) {
            RoInv* v = ro_inv_slot(c, p->index);
            if (v) {
                v->amount = p->amount;
                if (v->amount <= 0) ro_inv_take(c, p->index, 0);
            }
            c->potions++;
        }
        break;
    }
    case RO_HEADER_ZC_ITEM_THROW_ACK:
        ro_inv_take(c, ((const RO_PACKET_ZC_ITEM_THROW_ACK*)buf)->index, ((const RO_PACKET_ZC_ITEM_THROW_ACK*)buf)->count);
        break;
    case RO_HEADER_ZC_DELETE_ITEM_FROM_BODY:
        ro_inv_take(c, ((const RO_PACKET_ZC_DELETE_ITEM_FROM_BODY*)buf)->index,
                    ((const RO_PACKET_ZC_DELETE_ITEM_FROM_BODY*)buf)->count);
        break;
    case RO_HEADER_ZC_PAR_CHANGE: case RO_HEADER_ZC_LONGPAR_CHANGE: {
        const RO_PACKET_ZC_PAR_CHANGE* p = (const RO_PACKET_ZC_PAR_CHANGE*)(buf);
        ro_set_status(c, p->varID, p->count);
        break;
    }
#ifdef RO_HAVE_PACKET_ZC_LONGLONGPAR_CHANGE
    case RO_HEADER_ZC_LONGLONGPAR_CHANGE: {  // exp and the exp to the next level, from 2017-08 on
        const RO_PACKET_ZC_LONGLONGPAR_CHANGE* p = (const RO_PACKET_ZC_LONGLONGPAR_CHANGE*)(buf);
        ro_set_status(c, p->varID, p->amount);
        break;
    }
#endif
    case RO_HEADER_ZC_STATUS: {
        const RO_PACKET_ZC_STATUS* p = (const RO_PACKET_ZC_STATUS*)(buf);
        c->status_points = p->point;
        c->str = p->str; c->agi = p->agi; c->vit = p->vit; c->int_ = p->int_; c->dex = p->dex; c->luk = p->luk;
        break;
    }
    case RO_HEADER_ZC_STATUS_CHANGE_ACK:  // <type>.W <ok>.B <value>.B after a stat raise
        if (buf[4]) ro_set_status(c, ro_u16(buf + 2), buf[5]);
        break;
    case RO_HEADER_ZC_COUPLESTATUS: {  // <type>.L <base>.L <bonus>.L
        uint32_t type, base;
        memcpy(&type, buf + 2, 4);
        memcpy(&base, buf + 6, 4);
        ro_set_status(c, (int)type, base);
        break;
    }
    case RO_HEADER_ZC_ATTACK_RANGE: c->attack_range = ((const RO_PACKET_ZC_ATTACK_RANGE*)buf)->currentAttRange; break;
    // Status icons, the bot's own: <efst>.W <id>.L <on>.B, then the times as the client version has them.
#ifdef RO_TYPE_status_changeType
    case RO_TYPE_status_changeType:  // ZC_MSG_STATE_CHANGE3: <total>.L <left>.L <val>.3L
        if (ro_u32(buf + 4) == c->aid) ro_effect_set(c, ro_u16(buf + 2), buf[8], ro_u32(buf + 9), ro_u32(buf + 13));
        break;
#endif
#ifdef RO_TYPE_status_change2Type
    case RO_TYPE_status_change2Type:  // ZC_MSG_STATE_CHANGE2: <left>.L <val>.3L
        if (ro_u32(buf + 4) == c->aid) ro_effect_set(c, ro_u16(buf + 2), buf[8], 0, ro_u32(buf + 9));
        break;
#endif
#ifdef RO_TYPE_status_change_endType
    case RO_TYPE_status_change_endType:  // ZC_MSG_STATE_CHANGE: no times (an end, or timers off)
        if (ro_u32(buf + 4) == c->aid) ro_effect_set(c, ro_u16(buf + 2), buf[8], 0, 0);
        break;
#endif
#ifdef RO_HEADER_ZC_STATE_CHANGE
    case RO_HEADER_ZC_STATE_CHANGE: {  // <id>.L <body>.W <health>.W <option>.L <pk>.B, for anyone in sight
        uint32_t id = ro_u32(buf + 2);
        int16_t opt1 = (int16_t)ro_u16(buf + 6), opt2 = (int16_t)ro_u16(buf + 8);
        int32_t option = (int32_t)ro_u32(buf + 10);
        RoUnit* u = id == c->aid ? NULL : ro_unit_find(c, id);
        if (id == c->aid) {
            c->opt1 = opt1;
            c->opt2 = opt2;
            c->option = option;
        } else if (u) {
            u->opt1 = opt1;
            u->opt2 = opt2;
            u->option = option;
        }
        break;
    }
#endif
    case RO_ZC_LOCKSTEP_DONE: {  // <step>.L <tick>.L: the step has been sent in full
        uint32_t tick;
        memcpy(&c->steps_done, buf + 2, 4);
        memcpy(&tick, buf + 6, 4);
        c->now = tick;
        c->step_pending = 0;
        c->lockstep_joined = 1;
        break;
    }
    case 0x8e: case 0x8d: case 0x9a: case 0x2c1:  // chat, broadcasts, NPC talk (the same ids in every version)
        if (c->verbose) {
            int off = cmd == 0x8d || cmd == 0x2c1 ? 8 : 4;
            if (cmd == 0x2c1) off = 12;
            if (len > off) fprintf(stderr, "[%s] chat 0x%04x: %.*s\n", c->user, cmd, len - off, (const char*)buf + off);
        }
        break;
    default: break;
    }
}

// Read whatever the map server sent and apply it. Returns -1 once the client is in RO_ERROR.
RO_API int ro_poll(RoClient* c) {
    if (!c->lockstep) c->now = ro_ms();
    if (c->state == RO_ERROR || c->fd < 0) return -1;
    for (;;) {
        if (c->rlen == RO_RBUF) return ro_fail(c, "receive buffer full");
        ssize_t n = recv(c->fd, c->rbuf + c->rlen, RO_RBUF - c->rlen, 0);
        if (n > 0) {
            c->rlen += (int)n;
            continue;
        }
        if (n == 0) return ro_fail(c, "map server closed the connection");
        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
        if (errno == EINTR) continue;
        return ro_fail(c, "recv: %s", strerror(errno));
    }
    int off = 0;
    while (c->rlen - off >= 2) {
        const uint8_t* p = c->rbuf + off;
        int cmd = ro_u16(p), len = ro_packet_len(cmd);
        if (len == 0) {
            char hex[3 * 24 + 1] = {0};
            for (int i = 0; i < 24 && off + i < c->rlen; i++) sprintf(hex + 3 * i, "%02x ", p[i]);
            c->unknown_packets++;
            return ro_fail(c, "unknown packet 0x%04x from the map server: %s", cmd, hex);
        }
        if (len < 0) {
            if (c->rlen - off < 4) break;
            len = ro_u16(p + 2);
            if (len < 4) return ro_fail(c, "bad length %d for packet 0x%04x", len, cmd);
        }
        if (c->rlen - off < len) break;
        ro_handle(c, p, len);
        if (c->state == RO_ERROR) return -1;
        off += len;
    }
    memmove(c->rbuf, c->rbuf + off, c->rlen - off);
    c->rlen -= off;
    if (c->state == RO_IN_GAME && c->now - c->last_ticksend > 10000) ro_ticksend(c);
    return 0;
}

// ---- lockstep over many bots ----------------------------------------------------------------

#define RO_MAX_STEP_BOTS 1024

// Wait until every bot in cs[] (all on one world, all joined) has the DONE of step `target`,
// sending READY for each bot below it that is not already waiting. -1 on timeout.
RO_API int ro_lockstep_reach(RoClient** cs, int n, uint32_t target, double timeout_ms) {
    struct pollfd fds[RO_MAX_STEP_BOTS];
    double t0 = ro_ms();
    for (;;) {
        int k = 0, behind = 0;
        for (int i = 0; i < n && i < RO_MAX_STEP_BOTS; i++) {
            RoClient* c = cs[i];
            if (c->state != RO_IN_GAME) continue;
            if (c->step_pending) ro_poll(c);
            if (c->state != RO_IN_GAME) continue;
            if (!c->step_pending && c->steps_done < target) ro_lockstep_ready(c);
            if (c->steps_done < target) {
                behind++;
                fds[k].fd = c->fd;
                fds[k].events = POLLIN;
                k++;
            }
        }
        if (!behind) return 0;
        if (ro_ms() - t0 > timeout_ms) return -1;
        poll(fds, k, 5);
    }
}

// One step for every bot: the step after the newest one any bot has seen.
RO_API int ro_lockstep_step_all(RoClient** cs, int n, double timeout_ms) {
    uint32_t newest = 0;
    for (int i = 0; i < n; i++)
        if (cs[i]->state == RO_IN_GAME && cs[i]->steps_done > newest) newest = cs[i]->steps_done;
    return ro_lockstep_reach(cs, n, newest + 1, timeout_ms);
}

// Join the world's step barrier and end with every bot at the same step. Until every bot has had
// a DONE, a bot whose DONE arrives sends READY again at once, so steps keep going for the bots
// already in while the others' first READYs arrive; then one more step for all of them.
RO_API int ro_lockstep_join(RoClient** cs, int n, double timeout_ms) {
    struct pollfd fds[RO_MAX_STEP_BOTS];
    double t0 = ro_ms();
    for (int i = 0; i < n; i++) {
        cs[i]->lockstep = 1;
        if (cs[i]->state == RO_IN_GAME && !cs[i]->step_pending) ro_lockstep_ready(cs[i]);
    }
    for (;;) {
        int k = 0, waiting = 0;
        for (int i = 0; i < n && i < RO_MAX_STEP_BOTS; i++) {
            RoClient* c = cs[i];
            if (c->state != RO_IN_GAME) continue;
            if (c->step_pending) ro_poll(c);
            if (c->state != RO_IN_GAME) continue;
            if (!c->lockstep_joined) waiting++;
            if (!c->step_pending) ro_lockstep_ready(c);
            fds[k].fd = c->fd;
            fds[k].events = POLLIN;
            k++;
        }
        if (!waiting) break;
        if (ro_ms() - t0 > timeout_ms) return -1;
        poll(fds, k, 5);
    }
    return ro_lockstep_step_all(cs, n, timeout_ms);
}
