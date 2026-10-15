// Exact typed labels from the existing engine. Copying its struct permits cheap
// counterfactual steps without changing the engine or serializing snapshots.
#include "../ocean/decision_snake/engine/snake.c"
#include <assert.h>
#include <stdio.h>

#define HORIZON 4
#define EPISODE_STEPS 64

const char* directions[] = {"up", "down", "left", "right"};

int food_distance(IBSnake env, int remaining) {
    int best = HORIZON + 1;
    if (!remaining) {
        return best;
    }
    uint32_t legal = ib_snake_legal_actions(&env);
    for (int action = 0; action < 4; action++) {
        if (!(legal & (1u << action))) {
            continue;
        }
        IBSnake next = env;
        ib_snake_step(&next, action);
        if (next.score > env.score) {
            return 1;
        }
        if (!next.terminated && !next.truncated) {
            int distance = 1 + food_distance(next, remaining - 1);
            if (distance < best) {
                best = distance;
            }
        }
    }
    return best;
}

uint32_t sample(uint32_t* rng) {
    *rng = *rng * 1664525u + 1013904223u;
    return *rng;
}

void write_row(FILE* out, IBSnake* env, int state_id, uint32_t seed, int* history,
    int type, int* offered, int count, int action, int target, int32_t* board) {
    fprintf(out, "{\"state\":\"10x10 Snake. Coordinates row,column from top left, starting at 0. "
        "Head=(%d,%d). Body neck-to-tail=", env->snake[0] / 10, env->snake[0] % 10);
    for (int i = 1; i < env->length; i++) {
        fprintf(out, "%s(%d,%d)", i == 1 ? "" : ",", env->snake[i] / 10, env->snake[i] % 10);
    }
    fprintf(out, ". Food=(%d,%d). Neighbor cells:", env->food / 10, env->food % 10);
    for (int move = 0; move < 4; move++) {
        int cell = next_position(env, move);
        const char* value = cell < 0 ? "wall" : board[cell] == -1 ? "food" :
            board[cell] == 0 ? "empty" : board[cell] == env->length ? "tail" :
            board[cell] == 2 ? "neck" : "body";
        fprintf(out, " %s=%s;", directions[move], value);
    }
    fputs("\",\"questions\":{\"q\":{", out);
    if (type == 0) {
        fputs("\"type\":\"choice\",\"instructions\":\"Which offered first move reaches the "
            "current food in the fewest total moves? Continue with any legal moves, "
            "for at most 4 total moves.\",\"criteria\":{", out);
        for (int i = 0; i < count; i++) {
            fprintf(out, "%s\"%s\":\"move %s\"", i ? "," : "", directions[offered[i]],
                directions[offered[i]]);
        }
        fputc('}', out);
    } else if (type == 1) {
        fprintf(out, "\"type\":\"noul\",\"instructions\":\"Would attempting %s now collide "
            "with a wall or the snake? A tail cell that departs on this move is safe.\"",
            directions[action]);
    } else {
        fprintf(out, "\"type\":\"score\",\"instructions\":\"After first moving %s, how many "
            "moves until the current food can first be eaten? Count the first move and use "
            "the shortest legal continuation within 4 total moves.\",\"criteria\":["
            "\"1 move\",\"2 moves\",\"3-4 moves\",\"not within 4 moves\"]",
            directions[action]);
    }
    fprintf(out, "}},\"target\":%d,\"state_id\":%d,\"board\":[", target, state_id);
    for (int i = 0; i < 100; i++) {
        fprintf(out, "%s%d", i ? "," : "", board[i]);
    }
    fprintf(out, "],\"reset_seed\":%u,\"history\":[", seed);
    for (int i = 0; i < env->steps; i++) {
        fprintf(out, "%s%d", i ? "," : "", history[i]);
    }
    fputs("]}\n", out);
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 6) {
        fprintf(stderr, "Usage: %s TRAIN.jsonl HELDOUT.jsonl [TRAIN_STATES=512] "
            "[HELDOUT_STATES=128] [SEED=73]\n", argv[0]);
        return 2;
    }
    int train = argc > 3 ? atoi(argv[3]) : 512;
    int heldout = argc > 4 ? atoi(argv[4]) : 128;
    uint32_t rng = argc > 5 ? strtoul(argv[5], NULL, 10) : 73;
    assert(train > 0 && heldout > 0);
    FILE* outputs[] = {fopen(argv[1], "wx"), fopen(argv[2], "wx")};
    if (!outputs[0] || !outputs[1]) {
        perror("create dataset");
        return 1;
    }
    int total = train + heldout;
    int32_t (*boards)[100] = calloc(total, sizeof(*boards));
    int counts[2][3][4] = {0};
    int history[EPISODE_STEPS];
    uint32_t seed = sample(&rng);
    IBSnake env;
    ib_snake_reset(&env, seed, EPISODE_STEPS);
    for (int written = 0, attempts = 0; written < total; attempts++) {
        assert(attempts < 10000000);
        if (env.terminated || env.truncated || env.steps > EPISODE_STEPS - HORIZON) {
            seed = sample(&rng);
            ib_snake_reset(&env, seed, EPISODE_STEPS);
        }
        int32_t board[100];
        ib_snake_observe(&env, board, 100);
        int legal[3], nlegal = 0, distances[4], collisions[4] = {0};
        uint32_t mask = ib_snake_legal_actions(&env);
        for (int action = 0; action < 4; action++) {
            distances[action] = HORIZON + 1;
            if (!(mask & (1u << action))) {
                continue;
            }
            legal[nlegal++] = action;
            IBSnake next = env;
            ib_snake_step(&next, action);
            collisions[action] = next.outcome == -1;
            if (next.score > env.score) {
                distances[action] = 1;
            } else if (!next.terminated && !next.truncated) {
                int distance = 1 + food_distance(next, HORIZON - 1);
                if (distance <= HORIZON) {
                    distances[action] = distance;
                }
            }
        }
        int split = written >= train;
        int index = split ? written - train : written;
        int score_target = index % 4, noul_target = (index / 4) % 2;
        int score_action = -1, noul_action = -1, best = legal[0];
        int score_seen = 0, noul_seen = 0;
        for (int i = 0; i < nlegal; i++) {
            int action = legal[i], distance = distances[action];
            int bucket = distance <= 2 ? distance - 1 : distance <= HORIZON ? 2 : 3;
            if (bucket == score_target && sample(&rng) % ++score_seen == 0) {
                score_action = action;
            }
            if (collisions[action] == noul_target && sample(&rng) % ++noul_seen == 0) {
                noul_action = action;
            }
            if (distance < distances[best]) {
                best = action;
            }
        }
        int count = 2 + index % 2, offered[3] = {best}, available = 1;
        for (int i = 0; i < nlegal; i++) {
            if (distances[legal[i]] > distances[best]) {
                offered[available++] = legal[i];
            }
        }
        int duplicate = 0;
        for (int i = 0; i < written; i++) {
            if (!memcmp(board, boards[i], sizeof(board))) {
                duplicate = 1;
                break;
            }
        }
        if (!duplicate && score_action >= 0 && noul_action >= 0 &&
            distances[best] <= HORIZON && available >= count) {
            memcpy(boards[written], board, sizeof(board));
            if (available == 3 && sample(&rng) % 2) {
                int swap = offered[1];
                offered[1] = offered[2];
                offered[2] = swap;
            }
            int target = (index / 2) % count;
            int swap = offered[target];
            offered[target] = best;
            offered[0] = swap;
            write_row(outputs[split], &env, written, seed, history, 0,
                offered, count, 0, target, board);
            write_row(outputs[split], &env, written, seed, history, 1,
                NULL, 0, noul_action, noul_target, board);
            write_row(outputs[split], &env, written, seed, history, 2,
                NULL, 0, score_action, score_target, board);
            counts[split][0][target]++;
            counts[split][1][noul_target]++;
            counts[split][2][score_target]++;
            written++;
        }
        int action = best;
        if (distances[best] > HORIZON || sample(&rng) % 4 == 0) {
            int safe[3], nsafe = 0;
            for (int i = 0; i < nlegal; i++) {
                if (!collisions[legal[i]]) {
                    safe[nsafe++] = legal[i];
                }
            }
            action = nsafe ? safe[sample(&rng) % nsafe] : legal[sample(&rng) % nlegal];
        }
        history[env.steps] = action;
        ib_snake_step(&env, action);
    }
    for (int split = 0; split < 2; split++) {
        if (ferror(outputs[split]) || fclose(outputs[split])) {
            perror("write dataset");
            return 1;
        }
        printf("%s: %d distinct states, %d rows; choice targets [%d,%d,%d]; "
            "noul [%d,%d]; score [%d,%d,%d,%d]\n", split ? "heldout" : "train",
            split ? heldout : train, 3 * (split ? heldout : train),
            counts[split][0][0], counts[split][0][1], counts[split][0][2],
            counts[split][1][0], counts[split][1][1], counts[split][2][0],
            counts[split][2][1], counts[split][2][2], counts[split][2][3]);
    }
    free(boards);
    return 0;
}
