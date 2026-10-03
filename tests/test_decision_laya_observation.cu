// CPU-only environment/packing validation. Compile with NVCC for the included
// model declarations; execution does not create a CUDA context or use a GPU.
#include "../src/ini.h"
#define PUF_HEADLESS
#include "../ocean/decision_laya/decision_laya.h"
#include <cassert>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>

static int validate(const unsigned char* observations) {
    const auto& bundle = decision_policy_context->bundle;
    int length = observations[0] | (observations[1] << 8);
    assert(length > 0 && length <= bundle.max_len);
    assert(observations[DECISION_POLICY_QTYPE_OFFSET] == 0);
    for (int k = 0; k < 4; ++k) {
        int marker = observations[2 + 2 * k] | (observations[3 + 2 * k] << 8);
        assert(marker < length);
        uint32_t token = 0;
        for (int byte = 0; byte < 4; ++byte)
            token |= uint32_t(observations[DECISION_POLICY_HEADER_BYTES + 4 * marker + byte]) << (8 * byte);
        assert(token == bundle.mask_id);
    }
    return length;
}

static std::string decode(const unsigned char* observations) {
    const int length = validate(observations);
    std::vector<uint32_t> ids(length);
    for (int i = 0; i < length; ++i)
        for (int byte = 0; byte < 4; ++byte)
            ids[i] |= uint32_t(observations[DECISION_POLICY_HEADER_BYTES + 4 * i + byte]) << (8 * byte);
    return decision_policy_context->tokenizer.Decode(ids, false);
}

static void check_spatial_format() {
    // Captured original seed-670001 state, before adding the optional format.
    const std::string original =
        "Snake on a 10 by 10 grid. Rows run top to bottom; columns left to right. "
        "0=empty, -1=food, 1=head, 2=neck, larger numbers follow the body toward the tail. "
        "Step 0 of 500. Board:\n"
        "0,0,0,0,0,0,0,0,0,0\n"
        "0,0,0,0,0,0,0,0,0,0\n"
        "0,0,0,0,0,0,0,0,0,0\n"
        "0,0,0,0,0,0,0,0,0,0\n"
        "0,0,0,0,0,0,0,0,0,0\n"
        "0,0,-1,3,2,1,0,0,0,0\n"
        "0,0,0,0,0,0,0,0,0,0\n"
        "0,0,0,0,0,0,0,0,0,0\n"
        "0,0,0,0,0,0,0,0,0,0\n"
        "0,0,0,0,0,0,0,0,0,0\n";
    IBSnake* snake = ib_snake_create(670001, 500);
    assert(snake);
    int32_t board[100];
    assert(ib_snake_observe(snake, board, 100) == 0);
    unsigned char default_observation[OBS_SIZE], legacy[OBS_SIZE], spatial[OBS_SIZE], expected[OBS_SIZE];
    decision_policy_encode(original,
        "Choose the next move. Eat food and avoid the walls and snake body.",
        {"Move up", "Move down", "Move left", "Move right"}, expected);
    decision_laya_encode(board, 0, 500, default_observation);
    decision_laya_encode(board, 0, 500, legacy, 0);
    assert(memcmp(default_observation, expected, OBS_SIZE) == 0);
    assert(memcmp(legacy, expected, OBS_SIZE) == 0);
    decision_laya_encode(board, 0, 500, spatial, 1);
    const std::string rendered = decode(spatial);
    assert(rendered.find("Coordinates are zero-based (row, column). Head: (5, 5). "
        "Food: (5, 2). Food minus head: row +0, column -3. ") != std::string::npos);
    assert(rendered.find("Negative row=up; positive row=down; negative column=left; positive column=right.\n") != std::string::npos);
    assert(rendered.find(original) != std::string::npos);
    // The question/options and their marker positions are unchanged.
    const std::string decoded_legacy = decode(legacy);
    const size_t state_start = decoded_legacy.find("Snake on a 10 by 10 grid.");
    assert(rendered.compare(0, state_start, decoded_legacy, 0, state_start) == 0);
    assert(memcmp(legacy + 2, spatial + 2, DECISION_POLICY_HEADER_BYTES - 2) == 0);
    std::cout << "initial legacy_tokens=" << validate(legacy)
              << " coordinate_tokens=" << validate(spatial) << '\n';
    ib_snake_destroy(snake);

    const int heads[] = {0, 9, 90, 99};
    const char* offsets[] = {
        "Food minus head: row +9, column +9.",
        "Food minus head: row +9, column -9.",
        "Food minus head: row -9, column +9.",
        "Food minus head: row -9, column -9.",
    };
    for (int k = 0; k < 4; ++k) {
        memset(board, 0, sizeof(board));
        const int head = heads[k], toward_center = head % 10 == 0 ? 1 : -1;
        board[head] = 1; board[head + toward_center] = 2;
        board[head + 2 * toward_center] = 3; board[99 - head] = -1;
        decision_laya_encode(board, 499, 500, spatial, 1);
        assert(decode(spatial).find(offsets[k]) != std::string::npos);
    }

    // Geometrically valid ordered bodies on a Hamiltonian cycle cover every
    // head cell and every supported body length. Two empty-cell choices cover
    // opposing food offsets; INT_MAX exercises the longest supported step text.
    int path[100], used = 0;
    for (int row = 0; row < 10; ++row) path[used++] = 10 * row;
    for (int row = 9; row >= 0; --row)
        for (int k = 0; k < 9; ++k)
            path[used++] = row * 10 + ((row % 2) ? k + 1 : 9 - k);
    assert(used == 100);
    for (int i = 0; i < 100; ++i) {
        const int a = path[i], b = path[(i + 1) % 100];
        assert(abs(a / 10 - b / 10) + abs(a % 10 - b % 10) == 1);
    }
    int maximum = 0, cases = 0;
    for (int length = 3; length <= 100; ++length) {
        for (int start = 0; start < 100; ++start) {
            for (int direction : {-1, 1}) {
                memset(board, 0, sizeof(board));
                for (int i = 0; i < length; ++i) board[path[(start + i) % 100]] = i + 1;
                for (int k = 0; k < 100; ++k) {
                    const int cell = direction > 0 ? k : 99 - k;
                    if (board[cell] == 0) { board[cell] = -1; break; }
                }
                decision_laya_encode(board, INT_MAX, INT_MAX, spatial, 1);
                maximum = std::max(maximum, validate(spatial));
                ++cases;
                if (length == 100) {
                    const std::string full = decode(spatial);
                    assert(full.find("Food: none. ") != std::string::npos);
                    assert(full.find("Food minus head:") == std::string::npos);
                }
            }
        }
    }
    bool rejected = false;
    try { decision_laya_encode(board, 0, 500, spatial, 2); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    // The serializer rejects over-budget state instead of silently dropping cells.
    const int original_limit = decision_policy_context->bundle.max_len;
    decision_policy_context->bundle.max_len = 192;
    rejected = false;
    try { decision_laya_encode(board, 0, 500, spatial, 1); }
    catch (const std::invalid_argument&) { rejected = true; }
    decision_policy_context->bundle.max_len = original_limit;
    assert(rejected);
    std::cout << "legacy bytes preserved; coordinate format cases=" << cases
              << " maximum_tokens=" << maximum << '\n';
}

static void check_format_validation() {
    const char* invalid[] = {"-1", "0.5", "2", "inf", "nan", "bad", "1,0", "", "1oops"};
    for (const char* value : invalid) {
        pid_t child = fork();
        assert(child >= 0);
        if (child == 0) {
            assert(freopen("/dev/null", "w", stderr));
            Dict kwargs{};
            dict_set(&kwargs, "max_steps", 500);
            puf_ini_set(&kwargs, "observation_format", value);
            Env env{}; puf_init(&env, &kwargs);
            _exit(0);
        }
        int status = 0;
        assert(waitpid(child, &status, 0) == child);
        assert(WIFEXITED(status) && WEXITSTATUS(status) == 1);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: test_decision_laya_observation BUNDLE\n"; return 2; }
    decision_policy_context.reset(new DecisionPolicyContext(argv[1]));
    unsigned char observations[OBS_SIZE], expected[OBS_SIZE], masks[4];
    int32_t board[100]{};
    for (int fixture = 0; fixture < 3; ++fixture) {
        for (int i = 0; i < 100; ++i) board[i] = fixture == 0 ? 0 : fixture == 1 ? i + 1 : -(i % 2);
        if (fixture == 0) { board[45] = 1; board[46] = 2; board[47] = 3; board[72] = -1; }
        decision_laya_encode(board, 499, 500, observations);
        std::cout << "fixture=" << fixture << " tokens=" << validate(observations) << '\n';
    }
    check_spatial_format();
    std::cout.flush();
    check_format_validation();

    for (int format = 0; format <= 1; ++format) {
        Dict kwargs{};
        dict_set(&kwargs, "max_steps", 1);
        dict_set(&kwargs, "num_agents", 1);
        // Omission must retain the exact legacy default.
        if (format) puf_ini_set(&kwargs, "observation_format", "1");
        Env env{}; env.rng = 73;
        puf_init(&env, &kwargs);
        assert(env.observation_format == format);
        float action = 0, reward = 0, terminal = 0;
        env.agents[0].observations = observations;
        env.agents[0].actions = &action;
        env.agents[0].rewards = &reward;
        env.agents[0].terminals = &terminal;
        env.agents[0].action_mask = masks;
        puf_reset(&env); validate(observations);
        ib_snake_observe(env.snake, board, 100);
        int head = 0;
        while (board[head] != 1) ++head;
        const int dr[4] = {-1, 1, 0, 0}, dc[4] = {0, 0, -1, 1};
        int selected = -1;
        for (int k = 0; k < 4; ++k) {
            int r = head / 10 + dr[k], c = head % 10 + dc[k];
            if (masks[k] && r >= 0 && r < 10 && c >= 0 && c < 10 && board[r * 10 + c] <= 0) { selected = k; break; }
        }
        assert(selected >= 0);
        IBSnake* reference = ib_snake_create(73, 1);
        assert(reference && ib_snake_step(reference, selected) == 0);
        ib_snake_observe(reference, board, 100);
        decision_laya_encode(board, 1, 1, expected, format);
        action = selected;
        puf_step(&env);
        assert(terminal == 1 && env.transition.truncated && !env.transition.terminated);
        assert(memcmp(expected, env.transition.observations, OBS_SIZE) == 0);
        assert(ib_snake_reset(reference, 74, 1) == 0);
        ib_snake_observe(reference, board, 100);
        decision_laya_encode(board, 0, 1, expected, format);
        assert(memcmp(expected, observations, OBS_SIZE) == 0);
        assert(memcmp(observations, env.transition.observations, OBS_SIZE) != 0);
        validate(env.transition.observations); validate(observations);
        ib_snake_destroy(reference); puf_close(&env);
        std::cout << "format=" << format << " timeout final tokens and autoreset tokens preserved independently\n";
    }
}
