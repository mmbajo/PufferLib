// Exercise the process supervisor without initializing CUDA or requiring GPUs.
// Only NCCL's unique-ID API is replaced; the production launcher is unchanged.
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

struct ncclUniqueId { unsigned char value[128]; };
using ncclResult_t = int;
constexpr int ncclSuccess = 0;
static bool fail_unique_id = false;
static int ncclGetUniqueId(ncclUniqueId* id) {
    if (fail_unique_id) return 1;
    memset(id, 42, sizeof(*id));
    return ncclSuccess;
}
static const char* ncclGetErrorString(int) { return "expected test unique-ID failure"; }
struct TrainContext {
    int rank, world_size, gpu_id, artifact_owner;
    ncclUniqueId* nccl_id;
};
#include "../src/native_distributed.h"

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Result {
    int rank, world_size, gpu_id, artifact_owner;
    // Larger than a typical pipe buffer: verify complete result transfers.
    unsigned char payload[128 * 1024];
};
enum Mode { SUCCESS, RETURN_ERROR, EXIT_WITHOUT_RESULT, ABORT_RANK, EMPTY_RESULT, WAIT_FOR_INTERRUPT };
struct Task { Mode mode; int offset; };

int worker(TrainContext* context, void* output, void* opaque) {
    auto task = *static_cast<Task*>(opaque);
    if (task.mode == WAIT_FOR_INTERRUPT) {
        poll(nullptr, 0, 5000);
        return 0;
    }
    if (task.mode != SUCCESS && task.mode != EMPTY_RESULT) {
        if (context->rank == 1) {
            if (task.mode == EXIT_WITHOUT_RESULT) _exit(0);
            if (task.mode == ABORT_RANK) raise(SIGABRT);
            return 1;
        }
        // Failed-rank cleanup should terminate these peers immediately. A
        // broken supervisor has a bounded delay before the test reports failure.
        poll(nullptr, 0, 5000);
        return 0;
    }
    if (context->rank < 0 || context->rank >= context->world_size ||
            context->artifact_owner != (context->rank == 0)) return 1;
    const int expected = task.offset + (context->rank == 0 ? context->world_size - 1 : context->rank - 1);
    if (context->gpu_id != expected) return 1;
    if (context->world_size == 1) {
        if (context->nccl_id != nullptr) return 1;
    } else {
        if (!context->nccl_id) return 1;
        for (unsigned char byte : context->nccl_id->value) if (byte != 42) return 1;
    }
    if (task.mode == EMPTY_RESULT) return 0;
    auto* result = static_cast<Result*>(output);
    result->rank = context->rank;
    result->world_size = context->world_size;
    result->gpu_id = context->gpu_id;
    result->artifact_owner = context->artifact_owner;
    memset(result->payload, context->rank + 17, sizeof(result->payload));
    return 0;
}
void require_reaped() {
    int status = 0;
    errno = 0;
    require(waitpid(-1, &status, WNOHANG) == -1 && errno == ECHILD,
        "launcher left a live or unreaped child");
}
void success_cases() {
    for (int world_size : {1, 2, 4}) {
        Task task{SUCCESS, 2};
        std::vector<Result> results(world_size);
        require(puf_launch_native_ranks(world_size, task.offset, sizeof(Result),
            results.data(), worker, &task) == 0, "valid launcher invocation failed");
        for (int rank = 0; rank < world_size; ++rank) {
            const auto& result = results[rank];
            require(result.rank == rank && result.world_size == world_size,
                "rank result order or world size differs");
            require(result.gpu_id == task.offset + (rank == 0 ? world_size - 1 : rank - 1),
                "rank/device assignment differs");
            require(result.artifact_owner == (rank == 0), "artifact owner differs");
            for (unsigned char byte : result.payload)
                require(byte == rank + 17, "large rank result was truncated or mixed");
        }
        require_reaped();
        task.mode = EMPTY_RESULT;
        require(puf_launch_native_ranks(world_size, task.offset, 0, nullptr, worker, &task) == 0,
            "zero-byte result invocation failed");
        require_reaped();
    }
}
void failure_cases() {
    puts("Native launcher: testing expected failure reports");
    for (Mode mode : {RETURN_ERROR, EXIT_WITHOUT_RESULT, ABORT_RANK}) {
        Task task{mode, 0};
        std::vector<Result> results(4);
        auto begin = std::chrono::steady_clock::now();
        require(puf_launch_native_ranks(4, 0, sizeof(Result), results.data(), worker, &task) != 0,
            "failed rank was reported as successful");
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        require(elapsed < 3, "failed rank did not promptly stop its blocked peers");
        require_reaped();
    }
    fail_unique_id = true;
    Task task{EMPTY_RESULT, 0};
    require(puf_launch_native_ranks(4, 0, 0, nullptr, worker, &task) != 0,
        "unique-ID initialization failure was reported as success");
    fail_unique_id = false;
    require_reaped();
}

volatile sig_atomic_t restored_handler_calls = 0;
void restored_handler(int) { ++restored_handler_calls; }

void interruption_case() {
    // Install a harmless previous handler. The signaling helper retries so an
    // early signal during fork setup cannot race with launcher handler setup.
    struct sigaction original{}, marker{};
    require(sigaction(SIGTERM, nullptr, &original) == 0, "cannot read original SIGTERM handler");
    marker.sa_handler = restored_handler;
    marker.sa_flags = SA_RESTART;
    sigemptyset(&marker.sa_mask);
    sigaddset(&marker.sa_mask, SIGUSR1);
    require(sigaction(SIGTERM, &marker, nullptr) == 0, "cannot install test SIGTERM handler");
    const pid_t supervisor = getpid();
    pid_t helper = fork();
    if (helper < 0) {
        sigaction(SIGTERM, &original, nullptr);
        throw std::runtime_error("cannot fork SIGTERM helper");
    }
    if (helper == 0) {
#ifdef __linux__
        prctl(PR_SET_PDEATHSIG, SIGKILL);
        if (getppid() != supervisor) _exit(1);
#endif
        for (int attempt = 0; attempt < 100; ++attempt) {
            poll(nullptr, 0, 20);
            if (kill(supervisor, SIGTERM) != 0) _exit(1);
        }
        _exit(0);
    }
    Task task{WAIT_FOR_INTERRUPT, 0};
    auto begin = std::chrono::steady_clock::now();
    int result = puf_launch_native_ranks(4, 0, 0, nullptr, worker, &task);
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    // Reap the helper before checking that the launcher reaped all its children.
    kill(helper, SIGKILL);
    int helper_status = 0;
    pid_t reaped;
    do { reaped = waitpid(helper, &helper_status, 0); } while (reaped < 0 && errno == EINTR);
    struct sigaction after{};
    bool restored = sigaction(SIGTERM, nullptr, &after) == 0 &&
        after.sa_handler == restored_handler && (after.sa_flags & SA_RESTART) &&
        sigismember(&after.sa_mask, SIGUSR1) == 1;
    if (restored) {
        sig_atomic_t before = restored_handler_calls;
        raise(SIGTERM);
        restored = restored_handler_calls == before + 1;
    }
    int reset = sigaction(SIGTERM, &original, nullptr);
    require(reaped == helper, "could not reap SIGTERM helper");
    require(result != 0, "parent SIGTERM was reported as successful training");
    require(elapsed < 3, "parent SIGTERM did not promptly terminate blocked ranks");
    require(restored && reset == 0, "launcher did not restore the previous SIGTERM handler/mask/flags");
    require_reaped();
}
} // namespace

int main() {
    setbuf(stdout, nullptr);
    rlimit core_limit{0, 0};
    setrlimit(RLIMIT_CORE, &core_limit);
    try {
        success_cases();
        failure_cases();
        interruption_case();
        puts("Native launcher: PASS (1/2/4 ranks, device mapping, large/empty results, "
             "failed return, early zero exit, signal death, initialization failure, "
             "parent SIGTERM, restored signal handler, peer cleanup)");
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "Native launcher test: %s\n", error.what());
        return 1;
    }
}
