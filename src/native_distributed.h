#ifndef PUFFER_NATIVE_DISTRIBUTED_H
#define PUFFER_NATIVE_DISTRIBUTED_H

// Include after TrainContext. The supervising process must not initialize CUDA
// before calling this launcher: every CUDA/NCCL rank, including rank 0, is a child.
#include <poll.h>
#include <climits>
#include <cstdint>
#include <fcntl.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
#include <vector>

static volatile sig_atomic_t puf_native_launch_signal = 0;
static void puf_native_launch_interrupted(int signal) {
    puf_native_launch_signal = signal;
}

static bool puf_native_pipe_read(int fd, void* data, size_t bytes) {
    char* out = static_cast<char*>(data);
    while (bytes) {
        ssize_t count = read(fd, out, bytes);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        out += count;
        bytes -= count;
    }
    return true;
}

static bool puf_native_pipe_write(int fd, const void* data, size_t bytes) {
    const char* in = static_cast<const char*>(data);
    while (bytes) {
        ssize_t count = write(fd, in, bytes);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        in += count;
        bytes -= count;
    }
    return true;
}

// Each callback writes one fixed-size result. The parent receives results in
// rank order and returns 0 only if every callback and process succeeded. Ranks
// remain alive until all results arrive, so an early rank exit cannot tear down
// a communicator while another rank is still using it. On any failed rank the
// supervisor kills and reaps all remaining ranks, including a blocked rank 0.
static int puf_launch_native_ranks(int world_size, int gpu_offset,
        size_t result_bytes, void* results,
        int (*worker)(TrainContext*, void* result, void* user), void* user) {
    if (world_size < 1 || gpu_offset < 0 || gpu_offset > INT_MAX - (world_size - 1) ||
            result_bytes == SIZE_MAX || (result_bytes &&
                (!results || (size_t)world_size > SIZE_MAX / result_bytes)) || !worker) {
        fprintf(stderr, "Invalid native distributed launcher arguments\n");
        return -1;
    }
    std::vector<pid_t> pids(world_size, -1);
    std::vector<pollfd> reports(world_size);
    std::vector<int> writers(world_size, -1);
    std::vector<bool> received(world_size, false);
    std::vector<size_t> progress(world_size, 0);
    std::vector<std::vector<unsigned char>> messages(world_size,
        std::vector<unsigned char>(result_bytes + 1));
    int ids[2] = {-1, -1};
    int release[2] = {-1, -1};
    for (auto& report : reports) report.fd = -1;
    auto cleanup_fds = [&]() {
        for (auto& report : reports) if (report.fd >= 0) { close(report.fd); report.fd = -1; }
        for (int& fd : writers) if (fd >= 0) { close(fd); fd = -1; }
        for (int& fd : ids) if (fd >= 0) { close(fd); fd = -1; }
        for (int& fd : release) if (fd >= 0) { close(fd); fd = -1; }
    };
    auto kill_and_reap = [&]() {
        for (pid_t pid : pids) if (pid > 0) kill(pid, SIGKILL);
        for (pid_t& pid : pids) if (pid > 0) {
            while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {}
            pid = -1;
        }
    };
    if (pipe(release) || (world_size > 1 && pipe(ids))) {
        perror("native launcher pipe"); cleanup_fds(); return -1;
    }
    for (int rank = 0; rank < world_size; ++rank) {
        int pair[2];
        if (pipe(pair)) {
            perror("native launcher result pipe"); cleanup_fds(); return -1;
        }
        reports[rank] = {pair[0], POLLIN, 0};
        writers[rank] = pair[1];
    }
    const pid_t supervisor = getpid();
    fflush(nullptr);
    for (int rank = 0; rank < world_size; ++rank) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("native launcher fork"); kill_and_reap(); cleanup_fds(); return -1;
        }
        if (pid == 0) {
#ifdef __linux__
            // Avoid stranded GPU workers if the supervisor is killed outright.
            prctl(PR_SET_PDEATHSIG, SIGKILL);
            if (getppid() != supervisor) _exit(1);
#endif
            close(release[1]);
            for (int other = 0; other < world_size; ++other) {
                close(reports[other].fd);
                if (other != rank) close(writers[other]);
            }
            ncclUniqueId id;
            if (world_size > 1) {
                if (rank == 0) {
                    close(ids[0]);
                    ncclResult_t status = ncclGetUniqueId(&id);
                    if (status != ncclSuccess) {
                        fprintf(stderr, "NCCL unique ID: %s\n", ncclGetErrorString(status));
                        _exit(1);
                    }
                    for (int other = 1; other < world_size; ++other)
                        if (!puf_native_pipe_write(ids[1], &id, sizeof(id))) _exit(1);
                    close(ids[1]);
                } else {
                    close(ids[1]);
                    if (!puf_native_pipe_read(ids[0], &id, sizeof(id))) _exit(1);
                    close(ids[0]);
                }
            }
            TrainContext context{};
            context.rank = rank;
            context.world_size = world_size;
            // Preserve launch_train's established rank/device assignment.
            context.gpu_id = gpu_offset + (rank == 0 ? world_size - 1 : rank - 1);
            context.artifact_owner = rank == 0;
            context.nccl_id = world_size > 1 ? &id : nullptr;
            void* result = calloc(1, result_bytes ? result_bytes : 1);
            if (!result || worker(&context, result, user) != 0) _exit(1);
            const char ready = 1;
            if (!puf_native_pipe_write(writers[rank], &ready, 1) ||
                    !puf_native_pipe_write(writers[rank], result, result_bytes)) _exit(1);
            close(writers[rank]);
            free(result);
            char value;
            ssize_t count;
            do { count = read(release[0], &value, 1); } while (count < 0 && errno == EINTR);
            close(release[0]);
            fflush(nullptr);
            _exit(count == 0 ? 0 : 1);
        }
        pids[rank] = pid;
    }
    for (int& fd : writers) { close(fd); fd = -1; }
    for (int& fd : ids) if (fd >= 0) { close(fd); fd = -1; }
    close(release[0]); release[0] = -1;
    // Never block while reading a partial result: even a stopped reporter must
    // not prevent detection of a different failed rank or a termination signal.
    for (auto& report : reports) {
        int flags = fcntl(report.fd, F_GETFL);
        if (flags < 0 || fcntl(report.fd, F_SETFL, flags | O_NONBLOCK) < 0) {
            perror("native launcher nonblocking result pipe");
            kill_and_reap(); cleanup_fds(); return -1;
        }
    }

    struct sigaction handler{}, previous_int{}, previous_term{};
    handler.sa_handler = puf_native_launch_interrupted;
    sigemptyset(&handler.sa_mask);
    puf_native_launch_signal = 0;
    sigaction(SIGINT, &handler, &previous_int);
    sigaction(SIGTERM, &handler, &previous_term);
    bool failed = false;
    int complete = 0;
    while (complete < world_size && !failed && !puf_native_launch_signal) {
        int ready = poll(reports.data(), reports.size(), 100);
        if (ready < 0 && errno != EINTR) { perror("native launcher poll"); failed = true; }
        for (int rank = 0; rank < world_size && !failed; ++rank) {
            if (!received[rank] && (reports[rank].revents & (POLLIN | POLLHUP | POLLERR))) {
                while (progress[rank] < messages[rank].size() && !puf_native_launch_signal) {
                    ssize_t count = read(reports[rank].fd, messages[rank].data() + progress[rank],
                        messages[rank].size() - progress[rank]);
                    if (count > 0) { progress[rank] += count; continue; }
                    if (count < 0 && errno == EINTR) continue;
                    if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
                    failed = true; break;
                }
                if (failed || (progress[rank] && messages[rank][0] != 1)) {
                    fprintf(stderr, "Native rank %d failed before reporting its result\n", rank);
                    failed = true; break;
                }
                if (progress[rank] == messages[rank].size()) {
                    if (result_bytes)
                        memcpy(static_cast<char*>(results) + rank * result_bytes,
                            messages[rank].data() + 1, result_bytes);
                    received[rank] = true; ++complete;
                    close(reports[rank].fd); reports[rank].fd = -1;
                }
            }
            int status = 0;
            pid_t exited = waitpid(pids[rank], &status, WNOHANG);
            if (exited > 0) {
                pids[rank] = -1;
                fprintf(stderr, "Native rank %d exited before all ranks completed (status %d)\n", rank, status);
                failed = true;
            }
        }
    }
    if (puf_native_launch_signal) {
        fprintf(stderr, "Native launch interrupted by signal %d\n", int(puf_native_launch_signal));
        failed = true;
    }
    if (failed) kill_and_reap();
    else {
        close(release[1]); release[1] = -1;
        for (pid_t& pid : pids) {
            int status = 0;
            pid_t exited;
            do { exited = waitpid(pid, &status, 0); } while (exited < 0 && errno == EINTR);
            if (exited < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) failed = true;
            pid = -1;
        }
    }
    sigaction(SIGINT, &previous_int, nullptr);
    sigaction(SIGTERM, &previous_term, nullptr);
    cleanup_fds();
    return failed ? -1 : 0;
}

#endif
