#ifndef PUFFER_NATIVE_DISTRIBUTED_H
#define PUFFER_NATIVE_DISTRIBUTED_H

// The supervisor owns no CUDA context. Every CUDA/NCCL rank is a child.
#include <poll.h>
#include <fcntl.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif

static volatile sig_atomic_t puf_native_launch_signal = 0;
static void puf_native_launch_interrupted(int signal) {
    puf_native_launch_signal = signal;
}

static bool puf_native_pipe_read(int fd, void* data, size_t bytes) {
    char* out = (char*)data;
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
    const char* in = (const char*)data;
    while (bytes) {
        ssize_t count = write(fd, in, bytes);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        in += count;
        bytes -= count;
    }
    return true;
}

// Callbacks write a nonempty fixed-size result. Ranks stay alive until all
// results arrive so an early exit cannot tear down a peer's communicator.
// Any failed rank causes the supervisor to kill and reap every remaining rank.
static int puf_launch_native_ranks(int world_size, int gpu_offset,
        size_t result_bytes, void* results,
        int (*worker)(TrainContext*, void* result, void* user), void* user) {
    pid_t* pids = (pid_t*)calloc(world_size, sizeof(pid_t));
    struct pollfd* reports = (struct pollfd*)calloc(world_size, sizeof(struct pollfd));
    int* writers = (int*)calloc(world_size, sizeof(int));
    size_t* progress = (size_t*)calloc(world_size, sizeof(size_t));
    assert(pids && reports && writers && progress);
    int ids[2] = {-1, -1}, release[2] = {-1, -1};
    struct sigaction handler = {}, previous_int = {}, previous_term = {};
    bool handlers_installed = false, failed = false;
    int complete = 0, status = -1;
    pid_t supervisor = getpid();
    for (int rank = 0; rank < world_size; ++rank) {
        pids[rank] = reports[rank].fd = writers[rank] = -1;
        reports[rank].events = POLLIN;
    }
    if (pipe(release) || (world_size > 1 && pipe(ids))) {
        perror("native launcher pipe");
        goto cleanup;
    }
    for (int rank = 0; rank < world_size; ++rank) {
        int pair[2];
        if (pipe(pair)) {
            perror("native launcher result pipe");
            goto cleanup;
        }
        reports[rank].fd = pair[0];
        writers[rank] = pair[1];
    }
    fflush(NULL);
    for (int rank = 0; rank < world_size; ++rank) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("native launcher fork");
            goto cleanup;
        }
        if (pid == 0) {
#ifdef __linux__
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
                    ncclResult_t code = ncclGetUniqueId(&id);
                    if (code != ncclSuccess) {
                        fprintf(stderr, "NCCL unique ID: %s\n", ncclGetErrorString(code));
                        _exit(1);
                    }
                    for (int other = 1; other < world_size; ++other) {
                        if (!puf_native_pipe_write(ids[1], &id, sizeof(id))) _exit(1);
                    }
                    close(ids[1]);
                } else {
                    close(ids[1]);
                    if (!puf_native_pipe_read(ids[0], &id, sizeof(id))) _exit(1);
                    close(ids[0]);
                }
            }
            TrainContext context = {};
            context.rank = rank;
            context.world_size = world_size;
            context.gpu_id = gpu_offset + (rank == 0 ? world_size - 1 : rank - 1);
            context.artifact_owner = rank == 0;
            context.nccl_id = world_size > 1 ? &id : NULL;
            void* result = calloc(1, result_bytes);
            if (!result || worker(&context, result, user) != 0) _exit(1);
            if (!puf_native_pipe_write(writers[rank], result, result_bytes)) _exit(1);
            close(writers[rank]);
            free(result);
            char value;
            ssize_t count;
            do { count = read(release[0], &value, 1); } while (count < 0 && errno == EINTR);
            close(release[0]);
            fflush(NULL);
            _exit(count == 0 ? 0 : 1);
        }
        pids[rank] = pid;
    }
    for (int rank = 0; rank < world_size; ++rank) {
        close(writers[rank]);
        writers[rank] = -1;
        int flags = fcntl(reports[rank].fd, F_GETFL);
        if (flags < 0 || fcntl(reports[rank].fd, F_SETFL, flags | O_NONBLOCK) < 0) {
            perror("native launcher nonblocking result pipe");
            goto cleanup;
        }
    }
    for (int i = 0; i < 2; ++i) {
        if (ids[i] >= 0) close(ids[i]);
        ids[i] = -1;
    }
    close(release[0]);
    release[0] = -1;
    handler.sa_handler = puf_native_launch_interrupted;
    sigemptyset(&handler.sa_mask);
    puf_native_launch_signal = 0;
    sigaction(SIGINT, &handler, &previous_int);
    sigaction(SIGTERM, &handler, &previous_term);
    handlers_installed = true;
    while (complete < world_size && !failed && !puf_native_launch_signal) {
        int ready = poll(reports, world_size, 100);
        if (ready < 0 && errno != EINTR) {
            perror("native launcher poll");
            failed = true;
        }
        for (int rank = 0; rank < world_size && !failed; ++rank) {
            if (reports[rank].fd >= 0 &&
                    (reports[rank].revents & (POLLIN | POLLHUP | POLLERR))) {
                // Partial reads never block detection of a failed/stopped peer.
                while (progress[rank] < result_bytes && !puf_native_launch_signal) {
                    ssize_t count = read(reports[rank].fd,
                        (char*)results + rank * result_bytes + progress[rank],
                        result_bytes - progress[rank]);
                    if (count > 0) {
                        progress[rank] += count;
                        continue;
                    }
                    if (count < 0 && errno == EINTR) continue;
                    if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
                    failed = true;
                    break;
                }
                if (failed) {
                    fprintf(stderr, "Native rank %d failed before reporting its result\n", rank);
                    break;
                }
                if (progress[rank] == result_bytes) {
                    ++complete;
                    close(reports[rank].fd);
                    reports[rank].fd = -1;
                }
            }
            int code = 0;
            pid_t exited = waitpid(pids[rank], &code, WNOHANG);
            if (exited > 0) {
                pids[rank] = -1;
                fprintf(stderr, "Native rank %d exited before all ranks completed (status %d)\n",
                    rank, code);
                failed = true;
            }
        }
    }
    if (puf_native_launch_signal) {
        fprintf(stderr, "Native launch interrupted by signal %d\n", (int)puf_native_launch_signal);
        failed = true;
    }
    if (!failed) {
        close(release[1]);
        release[1] = -1;
        for (int rank = 0; rank < world_size; ++rank) {
            int code = 0;
            pid_t exited;
            do { exited = waitpid(pids[rank], &code, 0); } while (exited < 0 && errno == EINTR);
            if (exited < 0 || !WIFEXITED(code) || WEXITSTATUS(code) != 0) failed = true;
            pids[rank] = -1;
        }
        status = failed ? -1 : 0;
    }
cleanup:
    if (status != 0) {
        for (int rank = 0; rank < world_size; ++rank) {
            if (pids[rank] > 0) kill(pids[rank], SIGKILL);
        }
    }
    for (int rank = 0; rank < world_size; ++rank) {
        if (pids[rank] > 0) {
            while (waitpid(pids[rank], NULL, 0) < 0 && errno == EINTR) {}
        }
        if (reports[rank].fd >= 0) close(reports[rank].fd);
        if (writers[rank] >= 0) close(writers[rank]);
    }
    for (int i = 0; i < 2; ++i) {
        if (ids[i] >= 0) close(ids[i]);
        if (release[i] >= 0) close(release[i]);
    }
    if (handlers_installed) {
        sigaction(SIGINT, &previous_int, NULL);
        sigaction(SIGTERM, &previous_term, NULL);
    }
    free(pids);
    free(reports);
    free(writers);
    free(progress);
    return status;
}

#endif
