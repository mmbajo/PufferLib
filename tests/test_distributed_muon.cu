// Focused optimizer test: real NCCL broadcasts, independent replicated state,
// several momentum updates, rectangular matrices, vectors and empty owners.
#include "../src/pufferl.cu"

#include <stdexcept>

static void require(bool okay, const char* message) {
    if (!okay) throw std::runtime_error(message);
}
static void gpu(cudaError_t error) {
    require(error == cudaSuccess, cudaGetErrorString(error));
}
static void nccl_ok(ncclResult_t error) {
    require(error == ncclSuccess, ncclGetErrorString(error));
}

static void transfer(int fd, void* bytes, size_t count, bool writing) {
    char* position = (char*)bytes;
    while (count) {
        ssize_t n = writing ? write(fd, position, count) : read(fd, position, count);
        if (n < 0 && errno == EINTR) continue;
        require(n > 0, "test child pipe closed unexpectedly");
        position += n;
        count -= n;
    }
}

static void host_layout() {
    // Both flat-vector count and aggregate count cross INT_MAX; no large
    // allocation is made. The matrix's individual cuBLAS dimensions still fit.
    Prec fields[] = {{.shape = {(int64_t)INT_MAX + 4097}},
                     {.shape = {32768, 65536}}, {.shape = {4}}};
    Allocator params = {};
    for (auto& field : fields) alloc_register(&params, &field);
    Muon plans[2] = {};
    for (int rank = 0; rank < 2; ++rank) {
        muon_plan(&plans[rank], &params, rank, 2, true);
        require(plans[rank].parameters[1].offset > INT_MAX,
            "parameter offset was narrowed");
        require(plans[rank].parameters[1].count == (int64_t)INT_MAX + 1,
            "matrix count was narrowed");
        require(plans[rank].replicated_momentum_elems == params.total_elems,
            "flat momentum count differs");
    }
    require(plans[0].momentum_elems + plans[1].momentum_elems == params.total_elems,
        "ownership does not partition all elements");
    require(muon_grid_size((int64_t)INT_MAX + 1) == 8388608,
        "launch geometry overflow above INT_MAX");
    require(muon_grid_size(INT64_MAX) == INT_MAX,
        "launch geometry overflow near INT64_MAX");
    require(muon_grid_size((int64_t)INT_MAX + 1, 256) == 256,
        "norm reduction block cap differs");
    for (auto& plan : plans) free(plan.parameters);
    free(params.regs);
    puts("PASS: Muon host layouts and launch counts above INT_MAX");
}

struct Result {
    std::vector<float> weights;
    int64_t momentum = 0, full_momentum = 0, scratch = 0, full_scratch = 0;
    double error = 0;
};

static void send_result(int fd, Result& result) {
    int64_t header[] = {(int64_t)result.weights.size(), result.momentum,
        result.full_momentum, result.scratch, result.full_scratch};
    transfer(fd, header, sizeof(header), true);
    transfer(fd, &result.error, sizeof(result.error), true);
    transfer(fd, result.weights.data(), result.weights.size() * sizeof(float), true);
}

static Result receive_result(int fd) {
    Result result;
    int64_t header[5];
    transfer(fd, header, sizeof(header), false);
    require(header[0] > 0 && header[0] < 10000, "invalid child result size");
    result.weights.resize(header[0]);
    result.momentum = header[1]; result.full_momentum = header[2];
    result.scratch = header[3]; result.full_scratch = header[4];
    transfer(fd, &result.error, sizeof(result.error), false);
    transfer(fd, result.weights.data(), result.weights.size() * sizeof(float), false);
    return result;
}

static void release(Muon& m, Allocator& a) {
    gpu(cudaFree(a.mem));
    gpu(cudaFree(m.lr));
    gpu(cudaFree(m.grad_norm));
    gpu(cudaFree(m.ns_norm));
    gpu(cudaFree(m.norm_partials));
    free(a.regs);
    free(m.parameters);
}

static Result rank_case(int rank, int world, ncclComm_t comm, bool empty_owner) {
    gpu(cudaSetDevice(rank));
    fprintf(stderr, "Muon test rank %d starting %s\n", rank,
        empty_owner ? "empty-owner case" : "mixed case");
    cublas_init_handle();
    cudaStream_t stream;
    gpu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<Prec> fields = empty_owner
        ? std::vector<Prec>{{.shape = {8, 8}}}
        : std::vector<Prec>{{.shape = {16, 8}}, {.shape = {8, 16}},
            {.shape = {32}}, {.shape = {4, 4}}, {.shape = {4}}};
    Allocator params = {}, reference_alloc = {}, sharded_alloc = {};
    for (auto& field : fields) alloc_register(&params, &field);
    Muon reference = {}, sharded = {};
    muon_init(&reference, &params, 0.95, &reference_alloc);
    muon_init(&sharded, &params, 0.95, &sharded_alloc, rank, world, comm, true);
    alloc_create(&reference_alloc);
    alloc_create(&sharded_alloc);
    int64_t count = params.total_elems;
    std::vector<float> initial(count), gradients(count), reference_values(count), actual(count);
    for (int64_t i = 0; i < count; ++i) initial[i] = 0.1f * sinf((float)i * 0.21f);
    Float wref = {.shape = {count}}, wlocal = {.shape = {count}};
    Prec gref = {.shape = {count}}, glocal = {.shape = {count}};
    gpu(cudaMalloc(&wref.data, count * sizeof(float)));
    gpu(cudaMalloc(&wlocal.data, count * sizeof(float)));
    gpu(cudaMalloc(&gref.data, count * sizeof(float)));
    gpu(cudaMalloc(&glocal.data, count * sizeof(float)));
    gpu(cudaMemcpy(wref.data, initial.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    gpu(cudaMemcpy(wlocal.data, initial.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    const float lr = 0.003f;
    gpu(cudaMemcpy(reference.lr, &lr, sizeof(lr), cudaMemcpyHostToDevice));
    gpu(cudaMemcpy(sharded.lr, &lr, sizeof(lr), cudaMemcpyHostToDevice));
    Result result;
    result.momentum = sharded.momentum_elems;
    result.full_momentum = reference.momentum_elems;
    result.scratch = sharded.scratch_elems;
    result.full_scratch = reference.scratch_elems;
    for (int step = 0; step < 5; ++step) {
        for (int64_t i = 0; i < count; ++i)
            gradients[i] = 0.2f * cosf((float)(i + 3 * step) * 0.17f)
                + 0.02f * (rank + 1);
        gpu(cudaMemcpyAsync(glocal.data, gradients.data(), count * sizeof(float),
            cudaMemcpyHostToDevice, stream));
        if (getenv("MUON_TEST_TRACE")) fprintf(stderr, "rank %d step %d gradient reduction\n", rank, step);
        if (world > 1) nccl_ok(ncclAllReduce(glocal.data, glocal.data,
            count, ncclFloat, ncclAvg, comm, stream));
        // Copy the exact common global gradient, so the comparison isolates
        // optimizer semantics from NCCL reduction rounding.
        gpu(cudaMemcpyAsync(gref.data, glocal.data, count * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));
        if (getenv("MUON_TEST_TRACE")) fprintf(stderr, "rank %d step %d reference update\n", rank, step);
        muon_step(&reference, wref, gref, 0.7f, stream);
        if (getenv("MUON_TEST_TRACE")) fprintf(stderr, "rank %d step %d distributed update\n", rank, step);
        muon_step(&sharded, wlocal, glocal, 0.7f, stream);
        gpu(cudaStreamSynchronize(stream));
        gpu(cudaMemcpy(reference_values.data(), wref.data, count * sizeof(float),
            cudaMemcpyDeviceToHost));
        gpu(cudaMemcpy(actual.data(), wlocal.data, count * sizeof(float), cudaMemcpyDeviceToHost));
        for (int64_t i = 0; i < count; ++i) {
            require(std::isfinite(actual[i]), "nonfinite optimizer result");
            result.error = std::max(result.error, fabs((double)actual[i] - reference_values[i]));
        }
        require(result.error <= 2e-6, "sharded update differs from replicated Muon");
        std::vector<float> reference_momentum(count), local_momentum(numel(sharded.mb.shape));
        gpu(cudaMemcpy(reference_momentum.data(), reference.mb.data, count * sizeof(float),
            cudaMemcpyDeviceToHost));
        gpu(cudaMemcpy(local_momentum.data(), sharded.mb.data,
            local_momentum.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int j = 0; j < params.num_regs; ++j) {
            const MuonParameter& p = sharded.parameters[j];
            if (p.owner != rank) continue;
            for (int64_t i = 0; i < p.count; ++i)
                require(local_momentum[p.momentum_offset + i] == reference_momentum[p.offset + i],
                    "owned momentum differs from replicated state");
        }
    }
    result.weights = std::move(actual);
    gpu(cudaFree(wref.data)); gpu(cudaFree(wlocal.data));
    gpu(cudaFree(gref.data)); gpu(cudaFree(glocal.data));
    release(reference, reference_alloc); release(sharded, sharded_alloc);
    free(params.regs);
    gpu(cudaStreamDestroy(stream));
    return result;
}

int main(int argc, char** argv) {
    pid_t child = -1;
    try {
        setvbuf(stdout, NULL, _IOLBF, 0);
        host_layout();
        if (argc == 2 && !strcmp(argv[1], "--host-only")) return 0;
        int world = argc == 2 ? atoi(argv[1]) : 1;
        require(world == 1 || world == 2, "usage: test_distributed_muon [1|2|--host-only]");
        // Match the production launch topology: fork before any CUDA/NCCL
        // initialization, one process per device. Separate host threads can
        // deadlock when CUDA's process-wide runtime locks meet a blocking
        // allocation/lazy cuBLAS init and an outstanding NCCL kernel.
        int ids[2] = {-1, -1}, results_pipe[2] = {-1, -1};
        int rank = 0;
        if (world > 1) {
            require(pipe(ids) == 0 && pipe(results_pipe) == 0, "pipe creation failed");
            child = fork();
            require(child >= 0, "fork failed");
            rank = child == 0 ? 1 : 0;
            close(ids[rank == 0 ? 0 : 1]);
            close(results_pipe[rank == 0 ? 1 : 0]);
        }
        int devices = 0;
        gpu(cudaGetDeviceCount(&devices));
        require(devices >= world, "insufficient GPUs");
        ncclComm_t comm = NULL;
        if (world > 1) {
            ncclUniqueId id;
            if (rank == 0) {
                nccl_ok(ncclGetUniqueId(&id));
                transfer(ids[1], &id, sizeof(id), true);
                close(ids[1]);
            } else {
                transfer(ids[0], &id, sizeof(id), false);
                close(ids[0]);
            }
            gpu(cudaSetDevice(rank));
            fprintf(stderr, "Muon test rank %d initializing NCCL\n", rank);
            nccl_ok(ncclCommInitRank(&comm, world, id, rank));
        }
        for (bool empty_owner : {false, true}) {
            std::vector<Result> results(world);
            results[rank] = rank_case(rank, world, comm, empty_owner);
            if (rank != 0) {
                send_result(results_pipe[1], results[rank]);
                continue;
            }
            if (world > 1) results[1] = receive_result(results_pipe[0]);
            int64_t owned = 0;
            for (int rank = 0; rank < world; ++rank) {
                owned += results[rank].momentum;
                require(results[rank].weights == results[0].weights,
                    "replicas differ after update broadcasts");
                if (world > 1 && !empty_owner)
                    require(results[rank].momentum < results[rank].full_momentum,
                        "momentum was not sharded");
                printf("rank=%d case=%s momentum=%ld/%ld scratch=%ld/%ld max_error=%.9g\n",
                    rank, empty_owner ? "empty-owner" : "mixed", results[rank].momentum,
                    results[rank].full_momentum, results[rank].scratch,
                    results[rank].full_scratch, results[rank].error);
            }
            require(owned == results[0].full_momentum, "momentum ownership sum differs");
            if (world > 1 && empty_owner) require(results[1].momentum == 0,
                "empty-owner fixture did not exercise an empty rank");
        }
        if (comm) nccl_ok(ncclCommDestroy(comm));
        if (rank != 0) {
            close(results_pipe[1]);
            return 0;
        }
        if (world > 1) {
            close(results_pipe[0]);
            int status = 0;
            require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
                "test child failed");
        }
        printf("PASS: %d-GPU distributed Muon parity, state ownership and empty ranks\n", world);
        return 0;
    } catch (const std::exception& e) {
        if (child > 0) {
            kill(child, SIGTERM);
            while (waitpid(child, NULL, 0) < 0 && errno == EINTR) {}
        }
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
