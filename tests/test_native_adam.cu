// Real torch.optim.Adam fixture parity; optional fork-per-GPU NCCL validation.
#include "../src/pufferl.cu"
#include <fstream>
#include <stdexcept>
#include <string>

static void check(bool okay, const char* message) {
    if (!okay) throw std::runtime_error(message);
}
static void gpu(cudaError_t status) { check(status == cudaSuccess, cudaGetErrorString(status)); }
static void comm_check(ncclResult_t status) { check(status == ncclSuccess, ncclGetErrorString(status)); }

static void host_checks() {
    Ini ini{};
    check(puf_optimizer_kind(&ini) == PUF_OPTIMIZER_MUON, "omitted optimizer changed default");
    puf_ini_put(&ini, "train.optimizer", "adam");
    check(puf_optimizer_kind(&ini) == PUF_OPTIMIZER_ADAM, "Adam selection failed");
    check(puf_adam_option(&ini, "adam_beta1", "0.9", 0, 1, false, true) == .9,
        "omitted beta changed default");
    check(muon_grid_size((int64_t)INT_MAX + 1) == 8388608,
        "optimizer launch narrows element counts above INT_MAX");
    const char* keys[] = {"optimizer", "adam_beta1", "adam_beta2", "adam_eps", "adam_weight_decay"};
    const char* bad[] = {"Adam", "1", "-0.1", "0", "-1"};
    for (int i = 0; i < 5; ++i) {
        pid_t child = fork(); check(child >= 0, "fork failed");
        if (!child) {
            puf_ini_set(puf_ini_section(&ini, "train", 1), keys[i], bad[i]);
            if (i == 0) puf_optimizer_kind(&ini);
            else puf_adam_option(&ini, keys[i], "0", 0, i <= 2 ? 1 : INFINITY, i == 3, i <= 2);
            _exit(0);
        }
        int status; check(waitpid(child, &status, 0) == child, "wait failed");
        check(WIFEXITED(status) && WEXITSTATUS(status) != 0, "invalid optimizer option accepted");
    }
    for (const char* value : {"nan", "inf", "true", "0,1", "0.9junk", "1e-1000"}) {
        pid_t child = fork(); check(child >= 0, "fork failed");
        if (!child) {
            puf_ini_put(&ini, "train.adam_beta1", value);
            puf_adam_option(&ini, "adam_beta1", "0.9", 0, 1, false, true);
            _exit(0);
        }
        int status; check(waitpid(child, &status, 0) == child, "wait failed");
        check(WIFEXITED(status) && WEXITSTATUS(status) != 0, "malformed optimizer scalar accepted");
    }
    puf_ini_free(&ini);
    puts("PASS: native optimizer defaults, selection, malformed controls and 64-bit counts");
    fflush(stdout);
}

static std::vector<float> read_values(std::ifstream& file, int n) {
    std::vector<float> values(n);
    for (float& value : values) { file >> value; check(file.good(), "truncated reference fixture"); }
    return values;
}

static void near_array(const float* device, const std::vector<float>& expected,
        const char* label, double& maximum) {
    std::vector<float> actual(expected.size());
    gpu(cudaMemcpy(actual.data(), device, actual.size()*sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < actual.size(); ++i) {
        double difference = fabs((double)actual[i]-expected[i]);
        maximum = std::max(maximum, difference);
        if (!std::isfinite(actual[i]) || difference > 2e-7 + 8e-6*fabs(expected[i])) {
            fprintf(stderr, "%s[%zu]: native %.9g, torch %.9g\n", label, i, actual[i], expected[i]);
            throw std::runtime_error("native Adam differs from PyTorch reference");
        }
    }
    check(actual.back() == 0, "optimizer changed zero padding");
}

static void rank_test(const char* path, int rank, int world, ncclUniqueId id) {
    gpu(cudaSetDevice(rank));
    ncclComm_t comm = nullptr;
    if (world > 1) comm_check(ncclCommInitRank(&comm, world, id, rank));
    cudaStream_t stream; gpu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // Re-read each fixture for eager and graph execution. DP uses eager updates;
    // graph coverage verifies that Adam's counter increments on replay.
    for (bool graph_mode : {false, true}) {
        if (world > 1 && graph_mode) continue;
        std::ifstream file(path);
        std::string magic; int cases; file >> magic >> cases;
        check(magic == "ADAMREF1" && cases > 0 && cases < 100, "invalid fixture header");
        for (int c = 0; c < cases; ++c) {
            std::string name; int n, steps; double beta1, beta2, eps, decay; float max_norm;
            file >> name >> n >> steps >> beta1 >> beta2 >> eps >> decay >> max_norm;
            check(n == 12 && steps == 12, "unexpected test fixture dimensions");
            auto initial = read_values(file, n);
            Prec matrix{.shape = {2, 4}}, bias{.shape = {4}};
            Allocator parameters{}, state{};
            alloc_register(&parameters, &matrix); alloc_register(&parameters, &bias);
            Adam optimizer{}; adam_init(&optimizer, &parameters, beta1, beta2, eps, decay, &state);
            alloc_create(&state);
            Float weights{.shape = {n}}; Prec gradients{.shape = {n}};
            gpu(cudaMalloc(&weights.data, n*sizeof(float)));
            gpu(cudaMalloc(&gradients.data, n*sizeof(float)));
            gpu(cudaMemcpy(weights.data, initial.data(), n*sizeof(float), cudaMemcpyHostToDevice));
            cudaGraph_t graph = nullptr; cudaGraphExec_t executable = nullptr;
            if (graph_mode) {
                gpu(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                adam_step(&optimizer, weights, gradients, max_norm, stream);
                gpu(cudaStreamEndCapture(stream, &graph));
                gpu(cudaGraphInstantiate(&executable, graph, 0));
            }
            double weight_error = 0, moment_error = 0, variance_error = 0;
            for (int step = 0; step < steps; ++step) {
                float lr; file >> lr;
                auto gradient = read_values(file, n);
                auto expected_weights = read_values(file, n);
                auto expected_first = read_values(file, n);
                auto expected_second = read_values(file, n);
                // Different rank gradients average to exactly the CPU fixture.
                if (world > 1) for (float& value : gradient) value *= rank == 0 ? 0.f : 2.f;
                gpu(cudaMemcpyAsync(optimizer.lr, &lr, sizeof(lr), cudaMemcpyHostToDevice, stream));
                gpu(cudaMemcpyAsync(gradients.data, gradient.data(), n*sizeof(float), cudaMemcpyHostToDevice, stream));
                if (world > 1) comm_check(ncclAllReduce(gradients.data, gradients.data,
                    n, ncclFloat, ncclAvg, comm, stream));
                if (graph_mode) gpu(cudaGraphLaunch(executable, stream));
                else adam_step(&optimizer, weights, gradients, max_norm, stream);
                gpu(cudaStreamSynchronize(stream));
                uint64_t actual_step; gpu(cudaMemcpy(&actual_step, optimizer.step, sizeof(actual_step), cudaMemcpyDeviceToHost));
                check(actual_step == (uint64_t)step+1, "Adam bias-correction counter is incorrect");
                near_array(weights.data, expected_weights, "weights", weight_error);
                near_array(optimizer.first_moment.data, expected_first, "first_moment", moment_error);
                near_array(optimizer.second_moment.data, expected_second, "second_moment", variance_error);
            }
            printf("PASS: rank%d %s %s, max_abs weights=%g first=%g second=%g\n",
                rank, name.c_str(), graph_mode ? "graph" : "eager", weight_error, moment_error, variance_error);
            if (executable) gpu(cudaGraphExecDestroy(executable));
            if (graph) gpu(cudaGraphDestroy(graph));
            gpu(cudaFree(weights.data)); gpu(cudaFree(gradients.data)); gpu(cudaFree(state.mem));
            gpu(cudaFree(optimizer.lr)); gpu(cudaFree(optimizer.step)); gpu(cudaFree(optimizer.grad_norm));
            gpu(cudaFree(optimizer.norm_partials)); gpu(cudaFree(optimizer.update_scalars));
            free(parameters.regs); free(state.regs);
        }
    }
    gpu(cudaStreamDestroy(stream));
    if (comm) comm_check(ncclCommDestroy(comm));
}

static int run_rank(TrainContext* context, void*, void* argument) {
    try {
        ncclUniqueId id{};
        if (context->nccl_id) id = *context->nccl_id;
        rank_test(static_cast<const char*>(argument), context->rank, context->world_size, id);
        fflush(stdout);
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "rank%d: %s\n", context->rank, error.what());
        return 1;
    }
}

int main(int argc, char** argv) {
    try {
        host_checks();
        if (argc == 2 && !strcmp(argv[1], "--host-only")) return 0;
        check(argc == 2 || (argc == 3 && !strcmp(argv[2], "--two-gpus")),
            "usage: test_native_adam REFERENCE [--two-gpus] | --host-only");
        int world = argc == 3 ? 2 : 1;
        // Match production: fork before NCCL/CUDA initialization, and use its
        // failure propagation instead of inheriting bootstrap threads in forks.
        std::vector<int> results(world);
        check(puf_launch_native_ranks(world, 0, sizeof(int), results.data(), run_rank, argv[1]) == 0,
            "Adam distributed test failed");
        puts("PASS: native Adam matches PyTorch weights, moments, counters and zero padding");
        return 0;
    } catch (const std::exception& error) { fprintf(stderr, "%s\n", error.what()); return 1; }
}
