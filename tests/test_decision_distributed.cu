// Two-GPU integration of the native Transformer adapter, NCCL and real PPO.
// The supervisor never initializes CUDA: each trial starts fresh rank processes.
#undef PUFFERLIB_BUILD_MAIN
#include "../src/pufferl.cu"

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int world = 2;
constexpr int local_batch = 4;
constexpr int updates = 3;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void gpu(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void collective(ncclResult_t error) {
    if (error != ncclSuccess) throw std::runtime_error(ncclGetErrorString(error));
}
template<class T> std::vector<T> download(const T* data, size_t count) {
    std::vector<T> result(count);
    gpu(cudaMemcpy(result.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
    return result;
}
template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) { gpu(cudaMalloc(&data, count * sizeof(T))); }
    explicit DeviceBuffer(const std::vector<T>& values) : DeviceBuffer(values.size()) {
        gpu(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
DecisionPolicyWeights* weights(PuffeRL* p) {
    return static_cast<DecisionPolicyWeights*>(p->policies[0].weights.encoder);
}
DecisionPolicyActivations* training(PuffeRL* p) {
    return static_cast<DecisionPolicyActivations*>(p->train_activs.encoder);
}
void close_enough(float actual, float expected, const std::string& label,
        float absolute = 4e-5f, float relative = 5e-4f) {
    require(std::isfinite(actual) && std::isfinite(expected) &&
        std::abs(actual - expected) <= absolute + relative * std::abs(expected),
        label + ": actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected));
}

void check_rng_seed_partition() {
    for (int seed : {0, 73, -1, INT_MAX}) {
        for (int buffers : {1, 2, 4}) {
            std::set<uint64_t> action_seeds;
            for (int rank = 0; rank < 8; ++rank) {
                for (int buffer = 0; buffer < buffers; ++buffer) {
                    const uint64_t value = puf_action_seed(seed, rank, buffers, buffer);
                    require(action_seeds.insert(value).second,
                        "action RNG streams overlap across ranks or collection buffers");
                    if (rank == 0)
                        require(value == (uint64_t)(ulong)seed + buffer,
                            "single-GPU action seeds changed");
                    if (buffers == 1)
                        require(value == (uint64_t)(ulong)seed + rank,
                            "one-buffer action seeds changed");
                }
            }
            // Explicit historical collision: rank 0's second buffer and rank
            // 1's first buffer formerly both used seed + 1.
            if (buffers > 1)
                require(puf_action_seed(seed, 0, buffers, 1) != puf_action_seed(seed, 1, buffers, 0),
                    "rank/buffer action-seed collision regressed");
        }
        std::set<uint32_t> environment_seeds;
        for (int rank = 0; rank < 8; ++rank)
            for (int env = 0; env < 16; ++env)
                require(environment_seeds.insert(puf_environment_seed(seed, rank, 16, env)).second,
                    "initial environment seeds overlap across ranks");
    }
}
std::vector<obs_t> fixed_observations(int first, int count) {
    std::vector<obs_t> result(size_t(count) * OBS_SIZE);
    for (int i = 0; i < count; ++i) {
        const int n = first + i;
        decision_policy_encode("CartPole state " + std::to_string(n) +
            ": cart position " + std::to_string(n - 3) +
            ", pole angle " + std::to_string(7 - n) + ".",
            "Keep the pole upright.", {"Push left", "Push right"},
            result.data() + size_t(i) * OBS_SIZE);
    }
    return result;
}
float derivative(int row, int channel) {
    return channel == DECISION_ACTIONS ? .17f + .011f * row :
        .023f * (row + 1) * (channel + 1) - .049f;
}

// This oracle decodes packed bytes on the CPU, owns separate weights and
// activations, and evaluates the union of both ranks' samples in one batch.
void check_concatenated_oracle(PuffeRL* p, const std::vector<float>& averaged) {
    const int B = local_batch * world, K = DECISION_ACTIONS;
    const int T = decision_policy_context->bundle.max_len;
    auto* w = weights(p);
    pretrained::DecisionModel oracle(w->config, B, T, K);
    for (size_t i = 0; i < w->registry.size(); ++i)
        gpu(cudaMemcpy(oracle.parameters()[i].data, w->parameters[i].data,
            w->registry[i].count * sizeof(float), cudaMemcpyDeviceToDevice));
    auto observations = fixed_observations(0, B);
    std::vector<int> ids(B * T, decision_policy_context->bundle.pad_id), mask(B * T, 0);
    std::vector<int> markers(B * K), marker_mask(B * K, 1), qtype(B);
    for (int b = 0; b < B; ++b) {
        const auto* row = observations.data() + size_t(b) * OBS_SIZE;
        int length = row[0] | (row[1] << 8);
        require(length > 0 && length <= T, "invalid oracle token sequence");
        qtype[b] = row[DECISION_POLICY_QTYPE_OFFSET];
        for (int k = 0; k < K; ++k) {
            markers[b * K + k] = row[2 + 2 * k] | (row[3 + 2 * k] << 8);
            require(markers[b * K + k] < length, "oracle marker outside sequence");
        }
        for (int t = 0; t < length; ++t) {
            uint32_t token = 0;
            for (int byte = 0; byte < 4; ++byte)
                token |= uint32_t(row[DECISION_POLICY_HEADER_BYTES + 4 * t + byte]) << (8 * byte);
            ids[b * T + t] = token; mask[b * T + t] = 1;
        }
    }
    std::vector<float> dlogits(B * K), dvalues(B);
    for (int b = 0; b < B; ++b) {
        for (int k = 0; k < K; ++k)
            dlogits[b * K + k] = derivative(b, k) / (B * decision_policy_context->temperature);
        dvalues[b] = derivative(b, K) / B;
    }
    DeviceBuffer<int> d_ids(ids), d_mask(mask), d_markers(markers), d_marker_mask(marker_mask), d_qtype(qtype);
    DeviceBuffer<float> d_logits(dlogits), d_values(dvalues);
    oracle.forward(d_ids.data, d_mask.data, nullptr, d_markers.data,
        d_marker_mask.data, d_qtype.data, B, T, K);
    oracle.zero_grad();
    oracle.backward(d_logits.data, nullptr, d_values.data);
    gpu(cudaDeviceSynchronize());
    size_t offset = 0;
    float max_gradient = 0;
    for (size_t i = 0; i < w->registry.size(); ++i) {
        const auto expected = download(oracle.parameters()[i].grad, w->registry[i].count);
        for (size_t j = 0; j < expected.size(); ++j) {
            close_enough(averaged[offset + j], expected[j], "global batch gradient " + w->registry[i].name);
            max_gradient = std::max(max_gradient, std::abs(expected[j]));
        }
        size_t padded = numel(w->parameters[i].shape);
        for (size_t j = expected.size(); j < padded; ++j)
            require(averaged[offset + j] == 0, "gradient alignment padding changed");
        offset += padded;
    }
    require(offset == averaged.size() && max_gradient > 1e-5f, "empty global-gradient oracle");
}

void check_gradient_average(PuffeRL* p, int rank) {
    const size_t count = numel(p->grad.shape);
    const float old_temperature = decision_policy_context->temperature;
    decision_policy_context->temperature = 1.7f;
    auto observations = fixed_observations(rank * local_batch, local_batch);
    DeviceBuffer<float> input(std::vector<float>(observations.begin(), observations.end()));
    std::vector<float> gradient(local_batch * (DECISION_ACTIONS + 1));
    for (int b = 0; b < local_batch; ++b)
        for (int k = 0; k <= DECISION_ACTIONS; ++k)
            gradient[b * (DECISION_ACTIONS + 1) + k] = derivative(rank * local_batch + b, k) / local_batch;
    DeviceBuffer<float> d_gradient(gradient), all_gradients(count * world);
    decision_policy_forward(weights(p), training(p),
        Prec{.data = input.data, .shape = {local_batch, OBS_SIZE}}, p->default_stream);
    decision_policy_backward(weights(p), training(p),
        Prec{.data = d_gradient.data, .shape = {local_batch, DECISION_ACTIONS + 1}}, p->default_stream);
    gpu(cudaDeviceSynchronize());
    collective(ncclAllGather(p->grad.data, all_gradients.data, count, ncclFloat,
        p->nccl_comm, p->default_stream));
    collective(ncclAllReduce(p->grad.data, p->grad.data, count, ncclFloat, ncclAvg,
        p->nccl_comm, p->default_stream));
    gpu(cudaDeviceSynchronize());
    const auto independent = download(all_gradients.data, count * world);
    const auto averaged = download(p->grad.data, count);
    for (size_t i = 0; i < count; ++i)
        close_enough(averaged[i], .5f * (independent[i] + independent[count + i]),
            "NCCL mean differs from independently averaged gradients", 1e-7f, 1e-6f);
    if (rank == 0) check_concatenated_oracle(p, averaged);
    // Both ranks finish the oracle before reusing the training workspaces.
    collective(ncclAllReduce(p->grad.data, p->grad.data, 1, ncclFloat, ncclAvg,
        p->nccl_comm, p->default_stream));
    gpu(cudaDeviceSynchronize());
    decision_policy_context->temperature = old_temperature;
}

std::vector<float> check_replicas(PuffeRL* p, const std::string& label) {
    const size_t count = numel(p->policies[0].param.shape);
    DeviceBuffer<float> copies(count * world);
    collective(ncclAllGather(p->policies[0].param.data, copies.data, count, ncclFloat,
        p->nccl_comm, p->default_stream));
    gpu(cudaDeviceSynchronize());
    auto host = download(copies.data, count * world);
    for (size_t i = 0; i < count; ++i) {
        require(std::isfinite(host[i]) && std::isfinite(host[count + i]), label + ": nonfinite weight");
        require(host[i] == host[count + i], label + ": parameter replicas differ");
    }
    host.resize(count);
    return host;
}
void check_distinct_environments(PuffeRL* p) {
    std::vector<obs_t> observation(p->vec->observations, p->vec->observations + OBS_SIZE);
    DeviceBuffer<obs_t> source(observation), copies(size_t(OBS_SIZE) * world);
    collective(ncclAllGather(source.data, copies.data, OBS_SIZE, ncclUint8,
        p->nccl_comm, p->default_stream));
    gpu(cudaDeviceSynchronize());
    auto host = download(copies.data, size_t(OBS_SIZE) * world);
    require(!std::equal(host.begin(), host.begin() + OBS_SIZE, host.begin() + OBS_SIZE),
        "ranks started with identical CartPole states; environment seed streams overlap");
}

void check_global_metrics(PuffeRL* p, int rank) {
    require(puf_dp_log_requested(p, rank == 0), "rank-zero logging request was not shared");
    require(!puf_dp_log_requested(p, rank != 0), "non-owner rank changed the logging cadence");
    for (int fixture = 0; fixture < 2; ++fixture) {
        // Fixture 0 has unequal counts (1 and 3); fixture 1 has an empty rank 0.
        // Averaging rank-local means would give the wrong answers in both cases.
        Log log{};
        log.n = fixture == 0 ? (rank == 0 ? 1 : 3) : (rank == 0 ? 0 : 2);
        log.score = fixture == 0 ? (rank == 0 ? 2 : 24) : (rank == 0 ? 0 : 14);
        log.episode_length = fixture == 0 ? (rank == 0 ? 4 : 30) : (rank == 0 ? 0 : 18);
        p->vec->envs[0].log = log;
        std::vector<float> losses(NUM_LOSSES);
        for (int i = 0; i < LOSS_N; ++i) losses[i] = log.score * (i + 1);
        losses[LOSS_N] = log.n;
        gpu(cudaMemcpy(p->losses, losses.data(), losses.size() * sizeof(float), cudaMemcpyHostToDevice));
        for (int i = 0; i < NUM_PROF; ++i) p->profile.accum[i] = rank == 0 ? 100 + i : 300 + 2 * i;
        double elapsed = rank == 0 ? .25 : .75;
        Dict output{};
        puf_train_log(p, &output, &elapsed);
        close_enough(dict_get(&output, "env/n"), fixture == 0 ? 4 : 2, "global episode count");
        close_enough(dict_get(&output, "env/score"), fixture == 0 ? 6.5f : 7, "weighted global score");
        close_enough(dict_get(&output, "env/episode_length"), fixture == 0 ? 8.5f : 9,
            "weighted global episode length");
        for (int i = 0; i < LOSS_N; ++i)
            close_enough(dict_get(&output, LOSS_NAMES[i]), (fixture == 0 ? 6.5f : 7) * (i + 1),
                "weighted global PPO loss");
        require(elapsed == .75, "global logging elapsed time is not the slowest rank");
        for (int i = 0; i < NUM_PROF; ++i)
            require(p->profile.accum[i] == 300 + 2 * i, "global profile missed the slowest rank");
        require(p->vec->envs[0].log.n == 0, "global logging did not clear local episode counters");
        auto cleared = download(p->losses, NUM_LOSSES);
        for (float value : cleared) require(value == 0, "global logging did not clear local PPO counters");
        dict_clear(&output);
    }
    memset(p->profile.accum, 0, sizeof(p->profile.accum));
}

template<class T> uint64_t hash_device(uint64_t hash, const T* source, size_t count) {
    auto values = download(source, count);
    const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
    for (size_t i = 0; i < values.size() * sizeof(T); ++i) {
        hash ^= bytes[i]; hash *= UINT64_C(1099511628211);
    }
    return hash;
}
uint64_t rollout_hash(PuffeRL* p) {
    uint64_t hash = UINT64_C(14695981039346656037);
    hash = hash_device(hash, p->rollouts.observations.data, numel(p->rollouts.observations.shape));
    hash = hash_device(hash, p->rollouts.actions.data, numel(p->rollouts.actions.shape));
    // Timeout-adjusted rewards include V(final_state), so low-order numerical
    // differences between valid optimizer modes may change those float bits.
    // CartPole's raw rewards are determined by these same states/actions/dones.
    return hash_device(hash, p->rollouts.terminals.data, numel(p->rollouts.terminals.shape));
}
std::vector<float> predictions(PuffeRL* p, int rank) {
    auto observations = fixed_observations(rank * local_batch, local_batch);
    DeviceBuffer<float> input(std::vector<float>(observations.begin(), observations.end()));
    Prec output = decision_policy_forward(weights(p), training(p),
        Prec{.data = input.data, .shape = {local_batch, OBS_SIZE}}, p->default_stream);
    gpu(cudaDeviceSynchronize());
    return download(output.data, local_batch * (DECISION_ACTIONS + 1));
}
void check_checkpoint(PuffeRL* p, int rank, const std::string& path) {
    const auto expected_weights = download(p->policies[0].param.data, numel(p->policies[0].param.shape));
    const auto expected_predictions = predictions(p, rank);
    puf_save_weights(p, path.c_str());
    float changed = expected_weights[0] + 1;
    gpu(cudaMemcpy(p->policies[0].param.data, &changed, sizeof(changed), cudaMemcpyHostToDevice));
    pufferl_load_policy(p, 0, path.c_str());
    require(download(p->policies[0].param.data, expected_weights.size()) == expected_weights,
        "checkpoint did not restore the complete parameter buffer");
    require(predictions(p, rank) == expected_predictions, "checkpoint reload changed predictions");
}

struct Report {
    uint64_t trajectory[updates]{};
    uint64_t parameters = 0;
    uint64_t momentum = 0;
    uint64_t scratch = 0;
    uint64_t owned_parameters = 0;
};
std::string prefix(const std::string& directory, bool distributed, int rank) {
    return directory + (distributed ? "/distributed" : "/replicated") + "-rank" + std::to_string(rank);
}
template<class T> void save_data(const std::string& path, const T* data, size_t count) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    require(bool(file), "cannot open " + path);
    file.write(reinterpret_cast<const char*>(data), count * sizeof(T));
    require(bool(file), "cannot write " + path);
}
template<class T> std::vector<T> load_data(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(bool(file), "cannot open " + path);
    auto bytes = file.tellg();
    require(bytes > 0 && size_t(bytes) % sizeof(T) == 0, "invalid artifact " + path);
    std::vector<T> result(size_t(bytes) / sizeof(T));
    file.seekg(0); file.read(reinterpret_cast<char*>(result.data()), bytes);
    require(bool(file), "cannot read " + path);
    return result;
}

int run_rank(const std::string& bundle, const std::string& directory,
        bool distributed, TrainContext* context) {
    const int rank = context->rank;
    PuffeRL* p = nullptr;
    Ini ini{};
    try {
        int devices = 0;
        gpu(cudaGetDeviceCount(&devices));
        require(devices >= world, "this test requires two visible CUDA GPUs; no test was skipped");
        puf_ini_load_env(&ini, PUFFER_ENV_NAME, 0, nullptr);
        const char* overrides[][2] = {
            {"base.async", "0"}, {"base.cudagraphs", "-1"}, {"base.seed", "73"},
            {"vec.total_agents", "1"}, {"vec.num_buffers", "1"}, {"vec.num_threads", "1"},
            {"env.max_steps", "3"}, {"train.horizon", "4"}, {"train.minibatch_size", "4"},
            {"train.total_timesteps", "24"}, {"train.gpus", "2"}, {"train.gamma", "0.99"},
            {"train.anneal_lr", "0"}, {"train.anneal_ent_coef", "0"},
            {"train.learning_rate", "0.0001"}, {"train.replay_ratio", "1"},
        };
        for (const auto& item : overrides) puf_ini_put(&ini, item[0], item[1]);
        puf_ini_put(&ini, "policy.bundle", bundle.c_str());
        puf_ini_put(&ini, "train.distributed_optimizer", distributed ? "1" : "0");
        p = create_pufferl(&ini, context);
        gpu(cudaDeviceSynchronize());
        const size_t count = numel(p->policies[0].param.shape);
        require(count < 5000000, "use the small BERT fixture, not full Laya, for independent gradient oracles");
        require(p->muon.distributed_optimizer == distributed, "distributed optimizer option was not applied");
        require(size_t(p->muon.replicated_momentum_elems) == count, "optimizer parameter count mismatch");
        if (distributed) require(size_t(p->muon.momentum_elems) < count, "momentum was not sharded");
        else require(size_t(p->muon.momentum_elems) == count, "replicated optimizer lost momentum entries");

        auto initial = check_replicas(p, "initialization");
        check_distinct_environments(p);
        check_global_metrics(p, rank);
        check_gradient_average(p, rank);
        Report report{};
        report.parameters = count;
        report.momentum = p->muon.momentum_elems;
        report.scratch = p->muon.scratch_elems;
        report.owned_parameters = p->muon.owned_parameter_count;
        for (int update = 0; update < updates; ++update) {
            rollouts(p);
            report.trajectory[update] = rollout_hash(p);
            train_impl(p, nullptr);
            gpu(cudaDeviceSynchronize());
            auto current = check_replicas(p, "PPO update " + std::to_string(update + 1));
            require(current != initial, "PPO did not update model parameters");
            auto loss = download(p->losses, NUM_LOSSES);
            for (float value : loss) require(std::isfinite(value), "PPO produced a nonfinite loss");
            require(loss[LOSS_N] > 0, "PPO loss counter did not advance");
        }
        require(p->global_step == local_batch * updates && p->epoch == updates,
            "actual collector/PPO update count differs from test fixture");
        const auto stem = prefix(directory, distributed, rank);
        check_checkpoint(p, rank, stem + ".bin");
        check_replicas(p, "checkpoint reload");
        save_data(stem + ".report", &report, 1);
        printf("DISTRIBUTED_TEST rank=%d optimizer=%s parameters=%zu momentum_elements=%llu "
               "scratch_elements=%llu owned_tensors=%llu updates=%d passed\n", rank,
            distributed ? "distributed" : "replicated", count,
            (unsigned long long)report.momentum, (unsigned long long)report.scratch,
            (unsigned long long)report.owned_parameters, updates);
        // Explicit teardown is safe here: CUDA graphs are disabled and all
        // ranks completed the final collective. Normal training owns its exit.
        close_pufferl(p);
        collective(ncclCommDestroy(p->nccl_comm));
        p = nullptr;
        puf_ini_free(&ini);
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "Distributed decision test rank %d (%s): %s\n", rank,
            distributed ? "distributed" : "replicated", error.what());
        if (p && p->nccl_comm) ncclCommAbort(p->nccl_comm);
        return 1;
    }
}

void run_trial(const std::string& bundle, const std::string& directory, bool distributed) {
    struct Task { const std::string& bundle; const std::string& directory; bool distributed; };
    Task task{bundle, directory, distributed};
    auto worker = [](TrainContext* context, void*, void* user) -> int {
        auto& task = *static_cast<Task*>(user);
        return run_rank(task.bundle, task.directory, task.distributed, context);
    };
    require(puf_launch_native_ranks(world, 0, 0, nullptr, worker, &task) == 0,
        "a distributed test worker failed");
}

void check_launcher_failure() {
    printf("DISTRIBUTED_TEST exercising expected rank failure and peer cleanup\n");
    auto worker = [](TrainContext* context, void*, void*) -> int {
        if (context->rank == 1) return 1;
        // The supervisor should kill this blocked peer promptly. A broken
        // supervisor eventually returns, so this check fails instead of hanging.
        poll(nullptr, 0, 10000);
        return 0;
    };
    double begin = wall_clock();
    require(puf_launch_native_ranks(world, 0, 0, nullptr, worker, nullptr) != 0,
        "launcher reported success after a failed rank");
    require(wall_clock() - begin < 5, "launcher did not promptly terminate a blocked peer");
}

void compare_trials(const std::string& directory) {
    uint64_t total_momentum = 0, parameter_count = 0;
    float max_difference = 0;
    for (int rank = 0; rank < world; ++rank) {
        auto replicated = load_data<float>(prefix(directory, false, rank) + ".bin");
        auto distributed = load_data<float>(prefix(directory, true, rank) + ".bin");
        auto normal_report = load_data<Report>(prefix(directory, false, rank) + ".report");
        auto sharded_report = load_data<Report>(prefix(directory, true, rank) + ".report");
        require(normal_report.size() == 1 && sharded_report.size() == 1, "invalid rank report");
        require(replicated.size() == distributed.size(), "optimizer modes produced different parameter counts");
        for (size_t i = 0; i < replicated.size(); ++i) {
            close_enough(distributed[i], replicated[i], "distributed/replicated PPO weight parity", 3e-6f, 3e-5f);
            max_difference = std::max(max_difference, std::abs(distributed[i] - replicated[i]));
        }
        for (int i = 0; i < updates; ++i)
            require(normal_report[0].trajectory[i] == sharded_report[0].trajectory[i],
                "optimizer trials collected different observations/actions/terminals");
        total_momentum += sharded_report[0].momentum;
        parameter_count = sharded_report[0].parameters;
    }
    require(total_momentum == parameter_count, "distributed ranks do not own exactly one complete momentum buffer");
    printf("DISTRIBUTED_TEST passed: two GPUs, independent concatenated-batch gradient oracle, "
           "rank-distinct environments, global weighted metrics, three real PPO updates per optimizer, identical rank replicas, "
           "checkpoint reload, momentum ownership and optimizer parity (max_abs_difference=%.9g). "
           "Artifacts: %s\n", max_difference, directory.c_str());
}
} // namespace

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);
    try {
        check_rng_seed_partition();
        if (argc == 2 && strcmp(argv[1], "--host") == 0) {
            puts("Distributed seed partition: PASS (distinct rank/buffer streams and legacy seed compatibility)");
            return 0;
        }
        require(argc == 2 || argc == 3,
            "usage: test_decision_distributed SMALL_BERT_BUNDLE [OUTPUT_DIRECTORY] (requires two GPUs), or --host");
        char temporary[] = "/tmp/puffer-decision-distributed.XXXXXX";
        std::string directory;
        if (argc == 3) { directory = argv[2]; mkdir_p(directory.c_str()); }
        else {
            require(mkdtemp(temporary) != nullptr, "cannot create test artifact directory");
            directory = temporary;
        }
        run_trial(argv[1], directory, false);
        run_trial(argv[1], directory, true);
        check_launcher_failure();
        compare_trials(directory);
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "Distributed decision integration: %s\n", error.what());
        return 1;
    }
}
