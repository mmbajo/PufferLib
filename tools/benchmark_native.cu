// Time the actual synchronous collector and PPO/Muon learner after warmup.
// No dashboard, evaluation, or checkpoint writes occur inside the timed loop.
#undef PUFFERLIB_BUILD_MAIN
#include "../src/pufferl.cu"
#include "../src/native_distributed.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <vector>

static void benchmark_cuda(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
static void benchmark_nccl(ncclResult_t error) {
    if (error != ncclSuccess) throw std::runtime_error(ncclGetErrorString(error));
}
static void benchmark_sync() { benchmark_cuda(cudaDeviceSynchronize()); }

struct BenchmarkOptions {
    Ini* ini;
    double seconds = 10;
    int warmup = 3;
    int updates = 0;
    bool check_replicas = false;
};

struct BenchmarkResult {
    char gpu[256];
    int rank, gpu_id, agents, buffers, threads, horizon, minibatch;
    int warmup_updates, updates, padded_tokens;
    bool cuda_graphs, distributed_optimizer;
    long long params, steps, token_sum, token_count;
    long long optimizer_momentum_bytes, replicated_optimizer_momentum_bytes;
    long long optimizer_scratch_elements;
    int token_min, token_max;
    double replay_ratio, startup, warmup, elapsed, rollout, learner, coordination;
    double replica_check, device_used_gib, profile[NUM_PROF];
};

// Bound temporary memory independently of model size. Compare every FP32 master
// parameter byte (including padding) against rank 0, and verify finiteness. The
// collective result makes every rank fail consistently after completing checks.
static void benchmark_check_replicas(PuffeRL* p, int* control) {
    const size_t chunk = 1024 * 1024;
    Float weights = p->policies[0].master_weights;
    std::vector<float> local(chunk), reference(chunk);
    float* broadcast = nullptr;
    benchmark_cuda(cudaMalloc(reinterpret_cast<void**>(&broadcast), chunk * sizeof(float)));
    int valid = 1;
    for (long long offset = 0, size = numel(weights.shape); offset < size; offset += chunk) {
        size_t count = std::min<long long>(chunk, size - offset);
        benchmark_cuda(cudaMemcpy(local.data(), weights.data + offset,
            count * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) if (!std::isfinite(local[i])) valid = 0;
        if (p->hypers.world_size > 1) {
            benchmark_nccl(ncclBroadcast(weights.data + offset, broadcast, count,
                ncclFloat, 0, p->nccl_comm, p->train_stream));
            benchmark_cuda(cudaStreamSynchronize(p->train_stream));
            benchmark_cuda(cudaMemcpy(reference.data(), broadcast,
                count * sizeof(float), cudaMemcpyDeviceToHost));
            if (memcmp(local.data(), reference.data(), count * sizeof(float))) valid = 0;
        }
    }
    benchmark_cuda(cudaFree(broadcast));
    benchmark_cuda(cudaMemcpy(control, &valid, sizeof(valid), cudaMemcpyHostToDevice));
    if (p->hypers.world_size > 1)
        benchmark_nccl(ncclAllReduce(control, control, 1, ncclInt, ncclMin,
            p->nccl_comm, p->train_stream));
    benchmark_cuda(cudaStreamSynchronize(p->train_stream));
    benchmark_cuda(cudaMemcpy(&valid, control, sizeof(valid), cudaMemcpyDeviceToHost));
    if (!valid) throw std::runtime_error("replica master weights differ or contain nonfinite values");
}

static int benchmark_rank(TrainContext* context, void* destination, void* user) {
    auto& options = *static_cast<BenchmarkOptions*>(user);
    auto& result = *static_cast<BenchmarkResult*>(destination);
    Ini* ini = options.ini;
    try {
        double begin = wall_clock();
        PuffeRL* p = create_pufferl(ini, context);
        int* control = nullptr;
        benchmark_cuda(cudaMalloc(reinterpret_cast<void**>(&control), sizeof(int)));
        benchmark_sync();
        result.startup = wall_clock() - begin;
        result.rank = context->rank;
        result.gpu_id = context->gpu_id;
        if (options.check_replicas) {
            begin = wall_clock(); benchmark_check_replicas(p, control);
            result.replica_check += wall_clock() - begin;
        }
        begin = wall_clock();
        for (int i = 0; i < options.warmup; ++i) { rollouts(p); train_impl(p, nullptr); }
        benchmark_sync();
        result.warmup = wall_clock() - begin;
        result.warmup_updates = options.warmup;
        memset(p->profile.accum, 0, sizeof(p->profile.accum));
        benchmark_cuda(cudaMemset(p->losses, 0, NUM_LOSSES * sizeof(float)));
        benchmark_cuda(cudaMemset(control, 0, sizeof(int)));
        // Start all timers only after every rank has completed startup/warmup.
        if (context->world_size > 1)
            benchmark_nccl(ncclAllReduce(control, control, 1, ncclInt, ncclSum,
                p->nccl_comm, p->train_stream));
        benchmark_sync();
        long initial_steps = p->global_step;
        result.token_min = INT_MAX;
        begin = wall_clock();
        while (true) {
            double tick = wall_clock();
            rollouts(p);
            double middle = wall_clock();
            train_impl(p, nullptr);
            double end = wall_clock();
            result.rollout += middle - tick;
            result.learner += end - middle;
            ++result.updates;
#ifdef PUFFER_DECISION_POLICY
            // Samples current observations once per rollout, not every decision.
            for (int i = 0; i < p->hypers.total_agents; ++i) {
                auto* obs = p->vec->observations + size_t(i) * OBS_SIZE;
                int length = obs[0] | (obs[1] << 8);
                result.token_min = std::min(result.token_min, length);
                result.token_max = std::max(result.token_max, length);
                result.token_sum += length; ++result.token_count;
            }
#endif
            if (options.updates) {
                if (result.updates >= options.updates) break;
            } else if (context->world_size == 1) {
                if (wall_clock() - begin >= options.seconds) break;
            } else {
                // Only rank 0 decides when a duration trial ends. Independent
                // clocks/loop conditions can leave peers hung in gradient NCCL.
                double coordinate_begin = wall_clock();
                int stop = context->rank == 0 && coordinate_begin - begin >= options.seconds;
                benchmark_cuda(cudaMemcpyAsync(control, &stop, sizeof(stop),
                    cudaMemcpyHostToDevice, p->train_stream));
                benchmark_nccl(ncclBroadcast(control, control, 1, ncclInt,
                    0, p->nccl_comm, p->train_stream));
                benchmark_cuda(cudaMemcpyAsync(&stop, control, sizeof(stop),
                    cudaMemcpyDeviceToHost, p->train_stream));
                benchmark_cuda(cudaStreamSynchronize(p->train_stream));
                result.coordination += wall_clock() - coordinate_begin;
                if (stop) break;
            }
        }
        benchmark_sync();
        result.elapsed = wall_clock() - begin;
        result.steps = p->global_step - initial_steps;
        float losses[NUM_LOSSES];
        benchmark_cuda(cudaMemcpy(losses, p->losses, sizeof(losses), cudaMemcpyDeviceToHost));
        for (float loss : losses)
            if (!std::isfinite(loss)) throw std::runtime_error("nonfinite training loss");
        if (losses[LOSS_N] <= 0 || result.steps <= 0)
            throw std::runtime_error("benchmark did not execute PPO updates");
        cudaDeviceProp device{};
        benchmark_cuda(cudaGetDeviceProperties(&device, context->gpu_id));
        snprintf(result.gpu, sizeof(result.gpu), "%s", device.name);
        size_t free_bytes = 0, total_bytes = 0;
        benchmark_cuda(cudaMemGetInfo(&free_bytes, &total_bytes));
        result.device_used_gib = (total_bytes - free_bytes) / (1024.0 * 1024 * 1024);
        result.params = numel(p->policies[0].param.shape);
        result.agents = p->hypers.total_agents;
        result.buffers = p->hypers.num_buffers;
        result.threads = puf_ini_get(ini, "vec", "num_threads");
        result.horizon = p->hypers.horizon;
        result.minibatch = p->hypers.minibatch_size;
        result.replay_ratio = p->hypers.replay_ratio;
        result.cuda_graphs = p->hypers.cudagraphs;
        result.distributed_optimizer = p->muon.distributed_optimizer;
        result.optimizer_momentum_bytes = p->muon.momentum_elems * sizeof(float);
        result.replicated_optimizer_momentum_bytes = p->muon.replicated_momentum_elems * sizeof(float);
        result.optimizer_scratch_elements = p->muon.scratch_elems;
        for (int i = 0; i < NUM_PROF; ++i) result.profile[i] = p->profile.accum[i] / 1000.0;
#ifdef PUFFER_DECISION_POLICY
        result.padded_tokens = decision_policy_context->bundle.max_len;
#endif
        if (options.check_replicas) {
            begin = wall_clock(); benchmark_check_replicas(p, control);
            result.replica_check += wall_clock() - begin;
        }
        benchmark_cuda(cudaFree(control));
        close_pufferl(p);
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "Native training benchmark rank %d: %s\n", context->rank, error.what());
        // The parent terminates blocked peers; cleanup must not enter NCCL here.
        return 1;
    }
}

static void benchmark_print(const std::vector<BenchmarkResult>& ranks, bool check_replicas,
        bool fixed_updates) {
    BenchmarkResult aggregate = ranks[0];
    const auto& first = ranks[0];
    int gpus = ranks.size();
    long long total_momentum = first.optimizer_momentum_bytes;
    long long max_momentum = first.optimizer_momentum_bytes;
    for (int rank = 1; rank < gpus; ++rank) {
        const auto& current = ranks[rank];
        if (current.updates != first.updates || current.params != first.params)
            throw std::runtime_error("inconsistent benchmark rank results");
        aggregate.steps += current.steps;
        aggregate.token_sum += current.token_sum;
        aggregate.token_count += current.token_count;
        aggregate.token_min = std::min(aggregate.token_min, current.token_min);
        aggregate.token_max = std::max(aggregate.token_max, current.token_max);
        aggregate.startup = std::max(aggregate.startup, current.startup);
        aggregate.warmup = std::max(aggregate.warmup, current.warmup);
        aggregate.elapsed = std::max(aggregate.elapsed, current.elapsed);
        aggregate.rollout = std::max(aggregate.rollout, current.rollout);
        aggregate.learner = std::max(aggregate.learner, current.learner);
        aggregate.coordination = std::max(aggregate.coordination, current.coordination);
        aggregate.replica_check = std::max(aggregate.replica_check, current.replica_check);
        aggregate.device_used_gib = std::max(aggregate.device_used_gib, current.device_used_gib);
        total_momentum += current.optimizer_momentum_bytes;
        max_momentum = std::max(max_momentum, current.optimizer_momentum_bytes);
        for (int i = 0; i < NUM_PROF; ++i)
            aggregate.profile[i] = std::max(aggregate.profile[i], current.profile[i]);
    }
    printf("BENCHMARK {\"env\":\"%s\",\"gpu\":\"%s\",\"precision\":\"%s\","
           "\"params\":%lld,\"agents\":%d,\"buffers\":%d,\"threads\":%d,"
           "\"horizon\":%d,\"minibatch\":%d,\"replay_ratio\":%.8g,\"cuda_graphs\":%s,"
           "\"gpus\":%d,\"global_agents\":%lld,\"global_rollout_batch\":%lld,"
           "\"global_minibatch\":%lld,\"distributed_optimizer\":%s,\"fixed_updates\":%s,"
           "\"startup_seconds\":%.6f,\"warmup_updates\":%d,\"warmup_seconds\":%.6f,"
           "\"updates\":%d,\"steps\":%lld,\"global_steps\":%lld,\"seconds\":%.6f,"
           "\"steps_per_second\":%.6f,\"rollout_wall_seconds\":%.6f,\"learner_wall_seconds\":%.6f,"
           "\"coordination_seconds\":%.6f,\"device_used_gib\":%.6f,\"finite_losses\":true,"
           "\"replicas_checked\":%s,\"replica_check_seconds\":%.6f,"
           "\"optimizer_momentum_bytes_max_rank\":%lld,\"optimizer_momentum_bytes_total\":%lld,"
           "\"replicated_optimizer_momentum_bytes_per_rank\":%lld",
           PUFFER_ENV_NAME, first.gpu, USE_BF16 ? "bf16" : "fp32", first.params,
           first.agents, first.buffers, first.threads, first.horizon, first.minibatch,
           first.replay_ratio, first.cuda_graphs ? "true" : "false", gpus,
           1LL * first.agents * gpus, 1LL * first.agents * first.horizon * gpus,
           1LL * first.minibatch * gpus, first.distributed_optimizer ? "true" : "false",
           fixed_updates ? "true" : "false", aggregate.startup, first.warmup_updates,
           aggregate.warmup, first.updates, aggregate.steps, aggregate.steps, aggregate.elapsed,
           aggregate.steps / aggregate.elapsed, aggregate.rollout, aggregate.learner,
           aggregate.coordination, aggregate.device_used_gib, check_replicas ? "true" : "false",
           aggregate.replica_check, max_momentum, total_momentum,
           first.replicated_optimizer_momentum_bytes);
    if (check_replicas) printf(",\"replicas_equal\":true,\"finite_master_weights\":true");
    for (int i = 0; i < NUM_PROF; ++i)
        printf(",\"profile_%s_seconds\":%.6f", PROF_NAMES[i], aggregate.profile[i]);
    if (aggregate.token_count)
        printf(",\"sampled_tokens_min\":%d,\"sampled_tokens_mean\":%.3f,\"sampled_tokens_max\":%d",
            aggregate.token_min, double(aggregate.token_sum) / aggregate.token_count, aggregate.token_max);
    if (first.padded_tokens) printf(",\"padded_tokens\":%d", first.padded_tokens);
    printf(",\"ranks\":[");
    for (int rank = 0; rank < gpus; ++rank) {
        const auto& current = ranks[rank];
        printf("%s{\"rank\":%d,\"gpu_id\":%d,\"steps\":%lld,\"seconds\":%.6f,"
               "\"startup_seconds\":%.6f,\"warmup_seconds\":%.6f,"
               "\"rollout_wall_seconds\":%.6f,\"learner_wall_seconds\":%.6f,"
               "\"coordination_seconds\":%.6f,\"device_used_gib\":%.6f,"
               "\"optimizer_momentum_bytes\":%lld,\"optimizer_scratch_elements\":%lld}",
               rank ? "," : "", current.rank, current.gpu_id, current.steps, current.elapsed,
               current.startup, current.warmup, current.rollout, current.learner,
               current.coordination, current.device_used_gib, current.optimizer_momentum_bytes,
               current.optimizer_scratch_elements);
    }
    printf("]}\n");
}

static int benchmark_integer(const char* value) {
    size_t consumed = 0;
    int parsed = std::stoi(value, &consumed);
    if (value[consumed]) throw std::invalid_argument("invalid benchmark integer");
    return parsed;
}

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);
    Ini ini{};
    BenchmarkOptions options{};
    options.ini = &ini;
    bool explicit_seconds = false;
    std::vector<char*> overrides;
    try {
        for (int i = 1; i < argc; ++i) {
            if (!strncmp(argv[i], "--seconds=", 10)) {
                size_t consumed = 0;
                options.seconds = std::stod(argv[i] + 10, &consumed);
                if (argv[i][10 + consumed]) throw std::invalid_argument("invalid --seconds");
                explicit_seconds = true;
            } else if (!strncmp(argv[i], "--warmup=", 9)) options.warmup = benchmark_integer(argv[i] + 9);
            else if (!strncmp(argv[i], "--updates=", 10)) {
                options.updates = benchmark_integer(argv[i] + 10);
                if (options.updates < 1) throw std::invalid_argument("require --updates >= 1");
            } else if (!strncmp(argv[i], "--check-replicas=", 17)) {
                int value = benchmark_integer(argv[i] + 17);
                if (value != 0 && value != 1) throw std::invalid_argument("--check-replicas must be 0 or 1");
                options.check_replicas = value;
            } else overrides.push_back(argv[i]);
        }
        if (!std::isfinite(options.seconds) || options.seconds <= 0 || options.warmup < 1)
            throw std::invalid_argument("require --seconds > 0 and --warmup >= 1");
        if (explicit_seconds && options.updates)
            throw std::invalid_argument("choose either --seconds or --updates");
        puf_ini_load_env(&ini, PUFFER_ENV_NAME, overrides.size(), overrides.data());
        // Measure the same sequential training path for every policy. Duration
        // trials do not have a predetermined schedule endpoint.
        puf_ini_put(&ini, "base.async", "0");
        puf_ini_put(&ini, "train.anneal_lr", "0");
        puf_ini_put(&ini, "train.anneal_ent_coef", "0");
        if (puf_ini_get(&ini, "vec", "num_policies") != 1 || puf_ini_get(&ini, "selfplay", "enabled") != 0)
            throw std::invalid_argument("benchmark requires one policy and no self-play");
        int gpus = puf_ini_get(&ini, "train", "gpus");
        int offset = puf_ini_get(&ini, "base", "gpu_offset");
        int mb = puf_ini_get(&ini, "train", "minibatch_size");
        int horizon = puf_ini_get(&ini, "train", "horizon");
        int agents = puf_ini_get(&ini, "vec", "total_agents");
        int buffers = puf_ini_get(&ini, "vec", "num_buffers");
        if (gpus < 1 || offset < 0 || agents < 1 || buffers < 1 || agents % buffers ||
                horizon < 1 || horizon % ADV_VEC_WIDTH || mb < 1 || mb % horizon ||
                static_cast<long long>(mb) > static_cast<long long>(horizon) * agents || agents % (mb / horizon))
            throw std::invalid_argument("invalid per-rank GPU, rollout or minibatch configuration");
        std::vector<BenchmarkResult> results(gpus);
        if (puf_launch_native_ranks(gpus, offset, sizeof(BenchmarkResult), results.data(), benchmark_rank, &options))
            throw std::runtime_error("distributed benchmark failed; no throughput result emitted");
        benchmark_print(results, options.check_replicas, options.updates != 0);
        puf_ini_free(&ini);
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "Native training benchmark: %s\n", error.what());
        puf_ini_free(&ini);
        return 1;
    }
}
