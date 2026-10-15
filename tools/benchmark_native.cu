// Time the actual synchronous collector and PPO/Muon learner after warmup.
// No dashboard, evaluation, or checkpoint writes occur inside the timed loop.
#undef PUFFERLIB_BUILD_MAIN
#include "../src/pufferl.cu"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

static void benchmark_sync() {
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);
    double seconds = 10;
    int warmup = 3;
    std::vector<char*> overrides;
    Ini ini{};
    PuffeRL* p = nullptr;
    try {
        for (int i = 1; i < argc; ++i) {
            if (!strncmp(argv[i], "--seconds=", 10)) seconds = std::stod(argv[i] + 10);
            else if (!strncmp(argv[i], "--warmup=", 9)) warmup = std::stoi(argv[i] + 9);
            else overrides.push_back(argv[i]);
        }
        if (!std::isfinite(seconds) || seconds <= 0 || warmup < 1)
            throw std::invalid_argument("require --seconds > 0 and --warmup >= 1");
        puf_ini_load_env(&ini, PUFFER_ENV_NAME, overrides.size(), overrides.data());
        // Measure the same sequential training path for every policy.
        puf_ini_put(&ini, "base.async", "0");
        puf_ini_put(&ini, "train.anneal_lr", "0");
        puf_ini_put(&ini, "train.anneal_ent_coef", "0");
        if (puf_ini_get(&ini, "vec", "num_policies") != 1 ||
                puf_ini_get(&ini, "selfplay", "enabled") != 0 ||
                puf_ini_get(&ini, "train", "gpus") != 1)
            throw std::invalid_argument("benchmark supports one GPU and one policy without self-play");
        TrainContext context{}; context.world_size = 1;
        context.gpu_id = puf_ini_get(&ini, "base", "gpu_offset");
        double begin = wall_clock();
        p = create_pufferl(&ini, &context);
        benchmark_sync();
        double startup = wall_clock() - begin;
        begin = wall_clock();
        for (int i = 0; i < warmup; ++i) { rollouts(p); train_impl(p, nullptr); }
        benchmark_sync();
        double warmup_seconds = wall_clock() - begin;
        memset(p->profile.accum, 0, sizeof(p->profile.accum));
        cudaMemset(p->losses, 0, NUM_LOSSES * sizeof(float));
        benchmark_sync();
        long initial_steps = p->global_step;
        int updates = 0;
        double rollout_wall = 0, learner_wall = 0;
        size_t token_sum = 0, token_count = 0;
        int token_min = 2147483647, token_max = 0;
        begin = wall_clock();
        do {
            double tick = wall_clock();
            rollouts(p);
            double middle = wall_clock();
            train_impl(p, nullptr);
            double end = wall_clock();
            rollout_wall += middle - tick;
            learner_wall += end - middle;
            ++updates;
#ifdef PUFFER_DECISION_POLICY
            // Samples current observations once per rollout, not every decision.
            for (int i = 0; i < p->hypers.total_agents; ++i) {
                auto* obs = p->vec->observations + size_t(i) * OBS_SIZE;
                int length = obs[0] | (obs[1] << 8);
                token_min = std::min(token_min, length);
                token_max = std::max(token_max, length);
                token_sum += length; ++token_count;
            }
#endif
        } while (wall_clock() - begin < seconds);
        benchmark_sync();
        double elapsed = wall_clock() - begin;
        long steps = p->global_step - initial_steps;
        float losses[NUM_LOSSES];
        cudaMemcpy(losses, p->losses, sizeof(losses), cudaMemcpyDeviceToHost);
        for (float loss : losses)
            if (!std::isfinite(loss)) throw std::runtime_error("nonfinite training loss");
        if (losses[LOSS_N] <= 0 || steps <= 0)
            throw std::runtime_error("benchmark did not execute PPO updates");
        int device_id = 0; cudaGetDevice(&device_id);
        cudaDeviceProp device{}; cudaGetDeviceProperties(&device, device_id);
        size_t free_bytes = 0, total_bytes = 0; cudaMemGetInfo(&free_bytes, &total_bytes);
        printf("BENCHMARK {\"env\":\"%s\",\"gpu\":\"%s\",\"precision\":\"%s\","
               "\"params\":%ld,\"agents\":%d,\"buffers\":%d,\"threads\":%d,"
               "\"horizon\":%d,\"minibatch\":%d,\"replay_ratio\":%.8g,\"cuda_graphs\":%s,"
               "\"startup_seconds\":%.6f,\"warmup_updates\":%d,\"warmup_seconds\":%.6f,"
               "\"updates\":%d,\"steps\":%ld,\"seconds\":%.6f,\"steps_per_second\":%.6f,"
               "\"rollout_wall_seconds\":%.6f,\"learner_wall_seconds\":%.6f,"
               "\"device_used_gib\":%.6f,\"finite_losses\":true",
               PUFFER_ENV_NAME, device.name, USE_BF16 ? "bf16" : "fp32",
               numel(p->policies[0].param.shape), p->hypers.total_agents,
               p->hypers.num_buffers, (int)puf_ini_get(&ini, "vec", "num_threads"),
               p->hypers.horizon, p->hypers.minibatch_size, (double)p->hypers.replay_ratio,
               p->hypers.cudagraphs ? "true" : "false", startup, warmup, warmup_seconds,
               updates, steps, elapsed, steps / elapsed, rollout_wall, learner_wall,
               (total_bytes - free_bytes) / (1024.0 * 1024 * 1024));
        for (int i = 0; i < NUM_PROF; ++i)
            printf(",\"profile_%s_seconds\":%.6f", PROF_NAMES[i], p->profile.accum[i] / 1000.0);
        if (token_count) printf(",\"sampled_tokens_min\":%d,\"sampled_tokens_mean\":%.3f,"
                               "\"sampled_tokens_max\":%d", token_min,
                               double(token_sum) / token_count, token_max);
#ifdef PUFFER_DECISION_POLICY
        printf(",\"padded_tokens\":%d", decision_policy_context->bundle.max_len);
#endif
        printf("}\n");
        close_pufferl(p); p = nullptr;
        puf_ini_free(&ini);
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "Native training benchmark: %s\n", error.what());
        if (p) close_pufferl(p);
        puf_ini_free(&ini);
        return 1;
    }
}
