// Evaluation-only CartPole runner: exact native engine, decision adapter and
// Puffer categorical sampler; independent, reproducible RNG seeds per episode.
#undef PUFFERLIB_BUILD_MAIN
#include "../src/pufferl.cu"
#include <algorithm>
#include <climits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef PUFFER_DECISION_CARTPOLE
#error "build for decision_cartpole"
#endif

static void eval_cuda(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
static uint64_t eval_integer(const char* value) {
    if (!*value || *value == '-') throw std::invalid_argument("expected unsigned integer");
    size_t used = 0; uint64_t result = std::stoull(value, &used);
    if (value[used]) throw std::invalid_argument("invalid integer suffix");
    return result;
}
static uint32_t eval_environment_seed(uint32_t base, uint32_t episode) {
    return base + episode;
}
static std::string eval_json(const std::string& value) {
    cJSON* node = cJSON_CreateString(value.c_str());
    char* encoded = cJSON_PrintUnformatted(node);
    if (!encoded) throw std::runtime_error("JSON allocation failed");
    std::string result(encoded); free(encoded); cJSON_Delete(node); return result;
}
__global__ static void eval_seed_action(curandStatePhilox4_32_10_t* states,
        int slot, unsigned long long seed) {
    if (threadIdx.x == 0) curand_init(seed, 0, 0, states + slot);
}

static void eval_load_weights(const std::string& path, float* weights, int64_t elements) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != std::streamoff(elements * sizeof(float)))
        throw std::runtime_error("checkpoint byte count differs from exact native parameter layout");
    input.seekg(0);
    std::vector<float> chunk(1024 * 1024);
    for (int64_t offset = 0; offset < elements; offset += chunk.size()) {
        size_t count = std::min<int64_t>(chunk.size(), elements - offset);
        if (!input.read(reinterpret_cast<char*>(chunk.data()), count * sizeof(float)))
            throw std::runtime_error("checkpoint read failed");
        for (size_t i = 0; i < count; ++i)
            if (!std::isfinite(chunk[i])) throw std::runtime_error("nonfinite checkpoint weight");
        eval_cuda(cudaMemcpy(weights + offset, chunk.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    }
}

struct EvalEpisode {
    int id = -1, steps = 0, left_actions = 0, right_actions = 0;
    uint32_t env_seed = 0;
    uint64_t action_seed = 0;
    float initial[4]{};
    double reward = 0, max_abs_theta_observed = 0, p_right_sum = 0;
};

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);
    Ini ini{};
    try {
        uint64_t episodes = 256, batch = 8, env_seed = 170001, action_seed = 270001, offset = 0;
        std::string checkpoint;
        std::vector<char*> overrides;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg.rfind("--episodes=", 0) == 0) episodes = eval_integer(argv[i] + 11);
            else if (arg.rfind("--batch=", 0) == 0) batch = eval_integer(argv[i] + 8);
            else if (arg.rfind("--env-seed=", 0) == 0) env_seed = eval_integer(argv[i] + 11);
            else if (arg.rfind("--action-seed=", 0) == 0) action_seed = eval_integer(argv[i] + 14);
            else if (arg.rfind("--episode-offset=", 0) == 0) offset = eval_integer(argv[i] + 17);
            else if (arg.rfind("--weights=", 0) == 0) checkpoint = arg.substr(10);
            else overrides.push_back(argv[i]);
        }
        if (!episodes || episodes > INT_MAX || !batch || batch > INT_MAX || batch > episodes ||
                env_seed > UINT32_MAX || offset > UINT32_MAX || episodes - 1 > UINT32_MAX - offset ||
                env_seed > UINT32_MAX - offset - (episodes - 1) ||
                action_seed > UINT64_MAX - offset - (episodes - 1))
            throw std::invalid_argument("invalid episode/batch/seed range");
        puf_ini_load_env(&ini, "decision_cartpole", overrides.size(), overrides.data());
        int device = puf_ini_get(&ini, "base", "gpu_offset");
        eval_cuda(cudaSetDevice(device));
        decision_policy_configure(&ini);
        Dict* env_config = puf_ini_section(&ini, "env", 0);
        const int B = batch;
        auto* weights = static_cast<DecisionPolicyWeights*>(decision_policy_weights(nullptr));
        Allocator parameter_alloc{}, activation_alloc{};
        DecisionPolicyActivations activations{};
        decision_policy_register_parameters(weights, &parameter_alloc);
        decision_policy_register_rollout(weights, &activations, &activation_alloc, B);
        alloc_create(&parameter_alloc); alloc_create(&activation_alloc);
        const ulong model_init_seed = puf_ini_get(&ini, "base", "seed");
        ulong init_seed = model_init_seed;
        decision_policy_initialize(weights, &init_seed, nullptr);
        if (!checkpoint.empty())
            eval_load_weights(checkpoint, static_cast<float*>(parameter_alloc.mem), parameter_alloc.total_elems);

        std::vector<Env> envs(B);
        std::vector<EvalEpisode> active(B);
        std::vector<obs_t> observations(size_t(B) * OBS_SIZE);
        std::vector<float> host_input(observations.size()), host_actions(B), rewards(B), terminals(B), host_logits(B * 3);
        Prec input{.shape = {B, OBS_SIZE}};
        float *actions = nullptr, *probabilities = nullptr, *values = nullptr, *mask = nullptr;
        int* sizes = nullptr;
        curandStatePhilox4_32_10_t* states = nullptr;
        eval_cuda(cudaMalloc(&input.data, host_input.size() * sizeof(float)));
        eval_cuda(cudaMalloc(&actions, B * sizeof(float)));
        eval_cuda(cudaMalloc(&probabilities, B * sizeof(float)));
        eval_cuda(cudaMalloc(&values, B * sizeof(float)));
        eval_cuda(cudaMalloc(&mask, B * 2 * sizeof(float)));
        eval_cuda(cudaMalloc(&sizes, sizeof(int)));
        eval_cuda(cudaMalloc(&states, B * sizeof(*states)));
        int two = 2; std::vector<float> all_legal(B * 2, 1);
        eval_cuda(cudaMemcpy(sizes, &two, sizeof(two), cudaMemcpyHostToDevice));
        eval_cuda(cudaMemcpy(mask, all_legal.data(), all_legal.size() * sizeof(float), cudaMemcpyHostToDevice));
        for (int slot = 0; slot < B; ++slot) {
            puf_init(&envs[slot], env_config);
            envs[slot].agents[0].observations = observations.data() + size_t(slot) * OBS_SIZE;
            envs[slot].agents[0].actions = &host_actions[slot];
            envs[slot].agents[0].rewards = &rewards[slot];
            envs[slot].agents[0].terminals = &terminals[slot];
        }
        auto start_episode = [&](int slot, int id) {
            EvalEpisode episode{}; episode.id = id;
            uint32_t identifier = offset + id;
            episode.env_seed = eval_environment_seed(env_seed, identifier);
            episode.action_seed = action_seed + identifier;
            Env& env = envs[slot]; env.rng = episode.env_seed;
            memset(&env.log, 0, sizeof(env.log));
            puf_reset(&env);
            episode.initial[0] = env.x; episode.initial[1] = env.x_dot;
            episode.initial[2] = env.theta; episode.initial[3] = env.theta_dot;
            active[slot] = episode;
            eval_seed_action<<<1, 1>>>(states, slot, episode.action_seed);
        };
        for (int slot = 0; slot < B; ++slot) start_episode(slot, slot);
        printf("{\"type\":\"protocol\",\"version\":1,\"environment\":\"decision_cartpole\","
               "\"bundle\":%s,\"checkpoint\":%s,\"episodes\":%llu,\"episode_offset\":%llu,"
               "\"batch\":%d,\"model_init_seed\":%llu,\"environment_seed_base\":%llu,\"action_seed_base\":%llu,"
               "\"sampling\":\"puffer_philox_categorical\",\"max_steps\":%d,\"temperature\":%.9g,"
               "\"padded_tokens\":%d,\"bundle_max_tokens\":%d,\"zero_init_critic\":%s,"
               "\"params\":%ld,\"cart_mass\":%.9g,\"pole_mass\":%.9g,"
               "\"pole_length\":%.9g,\"gravity\":%.9g,\"force_mag\":%.9g,\"dt\":%.9g}\n",
               eval_json(decision_policy_context->path).c_str(), eval_json(checkpoint).c_str(),
               (unsigned long long)episodes, (unsigned long long)offset, B, (unsigned long long)model_init_seed,
               (unsigned long long)env_seed, (unsigned long long)action_seed, envs[0].max_steps,
               decision_policy_context->temperature, decision_policy_context->execution_tokens,
               decision_policy_context->bundle.max_len,
               decision_policy_context->zero_init_critic ? "true" : "false",
               parameter_alloc.total_elems, envs[0].cart_mass, envs[0].pole_mass,
               envs[0].pole_length, envs[0].gravity, envs[0].force_mag, envs[0].tau);
        int next = B, complete = 0;
        double total_reward = 0, total_length = 0, total_length_sq = 0;
        int pole_ends = 0, cart_ends = 0, pure_time_limits = 0;
        const double begin = wall_clock();
        while (complete < int(episodes)) {
            for (size_t i = 0; i < observations.size(); ++i) host_input[i] = observations[i];
            eval_cuda(cudaMemcpy(input.data, host_input.data(), host_input.size() * sizeof(float), cudaMemcpyHostToDevice));
            Prec output = decision_policy_forward(weights, &activations, input, nullptr);
            sample_logits<<<grid_size(B), BLOCK_SIZE>>>(output, Prec{}, sizes, actions, actions,
                probabilities, values, states, mask, 2);
            eval_cuda(cudaMemcpy(host_actions.data(), actions, B * sizeof(float), cudaMemcpyDeviceToHost));
            eval_cuda(cudaMemcpy(host_logits.data(), output.data, host_logits.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (int slot = 0; slot < B; ++slot) {
                EvalEpisode& ep = active[slot]; if (ep.id < 0) continue;
                Env& env = envs[slot];
                float left = host_logits[slot * 3], right = host_logits[slot * 3 + 1];
                if (!std::isfinite(left) || !std::isfinite(right))
                    throw std::runtime_error("nonfinite policy logits");
                ep.p_right_sum += 1.0 / (1.0 + std::exp(double(left) - right));
                ep.max_abs_theta_observed = std::max(ep.max_abs_theta_observed, double(std::abs(env.theta)));
                ep.left_actions += host_actions[slot] == 0; ep.right_actions += host_actions[slot] == 1;
                puf_step(&env); ++ep.steps; ep.reward += rewards[slot];
                if (!env.transition.terminated && !env.transition.truncated) continue;
                if (env.log.n != 1 || env.transition.steps != ep.steps)
                    throw std::runtime_error("episode boundary accounting mismatch");
                const bool pole = env.log.pole_angle_termination != 0;
                const bool cart = env.log.x_threshold_termination != 0;
                const bool cap = env.transition.truncated;
                const bool terminated = env.transition.terminated;
                printf("{\"type\":\"episode\",\"episode\":%llu,\"environment_seed\":%u,\"action_seed\":%llu,"
                       "\"initial_state\":[%.9g,%.9g,%.9g,%.9g],\"length\":%d,\"return\":%.9g,"
                       "\"terminated\":%s,\"time_limit_reached\":%s,\"pure_timeout\":%s,"
                       "\"pole_angle_termination\":%s,\"cart_position_termination\":%s,"
                       "\"max_abs_theta_observed\":%.9g,\"left_actions\":%d,\"right_actions\":%d,"
                       "\"mean_probability_right\":%.9g}\n",
                       (unsigned long long)(offset + ep.id), ep.env_seed, (unsigned long long)ep.action_seed,
                       ep.initial[0], ep.initial[1], ep.initial[2], ep.initial[3], ep.steps, ep.reward,
                       terminated ? "true" : "false", cap ? "true" : "false",
                       cap && !terminated ? "true" : "false", pole ? "true" : "false", cart ? "true" : "false",
                       ep.max_abs_theta_observed, ep.left_actions, ep.right_actions, ep.p_right_sum / ep.steps);
                ++complete; total_reward += ep.reward; total_length += ep.steps;
                total_length_sq += double(ep.steps) * ep.steps;
                pole_ends += pole; cart_ends += cart; pure_time_limits += cap && !terminated;
                if (next < int(episodes)) start_episode(slot, next++);
                else ep.id = -1;
            }
        }
        const double mean = total_length / episodes;
        printf("{\"type\":\"summary\",\"episodes\":%d,\"mean_length\":%.9g,\"mean_return\":%.9g,"
               "\"length_standard_error\":%.9g,\"pole_angle_terminations\":%d,"
               "\"cart_position_terminations\":%d,\"pure_timeouts\":%d,\"evaluation_seconds\":%.6f}\n",
               complete, mean, total_reward / episodes,
               episodes > 1 ? std::sqrt(std::max(0.0, total_length_sq - episodes * mean * mean) /
                   (episodes * (episodes - 1))) : 0,
               pole_ends, cart_ends, pure_time_limits, wall_clock() - begin);
        puf_ini_free(&ini);
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "Decision learning evaluation: %s\n", error.what());
        puf_ini_free(&ini); return 1;
    }
}
