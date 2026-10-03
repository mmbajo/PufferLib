// Evaluation-only Snake runner: exact native engine, decision adapter and
// Puffer categorical sampler; independent, reproducible RNG seeds per episode.
#undef PUFFERLIB_BUILD_MAIN
#include "../src/pufferl.cu"
#include "../vendor/cJSON.h"
#ifndef PUFFER_DECISION_POLICY
#include "../src/decision_checkpoint.cuh"
#endif
#include <algorithm>
#include <climits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef PUFFER_DECISION_SNAKE
#error "build for decision_laya or decision_snake"
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

// First legal maximum wins ties. Collision actions remain legal whenever the
// environment's reverse-only mask admits them.
__global__ static void eval_greedy(Prec output, const float* mask, float* actions) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= output.shape[0]) return;
    int best = -1; float maximum = -INFINITY;
    for (int action = 0; action < IB_SNAKE_ACTIONS; ++action) {
        float value = output.data[row * (IB_SNAKE_ACTIONS + 1) + action];
        if (mask[row * IB_SNAKE_ACTIONS + action] && (best < 0 || value > maximum)) {
            maximum = value; best = action;
        }
    }
    actions[row] = float(best);
}

#ifndef PUFFER_DECISION_POLICY
// Board models do not use imported-policy controls. Treat absent options as
// their documented zero defaults, and reject malformed values explicitly.
static void eval_require_zero_option(Ini* ini, const char* section, const char* key) {
    DictItem* item = dict_find(puf_ini_section(ini, section, 0), key);
    if (!item) return;
    double value = item->value;
    bool valid = item->len == 0;
    if (item->str) {
        char* end = nullptr;
        errno = 0;
        value = strtod(item->str, &end);
        valid = valid && end != item->str && errno != ERANGE;
        while (end && isspace((unsigned char)*end)) ++end;
        valid = valid && end && !*end;
    }
    if (!valid || value != 0)
        throw std::invalid_argument(std::string(section) + "." + key +
            " must be zero for the board evaluator");
}

// PUFDT stores architecture and named unpadded tensors. Preserve all exported
// pooling/coordinate variants while binding the same native encoder callbacks
// and padded parameter layout used for ordinary Puffer flat checkpoints.
static DecisionSnakeWeights* eval_named_layout(const decision::Config& config) {
    auto* weights = new DecisionSnakeWeights;
    weights->config = config;
    decision::Model layout(config, 1);
    for (const auto& parameter : layout.parameters()) {
        Prec tensor{};
        for (size_t i = 0; i < parameter.shape.size(); ++i) tensor.shape[i] = parameter.shape[i];
        if (parameter.count % 4) {
            if (parameter.shape.size() != 1) throw std::runtime_error("unaligned named model matrix");
            tensor.shape[0] = (parameter.count + 3) & ~size_t(3);
        }
        weights->parameters.push_back(tensor);
        weights->counts.push_back(parameter.count);
    }
    return weights;
}
static void eval_load_named(DecisionSnakeWeights* weights, const std::string& path) {
    decision::Model source(weights->config, 1);
    decision::checkpoint::load(source, path);
    if (source.parameters().size() != weights->parameters.size())
        throw std::runtime_error("named checkpoint registry differs from native layout");
    for (size_t i = 0; i < weights->parameters.size(); ++i)
        eval_cuda(cudaMemcpy(weights->parameters[i].data, source.parameters()[i].data,
            source.parameters()[i].count * sizeof(float), cudaMemcpyDeviceToDevice));
}
#endif

struct EvalEpisode {
    int id = -1, steps = 0, action_counts[IB_SNAKE_ACTIONS]{};
    uint32_t env_seed = 0;
    uint64_t action_seed = 0;
    int32_t initial_board[IB_SNAKE_CELLS]{};
    unsigned char initial_mask[IB_SNAKE_ACTIONS]{};
    double reward = 0, probability_sum[IB_SNAKE_ACTIONS]{};
};

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);
    Ini ini{};
    try {
        uint64_t episodes = 1024, batch = 16, env_seed = 670001, action_seed = 770001, offset = 0;
        std::string checkpoint, named_checkpoint, sampling = "sampled";
        std::vector<char*> overrides;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg.rfind("--episodes=", 0) == 0) episodes = eval_integer(argv[i] + 11);
            else if (arg.rfind("--batch=", 0) == 0) batch = eval_integer(argv[i] + 8);
            else if (arg.rfind("--env-seed=", 0) == 0) env_seed = eval_integer(argv[i] + 11);
            else if (arg.rfind("--action-seed=", 0) == 0) action_seed = eval_integer(argv[i] + 14);
            else if (arg.rfind("--episode-offset=", 0) == 0) offset = eval_integer(argv[i] + 17);
            else if (arg.rfind("--weights=", 0) == 0) checkpoint = arg.substr(10);
            else if (arg.rfind("--pufdt=", 0) == 0) named_checkpoint = arg.substr(8);
            else if (arg.rfind("--sampling=", 0) == 0) sampling = arg.substr(11);
            else overrides.push_back(argv[i]);
        }
        if (!episodes || episodes > INT_MAX || !batch || batch > INT_MAX || batch > episodes ||
                env_seed > UINT32_MAX || offset > UINT32_MAX || episodes - 1 > UINT32_MAX - offset ||
                env_seed > UINT32_MAX - offset - (episodes - 1) ||
                action_seed > UINT64_MAX - offset - (episodes - 1))
            throw std::invalid_argument("invalid episode/batch/seed range");
        if (sampling != "sampled" && sampling != "greedy" && sampling != "random")
            throw std::invalid_argument("--sampling must be sampled, greedy or random");
        if ((!checkpoint.empty() && !named_checkpoint.empty()) ||
                (sampling == "random" && (!checkpoint.empty() || !named_checkpoint.empty())))
            throw std::invalid_argument("choose one checkpoint format; random sampling takes no checkpoint");
        puf_ini_load_env(&ini, PUFFER_ENV_NAME, overrides.size(), overrides.data());
        int device = puf_ini_get(&ini, "base", "gpu_offset");
        eval_cuda(cudaSetDevice(device));
        const int B = batch, K = IB_SNAKE_ACTIONS;
        std::string bundle, model, representation, pooling, coordinates;
        bool zero_init_critic = false;
        int observation_format = 0, padded_tokens = 101, bundle_max_tokens = 101;
#ifdef PUFFER_DECISION_POLICY
        if (!named_checkpoint.empty()) throw std::invalid_argument("--pufdt requires the decision_snake board build");
        decision_policy_configure(&ini);
        bundle = decision_policy_context->path;
        model = decision_policy_context->bundle.kind == "laya" ? "laya" : decision_policy_context->bundle.encoder.family;
        pooling = "decision_options"; coordinates = "prompt";
        zero_init_critic = decision_policy_context->zero_init_critic;
        padded_tokens = decision_policy_context->execution_tokens;
        bundle_max_tokens = decision_policy_context->bundle.max_len;
#else
        eval_require_zero_option(&ini, "env", "observation_format");
        eval_require_zero_option(&ini, "policy", "zero_init_critic");
        eval_require_zero_option(&ini, "policy", "sequence_length");
        model = "board_transformer"; representation = "board_tokens";
        if (!named_checkpoint.empty()) {
            const auto config = decision::checkpoint::config(named_checkpoint);
            puf_ini_put(&ini, "policy.hidden_size", std::to_string(config.width).c_str());
            puf_ini_put(&ini, "policy.num_layers", std::to_string(config.layers).c_str());
        }
#endif
        Dict* env_config = puf_ini_section(&ini, "env", 0);
        Encoder encoder{};
        encoder.in_dim = OBS_SIZE;
        encoder.out_dim = puf_ini_get(&ini, "policy", "hidden_size");
        encoder.num_layers = puf_ini_get(&ini, "policy", "num_layers");
        create_custom_encoder(&encoder);
        void* weights = nullptr;
#ifndef PUFFER_DECISION_POLICY
        if (!named_checkpoint.empty()) weights = eval_named_layout(decision::checkpoint::config(named_checkpoint));
#endif
        if (!weights) weights = encoder.create_weights(&encoder);
        Allocator parameter_alloc{}, activation_alloc{};
        void* activations = calloc(1, encoder.activation_size);
        if (!activations) throw std::runtime_error("encoder activation allocation failed");
        encoder.reg_params(weights, &parameter_alloc);
        encoder.reg_rollout(weights, activations, &activation_alloc, B);
        alloc_create(&parameter_alloc); alloc_create(&activation_alloc);
        const ulong model_init_seed = puf_ini_get(&ini, "base", "seed");
        ulong init_seed = model_init_seed;
        encoder.init_weights(weights, &init_seed, nullptr);
        if (!checkpoint.empty())
            eval_load_weights(checkpoint, static_cast<float*>(parameter_alloc.mem), parameter_alloc.total_elems);
#ifndef PUFFER_DECISION_POLICY
        auto* board_weights = static_cast<DecisionSnakeWeights*>(weights);
        pooling = board_weights->config.head_pooling ? "head" : "cls";
        coordinates = board_weights->config.relative_coordinates ? "head-relative" : "absolute";
        if (!named_checkpoint.empty()) {
            eval_load_named(board_weights, named_checkpoint);
            checkpoint = named_checkpoint;
        }
#endif

        std::vector<Env> envs(B);
        std::vector<EvalEpisode> active(B);
        std::vector<obs_t> observations(size_t(B) * OBS_SIZE);
        std::vector<unsigned char> masks(size_t(B) * K);
        std::vector<float> host_input(observations.size()), host_mask(size_t(B) * K);
        std::vector<float> host_actions(B), rewards(B), terminals(B), host_logits(size_t(B) * (K + 1));
        Prec input{.shape = {B, OBS_SIZE}};
        Prec random_output{.shape = {B, K + 1}};
        float *actions = nullptr, *probabilities = nullptr, *values = nullptr, *mask = nullptr;
        int* sizes = nullptr;
        curandStatePhilox4_32_10_t* states = nullptr;
        eval_cuda(cudaMalloc(&input.data, host_input.size() * sizeof(float)));
        if (sampling == "random") {
            eval_cuda(cudaMalloc(&random_output.data, size_t(B) * (K + 1) * sizeof(float)));
            eval_cuda(cudaMemset(random_output.data, 0, size_t(B) * (K + 1) * sizeof(float)));
        }
        eval_cuda(cudaMalloc(&actions, B * sizeof(float)));
        eval_cuda(cudaMalloc(&probabilities, B * sizeof(float)));
        eval_cuda(cudaMalloc(&values, B * sizeof(float)));
        eval_cuda(cudaMalloc(&mask, host_mask.size() * sizeof(float)));
        eval_cuda(cudaMalloc(&sizes, sizeof(int)));
        eval_cuda(cudaMalloc(&states, B * sizeof(*states)));
        eval_cuda(cudaMemcpy(sizes, &K, sizeof(K), cudaMemcpyHostToDevice));
        for (int slot = 0; slot < B; ++slot) {
            puf_init(&envs[slot], env_config);
            envs[slot].agents[0].observations = observations.data() + size_t(slot) * OBS_SIZE;
            envs[slot].agents[0].actions = &host_actions[slot];
            envs[slot].agents[0].rewards = &rewards[slot];
            envs[slot].agents[0].terminals = &terminals[slot];
            envs[slot].agents[0].action_mask = masks.data() + size_t(slot) * K;
        }
        observation_format = envs[0].observation_format;
#ifdef PUFFER_DECISION_POLICY
        representation = observation_format == 1 ? "text_coordinates_grid" : "text_grid";
#endif
        auto start_episode = [&](int slot, int id) {
            EvalEpisode episode{}; episode.id = id;
            uint32_t identifier = offset + id;
            episode.env_seed = eval_environment_seed(env_seed, identifier);
            episode.action_seed = action_seed + identifier;
            Env& env = envs[slot];
            memset(&env.log, 0, sizeof(env.log));
            if (decision_snake_reset(&env, episode.env_seed) != 0 ||
                    ib_snake_observe(env.snake, episode.initial_board, IB_SNAKE_CELLS) != 0)
                throw std::runtime_error("explicit Snake reset/initial observation failed");
            memcpy(episode.initial_mask, env.agents[0].action_mask, sizeof(episode.initial_mask));
            active[slot] = episode;
            eval_seed_action<<<1, 1>>>(states, slot, episode.action_seed);
        };
        for (int slot = 0; slot < B; ++slot) start_episode(slot, slot);
        const std::string sampler = sampling == "sampled" ? "puffer_philox_categorical" :
            sampling == "greedy" ? "greedy_first_max" : "puffer_philox_uniform_legal";
        printf("{\"type\":\"protocol\",\"version\":1,\"environment\":%s,"
               "\"model\":%s,\"representation\":%s,\"observation_format\":%d,"
               "\"zero_init_critic\":%s,\"model_width\":%d,\"model_layers\":%d,\"pooling\":%s,\"coordinates\":%s,"
               "\"bundle\":%s,\"checkpoint\":%s,\"episodes\":%llu,\"episode_offset\":%llu,"
               "\"batch\":%d,\"model_init_seed\":%llu,\"environment_seed_base\":%llu,\"action_seed_base\":%llu,"
               "\"sampling\":%s,\"sampling_mode\":%s,\"max_steps\":%d,\"temperature\":%.9g,"
               "\"padded_tokens\":%d,\"bundle_max_tokens\":%d,\"params\":%ld,\"grid_size\":10,\"action_count\":4,"
               "\"primary_metric\":\"food_score\",\"mask_rule\":\"reverse_only\"}\n",
               eval_json(PUFFER_ENV_NAME).c_str(), eval_json(model).c_str(), eval_json(representation).c_str(),
               observation_format, zero_init_critic ? "true" : "false", encoder.out_dim, encoder.num_layers,
               eval_json(pooling).c_str(), eval_json(coordinates).c_str(),
               eval_json(bundle).c_str(), eval_json(checkpoint).c_str(),
               (unsigned long long)episodes, (unsigned long long)offset, B, (unsigned long long)model_init_seed,
               (unsigned long long)env_seed, (unsigned long long)action_seed,
               eval_json(sampler).c_str(), eval_json(sampling).c_str(), envs[0].max_steps,
#ifdef PUFFER_DECISION_POLICY
               decision_policy_context->temperature,
#else
               1.0,
#endif
               padded_tokens, bundle_max_tokens, parameter_alloc.total_elems);
        int next = B, complete = 0;
        double total_reward = 0, total_length = 0, total_food = 0, total_food_sq = 0;
        int collisions = 0, full_boards = 0, pure_timeouts = 0;
        const double begin = wall_clock();
        while (complete < int(episodes)) {
            for (size_t i = 0; i < observations.size(); ++i) host_input[i] = observations[i];
            for (int slot = 0; slot < B; ++slot)
                for (int k = 0; k < K; ++k)
                    // Completed slots have zero engine masks; they are not stepped
                    // again. An all-legal dummy row keeps ignored samples well-defined.
                    host_mask[slot * K + k] = active[slot].id < 0 ? 1 : masks[slot * K + k];
            eval_cuda(cudaMemcpy(input.data, host_input.data(), host_input.size() * sizeof(float), cudaMemcpyHostToDevice));
            eval_cuda(cudaMemcpy(mask, host_mask.data(), host_mask.size() * sizeof(float), cudaMemcpyHostToDevice));
            // Random is a sampler control: it does no policy forward computation.
            Prec output = sampling == "random" ? random_output : encoder.forward(weights, activations, input, nullptr);
            if (sampling == "greedy") eval_greedy<<<grid_size(B), BLOCK_SIZE>>>(output, mask, actions);
            else sample_logits<<<grid_size(B), BLOCK_SIZE>>>(output, Prec{}, sizes, actions, actions,
                probabilities, values, states, mask, K);
            eval_cuda(cudaMemcpy(host_actions.data(), actions, B * sizeof(float), cudaMemcpyDeviceToHost));
            eval_cuda(cudaMemcpy(host_logits.data(), output.data, host_logits.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (int slot = 0; slot < B; ++slot) {
                EvalEpisode& ep = active[slot]; if (ep.id < 0) continue;
                Env& env = envs[slot];
                double maximum = -INFINITY;
                for (int k = 0; k < K; ++k) {
                    double logit = host_logits[slot * (K + 1) + k];
                    if (!std::isfinite(logit)) throw std::runtime_error("nonfinite policy logit");
                    if (masks[slot * K + k]) maximum = std::max(maximum, logit);
                }
                if (!std::isfinite(maximum)) throw std::runtime_error("active Snake has no legal action");
                double probability[K], normalizer = 0;
                for (int k = 0; k < K; ++k) {
                    probability[k] = masks[slot * K + k] ?
                        std::exp(double(host_logits[slot * (K + 1) + k]) - maximum) : 0;
                    normalizer += probability[k];
                }
                float sampled = host_actions[slot];
                if (!(sampled >= 0 && sampled < K) || sampled != float(int(sampled)) ||
                        !masks[slot * K + int(sampled)])
                    throw std::runtime_error("categorical sampler selected an invalid/reversed Snake action");
                for (int k = 0; k < K; ++k)
                    ep.probability_sum[k] += sampling == "greedy" ? double(k == int(sampled)) : probability[k] / normalizer;
                ++ep.action_counts[int(sampled)];
                if (decision_snake_step(&env, int(sampled)) != 0)
                    throw std::runtime_error("explicit Snake step rejected sampled action");
                ++ep.steps; ep.reward += rewards[slot];
                if (!env.transition.terminated && !env.transition.truncated) continue;
                const auto& transition = env.transition;
                const bool collision = transition.outcome == -1;
                const bool full = transition.outcome == 1;
                const bool terminated = transition.terminated;
                const bool truncated = transition.truncated;
                const bool cap = ep.steps >= env.max_steps;
                const bool timeout = truncated && !terminated;
                const int food = int(transition.score);
                if (env.log.n != 1 || transition.steps != ep.steps ||
                        transition.seed != ep.env_seed || transition.episode_return != ep.reward ||
                        transition.score != food || food < 0 || food > 97 ||
                        ep.reward != food - (collision ? 1 : 0) ||
                        terminated != (collision || full) || (!terminated && !timeout) ||
                        (full != (food == 97)))
                    throw std::runtime_error("Snake episode outcome/accounting mismatch");
                printf("{\"type\":\"episode\",\"episode\":%llu,\"environment_seed\":%u,\"action_seed\":%llu,"
                       "\"initial_board\":[", (unsigned long long)(offset + ep.id), ep.env_seed,
                       (unsigned long long)ep.action_seed);
                for (int i = 0; i < IB_SNAKE_CELLS; ++i) printf("%s%d", i ? "," : "", ep.initial_board[i]);
                printf("],\"initial_action_mask\":[%d,%d,%d,%d],\"length\":%d,\"food_score\":%d,\"return\":%.9g,"
                       "\"terminated\":%s,\"truncated\":%s,\"time_limit_reached\":%s,\"pure_timeout\":%s,"
                       "\"outcome\":%d,\"collision\":%s,\"full_board\":%s,"
                       "\"action_counts\":[%d,%d,%d,%d],\"mean_action_probabilities\":[%.9g,%.9g,%.9g,%.9g]}\n",
                       ep.initial_mask[0], ep.initial_mask[1], ep.initial_mask[2], ep.initial_mask[3],
                       ep.steps, food, ep.reward, terminated ? "true" : "false", truncated ? "true" : "false",
                       cap ? "true" : "false", timeout ? "true" : "false", transition.outcome,
                       collision ? "true" : "false", full ? "true" : "false",
                       ep.action_counts[0], ep.action_counts[1], ep.action_counts[2], ep.action_counts[3],
                       ep.probability_sum[0] / ep.steps, ep.probability_sum[1] / ep.steps,
                       ep.probability_sum[2] / ep.steps, ep.probability_sum[3] / ep.steps);
                ++complete; total_reward += ep.reward; total_length += ep.steps;
                total_food += food; total_food_sq += double(food) * food;
                collisions += collision; full_boards += full; pure_timeouts += timeout;
                if (next < int(episodes)) start_episode(slot, next++);
                else ep.id = -1;
            }
        }
        const double mean = total_food / episodes;
        printf("{\"type\":\"summary\",\"episodes\":%d,\"mean_food_score\":%.9g,\"mean_return\":%.9g,"
               "\"mean_length\":%.9g,\"food_score_standard_error\":%.9g,\"collisions\":%d,"
               "\"full_boards\":%d,\"pure_timeouts\":%d,\"evaluation_seconds\":%.6f}\n",
               complete, mean, total_reward / episodes, total_length / episodes,
               episodes > 1 ? std::sqrt(std::max(0.0, total_food_sq - episodes * mean * mean) /
                   (episodes * (episodes - 1))) : 0,
               collisions, full_boards, pure_timeouts, wall_clock() - begin);
        for (auto& env : envs) puf_close(&env);
        puf_ini_free(&ini);
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "Snake learning evaluation: %s\n", error.what());
        puf_ini_free(&ini); return 1;
    }
}
