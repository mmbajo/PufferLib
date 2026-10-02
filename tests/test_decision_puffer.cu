// Integration test of the actual native collector, policy callbacks, PPO and
// optimizer. Build/run from the repository root; no training CLI main is used.
#include "../src/pufferl.cu"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

static void gpu(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

template<class T> static std::vector<T> download(const T* data, size_t count) {
    std::vector<T> values(count);
    gpu(cudaMemcpy(values.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
    return values;
}

static DecisionSnakeWeights* weights(PuffeRL* p) {
    return static_cast<DecisionSnakeWeights*>(p->policies[0].weights.encoder);
}

static DecisionSnakeActivations* training(PuffeRL* p) {
    return static_cast<DecisionSnakeActivations*>(p->train_activs.encoder);
}

static void set_parameter(PuffeRL* p, const std::string& name, const std::vector<float>& values) {
    auto* w = weights(p);
    const auto& layout = training(p)->workspace->model->parameters();
    for (size_t i = 0; i < layout.size(); ++i) {
        if (layout[i].name != name) continue;
        require(w->counts[i] == values.size(), "fixture parameter size mismatch");
        gpu(cudaMemcpy(w->parameters[i].data, values.data(), values.size() * sizeof(float),
                       cudaMemcpyHostToDevice));
        return;
    }
    throw std::runtime_error("fixture parameter not found: " + name);
}

static std::vector<uint32_t> reset_instances(PuffeRL* p, uint32_t base) {
    require(!p->hypers.async, "fixture requires synchronous native collection");
    std::vector<uint32_t> seeds(p->hypers.total_agents);
    for (int i = 0; i < p->vec->size; ++i) {
        Env* env = &p->vec->envs[i];
        size_t row = (env->agents[0].observations - p->vec->observations) / OBS_SIZE;
        seeds[row] = base + (uint32_t)(1000 * row);
        require(decision_snake_reset(env, seeds[row]) == 0, "seeded fixture reset failed");
    }
    cpu_upload(p, 0, p->hypers.total_agents, p->default_stream);
    gpu(cudaDeviceSynchronize());
    return seeds;
}

static void check_padding_and_layout(PuffeRL* p, bool after_training) {
    auto* w = weights(p);
    auto* a = training(p);
    size_t offset = 0, padding = 0;
    for (size_t i = 0; i < w->parameters.size(); ++i) {
        size_t count = (size_t)numel(w->parameters[i].shape);
        require(w->parameters[i].data == p->policies[0].param.data + offset,
                "parameter packing has an unregistered alignment gap");
        require(a->workspace->gradients[i].data == p->grad.data + offset,
                "gradient packing differs from parameter packing");
        auto values = download(w->parameters[i].data, count);
        for (float value : values) require(std::isfinite(value), "nonfinite native parameter");
        std::vector<float> gradients;
        if (after_training) {
            gradients = download(a->workspace->gradients[i].data, count);
            for (float gradient : gradients)
                require(std::isfinite(gradient), "nonfinite native gradient, including padding");
        }
        for (size_t j = w->counts[i]; j < count; ++j) {
            ++padding;
            require(values[j] == 0, "optimizer changed padded parameter bytes");
            if (after_training) require(gradients[j] == 0, "padded gradient bytes are nonzero");
        }
        offset += count;
    }
    require(padding == 3, "fixture must cover the padded scalar value bias");
    require(offset == (size_t)numel(p->policies[0].param.shape), "flat parameter extent mismatch");
    require(offset == (size_t)numel(p->grad.shape), "flat gradient extent mismatch");
}

// Reconstruct pre-reset boards from episode seeds and the actions actually
// sampled by the core. A separate model workspace then evaluates those boards,
// bypassing all Puffer final-observation upload/selection/packing code.
static std::vector<float> check_rollout(PuffeRL* p, const std::vector<uint32_t>& seeds,
                                       bool constant_value) {
    const int T = p->hypers.horizon, B = p->hypers.total_agents;
    auto rewards = download(p->rollouts.rewards.data, T * B);
    auto terminals = download(p->rollouts.terminals.data, T * B);
    auto actions = download(p->rollouts.actions.data, T * B);
    auto observations = download(p->rollouts.observations.data, T * B * OBS_SIZE);
    std::vector<int> final_boards((T - 1) * B * OBS_SIZE), reset_boards(final_boards.size());
    std::vector<float> raw((T - 1) * B);
    for (int b = 0; b < B; ++b) {
        require(rewards[b] == 0 && terminals[b] == 0, "initial rollout row is not a clean reset");
        for (int t = 1; t < T; ++t) {
            int row = (t - 1) * B + b;
            IBSnake* reference = ib_snake_create(seeds[b] + (uint32_t)t - 1, 1);
            require(reference != nullptr, "reference environment allocation failed");
            int action = (int)actions[row];
            require(actions[row] == action && action == 0, "fixture must sample legal up actions");
            require(ib_snake_step(reference, action) == 0, "reference action rejected");
            require(ib_snake_truncated(reference) && !ib_snake_terminated(reference),
                    "fixture must end through timeout");
            raw[row] = (float)ib_snake_reward(reference);
            require(ib_snake_observe(reference, final_boards.data() + row * OBS_SIZE, OBS_SIZE) == 0,
                    "reference final observation failed");
            require(ib_snake_reset(reference, seeds[b] + (uint32_t)t, 1) == 0,
                    "reference next reset failed");
            require(ib_snake_observe(reference, reset_boards.data() + row * OBS_SIZE, OBS_SIZE) == 0,
                    "reference reset observation failed");
            ib_snake_destroy(reference);
            require(terminals[t * B + b] == 1, "timeout must terminate the advantage trace");
            for (int cell = 0; cell < OBS_SIZE; ++cell)
                require(observations[(t * B + b) * OBS_SIZE + cell]
                            == reset_boards[row * OBS_SIZE + cell] + 1,
                        "rollout observation is not the next episode's encoded reset board");
        }
    }

    auto* w = weights(p);
    decision::Model oracle(w->config, (T - 1) * B);
    for (size_t i = 0; i < w->parameters.size(); ++i)
        gpu(cudaMemcpy(oracle.parameters()[i].data, w->parameters[i].data,
            w->counts[i] * sizeof(float), cudaMemcpyDeviceToDevice));
    int* boards = nullptr;
    gpu(cudaMalloc(&boards, final_boards.size() * sizeof(int)));
    gpu(cudaMemcpy(boards, final_boards.data(), final_boards.size() * sizeof(int), cudaMemcpyHostToDevice));
    auto values = download(oracle.forward(boards, (T - 1) * B).values, (T - 1) * B);
    gpu(cudaMemcpy(boards, reset_boards.data(), reset_boards.size() * sizeof(int), cudaMemcpyHostToDevice));
    auto reset_values = download(oracle.forward(boards, (T - 1) * B).values, (T - 1) * B);
    gpu(cudaFree(boards));
    float largest_board_effect = 0;
    bool saw_growth = false;
    for (int t = 1; t < T; ++t) for (int b = 0; b < B; ++b) {
        int row = (t - 1) * B + b;
        float expected = raw[row] + p->hypers.gamma * values[row];
        require(std::abs(rewards[t * B + b] - expected) < 1e-4f,
                "timeout reward differs from raw reward plus value of the pre-reset final board");
        largest_board_effect = std::max(largest_board_effect, std::abs(values[row] - reset_values[row]));
        saw_growth = saw_growth || raw[row] == 1;
        if (constant_value) {
            require(values[row] == 2, "fixture value head must produce exactly two");
            require(rewards[t * B + b] > 1, "timeout bootstrap was clamped to one");
        }
    }
    if (constant_value) require(saw_growth, "fixture must include food eaten on a timeout");
    else require(largest_board_effect > 1e-3f, "value fixture cannot distinguish final and reset boards");
    return rewards;
}

int main() {
    PuffeRL* p = nullptr;
    Ini ini = {};
    try {
        puf_ini_load_env(&ini, "decision_snake", 0, nullptr);
        const char* overrides[][2] = {
            {"base.async", "0"}, {"base.cudagraphs", "-1"},
            {"vec.total_agents", "8"}, {"vec.num_buffers", "2"}, {"vec.num_threads", "2"},
            {"env.max_steps", "1"}, {"policy.hidden_size", "8"}, {"policy.num_layers", "1"},
            {"train.horizon", "4"}, {"train.minibatch_size", "32"},
            {"train.total_timesteps", "128"}, {"train.gamma", "0.99"},
            {"train.anneal_lr", "0"}, {"train.replay_ratio", "1"},
        };
        for (const auto& item : overrides) puf_ini_put(&ini, item[0], item[1]);
        TrainContext context = {};
        context.world_size = 1;
        p = create_pufferl(&ini, &context);
        require(p->policies[0].master_weights.data == p->policies[0].param.data,
                "FP32 test requires shared master and inference parameters");
        check_padding_and_layout(p, false);
        set_parameter(p, "action_head.weight", std::vector<float>(32, 0));
        set_parameter(p, "action_head.bias", {80, -80, -80, -80});
        set_parameter(p, "value_head.weight", std::vector<float>(8, 0));
        set_parameter(p, "value_head.bias", {2});
        auto before = download(p->policies[0].param.data, numel(p->policies[0].param.shape));
        auto seeds = reset_instances(p, 70); // Seed 70 places food directly above the initial head.
        rollouts(p);
        gpu(cudaDeviceSynchronize());
        auto expected_rewards = check_rollout(p, seeds, true);
        train_impl(p, nullptr);
        gpu(cudaDeviceSynchronize());
        require(p->epoch == 1, "native training did not advance an epoch");
        auto transposed = download(p->train_rollouts.rewards.data, 32);
        for (int t = 0; t < 4; ++t) for (int b = 0; b < 8; ++b)
            require(transposed[b * 4 + t] == expected_rewards[t * 8 + b],
                    "native training changed or re-clamped bootstrapped rewards");
        check_padding_and_layout(p, true);
        auto after = download(p->policies[0].param.data, before.size());
        require(before != after, "native PPO/Muon update left every Transformer parameter unchanged");

        // A constant value proves reward scaling; a board-dependent value also
        // catches accidentally bootstrapping from the autoreset observation.
        set_parameter(p, "action_head.weight", std::vector<float>(32, 0));
        set_parameter(p, "action_head.bias", {80, -80, -80, -80});
        set_parameter(p, "value_head.weight", {.2f, -.3f, .1f, .4f, -.15f, .25f, -.35f, .05f});
        set_parameter(p, "value_head.bias", {.5f});
        seeds = reset_instances(p, 170);
        rollouts(p);
        gpu(cudaDeviceSynchronize());
        check_rollout(p, seeds, false);
        close_pufferl(p);
        puf_ini_free(&ini);
        std::cout << "Native Puffer Transformer: timeout bootstrap, final boards, PPO update and padding passed\n";
        return 0;
    } catch (const std::exception& error) {
        if (p) close_pufferl(p);
        puf_ini_free(&ini);
        std::cerr << "Native Puffer integration: " << error.what() << '\n';
        return 1;
    }
}
