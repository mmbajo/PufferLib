// Compile this integration test for each native text decision environment.
// It exercises the actual native collector,
// shared model parameters, decoder, PPO update, and checkpoint API.
#include "../src/pufferl.cu"

#include <algorithm>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
static void gpu(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
template<class T> static std::vector<T> download(const T* data, size_t count) {
    std::vector<T> result(count);
    gpu(cudaMemcpy(result.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
    return result;
}
template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(const std::vector<T>& values) {
        gpu(cudaMalloc(&data, values.size() * sizeof(T)));
        gpu(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
static DecisionPolicyWeights* weights(PuffeRL* p) {
    return static_cast<DecisionPolicyWeights*>(p->policies[0].weights.encoder);
}
static DecisionPolicyActivations* training(PuffeRL* p) {
    return static_cast<DecisionPolicyActivations*>(p->train_activs.encoder);
}
static void close_enough(float actual, float expected, const std::string& what,
        float absolute = 2e-5f, float relative = 2e-4f) {
    require(std::isfinite(actual) && std::isfinite(expected) &&
            std::abs(actual - expected) <= absolute + relative * std::abs(expected), what);
}
static void set_parameter(PuffeRL* p, const std::string& name,
        const std::vector<float>& values) {
    auto* w = weights(p);
    for (size_t i = 0; i < w->registry.size(); ++i) {
        if (w->registry[i].name != name) continue;
        require(values.size() == w->registry[i].count, "fixture parameter size mismatch");
        gpu(cudaMemcpy(w->parameters[i].data, values.data(), values.size() * sizeof(float),
                       cudaMemcpyHostToDevice));
        return;
    }
    throw std::runtime_error("fixture parameter missing: " + name);
}
static void check_registry(PuffeRL* p, bool gradients) {
    auto* w = weights(p);
    auto* a = training(p);
    size_t offset = 0, padding = 0;
    for (size_t i = 0; i < w->registry.size(); ++i) {
        size_t count = numel(w->parameters[i].shape);
        require(w->parameters[i].data == p->policies[0].param.data + offset,
                "parameter packing contains an unregistered gap");
        require(a->workspace->gradients[i].data == p->grad.data + offset,
                "gradient registry differs from parameter registry");
        auto values = download(w->parameters[i].data, count);
        auto grads = gradients ? download(a->workspace->gradients[i].data, count) : std::vector<float>();
        for (size_t j = 0; j < count; ++j) {
            require(std::isfinite(values[j]), "nonfinite parameter " + w->registry[i].name);
            if (gradients) require(std::isfinite(grads[j]), "nonfinite shared gradient");
            if (j >= w->registry[i].count) {
                ++padding;
                require(values[j] == 0 && (!gradients || grads[j] == 0), "nonzero alignment padding");
            }
        }
        offset += count;
    }
    require(padding > 0, "fixture must exercise scalar/bias alignment padding");
    require(offset == size_t(numel(p->policies[0].param.shape)) &&
            offset == size_t(numel(p->grad.shape)), "flat registry extent mismatch");
}

// Decode bytes on the CPU independently of the policy's CUDA unpacker. The
// oracle owns its own weights and activation/gradient storage.
struct Oracle {
    pretrained::DecisionModel model;
    int T;
    explicit Oracle(PuffeRL* p, int batch)
        : model(weights(p)->config, batch, decision_policy_context->bundle.max_len,
                DECISION_ACTIONS), T(decision_policy_context->bundle.max_len) {
        auto* w = weights(p);
        for (size_t i = 0; i < w->registry.size(); ++i)
            gpu(cudaMemcpy(model.parameters()[i].data, w->parameters[i].data,
                w->registry[i].count * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    std::vector<float> forward(const std::vector<obs_t>& observations) {
        int B = observations.size() / OBS_SIZE;
        std::vector<int> ids(B * T, decision_policy_context->bundle.pad_id), mask(B * T, 0);
        std::vector<int> markers(B * DECISION_ACTIONS), marker_mask(markers.size(), 1), qtype(B);
        for (int b = 0; b < B; ++b) {
            const obs_t* row = observations.data() + size_t(b) * OBS_SIZE;
            int length = row[0] | (row[1] << 8);
            require(length > 0 && length <= T, "invalid packed observation length");
            qtype[b] = row[DECISION_POLICY_QTYPE_OFFSET];
            for (int k = 0; k < DECISION_ACTIONS; ++k) {
                int position = row[2 + 2 * k] | (row[3 + 2 * k] << 8);
                require(position < length, "marker outside token sequence");
                markers[b * DECISION_ACTIONS + k] = position;
            }
            for (int t = 0; t < length; ++t) {
                uint32_t token = 0;
                for (int byte = 0; byte < 4; ++byte)
                    token |= uint32_t(row[DECISION_POLICY_HEADER_BYTES + 4 * t + byte]) << (8 * byte);
                ids[b * T + t] = token;
                mask[b * T + t] = 1;
            }
            for (int k = 0; k < DECISION_ACTIONS; ++k)
                require(ids[b * T + markers[b * DECISION_ACTIONS + k]] ==
                        int(decision_policy_context->bundle.mask_id), "marker does not select MASK token");
        }
        DeviceBuffer<int> d_ids(ids), d_mask(mask), d_markers(markers), d_marker_mask(marker_mask), d_qtype(qtype);
        auto output = model.forward(d_ids.data, d_mask.data, nullptr, d_markers.data,
            d_marker_mask.data, d_qtype.data, B, T, DECISION_ACTIONS);
        auto logits = download(output.logits, B * DECISION_ACTIONS);
        auto values = download(output.values, B);
        std::vector<float> packed(B * (DECISION_ACTIONS + 1));
        for (int b = 0; b < B; ++b) {
            for (int k = 0; k < DECISION_ACTIONS; ++k)
                packed[b * (DECISION_ACTIONS + 1) + k] =
                    logits[b * DECISION_ACTIONS + k] / decision_policy_context->temperature;
            packed[b * (DECISION_ACTIONS + 1) + DECISION_ACTIONS] = values[b];
        }
        return packed;
    }
};

static std::vector<obs_t> live_observations(PuffeRL* p) {
    return {p->vec->observations, p->vec->observations + size_t(p->hypers.total_agents) * OBS_SIZE};
}
static void check_forward_backward(PuffeRL* p) {
    const int B = p->hypers.total_agents, K = DECISION_ACTIONS;
    auto observations = live_observations(p);
    DeviceBuffer<float> input(std::vector<float>(observations.begin(), observations.end()));
    float original_temperature = decision_policy_context->temperature;
    decision_policy_context->temperature = 1.7f; // Exercise both forward and backward calibration.
    Oracle oracle(p, B);
    auto expected = oracle.forward(observations);
    Prec output = decision_policy_forward(weights(p), training(p),
        Prec{.data = input.data, .shape = {B, OBS_SIZE}}, p->default_stream);
    require(output.shape[0] == B && output.shape[1] == K + 1, "policy output action/value shape mismatch");
    auto actual = download(output.data, expected.size());
    for (size_t i = 0; i < actual.size(); ++i) close_enough(actual[i], expected[i], "policy/oracle forward mismatch");

    std::vector<float> packed_gradient(B * (K + 1)), logits_gradient(B * K), value_gradient(B);
    for (int b = 0; b < B; ++b) {
        for (int k = 0; k < K; ++k) {
            float value = .13f * (b + 1) * (k + 1) - .21f;
            packed_gradient[b * (K + 1) + k] = value;
            logits_gradient[b * K + k] = value / decision_policy_context->temperature;
        }
        packed_gradient[b * (K + 1) + K] = value_gradient[b] = .2f * (b + 1);
    }
    DeviceBuffer<float> grad(packed_gradient), dlogits(logits_gradient), dvalues(value_gradient);
    decision_policy_backward(weights(p), training(p),
        Prec{.data = grad.data, .shape = {B, K + 1}}, p->default_stream);
    oracle.model.zero_grad();
    oracle.model.backward(dlogits.data, nullptr, dvalues.data);
    gpu(cudaDeviceSynchronize());
    for (size_t i = 0; i < weights(p)->registry.size(); ++i) {
        auto expected_grad = download(oracle.model.parameters()[i].grad, weights(p)->registry[i].count);
        auto actual_grad = download(training(p)->workspace->gradients[i].data, expected_grad.size());
        for (size_t j = 0; j < actual_grad.size(); ++j)
            close_enough(actual_grad[j], expected_grad[j],
                "shared parameter gradient mismatch: " + weights(p)->registry[i].name, 4e-5f, 5e-4f);
    }
    check_registry(p, true);
    decision_policy_context->temperature = original_temperature;
}

static int safe_fixture_action(Env* env) {
#ifdef PUFFER_DECISION_LAYA
    int32_t board[100];
    require(ib_snake_observe(env->snake, board, 100) == 0, "cannot inspect fixture board");
    int head = 0;
    while (head < 100 && board[head] != 1) ++head;
    const int dr[4] = {-1, 1, 0, 0}, dc[4] = {0, 0, -1, 1};
    for (int k = 0; k < 4; ++k) {
        int r = head / 10 + dr[k], c = head % 10 + dc[k];
        if (env->agents[0].action_mask[k] && r >= 0 && r < 10 && c >= 0 && c < 10 && board[10 * r + c] <= 0)
            return k;
    }
    throw std::runtime_error("fixture board has no safe timeout action");
#elif defined(PUFFER_DECISION_LIGHTSOUT)
    int32_t board[IB_LIGHTSOUT_CELLS];
    require(ib_lightsout_observe(env->lightsout, board, IB_LIGHTSOUT_CELLS) == 0,
            "cannot read Lights Out timeout fixture");
    for (int action = 0; action < DECISION_ACTIONS; ++action) {
        bool solves = true;
        for (int cell = 0; cell < IB_LIGHTSOUT_CELLS; ++cell) {
            bool toggles = std::abs(cell / 5 - action / 5) +
                           std::abs(cell % 5 - action % 5) <= 1;
            if (board[cell] != int(toggles)) solves = false;
        }
        if (!solves) return action;
    }
    throw std::runtime_error("Lights Out fixture has no non-solving action");
#else
    for (int action = 0; action < DECISION_ACTIONS; ++action)
        if (env->agents[0].action_mask[action]) return action;
    throw std::runtime_error("fixture has no legal action");
#endif
}

static void check_timeout(PuffeRL* p, bool constant_value) {
    int B = p->hypers.total_agents, T = p->hypers.horizon, D = weights(p)->config.encoder.width;
    std::vector<float> critic(D, 0);
    if (!constant_value) for (int i = 0; i < D; ++i) critic[i] = (i % 2 ? -10.0f : 7.0f) / D;
    set_parameter(p, "value_head.weight", critic);
    set_parameter(p, "value_head.bias", {constant_value ? 2.0f : .5f});
    env_restart(p);
    Oracle oracle(p, B);
    std::vector<float> expected_rewards(B, 0);
    float largest_state_effect = 0;
    for (int t = 0; t < T; ++t) {
        pufferl_forward_step(p, 0, t, p->default_stream);
        gpu(cudaDeviceSynchronize());
        auto rewards = download(p->rollouts.rewards.data + t * B, B);
        auto terminals = download(p->rollouts.terminals.data + t * B, B);
        for (int b = 0; b < B; ++b) {
            close_enough(rewards[b], expected_rewards[b], "timeout bootstrap did not use saved final tokens");
            require(terminals[b] == (t ? 1 : 0), "timeout must end GAE recursion despite autoreset");
            if (t && constant_value) require(rewards[b] > 1, "timeout bootstrap was clamped after addition");
        }
        std::vector<obs_t> finals(B * OBS_SIZE);
        std::vector<float> raw(B);
        for (int i = 0; i < p->vec->size; ++i) {
            Env* env = &p->vec->envs[i];
            Agent* agent = &env->agents[0];
            int row = (agent->observations - p->vec->observations) / OBS_SIZE;
            agent->actions[0] = safe_fixture_action(env);
            puf_step(env);
            const obs_t* final = puf_truncation_observation(env, agent);
            require(final && agent->terminals[0] == 1, "fixture must produce a pure timeout");
            require(std::memcmp(final, agent->observations, OBS_SIZE) != 0,
                    "final and autoreset tokens unexpectedly coincide");
            std::copy(final, final + OBS_SIZE, finals.begin() + size_t(row) * OBS_SIZE);
            raw[row] = agent->rewards[0];
        }
        auto final_output = oracle.forward(finals);
        auto reset_output = oracle.forward(live_observations(p));
        for (int b = 0; b < B; ++b) {
            float value = final_output[b * (DECISION_ACTIONS + 1) + DECISION_ACTIONS];
            float reset_value = reset_output[b * (DECISION_ACTIONS + 1) + DECISION_ACTIONS];
            expected_rewards[b] = std::clamp(raw[b], -1.0f, 1.0f) + p->hypers.gamma * value;
            largest_state_effect = std::max(largest_state_effect, std::abs(value - reset_value));
            if (constant_value) require(value == 2, "constant critic fixture did not produce two");
        }
        cpu_upload(p, 0, B, p->default_stream);
        gpu(cudaDeviceSynchronize());
    }
    if (!constant_value) require(largest_state_effect > 1e-6f, "critic cannot distinguish final and reset tokens");
}

static void check_training_checkpoint(PuffeRL* p) {
    env_restart(p);
    auto before = download(p->policies[0].param.data, numel(p->policies[0].param.shape));
    rollouts(p);
    gpu(cudaDeviceSynchronize());
    auto rewards = download(p->rollouts.rewards.data, p->hypers.horizon * p->hypers.total_agents);
    train_impl(p, nullptr);
    gpu(cudaDeviceSynchronize());
    require(p->epoch == 1, "PPO did not advance an epoch");
    auto transposed = download(p->train_rollouts.rewards.data, rewards.size());
    for (int t = 0; t < p->hypers.horizon; ++t) for (int b = 0; b < p->hypers.total_agents; ++b)
        require(transposed[b * p->hypers.horizon + t] == rewards[t * p->hypers.total_agents + b],
                "PPO transpose changed bootstrapped rewards");
    check_registry(p, true);
    auto after = download(p->policies[0].param.data, before.size());
    std::map<std::string, size_t> changed;
    size_t offset = 0;
    for (size_t i = 0; i < weights(p)->registry.size(); ++i) {
        const auto& spec = weights(p)->registry[i];
        std::string group = spec.name.substr(0, spec.name.find('.'));
        for (size_t j = 0; j < spec.count; ++j) changed[group] += before[offset + j] != after[offset + j];
        offset += numel(weights(p)->parameters[i].shape);
    }
    for (const char* group : {"encoder", "scorer", "value_head"})
        require(changed[group] > 0, std::string("PPO update did not reach ") + group);
    require(changed["act_head"] == 0, "PPO modified unused act/escalate head");

    auto observations = live_observations(p);
    DeviceBuffer<float> input(std::vector<float>(observations.begin(), observations.end()));
    auto evaluate = [&]() {
        auto output = decision_policy_forward(weights(p), training(p),
            Prec{.data = input.data, .shape = {p->hypers.total_agents, OBS_SIZE}}, p->default_stream);
        return download(output.data, p->hypers.total_agents * (DECISION_ACTIONS + 1));
    };
    auto expected = evaluate();
    char checkpoint[] = "/tmp/puffer-decision-policy.XXXXXX";
    int descriptor = mkstemp(checkpoint);
    require(descriptor >= 0, "cannot create temporary checkpoint");
    close(descriptor);
    puf_save_weights(p, checkpoint);
    set_parameter(p, "value_head.bias", {123});
    require(evaluate() != expected, "checkpoint reload fixture failed to change outputs");
    pufferl_load_policy(p, 0, checkpoint);
    unlink(checkpoint);
    require(download(p->policies[0].param.data, after.size()) == after, "checkpoint reload changed flat weights");
    require(evaluate() == expected, "checkpoint reload changed predictions");
}

int main(int argc, char** argv) {
    PuffeRL* p = nullptr;
    Ini ini{};
    try {
        require(argc == 2, "usage: test_decision_policy BUNDLE (use a small BERT fixture)");
        puf_ini_load_env(&ini, PUFFER_ENV_NAME, 0, nullptr);
        const char* overrides[][2] = {
            {"base.async", "0"}, {"base.cudagraphs", "-1"},
            {"vec.total_agents", "2"}, {"vec.num_buffers", "1"}, {"vec.num_threads", "1"},
            {"env.max_steps", "1"}, {"train.horizon", "4"}, {"train.minibatch_size", "8"},
            {"train.total_timesteps", "32"}, {"train.gamma", "0.99"},
            {"train.anneal_lr", "0"}, {"train.replay_ratio", "1"},
        };
        for (const auto& item : overrides) puf_ini_put(&ini, item[0], item[1]);
        puf_ini_put(&ini, "policy.bundle", argv[1]);
        TrainContext context{}; context.world_size = 1;
        p = create_pufferl(&ini, &context);
        require(p->policies[0].master_weights.data == p->policies[0].param.data,
                "FP32 master and policy parameters must be shared");
        check_registry(p, false);
        check_forward_backward(p);
        check_timeout(p, true);
        check_timeout(p, false);
        check_training_checkpoint(p);
        close_pufferl(p); p = nullptr;
        puf_ini_free(&ini);
        std::cout << PUFFER_ENV_NAME << ": K=" << DECISION_ACTIONS
                  << " packing, gradients, timeout bootstrap, PPO and checkpoint reload passed\n";
        return 0;
    } catch (const std::exception& error) {
        if (p) close_pufferl(p);
        puf_ini_free(&ini);
        std::cerr << "Native decision policy integration: " << error.what() << '\n';
        return 1;
    }
}
