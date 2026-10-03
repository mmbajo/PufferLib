// Reuse the integration test's native-core helpers and checkpoint/PPO check.
// This executable has a separate main and exercises optional initialization and
// padding controls against the exact same model parameters and observations.
#define main decision_policy_existing_integration_main
#include "test_decision_policy.cu"
#undef main

static void host_options() {
    Ini ini{};
    puf_ini_set(puf_ini_section(&ini, "policy", 1), "bundle", "unused");
    require(decision_policy_integer_option(&ini, "zero_init_critic", 1) == 0,
        "omitted critic setting must default to zero");
    require(decision_policy_integer_option(&ini, "sequence_length", 2048) == 0,
        "omitted sequence setting must default to bundle length");
    for (const char* value : {"0", "1", "1.0"}) {
        puf_ini_put(&ini, "policy.zero_init_critic", value);
        require(decision_policy_integer_option(&ini, "zero_init_critic", 1) == atoi(value),
            "numeric critic setting not accepted");
    }
    for (const char* value : {"2", "-1", "0.5", "nan", "inf", "true", "false", "garbage", "0,1"}) {
        puf_ini_put(&ini, "policy.zero_init_critic", value);
        bool rejected = false;
        try { decision_policy_integer_option(&ini, "zero_init_critic", 1); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, std::string("invalid critic setting accepted: ") + value);
    }
    for (const char* value : {"-1", "2049", "383.5", "nan", "true", "384x"}) {
        puf_ini_put(&ini, "policy.sequence_length", value);
        bool rejected = false;
        try { decision_policy_integer_option(&ini, "sequence_length", 2048); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, std::string("invalid sequence setting accepted: ") + value);
    }
    puf_ini_free(&ini);
    std::cout << "PASS: optional controls default off and reject malformed values\n";
}

static bool critic_parameter(const std::string& name) {
    return name.compare(0, 11, "value_head.") == 0;
}

int main(int argc, char** argv) {
    PuffeRL* p = nullptr;
    Ini ini{};
    try {
        host_options();
        if (argc == 2 && !strcmp(argv[1], "--host-only")) return 0;
        require(argc >= 2 && argc <= 4,
            "usage: test_decision_policy_options BUNDLE [--forward-only] [--coordinates] | --host-only");
        bool forward_only = false, coordinates = false;
        for (int i = 2; i < argc; ++i) {
            if (!strcmp(argv[i], "--forward-only") && !forward_only) forward_only = true;
            else if (!strcmp(argv[i], "--coordinates") && !coordinates) coordinates = true;
            else throw std::invalid_argument("unknown or duplicate test option");
        }
        puf_ini_load_env(&ini, PUFFER_ENV_NAME, 0, nullptr);
        const char* overrides[][2] = {
            {"base.async", "0"}, {"base.cudagraphs", "-1"}, {"base.seed", "73"},
            {"vec.total_agents", "2"}, {"vec.num_buffers", "1"}, {"vec.num_threads", "1"},
            {"env.max_steps", "500"}, {"train.horizon", "4"}, {"train.minibatch_size", "8"},
            {"train.total_timesteps", "32"}, {"train.learning_rate", "0.001"},
            {"train.anneal_lr", "0"}, {"train.replay_ratio", "1"},
            {"policy.zero_init_critic", "1"}, {"policy.sequence_length", "0"},
        };
        for (const auto& item : overrides) puf_ini_put(&ini, item[0], item[1]);
        if (coordinates) puf_ini_put(&ini, "env.observation_format", "1");
        puf_ini_put(&ini, "policy.bundle", argv[1]);
        decision_policy_configure(&ini);
        bool oversized_rejected = false;
        try {
            DecisionPolicyContext invalid(argv[1], true, decision_policy_context->bundle.max_len + 1);
        } catch (const std::invalid_argument&) { oversized_rejected = true; }
        require(oversized_rejected, "execution length above bundle limit accepted");
        puf_ini_put(&ini, "policy.zero_init_critic", "0");
        bool switch_rejected = false;
        try { decision_policy_configure(&ini); }
        catch (const std::invalid_argument&) { switch_rejected = true; }
        require(switch_rejected, "live process changed initialization setting");
        puf_ini_put(&ini, "policy.zero_init_critic", "1");
        TrainContext context{}; context.world_size = 1;
        p = create_pufferl(&ini, &context);
        require(decision_policy_context->zero_init_critic && weights(p)->zero_init_critic,
            "critic flag did not reach native policy weights");
        int original_tokens = decision_policy_context->execution_tokens;
        require(original_tokens == decision_policy_context->bundle.max_len,
            "default sequence length changed");
        const int B = p->hypers.total_agents, K = DECISION_ACTIONS;
        auto observations = live_observations(p);
        int actual_tokens = 0;
        for (int b = 0; b < B; ++b) {
            const obs_t* row = observations.data() + size_t(b) * OBS_SIZE;
            actual_tokens = std::max(actual_tokens, int(row[0]) | (int(row[1]) << 8));
        }
        int reduced_tokens = 384;
        require(actual_tokens <= reduced_tokens && reduced_tokens < original_tokens,
            "test bundle/observations require real padding above 384 tokens");
        DeviceBuffer<float> input(std::vector<float>(observations.begin(), observations.end()));
        Prec input_view{.data = input.data, .shape = {B, OBS_SIZE}};
        auto zero_output = download(decision_policy_forward(weights(p), training(p),
            input_view, p->default_stream).data, B * (K + 1));
        for (int b = 0; b < B; ++b)
            require(zero_output[b * (K + 1) + K] == 0, "zero critic produces a nonzero value");

        // The old initializer remains an independent reference: same seed,
        // imported bundle and padded registry, with only this flag disabled.
        auto* reference = static_cast<DecisionPolicyWeights*>(decision_policy_weights(nullptr));
        reference->zero_init_critic = false;
        Allocator reference_params{}, reference_acts{};
        decision_policy_register_parameters(reference, &reference_params);
        alloc_create(&reference_params);
        ulong seed = 73;
        decision_policy_initialize(reference, &seed, p->default_stream);
        require(seed == 74, "initializer must consume exactly one native seed");
        size_t changed_critic = 0;
        for (size_t i = 0; i < reference->registry.size(); ++i) {
            size_t count = numel(reference->parameters[i].shape);
            auto normal = download(reference->parameters[i].data, count);
            auto zero = download(weights(p)->parameters[i].data, count);
            require(count == size_t(numel(weights(p)->parameters[i].shape)), "parameter layout changed");
            for (size_t j = 0; j < count; ++j) {
                if (critic_parameter(reference->registry[i].name)) {
                    require(zero[j] == 0, "zero critic or its padding is nonzero");
                    changed_critic += normal[j] != zero[j];
                } else require(memcmp(&normal[j], &zero[j], sizeof(float)) == 0,
                    "critic initialization changed another parameter or RNG sequence");
            }
        }
        require(changed_critic > 0, "normal initialization reference already had zero critic weights");
        DecisionPolicyActivations reference_activation{};
        decision_policy_register_rollout(reference, &reference_activation, &reference_acts, B);
        alloc_create(&reference_acts);
        auto normal_output = download(decision_policy_forward(reference, &reference_activation,
            input_view, p->default_stream).data, B * (K + 1));
        for (int b = 0; b < B; ++b) for (int k = 0; k < K; ++k) {
            size_t i = b * (K + 1) + k;
            require(memcmp(&normal_output[i], &zero_output[i], sizeof(float)) == 0,
                "zero critic changed initial policy logits");
        }
        std::cout << "PASS: noncritic parameters and policy logits byte-identical; initial values zero\n";

        // Keep the same parameter buffers; only padding/workspace extent differs.
        decision_policy_context->execution_tokens = reduced_tokens;
        DecisionPolicyActivations reduced_activation{};
        Allocator reduced_acts{}, reduced_grads{};
        if (forward_only)
            decision_policy_register_rollout(weights(p), &reduced_activation, &reduced_acts, B);
        else decision_policy_register_train(weights(p), &reduced_activation, &reduced_acts, &reduced_grads, B);
        alloc_create(&reduced_acts);
        if (!forward_only) alloc_create(&reduced_grads);
        auto short_output = download(decision_policy_forward(weights(p), &reduced_activation,
            input_view, p->default_stream).data, B * (K + 1));
        double maximum_output_difference = 0;
        for (size_t i = 0; i < short_output.size(); ++i) {
            close_enough(short_output[i], zero_output[i], "padding-only forward mismatch", 2e-4f, 2e-4f);
            maximum_output_difference = std::max(maximum_output_difference,
                fabs(double(short_output[i]) - zero_output[i]));
        }
        // Also compare the historical nonzero critic; a zero value alone would
        // not establish padding parity for the existing default value path.
        reduced_activation.workspace->bound = false;
        auto short_normal_output = download(decision_policy_forward(reference, &reduced_activation,
            input_view, p->default_stream).data, B * (K + 1));
        for (size_t i = 0; i < short_normal_output.size(); ++i) {
            close_enough(short_normal_output[i], normal_output[i], "default-critic padding mismatch", 2e-4f, 2e-4f);
            maximum_output_difference = std::max(maximum_output_difference,
                fabs(double(short_normal_output[i]) - normal_output[i]));
        }
        reduced_activation.workspace->bound = false;
        // Packing an input that exceeds the execution budget must fail without
        // touching existing observation storage or silently truncating its state.
        std::vector<obs_t> guard(OBS_SIZE, 0xA5);
        std::vector<std::string> options{"Move up", "Move down", "Move left", "Move right"};
        decision_policy_context->execution_tokens = 1;
        bool rejected = false;
        try { decision_policy_encode("Snake test state", "Choose the next move.", options, guard.data()); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected && std::all_of(guard.begin(), guard.end(), [](obs_t x) { return x == 0xA5; }),
            "insufficient execution budget truncated or modified observation storage");
        decision_policy_context->execution_tokens = original_tokens;
        std::cout << "PASS: padding " << original_tokens << " -> " << reduced_tokens
                  << ", actual tokens=" << actual_tokens << ", max output difference="
                  << maximum_output_difference << "; insufficient budget rejected\n";
        if (!forward_only) {
            std::vector<float> mixed(B * (K + 1));
            for (int b = 0; b < B; ++b) {
                for (int k = 0; k < K; ++k) mixed[b*(K+1)+k] = .03f*(b+1)*(k+1) - .08f;
                mixed[b*(K+1)+K] = .25f*(b+1);
            }
            DeviceBuffer<float> mixed_gradient(mixed);
            Prec gradient_view{.data = mixed_gradient.data, .shape = {B, K + 1}};
            decision_policy_forward(weights(p), training(p), input_view, p->default_stream);
            decision_policy_backward(weights(p), training(p), gradient_view, p->default_stream);
            decision_policy_context->execution_tokens = reduced_tokens;
            decision_policy_forward(weights(p), &reduced_activation, input_view, p->default_stream);
            decision_policy_backward(weights(p), &reduced_activation, gradient_view, p->default_stream);
            gpu(cudaDeviceSynchronize());
            double maximum_gradient_difference = 0;
            for (size_t i = 0; i < weights(p)->registry.size(); ++i) {
                auto full = download(training(p)->workspace->gradients[i].data, weights(p)->registry[i].count);
                auto short_grad = download(reduced_activation.workspace->gradients[i].data, full.size());
                for (size_t j = 0; j < full.size(); ++j) {
                    close_enough(short_grad[j], full[j], "padding-only gradient mismatch", 4e-5f, 5e-4f);
                    maximum_gradient_difference = std::max(maximum_gradient_difference,
                        fabs(double(short_grad[j]) - full[j]));
                }
            }
            decision_policy_context->execution_tokens = original_tokens;
            std::cout << "PASS: all padding-only parameter gradients; max difference="
                      << maximum_gradient_difference << '\n';
            std::vector<float> critic_gradient(B * (K + 1), 0);
            for (int b = 0; b < B; ++b) critic_gradient[b*(K+1)+K] = .25f*(b+1);
            DeviceBuffer<float> only_value(critic_gradient);
            Prec value_gradient{.data = only_value.data, .shape = {B, K + 1}};
            decision_policy_forward(weights(p), training(p), input_view, p->default_stream);
            decision_policy_backward(weights(p), training(p), value_gradient, p->default_stream);
            bool critic_nonzero = false;
            for (size_t i = 0; i < weights(p)->registry.size(); ++i) {
                auto grad = download(training(p)->workspace->gradients[i].data, weights(p)->registry[i].count);
                for (float value : grad) {
                    require(std::isfinite(value), "nonfinite initial critic gradient");
                    if (critic_parameter(weights(p)->registry[i].name)) critic_nonzero |= value != 0;
                    else require(value == 0, "zero critic transmitted its initial gradient to policy/encoder");
                }
            }
            require(critic_nonzero, "zero initialization blocked critic learning");
            muon_step(&p->muon, p->policies[0].master_weights, p->grad, 1.5f, p->default_stream);
            gpu(cudaStreamSynchronize(p->default_stream));
            auto learned_output = download(decision_policy_forward(weights(p), training(p),
                input_view, p->default_stream).data, B * (K + 1));
            require(learned_output[K] != 0, "Muon did not update the zero critic");
            decision_policy_backward(weights(p), training(p), value_gradient, p->default_stream);
            bool encoder_nonzero = false;
            for (size_t i = 0; i < weights(p)->registry.size(); ++i) {
                if (weights(p)->registry[i].name.compare(0, 8, "encoder.") != 0) continue;
                for (float value : download(training(p)->workspace->gradients[i].data, weights(p)->registry[i].count))
                    encoder_nonzero |= value != 0;
            }
            require(encoder_nonzero, "trained critic cannot send later gradients into the encoder");
            check_training_checkpoint(p);
            std::cout << "PASS: initial critic gradient isolation, nonzero critic update, later encoder gradients, PPO and checkpoint reload\n";
        }
        close_pufferl(p); p = nullptr;
        puf_ini_free(&ini);
        return 0;
    } catch (const std::exception& error) {
        if (p) close_pufferl(p);
        puf_ini_free(&ini);
        std::cerr << "Decision policy options: " << error.what() << '\n';
        return 1;
    }
}
