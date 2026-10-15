#include "pretrained_fixture.h"
#include "../src/pretrained_decision.cuh"

int main(int argc, char** argv) {
    try {
        require(argc == 4 || argc == 5, "usage: decision-test weights.bin input.bin output.bin [fp32|tf32]");
        std::ifstream input(argv[2], std::ios::binary);
        std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
        require((bool)input && (bool)output, "cannot open decision fixture files");
        if (std::string(argv[1]) == "--loss-only") {
            int batch = read<uint32_t>(input), options = read<uint32_t>(input);
            int count = batch * options;
            Device<float> logits(count), temperatures(batch), losses(batch), gradient(count);
            Device<int> mask(count), targets(batch);
            logits.load(input); mask.load(input); targets.load(input); temperatures.load(input);
            for (bool calibrated : {true, false}) {
                pretrained::decision_loss(logits.data, mask.data, targets.data,
                    calibrated ? temperatures.data : nullptr, losses.data, gradient.data,
                    batch, options);
                dump(output, losses.data, batch); dump(output, gradient.data, count);
                pretrained::decision_loss(logits.data, mask.data, targets.data,
                    calibrated ? temperatures.data : nullptr, losses.data, nullptr,
                    batch, options);
                dump(output, losses.data, batch);
            }
        } else if (std::string(argv[1]) == "--adam-only") {
            int count = read<uint32_t>(input), steps = read<uint32_t>(input);
            float lr = read<float>(input);
            Device<float> weights(count), gradient(count), first(count), second(count);
            weights.load(input);
            gpu(cudaMemset(first.data, 0, count * sizeof(float)));
            gpu(cudaMemset(second.data, 0, count * sizeof(float)));
            for (int step = 1; step <= steps; ++step) {
                gradient.load(input);
                pretrained::decision_adam(weights.data, gradient.data, first.data,
                    second.data, count, lr, step);
                dump(output, weights.data, count);
                dump(output, first.data, count); dump(output, second.data, count);
            }
        } else {
            std::ifstream weights(argv[1], std::ios::binary);
            require((bool)weights && (bool)input && (bool)output, "cannot open decision fixture files");
            pretrained::DecisionConfig config;
            config.encoder = load_config(weights);
            if (argc == 5) config.encoder.gemm_precision = pretrained::parse_gemm_precision(argv[4]);
            std::string magic = read_string(input, 8);
            bool supervised = magic == std::string("PUFPDI2\0", 8);
            require(supervised || magic == std::string("PUFPDI1\0", 8), "invalid decision input magic");
            config.head_layers = read<uint32_t>(input);
            config.n_act = read<uint32_t>(input);
            config.value_head = read<uint32_t>(input);
            int batch = read<uint32_t>(input), tokens = read<uint32_t>(input), options = read<uint32_t>(input);
            require(batch > 0 && batch <= 16 && tokens > 0 && tokens <= 512 && options > 0 && options <= 64,
                    "invalid decision fixture dimensions");
            Device<int> ids(batch * tokens), mask(batch * tokens), types(batch * tokens);
            Device<int> markers(batch * options), marker_mask(batch * options), qtype(batch);
            Device<float> d_logits(batch * options), d_act(batch * config.n_act), d_values(batch);
            Device<int> targets(batch);
            Device<float> temperatures(batch), losses(batch);
            ids.load(input); mask.load(input); types.load(input);
            markers.load(input); marker_mask.load(input); qtype.load(input);
            if (supervised) {
                targets.load(input); temperatures.load(input);
            } else {
                d_logits.load(input); d_act.load(input); d_values.load(input);
            }
            require(input.peek() == std::char_traits<char>::eof(), "trailing decision fixture input");
            pretrained::DecisionModel model(config, batch + 1, tokens + 3, options + 2);
            load_parameters(model, weights);
            model.zero_grad();
            auto prediction = model.forward(ids.data, mask.data, types.data, markers.data,
                                            marker_mask.data, qtype.data, batch, tokens, options);
            gpu(cudaDeviceSynchronize());
            output.write("PUFPDO1\0", 8);
            write<uint32_t>(output, batch); write<uint32_t>(output, options);
            write<uint32_t>(output, config.n_act); write<uint32_t>(output, config.value_head);
            dump(output, prediction.logits, batch * options);
            dump(output, prediction.act_logits, batch * config.n_act);
            if (config.value_head) dump(output, prediction.values, batch);
            if (supervised) {
                pretrained::decision_loss(prediction.logits, marker_mask.data, targets.data,
                    temperatures.data, losses.data, d_logits.data, batch, options);
                dump(output, losses.data, batch);
                dump(output, d_logits.data, batch * options);
                model.backward(d_logits.data);
            } else {
                model.backward(d_logits.data, d_act.data, config.value_head ? d_values.data : nullptr);
            }
            gpu(cudaDeviceSynchronize());
            write<uint32_t>(output, (uint32_t)model.parameters().size());
            for (const auto& parameter : model.parameters()) {
                write<uint32_t>(output, (uint32_t)parameter.name.size());
                output.write(parameter.name.data(), parameter.name.size());
                write<uint64_t>(output, parameter.count);
                dump(output, parameter.grad, parameter.count);
            }
        }
        require(input.peek() == std::char_traits<char>::eof(), "trailing decision fixture input");
        output.flush(); require((bool)output, "decision output write failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pretrained decision parity: " << error.what() << '\n';
        return 1;
    }
}
