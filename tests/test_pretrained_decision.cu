#include "pretrained_fixture.h"
#include "../src/pretrained_decision.cuh"

int main(int argc, char** argv) {
    try {
        require(argc == 4, "usage: decision-test weights.bin input.bin output.bin");
        std::ifstream weights(argv[1], std::ios::binary), input(argv[2], std::ios::binary);
        std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
        require((bool)weights && (bool)input && (bool)output, "cannot open decision fixture files");
        pretrained::DecisionConfig config;
        config.encoder = load_config(weights);
        require(read_string(input, 8) == std::string("PUFPDI1\0", 8), "invalid decision input magic");
        config.head_layers = read<uint32_t>(input);
        config.n_act = read<uint32_t>(input);
        config.value_head = read<uint32_t>(input);
        int batch = read<uint32_t>(input), tokens = read<uint32_t>(input), options = read<uint32_t>(input);
        require(batch > 0 && batch <= 16 && tokens > 0 && tokens <= 512 && options > 0 && options <= 64,
                "invalid decision fixture dimensions");
        Device<int> ids(batch * tokens), mask(batch * tokens), types(batch * tokens);
        Device<int> markers(batch * options), marker_mask(batch * options), qtype(batch);
        Device<float> d_logits(batch * options), d_act(batch * config.n_act), d_values(batch);
        ids.load(input); mask.load(input); types.load(input);
        markers.load(input); marker_mask.load(input); qtype.load(input);
        d_logits.load(input); d_act.load(input); d_values.load(input);
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
        model.backward(d_logits.data, d_act.data, config.value_head ? d_values.data : nullptr);
        gpu(cudaDeviceSynchronize());
        write<uint32_t>(output, (uint32_t)model.parameters().size());
        for (const auto& parameter : model.parameters()) {
            write<uint32_t>(output, (uint32_t)parameter.name.size());
            output.write(parameter.name.data(), parameter.name.size());
            write<uint64_t>(output, parameter.count);
            dump(output, parameter.grad, parameter.count);
        }
        output.flush(); require((bool)output, "decision output write failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pretrained decision parity: " << error.what() << '\n';
        return 1;
    }
}
