#include "pretrained_fixture.h"

int main(int argc, char** argv) {
    try {
        require(argc == 4, "usage: pretrained-test weights.bin input.bin output.bin");
        std::ifstream weights(argv[1], std::ios::binary), input(argv[2], std::ios::binary);
        std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
        require((bool)weights && (bool)input && (bool)output, "cannot open encoder test files");
        auto config = load_config(weights);
        require(read_string(input, 8) == std::string("PUFPRI1\0", 8), "invalid input fixture magic");
        int batch = read<uint32_t>(input), tokens = read<uint32_t>(input);
        require(batch > 0 && batch <= 16 && tokens > 0 && tokens <= 512, "invalid fixture dimensions");
        size_t elements = (size_t)batch * tokens * config.width;
        Device<int> ids(batch * tokens), mask(batch * tokens), types(batch * tokens);
        Device<float> hidden(elements), gradient(elements);
        ids.load(input); mask.load(input); types.load(input); hidden.load(input); gradient.load(input);
        require(input.peek() == std::char_traits<char>::eof(), "trailing fixture input");
        pretrained::Encoder model(config, batch + 1, tokens + 3);
        load_parameters(model, weights);
        model.zero_grad();
        float* prediction = config.family == "laya_head"
            ? model.forward_hidden(hidden.data, mask.data, batch, tokens)
            : model.forward(ids.data, mask.data, types.data, batch, tokens);
        gpu(cudaDeviceSynchronize());
        // Production inference binds the same weights and reuses one layer's scratch.
        pretrained::Encoder inference(config, batch + 1, tokens + 3, nullptr, false, false);
        std::vector<float*> data;
        for (const auto& parameter : model.parameters()) data.push_back(parameter.data);
        inference.bind_parameters(data);
        float* evaluated = config.family == "laya_head"
            ? inference.forward_hidden(hidden.data, mask.data, batch, tokens)
            : inference.forward(ids.data, mask.data, types.data, batch, tokens);
        std::vector<float> training_output(elements), inference_output(elements);
        gpu(cudaMemcpy(training_output.data(), prediction, elements * sizeof(float), cudaMemcpyDeviceToHost));
        gpu(cudaMemcpy(inference_output.data(), evaluated, elements * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < elements; ++i)
            require(std::abs(training_output[i] - inference_output[i]) <= 1e-6f,
                    "inference workspace differs from trainable forward");
        output.write("PUFPRO1\0", 8);
        write<uint32_t>(output, batch); write<uint32_t>(output, tokens);
        write<uint32_t>(output, config.width); write<uint32_t>(output, config.layers);
        dump(output, prediction, elements);
        for (int i = 0; i < config.layers; ++i) dump(output, model.layer_output(i), elements);
        float* input_gradient = model.backward(gradient.data);
        gpu(cudaDeviceSynchronize());
        dump(output, config.family == "laya_head" ? input_gradient : nullptr, elements);
        write<uint32_t>(output, (uint32_t)model.parameters().size());
        for (const auto& parameter : model.parameters()) {
            write<uint32_t>(output, (uint32_t)parameter.name.size());
            output.write(parameter.name.data(), parameter.name.size());
            write<uint64_t>(output, parameter.count);
            dump(output, parameter.grad, parameter.count);
        }
        output.flush(); require((bool)output, "encoder test output write failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pretrained encoder parity: " << error.what() << '\n';
        return 1;
    }
}
