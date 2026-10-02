#ifndef PUFFER_TEST_PRETRAINED_FIXTURE_H
#define PUFFER_TEST_PRETRAINED_FIXTURE_H

#include "../src/pretrained_encoder.cuh"

#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
static void gpu(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> static T read(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    require((bool)in, "truncated encoder fixture");
    return value;
}
template<class T> static void write(std::ostream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
static std::string read_string(std::istream& in, uint32_t count) {
    require(count <= 4096, "invalid encoder fixture string length");
    std::string value(count, '\0');
    if (count) in.read(&value[0], count);
    require((bool)in, "truncated encoder fixture string");
    return value;
}
template<class T> struct Device {
    T* data = nullptr;
    size_t count;
    explicit Device(size_t n) : count(n) { gpu(cudaMalloc(&data, n * sizeof(T))); }
    ~Device() { cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void load(std::istream& in) {
        std::vector<T> host(count);
        in.read(reinterpret_cast<char*>(host.data()), count * sizeof(T));
        require((bool)in, "truncated encoder fixture tensor");
        gpu(cudaMemcpy(data, host.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
};
static void dump(std::ostream& out, const float* data, size_t count) {
    std::vector<float> host(count, 0);
    if (data) gpu(cudaMemcpy(host.data(), data, count * sizeof(float), cudaMemcpyDeviceToHost));
    out.write(reinterpret_cast<char*>(host.data()), count * sizeof(float));
}

static pretrained::Config load_config(std::istream& in) {
    require(read_string(in, 8) == std::string("PUFPRE1\0", 8), "invalid model fixture magic");
    int family = read<int>(in);
    require(family >= 0 && family <= 2, "invalid fixture family");
    pretrained::Config config;
    config.family = family == 0 ? "modernbert" : family == 1 ? "bert" : "laya_head";
    config.width = read<int>(in); config.layers = read<int>(in);
    config.heads = read<int>(in); config.intermediate = read<int>(in);
    config.vocab = read<int>(in); config.max_positions = read<int>(in);
    config.type_vocab = read<int>(in); config.pad_token_id = read<int>(in);
    config.epsilon = read<float>(in);
    config.attention_bias = read<int>(in); config.mlp_bias = read<int>(in); config.norm_bias = read<int>(in);
    config.local_window = read<int>(in); config.global_every = read<int>(in);
    int layer_types = read<int>(in);
    config.global_rope_theta = read<float>(in); config.local_rope_theta = read<float>(in);
    require(layer_types >= 0 && layer_types <= 128, "invalid fixture layer types");
    for (int i = 0; i < layer_types; ++i) config.layer_types.push_back(read<int>(in));
    return config;
}

template<class Model> static void load_parameters(Model& model, std::istream& in) {
    std::map<std::string, const pretrained::Parameter*> expected;
    for (const auto& parameter : model.parameters()) expected.emplace(parameter.name, &parameter);
    require(read<uint32_t>(in) == expected.size(), "fixture/native parameter count mismatch");
    while (!expected.empty()) {
        std::string name = read_string(in, read<uint32_t>(in));
        auto found = expected.find(name);
        if (found == expected.end()) throw std::runtime_error("unknown/repeated parameter: " + name);
        const auto& parameter = *found->second;
        require(read<uint32_t>(in) == parameter.shape.size(), "fixture/native tensor rank mismatch");
        for (int dim : parameter.shape)
            require(read<uint32_t>(in) == (uint32_t)dim, "fixture/native tensor shape mismatch");
        require(read<uint64_t>(in) == parameter.count, "fixture/native tensor count mismatch");
        std::vector<float> values(parameter.count);
        in.read(reinterpret_cast<char*>(values.data()), parameter.count * sizeof(float));
        require((bool)in, "truncated fixture weights");
        gpu(cudaMemcpy(parameter.data, values.data(), parameter.count * sizeof(float), cudaMemcpyHostToDevice));
        expected.erase(found);
    }
    require(in.peek() == std::char_traits<char>::eof(), "trailing fixture weights");
}


#endif
