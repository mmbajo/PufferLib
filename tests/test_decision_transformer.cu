// Test executable only; no production dependency on Python or PyTorch.
#include "../src/decision_checkpoint.cuh"

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using decision::checkpoint::check;
using decision::checkpoint::read;
using decision::checkpoint::write;

template <class T> struct DeviceArray {
    T* data = nullptr;
    size_t count;
    explicit DeviceArray(size_t n) : count(n) {
        decision::transformer_detail::cuda_check(cudaMalloc(&data, n * sizeof(T)));
    }
    ~DeviceArray() { cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    void load(std::istream& input) {
        std::vector<T> values(count);
        input.read(reinterpret_cast<char*>(values.data()), count * sizeof(T));
        check((bool)input, "truncated test input");
        decision::transformer_detail::cuda_check(cudaMemcpy(data, values.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
};

static void dump_floats(std::ostream& output, const float* device, size_t count) {
    std::vector<float> values(count, 0);
    if (device) decision::transformer_detail::cuda_check(cudaMemcpy(values.data(), device,
        count * sizeof(float), cudaMemcpyDeviceToHost));
    output.write(reinterpret_cast<const char*>(values.data()), count * sizeof(float));
    check((bool)output, "test output write failed");
}

// Deliberately simple test-only SGD: production uses PufferLib's optimizer.
__global__ void squared_error_gradient(const float* output, const float* target,
        float* gradient, int count, int batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) gradient[i] = (output[i] - target[i]) / batch;
}
__global__ void sgd_update(float* weight, const float* gradient, size_t count, float rate) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) weight[i] -= rate * gradient[i];
}

// Hold a nonblocking stream long enough for an incorrectly ordered default-
// stream checkpoint copy to observe the preceding parameter revision.
__global__ void checkpoint_delay(unsigned long long cycles) {
    unsigned long long begin = clock64();
    while (clock64() - begin < cycles) {}
}
__global__ void checkpoint_fill(float* data, size_t count, float value) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) data[i] = value;
}

static void checkpoint_case(const std::string& initial_path,
        const std::string& corrupt_path, const std::string& output_path) {
    using decision::transformer_detail::cuda_check;
    struct Stream {
        cudaStream_t value = nullptr;
        Stream() { cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
        ~Stream() { cudaStreamDestroy(value); }
    } stream;
    decision::Model model(decision::checkpoint::config(initial_path), 1, 1, stream.value);
    decision::checkpoint::load(model, initial_path);
    cudaDeviceProp properties;
    int device;
    cuda_check(cudaGetDevice(&device));
    cuda_check(cudaGetDeviceProperties(&properties, device));
    const unsigned long long delay_cycles = static_cast<unsigned long long>(properties.clockRate) * 100;
    auto fill = [&](bool expected) {
        checkpoint_delay<<<1, 1, 0, stream.value>>>(delay_cycles);
        for (size_t i = 0; i < model.parameters().size(); ++i) {
            const auto& p = model.parameters()[i];
            float value = expected ? (i + 1) * 0.03125f : -100.0f;
            checkpoint_fill<<<(p.count + 255) / 256, 256, 0, stream.value>>>(p.data, p.count, value);
        }
        cuda_check(cudaGetLastError());
    };
    auto verify = [&]() {
        cuda_check(cudaStreamSynchronize(stream.value));
        for (size_t i = 0; i < model.parameters().size(); ++i) {
            const auto& p = model.parameters()[i];
            std::vector<float> host(p.count);
            cuda_check(cudaMemcpy(host.data(), p.data, p.count * sizeof(float), cudaMemcpyDeviceToHost));
            for (float value : host)
                check(value == (i + 1) * 0.03125f, "checkpoint did not preserve the ordered parameter revision");
        }
    };
    fill(true);
    decision::checkpoint::save(model, output_path + ".checkpoint");
    // These writes must complete before the checkpoint is loaded; they must not
    // run afterward and silently overwrite the restored parameters.
    fill(false);
    decision::checkpoint::load(model, output_path + ".checkpoint");
    verify();
    bool rejected = false;
    try {
        decision::checkpoint::load(model, corrupt_path);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("nonfinite") != std::string::npos;
    }
    check(rejected, "checkpoint with a nonfinite final tensor was not rejected");
    verify(); // Earlier valid tensors from the rejected file must not be applied.
}

static void model_case(const std::string& mode, const std::string& checkpoint,
        std::istream& input, std::ostream& output, const std::string& output_path) {
    char magic[8]; input.read(magic, 8);
    check(std::string(magic, 8) == std::string("PUFDTI1\0", 8), "invalid test input magic");
    uint32_t batch = read<uint32_t>(input), iterations = read<uint32_t>(input);
    float learning_rate = read<float>(input);
    (void)read<float>(input); // Reserved for test input format compatibility.
    check(batch > 0 && batch <= 16 && iterations > 0 && iterations <= 100, "invalid test dimensions");
    DeviceArray<int> boards(batch * 100);
    DeviceArray<float> dlogits(batch * 4), dvalues(batch);
    boards.load(input); dlogits.load(input); dvalues.load(input);

    // Model calls below intentionally use the same public interface as the
    // production architecture adapter; tensors are CUDA buffers throughout.
    decision::Config config = decision::checkpoint::config(checkpoint);
    std::vector<std::unique_ptr<DeviceArray<float>>> external_data, external_gradients;
    // Active batches need not fill the workspace's allocated capacity.
    auto owner = std::make_unique<decision::Model>(config, batch + 2);
    auto& model = *owner;
    decision::checkpoint::load(model, checkpoint);
    if (mode == "bound") {
        std::vector<float*> data, gradients;
        for (const auto& parameter : model.parameters()) {
            external_data.emplace_back(new DeviceArray<float>(parameter.count));
            external_gradients.emplace_back(new DeviceArray<float>(parameter.count));
            decision::transformer_detail::cuda_check(cudaMemcpy(external_data.back()->data,
                parameter.data, parameter.count * sizeof(float), cudaMemcpyDeviceToDevice));
            data.push_back(external_data.back()->data);
            gradients.push_back(external_gradients.back()->data);
        }
        model.bind_parameters(data, gradients);
    }
    decision::checkpoint::save(model, output_path + ".checkpoint");
    DeviceArray<float> loss_logits(batch * 4), loss_values(batch);

    output.write("PUFDTR1\0", 8);
    write<uint32_t>(output, batch);
    write<uint32_t>(output, iterations);
    write<uint32_t>(output, (uint32_t)model.parameters().size());
    for (uint32_t step = 0; step < iterations; ++step) {
        model.zero_grad();
        auto prediction = model.forward(boards.data, batch);
        if (mode == "overfit") {
            squared_error_gradient<<<(batch*4+255)/256,256>>>(prediction.logits,
                dlogits.data, loss_logits.data, batch*4, batch);
            squared_error_gradient<<<(batch+255)/256,256>>>(prediction.values,
                dvalues.data, loss_values.data, batch, batch);
            model.backward(loss_logits.data, loss_values.data);
        } else {
            model.backward(dlogits.data, dvalues.data);
            if (mode == "accumulate") model.backward(dlogits.data, dvalues.data);
        }
        decision::transformer_detail::cuda_check(cudaDeviceSynchronize());
        dump_floats(output, prediction.logits, batch * 4);
        dump_floats(output, prediction.values, batch);
        write<float>(output, 0); // Reserved result slot.
        if (mode == "overfit") {
            for (const auto& parameter : model.parameters())
                sgd_update<<<(parameter.count+255)/256,256>>>(parameter.data,
                    parameter.grad, parameter.count, learning_rate);
        }
        decision::transformer_detail::cuda_check(cudaDeviceSynchronize());
        for (size_t index = 0; index < model.parameters().size(); ++index) {
            const auto& parameter = model.parameters()[index];
            write<uint32_t>(output, (uint32_t)parameter.name.size());
            output.write(parameter.name.data(), parameter.name.size());
            write<uint64_t>(output, parameter.count);
            dump_floats(output, parameter.grad, parameter.count);
            dump_floats(output, parameter.data, parameter.count);
        }
    }
    owner.reset();
    if (!external_data.empty()) {
        // Puffer owns these buffers. Destroying a bound model must leave both
        // its parameters and gradients alive for the trainer's allocator.
        float sample;
        decision::transformer_detail::cuda_check(cudaMemcpy(&sample, external_data[0]->data,
            sizeof(float), cudaMemcpyDeviceToHost));
        decision::transformer_detail::cuda_check(cudaMemcpy(&sample, external_gradients[0]->data,
            sizeof(float), cudaMemcpyDeviceToHost));
    }
}

int main(int argc, char** argv) {
    try {
        check(argc == 5, "usage: native-test forward|bound|accumulate|overfit|checkpoint checkpoint input output");
        std::ifstream input(argv[3], std::ios::binary);
        std::ofstream output(argv[4], std::ios::binary | std::ios::trunc);
        check((bool)input && (bool)output, "cannot open native test files");
        std::string mode = argv[1];
        check(mode == "forward" || mode == "bound" || mode == "accumulate" || mode == "overfit" || mode == "checkpoint",
            "unknown test mode");
        if (mode == "checkpoint") {
            checkpoint_case(argv[2], argv[3], argv[4]);
            output << "checkpoint stream and corruption checks passed\n";
        } else {
            model_case(mode, argv[2], input, output, argv[4]);
        }
        output.flush(); check((bool)output, "native test output flush failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "decision transformer test: " << error.what() << '\n';
        return 1;
    }
}
