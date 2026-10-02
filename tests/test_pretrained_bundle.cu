#include "pretrained_fixture.h"
#include "../src/pretrained_bundle.cuh"
#include "../src/pretrained_decision.cuh"

int main(int argc, char** argv) {
    try {
        require(argc == 4 || argc == 5, "usage: bundle-test dump|validate|forward BUNDLE OUTPUT [INPUT]");
        std::string mode(argv[1]);
        pretrained::Bundle bundle(argv[2]);
        pretrained::DecisionConfig config{bundle.encoder, bundle.head_layers, bundle.n_act, false};
        auto specs = pretrained::decision_parameter_specs(config);
        bundle.validate_parameters(specs);
        std::ofstream out(argv[3], std::ios::binary | std::ios::trunc);
        require((bool)out, "cannot open bundle test output");
        if (mode == "dump" || mode == "validate") {
            out.write("PUFPBD1\0", 8);
            write<uint32_t>(out, config.encoder.family == "bert" ? 1 : 0);
            for (int n : {config.encoder.width, config.encoder.layers, config.encoder.heads, config.encoder.vocab,
                          config.head_layers, config.n_act}) write<uint32_t>(out, n);
            for (auto n : {bundle.cls_id, bundle.sep_id, bundle.mask_id, bundle.pad_id}) write<uint32_t>(out, n);
            if (mode == "validate") return 0;
            std::vector<pretrained::Parameter> loaded;
            for (const auto& p : specs)
                if (bundle.kind == "laya" || p.name.compare(0, 8, "encoder.") == 0) loaded.push_back(p);
            write<uint32_t>(out, loaded.size());
            for (const auto& p : loaded) {
                write<uint32_t>(out, p.name.size()); out.write(p.name.data(), p.name.size());
                auto values = bundle.read_parameter(p.name);
                write<uint64_t>(out, values.size());
                out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
            }
        } else if (mode == "forward") {
            require(argc == 5, "forward mode requires input fixture");
            std::ifstream input(argv[4], std::ios::binary);
            require(read_string(input, 8) == std::string("PUFPDI1\0", 8), "invalid decision input magic");
            require(read<uint32_t>(input) == (uint32_t)config.head_layers, "bundle head mismatch");
            require(read<uint32_t>(input) == (uint32_t)config.n_act, "bundle acts mismatch");
            require(read<uint32_t>(input) == 0, "bundle oracle does not add a critic");
            int B = read<uint32_t>(input), T = read<uint32_t>(input), K = read<uint32_t>(input);
            require(B > 0 && B <= 16 && T > 0 && T <= 512 && K > 0 && K <= 64, "invalid input dimensions");
            Device<int> ids(B * T), mask(B * T), types(B * T), markers(B * K), marker_mask(B * K), qtype(B);
            ids.load(input); mask.load(input); types.load(input); markers.load(input); marker_mask.load(input); qtype.load(input);
            pretrained::DecisionModel model(config, B, T, K, nullptr, false);
            model.initialize_new_heads(1);
            bundle.load_parameters(model.parameters());
            auto result = model.forward(ids.data, mask.data, types.data, markers.data, marker_mask.data, qtype.data, B, T, K);
            gpu(cudaDeviceSynchronize());
            out.write("PUFPDO1\0", 8);
            write<uint32_t>(out, B); write<uint32_t>(out, K);
            write<uint32_t>(out, config.n_act); write<uint32_t>(out, 0);
            dump(out, result.logits, B * K); dump(out, result.act_logits, B * config.n_act);
        } else throw std::runtime_error("unknown bundle test mode");
        out.flush(); require((bool)out, "bundle test output write failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pretrained bundle parity: " << error.what() << '\n';
        return 1;
    }
}
