// Verify a native Snake PPO run updates the imported model and critic while
// preserving the unused act/escalate head. Also supports encoder-only imports
// with freshly initialized decision heads. Uses no GPU at runtime.
#include "../src/pretrained_bundle.cuh"
#include "../src/pretrained_decision.cuh"
#include <fstream>
#include <iostream>

struct Changes {
    size_t tensors = 0, changed_tensors = 0, changed_values = 0;
    double maximum_delta = 0;
};

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 4)
            throw std::invalid_argument("usage: test_laya_checkpoint BUNDLE PUFFER_CHECKPOINT [INITIAL_SEED=73]");
        pretrained::Bundle bundle(argv[1]);
        pretrained::DecisionConfig config{bundle.encoder, bundle.head_layers, bundle.n_act, true};
        auto registry = pretrained::decision_parameter_specs(config);
        bundle.validate_parameters(registry);
        size_t expected = 0;
        for (const auto& p : registry) {
            if (p.count % 4 && p.shape.size() != 1) throw std::logic_error("unaligned matrix registry");
            expected += ((p.count + 3) & ~size_t(3)) * sizeof(float);
        }
        std::ifstream input(argv[2], std::ios::binary | std::ios::ate);
        if (!input || input.tellg() < 0 || static_cast<size_t>(input.tellg()) != expected)
            throw std::runtime_error("checkpoint size differs from padded native registry");
        input.seekg(0);
        std::mt19937_64 rng(argc == 4 ? std::stoull(argv[3]) : 73);
        std::normal_distribution<float> normal(0, 0.02f);
        std::map<std::string, Changes> groups;
        for (const auto& p : registry) {
            bool critic = p.name.compare(0, 11, "value_head.") == 0;
            bool fresh = critic || (bundle.kind == "encoder" && p.name.compare(0, 8, "encoder.") != 0);
            auto before = fresh ? std::vector<float>(p.count) : bundle.read_parameter(p.name);
            if (fresh) {
                bool bias = p.name.size() >= 4 && p.name.compare(p.name.size() - 4, 4, "bias") == 0;
                bool norm = p.name.find("norm") != std::string::npos || p.name == "scorer.0.weight";
                for (auto& value : before) value = bias ? 0 : norm ? 1 : normal(rng);
            }
            size_t padded = (p.count + 3) & ~size_t(3);
            std::vector<float> after(padded);
            input.read(reinterpret_cast<char*>(after.data()), padded * sizeof(float));
            if (!input) throw std::runtime_error("truncated checkpoint tensor: " + p.name);
            for (float value : after)
                if (!std::isfinite(value)) throw std::runtime_error("nonfinite checkpoint tensor: " + p.name);
            for (size_t i = p.count; i < padded; ++i)
                if (after[i] != 0) throw std::runtime_error("nonzero scalar/bias alignment padding: " + p.name);
            std::string category = critic ? "critic" : p.name.substr(0, p.name.find('.'));
            auto& changes = groups[category];
            ++changes.tensors;
            size_t changed = 0;
            for (size_t i = 0; i < p.count; ++i) {
                if (after[i] != before[i]) ++changed;
                changes.maximum_delta = std::max(changes.maximum_delta, std::abs(double(after[i]) - before[i]));
            }
            changes.changed_values += changed;
            changes.changed_tensors += changed != 0;
            if (category == "act_head" && std::memcmp(after.data(), before.data(), p.count * sizeof(float)) != 0)
                throw std::runtime_error("unused imported act/escalate head changed: " + p.name);
        }
        for (const char* category : {"encoder", "scorer", "critic"})
            if (!groups[category].changed_values)
                throw std::runtime_error(std::string("no training update reached ") + category);
        if (!groups["act_head"].tensors) throw std::runtime_error("missing act head");
        for (const auto& group : groups) {
            const auto& c = group.second;
            std::cout << group.first << ": tensors=" << c.tensors << " changed_tensors=" << c.changed_tensors
                      << " changed_values=" << c.changed_values << " max_delta=" << c.maximum_delta << '\n';
        }
        std::cout << "finite checkpoint, correct flat padding, expected training branches verified\n";
    } catch (const std::exception& error) {
        std::cerr << "checkpoint audit: " << error.what() << '\n'; return 1;
    }
}
