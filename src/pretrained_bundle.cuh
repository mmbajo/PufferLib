#ifndef PUFFER_PRETRAINED_BUNDLE_CUH
#define PUFFER_PRETRAINED_BUNDLE_CUH

#include "pretrained_encoder.cuh"
#include "pretrained_json.h"
#include "pretrained_tokenizer.h"
#include <cstring>
#include <fcntl.h>
#include <map>
#include <memory>
#include <set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace pretrained {

// Offline-imported safetensors stay in their original dtype on disk. File maps
// are read-only; upload uses a bounded conversion buffer, never a full FP32 copy.
class Bundle {
public:
    explicit Bundle(const std::string& path);
    Config encoder;
    std::string kind, tokenizer_path, tokenizer_config_path;
    int head_layers = 2, n_act = 2, max_len = 512, head_max_len = 192;
    std::vector<float> temperature{1, 1, 1};
    std::vector<float> checkpoint_temperature;
    std::map<std::string, float> temperature_by_options;
    std::string cls_token, sep_token, mask_token, pad_token;
    uint32_t cls_id = 0, sep_id = 0, mask_id = 0, pad_id = 0;
    // Source classifier/pooler weights deliberately not used by encoder imports.
    std::vector<std::string> ignored_parameters;

    // All registry names are canonical (encoder.* plus Laya head names).
    // Validates every source tensor for finiteness before any GPU mutation.
    void validate_parameters(const std::vector<Parameter>& parameters) const;
    void load_parameters(const std::vector<Parameter>& parameters,
        cudaStream_t stream = nullptr) const;
    std::vector<float> read_parameter(const std::string& canonical_name) const;

private:
    struct MappedFile;
    struct Tensor {
        std::vector<int> shape;
        std::string dtype, file;
        size_t count = 0, start = 0, bytes = 0;
        std::shared_ptr<MappedFile> mapping;
    };
    std::string root_, source_encoder_prefix_;
    std::map<std::string, Tensor> tensors_;
    std::string source_name(const std::string& canonical_name) const;
    bool loadable(const std::string& canonical_name) const;
    static float value(const Tensor& tensor, size_t index);
    static void finite(const Tensor& tensor, const std::string& name);
    void parse_config(const cJSON* config);
    void read_weights(const cJSON* manifest);
};

namespace bundle_detail {
inline void check(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error("pretrained bundle: " + message);
}
inline bool starts(const std::string& text, const std::string& prefix) {
    return text.compare(0, prefix.size(), prefix) == 0;
}
inline std::string sibling(const std::string& root, const std::string& file) {
    check(!file.empty() && file != "." && file != ".." &&
        file.find_first_of("/\\") == std::string::npos, "bundle files must be siblings");
    return root + "/" + file;
}
inline std::vector<int> shape(const cJSON* node) {
    check(cJSON_IsArray(node), "tensor shape must be an array");
    const int rank = cJSON_GetArraySize(node);
    check(rank >= 1 && rank <= 4, "unsupported tensor rank");
    std::vector<int> result;
    for (const auto* item = node->child; item; item = item->next)
        result.push_back(static_cast<int>(json::as_integer(item, 1, INT32_MAX)));
    return result;
}
inline size_t count(const std::vector<int>& dims) {
    size_t result = 1;
    for (int dim : dims) {
        check(dim > 0 && result <= std::numeric_limits<size_t>::max() / size_t(dim),
            "tensor element count overflows");
        result *= size_t(dim);
    }
    return result;
}
inline std::string special(const cJSON* config, const char* key) {
    const auto* value = json::required(config, key);
    return cJSON_IsObject(value) ? json::string(value, "content") : json::as_string(value);
}
inline uint32_t special_id(const Tokenizer& tokenizer, const std::string& token, int vocab) {
    const auto id = tokenizer.TokenId(token);
    check(!token.empty() && id.has_value() && *id < static_cast<uint32_t>(vocab),
        "special token absent or outside embedding vocabulary: " + token);
    return *id;
}
inline bool generic_head(const std::string& name, const std::string& encoder_prefix) {
    if (starts(name, encoder_prefix + "pooler.")) return true;
    for (const char* prefix : {"cls.", "classifier.", "head.", "decoder.", "lm_head."})
        if (starts(name, prefix)) return true;
    return false;
}
} // namespace bundle_detail

struct Bundle::MappedFile {
    int fd = -1;
    size_t size = 0;
    const unsigned char* data = nullptr;
    explicit MappedFile(const std::string& path) {
        using bundle_detail::check;
        fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        check(fd >= 0, "cannot open " + path);
        struct stat info{};
        if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size < 10 ||
                uint64_t(info.st_size) > std::numeric_limits<size_t>::max()) {
            close(fd); fd = -1;
            throw std::runtime_error("pretrained bundle: invalid safetensors file " + path);
        }
        size = static_cast<size_t>(info.st_size);
        void* mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) {
            close(fd); fd = -1;
            throw std::runtime_error("pretrained bundle: cannot map " + path);
        }
        data = static_cast<const unsigned char*>(mapped);
    }
    ~MappedFile() {
        if (data) munmap(const_cast<unsigned char*>(data), size);
        if (fd >= 0) close(fd);
    }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
};

inline Bundle::Bundle(const std::string& path) : root_(path) {
    using namespace bundle_detail;
    uint32_t endian = 1;
    check(*reinterpret_cast<unsigned char*>(&endian) == 1, "little-endian host required");
    const auto manifest = json::read(root_ + "/manifest.json");
    const auto* m = manifest.get();
    check(json::string(m, "format") == "puffer-pretrained-v1", "unsupported manifest format");
    kind = json::string(m, "kind");
    check(kind == "laya" || kind == "encoder", "unsupported bundle kind");
    source_encoder_prefix_ = json::string(m, "source_encoder_prefix");
    check(kind != "laya" || source_encoder_prefix_ == "encoder.", "invalid Laya encoder namespace");
    const auto config = json::read(sibling(root_, json::string(m, "encoder_config")));
    parse_config(config.get());
    check(json::string(m, "family") == encoder.family, "manifest/config model family mismatch");
    const auto decision = json::read(sibling(root_, json::string(m, "decision_config")));
    const auto* d = decision.get();
    head_layers = json::integer(d, "head_layers", 2, 0, 32);
    max_len = json::integer(d, "max_len", 512, 1, encoder.max_positions);
    head_max_len = json::integer(d, "head_max_len", 192, 1, max_len);
    if (const auto* costs = json::optional(d, "act_costs")) {
        check(cJSON_IsObject(costs), "act_costs must be an object");
        n_act = 1 + cJSON_GetArraySize(costs);
    }
    n_act = json::integer(d, "n_act", n_act, 1, 256);
    if (const auto* temps = json::optional(d, "temperature")) {
        check(cJSON_IsArray(temps) && cJSON_GetArraySize(temps) == 3, "temperature requires three values");
        for (int i = 0; i < 3; ++i) {
            temperature[i] = static_cast<float>(json::as_number(cJSON_GetArrayItem(temps, i)));
            check(std::isfinite(temperature[i]) && temperature[i] > 0, "invalid temperature");
        }
    }
    if (const auto* temps = json::optional(d, "temperature_by_options")) {
        check(cJSON_IsObject(temps), "temperature_by_options must be an object");
        for (const auto* item = temps->child; item; item = item->next) {
            const float t = static_cast<float>(json::as_number(item));
            check(std::isfinite(t) && t > 0, "invalid option-count temperature");
            temperature_by_options.emplace(item->string, t);
        }
    }
    tokenizer_path = sibling(root_, json::string(m, "tokenizer"));
    tokenizer_config_path = sibling(root_, json::string(m, "tokenizer_config"));
    const auto tokenizer_config = json::read(tokenizer_config_path);
    Tokenizer tokenizer(tokenizer_path);
    check(tokenizer.MaxTokenId() < static_cast<uint32_t>(encoder.vocab),
        "tokenizer contains IDs outside embedding vocabulary");
    cls_token = special(tokenizer_config.get(), "cls_token");
    sep_token = special(tokenizer_config.get(), "sep_token");
    mask_token = special(tokenizer_config.get(), "mask_token");
    pad_token = special(tokenizer_config.get(), "pad_token");
    cls_id = special_id(tokenizer, cls_token, encoder.vocab);
    sep_id = special_id(tokenizer, sep_token, encoder.vocab);
    mask_id = special_id(tokenizer, mask_token, encoder.vocab);
    pad_id = special_id(tokenizer, pad_token, encoder.vocab);
    // The tokenizer determines packing IDs. Keep the encoder's own padding_idx
    // for embedding gradients: replacing it would change the imported model.
    read_weights(m);
    if (kind == "laya") {
        const auto found = tensors_.find("temperature");
        check(found != tensors_.end() && found->second.shape == std::vector<int>{3},
            "Laya temperature buffer must have shape [3]");
        checkpoint_temperature = read_parameter("temperature");
        for (float t : checkpoint_temperature) check(t > 0, "invalid source temperature buffer");
    }
}

inline void Bundle::parse_config(const cJSON* c) {
    using bundle_detail::check;
    encoder.family = json::string(c, "model_type");
    check(encoder.family == "bert" || encoder.family == "modernbert", "unsupported encoder family");
    encoder.width = static_cast<int>(json::as_integer(json::required(c, "hidden_size"), 1, 16384));
    encoder.layers = static_cast<int>(json::as_integer(json::required(c, "num_hidden_layers"), 1, 128));
    encoder.heads = static_cast<int>(json::as_integer(json::required(c, "num_attention_heads"), 1, 16384));
    encoder.intermediate = static_cast<int>(json::as_integer(json::required(c, "intermediate_size"), 1, 65536));
    encoder.vocab = static_cast<int>(json::as_integer(json::required(c, "vocab_size"), 1, INT32_MAX));
    encoder.max_positions = static_cast<int>(json::as_integer(json::required(c, "max_position_embeddings"), 1, INT32_MAX));
    const auto* padding = json::optional(c, "pad_token_id");
    encoder.pad_token_id = padding && !cJSON_IsNull(padding) ?
        static_cast<int>(json::as_integer(padding, 0, encoder.vocab - 1)) : -1;
    check(encoder.width % encoder.heads == 0, "hidden size must divide into attention heads");
    if (const auto* dimension = json::optional(c, "head_dim")) {
        if (!cJSON_IsNull(dimension))
            check(json::as_integer(dimension, 1, 16384) == encoder.width / encoder.heads,
                "head_dim must match hidden_size/num_attention_heads");
    }
    check(!json::boolean(c, "is_decoder", false) && !json::boolean(c, "add_cross_attention", false),
        "decoder/cross-attention encoders are unsupported");
    if (encoder.family == "bert") {
        check(json::string(c, "position_embedding_type", "absolute") == "absolute", "BERT relative positions unsupported");
        check(json::string(c, "hidden_act", "gelu") == "gelu", "BERT requires exact GELU");
        encoder.type_vocab = json::integer(c, "type_vocab_size", 2, 1);
        encoder.epsilon = json::number(c, "layer_norm_eps", 1e-12f);
        encoder.attention_bias = encoder.mlp_bias = encoder.norm_bias = true;
    } else {
        check((encoder.width / encoder.heads) % 2 == 0, "ModernBERT head dimension must be even");
        check(json::string(c, "hidden_activation", "gelu") == "gelu", "ModernBERT requires GELU gating");
        encoder.epsilon = json::number(c, "norm_eps", json::number(c, "layer_norm_eps", 1e-5f));
        encoder.attention_bias = json::boolean(c, "attention_bias", false);
        encoder.mlp_bias = json::boolean(c, "mlp_bias", false);
        encoder.norm_bias = json::boolean(c, "norm_bias", false);
        encoder.local_window = json::integer(c, "local_attention", 128, 1);
        encoder.global_every = json::integer(c, "global_attn_every_n_layers", 3, 1);
        encoder.global_rope_theta = json::number(c, "global_rope_theta", 160000.f);
        encoder.local_rope_theta = json::number(c, "local_rope_theta", 10000.f);
        if (const auto* scaled = json::optional(c, "rope_scaling"))
            check(cJSON_IsNull(scaled), "scaled rotary positions unsupported");
        if (const auto* ropes = json::optional(c, "rope_parameters")) {
            check(cJSON_IsObject(ropes), "rope_parameters must be an object");
            for (const auto* item = ropes->child; item; item = item->next) {
                check(std::string(item->string) == "full_attention" ||
                    std::string(item->string) == "sliding_attention", "unknown rotary attention type");
                check(json::string(item, "rope_type", "default") == "default", "scaled rotary positions unsupported");
                float& theta = std::string(item->string) == "full_attention" ?
                    encoder.global_rope_theta : encoder.local_rope_theta;
                theta = json::number(item, "rope_theta", theta);
            }
        }
        if (const auto* layers = json::optional(c, "layer_types")) {
            check(cJSON_IsArray(layers) && cJSON_GetArraySize(layers) == encoder.layers, "invalid layer_types length");
            for (const auto* item = layers->child; item; item = item->next) {
                const std::string type = json::as_string(item);
                check(type == "full_attention" || type == "sliding_attention", "unknown attention layer type");
                encoder.layer_types.push_back(type == "sliding_attention");
            }
        }
        check(encoder.global_rope_theta > 0 && encoder.local_rope_theta > 0, "invalid rotary base");
    }
    check(encoder.epsilon > 0, "normalization epsilon must be positive");
}

inline void Bundle::read_weights(const cJSON* manifest) {
    using namespace bundle_detail;
    const auto* specs = json::required(manifest, "tensors");
    check(cJSON_IsObject(specs), "manifest tensors must be an object");
    std::set<std::string> files;
    const auto* weights = json::array(manifest, "weights");
    check(cJSON_GetArraySize(weights) > 0, "no safetensors shards");
    for (const auto* item = weights->child; item; item = item->next) {
        const std::string name = json::string(item, "file");
        check(files.insert(name).second, "duplicate weight file");
        auto mapping = std::make_shared<MappedFile>(sibling(root_, name));
        uint64_t header_bytes = 0;
        std::memcpy(&header_bytes, mapping->data, 8);
        check(header_bytes >= 2 && header_bytes <= (32u << 20) && header_bytes <= mapping->size - 8,
            "invalid safetensors header size");
        const size_t data_start = 8 + static_cast<size_t>(header_bytes);
        const auto header = json::parse(std::string(reinterpret_cast<const char*>(mapping->data + 8), header_bytes));
        check(cJSON_IsObject(header.get()), "safetensors header must be object");
        std::vector<std::pair<size_t, size_t>> intervals;
        for (const auto* tensor = header->child; tensor; tensor = tensor->next) {
            const std::string key = tensor->string;
            if (key == "__metadata__") {
                check(cJSON_IsObject(tensor), "invalid safetensors metadata");
                for (const auto* entry = tensor->child; entry; entry = entry->next)
                    check(cJSON_IsString(entry), "safetensors metadata values must be strings");
                continue;
            }
            Tensor t;
            t.mapping = mapping;
            t.file = name;
            t.dtype = json::string(tensor, "dtype");
            check(t.dtype == "F32" || t.dtype == "F16" || t.dtype == "BF16", "unsupported tensor dtype: " + key);
            t.shape = shape(json::required(tensor, "shape"));
            t.count = count(t.shape);
            const size_t element_bytes = t.dtype == "F32" ? 4 : 2;
            check(t.count <= std::numeric_limits<size_t>::max() / element_bytes, "tensor byte count overflows");
            t.bytes = t.count * element_bytes;
            const auto* offsets = json::array(tensor, "data_offsets");
            check(cJSON_GetArraySize(offsets) == 2, "invalid tensor offsets");
            const size_t begin = static_cast<size_t>(json::as_integer(cJSON_GetArrayItem(offsets, 0)));
            const size_t end = static_cast<size_t>(json::as_integer(cJSON_GetArrayItem(offsets, 1)));
            check(begin <= end && end <= mapping->size - data_start && end - begin == t.bytes,
                "tensor offsets/shape mismatch: " + key);
            t.start = data_start + begin;
            intervals.emplace_back(begin, end);
            const auto* advertised = json::required(specs, key.c_str());
            check(json::string(advertised, "dtype") == t.dtype &&
                json::string(advertised, "file") == name &&
                shape(json::required(advertised, "shape")) == t.shape,
                "manifest/header tensor mismatch: " + key);
            check(tensors_.emplace(key, std::move(t)).second, "duplicate tensor across shards: " + key);
        }
        std::sort(intervals.begin(), intervals.end());
        size_t previous = 0;
        for (const auto& interval : intervals) {
            check(interval.first == previous, "overlapping or noncontiguous safetensors data");
            previous = interval.second;
        }
        check(previous == mapping->size - data_start, "unindexed/trailing safetensors data");
    }
    check(!tensors_.empty() && tensors_.size() == static_cast<size_t>(cJSON_GetArraySize(specs)),
        "manifest/source tensor count mismatch");
    if (kind == "encoder") {
        for (const auto& item : tensors_)
            if (generic_head(item.first, source_encoder_prefix_)) ignored_parameters.push_back(item.first);
    }
}

inline std::string Bundle::source_name(const std::string& name) const {
    if (bundle_detail::starts(name, "encoder.")) return source_encoder_prefix_ + name.substr(8);
    return name;
}
inline bool Bundle::loadable(const std::string& name) const {
    if (name == "value_head.weight" || name == "value_head.bias") return false;
    return kind == "laya" || bundle_detail::starts(name, "encoder.");
}
inline float Bundle::value(const Tensor& t, size_t index) {
    const unsigned char* data = t.mapping->data + t.start;
    uint32_t bits = 0;
    if (t.dtype == "F32") {
        std::memcpy(&bits, data + index * 4, 4);
    } else {
        uint16_t half = 0;
        std::memcpy(&half, data + index * 2, 2);
        if (t.dtype == "BF16") bits = uint32_t(half) << 16;
        else {
            const uint32_t sign = uint32_t(half & 0x8000) << 16;
            const uint32_t exponent = (half >> 10) & 31;
            const uint32_t mantissa = half & 1023;
            if (exponent == 0) {
                const float result = std::ldexp(static_cast<float>(mantissa), -24);
                return sign ? -result : result;
            }
            bits = sign | ((exponent == 31 ? 255 : exponent + 112) << 23) | (mantissa << 13);
        }
    }
    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}
inline void Bundle::finite(const Tensor& t, const std::string& name) {
    const unsigned char* data = t.mapping->data + t.start;
    if (t.dtype == "F32") {
        for (size_t i = 0; i < t.count; ++i) {
            uint32_t bits;
            std::memcpy(&bits, data + 4 * i, 4);
            if ((bits & 0x7f800000u) == 0x7f800000u)
                bundle_detail::check(false, "nonfinite tensor: " + name);
        }
    } else {
        const uint16_t mask = t.dtype == "F16" ? 0x7c00 : 0x7f80;
        for (size_t i = 0; i < t.count; ++i) {
            uint16_t bits;
            std::memcpy(&bits, data + 2 * i, 2);
            if ((bits & mask) == mask) bundle_detail::check(false, "nonfinite tensor: " + name);
        }
    }
}
inline std::vector<float> Bundle::read_parameter(const std::string& name) const {
    const auto found = tensors_.find(source_name(name));
    bundle_detail::check(found != tensors_.end(), "missing tensor: " + name);
    finite(found->second, name);
    std::vector<float> result(found->second.count);
    for (size_t i = 0; i < result.size(); ++i) result[i] = value(found->second, i);
    return result;
}
inline void Bundle::validate_parameters(const std::vector<Parameter>& parameters) const {
    using namespace bundle_detail;
    std::set<std::string> names, loaded;
    for (const auto& p : parameters) {
        check(names.insert(p.name).second, "duplicate destination parameter: " + p.name);
        check(p.count == count(p.shape), "destination shape/count mismatch: " + p.name);
        if (!loadable(p.name)) {
            check(p.name == "value_head.weight" || p.name == "value_head.bias" ||
                starts(p.name, "head.layers.") || starts(p.name, "scorer.") ||
                starts(p.name, "act_head.") || p.name == "type_emb.weight",
                "unknown new decision parameter: " + p.name);
            continue;
        }
        const std::string source = source_name(p.name);
        const auto found = tensors_.find(source);
        check(found != tensors_.end(), "missing source parameter: " + source);
        check(found->second.shape == p.shape && found->second.count == p.count,
            "source/destination shape mismatch: " + p.name);
        check(loaded.insert(source).second, "duplicate mapped source parameter: " + source);
    }
    check(!loaded.empty(), "no encoder parameters requested");
    for (const auto& item : tensors_) {
        const auto& name = item.first;
        const bool allowed_buffer = kind == "laya" && name == "temperature";
        const bool ignored_head = kind == "encoder" && generic_head(name, source_encoder_prefix_);
        check(loaded.count(name) || allowed_buffer || ignored_head, "unexpected/unused source parameter: " + name);
        finite(item.second, name);
    }
}
inline void Bundle::load_parameters(const std::vector<Parameter>& parameters, cudaStream_t stream) const {
    using bundle_detail::check;
    validate_parameters(parameters);
    // Validate destination pointers before the first upload too.
    for (const auto& p : parameters)
        if (loadable(p.name)) check(p.data != nullptr, "null destination parameter: " + p.name);
    decision::transformer_detail::cuda_check(cudaStreamSynchronize(stream));
    constexpr size_t chunk_elements = 1u << 20;
    std::vector<float> staging(chunk_elements);
    for (const auto& p : parameters) {
        if (!loadable(p.name)) continue;
        const Tensor& tensor = tensors_.at(source_name(p.name));
        for (size_t offset = 0; offset < p.count; offset += chunk_elements) {
            const size_t length = std::min(chunk_elements, p.count - offset);
            for (size_t i = 0; i < length; ++i) staging[i] = value(tensor, offset + i);
            const auto copy = cudaMemcpyAsync(p.data + offset, staging.data(), length * sizeof(float),
                cudaMemcpyHostToDevice, stream);
            // Always finish any queued copy before the staging allocation dies.
            const auto sync = cudaStreamSynchronize(stream);
            decision::transformer_detail::cuda_check(copy);
            decision::transformer_detail::cuda_check(sync);
        }
    }
}

} // namespace pretrained
#endif
