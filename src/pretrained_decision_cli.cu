// Native JSONL inference for an imported Laya model or a new decision head on a
// supported pretrained encoder. No Python or PyTorch is loaded by this program.
#include "pretrained_decision.cuh"
#include "pretrained_input.h"
#include <charconv>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

namespace {
using namespace pretrained;

// cJSON stores all numbers as doubles, which loses integer precision and the
// distinction between 1 and 1.0. Keep their original lexical types for the
// Python-compatible rendering used by Laya before tokenization.
using NumberText = std::unordered_map<const cJSON*, std::string>;
std::string render_float(const std::string& token) {
    double value = std::strtod(token.c_str(), nullptr);
    json::check(std::isfinite(value), "nonfinite request number");
    if (value == 0) return std::signbit(value) ? "-0.0" : "0.0";
    char buffer[64];
    auto converted = std::to_chars(buffer, buffer + sizeof(buffer), std::abs(value), std::chars_format::scientific);
    json::check(converted.ec == std::errc(), "cannot render request number");
    std::string scientific(buffer, converted.ptr);
    size_t e = scientific.find('e');
    int exponent = std::stoi(scientific.substr(e + 1));
    std::string digits = scientific.substr(0, e);
    digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());
    std::string result;
    if (exponent >= -4 && exponent < 16) {
        int point = exponent + 1;
        if (point <= 0) result = "0." + std::string(-point, '0') + digits;
        else if (point >= static_cast<int>(digits.size()))
            result = digits + std::string(point - digits.size(), '0') + ".0";
        else result = digits.substr(0, point) + "." + digits.substr(point);
    } else {
        result = digits.substr(0, 1);
        if (digits.size() > 1) result += "." + digits.substr(1);
        auto magnitude = std::to_string(std::abs(exponent));
        result += std::string("e") + (exponent < 0 ? "-" : "+") +
            (magnitude.size() < 2 ? "0" : "") + magnitude;
    }
    return (std::signbit(value) ? "-" : "") + result;
}
void number_nodes(const cJSON* node, std::vector<const cJSON*>& nodes) {
    if (cJSON_IsNumber(node)) nodes.push_back(node);
    for (auto child = node->child; child; child = child->next) number_nodes(child, nodes);
}
NumberText request_numbers(const std::string& source, const cJSON* root) {
    std::vector<const cJSON*> nodes;
    number_nodes(root, nodes);
    NumberText numbers;
    size_t index = 0;
    const auto digit = [](char c) { return c >= '0' && c <= '9'; };
    for (size_t i = 0; i < source.size();) {
        if (source[i] == '"') {
            ++i;
            while (i < source.size()) {
                char c = source[i++];
                if (c == '\\') ++i;
                else if (c == '"') break;
            }
        } else if (source[i] == '-' || digit(source[i])) {
            size_t start = i;
            if (source[i] == '-') ++i;
            json::check(i < source.size() && digit(source[i]), "invalid JSON number");
            if (source[i] == '0') ++i;
            else while (i < source.size() && digit(source[i])) ++i;
            bool floating = false;
            if (i < source.size() && source[i] == '.') {
                floating = true; ++i;
                json::check(i < source.size() && digit(source[i]), "invalid JSON fraction");
                while (i < source.size() && digit(source[i])) ++i;
            }
            if (i < source.size() && (source[i] == 'e' || source[i] == 'E')) {
                floating = true; ++i;
                if (i < source.size() && (source[i] == '+' || source[i] == '-')) ++i;
                json::check(i < source.size() && digit(source[i]), "invalid JSON exponent");
                while (i < source.size() && digit(source[i])) ++i;
            }
            json::check(i == source.size() || std::string(" \t\r\n,]}").find(source[i]) != std::string::npos,
                "invalid JSON number suffix");
            json::check(index < nodes.size(), "request number count mismatch");
            std::string token = source.substr(start, i - start);
            numbers.emplace(nodes[index++], floating ? render_float(token) : token == "-0" ? "0" : token);
        } else ++i;
    }
    json::check(index == nodes.size(), "request number count mismatch");
    return numbers;
}

// Match Python json.dumps' separators and numeric rendering. Strings are
// passed directly to the tokenizer; structured values use this JSON form.
std::string render_json(const cJSON* value, const NumberText& numbers) {
    if (cJSON_IsNumber(value)) return numbers.at(value);
    if (!cJSON_IsArray(value) && !cJSON_IsObject(value)) {
        char* text = cJSON_PrintUnformatted(value);
        if (!text) throw std::runtime_error("cannot serialize request value");
        std::string result(text); cJSON_free(text); return result;
    }
    bool object = cJSON_IsObject(value);
    std::string result = object ? "{" : "[";
    for (auto item = value->child; item; item = item->next) {
        if (item != value->child) result += ", ";
        if (object) {
            json::Document key(cJSON_CreateString(item->string), cJSON_Delete);
            result += render_json(key.get(), numbers) + ": ";
        }
        result += render_json(item, numbers);
    }
    result += object ? "}" : "]";
    return result;
}
std::string criterion(const cJSON* value, const NumberText& numbers) {
    return cJSON_IsString(value) ? json::as_string(value) : render_json(value, numbers);
}
const cJSON* field(const cJSON* object, const char* name, const char* alias) {
    auto value = json::optional(object, name);
    return value ? value : json::optional(object, alias);
}
double rounded(double value) { return std::round(value * 10000.0) / 10000.0; }
void number(cJSON* object, const char* key, double value) {
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite model output");
    cJSON_AddNumberToObject(object, key, rounded(value));
}
template<class T> struct Device {
    T* data = nullptr;
    explicit Device(const std::vector<T>& values) {
        decision::transformer_detail::cuda_check(cudaMalloc(&data, values.size() * sizeof(T)));
        decision::transformer_detail::cuda_check(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~Device() { if (data) cudaFree(data); }
};
std::vector<float> download(const float* values, size_t size) {
    std::vector<float> result(size);
    decision::transformer_detail::cuda_check(cudaMemcpy(result.data(), values, size * sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}
void load_puffer_weights(DecisionModel& model, const std::string& path) {
    // Puffer pads each registered FP32 tensor to a 16-byte boundary. The
    // imported bundle supplies the architecture; the flat file supplies weights.
    size_t expected = 0;
    for (const auto& p : model.parameters()) expected += ((p.count + 3) & ~size_t(3)) * sizeof(float);
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() < 0 || static_cast<size_t>(input.tellg()) != expected)
        throw std::runtime_error("flat checkpoint size does not match this bundle plus its Puffer critic");
    std::vector<float> chunk(1 << 20);
    input.seekg(0);
    for (size_t offset = 0; offset < expected / sizeof(float);) {
        size_t count = std::min(chunk.size(), expected / sizeof(float) - offset);
        input.read(reinterpret_cast<char*>(chunk.data()), count * sizeof(float));
        if (!input) throw std::runtime_error("truncated flat checkpoint");
        for (size_t i = 0; i < count; ++i)
            if (!std::isfinite(chunk[i])) throw std::runtime_error("nonfinite flat checkpoint weight");
        offset += count;
    }
    // Validate every weight before modifying the live model.
    input.clear(); input.seekg(0);
    for (const auto& p : model.parameters()) {
        for (size_t offset = 0; offset < p.count;) {
            size_t count = std::min(chunk.size(), p.count - offset);
            input.read(reinterpret_cast<char*>(chunk.data()), count * sizeof(float));
            if (!input) throw std::runtime_error("flat checkpoint changed during loading");
            decision::transformer_detail::cuda_check(cudaMemcpy(p.data + offset, chunk.data(),
                count * sizeof(float), cudaMemcpyHostToDevice));
            offset += count;
        }
        input.seekg((((p.count + 3) & ~size_t(3)) - p.count) * sizeof(float), std::ios::cur);
    }
}
template<class T> void array(cJSON* object, const char* key, const std::vector<T>& values) {
    auto output = cJSON_AddArrayToObject(object, key);
    for (auto value : values) cJSON_AddItemToArray(output, cJSON_CreateNumber(value));
}

struct Question {
    std::string type, instructions;
    std::vector<std::string> options, labels, legend;
    std::vector<int> order;
};
std::string strip_label(const std::string& label) {
    // Python str.strip's Unicode whitespace set, with byte offsets retained.
    size_t first = label.size(), last = 0;
    for (size_t i = 0; i < label.size();) {
        size_t start = i;
        unsigned char byte = label[i++];
        uint32_t cp = byte;
        int continuation = (byte & 0xe0) == 0xc0 ? 1 : (byte & 0xf0) == 0xe0 ? 2 : (byte & 0xf8) == 0xf0 ? 3 : 0;
        if (continuation) {
            cp = byte & ((1u << (6 - continuation)) - 1);
            for (int n = 0; n < continuation; ++n) {
                json::check(i < label.size() && (static_cast<unsigned char>(label[i]) & 0xc0) == 0x80,
                    "invalid UTF-8 label");
                cp = (cp << 6) | (static_cast<unsigned char>(label[i++]) & 0x3f);
            }
        }
        bool space = (cp >= 9 && cp <= 13) || (cp >= 28 && cp <= 32) || cp == 0x85 || cp == 0xa0 ||
            cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 ||
            cp == 0x202f || cp == 0x205f || cp == 0x3000;
        if (!space) { first = std::min(first, start); last = i; }
    }
    return last ? label.substr(first, last - first) : "";
}
Question parse_question(const cJSON* value, const NumberText& numbers) {
    Question question;
    question.type = json::as_string(field(value, "type", "t"));
    question.instructions = json::as_string(field(value, "instructions", "ins"));
    auto labels = json::optional(value, "labels");
    if (labels && cJSON_IsNull(labels)) labels = nullptr;
    json::check(!labels || question.type == "noul", "labels are only supported for noul");
    if (auto order = json::optional(value, "option_order")) {
        json::check(cJSON_IsArray(order), "option_order must be an array");
        for (auto item = order->child; item; item = item->next)
            question.order.push_back(json::as_integer(item, 0, 4095));
    }
    auto criteria = field(value, "criteria", "crit");
    if (question.type == "choice") {
        json::check(cJSON_IsObject(criteria) && criteria->child, "choice criteria must be a nonempty object");
        for (auto item = criteria->child; item; item = item->next) {
            question.labels.push_back(item->string);
            bool empty = cJSON_IsNull(item) || (cJSON_IsString(item) && !item->valuestring[0]);
            question.options.push_back(std::string(item->string) + (empty ? "" : ": " + criterion(item, numbers)));
        }
    } else if (question.type == "score") {
        json::check(cJSON_IsArray(criteria) && criteria->child, "score criteria must be a nonempty array");
        for (auto item = criteria->child; item; item = item->next) {
            std::string level = std::to_string(question.labels.size());
            question.labels.push_back(level); question.legend.push_back(criterion(item, numbers));
            question.options.push_back("level " + level + ": " + question.legend.back());
        }
    } else if (question.type == "noul") {
        json::check(!criteria || cJSON_IsNull(criteria) || cJSON_IsObject(criteria), "noul criteria must be an object");
        if (criteria && cJSON_IsObject(criteria))
            for (auto item = criteria->child; item; item = item->next)
                json::check(std::string(item->string) == "false" || std::string(item->string) == "true", "unknown noul criterion");
        if (labels) {
            json::check(cJSON_IsObject(labels) && cJSON_GetArraySize(labels) == 2,
                "noul labels must have exactly false and true keys");
        }
        const char* keys[] = {"false", "true"};
        const char* defaults[] = {"no, the statement does not hold", "yes, the statement holds"};
        for (int i = 0; i < 2; ++i) {
            std::string label = labels ? strip_label(json::string(labels, keys[i])) : keys[i];
            json::check(!label.empty(), "noul labels cannot be empty");
            auto item = criteria && cJSON_IsObject(criteria) ? json::optional(criteria, keys[i]) : nullptr;
            bool empty = !item || cJSON_IsNull(item) || (cJSON_IsString(item) && !item->valuestring[0]);
            question.labels.push_back(label);
            question.options.push_back(label + ": " + (empty ? defaults[i] : criterion(item, numbers)));
        }
        json::check(question.labels[0] != question.labels[1], "noul labels must be distinct");
    } else throw std::invalid_argument("question type must be choice, score or noul");
    return question;
}

json::Document predict(const std::string& request, Bundle& bundle, Tokenizer& tokenizer,
        DecisionModel& model, bool raw, int token_limit) {
    auto document = json::parse(request);
    auto root = document.get();
    const auto numbers = request_numbers(request, root);
    auto state = json::required(root, "state");
    auto questions = json::required(root, "questions");
    json::check(cJSON_IsObject(questions) && questions->child, "questions must be a nonempty object");
    std::string state_text = criterion(state, numbers);
    json::Document response(cJSON_CreateObject(), cJSON_Delete);
    auto answers = cJSON_AddObjectToObject(response.get(), "answers");
    size_t input_tokens = 0;
    for (auto item = questions->child; item; item = item->next) {
        auto question = parse_question(item, numbers);
        SequenceOptions options;
        options.max_len = token_limit;
        options.option_order = question.order;
        auto sequence = build_sequence(bundle, tokenizer, state_text, question.type,
            question.instructions, question.options, options);
        int T = sequence.ids.size(), K = sequence.marker_positions.size();
        Device<int> ids(sequence.ids), attention(std::vector<int>(T, 1));
        Device<int> markers(sequence.marker_positions), valid(std::vector<int>(K, 1));
        Device<int> types(std::vector<int>{sequence.qtype});
        auto result = model.forward(ids.data, attention.data, nullptr, markers.data, valid.data, types.data, 1, T, K);
        auto logits = download(result.logits, K), acts = download(result.act_logits, bundle.n_act);
        auto calibrated = calibrate(bundle, sequence.qtype, logits, acts, question.order);
        const auto& p = calibrated.probabilities;
        const auto& act = calibrated.act_probabilities;
        input_tokens += T;
        size_t best = calibrated.prediction_index;
        auto answer = cJSON_AddObjectToObject(answers, item->string);
        cJSON_AddStringToObject(answer, "type", question.type.c_str());
        number(answer, "answer_confidence", p[best]);
        auto action = cJSON_AddObjectToObject(answer, "action");
        number(action, "act_probability", act[0]);
        if (question.type == "noul") {
            number(answer, "noul", p[1]); number(answer, "confidence", p[best]);
        } else {
            auto output_p = cJSON_AddObjectToObject(answer, "probabilities");
            for (int k = 0; k < K; ++k) number(output_p, question.labels[k].c_str(), p[k]);
            number(answer, "confidence", calibrated.confidence);
            if (question.type == "choice") cJSON_AddStringToObject(answer, "choice", question.labels[best].c_str());
            else {
                double score = 0;
                auto legend = cJSON_AddObjectToObject(answer, "legend");
                for (int k = 0; k < K; ++k) {
                    score += k * p[k];
                    cJSON_AddStringToObject(legend, question.labels[k].c_str(), question.legend[k].c_str());
                }
                number(answer, "score", score);
            }
        }
        if (raw) {
            auto debug = cJSON_AddObjectToObject(answer, "raw");
            array(debug, "logits", logits); array(debug, "act_logits", acts);
            array(debug, "input_ids", sequence.ids); array(debug, "marker_positions", sequence.marker_positions);
            cJSON_AddNumberToObject(debug, "temperature", calibrated.temperature);
        }
        const auto& stats = sequence.stats;
        if (stats.truncated || stats.instruction_tokens_used < stats.instruction_tokens ||
                stats.option_tokens_used < stats.option_tokens) {
            auto truncation = cJSON_AddObjectToObject(answer, "truncation");
            cJSON_AddNumberToObject(truncation, "state_tokens_dropped", stats.state_tokens_dropped);
            cJSON_AddNumberToObject(truncation, "instruction_tokens_dropped", stats.instruction_tokens - stats.instruction_tokens_used);
            cJSON_AddNumberToObject(truncation, "option_tokens_dropped", stats.option_tokens - stats.option_tokens_used);
        }
    }
    auto usage = cJSON_AddObjectToObject(response.get(), "usage");
    cJSON_AddNumberToObject(usage, "input_tokens", input_tokens);
    cJSON_AddNumberToObject(usage, "output_tokens", 0);
    return response;
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::string directory, input_path, weights_path;
        int max_tokens = 0;
        bool raw = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--raw") raw = true;
            else if (arg == "--help") {
                std::cout << "Usage: pretrained_decision --bundle DIR [--weights PUFFER.bin] [--input JSONL] [--max-tokens N] [--raw]\n";
                return 0;
            } else if ((arg == "--bundle" || arg == "--input" || arg == "--max-tokens" || arg == "--weights") && i + 1 < argc) {
                std::string value = argv[++i];
                if (arg == "--bundle") directory = value;
                else if (arg == "--input") input_path = value;
                else if (arg == "--weights") weights_path = value;
                else {
                    size_t consumed = 0;
                    max_tokens = std::stoi(value, &consumed);
                    if (consumed != value.size() || max_tokens < 1)
                        throw std::invalid_argument("invalid --max-tokens");
                }
            } else throw std::invalid_argument("unknown or incomplete argument: " + arg);
        }
        if (directory.empty()) throw std::invalid_argument("--bundle is required");
        Bundle bundle(directory);
        Tokenizer tokenizer(bundle.tokenizer_path);
        if (!max_tokens) max_tokens = bundle.max_len;
        if (max_tokens < 1 || max_tokens > bundle.encoder.max_positions) throw std::invalid_argument("invalid --max-tokens");
        DecisionConfig config{bundle.encoder, bundle.head_layers, bundle.n_act, !weights_path.empty()};
        DecisionModel model(config, 1, max_tokens, std::min(4096, max_tokens), nullptr, false);
        model.initialize_new_heads(1, bundle.kind == "laya");
        if (weights_path.empty()) bundle.load_parameters(model.parameters());
        else { bundle.validate_parameters(model.parameters()); load_puffer_weights(model, weights_path); }
        std::ifstream file;
        if (!input_path.empty()) { file.open(input_path); if (!file) throw std::runtime_error("cannot open --input"); }
        std::istream& input = input_path.empty() ? std::cin : file;
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty()) continue;
            auto result = predict(line, bundle, tokenizer, model, raw, max_tokens);
            char* text = cJSON_PrintUnformatted(result.get());
            if (!text) throw std::runtime_error("cannot serialize prediction");
            std::cout << text << '\n'; cJSON_free(text);
            std::cout.flush();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pretrained_decision: " << error.what() << '\n'; return 1;
    }
}
