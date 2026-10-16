// Native JSONL inference and typed supervision for imported decision models.
// No Python or PyTorch is loaded by this program.
#include "pretrained_decision.cuh"
#include "pretrained_input.h"
#include <charconv>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <cstdio>

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
    for (const auto& p : model.parameters()) {
        for (size_t offset = 0; offset < p.count;) {
            size_t count = std::min(chunk.size(), p.count - offset);
            input.read(reinterpret_cast<char*>(chunk.data()), count * sizeof(float));
            if (!input) throw std::runtime_error("truncated flat checkpoint");
            for (size_t i = 0; i < count; ++i)
                if (!std::isfinite(chunk[i]))
                    throw std::runtime_error("nonfinite flat checkpoint weight");
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

Sequence pack_question(const Bundle& bundle, const Tokenizer& tokenizer,
        const std::string& state, const Question& question, int token_limit) {
    SequenceOptions options;
    options.max_len = token_limit;
    options.option_order = question.order;
    return build_sequence(bundle, tokenizer, state, question.type,
        question.instructions, question.options, options);
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
        auto sequence = pack_question(bundle, tokenizer, state_text, question, token_limit);
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
struct SupervisedRow {
    Sequence sequence;
    std::string state;
    std::vector<int> order;
    int target;
    float temperature;
};

std::vector<SupervisedRow> read_supervision(const std::string& path, const Bundle& bundle,
        const Tokenizer& tokenizer, int token_limit) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open supervision data: " + path);
    std::vector<SupervisedRow> rows;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        auto document = json::parse(line);
        auto root = document.get();
        const auto numbers = request_numbers(line, root);
        auto questions = json::required(root, "questions");
        json::check(cJSON_IsObject(questions) && cJSON_GetArraySize(questions) == 1,
            "supervision requires exactly one question per row");
        auto question = parse_question(questions->child, numbers);
        SupervisedRow row;
        row.state = criterion(json::required(root, "state"), numbers);
        row.sequence = pack_question(bundle, tokenizer, row.state, question, token_limit);
        json::check(!row.sequence.stats.truncated, "supervision state exceeds token budget");
        int target = json::as_integer(json::required(root, "target"),
            0, question.options.size() - 1);
        row.order = option_permutation(question.options.size(), question.order);
        row.target = std::find(row.order.begin(), row.order.end(), target) - row.order.begin();
        row.temperature = decision_temperature(bundle, row.sequence.qtype, row.order.size());
        rows.push_back(std::move(row));
    }
    json::check(!rows.empty(), "supervision data is empty");
    return rows;
}

struct SupervisedBatch {
    std::vector<int> host;
    std::vector<float> temperatures;
    int* input;
    float* output;
};
struct SupervisedStats {
    int questions = 0, correct = 0;
    double loss = 0;
};

std::vector<SupervisedStats> supervised_pass(DecisionModel& model,
        const std::vector<SupervisedRow>& rows, const std::vector<int>& order,
        int batch_size, int pad, SupervisedBatch* batch, int validation_epoch = -1,
        float* moments = nullptr, size_t parameters = 0, float learning_rate = 0,
        int* updates = nullptr) {
    using decision::transformer_detail::cuda_check;
    std::vector<SupervisedStats> stats(3);
    for (size_t start = 0; start < rows.size(); start += batch_size) {
        int B = std::min((size_t)batch_size, rows.size() - start), T = 0, K = 0;
        for (int b = 0; b < B; ++b) {
            const auto& sequence = rows[order[start + b]].sequence;
            T = std::max(T, (int)sequence.ids.size());
            K = std::max(K, (int)sequence.marker_positions.size());
        }
        size_t count = 2 * B * T + 2 * B * K + 2 * B;
        std::fill(batch->host.begin(), batch->host.begin() + count, 0);
        int* ids = batch->host.data();
        int* attention = ids + B * T;
        int* markers = attention + B * T;
        int* valid = markers + B * K;
        int* types = valid + B * K;
        int* targets = types + B;
        std::fill(ids, attention, pad);
        for (int b = 0; b < B; ++b) {
            const auto& row = rows[order[start + b]];
            const auto& sequence = row.sequence;
            std::copy(sequence.ids.begin(), sequence.ids.end(), ids + b * T);
            std::fill(attention + b * T, attention + b * T + sequence.ids.size(), 1);
            std::copy(sequence.marker_positions.begin(), sequence.marker_positions.end(),
                markers + b * K);
            std::fill(valid + b * K, valid + b * K + sequence.marker_positions.size(), 1);
            types[b] = sequence.qtype;
            targets[b] = row.target;
            batch->temperatures[b] = row.temperature;
        }
        cuda_check(cudaMemcpy(batch->input, ids, count * sizeof(int), cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(batch->output, batch->temperatures.data(),
            B * sizeof(float), cudaMemcpyHostToDevice));
        int* d_attention = batch->input + (attention - ids);
        int* d_markers = batch->input + (markers - ids);
        int* d_valid = batch->input + (valid - ids);
        int* d_types = batch->input + (types - ids);
        int* d_targets = batch->input + (targets - ids);
        auto output = model.forward(batch->input, d_attention, nullptr,
            d_markers, d_valid, d_types, B, T, K);
        float* losses = batch->output + B;
        float* gradient = losses + B;
        decision_loss(output.logits, d_valid, d_targets, batch->output,
            losses, moments ? gradient : nullptr, B, K);
        auto host_losses = download(losses, B);
        auto logits = download(output.logits, B * K);
        for (int b = 0; b < B; ++b) {
            const auto& row = rows[order[start + b]];
            int size = row.sequence.marker_positions.size(), best = 0;
            auto probabilities = probability_softmax(std::vector<float>(
                logits.begin() + b * K, logits.begin() + b * K + size), row.temperature);
            for (int k = 1; k < size; ++k) {
                if (probabilities[k] > probabilities[best] ||
                        (probabilities[k] == probabilities[best] &&
                        row.order[k] < row.order[best])) {
                    best = k;
                }
            }
            auto& total = stats[row.sequence.qtype];
            ++total.questions;
            total.correct += best == row.target;
            total.loss += host_losses[b];
            if (validation_epoch >= 0) {
                printf("{\"event\":\"validation_row\",\"epoch\":%d,\"row\":%d,"
                    "\"correct\":%d,\"log_loss\":%.17g}\n", validation_epoch,
                    order[start + b], best == row.target, (double)host_losses[b]);
            }
        }
        if (moments) {
            model.zero_grad();
            model.backward(gradient);
            ++*updates;
            size_t offset = 0;
            for (const auto& p : model.parameters()) {
                decision_adam(p.data, p.grad, moments + offset, moments + parameters + offset,
                    p.count, learning_rate, *updates);
                offset += p.count;
            }
        }
    }
    return stats;
}

void print_supervised_evaluation(const char* split, int epoch,
        const std::vector<SupervisedRow>& rows, const std::vector<SupervisedStats>& stats) {
    std::set<std::string> states;
    for (const auto& row : rows) states.insert(row.state);
    printf("{\"event\":\"evaluation\",\"split\":\"%s\",\"epoch\":%d,"
        "\"states\":%zu,\"questions\":%zu,\"types\":{", split, epoch, states.size(), rows.size());
    bool first = true;
    for (int type = 0; type < 3; ++type) {
        const auto& total = stats[type];
        if (!total.questions) continue;
        printf("%s\"%s\":{\"questions\":%d,\"accuracy\":%.17g,\"log_loss\":%.17g}",
            first ? "" : ",", question_type_name(type), total.questions,
            (double)total.correct / total.questions, total.loss / total.questions);
        first = false;
    }
    printf("}}\n");
    fflush(stdout);
}

void save_puffer_weights(DecisionModel& model, const std::string& path) {
    FILE* output = fopen(path.c_str(), "wbx");
    if (!output) throw std::runtime_error("cannot create --save-weights: " + path);
    std::vector<float> chunk(1 << 20);
    const float padding[3] = {};
    for (const auto& p : model.parameters()) {
        for (size_t offset = 0; offset < p.count;) {
            size_t count = std::min(chunk.size(), p.count - offset);
            decision::transformer_detail::cuda_check(cudaMemcpy(chunk.data(), p.data + offset,
                count * sizeof(float), cudaMemcpyDeviceToHost));
            if (fwrite(chunk.data(), sizeof(float), count, output) != count)
                throw std::runtime_error("cannot write --save-weights");
            offset += count;
        }
        size_t count = ((p.count + 3) & ~size_t(3)) - p.count;
        if (fwrite(padding, sizeof(float), count, output) != count)
            throw std::runtime_error("cannot write --save-weights padding");
    }
    if (fclose(output)) throw std::runtime_error("cannot close --save-weights");
}

void supervise(DecisionModel& model, const Bundle& bundle,
        const std::vector<SupervisedRow>& train, const std::vector<SupervisedRow>& validation,
        int max_tokens, int max_options, int batch_size, int epochs, float learning_rate,
        unsigned seed, const std::string& save_path) {
    using decision::transformer_detail::cuda_check;
    SupervisedBatch batch;
    batch.host.resize(2 * batch_size * (max_tokens + max_options + 1));
    batch.temperatures.resize(batch_size);
    cuda_check(cudaMalloc(&batch.input, batch.host.size() * sizeof(int)));
    cuda_check(cudaMalloc(&batch.output, batch_size * (max_options + 2) * sizeof(float)));
    size_t parameters = 0;
    for (const auto& p : model.parameters()) parameters += p.count;
    float* moments = nullptr;
    cuda_check(cudaMalloc(&moments, 2 * parameters * sizeof(float)));
    cuda_check(cudaMemset(moments, 0, 2 * parameters * sizeof(float)));
    std::vector<int> train_order(train.size()), valid_order(validation.size());
    std::iota(train_order.begin(), train_order.end(), 0);
    std::iota(valid_order.begin(), valid_order.end(), 0);
    int updates = 0;
    auto evaluate = [&](int epoch) {
        print_supervised_evaluation("train", epoch, train, supervised_pass(model, train,
            train_order, batch_size, bundle.pad_id, &batch));
        print_supervised_evaluation("validation", epoch, validation, supervised_pass(model,
            validation, valid_order, batch_size, bundle.pad_id, &batch, epoch));
    };
    evaluate(0);
    std::mt19937 random(seed);
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::shuffle(train_order.begin(), train_order.end(), random);
        auto stats = supervised_pass(model, train, train_order, batch_size, bundle.pad_id,
            &batch, -1, moments, parameters, learning_rate, &updates);
        double loss = 0;
        for (const auto& total : stats) loss += total.loss;
        printf("{\"event\":\"train\",\"epoch\":%d,\"updates\":%d,\"questions\":%zu,"
            "\"mean_log_loss\":%.17g}\n", epoch, updates, train.size(), loss / train.size());
        fflush(stdout);
    }
    if (epochs) evaluate(epochs);
    save_puffer_weights(model, save_path);
    cuda_check(cudaFree(moments));
    cuda_check(cudaFree(batch.input));
    cuda_check(cudaFree(batch.output));
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::string directory, input_path, weights_path, train_path, validation_path, save_path;
        int epochs = 1, batch_size = 16, seed = 1;
        float learning_rate = .001f;
        int max_tokens = 0;
        bool raw = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--raw") raw = true;
            else if (arg == "--help") {
                std::cout << "Usage: pretrained_decision --bundle DIR [--weights PUFFER.bin] "
                    "[--input JSONL] [--max-tokens N] [--raw]\n"
                    "Supervision: --train JSONL --validation JSONL --save-weights PUFFER.bin "
                    "[--epochs N] [--batch-size N] [--learning-rate F] [--seed N]\n";
                return 0;
            } else if ((arg == "--bundle" || arg == "--input" || arg == "--max-tokens" ||
                    arg == "--weights" || arg == "--train" || arg == "--validation" ||
                    arg == "--save-weights" || arg == "--epochs" || arg == "--batch-size" ||
                    arg == "--learning-rate" || arg == "--seed") && i + 1 < argc) {
                std::string value = argv[++i];
                if (arg == "--bundle") directory = value;
                else if (arg == "--input") input_path = value;
                else if (arg == "--weights") weights_path = value;
                else if (arg == "--train") train_path = value;
                else if (arg == "--validation") validation_path = value;
                else if (arg == "--save-weights") save_path = value;
                else {
                    size_t consumed = 0;
                    if (arg == "--learning-rate") {
                        learning_rate = std::stof(value, &consumed);
                        if (!std::isfinite(learning_rate) || learning_rate <= 0)
                            throw std::invalid_argument("invalid --learning-rate");
                    } else {
                        int number = std::stoi(value, &consumed);
                        if (number < 0 || (arg == "--max-tokens" && !number))
                            throw std::invalid_argument("invalid " + arg);
                        if (arg == "--max-tokens") max_tokens = number;
                        else if (arg == "--epochs") epochs = number;
                        else if (arg == "--batch-size") batch_size = number;
                        else seed = number;
                    }
                    if (consumed != value.size()) throw std::invalid_argument("invalid " + arg);
                }
            } else throw std::invalid_argument("unknown or incomplete argument: " + arg);
        }
        if (directory.empty()) throw std::invalid_argument("--bundle is required");
        Bundle bundle(directory);
        Tokenizer tokenizer(bundle.tokenizer_path);
        if (!max_tokens) max_tokens = bundle.max_len;
        if (max_tokens < 1 || max_tokens > bundle.encoder.max_positions) throw std::invalid_argument("invalid --max-tokens");
        bool training = !train_path.empty();
        if (training && (validation_path.empty() || save_path.empty() ||
                !input_path.empty() || raw))
            throw std::invalid_argument(
                "--train requires --validation and --save-weights, without --input/--raw");
        if (!training && (!validation_path.empty() || !save_path.empty()))
            throw std::invalid_argument("--validation and --save-weights require --train");
        std::vector<SupervisedRow> train, validation;
        int max_options = std::min(4096, max_tokens);
        if (training) {
            train = read_supervision(train_path, bundle, tokenizer, max_tokens);
            validation = read_supervision(validation_path, bundle, tokenizer, max_tokens);
            std::set<std::string> training_states;
            max_options = 0;
            for (const auto& row : train) {
                training_states.insert(row.state);
                max_options = std::max(max_options, (int)row.sequence.marker_positions.size());
            }
            for (const auto& row : validation) {
                json::check(!training_states.count(row.state),
                    "training and validation states overlap");
                max_options = std::max(max_options, (int)row.sequence.marker_positions.size());
            }
        }
        DecisionConfig config{bundle.encoder, bundle.head_layers, bundle.n_act,
            training || !weights_path.empty()};
        DecisionModel model(config, training ? batch_size : 1, max_tokens,
            max_options, nullptr, training);
        model.initialize_new_heads(seed, bundle.kind == "laya");
        if (weights_path.empty()) bundle.load_parameters(model.parameters());
        else load_puffer_weights(model, weights_path);
        if (training) {
            supervise(model, bundle, train, validation, max_tokens, max_options,
                batch_size, epochs, learning_rate, seed, save_path);
            return 0;
        }
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
