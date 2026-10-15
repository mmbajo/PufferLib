#ifndef PUFFER_PRETRAINED_INPUT_H
#define PUFFER_PRETRAINED_INPUT_H

#include "pretrained_bundle.cuh"
#include <numeric>

namespace pretrained {

// Sequence and calibration semantics follow laya/common.py at
// fa9a2a7070b1789912a49ae24603bbfb1a78b001. Callers render structured state and
// criteria before entry; strings are UTF-8 and preserve embedded NUL.
struct SequenceOptions {
    int max_len = 0, head_max_len = 0; // zero uses the bundle's stored budgets
    bool truncate_left = false;
    bool reject_collapsed_options = true;
    bool reject_truncated_state = false;
    std::vector<int> option_order; // slot -> semantic option; empty is identity
};
struct SequenceStats {
    size_t options = 0, options_distinct = 0;
    int tokens_per_option = -1; // includes [MASK], -1 = no budget recapping
    size_t instruction_tokens = 0, instruction_tokens_used = 0;
    size_t option_tokens = 0, option_tokens_used = 0; // excludes marker tokens
    size_t state_tokens = 0, state_tokens_used = 0, state_tokens_dropped = 0;
    bool truncated = false;
};
struct Sequence {
    std::vector<int> ids, marker_positions;
    int qtype = 0;
    SequenceStats stats;
};

inline int question_type(const std::string& name) {
    if (name == "choice") return 0;
    if (name == "score") return 1;
    if (name == "noul") return 2;
    throw std::invalid_argument("question type must be choice, score or noul");
}
inline const char* question_type_name(int type) {
    if (type < 0 || type > 2) throw std::invalid_argument("invalid question type ID");
    return type == 0 ? "choice" : type == 1 ? "score" : "noul";
}
inline std::string sanitize_mask(const std::string& text, const std::string& mask) {
    if (mask.empty()) throw std::invalid_argument("empty mask token");
    std::string result;
    size_t start = 0;
    for (;;) {
        const size_t next = text.find(mask, start);
        if (next == std::string::npos) { result.append(text, start, std::string::npos); break; }
        result.append(text, start, next - start);
        result.push_back(' ');
        start = next + mask.size();
    }
    return result;
}
inline std::vector<int> option_permutation(size_t count, const std::vector<int>& order) {
    if (count < 1 || count > 4096) throw std::invalid_argument("option count must be in [1,4096]");
    if (order.empty()) {
        std::vector<int> identity(count);
        std::iota(identity.begin(), identity.end(), 0);
        return identity;
    }
    if (order.size() != count) throw std::invalid_argument("option_order must be a complete permutation");
    std::vector<bool> seen(count, false);
    for (int i : order) {
        if (i < 0 || size_t(i) >= count || seen[i])
            throw std::invalid_argument("option_order must be a complete permutation");
        seen[i] = true;
    }
    return order;
}

inline Sequence build_sequence(const Bundle& bundle, const Tokenizer& tokenizer,
        const std::string& state, const std::string& type, const std::string& instructions,
        const std::vector<std::string>& rendered_options, const SequenceOptions& options = {}) {
    Sequence result;
    result.qtype = question_type(type);
    const int max_len = options.max_len ? options.max_len : bundle.max_len;
    const int head_max_len = options.head_max_len ? options.head_max_len : bundle.head_max_len;
    if (max_len < 1 || max_len > bundle.encoder.max_positions || head_max_len < 1 || head_max_len > max_len)
        throw std::invalid_argument("invalid sequence token budgets");
    const auto order = option_permutation(rendered_options.size(), options.option_order);
    if (result.qtype == 2 && order.size() != 2)
        throw std::invalid_argument("noul requires exactly false and true options");
    auto head = tokenizer.Encode(type + " question: " + sanitize_mask(instructions, bundle.mask_token), false);
    result.stats.instruction_tokens = head.size();
    std::vector<std::vector<uint32_t>> spans;
    size_t option_length = 0;
    for (int i : order) {
        auto ids = tokenizer.Encode(" " + sanitize_mask(rendered_options[i], bundle.mask_token), false);
        result.stats.option_tokens += ids.size();
        if (ids.size() > 48) ids.resize(48);
        ids.insert(ids.begin(), bundle.mask_id);
        option_length += ids.size();
        spans.push_back(std::move(ids));
    }
    int64_t option_budget = int64_t(head_max_len) - int64_t(option_length);
    if (option_budget < 16) {
        const int per = std::max(4, (head_max_len - 16) / static_cast<int>(spans.size()));
        result.stats.tokens_per_option = per;
        option_length = 0;
        for (auto& span : spans) {
            if (span.size() > size_t(per)) span.resize(per);
            option_length += span.size();
        }
        option_budget = int64_t(head_max_len) - int64_t(option_length);
    }
    head.resize(std::min(head.size(), static_cast<size_t>(std::max<int64_t>(8, option_budget))));
    result.stats.instruction_tokens_used = head.size();
    result.stats.options = spans.size();
    result.stats.options_distinct = std::set<std::vector<uint32_t>>(spans.begin(), spans.end()).size();
    result.stats.option_tokens_used = option_length - spans.size();
    if (options.reject_collapsed_options && result.stats.options_distinct != spans.size())
        throw std::invalid_argument("token budget/tokenizer collapses distinct options to identical token spans");
    result.ids.push_back(static_cast<int>(bundle.cls_id));
    for (uint32_t id : head) result.ids.push_back(static_cast<int>(id));
    result.ids.push_back(static_cast<int>(bundle.sep_id));
    for (const auto& span : spans) {
        result.marker_positions.push_back(static_cast<int>(result.ids.size()));
        for (uint32_t id : span) result.ids.push_back(static_cast<int>(id));
    }
    result.ids.push_back(static_cast<int>(bundle.sep_id));
    // A too-large option set can exceed max_len even after per-option capping.
    // Refuse a partially cut option, not only a missing marker.
    if (result.ids.size() > size_t(max_len))
        throw std::invalid_argument("question/options exceed sequence budget; increase max_len or reduce options");
    const size_t room = result.ids.size() < size_t(max_len) ? size_t(max_len) - result.ids.size() - 1 : 0;
    const auto state_ids = tokenizer.Encode(sanitize_mask(state, bundle.mask_token), false);
    result.stats.state_tokens = state_ids.size();
    result.stats.state_tokens_used = std::min(room, state_ids.size());
    result.stats.state_tokens_dropped = state_ids.size() - result.stats.state_tokens_used;
    result.stats.truncated = result.stats.state_tokens_dropped != 0;
    if (options.reject_truncated_state && result.stats.truncated)
        throw std::invalid_argument("state exceeds sequence budget; state tokens would be dropped");
    const size_t start = options.truncate_left ? result.stats.state_tokens_dropped : 0;
    for (size_t i = 0; i < result.stats.state_tokens_used; ++i)
        result.ids.push_back(static_cast<int>(state_ids[start + i]));
    result.ids.push_back(static_cast<int>(bundle.sep_id));
    if (result.ids.size() > size_t(max_len)) result.ids.resize(max_len);
    return result;
}

inline float clamp_temperature(float temperature) {
    if (!std::isfinite(temperature)) return 1.f;
    return std::min(5.f, std::max(.5f, temperature));
}
inline float decision_temperature(const Bundle& bundle, int qtype, size_t count) {
    const char* type = question_type_name(qtype);
    if (!count) throw std::invalid_argument("cannot calibrate zero options");
    const char* bucket = count <= 2 ? "2" : count <= 5 ? "3-5" : count <= 10 ? "6-10" : "11+";
    const auto found = bundle.temperature_by_options.find(std::string(type) + ":" + bucket);
    return clamp_temperature(found != bundle.temperature_by_options.end() ? found->second : bundle.temperature.at(qtype));
}
inline std::vector<float> probability_softmax(const std::vector<float>& logits, float temperature = 1.f) {
    if (logits.empty() || !std::isfinite(temperature) || temperature <= 0)
        throw std::invalid_argument("invalid softmax inputs");
    float maximum = -std::numeric_limits<float>::infinity();
    for (float value : logits) {
        if (!std::isfinite(value)) throw std::invalid_argument("nonfinite decision logit");
        const float scaled = value / temperature;
        if (!std::isfinite(scaled)) throw std::invalid_argument("decision logit overflows temperature scaling");
        maximum = std::max(maximum, scaled);
    }
    std::vector<float> result(logits.size());
    float sum = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        result[i] = std::exp(logits[i] / temperature - maximum);
        sum += result[i];
    }
    for (float& value : result) value /= sum;
    return result;
}
struct CalibratedDecision {
    std::vector<float> probabilities, act_probabilities;
    float temperature = 1, confidence = 0, answer_confidence = 0, expected_score = 0;
    size_t prediction_index = 0;
};
inline CalibratedDecision calibrate(const Bundle& bundle, int qtype,
        const std::vector<float>& logits, const std::vector<float>& act_logits = {},
        const std::vector<int>& option_order = {}) {
    if (qtype == 2 && logits.size() != 2) throw std::invalid_argument("noul requires two logits");
    CalibratedDecision result;
    result.temperature = decision_temperature(bundle, qtype, logits.size());
    const auto raw = probability_softmax(logits, result.temperature);
    const auto order = option_permutation(raw.size(), option_order);
    result.probabilities.resize(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) result.probabilities[order[i]] = raw[i];
    const auto best = std::max_element(result.probabilities.begin(), result.probabilities.end());
    result.prediction_index = static_cast<size_t>(best - result.probabilities.begin());
    result.answer_confidence = *best;
    float entropy = 0;
    for (size_t i = 0; i < result.probabilities.size(); ++i) {
        const float p = result.probabilities[i];
        entropy -= p * std::log(std::max(1e-12f, p));
        result.expected_score += static_cast<float>(i) * p;
    }
    result.confidence = qtype == 2 ? result.answer_confidence : raw.size() < 2 ? 1.f :
        std::min(1.f, std::max(0.f, 1.f - entropy / std::log(static_cast<float>(raw.size()))));
    if (!act_logits.empty()) result.act_probabilities = probability_softmax(act_logits);
    return result;
}

} // namespace pretrained
#endif
