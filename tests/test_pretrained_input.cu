#include "../src/pretrained_input.h"
#include <iostream>

static void array(cJSON* root, const char* name, const std::vector<int>& values) {
    auto* output = cJSON_AddArrayToObject(root, name);
    for (int value : values) cJSON_AddItemToArray(output, cJSON_CreateNumber(value));
}
static void array(cJSON* root, const char* name, const std::vector<float>& values) {
    auto* output = cJSON_AddArrayToObject(root, name);
    for (float value : values) cJSON_AddItemToArray(output, cJSON_CreateNumber(value));
}
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    try {
        pretrained::Bundle bundle(argv[1]);
        pretrained::Tokenizer tokenizer(bundle.tokenizer_path);
        auto records = pretrained::json::read(argv[2]);
        for (auto* record = records->child; record; record = record->next) {
            auto* output = cJSON_CreateObject();
            try {
                std::vector<std::string> options;
                for (auto* option = pretrained::json::array(record, "options")->child; option; option = option->next)
                    options.push_back(pretrained::json::as_string(option));
                pretrained::SequenceOptions settings;
                settings.max_len = pretrained::json::integer(record, "max_len", 0);
                settings.head_max_len = pretrained::json::integer(record, "head_max_len", 0);
                settings.truncate_left = pretrained::json::boolean(record, "truncate_left", false);
                settings.reject_truncated_state = pretrained::json::boolean(record, "reject_truncated_state", false);
                if (auto* order = pretrained::json::optional(record, "order"))
                    for (auto* index = order->child; index; index = index->next)
                        settings.option_order.push_back(pretrained::json::as_integer(index));
                auto sequence = pretrained::build_sequence(bundle, tokenizer,
                    pretrained::json::string(record, "state"), pretrained::json::string(record, "type"),
                    pretrained::json::string(record, "instructions"), options, settings);
                array(output, "ids", sequence.ids);
                array(output, "markers", sequence.marker_positions);
                cJSON_AddNumberToObject(output, "options_distinct", sequence.stats.options_distinct);
                cJSON_AddNumberToObject(output, "tokens_per_option", sequence.stats.tokens_per_option);
                cJSON_AddNumberToObject(output, "state_tokens", sequence.stats.state_tokens);
                cJSON_AddNumberToObject(output, "state_tokens_used", sequence.stats.state_tokens_used);
                cJSON_AddNumberToObject(output, "state_tokens_dropped", sequence.stats.state_tokens_dropped);
                std::vector<float> logits;
                for (auto* value = pretrained::json::array(record, "logits")->child; value; value = value->next)
                    logits.push_back(pretrained::json::as_number(value));
                auto calibrated = pretrained::calibrate(bundle, sequence.qtype, logits, {0.5f, -0.5f}, settings.option_order);
                cJSON_AddNumberToObject(output, "temperature", calibrated.temperature);
                cJSON_AddNumberToObject(output, "prediction_index", calibrated.prediction_index);
                cJSON_AddNumberToObject(output, "confidence", calibrated.confidence);
                cJSON_AddNumberToObject(output, "answer_confidence", calibrated.answer_confidence);
                cJSON_AddNumberToObject(output, "expected_score", calibrated.expected_score);
                array(output, "probabilities", calibrated.probabilities);
                array(output, "act_probabilities", calibrated.act_probabilities);
            } catch (const std::exception& error) {
                cJSON_AddStringToObject(output, "error", error.what());
            }
            char* serialized = cJSON_PrintUnformatted(output);
            std::cout << serialized << '\n';
            cJSON_free(serialized);
            cJSON_Delete(output);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
