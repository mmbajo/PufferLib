#ifndef PUFFER_PRETRAINED_JSON_H
#define PUFFER_PRETRAINED_JSON_H

#include "../vendor/cJSON.h"
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

namespace pretrained { namespace json {
using Document = std::unique_ptr<cJSON, decltype(&cJSON_Delete)>;

inline void check(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error("pretrained JSON: " + message);
}
inline void unique_keys(const cJSON* node) {
    if (cJSON_IsObject(node)) {
        std::set<std::string> keys;
        for (auto child = node->child; child; child = child->next) {
            check(child->string && keys.insert(child->string).second, "duplicate object key");
            unique_keys(child);
        }
    } else if (cJSON_IsArray(node)) {
        for (auto child = node->child; child; child = child->next) unique_keys(child);
    }
}
inline Document parse(const std::string& text) {
    check(text.find('\0') == std::string::npos, "embedded NUL in JSON source");
    // cJSON exposes strings as C strings and cannot preserve escaped NUL.
    // Bundle metadata has no need for it; fail rather than truncate a path/key.
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] != '\\') continue;
        check(text.compare(i, 6, "\\u0000") != 0, "NUL in JSON string");
        ++i;
    }
    Document result(cJSON_ParseWithLengthOpts(text.c_str(), text.size() + 1, nullptr, 1), cJSON_Delete);
    check(bool(result), "invalid or trailing JSON");
    unique_keys(result.get());
    return result;
}
inline Document read(const std::string& path, size_t max_bytes = 32u << 20) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    check(bool(input), "cannot open " + path);
    const auto size = input.tellg();
    check(size >= 0 && static_cast<uint64_t>(size) <= max_bytes, "file too large: " + path);
    std::string text(static_cast<size_t>(size), '\0');
    input.seekg(0);
    input.read(text.data(), text.size());
    check(bool(input), "cannot read " + path);
    return parse(text);
}
inline const cJSON* optional(const cJSON* object, const char* key) {
    check(cJSON_IsObject(object), "expected object");
    return cJSON_GetObjectItemCaseSensitive(object, key);
}
inline const cJSON* required(const cJSON* object, const char* key) {
    const auto* value = optional(object, key);
    check(value != nullptr, std::string("missing ") + key);
    return value;
}
inline std::string as_string(const cJSON* value) {
    check(cJSON_IsString(value) && value->valuestring, "expected string");
    return value->valuestring;
}
inline double as_number(const cJSON* value) {
    check(cJSON_IsNumber(value) && std::isfinite(value->valuedouble), "expected finite number");
    return value->valuedouble;
}
inline int64_t as_integer(const cJSON* value, int64_t minimum = 0,
        int64_t maximum = 9007199254740991LL) {
    const double number = as_number(value);
    check(number >= minimum && number <= maximum && std::floor(number) == number,
        "integer outside supported range");
    return static_cast<int64_t>(number);
}
inline std::string string(const cJSON* object, const char* key) {
    return as_string(required(object, key));
}
inline std::string string(const cJSON* object, const char* key, const std::string& fallback) {
    const auto* value = optional(object, key);
    return value ? as_string(value) : fallback;
}
inline int integer(const cJSON* object, const char* key, int fallback,
        int minimum = 0, int maximum = std::numeric_limits<int>::max()) {
    const auto* value = optional(object, key);
    if (value) return static_cast<int>(as_integer(value, minimum, maximum));
    check(fallback >= minimum && fallback <= maximum, std::string("default outside range: ") + key);
    return fallback;
}
inline float number(const cJSON* object, const char* key, float fallback) {
    const auto* value = optional(object, key);
    const float result = value ? static_cast<float>(as_number(value)) : fallback;
    check(std::isfinite(result), "number outside float range");
    return result;
}
inline bool boolean(const cJSON* object, const char* key, bool fallback) {
    const auto* value = optional(object, key);
    if (!value) return fallback;
    check(cJSON_IsBool(value), std::string("expected boolean ") + key);
    return cJSON_IsTrue(value);
}
inline const cJSON* array(const cJSON* object, const char* key) {
    const auto* value = required(object, key);
    check(cJSON_IsArray(value), std::string("expected array ") + key);
    return value;
}
}} // namespace pretrained::json
#endif
