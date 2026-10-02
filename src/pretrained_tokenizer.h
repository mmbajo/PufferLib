#ifndef PUFFER_PRETRAINED_TOKENIZER_H
#define PUFFER_PRETRAINED_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Native Hugging Face tokenizers 0.23.2. All byte strings are length-delimited
 * UTF-8, including paths. A handle supports concurrent read-only operations.
 * Load clears saved padding/truncation; model callers own those decisions.
 * Successful calls return 0, failures return -1 and allocate *error (if given).
 * Free errors with puf_tokenizer_string_free, never free()/delete[]. */
typedef struct PufTokenizer PufTokenizer;
typedef struct {
    uint32_t* ids;
    uint32_t* type_ids;
    uint32_t* attention_mask;
    size_t length;
} PufTokenEncoding;

int puf_tokenizer_from_file(const char* path, size_t length,
    PufTokenizer** out, char** error);
int puf_tokenizer_from_json(const char* json, size_t length,
    PufTokenizer** out, char** error);
void puf_tokenizer_free(PufTokenizer* tokenizer);
int puf_tokenizer_encode(const PufTokenizer* tokenizer,
    const char* text, size_t text_length, const char* pair, size_t pair_length,
    int has_pair, int add_special_tokens, PufTokenEncoding* out, char** error);
void puf_tokenizer_encoding_free(PufTokenEncoding* encoding);
int puf_tokenizer_decode(const PufTokenizer* tokenizer,
    const uint32_t* ids, size_t length, int skip_special_tokens,
    char** out, size_t* out_length, char** error);
void puf_tokenizer_string_free(char* string);
/* Decoded text may contain NUL: free with its length, not string_free. */
void puf_tokenizer_bytes_free(char* bytes, size_t length);
/* token_id reports absence with found=0; absence is not an error. */
int puf_tokenizer_token_id(const PufTokenizer* tokenizer,
    const char* token, size_t length, uint32_t* id, int* found, char** error);
int puf_tokenizer_vocab_size(const PufTokenizer* tokenizer,
    int with_added_tokens, size_t* size, char** error);
/* Checks actual vocabulary membership, including potentially sparse IDs. */
int puf_tokenizer_has_id(const PufTokenizer* tokenizer,
    uint32_t id, int* found, char** error);
int puf_tokenizer_max_id(const PufTokenizer* tokenizer,
    uint32_t* id, char** error);

#ifdef __cplusplus
}

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pretrained {

struct TokenEncoding {
    std::vector<uint32_t> ids;
    std::vector<uint32_t> type_ids;
    std::vector<uint32_t> attention_mask;
};

class Tokenizer {
public:
    explicit Tokenizer(const std::string& path) {
        char* error = nullptr;
        const int status = puf_tokenizer_from_file(path.data(), path.size(), &handle_, &error);
        Check(status, error);
    }
    static Tokenizer FromJson(const std::string& json) {
        Tokenizer result;
        char* error = nullptr;
        const int status = puf_tokenizer_from_json(json.data(), json.size(), &result.handle_, &error);
        Check(status, error);
        return result;
    }
    ~Tokenizer() { puf_tokenizer_free(handle_); }
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    Tokenizer& operator=(Tokenizer&& other) noexcept {
        if (this != &other) {
            puf_tokenizer_free(handle_);
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }
    TokenEncoding EncodeDetailed(const std::string& text,
            bool add_special_tokens = true) const {
        return EncodeImpl(text, nullptr, add_special_tokens);
    }
    TokenEncoding EncodePairDetailed(const std::string& text, const std::string& pair,
            bool add_special_tokens = true) const {
        return EncodeImpl(text, &pair, add_special_tokens);
    }
    std::vector<uint32_t> Encode(const std::string& text,
            bool add_special_tokens = true) const {
        return EncodeDetailed(text, add_special_tokens).ids;
    }
    std::vector<uint32_t> EncodePair(const std::string& text, const std::string& pair,
            bool add_special_tokens = true) const {
        return EncodePairDetailed(text, pair, add_special_tokens).ids;
    }
    std::string Decode(const std::vector<uint32_t>& ids,
            bool skip_special_tokens = true) const {
        char* text = nullptr;
        size_t length = 0;
        char* error = nullptr;
        const int status = puf_tokenizer_decode(handle_, ids.data(), ids.size(),
            skip_special_tokens, &text, &length, &error);
        Check(status, error);
        try {
            std::string result(text, length);
            puf_tokenizer_bytes_free(text, length);
            return result;
        } catch (...) { puf_tokenizer_bytes_free(text, length); throw; }
    }
    std::optional<uint32_t> TokenId(const std::string& token) const {
        uint32_t id = 0;
        int found = 0;
        char* error = nullptr;
        const int status = puf_tokenizer_token_id(handle_, token.data(), token.size(),
            &id, &found, &error);
        Check(status, error);
        return found ? std::optional<uint32_t>(id) : std::nullopt;
    }
    size_t VocabSize(bool with_added_tokens = true) const {
        size_t size = 0;
        char* error = nullptr;
        const int status = puf_tokenizer_vocab_size(handle_, with_added_tokens, &size, &error);
        Check(status, error);
        return size;
    }
    bool HasId(uint32_t id) const {
        int found = 0;
        char* error = nullptr;
        const int status = puf_tokenizer_has_id(handle_, id, &found, &error);
        Check(status, error);
        return found != 0;
    }
    uint32_t MaxTokenId() const {
        uint32_t id = 0;
        char* error = nullptr;
        const int status = puf_tokenizer_max_id(handle_, &id, &error);
        Check(status, error);
        return id;
    }

private:
    Tokenizer() = default;
    static void Check(int status, char* error) {
        if (status != 0) {
            const std::string message = error ? error : "Native tokenizer operation failed";
            puf_tokenizer_string_free(error);
            throw std::runtime_error(message);
        }
        puf_tokenizer_string_free(error);
    }
    TokenEncoding EncodeImpl(const std::string& text, const std::string* pair,
            bool special) const {
        PufTokenEncoding raw{};
        char* error = nullptr;
        const int status = puf_tokenizer_encode(handle_, text.data(), text.size(),
            pair ? pair->data() : nullptr, pair ? pair->size() : 0,
            pair != nullptr, special, &raw, &error);
        Check(status, error);
        try {
            TokenEncoding result;
            if (raw.length) {
                result.ids.assign(raw.ids, raw.ids + raw.length);
                result.type_ids.assign(raw.type_ids, raw.type_ids + raw.length);
                result.attention_mask.assign(raw.attention_mask, raw.attention_mask + raw.length);
            }
            puf_tokenizer_encoding_free(&raw);
            return result;
        } catch (...) { puf_tokenizer_encoding_free(&raw); throw; }
    }
    PufTokenizer* handle_ = nullptr;
};

} // namespace pretrained
#endif
#endif
