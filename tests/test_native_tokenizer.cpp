#include "../src/pretrained_tokenizer.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

static uint32_t read_u32(std::istream& input) {
    unsigned char bytes[4];
    input.read(reinterpret_cast<char*>(bytes), 4);
    if (!input) throw std::runtime_error("truncated tokenizer fixture");
    return (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 |
        (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24;
}

static void write_u32(std::ostream& output, uint32_t value) {
    unsigned char bytes[] = {(unsigned char)value, (unsigned char)(value >> 8),
        (unsigned char)(value >> 16), (unsigned char)(value >> 24)};
    output.write(reinterpret_cast<char*>(bytes), 4);
}

static std::string read_string(std::istream& input, uint32_t size) {
    if (size > 1024 * 1024) throw std::runtime_error("tokenizer fixture text is too large");
    std::string text(size, '\0');
    if (size) input.read(&text[0], size);
    if (!input) throw std::runtime_error("truncated tokenizer fixture text");
    return text;
}

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: tokenizer-test tokenizer.json input.bin output.bin");
        pretrained::Tokenizer tokenizer(argv[1]);
        std::ifstream input(argv[2], std::ios::binary);
        std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
        if (!input || !output) throw std::runtime_error("cannot open tokenizer test files");
        if (read_string(input, 8) != std::string("PUFTOK1\0", 8))
            throw std::runtime_error("invalid tokenizer fixture magic");
        uint32_t records = read_u32(input);
        if (records > 10000) throw std::runtime_error("too many tokenizer fixture records");
        write_u32(output, records);
        write_u32(output, (uint32_t)tokenizer.VocabSize(false));
        write_u32(output, (uint32_t)tokenizer.VocabSize(true));
        uint32_t lookups = read_u32(input);
        if (lookups > 1000) throw std::runtime_error("too many tokenizer lookup fixtures");
        write_u32(output, lookups);
        for (uint32_t i = 0; i < lookups; ++i) {
            auto id = tokenizer.TokenId(read_string(input, read_u32(input)));
            write_u32(output, id ? 1 : 0);
            write_u32(output, id.value_or(0));
            if (id && !tokenizer.HasId(*id)) throw std::runtime_error("token lookup returned absent vocabulary ID");
        }
        if (tokenizer.HasId(UINT32_MAX)) throw std::runtime_error("invalid ID unexpectedly exists in test vocabulary");
        for (uint32_t i = 0; i < records; ++i) {
            bool special = read_u32(input) != 0;
            std::string text = read_string(input, read_u32(input));
            uint32_t pair_size = read_u32(input);
            auto encoding = pair_size == UINT32_MAX ? tokenizer.EncodeDetailed(text, special) :
                tokenizer.EncodePairDetailed(text, read_string(input, pair_size), special);
            if (encoding.ids.size() != encoding.type_ids.size() ||
                    encoding.ids.size() != encoding.attention_mask.size())
                throw std::runtime_error("native tokenizer returned mismatched metadata lengths");
            write_u32(output, (uint32_t)encoding.ids.size());
            for (const auto* values : {&encoding.ids, &encoding.type_ids, &encoding.attention_mask})
                for (uint32_t value : *values) write_u32(output, value);
            for (uint32_t id : encoding.ids)
                if (!tokenizer.HasId(id)) throw std::runtime_error("encoded token is absent from vocabulary");
            for (bool skip_special : {false, true}) {
                std::string decoded = tokenizer.Decode(encoding.ids, skip_special);
                write_u32(output, (uint32_t)decoded.size());
                output.write(decoded.data(), decoded.size());
            }
        }
        output.flush();
        if (!output) throw std::runtime_error("tokenizer fixture output write failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native tokenizer parity: " << error.what() << '\n';
        return 1;
    }
}
