// CPU-only packed observation checks. Compile separately with DECISION_ACTIONS
// set to 2, 4, 8 and 25; the larger cases extend the original 16-byte header.
#ifndef DECISION_ACTIONS
#define DECISION_ACTIONS 8
#endif
#include "../src/decision_policy.h"
#include <cassert>
#include <iostream>

static uint16_t read_u16(const unsigned char* source) {
    return uint16_t(source[0]) | (uint16_t(source[1]) << 8);
}
static uint32_t read_u32(const unsigned char* source) {
    return uint32_t(source[0]) | (uint32_t(source[1]) << 8) |
        (uint32_t(source[2]) << 16) | (uint32_t(source[3]) << 24);
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: test_decision_policy_layout BUNDLE\n"; return 2; }
    decision_policy_context.reset(new DecisionPolicyContext(argv[1]));
    const auto& context = *decision_policy_context;
    const std::string state = "Choose a numbered action for this test state.";
    const std::string instructions = "Select one of the available actions.";
    std::vector<std::string> actions;
    for (int i = 0; i < DECISION_ACTIONS; ++i)
        actions.push_back("Action " + std::to_string(i));
    pretrained::SequenceOptions sequence_options;
    sequence_options.reject_truncated_state = true;
    const auto reference = pretrained::build_sequence(context.bundle, context.tokenizer,
        state, "choice", instructions, actions, sequence_options);

    std::vector<unsigned char> storage(OBS_SIZE + 32, 0xa5);
    unsigned char* packed = storage.data() + 16;
    decision_policy_encode(state, instructions, actions, packed);
    for (int i = 0; i < 16; ++i) {
        assert(storage[i] == 0xa5);
        assert(storage[16 + OBS_SIZE + i] == 0xa5);
    }

    // Check the documented wire offsets independently of the encoder's macros.
    const int header = DECISION_ACTIONS <= 6 ? 16 : 4 + 2 * DECISION_ACTIONS;
    const int qtype = 2 + 2 * DECISION_ACTIONS;
    assert(DECISION_POLICY_HEADER_BYTES == header);
    assert(DECISION_POLICY_QTYPE_OFFSET == qtype && qtype < header);
    assert(OBS_SIZE == header + 4 * 2048);
    assert(read_u16(packed) == reference.ids.size());
    for (int k = 0; k < DECISION_ACTIONS; ++k) {
        const int marker = read_u16(packed + 2 + 2 * k);
        assert(marker == reference.marker_positions[k]);
        assert(marker < int(reference.ids.size()));
        assert(read_u32(packed + header + 4 * marker) == context.bundle.mask_id);
    }
    assert(packed[qtype] == 0);
    for (int i = qtype + 1; i < header; ++i) assert(packed[i] == 0);
    for (size_t t = 0; t < reference.ids.size(); ++t)
        assert(read_u32(packed + header + 4 * t) == uint32_t(reference.ids[t]));
    for (size_t i = header + 4 * reference.ids.size(); i < OBS_SIZE; ++i)
        assert(packed[i] == 0);

    // Rejection must leave the last valid observation intact.
    const auto valid_storage = storage;
    actions.pop_back();
    bool rejected = false;
    try { decision_policy_encode(state, instructions, actions, packed); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected && storage == valid_storage);
    actions.assign(DECISION_ACTIONS, "Identical action");
    rejected = false;
    try { decision_policy_encode(state, instructions, actions, packed); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected && storage == valid_storage);

    std::cout << "actions=" << DECISION_ACTIONS << " header=" << header
              << " qtype=" << qtype << " tokens=" << reference.ids.size()
              << " packing and option validation passed\n";
}
