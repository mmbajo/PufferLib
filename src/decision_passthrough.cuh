#ifndef PUFFER_DECISION_PASSTHROUGH_CUH
#define PUFFER_DECISION_PASSTHROUGH_CUH

#include <stdexcept>

// Decision encoders emit [action logits..., value] directly. The recurrent
// network and linear decoder stages therefore only pass through those values.
static void decision_no_params(void*, Allocator*) {}
static void decision_no_init(void*, ulong*, cudaStream_t) {}
static void decision_no_rollout(void*, void*, Allocator*, int) {}
static void decision_no_train(void*, void*, Allocator*, Allocator*, int) {}
static void* decision_identity_weights(void*) { return calloc(1, 1); }
static Prec decision_identity_forward(void*, Prec input, Prec, void*, cudaStream_t) { return input; }
static Prec decision_identity_train(void*, Prec input, Prec, Prec, void*, int, cudaStream_t) { return input; }
static Prec decision_identity_backward(void*, Prec gradient, void*, cudaStream_t) { return gradient; }
static void create_decision_network(Network* network) {
    network->forward = decision_identity_forward;
    network->forward_train = decision_identity_train;
    network->backward = decision_identity_backward;
    network->create_weights = decision_identity_weights;
    network->reg_params = decision_no_params;
    network->reg_train = decision_no_train;
    network->reg_rollout = decision_no_rollout;
    network->init_weights = decision_no_init;
}

// Retain the stock DecoderWeights layout: the runner reads continuous/logstd
// through it even when custom decoder callbacks are installed.
static void* decision_decoder_weights(void* self) {
    auto* decoder = static_cast<Decoder*>(self);
    if (decoder->continuous || decoder->output_dim < 2)
        throw std::invalid_argument("decision policies require discrete actions");
    auto* weights = static_cast<DecoderWeights*>(calloc(1, sizeof(DecoderWeights)));
    weights->output_dim = decoder->output_dim;
    weights->hidden_dim = decoder->output_dim + 1;
    weights->continuous = false;
    return weights;
}
static Prec decision_decoder_forward(void*, void*, Prec input, cudaStream_t) { return input; }
static void decision_decoder_train(void* opaque, void* activations,
        Allocator* acts, Allocator*, int batch) {
    auto* weights = static_cast<DecoderWeights*>(opaque);
    auto* a = static_cast<DecoderActivations*>(activations);
    a->grad_out = {.shape = {batch, weights->hidden_dim}};
    alloc_register(acts, &a->grad_out);
}
static Prec decision_decoder_backward(void* opaque, void* activations,
        Float logits, Float, Float values, cudaStream_t stream) {
    auto* weights = static_cast<DecoderWeights*>(opaque);
    auto* a = static_cast<DecoderActivations*>(activations);
    int actions = weights->output_dim, stride = weights->hidden_dim;
    int batch = static_cast<int>(numel(logits.shape) / actions);
    assemble_decoder_grad<<<grid_size(batch * stride), BLOCK_SIZE, 0, stream>>>(
        a->grad_out.data, logits.data, values.data, batch, actions, stride);
    return Prec{.data = a->grad_out.data, .shape = {batch, stride}};
}
static void create_decision_decoder(Decoder* decoder) {
    decoder->forward = decision_decoder_forward;
    decoder->backward = decision_decoder_backward;
    decoder->create_weights = decision_decoder_weights;
    decoder->reg_params = decision_no_params;
    decoder->reg_train = decision_decoder_train;
    decoder->reg_rollout = decision_no_rollout;
    decoder->init_weights = decision_no_init;
}

#endif
