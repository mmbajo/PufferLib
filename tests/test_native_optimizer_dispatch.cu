// Real board-policy initialization and default/explicit optimizer dispatch.
// Fixed external gradients isolate this check from atomic backward reductions.
#include "../src/pufferl.cu"
#include <stdexcept>

static void ensure(bool okay, const char* why) {
    if (!okay) throw std::runtime_error(why);
}
static std::vector<float> host(const float* source, int64_t n) {
    std::vector<float> result(n);
    ensure(cudaMemcpy(result.data(), source, n*sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess,
        "GPU download failed");
    return result;
}
static void identical(const float* a, const float* b, int64_t n, const char* why) {
    auto first = host(a,n), second = host(b,n);
    ensure(!memcmp(first.data(), second.data(), n*sizeof(float)), why);
}
static PuffeRL* create(Ini* ini, const char* optimizer) {
    puf_ini_load_env(ini, "decision_snake", 0, nullptr);
    const char* settings[][2] = {{"base.seed","73"},{"base.async","0"},
        {"base.cudagraphs","-1"},{"vec.total_agents","2"},{"vec.num_buffers","1"},
        {"vec.num_threads","1"},{"train.horizon","4"},{"train.minibatch_size","8"},
        {"train.total_timesteps","32"},{"train.learning_rate","0.003"},
        {"train.anneal_lr","0"},{"train.distributed_optimizer","0"}};
    for (auto& setting : settings) puf_ini_put(ini, setting[0], setting[1]);
    if (optimizer) puf_ini_put(ini, "train.optimizer", optimizer);
    TrainContext context{}; context.world_size = 1;
    return create_pufferl(ini, &context);
}
int main() {
    try {
        Ini ini{}, reference_ini{}, adam_ini{};
        PuffeRL* actual = create(&ini, nullptr);
        PuffeRL* reference = create(&reference_ini, "muon");
        PuffeRL* adam = create(&adam_ini, "adam");
        ensure(actual->hypers.optimizer == PUF_OPTIMIZER_MUON &&
            reference->hypers.optimizer == PUF_OPTIMIZER_MUON, "default selection changed");
        int64_t count = numel(actual->policies[0].master_weights.shape);
        ensure(count == 33384, "unexpected board fixture layout");
        identical(actual->policies[0].master_weights.data, reference->policies[0].master_weights.data,
            count, "default/explicit model initialization differs");
        identical(actual->policies[0].master_weights.data, adam->policies[0].master_weights.data,
            count, "Adam selection changed model initialization");
        ensure(adam->hypers.optimizer == PUF_OPTIMIZER_ADAM && puf_optimizer_lr(adam) == adam->adam.lr,
            "Adam selection or learning-rate dispatch failed");
        cudaMemset(adam->grad.data, 0, count*sizeof(float));
        puf_optimizer_step(adam, adam->policies[0].master_weights, adam->grad, 1.5f, adam->default_stream);
        ensure(cudaDeviceSynchronize() == cudaSuccess, "Adam dispatch failed");
        uint64_t adam_count=0;
        cudaMemcpy(&adam_count, adam->adam.step, sizeof(adam_count), cudaMemcpyDeviceToHost);
        ensure(adam_count==1, "Adam dispatch failed to increment its counter");
        identical(actual->policies[0].master_weights.data, adam->policies[0].master_weights.data,
            count, "initial zero-gradient Adam update changed weights");
        puts("PASS: default Muon/explicit Muon/Adam initialization byte-identical; Adam dispatch and LR verified");
        for (int step = 0; step < 4; ++step) {
            std::vector<float> gradient(count);
            for (int64_t i = 0; i < count; ++i)
                gradient[i] = step == 2 ? 0 : .01f*sinf((float)(i+step)*.173f);
            cudaMemcpy(actual->grad.data, gradient.data(), count*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(reference->grad.data, gradient.data(), count*sizeof(float), cudaMemcpyHostToDevice);
            puf_optimizer_step(actual, actual->policies[0].master_weights, actual->grad,
                1.5f, actual->default_stream);
            muon_step(&reference->muon, reference->policies[0].master_weights, reference->grad,
                1.5f, reference->default_stream);
            ensure(cudaDeviceSynchronize() == cudaSuccess, "optimizer update failed");
            identical(actual->policies[0].master_weights.data, reference->policies[0].master_weights.data,
                count, "Muon dispatch changed weights");
            identical(actual->muon.mb.data, reference->muon.mb.data, count,
                "Muon dispatch changed momentum");
        }
        puts("PASS: four fixed-gradient Muon dispatch updates and momenta byte-identical");
        close_pufferl(actual); close_pufferl(reference); close_pufferl(adam);
        puf_ini_free(&ini); puf_ini_free(&reference_ini); puf_ini_free(&adam_ini);
        return 0;
    } catch (const std::exception& error) { fprintf(stderr, "%s\n", error.what()); return 1; }
}
