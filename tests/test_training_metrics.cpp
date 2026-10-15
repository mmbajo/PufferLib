#include "training_metrics.h"
#include <limits>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    TrainingMetrics writer = metrics_open(argv[2]);
    if (!strcmp(argv[1], "disabled")) return writer.file ? 3 : 0;
    if (!strcmp(argv[1], "create")) {
        metrics_close(&writer);
        return 0;
    }
    Ini ini = {};
    puf_ini_set(puf_ini_section(&ini, "base", 1), "seed", "73");
    puf_ini_set(puf_ini_section(&ini, "base", 1), "run_id", "line\nquote\"slash\\tab\t\x01");
    puf_ini_set(puf_ini_section(&ini, "train", 1), "optimizer", "muon");
    puf_ini_set(puf_ini_section(&ini, "policy", 1), "bundle", "models/日本語");
    puf_ini_set(puf_ini_section(&ini, "policy", 1), "gemm_precision", "tf32");
    puf_ini_set(puf_ini_section(&ini, "env", 1), "max_steps", "500");
    puf_ini_set(puf_ini_section(&ini, "selfplay", 1), "num_policies", "2");
    metrics_run(&writer, &ini, 4);
    Dict metrics = {};
    dict_set(&metrics, "epoch", 9);
    dict_set(&metrics, "agent_steps", 9216);
    dict_set(&metrics, "uptime", 12.345678901234567);
    dict_set(&metrics, "env/n", 0);
    dict_set(&metrics, "loss/kl", 1.2345678901234567e-9);
    dict_set(&metrics, "loss/entropy", 1.0986122886681098);
    dict_set(&metrics, "loss/escaped\"\\\n\x01", -0.25);
    if (!strcmp(argv[1], "nonfinite")) {
        dict_set(&metrics, "loss/value", std::numeric_limits<double>::quiet_NaN());
        dict_set(&metrics, "loss/policy", std::numeric_limits<double>::infinity());
        dict_set(&metrics, "loss/total", -std::numeric_limits<double>::infinity());
    }
    metrics_log(&writer, &metrics);
    dict_set(&metrics, "env/n", 3);
    dict_set(&metrics, "env/score", 2.5);
    metrics_log(&writer, &metrics);
    metrics_close(&writer);
    dict_clear(&metrics);
    for (int s = 0; s < ini.num_sections; ++s) dict_clear(&ini.sections[s]);
    free(ini.sections);
    return 0;
}
