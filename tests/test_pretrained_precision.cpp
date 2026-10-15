#include "../src/ini.h"
#include "../src/pretrained_encoder.cuh"
#include <iostream>

template<class F> static void rejected(F&& function) {
    try { function(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid precision configuration accepted");
}

int main() {
    using namespace pretrained;
    try {
        for (const char* name : {"fp32", "tf32"}) {
            if (strcmp(gemm_precision_name(parse_gemm_precision(name)), name)) {
                throw std::runtime_error("precision name did not round trip");
            }
        }
        for (const char* name : {"", "FP32", "TF32", "bf16", "0", "1",
                "fp32,tf32", " tf32", "tf32 "}) {
            rejected([&] { parse_gemm_precision(name); });
        }
        // Invalid overrides fail before configure_gemm_precision touches the
        // cuBLAS handle, so this boundary check needs no GPU context.
        for (const char* value : {"", "0", "true", "false", "garbage", "2"}) {
            setenv("NVIDIA_TF32_OVERRIDE", value, 1);
            rejected([] { detail::configure_gemm_precision(nullptr, GemmPrecision::TF32); });
        }
        for (const char* env : {"decision_laya", "decision_cartpole", "decision_snake"}) {
            Ini defaults{};
            puf_ini_load_env(&defaults, env, 0, nullptr);
            if (parse_gemm_precision(puf_ini_get_str(&defaults, "policy", "gemm_precision"))
                    != GemmPrecision::FP32) {
                throw std::runtime_error("environment default is not strict FP32");
            }
            puf_ini_free(&defaults);
            for (const char* mode : {"fp32", "tf32"}) {
                Ini command{};
                std::string option = std::string("--policy.gemm_precision=") + mode;
                char* argv[] = {option.data()};
                puf_ini_load_env(&command, env, 1, argv);
                if (parse_gemm_precision(puf_ini_get_str(&command, "policy", "gemm_precision"))
                        != parse_gemm_precision(mode)) {
                    throw std::runtime_error("CLI GEMM precision was not applied");
                }
                puf_ini_free(&command);
            }
        }
        std::cout << "PASS: GEMM precision parsing, CLI defaults and overrides\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
