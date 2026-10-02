#!/usr/bin/env bash
# Builds CPU-only tests; a CUDA toolkit is needed to parse the shared policy.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cuda="${CUDA_HOME:-${CUDA_PATH:-}}"
nvcc="${CUDACXX:-${cuda:+$cuda/bin/nvcc}}"
nvcc="$(command -v "${nvcc:-nvcc}" || true)"
if [[ -z "$nvcc" ]]; then echo "Set CUDA_HOME or CUDACXX to a CUDA toolkit" >&2; exit 1; fi
cuda="${cuda:-$(dirname -- "$(dirname -- "$nvcc")")}"
cuda="$(cd -- "$cuda" && pwd)"
mkdir -p "$root/build"
temporary="$(mktemp -d "$root/build/.decision-2048-tests.XXXXXX")"
trap 'rm -rf -- "$temporary"' EXIT

"${CC:-cc}" -std=c11 -O2 -Wall -Wextra -Werror \
    "$root/ocean/decision_2048/engine/tests/test_g2048.c" -lm -o "$temporary/test_engine"
"${CC:-cc}" -std=c11 -O2 -c "$root/tests/test_decision_2048_engine.c" -o "$temporary/engine.o"
"${CC:-cc}" -std=c11 -O2 -c "$root/vendor/cJSON.c" -o "$temporary/json.o"
archive="$root/build/libpuffer_tokenizer.a"
rebuild=0
for source in "$root/tools/native_tokenizer/Cargo.toml" "$root/tools/native_tokenizer/Cargo.lock" \
        "$root/tools/native_tokenizer/src/lib.rs"; do
    if [[ ! -f "$archive" || "$source" -nt "$archive" ]]; then rebuild=1; fi
done
if [[ "$rebuild" == 1 ]]; then "$root/tools/build_native_tokenizer.sh"; fi
flags=(-std=c++17 -O2 --threads 0 -arch="${NVCC_ARCH:-sm_80}" -I"$root/src"
       -L"$cuda/lib64" -Xlinker -rpath -Xlinker "$cuda/lib64")
for dir in "$cuda/../math_libs" "$cuda/../../math_libs/$(basename "$cuda")"; do
    if [[ -f "$dir/include/cublas_v2.h" ]]; then
        flags+=(-I"$dir/include" -L"$dir/lib64" -Xlinker -rpath -Xlinker "$dir/lib64")
        break
    fi
done
"$nvcc" "${flags[@]}" "$root/tests/test_decision_2048_env.cu" \
    "$temporary/engine.o" "$temporary/json.o" "$archive" \
    -lcublas -lcudart -lm -ldl -lpthread -o "$temporary/test_adapter"
mv "$temporary/test_engine" "$root/build/test_decision_2048_engine"
mv "$temporary/test_adapter" "$root/build/test_decision_2048_env"
echo "Built build/test_decision_2048_engine and build/test_decision_2048_env"
