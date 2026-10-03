#!/usr/bin/env bash
# Build the CPU-only Laya snapshot test. NVCC is needed for model declarations;
# running the result uses the tokenizer and environment only, never a GPU.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$root/build/test_decision_laya_state}"
if [[ $# -gt 1 ]]; then echo "Usage: $0 [OUT]" >&2; exit 2; fi
mkdir -p -- "$(dirname -- "$out")"
temporary="$(mktemp -d "$(dirname -- "$out")/.state-test-build.XXXXXX")"
trap 'rm -rf -- "$temporary"' EXIT
archive="$root/build/libpuffer_tokenizer.a"
rebuild=0
for source in "$root/tools/native_tokenizer/Cargo.toml" \
        "$root/tools/native_tokenizer/Cargo.lock" "$root/tools/native_tokenizer/src/lib.rs"; do
    if [[ ! -f "$archive" || "$source" -nt "$archive" ]]; then rebuild=1; fi
done
if [[ "$rebuild" == 1 ]]; then "$root/tools/build_native_tokenizer.sh"; fi
cuda="${CUDA_HOME:-${CUDA_PATH:-}}"
nvcc="${CUDACXX:-${cuda:+$cuda/bin/nvcc}}"
nvcc="$(command -v "${nvcc:-nvcc}" || true)"
if [[ -z "$nvcc" ]]; then echo "Set CUDA_HOME or CUDACXX to a CUDA toolkit" >&2; exit 1; fi
cuda="${cuda:-$(dirname -- "$(dirname -- "$nvcc")")}"
cuda="$(cd -- "$cuda" && pwd)"
flags=(-std=c++17 -O2 --threads 0 -I"$root/src" -L"$cuda/lib64"
    -Xlinker -rpath -Xlinker "$cuda/lib64")
if [[ -n "${NVCC_ARCH:-}" ]]; then flags+=(-arch="$NVCC_ARCH");
else flags+=('-gencode=arch=compute_80,code=[sm_80,compute_80]'); fi
for directory in "$cuda/../math_libs" "$cuda/../../math_libs/$(basename "$cuda")"; do
    if [[ -f "$directory/include/cublas_v2.h" ]]; then
        flags+=(-I"$directory/include" -L"$directory/lib64"
            -Xlinker -rpath -Xlinker "$directory/lib64"); break
    fi
done
"${CC:-cc}" -std=c11 -O2 -c "$root/ocean/decision_snake/engine/snake.c" -o "$temporary/snake.o"
"${CC:-cc}" -std=c11 -O2 -c "$root/vendor/cJSON.c" -o "$temporary/json.o"
"$nvcc" "${flags[@]}" "$root/tests/test_decision_laya_state.cu" \
    "$temporary/snake.o" "$temporary/json.o" "$archive" \
    -lcublas -lcudart -lm -ldl -lpthread -o "$temporary/test"
mv -- "$temporary/test" "$out"
echo "Built $out (CPU execution: $out BUNDLE)"
