#!/usr/bin/env bash
# CPU execution only; the adapter includes CUDA declarations and needs nvcc.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$root/build"
temporary="$(mktemp -d "$root/build/.connect4-test-build.XXXXXX")"
trap 'rm -rf -- "$temporary"' EXIT
cc="${CC:-cc}"
"$cc" -std=c11 -O2 -Wall -Wextra -Werror \
    "$root/ocean/decision_connect4/engine/tests/test_connect4.c" -lm \
    -o "$temporary/test_connect4_engine"
mv "$temporary/test_connect4_engine" "$root/build/test_connect4_engine"
cuda="${CUDA_HOME:-${CUDA_PATH:-}}"
nvcc="${CUDACXX:-${cuda:+$cuda/bin/nvcc}}"
nvcc="$(command -v "${nvcc:-nvcc}" || true)"
if [[ -z "$nvcc" ]]; then echo "Set CUDA_HOME or CUDACXX to a CUDA toolkit" >&2; exit 1; fi
cuda="${cuda:-$(dirname -- "$(dirname -- "$nvcc")")}"
cuda="$(cd -- "$cuda" && pwd)"
flags=(-std=c++17 -O2 --threads 0 -I"$root/src" -L"$cuda/lib64"
    -Xlinker -rpath -Xlinker "$cuda/lib64")
for dir in "$cuda/../math_libs" "$cuda/../../math_libs/$(basename "$cuda")"; do
    if [[ -f "$dir/include/cublas_v2.h" ]]; then
        flags+=(-I"$dir/include" -L"$dir/lib64" -Xlinker -rpath -Xlinker "$dir/lib64"); break
    fi
done
"$cc" -std=c11 -O2 -c "$root/ocean/decision_connect4/engine/connect4.c" -o "$temporary/engine.o"
"$cc" -std=c11 -O2 -c "$root/vendor/cJSON.c" -o "$temporary/json.o"
archive="$root/build/libpuffer_tokenizer.a"
rebuild=0
for source in "$root/tools/native_tokenizer/Cargo.toml" \
        "$root/tools/native_tokenizer/Cargo.lock" "$root/tools/native_tokenizer/src/lib.rs"; do
    if [[ ! -f "$archive" || "$source" -nt "$archive" ]]; then rebuild=1; fi
done
if [[ "$rebuild" == 1 ]]; then "$root/tools/build_native_tokenizer.sh"; fi
"$nvcc" "${flags[@]}" "$root/tests/test_decision_connect4_env.cu" \
    "$temporary/engine.o" "$temporary/json.o" "$archive" \
    -lcublas -lcudart -lm -ldl -lpthread -o "$temporary/test_decision_connect4_env"
mv "$temporary/test_decision_connect4_env" "$root/build/test_decision_connect4_env"
echo "Built build/test_connect4_engine and build/test_decision_connect4_env"
