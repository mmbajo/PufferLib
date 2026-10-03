#!/usr/bin/env bash
# Build full-state GPU checkpoint tests for either Snake policy adapter.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 || $# -gt 2 ]]; then echo "Usage: $0 decision_snake|decision_laya [OUT]" >&2; exit 2; fi
eval_env="$1"
case "$eval_env" in decision_snake|decision_laya) ;; *) exit 2;; esac
out="${2:-$root/build/test_native_checkpoint_$eval_env}"
cuda="${CUDA_HOME:-${CUDA_PATH:-}}"
nvcc="${CUDACXX:-${cuda:+$cuda/bin/nvcc}}"
nvcc="$(command -v "${nvcc:-nvcc}" || true)"
if [[ -z "$nvcc" ]]; then echo "CUDA compiler not found; set CUDA_HOME or CUDACXX" >&2; exit 1; fi
cuda="${cuda:-$(dirname -- "$(dirname -- "$nvcc")")}"
cuda="$(cd -- "$cuda" && pwd)"
raylib="${RAYLIB_HOME:-$root/raylib-5.5_linux_amd64}"
if [[ ! -f "$raylib/lib/libraylib.a" ]]; then
    echo "Run ./build.sh decision_laya first, or set RAYLIB_HOME" >&2; exit 1
fi
flags=(-std=c++17 -O2 --threads 0 -I"$root" -I"$root/src" -I"$root/vendor"
    -I"$raylib/include" -L"$cuda/lib64" -Xlinker -rpath -Xlinker "$cuda/lib64"
    -DPRECISION_FLOAT "-DPUFFER_${eval_env^^}" -DPLATFORM_DESKTOP
    "-DENV_HEADER=\"ocean/$eval_env/$eval_env.h\""
    "-DPUFFER_ENV_NAME=\"$eval_env\"" "-DENV_NAME=$eval_env"
    -Xcompiler=-fopenmp -Xcompiler=-Wno-narrowing
    --diag-suppress=2361 --diag-suppress=111 --diag-suppress=128)
if [[ -n "${NVCC_ARCH:-}" ]]; then flags+=(-arch="$NVCC_ARCH"); else
    flags+=('-gencode=arch=compute_80,code=[sm_80,compute_80]')
    case "$("$nvcc" --list-gpu-code)" in *sm_90*) flags+=('-gencode=arch=compute_90,code=sm_90');; esac
fi
for dir in "$cuda/../math_libs" "$cuda/../../math_libs/$(basename "$cuda")"; do
    if [[ -f "$dir/include/cublas_v2.h" ]]; then
        flags+=(-I"$dir/include" -L"$dir/lib64" -Xlinker -rpath -Xlinker "$dir/lib64"); break
    fi
done
nccl_roots=("${NCCL_HOME:-${NCCL_ROOT:-}}" "$cuda/../comm_libs/nccl" "$cuda/../../comm_libs/nccl")
for dir in "${NCCL_INCLUDE_DIR:-}" "${nccl_roots[@]/%//include}" /usr/include "$cuda/include"; do
    if [[ -f "$dir/nccl.h" ]]; then flags+=(-I"$dir"); break; fi
done
for dir in "${NCCL_LIB_DIR:-}" "${nccl_roots[@]/%//lib}" "${nccl_roots[@]/%//lib64}" /usr/lib/x86_64-linux-gnu "$cuda/lib64"; do
    if [[ -f "$dir/libnccl.so" ]]; then flags+=(-L"$dir" -Xlinker -rpath -Xlinker "$dir"); break; fi
done
if [[ -f "$cuda/lib64/stubs/libnvidia-ml.so" ]]; then flags+=(-L"$cuda/lib64/stubs"); fi
if [[ -f "$cuda/lib64/libnvJitLink.so" ]]; then flags+=(-Xlinker=--no-as-needed,-lnvJitLink,--as-needed); fi
libraries=("$raylib/lib/libraylib.a" -lcublas -lcudart -lnccl -lnvidia-ml
    -lcusolver -lcurand -lgomp -ldl -lm -lpthread)
if [[ "$("${CC:-cc}" -print-file-name=libGL.so)" == libGL.so ]]; then
    libraries+=(-l:libGL.so.1)
else libraries+=(-lGL); fi
objects=()
if [[ "$eval_env" == decision_laya ]]; then
    archive="$root/build/libpuffer_tokenizer.a"
    for source in "$root/tools/native_tokenizer/Cargo.toml" "$root/tools/native_tokenizer/Cargo.lock" \
            "$root/tools/native_tokenizer/src/lib.rs"; do
        if [[ ! -f "$archive" || "$source" -nt "$archive" ]]; then
            "$root/tools/build_native_tokenizer.sh"; break
        fi
    done
    objects+=("$archive")
fi
mkdir -p -- "$(dirname -- "$out")"
temporary="$(mktemp -d "$(dirname -- "$out")/.checkpoint-build.XXXXXX")"
trap 'rm -rf -- "$temporary"' EXIT
"${CC:-cc}" -std=c11 -O2 -c "$root/vendor/cJSON.c" -o "$temporary/json.o"
"${CC:-cc}" -std=c11 -O2 -c "$root/ocean/decision_snake/engine/snake.c" -o "$temporary/snake.o"
"$nvcc" "${flags[@]}" "$root/tests/test_native_checkpoint.cu" \
    "${objects[@]}" "$temporary/json.o" "$temporary/snake.o" "${libraries[@]}" -o "$temporary/test"
mv -- "$temporary/test" "$out"
echo "Built $out"
