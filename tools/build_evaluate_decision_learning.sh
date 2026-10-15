#!/usr/bin/env bash
# Build the inference-only native CartPole paired-evaluation runner.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 1 ]]; then echo "Usage: $0 [OUT]" >&2; exit 2; fi
out="${1:-$root/build/evaluate_decision_learning}"
cuda="${CUDA_HOME:-${CUDA_PATH:-}}"
nvcc="${CUDACXX:-${cuda:+$cuda/bin/nvcc}}"
nvcc="$(command -v "${nvcc:-nvcc}" || true)"
if [[ -z "$nvcc" ]]; then echo "CUDA compiler not found; set CUDA_HOME or CUDACXX" >&2; exit 1; fi
cuda="${cuda:-$(dirname -- "$(dirname -- "$nvcc")")}"
cuda="$(cd -- "$cuda" && pwd)"
raylib="${RAYLIB_HOME:-$root/raylib-5.5_linux_amd64}"
if [[ ! -f "$raylib/lib/libraylib.a" ]]; then
    echo "Run ./build.sh decision_cartpole first, or set RAYLIB_HOME" >&2; exit 1
fi
flags=(-std=c++17 -O2 --threads 0 -I"$root" -I"$root/src" -I"$root/vendor"
    -I"$raylib/include" -L"$cuda/lib64" -Xlinker -rpath -Xlinker "$cuda/lib64"
    -DPRECISION_FLOAT -DPUFFER_DECISION_CARTPOLE -DPLATFORM_DESKTOP
    '-DENV_HEADER="ocean/decision_cartpole/decision_cartpole.h"'
    '-DPUFFER_ENV_NAME="decision_cartpole"' -DENV_NAME=decision_cartpole
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
archive="$root/build/libpuffer_tokenizer.a"
for source in "$root/tools/native_tokenizer/Cargo.toml" "$root/tools/native_tokenizer/Cargo.lock" \
        "$root/tools/native_tokenizer/src/lib.rs"; do
    if [[ ! -f "$archive" || "$source" -nt "$archive" ]]; then
        "$root/tools/build_native_tokenizer.sh"; break
    fi
done
mkdir -p -- "$(dirname -- "$out")"
temporary="$(mktemp -d "$(dirname -- "$out")/.distributed-test-build.XXXXXX")"
trap 'rm -rf -- "$temporary"' EXIT
"${CC:-cc}" -std=c11 -O2 -c "$root/vendor/cJSON.c" -o "$temporary/json.o"
"$nvcc" "${flags[@]}" "$root/tools/evaluate_decision_learning.cu" \
    "$archive" "$temporary/json.o" "${libraries[@]}" -o "$temporary/test"
mv -- "$temporary/test" "$out"
echo "Built $out (single-GPU inference; --policy.bundle=BUNDLE [--weights=CHECKPOINT])"
