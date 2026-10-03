#!/usr/bin/env bash
# Build real board-policy initialization and Muon dispatch regression test.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$root/build/test_native_optimizer_dispatch}"
cuda="${CUDA_HOME:-${CUDA_PATH:-}}"
nvcc="${CUDACXX:-${cuda:+$cuda/bin/nvcc}}"
nvcc="$(command -v "${nvcc:-nvcc}" || true)"
if [[ -z "$nvcc" ]]; then echo 'Set CUDA_HOME or CUDACXX' >&2; exit 1; fi
cuda="${cuda:-$(dirname -- "$(dirname -- "$nvcc")")}"
cuda="$(cd -- "$cuda" && pwd)"
raylib="${RAYLIB_HOME:-$root/raylib-5.5_linux_amd64}"
flags=(-std=c++17 -O2 --threads 0 -DPRECISION_FLOAT -DPUFFER_DECISION_SNAKE
    -DPLATFORM_DESKTOP '-DENV_HEADER="ocean/decision_snake/decision_snake.h"'
    '-DPUFFER_ENV_NAME="decision_snake"' -DENV_NAME=decision_snake
    -I"$root" -I"$root/src" -I"$root/vendor" -I"$raylib/include"
    -L"$cuda/lib64" -Xlinker -rpath -Xlinker "$cuda/lib64"
    -Xcompiler=-fopenmp -Xcompiler=-Wno-narrowing
    --diag-suppress=2361 --diag-suppress=111 --diag-suppress=128)
if [[ -n "${NVCC_ARCH:-}" ]]; then flags+=(-arch="$NVCC_ARCH")
else
    flags+=('-gencode=arch=compute_80,code=[sm_80,compute_80]')
    case "$("$nvcc" --list-gpu-code)" in
        *sm_90*) flags+=('-gencode=arch=compute_90,code=sm_90') ;;
    esac
fi
for dir in "$cuda/../math_libs" "$cuda/../../math_libs/$(basename "$cuda")"; do
    if [[ -f "$dir/include/cublas_v2.h" ]]; then
        flags+=(-I"$dir/include" -L"$dir/lib64" -Xlinker -rpath -Xlinker "$dir/lib64")
        break
    fi
done
nccl_roots=("${NCCL_HOME:-${NCCL_ROOT:-}}" "$cuda/../comm_libs/nccl" "$cuda/../../comm_libs/nccl")
for dir in "${NCCL_INCLUDE_DIR:-}" "${nccl_roots[@]/%//include}"; do
    if [[ -f "$dir/nccl.h" ]]; then flags+=(-I"$dir"); break; fi
done
for dir in "${NCCL_LIB_DIR:-}" "${nccl_roots[@]/%//lib}" "${nccl_roots[@]/%//lib64}"; do
    if [[ -f "$dir/libnccl.so" ]]; then
        flags+=(-L"$dir" -Xlinker -rpath -Xlinker "$dir"); break
    fi
done
if [[ -f "$cuda/lib64/stubs/libnvidia-ml.so" ]]; then flags+=(-L"$cuda/lib64/stubs"); fi
if [[ -f "$cuda/lib64/libnvJitLink.so" ]]; then
    flags+=(-Xlinker=--no-as-needed,-lnvJitLink,--as-needed)
fi
gl=-lGL
if [[ "$("${CC:-cc}" -print-file-name=libGL.so)" == libGL.so ]]; then gl=-l:libGL.so.1; fi
mkdir -p -- "$(dirname -- "$out")"
temporary="$(mktemp -d "$(dirname -- "$out")/.adam-test-build.XXXXXX")"
trap 'rm -rf -- "$temporary"' EXIT
"${CC:-cc}" -std=c11 -O2 -c "$root/ocean/decision_snake/engine/snake.c" -o "$temporary/snake.o"
"$nvcc" "${flags[@]}" "$root/tests/test_native_optimizer_dispatch.cu" \
    "$temporary/snake.o" "$raylib/lib/libraylib.a" -lcublas -lcudart -lnccl -lnvidia-ml \
    -lcusolver -lcurand -lgomp -ldl -lm -lpthread "$gl" -o "$temporary/test"
mv -- "$temporary/test" "$out"
echo "Built $out"
