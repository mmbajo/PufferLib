#!/usr/bin/env bash
# Build the independent model oracle or the test that includes the native core.
# The Puffer test uses Raylib already fetched by ./build.sh decision_snake.
set -euo pipefail

kind="${1:-puffer}"
case "$kind" in
    puffer|transformer) ;;
    *) echo "Usage: $0 [puffer|transformer] [OUT]" >&2; exit 2 ;;
esac
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="${2:-$root/build/test_decision_$kind}"
cuda="${CUDA_HOME:-${CUDA_PATH:-}}"
nvcc="${CUDACXX:-${cuda:+$cuda/bin/nvcc}}"
nvcc="$(command -v "${nvcc:-nvcc}" || true)"
if [[ -z "$nvcc" ]]; then
    echo "CUDA compiler not found; set CUDA_HOME or CUDACXX" >&2
    exit 1
fi
cuda="${cuda:-$(dirname -- "$(dirname -- "$nvcc")")}"
cuda="$(cd -- "$cuda" && pwd)"
flags=(-std=c++17 -O2 --threads 0 -I"$root" -I"$root/src"
       -L"$cuda/lib64" -Xlinker -rpath -Xlinker "$cuda/lib64")
if [[ -n "${NVCC_ARCH:-}" ]]; then
    flags+=(-arch="$NVCC_ARCH")
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
libraries=(-lcublas)
objects=()
mkdir -p -- "$(dirname -- "$out")"
temporary="$(mktemp -d "$(dirname -- "$out")/.decision-test-build.XXXXXX")"
trap 'rm -rf -- "$temporary"' EXIT

if [[ "$kind" == puffer ]]; then
    raylib="${RAYLIB_HOME:-$root/raylib-5.5_linux_amd64}"
    if [[ ! -f "$raylib/lib/libraylib.a" ]]; then
        echo "Run ./build.sh decision_snake first to fetch Raylib, or set RAYLIB_HOME" >&2
        exit 1
    fi
    nccl_roots=("${NCCL_HOME:-${NCCL_ROOT:-}}" "$cuda/../comm_libs/nccl" "$cuda/../../comm_libs/nccl")
    for dir in "${NCCL_INCLUDE_DIR:-}" "${nccl_roots[@]/%//include}"; do
        if [[ -f "$dir/nccl.h" ]]; then flags+=(-I"$dir"); break; fi
    done
    for dir in "${NCCL_LIB_DIR:-}" "${nccl_roots[@]/%//lib}" "${nccl_roots[@]/%//lib64}"; do
        if [[ -f "$dir/libnccl.so" ]]; then
            flags+=(-L"$dir" -Xlinker -rpath -Xlinker "$dir")
            break
        fi
    done
    if [[ -f "$cuda/lib64/stubs/libnvidia-ml.so" ]]; then
        flags+=(-L"$cuda/lib64/stubs")
    fi
    if [[ -f "$cuda/lib64/libnvJitLink.so" ]]; then
        flags+=(-Xlinker=--no-as-needed,-lnvJitLink,--as-needed)
    fi
    flags+=(-DPRECISION_FLOAT -DPUFFER_DECISION_SNAKE -DPLATFORM_DESKTOP
            '-DENV_HEADER="ocean/decision_snake/decision_snake.h"'
            '-DPUFFER_ENV_NAME="decision_snake"' -DENV_NAME=decision_snake
            -I"$root/vendor" -I"$raylib/include" -Xcompiler=-fopenmp
            -Xcompiler=-Wno-narrowing --diag-suppress=2361
            --diag-suppress=111 --diag-suppress=128)
    # NVCC's default Linux host compiler is GCC, whose OpenMP runtime is libgomp.
    libraries+=("$raylib/lib/libraylib.a" -lcudart -lnccl -lnvidia-ml
                -lcusolver -lcurand -lgomp -ldl -lm -lpthread)
    if [[ "$("${CC:-cc}" -print-file-name=libGL.so)" == libGL.so ]]; then
        libraries+=(-l:libGL.so.1)
    else
        libraries+=(-lGL)
    fi
    "${CC:-cc}" -std=c11 -O2 -c "$root/ocean/decision_snake/engine/snake.c" \
        -o "$temporary/snake.o"
    objects+=("$temporary/snake.o")
fi

"$nvcc" "${flags[@]}" "$root/tests/test_decision_$kind.cu" \
    "${objects[@]}" "${libraries[@]}" -o "$temporary/test"
mv -- "$temporary/test" "$out"
echo "Built $out"
