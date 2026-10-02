#!/usr/bin/env bash
# Build the independent model oracle or the test that includes the native core.
# The Puffer test uses Raylib already fetched by ./build.sh decision_snake.
set -euo pipefail

kind="${1:-puffer}"
case "$kind" in
    puffer|transformer|policy-cartpole|policy-laya) ;;
    *) echo "Usage: $0 [puffer|transformer|policy-cartpole|policy-laya] [OUT]" >&2; exit 2 ;;
esac
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
name="test_decision_${kind//-/_}"
out="${2:-$root/build/$name}"
source="$root/tests/test_decision_$kind.cu"
env=decision_snake
case "$kind" in
    policy-cartpole) env=decision_cartpole; source="$root/tests/test_decision_policy.cu" ;;
    policy-laya) env=decision_laya; source="$root/tests/test_decision_policy.cu" ;;
esac
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

if [[ "$kind" != transformer ]]; then
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
    flags+=(-DPRECISION_FLOAT "-DPUFFER_${env^^}" -DPLATFORM_DESKTOP
            "-DENV_HEADER=\"ocean/$env/$env.h\""
            "-DPUFFER_ENV_NAME=\"$env\"" "-DENV_NAME=$env"
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
    if [[ "$env" != decision_cartpole ]]; then
        "${CC:-cc}" -std=c11 -O2 -c "$root/ocean/decision_snake/engine/snake.c" \
            -o "$temporary/snake.o"
        objects+=("$temporary/snake.o")
    fi
    if [[ "$kind" == policy-* ]]; then
        archive="$root/build/libpuffer_tokenizer.a"
        rebuild=0
        for item in "$root/tools/native_tokenizer/Cargo.toml" \
                "$root/tools/native_tokenizer/Cargo.lock" "$root/tools/native_tokenizer/src/lib.rs"; do
            if [[ ! -f "$archive" || "$item" -nt "$archive" ]]; then rebuild=1; fi
        done
        if [[ "$rebuild" == 1 ]]; then "$root/tools/build_native_tokenizer.sh"; fi
        "${CC:-cc}" -std=c11 -O2 -c "$root/vendor/cJSON.c" -o "$temporary/json.o"
        objects+=("$archive" "$temporary/json.o")
    fi
fi

"$nvcc" "${flags[@]}" "$source" \
    "${objects[@]}" "${libraries[@]}" -o "$temporary/test"
mv -- "$temporary/test" "$out"
echo "Built $out"
