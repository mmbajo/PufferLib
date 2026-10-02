#!/usr/bin/env bash
# Build native oracle harnesses. Running GPU tests is a separate, explicit step.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
kind="${1:-all}"
case "$kind" in
    all)
        if [[ $# -gt 1 ]]; then echo "all does not accept an output path" >&2; exit 2; fi
        for target in tokenizer encoder decision bundle input observation checkpoint; do "$0" "$target"; done
        exit 0 ;;
    encoder|decision|bundle|input|tokenizer|observation|checkpoint) ;;
    *) echo "Usage: $0 [all|encoder|decision|bundle|input|tokenizer|observation|checkpoint] [OUT]" >&2; exit 2 ;;
esac
case "$kind" in
    tokenizer) name=test_native_tokenizer ;;
    observation) name=test_decision_laya_observation ;;
    checkpoint) name=test_laya_checkpoint ;;
    *) name="test_pretrained_$kind" ;;
esac
out="${2:-$root/build/$name}"
mkdir -p -- "$(dirname -- "$out")"
temporary="$(mktemp -d "$(dirname -- "$out")/.pretrained-test-build.XXXXXX")"
trap 'rm -rf -- "$temporary"' EXIT

objects=()
libraries=(-lm -ldl -lpthread)
if [[ "$kind" == tokenizer || "$kind" == bundle || "$kind" == input ||
        "$kind" == observation || "$kind" == checkpoint ]]; then
    archive="$root/build/libpuffer_tokenizer.a"
    rebuild=0
    for source in "$root/tools/native_tokenizer/Cargo.toml" \
            "$root/tools/native_tokenizer/Cargo.lock" "$root/tools/native_tokenizer/src/lib.rs"; do
        if [[ ! -f "$archive" || "$source" -nt "$archive" ]]; then rebuild=1; fi
    done
    if [[ "$rebuild" == 1 ]]; then "$root/tools/build_native_tokenizer.sh"; fi
    objects+=("$archive")
fi
if [[ "$kind" == tokenizer ]]; then
    "${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror \
        "$root/tests/test_native_tokenizer.cpp" "${objects[@]}" "${libraries[@]}" -o "$temporary/test"
else
    cuda="${CUDA_HOME:-${CUDA_PATH:-}}"
    nvcc="${CUDACXX:-${cuda:+$cuda/bin/nvcc}}"
    nvcc="$(command -v "${nvcc:-nvcc}" || true)"
    if [[ -z "$nvcc" ]]; then echo "Set CUDA_HOME or CUDACXX to a CUDA toolkit" >&2; exit 1; fi
    cuda="${cuda:-$(dirname -- "$(dirname -- "$nvcc")")}"
    cuda="$(cd -- "$cuda" && pwd)"
    flags=(-std=c++17 -O2 --threads 0 -I"$root/src" -L"$cuda/lib64"
           -Xlinker -rpath -Xlinker "$cuda/lib64")
    if [[ -n "${NVCC_ARCH:-}" ]]; then
        flags+=(-arch="$NVCC_ARCH")
    else
        flags+=('-gencode=arch=compute_80,code=[sm_80,compute_80]')
        case "$("$nvcc" --list-gpu-code)" in *sm_90*) flags+=('-gencode=arch=compute_90,code=sm_90');; esac
    fi
    for dir in "$cuda/../math_libs" "$cuda/../../math_libs/$(basename "$cuda")"; do
        if [[ -f "$dir/include/cublas_v2.h" ]]; then
            flags+=(-I"$dir/include" -L"$dir/lib64" -Xlinker -rpath -Xlinker "$dir/lib64"); break
        fi
    done
    if [[ "$kind" == bundle || "$kind" == input || "$kind" == observation || "$kind" == checkpoint ]]; then
        "${CC:-cc}" -std=c11 -O2 -c "$root/vendor/cJSON.c" -o "$temporary/json.o"
        objects+=("$temporary/json.o")
    fi
    if [[ "$kind" == observation ]]; then
        "${CC:-cc}" -std=c11 -O2 -c "$root/ocean/decision_snake/engine/snake.c" -o "$temporary/snake.o"
        objects+=("$temporary/snake.o")
    fi
    "$nvcc" "${flags[@]}" "$root/tests/$name.cu" \
        "${objects[@]}" -lcublas -lcudart "${libraries[@]}" -o "$temporary/test"
fi
mv -- "$temporary/test" "$out"
echo "Built $out"
