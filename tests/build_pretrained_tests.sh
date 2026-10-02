#!/usr/bin/env bash
# Build native oracle harnesses. Running GPU tests is a separate, explicit step.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
kind="${1:-all}"
case "$kind" in
    all)
        if [[ $# -gt 1 ]]; then echo "all does not accept an output path" >&2; exit 2; fi
        for target in tokenizer encoder decision bundle input observation checkpoint cartpole-stock cartpole-env lightsout-env layout-2 layout-4 layout-8 layout-25 policy-cartpole policy-laya policy-connect4 policy-lightsout policy-2048; do "$0" "$target"; done
        exit 0 ;;
    policy-cartpole|policy-laya|policy-connect4|policy-lightsout|policy-2048)
        exec "$root/tests/build_decision_tests.sh" "$@" ;;
    encoder|decision|bundle|input|tokenizer|observation|checkpoint|cartpole-stock|cartpole-env|lightsout-env|layout-2|layout-4|layout-8|layout-25) ;;
    *) echo "Usage: $0 [all|encoder|decision|bundle|input|tokenizer|observation|checkpoint|cartpole-stock|cartpole-env|lightsout-env|layout-{2,4,8,25}|policy-{cartpole,laya,connect4,lightsout,2048}] [OUT]" >&2; exit 2 ;;
esac
case "$kind" in
    tokenizer) name=test_native_tokenizer ;;
    observation) name=test_decision_laya_observation ;;
    checkpoint) name=test_laya_checkpoint ;;
    cartpole-stock) name=test_decision_cartpole_stock ;;
    cartpole-env) name=test_decision_cartpole_env ;;
    lightsout-env) name=test_decision_lightsout_env ;;
    layout-*) name="test_decision_policy_layout_${kind#layout-}" ;;
    *) name="test_pretrained_$kind" ;;
esac
out="${2:-$root/build/$name}"
mkdir -p -- "$(dirname -- "$out")"
temporary="$(mktemp -d "$(dirname -- "$out")/.pretrained-test-build.XXXXXX")"
trap 'rm -rf -- "$temporary"' EXIT

if [[ "$kind" == cartpole-stock ]]; then
    "${CXX:-c++}" -std=c++17 -O2 -x c++ -DTEST_STOCK_CARTPOLE -I"$root/src" \
        "$root/tests/test_decision_cartpole_env.cu" -o "$temporary/test"
    mv -- "$temporary/test" "$out"
    echo "Built $out"
    exit 0
fi

source="$root/tests/$name.cu"
case "$kind" in
    layout-*) source="$root/tests/test_decision_policy_layout.cu" ;;
esac

objects=()
libraries=(-lm -ldl -lpthread)
needs_bundle=0
case "$kind" in
    bundle|input|observation|checkpoint|cartpole-env|lightsout-env|layout-*) needs_bundle=1 ;;
esac
if [[ "$kind" == tokenizer || "$needs_bundle" == 1 ]]; then
    archive="$root/build/libpuffer_tokenizer.a"
    rebuild=0
    for tokenizer_source in "$root/tools/native_tokenizer/Cargo.toml" \
            "$root/tools/native_tokenizer/Cargo.lock" "$root/tools/native_tokenizer/src/lib.rs"; do
        if [[ ! -f "$archive" || "$tokenizer_source" -nt "$archive" ]]; then rebuild=1; fi
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
    if [[ "$kind" == layout-* ]]; then flags+=("-DDECISION_ACTIONS=${kind#layout-}"); fi
    if [[ "$needs_bundle" == 1 ]]; then
        "${CC:-cc}" -std=c11 -O2 -c "$root/vendor/cJSON.c" -o "$temporary/json.o"
        objects+=("$temporary/json.o")
    fi
    if [[ "$kind" == observation ]]; then
        "${CC:-cc}" -std=c11 -O2 -c "$root/ocean/decision_snake/engine/snake.c" -o "$temporary/snake.o"
        objects+=("$temporary/snake.o")
    fi
    if [[ "$kind" == lightsout-env ]]; then
        "${CC:-cc}" -std=c11 -O2 -c "$root/ocean/decision_lightsout/engine/lightsout.c" -o "$temporary/lightsout.o"
        objects+=("$temporary/lightsout.o")
    fi
    "$nvcc" "${flags[@]}" "$source" \
        "${objects[@]}" -lcublas -lcudart "${libraries[@]}" -o "$temporary/test"
fi
mv -- "$temporary/test" "$out"
echo "Built $out"
