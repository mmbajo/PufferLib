#!/usr/bin/env bash
# Standalone native typed-decision inference; does not link Puffer's game UI.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$root/build/pretrained_decision}"
cuda="${CUDA_HOME:-${CUDA_PATH:-}}"
nvcc="${CUDACXX:-${cuda:+$cuda/bin/nvcc}}"
nvcc="$(command -v "${nvcc:-nvcc}" || true)"
if [[ -z "$nvcc" ]]; then echo "Set CUDA_HOME or CUDACXX to a CUDA toolkit" >&2; exit 1; fi
cuda="${cuda:-$(dirname -- "$(dirname -- "$nvcc")")}"
cuda="$(cd -- "$cuda" && pwd)"
flags=(-std=c++17 -O2 --threads 0 -I"$root/src" -L"$cuda/lib64"
       -Xlinker -rpath -Xlinker "$cuda/lib64")
if [[ -n "${NVCC_ARCH:-}" ]]; then flags+=(-arch="$NVCC_ARCH"); else
    flags+=('-gencode=arch=compute_80,code=[sm_80,compute_80]')
    case "$("$nvcc" --list-gpu-code)" in *sm_90*) flags+=('-gencode=arch=compute_90,code=sm_90');; esac
fi
for dir in "$cuda/../math_libs" "$cuda/../../math_libs/$(basename "$cuda")"; do
    if [[ -f "$dir/include/cublas_v2.h" ]]; then
        flags+=(-I"$dir/include" -L"$dir/lib64" -Xlinker -rpath -Xlinker "$dir/lib64"); break
    fi
done
archive="${PUFFER_TOKENIZER_LIB:-$root/build/libpuffer_tokenizer.a}"
if [[ ! -f "$archive" || "$root/tools/native_tokenizer/Cargo.toml" -nt "$archive" ||
      "$root/tools/native_tokenizer/Cargo.lock" -nt "$archive" ||
      "$root/tools/native_tokenizer/src/lib.rs" -nt "$archive" ]]; then
    "$root/tools/build_native_tokenizer.sh" "$archive"
fi
mkdir -p -- "$(dirname -- "$out")"
temporary="$(mktemp -d "$(dirname -- "$out")/.pretrained-build.XXXXXX")"
trap 'rm -rf -- "$temporary"' EXIT
"${CC:-cc}" -std=c11 -O2 -c "$root/vendor/cJSON.c" -o "$temporary/json.o"
"$nvcc" "${flags[@]}" "$root/src/pretrained_decision_cli.cu" "$temporary/json.o" \
    "$archive" -lcublas -lcudart -ldl -lpthread -lm -o "$temporary/model"
mv -- "$temporary/model" "$out"
echo "Built $out"
