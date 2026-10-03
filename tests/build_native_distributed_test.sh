#!/usr/bin/env bash
# CPU-only: no CUDA toolkit or GPU allocation is required.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 1 ]]; then echo "Usage: $0 [OUT]" >&2; exit 2; fi
out="${1:-$root/build/test_native_distributed}"
mkdir -p -- "$(dirname -- "$out")"
"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror \
    "$root/tests/test_native_distributed.cpp" -o "$out"
echo "Built $out (CPU-only launcher tests)"
