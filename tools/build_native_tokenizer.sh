#!/usr/bin/env bash
# Build a native static tokenizer library; never installs a toolchain/packages.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARGO="${CARGO:-cargo}"
if ! command -v "$CARGO" >/dev/null 2>&1; then
    echo "Rust cargo is required to build the native tokenizer (tested with Rust 1.90.0)." >&2
    echo "Set CARGO to its executable; see tools/native_tokenizer/README.md." >&2
    exit 1
fi
OUT="${1:-$ROOT/build/libpuffer_tokenizer.a}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/build/native_tokenizer}"
"$CARGO" build --manifest-path "$ROOT/tools/native_tokenizer/Cargo.toml" --release --locked
mkdir -p "$(dirname "$OUT")"
cp "$CARGO_TARGET_DIR/release/libpuffer_tokenizer.a" "$OUT"
echo "Built $OUT"
