#!/usr/bin/env bash
# Build libtierinfer_mmap.so against an existing llama.cpp checkout and build.
# Nothing in the llama.cpp tree is modified; the shim is preloaded into its binaries.
set -euo pipefail
LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp-qwen38}"
LLAMA_BUILD="${LLAMA_BUILD:-$LLAMA_CPP/build}"
OUT="${OUT:-$(cd "$(dirname "$0")/../.." && pwd)/build/libtierinfer_mmap.so}"
for f in "$LLAMA_CPP/include/llama.h" "$LLAMA_CPP/ggml/include/ggml.h" "$LLAMA_BUILD/bin/libllama.so"; do
    [ -f "$f" ] || { echo "missing $f — set LLAMA_CPP / LLAMA_BUILD" >&2; exit 2; }
done
mkdir -p "$(dirname "$OUT")"
gcc -std=gnu11 -O2 -Wall -Wextra -fPIC -shared \
    -I"$LLAMA_CPP/include" -I"$LLAMA_CPP/ggml/include" \
    "$(dirname "$0")/tierinfer_mmap.c" \
    -L"$LLAMA_BUILD/bin" -lllama -lggml-base -ldl -lpthread \
    -Wl,-rpath,"$LLAMA_BUILD/bin" \
    -o "$OUT"
echo "built $OUT"
