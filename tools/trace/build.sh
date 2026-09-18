#!/usr/bin/env bash
# Build tierinfer-trace against an existing llama.cpp checkout and build.
#
# Default target is the b10482 build the 480B validation runs on; the b9888
# tree at ~/llama.cpp still builds with LLAMA_CPP=$HOME/llama.cpp.
# Nothing in the llama.cpp tree is modified or rebuilt: this links against the
# headers and shared objects that are already there. Point LLAMA_CPP at a
# different checkout to build against another version.
set -euo pipefail

LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp-qwen38}"
LLAMA_BUILD="${LLAMA_BUILD:-$LLAMA_CPP/build}"
OUT="${OUT:-$(cd "$(dirname "$0")/../.." && pwd)/build/tierinfer-trace}"

for f in "$LLAMA_CPP/include/llama.h" "$LLAMA_CPP/ggml/include/ggml.h" \
         "$LLAMA_BUILD/bin/libllama.so"; do
    [ -f "$f" ] || { echo "missing $f — set LLAMA_CPP / LLAMA_BUILD" >&2; exit 2; }
done

mkdir -p "$(dirname "$OUT")"
g++ -std=c++17 -O2 -Wall -Wextra \
    -I"$LLAMA_CPP/include" -I"$LLAMA_CPP/ggml/include" \
    "$(dirname "$0")/tierinfer_trace.cpp" \
    -L"$LLAMA_BUILD/bin" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$LLAMA_BUILD/bin" \
    -o "$OUT"

echo "built $OUT against $(readlink -f "$LLAMA_BUILD/bin/libllama.so")"
