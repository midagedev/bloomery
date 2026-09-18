#!/usr/bin/env bash
# Builds tools/ref/q3k_ref.cpp on the box against ik_llama.cpp's libggml
# (CUDA backend compiled in). Override the ik checkout with IK=<path>.
set -euo pipefail
IK=${IK:-/home/user/ik_llama.cpp}
CUDA=${CUDA_HOME:-/usr/local/cuda}
OUT=${OUT:-/root/mulle-data/q3k_ref}
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$(dirname "$OUT")"
g++ -O3 -std=c++17 \
    -I"$IK/ggml/include" \
    -I"$CUDA/include" \
    "$HERE/q3k_ref.cpp" \
    -L"$IK/build/ggml/src" -lggml \
    -L"$CUDA/lib64" -lcudart \
    -Wl,-rpath,"$IK/build/ggml/src" \
    -Wl,-rpath,"$CUDA/lib64" \
    -o "$OUT"
echo "built $OUT"
