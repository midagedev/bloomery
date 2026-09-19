#!/usr/bin/env bash
# Build the stage-1 dequantization oracle on the box (round 1-1).
# Runs under tools/box.sh (toolchain env already sourced).
# NOTE: the runner rsyncs this tree with --delete before every command, so
# the binary must live OUTSIDE the tree; it goes to $BLOOMERY_DATA/bin.
set -euo pipefail
: "${IK:=/home/user/ik_llama.cpp}"
BLOOMERY_DATA=${BLOOMERY_DATA:-/root/bloomery-data}
OUT=${DEQUANT_OUT:-$BLOOMERY_DATA/bin}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT"
g++ -std=c++17 -O2 -o "$OUT/dequant_ref" "$HERE/tools/ref/dequant_ref.cpp" \
  -I"$IK/ggml/include" \
  -L"$IK/build/ggml/src" -lggml \
  -Wl,-rpath,"$IK/build/ggml/src"
echo "built $OUT/dequant_ref"
