#!/usr/bin/env bash
# Build the 32-prompt ik reference on the box. Mirrors tools/ref/build.sh.
# The runner rsyncs this tree with --delete, so the binary lives outside it.
set -euo pipefail
: "${IK:=/home/user/ik_llama.cpp}"
BLOOMERY_DATA=${BLOOMERY_DATA:-/root/bloomery-data}
OUT=${ARGMAX_OUT:-$BLOOMERY_DATA/bin}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT"
g++ -std=c++17 -O2 -o "$OUT/argmax_ref" "$HERE/tools/ref/argmax_ref.cpp" \
  -I"$IK/ggml/include" -I"$IK/include" -I"$IK/common" -I"$IK/src" \
  -L"$IK/build/common" -L"$IK/build/src" -L"$IK/build/ggml/src" \
  -lcommon -lllama -lggml \
  -Wl,-rpath,"$IK/build/src" -Wl,-rpath,"$IK/build/ggml/src"
echo "built $OUT/argmax_ref"
