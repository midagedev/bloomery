#!/usr/bin/env bash
# Build the stage-1 oracle instrument on the box. Mirrors tools/ref/build.sh.
# The runner rsyncs this tree with --delete, so the binary lives outside it.
set -euo pipefail
: "${IK:=/home/user/ik_llama.cpp}"
MULLE_DATA=${MULLE_DATA:-/root/mulle-data}
OUT=${DUMP_OUT:-$MULLE_DATA/bin}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT"
g++ -std=c++17 -O2 -o "$OUT/dump_ref" "$HERE/tools/ref/dump_ref.cpp" \
  -I"$IK/ggml/include" -I"$IK/include" -I"$IK/common" -I"$IK/src" \
  -L"$IK/build/common" -L"$IK/build/src" -L"$IK/build/ggml/src" \
  -lcommon -lllama -lggml \
  -Wl,-rpath,"$IK/build/src" -Wl,-rpath,"$IK/build/ggml/src"
echo "built $OUT/dump_ref"
