#!/usr/bin/env bash
# Build the stage-2 CPU reference harness on the box.
# Runs under tools/box.sh (toolchain env already sourced).
# NOTE: the runner rsyncs this tree with --delete before every command, so
# the binary must live OUTSIDE the tree; it goes to $MULLE_DATA/bin (default /root/mulle-data).
set -euo pipefail
: "${IK:=/home/user/ik_llama.cpp}"
MULLE_DATA=${MULLE_DATA:-/root/mulle-data}
OUT=${Q3K_OUT:-$MULLE_DATA/bin}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT"
g++ -std=c++17 -O2 -o "$OUT/q3k_cpu_ref" "$HERE/tools/ref/q3k_cpu_ref.cpp" \
  -I"$IK/ggml/include" \
  -L"$IK/build/ggml/src" -lggml \
  -Wl,-rpath,"$IK/build/ggml/src"
echo "built $OUT/q3k_cpu_ref"
