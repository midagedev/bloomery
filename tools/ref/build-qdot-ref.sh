#!/usr/bin/env bash
# Build and run the four x4 reference harnesses behind gate-qdot (on the box).
# Runs under tools/box.sh (toolchain env already sourced).
#
# These four link ik's own kernel tables, not just libggml: each one calls the
# entry the oracle dispatches (iqk_set_kernels_kquants /
# iqk_set_kernels_legacy_quants) on the first aligned tensor of its type and
# writes $BLOOMERY_DATA/ref/<name>-ik-dot.txt. gate-qdot's four hw tests read
# those dumps; without them the tests fail on a missing file, which is a
# standing red, not a gate.
#
# Unlike build.sh the binaries are built AND run here: the dump is the product,
# the binary is scaffolding. Both go outside the tree, which tools/box.sh
# rsyncs with --delete before every command.
#
# IQK_IMPLEMENT plus the three ik include roots is what makes the kernel table
# visible; -mavx2 -mfma -mf16c is the ISA those kernels are written for (a
# build without them does not compile).
set -euo pipefail
: "${IK:=/home/user/ik_llama.cpp}"
BLOOMERY_DATA=${BLOOMERY_DATA:-/root/bloomery-data}
OUT=${Q3K_OUT:-$BLOOMERY_DATA/bin}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT" "$BLOOMERY_DATA/ref"

for name in q4k_x4_ref q6k_x4_ref q5f0_ref q5f1_ref; do
  g++ -std=c++17 -O2 -mavx2 -mfma -mf16c -o "$OUT/$name" "$HERE/tools/ref/$name.cpp" \
    -I"$IK/ggml/include" \
    -I"$IK/ggml/src" \
    -I"$IK/ggml/src/iqk" \
    -L"$IK/build/ggml/src" -lggml \
    -Wl,-rpath,"$IK/build/ggml/src"
  echo "built $OUT/$name"
done

# The dumps are the output of THIS ik build: rerun this script when ik moves.
for name in q4k_x4_ref q6k_x4_ref q5f0_ref q5f1_ref; do
  BLOOMERY_DATA="$BLOOMERY_DATA" "$OUT/$name"
done

ls -l "$BLOOMERY_DATA"/ref/q4k-x4-ik-dot.txt "$BLOOMERY_DATA"/ref/q6k-x4-ik-dot.txt \
      "$BLOOMERY_DATA"/ref/q5f0-ik-dot.txt "$BLOOMERY_DATA"/ref/q5f1-ik-dot.txt
