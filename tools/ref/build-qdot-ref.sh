#!/usr/bin/env bash
# Build the x4 harnesses that link ik's own kernel tables, and run the four
# reference ones (on the box). Runs under tools/box.sh (toolchain env already
# sourced).
#
# These link ik's own kernel tables, not just libggml: each one calls the
# entry the oracle dispatches (iqk_set_kernels_kquants /
# iqk_set_kernels_legacy_quants).
#
# The four *_ref harnesses dump $BLOOMERY_DATA/ref/<name>-ik-dot.txt from the
# first aligned tensor of their type; gate-qdot's four hw tests read those
# dumps, and without them the tests fail on a missing file, which is a standing
# red, not a gate. Unlike build.sh they are built AND run here: the dump is the
# product, the binary is scaffolding.
#
# The four *_rate harnesses are built but NOT run: each is a timed kernel-rate
# bench over a synthetic shape, and a measurement belongs to a quiet machine and
# a lease, never to a build recipe. Building them here is what keeps them
# compiling with the rest.
#
# Everything goes outside the tree, which tools/box.sh rsyncs with --delete
# before every command. IK, the data root and the ggml flags come from
# ref-build-common.sh.
#
# IQK_IMPLEMENT plus the three ik include roots is what makes the kernel table
# visible; -mavx2 -mfma -mf16c is the ISA those kernels are written for (a
# build without them does not compile).
set -euo pipefail
# shellcheck source=tools/ref/ref-build-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-build-common.sh"
OUT=${Q3K_OUT:-$REF_BIN}
mkdir -p "$OUT" "$BLOOMERY_DATA/ref"

DUMPERS="q4k_x4_ref q6k_x4_ref q5f0_ref q5f1_ref"
RATES="q4k_x4_rate q6k_x4_rate q5f0_rate q5f1_rate"

# Same flags for both sets: the rate harnesses include the same ik headers under
# the same IQK_IMPLEMENT as their _ref twins.
for name in $DUMPERS $RATES; do
  ref_cxx -mavx2 -mfma -mf16c -o "$OUT/$name" "$HERE/tools/ref/$name.cpp" \
    "${REF_GGML_INC[@]}" -I"$IK/ggml/src" -I"$IK/ggml/src/iqk" "${REF_GGML_LINK[@]}"
  echo "built $OUT/$name"
done

# The dumps are the output of THIS ik build: rerun this script when ik moves.
for name in $DUMPERS; do
  BLOOMERY_DATA="$BLOOMERY_DATA" "$OUT/$name"
done

ls -l "$BLOOMERY_DATA"/ref/q4k-x4-ik-dot.txt "$BLOOMERY_DATA"/ref/q6k-x4-ik-dot.txt \
      "$BLOOMERY_DATA"/ref/q5f0-ik-dot.txt "$BLOOMERY_DATA"/ref/q5f1-ik-dot.txt
