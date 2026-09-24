#!/usr/bin/env bash
# Build the harnesses that link ik's own kernel tables, and run the seven
# reference ones (on the box). Runs under tools/box.sh (toolchain env already
# sourced).
#
# These link ik's own kernel tables, not just libggml: each one calls the
# entry the oracle dispatches (iqk_set_kernels_kquants /
# iqk_set_kernels_legacy_quants / iqk_set_kernels_iquants).
#
# The seven *_ref harnesses dump $BLOOMERY_DATA/ref/<name>-ik-dot.txt from the
# first aligned tensor of their type; gate-qdot's hw tests read those dumps,
# and without them the tests fail on a missing file, which is a standing red,
# not a gate. Unlike build.sh they are built AND run here: the dump is the
# product, the binary is scaffolding. Four read the reference model; q5k_x4_ref
# reads the file named by BLOOMERY_Q5K_MODEL (default: the V4.1 first shard,
# which holds blk.0.ffn_down_exps.weight — the reference model has no Q5_K
# tensor), and also dumps ggml's to_float of that tensor's first rows
# (q5k-v41-dequant.raw/.meta) for gate-qdot's dequant test. iq3xxs_ref and
# mxfp4_x4_ref read the file named by BLOOMERY_V4_MODEL (default: the V4-Flash
# first data shard, whose first IQ3_XXS and MXFP4 tensors are
# blk.0.ffn_gate_exps.weight and blk.0.ffn_down_exps.weight; gate-qdot reads the
# same variable with the same default).
#
# mxfp4_ref links libggml alone and dumps ggml's to_float of the first rows of one MXFP4 expert
# tensor of the DSpark draft (BLOOMERY_DSPARK_MODEL, default the tl37 file) to
# mxfp4-dspark-dequant.raw/.meta for gate-dspark-read; the .meta names the ik tree and commit.
#
# The seven *_rate harnesses are built but NOT run: each is a timed kernel-rate
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
# The V4.1 first shard (the deepseek41 profile's choice, exported by tools/box.sh) unless
# BLOOMERY_Q5K_MODEL names another.
Q5K_MODEL=${BLOOMERY_Q5K_MODEL:-${BLOOMERY_V41_MODEL:?BLOOMERY_V41_MODEL unset — run through tools/box.sh, which exports it from the deepseek41 profile}}
DSPARK_MODEL=${BLOOMERY_DSPARK_MODEL:-/models/DeepSeek-V4.1-Flash-DSpark/DeepSeek-V4.1-Flash-Fp8-128x742M-MXFP4_MOE.tl37.gguf}
V4_MODEL=${BLOOMERY_V4_MODEL:-/models/DeepSeek-V4-Flash-0731-UD-Q3_K_M/DeepSeek-V4-Flash-0731-UD-Q3_K_M-00002-of-00004.gguf}
mkdir -p "$OUT" "$BLOOMERY_DATA/ref"

DUMPERS="q4k_x4_ref q6k_x4_ref q5f0_ref q5f1_ref q5k_x4_ref iq3xxs_ref mxfp4_x4_ref"
RATES="q4k_x4_rate q6k_x4_rate q5f0_rate q5f1_rate q5k_x4_rate iq3xxs_rate mxfp4_x4_rate"

# Same flags for both sets: the rate harnesses include the same ik headers under
# the same IQK_IMPLEMENT as their _ref twins.
for name in $DUMPERS $RATES; do
  ref_cxx -mavx2 -mfma -mf16c -o "$OUT/$name" "$HERE/tools/ref/$name.cpp" \
    "${REF_GGML_INC[@]}" -I"$IK/ggml/src" -I"$IK/ggml/src/iqk" "${REF_GGML_LINK[@]}"
  echo "built $OUT/$name"
done

# The dumps are the output of THIS ik build: rerun this script when ik moves.
for name in $DUMPERS; do
  case $name in
    q5k_x4_ref) BLOOMERY_DATA="$BLOOMERY_DATA" "$OUT/$name" "$Q5K_MODEL" ;;
    iq3xxs_ref|mxfp4_x4_ref) BLOOMERY_DATA="$BLOOMERY_DATA" "$OUT/$name" "$V4_MODEL" ;;
    *) BLOOMERY_DATA="$BLOOMERY_DATA" "$OUT/$name" ;;
  esac
done

GGML_BUILD="$IK@$(git -c safe.directory='*' -C "$IK" rev-parse --short=8 HEAD 2>/dev/null || echo unknown-commit)"
ref_cxx -DREF_GGML_BUILD="\"$GGML_BUILD\"" -o "$OUT/mxfp4_ref" "$HERE/tools/ref/mxfp4_ref.cpp" \
  "${REF_GGML_INC[@]}" "${REF_GGML_LINK[@]}"
echo "built $OUT/mxfp4_ref"
BLOOMERY_DATA="$BLOOMERY_DATA" "$OUT/mxfp4_ref" "$DSPARK_MODEL"

ls -l "$BLOOMERY_DATA"/ref/q4k-x4-ik-dot.txt "$BLOOMERY_DATA"/ref/q6k-x4-ik-dot.txt \
      "$BLOOMERY_DATA"/ref/q5f0-ik-dot.txt "$BLOOMERY_DATA"/ref/q5f1-ik-dot.txt \
      "$BLOOMERY_DATA"/ref/q5k-x4-ik-dot.txt "$BLOOMERY_DATA"/ref/q5k-v41-dequant.raw \
      "$BLOOMERY_DATA"/ref/iq3xxs-ik-dot.txt "$BLOOMERY_DATA"/ref/mxfp4-x4-ik-dot.txt \
      "$BLOOMERY_DATA"/ref/q5k-v41-dequant.meta \
      "$BLOOMERY_DATA"/ref/mxfp4-dspark-dequant.raw "$BLOOMERY_DATA"/ref/mxfp4-dspark-dequant.meta
