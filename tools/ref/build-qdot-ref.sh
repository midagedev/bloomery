#!/usr/bin/env bash
# Build the harnesses that link ik's own kernel tables, and run the thirteen
# reference ones (on the box). Runs under tools/box.sh (toolchain env already
# sourced).
#
#   bash tools/ref/build-qdot-ref.sh              all of them, and the rate harnesses
#   bash tools/ref/build-qdot-ref.sh NAME…        only those dumpers (iq4xs_ref: just build-ref-iq4xs); a NAME that is
#                                                 not one dumper, or holds a space, is exit 64 and nothing built
#   bash tools/ref/build-qdot-ref.sh --self-test  the NAME rules on a stand-in compiler, no ik, no box
#
# These link ik's own kernel tables, not just libggml: each one calls the
# entry the oracle dispatches (iqk_set_kernels_kquants /
# iqk_set_kernels_legacy_quants / iqk_set_kernels_iquants).
#
# The *_ref harnesses dump $BLOOMERY_DATA/ref/<name>-ik-dot.txt from the
# first aligned tensor of their type (q8f0_ref from a synthetic block set it
# writes into the dump, reading no model); gate-qdot's hw tests read those dumps,
# and without them the tests fail on a missing file, which is a standing red,
# not a gate. Unlike build.sh they are built AND run here: the dump is the
# product, the binary is scaffolding. Four read the reference model; q5k_x4_ref
# reads the file named by BLOOMERY_Q5K_MODEL (default: the V4.1 first shard,
# which holds blk.0.ffn_down_exps.weight — the reference model has no Q5_K
# tensor), and also dumps ggml's to_float of that tensor's first rows
# (q5k-v41-dequant.raw/.meta) for gate-qdot's dequant test. iq3xxs_ref reads the
# file named by BLOOMERY_IQ3XXS_MODEL (default: the Qwen3.8 UD-Q3_K_XL second
# shard, whose first IQ3_XXS tensor is blk.0.ffn_gate_exps.weight, k = 2560) and
# mxfp4_x4_ref the file named by BLOOMERY_MXFP4_MODEL (default: the V4.1 DSpark
# draft's tl37 file, whose first MXFP4 tensor is blk.0.ffn_gate_exps.weight,
# k = 5120); gate-qdot reads the same variables with the same defaults.
# BLOOMERY_V4_MODEL, which named one file for both, is refused when set (exit 64).
# iq4nl_ref reads the file named by
# BLOOMERY_QWEN_Q4_MODEL (default: the Qwen3.8 UD-Q4_K_XL second shard, whose
# first IQ4_NL tensor is per_layer_token_embd.weight) and iq4xs_ref the file
# named by BLOOMERY_QWEN_Q3_MODEL (default: the Qwen3.8 UD-Q3_K_XL second shard,
# which holds the one IQ4_XS layer's experts) — until that file's .done sentinel
# exists iq4xs_ref dumps synthetic blocks under the name synthetic-iq4_xs and
# says so. iq4xs_ref alone links its own ik tree, the commit tools/ref/iq4xs-ik.pin names
# (tools/ref/ik-pin-tree.sh makes it): the shared tree predates the signed AVX2 path that qdot's
# IQ4_XS kernel follows. Its dump has an iq4xs-ik-dot.meta naming that commit, published with the
# dump under iq4xs-ik-dot.lock, and gate-qdot refuses a dump whose meta is missing or names
# another. (iq4xs_rate stays on the shared tree:
# qdot-rate-floors.tsv's IQ4_XS floor was derived against it.) q8f0_ref reads no model: its
# synthetic block set is written into its own dump. iq3s_ref reads no model either: there is
# no IQ3_S file on the box, so it quantizes seeded random rows through ggml and adds
# random-code rows (every grid index and sign byte), and dumps them under the name
# synthetic-iq3_s. q2k_ref and iq2xs_ref read
# the GLM-5.3-Flash UD-Q2_K_XL file: iq2xs_ref the file named by GLM_Q2_IQ2XS_MODEL (default its
# second shard, which holds the first IQ2_XS routed stack) and q2k_ref the file named by
# GLM_Q2_Q2K_MODEL (default its fourth shard, whose first Q2_K tensor is the NextN layer's
# blk.45.ffn_gate_exps.weight). Each dumps the tensor's first 64 real rows under its real name and
# appends 32 synthetic random-code rows (every grid index and sign byte for IQ2_XS, every scale and
# min nibble for Q2_K) after them, all 96 in the dump; the dump has no per-row marker.
#
# mxfp4_ref links libggml alone and dumps ggml's to_float of the first rows of one MXFP4 expert
# tensor of the DSpark draft (BLOOMERY_DSPARK_MODEL, default the tl37 file) to
# mxfp4-dspark-dequant.raw/.meta for gate-dspark-read; the .meta names the ik tree and commit.
#
# The twelve *_rate harnesses are built but NOT run: each is a timed kernel-rate
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
# The retired name is refused by name, with any value, before anything is built.
if [ -n "${BLOOMERY_V4_MODEL+x}" ]; then
  echo "build-qdot-ref: BLOOMERY_V4_MODEL is retired: name the IQ3_XXS file with BLOOMERY_IQ3XXS_MODEL and the MXFP4 file with BLOOMERY_MXFP4_MODEL" >&2
  exit 64
fi
if [ "${1:-}" = --self-test ]; then
  set +e
  t=$(mktemp -d "${TMPDIR:-/tmp}/build-qdot-ref-test.XXXXXX") || exit 70
  trap 'rm -rf "$t"' EXIT
  bad=0
  check() { # NAME GOT WANT
    [ "$2" = "$3" ] && return 0
    echo "build-qdot-ref self-test FAIL $1: got '$2', want '$3'" >&2
    bad=$((bad + 1))
  }
  # A g++ that writes an executable at its -o path, which records its arguments beside itself: the selection
  # is what the run built and the files each dumper was handed, no ik.
  mkdir "$t/stub"
  cat > "$t/stub/g++" << 'EOF'
#!/bin/sh
while [ $# -gt 0 ]; do [ "$1" = -o ] && out=$2; shift; done
printf '#!/bin/sh\nprintf "%%s\\n" "$*" > "$0.args"\n' > "$out"
chmod +x "$out"
EOF
  chmod +x "$t/stub/g++"
  # run ARGS…: the script with those arguments in a fresh output directory; RC and BUILT (what it built there).
  run() {
    rm -rf "${t:?}/bin" "${t:?}/data"
    PATH=$t/stub:$PATH BLOOMERY_V41_MODEL=x BLOOMERY_DATA=$t/data Q3K_OUT=$t/bin bash "${BASH_SOURCE[0]}" "$@" > "$t/out" 2>&1
    RC=$?
    BUILT=$(find "$t/bin" -type f ! -name '*.args' -exec basename {} \; 2> /dev/null | sort | tr '\n' ' ')
  }
  run q4k_x4_ref
  check one-rc "$RC" 0
  check one-built "$BUILT" 'q4k_x4_ref '
  run q4k_x4_ref q6k_x4_ref
  check two-rc "$RC" 0
  check two-built "$BUILT" 'q4k_x4_ref q6k_x4_ref '
  # Not one dumper: refused by name before anything is built. Two names in one argument are not one dumper.
  for a in 'q4k_x4_ref q6k_x4_ref' 'q4k_x4_ref ' ' q4k_x4_ref' q4k bogus ''; do
    run "$a"
    check "refused-rc '$a'" "$RC" 64
    check "refused-built '$a'" "$BUILT" ''
  done
  run q4k_x4_ref bogus
  check late-bad-rc "$RC" 64
  check late-bad-built "$BUILT" ''
  # A type's file is its own variable: swapped, the IQ3_XXS harness would read the MXFP4 file.
  BLOOMERY_IQ3XXS_MODEL=/x/iq3.gguf BLOOMERY_MXFP4_MODEL=/x/mx.gguf run iq3xxs_ref mxfp4_x4_ref
  check files-rc "$RC" 0
  check files-built "$BUILT" 'iq3xxs_ref mxfp4_x4_ref '
  check iq3xxs-file "$(cat "$t/bin/iq3xxs_ref.args" 2> /dev/null)" /x/iq3.gguf
  check mxfp4-file "$(cat "$t/bin/mxfp4_x4_ref.args" 2> /dev/null)" /x/mx.gguf
  # The retired one-file name is refused by name, whatever its value, before anything is built.
  for v in x ''; do
    export BLOOMERY_V4_MODEL=$v
    run q4k_x4_ref
    unset BLOOMERY_V4_MODEL
    check "retired-rc '$v'" "$RC" 64
    check "retired-built '$v'" "$BUILT" ''
    check "retired-names '$v'" "$(grep -c 'BLOOMERY_IQ3XXS_MODEL.*BLOOMERY_MXFP4_MODEL' "$t/out")" 1
  done
  echo "build-qdot-ref self-test: $([ "$bad" = 0 ] && echo ok || echo FAIL) ($bad failures)"
  [ "$bad" = 0 ]
  exit
fi
# shellcheck source=tools/ref/ref-build-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/ref-build-common.sh"
OUT=${Q3K_OUT:-$REF_BIN}
# The V4.1 first shard (the deepseek41 profile's choice, exported by tools/box.sh) unless
# BLOOMERY_Q5K_MODEL names another.
Q5K_MODEL=${BLOOMERY_Q5K_MODEL:-${BLOOMERY_V41_MODEL:?BLOOMERY_V41_MODEL unset — run through tools/box.sh, which exports it from the deepseek41 profile}}
DSPARK_MODEL=${BLOOMERY_DSPARK_MODEL:-/models/DeepSeek-V4.1-Flash-DSpark/DeepSeek-V4.1-Flash-Fp8-128x742M-MXFP4_MOE.tl37.gguf}
IQ3XXS_MODEL=${BLOOMERY_IQ3XXS_MODEL:-/models/Qwen3.8-Flash-Next-UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00002-of-00003.gguf}
MXFP4_MODEL=${BLOOMERY_MXFP4_MODEL:-/models/DeepSeek-V4.1-Flash-DSpark/DeepSeek-V4.1-Flash-Fp8-128x742M-MXFP4_MOE.tl37.gguf}
QWEN_Q4_MODEL=${BLOOMERY_QWEN_Q4_MODEL:-/models/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf}
QWEN_Q3_MODEL=${BLOOMERY_QWEN_Q3_MODEL:-/models/Qwen3.8-Flash-Next-UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00002-of-00003.gguf}
GLM_Q2_IQ2XS_MODEL=${GLM_Q2_IQ2XS_MODEL:-/models/GLM-5.3-Flash-UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-00002-of-00004.gguf}
GLM_Q2_Q2K_MODEL=${GLM_Q2_Q2K_MODEL:-/models/GLM-5.3-Flash-UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-00004-of-00004.gguf}
mkdir -p "$OUT" "$BLOOMERY_DATA/ref"

DUMPERS="q4k_x4_ref q6k_x4_ref q5f0_ref q5f1_ref q5k_x4_ref iq3xxs_ref mxfp4_x4_ref q8f0_ref iq4nl_ref iq3s_ref q2k_ref iq2xs_ref"
RATES="q4k_x4_rate q6k_x4_rate q5f0_rate q5f1_rate q5k_x4_rate iq3xxs_rate mxfp4_x4_rate iq4nl_rate iq4xs_rate iq3s_rate q2k_rate iq2xs_rate"

# NAME… selects dumpers; iq4xs_ref is one (below). No name: every dumper, every rate harness and mxfp4_ref.
ONLY=("$@")
for name in "${ONLY[@]}"; do
  if [[ $name == *[[:space:]]* || " $DUMPERS iq4xs_ref " != *" $name "* ]]; then
    echo "build-qdot-ref: '$name' is not one dumper ($DUMPERS iq4xs_ref)" >&2
    exit 64
  fi
done
want() {
  [ ${#ONLY[@]} -eq 0 ] && return 0
  local n
  for n in "${ONLY[@]}"; do [ "$n" = "$1" ] && return 0; done
  return 1
}

# Same flags for both sets: the rate harnesses include the same ik headers under
# the same IQK_IMPLEMENT as their _ref twins.
for name in $DUMPERS $RATES; do
  want "$name" || continue
  ref_cxx -mavx2 -mfma -mf16c -o "$OUT/$name" "$HERE/tools/ref/$name.cpp" \
    "${REF_GGML_INC[@]}" -I"$IK/ggml/src" -I"$IK/ggml/src/iqk" "${REF_GGML_LINK[@]}"
  echo "built $OUT/$name"
done

# The dumps are the output of THIS ik build: rerun this script when ik moves.
for name in $DUMPERS; do
  want "$name" || continue
  case $name in
    q5k_x4_ref) BLOOMERY_DATA="$BLOOMERY_DATA" "$OUT/$name" "$Q5K_MODEL" ;;
    iq3xxs_ref) BLOOMERY_DATA="$BLOOMERY_DATA" "$OUT/$name" "$IQ3XXS_MODEL" ;;
    mxfp4_x4_ref) BLOOMERY_DATA="$BLOOMERY_DATA" "$OUT/$name" "$MXFP4_MODEL" ;;
    iq4nl_ref) BLOOMERY_DATA="$BLOOMERY_DATA" "$OUT/$name" "$QWEN_Q4_MODEL" ;;
    iq2xs_ref) BLOOMERY_DATA="$BLOOMERY_DATA" "$OUT/$name" "$GLM_Q2_IQ2XS_MODEL" ;;
    q2k_ref) BLOOMERY_DATA="$BLOOMERY_DATA" "$OUT/$name" "$GLM_Q2_Q2K_MODEL" ;;
    *) BLOOMERY_DATA="$BLOOMERY_DATA" "$OUT/$name" ;;
  esac
done

# iq4xs_ref links the tree of tools/ref/iq4xs-ik.pin, never $IK: the include roots and the
# libggml are that tree's, and the harness carries its commit into the dump's .meta.
if want iq4xs_ref; then
  IQ4XS_PIN_FILE=$HERE/tools/ref/iq4xs-ik.pin
  IQ4XS_IK=$(bash "$HERE/tools/ref/ik-pin-tree.sh" "$IQ4XS_PIN_FILE")
  read -r IQ4XS_COMMIT < "$IQ4XS_PIN_FILE"
  ref_cxx -mavx2 -mfma -mf16c -DREF_IK_COMMIT="\"$IQ4XS_COMMIT\"" -o "$OUT/iq4xs_ref" "$HERE/tools/ref/iq4xs_ref.cpp" \
    -I"$IQ4XS_IK/ggml/include" -I"$IQ4XS_IK/ggml/src" -I"$IQ4XS_IK/ggml/src/iqk" \
    -L"$IQ4XS_IK/build/ggml/src" -lggml -Wl,-rpath,"$IQ4XS_IK/build/ggml/src"
  echo "built $OUT/iq4xs_ref"
  BLOOMERY_DATA="$BLOOMERY_DATA" "$OUT/iq4xs_ref" "$QWEN_Q3_MODEL"
fi
[ ${#ONLY[@]} -eq 0 ] || exit 0

GGML_BUILD="$IK@$(git -c safe.directory='*' -C "$IK" rev-parse --short=8 HEAD 2>/dev/null || echo unknown-commit)"
ref_cxx -DREF_GGML_BUILD="\"$GGML_BUILD\"" -o "$OUT/mxfp4_ref" "$HERE/tools/ref/mxfp4_ref.cpp" \
  "${REF_GGML_INC[@]}" "${REF_GGML_LINK[@]}"
echo "built $OUT/mxfp4_ref"
BLOOMERY_DATA="$BLOOMERY_DATA" "$OUT/mxfp4_ref" "$DSPARK_MODEL"

ls -l "$BLOOMERY_DATA"/ref/q4k-x4-ik-dot.txt "$BLOOMERY_DATA"/ref/q6k-x4-ik-dot.txt \
      "$BLOOMERY_DATA"/ref/q5f0-ik-dot.txt "$BLOOMERY_DATA"/ref/q5f1-ik-dot.txt \
      "$BLOOMERY_DATA"/ref/q5k-x4-ik-dot.txt "$BLOOMERY_DATA"/ref/q5k-v41-dequant.raw \
      "$BLOOMERY_DATA"/ref/iq3xxs-ik-dot.txt "$BLOOMERY_DATA"/ref/mxfp4-x4-ik-dot.txt \
      "$BLOOMERY_DATA"/ref/q8f0-ik-dot.txt \
      "$BLOOMERY_DATA"/ref/iq4nl-ik-dot.txt "$BLOOMERY_DATA"/ref/iq4xs-ik-dot.txt \
      "$BLOOMERY_DATA"/ref/iq4xs-ik-dot.meta \
      "$BLOOMERY_DATA"/ref/iq3s-ik-dot.txt \
      "$BLOOMERY_DATA"/ref/q2k-ik-dot.txt "$BLOOMERY_DATA"/ref/iq2xs-ik-dot.txt \
      "$BLOOMERY_DATA"/ref/q5k-v41-dequant.meta \
      "$BLOOMERY_DATA"/ref/mxfp4-dspark-dequant.raw "$BLOOMERY_DATA"/ref/mxfp4-dspark-dequant.meta
