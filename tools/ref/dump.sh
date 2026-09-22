#!/usr/bin/env bash
# Produce the stage-1 oracle reference set: ik_llama.cpp's intermediate tensors for a
# fixed token sequence, as raw f32, into $BLOOMERY_DATA/ref/.
#
# The gate for rounds 1-2 through 1-5 reads what this writes. Run it once per model;
# re-run it whenever the ik build changes, because the reference is that build's output.
#
# The token sequence is a literal here, not a prompt. dump_ref does not tokenize on
# purpose — if it did, a tokenizer difference between it and bloomery would appear as a
# numeric difference in every downstream tensor and read as a kernel bug.
# These ids are "The capital of France is" under this model's tokenizer, read once with
# llama-tokenize --ids on the box (2026-09-19). Changing them invalidates the whole set.
#
# CUDA is switched off, not merely unused: with the CUDA backend registered, ik splits
# the graph (measured 2026-09-19: 351 splits with -ngl 0, 1 split with CUDA hidden) and
# the reference would then be a GPU reduction order that our CPU rounds cannot match.
#
# BLOOMERY_REF_BACKEND=cuda writes the GPU engine's oracle instead: the same dumper, the
# same tokens, every layer offloaded (-ngl 99) on the card the box env pins, into
# $BLOOMERY_DATA/ref_cuda/. The CPU set is never touched by that run. The GPU kernels use
# q8_1 activations like ik's CUDA path, so their band is against this set, not the CPU one.
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
BLOOMERY_DATA=${BLOOMERY_DATA:-/root/bloomery-data}
BACKEND=${BLOOMERY_REF_BACKEND:-cpu}
case $BACKEND in
  cpu)  SET=ref;      NGL=0;  HIDE_CUDA=1 ;;
  cuda) SET=ref_cuda; NGL=99; HIDE_CUDA=0 ;;
  *) echo "dump.sh: BLOOMERY_REF_BACKEND must be cpu or cuda, got '$BACKEND'" >&2; exit 2 ;;
esac
# Output-set override: the backend still picks the offload depth and CUDA visibility,
# BLOOMERY_REF_SET only renames the destination (e.g. ref_cuda_v2), so an instrumented
# dumper can produce a second set beside the one the gates read without touching it.
SET=${BLOOMERY_REF_SET:-$SET}
MODEL=${BLOOMERY_REF_MODEL:-/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf}
TOKENS=${BLOOMERY_REF_TOKENS:-100000,549,6077,280,7239,317}
BIN="$BLOOMERY_DATA/bin/dump_ref"
[ -x "$BIN" ] || { echo "no dump_ref at $BIN — run: just build-ref-dump" >&2; exit 2; }

# Stage, then swap. The old set survives a failed run: `rm -f *.f32` up front used to mean
# that a dumper killed halfway left a half-set with nothing to compare it against, and on
# 2026-09-19 that is exactly what happened (a round ran the binary under gdb; every
# breakpoint killed it mid-write). A dump that does not finish must cost nothing.
REF="$BLOOMERY_DATA/$SET"
STAGE="$BLOOMERY_DATA/$SET.staging"
rm -rf "$STAGE"
mkdir -p "$STAGE"

# Which ik build this set is the output of — recorded in the manifest, because the
# reference is that build's answer and nothing else's.
# $IK comes from ref-paths.sh, the file build-dump.sh reads too — the tree this binary was
# linked against. Not $HOME/ik_llama.cpp: the dump runs as root and the tree is the serving user's.
BUILD=$(git -C "$IK" rev-parse --short HEAD 2>/dev/null || echo unknown)

if [ "$HIDE_CUDA" = 1 ]; then export CUDA_VISIBLE_DEVICES=""; fi
BLOOMERY_REF_WRITE=1 BLOOMERY_REF_DIR="$STAGE" BLOOMERY_REF_BUILD="$BUILD" \
  "$BIN" -m "$MODEL" --tokens "$TOKENS" -ngl "$NGL" -c 512 -t 32

# The trailer is the dumper's completion proof; without it the staged set is not installed.
grep -q '^# complete' "$STAGE/MANIFEST.tsv" || {
    echo "dump.sh: the staged set has no completion trailer — not installing it" >&2
    exit 1
}
rm -rf "$REF.old"
if [ -d "$REF" ]; then mv "$REF" "$REF.old"; fi
mv "$STAGE" "$REF"
rm -rf "$REF.old"
grep -c '^tensor' "$REF/MANIFEST.tsv" | xargs echo "reference tensors:"
echo "build: $BUILD  backend: $BACKEND  set: $REF"
