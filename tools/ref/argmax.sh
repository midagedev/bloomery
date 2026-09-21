#!/usr/bin/env bash
# ik's greedy next token for tools/ref/prompts.tsv, written under $BLOOMERY_DATA.
#
# Same conditions as the oracle dump, for the same reason: CUDA is hidden, not merely
# unused. With the CUDA backend registered ik splits the graph even at -ngl 0 (measured
# 2026-09-19: 351 splits vs 1), and the reference would then be a GPU reduction order that
# the CPU forward cannot match.
#
# BLOOMERY_REF_BACKEND=cuda runs the GPU engine's token reference instead: CUDA left
# visible on the card the box env pins, every layer offloaded (-ngl 99) — dump.sh's
# card selection and offload depth. Two deviations from dump.sh's invocation, both
# forced by the same measured fact: ik's CUDA backend writes prompt-independent logits
# for batch prefill of nine tokens and up in this build (llama-cli at -ngl 99 answers
# "emanoicisananan..." to "The first president of the United States was"; one token at
# a time, or CUDA hidden, answers " George Washington"). So the cuda path feeds the
# prompt through argmax_ref --step-prefill (M=1 kernels, same tokens/positions/KV), and
# it needs no -mla/-fa flags: they do not touch the defect, and fused MoE is this
# build's default (there is no -fmoe; only -no-fmoe). Set BLOOMERY_REF_BATCH_PREFILL=1
# to re-test batch prefill after an ik upgrade; a fixed build should flip this default
# back and regenerate the files.
#
# BLOOMERY_REF_GEN=N appends N greedy steps (argmax_ref --gen): the file becomes
# greedy-ik-<backend>-N.tsv. Without it the plain argmax set is written:
# argmax-ik.tsv (cpu) / argmax-ik-cuda.tsv (cuda). cpu with N=0 is the original file,
# byte for byte.
#
# This writes ONE file and never touches $BLOOMERY_DATA/ref. The check at the end says so
# out loud: the oracle's manifest must be byte-identical afterwards.
set -euo pipefail
BLOOMERY_DATA=${BLOOMERY_DATA:-/root/bloomery-data}
BACKEND=${BLOOMERY_REF_BACKEND:-cpu}
GEN=${BLOOMERY_REF_GEN:-0}
MODEL=${BLOOMERY_REF_MODEL:-/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf}
BIN="$BLOOMERY_DATA/bin/argmax_ref"
HERE=$(cd "$(dirname "$0")/../.." && pwd)
PROMPTS=${BLOOMERY_PROMPTS:-$HERE/tools/ref/prompts.tsv}
STEP=()
case $BACKEND in
  cpu)  NGL=0;  HIDE_CUDA=1; NAME=argmax-ik.tsv ;;
  cuda) NGL=99; HIDE_CUDA=0; NAME=argmax-ik-cuda.tsv
        [ "${BLOOMERY_REF_BATCH_PREFILL:-0}" = 1 ] || STEP=(--step-prefill) ;;
  *) echo "argmax.sh: BLOOMERY_REF_BACKEND must be cpu or cuda, got '$BACKEND'" >&2; exit 2 ;;
esac
if [ "$GEN" -gt 0 ]; then NAME="greedy-ik-$BACKEND-$GEN.tsv"; fi
OUT=${BLOOMERY_ARGMAX_OUT:-$BLOOMERY_DATA/$NAME}
# The context must hold the longest prompt (56 today) plus every generated step.
CTX=$((512 + GEN))
[ -x "$BIN" ] || { echo "no argmax_ref at $BIN — run: just build-argmax" >&2; exit 2; }

before=$(md5sum "$BLOOMERY_DATA/ref/MANIFEST.tsv" | cut -d' ' -f1)
if [ "$HIDE_CUDA" = 1 ]; then export CUDA_VISIBLE_DEVICES=""; fi
if [ "$GEN" -gt 0 ]; then
  "$BIN" -m "$MODEL" --prompts "$PROMPTS" --gen "$GEN" "${STEP[@]}" -ngl "$NGL" -c "$CTX" -t 32 > "$OUT.partial"
else
  "$BIN" -m "$MODEL" --prompts "$PROMPTS" "${STEP[@]}" -ngl "$NGL" -c "$CTX" -t 32 > "$OUT.partial"
fi
mv "$OUT.partial" "$OUT"
after=$(md5sum "$BLOOMERY_DATA/ref/MANIFEST.tsv" | cut -d' ' -f1)
[ "$before" = "$after" ] || { echo "argmax.sh: the oracle manifest CHANGED — this tool must never write there" >&2; exit 1; }
echo "wrote $OUT ($(grep -vc '^#' "$OUT") rows, backend=$BACKEND gen=$GEN); oracle manifest unchanged"
