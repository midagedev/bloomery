#!/usr/bin/env bash
# ik's greedy next token for tools/ref/prompts.tsv, written to $BLOOMERY_DATA/argmax-ik.tsv.
#
# Same conditions as the oracle dump, for the same reason: CUDA is hidden, not merely
# unused. With the CUDA backend registered ik splits the graph even at -ngl 0 (measured
# 2026-09-19: 351 splits vs 1), and the reference would then be a GPU reduction order that
# the CPU forward cannot match.
#
# This writes ONE file and never touches $BLOOMERY_DATA/ref. The check at the end says so
# out loud: the oracle's manifest must be byte-identical afterwards.
set -euo pipefail
BLOOMERY_DATA=${BLOOMERY_DATA:-/root/bloomery-data}
MODEL=${BLOOMERY_REF_MODEL:-/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf}
BIN="$BLOOMERY_DATA/bin/argmax_ref"
HERE=$(cd "$(dirname "$0")/../.." && pwd)
PROMPTS=${BLOOMERY_PROMPTS:-$HERE/tools/ref/prompts.tsv}
OUT=${BLOOMERY_ARGMAX_OUT:-$BLOOMERY_DATA/argmax-ik.tsv}
[ -x "$BIN" ] || { echo "no argmax_ref at $BIN — run: just build-argmax" >&2; exit 2; }

before=$(md5sum "$BLOOMERY_DATA/ref/MANIFEST.tsv" | cut -d' ' -f1)
CUDA_VISIBLE_DEVICES="" "$BIN" -m "$MODEL" --prompts "$PROMPTS" -ngl 0 -c 512 -t 32 > "$OUT.partial"
mv "$OUT.partial" "$OUT"
after=$(md5sum "$BLOOMERY_DATA/ref/MANIFEST.tsv" | cut -d' ' -f1)
[ "$before" = "$after" ] || { echo "argmax.sh: the oracle manifest CHANGED — this tool must never write there" >&2; exit 1; }
echo "wrote $OUT ($(grep -vc '^#' "$OUT") rows); oracle manifest unchanged"
