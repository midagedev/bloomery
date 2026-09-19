#!/usr/bin/env bash
# Produce the stage-1 oracle reference set: ik_llama.cpp's intermediate tensors for a
# fixed token sequence, as raw f32, into $MULLE_DATA/ref/.
#
# The gate for rounds 1-2 through 1-5 reads what this writes. Run it once per model;
# re-run it whenever the ik build changes, because the reference is that build's output.
#
# The token sequence is a literal here, not a prompt. dump_ref does not tokenize on
# purpose — if it did, a tokenizer difference between it and mulle would appear as a
# numeric difference in every downstream tensor and read as a kernel bug.
# These ids are "The capital of France is" under this model's tokenizer, read once with
# llama-tokenize --ids on the box (2026-09-19). Changing them invalidates the whole set.
#
# CUDA is switched off, not merely unused: with the CUDA backend registered, ik splits
# the graph (measured 2026-09-19: 351 splits with -ngl 0, 1 split with CUDA hidden) and
# the reference would then be a GPU reduction order that our CPU rounds cannot match.
set -euo pipefail
MULLE_DATA=${MULLE_DATA:-/root/mulle-data}
MODEL=${MULLE_REF_MODEL:-/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf}
TOKENS=${MULLE_REF_TOKENS:-100000,549,6077,280,7239,317}
BIN="$MULLE_DATA/bin/dump_ref"
[ -x "$BIN" ] || { echo "no dump_ref at $BIN — run: just build-ref-dump" >&2; exit 2; }
rm -f "$MULLE_DATA/ref"/*.f32 "$MULLE_DATA/ref"/MANIFEST.tsv
CUDA_VISIBLE_DEVICES="" "$BIN" -m "$MODEL" --tokens "$TOKENS" -ngl 0 -c 512 -t 32
grep -c '^tensor' "$MULLE_DATA/ref/MANIFEST.tsv" | xargs echo "reference tensors:"
