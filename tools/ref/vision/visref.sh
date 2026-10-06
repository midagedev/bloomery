#!/usr/bin/env bash
# The V4.1 vision fork comparison's set (`just dump-ref-visref`): the reference engine's side of "does the engine see
# an image as the reference does". Both engines get the same ids and the same image rows — the official encoder's
# (dump_vision.py's set deepseek41v-scenes) — so the encoder is out of the comparison and the text model's image
# rules are in it: the rows at the span's positions, exp_probs_b_vl in the expert pick there, no engram row there
# and a blocked lookback after it.
#
#   BLOOMERY_REMOTE='~/repo/bloomery-visref' tools/box.sh 'bash tools/ref/vision/visref.sh'
#
# Steps: visref_cases.py renders and tokenizes each case of visref-cases.tsv with the checkpoint's own template
# and tokenizer and splices the image's rows in, with two text controls per image (the span removed; the span's
# positions holding prose ids); visref_fork (build-fork.sh) decodes each case greedily on the V4.1 file the tree
# runs, keeping the logits of the answer's first KEEP tokens; visref_cases.py writes MANIFEST.tsv. The set goes to
# $BLOOMERY_DATA/ref-visref/deepseek41-scenes through a staging directory, `# complete` last; crates/refset reads it
# (`refset::visref`, family visref-deepseek41).
#
# The fork runs through tools/gpu-gate.sh from its build root (target/release/visref_fork): the card's gate lock
# and the V4.1 load lock, under VISREF_BOUND seconds (default 1500). Not timed, no lease.
# Environment: VISION_PYTHON (default /home/user/ft/bin/python3), VISION_CKPT (default
# /models/DeepSeek-V4.1-Flash-fp8), VISREF_FORK_ROOT (build-fork.sh's), VISREF_GEN (greedy tokens per
# case, default 256), VISREF_KEEP (logits rows kept, default 64).
set -euo pipefail
HERE=$(cd "${BASH_SOURCE[0]%/*}" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
# shellcheck source=tools/ref/ref-paths.sh
source "$HERE/../ref-paths.sh"
PY=${VISION_PYTHON:-/home/user/ft/bin/python3}
CKPT=${VISION_CKPT:-/models/DeepSeek-V4.1-Flash-fp8}
ROOT=${VISREF_FORK_ROOT:-$HOME/repo/bloomery-visref-fork}
GEN=${VISREF_GEN:-256}
KEEP=${VISREF_KEEP:-64}
BOUND=${VISREF_BOUND:-1500}
for v in GEN KEEP BOUND; do
  case ${!v} in '' | *[!0-9]* | 0) echo "visref.sh: $v must be a positive integer, got '${!v}'" >&2; exit 64 ;; esac
done
V41=${BLOOMERY_V41_MODEL:?visref.sh runs under tools/box.sh, which exports BLOOMERY_V41_MODEL}
FORK_SHA=$(sed -n 's/^FORK_SHA=//p' "$HERE/build-fork.sh")
[ "$(cat "$ROOT/target/release/visref_fork.build" 2> /dev/null)" = "fork $FORK_SHA" ] || {
  echo "visref.sh: no visref_fork of fork $FORK_SHA under $ROOT — run: bash tools/ref/vision/build-fork.sh" >&2
  exit 2
}
VSET=$BLOOMERY_DATA/ref-vision/deepseek41v-scenes
grep -q '^# complete' "$VSET/MANIFEST.tsv" 2> /dev/null || {
  echo "visref.sh: no complete $VSET — run: BLOOMERY_VISION_SET=deepseek41v-scenes VISION_IMAGES=tools/ref/vision/scenes bash tools/ref/vision/dump-vision.sh" >&2
  exit 2
}
SET=$BLOOMERY_DATA/ref-visref/deepseek41-scenes
mkdir -p "${SET%/*}"
# A staging directory of this run's own: a fork left running by a killed run writes into its own.
STAGE=$(mktemp -d "$SET.staging.XXXXXX")
"$PY" "$HERE/visref_cases.py" cases --ckpt "$CKPT" --vision-set "$VSET" --cases "$HERE/visref-cases.tsv" \
  --prose "$BLOOMERY_DATA/engram/corpus-prose.ids" --gen "$GEN" --out "$STAGE/cases"
mv "$STAGE/cases"/* "$STAGE"/
rmdir "$STAGE/cases"
rc=0
(cd "$ROOT" && BLOOMERY_GATE_V41_LOAD=1 BLOOMERY_GATE_BOUND=$BOUND bash "$REPO/tools/gpu-gate.sh" visref_fork \
  "$V41" "$STAGE" "$STAGE" "$KEEP") > "$STAGE/fork.log" 2> "$STAGE/fork-stderr.log" || rc=$?
cat "$STAGE/fork.log"
if [ "$rc" != 0 ]; then
  tail -n 30 "$STAGE/fork-stderr.log" >&2
  echo "visref.sh: visref_fork rc $rc — not installing (staging kept in $STAGE)" >&2
  exit "$rc"
fi
device=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" | head -n 1)
"$PY" "$HERE/visref_cases.py" manifest --set "$STAGE" --fork-log "$STAGE/fork.log" --vision-set "$VSET" \
  --model "$V41" --build "$FORK_SHA" --keep "$KEEP" --device "$device" \
  --placement "every layer on the card, the routed experts in host memory, no op offload, flash attention, 32 threads, n_batch 512"
chmod 755 "$STAGE"
rm -rf "$SET.old"
if [ -d "$SET" ]; then mv "$SET" "$SET.old"; fi
mv "$STAGE" "$SET"
rm -rf "$SET.old"
echo "set: $SET"
grep -E '^(# (model|build|rows|n_vocab|n_embd|keep|device|complete)|case)' "$SET/MANIFEST.tsv"
for f in "$SET"/*.answer.txt; do
  echo "--- ${f##*/}"
  cat "$f"
  echo
done
