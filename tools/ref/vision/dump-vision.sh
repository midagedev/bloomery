#!/usr/bin/env bash
# The V4.1 vision oracle set: the official checkpoint's preprocessing, ViT and aligner over the fixed
# images in tools/ref/vision/images/, into $BLOOMERY_DATA/ref-vision/<set>/ (dump_vision.py's header
# lists the files). The gates of crates/vision read it.
#
# Where the set came from is the checkpoint, not an ik build: dump_vision.py refuses a tree whose
# shards or reference files differ from revision dba1be0a…, and MANIFEST.tsv names that revision, the
# shard sha256s, torch, Pillow (the resampler the preprocessing gate is a port of), the card and the
# mmproj file whose header the name-table gate reads.
#
# The dump runs twice into two staging directories and installs only when every file of the two is
# byte-identical: a set that a rerun would not reproduce cannot back a bit-exact gate. The comparison is
# printed (the md5 list of each run and `cmp` over every file).
#
# It runs on the card the box env pins (the 3090) under the gate lock the 3090 gates serialize on
# (tools/gpu-gate.sh): the ViT and aligner weights are about 1 GB of card memory and a gate that
# fills the card must not land beside it. Each run is bounded by BLOOMERY_VISION_BOUND seconds
# (default 600; a run takes well under a minute).
#
# Environment: VISION_PYTHON (default /home/user/ft/bin/python3, the torch environment), VISION_CKPT
# (default /models/DeepSeek-V4.1-Flash-fp8), VISION_MMPROJ (default: smalinin's BF16 mmproj under the
# checkpoint's mmproj/), BLOOMERY_VISION_SET (default deepseek41v).
set -euo pipefail
HERE=${BASH_SOURCE[0]%/*}
# shellcheck source=tools/ref/ref-paths.sh
source "$HERE/../ref-paths.sh"
PY=${VISION_PYTHON:-/home/user/ft/bin/python3}
CKPT=${VISION_CKPT:-/models/DeepSeek-V4.1-Flash-fp8}
MMPROJ=${VISION_MMPROJ:-$CKPT/mmproj/mmproj-DeepSeek-V4.1-Flash-BF16.gguf}
SET=${BLOOMERY_VISION_SET:-deepseek41v}
BOUND=${BLOOMERY_VISION_BOUND:-600}
case $SET in
  */*|.*|'') echo "dump-vision.sh: '$SET' cannot name a set" >&2; exit 2 ;;
esac
case $BOUND in
  '' | *[!0-9]* | 0) echo "dump-vision.sh: BLOOMERY_VISION_BOUND must be a positive integer, got '$BOUND'" >&2; exit 64 ;;
esac
[ -x "$PY" ] || { echo "dump-vision.sh: no python at $PY" >&2; exit 2; }
[ -f "$MMPROJ" ] || { echo "dump-vision.sh: no mmproj at $MMPROJ" >&2; exit 2; }
IMAGES=$(cd "$HERE/images" && pwd)

ROOT="$BLOOMERY_DATA/ref-vision"
REF="$ROOT/$SET"
STAGE="$ROOT/$SET.staging"
mkdir -p "$ROOT"
rm -rf "$STAGE"
mkdir -p "$STAGE"

exec 8>/root/bloomery-gate.lock
echo "[gate-lock] waiting for /root/bloomery-gate.lock ..."
flock -w 1800 8 || { echo "[gate-lock] timed out after 30 min" >&2; exit 75; }
echo "[gate-lock] held"

for run in a b; do
  echo "dump-vision.sh: run $run"
  timeout --kill-after=10 "$BOUND" "$PY" "$HERE/dump_vision.py" \
    --ckpt "$CKPT" --images "$IMAGES" --mmproj "$MMPROJ" --out "$STAGE/$run"
done
exec 8>&-

for run in a b; do
  grep -q '^# complete' "$STAGE/$run/MANIFEST.tsv" || {
    echo "dump-vision.sh: run $run has no completion trailer — not installing" >&2
    exit 1
  }
done
echo "dump-vision.sh: md5 of every file, run a then run b"
(cd "$STAGE/a" && md5sum -- * | sort -k2) > "$STAGE/a.md5"
(cd "$STAGE/b" && md5sum -- * | sort -k2) > "$STAGE/b.md5"
paste -d' ' "$STAGE/a.md5" "$STAGE/b.md5" | awk '{ print ($1 == $3 ? "same" : "DIFF"), $1, $2 }'
differ=0
for f in "$STAGE"/a/*; do
  n=${f##*/}
  cmp -s "$f" "$STAGE/b/$n" || { echo "dump-vision.sh: $n differs between the two runs" >&2; differ=1; }
done
[ "$(ls "$STAGE/a" | wc -l)" = "$(ls "$STAGE/b" | wc -l)" ] || { echo "dump-vision.sh: the runs wrote different file lists" >&2; differ=1; }
if [ "$differ" = 1 ]; then
  echo "dump-vision.sh: the two runs are not byte-identical — not installing (staging kept in $STAGE)" >&2
  exit 1
fi
echo "dump-vision.sh: run a and run b are byte-identical ($(wc -l < "$STAGE/a.md5") files)"

rm -rf "$REF.old"
if [ -d "$REF" ]; then mv "$REF" "$REF.old"; fi
mv "$STAGE/a" "$REF"
rm -rf "$REF.old" "$STAGE"
echo "set: $REF"
head -n 16 "$REF/MANIFEST.tsv"
grep '^# complete' "$REF/MANIFEST.tsv"
