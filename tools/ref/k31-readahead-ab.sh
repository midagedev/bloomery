#!/usr/bin/env bash
# read_ahead_kb A/B for K3.1 at --place bp (docs/cards/k31-readahead-ab.card): BLOOMERY_AB_ROUNDS rounds (default 2) of two arms,
# base (128) and changed (K31_RA_KB, default 2048), the order rotated every round, each arm one tools/ref/k31-nvme-probe.sh run.
# Runs on the box under tools/ref/lease-hold.sh --card. The setting is restored to its value at start on every exit.
set -u
DEV=${K31_RA_DEV:-nvme0n1}; RA=/sys/block/$DEV/queue/read_ahead_kb
CHANGED=${K31_RA_KB:-2048}; ROUNDS=${BLOOMERY_AB_ROUNDS:-2}
OUT=${K31_RA_OUT:-/root/k31-readahead-ab}; mkdir -p "$OUT"
ORIG=$(cat "$RA") || { echo "k31-readahead-ab: cannot read $RA" >&2; exit 66; }
restore() { echo "$ORIG" > "$RA"; echo "[ra] restored $DEV read_ahead_kb=$(cat "$RA")"; }
trap restore EXIT
echo "[ra] $DEV read_ahead_kb at start=$ORIG base=$ORIG changed=$CHANGED rounds=$ROUNDS"
for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) = 1 ]; then arms="base changed"; else arms="changed base"; fi
  for arm in $arms; do
    if [ "$arm" = base ]; then echo "$ORIG" > "$RA"; else echo "$CHANGED" > "$RA"; fi
    d="$OUT/r$r-$arm"; echo "[ra] round $r arm $arm read_ahead_kb=$(cat "$RA") -> $d"
    K31_PROBE_OUT="$d" bash tools/ref/k31-nvme-probe.sh > "$d.log" 2>&1; rc=$?
    echo "[ra] r$r $arm rc=$rc $(grep -h '^pp_tok_s' "$d.log" | cut -c1-120) $(grep -h '^pgmajfault' "$d.log") $(grep -h '^proc_io' "$d.log")"
    grep -h '^plan ' "$d/summary.txt" | sed -E 's/.*(host_experts=[0-9]+).*(paged=[0-9]+).*/[ra] r'"$r"' '"$arm"' \1 \2/'
  done
done
