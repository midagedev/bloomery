#!/usr/bin/env bash
# One K3.1 bp decode run of generate_kimi_k3 under BLOOMERY_STEP_STATS=1 with iostat beside it, no lease (a counter read).
# Usage (on the box, from the tree): tools/ref/k31-step-stats.sh [OUT] ; env K31_SS_D (prompt ids, 64), K31_SS_N (ids
# generated, 48), K31_SS_ROOM (232G), K31_SS_MODEL (the K3.1 first shard), K31_SS_ENV (extra NAME=value words).
set -u
OUT=${1:-/root/k31-step-stats}
D=${K31_SS_D:-64}; N=${K31_SS_N:-48}; ROOM=${K31_SS_ROOM:-232G}
MODEL=${K31_SS_MODEL:-/models/Kimi-K3.1-Pro-IQ1_S/Kimi-K3.1-Pro-IQ1_S-00001-of-00031.gguf}
IDS=${BLOOMERY_DATA:-/root/bloomery-data}/kimi-k3/corpus-prose.ids
BIN=target/release/generate_kimi_k3
die() { echo "k31-step-stats: $1" >&2; exit "${2:-1}"; }
[ -x "$BIN" ] || die "no $BIN (build it first)" 66
[ -f "$MODEL" ] || die "no model at $MODEL" 66
[ -f "$IDS" ] || die "no ids at $IDS" 66
mkdir -p "$OUT"
# one K3.1 process on the box at a time (each loads ~250 GB of host RAM and both cards): the probe lock, waited for
exec 9>/root/k31-probe.lock; flock -w 7200 9 || die "k31-probe.lock held for 2 h" 75
TOKENS=$(head -n "$D" "$IDS" | paste -sd, -)
[ "$(echo "$TOKENS" | tr ',' '\n' | wc -l)" = "$D" ] || die "ids file shorter than $D"
sha256sum "$BIN" | tee "$OUT/summary.txt"
echo "MemAvailable_kB=$(awk '/MemAvailable/{print $2}' /proc/meminfo) Cached_kB=$(awk '/^Cached/{print $2}' /proc/meminfo)" | tee -a "$OUT/summary.txt"
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tee "$OUT/gpu-before.txt"
iostat -x -t 1 nvme0n1 nvme1n1 > "$OUT/iostat.log" 2>&1 &
IPID=$!
t0=$(date +%s)
echo "start=$(date -Iseconds) D=$D N=$N room=$ROOM" | tee -a "$OUT/summary.txt"
env BLOOMERY_REF_MODEL="$MODEL" BLOOMERY_HOST_ROOM="$ROOM" BLOOMERY_STEP_STATS=1 ${K31_SS_ENV:-} \
  timeout --kill-after=10 3600 "$BIN" --place bp --ctx 2048 --mode graph --tokens "$TOKENS" -n "$N" \
  > "$OUT/out.txt" 2> "$OUT/err.txt"
rc=$?
echo "rc=$rc wall_s=$(( $(date +%s) - t0 ))" | tee -a "$OUT/summary.txt"
kill "$IPID" 2>/dev/null
grep -m1 '^plan ' "$OUT/err.txt" | tee -a "$OUT/summary.txt"
grep '^load \|^stat summary\|^nvtier ' "$OUT/out.txt" "$OUT/err.txt" | tee -a "$OUT/summary.txt"
exit $rc
