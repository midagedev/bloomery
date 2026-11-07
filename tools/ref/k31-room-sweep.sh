#!/usr/bin/env bash
# Three K3.1 bp requests at host rooms 252G, 232G, 212G (docs/cards/k31-room-sweep.card), each one tools/ref/k31-nvme-probe.sh run.
set -u
for room in ${K31_ROOMS:-252G 232G 212G}; do
  d=/root/k31-room-sweep/$room; mkdir -p "$d"
  K31_PROBE_ENV="BLOOMERY_HOST_ROOM=$room" K31_PROBE_OUT="$d" bash tools/ref/k31-nvme-probe.sh > "$d.log" 2>&1; rc=$?
  echo "[room] $room rc=$rc $(grep -h '^pp_tok_s' "$d.log" | cut -c1-110) $(grep -h '^pgmajfault' "$d.log") $(grep -h '^proc_io' "$d.log" | cut -c1-40)"
  grep -h '^plan ' "$d/summary.txt" | sed -E 's/.*(host_experts=[0-9]+ \([0-9]+ B\)).*(paged=[0-9]+ arena=[0-9]+).*/[room] '"$room"' \1 \2/'
  grep -h '^decode  nvme0n1' "$d.log" | sed "s/^/[room] $room /"
done
