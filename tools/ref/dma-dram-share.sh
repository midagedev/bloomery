#!/usr/bin/env bash
# What a host-to-card copy costs the host tier's DRAM reads, under the machine-wide lease (lead-only:
# `just time-dma-dram`). The host tier's bench (bench_v41_host --time) runs in the conditions --conds
# names (default alone,pageable,staged), interleaved, the order rotated each round:
#
#   alone        the bench by itself
#   pageable     beside h2d_probe's pageable-loop: copies from a warm map, which the driver stages
#   staged       beside h2d_probe's staged-loop: the streaming design, fill threads copying the map into
#                a ring of pinned chunks that the card reads while the next fills
#   staging:<n>  beside the residency machine's copy path (crates/gpu/src/host/swap.rs): one thread
#                memcpys --block bytes (one V4.1 expert) from an anonymous pageable source into a ring
#                of 4 pinned chunks and the card DMAs each, n copies a token at --token-ms a token
#                (n x 1000 / token-ms copies a second); `staging:max` unpaced
#   direct:<n>   the same source registered (cuMemHostRegister), each block one DMA read with no host
#                copy, 4 in flight; `direct:max` unpaced
#   odirect:<n>  the O_DIRECT candidate: each block pread from the model file (h2d_probe's --od-file
#                default) into a pinned ring slot and one DMA to the card — the NVMe controller's DMA
#                writes host DRAM, the card's DMA reads it, no CPU copy — blocks at a fixed stride
#                over the file, n copies a token paced as the others; `odirect:max` unpaced
#
# The staging and direct conditions read an anonymous window of --anon-bytes (60 blocks), first-touched
# by the probe before its loop starts: they touch no model file. The odirect condition reads the model
# file itself with O_DIRECT — no page cache, the only condition that touches a model file. Against a
# DRAM-bound leg (V4.1's
# engine:5, ~200 host slots a token like the chat decode) their k is the design's c: host-slot
# equivalents one copy costs, since a slot and a copy are both one expert's bytes and the leg's time is
# its bytes over its rate, c = Δt × H / (copies a token × t_off) / B = (U_alone - U_with) / C_with.
#
# This box refuses to register the file mapping (rig-log 09-25#m2-h2d-a6000), so every copy crosses
# DRAM more than once: the source read, the staging write and the card's read of it, and a store that
# is not non-temporal reads the destination first. The run measures the union's GB/s beside each copy,
# the copy's own GB/s over the same span, and their ratio k = (U_alone - U_with) / C_with, the union's
# read bytes a copied byte displaces — the DRAM passes a copy costs when the union holds DRAM at its
# ceiling (bench_v41_host's engine arms read 132-137 GB/s alone; STREAM is 147.7).
#
# Before the rounds each copy runs alone for --copy-alone seconds (its rate without the bench). The bench
# runs at --threads (default: the physical cores less one per L3 group) and the copy is pinned with
# taskset to the cores the pool leaves free (--copy-cpus, default the last physical core of each L3
# group, which the pool's CCD-spread order leaves free at that count); a round whose bench prints a
# pinned thread on one of those cpus is marked [overlap] and left out of the summary. The copy prints
# its GB/s every second with the epoch time; the runner stamps every bench line with the epoch time,
# and the copy's rate for an arm-round is the bytes of the intervals inside that arm-round's span (from
# the bench line before it to its `time` line) over their time; a span the copy did not cover from end
# to end is marked [copy-gap].
#
# Rows:   P2 round=<r> cond=<c> arm=<a> union_gbps=<U> copy_gbps=<C|-> admissible=<yes|no>[ tags]
# Close:  P2 summary arm=<a> U_alone=<median> | <cond> U=<> C=<> k=<> n=<> | … | copy alone <cond>=<> …
#         P2 c arm=<a> cond=<staging:n|direct:n|odirect:n> C=<> copies_per_s=<> c=<mean> lo=<> hi=<> n=<>
# Medians over the rounds without tags and with admissible=yes; k per round against that round's alone;
# c the mean of those per-round k with its 95 % t interval (tools/ref/tdist.py), for the staging,
# direct and odirect conditions.
#
# Arguments (all optional): --rounds N (4), --seconds S (24, the bench's timed seconds per arm), --warmup W
# (3), --arms A,B (engine:6; bench_v41_host's arm grammar, passed through), --threads T, --copy-cpus LIST,
# --chunk BYTES (67108864), --ring R (4), --fill-threads T (4) (pageable and staged), --conds LIST
# (alone,pageable,staged), --token-ms MS (37.5, the chat decode token), --block BYTES (16773120) and
# --anon-bytes BYTES (1006387200) (staging and direct), --copy-alone S (12), --parse-only (print the plan
# and exit 0 before any check or lease). The rows, the close and the core layout are
# tools/ref/dma-dram-share.py's (its self-test runs in check-recipes). Each process under the lease runs under timeout: the bench under
# BLOOMERY_HOST_BOUND (600 s), the copy under that plus 60 s, and it stops earlier at its stop file.
# Every pid the runner starts is written at spawn to <out>/pids (the `timeout` wrapper's pid: a TERM to
# it reaches the probe); the runner waits on those pids and signals no other. The logs go to
# target/dma-dram-share/<utc>/. The run's card: BLOOMERY_LEASE_CARD through BLOOMERY_BOX_ENV
# (docs/cards/dma-dram-share.card).
set -euo pipefail
HERE=${BASH_SOURCE[0]%/*}

# p2 <mode> <args…>: tools/ref/dma-dram-share.py, the rows, the close and the core layout.
p2() { python3 "$HERE/dma-dram-share.py" "$@"; }

ROUNDS=4 SECS=24 WARMUP=3 ARMS=engine:6 THREADS='' COPY_CPUS='' CHUNK=67108864 RING=4 FILL=4 ALONE_S=12
CONDS=alone,pageable,staged TOKEN_MS=37.5 BLOCK=16773120 ANON=1006387200
PARSE_ONLY=0
while [ $# -gt 0 ]; do
  case $1 in
    --parse-only) PARSE_ONLY=1; shift; continue ;;
    --rounds | --seconds | --warmup | --arms | --threads | --copy-cpus | --chunk | --ring | --fill-threads | --copy-alone | \
      --conds | --token-ms | --block | --anon-bytes)
      [ $# -ge 2 ] || { echo "dma-dram-share.sh: $1 takes a value" >&2; exit 64; } ;;
    *) echo "dma-dram-share.sh: unknown argument '$1'" >&2; exit 64 ;;
  esac
  case $1 in
    --rounds) ROUNDS=$2 ;;
    --seconds) SECS=$2 ;;
    --warmup) WARMUP=$2 ;;
    --arms) ARMS=$2 ;;
    --threads) THREADS=$2 ;;
    --copy-cpus) COPY_CPUS=$2 ;;
    --chunk) CHUNK=$2 ;;
    --ring) RING=$2 ;;
    --fill-threads) FILL=$2 ;;
    --copy-alone) ALONE_S=$2 ;;
    --conds) CONDS=$2 ;;
    --token-ms) TOKEN_MS=$2 ;;
    --block) BLOCK=$2 ;;
    --anon-bytes) ANON=$2 ;;
  esac
  shift 2
done
BOUND=${BLOOMERY_HOST_BOUND:-600}
for v in "$ROUNDS" "$SECS" "$WARMUP" "$CHUNK" "$RING" "$FILL" "$ALONE_S" "$BLOCK" "$ANON" "$BOUND" ${THREADS:+"$THREADS"}; do
  case $v in '' | *[!0-9]* | 0) echo "dma-dram-share.sh: counts, seconds and bytes are positive integers, got '$v'" >&2; exit 64 ;; esac
done
case $COPY_CPUS in '' | [0-9]*) ;; *) echo "dma-dram-share.sh: --copy-cpus is a cpu list like 7,15,23,31" >&2; exit 64 ;; esac
# The conditions and each one's h2d_probe arguments (dma-dram-share.py plan: the grammar's one parser).
PLAN=$(p2 plan "$CONDS" "$TOKEN_MS" "$CHUNK" "$RING" "$FILL" "$BLOCK" "$ANON") || exit 64
IFS=',' read -r -a conds <<< "$CONDS"

if [ "$PARSE_ONLY" = 1 ]; then
  echo "[parse] rounds=$ROUNDS seconds=$SECS warmup=$WARMUP arms=$ARMS threads=${THREADS:-auto} copy-cpus=${COPY_CPUS:-auto} chunk=$CHUNK ring=$RING fill-threads=$FILL copy-alone=$ALONE_S bound=$BOUND"
  printf '%s\n' "$PLAN" | sed 's/^/[parse] /'
  exit 0
fi

# shellcheck source=tools/ref/ref-paths.sh
source "$HERE/ref-paths.sh"
# shellcheck source=tools/ref/timing-card.sh
source "$HERE/timing-card.sh"
PROBE=target/release/h2d_probe
BENCH=target/release/bench_v41_host
assert_fresh_binary "$PROBE" || exit $?
PROBE_SHA=$BIN_SHA
assert_fresh_binary "$BENCH" || exit $?
BENCH_SHA=$BIN_SHA
read -r auto_threads auto_cpus <<< "$(p2 topology)"
THREADS=${THREADS:-$auto_threads}
COPY_CPUS=${COPY_CPUS:-$auto_cpus}
OUT=target/dma-dram-share/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
PIDS=$OUT/pids
: > "$PIDS"
echo "[p2] out=$OUT threads=$THREADS copy-cpus=$COPY_CPUS arms=$ARMS rounds=$ROUNDS seconds=$SECS chunk=$CHUNK ring=$RING fill-threads=$FILL conds=$CONDS token-ms=$TOKEN_MS block=$BLOCK anon-bytes=$ANON probe=$PROBE_SHA bench=$BENCH_SHA"
printf '%s\n' "$PLAN" | sed 's/^/[p2] /'

CPID=
STOP=
# On any exit a copy still running is asked to stop by its file, then its timeout wrapper is signalled.
# shellcheck disable=SC2329 # the EXIT trap below runs it
cleanup() {
  if [ -n "$CPID" ] && kill -0 "$CPID" 2> /dev/null; then
    touch "$STOP"
    sleep 3
    kill -0 "$CPID" 2> /dev/null && kill "$CPID" 2> /dev/null
    wait "$CPID" 2> /dev/null || true
  fi
}
trap cleanup EXIT

# probe_args <cond>: the cond's h2d_probe arguments, from the plan.
probe_args() {
  printf '%s\n' "$PLAN" | awk -v c="$1" '$1 == "cond=" c { $1 = ""; print; exit }'
}

# start_copy <cond> <log>: the cond's copy under timeout and taskset, its pid recorded, back once its
# first interval line is out (the loop at its rate).
start_copy() {
  local cond=$1 log=$2 i args
  read -r -a args <<< "$(probe_args "$cond")"
  STOP=$OUT/stop.${cond//:/-}.$RANDOM
  CUDA_VISIBLE_DEVICES=$TIMING_GPU timeout --kill-after=10 $((BOUND + 60)) taskset -c "$COPY_CPUS" \
    "$PROBE" "${args[@]}" --seconds "$BOUND" --stop-file "$STOP" > "$log" 2>&1 &
  CPID=$!
  echo "$CPID $(cat "/proc/$CPID/comm" 2> /dev/null || echo '?') $cond $(now)" >> "$PIDS"
  for ((i = 0; i < 240; i++)); do
    grep -q '^h2d loop .* interval_ms=' "$log" && return 0
    if ! kill -0 "$CPID" 2> /dev/null; then
      echo "[p2] the $cond copy ended before its first interval:" >&2
      cat "$log" >&2
      return 1
    fi
    sleep 0.5
  done
  echo "[p2] the $cond copy printed no interval in 120 s" >&2
  return 1
}

# stop_copy: its stop file, then wait on its pid; the copy's rc.
stop_copy() {
  local r=0
  touch "$STOP"
  wait "$CPID" || r=$?
  CPID=
  return "$r"
}

# stamp: every line with the epoch milliseconds it was read at, tab-separated.
stamp() {
  local line
  while IFS= read -r line; do
    printf '%s\t%s\n' "$((${EPOCHREALTIME/./} / 1000))" "$line"
  done
}

# bench <log>: one bench run, its lines stamped; its rc.
bench() {
  local r
  set +e
  BLOOMERY_THREADS=$THREADS BLOOMERY_HOST_LEASE=1 lease_bounded "$BOUND" "$BENCH" --time --rounds 1 \
    --seconds "$SECS" --warmup "$WARMUP" --arms "$ARMS" 2>&1 | stamp > "$1"
  r=${PIPESTATUS[0]}
  set -e
  return "$r"
}

WITNESS=(head loadavg pressure-cpu pressure-io card gpus gpu-apps busiest lock-holder cpu-freq meminfo model)
lease_take
witness pre
rc=0
: > "$OUT/alone"
for cond in "${conds[@]}"; do
  [ "$cond" != alone ] || continue
  log=$OUT/alone.${cond//:/-}.log
  start_copy "$cond" "$log" || { rc=1; break; }
  sleep "$ALONE_S"
  stop_copy || rc=$?
  echo "$cond $(p2 alone "$log")" | tee -a "$OUT/alone"
done
: > "$OUT/rows"
nc=${#conds[@]}
for ((r = 1; r <= ROUNDS && rc == 0; r++)); do
  for ((j = 0; j < nc; j++)); do
    cond=${conds[$(((j + r - 1) % nc))]}
    log=$OUT/r$r.${cond//:/-}
    echo "--- round $r $cond $(now) load=$(cut -d' ' -f1-3 /proc/loadavg) io=$(grep '^some' /proc/pressure/io | cut -d' ' -f2)"
    copy=-
    if [ "$cond" != alone ]; then
      copy=$log.copy
      start_copy "$cond" "$copy" || { rc=1; break; }
    fi
    br=0
    bench "$log.bench" || br=$?
    if [ "$cond" != alone ]; then stop_copy || rc=$?; fi
    if [ "$br" != 0 ]; then
      echo "[p2] bench rc=$br in round $r $cond; its log:" >&2
      cut -f2- "$log.bench" | tail -n 20 >&2
      rc=$br
      break
    fi
    p2 row "$r" "$cond" "$log.bench" "$copy" "$COPY_CPUS" | tee -a "$OUT/rows"
  done
done
witness post
p2 summary "$OUT/rows" "$OUT/alone" "$CONDS" "$BLOCK"
echo "dma-dram-share rc=$rc out=$OUT"
exit "$rc"
