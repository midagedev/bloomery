#!/usr/bin/env bash
# GLM-5.3-Flash decode kernel table: where one decode step's time goes, asked of Nsight Systems on
# the timing card (run on the box, under the lease, lead-only; `just nsys-gpu-glm5next [D…]`).
#
# nsys-gpu.sh and nsys-ds41.sh profile other binaries and command lines; this one profiles
# depth-glm5next.sh's ours arm as it is timed: generate_glm5next --tokens <the first D ids of
# GLM_PROSE> -n N --ctx C --place a --time (C BLOOMERY_GEN_CTX, default 2048; N BLOOMERY_NSYS_N,
# default 8), traced with -t cuda and graph nodes as kernels (--cuda-graph-trace=node), no CPU sampling
# or context-switch trace (the host tier spins a pool; sampling would perturb the joins). GLM feeds
# its prompt one decode step a position, each a replay of the step graph, so the trace holds
# D + N - 1 replays; the kernel table (tools/ref/nsys-table.py, one row per kernel and grid) divides by
# the cuGraphLaunch calls, the prompt's replays included, and again by D + N - 1 (--steps), so a count
# of graph launches other than one a step shows as two different headers. That is a step's mean over depths 0 .. D + N - 2
# — the gemv and expert rows are depth-flat, the latent attention rows are not, so read those at the
# window's mean depth. After the table: the run's `time prompt`, `time step` and SMOKE lines (the
# step walls the kernel sum stands beside) and the rows whose kernel name holds q8_0 (the Q8_0 gemv of
# the card trunk) summed per step.
#
#   just nsys-gpu-glm5next 512
#   BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 just nsys-gpu-glm5next 512     # the command lines, no lease
#   bash tools/ref/nsys-glm5next.sh --analyze <out>.sqlite <out>.txt <D+N-1>  # the tables again, no profile
#
# Environment: BLOOMERY_NSYS_N, BLOOMERY_GEN_CTX, BLOOMERY_GEN_BIN (default
# target/release/generate_glm5next), BLOOMERY_NSYS_OUT (default $BLOOMERY_DATA/nsys),
# BLOOMERY_ARM_BOUND (seconds one profile or export may run, default 900), BLOOMERY_DRY.
set -uo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
TABLE=${BASH_SOURCE[0]%/*}/nsys-table.py

analyze() { # analyze <sqlite> <run log> <replays>
  local db=$1 log=$2 r=$3
  echo "--- the run's walls ($log)"
  grep -E '^(time prompt|time step|SMOKE) ' "$log" | sed 's/^/    /'
  echo "--- kernels, per step (nsys-table.py: steps = the cuGraphLaunch calls; $r replays expected)"
  lease_bounded "$LEASE_ARM_BOUND" python3 "$TABLE" "$db" > "$db.table.txt" || return $?
  cat "$db.table.txt"
  echo "--- the same trace over the $r replays the command ran (--steps $r): the header only"
  lease_bounded "$LEASE_ARM_BOUND" python3 "$TABLE" "$db" --steps "$r" > "$db.table-r.txt" || return $?
  sed -n 1p "$db.table-r.txt"
  echo "--- the Q8_0 gemv rows (--steps $r)"
  lease_bounded "$LEASE_ARM_BOUND" python3 "$TABLE" "$db" --steps "$r" --pattern q8_0 || return $?
}

if [ "${1:-}" = --analyze ]; then
  [ $# = 4 ] || { echo "usage: nsys-glm5next.sh --analyze <sqlite> <run log> <replays>" >&2; exit 64; }
  # shellcheck source=tools/ref/lease.sh
  source "${BASH_SOURCE[0]%/*}/lease.sh"
  analyze "$2" "$3" "$4"
  exit $?
fi

[ "$MODEL_NAME" = glm5next ] || {
  echo "nsys-glm5next.sh: the profile is $MODEL_NAME — BLOOMERY_MODEL=glm5next on the Mac side" >&2
  exit 64
}
BIN=${BLOOMERY_GEN_BIN:-target/release/generate_glm5next}
NGEN=${BLOOMERY_NSYS_N:-8}
CTX=${BLOOMERY_GEN_CTX:-2048}
OUTDIR=${BLOOMERY_NSYS_OUT:-$BLOOMERY_DATA/nsys}
NSYS=${NSYS:-/usr/local/cuda/bin/nsys}
DRY=${BLOOMERY_DRY:-}
PROSE=$BLOOMERY_DATA/$GLM_PROSE
DEPTHS=("$@")
[ ${#DEPTHS[@]} -gt 0 ] || DEPTHS=(512)
for v in "N:$NGEN" "CTX:$CTX" "${DEPTHS[@]/#/D:}"; do
  case ${v#*:} in '' | *[!0-9]* | 0*) echo "nsys-glm5next.sh: ${v%%:*} is a positive integer, got '${v#*:}'" >&2; exit 64 ;; esac
done
[ "$NGEN" -ge 2 ] || { echo "nsys-glm5next.sh: BLOOMERY_NSYS_N is at least 2 (a generated step after token 0)" >&2; exit 64; }
for d in "${DEPTHS[@]}"; do
  [ $((d + NGEN)) -le "$CTX" ] || { echo "nsys-glm5next.sh: depth $d + $NGEN steps is past the context $CTX the plan serves" >&2; exit 64; }
done
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
[ "$TIMING_GPU" = "$GPU_A6000" ] || { echo "nsys-glm5next.sh: --place a loads on the A6000, the timing card is $TIMING_GPU" >&2; exit 64; }

profile_cmd() { # profile_cmd <depth> <ids>: into CMD
  CMD=(timeout --kill-after=10 "$LEASE_ARM_BOUND" "$NSYS" profile -t cuda --cuda-graph-trace=node --cuda-event-trace=false
       --sample=none --cpuctxsw=none -o "$out" --force-overwrite true
       "$BIN" --tokens "$2" -n "$NGEN" --ctx "$CTX" --place a --time)
}

if [ -n "$DRY" ]; then
  echo "[dry] model=$MODEL bin=$BIN n=$NGEN ctx=$CTX depths='${DEPTHS[*]}' out=$OUTDIR timing_gpu=$TIMING_GPU"
  for d in "${DEPTHS[@]}"; do
    out="$OUTDIR/<name>"
    profile_cmd "$d" "<the first $d ids of $GLM_PROSE>"
    echo "[dry] depth $d: ${CMD[*]}"
  done
  exit 0
fi

[ -f "$PROSE" ] || { echo "nsys-glm5next.sh: no prose ids at $PROSE" >&2; exit 66; }
[ "$(sha256sum "$PROSE" | cut -d' ' -f1)" = "$GLM_PROSE_SHA256" ] ||
  { echo "nsys-glm5next.sh: $PROSE is not the prose the profile pins (sha256 $GLM_PROSE_SHA256)" >&2; exit 65; }
for d in "${DEPTHS[@]}"; do
  [ "$d" -le "$(wc -l < "$PROSE")" ] || { echo "nsys-glm5next.sh: $PROSE holds fewer than $d ids" >&2; exit 64; }
done
assert_fresh_binary "$BIN" || exit $?
[ -x "$NSYS" ] || { echo "no nsys at $NSYS" >&2; exit 2; }
mkdir -p "$OUTDIR"

WITNESS=(head-open indent card model)
lease_take
echo "[config] nsys=$($NSYS --version) n=$NGEN ctx=$CTX depths='${DEPTHS[*]}' out=$OUTDIR"
witness pre
rc_all=0
for d in "${DEPTHS[@]}"; do
  out="$OUTDIR/nsys-glm5next-d${d}-n${NGEN}-$(date -u +%H%M%S)"
  profile_cmd "$d" "$(head -n "$d" "$PROSE" | paste -sd, -)"
  echo
  echo "=== depth $d (ctx $CTX, place a, $((d + NGEN - 1)) replays expected) -> $out.nsys-rep"
  guard_other
  witness "pre d=$d"
  "${CMD[@]}" > "$out.txt" 2>&1
  rc=$?
  witness "post d=$d"
  echo "[rc] $rc${OTHER_BUSY_TAG}"
  [ $rc -eq 0 ] || { rc_all=$rc; echo "--- the last 20 lines"; tail -n 20 "$out.txt"; continue; }
  lease_bounded "$LEASE_ARM_BOUND" "$NSYS" export --type sqlite --force-overwrite true -o "$out.sqlite" "$out.nsys-rep" > "$out.export.txt" 2>&1 ||
    { rc_all=$?; echo "[export] failed"; tail -n 10 "$out.export.txt"; continue; }
  analyze "$out.sqlite" "$out.txt" $((d + NGEN - 1)) || rc_all=$?
  echo "--- sources: $out.nsys-rep, $out.sqlite, $out.txt"
done
witness post
echo "[lease] released at $(now)"
exit $rc_all
