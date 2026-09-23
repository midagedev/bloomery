#!/usr/bin/env bash
# Host binary gate runner: the lock-free twin of tools/gpu-gate.sh, for a gate or check binary that
# touches no card. Runs on the box, from the directory box.sh entered: target/release/<name> under
# the same bound as every gate, and its exit code, unchanged. The arguments after the name go to the
# binary as they are.
#   ./tools/box.sh 'cargo build --release -p bloomery-gpu-gates --bin gate_deepseek41_plan && bash tools/host-gate.sh gate_deepseek41_plan'
#
# It takes no lock: a host gate holds no card, so it neither waits on the GPU gate lock nor blocks
# the tracks queued on it. The bound is the point — a recipe that runs `timeout … target/release/…`
# itself leaves the bound and the exit code to be spelled once per recipe, and tools/check-recipes.sh
# refuses such a line.
#
# Exit code: the binary's. 124 = ended by TERM at the bound, 137 = ignored TERM and ended by
# --kill-after's KILL, 64 = usage, 2 = no binary.
# Environment: BLOOMERY_GATE_BOUND (seconds, default 900 — the lever of tools/gate.sh and gpu-gate.sh).
set -uo pipefail
NAME=${1:-}
[ -n "$NAME" ] || { echo "usage: host-gate.sh <target/release binary> [args...]" >&2; exit 64; }
shift
BOUND=${BLOOMERY_GATE_BOUND:-900}
case "$BOUND" in
  '' | *[!0-9]* | 0) echo "host-gate.sh: BLOOMERY_GATE_BOUND must be a positive integer, got '$BOUND'" >&2; exit 64 ;;
esac
EXE=./target/release/$NAME
[ -x "$EXE" ] || { echo "host-gate.sh: no $EXE — the recipe builds it before calling this" >&2; exit 2; }
timeout --kill-after=10 "$BOUND" "$EXE" "$@"
rc=$?
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  echo "HOST GATE TIMED OUT: $NAME after the ${BOUND}s bound (exit $rc) — a gate that hangs is a red gate, not a silent one" >&2
elif [ "$rc" -ne 0 ]; then
  echo "HOST GATE RED: $NAME (exit $rc)" >&2
fi
exit "$rc"
