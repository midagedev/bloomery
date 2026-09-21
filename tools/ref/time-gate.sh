#!/usr/bin/env bash
# Graph-replay timing of one gate binary under the machine-wide lease: `<gate> --time` prints
# us/replay of its captured graphs plus empty-graph references. Lead-only. Runs on the box under
# tools/box.sh after the matching `gate-gpu-*` recipe has built the binary. Witness blocks bracket
# the timed run so a reader can tell a quiet box from a contended one (docs/quiet-machine.md in
# rig-log). Usage: time-gate.sh <gate_bin_name> [extra args...]
#   time-gate.sh gate_p8              -> gate_p8 --time      (the historical default)
#   time-gate.sh gate_p8 --profile    -> gate_p8 --profile   (gate_p8's per-op table)
# Extra args replace the default `--time` wholesale; the lease and witness blocks wrap whatever
# runs, unchanged.
set -euo pipefail
export BLOOMERY_DATA=${BLOOMERY_DATA:-/root/bloomery-data}
LOCK=/root/bloomery-cpu.lock
GPU_UUID=GPU-307fa0f6-daae-24e5-6fd3-cd50620de6b1
NAME=${1:?usage: time-gate.sh <gate_bin_name> [extra args...]}
shift
ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then
  ARGS=(--time)
fi
BIN=target/release/$NAME
[ -x "$BIN" ] || { echo "no $BIN — run the matching just gate-gpu-* recipe first" >&2; exit 2; }
witness() {
  echo "--- witness $1 $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw,clocks.sm --format=csv
  echo "compute-apps-3090:"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv -i "$GPU_UUID"
  echo "loadavg: $(cat /proc/loadavg)"
  echo "pressure-io: $(grep '^some' /proc/pressure/io | head -n1)"
  echo "llm.service: $(systemctl is-active llm.service || true)"
}
exec 9>"$LOCK"
echo "[lease] waiting for $LOCK ..."
flock -w 1800 9 || { echo "[lease] timed out after 30 min"; exit 75; }
echo "[lease] held by pid $$ at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
witness pre
timeout --kill-after=10 600 "$BIN" "${ARGS[@]}"
rc=$?
witness post
echo "$NAME ${ARGS[*]} rc=$rc"
exit $rc
