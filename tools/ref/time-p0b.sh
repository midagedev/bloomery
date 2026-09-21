#!/usr/bin/env bash
# P0b step timing under the machine-wide lease: op 8-node graph vs fused 4-node graph, us/replay,
# plus the 4- and 8-node empty-graph references gate_p0b prints. Lead-only. Runs on the box under
# tools/box.sh after `gate-gpu-p0b` has built the binary. Witness blocks bracket the timed run so a
# reader can tell a quiet box from a contended one (docs/quiet-machine.md in rig-log).
set -euo pipefail
export BLOOMERY_DATA=${BLOOMERY_DATA:-/root/bloomery-data}
LOCK=/root/bloomery-cpu.lock
GPU_UUID=GPU-307fa0f6-daae-24e5-6fd3-cd50620de6b1
BIN=target/release/gate_p0b
[ -x "$BIN" ] || { echo "no $BIN — run: just gate-gpu-p0b first" >&2; exit 2; }
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
timeout --kill-after=10 600 "$BIN" --time
rc=$?
witness post
echo "gate_p0b --time rc=$rc"
exit $rc
