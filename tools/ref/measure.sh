#!/usr/bin/env bash
# Stage-0 back-to-back measurement: ggml reference, then the rust kernel,
# same card (RTX 3090), same minute, one invocation.
# Runs on the box under tools/box.sh. Prints witness blocks around each
# engine's timed section.
set -euo pipefail
GPU_UUID=GPU-307fa0f6-daae-24e5-6fd3-cd50620de6b1

wait_gpu() {
  local i out lines
  for i in $(seq 1 90); do
    out=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv -i "$GPU_UUID")
    lines=$(echo "$out" | tail -n +2 | grep -cv '^[[:space:]]*$' || true)
    if [ "$lines" -eq 0 ]; then
      echo "[wait_gpu] 3090 idle, proceeding"
      echo "$out"
      return 0
    fi
    echo "[wait_gpu] 3090 busy ($i/90):" >&2
    echo "$out" >&2
    sleep 10
  done
  echo "[wait_gpu] proceeding after 15 min wait" >&2
}

witness() {
  echo "--- witness $1 $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw --format=csv
  echo "compute-apps-3090:"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv -i "$GPU_UUID"
  echo "loadavg: $(cat /proc/loadavg)"
  echo "pressure-io avg10: $(grep '^some' /proc/pressure/io | head -n 1)"
}

bash tools/ref/build.sh
export BLOOMERY_DATA=${BLOOMERY_DATA:-/root/bloomery-data}
mkdir -p "$BLOOMERY_DATA"

wait_gpu
witness pre-ref
"$BLOOMERY_DATA/bin/q3k_ref"
witness post-ref

wait_gpu
witness pre-rust
cd crates/q3k-gemv && cargo oxide run --arch sm_86
cd - > /dev/null
witness post-rust
