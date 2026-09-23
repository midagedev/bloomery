#!/usr/bin/env bash
# Stage-0 back-to-back measurement: ggml reference, then the rust kernel,
# same card (RTX 3090), same minute, one invocation.
# Runs on the box under tools/box.sh. Prints witness blocks around each
# engine's timed section.
set -euo pipefail
# BLOOMERY_DATA defaults in ref-paths.sh, as in every runner here (BLOOMERY_DATA overrides it).
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# This runner stays on the 3090: the stage-0 rates it has recorded are that card's, and two
# cards' numbers never share a table. timing-card.sh is not sourced here on purpose — it
# exports CUDA_VISIBLE_DEVICES for the timing card, which would move this runner silently.
# shellcheck source=tools/ref/cards.sh
source "${BASH_SOURCE[0]%/*}/cards.sh"
# The witness fields; this runner waits for an idle 3090 instead of taking the lease.
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
GPU_UUID=$GPU_3090

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

WITNESS=(head stage0-gpus stage0-apps loadavg model pressure-io-avg10)

bash tools/ref/build.sh
export BLOOMERY_DATA
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
