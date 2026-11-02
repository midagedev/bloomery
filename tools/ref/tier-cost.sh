#!/usr/bin/env bash
# Plan (b') local tier versus a tier_server on this machine's first card.
#   bash tools/ref/tier-cost.sh [--dry-run] [GIB [ROUNDS]]
# Build gate_qwen38_nodes and tier_server first. Requires the qwen4exp model and corpus.
set -euo pipefail
source "${BASH_SOURCE[0]%/*}/lease.sh"
if [ "${1:-}" = --self-test ]; then
  exec python3 "$LEASE_TREE/tools/ref/tier-cost.py" --self-test
fi
dry=0
if [ "${1:-}" = --dry-run ]; then dry=1; shift; fi
[ "$#" -le 2 ] || { echo 'usage: tier-cost.sh [--dry-run] [GIB [ROUNDS]]' >&2; exit 64; }
gib=${1:-16}; ROUNDS=${2:-6}
[[ $gib =~ ^[1-9][0-9]*$ ]] && [ "$gib" -le 16 ] || { echo 'tier-cost: GIB must be 1..16' >&2; exit 64; }
[[ $ROUNDS =~ ^[1-9][0-9]*$ ]] && [ "$ROUNDS" -ge 2 ] && [ "$ROUNDS" -le 100 ] || { echo 'tier-cost: ROUNDS must be 2..100' >&2; exit 64; }
BIN=$LEASE_TREE/target/release/gate_qwen38_nodes
echo 'one machine, loopback: not a two-machine timing'
if [ "$dry" = 1 ]; then
  echo 'DRY RUN: no lease or GPU load; commands only, not timing evidence'
  for arm in local node; do printf '%q --measure %s %s\n' "$BIN" "$arm" "$gib"; done
  printf 'rounds=%s order=local,node then node,local\n' "$ROUNDS"
  exit 0
fi
cd "$LEASE_TREE"
TIMING_LOCAL_RUNNER=1
source tools/ref/timing-card.sh
assert_fresh_binary "$BIN"
assert_fresh_binary "$LEASE_TREE/target/release/tier_server"
MODEL=${BLOOMERY_REF_MODEL:?set BLOOMERY_REF_MODEL}
MODEL_NAME=qwen4exp
WITNESS=(head-open indent card gpu-apps model busiest)
out=$(mktemp -d "${TMPDIR:-/tmp}/tier-cost.XXXXXX")
echo "logs=$out"
"$BIN" --records-schema > "$out/schema.jsonl"
trap 'lease_release' EXIT
timing_cards_before_lease
lease_take
timing_cards_start
tier_head=${BLOOMERY_GIT_COMMIT:-$(git rev-parse HEAD)}
printf 'local_commit=%s\nnode_commit=%s\nmodel=%s\n' "$tier_head" "$tier_head" "$MODEL"
sha256sum "$BIN" target/release/tier_server
lease_bounded "$LEASE_ARM_BOUND" nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
witness pre
for ((r=1; r<=ROUNDS; r++)); do
  arms=(local node)
  if [ "$((r % 2))" = 0 ]; then arms=(node local); fi
  for arm in "${arms[@]}"; do
    witness "pre r$r $arm"
    lease_bounded "$LEASE_ARM_BOUND" "$BIN" --measure "$arm" "$gib" > "$out/r$r-$arm.log" 2>&1
    witness "post r$r $arm"
    if ! timing_cards_arm ''; then echo "tier-cost: invalid witness: $TWOCARD_WHY" >&2; exit 1; fi
    cat "$out/r$r-$arm.log"
  done
  python3 tools/ref/tier-cost.py "$out" "$r"
done
witness post
python3 tools/ref/tier-cost.py "$out" "$ROUNDS"
