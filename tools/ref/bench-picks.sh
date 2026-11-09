#!/usr/bin/env bash
# The kernel-pick bench under the machine-wide lease: `bench_picks --time` forces every candidate of each pick pair on the
# same shape and prints the `kcal` rows tools/ref/kernel-model.py reads. Lead-only. Runs under tools/box.sh after
# `just bench-picks` (or the build line in its recipe) has built the binary; witness blocks bracket the timed run.
# On the box the timing card is the A6000 (timing-card.sh); on a machine with one card, under tools/box.sh's local mode
# (BLOOMERY_BOX_LOCAL=1), the card in view and the lease in BLOOMERY_LOCK_DIR. A prediction card is required
# (BLOOMERY_LEASE_CARD=docs/cards/<slug>.card).
# Usage: bench-picks.sh [bench_picks args…]     --time is added unless an argument is --time or --check; `--arm PREFIX` keeps the arms of that label prefix
set -euo pipefail
TIMING_LOCAL_RUNNER=1
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
ARGS=("$@")
case " ${ARGS[*]-} " in
  *" --time "* | *" --check "*) ;;
  *) ARGS=(--time "${ARGS[@]}") ;;
esac
BIN=target/release/bench_picks
assert_fresh_binary "$BIN" || exit $?
WITNESS=(head-open indent card gpu-apps binary busiest)
timing_cards_before_lease
lease_take
timing_cards_start
witness pre
# `|| rc=$?`: a failed timed run still prints its post witness and rc line (see time-gate.sh).
rc=0
run_bounded timeout --kill-after=10 1500 "$BIN" "${ARGS[@]}" || rc=$?
if ! timing_cards_arm ''; then
  echo "bench-picks: invalid witness: $TWOCARD_WHY" >&2
  rc=1
fi
witness post
echo "bench_picks ${ARGS[*]} rc=$rc"
exit $rc
