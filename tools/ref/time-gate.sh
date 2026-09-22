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
# The card pin, the witness lines and the stale-binary refusal are the same code the three
# depth/profile runners use; this runner used to pin the 3090 by hand and print no card name.
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
NAME=${1:?usage: time-gate.sh <gate_bin_name> [extra args...]}
shift
ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then
  ARGS=(--time)
fi
BIN=target/release/$NAME
assert_fresh_binary "$BIN" || exit $?
witness() {
  echo "--- witness $1 $(now) ---"
  witness_card
}
exec 9>"$LOCK"
echo "[lease] waiting for $LOCK ..."
flock -w 1800 9 || { echo "[lease] timed out after 30 min"; exit 75; }
echo "[lease] held by pid $$ at $(now)"
witness pre
# `|| rc=$?`, not a bare `rc=$?`: under `set -e` a non-zero gate exits the script on the
# spot, and the post witness and the rc line never print (measured 2026-09-21 — a FAIL-first
# timing run lost its post witness). A failing timed run must still be a complete record.
rc=0
timeout --kill-after=10 600 "$BIN" "${ARGS[@]}" || rc=$?
witness post
echo "$NAME ${ARGS[*]} rc=$rc"
exit $rc
