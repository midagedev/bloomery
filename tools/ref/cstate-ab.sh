#!/usr/bin/env bash
# Idle-state A/B: the deepest cpuidle state (state2, C2 on this box: 18 us exit latency) enabled
# versus disabled on every CPU, the two arms interleaved inside one lease and their order rotated
# every round. Lead-only; a MEASUREMENT. Runs on the box under tools/box.sh after the recipe has
# built the binary.
#
#   cstate-ab.sh <rounds> <command...>
#   cstate-ab.sh 6 env BLOOMERY_HYBRID_NL=32 target/release/generate -n 64 --time
#
# The one thing it changes is fenced: every exit path writes back each CPU's `disable` flag as it
# read it at start (EXIT trap) and prints the flags it left. The command runs once per arm per
# round from the repo root with the timing card pinned (timing-card.sh); of its output only the
# lines matching CSTATE_AB_KEEP (grep -E; default: generate's SMOKE footer and bench_join's timed
# lines) are printed, each behind a `<state>=<on|off> round=<r> |` tag, so the log answers per arm.
set -euo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
export BLOOMERY_DATA
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
ROUNDS=${1:?usage: cstate-ab.sh <rounds> <command...>}
shift
[ $# -gt 0 ] || { echo "usage: cstate-ab.sh <rounds> <command...>" >&2; exit 64; }
# A leading zero is refused with the rest: bash arithmetic would read 08 as octal.
case $ROUNDS in
  '' | *[!0-9]* | 0*) echo "cstate-ab: ROUNDS is a positive integer, got '$ROUNDS'" >&2; exit 64 ;;
esac
KEEP=${CSTATE_AB_KEEP:-'^SMOKE|^time |mech='}
STATE=${CSTATE_AB_STATE:-state2}
FILES=(/sys/devices/system/cpu/cpu[0-9]*/cpuidle/"$STATE"/disable)
[ -e "${FILES[0]}" ] || { echo "cstate-ab: no idle state $STATE on this machine" >&2; exit 2; }
for a in "$@"; do
  if [[ $a == target/release/* ]]; then
    assert_fresh_binary "$a" || exit $?
    break
  fi
done

ORIG=()
for f in "${FILES[@]}"; do ORIG+=("$(cat "$f")"); done
set_disable() {
  local f
  for f in "${FILES[@]}"; do echo "$1" > "$f"; done
}
restore() {
  local i
  for i in "${!FILES[@]}"; do echo "${ORIG[$i]}" > "${FILES[$i]}"; done
  echo "[cstate-ab] restored: $STATE disable flags $(cat "${FILES[@]}" | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
}
trap restore EXIT

WITNESS=(head indent card model busiest)
lease_take
guard_other
d=/sys/devices/system/cpu/cpu0/cpuidle/$STATE
echo "[cstate-ab] $STATE=$(cat "$d/name") exit latency $(cat "$d/latency") us, residency $(cat "$d/residency") us; ${#FILES[@]} CPUs; rounds=$ROUNDS; cmd: $*"
witness pre
for ((r = 1; r <= ROUNDS; r++)); do
  if ((r % 2)); then order=(on off); else order=(off on); fi
  for arm in "${order[@]}"; do
    if [ "$arm" = off ]; then set_disable 1; else set_disable 0; fi
    rc=0
    out=$(timeout --kill-after=10 600 "$@" 2>&1) || rc=$?
    grep -E "$KEEP" <<< "$out" | sed "s/^/$STATE=$arm round=$r | /" || true
    if [ "$rc" != 0 ]; then
      echo "$STATE=$arm round=$r rc=$rc; last lines:"
      tail -n 5 <<< "$out"
      witness post
      exit "$rc"
    fi
  done
done
witness post
