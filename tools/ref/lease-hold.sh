#!/usr/bin/env bash
# Hold the machine-wide lease for a job that has no timing runner: a sysctl A/B, a probe beside a
# concurrent load, a conversion that needs the box to itself. The one sanctioned way; a raw `flock`
# on the lease file is not a lease — it carries no card and no witness, so its log cannot say what
# the run was for or what it predicted.
#
#   tools/ref/lease-hold.sh --card docs/cards/<slug>.card -- <command…>
#
# The card goes through lease_take (tools/ref/lease.sh), the same check every runner gets: its path,
# sha256 and body go into the log as [lease] lines, and a missing or refused card exits with
# tools/ref/card.py's code before the lease is taken. An ab card is checked at its own `rounds` (or
# BLOOMERY_AB_ROUNDS): the command's arms are its own to count. The lease is held on descriptor 9 for
# the command's life, `witness pre` and `witness post` bracket the command, and the command's exit
# code is this script's. The command runs with descriptor 9 closed, so nothing it leaves running can
# keep the lease: the lease ends when this process does, on every exit path — the kernel closes the
# descriptor even when the process is killed. The command must not take the lease itself: it runs
# with BLOOMERY_LEASE_HELD set to this script's pid, and lease_take refuses under it (exit 64) rather
# than wait on the lease held here. Runs on the box, through tools/box.sh.
set -euo pipefail
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
USAGE='usage: tools/ref/lease-hold.sh --card docs/cards/<slug>.card -- <command…>'
card=
while [ $# -gt 0 ]; do
  case $1 in
    --card)
      [ $# -ge 2 ] || { echo "lease-hold: --card needs a path; $USAGE" >&2; exit 64; }
      card=$2
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "lease-hold: unknown argument '$1'; $USAGE" >&2
      exit 64
      ;;
  esac
done
[ -n "$card" ] || { echo "lease-hold: no --card; $USAGE" >&2; exit 64; }
[ $# -gt 0 ] || { echo "lease-hold: no command after --; $USAGE" >&2; exit 64; }
if [ -n "${BLOOMERY_LEASE_CARD:-}" ] && [ "$BLOOMERY_LEASE_CARD" != "$card" ]; then
  echo "lease-hold: --card $card and BLOOMERY_LEASE_CARD=$BLOOMERY_LEASE_CARD name two cards" >&2
  exit 64
fi
BLOOMERY_LEASE_CARD=$card
# Machine-wide fields only: the command is not this script's to describe. head-epoch gives the
# seconds a command's own log can be lined up against.
WITNESS=(head-epoch indent loadavg pressure-cpu pressure-io meminfo pgmajfault gpu-apps busiest lock-holder)
lease_take
trap 'lease_release; echo "[lease] released at $(now)"' EXIT
echo "[lease-hold] command: $*"
witness pre
t0=$(date +%s)
rc=0
BLOOMERY_LEASE_HELD=$$ "$@" 9>&- || rc=$?
witness post
echo "[lease-hold] rc=$rc held=$(($(date +%s) - t0)) s"
exit "$rc"
