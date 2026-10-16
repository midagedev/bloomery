#!/usr/bin/env bash
# shellcheck shell=bash
# The bound of the gate runners (tools/gate.sh, tools/gpu-gate.sh, tools/host-gate.sh): the seconds
# a gate's binary may run before `timeout` ends it, BLOOMERY_GATE_BOUND, 900 when unset, and
# run_bounded, which runs the bounded command so that an ended runner takes it with it (also
# tools/ref/lease.sh's lease_bounded, under a bound of its own), and lease_stop, the process-group
# stop shared by run_bounded and the lease runners. Sourced, never executed; one parser, so the
# three runners take and refuse the same values.
#
# The value is a base-10 whole number from 1 up, with no sign and no leading zero; anything else is
# refused by name, set-and-empty included. GNU timeout reads 0 (00 as well) as no bound at all, so a
# gate handed one hangs as long as it likes; a value timeout itself refuses (a sign, a word) would end
# the runner with timeout's 125, reported as a red gate.
#
#   gate_bound <runner>   BOUND=<the value>, or the refusal on stderr and 64
gate_bound() {
  local v=${BLOOMERY_GATE_BOUND-900}
  if ! [[ $v =~ ^[1-9][0-9]*$ ]]; then
    echo "$1: BLOOMERY_GATE_BOUND is whole seconds from 1 up, base 10 with no sign or leading zero (unset: 900), got '$v'" >&2
    return 64
  fi
  # shellcheck disable=SC2034 # the sourcing runner reads it
  BOUND=$v
}

# lease_stop <timeout pid>: TERM the monitor, then its surviving group once the monitor exits,
# within LEASE_STOP_GRACE seconds (9 by default), reserving up to 2 s for KILL and reap.
# This leaves a margin before an enclosing timeout's 10 s KILL. run_bounded halves the budget
# inherited by its command, so nested cleanup finishes before its parent's. A monitor
# that exited without forwarding TERM can leave a live group behind. Zombies hold no descriptors;
# only live processes keep cleanup waiting. The caller ignores further signals.
# A failed process read skips its wait (KILL follows TERM without the grace) and returns 70 without
# reaping the monitor.
lease_stop() {
  local pid=$1 sig target err table p state live monitor failed=0 deadline limit term_deadline
  local grace=${LEASE_STOP_GRACE:-9}
  deadline=$((SECONDS + grace))
  term_deadline=$((SECONDS + (grace > 2 ? grace - 2 : 0)))
  for sig in TERM KILL; do
    limit=$deadline
    [ "$sig" != TERM ] || limit=$term_deadline
    for target in "$pid" "-$pid"; do
      if err=$(LC_ALL=C kill -s "$sig" -- "$target" 2>&1); then :; else
        case $err in
          *'No such process'*) ;;
          *) printf '[bound] cannot send %s to %s: %s\n' "$sig" "$target" "$err" >&2; failed=70 ;;
        esac
      fi
      if [ "$sig" = KILL ] && [ "$target" = "$pid" ]; then continue; fi
      while :; do
        if ! table=$(
          set -o pipefail
          ps -eo pid=,pgid=,stat= | awk -v pid="$pid" '$1 == pid || $2 == pid { print $1, $3 }'
        ); then
          echo "[bound] cannot read processes while stopping $pid" >&2
          failed=70
          break 2
        fi
        live='' monitor=''
        while IFS=$' \t\n' read -r p state; do
          if [ -n "$p" ] && [[ $state != Z* ]]; then
            live=1
            [ "$p" != "$pid" ] || monitor=1
          fi
        done <<< "$table"
        [ -n "$live" ] || break 3
        if [ "$sig" = TERM ] && [ "$target" = "$pid" ] && [ -z "$monitor" ]; then break; fi
        if [ "$SECONDS" -ge "$limit" ]; then
          if [ "$sig" = KILL ]; then
            echo "[bound] $pid or its process group still alive after its $grace s cleanup budget" >&2
            return 70
          fi
          break 2
        fi
        sleep 0.1 9>&-
      done
    done
  done
  [ "$failed" = 0 ] || return "$failed"
  wait "$pid" 2> /dev/null || :
  return "$failed"
}

# run_bounded <command…>: the bounded command (`timeout … <binary>`) as a child the runner waits for,
# returning a nonzero child status or, when the child succeeds, any cleanup failure.
# The status contract is GNU coreutils timeout's: after a TERM from outside it returns the child's code.
# uutils coreutils timeout 0.8.0 returns 143 there instead, so under it a cancelled run reports 143.
# timeout runs the binary in a process group of its own, which a signal to the runner's group never
# reaches. So a TERM or HUP to the runner (an INT too, unless the runner was started with & by a shell,
# which leaves INT ignored) goes on to the child as a TERM, and the
# runner waits for the child and its process group before going on. The TERM reaches the group too,
# including a command GNU timeout started before returning from fork. lease_stop bounds that cleanup
# within the inherited cleanup budget, including KILL and reap. The child keeps the runner's stdin,
# which a command run with & would not.
# The trap is set before the child starts: one that comes before the child's pid is known is held in
# BIN_SIGNAL and passed on as soon as the pid is, once (BIN_SENT says the trap has passed it on
# already). Once it has passed a signal on, the trap ignores further ones (GNU
# timeout sends its TERM to the runner and again to the runner's group), so the second wait is not
# cut short and reads the child's status; it ignores none before the child starts, which would start
# the child with them ignored. A TERM to a child that has already ended is no error, under set -e
# too, and the pid is cleared once the child is reaped, so a late signal reaches no other process.
# The caller's own traps on the three signals are back when it returns, and BIN_SIGNAL names the
# signal that was passed on (empty: none), for a caller that stops on it.
run_bounded() {
  local traps='' saved s rc=0 BIN_PID='' BIN_SENT=''
  local LEASE_STOP_GRACE=${LEASE_STOP_GRACE:-9}
  # shellcheck disable=SC2034 # the caller reads it
  BIN_SIGNAL=
  case $LEASE_STOP_GRACE in
    [1-9]) ;;
    *) echo "[bound] no cleanup budget for another bounded command: '$LEASE_STOP_GRACE'" >&2; return 70 ;;
  esac
  for s in TERM INT HUP; do
    saved=$(trap -p "$s")
    traps+="${saved:-trap - $s}"$'\n'
  done
  for s in TERM INT HUP; do
    # shellcheck disable=SC2064 # the signal's name now, the pid when the trap runs
    trap "BIN_SIGNAL=$s; [ -z \"\$BIN_PID\" ] || { trap '' TERM INT HUP; BIN_SENT=1; lease_stop \"\$BIN_PID\" || rc=\$?; }" "$s"
  done
  LEASE_STOP_GRACE=$((LEASE_STOP_GRACE / 2)) "$@" <&0 &
  BIN_PID=$!
  if [ -n "$BIN_SIGNAL" ]; then
    trap '' TERM INT HUP
    [ -n "$BIN_SENT" ] || lease_stop "$BIN_PID" || rc=$?
  fi
  wait "$BIN_PID" || true
  wait "$BIN_PID" || rc=$?
  BIN_PID=''
  eval "$traps"
  return "$rc"
}
