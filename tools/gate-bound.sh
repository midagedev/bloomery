#!/usr/bin/env bash
# shellcheck shell=bash
# The bound of the gate runners (tools/gate.sh, tools/gpu-gate.sh, tools/host-gate.sh): the seconds
# a gate's binary may run before `timeout` ends it, BLOOMERY_GATE_BOUND, 900 when unset, and
# run_bounded, which runs the bounded command so that an ended runner takes it with it (also
# tools/ref/lease.sh's lease_bounded, under a bound of its own). Sourced, never executed; one parser,
# so the three runners take and refuse the same values.
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

# lease_stop <timeout pid>: TERM the monitor and its group, wait up to 10 s before KILL, then reap
# the monitor. Allow another 10 s for KILL to finish, reporting a group still alive then. A monitor
# that exited without forwarding TERM can leave a live group behind. Zombies hold no descriptors;
# only live processes keep cleanup waiting. The caller ignores further signals.
lease_stop() {
  local pid=$1 sig target err table p pg state live failed=0 deadline
  for sig in TERM KILL; do
    for target in "$pid" "-$pid"; do
      if err=$(LC_ALL=C kill -s "$sig" -- "$target" 2>&1); then :; else
        case $err in
          *'No such process'*) ;;
          *) printf '[bound] cannot send %s to %s: %s\n' "$sig" "$target" "$err" >&2; failed=70 ;;
        esac
      fi
    done
    deadline=$((SECONDS + 10))
    while :; do
      if ! table=$(ps -eo pid=,pgid=,stat=); then
        echo "[bound] cannot read processes while stopping $pid" >&2
        failed=70
        break
      fi
      live=''
      while IFS=$' \t\n' read -r p pg state; do
        if { [ "$p" = "$pid" ] || [ "$pg" = "$pid" ]; } && [[ $state != Z* ]]; then live=1; break; fi
      done <<< "$table"
      [ -n "$live" ] || break 2
      if [ "$SECONDS" -ge "$deadline" ]; then
        if [ "$sig" = KILL ]; then
          echo "[bound] $pid or its process group still alive 10 s after KILL" >&2
          return 70
        fi
        break
      fi
      sleep 0.1 9>&-
    done
  done
  [ "$failed" = 0 ] || return "$failed"
  wait "$pid" 2> /dev/null || :
  return "$failed"
}

# run_bounded <command…>: the bounded command (`timeout … <binary>`) as a child the runner waits for,
# its return the child's code. timeout runs the binary in a process group of its own, which a signal
# to the runner's group never reaches. So a TERM or HUP to the runner (an INT too, unless the runner
# was started with & by a shell, which leaves INT ignored) goes on to the child as a TERM, and the
# runner waits for the child and its process group before going on. The TERM reaches the group too,
# including a command GNU timeout started before returning from fork. lease_stop
# bounds that cleanup at 10 s before KILL. The child keeps the runner's stdin, which a command run
# with & would not. The trap is set before the child starts: one that comes before the child's
# pid is known is held in BIN_SIGNAL and passed on as soon as the pid is, once (BIN_SENT says the trap
# has passed it on already). Once it has passed a signal on, the trap ignores further ones (GNU
# timeout sends its TERM to the runner and again to the runner's group), so the second wait is not
# cut short and reads the child's status; it ignores none before the child starts, which would start
# the child with them ignored. A TERM to a child that has already ended is no error, under set -e
# too, and the pid is cleared once the child is reaped, so a late signal reaches no other process.
# The caller's own traps on the three signals are back when it returns, and BIN_SIGNAL names the
# signal that was passed on (empty: none), for a caller that stops on it.
run_bounded() {
  local traps s rc=0 BIN_PID='' BIN_SENT=''
  traps=$(trap -p TERM INT HUP)
  # shellcheck disable=SC2034 # the caller reads it
  BIN_SIGNAL=
  for s in TERM INT HUP; do
    # shellcheck disable=SC2064 # the signal's name now, the pid when the trap runs
    trap "BIN_SIGNAL=$s; [ -z \"\$BIN_PID\" ] || { trap '' TERM INT HUP; BIN_SENT=1; lease_stop \"\$BIN_PID\" || :; }" "$s"
  done
  "$@" <&0 &
  BIN_PID=$!
  if [ -n "$BIN_SIGNAL" ]; then
    trap '' TERM INT HUP
    [ -n "$BIN_SENT" ] || lease_stop "$BIN_PID" || :
  fi
  wait "$BIN_PID" || true
  wait "$BIN_PID" || rc=$?
  BIN_PID=''
  trap - TERM INT HUP
  eval "$traps"
  return "$rc"
}
