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

# run_bounded <command…>: the bounded command (`timeout … <binary>`) as a child the runner waits for,
# its return the child's code. timeout runs the binary in a process group of its own, which a signal
# to the runner's group never reaches. So a TERM or HUP to the runner (an INT too, unless the runner
# was started with & by a shell, which leaves INT ignored) goes on to the child as a TERM, and the
# runner goes on only after the child has ended: an ended run leaves no binary holding its locks and
# its output. The child keeps the runner's stdin, which a command run with & would not. The trap is
# set before the child starts, so no signal falls between the two: one that comes before the child's
# pid is known is held in BIN_SIGNAL and passed on as soon as the pid is. Once it has passed a signal
# on, the trap ignores further ones (GNU timeout sends its TERM to the runner and again to the
# runner's group), so the second wait is not cut short and reads the child's status; it ignores none
# before the child starts, which would start the child with them ignored. The caller's own traps on
# the three signals are back when it returns, and BIN_SIGNAL names the signal that was passed on
# (empty: none), for a caller that stops on it.
run_bounded() {
  local traps s rc=0 BIN_PID=''
  traps=$(trap -p TERM INT HUP)
  # shellcheck disable=SC2034 # the caller reads it
  BIN_SIGNAL=
  for s in TERM INT HUP; do
    # shellcheck disable=SC2064 # the signal's name now, the pid when the trap runs
    trap "BIN_SIGNAL=$s; [ -z \"\$BIN_PID\" ] || { trap '' TERM INT HUP; kill -TERM \"\$BIN_PID\" 2> /dev/null; }" "$s"
  done
  "$@" <&0 &
  BIN_PID=$!
  if [ -n "$BIN_SIGNAL" ]; then
    trap '' TERM INT HUP
    kill -TERM "$BIN_PID" 2> /dev/null
  fi
  wait "$BIN_PID" || true
  wait "$BIN_PID" || rc=$?
  trap - TERM INT HUP
  eval "$traps"
  return "$rc"
}
