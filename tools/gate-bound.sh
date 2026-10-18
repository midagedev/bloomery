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

# lease_bound <seconds>: 0 for a positive whole number of seconds, else a named refusal on stderr and 64,
# before anything runs. The one parser of an arm's bound beside gate_bound (the gate's own).
lease_bound() {
  local bound=${1:-}
  case $bound in
    '' | *[!0-9]* | 0*)
      echo "[bound] lease_bounded: the bound is a positive number of seconds, got '$bound' (BLOOMERY_ARM_BOUND?)" >&2
      return 64
      ;;
  esac
}

# bounded <seconds> <command…>: (with LEASE_BOUNDED_OUT=<file> set for the call, the command's stdout and stderr go
# to that file, stderr to LEASE_BOUNDED_ERR when that is set too) the command under `timeout --kill-after=G
# <seconds>`, its exit code returned: the foreground form every runner outside the gate runners uses, so none writes
# its own timeout. tools/ref/lease.sh's lease_bounded is this function under the lease runners' name; it needs no lease.
# G is the inherited LEASE_STOP_GRACE (9 s at the root); run_bounded gives the command half that cleanup budget,
# leaving time for this shell to reap it before an enclosing KILL. A command cut off at its bound prints `[bound]` on
# stderr: a child that hangs is a failed run, not a slow one. A TERM, INT or HUP to the runner stops timeout and its
# process group through run_bounded, then the runner stops on that signal as it would have (by its own trap, if it set
# one), not at its next command. Where it runs in a subshell (`out=$(bounded …)`, a pipeline), a signal to the
# runner's pid alone does not reach it: the runner ends and the command runs on to its bound, holding whatever
# descriptors it inherited (the lease, a gate lock). So a runner calls it in its own shell and takes the output from
# a file (lease_capture, tools/ref/lease.sh, or LEASE_BOUNDED_OUT). The command is an executable, not a shell function
# or an assignment. LEASE_BOUNDED_PIDFILE, when set, records timeout's PID from its child before the executable starts.
bounded() {
  local bound=${1:-} rc=0 grace=${LEASE_STOP_GRACE:-9}
  lease_bound "$bound" || return $?
  shift
  if [ -n "${LEASE_BOUNDED_PIDFILE:-}" ]; then
    set -- bash -c 'printf "%s\n" "$PPID" > "$1" || exit $?; shift; exec "$@"' _ "$LEASE_BOUNDED_PIDFILE" "$@"
  fi
  if [ -n "${LEASE_BOUNDED_OUT:-}" ] && [ -n "${LEASE_BOUNDED_ERR:-}" ]; then
    run_bounded timeout --kill-after="$grace" "$bound" "$@" > "$LEASE_BOUNDED_OUT" 2> "$LEASE_BOUNDED_ERR" || rc=$?
  elif [ -n "${LEASE_BOUNDED_OUT:-}" ]; then
    run_bounded timeout --kill-after="$grace" "$bound" "$@" > "$LEASE_BOUNDED_OUT" 2>&1 || rc=$?
  else
    run_bounded timeout --kill-after="$grace" "$bound" "$@" || rc=$?
  fi
  case $rc in
    124) echo "[bound] $1 cut off at its ${bound} s bound (rc 124)" >&2 ;;
    137) echo "[bound] $1 killed (rc 137: the bound's KILL ${bound} s + $grace s in, or another SIGKILL)" >&2 ;;
  esac
  [ -z "$BIN_SIGNAL" ] || kill -s "$BIN_SIGNAL" $$ 2> /dev/null
  return "$rc"
}

# The background form of the bounded command, for a runner that works beside its child: reads its output, feeds
# its stdin, attaches a profiler to it, keeps a server up across requests. It is run_bounded's guarantees split in
# two, so a runner writes no trap, pid bookkeeping or grace period of its own:
#   bounded_bg <seconds> <command…>   starts `timeout --kill-after=G <seconds> <command…>` (G the inherited
#                                     LEASE_STOP_GRACE, 9 at the root; the command gets half of it for its own
#                                     cleanup). BG_PID is timeout's pid, which leads a process group of its own.
#                                     BG_IN and BG_OUT (or BG_OUT_APPEND), when set for the call, are the command's
#                                     stdin and its stdout and stderr, opened by the child, so a FIFO's open waits
#                                     there and never in the runner; without BG_IN the command keeps the runner's stdin.
#                                     Returns 64 for a bound that is no positive whole number, 70 when a command is
#                                     already live (one at a time: the traps are one set) or LEASE_STOP_GRACE leaves
#                                     no budget.
#   bounded_bg_wait                   waits for the command to end by itself: its status in BG_RC and returned; a
#                                     cleanup failure of a signal stop is 70 and ends the runner.
#   bounded_bg_stop                   TERM, then KILL, for the command and its group within the cleanup budget
#                                     (lease_stop): returns 0, or 70 when something outlived the budget; BG_RC is
#                                     the command's status (143 after the TERM).
# The traps on TERM, INT and HUP are set before the child starts. A signal while the command lives stops the command
# and its group, restores the caller's traps and raises the signal again, so the runner ends as it would have without
# the child (by its own trap, if it set one); one that comes before the pid is known is held and stops the command
# the moment the pid is. The caller's own traps are back when the command has ended by either call. A runner whose
# EXIT trap or error path leaves while a command lives calls bounded_bg_stop first. These functions keep their locals
# under a bg_ prefix: a caller's EXIT trap reads its own `rc` and runs beside them in the exiting function's scope.
BG_PID='' BG_RC=0 BG_SIGNAL='' BG_TRAPS='' BG_LIVE='' BG_IN='' BG_OUT='' BG_OUT_APPEND=''
bounded_bg() {
  local bg_bound=${1:-} bg_grace=${LEASE_STOP_GRACE:-9} bg_s bg_saved
  lease_bound "$bg_bound" || return $?
  shift
  case $bg_grace in
    [1-9]) ;;
    *) echo "[bound] no cleanup budget for a bounded command: '$bg_grace'" >&2; return 70 ;;
  esac
  if [ -n "$BG_LIVE" ]; then
    echo "[bound] bounded_bg: the command of pid ${BG_PID:-?} still runs; one background command at a time" >&2
    return 70
  fi
  if [ -n "$BG_OUT" ] && [ -n "$BG_OUT_APPEND" ]; then
    echo "[bound] bounded_bg: BG_OUT and BG_OUT_APPEND both set" >&2
    return 64
  fi
  BG_TRAPS='' BG_PID='' BG_SIGNAL='' BG_RC=0 BG_LIVE=1
  for bg_s in TERM INT HUP; do
    bg_saved=$(trap -p "$bg_s")
    BG_TRAPS+="${bg_saved:-trap - $bg_s}"$'\n'
  done
  for bg_s in TERM INT HUP; do
    # shellcheck disable=SC2064 # the signal's name now, the pid when the trap runs
    trap "BG_SIGNAL=$bg_s; [ -z \"\$BG_PID\" ] || bounded_bg_stop" "$bg_s"
  done
  (
    [ -z "$BG_IN" ] || exec < "$BG_IN" || exit 70
    [ -z "$BG_OUT" ] || exec > "$BG_OUT" 2>&1 || exit 70
    [ -z "$BG_OUT_APPEND" ] || exec >> "$BG_OUT_APPEND" 2>&1 || exit 70
    LEASE_STOP_GRACE=$((bg_grace / 2)) exec timeout --kill-after="$bg_grace" "$bg_bound" "$@"
  ) <&0 &
  BG_PID=$!
  [ -z "$BG_SIGNAL" ] || bounded_bg_stop
  return 0
}

# __bounded_bg_end <command's status> <cleanup status>: the command is gone — its pid cleared before the traps come
# back (a late signal reaches no other process), its status kept, and a signal that came until the caller's traps are
# back, the one that stopped the command included, raised again for the caller's handler; a cleanup failure ends the
# runner first, which the signal would hide. The pending signal is read after the traps are back, so one that arrives
# between the pid's clearing and the restore is not lost.
__bounded_bg_end() {
  local bg_sig
  BG_PID='' BG_LIVE='' BG_RC=$1
  eval "$BG_TRAPS"
  BG_TRAPS=''
  bg_sig=$BG_SIGNAL
  if [ -n "$bg_sig" ]; then
    [ "$2" = 0 ] || exit "$2"
    kill -s "$bg_sig" $$
  fi
}

bounded_bg_stop() {
  local bg_rc=0 bg_status=0
  [ -n "$BG_LIVE" ] || return 0
  trap '' TERM INT HUP
  if [ -n "$BG_PID" ]; then
    lease_stop "$BG_PID" || bg_rc=$?
    # A stop that failed leaves a process no signal ends (uninterruptible I/O): waiting on it has no bound.
    if [ "$bg_rc" = 0 ]; then wait "$BG_PID" 2> /dev/null || bg_status=$?; fi
  fi
  __bounded_bg_end "$bg_status" "$bg_rc"
  return "$bg_rc"
}

bounded_bg_wait() {
  local bg_rc=0
  [ -n "$BG_LIVE" ] || return "$BG_RC"
  # A signal cuts a wait short and its trap stops the command: nothing is left to wait for then.
  if [ -n "$BG_PID" ]; then
    wait "$BG_PID" || bg_rc=$?
    [ -n "$BG_PID" ] || return "$BG_RC"
  fi
  __bounded_bg_end "$bg_rc" 0
  return "$bg_rc"
}

# `bash tools/gate-bound.sh --self-test`: the cancellation table of the owner, one case per way a runner has
# mishandled a bounded child (issue #313's checklist; the case's name is its row). Each case starts a runner that holds
# a child with a grandchild under the owner's foreground form (bounded) and its background form (bounded_bg), waits for
# the child's readiness signal (never a longer sleep), stops the runner with TERM, and asserts the runner's status, that
# the child and the grandchild are gone, and what the row names. The monitor is tools/ref/cancel-stubs.sh's stand-in
# for GNU timeout, which forwards the TERM or, unset, exits on it and leaves its child's group. A Mac has no timeout:
# the stand-in is the only one used, and the box's GNU timeout runs the same table through `gnu` rows when it is there.
#   named-owner        tools/check-bounded.sh: no runner keeps its own timeout (its self-test and the tree scan)
#   decoder-child      plain: TERM to a runner whose child holds a grandchild ends both (a stop that reached the
#                      runner and not the decoder leaves the decoder running)
#   monitor-children   monitor-exit: the monitor exits on the TERM and its child's group is stopped all the same
#   trap-before-child  boundary: the TERM comes between the child's fork and the pid's assignment
#   pid-cleared        reaped: the TERM comes after the pid is cleared and before the caller's traps are back, and a
#                      child slow to end (slow) is waited for, so the runner is not gone while it still shuts down
#   timeout-fork       fork-exit is the monitor that exits before it reached its command: monitor-exit; the real
#                      timeout's own case is lease.sh's `fork` form and stack-watch.sh's fork-exit rows
#   cleanup-status     cleanup-fail: a stop that fails is the runner's status (70), a signal's included
#   cleanup-alive      a stop that fails with the monitor alive returns 70 at once, with no wait on it
#   output-setup       fifo-fail: a stdin or output file that cannot be opened ends the command, not the runner, and
#                      starts nothing that outlives the call
#   output-filter      lease.sh --self-test, bounded-filter-open-term-*: the filter's guard stands before it starts
#   no-group           a child that leads no process group of its own is stopped, and no other group is signalled
#   nested-grace       two supervisors, one inside the other, finish inside the outer's budget (the inner's is half)
#   exit-handler       lease.sh --self-test, bounded-capture-exit-trap and bounded-out-clean: a runner's EXIT handler
#                      that exits still runs the capture directory's cleanup
gb_self_test() {
  local self t bad=0 s0 form mode trp tag n left p
  self=$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")
  # shellcheck source=tools/ref/cancel-stubs.sh
  source "${self%/*}/ref/cancel-stubs.sh"
  t=$(mktemp -d "${TMPDIR:-/tmp}/gate-bound-test.XXXXXX") || return 70
  # shellcheck disable=SC2064 # the path is fixed now
  trap "rm -rf '$t'" EXIT
  check() { if [ "$2" = "$3" ]; then echo "ok     $1"; else echo "FAIL   $1: got [$2], want [$3]"; bad=$((bad + 1)); fi; }
  mkdir "$t/bin"
  cancel_timeout "$t/bin/timeout"
  chmod +x "$t/bin/timeout"
  # The child: records its pid and its monitor's, leaves a grandchild, then waits in `wait` (a trapped signal ends it).
  cat > "$t/held" << 'EOF'
#!/bin/sh
case ${HELD_MODE:-term} in
  term) trap 'exit 7' TERM ;;
  zero) trap 'exit 0' TERM ;;
  slow) trap 'sleep 1; exit 7' TERM ;;
  ignore) trap '' TERM INT HUP ;;
esac
sleep 1000 &
echo $! > "$1.gc"
echo "$$ $PPID" > "$1"
while :; do wait; done
EOF
  chmod +x "$t/held"
  # The runner of a case: the owner's file $GB, the caller's trap, then the form's call; "after" marks a runner that went on.
  cat > "$t/runner" << 'EOF'
source "$GB"
[ -z "${TRAP_SH:-}" ] || eval "$TRAP_SH"
case $FORM in
  fg) bounded 60 "$HELD" "$PIDF"; echo "rc=$?" ;;
  bg) bounded_bg 60 "$HELD" "$PIDF"; bounded_bg_wait; echo "rc=$?" ;;
esac
echo after
EOF
  # alive <pid>: running, not a zombie.
  alive() { local st; st=$(ps -o stat= -p "$1" 2> /dev/null) || return 1; case $st in '' | *Z*) return 1 ;; esac; }
  # leftover <pid file>: the names of the child, its monitor and its grandchild that still run.
  leftover() {
    local f=$1 sp ip gc left=''
    if [ -s "$f" ]; then
      read -r sp ip < "$f"
      alive "$sp" && left="$left child"
      alive "$ip" && left="$left monitor"
      gc=$(cat "$f.gc" 2> /dev/null) && [ -n "$gc" ] && alive "$gc" && left="$left grandchild"
    else
      left=' no-pid-file'
    fi
    echo "${left# }"
  }
  # reap <pid file>: signal only a survivor of a failed case, after its assertions.
  reap() {
    local sp ip gc
    [ -s "$1" ] || return 0
    read -r sp ip < "$1"
    gc=$(cat "$1.gc" 2> /dev/null) || gc=''
    for p in $gc "$ip" "$sp"; do [ -z "$p" ] || ! alive "$p" || kill -KILL "$p" 2> /dev/null || :; done
  }
  # run_case <tag> <form> <held mode> <forward 1|0> <caller trap> [<owner file> [<runner file>]]: the runner on its own
  # until the child is ready, then TERM; RC, OUT and LEFT, ELAPSED.
  run_case() {
    local tag=$1 form=$2 mode=$3 fwd=$4 trp=$5 gb=${6:-$self} runner=${7:-$t/runner} i=0 rp ready=no
    rm -f "$t/$tag.pid" "$t/$tag.pid.gc"
    env GB="$gb" FORM="$form" HELD="$t/held" PIDF="$t/$tag.pid" HELD_MODE="$mode" TRAP_SH="$trp" LEASE_STOP_GRACE=4 \
      ${fwd:+CANCEL_TIMEOUT_FORWARD=$fwd} PATH="$t/bin:$PATH" bash "$runner" > "$t/$tag.out" 2>&1 &
    rp=$!
    until { cancel_ready "$t/$tag.pid" && ready=yes; } || [ "$i" = 100 ]; do sleep 0.1; i=$((i + 1)); done
    check "$tag-ready" "$ready" yes
    s0=$SECONDS
    kill -TERM "$rp" 2> /dev/null
    RC=0
    { wait "$rp" || RC=$?; } 2> /dev/null # the shell's "Terminated" notice for the killed runner
    ELAPSED=$((SECONDS - s0))
    LEFT=$(leftover "$t/$tag.pid")
    OUT=$(cat "$t/$tag.out")
  }
  for form in fg bg; do
    for trp in '' "trap 'echo trap ran; exit 143' TERM"; do
      tg=${trp:+-trap}
      # decoder-child, trap-before-child (the caller's trap stands in the signal's path): the child, its grandchild.
      tag=plain-$form$tg
      run_case "$tag" "$form" term 1 "$trp"
      check "$tag-rc" "$RC" 143
      check "$tag-left" "${LEFT:-none}" none
      check "$tag-trap" "$(grep -c '^trap ran$' <<< "$OUT")" "$([ -n "$trp" ] && echo 1 || echo 0)"
      check "$tag-no-after" "$(grep -c '^after$' <<< "$OUT")" 0
      reap "$t/$tag.pid"
    done
    # monitor-children, timeout-fork: the monitor exits on the TERM and leaves its child's group.
    tag=monitor-exit-$form
    run_case "$tag" "$form" term 0 ''
    check "$tag-rc" "$RC" 143
    check "$tag-left" "${LEFT:-none}" none
    reap "$t/$tag.pid"
    # a child that ignores TERM ends by the KILL inside the cleanup budget (4 s here, 9 at the root).
    tag=ignore-term-$form
    run_case "$tag" "$form" ignore 1 ''
    check "$tag-rc" "$RC" 143
    check "$tag-left" "${LEFT:-none}" none
    check "$tag-budget" "$([ "$ELAPSED" -le 8 ] && echo inside || echo "$ELAPSED s")" inside
    reap "$t/$tag.pid"
    # pid-cleared: a child slow to end is waited for: the runner is not gone while it still shuts down.
    tag=slow-exit-$form
    run_case "$tag" "$form" slow 1 ''
    check "$tag-rc" "$RC" 143
    check "$tag-left" "${LEFT:-none}" none
    reap "$t/$tag.pid"
  done
  # trap-before-child: the TERM falls between the fork and the pid's assignment, delivered by the owner's own file with a
  # `kill -TERM $$` put there; the child must end all the same.
  awk '$0 == "  BIN_PID=$!" || $0 == "  BG_PID=$!" { print "  kill -TERM $$" } { print }' "$self" > "$t/boundary-owner.sh"
  check boundary-injected "$(grep -c '^  kill -TERM \$\$$' "$t/boundary-owner.sh")" 2
  cat > "$t/boundary-runner" << 'EOF'
source "$GB"
case $FORM in
  fg) bounded 60 "$HELD" "$PIDF"; echo "rc=$?" ;;
  bg) bounded_bg 60 "$HELD" "$PIDF"; bounded_bg_wait; echo "rc=$?" ;;
esac
echo after
EOF
  for form in fg bg; do
    tag=boundary-$form
    rm -f "$t/$tag.pid" "$t/$tag.pid.gc"
    RC=0
    { env GB="$t/boundary-owner.sh" FORM="$form" HELD="$t/held" PIDF="$t/$tag.pid" HELD_MODE=term LEASE_STOP_GRACE=4 \
      CANCEL_TIMEOUT_FORWARD=1 PATH="$t/bin:$PATH" bash "$t/boundary-runner" > "$t/$tag.out" 2>&1 || RC=$?; } 2> /dev/null
    check "$tag-rc" "$RC" 143
    check "$tag-no-after" "$(grep -c '^after$' "$t/$tag.out")" 0
    # The child may not have written its pid yet when the runner ended: give it the time a stop takes, then read.
    n=0
    while [ ! -s "$t/$tag.pid" ] && [ "$n" -lt 20 ]; do sleep 0.1; n=$((n + 1)); done
    sleep 0.3
    left=$(leftover "$t/$tag.pid" | sed 's/^no-pid-file$/none/')
    check "$tag-left" "${left:-none}" none
    reap "$t/$tag.pid"
  done
  # pid-cleared (bg): the TERM falls after the pid is cleared and before the caller's traps are back (a DEBUG trap on
  # the owner's restore); the runner ends on the signal and starts no next command.
  cat > "$t/reaped-runner" << 'EOF'
source "$GB"
set -T
trap 'if [ "${FUNCNAME[0]:-}" = __bounded_bg_end ] && [ "$BASH_COMMAND" = "eval \"\$BG_TRAPS\"" ] && [ -z "$BG_PID" ]; then
        trap - DEBUG
        kill -s TERM $$
      fi' DEBUG
bounded_bg 60 true
bounded_bg_wait
echo after
EOF
  RC=0
  { env GB="$self" PATH="$t/bin:$PATH" CANCEL_TIMEOUT_FORWARD=1 bash "$t/reaped-runner" > "$t/reaped.out" 2>&1 || RC=$?; } 2> /dev/null
  check reaped-rc "$RC" 143
  check reaped-no-after "$(grep -c '^after$' "$t/reaped.out")" 0
  # cleanup-status: a stop that fails is the status, by an explicit stop and by a signal's (the runner ends on 70).
  cat > "$t/cleanup-runner" << 'EOF'
source "$GB"
eval "$(declare -f lease_stop | sed '1s/lease_stop/__real_lease_stop/')"
lease_stop() { __real_lease_stop "$@"; return 70; }
bounded_bg 60 "$HELD" "$PIDF"
if [ "$MODE" = explicit ]; then
  until [ -s "$PIDF" ]; do sleep 0.05; done
  rc=0
  bounded_bg_stop || rc=$?
  echo "stop rc=$rc live=[$BG_LIVE]"
  exit 0
fi
bounded_bg_wait
echo after
EOF
  for mode in explicit signal; do
    tag=cleanup-$mode
    rm -f "$t/$tag.pid" "$t/$tag.pid.gc"
    env GB="$self" MODE=$mode HELD="$t/held" PIDF="$t/$tag.pid" HELD_MODE=zero LEASE_STOP_GRACE=4 CANCEL_TIMEOUT_FORWARD=1 \
      PATH="$t/bin:$PATH" bash "$t/cleanup-runner" > "$t/$tag.out" 2>&1 &
    rp=$!
    if [ "$mode" = signal ]; then
      n=0
      until cancel_ready "$t/$tag.pid" || [ "$n" = 100 ]; do sleep 0.1; n=$((n + 1)); done
      kill -TERM "$rp" 2> /dev/null
    fi
    RC=0
    { wait "$rp" || RC=$?; } 2> /dev/null # the shell's "Terminated" notice for the killed runner
    case $mode in
      explicit) check "$tag-rc" "$RC|$(grep -c '^stop rc=70 live=\[\]$' "$t/$tag.out")" '0|1' ;;
      signal) check "$tag-rc" "$RC|$(grep -c '^after$' "$t/$tag.out")" '70|0' ;;
    esac
    left=$(leftover "$t/$tag.pid")
    check "$tag-left" "${left:-none}" none
    reap "$t/$tag.pid"
  done
  # cleanup-alive: a stop that fails with the monitor still alive returns 70 at once and never waits on it.
  cat > "$t/alive-runner" << 'EOF'
source "$GB"
lease_stop() { return 70; }
bounded_bg 60 "$HELD" "$PIDF"
until [ -s "$PIDF" ]; do sleep 0.05; done
s=$SECONDS rc=0
bounded_bg_stop || rc=$?
echo "stop rc=$rc took=$((SECONDS - s))"
EOF
  rm -f "$t/cleanup-alive.pid" "$t/cleanup-alive.pid.gc"
  RC=0
  env GB="$self" HELD="$t/held" PIDF="$t/cleanup-alive.pid" HELD_MODE=zero LEASE_STOP_GRACE=4 CANCEL_TIMEOUT_FORWARD=1 \
    PATH="$t/bin:$PATH" bash "$t/alive-runner" > "$t/cleanup-alive.out" 2>&1 || RC=$?
  check cleanup-alive-rc "$RC|$(grep -cE '^stop rc=70 took=[0-2]$' "$t/cleanup-alive.out")" '0|1'
  reap "$t/cleanup-alive.pid"
  # output-setup: a stdin or output file that cannot be opened ends the command, not the runner, and starts nothing.
  RC=0
  env HELD="$t/held" PIDF="$t/fifo.pid" PATH="$t/bin:$PATH" NOPE="$t/no-such-dir" bash -c \
    'source "$1"; BG_IN=$NOPE/go bounded_bg 5 "$HELD" "$PIDF"; bounded_bg_wait' _ "$self" > "$t/fifo.out" 2>&1 || RC=$?
  check fifo-in-rc "$([ "$RC" != 0 ] && echo nonzero || echo 0)" nonzero
  check fifo-in-nothing-started "$([ -e "$t/fifo.pid" ] && echo started || echo none)" none
  RC=0
  env HELD="$t/held" PIDF="$t/fifo2.pid" PATH="$t/bin:$PATH" NOPE="$t/no-such-dir" bash -c \
    'source "$1"; BG_OUT=$NOPE/out bounded_bg 5 "$HELD" "$PIDF"; bounded_bg_wait' _ "$self" > "$t/fifo2.out" 2>&1 || RC=$?
  check fifo-out-rc "$([ "$RC" != 0 ] && echo nonzero || echo 0)" nonzero
  check fifo-out-nothing-started "$([ -e "$t/fifo2.pid" ] && echo started || echo none)" none
  RC=0
  env HELD="$t/held" PIDF="$t/fifo3.pid" PATH="$t/bin:$PATH" NOPE="$t/no-such-dir" bash -c \
    'source "$1"; LEASE_BOUNDED_OUT=$NOPE/out bounded 5 "$HELD" "$PIDF"' _ "$self" > "$t/fifo3.out" 2>&1 || RC=$?
  check fifo-fg-rc "$([ "$RC" != 0 ] && echo nonzero || echo 0)" nonzero
  check fifo-fg-nothing-started "$([ -e "$t/fifo3.pid" ] && echo started || echo none)" none
  # no-group: a child that leads no group of its own is stopped, and a process in the runner's own group lives on.
  cat > "$t/nogroup-runner" << 'EOF'
source "$GB"
sleep 1000 &
echo $! > "$SENTINEL"
LEASE_STOP_GRACE=4 run_bounded "$HELD" "$PIDF"
echo after
EOF
  tag=nogroup
  rm -f "$t/$tag.pid" "$t/$tag.pid.gc"
  env GB="$self" HELD="$t/held" PIDF="$t/$tag.pid" SENTINEL="$t/sentinel.pid" HELD_MODE=term LEASE_STOP_GRACE=4 bash "$t/nogroup-runner" > "$t/$tag.out" 2>&1 &
  rp=$!
  n=0
  until [ -s "$t/$tag.pid" ] || [ "$n" = 100 ]; do sleep 0.1; n=$((n + 1)); done
  sleep 0.2
  kill -TERM "$rp" 2> /dev/null
  RC=0
  { wait "$rp" || RC=$?; } 2> /dev/null # the shell's "Terminated" notice for the killed runner
  p=$(cat "$t/sentinel.pid" 2> /dev/null) || p=''
  check nogroup-sentinel "$({ [ -n "$p" ] && alive "$p"; } && echo alive || echo dead)" alive
  check nogroup-child "$(leftover "$t/$tag.pid" | sed 's/ grandchild//; s/^grandchild$//')" ''
  [ -z "$p" ] || kill -KILL "$p" 2> /dev/null || :
  reap "$t/$tag.pid"
  # nested-grace: an outer and an inner bounded command, the child ignoring TERM: all gone inside the outer's budget.
  cat > "$t/nested-runner" << 'EOF'
source "$GB"
bounded 60 bash -c 'source "$1"; bounded 60 "$2" "$3"' _ "$GB" "$HELD" "$PIDF"
echo after
EOF
  tag=nested
  rm -f "$t/$tag.pid" "$t/$tag.pid.gc"
  run_case "$tag" fg ignore 1 '' "$self" "$t/nested-runner"
  check "$tag-left" "${LEFT:-none}" none
  check "$tag-budget" "$([ "$ELAPSED" -le 8 ] && echo inside || echo "$ELAPSED s")" inside
  reap "$t/$tag.pid"
  echo "gate-bound self-test: $([ "$bad" = 0 ] && echo ok || echo FAIL) ($bad failures)"
  [ "$bad" = 0 ]
}
if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "${1:-}" = --self-test ]; then
  gb_self_test
  exit $?
fi
