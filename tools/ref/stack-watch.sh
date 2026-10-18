#!/usr/bin/env bash
# Stack watch: run a command, pass its output through, and when the output stops growing for QUIET
# seconds, dump the thread stacks of the process named COMM among the command's descendants, then end
# that process (TERM, KILL ten seconds later) — so a hang leaves the place it hangs in its log instead
# of dying silent at a runner's bound.
#
#   bash tools/ref/stack-watch.sh QUIET COMM -- CMD [ARGS...]
#   bash tools/ref/stack-watch.sh --self-test
#
# Blocks: a gate or a run that stops inside a call with no deadline (a driver enqueue behind a stream
# wait nothing releases, 2026-09-29) and is killed by the runner's bound with no trace of where.
# The watched process is found only among the descendants of the pid this script spawned, by its comm
# (never by a pattern over every process), and is the only one signalled. Stacks: `gdb -batch -ex
# 'thread apply all bt'` when gdb is on PATH, else every thread's /proc/<pid>/task/*/{comm,wchan,stack},
# else a line saying neither is there. The dump goes to stderr, each line prefixed `stack-watch:`.
# QUIET counts from the last byte the command wrote; the command's own output reaches stdout within a
# second of being written (stdout and stderr merged). A TERM or HUP to the watch goes on to the command and
# to a dump under way as a TERM; so does an INT when the watch runs in the foreground (bash starts a command
# run with & with INT ignored). Sourcing this script runs the watch in the caller's shell; the caller
# restores its signal traps afterward. Exit: the command's code (143 or 137 when the watch ended it),
# 64 on a bad command line.
set -uo pipefail
# shellcheck source=tools/gate-bound.sh
source "${BASH_SOURCE[0]%/*}/../gate-bound.sh" || exit 70

usage() {
  echo "usage: stack-watch.sh QUIET COMM -- CMD [ARGS...]  (QUIET whole seconds from 1)" >&2
  exit 64
}

# Every descendant of $1 (itself first), from a `pid ppid comm` table read once.
descendants() {
  local root=$1 table
  table=$(ps -A -o pid= -o ppid= -o comm=) || return 1
  awk -v root="$root" '
    { pid[NR] = $1; ppid[NR] = $2; n = NR }
    END {
      want[root] = 1; print root
      do {
        grew = 0
        for (i = 1; i <= n; i++) if (want[ppid[i]] && !want[pid[i]]) { want[pid[i]] = 1; print pid[i]; grew = 1 }
      } while (grew)
    }' <<< "$table"
}

# True while pid $1 runs: one that has ended and is not yet reaped (a zombie, state Z) has ended, and
# its parent may never reap it.
alive() {
  local s
  if ! s=$(ps -o stat= -p "$1" 2> /dev/null); then
    kill -0 "$1" 2> /dev/null || return 1
    printf 'stack-watch: cannot read state of existing pid %s\n' "$1" >&2
    return 0
  fi
  s=${s#"${s%%[! ]*}"}
  [ -n "$s" ] && [ "${s:0:1}" != Z ]
}

# The comm of pid $1 (the last path part: ps prints a path on some systems).
comm_of() {
  local c
  c=$(ps -o comm= -p "$1" 2> /dev/null) || return 1
  c=${c##*/}
  printf '%s\n' "${c:0:15}"
}

# dump_stacks <pid>: a gdb dump runs in the background as the watch's $dump and is waited for, so a TERM
# to the watch is heard during it and the dump ends before the watch goes on; a dump started as the TERM
# came gets a TERM too, unless the trap sent it one already (the watch's trap sets $stop and $told and
# reads $pid, $dump and $spawning, so none is a local here). The dump is bounded at 120 s by a TERM and
# 10 s later by a KILL. Its output goes to a file beside the watch's log and then to stderr, so no
# child of the dump outlives it.
dump_stacks() {
  local target=$1 t
  if command -v gdb > /dev/null; then
    run_bounded timeout --kill-after=10 120 gdb -p "$target" -batch -ex 'thread apply all bt' > "$log.gdb" 2>&1 &
    dump=$!
    spawning=0
    if [ "$stop" = 1 ]; then
      trap '' TERM INT HUP
      [ "$told" = "$pid $dump" ] || kill -TERM "$dump" 2> /dev/null || :
    fi
    wait "$dump" || true
    wait "$dump" || true
    dump=''
    sed 's/^/stack-watch: /' "$log.gdb" >&2
    rm -f "$log.gdb"
  elif [ -d "/proc/$target/task" ]; then
    for t in /proc/"$target"/task/*; do
      echo "stack-watch: == ${t##*/} $(cat "$t/comm" 2> /dev/null) wchan=$(cat "$t/wchan" 2> /dev/null)" >&2
      sed 's/^/stack-watch:    /' "$t/stack" >&2 2> /dev/null
    done
  else
    echo "stack-watch: no stack source here (no gdb on PATH, no /proc/$target/task)" >&2
  fi
}

watch() {
  local quiet=$1 comm=$2 log pid='' sent=0 size last_change=$SECONDS target='' rc c p dump='' stop=0 spawning=0 told=''
  shift 3
  log=$(mktemp "${TMPDIR:-/tmp}/stack-watch.XXXXXX") || return 70
  # A command under timeout runs in a process group of its own, which a signal to the watch's group never reaches: a TERM or HUP
  # to the watch (an INT too, in the foreground) goes on to the command and to a dump under way, and the watch goes on only after
  # each has ended; after it the watch starts no dump and waits for the command. The trap is set before the command starts, so no
  # signal falls between the two: one that comes before the command's pid is known sets $stop, and the command gets its TERM as
  # soon as the pid is, unless the trap sent it one already ($told names the pids it sent to). The signal cuts short the wait it
  # arrives in; once it has a pid to pass it to, the trap ignores further signals, so each wait is made twice and the second reads
  # the child's status. It ignores none while a child may still start ($spawning: before the command's pid is known, and from the
  # check before a dump until the dump's pid is), which would start that child with them ignored. A TERM to a child that has
  # already ended is no error, under set -e too.
  trap 'stop=1; [ -z "$pid" ] || { [ "$spawning" = 1 ] || trap "" TERM INT HUP; told="$pid $dump"; kill -TERM "$pid" ${dump:+"$dump"} 2> /dev/null || :; }' TERM INT HUP
  (set -m; run_bounded "$@") > "$log" 2>&1 &
  pid=$!
  if [ "$stop" = 1 ]; then
    trap '' TERM INT HUP
    [ -n "$told" ] || kill -TERM "$pid" 2> /dev/null || :
  fi
  # Pass new bytes of the log through; true when there were any.
  flush_log() {
    size=$(wc -c < "$log" | tr -d ' ')
    [ "$size" -gt "$sent" ] || return 1
    tail -c "+$((sent + 1))" "$log" | head -c "$((size - sent))"
    sent=$size
  }
  while [ "$stop" = 0 ] && kill -0 "$pid" 2> /dev/null; do
    sleep 1
    flush_log && last_change=$SECONDS
    if [ $((SECONDS - last_change)) -ge "$quiet" ]; then
      target=''
      for p in $(descendants "$pid"); do
        c=$(comm_of "$p") || continue
        [ "$c" = "${comm:0:15}" ] && { target=$p; break; }
      done
      if [ -z "$target" ]; then
        echo "stack-watch: no output for ${quiet} s and no process named $comm under $pid: nothing dumped" >&2
        last_change=$SECONDS
        continue
      fi
      spawning=1
      if [ "$stop" = 1 ]; then
        spawning=0
        break
      fi
      echo "stack-watch: no output for ${quiet} s: the stacks of $target ($comm), then TERM" >&2
      dump_stacks "$target"
      spawning=0
      kill -TERM "$target" 2> /dev/null || :
      for _ in 1 2 3 4 5 6 7 8 9 10; do alive "$target" || break; sleep 1; done
      if alive "$target"; then kill -KILL "$target" 2> /dev/null || :; fi
      break
    fi
  done
  rc=0
  wait "$pid" || true
  wait "$pid" || rc=$?
  pid=''
  flush_log || true
  rm -f "$log"
  return "$rc"
}

# The watch on stub commands: output passes through and the code is the command's; a quiet stub named
# by its comm is dumped and ended; a quiet command with no such process is left to finish; a TERM during
# a dump or during the last wait ends the dump and the command, and the watch returns after both, with the
# command's code; after a TERM before the quiet time no dump starts; a TERM before or after the watch
# reads the command's PID ends it.
# One line a case, a failed one with the last line of what it read; exit 0 iff none failed.
self_test() {
  local self t bad=0 out rc w k s left phase quiet ready before lk f sp ip watcher child
  self=$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")
  # shellcheck source=tools/ref/cancel-stubs.sh
  source "${self%/*}/cancel-stubs.sh"
  t=$(mktemp -d "${TMPDIR:-/tmp}/stack-watch-test.XXXXXX") || return 70
  # shellcheck disable=SC2064 # the path is fixed now
  trap "rm -rf '$t'" EXIT
  case_() { # name, want rc, want pattern, rc, file
    if [ "$4" = "$2" ] && grep -Eq -- "$3" "$5"; then echo "ok     $1"; else
      echo "FAIL   $1 (rc $4, want $2; last line: $(tail -n 1 "$5"); output:)"; sed 's/^/         /' "$5"; bad=$((bad + 1))
    fi
  }
  mkdir "$t/ps-bin" || return 70
  printf '%s\n' '#!/bin/sh' 'exit 2' > "$t/ps-bin/ps" || return 70
  chmod +x "$t/ps-bin/ps" || return 70
  rc=0; PATH=$t/ps-bin:$PATH alive "$$" > "$t/ps-failure" 2>&1 || rc=$?
  case_ 'a failed state read of an existing pid: alive, with a diagnostic' 0 \
    '^stack-watch: cannot read state of existing pid [0-9]+$' "$rc" "$t/ps-failure"
  rc=0; bash "$self" 3 sleep -- sh -c 'echo one; echo two >&2; exit 3' > "$t/a" 2>&1 || rc=$?
  case_ 'a command that ends: its output through, its code' 3 '^two$' "$rc" "$t/a"
  rc=0; bash "$self" 2 sleep -- sh -c 'echo start; exec sleep 30' > "$t/b" 2>&1 || rc=$?
  case_ 'a quiet command: its sleep dumped and ended, TERM' 143 'no output for 2 s: the stacks of [0-9]+ \(sleep\), then TERM' "$rc" "$t/b"
  case_ '  … its output before the stop kept' 143 '^start$' "$rc" "$t/b"
  rc=0; bash "$self" 1 nosuchcomm -- sh -c 'sleep 3; echo late' > "$t/c" 2>&1 || rc=$?
  case_ 'a quiet command with no process of that comm: named, left to finish' 0 'no process named nosuchcomm' "$rc" "$t/c"
  case_ '  … and its later output kept' 0 '^late$' "$rc" "$t/c"
  rc=0; bash "$self" x sleep -- true > "$t/d" 2>&1 || rc=$?
  case_ 'QUIET not a whole number: 64' 64 'usage' "$rc" "$t/d"
  # A TERM during a dump (a gdb that holds, and ends six seconds after its TERM) ends the dump and the command, and a second
  # TERM while the watch waits does not cut that wait short: the watch returns after both, with the command's code (7, two
  # seconds after its TERM).
  mkdir "$t/bin"
  printf '%s\n' '#!/bin/sh' "[ -z \"\${SW_HOLD:-}\" ] || trap 'sleep 6; exit 0' TERM" 'echo $$ > "$SW_GDB"' \
    '[ -n "${SW_HOLD:-}" ] || exit 0' 'while :; do sleep 1; done' > "$t/bin/gdb"
  printf '%s\n' 'echo $$ > "$SW_CMD"' "trap 'sleep 2; exit 7' TERM" 'sleep 30 &' 'wait' > "$t/cmd"
  chmod +x "$t/bin/gdb"
  command -v timeout > /dev/null || cp "${self%/*}/stub-timeout.py" "$t/bin/timeout"
  SW_HOLD=1 SW_GDB=$t/gdb.pid SW_CMD=$t/cmd.pid PATH=$t/bin:$PATH bash "$self" 1 sleep -- sh "$t/cmd" > "$t/e" 2>&1 &
  w=$! k=0
  until [ -s "$t/gdb.pid" ] || [ "$k" = 50 ]; do sleep 0.2; k=$((k + 1)); done
  s=$SECONDS
  kill -TERM "$w"
  sleep 1.5
  kill -TERM "$w"
  rc=0; wait "$w" || rc=$?
  left=
  alive "$(cat "$t/gdb.pid" 2> /dev/null)" && left="$left gdb"
  alive "$(cat "$t/cmd.pid")" && left="$left command"
  echo "dumping: $([ -s "$t/gdb.pid" ] && echo yes || echo no), ended in $((SECONDS - s)) s, left:${left:- none}" >> "$t/e"
  # PIN(2026-10-15): was [0-7] s: the stub gdb ends six seconds after its TERM, so the watch returns about 6 s after the TERM;
  # a gdb the TERM does not reach holds until the dump's bound (a TERM at 120 s), far past 19 s.
  case_ 'a TERM during the dump ends it and the command, their code kept' 7 'dumping: yes, ended in 1?[0-9] s, left: none$' "$rc" "$t/e"
  # A TERM while the watch waits for the command, after it has ended the command's quiet process (the command goes on after
  # it), ends the command, and the watch returns after it, with its code.
  printf '%s\n' 'echo $$ > "$SW_CMD"' "trap 'sleep 2; exit 7' TERM" 'sleep 30 &' 'wait' 'echo > "$SW_AFTER"' \
    'while :; do sleep 1; done' > "$t/cmd2"
  SW_GDB=$t/gdb2.pid SW_CMD=$t/cmd2.pid SW_AFTER=$t/after PATH=$t/bin:$PATH bash "$self" 1 sleep -- sh "$t/cmd2" > "$t/f" 2>&1 &
  w=$! k=0
  until [ -e "$t/after" ] || [ "$k" = 50 ]; do sleep 0.2; k=$((k + 1)); done
  sleep 2
  kill -TERM "$w"
  rc=0; wait "$w" || rc=$?
  left=
  alive "$(cat "$t/cmd2.pid")" && left=' command'
  echo "quiet process ended: $([ -e "$t/after" ] && echo yes || echo no), left:${left:- none}" >> "$t/f"
  case_ 'a TERM during the last wait ends the command, its code kept' 7 'quiet process ended: yes, left: none$' "$rc" "$t/f"
  # A TERM before the quiet time, to a command that prints every 0.2 s until its TERM and then takes five seconds to end (longer
  # than the quiet time): the watch starts no dump after the TERM, and returns when the command ends, with its code. The printing,
  # not the TERM's timing, keeps the quiet time from running out before the TERM.
  # PIN(2026-10-18): the wall band is 4 to 9 s. The command's trap sleeps 5 s, so no load ends the watch before 5 s, and a watch that went on
  # to the quiet path would end the trap's sleep early, under 4 s. The watch returns one 1 s poll after the command ends, 7 s at
  # most unloaded; 8 s was read on a loaded Mac with the command's own 5 s unchanged, so the ceiling takes a 2 s margin. A late return
  # past 9 s is the only mode it still tells apart (a dump is `dumped: yes`, an early return is under the floor).
  printf '%s\n' "trap 'sleep 5; exit 7' TERM" 'echo $$ > "$SW_CMD"' 'while :; do echo tick; sleep 0.2; done' > "$t/cmd3"
  SW_GDB=$t/gdb3.pid SW_CMD=$t/cmd3.pid PATH=$t/bin:$PATH bash "$self" 2 sleep -- sh "$t/cmd3" > "$t/g" 2>&1 &
  w=$! k=0
  until [ -s "$t/cmd3.pid" ] || [ "$k" = 50 ]; do sleep 0.2; k=$((k + 1)); done
  sleep 1
  s=$SECONDS
  kill -TERM "$w"
  rc=0; wait "$w" || rc=$?
  left=
  alive "$(cat "$t/cmd3.pid")" && left=' command'
  echo "dumped: $([ -e "$t/gdb3.pid" ] && echo yes || echo no), ended in $((SECONDS - s)) s, left:${left:- none}" >> "$t/g"
  case_ 'a TERM before the quiet time: no dump after it, the command awaited, its code kept' 7 \
    'dumped: no, ended in [4-9] s, left: none$' "$rc" "$t/g"
  # The startup copy holds the PID-assignment window open until TERM has been sent.
  mkdir -p "$t/startup/ref" || return 70
  cp "${self%/*}/../gate-bound.sh" "$t/startup/gate-bound.sh" || return 70
  awk '
    { print }
    $0 == "  (set -m; run_bounded \"$@\") > \"$log\" 2>&1 &" {
      gaps++
      print "  echo gap > \"$START_READY\""
      print "  while [ ! -e \"$START_RELEASE\" ]; do sleep 0.01 9>&-; done"
      print "  echo \"startup stop=$stop pid=${pid:-unset}\""
    }
    END { if (gaps != 1) exit 1 }
  ' "$self" > "$t/startup/ref/stack-watch.sh" || return 70
  # Without TERM the stub ends with 0, so a missing forward fails the expected status rather than hanging.
  printf '%s\n' "trap 'exit 7' TERM" 'sleep 30 &' 'echo $! > "$SW_CHILD"' 'echo "$$ $PPID" > "$SW_CMD"' \
    'i=0; while [ "$i" -lt 20 ]; do sleep 1; i=$((i + 1)); done' > "$t/cmd4" || return 70
  for phase in started startup; do
    watcher=$self
    [ "$phase" != startup ] || watcher=$t/startup/ref/stack-watch.sh
    SW_CMD=$t/$phase.pid SW_CHILD=$t/$phase.child START_READY=$t/gap START_RELEASE=$t/go \
      bash "$watcher" 30 sleep -- sh "$t/cmd4" > "$t/h-$phase" 2>&1 &
    w=$! k=0 ready=no
    until [ "$k" = 100 ]; do
      if cancel_ready "$t/$phase.pid" && [ -s "$t/$phase.child" ] && { [ "$phase" != startup ] || [ -s "$t/gap" ]; }; then
        ready=yes
        break
      fi
      sleep 0.1
      k=$((k + 1))
    done
    kill -TERM "$w" 2> /dev/null || { echo 'TERM delivery failed' >&2; bad=$((bad + 1)); }
    [ "$phase" != startup ] || touch "$t/go" || return 70
    rc=0; wait "$w" || rc=$?
    left=''
    if [ -s "$t/$phase.pid" ]; then
      read -r sp ip < "$t/$phase.pid" || return 70
      alive "$sp" && left=' command'
    fi
    echo "command ready: $ready, left:${left:- none}" >> "$t/h-$phase"
    [ -z "$left" ] || kill -TERM "$sp" 2> /dev/null || return 70
    case_ "a TERM with the watch $phase: the command ended, its code kept" 7 \
      '^command ready: yes, left: none$' "$rc" "$t/h-$phase"
    left=missing
    if [ -s "$t/$phase.child" ]; then
      read -r child < "$t/$phase.child" || return 70
      if alive "$child"; then left=alive; else left=ended; fi
    fi
    echo "background descendant: $left" >> "$t/h-$phase"
    case_ "a TERM with the watch $phase: its background descendant ended" 7 \
      '^background descendant: ended$' "$rc" "$t/h-$phase"
    [ "$left" != alive ] || kill -TERM "$child" 2> /dev/null || return 70
    if [ "$phase" = startup ]; then
      case_ '  TERM recorded before PID assignment' 7 '^startup stop=1 pid=unset$' "$rc" "$t/h-$phase"
    fi
  done
  # A timeout that exits on TERM without forwarding it leaves its child holding descriptor 9 in
  # its own group. Cancelling the watch waits for that group, for both the command and a gdb dump.
  command -v flock > /dev/null || PATH=${self%/*}/card-tests/bin:$PATH
  mkdir "$t/fork-bin"
  cancel_timeout "$t/fork-bin/timeout"
  printf '%s\n' '#!/bin/sh' 'trap '\''trap "" TERM INT HUP; sleep 1; exit 7'\'' TERM' \
    'echo "$$ $PPID" > "$1"' 'while :; do sleep 30; done' > "$t/fork-held"
  printf '%s\n' '#!/bin/sh' 'exec "$SW_HELD" "$SW_DUMP"' > "$t/fork-bin/gdb"
  chmod +x "$t/fork-bin/timeout" "$t/fork-bin/gdb" "$t/fork-held"
  for phase in command dump; do
    quiet=30
    [ "$phase" != dump ] || quiet=1
    SW_HELD=$t/fork-held SW_DUMP=$t/fork-$phase.gdb PATH=$t/fork-bin:$PATH \
      bash -c 'exec 9> "$1" && flock -n 9 && exec bash "$2" "$3" sleep -- timeout --kill-after=10 30 "$4" "$5"' \
      _ "$t/fork-$phase.lock" "$self" "$quiet" "$t/fork-held" "$t/fork-$phase.cmd" > "$t/fork-$phase.log" 2>&1 &
    w=$! k=0 ready=no
    until [ "$k" = 100 ]; do
      if cancel_ready "$t/fork-$phase.cmd" && { [ "$phase" != dump ] || cancel_ready "$t/fork-$phase.gdb"; }; then
        ready=yes
        break
      fi
      sleep 0.1
      k=$((k + 1))
    done
    before=free
    flock -n "$t/fork-$phase.lock" true 2> /dev/null || before=held
    kill -TERM "$w" 2> /dev/null || :
    rc=0; wait "$w" || rc=$?
    left=''
    for f in "$t/fork-$phase.cmd" "$t/fork-$phase.gdb"; do
      [ -s "$f" ] && read -r sp ip < "$f" || continue
      alive "$sp" && left="$left child"
      alive "$ip" && left="$left timeout"
    done
    lk=free
    flock -n "$t/fork-$phase.lock" true 2> /dev/null || lk=held
    echo "ready: $ready, lock before: $before" >> "$t/fork-$phase.log"
    echo "left:${left:- none}, lock after: $lk" >> "$t/fork-$phase.log"
    case_ "a TERM with a fork-exit timeout ($phase): child ended, inherited lock free" 143 \
      '^left: none, lock after: free$' "$rc" "$t/fork-$phase.log"
    case_ "  $phase and its timeout ready with the lock held" 143 '^ready: yes, lock before: held$' "$rc" "$t/fork-$phase.log"
    for f in "$t/fork-$phase.cmd" "$t/fork-$phase.gdb"; do
      [ -s "$f" ] && read -r sp ip < "$f" || continue
      if alive "$sp" || kill -0 "$ip" 2> /dev/null; then lease_stop "$ip" || :; fi
    done
  done
  [ "$bad" = 0 ] && echo "stack-watch self-test: ok" || echo "stack-watch self-test: $bad failed"
  [ "$bad" = 0 ]
}

if [ "${1:-}" = --self-test ]; then
  self_test
  exit
fi
[ $# -ge 4 ] && [ "$3" = -- ] || usage
case "$1" in '' | *[!0-9]* | 0) usage ;; esac
watch "$@"
