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
# run with & with INT ignored, and tools/gpu-gate.sh, which starts the watch so, sends it a TERM for its own
# INT). Exit: the command's code (143 or 137 when the watch ended it), 64 on a bad command line.
set -uo pipefail

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
  s=$(ps -o stat= -p "$1" 2> /dev/null) || return 1
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
# came gets a TERM too (the watch's trap sets $stop and reads $pid and $dump, so none is a local here).
dump_stacks() {
  local target=$1 t
  if command -v gdb > /dev/null; then
    timeout 120 gdb -p "$target" -batch -ex 'thread apply all bt' > >(sed 's/^/stack-watch: /' >&2) 2>&1 &
    dump=$!
    [ "$stop" = 0 ] || kill -TERM "$dump" 2> /dev/null
    wait "$dump" || true
    wait "$dump" || true
    dump=''
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
  local quiet=$1 comm=$2 log pid='' sent=0 size last_change=$SECONDS target='' rc c p dump='' stop=0
  shift 3
  log=$(mktemp "${TMPDIR:-/tmp}/stack-watch.XXXXXX") || return 70
  # A command under timeout runs in a process group of its own, which a signal to the watch's group never reaches: a TERM or HUP
  # to the watch (an INT too, in the foreground) goes on to the command and to a dump under way, and the watch goes on only after
  # each has ended; after it the watch starts no dump and waits for the command. The trap is set before the command starts, so no
  # signal falls between the two: one that comes before the command's pid is known sets $stop, and the command gets its TERM as
  # soon as the pid is. The signal cuts short the wait it arrives in; once it has a pid to pass it to, the trap ignores further
  # signals (none before the command starts, which would start it with them ignored), so each wait is made twice and the second
  # reads the child's status.
  trap 'stop=1; [ -z "$pid" ] || { trap "" TERM INT HUP; kill -TERM "$pid" ${dump:+"$dump"} 2> /dev/null; }' TERM INT HUP
  "$@" > "$log" 2>&1 &
  pid=$!
  if [ "$stop" = 1 ]; then
    trap '' TERM INT HUP
    kill -TERM "$pid" 2> /dev/null
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
      [ "$stop" = 0 ] || break
      echo "stack-watch: no output for ${quiet} s: the stacks of $target ($comm), then TERM" >&2
      dump_stacks "$target"
      kill -TERM "$target" 2> /dev/null
      for _ in 1 2 3 4 5 6 7 8 9 10; do alive "$target" || break; sleep 1; done
      alive "$target" && kill -KILL "$target" 2> /dev/null
      break
    fi
  done
  rc=0
  wait "$pid" || true
  wait "$pid" || rc=$?
  flush_log || true
  rm -f "$log"
  return "$rc"
}

# The watch on stub commands: output passes through and the code is the command's; a quiet stub named
# by its comm is dumped and ended; a quiet command with no such process is left to finish; a TERM during
# a dump or during the last wait ends the dump and the command, and the watch returns after both, with the
# command's code; after a TERM before the quiet time no dump starts; a TERM as the command starts ends it.
# One line a case, a failed one with the last line of what it read; exit 0 iff none failed.
self_test() {
  local self t bad=0 out rc w k s left
  self=$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")
  t=$(mktemp -d "${TMPDIR:-/tmp}/stack-watch-test.XXXXXX") || return 70
  # shellcheck disable=SC2064 # the path is fixed now
  trap "rm -rf '$t'" EXIT
  case_() { # name, want rc, want pattern, rc, file
    if [ "$4" = "$2" ] && grep -Eq -- "$3" "$5"; then echo "ok     $1"; else
      echo "FAIL   $1 (rc $4, want $2; last line: $(tail -n 1 "$5"); output:)"; sed 's/^/         /' "$5"; bad=$((bad + 1))
    fi
  }
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
  printf '%s\n' '#!/bin/sh' 'echo $$ > "$SW_GDB"' '[ -n "${SW_HOLD:-}" ] || exit 0' "trap 'sleep 6; exit 0' TERM" \
    'while :; do sleep 1; done' > "$t/bin/gdb"
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
  kill -0 "$(cat "$t/gdb.pid" 2> /dev/null)" 2> /dev/null && left="$left gdb"
  kill -0 "$(cat "$t/cmd.pid")" 2> /dev/null && left="$left command"
  echo "dumping: $([ -s "$t/gdb.pid" ] && echo yes || echo no), ended in $((SECONDS - s)) s, left:${left:- none}" >> "$t/e"
  # PIN(2026-10-15): was [0-7] s: the stub gdb ends six seconds after its TERM, so the watch returns about 6 s after the TERM;
  # a gdb the TERM does not reach holds until the dump's 120 s bound, far past 19 s.
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
  kill -0 "$(cat "$t/cmd2.pid")" 2> /dev/null && left=' command'
  echo "quiet process ended: $([ -e "$t/after" ] && echo yes || echo no), left:${left:- none}" >> "$t/f"
  case_ 'a TERM during the last wait ends the command, its code kept' 7 'quiet process ended: yes, left: none$' "$rc" "$t/f"
  # A TERM before the quiet time, to a command that takes five seconds to end after its TERM (longer than the quiet time):
  # the watch starts no dump after the TERM, and returns when the command ends, with its code.
  printf '%s\n' 'echo $$ > "$SW_CMD"' "trap 'sleep 5; exit 7' TERM" 'echo start' 'sleep 30 &' 'wait' > "$t/cmd3"
  SW_GDB=$t/gdb3.pid SW_CMD=$t/cmd3.pid PATH=$t/bin:$PATH bash "$self" 2 sleep -- sh "$t/cmd3" > "$t/g" 2>&1 &
  w=$! k=0
  until [ -s "$t/cmd3.pid" ] || [ "$k" = 50 ]; do sleep 0.2; k=$((k + 1)); done
  sleep 0.5
  s=$SECONDS
  kill -TERM "$w"
  rc=0; wait "$w" || rc=$?
  left=
  kill -0 "$(cat "$t/cmd3.pid")" 2> /dev/null && left=' command'
  echo "dumped: $([ -e "$t/gdb3.pid" ] && echo yes || echo no), ended in $((SECONDS - s)) s, left:${left:- none}" >> "$t/g"
  case_ 'a TERM before the quiet time: no dump after it, the command awaited, its code kept' 7 \
    'dumped: no, ended in [4-7] s, left: none$' "$rc" "$t/g"
  # A TERM as soon as the command runs (it sets its trap, then writes its pid): the command gets it, and the watch returns after
  # the command, with its code. A command still there once the checks are made is let go.
  printf '%s\n' "trap 'exit 7' TERM" 'echo $$ > "$SW_CMD"' 'while :; do sleep 1; done' > "$t/cmd4"
  SW_CMD=$t/cmd4.pid bash "$self" 30 sleep -- sh "$t/cmd4" > "$t/h" 2>&1 &
  w=$! k=0
  until [ -s "$t/cmd4.pid" ] || [ "$k" = 100 ]; do sleep 0.1; k=$((k + 1)); done
  kill -TERM "$w"
  rc=0; wait "$w" || rc=$?
  left=
  kill -0 "$(cat "$t/cmd4.pid" 2> /dev/null)" 2> /dev/null && left=' command'
  echo "command ran: $([ -s "$t/cmd4.pid" ] && echo yes || echo no), left:${left:- none}" >> "$t/h"
  [ -z "$left" ] || kill -TERM "$(cat "$t/cmd4.pid")" 2> /dev/null
  case_ 'a TERM as the command starts: the command ended, its code kept' 7 'command ran: yes, left: none$' "$rc" "$t/h"
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
