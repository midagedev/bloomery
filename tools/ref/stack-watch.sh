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
# second of being written (stdout and stderr merged). Exit: the command's code (143 or 137 when the watch
# ended it), 64 on a bad command line.
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

# The comm of pid $1 (the last path part: ps prints a path on some systems).
comm_of() {
  local c
  c=$(ps -o comm= -p "$1" 2> /dev/null) || return 1
  c=${c##*/}
  printf '%s\n' "${c:0:15}"
}

dump_stacks() {
  local pid=$1 t
  if command -v gdb > /dev/null; then
    timeout 120 gdb -p "$pid" -batch -ex 'thread apply all bt' 2>&1 | sed 's/^/stack-watch: /' >&2
  elif [ -d "/proc/$pid/task" ]; then
    for t in /proc/"$pid"/task/*; do
      echo "stack-watch: == ${t##*/} $(cat "$t/comm" 2> /dev/null) wchan=$(cat "$t/wchan" 2> /dev/null)" >&2
      sed 's/^/stack-watch:    /' "$t/stack" >&2 2> /dev/null
    done
  else
    echo "stack-watch: no stack source here (no gdb on PATH, no /proc/$pid/task)" >&2
  fi
}

watch() {
  local quiet=$1 comm=$2 log pid sent=0 size last_change=$SECONDS target='' rc c p
  shift 3
  log=$(mktemp "${TMPDIR:-/tmp}/stack-watch.XXXXXX") || return 70
  "$@" > "$log" 2>&1 &
  pid=$!
  # Pass new bytes of the log through; true when there were any.
  flush_log() {
    size=$(wc -c < "$log" | tr -d ' ')
    [ "$size" -gt "$sent" ] || return 1
    tail -c "+$((sent + 1))" "$log" | head -c "$((size - sent))"
    sent=$size
  }
  while kill -0 "$pid" 2> /dev/null; do
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
      echo "stack-watch: no output for ${quiet} s: the stacks of $target ($comm), then TERM" >&2
      dump_stacks "$target"
      kill -TERM "$target" 2> /dev/null
      for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$target" 2> /dev/null || break; sleep 1; done
      kill -0 "$target" 2> /dev/null && kill -KILL "$target" 2> /dev/null
      break
    fi
  done
  rc=0
  wait "$pid" || rc=$?
  flush_log || true
  rm -f "$log"
  return "$rc"
}

# The watch on stub commands: output passes through and the code is the command's; a quiet stub named
# by its comm is dumped and ended; a quiet command with no such process is left to finish. One line a
# case; exit 0 iff none failed.
self_test() {
  local self t bad=0 out rc
  self=$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")
  t=$(mktemp -d "${TMPDIR:-/tmp}/stack-watch-test.XXXXXX") || return 70
  # shellcheck disable=SC2064 # the path is fixed now
  trap "rm -rf '$t'" EXIT
  case_() { # name, want rc, want pattern, rc, file
    if [ "$4" = "$2" ] && grep -Eq -- "$3" "$5"; then echo "ok     $1"; else
      echo "FAIL   $1 (rc $4, want $2; output:)"; sed 's/^/         /' "$5"; bad=$((bad + 1))
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
