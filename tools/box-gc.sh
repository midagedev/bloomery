#!/usr/bin/env bash
# Orphan collector for one track's remote directory. Runs ON THE BOX — either through
# tools/box.sh (`just box-gc`) or piped over ssh stdin (`ssh box 'bash -s -- --dry-run <dir>'
# < tools/box-gc.sh`, which is how box-tracks.sh reuses it without the script being there).
#
#   bash tools/box-gc.sh [--dry-run|--check|--kill] [root]     root defaults to $PWD
#
# --dry-run lists and exits 0 (what a human runs), --check lists and exits 10 when anything
# matched (what box-tracks.sh asks before deleting a directory), --kill collects.
#
# It selects a process by `readlink /proc/<pid>/exe`, never by matching a command line.
# `pgrep -f "<dir>/target"` also matches every shell that carries that pattern in its own
# argv — the scanning shell itself, and the ssh handler that started it. Signalling what such
# a match returns has frozen a session before (a -STOP went to the Tailscale ssh child and to
# the loop's own shell). This scan reads exe links only, and self plus every ancestor up to
# pid 1 is excluded before the prefix test, so neither can be selected even in principle.
# An exe that reads "<path> (deleted)" is still a match: a rebuilt-away orphan is exactly the
# process this is here to collect.
#
# Killing is TERM, then up to five seconds of `kill -0` on the pids we were handed at start,
# then KILL. Signals go to those pids and to nothing found later: a pid discovered by a
# pattern after the fact is the class of mistake above.
set -uo pipefail
MODE=--dry-run
case "${1:-}" in
  --dry-run|--check|--kill) MODE=$1; shift ;;
  -*) echo "box-gc: unknown flag $1 (use --dry-run, --check or --kill)" >&2; exit 64 ;;
esac
ROOT=${1:-$PWD}
PREFIX="$ROOT/target"

self=$$
anc=" $self "
p=$self
while [ -n "$p" ] && [ "$p" != 0 ] && [ "$p" != 1 ]; do
  # PPid from status, not field 4 of stat: a comm containing a space or a ')' shifts every
  # field of stat, and the ancestor walk then stops at the wrong pid — which would put the
  # scanning shell's own parent back among the candidates.
  p=$(awk '/^PPid:/{print $2}' "/proc/$p/status" 2>/dev/null)
  [ -n "$p" ] || break
  anc="$anc$p "
done
echo "box-gc: mode=$MODE prefix=$PREFIX self=$self never-candidates=[${anc# }]"

pids=()
exes=()
for d in /proc/[0-9]*; do
  pid=${d#/proc/}
  case "$anc" in *" $pid "*) continue ;; esac
  exe=$(readlink "$d/exe" 2>/dev/null) || continue
  case "$exe" in
    "$PREFIX"/*) pids+=("$pid"); exes+=("$exe") ;;
  esac
done

for i in "${!pids[@]}"; do
  echo "proc ${pids[$i]} ${exes[$i]}"
done
echo "box-gc: found ${#pids[@]} process(es) under $PREFIX"

if [ "${#pids[@]}" -eq 0 ]; then
  echo gc-done
  exit 0
fi
if [ "$MODE" = --dry-run ]; then
  exit 0
fi
if [ "$MODE" = --check ]; then
  # rc 10 = "some are running": box-tracks.sh reads this to leave a live track's directory alone.
  exit 10
fi

for pid in "${pids[@]}"; do
  kill -TERM "$pid" 2>/dev/null && echo "term $pid"
done
for _ in 1 2 3 4 5; do
  alive=0
  for pid in "${pids[@]}"; do
    kill -0 "$pid" 2>/dev/null && alive=1
  done
  [ "$alive" = 1 ] || break
  sleep 1
done
for pid in "${pids[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid" 2>/dev/null && echo "kill $pid (did not exit on TERM)"
  fi
done
echo gc-done
