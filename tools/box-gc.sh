#!/usr/bin/env bash
# Orphan collector for one track's remote directory. Runs ON THE BOX, piped over ssh stdin, so the
# script need not be there: `just box-gc` sends it through tools/box.sh's read-only path (which syncs
# nothing and runs in the remote directory as it is), box-tracks.sh over plain ssh (`ssh box 'bash -s
# -- --dry-run <dir>' < tools/box-gc.sh`).
#
#   bash tools/box-gc.sh [--dry-run|--check|--kill|--self-test] [root]     root defaults to $PWD
#
# --dry-run lists and exits 0 (what a human runs), --check lists and exits 10 when anything
# matched (what box-tracks.sh asks before deleting a directory), --kill collects. --check also
# counts a process whose cwd is under root: a build in progress runs cargo and rustc from the
# toolchain, not from target/, so the exe test alone calls that directory idle. --kill never
# selects by cwd — a build is not an orphan.
# --self-test runs the selection on a fake /proc tree (a copy of this script whose PROC line below
# points at it), one line per case; no signal is sent (--dry-run and --check only — --kill signals
# the pids those two print, over the same scan).
#
# It selects a process by `readlink /proc/<pid>/exe`, never by a pattern over the process table.
# `pgrep -f "<dir>/target"` also matches every shell that carries that pattern in its own
# argv — the scanning shell itself, and the ssh handler that started it. Signalling what such
# a match returns has frozen a session before (a -STOP went to the Tailscale ssh child and to
# the loop's own shell). This scan reads exe links only — one exception, the gpu-gate.sh waiter
# below, whose command line is read of a process the scan has already reached, never matched
# against the table — and self plus every ancestor up to pid 1 is excluded before the prefix
# test, so neither can be selected even in principle.
# An exe that reads "<path> (deleted)" is still a match: a rebuilt-away orphan is exactly the
# process this is here to collect.
#
# A `bash tools/gpu-gate.sh` waiting on a card lock or the batch hold — a round that stopped its
# batch leaves one on the box, and it starts its gate the moment the hold drops — has exe bash,
# so no exe test can see it. It counts as this track's when its exe is bash, its command line
# runs tools/gpu-gate.sh and its cwd is inside root.
#
# The reference harnesses (dump_ref, argmax_ref, the *_ref and *_rate kernels, router_trace) are
# built outside every track's target/, into the shared $BLOOMERY_DATA/bin and
# $BLOOMERY_DATA/router/bin, so their exe alone says nothing about whose they are. A runner starts
# them from its track's root (tools/box.sh cd's there), so a harness counts as this track's when its
# exe is in one of those two directories AND its cwd is inside root: another track's harness, or
# the lead's, has its cwd in that track's own directory and is never selected. Without
# BLOOMERY_DATA in the environment (box-tracks.sh pipes this script over plain ssh) only target/
# is scanned; --check still sees such a harness through its cwd.
#
# Killing is TERM, then up to five seconds of `kill -0` on the pids we were handed at start,
# then KILL. Signals go to those pids and to nothing found later: a pid discovered by a
# pattern after the fact is the class of mistake above.
set -uo pipefail

# --self-test (the header): the selection rules on a fake /proc tree, no box, no signal.
self_test() {
  local self t n=0 bad=0 out rc
  self=$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")
  t=$(mktemp -d "${TMPDIR:-/tmp}/box-gc-test.XXXXXX") || { echo "box-gc: self-test: no temporary directory" >&2; return 70; }
  # shellcheck disable=SC2064 # the path is fixed now
  trap "rm -rf '$t'" EXIT
  # The copy under test: its PROC line points at the fake tree (the sed target is that one line; if
  # it stops matching, the guard below ends the test rather than scanning a real /proc).
  sed "s|^PROC=/proc$|PROC=$t/proc|" "$self" > "$t/box-gc.sh"
  grep -q "^PROC=$t/proc\$" "$t/box-gc.sh" || { echo "box-gc: self-test: the copy did not patch (PROC's line moved?)" >&2; return 70; }
  export BLOOMERY_DATA=$t/data # the fake harness dirs, not the caller's
  mkdir -p "$t/root/target/release" "$t/other" "$t/proc" "$t/data/bin"
  : > "$t/root/target/release/fakegate"
  : > "$t/data/bin/dump_ref"
  # fake <pid> <exe> <cwd> <argv…>: one process of the fake tree. The pids sit far above any live
  # range — nothing here is ever signalled (the header's modes).
  fake() {
    local pid=$1 exe=$2 cwd=$3
    shift 3
    mkdir -p "$t/proc/$pid"
    ln -s "$exe" "$t/proc/$pid/exe"
    ln -s "$cwd" "$t/proc/$pid/cwd"
    printf '%s\0' "$@" > "$t/proc/$pid/cmdline"
  }
  fake 4194301 "$t/root/target/release/fakegate" "$t/root" ./target/release/fakegate
  fake 4194302 /bin/bash "$t/root" bash tools/gpu-gate.sh gen_x        # the waiter this fix is for
  fake 4194303 /bin/bash "$t/other" bash tools/gpu-gate.sh gen_x       # another track's waiter
  fake 4194304 /bin/bash "$t/root" bash -c 'cargo oxide build --bin gate_x' # this track's build shell
  fake 4194305 /usr/bin/grep "$t/root" grep -rl tools/gpu-gate.sh .    # a non-shell whose argv names the script
  fake 4194306 "$t/data/bin/dump_ref" "$t/root" "$t/data/bin/dump_ref" --dump # a harness runner of this track
  pass() { n=$((n + 1)); echo "ok $1"; }
  fail() {
    n=$((n + 1)) bad=$((bad + 1))
    echo "FAIL $1: ${2:-}"
    printf '%s\n' "$out" | sed 's/^/    | /'
  }
  out=$(bash "$t/box-gc.sh" --dry-run "$t/root" 2>&1)
  rc=$?
  if [ "$rc" = 0 ] && grep -Eq '^proc 4194302 /bin/bash: bash tools/gpu-gate\.sh gen_x \(cwd .*/root\)$' <<< "$out"; then
    pass 'dry run: this track'"'"'s waiting bash tools/gpu-gate.sh is listed'
  else fail 'dry run: this track'"'"'s waiting bash tools/gpu-gate.sh is listed' "rc $rc"; fi
  if grep -q '4194303' <<< "$out"; then fail 'dry run: another track'"'"'s waiter never' 'it is listed'
  else pass 'dry run: another track'"'"'s waiter never'; fi
  if grep -q '4194304' <<< "$out"; then fail 'dry run: the track'"'"'s own build shell never' 'it is listed'
  else pass 'dry run: the track'"'"'s own build shell never'; fi
  if grep -q '4194305' <<< "$out"; then fail 'dry run: a non-shell whose argv names the script never' 'it is listed'
  else pass 'dry run: a non-shell whose argv names the script never'; fi
  if grep -q '^proc 4194301 ' <<< "$out" && grep -q '^proc 4194306 ' <<< "$out" && grep -q 'found 3 process(es)' <<< "$out"; then
    pass 'dry run: the target/ orphan and the harness runner are still listed (3 in all)'
  else fail 'dry run: the target/ orphan and the harness runner are still listed (3 in all)' 'see the listing'; fi
  out=$(bash "$t/box-gc.sh" --check "$t/root" 2>&1)
  rc=$?
  if [ "$rc" = 10 ] && grep -q 'found 5 process(es)' <<< "$out" && grep -q '^proc 4194304 ' <<< "$out"; then
    pass 'check: rc 10, and every cwd inside root counted beside them (5 in all)'
  else fail 'check: rc 10, and every cwd inside root counted beside them (5 in all)' "rc $rc"; fi
  echo "box-gc self-test: $((n - bad)) of $n ok"
  [ "$bad" = 0 ]
}

MODE=--dry-run
case "${1:-}" in
  --dry-run|--check|--kill) MODE=$1; shift ;;
  --self-test)
    self_test
    exit $? ;;
  -*) echo "box-gc: unknown flag $1 (use --dry-run, --check, --kill or --self-test)" >&2; exit 64 ;;
esac
ROOT=${1:-$PWD}
PREFIX="$ROOT/target"
# The self-test's copy seds the next line to its fake tree — no BLOOMERY_* name for it: every such
# name is a row of the lever registry (tools/check-levers.sh).
PROC=/proc
HARNESS=()
if [ -n "${BLOOMERY_DATA:-}" ]; then
  HARNESS=("$BLOOMERY_DATA/bin" "$BLOOMERY_DATA/router/bin")
fi

self=$$
anc=" $self "
p=$self
# Bounded: a PPid chain that never reaches 1 would spin here forever, and this scan runs at
# the start and end of every track.
steps=0
while [ -n "$p" ] && [ "$p" != 0 ] && [ "$p" != 1 ]; do
  steps=$((steps + 1))
  if [ "$steps" -gt 64 ]; then
    echo "box-gc: ancestor walk exceeded 64 steps at pid $p — never-candidates may be incomplete" >&2
    break
  fi
  # PPid from status, not field 4 of stat: a comm containing a space or a ')' shifts every
  # field of stat, and the ancestor walk then stops at the wrong pid — which would put the
  # scanning shell's own parent back among the candidates.
  p=$(awk '/^PPid:/{print $2}' "$PROC/$p/status" 2>/dev/null)
  [ -n "$p" ] || break
  anc="$anc$p "
done
echo "box-gc: mode=$MODE prefix=$PREFIX harness=[${HARNESS[*]}] (cwd inside $ROOT) self=$self never-candidates=[${anc# }]"

pids=()
exes=()
for d in "$PROC"/[0-9]*; do
  pid=${d##*/}
  case "$anc" in *" $pid "*) continue ;; esac
  exe=$(readlink "$d/exe" 2>/dev/null) || continue
  case "$exe" in
    "$PREFIX"/*) pids+=("$pid"); exes+=("$exe"); continue ;;
  esac
  harness=0
  for h in "${HARNESS[@]}"; do
    case "$exe" in "$h"/*) harness=1 ;; esac
  done
  if [ "$harness" = 1 ]; then
    cwd=$(readlink "$d/cwd" 2>/dev/null) || continue
    case "$cwd" in
      "$ROOT" | "$ROOT"/*) pids+=("$pid"); exes+=("$exe (cwd $cwd)"); continue ;;
    esac
  fi
  # gpu-gate.sh's waiter (the header): the exe tests above cannot see it. Three guards, one
  # failure mode each:
  #   exe bash — a `tail -f` of a gate log or a grep whose argv names the script is not the waiter
  #   cmdline tools/gpu-gate.sh — this track's own build shells (bash running cargo) are not orphans
  #   cwd inside root — another track's (or the lead's) waiter sits in its own remote dir
  if [ "${exe##*/}" = bash ]; then
    cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || cmd=
    case "$cmd" in
      *tools/gpu-gate.sh*)
        cwd=$(readlink "$d/cwd" 2>/dev/null) || cwd=
        case "$cwd" in
          "$ROOT" | "$ROOT"/*) pids+=("$pid"); exes+=("$exe: ${cmd% } (cwd $cwd)"); continue ;;
        esac
        ;;
    esac
  fi
  if [ "$MODE" = --check ]; then
    cwd=$(readlink "$d/cwd" 2>/dev/null) || continue
    case "$cwd" in
      "$ROOT" | "$ROOT"/*) pids+=("$pid"); exes+=("$exe (cwd $cwd)") ;;
    esac
  fi
done

for i in "${!pids[@]}"; do
  echo "proc ${pids[$i]} ${exes[$i]}"
done
echo "box-gc: found ${#pids[@]} process(es) under $PREFIX, in the harness dirs or waiting on tools/gpu-gate.sh (--check also counts a cwd inside $ROOT)"

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
