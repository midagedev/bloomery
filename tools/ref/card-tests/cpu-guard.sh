#!/usr/bin/env bash
# The cpu guard's tests (guard_cpu, cpu_busy_reading: tools/ref/lease.sh), against a copy of lease.sh
# beside this tree's lease-probe.sh in a temporary directory.
#
#   tools/ref/card-tests/cpu-guard.sh [LEASE_SH]
#
# LEASE_SH is the lease.sh to test, this tree's by default; a base copy is the FAIL-first.
# Stub: a /proc tree of its own (BLOOMERY_LEASE_PROC) whose uptime and cpu times the test sets, so every
# reading is exact — the runner's own busy arm left out, a foreign cargo and a rustc started inside the
# interval counted, a reused pid counted from its own start, a reading under a second only starting the
# next interval, a tree that cannot be read a named exit. Runs on the Mac (bash 3.2) and on the box.
# Live, where /proc/self/stat exists (the box): two busy copies of `yes` under names of their own (so no
# real build on the box counts), one the test shell's child (its own arm: never tagged) and one
# reparented away from it (foreign: tagged), each started and killed by the pid this test recorded.
# One line per test, `ok <name>`, `FAIL <name>: <why>` or `skip <name>: <why>`; exit 0 iff none failed.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
LEASE=${1:-$ROOT/tools/ref/lease.sh}
tmp=${TMPDIR:-/tmp}
tmp=$(mktemp -d "${tmp%/}/cpu-guard.XXXXXX")
cleanup() {
  local f
  for f in "$tmp"/*.pid; do [ -f "$f" ] && kill "$(cat "$f")" 2> /dev/null; done
  rm -rf "$tmp"
}
trap cleanup EXIT
mkdir -p "$tmp/tree/tools/ref" "$tmp/bin"
cp "$LEASE" "$tmp/tree/tools/ref/lease.sh"
cp "$ROOT/tools/ref/lease-probe.sh" "$tmp/tree/tools/ref/"
L=$tmp/tree/tools/ref/lease.sh
n=0 failed=0

# expect <name> <want rc> <pattern> <script>: the script run by a bash that sourced the lease copy, its
# rc against <want rc> and its output (stdout and stderr) against grep -E <pattern>; a pattern that
# starts with `!` must not match.
expect() {
  local name=$1 want=$2 pat=$3 rc
  bash -c "source '$L' && $4" > "$tmp/out" 2>&1
  rc=$?
  n=$((n + 1))
  if [ "$rc" != "$want" ]; then
    failed=$((failed + 1))
    echo "FAIL $name: rc $rc, want $want"
    sed 's/^/    | /' "$tmp/out"
  elif [ "${pat#!}" != "$pat" ] && grep -Eq -- "${pat#!}" "$tmp/out"; then
    failed=$((failed + 1))
    echo "FAIL $name: a line matches /${pat#!}/"
    sed 's/^/    | /' "$tmp/out"
  elif [ "${pat#!}" = "$pat" ] && ! grep -Eq -- "$pat" "$tmp/out"; then
    failed=$((failed + 1))
    echo "FAIL $name: no line matches /$pat/"
    sed 's/^/    | /' "$tmp/out"
  else
    echo "ok $name"
  fi
}

# The stub tree's writers, for the scripts `expect` runs: `up <seconds>` sets the uptime, `ps1 <pid>
# <comm> <ppid> <start> <ticks>` a process's stat line (utime = ticks, stime 0), `gone <pid>` ends it.
# The runner is the script's own shell ($$): its child `timeout` (pid 101) runs generate_ds41 (102); cargo
# (201) and rustc (202) hang off pid 1. Every process exists at the interval's start, so the seed reading
# (named processes: none) is taken at once. Each script starts from an empty tree.
STUB='P=$BLOOMERY_LEASE_PROC; rm -rf "${P:?}"; mkdir -p "$P"
up() { echo "$1 0.00" > "$P/uptime"; }
ps1() { mkdir -p "$P/$1"; echo "$1 ($2) S $3 0 0 0 -1 0 0 0 0 0 $5 0 0 0 20 0 1 0 $4 0 0" > "$P/$1/stat"; }
gone() { rm -rf "${P:?}/$1"; }
up 100.00; ps1 1 systemd 0 1 50; ps1 $$ bash 1 900 10; ps1 101 timeout $$ 1000 1; ps1 102 generate_ds41 101 1001 1000
ps1 201 cargo 1 1100 500; ps1 202 rustc 201 1101 20
seed() { CPU_BUSY_COMMS=none; cpu_busy_sample || exit 3; CPU_BUSY_COMMS="cargo rustc generate_ds41"; }'
export BLOOMERY_LEASE_PROC=$tmp/proc
export CPU_TEST_STUB=$STUB

expect 'the runner'\''s own busy arm is left out' 0 '^tag=\[\] reading=\[0\.0 cargo 0\.0%;rustc 0\.0%;\] own=\[generate_ds41 300\.0%;\] span=\[10\.0\]$' \
  'eval "$CPU_TEST_STUB"; seed; up 110.00; ps1 102 generate_ds41 101 1001 4000; guard_cpu post
   echo "tag=[$CPU_BUSY_TAG] reading=[$CPU_BUSY_READING] own=[$CPU_BUSY_OWN] span=[$CPU_BUSY_SPAN]"'
expect 'a foreign cargo over the threshold tags the row' 0 '^\[cpu-busy\] .* post: 80\.0% > 50% of one cpu over 10\.0 s \[cargo 80\.0%;rustc 0\.0%;\] \(not counted, under the runner: generate_ds41 300\.0%;\)$' \
  'eval "$CPU_TEST_STUB"; seed; up 110.00; ps1 102 generate_ds41 101 1001 4000; ps1 201 cargo 1 1100 1300; guard_cpu post
   [ "$CPU_BUSY_TAG" = " [cpu-busy]" ]'
expect 'a foreign rustc started inside the interval counts whole' 0 'post: 70\.0% > 50% .* \[cargo 0\.0%;rustc 0\.0%;rustc 70\.0%;\]' \
  'eval "$CPU_TEST_STUB"; seed; up 110.00; ps1 203 rustc 201 10500 700; guard_cpu post'
expect 'a reused pid counts from its own start' 0 '^tag=\[\] reading=\[10\.0 cargo 10\.0%;rustc 0\.0%;\]' \
  'eval "$CPU_TEST_STUB"; seed; up 110.00; gone 201; ps1 201 cargo 1 10400 100; guard_cpu post
   echo "tag=[$CPU_BUSY_TAG] reading=[$CPU_BUSY_READING]"'
expect 'a foreign process of the same busy share under the threshold does not tag' 0 '^tag=\[\]$' \
  'eval "$CPU_TEST_STUB"; seed; up 110.00; ps1 201 cargo 1 1100 900; guard_cpu post; echo "tag=[$CPU_BUSY_TAG]"'
expect 'a call under a second after the last reading starts the next interval' 0 '^tag=\[\] span=\[\] then tag=\[ \[cpu-busy\]\] span=\[10\.0\]$' \
  'eval "$CPU_TEST_STUB"; seed; up 100.50; ps1 201 cargo 1 1100 1300; guard_cpu pre; a="tag=[$CPU_BUSY_TAG] span=[$CPU_BUSY_SPAN]"
   up 110.50; ps1 201 cargo 1 1100 2100; guard_cpu post; echo "$a then tag=[$CPU_BUSY_TAG] span=[$CPU_BUSY_SPAN]"'
expect 'cpu_busy_reading keeps its form' 0 '^0\.0 $' \
  'eval "$CPU_TEST_STUB"; CPU_BUSY_COMMS=none; cpu_busy_reading'
expect 'a process tree that cannot be read is a named exit' 70 '^guard_cpu: pre: no cpu reading: .*/no-proc cannot be read' \
  "BLOOMERY_LEASE_PROC='$tmp/no-proc' guard_cpu pre"
expect 'strict mode aborts on a foreign cargo' 75 'witness abort-cpu' \
  'eval "$CPU_TEST_STUB"; seed; up 110.00; ps1 201 cargo 1 1100 1300; witness() { echo "witness $1"; }; BLOOMERY_OTHER_STRICT=1 guard_cpu post'

if [ -r /proc/self/stat ]; then
  unset BLOOMERY_LEASE_PROC
  yes=$(command -v yes)
  cp "$yes" "$tmp/bin/gends41-own"
  cp "$yes" "$tmp/bin/cargo-foreign"
  export CPU_TEST_TMP=$tmp
  # The test shell starts its own arm and records its pid; the foreign one is started by a subshell that
  # exits at once, so its parent is no longer under the test shell.
  expect 'live: the runner'\''s own busy arm does not tag its row' 0 '!\[cpu-busy\]' \
    'CPU_BUSY_COMMS="gends41-own cargo-foreign"
     "$CPU_TEST_TMP/bin/gends41-own" > /dev/null & echo $! > "$CPU_TEST_TMP/own.pid"
     sleep 1.5; guard_cpu pre; sleep 1.5; guard_cpu post
     kill "$(cat "$CPU_TEST_TMP/own.pid")"; echo "own: $CPU_BUSY_OWN"'
  expect 'live: a foreign busy process tags the row' 0 '^\[cpu-busy\] .* post: .* \[cargo-foreign [0-9.]+%;\]' \
    'CPU_BUSY_COMMS="gends41-own cargo-foreign"
     ( "$CPU_TEST_TMP/bin/cargo-foreign" > /dev/null & echo $! > "$CPU_TEST_TMP/foreign.pid" )
     guard_cpu pre; sleep 1.5; guard_cpu post
     kill "$(cat "$CPU_TEST_TMP/foreign.pid")"; [ "$CPU_BUSY_TAG" = " [cpu-busy]" ]'
else
  echo "skip live: no /proc/self/stat here (the box runs it)"
fi
echo "cpu-guard: $n tests, $failed failed"
[ "$failed" = 0 ]
