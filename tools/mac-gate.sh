#!/usr/bin/env bash
# The Metal gate runner. It runs on a Mac: it takes the gate binary a recipe just built, runs it under the Metal lock
# and a bound, and returns its exit code unchanged. Arguments after the name go to the binary as they are.
#   ./tools/mac-gate.sh target/debug/gate_metal_device
#   ./tools/mac-gate.sh --self-test
#
# A name with a '/' is used as a path; without one, target/release/<name> is run (the release-profile recipe's form).
# Unlike the box's card locks, the Metal lock belongs to this script alone: ~/.cache/bloomery/metal-gate.lock, fcntl,
# a 30-minute wait. The timing lease and the witness measurements come with the Mac's timing runner (M3); here
# sw_vers and hw.model are printed for information only.
#
# Exit codes: the gate's own, unchanged. 124 = the bound passed and TERM ended the gate. 137 = the gate ignored TERM
# and was KILLed. 143, 130, 129 = the runner itself was ended by TERM, INT or HUP and took the gate's process group with
# it. 75 = the lock was not free within the wait (contention, not a gate failure). 69 = the lock file cannot be opened.
# 64 = usage, a bad bound, a host that is not Darwin arm64, BLOOMERY_CARD or GATE_CARD set (a box card name). 78 = macOS
# below 26.2, a macOS version that cannot be read, or no Metal device (the code the gate bin also uses when it is run
# off macOS). 2 = no such binary.
#
# Environment: BLOOMERY_GATE_BOUND (whole seconds, 900 unset; parsed by tools/gate-bound.sh like every gate runner's),
# MAC_GATE_LOCK (the lock file), MAC_GATE_LOCK_WAIT (seconds to wait for the lock, 1800), MAC_GATE_KILL_GRACE (seconds
# between TERM and KILL, 10), MAC_GATE_BIN_DIR (where a bare name is looked for, target/release).
# Test seams, read only while `--self-test` runs its children (it sets MAC_GATE_SELF_TEST): MAC_GATE_TEST_HOST,
# MAC_GATE_TEST_ARCH, MAC_GATE_TEST_OS, MAC_GATE_TEST_DEVICE (yes|no). They let the self-test run, and refuse, as
# another host, another macOS or a machine with no Metal device; they work on a non-Darwin host too.
set -u

PROG="$(basename "$0")"
PROG_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=tools/gate-bound.sh
. "$PROG_DIR/gate-bound.sh"

die() {
  # die <code> <message...>
  local code=$1
  shift
  echo "$PROG: $*" >&2
  exit "$code"
}

# seam <name>: the test seam's value, only while the self-test runs its children; empty otherwise.
seam() {
  if [ -n "${MAC_GATE_SELF_TEST:-}" ]; then
    eval "printf '%s' \"\${$1:-}\""
  fi
}

host_os() {
  local v
  v=$(seam MAC_GATE_TEST_HOST)
  if [ -n "$v" ]; then printf '%s' "$v"; else uname -s; fi
}

host_arch() {
  local v
  v=$(seam MAC_GATE_TEST_ARCH)
  if [ -n "$v" ]; then printf '%s' "$v"; else uname -m; fi
}

os_version() {
  local v
  v=$(seam MAC_GATE_TEST_OS)
  if [ -n "$v" ]; then printf '%s' "$v"; else sw_vers -productVersion 2>/dev/null || true; fi
}

lock_path() {
  if [ -n "${MAC_GATE_LOCK:-}" ]; then
    printf '%s' "$MAC_GATE_LOCK"
  else
    printf '%s' "$HOME/.cache/bloomery/metal-gate.lock"
  fi
}

lock_wait() {
  printf '%s' "${MAC_GATE_LOCK_WAIT:-1800}"
}

kill_grace() {
  printf '%s' "${MAC_GATE_KILL_GRACE:-10}"
}

# bound_seconds: BOUND, the bound in seconds; the refusal is 64. A bound with more than nine digits is refused too: the
# runner's timer cannot take a longer one.
BOUND=900
bound_seconds() {
  gate_bound "$PROG" || return 64
  if [ "${#BOUND}" -gt 9 ]; then
    echo "$PROG: BLOOMERY_GATE_BOUND has at most nine digits, got '$BOUND'" >&2
    return 64
  fi
}

resolve_binary() {
  # resolve_binary <name-or-path>
  local name=$1
  case "$name" in
    */*) printf '%s' "$name" ;;
    *) printf '%s/%s' "${MAC_GATE_BIN_DIR:-target/release}" "$name" ;;
  esac
}

refuse_host() {
  local os arch
  os=$(host_os)
  arch=$(host_arch)
  if [ "$os" != "Darwin" ]; then
    die 64 "this host is $os, not Darwin; the Metal gates run on a Mac"
  fi
  if [ "$arch" != "arm64" ]; then
    die 64 "this host is $arch, not arm64; the Metal gates run on an Apple silicon Mac"
  fi
  if [ -n "${BLOOMERY_CARD:-}" ] || [ -n "${GATE_CARD:-}" ]; then
    die 64 "BLOOMERY_CARD/GATE_CARD name a box card; a Metal gate has no card"
  fi
}

# The floor is 26.2. A version without a minor part ("26") has minor 0.
refuse_os() {
  local v major minor
  v=$(os_version)
  case "$v" in
    '' | *[!0-9.]*) die 78 "the macOS version could not be read" ;;
  esac
  major=${v%%.*}
  case "$v" in
    *.*)
      minor=${v#*.}
      minor=${minor%%.*}
      ;;
    *) minor=0 ;;
  esac
  case "$major" in
    '' | *[!0-9]*) die 78 "the macOS version '$v' could not be read" ;;
  esac
  case "$minor" in
    '' | *[!0-9]*) minor=0 ;;
  esac
  if [ "$major" -gt 26 ]; then
    return 0
  fi
  if [ "$major" -eq 26 ] && [ "$minor" -ge 2 ]; then
    return 0
  fi
  die 78 "macOS $v is below the floor 26.2 the Metal gates need"
}

refuse_device() {
  local v
  v=$(seam MAC_GATE_TEST_DEVICE)
  case "$v" in
    yes) return 0 ;;
    no) die 78 "this machine shows no Metal device" ;;
  esac
  if ! system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Metal Support"; then
    die 78 "this machine shows no Metal device"
  fi
}

witness() {
  echo "witness: os=$(os_version) model=$(sysctl -n hw.model 2>/dev/null || echo unknown)"
}

# The lock and the bound, in one python3: the Mac has neither GNU timeout nor flock, so python stands in. The gate runs
# in its own session, so the runner can stop its whole process group: at the bound (TERM, then KILL after the grace)
# and when the runner itself is ended by TERM, INT or HUP, so that no gate outlives its runner and overlaps the next
# one on the Metal device. Exit codes: 69 (the lock file cannot be opened), 75 (the lock is held), 124 (the bound
# passed and TERM ended the gate), 137 (the gate ignored TERM and was KILLed), 128 plus the signal (the runner was
# signalled), else the gate's own code.
run_under_lock() {
  # run_under_lock <lock> <wait-seconds> <bound-seconds> <grace-seconds> <binary> [args...]
  local lock=$1 wait_s=$2 bound=$3 grace=$4
  shift 4
  # exec: the python runner takes this script's pid, so a signal sent to the pid a caller knows reaches it.
  exec python3 - "$lock" "$wait_s" "$bound" "$grace" "$@" <<'PY'
import fcntl
import os
import signal
import subprocess
import sys
import time

lock_path, wait_s, bound, grace = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
cmd = sys.argv[5:]

try:
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
except OSError as e:
    print(f"mac-gate.sh: the lock file {lock_path} cannot be opened: {e}", file=sys.stderr)
    sys.exit(69)

signalled = []


def on_signal(signum, _frame):
    signalled.append(signum)


for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(sig, on_signal)

deadline = time.monotonic() + wait_s
while True:
    if signalled:
        sys.exit(128 + signalled[0])
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        break
    except OSError:
        if time.monotonic() >= deadline:
            print("mac-gate.sh: the metal gate lock is held by another run", file=sys.stderr)
            sys.exit(75)
        time.sleep(0.2)

# The gate leads a session of its own: its process group is its pid.
proc = subprocess.Popen(cmd, start_new_session=True)


def stop_group(sig):
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass  # the group is already gone


limit = time.monotonic() + bound
rc = None
while rc is None and not signalled and time.monotonic() < limit:
    rc = proc.poll()
    if rc is None:
        time.sleep(0.05)

if rc is not None:
    # The leader is gone; a child it left behind must not outlive the lock.
    stop_group(signal.SIGKILL)
    sys.exit(rc if rc >= 0 else 128 - rc)

# The bound passed or the runner was signalled: end the gate's whole group.
cause = 128 + signalled[0] if signalled else 124
stop_group(signal.SIGTERM)
end = time.monotonic() + grace
while proc.poll() is None and time.monotonic() < end:
    time.sleep(0.05)
if proc.poll() is None:
    stop_group(signal.SIGKILL)
    proc.wait()
    sys.exit(137)
stop_group(signal.SIGKILL)
sys.exit(cause)
PY
}

run_gate() {
  # run_gate <binary> [args...]
  local bin
  bin=$(resolve_binary "$1")
  shift
  [ -x "$bin" ] || die 2 "the gate binary $bin is not there"
  bound_seconds || exit $?
  local lock
  lock=$(lock_path)
  mkdir -p "$(dirname "$lock")" 2>/dev/null || true
  witness
  run_under_lock "$lock" "$(lock_wait)" "$BOUND" "$(kill_grace)" "$bin" "$@"
}

SELF_TEST_DIR=""
self_test() {
  local rc
  SELF_TEST_DIR=$(mktemp -d "${TMPDIR:-/tmp}/mac-gate-self.XXXXXX") || die 70 "no temporary directory"
  local dir=$SELF_TEST_DIR
  trap 'rm -rf "$SELF_TEST_DIR"' EXIT

  local lock="$dir/lock"
  local pass=0 fail=0

  # The children's environment: a Darwin arm64 host on macOS 26.2 with a Metal device, whatever this host is. A test
  # overrides one seam to see its refusal.
  local base_env=(MAC_GATE_SELF_TEST=1 MAC_GATE_TEST_HOST=Darwin MAC_GATE_TEST_ARCH=arm64 MAC_GATE_TEST_OS=26.2.1
    MAC_GATE_TEST_DEVICE=yes MAC_GATE_LOCK="$lock" MAC_GATE_LOCK_WAIT=1 MAC_GATE_KILL_GRACE=1)
  # shellcheck disable=SC2329 # invoked through check
  runner() {
    env "${base_env[@]}" "$@"
  }

  check() {
    # check <name> <expected-rc> <command...>
    local name=$1 want=$2
    shift 2
    "$@" >"$dir/out" 2>&1
    rc=$?
    if [ "$rc" = "$want" ]; then
      pass=$((pass + 1))
      echo "self-test $name: PASS (rc $rc)"
    else
      fail=$((fail + 1))
      echo "self-test $name: FAIL (rc $rc, wanted $want)"
      sed 's/^/  /' "$dir/out"
    fi
  }

  # Stub gates: one answers with its own code, one hangs, one ignores TERM and hangs, one records its pid and hangs.
  printf '#!/usr/bin/env bash\nexit 3\n' >"$dir/code3"
  printf '#!/usr/bin/env bash\nexit 78\n' >"$dir/code78"
  printf '#!/usr/bin/env bash\nsleep 30\n' >"$dir/hang"
  printf '#!/usr/bin/env bash\ntrap "" TERM\nwhile :; do sleep 1; done\n' >"$dir/stubborn"
  # shellcheck disable=SC2016 # the stub's own $$, not this shell's
  printf '#!/usr/bin/env bash\necho $$ >"$1"\nsleep 30\n' >"$dir/pidhang"
  # One leaves a TERM-ignoring child behind, records its pid and exits 0.
  # shellcheck disable=SC2016 # the stub's own $!, not this shell's
  printf '#!/usr/bin/env bash\n( trap "" TERM; exec sleep 30 ) &\necho $! >"$1"\nexit 0\n' >"$dir/orphan"
  # One leaves the same child, records its pid and hangs until TERM ends it.
  # shellcheck disable=SC2016 # the stub's own $!, not this shell's
  printf '#!/usr/bin/env bash\n( trap "" TERM; exec sleep 30 ) &\necho $! >"$1"\nsleep 30\n' >"$dir/hangorphan"
  chmod +x "$dir/code3" "$dir/code78" "$dir/hang" "$dir/stubborn" "$dir/pidhang" "$dir/orphan" "$dir/hangorphan"

  check "the binary's own code is returned" 3 runner "$PROG_DIR/mac-gate.sh" "$dir/code3"
  check "the binary's own 78 is returned" 78 runner "$PROG_DIR/mac-gate.sh" "$dir/code78"
  check "a missing binary is 2" 2 runner "$PROG_DIR/mac-gate.sh" "$dir/nothing"
  check "no argument is 64" 64 runner "$PROG_DIR/mac-gate.sh"
  check "a zero bound is 64" 64 runner BLOOMERY_GATE_BOUND=0 "$PROG_DIR/mac-gate.sh" "$dir/code3"
  check "a bound with a leading zero is 64" 64 runner BLOOMERY_GATE_BOUND=08 "$PROG_DIR/mac-gate.sh" "$dir/code3"
  check "a bound that is no number is 64" 64 runner BLOOMERY_GATE_BOUND=abc "$PROG_DIR/mac-gate.sh" "$dir/code3"
  check "a bound of twenty digits is 64" 64 \
    runner BLOOMERY_GATE_BOUND=99999999999999999999 "$PROG_DIR/mac-gate.sh" "$dir/code3"
  check "a non-Darwin host is 64" 64 runner MAC_GATE_TEST_HOST=Linux "$PROG_DIR/mac-gate.sh" "$dir/code3"
  check "a non-arm64 host is 64" 64 runner MAC_GATE_TEST_ARCH=x86_64 "$PROG_DIR/mac-gate.sh" "$dir/code3"
  check "BLOOMERY_CARD is 64" 64 runner BLOOMERY_CARD=3090 "$PROG_DIR/mac-gate.sh" "$dir/code3"
  check "GATE_CARD is 64" 64 runner GATE_CARD=a6000 "$PROG_DIR/mac-gate.sh" "$dir/code3"
  check "a macOS just below the floor is 78" 78 runner MAC_GATE_TEST_OS=26.1.9 "$PROG_DIR/mac-gate.sh" "$dir/code3"
  check "a macOS version without a minor part, below the floor, is 78" 78 \
    runner MAC_GATE_TEST_OS=26 "$PROG_DIR/mac-gate.sh" "$dir/code3"
  check "a macOS version that cannot be read is 78" 78 runner MAC_GATE_TEST_OS=abc "$PROG_DIR/mac-gate.sh" "$dir/code3"
  check "the floor itself is taken" 3 runner MAC_GATE_TEST_OS=26.2 "$PROG_DIR/mac-gate.sh" "$dir/code3"
  check "a macOS above the floor is taken" 3 runner MAC_GATE_TEST_OS=27 "$PROG_DIR/mac-gate.sh" "$dir/code3"
  check "no Metal device is 78" 78 runner MAC_GATE_TEST_DEVICE=no "$PROG_DIR/mac-gate.sh" "$dir/code3"
  check "the bound ends a hanging gate with 124" 124 runner BLOOMERY_GATE_BOUND=1 "$PROG_DIR/mac-gate.sh" "$dir/hang"
  check "a gate that ignores TERM is KILLed with 137" 137 \
    runner BLOOMERY_GATE_BOUND=1 "$PROG_DIR/mac-gate.sh" "$dir/stubborn"
  check "a lock file that cannot be opened is 69" 69 \
    runner MAC_GATE_LOCK="$dir" "$PROG_DIR/mac-gate.sh" "$dir/code3"

  # The runner ended by TERM takes the gate with it: the gate's pid is gone, and the runner answers 143. The runner is
  # started directly (env, then this script, then python, all by exec), so $! is the pid that gets the signal.
  local pidfile="$dir/pid" runner_pid gate_pid=""
  env "${base_env[@]}" "$PROG_DIR/mac-gate.sh" "$dir/pidhang" "$pidfile" >"$dir/out" 2>&1 &
  runner_pid=$!
  local waited=0
  while [ ! -s "$pidfile" ] && [ "$waited" -lt 100 ]; do
    sleep 0.1
    waited=$((waited + 1))
  done
  gate_pid=$(cat "$pidfile" 2>/dev/null || true)
  kill -TERM "$runner_pid" 2>/dev/null || true
  wait "$runner_pid" 2>/dev/null
  rc=$?
  if [ "$rc" = 143 ] && [ -n "$gate_pid" ] && ! kill -0 "$gate_pid" 2>/dev/null; then
    pass=$((pass + 1))
    echo "self-test a signalled runner takes its gate along: PASS (rc $rc)"
  else
    fail=$((fail + 1))
    echo "self-test a signalled runner takes its gate along: FAIL (rc $rc, wanted 143, gate pid '$gate_pid')"
    [ -z "$gate_pid" ] || kill -KILL "$gate_pid" 2>/dev/null || true
  fi

  # A held lock: the holder takes it, says so, and keeps it while the gate waits and gives up.
  local ready="$dir/ready" holder
  python3 -c "
import fcntl, os, sys, time
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o644)
fcntl.flock(fd, fcntl.LOCK_EX)
open(sys.argv[2], 'w').close()
time.sleep(4)
" "$lock" "$ready" &
  holder=$!
  waited=0
  while [ ! -e "$ready" ] && [ "$waited" -lt 100 ]; do
    sleep 0.1
    waited=$((waited + 1))
  done
  check "a held lock is 75" 75 runner "$PROG_DIR/mac-gate.sh" "$dir/code3"
  kill "$holder" 2>/dev/null || true
  wait "$holder" 2>/dev/null || true

  # A gate that leaves a TERM-ignoring child, on both ends of the runner: the leader exits, or the bound ends it. The
  # runner returns the gate's code and the child does not outlive it.
  # shellcheck disable=SC2329 # invoked below
  child_gone() {
    # child_gone <name> <pid file>
    local pid
    pid=$(cat "$2" 2>/dev/null)
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      pass=$((pass + 1))
      echo "self-test $1: PASS"
    else
      fail=$((fail + 1))
      echo "self-test $1: FAIL (pid ${pid:-none} still runs or was not written)"
      [ -n "$pid" ] && kill -KILL "$pid" 2>/dev/null
    fi
  }
  check "a gate that exits 0 and leaves a child is 0" 0 runner "$PROG_DIR/mac-gate.sh" "$dir/orphan" "$dir/orphan.pid"
  child_gone "the child a gate left is gone" "$dir/orphan.pid"
  check "the bound ends a gate that leaves a child with 124" 124 \
    runner BLOOMERY_GATE_BOUND=1 "$PROG_DIR/mac-gate.sh" "$dir/hangorphan" "$dir/hangorphan.pid"
  child_gone "the child of a gate the bound ended is gone" "$dir/hangorphan.pid"

  echo "self-test: $pass passed, $fail failed"
  [ "$fail" -eq 0 ] || exit 1
}

case "${1:-}" in
  --self-test)
    self_test
    exit $?
    ;;
  '')
    die 64 "usage: $PROG BIN [ARGS...] | $PROG --self-test"
    ;;
  --help | -h)
    echo "usage: $PROG BIN [ARGS...] | $PROG --self-test"
    exit 0
    ;;
esac

refuse_host
refuse_os
refuse_device
run_gate "$@"
