#!/usr/bin/env bash
# shellcheck shell=bash
# The lease's read side: everything that reads the machine-wide lease without taking it. Sourced,
# never executed, as tools/ref/lease.sh is (which sources this file first): it defines functions and
# variables, exports nothing, and nothing here may exit or fail at top level. It names no other file
# of the tree, so it is the whole of what tools/box.sh runs on the box before every command (the
# guard, lease_guard) — the take, the card, the witnesses and the runners' helpers stay in lease.sh.
#
#   ( cd <tree> && . tools/ref/lease-probe.sh && lease_guard <bound s> [owner] ) && <command>
#
# What it owns: the lease's file (LEASE_LOCK); its probe (lease_free, the only `flock` on the lease
# that is not a take); the holds (LEASE_HOLDS, lease_holds_up); the guard (lease_guard); naming the
# lease's holder to a waiter (lease_holders); the UTC stamp every [lease] line carries (now).
LEASE_LOCK=${BLOOMERY_LEASE_LOCK:-/root/bloomery-cpu.lock}

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# lease_free [file]: the probe of the lease, and its one owner. 0 when the lease is free, 1 when it is
# held, 2 when it cannot be tested (flock's own error, named on stderr). It takes a shared lock for
# the life of `true` (`flock -s -n`): shared probes never collide with one another, so it fails only
# while an exclusive lock is on the file — lease_take's descriptor 9, or a process that inherited it.
# An exclusive probe (`flock -n <file> true`) is a take for a few ms and reads a free lease as held
# whenever another probe is in at that instant (rig-log's netdata gate probes, shared, every 0.5 s).
# A probe that meets lease_take costs the runner a line: its `flock -n 9` fails for the probe's ms, and
# the wait loop's `flock -w 60 9` returns as soon as the probe lets go.
# Success means free, as for `flock -n <file> true`, so `if lease_free`, `lease_free ||` and
# `! lease_free` all read an untestable lease as not free. The file defaults to LEASE_LOCK.
lease_free() {
  local f=${1:-$LEASE_LOCK} rc=0
  flock -s -n -E 75 "$f" true || rc=$?
  case $rc in
    0) return 0 ;;
    75) return 1 ;;
  esac
  echo "[lease] $f cannot be tested (flock rc $rc): read as not free" >&2
  return 2
}

# The holds: a window that the lease alone does not cover. A sitting of several runs (a sitting script,
# tools/gpu-ab.py's arms) takes and releases the lease once per run, and between two runs the lease
# is free for the seconds the next run's rsync and build take. A hold is a file, up while it exists:
# /root/bloomery-<owner>-hold, one per owner, <owner> letters, digits and `_` (`03`: bloomery-03's
# sitting scripts). LEASE_HOLDS is the one pattern that names them, its `*` the owner
# (BLOOMERY_LEASE_HOLDS, for the stub tests).
LEASE_HOLDS=${BLOOMERY_LEASE_HOLDS:-/root/bloomery-*-hold}

# lease_holds_up [owner]: one line `<file> owner=<o> up=<age> since=<epoch>` per hold that is up,
# except <owner>'s: a hold's owner runs its own sitting through tools/box.sh while the hold is up. A
# LEASE_HOLDS that is not one `*` between a prefix and a suffix is refused (rc 64): a pattern that
# matches nothing would pass every command.
lease_holds_up() {
  local mine=${1:-} pre suf f o m s
  pre=${LEASE_HOLDS%%\**} suf=${LEASE_HOLDS#*\*}
  if [ "$pre" = "$LEASE_HOLDS" ] || [ -z "$pre" ] || [ -z "$suf" ] || [ "$suf" != "${suf#*\*}" ]; then
    echo "[guard] LEASE_HOLDS='$LEASE_HOLDS' is not <prefix>*<suffix> with one '*' (the owner)" >&2
    return 64
  fi
  # shellcheck disable=SC2086 # LEASE_HOLDS is a pattern; its expansion is the list of holds
  for f in $LEASE_HOLDS; do
    [ -e "$f" ] || continue
    o=${f#"$pre"} o=${o%"$suf"}
    [ -n "$mine" ] && [ "$o" = "$mine" ] && continue
    m=$(__lease_mtime "$f")
    if [ -n "$m" ]; then
      s=$(($(date +%s) - m))
      echo "$f owner=$o up=$((s / 3600))h$(printf %02d $((s % 3600 / 60)))m$(printf %02d $((s % 60)))s since=$m"
    else
      echo "$f owner=$o up=? since=?"
    fi
  done
}
# __lease_mtime <file>: its mtime in epoch seconds (GNU stat on the box, BSD stat on the Mac), or nothing.
__lease_mtime() { stat -c %Y "$1" 2> /dev/null || stat -f %m "$1" 2> /dev/null || true; }

# __lease_gives_way <owner> <holds>: 0 when <owner>'s own hold is up and another hold in <holds>
# (lease_holds_up's lines) went up before it — same second: the owner that sorts first went up first.
# Two sittings whose holds are both up wait on each other; the later one gives way at once.
__lease_gives_way() {
  local owner=$1 mine m f o since
  mine=${LEASE_HOLDS%%\**}$owner${LEASE_HOLDS#*\*}
  [ -n "$owner" ] && [ -e "$mine" ] || return 1
  m=$(__lease_mtime "$mine")
  [ -n "$m" ] || return 1
  while read -r f o _ since; do
    o=${o#owner=} since=${since#since=}
    case $since in '' | *[!0-9]*) continue ;; esac
    if [ "$since" -lt "$m" ] || { [ "$since" = "$m" ] && [[ $o < $owner ]]; }; then
      echo "[guard] $(now) hold $f (owner $o) went up before $mine: two sittings would wait on each other, so this one gives way — the command did not run (rc 75); take $mine down and start again after $o's sitting" >&2
      return 0
    fi
  done <<< "$2"
  return 1
}

# lease_guard <bound seconds> [owner]: what tools/box.sh runs on the box before its command, in the
# same ssh. 0 when neither the lease (lease_free) nor a hold other than <owner>'s is up: at once when
# that holds at the first look. Else the command waits: what is up goes to stderr at once and once a
# minute (the lease with its holders, lease_holders; each hold with its owner and age), it polls every
# LEASE_POLL seconds, and after a wait it starts only on two quiet polls in a row — the lease is free
# for seconds between the runs of a sitting, and a waiter must not start there. Still busy after
# <bound> seconds: 75, naming what is up. Bound 0 does not wait: 75 at once when busy. When <owner>'s
# own hold is up and another hold went up before it, 75 at once (__lease_gives_way): the two sittings
# would wait on each other until both bounds ran out. A lease that cannot be tested is 70 at once, a
# bad bound, owner or LEASE_POLL 64. It holds nothing: a probe is a few ms, and the command's own
# runner takes the lease after this returns.
LEASE_POLL=${BLOOMERY_LEASE_POLL:-30}
lease_guard() {
  local bound=${1:-} owner=${2:-} t0 el said=-1 quiet=0 waited=0 rc holds
  case $bound in
    '' | *[!0-9]*)
      echo "[guard] the bound is whole seconds (0: do not wait), got '$bound'" >&2
      return 64
      ;;
  esac
  case $owner in
    *[!A-Za-z0-9_]*)
      echo "[guard] an owner is letters, digits and _ (the <owner> of /root/bloomery-<owner>-hold), got '$owner'" >&2
      return 64
      ;;
  esac
  case $LEASE_POLL in
    '' | *[!0-9]* | 0)
      echo "[guard] BLOOMERY_LEASE_POLL is whole seconds above 0, got '$LEASE_POLL'" >&2
      return 64
      ;;
  esac
  t0=$(date +%s)
  while :; do
    rc=0
    lease_free || rc=$?
    if [ "$rc" = 2 ]; then
      echo "[guard] $(now) the lease cannot be tested (above): the command did not run (rc 70)" >&2
      return 70
    fi
    holds=$(lease_holds_up "$owner") || return $?
    ! __lease_gives_way "$owner" "$holds" || return 75
    el=$(($(date +%s) - t0))
    if [ "$rc" = 0 ] && [ -z "$holds" ]; then
      [ "$waited" = 1 ] || return 0
      quiet=$((quiet + 1))
      if [ "$quiet" -ge 2 ]; then
        # After a wait this line is also the liveness check: a caller that went away takes SIGPIPE
        # here, and the command does not start with nobody attached.
        echo "[guard] $(now) quiet on two polls in a row after ${el} s: the command starts" >&2
        return 0
      fi
    else
      quiet=0
      if [ "$el" -ge "$bound" ] || [ "$said" -lt 0 ] || [ $((el - said)) -ge 60 ]; then
        if [ "$bound" = 0 ]; then
          echo "[guard] $(now) the box is busy and BLOOMERY_BOX_WAIT=0 does not wait: the command did not run (rc 75):" >&2
        elif [ "$el" -ge "$bound" ]; then
          echo "[guard] $(now) still busy after ${el} s (the bound, BLOOMERY_BOX_WAIT=$bound): the command did not run (rc 75):" >&2
        elif [ "$said" -lt 0 ]; then
          echo "[guard] $(now) the box is busy; the command waits up to ${bound} s (BLOOMERY_BOX_WAIT), polling every ${LEASE_POLL} s:" >&2
        else
          echo "[guard] $(now) still busy after ${el} s:" >&2
        fi
        if [ "$rc" = 1 ]; then
          echo "[guard]   the timing lease $LEASE_LOCK is held (a sitting runs); its holders:" >&2
          lease_holders "$LEASE_LOCK" >&2
        fi
        [ -z "$holds" ] || sed 's/^/[guard]   hold /; s/$/ — its owner passes with BLOOMERY_HOLD_OWNER/' <<< "$holds" >&2
        said=$el
        [ "$el" -lt "$bound" ] || return 75
      fi
    fi
    waited=1
    sleep "$LEASE_POLL"
  done
}

# lease_holders <lock file>: who holds the lock, as lines a waiter prints. One line per process with
# a descriptor on the file (/proc/<pid>/fd, matched by device and inode): pid, comm, exe, cwd, time
# since it started, the BLOOMERY_LEASE_CARD of its environment, its argv, and whether that descriptor
# carries the lock (/proc/<pid>/fdinfo) or is only open — another waiter, or a probe (lease_free)
# caught between its open and its lock; a probe caught holding shows in /proc/locks as READ (shared),
# the lease as WRITE. Then the lock as /proc/locks has it: the pid that took it, which can be dead
# while the lock lives on in the descriptor — flock(1) takes it on a runner's descriptor 9 and exits,
# and a runner's child keeps it after the runner dies — so both are printed. The caller's own
# descriptor (a waiter's) is left out. A lock whose holder cannot be found is said to be so, never
# passed over.
# BLOOMERY_LEASE_PROC names another /proc tree, for the Mac stub tests (tools/ref/card-tests).
lease_holders() {
  python3 - "$1" "${BLOOMERY_LEASE_PROC:-/proc}" << 'PY'
import os
import sys

lock, proc = sys.argv[1], sys.argv[2]
say = lambda text: print(f'[lease]   {text}')
try:
    want = os.stat(lock)
except OSError as e:
    say(f'the holder of {lock} cannot be named: the file cannot be read ({e.strerror})')
    sys.exit(0)
key = (want.st_dev, want.st_ino)


def read(path, mode='r'):
    try:
        with open(path, mode) as f:
            return f.read()
    except OSError:
        return None


def link(path):
    try:
        return os.readlink(path)
    except OSError:
        return '?'


def elapsed(pid):
    stat, up = read(f'{proc}/{pid}/stat'), read(f'{proc}/uptime')
    try:
        start = int(stat[stat.rindex(')') + 2:].split()[19]) / os.sysconf('SC_CLK_TCK')
        s = int(float(up.split()[0]) - start)
    except (AttributeError, ValueError, IndexError):
        return '?'
    return f'{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s'


def describe(pid):
    env = read(f'{proc}/{pid}/environ', 'rb') or b''
    card = next((v[20:].decode(errors='replace') for v in env.split(b'\0') if v.startswith(b'BLOOMERY_LEASE_CARD=')), '')
    argv = (read(f'{proc}/{pid}/cmdline', 'rb') or b'').replace(b'\0', b' ').decode(errors='replace').strip()
    comm = (read(f'{proc}/{pid}/comm') or '?').strip()
    return (f'comm={comm} exe={link(f"{proc}/{pid}/exe")} cwd={link(f"{proc}/{pid}/cwd")} '
            f'elapsed={elapsed(pid)} card={card or "-"} args=[{argv[:160]}]')


def locked(pid, fd):
    info = read(f'{proc}/{pid}/fdinfo/{fd}') or ''
    return any(ln.startswith('lock:') and '->' not in ln for ln in info.split('\n'))


caller, me = os.getppid(), os.getpid()
found = []
try:
    names = os.listdir(proc)
except OSError as e:
    say(f'the holder of {lock} cannot be named: {proc} cannot be listed ({e.strerror})')
    sys.exit(0)
for name in sorted(names, key=lambda n: int(n) if n.isdigit() else -1):
    if not name.isdigit() or int(name) == me:
        continue
    pid = int(name)
    try:
        fds = os.listdir(f'{proc}/{pid}/fd')
    except OSError:
        continue
    held = opened = False
    for fd in fds:
        try:
            st = os.stat(f'{proc}/{pid}/fd/{fd}')
        except OSError:
            continue
        if (st.st_dev, st.st_ino) == key:
            if locked(pid, fd):
                held = True
            else:
                opened = True
    if held or (opened and pid != caller):
        found.append((pid, held))
for pid, held in found:
    what = 'holds it' if held else 'has it open without the lock (another waiter, or a test)'
    say(f'pid {pid} {what}{" (this process)" if pid == caller else ""}: {describe(pid)}')
locks = read(f'{proc}/locks')
takers, waiting = [], 0
for ln in (locks or '').split('\n'):
    f = ln.split()
    blocked = len(f) > 1 and f[1] == '->'
    if blocked:
        f = f[:1] + f[2:]
    if len(f) < 6:
        continue
    try:
        maj, mnr, ino = f[5].split(':')
        same = (int(maj, 16), int(mnr, 16), int(ino)) == (os.major(want.st_dev), os.minor(want.st_dev), want.st_ino)
    except ValueError:
        continue
    if same and blocked:
        waiting += 1
    elif same:
        takers.append((f[1], f[3], f[4]))
if locks is None:
    say(f'{proc}/locks cannot be read, so the pid that took the lock is not named')
for kind, mode, pid in takers:
    alive = os.path.isdir(f'{proc}/{pid}')
    state = 'alive' if alive else ('not running — a lock outlives the process that took it (for a runner, flock(1) '
                                   'itself, on the runner\'s descriptor 9); the holders are the processes above')
    say(f'{proc}/locks: {kind} {mode} taken by pid {pid} ({state})')
if waiting:
    say(f'{proc}/locks: {waiting} more blocked on it')
if not any(held for _, held in found) and not takers:
    say(f'no holder found: no process has {lock} locked in {proc} (released while this looked, or held from '
        'another pid namespace)')
PY
}
