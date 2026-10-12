#!/usr/bin/env bash
# bloomery — 박스(워크스테이션)에서 빌드·실행하는 러너.
# 소스 트리를 박스의 ~/repo/bloomery로 rsync한 뒤 인자를 그 디렉터리에서 실행한다.
# 환경(nightly, LLVM 21 타르볼, CUDA 13.3, 3090 핀)은 박스의 ~/bloomery-env.sh가 소유한다.
# 데이터·참조 바이너리 디렉터리는 BLOOMERY_DATA 하나가 소유한다. 병렬 트랙은 이 값과 REMOTE만 바꾼다.
#   tools/box.sh cargo oxide doctor
#   tools/box.sh cargo oxide run q3k_gemv --arch sm_86
#   tools/box.sh --hold up <owner> [reason…] | down <owner>   the hold on this side (below); `just box-hold`
#   tools/box.sh --self-test [SCRIPT]                        the hold's rules against a stub ssh and rsync, no box;
#                                                            SCRIPT (default: this one) is the box.sh under test
#
# The guard. Before the command, on the box and in the same ssh, lease_guard (tools/ref/lease-probe.sh)
# reads the timing lease (/root/bloomery-cpu.lock, through lease_free: a shared-lock probe) and every
# hold (/root/bloomery-<owner>-hold, up while the file exists). While either is up the command does
# not start: a build or a gate beside a sitting puts [cpu-busy] on the sitting's rows and voids them.
# It waits, naming what is up — the lease's holders, each hold's owner and age — at once and once a
# minute, polls every 30 s, and after a wait starts only on two quiet polls in a row (a sitting of
# several runs leaves the lease free for seconds between them). Still busy at the bound: exit 75,
# contention, naming what is up. A lease that cannot be tested: 70. The guard holds nothing while
# the command runs, so a runner in the command takes the lease after it as before.
#   BLOOMERY_BOX_WAIT=<s>     the bound, default 1800: the wait lease_take and gpu-gate.sh give up
#                             after. 0 does not wait: 75 at once when busy (gate-batch.sh's checks)
#   BLOOMERY_HOLD_OWNER=<o>   this call belongs to the owner of /root/bloomery-<o>-hold and passes that
#                             hold — a sitting script puts its hold up and runs its runners through
#                             box.sh; the lease and every other hold still stop it, and with its own
#                             hold up and another that went up first it gives way at once (75). It
#                             passes the owner's hold on this side (below) the same way
#   BLOOMERY_TIER=real|fixture  the tier the command runs in (unset: real, and every byte of the remote command as before). It
#                             names the tier the model file comes from: tools/ref/ref-paths.sh resolves BLOOMERY_REF_MODEL
#                             to the family's fixture under `fixture`, exports the fixture's card budget as BLOOMERY_CARD_BUDGET
#                             and the file as BLOOMERY_FIXTURE_MODEL (set only when the tier resolved one; tools/gpu-gate.sh
#                             skips its V4.1 load lock only for that file), and refuses a family with none (66) — it never runs
#                             one on its real file. The tier comes from the environment here or a BLOOMERY_TIER entry of
#                             BLOOMERY_BOX_ENV, which is read here, before the profile: an entry the profile had not seen would
#                             reach the binary and not the file. Both set to different tiers is 64.
#   BLOOMERY_FIXTURE_VARIANT=<tag>  the fixture variant (tools/fixture-variants.tsv): the family's fixture requantized to the tag's type
#                             map, under /models/fixture-variants/<tag> (beside the fixture root: the box manifest keys every file
#                             under the root into every fixture-tier item). Read as the tier is, from the environment here or a
#                             BLOOMERY_BOX_ENV entry, before the profile, for the same reason; only under BLOOMERY_TIER=fixture (the
#                             real tier has no variant: 64), and both set to different tags is 64. ref-paths.sh resolves the file
#                             and refuses a tag the table does not name.
#   BLOOMERY_ARCH=<sm_NN>     the CUDA arch (tools/arch.sh; the justfile's `arch` variable reads the same name, sm_86
#                             unset). Set, it is checked here before anything is synced and exported into the command, so a
#                             tool that runs on the box (tools/gate.sh, the scans, tools/release/build.sh) reads what the
#                             recipe's `--arch {{arch}}` expanded to on this side. A value that is no arch is 64, and so is a
#                             BLOOMERY_ARCH entry of BLOOMERY_BOX_ENV that is not this side's value (the default when it is
#                             unset): the arch is named in the environment, where the recipes read it. Unset with no such
#                             entry: every byte of the remote command as before, and tools/arch.sh is not read.
#   BLOOMERY_BOX_READONLY=1   a read — ps, cat, tail, ls, nvidia-smi, a status probe — runs without the
#                             guard and the hold on this side: the way to read a sitting's log, the owner's own
#                             included, while it runs. Refused (64) when the command names cargo, just, make, cmake, ninja
#                             or target/, so the opt-in cannot carry a build or one of our binaries. It
#                             syncs nothing and creates nothing: the command runs in the remote directory
#                             as it is, and a remote directory that is not there is refused (66) — a
#                             sync beside a sitting would put this tree's uncommitted edits under the
#                             sitting's running scripts. Its ssh is the only one that reads stdin, so a
#                             script can go over it (`just box-gc`: 'bash -s -- …' < tools/box-gc.sh)
#   BLOOMERY_BOX_LOCAL=1      the command runs here, in this tree, on a contributor's Linux machine with a card of its own
#                             (docs/contrib/local-check.md): the same command line as on the box, guard and profile and
#                             exports, run by bash with no ssh and no rsync. Its environment file is BLOOMERY_ENV_FILE (default
#                             ~/bloomery-env.sh; not there: 69). The timing lease, the holds and the gate locks live under
#                             BLOOMERY_LOCK_DIR (a path from /, default ~/.cache/bloomery/locks, made when missing),
#                             exported to the command, so the guard reads this machine's lease and holds; the environment
#                             file or BLOOMERY_BOX_ENV moving it after the guard is refused (64), and so is an environment
#                             file's BLOOMERY_ARCH that is not the one the recipe expanded. The card is what this machine
#                             shows: the box side sees BLOOMERY_BOX_CARD=local (tools/gpu-gate.sh's one local lock), any
#                             BLOOMERY_CARD but `both` runs on it, `both` is refused (64), and CUDA_VISIBLE_DEVICES is the
#                             caller's, or the environment file's when it sets one (a timing runner then pins its timing
#                             card: tools/ref/timing-card.sh). CUDA_OXIDE_BACKEND is unset after the
#                             environment file, so cargo oxide builds and keeps its own backend. BLOOMERY_REMOTE names no
#                             directory here and adds no CARGO_BUILD_JOBS. What box.sh reads on this side, its own outputs
#                             (BLOOMERY_REF_MODEL_PROFILE, the fixture tier's BLOOMERY_CARD_BUDGET and
#                             BLOOMERY_FIXTURE_MODEL) and the lease's file names reach the command only as box.sh exports
#                             them, as on the box. Refused (64): on a host that is not Linux, with BLOOMERY_BOX_QUEUE=1 (the
#                             queue is the box's), with a relative BLOOMERY_LOCK_DIR (two trees would lock apart), with /root
#                             however it is spelled (the guard would make /root/bloomery-cpu.lock, the file that marks the
#                             box), and where /root/bloomery-cpu.lock exists: on the box its locks under BLOOMERY_LOCK_DIR
#                             would not keep it off the box's sittings and gates
#
# The hold on this side. A hold is also a file on this machine, ~/.cache/bloomery/box-hold/<owner>
# (`just box-hold up <owner> [reason]` makes it and `down <owner>` removes it; its content is an optional
# one-line reason). While one is up, a call that is not a read ends 75 before anything is synced, queued or
# sent over ssh, naming the owner, the file's mtime and how to pass. 75 is contention, not a failure: the
# call did not run. Every call reads the directory afresh, so a caller that loops on 75 stops running its
# command at its next call. BLOOMERY_HOLD_OWNER=<o> passes the hold of <o>, BLOOMERY_BOX_READONLY=1 passes
# every hold, and the box-side guard above is unchanged. A file in the directory that is no owner's name,
# or no regular file, ends the call with 64 naming it: a hold that does not hold is never silent. The hold on
# this side refuses new calls; a call already waiting in the guard above waits on the box-side holds.
#
# The box queue. BLOOMERY_BOX_QUEUE=1 makes the command a job of the box's signalbox daemon (unset or 0:
# the path above, byte for byte). The daemon waits on the timing lease and the holds, so the job carries
# no guard; tools/boxq.py runs it (submit, follow its out and err apart, exit with its rc, cancel it on
# INT or TERM). A read (BLOOMERY_BOX_READONLY=1) never goes through the queue.
#   Sync. The tree goes to the remote directory under that directory's lock, /root/bloomery-queue/dir-<key>.lock
#   (<key> the remote directory's name): rsync runs behind `flock -n -x`, so a tree a job is running in is
#   never replaced under it. A sync that fails on that lock ends 83 naming it (the lock is probed once
#   after a failed rsync: rsync's own code is 12 for a lock and a network fault alike). A fresh id goes
#   along as the remote tree's .sync-id, from a temp dir (a file in the Mac tree would make every later
#   command `-dirty` and move the ledger keys that walk the tree).
#   Job. The assembled remote command, without the guard, runs as `flock -s -n -E 85 <the dir lock> bash -c
#   '…'`: the lock is held shared for the job's life, children included, and not taking it ends 85. Inside,
#   a .sync-id that is not this call's id (another sync replaced the tree after the submit) ends 80 before
#   anything runs. box.sh produces 80, 83 and 85 only, and tools/boxq.py passes them through; boxq.py's own
#   are 81, 84 and 130/143 (see its header). A queued job has no stdin, and BLOOMERY_BOX_WAIT and
#   BLOOMERY_HOLD_OWNER do not reach the daemon.
#   BLOOMERY_Q_CLASS=lead|round|sitting|fill   the job's class, which sets its priority (default round)
#   BLOOMERY_Q_CARD=3090|a6000|both|any|none   the lane (default BLOOMERY_CARD, else 3090; both set to
#                             different values: 64). `any` is the daemon's choice of a card: the card pick and
#                             BLOOMERY_BOX_CARD then read $SIGNALBOX_CARD, which the daemon sets in the job's
#                             environment. `none` holds no card and pins none beyond the default
#   BLOOMERY_Q_PRED_S=<s>     the job's predicted wall, whole seconds >= 1; unset prints `predicted unknown`
#                             and the job goes without one. The queue line `box.sh: queue <class> card=<lane>
#                             predicted <s> remote=<dir>` goes to stderr before the submit
set -euo pipefail
# The hold on this side (the header): one file a holder.
BOX_HOLD_DIR=$HOME/.cache/bloomery/box-hold

# hold_stamp <epoch>: the UTC stamp of an epoch second (BSD date -r on the Mac, GNU date -d elsewhere).
hold_stamp() {
  local f=+%Y-%m-%dT%H:%M:%SZ
  { [ "$(uname -s)" = Darwin ] && date -u -r "$1" "$f" 2> /dev/null; } || date -u -d "@$1" "$f"
}

# hold_since <file>: when the hold went up (its mtime, GNU stat then BSD stat) and how long ago.
hold_since() {
  local m s
  m=$(stat -c %Y "$1" 2> /dev/null || stat -f %m "$1" 2> /dev/null || true)
  case $m in
    '' | *[!0-9]*) printf 'at a time that could not be read' ;;
    *)
      s=$(($(date +%s) - m))
      [ "$s" -ge 0 ] || s=0
      printf 'since %s (%dh%02dm%02ds ago)' "$(hold_stamp "$m")" $((s / 3600)) $((s % 3600 / 60)) $((s % 60))
      ;;
  esac
}

# hold_reason <file>: the hold's first line (the optional reason), without control characters, at most 200 of them.
hold_reason() {
  head -n 1 "$1" 2> /dev/null | tr -d '\000-\037' | cut -c1-200 || true
}

# hold_check <owner>: 0 when no hold on this side other than <owner>'s is up. Else each hold is a line on stderr and the
# result is 75; a file of the directory that is no hold's file is 64, naming it.
hold_check() {
  local mine=$1 f o why lines=
  [ -d "$BOX_HOLD_DIR" ] || return 0
  for f in "$BOX_HOLD_DIR"/*; do
    [ -e "$f" ] || [ -L "$f" ] || continue # the pattern itself: the directory is empty
    o=${f##*/}
    case $o in
      *[!A-Za-z0-9_]*)
        echo "box.sh: $f is in the hold directory and is no owner's name (a hold is a file named <owner>: letters, digits, _): remove it" >&2
        return 64
        ;;
    esac
    if [ ! -f "$f" ]; then
      echo "box.sh: $f is in the hold directory and is no regular file: remove it" >&2
      return 64
    fi
    [ "$o" != "$mine" ] || continue
    why=$(hold_reason "$f")
    lines="${lines}box.sh: the box is on hold for $o $(hold_since "$f")${why:+: $why} — rc 75, contention, not a failure: the call did not run, and nothing was synced, queued or sent. $o passes with BLOOMERY_HOLD_OWNER=$o, a read passes with BLOOMERY_BOX_READONLY=1, and \`just box-hold down $o\` lifts it ($f)
"
  done
  [ -z "$lines" ] || { printf '%s' "$lines" >&2; return 75; }
  return 0
}

# hold_cmd up <owner> [reason…] | down <owner>: make or remove the hold's file. An owner that is no name is 64, an `up` of a hold
# that is up leaves it as it is (its time stays the time it went up), and a `down` of a hold that is not there is a named no-op.
hold_cmd() {
  local act=${1:-} o=${2:-} f reason cur
  case $act in
    up | down) ;;
    *) echo "usage: tools/box.sh --hold up <owner> [reason…] | down <owner>" >&2; return 64 ;;
  esac
  case $o in
    '' | *[!A-Za-z0-9_]*) echo "box-hold: an owner is letters, digits and _ (the file $BOX_HOLD_DIR/<owner>), got '$o'" >&2; return 64 ;;
  esac
  f=$BOX_HOLD_DIR/$o
  if [ "$act" = down ]; then
    if [ ! -e "$f" ] && [ ! -L "$f" ]; then
      echo "box-hold: no hold for $o ($f is not there): nothing to take down"
      return 0
    fi
    rm "$f" || return 73
    echo "box-hold: the hold for $o is down"
    return 0
  fi
  shift 2
  reason="$*"
  reason=$(printf '%s' "$reason" | tr '\n\r\t' '   ' | tr -d '\000-\037' | cut -c1-200)
  mkdir -p "$BOX_HOLD_DIR" || return 73
  if (set -C && printf '%s\n' "$reason" > "$f") 2> /dev/null; then
    echo "box-hold: the hold for $o is up ($f): a box.sh call that is not a read ends 75, but BLOOMERY_HOLD_OWNER=$o passes; \`just box-hold down $o\` lifts it"
  elif [ -e "$f" ] || [ -L "$f" ]; then
    cur=$(hold_reason "$f")
    echo "box-hold: the hold for $o is already up $(hold_since "$f")${cur:+, reason: $cur}: left as it is${reason:+ (the reason given now was not written: take it down first to change it)}"
  else
    echo "box-hold: $f cannot be made" >&2
    return 73
  fi
}

# self_test [SCRIPT]: the hold's rules (the header) on a copy of SCRIPT (default: this file) in a scratch tree, under a fake HOME,
# with ssh and rsync stubs ahead on PATH that record their arguments and reach nothing: a call the hold refuses leaves both logs empty.
self_test() {
  local script=${1:-$0} d t box hd fails=0 rc=0 out= n=0 lrc=0 lpid i
  [ -f "$script" ] || { echo "box.sh: --self-test: '$script' is no file" >&2; return 64; }
  d=$(mktemp -d "${TMPDIR:-/tmp}/box-selftest.XXXXXX")
  box=$d/tree/tools/box.sh hd=$d/home/.cache/bloomery/box-hold
  mkdir -p "$d/tree/tools" "$d/stub" "$d/home"
  cp "$script" "$box"
  printf '%s\n' 'cuda-device = { git = "x", rev = "0123456789ab" }' > "$d/tree/Cargo.toml"
  for t in ssh rsync; do
    printf '#!/bin/sh\nprintf "%%s\\n" "$*" >> "%s/%s.log"\n' "$d" "$t" > "$d/stub/$t"
    chmod +x "$d/stub/$t"
  done
  if [ "$(PATH="$d/stub:/usr/bin:/bin"; command -v ssh)" != "$d/stub/ssh" ]; then
    echo "box.sh: --self-test: the ssh stub is not first on PATH: nothing was run" >&2
    rm -rf "$d"
    return 70
  fi
  fail() { echo "box.sh: self-test: $*" >&2; fails=$((fails + 1)); }
  # call [VAR=value…] -- ARGS…: the scratch box.sh with ARGS, in an environment of PATH, HOME, TZ and the VARs only; rc and out are its result.
  call() {
    local envs=()
    while [ "$1" != -- ]; do envs+=("$1"); shift; done
    shift
    : > "$d/ssh.log"
    : > "$d/rsync.log"
    rc=0
    out=$(env -i PATH="$d/stub:/usr/bin:/bin" HOME="$d/home" TZ=UTC BLOOMERY_BOX=stub BLOOMERY_REMOTE='~/repo/x' ${envs[@]+"${envs[@]}"} bash "$box" "$@" 2>&1) || rc=$?
  }
  want() { [ "$rc" = "$1" ] || fail "$2: rc $rc, want $1: $out"; }
  has() { case $out in *"$1"*) ;; *) fail "$2: no '$1' in: $out" ;; esac; }
  hasnt() { case $out in *"$1"*) fail "$2: '$1' in: $out" ;; esac; }
  untouched() { { [ ! -s "$d/ssh.log" ] && [ ! -s "$d/rsync.log" ]; } || fail "$1: ssh or rsync was reached: $(cat "$d/ssh.log" "$d/rsync.log" | head -c 160)"; }
  reached() { grep -q -- 'echo hi' "$d/ssh.log" || fail "$1: the command never reached ssh"; }

  call -- 'echo hi'
  want 0 'no hold directory'; reached 'no hold directory'
  mkdir -p "$hd"
  call -- 'echo hi'
  want 0 'an empty hold directory'; reached 'an empty hold directory'
  call -- --hold up aa 'sitting 3'
  want 0 'hold up aa'; has 'the hold for aa is up' 'hold up aa'; untouched 'hold up aa'
  [ "$(cat "$hd/aa" 2> /dev/null)" = 'sitting 3' ] || fail "hold up aa: the file holds '$(cat "$hd/aa" 2> /dev/null)', not the reason"
  TZ=UTC touch -t 202610110530.00 "$hd/aa"
  # another owner's call ends 75 before any rsync or ssh, naming the owner, since when, the reason and how to pass
  call -- 'echo hi'
  want 75 'a hold'; untouched 'a hold'
  has 'the box is on hold for aa since 2026-10-11T05:30:00Z (' 'a hold names owner and time'
  has '): sitting 3 — rc 75, contention, not a failure' 'a hold says contention and its reason'
  has 'BLOOMERY_HOLD_OWNER=aa' 'a hold says how its owner passes'; has 'BLOOMERY_BOX_READONLY=1' 'a hold says how a read passes'
  has 'just box-hold down aa' 'a hold says how it ends'
  call BLOOMERY_BOX_QUEUE=1 -- 'echo hi'
  want 75 'a queued call'; untouched 'a queued call'
  call BLOOMERY_HOLD_OWNER=aa -- 'echo hi'
  want 0 'the owner'; reached 'the owner'
  call BLOOMERY_HOLD_OWNER=bb -- 'echo hi'
  want 75 'another owner'; untouched 'another owner'; has 'on hold for aa' 'another owner'
  call BLOOMERY_BOX_READONLY=1 -- 'echo hi'
  want 0 'a read'; reached 'a read'
  # two holds: the caller's own is passed, the other is named
  call -- --hold up cc
  want 0 'hold up cc'
  call BLOOMERY_HOLD_OWNER=aa -- 'echo hi'
  want 75 'two holds'; untouched 'two holds'; has 'on hold for cc' 'two holds'; hasnt 'on hold for aa' 'two holds'; has ' ago) — rc 75' 'a hold with no reason'
  call -- --hold down cc
  want 0 'hold down cc'
  # up again leaves the hold as it is; a reason of several lines or with control characters is one clean line
  call -- --hold up aa 'asked again'
  want 0 'hold up again'; has 'already up since 2026-10-11T05:30:00Z' 'hold up again'; has 'reason: sitting 3' 'hold up again'
  [ "$(cat "$hd/aa")" = 'sitting 3' ] || fail "hold up again: the file holds '$(cat "$hd/aa")'"
  call -- --hold down aa
  call -- --hold up aa $'x\ny\001z'
  [ "$(cat "$hd/aa")" = 'x yz' ] || fail "hold up with a messy reason: the file holds '$(cat "$hd/aa")', want one line 'x yz'"
  # down, and down of a hold that is not there
  call -- --hold down aa
  want 0 'hold down aa'; has 'the hold for aa is down' 'hold down aa'
  [ ! -e "$hd/aa" ] || fail 'hold down aa: the file is still there'
  call -- --hold down aa
  want 0 'hold down of nothing'; has 'no hold for aa' 'hold down of nothing'; has 'nothing to take down' 'hold down of nothing'
  call -- 'echo hi'
  want 0 'no hold again'; reached 'no hold again'
  # an owner that is no name: refused, nothing made
  for t in 'a b' 'a/b' '../x' 'a.b' ''; do
    call -- --hold up "$t"
    want 64 "hold up '$t'"; has 'letters, digits and _' "hold up '$t'"
    call -- --hold down "$t"
    want 64 "hold down '$t'"
  done
  [ -z "$(ls -A "$hd")" ] || fail "a refused hold up made $(ls -A "$hd")"
  call BLOOMERY_HOLD_OWNER='a b' -- 'echo hi'
  want 64 'a malformed BLOOMERY_HOLD_OWNER'; untouched 'a malformed BLOOMERY_HOLD_OWNER'; has 'letters, digits' 'a malformed BLOOMERY_HOLD_OWNER'
  # a file or directory of the hold directory that is no hold: 64 naming it; a dotfile is not looked at
  : > "$hd/stray.txt"
  call -- 'echo hi'
  want 64 'a stray file'; untouched 'a stray file'; has "$hd/stray.txt" 'a stray file'; has 'no owner' 'a stray file'
  rm "$hd/stray.txt"
  mkdir "$hd/dd"
  call -- 'echo hi'
  want 64 'a directory'; untouched 'a directory'; has 'no regular file' 'a directory'
  rmdir "$hd/dd"
  : > "$hd/.DS_Store"
  call -- 'echo hi'
  want 0 'a dotfile'; reached 'a dotfile'
  rm "$hd/.DS_Store"
  # a caller that loops on its command: the call after the hold goes up is the first one refused, and nothing runs after it
  : > "$d/ssh.log"
  (
    n=0
    lrc=0
    while [ "$n" -lt 15 ]; do
      env -i PATH="$d/stub:/usr/bin:/bin" HOME="$d/home" TZ=UTC BLOOMERY_BOX=stub BLOOMERY_REMOTE='~/repo/x' bash "$box" 'echo hi' > /dev/null 2>&1 || { lrc=$?; break; }
      n=$((n + 1))
      sleep 0.2
    done
    echo "$lrc $n" > "$d/loop"
  ) &
  lpid=$!
  i=0
  until grep -q 'echo hi' "$d/ssh.log"; do
    i=$((i + 1))
    [ "$i" -lt 100 ] || break
    sleep 0.1
  done
  printf '%s\n' 'loop' > "$hd/aa"
  wait "$lpid" || true
  read -r lrc n < "$d/loop" || { lrc=?; n=?; }
  [ "$lrc" = 75 ] || fail "a looping caller ended $lrc after $n runs, not 75 at its next call"
  [ "$n" = "$(grep -c 'echo hi' "$d/ssh.log")" ] || fail "a looping caller ran $n times and ssh got the command $(grep -c 'echo hi' "$d/ssh.log") times"
  rm -rf "$d"
  if [ "$fails" != 0 ]; then
    echo "box.sh: self-test $fails failed" >&2
    return 1
  fi
  echo "box.sh: self-test ok"
}

case ${1:-} in
  --hold) shift; rc=0; hold_cmd "$@" || rc=$?; exit "$rc" ;;
  --self-test) shift; rc=0; self_test "$@" || rc=$?; exit "$rc" ;;
esac
HOST=${BLOOMERY_BOX:-ws}
REMOTE=${BLOOMERY_REMOTE:-"~/repo/$(basename "$(cd "$(dirname "$0")/.." && pwd)")"}
HERE=$(cd "$(dirname "$0")/.." && pwd)
BOX_WAIT=${BLOOMERY_BOX_WAIT:-1800}
case $BOX_WAIT in
  '' | *[!0-9]*) echo "box.sh: BLOOMERY_BOX_WAIT is whole seconds (0: do not wait), got '$BOX_WAIT'" >&2; exit 64 ;;
esac
HOLD_OWNER=${BLOOMERY_HOLD_OWNER:-}
case $HOLD_OWNER in
  *[!A-Za-z0-9_]*) echo "box.sh: BLOOMERY_HOLD_OWNER is the <owner> of /root/bloomery-<owner>-hold (letters, digits, _), got '$HOLD_OWNER'" >&2; exit 64 ;;
esac
# The tier (the header), named before anything is synced: a refusal touches nothing on the box.
TIER_MAC=${BLOOMERY_TIER:-} TIER_BOX=
read -r -a tier_env <<< "${BLOOMERY_BOX_ENV:-}"
for kv in ${tier_env[@]+"${tier_env[@]}"}; do
  [ "${kv%%=*}" != BLOOMERY_TIER ] || TIER_BOX=${kv#*=}
done
for t in "$TIER_MAC" "$TIER_BOX"; do
  case "$t" in
    '' | real | fixture) ;;
    *) echo "box.sh: BLOOMERY_TIER is real or fixture (unset: real), got '$t'" >&2; exit 64 ;;
  esac
done
if [ -n "$TIER_MAC" ] && [ -n "$TIER_BOX" ] && [ "$TIER_MAC" != "$TIER_BOX" ]; then
  echo "box.sh: BLOOMERY_TIER is '$TIER_MAC' in the environment and '$TIER_BOX' in BLOOMERY_BOX_ENV: name one" >&2
  exit 64
fi
TIER=${TIER_BOX:-${TIER_MAC:-real}}
# Exported ahead of the profile below, so ref-paths.sh sees it; nothing is added for a command that names none.
TIERX=
[ -z "$TIER_MAC$TIER_BOX" ] || TIERX="export BLOOMERY_TIER=$TIER && "
# The fixture variant (the header), read the same way: a BLOOMERY_BOX_ENV entry is exported after the profile, too late for the file it names.
VAR_MAC=${BLOOMERY_FIXTURE_VARIANT:-} VAR_BOX=
for kv in ${tier_env[@]+"${tier_env[@]}"}; do
  [ "${kv%%=*}" != BLOOMERY_FIXTURE_VARIANT ] || VAR_BOX=${kv#*=}
done
if [ -n "$VAR_MAC" ] && [ -n "$VAR_BOX" ] && [ "$VAR_MAC" != "$VAR_BOX" ]; then
  echo "box.sh: BLOOMERY_FIXTURE_VARIANT is '$VAR_MAC' in the environment and '$VAR_BOX' in BLOOMERY_BOX_ENV: name one" >&2
  exit 64
fi
VARIANT=${VAR_BOX:-$VAR_MAC}
case "$VARIANT" in
  *[!a-z0-9]*) echo "box.sh: BLOOMERY_FIXTURE_VARIANT is a tag of tools/fixture-variants.tsv, lower-case letters and digits, got '$VARIANT'" >&2; exit 64 ;;
esac
if [ -n "$VARIANT" ] && [ "$TIER" != fixture ]; then
  echo "box.sh: BLOOMERY_FIXTURE_VARIANT=$VARIANT, but the tier is $TIER: a variant is a requantized fixture file, and nothing runs on a real file under a variant's name" >&2
  exit 64
fi
VARX=
[ -z "$VARIANT" ] || VARX="export BLOOMERY_FIXTURE_VARIANT=$VARIANT && "
# The arch (the header), named before anything is synced: a refusal touches nothing on the box.
ARCHX='' ARCH_ENTRY='' RECIPE_ARCH=sm_86
ARCH_SEEN=0
for kv in ${tier_env[@]+"${tier_env[@]}"}; do
  if [ "${kv%%=*}" = BLOOMERY_ARCH ]; then
    ARCH_SEEN=1 ARCH_ENTRY=${kv#*=}
  fi
done
if [ -n "${BLOOMERY_ARCH+x}" ] || [ "$ARCH_SEEN" = 1 ]; then
  # shellcheck source=tools/arch.sh
  source "$HERE/tools/arch.sh"
  bloomery_arch box.sh || exit $?
  RECIPE_ARCH=$ARCH
  [ -z "${BLOOMERY_ARCH+x}" ] || ARCHX="export BLOOMERY_ARCH=$ARCH && "
  if [ "$ARCH_SEEN" = 1 ] && [ "$ARCH_ENTRY" != "$ARCH" ]; then
    echo "box.sh: BLOOMERY_ARCH is '$ARCH_ENTRY' in BLOOMERY_BOX_ENV and '$ARCH' in the environment (unset: sm_86): name it in the environment, where the recipes read it" >&2
    exit 64
  fi
fi
QUEUE=0
case ${BLOOMERY_BOX_QUEUE:-0} in
  0) ;;
  1) QUEUE=1 ;;
  *) echo "box.sh: BLOOMERY_BOX_QUEUE is 1 (a job of the box queue), or 0 or unset (the command runs directly), got '$BLOOMERY_BOX_QUEUE'" >&2; exit 64 ;;
esac
# Local mode (the header): the command's directory is this tree, and nothing below syncs or ssh's.
LOCAL=${BLOOMERY_BOX_LOCAL:-0}
# shellcheck disable=SC2088 # the box's shell expands the '~'
DIR=$REMOTE ENVF='~/bloomery-env.sh'
case $LOCAL in
  0) ;;
  1)
    if [ "$QUEUE" = 1 ]; then
      echo "box.sh: BLOOMERY_BOX_QUEUE=1 makes the command a job of the box's queue, and BLOOMERY_BOX_LOCAL=1 runs it here: name one" >&2
      exit 64
    fi
    if [ "$(uname -s)" != Linux ]; then
      echo "box.sh: BLOOMERY_BOX_LOCAL=1 runs the box's commands here, and they need Linux (this is $(uname -s)); a Mac runs the static tier, just mac-static" >&2
      exit 64
    fi
    # shellcheck source=tools/ref/lease-probe.sh
    . "$HERE/tools/ref/lease-probe.sh"
    if on_box; then
      echo "box.sh: BLOOMERY_BOX_LOCAL=1 on the machine whose lease is $BOX_LEASE (the box): local mode's locks under BLOOMERY_LOCK_DIR would not keep its gates off the box's own; run box commands from another machine. If this machine is not the box, a lock script run as root without BLOOMERY_LOCK_DIR made that file (flock creates the file it probes): remove it" >&2
      exit 64
    fi
    LOCAL_ENV=${BLOOMERY_ENV_FILE:-$HOME/bloomery-env.sh}
    if [ ! -f "$LOCAL_ENV" ]; then
      echo "box.sh: BLOOMERY_BOX_LOCAL=1 sources the environment file BLOOMERY_ENV_FILE (default ~/bloomery-env.sh), and '$LOCAL_ENV' is not there: write the toolchain block of docs/BUILD.md into it" >&2
      exit 69
    fi
    LOCAL_ENV=$(cd "$(dirname "$LOCAL_ENV")" && pwd)/${LOCAL_ENV##*/}
    LOCK_DIR=${BLOOMERY_LOCK_DIR:-$HOME/.cache/bloomery/locks}
    case $LOCK_DIR in
      /*) ;;
      *) echo "box.sh: BLOOMERY_LOCK_DIR '$LOCK_DIR' is relative, and every tree on this machine has to take its locks in one directory: name it from / (default ~/.cache/bloomery/locks)" >&2; exit 64 ;;
    esac
    if box_lock_dir "$LOCK_DIR"; then
      echo "box.sh: BLOOMERY_LOCK_DIR=$LOCK_DIR is the box's lock directory: local mode's guard would make $BOX_LEASE there, the file that marks the box, and refuse every later call; name another (default ~/.cache/bloomery/locks)" >&2
      exit 64
    fi
    mkdir -p "$LOCK_DIR" || { echo "box.sh: BLOOMERY_LOCK_DIR '$LOCK_DIR' cannot be made" >&2; exit 69; }
    DIR=$(printf %q "$HERE") ENVF=$(printf %q "$LOCAL_ENV")
    ;;
  *) echo "box.sh: BLOOMERY_BOX_LOCAL is 1 (run here) or 0/unset (run on the box), got '$LOCAL'" >&2; exit 64 ;;
esac
READONLY=0
case ${BLOOMERY_BOX_READONLY:-0} in
  0) GUARD="( cd $DIR && . tools/ref/lease-probe.sh && lease_guard $BOX_WAIT $HOLD_OWNER ) && " ;;
  1)
    READONLY=1
    GUARD=
    ro_build='(^|[^A-Za-z0-9_])(cargo|just|make|cmake|ninja)([^A-Za-z0-9_]|$)'
    if [[ $* =~ $ro_build ]]; then
      echo "box.sh: BLOOMERY_BOX_READONLY=1 is for reads, and this command names '${BASH_REMATCH[2]}': run it without the opt-in, behind the guard" >&2
      exit 64
    fi
    case $* in
      *target/*) echo "box.sh: BLOOMERY_BOX_READONLY=1 is for reads, and this command names 'target/' (one of our binaries): run it without the opt-in, behind the guard" >&2; exit 64 ;;
    esac
    ;;
  *) echo "box.sh: BLOOMERY_BOX_READONLY is 1 (a read), or 0 or unset (behind the guard), got '$BLOOMERY_BOX_READONLY'" >&2; exit 64 ;;
esac
# A read never goes through the queue. The queue's settings (the header) are named before anything is synced: a refusal touches nothing.
[ "$READONLY" = 0 ] || QUEUE=0
QLOCK_DIR=/root/bloomery-queue
QKEY= QLOCK= QCLASS= QLANE= QPRED=
if [ "$QUEUE" = 1 ]; then
  QKEY=$(basename "$REMOTE")
  case $QKEY in
    '' | *[!A-Za-z0-9._-]*) echo "box.sh: the remote directory's name is the dir lock's key, [A-Za-z0-9._-]+, got '$QKEY' (BLOOMERY_REMOTE=$REMOTE)" >&2; exit 64 ;;
  esac
  QLOCK=$QLOCK_DIR/dir-$QKEY.lock
  QCLASS=${BLOOMERY_Q_CLASS:-round}
  case $QCLASS in
    lead | round | sitting | fill) ;;
    *) echo "box.sh: BLOOMERY_Q_CLASS is lead, round, sitting or fill, got '$QCLASS'" >&2; exit 64 ;;
  esac
  QPRED=${BLOOMERY_Q_PRED_S:-}
  case $QPRED in
    '') ;;
    0* | *[!0-9]*) echo "box.sh: BLOOMERY_Q_PRED_S is whole seconds >= 1 (unset: predicted unknown), got '$QPRED'" >&2; exit 64 ;;
  esac
  case ${BLOOMERY_CARD:-3090} in
    3090 | a6000 | both) ;;
    *) echo "box.sh: BLOOMERY_CARD must be 3090, a6000 or both (BLOOMERY_Q_CARD also takes any and none)" >&2; exit 64 ;;
  esac
  QLANE=${BLOOMERY_CARD:-3090}
  if [ -n "${BLOOMERY_Q_CARD:-}" ]; then
    case $BLOOMERY_Q_CARD in
      3090 | a6000 | both | any | none) ;;
      *) echo "box.sh: BLOOMERY_Q_CARD must be 3090, a6000, both, any or none, got '$BLOOMERY_Q_CARD'" >&2; exit 64 ;;
    esac
    if [ -n "${BLOOMERY_CARD:-}" ] && [ "$BLOOMERY_CARD" != "$BLOOMERY_Q_CARD" ]; then
      echo "box.sh: BLOOMERY_CARD is '$BLOOMERY_CARD' and BLOOMERY_Q_CARD is '$BLOOMERY_Q_CARD': name one lane" >&2
      exit 64
    fi
    QLANE=$BLOOMERY_Q_CARD
  fi
fi
# The hold on this side (the header): a read passes it, and a refused call has synced, queued and sent nothing.
if [ "$READONLY" = 0 ]; then
  hold_rc=0
  hold_check "$HOLD_OWNER" || hold_rc=$?
  [ "$hold_rc" = 0 ] || exit "$hold_rc"
fi
if [ "$LOCAL" = 1 ]; then
  :
elif [ "$READONLY" = 1 ]; then
  # A read runs in the remote directory as it is. -n: this lookup leaves stdin to the command's ssh.
  rc=0
  ssh -n "$HOST" "test -d $REMOTE" || rc=$?
  case $rc in
    0) ;;
    1) echo "box.sh: BLOOMERY_BOX_READONLY=1 runs the command in $REMOTE as it is and syncs nothing, and $HOST has no $REMOTE: nothing was synced there yet" >&2; exit 66 ;;
    *) echo "box.sh: ssh $HOST failed looking for $REMOTE (rc $rc)" >&2; exit "$rc" ;;
  esac
else
  ssh -n "$HOST" "mkdir -p $REMOTE"
  # 시각은 싣지 않는다(-t 없음) — 바뀐 파일은 내용 체크섬(-c)으로 고르고, 박스에 닿은 파일의 mtime은 박스 시계의 "지금"이 된다.
  # 맥의 mtime을 그대로 실으면 cargo가 낡은 바이너리를 내준다: 박스 시계가 맥보다 앞서 있어(실측 4.1초) 복원 직후의 touch조차
  # 직전 빌드 산출물보다 과거로 찍힌다(변이 바이너리가 두 번 그대로 돌았다).
  # 원격 루트의 *.ptx·*.ll은 cargo oxide가 빌드 중에 쓰는 산출물이다(`bloomery_gpu_deepseek41.ptx`, `….linked.opt.ll`) — 같은
  # 원격 디렉터리에서 빌드가 도는 사이 다른 box.sh 호출의 --delete가 그것을 지우면 빌드가 rc 101로 죽는다(dspark-q3k와
  # ds41splitk 라운드에서 한 번씩). `.oxide-artifacts/`(`embed.o` 등)도 같은 부류다(q3fix 라운드에서 한 번). 맥 트리에는 없으니
  # 삭제 대상에서 뺀다.
  # A queued call syncs behind the remote directory's lock and ships a fresh id with the tree (the header); the direct call adds neither.
  SYNC_ID= SYNC_TMP= SYNC_PATH=() SYNC_ID_SRC=()
  if [ "$QUEUE" = 1 ]; then
    SYNC_ID=$(od -An -N8 -tx1 /dev/urandom | tr -d ' \n')
    SYNC_TMP=$(mktemp -d "${TMPDIR:-/tmp}/box-sync.XXXXXX")
    printf '%s\n' "$SYNC_ID" > "$SYNC_TMP/.sync-id"
    SYNC_PATH=(--rsync-path="mkdir -p $QLOCK_DIR && flock -n -x $QLOCK rsync")
    SYNC_ID_SRC=("$SYNC_TMP/.sync-id")
  fi
  rc=0
  rsync -rlpgoDcz --delete --exclude target/ --exclude .git/ --exclude '/*.ptx' --exclude '/*.ll' --exclude '/.oxide-artifacts/' ${SYNC_PATH[@]+"${SYNC_PATH[@]}"} "$HERE"/ ${SYNC_ID_SRC[@]+"${SYNC_ID_SRC[@]}"} "$HOST:$REMOTE/" || rc=$?
  [ -z "$SYNC_TMP" ] || rm -rf "$SYNC_TMP"
  if [ "$rc" != 0 ] && [ "$QUEUE" = 1 ]; then
    # rsync's rc is 12 for a held lock and for a dead ssh alike: the lock decides (flock: 1 held, 0 free).
    prc=0
    ssh -n "$HOST" "mkdir -p $QLOCK_DIR && flock -n -x $QLOCK true" || prc=$?
    if [ "$prc" = 1 ]; then
      echo "box.sh: a job of this remote directory is running ($QLOCK is held on $HOST); its tree cannot be replaced under it: wait for it, or use another BLOOMERY_REMOTE" >&2
      exit 83
    fi
  fi
  [ "$rc" = 0 ] || exit "$rc"
fi
# 카드 선택. 기본은 env 파일의 3090 핀 그대로. BLOOMERY_CARD=a6000|both는 박스에서 이름으로 UUID를 찾아
# CUDA_VISIBLE_DEVICES를 덮어쓴다(both = 3090 먼저 → 디바이스 0이 3090). 두 카드 다 우리 것이다(야간 학습은
# 2026-09-21에 끝났고 llm.service는 꺼져 있다). A compute process on the A6000 with its gate lock and the timing
# lease both free is not one of ours: the command ends with rc 75 (below). llm.service 검사는 누가 다시 켰을 때의
# 안전장치다.
# 고른 값은 박스 쪽에 BLOOMERY_BOX_CARD로 넘어가고, tools/gpu-gate.sh가 그 카드의 게이트 락을 잡는다(both = 둘 다).
# card_pick <card>: the remote text that picks that card (3090, a6000 or both); a card of another name is refused (64).
card_pick() {
  case "$1" in
    3090) printf %s ":" ;;
    a6000|both) printf %s '
    A=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | grep "A6000" | cut -d, -f1)
    T=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | grep "3090" | cut -d, -f1)
    # a6000 alone needs only its own UUID — the 3090 has fallen off the bus twice on 2026-09-22 and must not
    # take the healthy card down with it. both needs both.
    [ -n "$A" ] || { echo "box.sh: A6000 lookup failed" >&2; exit 75; }
    [ "'"$1"'" != both ] || [ -n "$T" ] || { echo "box.sh: 3090 lookup failed (both)" >&2; exit 75; }
    if [ "$(systemctl is-active llm.service)" = active ]; then echo "box.sh: llm.service holds the A6000" >&2; exit 75; fi
    # A compute process on the A6000 while its gate lock or the timing lease is held is one of our runs: the
    # command goes on, and tools/gpu-gate.sh waits for the card lock inside its bound. One with both free is
    # a surprise (serving or training) and ends the command with 75. Shared probes: they never hold a lock.
    if [ -n "$(nvidia-smi -i "$A" --query-compute-apps=pid --format=csv,noheader)" ] &&
       flock -s -n /root/bloomery-gate-a6000.lock true && flock -s -n /root/bloomery-cpu.lock true; then
      echo "box.sh: the A6000 has compute processes and neither its gate lock nor the timing lease is held (serving or training?) — not taking it" >&2; exit 75; fi
    '"$( [ "$1" = both ] && echo 'export CUDA_VISIBLE_DEVICES="$T,$A"' || echo 'export CUDA_VISIBLE_DEVICES="$A"' )" ;;
    *) echo "box.sh: BLOOMERY_CARD must be 3090, a6000 or both" >&2; exit 64 ;;
  esac
}
CARD=${BLOOMERY_CARD:-3090}
# A queue lane (BLOOMERY_Q_CARD) of `any` or `none` takes today's default pin here; `any` then picks its card when the job starts (below).
case "$QLANE" in
  '') ;;
  any | none) CARD=3090 ;;
  *) CARD=$QLANE ;;
esac
PICK=$(card_pick "$CARD")
BOXCARD=$CARD
if [ "$QLANE" = any ]; then
  PICK="case \"\${SIGNALBOX_CARD:-}\" in
  3090) : ;;
  a6000) $(card_pick a6000)
  ;;
  *) echo \"box.sh: a job of lane any starts with SIGNALBOX_CARD 3090 or a6000 (the daemon sets it), got '\${SIGNALBOX_CARD:-}'\" >&2; exit 70 ;;
esac"
  BOXCARD='"$SIGNALBOX_CARD"'
fi
if [ "$LOCAL" = 1 ]; then
  if [ "$CARD" = both ]; then
    echo "box.sh: BLOOMERY_CARD=both puts the box's two cards in view, and BLOOMERY_BOX_LOCAL=1 runs on what this machine shows: a two-card run is the box's" >&2
    exit 64
  fi
  PICK=":" BOXCARD=local
fi
# The model the gates and GPU binaries open (BLOOMERY_REF_MODEL) is a property of the tool profile:
# every command runs with it exported from tools/ref/ref-paths.sh, and an unknown profile stops the
# command with that file's exit 64. The profile is picked on this side — BLOOMERY_MODEL here, or
# ref-paths.sh's default — and its name goes along as BLOOMERY_REF_MODEL_PROFILE, which the scripts
# inside take as their profile; one that picks another is refused (ref-paths.sh, exit 64), since
# the export would not follow it. BLOOMERY_MODEL itself is not exported: the model crate's test
# harnesses read that name as a model path. A caller's own BLOOMERY_REF_MODEL wins: one set on this
# side is carried over, one set inside the command overrides the export.
FWD=
if [ -n "${BLOOMERY_REF_MODEL:-}" ]; then
  FWD="export BLOOMERY_REF_MODEL=$(printf %q "$BLOOMERY_REF_MODEL") && "
fi
MODEL_PICK=
if [ -n "${BLOOMERY_MODEL:-}" ]; then
  MODEL_PICK="BLOOMERY_MODEL=$(printf %q "$BLOOMERY_MODEL") && "
fi
PROFILE="__p=\$(${MODEL_PICK}. tools/ref/ref-paths.sh && printf %s \"\$BLOOMERY_MODEL\") && export BLOOMERY_REF_MODEL_PROFILE=\"\$__p\" && __m=\$(. tools/ref/ref-paths.sh && printf %s \"\$MODEL\") && export BLOOMERY_REF_MODEL=\"\$__m\" && unset __p __m"
# The fixture tier also takes the fixture file's card budget and names the file, in one more read of ref-paths.sh each; a family
# whose real file stands (FIXTURE_FILE empty) exports neither, and a file that is not a whole fixture stops the command (65).
if [ "$TIER" = fixture ]; then
  PROFILE="$PROFILE && __f=\$(. tools/ref/ref-paths.sh && printf %s \"\$FIXTURE_FILE\") && __b=\$(. tools/ref/ref-paths.sh && fixture_budget) && { [ -z \"\$__f\" ] || export BLOOMERY_FIXTURE_MODEL=\"\$__f\" BLOOMERY_CARD_BUDGET=\"\$__b\"; } && unset __f __b"
fi
# The V4.1 file (BLOOMERY_V41_MODEL, and its directory as BLOOMERY_V41_DIR) is exported into every command
# whatever profile it picked: the deepseek41 profile owns the choice (its V41_MODEL), and the crates and
# scripts that open V4.1 under another profile (the tokenizer, engram, qdot and placement tests, the
# tokenizer and engram runners) read the export. A caller's own BLOOMERY_V41_MODEL is carried over and wins.
V41=
if [ -n "${BLOOMERY_V41_MODEL:-}" ]; then
  V41="export BLOOMERY_V41_MODEL=$(printf %q "$BLOOMERY_V41_MODEL") && "
fi
V41="${V41}__v=\$(. tools/ref/models/deepseek41.sh && printf %s \"\$V41_MODEL\") && export BLOOMERY_V41_MODEL=\"\$__v\" BLOOMERY_V41_DIR=\"\${__v%/*}\" && unset __v"
# The data directory (BLOOMERY_DATA) is exported into every command the same way: its default is
# ref-paths.sh's, read on the box from the synced tree, so no copy of it lives here. A caller's own
# BLOOMERY_DATA is carried over and wins.
DATA=
if [ -n "${BLOOMERY_DATA:-}" ]; then
  DATA="export BLOOMERY_DATA=$(printf %q "$BLOOMERY_DATA") && "
fi
DATA="${DATA}__d=\$(. tools/ref/ref-paths.sh && printf %s \"\$BLOOMERY_DATA\") && export BLOOMERY_DATA=\"\$__d\" && unset __d"
# BLOOMERY_BOX_ENV="NAME=value NAME2=value2" exports those variables into the command, so a lever arm
# reaches the binary through an unchanged recipe (tools/gpu-ab.py's env arms). Entries are split on
# spaces and quote nothing; each value is quoted for the remote shell. So a value with a space cannot
# cross: an entry that holds a quote, or one that is not an upper-case NAME=value (the second half of a
# spaced value), is refused by name — such a value goes in the command itself, or in an arm word (a
# server arm's +t<N>, +nopo<0|1>, +k<K>: tools/ref/lcpp-warm.sh).
ENVS=
read -r -a box_env <<< "${BLOOMERY_BOX_ENV:-}"
prev=
spaced="BLOOMERY_BOX_ENV splits on spaces and quotes nothing, so a value with a space does not cross it; put it in the command, or in an arm word (a server arm's +t<N>, +nopo<0|1>, +k<K>)"
for kv in ${box_env[@]+"${box_env[@]}"}; do
  name=${kv%%=*} after=
  [ -z "$prev" ] || after=" (after '$prev': the rest of a value with a space? $spaced)"
  case "$kv" in
    *[\"\']*) echo "box.sh: BLOOMERY_BOX_ENV entry '$kv' holds a quote: $spaced" >&2; exit 64 ;;
    *=*) ;;
    *) echo "box.sh: BLOOMERY_BOX_ENV entries are NAME=value, got '$kv'$after" >&2; exit 64 ;;
  esac
  case "$name" in
    '' | [0-9]* | *[![:upper:][:digit:]_]*) echo "box.sh: '$name' in BLOOMERY_BOX_ENV is not an upper-case variable name$after" >&2; exit 64 ;;
  esac
  ENVS="${ENVS}export $name=$(printf %q "${kv#*=}") && "
  prev=$kv
done
# 트랙 원격 디렉터리(BLOOMERY_REMOTE)의 빌드는 CARGO_BUILD_JOBS 기본 12: 파동의 4중 빌드가 32코어를 스래싱하지 않게.
# BLOOMERY_BOX_ENV로 주면 그 값이고, 메인 트리(BLOOMERY_REMOTE 없음)는 그대로다.
if [ "$LOCAL" = 0 ] && [ -n "${BLOOMERY_REMOTE:-}" ] && [[ " ${box_env[*]+${box_env[*]}} " != *" CARGO_BUILD_JOBS="* ]]; then
  ENVS="${ENVS}export CARGO_BUILD_JOBS=12 && "
fi
# cuda-oxide 백엔드(librustc_codegen_cuda.so)는 핀 rev마다 제 디렉터리에 둔다: ~/.cargo/cuda-oxide-bloomery/<rev>/.
# cargo-oxide의 기본 캐시(~/.cargo/cuda-oxide/)는 소스가 새로워 보이면 그 자리에서 다시 빌드하고, 그동안 다른 트랙의
# rustc가 그 .so를 mmap한 채 SIGBUS를 맞는다(nvlabs-ledger 19). rev로 고정한 경로의 파일은 한 번 놓이면 바뀌지 않는다.
# rev는 Cargo.toml [patch."https://github.com/NVIDIA/cuda-rust.git"]의 포크 rev이고, 그 절이 없으면
# [workspace.dependencies]의 rev다. cargo oxide를 부르는 명령은 그 파일이 없거나 옆의 source-rev.txt가 그 rev가 아니면 이름을 대고 멈춘다(rc 70) — CUDA_OXIDE_BACKEND로
# 고정하면 cargo-oxide는 백엔드와 의존성의 커밋을 대조하지 않으므로 그 대조는 여기가 한다. 새 rev의 백엔드는 그 rev의 체크아웃(~/.cargo/git/checkouts/
# cuda-oxide-*/<rev>)의 cuda-oxide/crates/rustc-codegen-cuda를 빌드해 같은 디렉터리의 임시 파일로 복사한 뒤 mv로 놓고, 그
# 체크아웃의 커밋을 source-rev.txt에 쓴다. 포크 커밋이 백엔드 크레이트의 의존 폐포를 건드리지 않은 핀 이동은 앞 rev의
# 백엔드를 같은 방법(임시 파일 + mv)으로 복사해 써도 된다. 그때는 사본의 md5와 근거를 그 디렉터리의 PROVENANCE에 적고,
# source-rev.txt는 그 .so를 지은 커밋이 아니라 그 .so가 섬기는 rev를 적는다.
OXREV=$(sed -n '/^\[patch\."https:\/\/github.com\/NVIDIA\/cuda-rust.git"\]/,/^\[workspace/s/^cuda-device = .*rev = "\([0-9a-f]*\)".*/\1/p' "$HERE/Cargo.toml")
[ -n "$OXREV" ] || OXREV=$(sed -n 's/^cuda-device = .*rev = "\([0-9a-f]*\)".*/\1/p' "$HERE/Cargo.toml" | head -1)
[ -n "$OXREV" ] || { echo "box.sh: no cuda-oxide rev in $HERE/Cargo.toml" >&2; exit 70; }
OXIDE="export CUDA_OXIDE_BACKEND=\$HOME/.cargo/cuda-oxide-bloomery/$OXREV/librustc_codegen_cuda.so && "
# PATH의 cargo-oxide도 그 핀 rev여야 한다: 더 오래된 cargo-oxide는 .cargo/cuda-oxide.toml의 [target.<triple>] 표를
# 아무 말 없이 무시해서, 호스트 코드가 CPU 플래그 없이 빌드된다. `cargo install --list`는 git 설치의 rev를 적는다
# (`cargo-oxide vX (https://...?rev=<rev>#<short>)`). 그 rev가 핀이 아니거나 rev가 없으면 이름을 대고 멈춘다(rc 70).
OXCLI='{ __c=$(cargo install --list 2>/dev/null | sed -n "s/^cargo-oxide v[^ ]* (.*#\([0-9a-f]\{7,\}\)).*/\1/p" | head -1); case OXREV in "$__c"*) [ -n "$__c" ] ;; *) false ;; esac || { echo "box.sh: the installed cargo-oxide is from rev ${__c:-none}, not the pin OXREV: reinstall it (docs/BUILD.md, cargo install ... --rev OXREV cargo-oxide)" >&2; exit 70; }; unset __c; } && '
OXCLI=${OXCLI//OXREV/$OXREV}
case "$*" in
  *"cargo oxide"*) OXIDE="${OXIDE}{ [ -f \"\$CUDA_OXIDE_BACKEND\" ] || { echo \"box.sh: no cuda-oxide backend for rev $OXREV at \$CUDA_OXIDE_BACKEND (tools/box.sh header)\" >&2; exit 70; }; } && { __r=\$(cat \"\${CUDA_OXIDE_BACKEND%/*}/source-rev.txt\" 2>/dev/null); [ \"\$__r\" = $OXREV ] || { echo \"box.sh: the backend at \$CUDA_OXIDE_BACKEND records source rev '\$__r', not $OXREV\" >&2; exit 70; }; unset __r; } && " ;;
esac
# Local mode unsets the environment file's backend: cargo oxide builds and keeps its own (the header).
[ "$LOCAL" = 0 ] || OXIDE="unset CUDA_OXIDE_BACKEND && "
# The guard probed LOCK_DIR and the recipe expanded its arch before the environment file and BLOOMERY_BOX_ENV ran; one of
# them moving either would put the command's locks where the guard did not look, or build for another arch.
LOCALX=
if [ "$LOCAL" = 1 ]; then
  LOCALX="{ [ \"\${BLOOMERY_LOCK_DIR-}\" = $(printf %q "$LOCK_DIR") ] || { echo \"box.sh: the environment file or BLOOMERY_BOX_ENV set BLOOMERY_LOCK_DIR after local mode's guard probed the lock directory; set it before tools/box.sh\" >&2; exit 64; }; } && "
  LOCALX+="{ [ \"\${BLOOMERY_ARCH:-sm_86}\" = $RECIPE_ARCH ] || { echo \"box.sh: the environment file set BLOOMERY_ARCH=\$BLOOMERY_ARCH, and the recipe expanded $RECIPE_ARCH (this side's BLOOMERY_ARCH, unset: sm_86): name it in the environment, where the recipes read it\" >&2; exit 64; }; } && "
fi
# cargo oxide를 스크립트 안에서 부르는 명령(gate.sh --oxide, release/build.sh)도 같은 확인을 거친다.
# release/build.sh --check는 바이너리만 읽고 빌드하지 않으므로 거치지 않는다. 그 부분만 빼고 보므로,
# 같은 명령 안의 다른 release/build.sh 빌드는 여전히 확인을 거친다.
# 확인은 BLOOMERY_BOX_ENV 뒤에 돈다: 거기서 CARGO_HOME이나 PATH를 바꾸면 빌드가 쓰는 그 설치를 확인한다.
# The same commands take their llc from tools/oxide-llc.sh, which holds the LLVM major (21) and the pick: it is exported as
# CUDA_OXIDE_LLC and named in one line, and an llc of another major, the Rust toolchain's own above all, is refused by
# name (rc 70) before cuda-oxide's own order can reach it. Local mode runs the same check on this machine.
OXLLC='{ __l=$(bash tools/oxide-llc.sh) || exit $?; export CUDA_OXIDE_LLC=$__l; unset __l; } && '
OXCHECK=
OXCMD=$*
case "${OXCMD//release\/build.sh --check/}" in
  *"cargo oxide"* | *"gate.sh --oxide"* | *"release/build.sh"*) OXCHECK=$OXCLI$OXLLC ;;
esac
# serve의 build.rs가 `/props`의 version에 새기는 커밋. 박스 사본에는 .git이 없어서(rsync가 뺀다) 여기서 넘긴다.
# 이 트리에 커밋에 없는 변경이 있으면 `-dirty`를 붙인다 — 그 바이너리는 그 커밋의 것이 아니다.
COMMIT=$(git -C "$HERE" rev-parse --short=8 HEAD 2>/dev/null || echo unknown)
[ -z "$(git -C "$HERE" status --porcelain 2>/dev/null | head -1)" ] || COMMIT="$COMMIT-dirty"
REMOTE_CMD="source $ENVF && { $PICK
} && cd $DIR && $V41 && $TIERX$VARX$FWD$PROFILE && $DATA && export BLOOMERY_GIT_COMMIT=$COMMIT BLOOMERY_BOX_CARD=$BOXCARD && $OXIDE$ARCHX$ENVS$OXCHECK$LOCALX$*"
if [ "$LOCAL" = 1 ]; then
  # ssh carries no environment to the box: the command runs without what this side read, BLOOMERY_MODEL above all (the model
  # crate's tests read that name as a model path), without box.sh's own outputs, which only the profile sets, and without
  # the lease's file names, which BLOOMERY_LOCK_DIR sets.
  exec env -u BLOOMERY_BOX -u BLOOMERY_REMOTE -u BLOOMERY_BOX_WAIT -u BLOOMERY_HOLD_OWNER -u BLOOMERY_BOX_ENV -u BLOOMERY_BOX_READONLY \
    -u BLOOMERY_BOX_LOCAL -u BLOOMERY_ENV_FILE -u BLOOMERY_CARD -u BLOOMERY_MODEL \
    -u BLOOMERY_BOX_QUEUE -u BLOOMERY_Q_CLASS -u BLOOMERY_Q_CARD -u BLOOMERY_Q_PRED_S \
    -u BLOOMERY_CARD_BUDGET -u BLOOMERY_FIXTURE_MODEL -u BLOOMERY_REF_MODEL_PROFILE -u BLOOMERY_LEASE_LOCK -u BLOOMERY_LEASE_HOLDS \
    bash -c "export BLOOMERY_LOCK_DIR=$(printf %q "$LOCK_DIR") && ${GUARD}$REMOTE_CMD"
fi
if [ "$QUEUE" = 0 ]; then
  ssh "$HOST" "${GUARD}$REMOTE_CMD"
  exit
fi
# The queued job (the header): the command as it runs directly, minus the guard, behind the stale-tree check and the dir lock.
# $REMOTE stays unquoted in `cat`: a quoted ~ would not expand.
STALE="[ \"\$(cat $REMOTE/.sync-id 2>/dev/null)\" = $SYNC_ID ] || { echo \"box.sh: $REMOTE was synced again after this job was queued (stale tree)\" >&2; exit 80; }"
INNER="$STALE
$REMOTE_CMD"
sq=\'\\\'\'
QUOTED=${INNER//\'/$sq}
JOB="flock -s -n -E 85 $QLOCK bash -c '$QUOTED'"
# <remote directory>-<the command's first word, as a name>-<the id's first 6>
first=${*%%[[:space:]]*}
first=$(printf %s "${first##*/}" | tr -c 'A-Za-z0-9._-' '_')
QNAME="$QKEY-${first:0:24}-${SYNC_ID:0:6}"
[ -n "$first" ] || QNAME="$QKEY-cmd-${SYNC_ID:0:6}"
qargs=(run --card "$QLANE" --class "$QCLASS" --name "$QNAME")
[ -z "$QPRED" ] || qargs+=(--expected-s "$QPRED")
PRED_TEXT=${QPRED:+$QPRED s}
echo "box.sh: queue $QCLASS card=$QLANE predicted ${PRED_TEXT:-unknown} remote=$REMOTE" >&2
exec python3 "$HERE/tools/boxq.py" ${qargs[@]+"${qargs[@]}"} -- "$JOB"
