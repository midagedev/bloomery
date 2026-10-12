#!/usr/bin/env bash
# bloomery — the Mac static tier on the second Mac. `just mac-static` takes 590-820 s on a Mac that also runs
# five Claude sessions (load 40-80, the disk 97 % full, a timing self-test inside check-recipes red from the
# load alone); the second Mac is idle and runs the same step list on a tree sent to it.
#
#   tools/mac-remote.sh [--tree DIR] [BASE]   `just mac-static BASE` for the worktree DIR (default: the tree
#                                             this script is in) on the second Mac; no BASE is the full run.
#                                             The remote's output (stdout and stderr, in one stream) comes
#                                             back on stdout, then the last 40 log lines of each red step; the exit
#                                             code is the remote run's.
#   tools/mac-remote.sh --self-test           the argument, config, track-name, dirty-tree, lock and ledger
#                                             rules and the ssh failure paths against a stub ssh, no remote
#
# The host is the one line of ~/.config/bloomery/mac-remote (`user@host`). It is not in the repository and
# never printed: every line that comes back is scrubbed of the account's home, user and host names. The
# remote's pieces and how they were made are tools/mac-remote/SETUP.md.
#
# One clone a track on the remote, ~/repo/bloomery-mr/<track>, <track> being DIR's directory name (two
# worktrees of one name share it; the lock below keeps their runs apart). A run is: a probe, one preflight
# (the toolchain, just, python, the env file), the lock, `git push` of HEAD and BASE to refs/mr/* of the
# clone (objects only), `git checkout --detach HEAD` there, rsync of DIR's content over it, `cargo fetch
# --locked`, `just mac-static BASE`, and the ledger lines the run appended copied back. The rsync compares by
# checksum and keeps the receiver's mtimes: a file whose content changed gets the remote's "now" and cargo
# rebuilds it, an unchanged one is left alone (the rule tools/box.sh syncs by). target/ stays on the remote,
# one per track: a track's first run is a cold build, the next ones warm.
#
# The ledger. A green shape's line (`<key> green <date> <tree>`) is appended to the local
# ~/.cache/bloomery/mac-static-ledger.tsv with `<track>@remote` as its tree: the reader
# (tools/recipes.py read_shape_ledger) takes exactly four fields, so a fifth `host=remote` field would make
# every pulled line malformed and ignored, and changing the reader moves every shape key (the file is in
# SHAPE_KEY_META). The remote keeps its own ledger and is not seeded from this one: a shape green here is
# checked once there. A key already green here is not appended again.
#
# The track lock is the remote directory ~/repo/bloomery-mr/.lock-<track>: the runner's host and pid, and the
# remote pid of the run once it started. A lock whose owner is a dead pid of this host is taken over, unless its
# remote run is still going (a killed runner leaves the remote run to finish). A live lock is waited for, up to
# TRACK_WAIT seconds (a second run of one tree, or two trees of one name, queue behind the first).
#
# The slot pool bounds the runs at once on the remote: SLOTS files ~/repo/bloomery-mr/.slot-<i>, each held by
# `lockf` (macOS's /usr/bin/lockf takes flock(2) through O_EXLOCK, so the kernel frees a slot when its holder
# dies, however it dies). A run takes the first free slot just before `just mac-static`, in its turn: it first takes
# the next ticket of the slot queue (~/repo/bloomery-mr/.slot-queue, one file a run, named by the ticket and holding
# the run's pid and start time) and tries the slots only while no live ticket is older than its own, so runs get
# slots in the order they came; a ticket whose run is gone (its pid dead, or now a process that started at another
# time) is dropped by the next run that reads the queue. At its start, before it takes a slot, it prints one line
# naming each slot: its holder (track and seconds held), `free`, or `starting` (held, the holder's file not written
# yet), and how many runs are ahead of it in the queue when any are; `free` is read from the lock itself, never from
# a holder file a killed run left behind. Until it has a slot it polls every SLOT_POLL s (3), prints the same line
# again every SLOT_SAY s (300, five minutes) and waits, never failing: a slot's holder is a run that ends. Two at once cost a run 5-8 % (SETUP.md has the walls);
# three at once is not measured.
#
# Exit: the remote run's code (mac-static's: the first red step's), or 0. 64: an argument, the config file
# (missing, or not one `user@host` line) or the tree this script does not take, or a BASE that is no commit.
# 65: the tree has a state rsync does not carry (a submodule, a symlink out of the tree, a tracked file under
# target/). 69: a prerequisite is missing on the remote, named with where SETUP.md makes it. 75: the remote
# is not reachable or the connection was lost mid-run (the remote run may still finish), or the track's
# lock was still held by a live run after TRACK_WAIT seconds.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
CONF=$HOME/.config/bloomery/mac-remote
LEDGER=$HOME/.cache/bloomery/mac-static-ledger.tsv
SSH_OPTS=(-o LogLevel=ERROR -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
RBASE=repo/bloomery-mr # the clones' directory, relative to the remote home
SLOTS=2 # runs at once on the remote; a further run waits for a slot
SLOT_SAY=300 # seconds between two prints of the slots' line while a run waits for a slot
SLOT_POLL=3 # seconds between a waiting run's two looks at the queue and the slots
TRACK_WAIT=1800 # seconds a run waits for its track's lock held by a live run

say() { printf '%s\n' "$*" >&2; }

# read_host: the config's one `user@host` line on stdout; 64 naming the file, never its content.
read_host() {
  local n line
  if [ ! -f "$CONF" ]; then
    say "mac-remote: no ~/.config/bloomery/mac-remote — one line, user@host, the second Mac (not in the repository)"
    return 64
  fi
  n=$(grep -cvE '^[[:space:]]*(#|$)' "$CONF" || true)
  line=$(grep -vE '^[[:space:]]*(#|$)' "$CONF" | head -1 || true)
  if [ "$n" != 1 ] || [[ ! $line =~ ^[A-Za-z0-9._-]+@[A-Za-z0-9._:-]+$ ]]; then
    say "mac-remote: ~/.config/bloomery/mac-remote must hold exactly one line, user@host (no spaces); it has $n non-comment line(s) and the first is not that form"
    return 64
  fi
  printf '%s\n' "$line"
}

# track_of DIR: the remote clone's name, DIR's directory name, or 64 when it is not one word of [A-Za-z0-9._-].
track_of() {
  local t=${1%/}
  t=${t##*/}
  if [[ ! $t =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    say "mac-remote: the tree's directory name '$t' is not a track name ([A-Za-z0-9][A-Za-z0-9._-]*): it names the remote clone"
    return 64
  fi
  printf '%s\n' "$t"
}

# commit_of TREE REV: the full sha of REV in TREE, or 64.
commit_of() {
  local sha
  sha=$(git -C "$1" rev-parse --verify -q "$2^{commit}" 2> /dev/null) || {
    say "mac-remote: '$2' is no commit in $1"
    return 64
  }
  printf '%s\n' "$sha"
}

# base_of TREE BASE: the BASE mac-static gets, every rev in it a full sha (`A..B` both ends), one per line:
# the argument, then each sha to push. An empty BASE prints nothing.
base_of() {
  local t=$1 b=$2 a z
  [ -n "$b" ] || return 0
  case $b in
    *..*)
      a=$(commit_of "$t" "${b%%..*}") || return $?
      z=$(commit_of "$t" "${b##*..}") || return $?
      printf '%s\n%s\n%s\n' "$a..$z" "$a" "$z"
      ;;
    -*) say "mac-remote: BASE '$b' looks like an option"; return 64 ;;
    *) a=$(commit_of "$t" "$b") || return $?; printf '%s\n%s\n' "$a" "$a" ;;
  esac
}

# norm PATH: PATH with . and .. resolved, no filesystem access (a link's target need not exist).
norm() {
  local IFS=/ p n
  local -a out=()
  for p in $1; do
    case $p in
      '' | .) ;;
      ..) n=${#out[@]}; if [ "$n" -gt 0 ]; then unset "out[$((n - 1))]"; out=(${out[@]+"${out[@]}"}); fi ;;
      *) out+=("$p") ;;
    esac
  done
  printf '/%s' ${out[@]+"${out[@]}"} | sed 's#^/\{2,\}#/#; s#^//*$#/#'
}

# dirt TREE: 65, naming each, when TREE holds a state the sync does not carry: a submodule (gitlink: the
# clone's checkout leaves it empty and rsync would fill it from a second repository), a symlink that
# leaves the tree (its target is another path on the other Mac), a tracked file under target/ (rsync
# excludes target/, so it would stay at HEAD's content while the tree's differs).
dirt() {
  local t rc=0 line f d tgt
  t=$(cd "$1" && pwd -P)
  while IFS= read -r line; do
    say "mac-remote: $t holds a submodule (gitlink ${line#*$'\t'}): the sync does not carry one"
    rc=65
  done < <(git -C "$t" ls-files -s | awk '$1 == "160000"')
  f=$(git -C "$t" ls-files | grep -m1 '^target/' || true)
  if [ -n "$f" ]; then
    say "mac-remote: $t tracks $f under target/, which the sync leaves out (target/ stays on the remote)"
    rc=65
  fi
  while IFS= read -r f; do
    [ -L "$t/$f" ] || continue
    tgt=$(readlink "$t/$f")
    case $tgt in /*) ;; *) tgt=$(dirname "$t/$f")/$tgt ;; esac
    d=$(norm "$tgt")
    case $d in
      "$t" | "$t"/*) ;;
      *) say "mac-remote: $t/$f is a symlink to $tgt, outside the tree: the other Mac has no such path"; rc=65 ;;
    esac
  done < <(git -C "$t" ls-files --cached --others --exclude-standard)
  return $rc
}

# sed_esc TEXT: TEXT as a literal in a sed regex.
sed_esc() { printf '%s' "$1" | sed 's/[][\.*^$#]/\\&/g'; }

# scrub HOME USER HOST: stdin to stdout with the remote account's home as ~ and its user and host names
# as <user> and <host>. Nothing of the three leaves this script.
scrub() {
  local -a ex=()
  [ -z "$1" ] || ex+=(-e "s#$(sed_esc "$1")#~#g")
  [ -z "$2" ] || ex+=(-e "s#$(sed_esc "$2")#<user>#g")
  [ -z "$3" ] || ex+=(-e "s#$(sed_esc "$3")#<host>#g")
  if [ ${#ex[@]} = 0 ]; then cat; else sed -u "${ex[@]}"; fi
}

# lock_verdict OWNER_HOST OWNER_PID RUN_ALIVE: `stale` when the lock's owner is a dead process of this host
# and its remote run is not going, else `busy`.
lock_verdict() {
  local me
  me=$(hostname -s | tr -c 'A-Za-z0-9.\n-' _)
  if [ "$1" != "$me" ] || [ "$3" = 1 ] || kill -0 "$2" 2> /dev/null; then echo busy; else echo stale; fi
}

# ledger_new LOCAL REMOTE_LINES TRACK: the lines of REMOTE_LINES that are green four-field ledger lines of
# TRACK's clone (the remote ledger is shared by every track) whose key is not in LOCAL, their tree field
# rewritten to TRACK@remote; a key once.
ledger_new() {
  awk -F'\t' -v want="$3" '
    FILENAME == ARGV[1] { seen[$1] = 1; next }
    NF == 4 && $4 == want && length($1) == 64 && $1 !~ /[^0-9a-f]/ && $2 == "green" && !($1 in seen) {
      seen[$1] = 1
      printf "%s\t%s\t%s\t%s@remote\n", $1, $2, $3, want
    }' "$1" "$2"
}

# The remote scripts, run as `bash -s -- ARGS` over ssh: the whole text is one group, so the shell has read
# it all before the first command and the commands may take stdin. The remote shell is zsh with a PATH of
# /usr/bin:/bin:/usr/sbin:/sbin and no profile of ours: each script sets its own PATH, and no file of the
# account's shell setup (the host also runs CI) is edited anywhere.
PREP='{
exec < /dev/null
echo "HOME=$HOME"
track=$1 owner=$2 channel=$3 want_just=$4 steal=$5
export PATH=$HOME/opt/bloomery-bin:$HOME/.cargo/bin:/usr/bin:/bin:/usr/sbin:/sbin
miss=0
need() { echo "missing: $*"; miss=1; }
case "$(uname -s) $(uname -m)" in
  "Darwin arm64") host=aarch64-apple-darwin ;;
  "Darwin x86_64") host=x86_64-apple-darwin ;;
  *) echo "missing: a macOS host (this is $(uname -s) $(uname -m))"; exit 69 ;;
esac
tc=$HOME/.rustup/toolchains/$channel-$host
for t in cargo rustc cargo-clippy clippy-driver cargo-fmt rustfmt; do [ -x "$tc/bin/$t" ] || need "$tc/bin/$t (rustup toolchain install $channel, SETUP.md step 3)"; done
ls "$tc"/lib/rustlib/x86_64-unknown-linux-gnu/lib/libstd-*.rlib > /dev/null 2>&1 || need "the x86_64-unknown-linux-gnu std in $channel (SETUP.md step 3)"
[ "$(just --version 2> /dev/null)" = "$want_just" ] || need "$want_just in PATH (SETUP.md step 2; found: $(just --version 2> /dev/null || echo none))"
python3 -c "import sys, numpy, PIL; sys.exit(sys.version_info < (3, 11))" 2> /dev/null || need "python3 >= 3.11 with numpy and Pillow in PATH (SETUP.md step 1; found: $(python3 --version 2> /dev/null || echo none))"
command -v git > /dev/null || need git
command -v rsync > /dev/null || need rsync
command -v lockf > /dev/null || need "lockf (/usr/bin/lockf: the run slots are its flock locks)"
if [ -f "$HOME/opt/bloomery-mac-env.sh" ]; then
  ( . "$HOME/opt/bloomery-mac-env.sh"
    [ -f "${CUDA_TOOLKIT_PATH:-}/include/cuda.h" ] || echo "missing: cuda.h under CUDA_TOOLKIT_PATH (SETUP.md step 4)"
    [ -f "${LIBCLANG_PATH:-}/libclang.dylib" ] || echo "missing: libclang.dylib under LIBCLANG_PATH (xcode-select --install)"
    sr=$(printf "%s\n" "${BINDGEN_EXTRA_CLANG_ARGS_x86_64_unknown_linux_gnu:-}" | tr " " "\n" | sed -n "s/^--sysroot=//p" | head -1)
    [ -f "$sr/usr/include/stdlib.h" ] || echo "missing: the Linux sysroot usr/include (SETUP.md step 4)" ) > "${TMPDIR:-/tmp}/mr-env.$$"
  if [ -s "${TMPDIR:-/tmp}/mr-env.$$" ]; then cat "${TMPDIR:-/tmp}/mr-env.$$"; miss=1; fi
  rm -f "${TMPDIR:-/tmp}/mr-env.$$"
else
  need "$HOME/opt/bloomery-mac-env.sh (SETUP.md step 4)"
fi
[ "$miss" = 0 ] || exit 69
base=$HOME/'"$RBASE"'
mkdir -p "$base"
lock=$base/.lock-$track
if [ "$steal" = 1 ]; then
  rp=$(cat "$lock/run-pid" 2> /dev/null || true)
  if [ -n "$rp" ] && kill -0 "$rp" 2> /dev/null; then echo "RUNNING=$rp"; exit 75; fi
  rm -rf "$lock"
fi
if ! mkdir "$lock" 2> /dev/null; then
  echo "HELD=$(cat "$lock/owner" 2> /dev/null)"
  rp=$(cat "$lock/run-pid" 2> /dev/null || true)
  if [ -n "$rp" ] && kill -0 "$rp" 2> /dev/null; then echo "RUNNING=$rp"; fi
  exit 75
fi
echo "$owner" > "$lock/owner"
[ -d "$base/$track/.git" ] || { mkdir -p "$base/$track"; git -C "$base/$track" init -q; }
n=0
[ ! -f "$HOME/.cache/bloomery/mac-static-ledger.tsv" ] || n=$(wc -l < "$HOME/.cache/bloomery/mac-static-ledger.tsv" | tr -d " ")
echo "LEDGER_N=$n"
exit 0
}'

CHECKOUT='{
exec < /dev/null
cd "$HOME/'"$RBASE"'/$1" || exit $?
# The last run'"'"'s rsync (-p) chmods every file it compares, so an unchanged file'"'"'s ctime moves past the index'"'"'s whenever
# that rsync ran a second or more after the checkout (a loaded remote); `checkout -f` compares stat data only and would
# rewrite those files, and cargo rebuilds what they feed. The refresh re-reads the content of a stat-dirty file first.
git update-index -q --refresh > /dev/null 2>&1
git checkout -q -f --detach "$2"
exit $?
}'

# The red steps of the last run, for the caller to read: mac-static prints one summary line a step and keeps each step's
# output in target/mac-static/<step>.log, which stays on the remote with the tree's target/.
RED='{
exec < /dev/null
d=$HOME/'"$RBASE"'/$1/target/mac-static
for f in "$d"/*.rc; do
  [ -f "$f" ] || continue
  n=$(basename "$f" .rc)
  r=$(cat "$f")
  [ "$r" != 0 ] || continue
  echo "mac-remote: red step $n (rc $r): the last 40 lines of target/mac-static/$n.log on the remote"
  tail -40 "$d/$n.log" 2> /dev/null
done
exit 0
}'

RUN='{
exec < /dev/null
track=$1 basearg=$2 channel=$3 slots=$4 say=$5 poll=$6
export PATH=$HOME/opt/bloomery-bin:$HOME/.cargo/bin:/usr/bin:/bin:/usr/sbin:/sbin
base=$HOME/'"$RBASE"'
echo $$ > "$base/.lock-$track/run-pid"
trap "rm -f \"$base/.lock-$track/run-pid\"" EXIT
cd "$base/$track" || exit 69
case "$(uname -m)" in arm64) host=aarch64-apple-darwin ;; *) host=x86_64-apple-darwin ;; esac
PATH=$HOME/.rustup/toolchains/$channel-$host/bin:$PATH
rc=0
cargo fetch --locked > "$base/.fetch-$track.log" 2>&1 || rc=$?
if [ "$rc" != 0 ]; then
  tail -5 "$base/.fetch-$track.log"
  echo "mac-remote: cargo fetch --locked failed on the remote (rc $rc): Cargo.lock is out of date for the tree (the box recipes need the same lock) or the registry or GitHub is not reachable from it"
  rm -f "$base/.fetch-$track.log"
  exit 69
fi
rm -f "$base/.fetch-$track.log"
if [ -n "$basearg" ]; then cmd=(just mac-static "$basearg"); else cmd=(just mac-static); fi
# One slot of the pool: the command runs under the slot file lockf and writes its own rc to a file, so an rc 75 of
# its own is not read as the lockf busy code (EX_TEMPFAIL, 75).
slot_inner() {
  local i=$1 waited=$2 holder=$3 rcf=$4
  shift 4
  rm -f "$ticket"
  echo "$track $(date +%s)" > "$holder"
  echo "mac-remote: run slot $((i + 1)) of $slots taken after $waited s"
  "$@"
  echo $? > "$rcf"
  rm -f "$holder"
}
export -f slot_inner
export track slots
holders() {
  local j=0 h now
  now=$(date +%s)
  while [ "$j" -lt "$slots" ]; do
    if lockf -k -s -t 0 "$base/.slot-$j" true; then
      printf " [slot %s: free]" "$((j + 1))"
    else
      h=$(cat "$base/.slot-$j.holder" 2> /dev/null || true)
      if [ -n "$h" ]; then set -- $h; printf " [slot %s: %s for %s s]" "$((j + 1))" "$1" "$((now - $2))"; else printf " [slot %s: starting]" "$((j + 1))"; fi
    fi
    j=$((j + 1))
  done
}
rcf=$base/.slot-rc-$track
rm -f "$rcf"
# The slot queue (header): the ticket is written whole under another name and renamed, so a reader never sees it empty.
qd=$base/.slot-queue
mkdir -p "$qd"
started() { ps -o lstart= -p "$1" 2> /dev/null | tr -s " " | sed "s/^ //; s/ \$//"; }
next_ticket() { local n; n=$(cat "$1" 2> /dev/null || echo 0); n=$((n + 1)); echo "$n" > "$1"; echo "$n"; }
export -f next_ticket
n=$(lockf -k -t 30 "$qd/.next.lock" bash -c "next_ticket \"\$1\"" _ "$qd/.next") || {
  echo "mac-remote: the ticket lock of the slot queue on the remote was not taken in 30 s"
  exit 69
}
ticket=$qd/$(printf "%012d" "$n").$track
printf "%s %s\n" "$$" "$(started $$)" > "$qd/.new-$$"
mv "$qd/.new-$$" "$ticket"
export ticket
trap "rm -f \"$base/.lock-$track/run-pid\" \"$ticket\"" EXIT
# The live tickets older than the ticket of this run, counted; a ticket whose run is gone is removed.
ahead() {
  local f p s k=0
  for f in "$qd"/[0-9]*; do
    [ -e "$f" ] || continue
    [ "$f" != "$ticket" ] || break
    read -r p s < "$f" 2> /dev/null || continue
    if [ -n "$p" ] && [ "$(started "$p")" = "$s" ]; then k=$((k + 1)); else rm -f "$f"; fi
  done
  echo "$k"
}
k=$(ahead)
if [ "$k" = 0 ]; then q=; else q=", $k ahead of it in the slot queue"; fi
echo "mac-remote: run slots:$(holders); $track$q"
w0=$SECONDS last=$SECONDS
while :; do
  k=$(ahead)
  if [ "$k" = 0 ]; then
    i=0
    while [ "$i" -lt "$slots" ]; do
      lockf -k -s -t 0 "$base/.slot-$i" bash -c "slot_inner \"\$@\"" _ "$i" "$((SECONDS - w0))" "$base/.slot-$i.holder" "$rcf" "${cmd[@]}"
      lrc=$?
      [ "$lrc" = 75 ] || break 2
      i=$((i + 1))
    done
  fi
  if [ $((SECONDS - last)) -ge "$say" ]; then
    if [ "$k" = 0 ]; then
      echo "mac-remote: all $slots run slots are taken:$(holders); $track waits ($((SECONDS - w0)) s)"
    else
      echo "mac-remote: run slots:$(holders); $track waits ($((SECONDS - w0)) s), $k ahead of it in the slot queue"
    fi
    last=$SECONDS
  fi
  sleep "$poll"
done
if [ "$lrc" != 0 ] || [ ! -f "$rcf" ]; then
  echo "mac-remote: the run slot failed on the remote (lockf rc $lrc, ${rcf##*/} $([ -f "$rcf" ] && echo present || echo missing))"
  rm -f "$rcf"
  exit 69
fi
rc=$(cat "$rcf")
rm -f "$rcf"
exit $rc
}'

# qargs ARG…: the words as one string the remote shell (zsh) reads back as the same words: ssh joins its arguments
# with spaces and the far shell splits them again.
qargs() {
  local a out=
  for a in "$@"; do out="$out$(printf '%q' "$a") "; done
  printf '%s' "$out"
}

# release_lock: drop the track's lock on the remote (never while a run may still be going).
release_lock() {
  ssh "${SSH_OPTS[@]}" "$R" "rm -rf $RBASE/.lock-$track" > /dev/null 2>&1 ||
    say "mac-remote: could not release the lock of $track on the remote (the next run takes it over once this runner's pid is gone)"
}

cleanup() {
  { [ "${LOCKED:-0}" = 0 ] || [ "${RUN_ACTIVE:-0}" = 1 ]; } || release_lock
  [ -z "${TMP:-}" ] || rm -rf "$TMP"
}

on_signal() {
  [ -z "${SSHPID:-}" ] || kill "$SSHPID" 2> /dev/null || true
  [ -z "${SCPID:-}" ] || kill "$SCPID" 2> /dev/null || true
  [ -z "${SLPID:-}" ] || kill "$SLPID" 2> /dev/null || true
  exit "$1"
}

sync_tree() {
  rsync -rlpc --delete --exclude=/.git --exclude=/target/ --exclude=__pycache__/ --exclude=.DS_Store \
    -e "ssh ${SSH_OPTS[*]}" "$TREE/" "$R:$RBASE/$track/"
}

# remote_main TREE BASE: the run; its return is the exit code.
remote_main() {
  local tree=$1 basearg=$2 r base_list n_pushed channel just_v owner ssh_rc t0=$SECONDS t_setup t_run
  local steal=0 tries=0 held running total appended tw0=$SECONDS twlast=-60
  local -a pushes=()
  R=$(read_host) || return $?
  RUSER=${R%@*}
  RHOST=${R#*@}
  RHOME=
  TREE=$(cd "$tree" 2> /dev/null && pwd -P) || { say "mac-remote: --tree '$tree' is not a directory"; return 64; }
  [ "$(git -C "$TREE" rev-parse --show-toplevel 2> /dev/null || true)" = "$TREE" ] ||
    { say "mac-remote: $TREE is not the root of a git worktree"; return 64; }
  track=$(track_of "$TREE") || return $?
  HEAD_SHA=$(commit_of "$TREE" HEAD) || return $?
  pushes=("$HEAD_SHA:refs/mr/head")
  BASE_ARG=
  if [ -n "$basearg" ]; then
    base_list=$(base_of "$TREE" "$basearg") || return $?
    BASE_ARG=$(printf '%s\n' "$base_list" | sed -n 1p)
    n_pushed=0
    for r in $(printf '%s\n' "$base_list" | sed 1d | sort -u); do
      n_pushed=$((n_pushed + 1))
      pushes+=("$r:refs/mr/base$n_pushed")
    done
  fi
  dirt "$TREE" || return $?
  channel=$(sed -nE 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' "$TREE/rust-toolchain.toml" 2> /dev/null | head -1)
  [ -n "$channel" ] || { say "mac-remote: no channel = \"…\" line in $TREE/rust-toolchain.toml"; return 64; }
  just_v=$(just --version) || { say "mac-remote: no just here to name the version the remote must have"; return 69; }
  owner="$(hostname -s | tr -c 'A-Za-z0-9.\n-' _)/$$"
  TMP=$(mktemp -d "${TMPDIR:-/tmp}/mac-remote.XXXXXX")
  trap cleanup EXIT
  trap 'on_signal 130' INT
  trap 'on_signal 143' TERM
  LOCKED=0 RUN_ACTIVE=0 SSHPID= SCPID= SLPID=
  export GIT_SSH_COMMAND="ssh ${SSH_OPTS[*]}"

  # the probe: unreachable is 75, whatever else is wrong
  ssh "${SSH_OPTS[@]}" "$R" true > "$TMP/probe.log" 2>&1 || {
    say "mac-remote: the second Mac is not reachable over ssh (BatchMode, 10 s): $(scrub '' "$RUSER" "$RHOST" < "$TMP/probe.log" | head -1)"
    return 75
  }

  # preflight, lock, clone and the ledger's size in one session
  printf '%s\n' "$PREP" > "$TMP/prep.sh"
  while :; do
    ssh_rc=0
    ssh "${SSH_OPTS[@]}" "$R" "bash -s -- $(qargs "$track" "$owner" "$channel" "$just_v" "$steal")" < "$TMP/prep.sh" > "$TMP/prep.out" 2>&1 || ssh_rc=$?
    RHOME=$(sed -n 's/^HOME=//p' "$TMP/prep.out" | head -1)
    case $ssh_rc in
      0) break ;;
      255) say "mac-remote: the second Mac dropped the ssh session during the preflight"; return 75 ;;
      69)
        say "mac-remote: the second Mac lacks a prerequisite (tools/mac-remote/SETUP.md makes each):"
        grep -v '^HOME=' "$TMP/prep.out" | scrub "$RHOME" "$RUSER" "$RHOST" >&2
        return 69
        ;;
      75)
        held=$(sed -n 's/^HELD=//p' "$TMP/prep.out" | head -1)
        running=$(sed -n 's/^RUNNING=//p' "$TMP/prep.out" | head -1)
        if [ -z "$running" ] && [ -n "$held" ] && [ "$tries" = 0 ] && [ "$(lock_verdict "${held%/*}" "${held##*/}" 0)" = stale ]; then
          steal=1 tries=1
          continue
        fi
        if [ $((SECONDS - tw0)) -ge "$TRACK_WAIT" ]; then
          say "mac-remote: track $track is still locked on the remote by ${held:-an unknown owner}${running:+, its run (remote pid $running) going} after $TRACK_WAIT s: one run a track at a time"
          return 75
        fi
        if [ $((SECONDS - twlast)) -ge 60 ]; then
          say "mac-remote: track $track is locked on the remote by ${held:-an unknown owner}${running:+, its run (remote pid $running) going}; waiting ($((SECONDS - tw0)) s)"
          twlast=$SECONDS
        fi
        steal=0 tries=0
        sleep 5 &
        SLPID=$!
        wait "$SLPID" || true
        SLPID=
        ;;
      *)
        say "mac-remote: the preflight ended rc $ssh_rc:"
        scrub "$RHOME" "$RUSER" "$RHOST" < "$TMP/prep.out" | tail -5 >&2
        return 75
        ;;
    esac
  done
  LOCKED=1
  LEDGER_N=$(sed -n 's/^LEDGER_N=//p' "$TMP/prep.out" | head -1)
  [[ $LEDGER_N =~ ^[0-9]+$ ]] || LEDGER_N=0
  say "mac-remote: track $track, HEAD ${HEAD_SHA:0:12}, base ${BASE_ARG:-none (full run)}"

  # objects, checkout, content
  git -C "$TREE" push -q --no-verify -f "$R:$RBASE/$track" "${pushes[@]}" > "$TMP/push.log" 2>&1 || {
    say "mac-remote: git push to the remote clone failed:"; scrub "$RHOME" "$RUSER" "$RHOST" < "$TMP/push.log" | tail -5 >&2; return 75
  }
  printf '%s\n' "$CHECKOUT" > "$TMP/checkout.sh"
  ssh "${SSH_OPTS[@]}" "$R" "bash -s -- $(qargs "$track" "$HEAD_SHA")" < "$TMP/checkout.sh" > "$TMP/checkout.log" 2>&1 || {
    say "mac-remote: git checkout --detach on the remote failed:"; scrub "$RHOME" "$RUSER" "$RHOST" < "$TMP/checkout.log" | tail -5 >&2; return 75
  }
  sync_tree > "$TMP/rsync.log" 2>&1 || {
    say "mac-remote: rsync to the remote failed:"; scrub "$RHOME" "$RUSER" "$RHOST" < "$TMP/rsync.log" | tail -5 >&2; return 75
  }
  t_setup=$((SECONDS - t0))

  # the run, its output scrubbed on the way back; the pids are ours, taken at spawn
  printf '%s\n' "$RUN" > "$TMP/run.sh"
  mkfifo "$TMP/out"
  scrub "$RHOME" "$RUSER" "$RHOST" < "$TMP/out" &
  SCPID=$!
  RUN_ACTIVE=1
  ssh "${SSH_OPTS[@]}" "$R" "bash -s -- $(qargs "$track" "$BASE_ARG" "$channel" "$SLOTS" "$SLOT_SAY" "$SLOT_POLL")" < "$TMP/run.sh" > "$TMP/out" 2>&1 &
  SSHPID=$!
  ssh_rc=0
  wait "$SSHPID" || ssh_rc=$?
  wait "$SCPID" || true
  SSHPID= SCPID=
  if [ "$ssh_rc" = 255 ]; then
    say "mac-remote: the ssh session of the run was lost (rc 255): the run may still be going on the remote, and the track stays locked until it ends"
    return 75
  fi
  RUN_ACTIVE=0
  t_run=$((SECONDS - t0 - t_setup))
  if [ "$ssh_rc" != 0 ]; then
    printf '%s\n' "$RED" > "$TMP/red.sh"
    ssh "${SSH_OPTS[@]}" "$R" "bash -s -- $(qargs "$track")" < "$TMP/red.sh" 2> /dev/null | scrub "$RHOME" "$RUSER" "$RHOST" || true
  fi

  # the ledger lines this track's run appended
  mkdir -p "$(dirname "$LEDGER")"
  [ -f "$LEDGER" ] || : > "$LEDGER"
  if ssh "${SSH_OPTS[@]}" "$R" "tail -n +$((LEDGER_N + 1)) .cache/bloomery/mac-static-ledger.tsv 2> /dev/null || true" > "$TMP/ledger.remote" 2> /dev/null; then
    ledger_new "$LEDGER" "$TMP/ledger.remote" "$track" > "$TMP/ledger.new"
    total=$(awk -F'\t' -v w="$track" 'NF == 4 && $4 == w && $2 == "green"' "$TMP/ledger.remote" | wc -l | tr -d ' ')
    appended=$(wc -l < "$TMP/ledger.new" | tr -d ' ')
    cat "$TMP/ledger.new" >> "$LEDGER"
    say "mac-remote: ledger: $total green shape line(s) this run, $appended appended here as $track@remote, $((total - appended)) already green here"
  else
    say "mac-remote: the ledger lines of the run were not copied back (ssh failed); the remote keeps them"
  fi
  say "mac-remote: wall $((SECONDS - t0)) s (setup $t_setup s, run $t_run s), remote rc $ssh_rc"
  return "$ssh_rc"
}

# fake_remote HOME CHANNEL: a remote home that passes the preflight, its tools stubs: just prints its arguments to
# just.args, appends a green ledger line under its directory's name and exits $STUB_JUST_RC.
fake_remote() {
  local rh=$1 ch=$2 h t
  case $(uname -m) in arm64) h=aarch64-apple-darwin ;; *) h=x86_64-apple-darwin ;; esac
  mkdir -p "$rh/.rustup/toolchains/$ch-$h/bin" "$rh/.rustup/toolchains/$ch-$h/lib/rustlib/x86_64-unknown-linux-gnu/lib" \
    "$rh/opt/bloomery-bin" "$rh/opt/cuda/include" "$rh/opt/clang" "$rh/opt/sr/usr/include"
  for t in cargo rustc cargo-clippy clippy-driver cargo-fmt rustfmt; do
    printf '#!/bin/sh\nexit 0\n' > "$rh/.rustup/toolchains/$ch-$h/bin/$t"
    chmod +x "$rh/.rustup/toolchains/$ch-$h/bin/$t"
  done
  : > "$rh/.rustup/toolchains/$ch-$h/lib/rustlib/x86_64-unknown-linux-gnu/lib/libstd-x.rlib"
  : > "$rh/opt/cuda/include/cuda.h"
  : > "$rh/opt/clang/libclang.dylib"
  : > "$rh/opt/sr/usr/include/stdlib.h"
  printf '%s\n' 'export CUDA_TOOLKIT_PATH=$HOME/opt/cuda LIBCLANG_PATH=$HOME/opt/clang' \
    'export BINDGEN_EXTRA_CLANG_ARGS_x86_64_unknown_linux_gnu="--sysroot=$HOME/opt/sr"' > "$rh/opt/bloomery-mac-env.sh"
  printf '#!/bin/sh\nexit 0\n' > "$rh/opt/bloomery-bin/python3"
  cat > "$rh/opt/bloomery-bin/just" << 'JUST'
#!/bin/sh
[ "$1" != --version ] || { echo "just 9.9.9"; exit 0; }
echo "$@" > "$HOME/just.args"
echo "stub just in $(basename "$PWD")"
basename "$PWD" >> "$HOME/just.order"
if [ "${STUB_JUST_RC:-0}" != 0 ]; then
  mkdir -p target/mac-static
  echo "${STUB_JUST_RC}" > target/mac-static/fake-step.rc
  echo "fake-step output line" > target/mac-static/fake-step.log
  echo 0 > target/mac-static/ok-step.rc
  echo "ok-step output line" > target/mac-static/ok-step.log
fi
mkdir -p "$HOME/.cache/bloomery"
printf '%064d\tgreen\td\t%s\n' "${STUB_KEY:-1}" "$(basename "$PWD")" | tr 0 a >> "$HOME/.cache/bloomery/mac-static-ledger.tsv"
exit "${STUB_JUST_RC:-0}"
JUST
  chmod +x "$rh/opt/bloomery-bin/python3" "$rh/opt/bloomery-bin/just"
}

self_test() {
  local fails=0 d out rc t fake stub repo me k1 k2 rh rt pid
  fail() { say "mac-remote self-test FAIL: $*"; fails=$((fails + 1)); }
  d=$(mktemp -d "${TMPDIR:-/tmp}/mac-remote-st.XXXXXX")
  fake=$d/home
  mkdir -p "$fake/.config/bloomery"
  me=$(hostname -s | tr -c 'A-Za-z0-9.\n-' _)

  # track names
  [ "$(track_of /a/b/bloomery-x/)" = bloomery-x ] || fail "track_of a path with a trailing slash"
  for t in 'a b' '.hid' '-x' 'a;b' 'a$b'; do
    rc=0; track_of "/x/$t" > /dev/null 2>&1 || rc=$?
    [ "$rc" = 64 ] || fail "track name '$t' ended rc $rc, not 64"
  done
  rc=0; track_of '' > /dev/null 2>&1 || rc=$?
  [ "$rc" = 64 ] || fail "an empty track name ended rc $rc, not 64"

  # the config: missing, with spaces, two lines — each 64, its content never printed
  rc=0; out=$(HOME=$fake "$HERE/tools/mac-remote.sh" HEAD 2>&1) || rc=$?
  [ "$rc" = 64 ] || fail "a missing config ended rc $rc, not 64"
  case $out in *"no ~/.config/bloomery/mac-remote"*) ;; *) fail "a missing config is not named: $out" ;; esac
  printf '%s\n' 'leak-sentinel one' > "$fake/.config/bloomery/mac-remote"
  rc=0; out=$(HOME=$fake "$HERE/tools/mac-remote.sh" HEAD 2>&1) || rc=$?
  [ "$rc" = 64 ] || fail "a config with spaces ended rc $rc"
  case $out in *leak-sentinel*) fail "the config's content was printed: $out" ;; esac
  printf '%s\n' 'u1@h1.example' 'u2@h2.example' > "$fake/.config/bloomery/mac-remote"
  rc=0; out=$(HOME=$fake "$HERE/tools/mac-remote.sh" HEAD 2>&1) || rc=$?
  [ "$rc" = 64 ] || fail "a config with two lines ended rc $rc"
  case $out in *h2.example* | *h1.example*) fail "a host name was printed: $out" ;; esac

  # arguments
  for t in '--bogus' '--tree' 'a b' '--self-test x'; do
    rc=0; HOME=$fake "$HERE/tools/mac-remote.sh" $t > /dev/null 2>&1 || rc=$?
    [ "$rc" = 64 ] || fail "arguments '$t' ended rc $rc, not 64"
  done

  # a scratch repo for the tree rules
  repo=$d/tree-one
  mkdir -p "$repo/target" "$repo/sub"
  git -C "$repo" init -q
  printf '%s\n' 'channel = "nightly-x"' > "$repo/rust-toolchain.toml"
  printf 'a\n' > "$repo/sub/f"
  printf 'g\n' > "$repo/sub/g"
  git -C "$repo" add rust-toolchain.toml sub/f sub/g
  git -C "$repo" -c user.name=t -c user.email=t@t commit -q -m one
  printf 'b\n' >> "$repo/sub/f"
  git -C "$repo" -c user.name=t -c user.email=t@t commit -q -am two
  out=$(dirt "$repo" 2>&1) || fail "a clean tree is refused: $out"
  [ "$(base_of "$repo" 'HEAD~1..HEAD' | wc -l | tr -d ' ')" = 3 ] || fail "an A..B base does not list its argument and both shas"
  rc=0; base_of "$repo" nosuchrev > /dev/null 2>&1 || rc=$?
  [ "$rc" = 64 ] || fail "a base that is no commit ended rc $rc, not 64"
  ln -s /etc/passwd "$repo/outlink"
  rc=0; out=$(dirt "$repo" 2>&1) || rc=$?
  [ "$rc" = 65 ] || fail "a symlink out of the tree ended rc $rc: $out"
  case $out in *outlink*) ;; *) fail "the escaping symlink is not named: $out" ;; esac
  rm "$repo/outlink"; ln -s sub/f "$repo/inlink"; ln -s ../tree-one/sub/f "$repo/uplink"
  out=$(dirt "$repo" 2>&1) || fail "a symlink inside the tree is refused: $out"
  rm "$repo/inlink" "$repo/uplink"
  git -C "$repo" update-index --add --cacheinfo "160000,$(git -C "$repo" rev-parse HEAD),subm"
  rc=0; out=$(dirt "$repo" 2>&1) || rc=$?
  [ "$rc" = 65 ] || fail "a gitlink ended rc $rc: $out"
  git -C "$repo" rm -q --cached subm
  printf 'x\n' > "$repo/target/t"
  git -C "$repo" add -f target/t
  rc=0; out=$(dirt "$repo" 2>&1) || rc=$?
  [ "$rc" = 65 ] || fail "a tracked file under target/ ended rc $rc: $out"
  case $out in *"target/t"*) ;; *) fail "the tracked target file is not named: $out" ;; esac
  git -C "$repo" rm -q --cached target/t
  [ "$(norm /a/b/../c/./d)" = /a/c/d ] || fail "norm /a/b/../c/./d gave $(norm /a/b/../c/./d)"
  [ "$(norm /a/../../b)" = /b ] || fail "norm /a/../../b gave $(norm /a/../../b)"
  [ "$(qargs 'a b' '' "x'y")" = "a\\ b '' x\\'y " ] || fail "qargs gave '$(qargs 'a b' '' "x'y")'"

  # scrubbing, the lock verdict and the ledger lines
  out=$(printf '%s\n' '/Users/uu/repo/x uu@hh.local' 'cargo at /Users/uu/.cargo' | scrub /Users/uu uu hh.local)
  case $out in */Users/uu* | *uu* | *hh.local*) fail "scrub left a name: $out" ;; esac
  case $out in "~/repo/x <user>@<host>"*) ;; *) fail "scrub's rewrite is wrong: $out" ;; esac
  [ "$(lock_verdict "$me" $$ 0)" = busy ] || fail "a live owner's lock is not busy"
  [ "$(lock_verdict other-host $$ 0)" = busy ] || fail "another host's lock is not busy"
  sleep 0 & pid=$!; wait "$pid"
  [ "$(lock_verdict "$me" "$pid" 0)" = stale ] || fail "a dead owner's lock is not stale"
  [ "$(lock_verdict "$me" "$pid" 1)" = busy ] || fail "a dead owner whose remote run is going is not busy"
  k1=$(printf '%064d' 1 | tr 0 a); k2=$(printf '%064d' 2 | tr 0 b)
  printf '%s\tgreen\td\tx\n' "$k1" > "$d/local.tsv"
  printf '%s\tgreen\td1\ttrk\n%s\tgreen\td2\ttrk\n%s\tgreen\td3\ttrk\nbad\tgreen\td\ttrk\n%s\tred\td\ttrk\n%s\tgreen\td\ttrk\textra\n%s\tgreen\td4\tother\n' \
    "$k1" "$k2" "$k2" "$k2" "$k2" "$k2" > "$d/remote.tsv"
  out=$(ledger_new "$d/local.tsv" "$d/remote.tsv" trk)
  [ "$out" = "$(printf '%s\tgreen\td2\ttrk@remote' "$k2")" ] || fail "ledger_new gave: $out"
  : > "$d/empty.tsv"
  [ "$(ledger_new "$d/empty.tsv" "$d/remote.tsv" trk | wc -l | tr -d ' ')" = 2 ] || fail "ledger_new against an empty local ledger"

  # the ssh failure paths and the whole flow against a stub ssh that runs the "remote" in a fake home
  stub=$d/stub
  rh=$d/rhome
  mkdir -p "$stub" "$rh"
  cat > "$stub/ssh" << 'STUB'
#!/bin/bash
# a stand-in for ssh: STUB_MODE=down fails like an unreachable host (naming it), else the remote is $STUB_HOME
while [ $# -gt 0 ]; do case $1 in -q) shift ;; -o | -l) shift 2 ;; -*) shift ;; *) shift; break ;; esac; done
[ "${STUB_MODE:-}" != down ] || { echo "ssh: connect to host stubhost.example port 22: Connection refused" >&2; exit 255; }
cd "$STUB_HOME" && HOME=$STUB_HOME exec bash -c "$*"
STUB
  printf '#!/bin/sh\necho "just 9.9.9"\n' > "$stub/just"
  chmod +x "$stub/ssh" "$stub/just"
  printf '%s\n' 'stubuser@stubhost.example' > "$fake/.config/bloomery/mac-remote"
  rc=0; out=$(PATH=$stub:$PATH STUB_MODE=down HOME=$fake "$HERE/tools/mac-remote.sh" --tree "$repo" 2>&1) || rc=$?
  [ "$rc" = 75 ] || fail "an unreachable remote ended rc $rc, not 75: $out"
  case $out in *stubhost* | *stubuser*) fail "the unreachable message printed the host: $out" ;; esac
  case $out in *"not reachable"*) ;; *) fail "the unreachable remote is not named: $out" ;; esac
  rc=0; out=$(PATH=$stub:$PATH STUB_HOME=$rh HOME=$fake "$HERE/tools/mac-remote.sh" --tree "$repo" 2>&1) || rc=$?
  [ "$rc" = 69 ] || fail "a remote with nothing installed ended rc $rc, not 69: $out"
  case $out in *"missing:"*) ;; *) fail "the missing prerequisite is not named: $out" ;; esac
  case $out in *stubhost* | *stubuser* | *"$rh"*) fail "the missing-prerequisite message printed the account: $out" ;; esac

  if [ "$(uname -s)" != Darwin ]; then # the preflight asks a macOS host
    echo "mac-remote: self-test ok (the flow against a fake remote needs macOS: skipped on $(uname -s))"
    rm -rf "$d"
    [ "$fails" = 0 ] || { say "mac-remote: self-test $fails failed"; return 1; }
    return 0
  fi
  fake_remote "$rh" nightly-x
  rt=$rh/repo/bloomery-mr/tree-one
  run() { PATH=$stub:$PATH STUB_HOME=$rh HOME=$fake "$HERE/tools/mac-remote.sh" --tree "$repo" "$@" 2>&1; }
  rc=0; out=$(run HEAD~1) || rc=$?
  [ "$rc" = 0 ] || fail "the first flow ended rc $rc: $out"
  case $out in *"stub just in tree-one"*) ;; *) fail "the run's output did not come back: $out" ;; esac
  case $out in *"run slot 1 of 2 taken after 0 s"*) ;; *) fail "the run did not take the first free slot: $out" ;; esac
  case $out in *"ledger: 1 green shape line(s) this run, 1 appended here as tree-one@remote, 0 already green here"*) ;; *) fail "the ledger line did not come back: $out" ;; esac
  case $out in *"$rh"* | *stubuser*) fail "the flow's output printed the account: $out" ;; esac
  [ "$(cat "$rh/just.args")" = "mac-static $(git -C "$repo" rev-parse HEAD~1)" ] || fail "mac-static got '$(cat "$rh/just.args")', not HEAD~1's sha"
  [ "$(git -C "$rt" rev-parse HEAD)" = "$(git -C "$repo" rev-parse HEAD)" ] || fail "the remote clone is not at HEAD"
  cmp -s "$repo/sub/f" "$rt/sub/f" || fail "the tree's content did not reach the remote"
  [ ! -e "$rt/target" ] || fail "target/ was synced"
  [ -d "$rt/.git" ] || fail "the remote clone has no .git directory"
  [ ! -d "$rh/repo/bloomery-mr/.lock-tree-one" ] || fail "the lock outlived the run"
  [ "$(tail -1 "$fake/.cache/bloomery/mac-static-ledger.tsv" | cut -f4)" = tree-one@remote ] || fail "the pulled ledger line has no @remote tree"
  # uncommitted edits arrive, a deleted file goes, an unchanged file keeps its mtime, a repeated key is not appended twice.
  # The touch (mtime kept, ctime moved) is the first run's rsync (-p) reaching the file a second after the checkout,
  # as on a loaded remote: the file's ctime then differs from the index's, and a bare `checkout -f` rewrote it.
  sleep 1; touch -m -r "$rt/rust-toolchain.toml" "$rt/rust-toolchain.toml"
  rt0=$(stat -f %m "$rt/rust-toolchain.toml")
  sleep 1
  printf 'edit\n' >> "$repo/sub/f"; rm "$repo/sub/g"; printf 'new\n' > "$repo/new.txt"
  rc=0; out=$(run) || rc=$?
  [ "$rc" = 0 ] || fail "the second flow ended rc $rc: $out"
  cmp -s "$repo/sub/f" "$rt/sub/f" || fail "an uncommitted edit did not reach the remote"
  [ "$(cat "$rt/new.txt")" = new ] || fail "an untracked file did not reach the remote"
  [ ! -e "$rt/sub/g" ] || fail "a deleted file stayed on the remote"
  [ "$(stat -f %m "$rt/rust-toolchain.toml")" = "$rt0" ] || fail "an unchanged file was rewritten (cargo would rebuild it)"
  case $out in *"ledger: 1 green shape line(s) this run, 0 appended here as tree-one@remote, 1 already green here"*) ;; *) fail "a repeated key was appended again: $out" ;; esac
  [ "$(cat "$rh/just.args")" = "mac-static" ] || fail "a run with no BASE gave mac-static '$(cat "$rh/just.args")'"
  # rc passes through (a child's own 75 too); a live lock is waited for (TRACK_WAIT 0 in the copy: refused at once), a dead
  # owner's lock is taken over unless its remote run is going
  rc=0; out=$(STUB_JUST_RC=3 STUB_KEY=2 run) || rc=$?
  [ "$rc" = 3 ] || fail "the remote's rc 3 came back as $rc"
  case $out in *"red step fake-step (rc 3)"*"fake-step output line"*) ;; *) fail "the red step's log did not come back: $out" ;; esac
  case $out in *"ok-step output line"*) fail "a green step's log came back: $out" ;; esac
  case $out in *"$rh"*) fail "the red step's log printed the account: $out" ;; esac
  rc=0; out=$(STUB_JUST_RC=75 STUB_KEY=3 run) || rc=$?
  [ "$rc" = 75 ] || fail "the remote's own rc 75 came back as $rc, not 75: $out"
  case $out in *"all 2 run slots are taken"*) fail "a child's own rc 75 was read as a busy slot: $out" ;; esac
  sed 's/^TRACK_WAIT=.*/TRACK_WAIT=0/' "$HERE/tools/mac-remote.sh" > "$d/mr0.sh"
  grep -q '^TRACK_WAIT=0$' "$d/mr0.sh" || fail "the TRACK_WAIT=0 copy was not made"
  run0() { PATH=$stub:$PATH STUB_HOME=$rh HOME=$fake bash "$d/mr0.sh" --tree "$repo" "$@" 2>&1; }
  mkdir -p "$rh/repo/bloomery-mr/.lock-tree-one"
  printf '%s\n' "$me/$$" > "$rh/repo/bloomery-mr/.lock-tree-one/owner"
  rc=0; out=$(run0) || rc=$?
  [ "$rc" = 75 ] || fail "a live lock ended rc $rc, not 75: $out"
  case $out in *"locked on the remote by $me/$$"*) ;; *) fail "the live lock's owner is not named: $out" ;; esac
  printf '%s\n' "$me/$pid" > "$rh/repo/bloomery-mr/.lock-tree-one/owner"
  rc=0; out=$(run) || rc=$?
  [ "$rc" = 0 ] || fail "a dead owner's lock was not taken over (rc $rc): $out"
  mkdir -p "$rh/repo/bloomery-mr/.lock-tree-one"
  printf '%s\n' "$me/$pid" > "$rh/repo/bloomery-mr/.lock-tree-one/owner"
  sleep 20 & pid=$!
  printf '%s\n' "$pid" > "$rh/repo/bloomery-mr/.lock-tree-one/run-pid"
  rc=0; out=$(run0) || rc=$?
  { kill "$pid"; wait "$pid"; } 2> /dev/null || true
  [ "$rc" = 75 ] || fail "a lock whose remote run is going ended rc $rc, not 75: $out"
  case $out in *"going"*) ;; *) fail "the running remote run is not named: $out" ;; esac
  # a live lock released while the run waits: the run goes on (TRACK_WAIT 1800 here)
  rm -rf "$rh/repo/bloomery-mr/.lock-tree-one"
  mkdir -p "$rh/repo/bloomery-mr/.lock-tree-one"
  printf '%s\n' "$me/$$" > "$rh/repo/bloomery-mr/.lock-tree-one/owner"
  run > "$d/wait-track.out" & pid=$!
  sleep 2
  rm -rf "$rh/repo/bloomery-mr/.lock-tree-one"
  rc=0; wait "$pid" || rc=$?
  [ "$rc" = 0 ] || fail "a run that waited for its track's lock ended rc $rc: $(cat "$d/wait-track.out")"
  grep -q "locked on the remote by $me/$$.*waiting" "$d/wait-track.out" || fail "the track-lock wait is not announced: $(cat "$d/wait-track.out")"
  # the slot pool (real lockf; a slot is taken when hold_slot returns and stays held until its stop file appears, so no wall-clock
  # guess decides a case, however late a loaded Mac starts the holder). Before it
  # takes a slot the run prints each slot's holder, `starting` (held, no holder file) or `free` (the lock is free, whatever a killed
  # holder's file says); with every slot held it repeats that line no sooner than SLOT_SAY seconds after the last print, and goes on
  # once a slot frees.
  sl=$rh/repo/bloomery-mr
  held=
  mkdir -p "$sl"
  hold_slot() { # hold_slot N [TRACK]: slot N is held when this returns, until $d/stop-N exists; with a TRACK, its holder file is there too
    rm -f "$d/stop-$1" "$d/held-$1"
    # the holder writes held-N under the lock, so its existence is the lock taken
    lockf -k -s -t 0 "$sl/.slot-$1" bash -c ': > "$2"; while [ ! -e "$1" ]; do sleep 0.2; done' _ "$d/stop-$1" "$d/held-$1" &
    hp=$!
    held="$held $hp"
    n=0
    until [ -e "$d/held-$1" ]; do
      n=$((n + 1))
      if [ "$n" -gt 600 ] || ! kill -0 "$hp" 2> /dev/null; then
        fail "slot $1's hold was not taken within 120 s"
        break
      fi
      sleep 0.2
    done
    [ -z "${2:-}" ] || printf '%s\n' "$2 $(date +%s)" > "$sl/.slot-$1.holder"
  }
  free_slots() { : > "$d/stop-0"; : > "$d/stop-1"; wait $held 2> /dev/null || true; held=; rm -f "$sl"/.slot-*.holder "$d"/stop-? "$d"/held-?; }
  [ "$(sed -n 's/^SLOT_SAY=\([0-9]*\).*/\1/p' "$HERE/tools/mac-remote.sh")" = 300 ] || fail "SLOT_SAY is not 300 (five minutes)"
  hold_slot 0 trackA
  rc=0; out=$(run) || rc=$?
  free_slots
  [ "$rc" = 0 ] || fail "a run beside one held slot ended rc $rc: $out"
  printf '%s\n' "$out" | grep -Eq "^mac-remote: run slots: \[slot 1: trackA for [0-9]+ s\] \[slot 2: free\]; tree-one$" || fail "the held slot's holder and the free slot are not named: $out"
  printf '%s\n' "$out" | grep -Eq "^mac-remote: run slot 2 of 2 taken after [01] s$" || fail "the run did not take the free slot 2: $out"
  case $out in *"run slots are taken"*) fail "a run that found a free slot said all were taken: $out" ;; esac
  hold_slot 0
  printf '%s\n' "ghost $(date +%s)" > "$sl/.slot-1.holder"
  rc=0; out=$(run) || rc=$?
  free_slots
  [ "$rc" = 0 ] || fail "a run beside a held slot with no holder file ended rc $rc: $out"
  printf '%s\n' "$out" | grep -Eq "^mac-remote: run slots: \[slot 1: starting\] \[slot 2: free\]; tree-one$" || fail "a held slot with no holder file is not 'starting', or a free slot with a left-behind holder file is not 'free': $out"
  sed 's/^SLOT_SAY=.*/SLOT_SAY=4/' "$HERE/tools/mac-remote.sh" > "$d/mr4.sh"
  grep -q '^SLOT_SAY=4$' "$d/mr4.sh" || fail "the SLOT_SAY=4 copy was not made"
  run4() { PATH=$stub:$PATH STUB_HOME=$rh HOME=$fake bash "$d/mr4.sh" --tree "$repo" "$@" 2>&1; }
  hold_slot 0 trackA
  hold_slot 1 trackB
  run4 > "$d/wait-slot.out" & pid=$!
  n=0
  until [ "$(grep -c '^mac-remote: all 2 run slots are taken' "$d/wait-slot.out")" -ge 2 ] || [ "$n" -ge 300 ]; do sleep 0.2; n=$((n + 1)); done
  free_slots
  rc=0; wait "$pid" || rc=$?
  [ "$rc" = 0 ] || fail "a run that waited for a slot ended rc $rc: $(cat "$d/wait-slot.out")"
  [ "$(grep -c '^mac-remote: run slots:' "$d/wait-slot.out")" = 1 ] || fail "the slots' line before the first slot is not printed exactly once: $(cat "$d/wait-slot.out")"
  grep -Eq "^mac-remote: run slots: \[slot 1: trackA for [0-9]+ s\] \[slot 2: trackB for [0-9]+ s\]; tree-one$" "$d/wait-slot.out" || fail "the held slots' holders are not named at the start: $(cat "$d/wait-slot.out")"
  grep -q "all 2 run slots are taken: \[slot 1: trackA for [0-9]* s\] \[slot 2: trackB for [0-9]* s\]; tree-one waits" "$d/wait-slot.out" ||
    fail "the busy slots' holders are not named: $(cat "$d/wait-slot.out")"
  w=$(grep -o 'tree-one waits ([0-9]* s)' "$d/wait-slot.out" | tr -dc '0-9\n' | tr '\n' ' ')
  set -- $w
  [ "$#" -ge 2 ] || fail "the wait line did not come twice: $(cat "$d/wait-slot.out")"
  [ "${1:-0}" -ge 4 ] || fail "the first wait line came ${1:-?} s after the start line, before SLOT_SAY (4 s): $(cat "$d/wait-slot.out")"
  [ $(( ${2:-0} - ${1:-0} )) -ge 4 ] || fail "two wait lines ${1:-?} s and ${2:-?} s into the wait are closer than SLOT_SAY (4 s): $(cat "$d/wait-slot.out")"
  grep -q "run slot 1 of 2 taken after [1-9][0-9]* s" "$d/wait-slot.out" || fail "the freed slot 1 was not taken after a wait: $(cat "$d/wait-slot.out")"
  [ ! -e "$rh/repo/bloomery-mr/.slot-0.holder" ] || fail "the run left its holder file"
  [ ! -e "$rh/repo/bloomery-mr/.slot-rc-tree-one" ] || fail "the run left its rc file"
  # the slot queue: a ticket whose run is gone does not hold a run back, and is dropped
  sleep 0 & gp=$!
  wait "$gp"
  printf '%s %s\n' "$gp" "Thu Jan 1 00:00:00 1970" > "$sl/.slot-queue/000000000000.ghost"
  rc=0; out=$(run) || rc=$?
  [ "$rc" = 0 ] || fail "a run behind a dead ticket ended rc $rc: $out"
  case $out in *"ahead of it"*) fail "a dead ticket was counted ahead of a run: $out" ;; esac
  [ ! -e "$sl/.slot-queue/000000000000.ghost" ] || fail "the dead ticket was not dropped"
  # three runs that start in that order, both slots held, get a slot in that order. When slot 1 frees, the first two
  # are stopped (SIGSTOP to the pid each wrote into its ticket at its start) and only the third, the last to come,
  # looks at the slots: it takes the free one unless the queue holds it back.
  sed 's/^SLOT_POLL=.*/SLOT_POLL=1/' "$HERE/tools/mac-remote.sh" > "$d/mrq.sh"
  grep -q '^SLOT_POLL=1$' "$d/mrq.sh" || fail "the SLOT_POLL=1 copy was not made"
  runq() { PATH=$stub:$PATH STUB_HOME=$rh HOME=$fake bash "$d/mrq.sh" --tree "$1" 2>&1; }
  for t in tree-two tree-three; do git clone -q "$repo" "$d/$t"; done
  rm -f "$rh/just.order"
  hold_slot 0 trackA
  hold_slot 1 trackB
  qpids=
  for t in tree-one tree-two tree-three; do
    runq "$d/$t" > "$d/queue-$t.out" &
    qpids="$qpids $!"
    n=0
    until ls "$sl/.slot-queue/"*".$t" > /dev/null 2>&1 || [ "$n" -ge 600 ]; do sleep 0.2; n=$((n + 1)); done
    [ "$n" -lt 600 ] || fail "$t's ticket did not appear within 120 s: $(cat "$d/queue-$t.out")"
  done
  stopped=
  for t in tree-one tree-two; do
    read -r p _ < "$(ls "$sl/.slot-queue/"*".$t")"
    kill -STOP "$p" && stopped="$stopped $p"
  done
  sleep 0.5
  : > "$d/stop-0"
  sleep 3
  grep -q "run slot [0-9]* of [0-9]* taken" "$d/queue-tree-three.out" && fail "the third run took the freed slot ahead of the two before it: $(cat "$d/queue-tree-three.out")"
  [ -z "$stopped" ] || kill -CONT $stopped
  rc=0
  for p in $qpids; do wait "$p" || rc=$?; done
  free_slots
  [ "$rc" = 0 ] || fail "a queued run ended rc $rc: $(cat "$d"/queue-*.out)"
  [ "$(tr '\n' ' ' < "$rh/just.order")" = "tree-one tree-two tree-three " ] || fail "the queued runs took slots in the order $(tr '\n' ' ' < "$rh/just.order"), not tree-one tree-two tree-three"
  grep -Eq "^mac-remote: run slots: .*; tree-two, 1 ahead of it in the slot queue$" "$d/queue-tree-two.out" || fail "the second run's place is not named: $(cat "$d/queue-tree-two.out")"
  grep -Eq "^mac-remote: run slots: .*; tree-three, 2 ahead of it in the slot queue$" "$d/queue-tree-three.out" || fail "the third run's place is not named: $(cat "$d/queue-tree-three.out")"
  ! ls "$sl/.slot-queue/"[0-9]* > /dev/null 2>&1 || fail "the queue kept a ticket after its run: $(ls "$sl/.slot-queue")"

  rm -rf "$d"
  if [ "$fails" != 0 ]; then
    say "mac-remote: self-test $fails failed"
    return 1
  fi
  echo "mac-remote: self-test ok"
}

TREE_ARG=$HERE
BASE_ARG_IN=
nb=0
while [ $# -gt 0 ]; do
  case $1 in
    --self-test) [ $# = 1 ] || { say "mac-remote.sh: --self-test takes nothing"; exit 64; }; self_test; exit $? ;;
    --tree) [ $# -ge 2 ] || { say "mac-remote.sh: --tree needs a directory"; exit 64; }; TREE_ARG=$2; shift 2 ;;
    -*) say "usage: tools/mac-remote.sh [--tree DIR] [BASE] | --self-test ('$1' is not one)"; exit 64 ;;
    *) nb=$((nb + 1)); BASE_ARG_IN=$1; shift ;;
  esac
done
[ "$nb" -le 1 ] || { say "mac-remote.sh: one optional BASE (a rev, or A..B), got $nb"; exit 64; }
remote_main "$TREE_ARG" "$BASE_ARG_IN"
