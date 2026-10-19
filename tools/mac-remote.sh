#!/usr/bin/env bash
# bloomery — the Mac static tier on the second Mac. `just mac-static` takes 590-820 s on a Mac that also runs
# five Claude sessions (load 40-80, the disk 97 % full, a timing self-test inside check-recipes red from the
# load alone); the second Mac is idle and runs the same step list on a tree sent to it.
#
#   tools/mac-remote.sh [--tree DIR] [BASE]   `just mac-static BASE` for the worktree DIR (default: the tree
#                                             this script is in) on the second Mac; no BASE is the full run, which
#                                             holds check-model-tax to origin/main: DIR's origin/main goes to the
#                                             clone as that ref, and a DIR without one is refused (64), as its own
#                                             mac-static would end rc 2 there.
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
# clone (objects only; with no BASE, DIR's origin/main too, as refs/remotes/origin/main), `git checkout --detach
# HEAD` there, rsync of DIR's content over it, `cargo fetch --locked`, `just mac-static BASE`, and the ledger lines the run
# appended copied back. The rsync compares by checksum and keeps the receiver's mtimes: a file whose content changed gets
# the remote's "now" and cargo rebuilds it, an unchanged one is left alone (the rule tools/box.sh syncs by). target/ stays on
# the remote, one per track: a track's first run is a cold build, the next ones warm.
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
# The lock also holds its start time and a heartbeat the remote run refreshes every 60 s; the remote run removes the
# lock itself when it ends (an ssh drop hangs it up and ends it), and a lock with no live remote run and a heartbeat
# older than LOCK_DEAD s is dead and reclaimed, the old owner logged. The pool and `--locks` are described below.
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
# (missing, or not one `user@host` line) or the tree this script does not take, or a BASE that is no commit, or no BASE in a
# tree with no origin/main.
# 65: the tree has a state rsync does not carry (a submodule, a symlink out of the tree, a tracked file under
# target/). 69: a prerequisite is missing on the remote, named with where SETUP.md makes it. 75: the remote
# is not reachable or the connection was lost mid-run (the remote run may still finish), or the track's
# lock was still held by a live run after TRACK_WAIT seconds.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
# BLOOMERY_MAC_REMOTE=<name> picks another Mac of the pool: ~/.config/bloomery/mac-remote.d/<name>, same one-line form;
# each Mac keeps its own slots, locks and clones, so the pool is just the choice of file.
if [ -n "${BLOOMERY_MAC_REMOTE:-}" ]; then
  [[ $BLOOMERY_MAC_REMOTE =~ ^[A-Za-z0-9._-]+$ ]] || { printf '%s\n' "mac-remote: BLOOMERY_MAC_REMOTE must be one word of [A-Za-z0-9._-]" >&2; exit 64; }
  CONF=$HOME/.config/bloomery/mac-remote.d/$BLOOMERY_MAC_REMOTE
else
  CONF=$HOME/.config/bloomery/mac-remote
fi
LEDGER=$HOME/.cache/bloomery/mac-static-ledger.tsv
SSH_OPTS=(-o LogLevel=ERROR -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
RBASE=repo/bloomery-mr # the clones' directory, relative to the remote home
SLOTS=2 # runs at once on the remote; a further run waits for a slot
SLOT_SAY=300 # seconds between two prints of the slots' line while a run waits for a slot
SLOT_POLL=3 # seconds between a waiting run's two looks at the queue and the slots
TRACK_WAIT=1800 # seconds a run waits for its track's lock held by a live run
PUSH_RETRY_S=5 # seconds before the one retry of a failed git push to the clone
LOCK_DEAD=${LOCK_DEAD:-600} # a lock with no live remote run and a heartbeat older than this is dead and reclaimed

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

# The pool. With BLOOMERY_MAC_REMOTE unset and files in ~/.config/bloomery/mac-remote.d (one `user@host` line
# each, the file's name the Mac's alias), a run picks its Mac itself: every Mac is surveyed at once over ssh with a
# short ConnectTimeout (POOL_CONNECT s); one that does not answer, or has less than POOL_MIN_GB free, is left out.
# Of the rest, a Mac with a free slot wins: first one whose clone of this track already has a target/ (warm), then
# the one with the most free slots, then the shortest slot queue. A slot counts as taken while this
# machine's runner that picked that Mac (a reservation in ~/.cache/bloomery/mac-pool, under one mkdir lock) has
# not taken a slot there yet, so two runners started together spread out. The claim itself stays the remote's
# lockf slot: a runner that loses a race queues on that Mac in its turn, never shares a slot. With no slot free
# anywhere it prints every Mac's slots and surveys again every POOL_POLL s, up to POOL_WAIT s, then queues on the
# least loaded Mac. Track locks whose run is gone are named (`stale?`) and never removed for another track.
POOL_DIR=$HOME/.config/bloomery/mac-remote.d
POOL_CONNECT=${POOL_CONNECT:-5}
POOL_MIN_GB=${POOL_MIN_GB:-15}
POOL_POLL=${POOL_POLL:-20}
POOL_WAIT=${POOL_WAIT:-1800}
RESV_DIR=$HOME/.cache/bloomery/mac-pool

SURVEY='{
exec < /dev/null
track=$1 slots=$2
base=$HOME/'"$RBASE"'
mkdir -p "$base"
echo "GB=$(df -g "$HOME" 2> /dev/null | awk "NR == 2 { print \$4 }")"
free=0 hold=
j=0
while [ "$j" -lt "$slots" ]; do
  if lockf -k -s -t 0 "$base/.slot-$j" true 2> /dev/null; then free=$((free + 1)); hold="$hold [slot $((j + 1)): free]"
  else h=$(cat "$base/.slot-$j.holder" 2> /dev/null || true); set -- $h; hold="$hold [slot $((j + 1)): ${1:-starting}]"; fi
  j=$((j + 1))
done
echo "FREE=$free"
echo "HOLD=$hold"
q=0
for f in "$base"/.slot-queue/[0-9]*; do [ -e "$f" ] || continue; read -r p s < "$f" 2> /dev/null; kill -0 "$p" 2> /dev/null && q=$((q + 1)); done
echo "QUEUE=$q"
if [ -d "$base/$track/target" ]; then echo WARM=1; else echo WARM=0; fi
for l in "$base"/.lock-*; do
  [ -d "$l" ] || continue
  rp=$(cat "$l/run-pid" 2> /dev/null || true)
  alive=0; [ -n "$rp" ] && kill -0 "$rp" 2> /dev/null && alive=1
  echo "LOCK=${l##*/.lock-} $(cat "$l/owner" 2> /dev/null || echo ?/0) $alive $(( $(date +%s) - $(cat "$l/heartbeat" 2> /dev/null || stat -f %m "$l" 2> /dev/null || date +%s) ))"
done
exit 0
}'

# pool_choose: survey records on stdin, one a Mac: `alias reachable gb free queue warm tracklocked reserved`;
# the alias to run on and why on stdout (`alias free|queue …`), or nothing when no Mac qualifies at all.
pool_choose() {
  awk -v min="$POOL_MIN_GB" '
    $2 != 1 || $3 + 0 < min { next }
    { eff = $4 - $8; if (eff < 0) eff = 0
      if (eff > 0 && $7 == 0) k = sprintf("1 %d %02d %04d %s", $6, eff, 9999 - $5, $1)
      else k = sprintf("0 %d %04d %02d %s", $7 == 0, 9999 - $5 - ($4 - eff), $4, $1)
      if (best == "" || k > best) { best = k; who = $1; why = (eff > 0 && $7 == 0) ? "free" : "queue"; w = $6; e = eff } }
    END { if (who != "") printf "%s %s %d %d\n", who, why, e, w }'
}

# lock_note LOCKLINE: a survey LOCK line (`track owner run-alive heartbeat-age`) as words for the log when the lock is
# dead (no live remote run, heartbeat older than LOCK_DEAD), or nothing for a live one.
lock_note() {
  local name owner alive age
  read -r name owner alive age <<< "$1"
  [ "$alive" = 0 ] && [ "${age:-0}" -ge "$LOCK_DEAD" ] || return 0
  echo "track $name: dead lock (owner $owner, no run going, heartbeat $age s old): the next run of $name reclaims it, or just mac-locks --clean"
}

resv_lock() { local n=0; mkdir -p "$RESV_DIR"; until mkdir "$RESV_DIR/.lock" 2> /dev/null; do n=$((n + 1)); [ "$n" -lt 100 ] || rm -rf "$RESV_DIR/.lock"; sleep 0.1; done; }
resv_unlock() { rmdir "$RESV_DIR/.lock" 2> /dev/null || true; }
# resv_count ALIAS HOLD: this machine's live reservations of ALIAS whose track does not hold a slot there yet.
resv_count() {
  local f p t k=0
  for f in "$RESV_DIR/$1".*; do
    [ -f "$f" ] || continue
    read -r p t < "$f" || continue
    kill -0 "$p" 2> /dev/null || { rm -f "$f"; continue; }
    case $2 in *": $t]"*) ;; *) k=$((k + 1)) ;; esac
  done
  echo "$k"
}

# pool_pick TRACK: sets CONF and MAC_NAME to the chosen Mac, or 64 / 75.
pool_pick() {
  local tr=$1 d f a out gb fr q w hold tl lines rec choice t0=$SECONDS last=-999 notes
  d=$(mktemp -d "${TMPDIR:-/tmp}/mac-pool.XXXXXX")
  local -a names=()
  for f in "$POOL_DIR"/*; do [ -f "$f" ] && [[ ${f##*/} =~ ^[A-Za-z0-9._-]+$ ]] && names+=("${f##*/}"); done
  [ ${#names[@]} -gt 0 ] || { rm -rf "$d"; say "mac-remote: no Mac in ~/.config/bloomery/mac-remote.d"; return 64; }
  while :; do
    for a in "${names[@]}"; do
      ( CONF=$POOL_DIR/$a; h=$(read_host 2> /dev/null) || exit 64
        ssh -o LogLevel=ERROR -o BatchMode=yes -o ConnectTimeout="$POOL_CONNECT" -o ServerAliveInterval=5 -o ServerAliveCountMax=2 \
          "$h" "bash -s -- $(qargs "$tr" "$SLOTS")" <<< "$SURVEY" > "$d/$a" 2> /dev/null ) &
    done
    wait
    resv_lock
    lines=
    notes=
    for a in "${names[@]}"; do
      out=$(cat "$d/$a" 2> /dev/null || true)
      gb=$(sed -n 's/^GB=//p' <<< "$out"); fr=$(sed -n 's/^FREE=//p' <<< "$out"); q=$(sed -n 's/^QUEUE=//p' <<< "$out")
      w=$(sed -n 's/^WARM=//p' <<< "$out"); hold=$(sed -n 's/^HOLD=//p' <<< "$out")
      if [ -z "$fr" ]; then lines+="$a 0 0 0 0 0 0 0"$'\n'; notes+="  $a: not reachable in $POOL_CONNECT s, skipped"$'\n'; continue; fi
      tl=0
      while IFS= read -r rec; do
        [ -n "$rec" ] || continue
        set -- $rec
        [ "$1" != "$tr" ] || [ "$3" = 0 ] || tl=1
        rec=$(lock_note "$rec"); [ -z "$rec" ] || notes+="  $a: $rec"$'\n'
      done <<< "$(sed -n 's/^LOCK=//p' <<< "$out")"
      lines+="$a 1 ${gb:-0} $fr ${q:-0} ${w:-0} $tl $(resv_count "$a" "$hold")"$'\n'
      [ "${gb:-0}" -ge "$POOL_MIN_GB" ] || notes+="  $a: ${gb:-?} GB free, under $POOL_MIN_GB, skipped"$'\n'
      rec=; [ "$w" != 1 ] || rec=", warm target/ for $tr"
      notes+="  $a:$hold, queue ${q:-0}$rec"$'\n'
    done
    choice=$(printf '%s' "$lines" | pool_choose)
    set -- $choice
    if [ -n "$choice" ] && { [ "$2" = free ] || [ $((SECONDS - t0)) -ge "$POOL_WAIT" ]; }; then
      echo "$$ $tr" > "$RESV_DIR/$1.$$"
      RESV_FILE=$RESV_DIR/$1.$$
      resv_unlock
      [ "$last" -ge 0 ] || printf 'mac-remote: pool:\n%s' "$notes" >&2
      MAC_NAME=$1
      CONF=$POOL_DIR/$1
      say "mac-remote: picked $1 ($([ "$2" = free ] && echo "$3 slot(s) free" || echo "all busy after $POOL_WAIT s, queueing")$([ "$4" != 1 ] || echo ", warm target/"))"
      rm -rf "$d"
      return 0
    fi
    resv_unlock
    if [ -z "$choice" ]; then printf 'mac-remote: pool:\n%s' "$notes" >&2; say "mac-remote: no Mac of the pool is reachable with $POOL_MIN_GB GB free"; rm -rf "$d"; return 75; fi
    if [ $((SECONDS - last)) -ge "$SLOT_SAY" ]; then
      printf 'mac-remote: all slots of the pool are taken (%s s):\n%s' "$((SECONDS - t0))" "$notes" >&2
      last=$SECONDS
    fi
    sleep "$POOL_POLL"
  done
}

# --locks [--clean]: every Mac's slots and track locks (the pool's, or the one Mac of ~/.config/bloomery/mac-remote),
# each lock `live` (its run goes, or its heartbeat is younger than LOCK_DEAD) or `dead`; --clean removes the dead ones.
LOCKS='{
exec < /dev/null
slots=$1 clean=$2 dead=$3
base=$HOME/'"$RBASE"'
mkdir -p "$base"
now=$(date +%s)
mtime() { stat -f %m "$1" 2> /dev/null || stat -c %Y "$1" 2> /dev/null || echo "$now"; }
j=0
while [ "$j" -lt "$slots" ]; do
  if ! command -v lockf > /dev/null; then st="unknown (no lockf)"
  elif lockf -k -s -t 0 "$base/.slot-$j" true 2> /dev/null; then st=free
  else h=$(cat "$base/.slot-$j.holder" 2> /dev/null || true); set -- $h; if [ -n "$h" ]; then st="$1 for $((now - $2)) s"; else st=starting; fi; fi
  echo "  slot $((j + 1)): $st"
  j=$((j + 1))
done
n=0
for l in "$base"/.lock-*; do
  [ -d "$l" ] || continue
  n=$((n + 1))
  t=${l##*/.lock-}
  rp=$(cat "$l/run-pid" 2> /dev/null || true)
  alive=0; [ -n "$rp" ] && kill -0 "$rp" 2> /dev/null && alive=1
  hb=$(cat "$l/heartbeat" 2> /dev/null || mtime "$l"); age=$((now - hb))
  if [ "$alive" = 1 ] || [ "$age" -lt "$dead" ]; then v=live; else v=dead; fi
  line="  lock $t: $v, owner $(cat "$l/owner" 2> /dev/null || echo unknown), run pid ${rp:-none}$([ "$alive" = 1 ] && echo " alive" || echo ""), heartbeat $age s old"
  if [ "$v" = dead ] && [ "$clean" = 1 ]; then rm -rf "$l" && line="$line: removed"; fi
  echo "$line"
done
[ "$n" -gt 0 ] || echo "  no track locks"
exit 0
}'

locks_main() {
  local clean=$1 a f rc=0 h
  local -a names=()
  if compgen -G "$POOL_DIR/*" > /dev/null; then
    for f in "$POOL_DIR"/*; do [ -f "$f" ] && names+=("${f##*/}"); done
  else
    names=(.)
  fi
  for a in "${names[@]}"; do
    if [ "$a" = . ]; then CONF=$HOME/.config/bloomery/mac-remote; a="the second Mac"; else CONF=$POOL_DIR/$a; fi
    h=$(read_host) || { rc=64; continue; }
    echo "$a:"
    ssh -o LogLevel=ERROR -o BatchMode=yes -o ConnectTimeout="$POOL_CONNECT" "$h" "bash -s -- $(qargs "$SLOTS" "$clean" "$LOCK_DEAD")" <<< "$LOCKS" 2> /dev/null |
      scrub "" "${h%@*}" "${h#*@}" || { echo "  not reachable in $POOL_CONNECT s"; rc=75; }
  done
  return $rc
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
track=$1 owner=$2 channel=$3 want_just=$4 steal=$5 dead=${6:-600}
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
mtime() { stat -f %m "$1" 2> /dev/null || stat -c %Y "$1" 2> /dev/null || date +%s; }
if [ -d "$lock" ]; then
  rp=$(cat "$lock/run-pid" 2> /dev/null || true)
  if [ -z "$rp" ] || ! kill -0 "$rp" 2> /dev/null; then
    hb=$(cat "$lock/heartbeat" 2> /dev/null || mtime "$lock")
    age=$(( $(date +%s) - ${hb:-0} ))
    if [ "$age" -ge "$dead" ]; then
      echo "RECLAIMED=owner $(cat "$lock/owner" 2> /dev/null || echo unknown), started $(cat "$lock/started" 2> /dev/null || echo unknown), run pid ${rp:-none} not alive, heartbeat $age s old"
      rm -rf "$lock"
    fi
  fi
fi
if ! mkdir "$lock" 2> /dev/null; then
  echo "HELD=$(cat "$lock/owner" 2> /dev/null)"
  rp=$(cat "$lock/run-pid" 2> /dev/null || true)
  if [ -n "$rp" ] && kill -0 "$rp" 2> /dev/null; then echo "RUNNING=$rp"; fi
  exit 75
fi
echo "$owner" > "$lock/owner"
date +%s > "$lock/started"
date +%s > "$lock/heartbeat"
[ -d "$base/$track/.git" ] || { mkdir -p "$base/$track"; git -C "$base/$track" init -q; }
n=0
[ ! -f "$HOME/.cache/bloomery/mac-static-ledger.tsv" ] || n=$(wc -l < "$HOME/.cache/bloomery/mac-static-ledger.tsv" | tr -d " ")
echo "LEDGER_N=$n"
exit 0
}'

CHECKOUT='{
exec < /dev/null
date +%s > "$HOME/'"$RBASE"'/.lock-$1/heartbeat" 2> /dev/null
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
# From here on the lock belongs to this run: a heartbeat every 60 s while it lives, and the whole lock removed when it ends,
# however it ends. An ssh drop (HUP) or a signal ends the run and everything it started (its process group), then the
# EXIT trap removes the lock, so a lost session leaves no lock behind.
lk=$base/.lock-$track
echo $$ > "$lk/run-pid"
( while kill -0 $$ 2> /dev/null; do date +%s > "$lk/heartbeat" 2> /dev/null || exit 0; sleep 60; done ) > /dev/null 2>&1 &
hb=$!
ticket=
trap "kill $hb 2> /dev/null; rm -f \"\$ticket\"; rm -rf \"$lk\"" EXIT
trap "trap \"\" HUP INT TERM PIPE; kill -TERM 0 2> /dev/null; exit 129" HUP INT TERM PIPE
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
      lockf -k -s -t 0 "$base/.slot-$i" bash -c "slot_inner \"\$@\"" _ "$i" "$((SECONDS - w0))" "$base/.slot-$i.holder" "$rcf" "${cmd[@]}" &
      # in the background and waited for, so a hangup runs the trap at once, not when the job ends
      lrc=0; wait $! || lrc=$?
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
  ssh "${SSH_OPTS[@]}" "$R" "l=$RBASE/.lock-$track; [ ! -d \$l ] || [ \"\$(cat \$l/owner 2> /dev/null)\" != $(qargs "$OWNER") ] || rm -rf \$l" > /dev/null 2>&1 ||
    say "mac-remote: could not release the lock of $track on the remote (the next run takes it over once this runner's pid is gone)"
}

cleanup() {
  { [ "${LOCKED:-0}" = 0 ] || [ "${RUN_ACTIVE:-0}" = 1 ]; } || release_lock
  [ -z "${TMP:-}" ] || rm -rf "$TMP"
  [ -z "${RESV_FILE:-}" ] || rm -f "$RESV_FILE"
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
  local tree=$1 basearg=$2 r base_list n_pushed om channel just_v owner ssh_rc t0=$SECONDS t_setup t_run
  local steal=0 tries=0 held running total appended tw0=$SECONDS twlast=-60
  local -a pushes=()
  TREE=$(cd "$tree" 2> /dev/null && pwd -P) || { say "mac-remote: --tree '$tree' is not a directory"; return 64; }
  [ "$(git -C "$TREE" rev-parse --show-toplevel 2> /dev/null || true)" = "$TREE" ] ||
    { say "mac-remote: $TREE is not the root of a git worktree"; return 64; }
  track=$(track_of "$TREE") || return $?
  MAC_NAME=${BLOOMERY_MAC_REMOTE:-}
  if [ -z "$MAC_NAME" ] && compgen -G "$POOL_DIR/*" > /dev/null; then
    pool_pick "$track" || return $?
  elif [ -n "$MAC_NAME" ]; then
    say "mac-remote: pinned to $MAC_NAME (BLOOMERY_MAC_REMOTE)"
  fi
  R=$(read_host) || return $?
  RUSER=${R%@*}
  RHOST=${R#*@}
  RHOME=
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
  else
    # No BASE: mac-static holds check-model-tax to origin/main (its base_rev), a ref the clone (a `git init` fed by these pushes)
    # lacks, so the remote run would end rc 2 where this tree's own run ends 0. The commit goes along as that ref.
    om=$(commit_of "$TREE" origin/main) ||
      { say "mac-remote: a run with no BASE holds check-model-tax to origin/main, which $TREE lacks: name a BASE"; return 64; }
    pushes+=("$om:refs/remotes/origin/main")
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
    ssh "${SSH_OPTS[@]}" "$R" "bash -s -- $(qargs "$track" "$owner" "$channel" "$just_v" "$steal" "$LOCK_DEAD")" < "$TMP/prep.sh" > "$TMP/prep.out" 2>&1 || ssh_rc=$?
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
  OWNER=$owner
  sed -n 's/^RECLAIMED=/mac-remote: reclaimed the dead lock of track '"$track"': /p' "$TMP/prep.out" | scrub "$RHOME" "$RUSER" "$RHOST" >&2
  LEDGER_N=$(sed -n 's/^LEDGER_N=//p' "$TMP/prep.out" | head -1)
  [[ $LEDGER_N =~ ^[0-9]+$ ]] || LEDGER_N=0
  say "mac-remote: track $track on ${MAC_NAME:-the second Mac}, HEAD ${HEAD_SHA:0:12}, base ${BASE_ARG:-none (full run)}"

  # objects, checkout, content
  push_clone "$track" || return $?
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
    say "mac-remote: the ssh session of the run was lost (rc 255): the remote run ends on the hangup and removes its own lock"
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

# fake_remote HOME CHANNEL: a remote home that passes the preflight, its tools stubs: uname answers Darwin and the Mac
# arch its toolchain is built for, just prints its arguments to just.args, appends a green ledger line under its
# directory's name and exits $STUB_JUST_RC.
fake_remote() {
  local rh=$1 ch=$2 h m t
  case $(uname -m) in
    arm64) h=aarch64-apple-darwin; m=arm64 ;;
    *) h=x86_64-apple-darwin; m=x86_64 ;;
  esac
  mkdir -p "$rh/.rustup/toolchains/$ch-$h/bin" "$rh/.rustup/toolchains/$ch-$h/lib/rustlib/x86_64-unknown-linux-gnu/lib" \
    "$rh/opt/bloomery-bin" "$rh/opt/cuda/include" "$rh/opt/clang" "$rh/opt/sr/usr/include"
  printf '#!/bin/sh\ncase $1 in -s) echo Darwin ;; -m) echo %s ;; *) /usr/bin/env -i PATH=/usr/bin:/bin uname "$@" ;; esac\n' "$m" > "$rh/opt/bloomery-bin/uname"
  chmod +x "$rh/opt/bloomery-bin/uname"
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
git rev-parse --verify -q 'origin/main^{commit}' > "$HOME/just.originmain" 2> /dev/null || echo none > "$HOME/just.originmain"
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

# push_clone TRACK: git push of HEAD and BASE (pushes) to TRACK's clone; a failed push is tried once more after
# PUSH_RETRY_S, then ends 75 with git's last lines. The push is a fresh ssh session right after the preflight's own
# succeeded, so a failure here is most often a dropped connection, not a missing clone (the preflight made it).
push_clone() {
  local n
  for n in 1 2; do
    git -C "$TREE" push -q --no-verify -f "$R:$RBASE/$1" "${pushes[@]}" > "$TMP/push.log" 2>&1 && return 0
    [ "$n" = 2 ] && break
    say "mac-remote: git push to the remote clone failed; trying once more in ${PUSH_RETRY_S} s"
    sleep "$PUSH_RETRY_S" &
    SLPID=$!
    wait "$SLPID" || true
    SLPID=
  done
  say "mac-remote: git push to the remote clone failed twice:"; scrub "$RHOME" "$RUSER" "$RHOST" < "$TMP/push.log" | tail -5 >&2
  return 75
}

self_test() {
  local fails=0 d out rc t fake stub repo me k1 k2 rh rt pid
  fail() { say "mac-remote self-test FAIL: $*"; fails=$((fails + 1)); }
  d=$(mktemp -d "${TMPDIR:-/tmp}/mac-remote-st.XXXXXX")
  fake=$d/home
  mkdir -p "$fake/.config/bloomery"
  me=$(hostname -s | tr -c 'A-Za-z0-9.\n-' _)

  # the push: a dropped first push is tried once more; two failures end 75 with git's lines
  mkdir -p "$d/gitbin"
  printf '#!/bin/sh\nc=$(cat "%s/n" 2>/dev/null || echo 0); c=$((c + 1)); echo $c > "%s/n"; echo git >> "%s/calls"\n[ "$c" -gt "${FAIL_N:-0}" ] && exit 0\necho "fatal: Could not read from remote repository."; exit 128\n' "$d/gitbin" "$d/gitbin" "$d/gitbin" > "$d/gitbin/git"
  printf '#!/bin/sh\necho "sleep $1" >> "%s/calls"\n' "$d/gitbin" > "$d/gitbin/sleep"
  chmod +x "$d/gitbin/git" "$d/gitbin/sleep"
  out=$( (TREE=$d R=st-user@st-host RHOME= RUSER=st-user RHOST=st-host TMP=$d pushes=(x) SLPID=; PATH=$d/gitbin:$PATH FAIL_N=1; export FAIL_N; push_clone t) 2>&1) && rc=0 || rc=$?
  [ "$rc" = 0 ] && [ "$(tr '\n' ' ' < "$d/gitbin/calls")" = "git sleep $PUSH_RETRY_S git " ] && grep -qF 'trying once more' <<< "$out" ||
    fail "a push that failed once was not retried after the wait: rc $rc, calls $(tr '\n' ' ' < "$d/gitbin/calls"): $out"
  rm -f "$d/gitbin/n" "$d/gitbin/calls"
  out=$( (TREE=$d R=st-user@st-host RHOME= RUSER=st-user RHOST=st-host TMP=$d pushes=(x) SLPID=; PATH=$d/gitbin:$PATH FAIL_N=9; export FAIL_N; push_clone t) 2>&1) && rc=0 || rc=$?
  [ "$rc" = 75 ] && [ "$(tr '\n' ' ' < "$d/gitbin/calls")" = "git sleep $PUSH_RETRY_S git " ] && grep -qF 'Could not read from remote repository' <<< "$out" ||
    fail "a push that failed twice did not end 75 with git's line after two tries and one wait: rc $rc, calls $(tr '\n' ' ' < "$d/gitbin/calls" 2>/dev/null): $out"
  rm -f "$d/gitbin/n" "$d/gitbin/calls"

  # the pool: the choice, the lock notes, the reservations, the pin, an unreachable pool
  out=$(printf '%s\n' 'a 1 40 1 0 0 0 0' 'b 1 40 2 0 0 0 0' | POOL_MIN_GB=15 pool_choose)
  [ "${out%% *}" = b ] || fail "the Mac with more free slots was not picked: $out"
  out=$(printf '%s\n' 'a 1 40 1 0 1 0 0' 'b 1 40 2 0 0 0 0' | POOL_MIN_GB=15 pool_choose)
  [ "${out%% *}" = a ] || fail "the warm Mac was not preferred: $out"
  out=$(printf '%s\n' 'a 1 10 2 0 1 0 0' 'b 0 0 0 0 0 0 0' 'c 1 40 1 0 0 0 0' | POOL_MIN_GB=15 pool_choose)
  [ "${out%% *}" = c ] || fail "a low-disk or unreachable Mac was not skipped: $out"
  out=$(printf '%s\n' 'a 1 40 1 0 1 0 1' 'b 1 40 1 0 0 0 0' | POOL_MIN_GB=15 pool_choose)
  [ "${out%% *}" = b ] || fail "a slot reserved by a runner here was counted free: $out"
  out=$(printf '%s\n' 'a 1 40 2 0 1 1 0' 'b 1 40 1 0 0 0 0' | POOL_MIN_GB=15 pool_choose)
  [ "${out%% *}" = b ] || fail "a Mac where the track's lock is held by a live run was picked: $out"
  out=$(printf '%s\n' 'a 1 40 0 3 1 0 0' 'b 1 40 0 1 0 0 0' | POOL_MIN_GB=15 pool_choose)
  [ "$out" = "b queue 0 0" ] || fail "with every slot taken the shortest queue was not picked: $out"
  out=$(printf '%s\n' 'a 0 0 0 0 0 0 0' 'b 1 5 2 0 0 0 0' | POOL_MIN_GB=15 pool_choose)
  [ -z "$out" ] || fail "a pool of no usable Mac chose $out"
  sleep 0 & pid=$!; wait "$pid"
  [ -z "$(LOCK_DEAD=600 lock_note "t1 x/1 1 9999")" ] || fail "a lock whose remote run goes was called dead"
  [ -z "$(LOCK_DEAD=600 lock_note "t1 x/1 0 30")" ] || fail "a lock with a fresh heartbeat was called dead"
  case $(LOCK_DEAD=600 lock_note "t1 x/1 0 700") in *"dead lock"*) ;; *) fail "a lock with no run and an old heartbeat is not named dead" ;; esac
  ( RESV_DIR=$d/resv; mkdir -p "$RESV_DIR"
    echo "$$ tA" > "$RESV_DIR/m1.$$"; echo "$pid tB" > "$RESV_DIR/m1.$pid"
    [ "$(resv_count m1 ' [slot 1: free]')" = 1 ] || fail "a live reservation was not counted, or a dead one was"
    [ ! -e "$RESV_DIR/m1.$pid" ] || fail "a dead runner's reservation was not dropped"
    [ "$(resv_count m1 ' [slot 1: tA]')" = 0 ] || fail "a reservation whose track holds a slot was still counted" )
  rc=0; out=$(HOME=$fake BLOOMERY_MAC_REMOTE='a b' "$HERE/tools/mac-remote.sh" HEAD 2>&1) || rc=$?
  [ "$rc" = 64 ] || fail "a BLOOMERY_MAC_REMOTE of two words ended rc $rc, not 64"
  rm -f "$fake/.config/bloomery/mac-remote"
  mkdir -p "$fake/.config/bloomery/mac-remote.d" "$d/sshbin"
  printf '%s\n' 'pool-leak-sentinel@nowhere.example' > "$fake/.config/bloomery/mac-remote.d/m1"
  printf '%s\n' 'pool-leak-sentinel@nowhere.example' > "$fake/.config/bloomery/mac-remote.d/m2"
  printf '#!/bin/sh\nexit 255\n' > "$d/sshbin/ssh"; chmod +x "$d/sshbin/ssh"
  rc=0; out=$(cd "$HERE" && HOME=$fake PATH=$d/sshbin:$PATH "$HERE/tools/mac-remote.sh" 2>&1) || rc=$?
  [ "$rc" = 75 ] || fail "a pool of unreachable Macs ended rc $rc, not 75: $out"
  case $out in *"m1: not reachable"*"m2: not reachable"*) ;; *) fail "the unreachable Macs are not named: $out" ;; esac
  case $out in *pool-leak-sentinel*|*nowhere.example*) fail "a pool host was printed: $out" ;; esac
  rc=0; out=$(cd "$HERE" && HOME=$fake BLOOMERY_MAC_REMOTE=m1 PATH=$d/sshbin:$PATH "$HERE/tools/mac-remote.sh" 2>&1) || rc=$?
  case $out in *"pinned to m1"*) ;; *) fail "the pin is not named: $out" ;; esac
  case $out in *"picked "*) fail "a pinned run surveyed the pool: $out" ;; esac
  rm -rf "$fake/.config/bloomery/mac-remote.d"

  # the lock's life, on Linux too (uname and lockf stubbed into the fake remote's PATH): a dead lock is reclaimed and
  # named, a live one is not; a run whose ssh drops (HUP) ends and removes its own lock; --locks --clean
  if command -v setsid > /dev/null; then
    local lr=$d/lockrem lb
    fake_remote "$lr" nightly-x
    printf '#!/bin/bash\nwhile [ "${1:-}" != "" ] && [ "${1#-}" != "$1" ]; do [ "$1" = -t ] && shift; shift; done\nshift\nexec "$@"\n' > "$lr/opt/bloomery-bin/lockf"
    printf '#!/bin/sh\n[ "$1" != --version ] || { echo "just 9.9.9"; exit 0; }\necho $$ > "$HOME/just.pid"\nsleep 30\n' > "$lr/opt/bloomery-bin/just"
    chmod +x "$lr/opt/bloomery-bin/lockf" "$lr/opt/bloomery-bin/just"
    lb=$lr/$RBASE
    mkdir -p "$lb/.lock-tdead" "$lb/.lock-tlive"
    echo "gone/1" > "$lb/.lock-tdead/owner"; echo $(( $(date +%s) - 900 )) > "$lb/.lock-tdead/heartbeat"
    echo "here/1" > "$lb/.lock-tlive/owner"; date +%s > "$lb/.lock-tlive/heartbeat"
    out=$(HOME=$lr bash -s -- tdead me/1 nightly-x "just 9.9.9" 0 600 <<< "$PREP" 2>&1) || fail "the preflight did not take a dead lock: $out"
    case $out in *"RECLAIMED=owner gone/1"*"heartbeat 9"*) ;; *) fail "the reclaim does not name the old owner and heartbeat: $out" ;; esac
    [ "$(cat "$lb/.lock-tdead/owner")" = me/1 ] || fail "the reclaimed lock is not the new owner's"
    rc=0; out=$(HOME=$lr bash -s -- tlive me/1 nightly-x "just 9.9.9" 0 600 <<< "$PREP" 2>&1) || rc=$?
    [ "$rc" = 75 ] || fail "a lock with a fresh heartbeat was taken (rc $rc): $out"
    [ "$(cat "$lb/.lock-tlive/owner")" = here/1 ] || fail "a live lock was overwritten"
    mkdir -p "$lb/tdead"
    HOME=$lr setsid bash -c 'bash -s -- tdead "" nightly-x 2 300 1 < "$0" > /dev/null 2>&1' <(printf '%s\n' "$RUN") &
    for _ in $(seq 100); do [ -s "$lr/just.pid" ] && break; sleep 0.1; done
    [ -s "$lr/just.pid" ] || fail "the stubbed run did not start"
    [ -s "$lb/.lock-tdead/heartbeat" ] || fail "the run wrote no heartbeat"
    kill -HUP "$(cat "$lb/.lock-tdead/run-pid")" 2> /dev/null || fail "no run pid to hang up"
    for _ in $(seq 50); do [ -d "$lb/.lock-tdead" ] || break; sleep 0.1; done
    [ ! -d "$lb/.lock-tdead" ] || fail "an ssh drop (HUP) left the run's lock behind"
    sleep 0.3
    ! kill -0 "$(cat "$lr/just.pid")" 2> /dev/null || fail "the run's job outlived the hangup"
    echo "gone/2" > "$lb/.lock-tlive/owner"; echo $(( $(date +%s) - 900 )) > "$lb/.lock-tlive/heartbeat"
    mkdir -p "$lb/.lock-tfresh"; date +%s > "$lb/.lock-tfresh/heartbeat"
    out=$(HOME=$lr bash -s -- 2 1 600 <<< "$LOCKS" 2>&1)
    case $out in *"lock tlive: dead"*"removed"*) ;; *) fail "--locks --clean did not remove the dead lock: $out" ;; esac
    [ ! -d "$lb/.lock-tlive" ] && [ -d "$lb/.lock-tfresh" ] || fail "--locks --clean removed the wrong locks: $out"
  fi

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
  git -C "$repo" update-ref refs/remotes/origin/main HEAD~1
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
  # a run with no BASE holds check-model-tax to origin/main (mac-static's base_rev): the clone has that ref at this tree's commit,
  # and a tree with no origin/main is refused (64) before anything is sent; a BASE needs no origin/main
  [ "$(cat "$rh/just.originmain")" = "$(git -C "$repo" rev-parse origin/main)" ] ||
    fail "the remote clone's origin/main is '$(cat "$rh/just.originmain")' on a run with no BASE, not this tree's $(git -C "$repo" rev-parse origin/main)"
  git -C "$repo" update-ref -d refs/remotes/origin/main
  rc=0; out=$(run) || rc=$?
  [ "$rc" = 64 ] || fail "a run with no BASE in a tree with no origin/main ended rc $rc, not 64: $out"
  case $out in *"holds check-model-tax to origin/main"*) ;; *) fail "the refusal does not say why: $out" ;; esac
  case $out in *"stub just in"*) fail "a run with no BASE and no origin/main reached the remote: $out" ;; esac
  rc=0; out=$(run HEAD~1) || rc=$?
  [ "$rc" = 0 ] || fail "a run with a BASE in a tree with no origin/main ended rc $rc: $out"
  git -C "$repo" update-ref refs/remotes/origin/main HEAD~1
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
  # the lock goes once the run has said it waits, not after a fixed sleep: a `sleep 2` lost that race on a loaded Mac under
  # check-recipes' parallel parts (2026-10-13), and the run then never printed its wait line. Bounded at 60 s, named.
  i=0
  until grep -q "locked on the remote by $me/$$.*waiting" "$d/wait-track.out" 2> /dev/null; do
    i=$((i + 1))
    if [ "$i" -gt 600 ] || ! kill -0 "$pid" 2> /dev/null; then
      fail "the track-lock wait was not announced within 60 s, or the run ended first: $(cat "$d/wait-track.out")"
      break
    fi
    sleep 0.1
  done
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
  printf '%s\n' "$out" | grep -E "^mac-remote: run slots: \[slot 1: trackA for [0-9]+ s\] \[slot 2: free\]; tree-one$" > /dev/null || fail "the held slot's holder and the free slot are not named: $out"
  printf '%s\n' "$out" | grep -E "^mac-remote: run slot 2 of 2 taken after [01] s$" > /dev/null || fail "the run did not take the free slot 2: $out"
  case $out in *"run slots are taken"*) fail "a run that found a free slot said all were taken: $out" ;; esac
  hold_slot 0
  printf '%s\n' "ghost $(date +%s)" > "$sl/.slot-1.holder"
  rc=0; out=$(run) || rc=$?
  free_slots
  [ "$rc" = 0 ] || fail "a run beside a held slot with no holder file ended rc $rc: $out"
  printf '%s\n' "$out" | grep -E "^mac-remote: run slots: \[slot 1: starting\] \[slot 2: free\]; tree-one$" > /dev/null || fail "a held slot with no holder file is not 'starting', or a free slot with a left-behind holder file is not 'free': $out"
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
    --locks) case ${2:-} in '') locks_main 0; exit $? ;; --clean) locks_main 1; exit $? ;; *) say "mac-remote.sh: --locks takes only --clean"; exit 64 ;; esac ;;
    --tree) [ $# -ge 2 ] || { say "mac-remote.sh: --tree needs a directory"; exit 64; }; TREE_ARG=$2; shift 2 ;;
    -*) say "usage: tools/mac-remote.sh [--tree DIR] [BASE] | --self-test ('$1' is not one)"; exit 64 ;;
    *) nb=$((nb + 1)); BASE_ARG_IN=$1; shift ;;
  esac
done
[ "$nb" -le 1 ] || { say "mac-remote.sh: one optional BASE (a rev, or A..B), got $nb"; exit 64; }
remote_main "$TREE_ARG" "$BASE_ARG_IN"
