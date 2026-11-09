#!/usr/bin/env bash
# bloomery — one command that retires every landed worktree beside the main tree and the copies it left elsewhere: its
# track directories on the box and the stale clones on the second Mac. The pieces are their own tools; this script only
# runs them in order and owns none of their rules.
#
#   tools/sweep.sh [--dry-run]   the four steps below (`just sweep`); --dry-run runs only the list modes and changes nothing
#   tools/sweep.sh --self-test   a scratch git repo with stub mac-disk, box-tracks, mac-remote and ssh; no box, no cargo
#
# 1. list   `tools/mac-disk.py report` (it runs du over every tree, so this step takes minutes on a full disk). Of its
#           rows, every LANDED or PRUNABLE worktree beside the main tree (<parent of main>/<main dir>-NAME, the tools/wt.sh
#           naming; paths from `git worktree list --porcelain`, joined on the basename the report prints) is picked. Any
#           other LANDED or PRUNABLE worktree (an Orca workspace under another directory) is never retired here: one line
#           `sweep: skip <path> (not beside the main tree)`, and a basename two worktrees share is skipped by name too.
# 2. retire `tools/mac-disk.py retire <absolute path>`, one call a name in report order: its refusals are its own and its
#           lines are printed as they are. Its LIVE refusal (a process has its cwd in the tree: a batch or static run
#           started after the listing) is a run in flight: retiring stops there, step 3 still runs for the names already
#           retired (their worktrees are gone, so no later sweep would name their box directories), step 4 is skipped,
#           and the sweep ends 75 with mac-disk's line. Any other refusal (DIRTY or AHEAD by now, a target/gate-batch/
#           left to be moved by hand) leaves that tree, the next names still retire, and the sweep ends 1 after the other
#           steps. tools/wt.sh's run sentinels (the mac-static and gate-batch marks of `just wt-retire`) are not read here.
# 3. box    for exactly the names retired: `tools/box-tracks.sh` once (the list) and then `tools/box-tracks.sh --remove`
#           with each retired name listed `stale` and every `stale` row whose name starts with `<name>-` (its aux trees,
#           stale once the track's worktree is gone). A live track or an aux row of another track is never passed, and
#           box-tracks re-reads its own list before it removes. Skipped with one line when no box is configured here:
#           BLOOMERY_BOX_LOCAL=1, or BLOOMERY_BOX unset and `ssh -G ws` resolving ws to itself (no ssh entry; tools/box.sh
#           falls back to `ws` without a check, and a contributor's machine has none).
# 4. clones `tools/mac-remote.sh --clones --clean` (stale clones of the second Mac, every Mac of mac-remote.d/), skipped
#           with one line when ~/.config/bloomery/mac-remote is absent and mac-remote.d/ holds no file.
# Then one line: `sweep: retired N (names), box dirs removed M, clones: <removed R, kept K>, skipped K`.
#
# --dry-run: step 2 runs mac-disk's `retire --dry-run`, step 3 the box-tracks list and prints the --remove line it would
# run (the retired name's live row and its aux rows count, as they become stale with the retire), step 4 prints its line.
#
# Exit: 0 done (nothing to do too). 75 a run in flight (step 2). 64 a bad argument or not in a git repo. Otherwise the
# first failing step's rc, named on the last line (`sweep: step <name> ended rc N`).
# Test seams, used by --self-test only: SWEEP_MAC_DISK and SWEEP_MAC_DISK_RUN (the mac-disk script and the interpreter that
# runs it), SWEEP_BOX_TRACKS and SWEEP_MAC_REMOTE (the scripts, run by bash); ssh comes from PATH, the configs from HOME.
# Runs under the Mac's bash 3.2: no associative arrays, no mapfile, no empty-array expansion. No `set -e`: every step's rc
# is tested by name.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd -P)
SELF=$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")
MAC_DISK=${SWEEP_MAC_DISK:-$HERE/tools/mac-disk.py}
MAC_DISK_RUN=${SWEEP_MAC_DISK_RUN:-python3}
BOX_TRACKS=${SWEEP_BOX_TRACKS:-$HERE/tools/box-tracks.sh}
MAC_REMOTE=${SWEEP_MAC_REMOTE:-$HERE/tools/mac-remote.sh}

say() { printf 'sweep: %s\n' "$*" >&2; }
out() { printf 'sweep: %s\n' "$*"; }

FAIL_RC=0 FAIL_STEP=
step_failed() { # STEP RC: the first failing step names the exit code
  say "step $1 ended rc $2"
  [ "$FAIL_RC" != 0 ] || { FAIL_RC=$2; FAIL_STEP=$1; }
}

# box_configured: 0 when this machine has a box to reach; BOX_WHY says why not.
box_configured() {
  local hn
  BOX_WHY=
  if [ "${BLOOMERY_BOX_LOCAL:-0}" = 1 ]; then BOX_WHY="BLOOMERY_BOX_LOCAL=1: this machine is its own box"; return 1; fi
  [ -z "${BLOOMERY_BOX:-}" ] || return 0
  hn=$(ssh -G ws 2> /dev/null | awk '$1 == "hostname" { print $2; exit }')
  if [ -n "$hn" ] && [ "$hn" != ws ]; then return 0; fi
  BOX_WHY="BLOOMERY_BOX is unset and ssh has no entry for ws"
  return 1
}

# remote_configured: 0 when mac-remote.sh has a Mac to reach.
remote_configured() {
  local f
  [ ! -f "$HOME/.config/bloomery/mac-remote" ] || return 0
  for f in "$HOME/.config/bloomery/mac-remote.d"/*; do [ -f "$f" ] && return 0; done
  return 1
}

# box_pick ROWS NAMES DRY: the box directories of NAMES (newline-separated) among ROWS (`state name owner`), one a line.
box_pick() {
  local rows=$1 names=$2 dry=$3 b state name owner
  while IFS= read -r b; do
    [ -n "$b" ] || continue
    while read -r state name owner; do
      [ -n "$name" ] || continue
      case $state in
        stale) [ "$name" = "$b" ] || case $name in "$b"-?*) ;; *) continue ;; esac ;;
        live) [ "$dry" = 1 ] && [ "$name" = "$b" ] || continue ;;
        aux) [ "$dry" = 1 ] && [ "$owner" = "$b" ] || continue ;;
        *) continue ;;
      esac
      printf '%s\n' "$name"
    done <<< "$rows"
  done <<< "$names"
}

sweep_main() {
  local dry=0 a wts main parent mb report rc rows state name paths n path picked='' skipped=0
  local retired='' nret=0 refused=0 inflight='' mout box_n=0 clones="skipped" list brows bpick bout rout
  for a in "$@"; do
    case $a in
      --dry-run) dry=1 ;;
      *) say "usage: tools/sweep.sh [--dry-run] | --self-test ('$a' is not one)"; return 64 ;;
    esac
  done
  wts=$(git worktree list --porcelain 2> /dev/null | sed -n 's/^worktree //p') || wts=
  [ -n "$wts" ] || { say "not in a git repo (git worktree list printed nothing)"; return 64; }
  main=${wts%%$'\n'*}
  parent=$(dirname "$main") mb=$(basename "$main")

  # 1. list
  report=$($MAC_DISK_RUN "$MAC_DISK" report)
  rc=$?
  if [ "$rc" != 0 ]; then step_failed list "$rc"; return "$rc"; fi
  rows=$(awk 'NR == 1 { next } NF == 0 { exit } { print $1, $NF }' <<< "$report")
  while read -r state name; do
    case $state in LANDED | PRUNABLE) ;; *) continue ;; esac
    paths=$(while IFS= read -r path; do [ "$(basename "$path")" != "$name" ] || printf '%s\n' "$path"; done <<< "$wts")
    n=$(grep -c . <<< "$paths")
    if [ "$n" != 1 ]; then
      out "skip $name ($n worktrees have this basename: $(tr '\n' ' ' <<< "$paths"))"
      skipped=$((skipped + 1))
      continue
    fi
    if [ "$(dirname "$paths")" = "$parent" ] && case $name in "$mb"-?*) true ;; *) false ;; esac; then
      picked="$picked$paths"$'\n'
    else
      out "skip $paths (not beside the main tree)"
      skipped=$((skipped + 1))
    fi
  done <<< "$rows"

  # 2. retire
  while IFS= read -r path; do
    [ -n "$path" ] || continue
    if [ "$dry" = 1 ]; then
      mout=$($MAC_DISK_RUN "$MAC_DISK" retire --dry-run "$path" 2>&1)
    else
      mout=$($MAC_DISK_RUN "$MAC_DISK" retire "$path" 2>&1)
    fi
    rc=$?
    [ -z "$mout" ] || printf '%s\n' "$mout"
    if [ "$rc" = 0 ]; then
      retired="$retired$(basename "$path")"$'\n'
      nret=$((nret + 1))
    elif [ "$rc" = 1 ]; then
      inflight=$(grep -E '^refused .*: LIVE: ' <<< "$mout" | head -n 1)
      [ -z "$inflight" ] || break
      refused=$((refused + 1))
    else
      step_failed retire "$rc"
      break
    fi
  done <<< "$picked"
  [ "$refused" = 0 ] || step_failed retire 1

  # 3. box
  if [ "$nret" = 0 ]; then
    out "box: nothing retired, nothing to remove"
  elif ! box_configured; then
    out "box: skip ($BOX_WHY)"
  else
    list=$(bash "$BOX_TRACKS")
    rc=$?
    if [ "$rc" != 0 ]; then
      step_failed box "$rc"
    else
      brows=$(awk '$1 == "live" || $1 == "aux" || $1 == "stale" { o = ""; if ($4 == "(live") { o = $NF; sub(/\)$/, "", o) } print $1, $3, o }' <<< "$list")
      bpick=$(box_pick "$brows" "$retired" "$dry")
      if [ -z "$bpick" ]; then
        out "box: no directory of the retired names on the box"
      elif [ "$dry" = 1 ]; then
        box_n=$(grep -c . <<< "$bpick")
        out "box: would run tools/box-tracks.sh --remove $(tr '\n' ' ' <<< "$bpick")"
      else
        # shellcheck disable=SC2046 # one word a directory name ([A-Za-z0-9._-], box-tracks refuses any other)
        bout=$(bash "$BOX_TRACKS" --remove $(tr '\n' ' ' <<< "$bpick"))
        rc=$?
        [ -z "$bout" ] || printf '%s\n' "$bout"
        box_n=$(grep -c '^removed ' <<< "$bout")
        [ "$rc" = 0 ] || step_failed box "$rc"
      fi
    fi
  fi

  # 4. clones
  if [ -n "$inflight" ]; then
    clones="not run (a run in flight)"
  elif ! remote_configured; then
    out "clones: skip (no ~/.config/bloomery/mac-remote)"
  elif [ "$dry" = 1 ]; then
    out "clones: would run tools/mac-remote.sh --clones --clean"
    clones="not run (dry run)"
  else
    rout=$(bash "$MAC_REMOTE" --clones --clean 2>&1)
    rc=$?
    [ -z "$rout" ] || printf '%s\n' "$rout"
    clones="removed $(grep -c ': removed$' <<< "$rout"), kept $(grep -c ': kept (' <<< "$rout")"
    [ "$rc" = 0 ] || step_failed clones "$rc"
  fi

  out "$([ "$dry" = 0 ] || echo 'dry run: ')retired $nret ($(printf '%s' "$retired" | tr '\n' ' ' | sed 's/ $//')), box dirs $([ "$dry" = 0 ] && echo removed || echo 'to remove') $box_n, clones: $clones, skipped $skipped$([ "$refused" = 0 ] || echo ", refused $refused")"
  if [ -n "$inflight" ]; then
    say "a run is in flight: $inflight — nothing more retired; the next sweep retries"
    return 75
  fi
  [ "$FAIL_RC" = 0 ] || say "ended rc $FAIL_RC: step $FAIL_STEP"
  return "$FAIL_RC"
}

self_test() {
  local fails=0 t main RC OUT
  fail() { say "self-test FAIL: $*"; fails=$((fails + 1)); }
  t=$(mktemp -d "${TMPDIR:-/tmp}/sweep-selftest.XXXXXX") || { say "self-test: no scratch directory"; return 1; }
  t=$(cd "$t" && pwd -P)
  # shellcheck disable=SC2064 # $t is expanded now on purpose
  trap "rm -rf '$t'" EXIT
  unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_COMMON_DIR GIT_PREFIX GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES
  unset BLOOMERY_BOX BLOOMERY_BOX_LOCAL
  export GIT_CEILING_DIRECTORIES
  GIT_CEILING_DIRECTORIES=$(dirname "$t") # a repo above $TMPDIR is never found from a scratch cwd
  export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null
  export SWEEP_MAC_DISK=$t/stub-disk.sh SWEEP_MAC_DISK_RUN=bash SWEEP_BOX_TRACKS=$t/stub-box.sh SWEEP_MAC_REMOTE=$t/stub-remote.sh
  export STUB_STATES=$t/states STUB_BOX_LIST=$t/box-list STUB_DISK_LOG=$t/disk.log STUB_BOX_LOG=$t/box.log STUB_REMOTE_LOG=$t/remote.log
  export HOME=$t/home PATH=$t/bin:$PATH
  main=$t/bloomery
  mkdir -p "$main" "$t/orca" "$t/bin" "$HOME/.config/bloomery"
  git -C "$main" init -q -b main
  git -C "$main" commit -q --allow-empty -m c1
  for n in landed pr dirty live; do git -C "$main" worktree add -q -b "$n" "$t/bloomery-$n" 2> /dev/null; done
  git -C "$main" worktree add -q -b orca "$t/orca/bloomery-orca" 2> /dev/null # an Orca workspace: a bloomery- name, another parent
  rm -rf "$t/bloomery-pr" # its directory gone: prunable
  printf 'user@host\n' > "$HOME/.config/bloomery/mac-remote"

  # mac-disk: report prints the states file as its rows; retire logs its arguments, refuses LIVE for the basenames in
  # STUB_INFLIGHT (mac-disk's own line) and otherwise prints what it did. box-tracks: the list file, and --remove logs.
  cat > "$t/stub-disk.sh" << 'STUB'
#!/bin/bash
if [ "${1:-report}" = report ]; then
  printf '%-8s %10s %10s %5s  %-24s worktree\n' state target tree age branch
  while read -r s b; do printf '%-8s %10s %10s %5s  %-24s %s\n' "$s" 1.0GiB 100MiB 0d br "$b"; done < "$STUB_STATES"
  printf '\nstates: …\n'
  exit 0
fi
printf '%s\n' "$*" >> "$STUB_DISK_LOG"
shift
dry=0; [ "$1" = --dry-run ] && { dry=1; shift; }
case " ${STUB_INFLIGHT:-} " in *" ${1##*/} "*) echo "refused $1: LIVE: a process has its cwd inside it"; exit 1 ;; esac
case " ${STUB_GATEBATCH:-} " in *" ${1##*/} "*) echo "refused $1: target/gate-batch/ exists and is not empty (1.0KiB at $1/target/gate-batch) — move or delete it by hand first"; exit 1 ;; esac
if [ "$dry" = 1 ]; then echo "$1: would remove the worktree (branch x kept)"; else echo "$1: worktree removed (branch x kept)"; fi
STUB
  cat > "$t/stub-box.sh" << 'STUB'
#!/bin/bash
printf '%s\n' "${*:-list}" >> "$STUB_BOX_LOG"
if [ "${1:-}" = --remove ]; then shift; for n in "$@"; do echo "removed $n"; done; exit 0; fi
cat "$STUB_BOX_LIST"
STUB
  cat > "$t/stub-remote.sh" << 'STUB'
#!/bin/bash
printf '%s\n' "$*" >> "$STUB_REMOTE_LOG"
printf 'the second Mac:\n  clone old: stale, last run 9 d ago, 3 GB: removed\n  clone cur: kept (a live lock), last run 0 d ago\n  40 GB free before, 43 GB after\n'
STUB
  # ssh -G: the hostname STUB_SSH_HOST names (ws: no entry). Nothing else is ever run through it.
  cat > "$t/bin/ssh" << 'STUB'
#!/bin/bash
[ "$1" = -G ] || { echo "stub ssh: only -G" >&2; exit 99; }
echo "hostname ${STUB_SSH_HOST:-ws}"
STUB
  chmod +x "$t/bin/ssh"
  printf 'LANDED bloomery-landed\nPRUNABLE bloomery-pr\nDIRTY bloomery-dirty\nLIVE bloomery-live\nLANDED bloomery-orca\nMAIN bloomery\n' > "$STUB_STATES"
  cat > "$STUB_BOX_LIST" << 'LIST'
live     1.2G  bloomery
stale    300M  bloomery-landed
stale     10M  bloomery-landed-base
live     400M  bloomery-landed-x
aux       20M  bloomery-landed-x-base (live track bloomery-landed-x)
stale    500M  bloomery-other
use    / 50 % of 1833 GiB, 900 GiB free
3 stale — delete by name: tools/box-tracks.sh --remove NAME…
LIST
  run() { rm -f "$STUB_DISK_LOG" "$STUB_BOX_LOG" "$STUB_REMOTE_LOG"; RC=0; OUT=$(cd "$main" && bash "$SELF" "$@" 2>&1) || RC=$?; }
  has() { case $OUT in *"$2"*) ;; *) fail "$1: no [$2] in: $OUT" ;; esac; }
  logis() { [ "$(cat "$2" 2> /dev/null)" = "$3" ] || fail "$1: $(basename "$2") is [$(cat "$2" 2> /dev/null)], want [$3]"; }

  # (a) only LANDED and PRUNABLE beside the main tree retire; the Orca workspace is skipped by name
  STUB_SSH_HOST=10.0.0.9 run
  [ "$RC" = 0 ] || fail "(a) rc $RC: $OUT"
  logis "(a) retire calls" "$STUB_DISK_LOG" "retire $t/bloomery-landed"$'\n'"retire $t/bloomery-pr"
  has "(a) the Orca path" "sweep: skip $t/orca/bloomery-orca (not beside the main tree)"
  has "(a) summary" "sweep: retired 2 (bloomery-landed bloomery-pr), box dirs removed 2, clones: removed 1, kept 1, skipped 1"
  logis "(a) mac-remote" "$STUB_REMOTE_LOG" "--clones --clean"
  # (b) box removal: exactly the retired names and their -suffix dirs listed stale; never the live track bloomery-landed-x
  logis "(b) box-tracks calls" "$STUB_BOX_LOG" "list"$'\n'"--remove bloomery-landed bloomery-landed-base"
  # (c) a run in flight: rc 75 at the first refusal; the first name refused touches neither box-tracks nor mac-remote
  STUB_SSH_HOST=10.0.0.9 STUB_INFLIGHT=bloomery-landed run
  [ "$RC" = 75 ] || fail "(c) rc $RC, want 75: $OUT"
  logis "(c) retire calls" "$STUB_DISK_LOG" "retire $t/bloomery-landed"
  [ ! -e "$STUB_BOX_LOG" ] || fail "(c) box-tracks was called: $(cat "$STUB_BOX_LOG")"
  [ ! -e "$STUB_REMOTE_LOG" ] || fail "(c) mac-remote was called"
  has "(c) the reason" "a run is in flight: refused $t/bloomery-landed: LIVE"
  # … and refused at the second, the first one's box dirs still go (its worktree is gone), mac-remote is not run
  STUB_SSH_HOST=10.0.0.9 STUB_INFLIGHT=bloomery-pr run
  [ "$RC" = 75 ] || fail "(c2) rc $RC, want 75: $OUT"
  logis "(c2) box-tracks calls" "$STUB_BOX_LOG" "list"$'\n'"--remove bloomery-landed bloomery-landed-base"
  [ ! -e "$STUB_REMOTE_LOG" ] || fail "(c2) mac-remote was called"
  # … while a target/gate-batch/ left behind is an ordinary refusal: the next name still retires, rc 1 at the end
  STUB_SSH_HOST=10.0.0.9 STUB_GATEBATCH=bloomery-landed run
  [ "$RC" = 1 ] || fail "(c3) rc $RC, want 1: $OUT"
  logis "(c3) retire calls" "$STUB_DISK_LOG" "retire $t/bloomery-landed"$'\n'"retire $t/bloomery-pr"
  logis "(c3) box-tracks calls" "$STUB_BOX_LOG" "list"
  logis "(c3) mac-remote" "$STUB_REMOTE_LOG" "--clones --clean"
  has "(c3) the summary" "sweep: retired 1 (bloomery-pr), box dirs removed 0, clones: removed 1, kept 1, skipped 1, refused 1"
  # (d) --dry-run calls no remover: mac-disk's dry run, the box list, no --remove, no mac-remote
  STUB_SSH_HOST=10.0.0.9 run --dry-run
  [ "$RC" = 0 ] || fail "(d) rc $RC: $OUT"
  logis "(d) retire calls" "$STUB_DISK_LOG" "retire --dry-run $t/bloomery-landed"$'\n'"retire --dry-run $t/bloomery-pr"
  logis "(d) box-tracks calls" "$STUB_BOX_LOG" "list"
  [ ! -e "$STUB_REMOTE_LOG" ] || fail "(d) mac-remote was called"
  has "(d) the box line" "box: would run tools/box-tracks.sh --remove bloomery-landed bloomery-landed-base"
  # (e) no box configured: step 3 skipped, rc 0; no mac-remote config: step 4 skipped
  run
  [ "$RC" = 0 ] || fail "(e) rc $RC: $OUT"
  [ ! -e "$STUB_BOX_LOG" ] || fail "(e) box-tracks was called with no box: $(cat "$STUB_BOX_LOG")"
  has "(e) the skip line" "sweep: box: skip (BLOOMERY_BOX is unset and ssh has no entry for ws)"
  STUB_SSH_HOST=10.0.0.9 BLOOMERY_BOX_LOCAL=1 run
  [ ! -e "$STUB_BOX_LOG" ] || fail "(e) box-tracks was called under BLOOMERY_BOX_LOCAL=1"
  rm "$HOME/.config/bloomery/mac-remote"
  run
  [ "$RC" = 0 ] || fail "(e2) rc $RC: $OUT"
  [ ! -e "$STUB_REMOTE_LOG" ] || fail "(e2) mac-remote was called with no config"
  has "(e2) the skip line" "sweep: clones: skip (no ~/.config/bloomery/mac-remote)"
  # usage
  run --bogus
  [ "$RC" = 64 ] || fail "usage: rc $RC, want 64"

  if [ "$fails" != 0 ]; then
    say "self-test $fails failed"
    return 1
  fi
  echo "sweep: self-test ok"
}

case ${1:-} in
  --self-test) [ $# = 1 ] || { say "--self-test takes nothing"; exit 64; }; self_test; exit $? ;;
esac
sweep_main "$@"
exit $?
