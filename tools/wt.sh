#!/usr/bin/env bash
# bloomery — a worktree that starts warm and is retired whole (the Mac, APFS). A tree builds in its own target/
# (tools/mac-check.sh), so a plain `git worktree add` starts every lint cold (~260 CPU s). `new` adds the worktree and
# seeds its target/ from the warm tree whose last green mac lint is nearest the base (tools/seed-target.sh: a clonefile
# copy, no disk until cargo rewrites a file); `retire` removes it through tools/mac-disk.py, which owns the refusals.
#
#   tools/wt.sh new NAME [BASE]          git worktree add -b NAME <repo parent>/<main dir>-NAME BASE (default HEAD), then seed
#   tools/wt.sh retire [--dry-run] NAME  mac-disk.py retire <repo parent>/<main dir>-NAME, and the bytes its target/ held
#   tools/wt.sh --self-test              scratch git repos under $TMPDIR, stub seed-target and mac-disk, no cargo, no box
#
# The repo is the one of the caller's cwd (`git rev-parse --git-common-dir`), never the one this script lives in: only the
# helpers (seed-target.sh, mac-disk.py) come from this script's own tree. NAME is the branch and the suffix of the
# directory (`bloomery-NAME` beside the main tree, as the existing worktrees are named).
#
# new. 1. The path must not exist (git accepts an empty directory, and a failed add leaves its `-b` branch behind
# when the path was not empty), and BASE must be a commit. 2. `git worktree add`; its rc is tested by name and a
# failure ends the script there (65), because a failed add once let the next commands run in the main tree. 3. The
# new path must be listed by `git worktree list --porcelain` on refs/heads/NAME (66 otherwise): an add that exits 0 and
# leaves nothing is also a stop. 4. CLAUDE.local.md is copied from the main tree (it imports the workstation rules, is
# gitignored, so a fresh worktree lacks it and a session there would run without the box, lock and landing rules); a
# missing source is one warning line, a copy that fails or differs stops with 67 before the seed, because a tree without the
# rules is not handed out. 5. The seed (below), then tools/seed-target.sh SEED NEW, whose units-and-bytes line is printed.
# 6. The new tree's `git worktree list` line is printed last, as the proof.
#
# The seed record. tools/mac-check.sh lint appends `<tree path>\t<HEAD sha>\t<unix time>\tlint` to
# ~/.cache/bloomery/seed.tsv (WT_SEED_TSV moves it) when it exits 0. Lint has one lane (the x86_64 cross check), so
# the fourth column is the mode, not a lane. The file is append-only: the last line of a path is that tree's last green lint,
# the earlier ones are history.
#
# The seed choice. Of the last lines per path, a tree is skipped, by name in the one-line summary, when: its directory or
# target/ is gone (a retired tree: its lines stay, `new` skips them by name, so no read-modify-write races an append); its
# sha is not an ancestor of BASE (or BASE itself, or not a commit of this repo); a mac-static run is in flight in it (its
# target/ is being written). Among the rest the cost is the commits between the seed's lint sha and BASE (git rev-list
# --count) plus the commits the seed's own HEAD has moved since that lint (its sources are not the lint's any more: files
# that differ from the new tree's rebuild their units); the least wins, the newer lint on a tie. No seed, or none fitting:
# one line naming why, no seeding, exit 0: the tree is made, cold.
# A seed is only as good as the environment of its lint: tools/mac-check.sh lint pins the commit word (BLOOMERY_GIT_COMMIT=
# static-check, or the caller's own), and a seed linted under another word, or one from a hand-written record of a tree
# linted before the pin, re-checks bloomery-serve and what depends on it in the new tree's first lint.
#
# Read-only seed. This script reads the seed through stat and the pid files and asks git only about the repo's object
# database and `git worktree list` (never `git status` or any command that refreshes an index in the seed). seed-target.sh
# copies by clonefile and writes only the new tree. The self-test proves the seed's `git status --porcelain` and the stat of
# every file under its target/ are identical before and after `new`.
#
# retire. The tree is refused (1) while a run is in flight in it. The sentinels, by tool:
#   mac-static   <tree>/target/mac-static/pids is the start mark, <step>.rc the end of a step (fmt-check, check-*, lint,
#                combos; the pids line `cargo-lane` stands for lint and combos). A step with a start mark and no .rc is
#                in flight when a pid recorded in pids is alive, or when no pid is on record; a run whose recorded pids
#                are all dead was killed: it is named and does not block (mac-static clears the marks at its next start).
#   gate-batch   no .rc: the batch dir is ~/.cache/bloomery/batches/<tree dir name>/<stamp>/ (--out under the tree's
#                target/ is read too), run.log is the start mark and its `^DONE ` line the end. Without it the batch is in
#                flight when a pid in lane-*.pid, lane-*.child or gpuhold.pid is alive, or when none is on record.
#   mac-check    no sentinel (its summary goes to stdout): a bare `just mac-lint` mid-run is mac-disk's LIVE refusal (its
#                cargo has its cwd in the tree).
# A refusal names the file; a mark left by a dead run is cleared by removing that file or directory. Everything else is
# mac-disk.py's, called once with the absolute path: LIVE, DIRTY, AHEAD, the main tree and a target symlink are its refusals
# and its rc 1; this script has no second copy of them. mac-disk removes target/ before the worktree; if a target/ is still
# at the path afterwards (a moved or kept one), it is removed here. The bytes are `du` of target/ before the call: a seeded
# target shares blocks with its seed, so the disk freed is what cargo rewrote since, which `df` shows. Branches are never
# deleted (mac-disk's rule). The seed lines of a retired tree stay; `new` skips them.
#
# Not here: the mac-remote clone (~/repo/bloomery-mr/<track>) belongs to tools/mac-remote.sh and may be in use by a slot.
#
# Exit: 0 done (`new` with no usable seed too). 1 retire refused (here or by mac-disk). 64 usage, a name or base that is not
# one, not in a git repo. 65 new: the path exists, or git worktree add failed. 66 new: after the add the path is not a worktree
# of this repo on the branch. 67 new: the tree is made and CLAUDE.local.md could not be copied. 69 mac-disk.py not found, or its own 69. 70 new: the tree is made and seed-target failed (its rc
# is in the line; the tree is cold).
# Test seams, used by --self-test only: WT_SEED_TSV, WT_SEED_TARGET (the seed-target path), WT_MAC_DISK and
# WT_MAC_DISK_RUN (the mac-disk script and the interpreter that runs it), WT_BATCH_ROOT (the gate-batch dir root).
# Runs under the Mac's bash 3.2: no associative arrays, no mapfile. No `set -e`: every step's rc is tested by name.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd -P)
SELF=$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")
SEED_TSV=${WT_SEED_TSV:-$HOME/.cache/bloomery/seed.tsv}
SEED_TARGET=${WT_SEED_TARGET:-$HERE/tools/seed-target.sh}
MAC_DISK=${WT_MAC_DISK:-$HERE/tools/mac-disk.py}
MAC_DISK_RUN=${WT_MAC_DISK_RUN:-python3}
BATCH_ROOT=${WT_BATCH_ROOT:-$HOME/.cache/bloomery/batches}

say() { printf 'wt: %s\n' "$*" >&2; }
out() { printf 'wt: %s\n' "$*"; }

# find_main: MAIN, the real path of the main worktree of the repo of the cwd; 64 and a line when the cwd is in none.
find_main() {
  local common
  common=$(git rev-parse --git-common-dir 2> /dev/null) || { say "not inside a git repository (the repo is the one of the cwd, $PWD)"; return 64; }
  common=$(cd "$common" 2> /dev/null && pwd -P) || { say "the git common dir '$common' is not reachable from $PWD"; return 64; }
  case $common in
    */.git) MAIN=${common%/.git} ;;
    *) say "the common git dir $common is not <main tree>/.git (a bare repo has no main tree)"; return 64 ;;
  esac
}

# tree_of NAME: TREE, the directory a NAME names beside the main tree.
tree_of() {
  TREE=$(dirname "$MAIN")/$(basename "$MAIN")-$1
}

valid_name() {
  case $1 in
    '' | -* | .* | *[!A-Za-z0-9._-]*) return 1 ;;
  esac
  git check-ref-format --branch "$1" > /dev/null 2>&1
}

# worktree_branch PATH: the branch `git worktree list --porcelain` shows for PATH; empty when PATH is not listed.
worktree_branch() {
  git -C "$MAIN" worktree list --porcelain | awk -v p="$1" '$1 == "worktree" { cur = substr($0, 10) } $1 == "branch" && cur == p { print $2; exit }'
}

# pid_scan FILE…: PIDS_SEEN pids recorded (the last number of each line), PIDS_LIVE of them alive, PIDS_LIST the live ones.
pid_scan() {
  local f p
  PIDS_SEEN=0 PIDS_LIVE=0 PIDS_LIST=
  for f in "$@"; do
    [ -f "$f" ] || continue
    for p in $(awk 'NF { print $NF }' "$f"); do
      case $p in '' | *[!0-9]*) continue ;; esac
      [ "$p" -gt 1 ] || continue
      PIDS_SEEN=$((PIDS_SEEN + 1))
      if kill -0 "$p" 2> /dev/null; then
        PIDS_LIVE=$((PIDS_LIVE + 1))
        PIDS_LIST="$PIDS_LIST $p"
      fi
    done
  done
}

# classify WHAT DIR HOW: one run with no sentinel; the pids just scanned decide. Appends to RUNS_BLOCK or RUNS_NOTE.
classify() {
  if [ "$PIDS_LIVE" -gt 0 ]; then
    RUNS_BLOCK="$RUNS_BLOCK$1 in flight: $2 ($3; pid$PIDS_LIST alive)"$'\n'
  elif [ "$PIDS_SEEN" = 0 ]; then
    RUNS_BLOCK="$RUNS_BLOCK$1 may be in flight: $2 ($3; no pid on record, so it cannot be shown dead: remove it if the run is gone)"$'\n'
  else
    RUNS_NOTE="$RUNS_NOTE$1 left by a dead run: $2 ($3; $PIDS_SEEN recorded pid(s), none alive)"$'\n'
  fi
}

# add_word LIST WORD: LIST (space-separated, a leading space) with WORD once.
add_word() {
  case "$1 " in *" $2 "*) printf '%s' "$1" ;; *) printf '%s %s' "$1" "$2" ;; esac
}

# static_run TREE: the mac-static run of TREE, per the header; RUNS_BLOCK and RUNS_NOTE gain a line when it has no end.
static_run() {
  local d=$1/target/mac-static name f missing= names
  [ -d "$d" ] || return 0
  names=$([ ! -f "$d/pids" ] || awk 'NF { print $1 }' "$d/pids")
  for f in "$d"/*.log; do
    [ -e "$f" ] && names="$names"$'\n'"$(basename "$f" .log)"
  done
  for name in $names; do
    case $name in
      cargo-lane) for f in lint combos; do [ -f "$d/$f.rc" ] || missing=$(add_word "$missing" "$f"); done ;;
      *) [ -f "$d/$name.rc" ] || missing=$(add_word "$missing" "$name") ;;
    esac
  done
  [ -n "$missing" ] || return 0
  pid_scan "$d/pids"
  classify "mac-static run" "$d" "no .rc for${missing% }"
}

# batch_dir DIR: one gate-batch dir; a run.log with no DONE line is a run with no end.
batch_dir() {
  grep -q '^DONE ' "$1/run.log" 2> /dev/null && return 0
  pid_scan "$1"/lane-*.pid "$1"/lane-*.child "$1/gpuhold.pid"
  classify "gate-batch run" "$1" "run.log has no DONE line"
}

# batch_runs TREE: every gate-batch dir of TREE (the default root by the tree's dir name, and --out dirs under target/).
batch_runs() {
  local f
  for f in "$BATCH_ROOT/$(basename "$1")"/*/run.log; do
    [ -f "$f" ] && batch_dir "$(dirname "$f")"
  done
  [ -d "$1/target" ] || return 0
  while IFS= read -r f; do
    batch_dir "$(dirname "$f")"
  done < <(find "$1/target" -maxdepth 3 -name run.log -type f 2> /dev/null)
}

# copy_local_md MAIN TREE: the header's step 4. 0 copied, or the source is missing (a warning line); 67 the copy failed or differs.
copy_local_md() {
  local src=$1/CLAUDE.local.md
  if [ ! -e "$src" ]; then
    say "warning: $src does not exist; $2 has no CLAUDE.local.md, so a session there lacks the workstation rules it imports"
    return 0
  fi
  if ! cp "$src" "$2/CLAUDE.local.md" 2> /dev/null || ! cmp -s "$src" "$2/CLAUDE.local.md"; then
    say "refused: could not copy $src into $2 (a tree without the workstation rules is not handed out); the tree is made, nothing after the copy ran"
    return 67
  fi
  out "copied CLAUDE.local.md from $1 (the workstation rules import; git ignores it)"
}

# pick_seed BASE_SHA DST_REAL: SEED_TREE SEED_SHA SEED_DIST SEED_DRIFT SEED_TS and return 0, or SEED_WHY and return 1.
pick_seed() {
  local base=$1 dst=$2 lines heads p sha ts kind full cur pr d drift eff best_eff= best_ts=0
  local total=0 gone=0 unrelated=0 busy=0
  SEED_TREE= SEED_SHA= SEED_DIST= SEED_DRIFT= SEED_TS= SEED_WHY=
  if [ ! -f "$SEED_TSV" ]; then
    SEED_WHY="no seed record ($SEED_TSV does not exist: no mac lint has been recorded)"
    return 1
  fi
  lines=$(awk -F'\t' 'NF >= 4 && $1 != "" && $2 != "" && $3 != "" && $4 == "lint" { if (!($1 in last)) order[++n] = $1; last[$1] = $0 } END { for (i = 1; i <= n; i++) print last[order[i]] }' "$SEED_TSV")
  if [ -z "$lines" ]; then
    SEED_WHY="no seed record: $SEED_TSV has no <path><TAB><sha><TAB><time><TAB>lint line"
    return 1
  fi
  heads=$(git -C "$MAIN" worktree list --porcelain | awk '$1 == "worktree" { cur = substr($0, 10) } $1 == "HEAD" { print cur "\t" $2 }')
  while IFS=$'\t' read -r p sha ts kind; do
    total=$((total + 1))
    if [ ! -d "$p/target" ]; then gone=$((gone + 1)); continue; fi
    pr=$(cd "$p" 2> /dev/null && pwd -P) || { gone=$((gone + 1)); continue; }
    if [ "$pr" = "$dst" ]; then gone=$((gone + 1)); continue; fi
    full=$(git -C "$MAIN" rev-parse --verify --quiet "${sha}^{commit}") || { unrelated=$((unrelated + 1)); continue; }
    git -C "$MAIN" merge-base --is-ancestor "$full" "$base" 2> /dev/null || { unrelated=$((unrelated + 1)); continue; }
    RUNS_BLOCK= RUNS_NOTE=
    static_run "$pr"
    if [ -n "$RUNS_BLOCK" ]; then busy=$((busy + 1)); continue; fi
    d=$(git -C "$MAIN" rev-list --count "$full..$base")
    cur=$(awk -F'\t' -v p="$pr" '$1 == p { print $2; exit }' <<< "$heads")
    drift=0
    if [ -n "$cur" ] && [ "$cur" != "$full" ]; then drift=$(git -C "$MAIN" rev-list --count "$full...$cur"); fi
    eff=$((d + drift))
    case $ts in '' | *[!0-9]*) ts=0 ;; esac
    if [ -z "$best_eff" ] || [ "$eff" -lt "$best_eff" ] || { [ "$eff" = "$best_eff" ] && [ "$ts" -gt "$best_ts" ]; }; then
      best_eff=$eff best_ts=$ts
      SEED_TREE=$pr SEED_SHA=$full SEED_DIST=$d SEED_DRIFT=$drift SEED_TS=$ts
    fi
  done <<< "$lines"
  if [ -z "$SEED_TREE" ]; then
    SEED_WHY="no seed fits ($total record(s): $gone gone or without target/, $unrelated not an ancestor of the base, $busy with a mac-static run in flight)"
    return 1
  fi
}

cmd_new() {
  local name base base_sha tree tree_real add_out add_rc src=0 wl age moved
  if [ $# -lt 1 ] || [ $# -gt 2 ]; then say "usage: tools/wt.sh new NAME [BASE]"; return 64; fi
  name=$1
  base=${2:-HEAD}
  find_main || return $?
  valid_name "$name" || { say "refused: '$name' is not a branch and directory name ([A-Za-z0-9._-], no leading . or -)"; return 64; }
  base_sha=$(git rev-parse --verify --quiet "${base}^{commit}") || { say "refused: base '$base' is not a commit of this repo"; return 64; }
  tree_of "$name"
  tree=$TREE
  if [ -e "$tree" ] || [ -L "$tree" ]; then
    say "refused: $tree already exists (git would adopt an empty directory); nothing ran"
    return 65
  fi
  add_out=$(git worktree add -b "$name" "$tree" "$base" 2>&1)
  add_rc=$?
  [ "$add_rc" = 0 ] || { say "refused: git worktree add -b $name $tree $base failed (rc $add_rc): ${add_out##*$'\n'}; nothing after it ran"; return 65; } # add-rc-check
  tree_real=$(cd "$tree" 2> /dev/null && pwd -P)
  [ -n "$tree_real" ] && [ "$(worktree_branch "$tree_real")" = "refs/heads/$name" ] || { say "refused: after the add, $tree is not a worktree of $MAIN on refs/heads/$name; nothing after it ran"; return 66; } # worktree-check
  out "worktree $tree on branch $name at ${base_sha:0:8}"
  copy_local_md "$MAIN" "$tree" || return $? # local-md-copy
  if pick_seed "$base_sha" "$tree_real"; then
    age=$(( ($(date +%s) - SEED_TS) / 60 ))
    moved=
    [ "$SEED_DRIFT" = 0 ] || moved=", its HEAD moved $SEED_DRIFT commit(s) since"
    out "seed $SEED_TREE: last green lint at ${SEED_SHA:0:8}, $SEED_DIST commit(s) behind ${base_sha:0:8}$moved, recorded $age min ago"
    bash "$SEED_TARGET" "$SEED_TREE" "$tree"; src=$? # seed-call
  else
    out "cold: $SEED_WHY; the tree is made, its first lint starts cold"
  fi
  wl=$(git -C "$MAIN" worktree list | awk -v p="$tree_real" '$1 == p { print; exit }')
  out "worktree list: $wl"
  if [ "$src" != 0 ]; then
    say "seed-target ended rc $src; $tree is made and cold"
    return 70
  fi
}

cmd_retire() {
  local dry=0 name= a tree bytes mrc=0 rest
  for a in "$@"; do
    case $a in
      --dry-run) dry=1 ;;
      -*) say "usage: tools/wt.sh retire [--dry-run] NAME ('$a' is not an option)"; return 64 ;;
      *) [ -z "$name" ] || { say "usage: tools/wt.sh retire [--dry-run] NAME (one NAME, got '$name' and '$a')"; return 64; }; name=$a ;;
    esac
  done
  [ -n "$name" ] || { say "usage: tools/wt.sh retire [--dry-run] NAME"; return 64; }
  find_main || return $?
  valid_name "$name" || { say "refused: '$name' is not a worktree name ([A-Za-z0-9._-], no leading . or -)"; return 64; }
  [ -f "$MAC_DISK" ] || { say "$MAC_DISK not found"; return 69; }
  tree_of "$name"
  tree=$TREE
  bytes=0
  if [ -d "$tree" ]; then
    RUNS_BLOCK= RUNS_NOTE=
    static_run "$tree"
    batch_runs "$tree"
    if [ -n "$RUNS_NOTE" ]; then
      printf '%s' "$RUNS_NOTE" | while IFS= read -r a; do out "note: $a"; done
    fi
    if [ -n "$RUNS_BLOCK" ]; then
      printf '%s' "$RUNS_BLOCK" | while IFS= read -r a; do say "refused retire $name: $a"; done
      return 1
    fi
    [ ! -d "$tree/target" ] || bytes=$(du -sk "$tree/target" 2> /dev/null | awk '{ print $1 * 1024 }')
  fi
  cd "$MAIN" || return 70
  $MAC_DISK_RUN "$MAC_DISK" retire $([ "$dry" = 0 ] || echo --dry-run) "$tree"
  mrc=$?
  [ "$mrc" = 0 ] || return "$mrc"
  if [ "$dry" = 1 ]; then
    out "dry run: nothing removed; target/ of $tree is $bytes bytes by du"
    return 0
  fi
  if [ -n "$(worktree_branch "$tree")" ]; then
    say "mac-disk ended 0 but $tree is still listed as a worktree; nothing more is removed here"
    return 70
  fi
  if [ -L "$tree/target" ]; then
    say "$tree/target is a symlink left behind; not followed"
  elif [ -e "$tree/target" ]; then
    rm -rf "$tree/target" # leftover-rm
    out "removed the target/ mac-disk left at $tree/target"
  fi
  [ ! -e "$tree" ] || out "note: $tree still exists (no longer a worktree)"
  rest=$(df -Pk "$MAIN" 2> /dev/null | awk 'NR == 2 { printf "%.1f", $4 / 1048576 }')
  out "retired $name: $tree is gone; its target/ held $bytes bytes by du before the call (blocks shared with its seed free only what cargo rewrote since); volume free ${rest:-?} GiB"
}

self_test() {
  local fails=0 t main m o i pid snap0 snap1 d b dead c1 c2 c3 c4 s1
  fail() { say "self-test FAIL: $*"; fails=$((fails + 1)); }
  t=$(mktemp -d "${TMPDIR:-/tmp}/wt-selftest.XXXXXX") || { say "self-test: no scratch directory"; return 1; }
  t=$(cd "$t" && pwd -P)
  # shellcheck disable=SC2064 # $t is expanded now on purpose
  trap "kill \$(jobs -p) 2> /dev/null; chmod -R u+rwx '$t' 2> /dev/null; rm -rf '$t'" EXIT
  unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_COMMON_DIR GIT_PREFIX GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES
  export GIT_CEILING_DIRECTORIES
  GIT_CEILING_DIRECTORIES=$(dirname "$t") # a repo above $TMPDIR is never found from a scratch cwd
  export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null
  export WT_SEED_TSV=$t/seed.tsv WT_SEED_TARGET=$t/stub-seed.sh WT_MAC_DISK=$t/stub-disk.sh WT_MAC_DISK_RUN=bash WT_BATCH_ROOT=$t/batches
  export STUB_SEED_LOG=$t/seed-calls.log STUB_DISK_LOG=$t/disk-calls.log WT_MARKER=$t/marker
  main=$t/bloomery
  mkdir -p "$main" "$t/batches"
  git -C "$main" init -q -b main
  printf 'target/\nCLAUDE.local.md\n' > "$main/.gitignore"
  git -C "$main" add .gitignore
  for i in 1 2 3 4; do
    printf '%s\n' "$i" > "$main/f"
    git -C "$main" add f
    git -C "$main" commit -q -m "c$i"
    eval "c$i=\$(git -C \"\$main\" rev-parse HEAD)"
  done
  git -C "$main" checkout -q -b side "$c1"
  printf 's\n' > "$main/s"
  git -C "$main" add s
  git -C "$main" commit -q -m s1
  s1=$(git -C "$main" rev-parse HEAD)
  git -C "$main" checkout -q main
  printf '@~/.claude/bloomery-workstation.md\nno trailing newline, \001 a control byte' > "$main/CLAUDE.local.md"

  # the stubs: seed-target records its cwd and arguments; mac-disk records its arguments and removes the worktree
  # the way mac-disk does (target/ first, then git worktree remove) unless STUB_DISK says otherwise
  cat > "$t/stub-seed.sh" << 'STUB'
#!/bin/bash
printf '%s\t%s\t%s\n' "$PWD" "$1" "$2" >> "$STUB_SEED_LOG"
if [ -f "$2/CLAUDE.local.md" ]; then echo present > "$STUB_SEED_LOG.md"; else echo absent > "$STUB_SEED_LOG.md"; fi
if [ -n "${STUB_SEED_RC:-}" ]; then echo "seed-target: refused (stub)" >&2; exit "$STUB_SEED_RC"; fi
echo "seed-target: mtime of 0 of 0 source files restored to the seed's (0 differ or are new and keep their own)"
echo "seed-target: seeded 3 units, 105 bytes"
STUB
  cat > "$t/stub-disk.sh" << 'STUB'
#!/bin/bash
printf '%s\t%s\n' "$PWD" "$*" >> "$STUB_DISK_LOG"
[ "$1" = retire ] && shift
dry=0; [ "$1" = --dry-run ] && { dry=1; shift; }
tree=$1
case ${STUB_DISK:-ok} in
  refuse) echo "refused $(basename "$tree"): LIVE: a process has its cwd inside it"; exit 1 ;;
  keep-target) git worktree remove --force "$tree" 2> /dev/null; mkdir -p "$tree/target/debug"; echo kept > "$tree/target/debug/f"; exit 0 ;;
esac
[ "$dry" = 1 ] && { echo "$(basename "$tree"): would remove the worktree"; exit 0; }
rm -rf "$tree/target"
git worktree remove --force "$tree" || exit 1
git worktree prune
echo "$(basename "$tree"): worktree removed"
STUB

  # the repo of every call is a scratch one: the common dir of the cwd must be under $t before a call is made
  scratch_ok() {
    local c
    c=$(cd "$1" 2> /dev/null && git rev-parse --git-common-dir 2> /dev/null) && c=$(cd "$1" && cd "$c" && pwd -P) || return 1
    case $c in "$t"/*) return 0 ;; *) return 1 ;; esac
  }
  scratch_ok "$main" || { fail "the scratch repo $main is not a repo under $t; no call is made"; return 1; }
  # run_wt SCRIPT CWD ARGS…: OUT and RC of `bash SCRIPT ARGS` run in CWD
  run_wt() {
    local script=$1 cwd=$2
    shift 2
    if ! scratch_ok "$cwd"; then
      fail "refusing to run: $cwd is not in a scratch repo under $t"
      OUT= RC=99
      return 0
    fi
    RC=0
    OUT=$(cd "$cwd" && bash "$script" "$@" 2>&1) || RC=$?
  }
  snap() { # the main repo's HEAD, branches, status, worktree list and the siblings beside it
    {
      git -C "$main" rev-parse HEAD
      git -C "$main" branch --list
      git -C "$main" status --porcelain
      git -C "$main" worktree list
      ls -A "$t" | sort
    } 2>&1
  }
  expect() { # WHAT RC WANT_RC OUT WANT_TEXT
    [ "$2" = "$3" ] || fail "$1 ended rc $2, not $3: $4"
    case $4 in *"$5"*) ;; *) fail "$1 does not say '$5': $4" ;; esac
  }
  rm_tree() { git -C "$main" worktree remove --force "$1" 2> /dev/null; git -C "$main" branch -D "$2" > /dev/null 2>&1; }
  : > "$t/seed.tsv"
  write_rec() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "${4:-lint}" >> "$t/seed.tsv"; }
  mkdir -p "$t/seedA/target"
  write_rec "$t/seedA" "$c4" 1000

  # usage and name refusals, outside a repo too
  RC=0; OUT=$(cd "$t" && bash "$SELF" new x 2>&1) || RC=$?
  expect "new outside a repo" "$RC" 64 "$OUT" "not inside a git repository"
  snap0=$(snap)
  for m in "new" "new ../x" "new a/b" "new -x" "new .x" "new 'a b'" "retire" "retire ../x" "retire a b" "retire --bogus x" "nope"; do
    eval "run_wt \"\$SELF\" \"\$main\" $m"
    [ "$RC" = 64 ] || fail "'$m' ended rc $RC, not 64: $OUT"
  done
  run_wt "$SELF" "$main" new okname nosuchbase
  expect "an unknown base" "$RC" 64 "$OUT" "is not a commit"
  [ "$(snap)" = "$snap0" ] || fail "a usage refusal changed the repo"
  [ ! -f "$STUB_SEED_LOG" ] || fail "a usage refusal called seed-target"

  # a path that exists (git adopts an empty directory, and leaves its -b branch behind when the path is not empty): refused
  # before the add; the repo is unchanged and nothing after it ran
  mkdir -p "$t/bloomery-full" "$t/bloomery-emptyd"
  echo x > "$t/bloomery-full/f"
  snap0=$(snap)
  for m in full emptyd; do
    run_wt "$SELF" "$main" new "$m"
    expect "new on the existing path $m" "$RC" 65 "$OUT" "already exists"
  done
  [ "$(snap)" = "$snap0" ] || fail "a refused existing path changed the repo (branch list, status or worktree list)"
  [ ! -f "$STUB_SEED_LOG" ] || fail "an existing path went on to seed-target"

  # an existing branch: git's refusal is the add's rc, named, and nothing after it runs (a seed record is present, so the next
  # step would be seed-target)
  git -C "$main" branch taken "$c2"
  snap0=$(snap)
  run_wt "$SELF" "$main" new taken
  expect "new on the existing branch" "$RC" 65 "$OUT" "git worktree add -b taken"
  case $OUT in *"nothing after it ran"*) ;; *) fail "the branch refusal does not say nothing after it ran: $OUT" ;; esac
  [ "$(snap)" = "$snap0" ] || fail "a refused existing branch changed the repo"
  [ ! -f "$STUB_SEED_LOG" ] || fail "an existing branch went on to seed-target"

  # a git whose add fails (rc 1), and one whose add exits 0 and creates nothing: each stops by name (65, 66)
  mkdir -p "$t/shim"
  cat > "$t/shim/git" << SHIM
#!/bin/sh
if [ "\$1" = worktree ] && [ "\$2" = add ]; then
  case "\${SHIM_ADD:-}" in
    fail) echo "fatal: shim says no" >&2; exit 1 ;;
    silent) exit 0 ;;
  esac
fi
exec $(command -v git) "\$@"
SHIM
  chmod +x "$t/shim/git"
  snap0=$(snap)
  RC=0; OUT=$(cd "$main" && PATH=$t/shim:$PATH SHIM_ADD=fail bash "$SELF" new shimfail 2>&1) || RC=$?
  expect "an add that fails" "$RC" 65 "$OUT" "shim says no"
  RC=0; OUT=$(cd "$main" && PATH=$t/shim:$PATH SHIM_ADD=silent bash "$SELF" new shimsilent 2>&1) || RC=$?
  expect "an add that exits 0 and creates nothing" "$RC" 66 "$OUT" "is not a worktree of $main on refs/heads/shimsilent"
  [ "$(snap)" = "$snap0" ] || fail "a stopped add changed the repo"
  [ ! -f "$STUB_SEED_LOG" ] || fail "a stopped add went on to seed-target"
  if [ "$(id -u)" != 0 ]; then
    chmod a-w "$t"
    RC=0; OUT=$(cd "$main" && bash "$SELF" new rofs 2>&1) || RC=$?
    chmod u+w "$t"
    expect "a parent that cannot be written" "$RC" 65 "$OUT" "nothing after it ran"
    git -C "$main" branch -D rofs > /dev/null 2>&1
  fi

  # FAIL-first: the guards one at a time, in mutants of this script. The add-rc line replaced by a marker command shows what
  # runs right after a failed add, and where: in the main tree. The two guards are layered, so each is also removed alone and
  # together; the post-add check stops what the rc check misses.
  for m in add-rc-check worktree-check seed-call local-md-copy; do
    [ "$(grep -c "# $m\$" "$SELF")" = 1 ] || fail "the anchor '# $m' is not on exactly one line of this script"
  done
  mkdir -p "$t/mut"
  sed 's|^.*# add-rc-check$|( pwd > "$WT_MARKER" ) # add-rc-check|' "$SELF" > "$t/mut/marker.sh"
  sed '/# add-rc-check$/d' "$SELF" > "$t/mut/no-rc.sh"
  # (the CLAUDE.local.md copy is also a stop after a failed add, its target directory being absent: these two mutants drop it
  # too, so the next step they show is seed-target's)
  sed -e '/# worktree-check$/d' -e '/# local-md-copy$/d' "$SELF" > "$t/mut/no-wt.sh"
  sed -e '/# add-rc-check$/d' -e '/# worktree-check$/d' -e '/# local-md-copy$/d' "$SELF" > "$t/mut/no-guards.sh"
  rm -f "$WT_MARKER" "$STUB_SEED_LOG"
  RC=0; OUT=$(cd "$main" && PATH=$t/shim:$PATH SHIM_ADD=fail bash "$t/mut/marker.sh" new mut1 2>&1) || RC=$?
  [ -f "$WT_MARKER" ] && [ "$(cat "$WT_MARKER")" = "$main" ] || fail "mutant (rc check replaced by a marker): the marker did not run in the main tree ($(cat "$WT_MARKER" 2> /dev/null))"
  expect "mutant: the post-add check alone" "$RC" 66 "$OUT" "is not a worktree"
  rm -f "$WT_MARKER" "$STUB_SEED_LOG"
  RC=0; OUT=$(cd "$main" && PATH=$t/shim:$PATH SHIM_ADD=fail bash "$t/mut/no-rc.sh" new mut2 2>&1) || RC=$?
  expect "mutant without the add-rc check" "$RC" 66 "$OUT" "is not a worktree"
  [ ! -f "$STUB_SEED_LOG" ] || fail "mutant without the add-rc check reached seed-target"
  RC=0; OUT=$(cd "$main" && PATH=$t/shim:$PATH SHIM_ADD=silent bash "$t/mut/no-wt.sh" new mut3 2>&1) || RC=$?
  [ -f "$STUB_SEED_LOG" ] || fail "mutant without the worktree check did not reach seed-target on an add that created nothing: $OUT"
  rm -f "$STUB_SEED_LOG"
  RC=0; OUT=$(cd "$main" && PATH=$t/shim:$PATH SHIM_ADD=fail bash "$t/mut/no-guards.sh" new mut4 2>&1) || RC=$?
  if [ -f "$STUB_SEED_LOG" ]; then
    [ "$(cut -f1 "$STUB_SEED_LOG")" = "$main" ] || fail "mutant without both guards: seed-target ran, but not in the main tree: $(cat "$STUB_SEED_LOG")"
  else
    fail "mutant without both guards did not run the next step after a failed add: $OUT"
  fi
  rm -f "$STUB_SEED_LOG" "$WT_MARKER"

  # no seed: exit 0, the cold line, the tree made, the proof line, seed-target not called
  mv "$t/seed.tsv" "$t/seed.tsv.away"
  run_wt "$SELF" "$main" new cold1 "$c3"
  expect "new with no seed record" "$RC" 0 "$OUT" "cold: no seed record"
  case $OUT in *"worktree list: $t/bloomery-cold1 "*"[cold1]"*) ;; *) fail "new does not print the worktree list line of the new tree: $OUT" ;; esac
  [ "$(git -C "$t/bloomery-cold1" rev-parse HEAD)" = "$c3" ] || fail "new cold1 c3 is not at c3"
  [ ! -f "$STUB_SEED_LOG" ] || fail "a cold new called seed-target"
  case $OUT in *"copied CLAUDE.local.md from $main"*) ;; *) fail "new does not say it copied CLAUDE.local.md: $OUT" ;; esac
  cmp -s "$main/CLAUDE.local.md" "$t/bloomery-cold1/CLAUDE.local.md" || fail "the new tree's CLAUDE.local.md is missing or not byte-identical to the main tree's"
  rm_tree "$t/bloomery-cold1" cold1
  mv "$t/seed.tsv.away" "$t/seed.tsv"

  # CLAUDE.local.md (the workstation rules import, ignored by git): a missing source is one warning and the tree is handed out;
  # a copy that fails stops with 67 before the seed (a seed record is present, so the next step would be seed-target); the
  # mutant without the copy leaves the tree without it, and the one that ignores the failure runs on to seed-target
  mv "$main/CLAUDE.local.md" "$main/CLAUDE.local.md.away"
  run_wt "$SELF" "$main" new nomd "$c4"
  expect "new with no CLAUDE.local.md in the main tree" "$RC" 0 "$OUT" "warning: $main/CLAUDE.local.md does not exist"
  [ ! -e "$t/bloomery-nomd/CLAUDE.local.md" ] || fail "a missing source produced a CLAUDE.local.md"
  rm_tree "$t/bloomery-nomd" nomd
  mv "$main/CLAUDE.local.md.away" "$main/CLAUDE.local.md"
  rm -f "$STUB_SEED_LOG"
  if [ "$(id -u)" != 0 ]; then
    chmod 000 "$main/CLAUDE.local.md"
    run_wt "$SELF" "$main" new mdfail "$c4"
    expect "new where CLAUDE.local.md cannot be copied" "$RC" 67 "$OUT" "could not copy $main/CLAUDE.local.md"
    [ -d "$t/bloomery-mdfail" ] || fail "the tree of a failed copy is not made"
    [ ! -f "$STUB_SEED_LOG" ] || fail "a failed CLAUDE.local.md copy went on to seed-target"
    [ "$(grep -c '# local-md-copy$' "$SELF")" = 1 ] || fail "the anchor '# local-md-copy' is not on exactly one line of this script"
    sed 's/ || return \$? # local-md-copy$/ || true # local-md-copy/' "$SELF" > "$t/mut/md-ignore.sh"
    cmp -s "$SELF" "$t/mut/md-ignore.sh" && fail "the md-ignore mutant did not apply"
    rm_tree "$t/bloomery-mdfail" mdfail
    RC=0; OUT=$(cd "$main" && bash "$t/mut/md-ignore.sh" new mdfail2 "$c4" 2>&1) || RC=$?
    [ -f "$STUB_SEED_LOG" ] || fail "mutant that ignores a failed copy did not run on to seed-target: the 67 case proves nothing"
    rm_tree "$t/bloomery-mdfail2" mdfail2
    chmod 644 "$main/CLAUDE.local.md"
    rm -f "$STUB_SEED_LOG"
  fi
  sed '/# local-md-copy$/d' "$SELF" > "$t/mut/md-none.sh"
  RC=0; OUT=$(cd "$main" && bash "$t/mut/md-none.sh" new mdnone "$c4" 2>&1) || RC=$?
  [ ! -e "$t/bloomery-mdnone/CLAUDE.local.md" ] || fail "mutant without the copy still has CLAUDE.local.md: the byte-identity check proves nothing"
  rm_tree "$t/bloomery-mdnone" mdnone
  rm -f "$STUB_SEED_LOG"

  # the choice: the last line of a path, the ancestors only, gone trees and busy trees skipped by name, the drift counted,
  # the newer lint on a tie. Base c4.
  : > "$t/seed.tsv"
  for i in B R P Q BUSY GONE NOTARGET SIDE; do mkdir -p "$t/seed$i/target"; done
  rm -rf "$t/seedGONE"
  rmdir "$t/seedNOTARGET/target"
  git -C "$main" worktree add -q -b p "$t/bloomery-p" "$c2"
  git -C "$main" worktree add -q -b q "$t/bloomery-q" "$c3"
  mkdir -p "$t/bloomery-p/target" "$t/bloomery-q/target"
  mkdir -p "$t/seedBUSY/target/mac-static"
  # the busy seed's "live" pid must outlive every pick below, however slow the host: a `sleep 60` died before the picks on
  # the second Mac (the wt block took 711 s under check-recipes' parallel parts, 2026-10-13), so BUSY counted as a dead run
  # and won every pick. It lives until this test kills it (below, or the EXIT trap's `jobs -p`).
  sleep 3600 & pid=$!
  printf 'fmt-check %s\ncargo-lane %s\n' "$pid" "$pid" > "$t/seedBUSY/target/mac-static/pids"
  : > "$t/seedBUSY/target/mac-static/fmt-check.log"
  write_rec "$t/seedB" "$c1" 100        # B: an older line, c1 (3 behind) ...
  write_rec "$t/seedB" "$c3" 200        # ... its last line c3 (1 behind) wins
  write_rec "$t/bloomery-p" "$c4" 300   # P: 0 behind, but its HEAD sits at c2: drifted by 2, cost 2
  write_rec "$t/bloomery-q" "$c3" 400   # Q: 1 behind, HEAD at c3: cost 1
  write_rec "$t/seedBUSY" "$c4" 500     # 0 behind, but a mac-static run is in flight
  write_rec "$t/seedGONE" "$c4" 600
  write_rec "$t/seedNOTARGET" "$c4" 700
  write_rec "$t/seedSIDE" "$s1" 800     # a commit that is not an ancestor of the base
  write_rec "$t/seedA" "deadbeef" 900   # not a commit of this repo
  printf 'junk line\n%s\t%s\t%s\tcombos\n' "$t/seedA" "$c4" 1000 >> "$t/seed.tsv" # not lint lines
  write_rec "$t/seedR" "$c3" 1500       # R: 1 behind, no worktree (HEAD unknown, no drift), newer than Q: wins the tie
  run_wt "$SELF" "$main" new pick1 "$c4"
  expect "new with a seed record" "$RC" 0 "$OUT" "seed-target: seeded 3 units, 105 bytes"
  case $OUT in *"wt: seed $t/seedR: last green lint at ${c3:0:8}, 1 commit(s) behind ${c4:0:8}"*) ;; *) fail "the pick is not seedR (cost 1, newer than Q on the tie): $OUT" ;; esac
  [ "$(cat "$STUB_SEED_LOG.md")" = present ] || fail "CLAUDE.local.md was not in the new tree when seed-target ran (the copy comes first)"
  cmp -s "$main/CLAUDE.local.md" "$t/bloomery-pick1/CLAUDE.local.md" || fail "a seeded new tree's CLAUDE.local.md is not byte-identical to the main tree's"
  [ "$(cat "$STUB_SEED_LOG")" = "$(printf '%s\t%s\t%s' "$main" "$t/seedR" "$t/bloomery-pick1")" ] || fail "seed-target did not get (seed, new tree) from the main tree's cwd: $(cat "$STUB_SEED_LOG")"
  rm_tree "$t/bloomery-pick1" pick1
  # without R: Q wins (cost 1) over P (distance 0, drift 2); without Q: P (cost 2) over B (cost 1)? no: B is cost 1 and older
  rm -rf "$t/seedR"
  rm -f "$STUB_SEED_LOG"
  run_wt "$SELF" "$main" new pick2 "$c4"
  case $OUT in *"wt: seed $t/bloomery-q: last green lint at ${c3:0:8}, 1 commit(s) behind"*) ;; *) fail "the pick is not Q (cost 1 beats P's distance 0 plus drift 2, and wins the tie over B by its newer lint): $OUT" ;; esac
  case $OUT in *"its HEAD moved"*) fail "a steady seed is printed as drifted: $OUT" ;; esac
  rm_tree "$t/bloomery-pick2" pick2
  rm -rf "$t/bloomery-q/target"
  run_wt "$SELF" "$main" new pick3 "$c4"
  case $OUT in *"wt: seed $t/seedB: last green lint at ${c3:0:8}, 1 commit(s) behind"*) ;; *) fail "the pick is not B's last line (c3, cost 1; its older line c1 would cost 3): $OUT" ;; esac
  rm_tree "$t/bloomery-pick3" pick3
  rm -rf "$t/seedB/target"
  run_wt "$SELF" "$main" new pick4 "$c4"
  case $OUT in *"wt: seed $t/bloomery-p: last green lint at ${c4:0:8}, 0 commit(s) behind ${c4:0:8}, its HEAD moved 2 commit(s) since"*) ;; *) fail "the pick is not the drifted P, named as drifted: $OUT" ;; esac
  rm_tree "$t/bloomery-pick4" pick4
  rm -rf "$t/bloomery-p/target"
  rm -f "$STUB_SEED_LOG"
  run_wt "$SELF" "$main" new pick5 "$c4"
  expect "new where no record fits" "$RC" 0 "$OUT" "cold: no seed fits (9 record(s): "
  case $OUT in *"1 with a mac-static run in flight"*) ;; *) fail "the busy seed is not counted by name: $OUT" ;; esac
  [ ! -f "$STUB_SEED_LOG" ] || fail "new called seed-target with no fitting seed"
  rm_tree "$t/bloomery-pick5" pick5
  kill "$pid" 2> /dev/null
  wait "$pid" 2> /dev/null
  rm -rf "$t/bloomery-p" "$t/bloomery-q"
  git -C "$main" worktree prune
  git -C "$main" branch -D p q > /dev/null 2>&1

  # seed-target refusing: the tree is made and cold, rc 70 names its rc
  : > "$t/seed.tsv"
  mkdir -p "$t/seedF/target"
  write_rec "$t/seedF" "$c4" 1000
  RC=0; OUT=$(cd "$main" && STUB_SEED_RC=67 bash "$SELF" new refused 2>&1) || RC=$?
  expect "seed-target refusing" "$RC" 70 "$OUT" "seed-target ended rc 67"
  [ -d "$t/bloomery-refused" ] || fail "the tree of a failed seed is not made"
  rm_tree "$t/bloomery-refused" refused

  # the seed is read-only: the real seed-target.sh (macOS), the main tree as the seed, a target/ of files with their own
  # mtimes; git status and the stat of every file under target/ are identical before and after `new`, and the new tree is seeded
  if [ "$(uname)" = Darwin ]; then
    mkdir -p "$main/target/debug/.fingerprint/u1" "$main/target/debug/build/p/h/fingerprint"
    printf 'a' > "$main/target/debug/.fingerprint/u1/lib-u1"
    printf 'bb' > "$main/target/debug/build/p/h/fingerprint/lib-p"
    find "$main/target" -exec touch -t 202001010000 {} +
    : > "$t/seed.tsv"
    write_rec "$main" "$c4" 1000
    readonly_state() {
      git --no-optional-locks -C "$main" status --porcelain
      find "$main/target" -exec stat -f '%m %z %N' {} + | sort
    }
    snap0=$(readonly_state)
    RC=0; OUT=$(cd "$main" && WT_SEED_TARGET=$HERE/tools/seed-target.sh bash "$SELF" new real4 2>&1) || RC=$?
    expect "new with the real seed-target" "$RC" 0 "$OUT" "seed-target: seeded 2 units, 3 bytes"
    snap1=$(readonly_state)
    [ "$snap0" = "$snap1" ] || fail "the seed's status or target/ stats changed across new"
    [ -f "$t/bloomery-real4/target/debug/build/p/h/fingerprint/lib-p" ] || fail "the new tree has no seeded target/"
    rm -rf "$t/bloomery-real4/target"
    rm_tree "$t/bloomery-real4" real4
    # FAIL-first for that instrument: a mutant that writes into the seed's target/ before the copy changes the state
    sed 's|^\(.*\)# seed-call$|touch "$SEED_TREE/target/zz-mutant"; \1# seed-call|' "$SELF" > "$t/mut/touch.sh"
    RC=0; OUT=$(cd "$main" && WT_SEED_TARGET=$HERE/tools/seed-target.sh bash "$t/mut/touch.sh" new real5 2>&1) || RC=$?
    [ "$(readonly_state)" != "$snap0" ] || fail "mutant that writes into the seed's target/ left its state unchanged: the read-only proof proves nothing"
    rm -f "$main/target/zz-mutant"
    rm -rf "$t/bloomery-real5/target"
    rm_tree "$t/bloomery-real5" real5
  else
    out "self-test: the real seed-target case is macOS only (skipped on $(uname))"
  fi
  rm -rf "$main/target"

  # retire: the in-flight rule, by tool. A tree to retire, with a target/ of 1 MiB.
  : > "$t/seed.tsv"
  retire_fixture() { # NAME: a worktree with a 1 MiB target/, no run marks
    rm_tree "$t/bloomery-$1" "$1"
    git -C "$main" worktree add -q -b "$1" "$t/bloomery-$1" "$c3"
    mkdir -p "$t/bloomery-$1/target"
    dd if=/dev/zero of="$t/bloomery-$1/target/blob" bs=1048576 count=1 2> /dev/null
  }
  retire_fixture rt
  rm -f "$STUB_DISK_LOG"
  sleep 3600 & pid=$! # killed below; see the busy seed above for why not a short sleep
  d=$t/bloomery-rt/target/mac-static
  mkdir -p "$d"
  printf 'fmt-check %s\ncargo-lane %s\n' "$pid" "$pid" > "$d/pids"
  : > "$d/fmt-check.log"
  run_wt "$SELF" "$main" retire rt
  expect "retire with a mac-static run in flight" "$RC" 1 "$OUT" "refused retire rt: mac-static run in flight: $d"
  case $OUT in *"no .rc for fmt-check lint combos"*"alive"*) ;; *) fail "the in-flight line does not name the missing .rc files and the live pid: $OUT" ;; esac
  [ ! -f "$STUB_DISK_LOG" ] || fail "retire called mac-disk with a mac-static run in flight"
  [ -d "$t/bloomery-rt" ] || fail "retire removed a tree with a run in flight"
  # a step with a log and no .rc, the start mark being the log alone (pids not written yet): no pid on record, in flight
  rm -f "$d/pids"
  run_wt "$SELF" "$main" retire rt
  expect "retire with a step log and no pids file" "$RC" 1 "$OUT" "no pid on record"
  # the same run killed: every recorded pid dead, named, not blocking
  true & wait $!
  dead=$!
  printf 'fmt-check %s\ncargo-lane %s\n' "$dead" "$dead" > "$d/pids"
  run_wt "$SELF" "$main" retire --dry-run rt
  expect "retire with a dead mac-static run" "$RC" 0 "$OUT" "note: mac-static run left by a dead run: $d"
  [ -f "$STUB_DISK_LOG" ] || fail "retire did not reach mac-disk with only a dead run's marks"
  rm -f "$STUB_DISK_LOG"
  # every step ended: .rc for fmt-check, lint and combos (cargo-lane stands for the last two)
  printf '0\n' > "$d/fmt-check.rc"; printf '0\n' > "$d/lint.rc"; printf '0\n' > "$d/combos.rc"
  : > "$d/lint.log"; : > "$d/combos.log"
  printf 'fmt-check %s\ncargo-lane %s\n' "$pid" "$pid" > "$d/pids"
  run_wt "$SELF" "$main" retire --dry-run rt
  expect "retire with every .rc present (a live pid in pids)" "$RC" 0 "$OUT" "dry run: nothing removed"
  [ "$(cut -f2 "$STUB_DISK_LOG")" = "retire --dry-run $t/bloomery-rt" ] || fail "mac-disk did not get 'retire --dry-run <abs path>': $(cat "$STUB_DISK_LOG")"
  [ "$(cut -f1 "$STUB_DISK_LOG")" = "$main" ] || fail "mac-disk was not run from the main tree: $(cat "$STUB_DISK_LOG")"
  # a gate-batch run: run.log with no DONE and a live lane pid blocks; with DONE it does not; no pid on record blocks
  rm -rf "$d"
  b=$t/batches/bloomery-rt/20260101-000000
  mkdir -p "$b"
  printf 'item rc=0 1s try=1 lane=A\n' > "$b/run.log"
  echo "$pid" > "$b/lane-A.pid"
  rm -f "$STUB_DISK_LOG"
  run_wt "$SELF" "$main" retire rt
  expect "retire with a gate-batch run in flight" "$RC" 1 "$OUT" "gate-batch run in flight: $b"
  [ ! -f "$STUB_DISK_LOG" ] || fail "retire called mac-disk with a gate-batch run in flight"
  rm -f "$b/lane-A.pid"
  run_wt "$SELF" "$main" retire rt
  expect "retire with a gate-batch run and no pid on record" "$RC" 1 "$OUT" "no pid on record"
  mkdir -p "$t/bloomery-rt/target/gate-batch/x"
  printf 'item rc=0 1s try=1 lane=A\n' > "$t/bloomery-rt/target/gate-batch/x/run.log"
  printf 'DONE total=1 red=0 tier=real wall=1s laneA=1s laneB=0s\n' >> "$b/run.log"
  run_wt "$SELF" "$main" retire rt
  expect "retire with a batch under target/ that has no DONE" "$RC" 1 "$OUT" "gate-batch run may be in flight: $t/bloomery-rt/target/gate-batch/x"
  printf 'DONE total=1 red=0 tier=real wall=1s laneA=1s laneB=0s\n' >> "$t/bloomery-rt/target/gate-batch/x/run.log"
  kill "$pid" 2> /dev/null
  wait "$pid" 2> /dev/null
  rm -f "$STUB_DISK_LOG"
  # nothing in flight: reaches mac-disk (which removes the tree), the bytes are the du before the call
  run_wt "$SELF" "$main" retire rt
  expect "retire with every run ended" "$RC" 0 "$OUT" "retired rt: $t/bloomery-rt is gone"
  case $OUT in *"its target/ held "[0-9]*" bytes by du"*) ;; *) fail "retire prints no du bytes: $OUT" ;; esac
  o=$(printf '%s\n' "$OUT" | sed -n 's/.*its target\/ held \([0-9]*\) bytes.*/\1/p')
  [ "${o:-0}" -ge 1048576 ] || fail "du before the call is $o bytes, under the 1 MiB blob"
  [ ! -e "$t/bloomery-rt" ] || fail "retire left the tree"
  [ -n "$(git -C "$main" branch --list rt)" ] || fail "retire deleted the branch"
  # mac-disk refusing passes through: its line, its rc, no 'retired' line, the tree stays
  retire_fixture rt2
  RC=0; OUT=$(cd "$main" && STUB_DISK=refuse bash "$SELF" retire rt2 2>&1) || RC=$?
  expect "retire where mac-disk refuses" "$RC" 1 "$OUT" "refused bloomery-rt2: LIVE"
  case $OUT in *"retired rt2"*) fail "retire claims a retire mac-disk refused: $OUT" ;; esac
  [ -d "$t/bloomery-rt2" ] || fail "the tree of a refused retire is gone"
  # a target/ that mac-disk left: removed here, and said so; FAIL-first: the mutant keeps it
  RC=0; OUT=$(cd "$main" && STUB_DISK=keep-target bash "$SELF" retire rt2 2>&1) || RC=$?
  expect "retire where mac-disk keeps a target/" "$RC" 0 "$OUT" "removed the target/ mac-disk left"
  [ ! -e "$t/bloomery-rt2/target" ] || fail "retire left the target/ mac-disk kept"
  [ "$(grep -c '# leftover-rm$' "$SELF")" = 1 ] || fail "the anchor '# leftover-rm' is not on exactly one line of this script"
  sed '/# leftover-rm$/d' "$SELF" > "$t/mut/keep.sh"
  retire_fixture rt3
  RC=0; OUT=$(cd "$main" && STUB_DISK=keep-target bash "$t/mut/keep.sh" retire rt3 2>&1) || RC=$?
  [ -e "$t/bloomery-rt3/target" ] || fail "mutant without the leftover removal removed it anyway"
  rm -rf "$t/bloomery-rt2" "$t/bloomery-rt3"
  git -C "$main" worktree prune
  # mac-disk runs from the main tree, whatever the caller's cwd (it finds the main repo from its cwd, and a process with its
  # cwd inside the tree would be a LIVE refusal of its own making)
  retire_fixture rt4
  retire_fixture other
  rm -f "$STUB_DISK_LOG"
  run_wt "$SELF" "$t/bloomery-other" retire --dry-run rt4
  expect "retire called from another worktree" "$RC" 0 "$OUT" "dry run: nothing removed"
  [ "$(cut -f1 "$STUB_DISK_LOG")" = "$main" ] || fail "mac-disk was run from $(cut -f1 "$STUB_DISK_LOG"), not from the main tree"
  rm -rf "$t/bloomery-rt4" "$t/bloomery-other"
  git -C "$main" worktree prune
  # one owner: this script holds no copy of mac-disk's checks
  if sed '/^self_test() {/,$d' "$SELF" | grep -v '^[[:space:]]*#' | grep -E 'lsof|LANDED|status --porcelain|origin/main' > /dev/null; then
    fail "this script carries a copy of a mac-disk check (lsof, LANDED, git status, origin)"
  fi

  # the real mac-disk behind it (a dry run, an origin for LANDED; its cwd scan is its own documented seam, MAC_DISK_LSOF_PATHS,
  # so the cases do not depend on the machine's processes): the interface is right, and its LIVE refusal and its rc 69 pass through
  if command -v python3 > /dev/null 2>&1; then
    git clone -q --bare "$main" "$t/origin.git"
    git -C "$main" remote add origin "$t/origin.git"
    git -C "$main" fetch -q origin
    retire_fixture dr
    RC=0; OUT=$(cd "$main" && MAC_DISK_LSOF_PATHS= WT_MAC_DISK=$HERE/tools/mac-disk.py WT_MAC_DISK_RUN=python3 bash "$SELF" retire --dry-run dr 2>&1) || RC=$?
    expect "retire --dry-run through the real mac-disk" "$RC" 0 "$OUT" "bloomery-dr: would remove the worktree"
    [ -d "$t/bloomery-dr" ] || fail "a dry run removed the tree"
    RC=0; OUT=$(cd "$main" && MAC_DISK_LSOF_PATHS=$t/bloomery-dr WT_MAC_DISK=$HERE/tools/mac-disk.py WT_MAC_DISK_RUN=python3 bash "$SELF" retire --dry-run dr 2>&1) || RC=$?
    expect "retire of a tree with a process in it, through the real mac-disk" "$RC" 1 "$OUT" "LIVE: a process has its cwd inside it"
    RC=0; OUT=$(cd "$main" && MAC_DISK_LSOF_FAIL=1 WT_MAC_DISK=$HERE/tools/mac-disk.py WT_MAC_DISK_RUN=python3 bash "$SELF" retire --dry-run dr 2>&1) || RC=$?
    expect "retire where the cwd scan fails, through the real mac-disk" "$RC" 69 "$OUT" "the cwd scan failed"
  else
    out "self-test: the real mac-disk cases need python3 (skipped)"
  fi

  if [ "$fails" != 0 ]; then
    say "self-test $fails failed"
    return 1
  fi
  echo "wt: self-test ok"
}

case ${1:-} in
  --self-test) [ $# = 1 ] || { say "--self-test takes nothing"; exit 64; }; self_test; exit $? ;;
  new) shift; cmd_new "$@"; exit $? ;;
  retire) shift; cmd_retire "$@"; exit $? ;;
  *) say "usage: tools/wt.sh new NAME [BASE] | retire [--dry-run] NAME | --self-test"; exit 64 ;;
esac
